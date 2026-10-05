"""
Vector backend auto-tiering — decision logic, factory, and the
VectorBackend abstraction.

The decision function is a PURE function (no embedders, no I/O) so the
tiering policy is fully testable without model downloads. Only tier 0
(memory) is implemented today; tiers 1-4 (npy / parallel / faiss /
elasticsearch) are interface-only and MUST degrade gracefully to memory
with a log — never raise, never silently pretend.
"""

import numpy as np
import pytest

from ontology.core.vector_backend import (
    VectorBackend,
    decide_backend_tier,
    create_backend,
    select_backend,
    TIER_MEMORY, TIER_NPY, TIER_PARALLEL, TIER_FAISS, TIER_DISTRIBUTED,
    NPY_PROMOTE, PARALLEL_PROMOTE, FAISS_PROMOTE, DISTRIBUTED_PROMOTE,
)


def fake_embed(texts):
    dim = 64
    import zlib
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in str(text).lower().split():
            vectors[i, zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    return vectors


# ─── 1. decide_backend_tier — thresholds (pure) ─────────────────────

class TestTierThresholds:
    def test_small_count_is_memory(self):
        assert decide_backend_tier(500) == TIER_MEMORY

    def test_just_below_npy_is_memory(self):
        assert decide_backend_tier(NPY_PROMOTE - 1) == TIER_MEMORY

    def test_npy_threshold(self):
        assert decide_backend_tier(NPY_PROMOTE) == TIER_NPY

    def test_parallel_threshold(self):
        assert decide_backend_tier(PARALLEL_PROMOTE) == TIER_PARALLEL

    def test_faiss_threshold(self):
        assert decide_backend_tier(FAISS_PROMOTE) == TIER_FAISS

    def test_distributed_threshold(self):
        assert decide_backend_tier(DISTRIBUTED_PROMOTE) == TIER_DISTRIBUTED

    def test_hundred_million_is_distributed(self):
        assert decide_backend_tier(100_000_000) == TIER_DISTRIBUTED


# ─── 2. override (param + env) ──────────────────────────────────────

class TestOverride:
    def test_param_override_forces_tier_regardless_of_count(self):
        assert decide_backend_tier(10, override="faiss") == TIER_FAISS
        assert decide_backend_tier(10_000_000, override="memory") == TIER_MEMORY

    def test_env_override_forces_tier(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_VECTOR_BACKEND", "memory")
        assert decide_backend_tier(10_000_000) == TIER_MEMORY

    def test_auto_override_uses_count(self):
        assert decide_backend_tier(NPY_PROMOTE, override="auto") == TIER_NPY

    def test_unknown_override_falls_back_to_auto(self):
        # never raises; ignores the garbage and decides by count
        assert decide_backend_tier(NPY_PROMOTE, override="banana") == TIER_NPY


# ─── 3. hysteresis (avoid boundary thrashing) ───────────────────────

class TestHysteresis:
    def test_holds_current_tier_within_band(self):
        # base would demote to memory, but within 0.8x of npy threshold → hold
        n = int(NPY_PROMOTE * 0.9)
        assert decide_backend_tier(n, current_tier=TIER_NPY) == TIER_NPY

    def test_demotes_when_clearly_below_band(self):
        n = int(NPY_PROMOTE * 0.5)
        assert decide_backend_tier(n, current_tier=TIER_NPY) == TIER_MEMORY

    def test_faiss_held_within_band(self):
        n = int(FAISS_PROMOTE * 0.9)
        assert decide_backend_tier(n, current_tier=TIER_FAISS) == TIER_FAISS

    def test_promotion_ignores_hysteresis(self):
        # growth always promotes immediately (no hold on the way up)
        assert decide_backend_tier(FAISS_PROMOTE, current_tier=TIER_NPY) == TIER_FAISS


# ─── 4. never-raises on bad input ───────────────────────────────────

class TestNeverRaises:
    def test_negative_count(self):
        assert decide_backend_tier(-5) == TIER_MEMORY

    def test_zero_count(self):
        assert decide_backend_tier(0) == TIER_MEMORY

    def test_non_numeric_count(self):
        assert decide_backend_tier("lots") == TIER_MEMORY


# ─── 5. create_backend — tier 0 real, tiers 1-4 degrade ─────────────

class TestCreateBackend:
    def test_memory_creates_semantic_index(self):
        b = create_backend(TIER_MEMORY, embed_fn=fake_embed)
        assert isinstance(b, VectorBackend)
        assert b.tier_name == TIER_MEMORY

    @pytest.mark.parametrize("tier", [TIER_NPY, TIER_PARALLEL, TIER_FAISS, TIER_DISTRIBUTED])
    def test_unimplemented_tier_degrades_to_working_backend(self, tier):
        # degrade to memory: still a VectorBackend that actually searches
        b = create_backend(tier, embed_fn=fake_embed)
        assert isinstance(b, VectorBackend)
        b.upsert("n1", "날씨 정보 제공", "agent")
        assert b.search("날씨")[0]["node_id"] == "n1"

    def test_unknown_tier_degrades(self):
        b = create_backend("banana", embed_fn=fake_embed)
        assert isinstance(b, VectorBackend)


# ─── 6. VectorBackend abstraction ───────────────────────────────────

class TestAbstraction:
    def test_semantic_index_is_a_vector_backend(self):
        from ontology.core.semantic_index import SemanticIndex
        assert isinstance(SemanticIndex(embed_fn=fake_embed), VectorBackend)

    def test_tier_name_is_memory(self):
        from ontology.core.semantic_index import SemanticIndex
        assert SemanticIndex(embed_fn=fake_embed).tier_name == TIER_MEMORY


# ─── 7. select_backend (decide + create + log) ──────────────────────

class TestSelectBackend:
    def test_small_returns_working_memory_backend(self):
        b = select_backend(3, embed_fn=fake_embed)
        assert isinstance(b, VectorBackend)
        b.upsert("n1", "날씨", "agent")
        assert b.search("날씨")[0]["node_id"] == "n1"

    def test_large_count_degrades_but_still_works(self):
        # decision = faiss, but unimplemented → degrades to a usable backend
        b = select_backend(FAISS_PROMOTE, embed_fn=fake_embed)
        assert isinstance(b, VectorBackend)
        b.upsert("n1", "가격 검색", "agent")
        assert b.search("가격")[0]["node_id"] == "n1"


# ─── 8. KG integration (drop-in) ────────────────────────────────────

class TestKGIntegration:
    def test_init_semantic_index_returns_vector_backend(self):
        import asyncio
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        kg = KnowledgeGraphEngine(fast_mode=True)
        asyncio.run(kg.add_concept("weather_agent", "agent",
                                   {"name": "weather_agent", "description": "도시 날씨 정보"}))
        backend = kg.init_semantic_index(embed_fn=fake_embed)
        assert isinstance(backend, VectorBackend)
        assert backend.tier_name == TIER_MEMORY  # tiny graph
        assert kg.semantic_search("날씨", top_k=3)
