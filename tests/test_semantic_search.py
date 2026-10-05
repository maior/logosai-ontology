"""
Semantic search tests — embedding index over graph nodes,
similarity entry + graph-traversal expansion to agents.

Unit tests use an injected deterministic fake embedder (no model download).
One integration test uses the real sentence-transformers model and is
skipped when the dependency is unavailable.
"""

import asyncio
import zlib

import numpy as np
import pytest


def run(coro):
    return asyncio.run(coro)


def fake_embed(texts):
    """Deterministic bag-of-tokens embedder: shared tokens → high cosine.

    Uses crc32 (not built-in hash) so results are stable across processes.
    """
    dim = 64
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in str(text).lower().split():
            vectors[i, zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    return vectors


@pytest.fixture()
def kg():
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
    return KnowledgeGraphEngine(fast_mode=True)


def seed_agents(kg):
    """weather / shopping agents with capabilities, tags, is_a hierarchy."""
    run(kg.add_concept("weather_agent", "agent", {
        "name": "weather_agent", "description": "도시 날씨 정보 제공"}))
    run(kg.add_concept("shopping_agent", "agent", {
        "name": "shopping_agent", "description": "상품 가격 검색 쇼핑"}))

    run(kg.add_concept("capability_forecast", "capability", {
        "name": "forecast", "display_name": "일기예보",
        "description": "도시 날씨 예보 제공"}))
    run(kg.add_concept("capability_price_search", "capability", {
        "name": "price_search", "display_name": "가격 검색",
        "description": "상품 가격 비교 검색"}))
    run(kg.add_relationship("weather_agent", "capability_forecast", "has_capability"))
    run(kg.add_relationship("shopping_agent", "capability_price_search", "has_capability"))

    run(kg.add_concept("tag_기상", "tag", {"name": "기상 날씨"}))
    run(kg.add_relationship("weather_agent", "tag_기상", "has_tag"))


# ─── 1. SemanticIndex (pure unit) ───────────────────────────────────

class TestSemanticIndex:
    def _index(self):
        from ontology.core.semantic_index import SemanticIndex
        return SemanticIndex(embed_fn=fake_embed)

    def test_search_ranks_relevant_node_first(self):
        idx = self._index()
        idx.upsert("cap_weather", "날씨 예보 제공", "capability")
        idx.upsert("cap_shopping", "상품 가격 검색", "capability")

        results = idx.search("날씨 알려줘", top_k=2)
        assert results[0]["node_id"] == "cap_weather"
        assert results[0]["score"] > 0

    def test_search_filters_by_node_type(self):
        idx = self._index()
        idx.upsert("a1", "날씨 정보", "agent")
        idx.upsert("t1", "날씨 정보", "tag")

        results = idx.search("날씨", node_types=["tag"])
        assert [r["node_id"] for r in results] == ["t1"]

    def test_upsert_skips_unchanged_text(self):
        idx = self._index()
        assert idx.upsert("n1", "같은 텍스트", "agent") is True
        assert idx.upsert("n1", "같은 텍스트", "agent") is False  # no re-embed
        assert idx.upsert("n1", "바뀐 텍스트", "agent") is True
        assert len(idx) == 1  # still one entry

    def test_empty_index_and_empty_query(self):
        idx = self._index()
        assert idx.search("아무거나") == []
        idx.upsert("n1", "텍스트", "agent")
        assert idx.search("") == []
        assert idx.search("   ") == []

    def test_top_k_larger_than_index(self):
        idx = self._index()
        idx.upsert("n1", "날씨", "agent")
        assert len(idx.search("날씨", top_k=100)) == 1

    def test_remove(self):
        idx = self._index()
        idx.upsert("n1", "날씨", "agent")
        idx.remove("n1")
        assert idx.search("날씨") == []
        assert len(idx) == 0

    def test_no_embedder_degrades_gracefully(self):
        # embed_fn=None + default disabled → search returns [], never raises
        from ontology.core.semantic_index import SemanticIndex
        idx = SemanticIndex(embed_fn=None, auto_default=False)
        assert idx.upsert("n1", "텍스트", "agent") is False
        assert idx.search("텍스트") == []


# ─── 2. Node text composition ───────────────────────────────────────

class TestComposeNodeText:
    def test_agent_text_has_name_and_description(self):
        from ontology.core.semantic_index import compose_node_text
        text = compose_node_text("weather_agent", {
            "type": "agent", "name": "weather_agent",
            "description": "도시 날씨 정보 제공"})
        assert "weather_agent" in text
        assert "도시 날씨 정보 제공" in text

    def test_capability_text_has_display_name_and_description(self):
        from ontology.core.semantic_index import compose_node_text
        text = compose_node_text("capability_forecast", {
            "type": "capability", "name": "forecast",
            "display_name": "일기예보", "description": "도시 날씨 예보 제공"})
        assert "일기예보" in text
        assert "도시 날씨 예보 제공" in text

    def test_missing_fields_do_not_crash(self):
        from ontology.core.semantic_index import compose_node_text
        text = compose_node_text("tag_기상", {"type": "tag"})
        assert isinstance(text, str) and "tag_기상" in text


# ─── 3. Engine-level semantic search ────────────────────────────────

class TestKGSemanticSearch:
    def test_semantic_search_finds_weather_nodes(self, kg):
        seed_agents(kg)
        kg.init_semantic_index(embed_fn=fake_embed)

        results = kg.semantic_search("날씨 알려줘", top_k=3)
        assert results, "results should not be empty"
        top_ids = [r["node_id"] for r in results]
        assert any("weather" in i or "forecast" in i or "기상" in i for i in top_ids)

    def test_find_agents_semantic_via_capability(self, kg):
        # query wording matches the capability description, not the agent name
        seed_agents(kg)
        kg.init_semantic_index(embed_fn=fake_embed)

        agents = kg.find_agents_semantic("도시 날씨 예보 제공", top_k=3)
        assert agents[0]["agent_id"] == "weather_agent"
        assert agents[0]["score"] > 0
        assert "matched_via" in agents[0]

    def test_find_agents_semantic_via_tag(self, kg):
        seed_agents(kg)
        kg.init_semantic_index(embed_fn=fake_embed)

        agents = kg.find_agents_semantic("기상 정보", top_k=3)
        assert "weather_agent" in [a["agent_id"] for a in agents]

    def test_find_agents_semantic_expands_is_a_descendants(self, kg):
        # agent holds the CHILD capability; query hits the PARENT capability text
        seed_agents(kg)
        run(kg.add_concept("capability_realtime_forecast", "capability", {
            "name": "realtime_forecast", "display_name": "실시간 예보",
            "description": "실시간 기상 레이더 예보"}))
        run(kg.add_concept("radar_agent", "agent", {
            "name": "radar_agent", "description": "레이더 관측"}))
        run(kg.add_relationship("radar_agent", "capability_realtime_forecast", "has_capability"))
        run(kg.add_relationship("capability_realtime_forecast", "capability_forecast", "is_a"))
        kg.init_semantic_index(embed_fn=fake_embed)

        # "도시 날씨 예보 제공" matches capability_forecast (parent) —
        # radar_agent holds only the child, must still be found
        agents = kg.find_agents_semantic("도시 날씨 예보 제공", top_k=5, min_score=0.01)
        assert "radar_agent" in [a["agent_id"] for a in agents]

    def test_min_score_filters_noise(self, kg):
        seed_agents(kg)
        kg.init_semantic_index(embed_fn=fake_embed)

        agents = kg.find_agents_semantic("도시 날씨 예보 제공", top_k=5, min_score=0.99)
        # perfect-match capability may survive; unrelated shopping must not
        assert "shopping_agent" not in [a["agent_id"] for a in agents]

    def test_refresh_picks_up_new_nodes(self, kg):
        seed_agents(kg)
        kg.init_semantic_index(embed_fn=fake_embed)
        assert kg.semantic_search("환율 계산") == [] or \
            all("환율" not in r["node_id"] for r in kg.semantic_search("환율 계산"))

        run(kg.add_concept("currency_agent", "agent", {
            "name": "currency_agent", "description": "환율 계산 변환"}))
        kg.refresh_semantic_index()

        results = kg.semantic_search("환율 계산", top_k=3)
        assert "currency_agent" in [r["node_id"] for r in results]

    def test_search_without_init_uses_lazy_default_or_empty(self, kg):
        # never raises even when no index was initialized explicitly
        seed_agents(kg)
        results = kg.semantic_search("날씨")
        assert isinstance(results, list)


# ─── 4. Real model integration (skipped without dependency) ────────

class TestRealModelIntegration:
    @pytest.mark.slow
    def test_real_model_ranks_weather_over_shopping(self, kg):
        pytest.importorskip("sentence_transformers")
        seed_agents(kg)
        kg.init_semantic_index()  # real default embedder

        # paraphrased query — no token overlap with node texts required
        agents = kg.find_agents_semantic("내일 비 오는지 알려주는 에이전트", top_k=3)
        assert agents, "real model should return results"
        ids = [a["agent_id"] for a in agents]
        assert ids[0] == "weather_agent"
        if "shopping_agent" in ids:
            weather_score = next(a["score"] for a in agents if a["agent_id"] == "weather_agent")
            shopping_score = next(a["score"] for a in agents if a["agent_id"] == "shopping_agent")
            assert weather_score > shopping_score
