"""
Vector backend auto-tiering — pick the right similarity-search engine for
the graph's size, so callers never configure infrastructure by hand.

Philosophy (same as the rest of the framework): a *smart default*, not
opaque magic.
  - Observable   — every selection is logged (which tier, how many nodes).
  - Overridable  — ONTOLOGY_VECTOR_BACKEND (or an override arg) forces a tier.
  - Never raises — an unimplemented/unavailable tier degrades to memory.
  - No thrash    — hysteresis holds the current tier near a boundary.

Three levers matter at different scales, hence the tiers:
  - memory  (<2K)        numpy brute-force, no persistence. Rebuild is instant.
  - npy     (2K–200K)    brute-force + on-disk .npy cache. Kills cold-start
                         re-embedding. Search is still exact and fast.
  - parallel(200K–1M)    sharded parallel matvec. Exact, no new dependency;
                         parallelism buys latency headroom.
  - faiss   (1M–50M)     FAISS ANN. Sub-linear search when brute-force
                         latency/RAM finally bite.
  - distributed(50M+)    external store (Elasticsearch, adapter) — raw vectors
                         no longer fit one machine; needs compression/sharding.

Only tier 0 (memory) is implemented today; the rest are interface-only and
degrade to memory. The decision logic below encodes all five so that adding
a real backend later is a drop-in with no change to callers.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger

# ─── Tier names (also the accepted ONTOLOGY_VECTOR_BACKEND values) ──────
TIER_MEMORY = "memory"
TIER_NPY = "npy"
TIER_PARALLEL = "parallel"
TIER_FAISS = "faiss"
TIER_DISTRIBUTED = "distributed"
# 'elasticsearch' 는 노드 수로 자동 승격되지 않는다 — 명시 선택 전용(override
# 또는 ONTOLOGY_VECTOR_BACKEND=elasticsearch). 노드 수가 아니라 **하이브리드가
# 필요한가**로 고르는 백엔드이기 때문이다: 조문번호·고유명사처럼 문자 그대로
# 맞아야 하는 질의는 임베딩만으로 놓친다. 청크 검색(chunk_index)이 주 사용처다.
TIER_ELASTICSEARCH = "elasticsearch"

# Promotion thresholds (node counts). Tunable — override any via env, e.g.
# ONTOLOGY_FAISS_PROMOTE=2000000. Defaults are order-of-magnitude estimates
# from brute-force numpy cost (N×384 MACs) and RAM (N×384×4B), not sacred.
NPY_PROMOTE = int(os.environ.get("ONTOLOGY_NPY_PROMOTE", 2_000))
PARALLEL_PROMOTE = int(os.environ.get("ONTOLOGY_PARALLEL_PROMOTE", 200_000))
FAISS_PROMOTE = int(os.environ.get("ONTOLOGY_FAISS_PROMOTE", 1_000_000))
DISTRIBUTED_PROMOTE = int(os.environ.get("ONTOLOGY_DISTRIBUTED_PROMOTE", 50_000_000))

# Demotion only below this fraction of a tier's promote threshold, so a node
# count hovering at a boundary does not rebuild the backend on every refresh.
HYSTERESIS = 0.8

_ORDER = {TIER_MEMORY: 0, TIER_NPY: 1, TIER_PARALLEL: 2,
          TIER_FAISS: 3, TIER_DISTRIBUTED: 4}

# 자동 승격 대상이 아닌, 명시 선택 전용 tier
_EXPLICIT_ONLY = {TIER_ELASTICSEARCH}

# descending: first threshold met wins
_THRESHOLDS = [
    (DISTRIBUTED_PROMOTE, TIER_DISTRIBUTED),
    (FAISS_PROMOTE, TIER_FAISS),
    (PARALLEL_PROMOTE, TIER_PARALLEL),
    (NPY_PROMOTE, TIER_NPY),
]

_PROMOTE_THRESHOLD = {
    TIER_NPY: NPY_PROMOTE,
    TIER_PARALLEL: PARALLEL_PROMOTE,
    TIER_FAISS: FAISS_PROMOTE,
    TIER_DISTRIBUTED: DISTRIBUTED_PROMOTE,
}


class VectorBackend(ABC):
    """Similarity-search index over graph nodes. The one seam every tier
    implements; SemanticIndex is the tier-0 (memory) implementation."""

    @abstractmethod
    def upsert(self, node_id: str, text: str, node_type: str) -> bool: ...

    @abstractmethod
    def remove(self, node_id: str) -> bool: ...

    @abstractmethod
    def build_from_graph(self, graph, node_types=...) -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5,
               node_types: Optional[List[str]] = None,
               min_score: float = 0.0) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    def tier_name(self) -> str:
        """The tier this instance actually is (after any degradation)."""
        return TIER_MEMORY


def _base_tier(node_count: int) -> str:
    for threshold, tier in _THRESHOLDS:
        if node_count >= threshold:
            return tier
    return TIER_MEMORY


def decide_backend_tier(node_count: Any, current_tier: Optional[str] = None,
                        override: Optional[str] = None) -> str:
    """Pure policy: choose a tier from node count. Never raises.

    override (or the ONTOLOGY_VECTOR_BACKEND env var) forces an explicit
    tier; "auto" or anything unrecognized falls through to count-based
    selection. current_tier enables hysteresis so a count near a boundary
    does not flip tiers on every refresh.
    """
    chosen = (override or os.environ.get("ONTOLOGY_VECTOR_BACKEND", "auto")
              or "auto").strip().lower()
    if chosen in _ORDER or chosen in _EXPLICIT_ONLY:  # explicit, forced
        return chosen

    try:
        n = max(0, int(node_count))
    except (TypeError, ValueError):
        n = 0

    target = _base_tier(n)

    # hysteresis: hold the current tier when a demotion is only marginal
    if current_tier in _ORDER and _ORDER[target] < _ORDER[current_tier]:
        promote_threshold = _PROMOTE_THRESHOLD.get(current_tier)
        if promote_threshold and n >= promote_threshold * HYSTERESIS:
            target = current_tier

    return target


def create_backend(tier: str, embed_fn=None,
                   namespace: str = "default") -> VectorBackend:
    """Instantiate a backend for the tier.

    실물: memory(tier 0), npy(tier 1), elasticsearch(명시 선택).
    미구현: parallel / faiss / distributed — memory 로 degrade 하되 무엇이
    떨어졌는지 반드시 로그로 말한다 ('no silent caps' 규칙).
    """
    from .semantic_index import SemanticIndex  # lazy: avoid import cycle

    if tier == TIER_MEMORY:
        return SemanticIndex(embed_fn=embed_fn)

    if tier == TIER_NPY:
        from .npy_backend import NpyBackend
        backend = NpyBackend(embed_fn=embed_fn, namespace=namespace)
        backend.load_from_disk()  # 캐시가 있으면 재임베딩을 건너뛴다
        return backend

    if tier == TIER_ELASTICSEARCH:
        from .es_backend import ElasticsearchBackend
        backend = ElasticsearchBackend(embed_fn=embed_fn, namespace=namespace)
        if backend.available():
            return backend
        logger.warning(
            "🔎 vector backend: 'elasticsearch' selected but the cluster is "
            "unreachable — using memory (brute-force, keyword matching lost)")
        return SemanticIndex(embed_fn=embed_fn)

    if tier in _ORDER:  # a real tier, just not built yet
        logger.info(
            f"🔎 vector backend: '{tier}' selected but not yet implemented "
            f"— using memory (brute-force). Install the matching extra when available.")
        return SemanticIndex(embed_fn=embed_fn)

    logger.warning(f"🔎 vector backend: unknown tier '{tier}' — using memory")
    return SemanticIndex(embed_fn=embed_fn)


def select_backend(node_count: Any, embed_fn=None,
                   current_tier: Optional[str] = None,
                   override: Optional[str] = None,
                   namespace: str = "default") -> VectorBackend:
    """Decide + create + log. The single entry point callers use."""
    tier = decide_backend_tier(node_count, current_tier=current_tier,
                               override=override)
    logger.info(f"🔎 vector backend: {tier} ({node_count} nodes)")
    return create_backend(tier, embed_fn=embed_fn, namespace=namespace)
