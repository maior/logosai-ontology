"""
aicoach-kb chunk retriever — 라이브 약관 청크를 aicoach 의 Elasticsearch 인덱스
에서 직접 읽어 온톨로지 chunk-hit 모양으로 돌려준다.

배경: aicoach-backed 네임스페이스(AI-Coach)는 GRAPH 를 이미 aicoach.kg_node/
kg_edge 에서 라이브 하이드레이트한다(graph_store.hydrate_graph_aicoach). 그러나
CHUNK(원문 인용)만은 온톨로지가 만든 로컬 npy 스냅샷에서 왔다 — 스냅샷은 곧
낡는다. 이 모듈은 CHUNK 채널을 aicoach 의 살아있는 색인 `aicoach-kb`(≈39K 문서)
로 돌려, /retrieve 의 인용이 실시간 약관을 가리키게 한다.

인터페이스는 ChunkIndex.search 와 동일하다 — `[(StoredChunk, score)]` 를
점수 내림차순으로. 그래서 GraphConditionedRetriever 의 chunk 채널에 그대로
꽂힌다(RRF/fusion 로직 무변경).

하이브리드 공식은 es_backend 와 같은 aicoach 골든셋 실측값을 그대로 쓴다:
    _score = BM25 × 0.1 + (cosineSimilarity + 1.0)
쿼리는 온톨로지 임베더(ko-sroberta, 768)로 임베딩한다 — aicoach-kb 의 vector 도
같은 계열이라 kNN 이 성립한다. 조문번호·고유명사처럼 문자 그대로 맞아야 하는
질의를 위해 BM25(text) 를 함께 섞는다.

커널 경계: 에이전트 스택을 import 하지 않는다. ES 는 es_backend/object_index 와
같은 `elasticsearch` 클라이언트 패턴, 임베더는 semantic_index 로만 닿는다.
ES/임베더가 없으면 절대 raise 하지 않고 [] 로 degrade — 온톨로지가 외부 인프라
때문에 같이 죽지 않는다.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .chunk_store import StoredChunk
from .es_backend import BM25_WEIGHT, DEFAULT_ES_URL, _COSINE_SHIFT, _SCRIPT_SOURCE

Hit = Tuple[StoredChunk, float]

# aicoach 라이브 청크 색인 — 네임스페이스별이 아니라 하나의 공유 인덱스다
# (aicoach 가 전 상품을 한 인덱스에 넣는다). env 로만 바꾼다.
DEFAULT_AICOACH_KB_INDEX = os.environ.get("ONTOLOGY_AICOACH_KB_INDEX", "aicoach-kb")

# 응답에 실을 provenance 필드 — vector 는 제외(무겁고 인용에 불필요).
_SOURCE_FIELDS = ["text", "source", "article", "layer", "tenant", "title"]


def build_aicoach_query(query: str, query_vector, top_k: int = 5) -> Dict[str, Any]:
    """aicoach-kb 하이브리드 검색 본문 — 순수 함수(I/O·임베더 없음).

    es_backend.build_hybrid_query 와 같은 script_score 구조지만 aicoach-kb 의
    필드(text/vector/source/article/layer/tenant/title)에 맞춘다. 후보군은
    text 매칭(should)으로 좁히고 script 로 cosine 을 얹어 재랭크한다 — aicoach 가
    recall@5 0.944 를 낸 방식 그대로.
    """
    qv = list(query_vector)
    return {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"bool": {"should": [{"match": {"text": query}}]}},
                "script": {"source": _SCRIPT_SOURCE,
                           "params": {"qv": qv, "bm25_w": BM25_WEIGHT}},
            }
        },
        "_source": _SOURCE_FIELDS,
    }


def hit_to_chunk(hit: Dict[str, Any]) -> StoredChunk:
    """ES 히트 → 온톨로지 StoredChunk.

    char_start/char_end 는 aicoach-kb 가 원문 오프셋을 안 들고 있으므로
    0/len(text) 로 둔다(축 2 의 불변식은 로컬 청크에만 적용된다). node_ids/trust
    는 aicoach-kb 에 없어 빈 값 — 그래프 연결(node_ids)은 채널 B 가 채운다.
    """
    src = hit.get("_source", {}) or {}
    text = src.get("text", "") or ""
    return StoredChunk(
        chunk_id=hit.get("_id", ""),
        text=text,
        source=src.get("source", "") or "",
        index=0,
        section=src.get("article", "") or "",
        char_start=0,
        char_end=len(text),
        node_ids=[],
        trust="",
        meta={"layer": src.get("layer", "") or "",
              "tenant": src.get("tenant", "") or "",
              "title": src.get("title", "") or ""},
    )


class AicoachChunkIndex:
    """aicoach-kb 위의 검색 — ChunkIndex.search 계약을 그대로 구현한다."""

    def __init__(self, namespace: str = "AI-Coach",
                 index: Optional[str] = None, url: Optional[str] = None,
                 embed_fn: Optional[Callable] = None, client=None,
                 auto_default: bool = True):
        self.namespace = namespace
        self.index = index or DEFAULT_AICOACH_KB_INDEX
        self._url = url or DEFAULT_ES_URL
        self._embed_fn = embed_fn
        self._auto_default = auto_default
        self._default_attempted = False
        self._client = client
        self._client_attempted = client is not None

    # ─── 지연 자원 ───────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None and not self._client_attempted:
            self._client_attempted = True
            try:
                from elasticsearch import Elasticsearch
                self._client = Elasticsearch(self._url, request_timeout=10)
                logger.info(f"🔎 aicoach-kb client created: {self._url}")
            except Exception as e:
                logger.warning(f"⚠️ aicoach-kb Elasticsearch unavailable ({e})")
                self._client = None
        return self._client

    @property
    def embed_fn(self):
        if (self._embed_fn is None and self._auto_default
                and not self._default_attempted):
            self._default_attempted = True
            from .semantic_index import _load_default_embed_fn
            self._embed_fn = _load_default_embed_fn()
        return self._embed_fn

    def available(self) -> bool:
        """ES 가 살아 있고 임베더가 있는가. 절대 raise 하지 않는다."""
        client = self.client
        if client is None or self.embed_fn is None:
            return False
        try:
            return bool(client.ping())
        except Exception:
            return False

    def _embed_one(self, text: str) -> Optional[List[float]]:
        if self.embed_fn is None:
            return None
        try:
            vector = np.atleast_2d(self.embed_fn([text])).astype(np.float32)[0]
            return [float(x) for x in vector]
        except Exception as e:
            logger.warning(f"⚠️ aicoach-kb query embed failed ({e})")
            return None

    # ─── 검색 (ChunkIndex 계약) ──────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0) -> List[Hit]:
        """aicoach-kb 하이브리드 검색. [(StoredChunk, score)] 내림차순.

        점수는 es_backend 와 같이 +1.0 시프트를 되돌려 cosine 스케일로 내보낸다
        — 융합은 순위(RRF)만 쓰지만, best_evidence 동점 해소가 이 값을 보므로
        나머지 채널과 스케일을 맞춘다. ES/임베더 부재 시 [] 로 degrade.
        """
        if not query or not query.strip():
            return []
        client = self.client
        if client is None:
            return []
        vector = self._embed_one(query)
        if vector is None:
            return []
        try:
            body = build_aicoach_query(query, vector, top_k=top_k)
            response = client.search(index=self.index, **body)
        except Exception as e:
            logger.warning(f"⚠️ aicoach-kb search failed ({e})")
            return []

        hits: List[Hit] = []
        for hit in response.get("hits", {}).get("hits", []):
            score = float(hit.get("_score", 0.0)) - _COSINE_SHIFT
            if score < min_score:
                continue
            hits.append((hit_to_chunk(hit), round(score, 4)))
        return hits


# ─── 네임스페이스 싱글턴 ────────────────────────────────────────────
# 임베더(ko-sroberta) 로드가 무거우므로 프로세스당 하나만 유지한다.

_indices: Dict[str, AicoachChunkIndex] = {}


def get_aicoach_chunk_index(namespace: str = "AI-Coach") -> AicoachChunkIndex:
    if namespace not in _indices:
        _indices[namespace] = AicoachChunkIndex(namespace=namespace)
        logger.info(f"🔎 aicoach-kb chunk index initialized "
                    f"(namespace={namespace}, index={_indices[namespace].index})")
    return _indices[namespace]


def reset_aicoach_chunk_indices() -> None:
    """싱글턴 초기화 — 테스트 격리용."""
    _indices.clear()
