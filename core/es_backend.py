"""
Elasticsearch 하이브리드 백엔드 — BM25(키워드) + dense_vector(의미) 결합.

축 3. 임베딩만으로는 고유명사·조문번호·코드처럼 **문자 그대로 맞아야 하는**
질의를 놓치고, BM25 만으로는 표현이 다른 같은 뜻을 놓친다. 둘을 한 질의에서
섞는 것이 하이브리드다.

블렌드 상수는 aicoach 가 KII 골든셋으로 **실측 튜닝한 값**을 그대로 이식했다
(aicoach backend/app/rag/search.py:20-22, recall@5 = 0.944, b∈[0.05,0.2] ·
A∈[1.5,2.5] 구간에서 plateau):

    _score = BM25 × 0.1 + (cosineSimilarity + 1.0)

- `+1.0` 은 cosine 을 [0,2] 로 올려 음수 점수를 없앤다 (ES script_score 는
  음수를 허용하지 않는다).
- `× 0.1` 은 BM25 원점수를 그 [0,2] 범위에 맞춰 눌러준다.
이 숫자들은 취향이 아니라 측정값이다. 바꾸려면 골든셋을 다시 돌려야 한다.

설계:
- 쿼리 조립은 **순수 함수**(build_hybrid_query / build_index_mapping) — 살아있는
  ES 없이 공식·필터·부스트를 전부 테스트할 수 있다. vector_backend 의 tiering
  정책을 순수 함수로 검증하는 것과 같은 이유다.
- 점수는 cosine 스케일로 되돌려 내보낸다. 안 그러면 tier 0 과 스케일이 달라
  기존 호출부의 min_score 계약이 조용히 깨진다.
- ES 는 외부 인프라다. 죽어도 raise 하지 않는다 — 온톨로지 서비스가 ES 때문에
  같이 죽는 것은 본말전도다.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .semantic_index import compose_node_text
from .vector_backend import TIER_ELASTICSEARCH, VectorBackend

# ─── 실측 튜닝 상수 (aicoach 골든셋) ────────────────────────────────
# 출처: aicoach backend/app/rag/search.py:20-22
#   "tuned on KII grounding golden set, recall@5 plateau 0.944
#    across b∈[0.05,0.2], A∈[1.5,2.5]"
BM25_WEIGHT = 0.1

_SCRIPT_SOURCE = (
    "_score * params.bm25_w + (cosineSimilarity(params.qv, 'vector') + 1.0)")

# cosine 을 [0,2] 로 올리려고 더한 값 — 결과를 내보낼 때 되돌린다
_COSINE_SHIFT = 1.0

DEFAULT_INDEX_PREFIX = os.environ.get("ONTOLOGY_ES_INDEX_PREFIX", "ontology")
DEFAULT_ES_URL = os.environ.get("ONTOLOGY_ES_URL", "http://localhost:9200")


def build_index_mapping(dim: int) -> Dict[str, Any]:
    """인덱스 매핑. aicoach rag/index.py:16-38 의 이식.

    한국어 BM25 에 `cjk` analyzer(bigram)를 쓰는 것은 **nori 플러그인이 없는
    공유 클러스터**를 전제한 aicoach 의 실측 선택이다 (rag/index.py:1-5).
    aicoach CLAUDE.md 는 nori 를 넣었다고 적었지만 코드는 cjk 다 — 문서가
    아니라 코드를 따랐다.
    """
    return {
        "mappings": {
            "properties": {
                "node_id": {"type": "keyword"},
                "node_type": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "cjk"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "int8_hnsw", "m": 16,
                                      "ef_construction": 100},
                },
            }
        }
    }


def build_hybrid_query(query: str, query_vector, top_k: int = 5,
                       node_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """하이브리드 검색 본문 조립 — 순수 함수 (I/O·임베더 없음).

    node_types 는 filter 절로 간다: 필터는 관련도에 섞이면 안 된다. must 에
    넣으면 타입 매칭 자체가 점수를 흔들어 블렌드 튜닝이 무의미해진다
    (aicoach 가 tenant/source 를 filter 에 두는 것과 같은 이유).
    """
    qv = list(query_vector)
    bool_query: Dict[str, Any] = {
        "must": [
            {
                "script_score": {
                    "query": {"bool": {"should": [{"match": {"text": query}}]}},
                    "script": {"source": _SCRIPT_SOURCE,
                               "params": {"qv": qv, "bm25_w": BM25_WEIGHT}},
                }
            }
        ]
    }
    if node_types:
        bool_query["filter"] = [{"terms": {"node_type": list(node_types)}}]

    return {"size": top_k, "query": {"bool": bool_query},
            "_source": ["node_id", "node_type"]}


class ElasticsearchBackend(VectorBackend):
    """ES 하이브리드 인덱스. VectorBackend 계약을 그대로 구현한다.

    계약이 (id, text, type) 이라 노드에도 청크에도 그대로 쓸 수 있다 —
    chunk_index.py 가 이 성질을 이용한다.
    """

    def __init__(self, client=None, index: Optional[str] = None,
                 embed_fn=None, dim: int = 768, auto_default: bool = True,
                 namespace: str = "default", url: Optional[str] = None):
        # ES 인덱스명은 소문자만 허용 — 대문자 네임스페이스(AI-Coach) 대응.
        self.index = (index or f"{DEFAULT_INDEX_PREFIX}-{namespace}").lower()
        self.dim = dim
        self._embed_fn = embed_fn
        self._auto_default = auto_default
        self._default_attempted = False
        self._client = client
        self._url = url or DEFAULT_ES_URL
        self._client_attempted = client is not None
        self._index_ready = False

    @property
    def tier_name(self) -> str:
        return TIER_ELASTICSEARCH

    # ─── 지연 자원 ───────────────────────────────────────────────────

    @property
    def client(self):
        """ES 클라이언트 — 첫 사용 때 연결한다. 실패하면 None."""
        if self._client is None and not self._client_attempted:
            self._client_attempted = True
            try:
                from elasticsearch import Elasticsearch
                self._client = Elasticsearch(self._url)
                logger.info(f"🔎 Elasticsearch client created: {self._url}")
            except Exception as e:
                logger.warning(f"⚠️ Elasticsearch unavailable ({e})")
                self._client = None
        return self._client

    @property
    def embed_fn(self):
        if self._embed_fn is None and self._auto_default and not self._default_attempted:
            self._default_attempted = True
            from .semantic_index import _load_default_embed_fn
            self._embed_fn = _load_default_embed_fn()
        return self._embed_fn

    def available(self) -> bool:
        """ES 가 살아 있는가. 절대 raise 하지 않는다."""
        client = self.client
        if client is None:
            return False
        try:
            return bool(client.ping())
        except Exception:
            return False

    def _ensure_index(self) -> bool:
        if self._index_ready:
            return True
        client = self.client
        if client is None:
            return False
        try:
            if not client.indices.exists(index=self.index):
                client.indices.create(index=self.index,
                                      **build_index_mapping(self.dim))
                logger.info(f"🔎 Elasticsearch index created: {self.index}")
            self._index_ready = True
            return True
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch index setup failed ({e})")
            return False

    def _embed_one(self, text: str) -> Optional[List[float]]:
        if self.embed_fn is None:
            return None
        try:
            vector = np.atleast_2d(self.embed_fn([text])).astype(np.float32)[0]
            return [float(x) for x in vector]
        except Exception as e:
            logger.warning(f"⚠️ Embedding failed ({e})")
            return None

    # ─── VectorBackend 계약 ──────────────────────────────────────────

    def upsert(self, node_id: str, text: str, node_type: str) -> bool:
        if not self._ensure_index():
            return False
        vector = self._embed_one(text)
        if vector is None:
            return False
        try:
            self.client.index(index=self.index, id=node_id,
                              document={"node_id": node_id, "node_type": node_type,
                                        "text": text, "vector": vector},
                              refresh=True)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch upsert failed ({node_id}): {e}")
            return False

    def remove(self, node_id: str) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            client.delete(index=self.index, id=node_id, ignore=[404], refresh=True)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch delete failed ({node_id}): {e}")
            return False

    def build_from_graph(self, graph, node_types=...) -> int:
        """⚠️ **잔존 결함**: 이 tier 는 사라진 노드를 정리하지 않는다.

        tier 0/1 은 `prune_to_graph` 로 그래프에서 없어진 노드의 벡터를 걷어내
        검색이 유령을 돌려주지 않게 한다(SemanticIndex 참고). 여기서 같은 일을
        하려면 live id 집합을 뺀 delete_by_query 가 필요한데, ES 가 없는 환경에서는
        검증할 수 없어 **테스트 없는 삭제 쿼리를 넣지 않았다** — 잘못된 delete
        범위는 인덱스를 통째로 날린다.

        ES tier 는 명시 선택 전용(ONTOLOGY_VECTOR_BACKEND=elasticsearch)이므로
        기본 경로는 영향받지 않는다. ES 를 쓰는 환경에서 노드를 지우면
        재색인(인덱스 재생성)이 필요하다.
        """
        from .semantic_index import DEFAULT_NODE_TYPES
        if node_types is ...:
            node_types = DEFAULT_NODE_TYPES

        indexed = 0
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "")
            if node_types is not None and node_type not in node_types:
                continue
            if self.upsert(node_id, compose_node_text(node_id, attrs), node_type):
                indexed += 1
        if indexed:
            logger.info(f"🔎 Elasticsearch index built: {indexed} nodes "
                        f"→ {self.index}")
        return indexed

    def search(self, query: str, top_k: int = 5,
               node_types: Optional[List[str]] = None,
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """하이브리드 검색. 점수는 cosine 스케일로 정규화해 돌려준다.

        ES 원점수는 BM25×0.1 + cosine + 1.0 이라 [0, 2+] 범위다. 그대로
        내보내면 tier 0(cosine, [-1,1])과 스케일이 달라 호출부의 min_score
        계약이 조용히 깨진다 (find_agents_semantic 은 min_score=0.1 을 쓴다).
        +1.0 시프트를 되돌리면 `cosine + 0.1×BM25` 가 되어 tier 0 과 같은
        의미의 임계값을 쓸 수 있다 — 키워드가 겹치면 그만큼 가산될 뿐이다.
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
            body = build_hybrid_query(query, vector, top_k=top_k,
                                      node_types=node_types)
            response = client.search(index=self.index, **body)
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch search failed ({e})")
            return []

        results: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            score = float(hit.get("_score", 0.0)) - _COSINE_SHIFT
            if score < min_score:
                continue
            source = hit.get("_source", {})
            results.append({"node_id": source.get("node_id", hit.get("_id", "")),
                            "node_type": source.get("node_type", ""),
                            "score": round(score, 4)})
        return results

    def __len__(self) -> int:
        client = self.client
        if client is None:
            return 0
        try:
            return int(client.count(index=self.index)["count"])
        except Exception:
            return 0
