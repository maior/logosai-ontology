"""
축 3 — Elasticsearch 하이브리드 백엔드.

앞선 분석의 가장 뼈아픈 사실: aicoach 는 온톨로지와 ES 를 **두 개의 평행한
검색 시스템**으로 두고 `source` 키워드 하나로만 이었다. 동의어 테이블
(kg/match.py:21-26)은 ES 까지 도달하지 않고, ES aggregation 은 코드베이스에
0건이다. 반대로 Logos 는 그래프와 임베딩을 한 곳에 갖고도 벡터 백엔드
5-tier 중 4개가 스텁이었다.

여기서는 aicoach 가 **골든셋으로 실측 튜닝한** 하이브리드 블렌드를 커널로
가져온다 (recall@5 = 0.944, aicoach rag/search.py:20-22):
    _score = BM25 × 0.1 + (cosine + 1.0)
이 상수는 취향이 아니라 측정값이므로 그대로 이식하고 출처를 남긴다.

쿼리 조립을 **순수 함수**로 뽑은 것은 이 파일의 핵심 설계다 — 살아있는 ES 없이
블렌드 공식·필터·부스트를 전부 검증할 수 있다 (test_vector_backend.py 가
tiering 정책을 순수 함수로 검증하는 것과 같은 이유).

고정하는 계약:
1. 블렌드 공식과 상수는 aicoach 실측값 그대로다.
2. 필터(node_type)는 점수에 섞이지 않는다 — filter 절에 간다.
3. 점수는 cosine 스케일로 정규화되어 나온다 — min_score 계약이 tier 0 과 호환된다.
4. ES 가 없거나 죽어도 raise 하지 않는다.
"""

import numpy as np
import pytest

from ontology.core.es_backend import (
    BM25_WEIGHT,
    ElasticsearchBackend,
    build_hybrid_query,
    build_index_mapping,
)
from ontology.core.vector_backend import TIER_ELASTICSEARCH


def fake_embed(texts):
    import zlib
    dim = 8
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in str(text).lower().split():
            vectors[i, zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    return vectors


# ─── 1. 쿼리 조립 (순수 함수) ────────────────────────────────────────

class TestHybridQueryShape:
    def test_blend_formula_matches_aicoach(self):
        """BM25 × 0.1 + (cosine + 1.0) — 골든셋 실측값 (recall@5 0.944)."""
        body = build_hybrid_query("청약철회", [0.1] * 8, top_k=5)
        script = body["query"]["bool"]["must"][0]["script_score"]["script"]
        assert script["source"] == (
            "_score * params.bm25_w + (cosineSimilarity(params.qv, 'vector') + 1.0)")
        assert script["params"]["bm25_w"] == BM25_WEIGHT

    def test_bm25_weight_is_the_measured_constant(self):
        assert BM25_WEIGHT == 0.1

    def test_query_vector_is_passed_to_script(self):
        qv = [0.5, 0.25] + [0.0] * 6
        body = build_hybrid_query("q", qv, top_k=3)
        script = body["query"]["bool"]["must"][0]["script_score"]["script"]
        assert script["params"]["qv"] == qv

    def test_keyword_side_matches_text(self):
        body = build_hybrid_query("청약철회", [0.0] * 8, top_k=5)
        should = body["query"]["bool"]["must"][0]["script_score"]["query"]["bool"]["should"]
        assert {"match": {"text": "청약철회"}} in should

    def test_top_k_becomes_size(self):
        assert build_hybrid_query("q", [0.0] * 8, top_k=7)["size"] == 7


class TestFiltersDoNotScore:
    """필터는 관련도 블렌드에 섞이면 안 된다 — aicoach 가 tenant/source 를
    filter 절에 두는 것과 같은 이유. must 에 넣으면 타입 매칭이 점수를 흔든다."""

    def test_node_types_go_to_filter(self):
        body = build_hybrid_query("q", [0.0] * 8, top_k=5,
                                  node_types=["agent", "capability"])
        filt = body["query"]["bool"]["filter"]
        assert {"terms": {"node_type": ["agent", "capability"]}} in filt

    def test_no_filter_when_no_node_types(self):
        body = build_hybrid_query("q", [0.0] * 8, top_k=5)
        assert not body["query"]["bool"].get("filter")

    def test_filter_is_not_in_must(self):
        body = build_hybrid_query("q", [0.0] * 8, top_k=5, node_types=["agent"])
        must = body["query"]["bool"]["must"]
        assert len(must) == 1
        assert "script_score" in must[0]


# ─── 2. 인덱스 매핑 ──────────────────────────────────────────────────

class TestIndexMapping:
    def test_dense_vector_config_matches_aicoach(self):
        mapping = build_index_mapping(dim=768)
        vector = mapping["mappings"]["properties"]["vector"]
        assert vector["type"] == "dense_vector"
        assert vector["dims"] == 768
        assert vector["similarity"] == "cosine"
        assert vector["index_options"]["type"] == "int8_hnsw"

    def test_korean_text_uses_cjk_analyzer(self):
        """공유 클러스터에 nori 플러그인이 없다는 aicoach 실측 그대로
        (rag/index.py:1-5). CLAUDE.md 는 nori 를 썼다고 적었지만 코드는
        cjk bigram 이다 — 문서가 아니라 코드를 따른다."""
        mapping = build_index_mapping(dim=768)
        assert mapping["mappings"]["properties"]["text"]["analyzer"] == "cjk"

    def test_node_type_is_keyword_not_text(self):
        mapping = build_index_mapping(dim=768)
        assert mapping["mappings"]["properties"]["node_type"]["type"] == "keyword"

    def test_dim_is_configurable(self):
        assert build_index_mapping(dim=384)["mappings"]["properties"]["vector"]["dims"] == 384


# ─── 3. 가짜 ES 클라이언트로 흐름 검증 ──────────────────────────────

class FakeIndices:
    def __init__(self, store):
        self._store = store
        self.created = []

    def exists(self, index):
        return index in self._store.created_indices

    def create(self, index, **body):
        self._store.created_indices.add(index)
        self.created.append((index, body))


class FakeES:
    """최소한의 ES 더블 — 백엔드가 실제로 부르는 표면만 흉내낸다."""

    def __init__(self, fail=False):
        self.docs = {}
        self.created_indices = set()
        self.indices = FakeIndices(self)
        self.searches = []
        self.fail = fail

    def ping(self):
        if self.fail:
            raise ConnectionError("es down")
        return True

    def index(self, index, id, document, refresh=None):
        if self.fail:
            raise ConnectionError("es down")
        self.docs[id] = document

    def delete(self, index, id, ignore=None, refresh=None):
        self.docs.pop(id, None)

    def count(self, index):
        return {"count": len(self.docs)}

    def search(self, index, **body):
        if self.fail:
            raise ConnectionError("es down")
        self.searches.append(body)
        hits = [
            {"_id": doc_id, "_score": 1.5,
             "_source": {"node_id": doc_id, "node_type": doc.get("node_type", "")}}
            for doc_id, doc in self.docs.items()
        ]
        return {"hits": {"hits": hits[: body.get("size", 10)]}}


@pytest.fixture
def backend():
    return ElasticsearchBackend(client=FakeES(), index="test-kb",
                                embed_fn=fake_embed, dim=8)


class TestBackendFlow:
    def test_tier_name(self, backend):
        assert backend.tier_name == TIER_ELASTICSEARCH

    def test_upsert_indexes_text_and_vector(self, backend):
        backend.upsert("weather_agent", "도시 날씨 예보", "agent")
        doc = backend.client.docs["weather_agent"]
        assert doc["text"] == "도시 날씨 예보"
        assert doc["node_type"] == "agent"
        assert len(doc["vector"]) == 8

    def test_index_is_created_once(self, backend):
        backend.upsert("a", "t", "agent")
        backend.upsert("b", "t", "agent")
        assert len(backend.client.indices.created) == 1

    def test_remove_deletes_doc(self, backend):
        backend.upsert("a", "t", "agent")
        assert backend.remove("a") is True
        assert "a" not in backend.client.docs

    def test_len_counts_docs(self, backend):
        backend.upsert("a", "t", "agent")
        backend.upsert("b", "t", "agent")
        assert len(backend) == 2

    def test_build_from_graph_indexes_nodes(self, backend):
        import networkx as nx
        graph = nx.MultiDiGraph()
        graph.add_node("weather_agent", type="agent", name="weather_agent",
                       description="날씨 예보")
        graph.add_node("shop_agent", type="agent", name="shop_agent",
                       description="쇼핑 검색")
        assert backend.build_from_graph(graph, node_types=None) == 2
        assert set(backend.client.docs) == {"weather_agent", "shop_agent"}

    def test_search_sends_hybrid_body(self, backend):
        backend.upsert("a", "날씨", "agent")
        backend.search("날씨 알려줘", top_k=3)
        body = backend.client.searches[-1]
        assert "script_score" in body["query"]["bool"]["must"][0]
        assert body["size"] == 3


class TestScoreNormalization:
    """ES 하이브리드 원점수는 BM25×0.1 + cosine + 1.0 이라 [0, 2+] 범위다.
    tier 0(cosine, [-1,1])과 스케일이 달라 그대로 내보내면 기존 호출부의
    min_score 계약이 조용히 깨진다 (find_agents_semantic 은 min_score=0.1).
    +1.0 시프트를 되돌려 cosine 스케일로 맞춘다."""

    def test_score_is_shifted_back_to_cosine_scale(self, backend):
        backend.upsert("a", "날씨", "agent")
        hits = backend.search("날씨", top_k=1)
        # FakeES 가 _score=1.5 를 돌려준다 → 1.5 - 1.0 = 0.5
        assert hits[0]["score"] == 0.5

    def test_min_score_filters_on_normalized_scale(self, backend):
        backend.upsert("a", "날씨", "agent")
        assert backend.search("날씨", top_k=1, min_score=0.9) == []
        assert len(backend.search("날씨", top_k=1, min_score=0.1)) == 1

    def test_result_shape_matches_tier0(self, backend):
        backend.upsert("a", "날씨", "agent")
        hit = backend.search("날씨", top_k=1)[0]
        assert set(hit) == {"node_id", "node_type", "score"}


class TestNeverRaises:
    """ES 는 외부 인프라다 — 죽어도 온톨로지 서비스는 살아야 한다."""

    def test_search_on_dead_es_returns_empty(self):
        backend = ElasticsearchBackend(client=FakeES(fail=True), index="i",
                                       embed_fn=fake_embed, dim=8)
        assert backend.search("q", top_k=3) == []

    def test_upsert_on_dead_es_returns_false(self):
        backend = ElasticsearchBackend(client=FakeES(fail=True), index="i",
                                       embed_fn=fake_embed, dim=8)
        assert backend.upsert("a", "t", "agent") is False

    def test_available_reports_dead_es(self):
        assert ElasticsearchBackend(client=FakeES(fail=True), index="i",
                                    embed_fn=fake_embed, dim=8).available() is False
        assert ElasticsearchBackend(client=FakeES(), index="i",
                                    embed_fn=fake_embed, dim=8).available() is True

    def test_search_without_embedder_returns_empty(self):
        backend = ElasticsearchBackend(client=FakeES(), index="i",
                                       embed_fn=None, dim=8, auto_default=False)
        assert backend.search("q", top_k=3) == []

    def test_blank_query_returns_empty(self, backend):
        assert backend.search("   ", top_k=3) == []
