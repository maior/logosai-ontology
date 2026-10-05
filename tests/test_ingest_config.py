"""인제스트 미리보기 설정(벡터화 전 교감) — 순수 함수 검증.

벡터화는 준-비가역·유상 작업이므로, 수집 전에 '무엇으로 벡터화되는가'
(임베딩 모델·청크·생성될 ES 인덱스·키워드검색 가능 여부)를 사용자에게
보여줘야 한다. build_ingest_config 는 그 payload 를 만드는 순수 함수.
"""

from ontology.core.ingest_config import build_ingest_config
from ontology.core.semantic_index import DEFAULT_EMBEDDING_MODEL


def test_reports_embedding_model_and_chunk_defaults():
    c = build_ingest_config(es_available=True, embedder_available=True,
                            namespace="ins_cancer_demo")
    assert c["embedding_model"] == DEFAULT_EMBEDDING_MODEL
    assert c["chunk_size"] == 800 and c["overlap"] == 120
    assert c["segment_mode"] == "auto"
    assert "exhaustive" in c["extraction_modes"] and "topic" in c["extraction_modes"]


def test_index_names_parameterized_by_namespace():
    c = build_ingest_config(es_available=True, embedder_available=True,
                            namespace="ins_cancer_demo")
    assert c["object_index"] == "ontology-obj-ins_cancer_demo"
    assert c["chunk_index"] == "chunks_ins_cancer_demo"


def test_namespace_placeholder_when_absent():
    # ns 미정(감식 시점)엔 패턴으로 — 프론트가 사용자가 입력한 ns 로 치환
    c = build_ingest_config(es_available=True, embedder_available=True)
    assert "{namespace}" in c["object_index"]
    assert "{namespace}" in c["chunk_index"]


def test_keyword_search_reflects_es_availability():
    on = build_ingest_config(es_available=True, embedder_available=True, namespace="x")
    off = build_ingest_config(es_available=False, embedder_available=True, namespace="x")
    assert on["keyword_search"] is True and on["es_available"] is True
    # ES 없으면 키워드(BM25) 하이브리드 불가 → 의미검색만
    assert off["keyword_search"] is False and off["es_available"] is False


def test_embedder_unavailable_flagged():
    c = build_ingest_config(es_available=True, embedder_available=False, namespace="x")
    assert c["embedder_available"] is False   # 벡터검색 불가(폴백 substring) 경고용


def test_custom_chunking_passthrough():
    c = build_ingest_config(es_available=True, embedder_available=True, namespace="x",
                            chunk_size=500, overlap=80, segment_mode="heading")
    assert c["chunk_size"] == 500 and c["overlap"] == 80
    assert c["segment_mode"] == "heading"
