"""ES 객체 인덱스 테스트 (축 5, P3).

순수 함수(매핑/쿼리/문서 변환)는 항상, ES 통합(sync/search/facets)은
ONTOLOGY_ES_URL 이 가리키는 ES 가 살아 있을 때만(없으면 skip) — es_backend
통합 테스트와 같은 관례.
"""
import networkx as nx
import pytest

from ontology.core import object_index as oi
from ontology.core.object_index import ObjectIndex

NS = "test_objindex"


def _graph():
    g = nx.MultiDiGraph()
    g.add_node("temple:불국사", type="temple", name="불국사", trust="authoritative",
               definition="신라 사찰", aliases=["Bulguksa"])
    g.add_node("heritage:석굴암", type="heritage", name="석굴암", trust="authoritative",
               aliases=[])
    g.add_node("place:경주", type="place", name="경주", trust="unknown")
    g.add_node("place:서울", type="place", name="서울", trust="unknown")
    return g


# ─── 순수 함수 ────────────────────────────────────────────────────────────
def test_mapping_shape():
    m = oi.build_object_mapping()["mappings"]["properties"]
    assert m["type"]["type"] == "keyword" and m["trust"]["type"] == "keyword"
    assert m["name"]["type"] == "text" and m["name"]["fields"]["kw"]["type"] == "keyword"


def test_search_body_query_vs_matchall():
    with_q = oi.build_search_body("경주", None, None)
    should = with_q["bool"]["must"][0]["bool"]["should"]
    # fuzzy 는 없어야 하고(오매칭 방지), substring wildcard 로 부분일치를 잡는다
    assert not any("fuzziness" in str(cl) for cl in should)
    wc = next(cl for cl in should if "wildcard" in cl)
    assert wc["wildcard"]["name.kw"]["value"] == "*경주*"
    no_q = oi.build_search_body("", "place", "unknown")
    assert "match_all" in no_q["bool"]["must"][0]
    terms = {list(f["term"].keys())[0] for f in no_q["bool"]["filter"]}
    assert terms == {"type", "trust"}


def test_graph_to_docs():
    docs = {d["node_id"]: d for d in oi.graph_to_docs(NS, _graph())}
    assert docs["temple:불국사"]["type"] == "temple"
    assert docs["temple:불국사"]["aliases"] == "Bulguksa"
    assert docs["place:경주"]["trust"] == "unknown"
    assert all(d["namespace"] == NS for d in docs.values())


# ─── ES 통합 (살아있을 때만) ──────────────────────────────────────────────
es_required = pytest.mark.skipif(
    not ObjectIndex(NS).available(), reason="ES 불가(ONTOLOGY_ES_URL) — skip")


@pytest.fixture
def indexed():
    idx = ObjectIndex(NS)
    idx.sync(oi.graph_to_docs(NS, _graph()))
    yield idx
    idx.delete()


@es_required
def test_search_all_and_facets(indexed):
    r = indexed.search(top_k=50)
    assert r["total"] == 4
    assert r["facets"]["type"] == {"place": 2, "temple": 1, "heritage": 1}
    assert r["facets"]["trust"] == {"authoritative": 2, "unknown": 2}


@es_required
def test_search_bm25_relevance(indexed):
    r = indexed.search(q="경주")
    assert r["total"] >= 1
    assert r["items"][0]["node_id"] == "place:경주"   # 최상위 관련도
    assert r["items"][0]["score"] is not None


@es_required
def test_search_filter_facets_reflect_filter(indexed):
    r = indexed.search(node_type="place")
    assert r["total"] == 2
    assert {i["node_id"] for i in r["items"]} == {"place:경주", "place:서울"}
    # 필터가 걸리면 파셋도 필터 결과를 반영
    assert r["facets"]["type"] == {"place": 2}
