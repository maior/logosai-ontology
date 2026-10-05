"""`sameAs` — 표기가 완전히 다른 동의어를 원문 근거로 잇는다.

**동기(실측)**: 관계 백필의 is_a 제안 11건 중 **3건이 동일시를 is_a 로 왜곡**했다.
원문은 명확히 같다고 말한다 — "‘계약 전 알릴 의무’라 하며, 상법상 ‘고지의무’와
**같습니다**". 그런데 허용 어휘에 `sameAs` 가 없어서 LLM 이 가장 가까운 `is_a` 로
밀어 넣었다. **어휘 부족이 만든 체계적 오류**이고, 술어 하나로 고쳐진다.

**왜 병합이 아니라 술어인가**: `고지의무`(상법)와 `계약 전 알릴 의무`(약관)를
합치면 **문서가 구별한 것을 지운다** — 두 법이 다른 이름을 쓴다는 사실이 사라진다.
그리고 병합은 묘비도 없이 되돌릴 수 없다. `sameAs` 는 비파괴적이고, 검색에서는
별칭과 같은 일을 한다. 중복 병합은 여전히 별도 경로(`merge_nodes`)로 남는다 —
사람이 "이건 정말 한 개체다"라고 판단할 때만.

**그리고 이건 중복 검사가 못 잡는 것을 잡는다**: `graph_health` 의 중복 클러스터는
이름 유사도 기반이라 `고지의무` vs `계약 전 알릴 의무` 처럼 **표기가 완전히 다른**
동의어를 못 본다(실측: 중복 클러스터 4개에 이 쌍이 없다).

**대칭이 이 술어의 핵심 성질이다.** 인접 확장은 `out_edges` 만 보므로
`A -sameAs-> B` 는 A 로 검색할 때만 확장된다. 동의어가 한 방향으로만 통하면
어느 쪽으로 적었는지라는 우연이 검색을 좌우한다.

**전이는 하지 않는다.** A=B, B=C 라고 A=C 로 넓히면 잘못된 sameAs 하나가 클러스터
전체를 오염시킨다. is_a 는 폐포를 쓰지만(계층은 틀려도 국소적) sameAs 는 1-hop 이다.
"""
import pytest

from ontology.core.graph_retrieval import (
    SYNONYM_PREDICATE,
    GraphConditionedRetriever,
)


@pytest.fixture()
def parts(tmp_path):
    import networkx as nx
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import ChunkStore

    g = nx.MultiDiGraph()
    for nid, name in (("T:계약 전 알릴 의무", "계약 전 알릴 의무"),
                      ("T:고지의무", "고지의무"),
                      ("T:제3의용어", "제3의용어"),
                      ("T:무관", "무관")):
        g.add_node(nid, type="T", name=name)
    # 원문이 "같다"고 말한 방향 그대로 적는다 (한 방향만)
    g.add_edge("T:고지의무", "T:계약 전 알릴 의무", predicate=SYNONYM_PREDICATE)
    # 전이 검사용: 계약 전 알릴 의무 = 제3의용어
    g.add_edge("T:계약 전 알릴 의무", "T:제3의용어", predicate=SYNONYM_PREDICATE)
    # 대조군 — 비대칭 술어는 out 방향만 확장되어야 한다(기존 계약)
    g.add_edge("T:무관", "T:고지의무", predicate="hasCondition")

    store = ChunkStore(namespace="synns", path=tmp_path / "c.jsonl")
    store.add(Chunk(text="계약 전 알릴 의무 조문 " + "본문 " * 20,
                    source="약관.pdf", index=0, char_start=0, char_end=80),
              node_ids=["T:계약 전 알릴 의무"])

    class _KG:
        graph = g

        def __init__(self, entry):
            self._entry = entry

        def semantic_search(self, query, top_k=5):
            return [{"node_id": self._entry, "score": 0.9}]

        def get_ancestors(self, node_id, predicate=None):
            return []

        def get_descendants(self, node_id, predicate=None):
            return []

    class _Index:
        def search(self, query, top_k=5):
            return []

    def make(entry):
        return GraphConditionedRetriever(namespace="synns", kg=_KG(entry),
                                        store=store, index=_Index())

    return make, g


class TestSynonymExpansion:
    def test_out_direction_expands(self, parts):
        make, g = parts
        exp = make("T:고지의무").expand("알려야 할 의무가 뭔가?")
        assert "T:계약 전 알릴 의무" in {n["node_id"]
                                        for n in exp.expanded_nodes}

    def test_in_direction_also_expands(self, parts):
        """**대칭이 핵심.** 원문이 어느 쪽으로 적었는지가 검색을 좌우하면 안 된다."""
        make, g = parts
        exp = make("T:계약 전 알릴 의무").expand("알려야 할 의무가 뭔가?")
        assert "T:고지의무" in {n["node_id"] for n in exp.expanded_nodes}

    def test_synonym_name_becomes_an_expansion_term(self, parts):
        """확장어에 들어가야 청크 채널이 그 표기의 조문을 찾는다."""
        make, g = parts
        exp = make("T:계약 전 알릴 의무").expand("알려야 할 의무가 뭔가?")
        assert "고지의무" in exp.terms
        assert "고지의무" in exp.expanded_query

    def test_via_marks_the_synonym_edge(self, parts):
        """왜 걸렸는지 보이지 않으면 순위를 설명할 수 없다 (이 파일의 계약)."""
        make, g = parts
        exp = make("T:계약 전 알릴 의무").expand("알려야 할 의무가 뭔가?")
        via = {n["node_id"]: n["via"] for n in exp.expanded_nodes}
        assert SYNONYM_PREDICATE in via["T:고지의무"]

    def test_not_transitive(self, parts):
        """A=B, B=C 여도 A 확장에 C 를 넣지 않는다 — 잘못된 sameAs 하나가
        클러스터 전체를 오염시키는 것을 막는다. 1-hop 만."""
        make, g = parts
        exp = make("T:고지의무").expand("알려야 할 의무가 뭔가?")
        assert "T:제3의용어" not in {n["node_id"] for n in exp.expanded_nodes}

    def test_asymmetric_predicates_still_out_only(self, parts):
        """기존 계약 유지 — hasCondition 은 역방향으로 확장되지 않는다.
        모든 술어를 대칭으로 만들면 확장이 무의미하게 폭발한다."""
        make, g = parts
        exp = make("T:고지의무").expand("알려야 할 의무가 뭔가?")
        assert "T:무관" not in {n["node_id"] for n in exp.expanded_nodes}

    def test_no_synonym_edges_is_harmless(self, parts):
        make, g = parts
        exp = make("T:무관").expand("무엇인가?")
        assert isinstance(exp.expanded_nodes, list)


class TestPredicateVocabulary:
    def test_same_as_is_offered_to_the_extractor(self):
        """어휘에 없으면 LLM 이 동일시를 is_a 로 왜곡한다 (실측 3/11)."""
        import networkx as nx
        from ontology.core.relation_backfill import allowed_predicates
        g = nx.MultiDiGraph()
        g.add_edge("a", "b", predicate="hasParty")
        preds = allowed_predicates(g)
        assert SYNONYM_PREDICATE in preds and "is_a" in preds
        assert "hasParty" in preds        # 데이터에서 온 것도 유지

    def test_vocabulary_still_comes_from_data(self):
        """하드코딩 어휘를 만들지 않는다 — 코드가 아는 둘만 예외로 더한다."""
        import networkx as nx
        from ontology.core.relation_backfill import allowed_predicates
        preds = set(allowed_predicates(nx.MultiDiGraph()))
        assert preds == {"is_a", SYNONYM_PREDICATE}
