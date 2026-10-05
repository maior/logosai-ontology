"""문서 뷰 — 문서를 **일급으로 조회·대조**하되, KG 노드로 만들지 않는다.

**왜 노드가 아닌가 (측정된 위험).** 허브가 순위를 오염시키는 것은 이 저장소에서
실측됐다 — PPR 원 질량 정렬에서 총칙 조문(허브 청크)이 정답을 밀어내 lift 교정이
필요했다. Document 를 노드로 만들어 청크 수백 개와 링크하면 **슈퍼허브**가 되어
채널 B(노드에 달린 청크)와 확산을 오염시킨다. "일급"의 실질 — 주소·조회·필터·대조 —
은 노드 없이 성립한다: 문서는 이미 `chunk.source` 축으로 데이터에 존재한다.

**대조가 이 모듈의 존재 이유다.** PROJ-A 용례: "제안요청서가 요구한 것 중 제안서가
빠뜨린 것". 개체의 문서 귀속 = 그 문서의 청크에 근거 링크가 있는가 — 결정적이고
LLM 0콜이다.
"""
import pytest

from ontology.core.document_view import compare_documents, document_views


class _Chunk:
    def __init__(self, chunk_id, source, node_ids=(), section="", text="본문",
                 char_start=0, char_end=10):
        self.chunk_id = chunk_id
        self.source = source
        self.node_ids = list(node_ids)
        self.section = section
        self.text = text
        self.char_start = char_start
        self.char_end = char_end


def _graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    for nid, name in (("T:공통", "공통"), ("T:요구만", "요구만"),
                      ("T:응답만", "응답만"), ("T:고아", "고아")):
        g.add_node(nid, type="T", name=name)
    return g


def _chunks():
    return [
        _Chunk("r1", "요구서.txt", ["T:요구만", "T:공통"], section="1장"),
        _Chunk("r2", "요구서.txt", ["T:공통"], section="2장"),
        _Chunk("r3", "요구서.txt", [], section="2장"),          # 미연결
        _Chunk("p1", "제안서.pdf", ["T:응답만", "T:공통"], section="개요"),
    ]


class TestDocumentViews:
    def test_groups_by_source_with_stats(self):
        views = document_views(_graph(), _chunks())
        by = {v["source"]: v for v in views}
        assert by["요구서.txt"]["chunks"] == 3
        assert by["요구서.txt"]["linked_chunks"] == 2
        assert by["요구서.txt"]["coverage"] == pytest.approx(2 / 3)
        assert by["요구서.txt"]["nodes"] == 2          # 요구만·공통
        assert by["제안서.pdf"]["nodes"] == 2          # 응답만·공통

    def test_sections_counted_not_dumped(self):
        """섹션 전체를 나열하면 문서 목록 응답이 원문 크기로 불어난다."""
        views = document_views(_graph(), _chunks())
        by = {v["source"]: v for v in views}
        assert by["요구서.txt"]["sections"] == 2       # 1장·2장

    def test_deterministic_order(self):
        assert [v["source"] for v in document_views(_graph(), _chunks())] \
            == sorted({"요구서.txt", "제안서.pdf"})

    def test_dangling_node_refs_ignored(self):
        """그래프에서 지워진 노드를 가리키는 링크는 세지 않는다 —
        세면 nodes 수가 유령을 포함해 거짓이 된다."""
        chunks = [_Chunk("c", "문서.txt", ["T:공통", "T:유령"])]
        views = document_views(_graph(), chunks)
        assert views[0]["nodes"] == 1

    def test_empty_is_safe(self):
        assert document_views(_graph(), []) == []

    def test_never_raises(self):
        class Boom:
            def nodes(self, data=False):
                raise RuntimeError("down")
        assert document_views(Boom(), _chunks()) == []


class TestCompareDocuments:
    def test_partitions_entities(self):
        """**이 함수의 존재 이유** — A에만 근거가 있는 개체 / B에만 / 공통."""
        res = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        assert [e["node_id"] for e in res["only_a"]] == ["T:요구만"]
        assert [e["node_id"] for e in res["only_b"]] == ["T:응답만"]
        assert [e["node_id"] for e in res["shared"]] == ["T:공통"]

    def test_orphan_nodes_are_not_attributed(self):
        """어느 문서에도 근거가 없는 노드는 대조 대상이 아니다 — 섞으면
        '빠뜨렸다'가 '원래 어디에도 없었다'와 구별되지 않는다."""
        res = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        all_ids = {e["node_id"] for part in ("only_a", "only_b", "shared")
                   for e in res[part]}
        assert "T:고아" not in all_ids

    def test_carries_evidence_for_review(self):
        """검수자가 판단하려면 근거 위치(청크·섹션)가 보여야 한다."""
        res = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        gap = res["only_a"][0]
        assert gap["name"] == "요구만"
        assert gap["evidence_a"][0]["chunk_id"] == "r1"
        assert gap["evidence_a"][0]["section"] == "1장"

    def test_unknown_source_is_loud(self):
        """오타 파일명이 빈 결과를 내면 '전부 커버됨'으로 오독된다."""
        res = compare_documents(_graph(), _chunks(), "없는파일.txt", "제안서.pdf")
        assert res["error"] == "source_not_found"
        assert "없는파일.txt" in res["detail"]

    def test_same_source_is_loud(self):
        res = compare_documents(_graph(), _chunks(), "요구서.txt", "요구서.txt")
        assert res["error"] == "invalid"

    def test_deterministic_order_within_parts(self):
        a = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        b = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        assert a == b

    def test_totals_reported(self):
        """개수 없이 목록만 주면 상한 잘림을 '전부'로 오해한다 (조용한 절단 금지)."""
        res = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        assert res["only_a_total"] == 1
        assert res["shared_total"] == 1


class TestCanonicalizedCompare:
    """대조는 **개체 정준화** 위에서 한다 — 같은 개념이 문서마다 다른 표기로
    추출되면 identity 대조는 그걸 전부 "차이"로 오보고한다.

    **실측이 요구했다.** PROJ-A 요구서↔제안서의 identity 공통은 6인데, 표기
    변형("통합저장소"↔"통합 저장소", "데이터 플랫폼 구축"↔"데이터플랫폼 구축")을
    묶으면 12다 — 공통의 절반이 "같은 개념, 다른 표기"로 차이에 잘못 계상되고
    있었다. LLM 추출은 문서마다 띄어쓰기가 흔들린다(비결정성의 알려진 모양).

    묶는 규칙 둘 — 둘 다 결정적이고 사람이 검증한 신호다:
      · **(타입, 정규화명) 동일** — graph_health.normalize_name 재사용
        (규칙을 두 벌 두면 '같다'의 정의가 갈라진다). 타입이 다르면 묶지 않는다.
      · **sameAs 엣지** — 사람이 검수해 넣은 동일시 선언.
    포함 관계("갑상선암" ⊂ "중증 갑상선암")는 **묶지 않는다** — 동등만.

    묶었으면 **members 로 드러낸다** — 조용히 합치면 잘못된 묶음을 볼 수 없다.
    """

    def _bi_graph(self):
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("S:데이터 플랫폼 구축", type="S", name="데이터 플랫폼 구축")
        g.add_node("S:데이터플랫폼 구축", type="S", name="데이터플랫폼 구축")
        g.add_node("T:고지의무", type="T", name="고지의무")
        g.add_node("T:계약 전 알릴 의무", type="T", name="계약 전 알릴 의무")
        g.add_edge("T:고지의무", "T:계약 전 알릴 의무", predicate="sameAs")
        g.add_node("D:갑상선암", type="D", name="갑상선암")
        g.add_node("D:중증 갑상선암", type="D", name="중증 갑상선암")
        g.add_node("X:플랫폼", type="X", name="플랫폼")
        g.add_node("Y:플랫폼", type="Y", name="플랫폼")
        return g

    def _bi_chunks(self):
        return [
            _Chunk("a1", "A.txt", ["S:데이터 플랫폼 구축", "T:고지의무",
                                   "D:갑상선암", "X:플랫폼"]),
            _Chunk("b1", "B.txt", ["S:데이터플랫폼 구축", "T:계약 전 알릴 의무",
                                   "D:중증 갑상선암", "Y:플랫폼"]),
        ]

    def test_whitespace_variants_group_as_shared(self):
        """**이 클래스의 존재 이유** — 표기 변형이 '차이'로 계상되지 않는다."""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        shared_names = {e["name"] for e in res["shared"]}
        assert any("플랫폼 구축" in n for n in shared_names)

    def test_sameas_pair_groups_as_shared(self):
        """사람이 검수해 넣은 동일시 선언은 대조에서도 같은 개념이다."""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        all_shared = {m for e in res["shared"]
                      for m in [e["node_id"], *e.get("members", [])]}
        assert "T:고지의무" in all_shared
        assert "T:계약 전 알릴 의무" in all_shared

    def test_containment_does_not_group(self):
        """'갑상선암' ⊂ '중증 갑상선암' 은 다른 개념이다 — 동등만 묶는다."""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        only_a = {e["node_id"] for e in res["only_a"]}
        only_b = {e["node_id"] for e in res["only_b"]}
        assert "D:갑상선암" in only_a and "D:중증 갑상선암" in only_b

    def test_same_name_different_type_does_not_group(self):
        """타입이 다르면 같은 이름이라도 묶지 않는다 — 자동 묶음은 보수적으로.
        (타입 무시 후보 발굴은 graph_health 중복 검사의 몫이다.)"""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        only_a = {e["node_id"] for e in res["only_a"]}
        only_b = {e["node_id"] for e in res["only_b"]}
        assert "X:플랫폼" in only_a and "Y:플랫폼" in only_b

    def test_grouped_entry_reveals_members(self):
        """조용히 합치면 잘못된 묶음을 볼 수 없다 — members 필수."""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        grouped = [e for e in res["shared"] if e.get("members")]
        assert grouped, "묶인 항목에 members 가 없다"
        for e in grouped:
            assert e["node_id"] not in e["members"]     # 대표는 중복 기재 금지

    def test_ungrouped_entry_has_no_members_key(self):
        """단독 항목에 빈 members 를 붙이면 응답만 불어난다."""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        solo = [e for e in res["only_a"] if e["node_id"] == "D:갑상선암"]
        assert solo and "members" not in solo[0]

    def test_group_evidence_is_union(self):
        """묶인 개념의 근거는 구성원 근거의 합집합 — 빠뜨리면 검수가 반쪽이 된다."""
        res = compare_documents(self._bi_graph(), self._bi_chunks(),
                                "A.txt", "B.txt")
        entry = next(e for e in res["shared"] if "플랫폼 구축" in e["name"])
        assert {ev["chunk_id"] for ev in entry["evidence_a"]} == {"a1"}
        assert {ev["chunk_id"] for ev in entry["evidence_b"]} == {"b1"}

    def test_identity_compare_unaffected_when_no_variants(self):
        """변형이 없으면 종전 결과와 완전히 같아야 한다 — 하위호환 관문."""
        res = compare_documents(_graph(), _chunks(), "요구서.txt", "제안서.pdf")
        assert [e["node_id"] for e in res["only_a"]] == ["T:요구만"]
        assert [e["node_id"] for e in res["shared"]] == ["T:공통"]


class TestCoverageMap:
    """coverage_map — 문서를 원문 순서대로 편 청크 스트립의 재료."""

    def _map(self, **kw):
        from ontology.core.document_view import coverage_map
        return coverage_map(_graph(), _chunks(), **kw)

    def test_orders_by_char_start_not_insertion(self):
        from ontology.core.document_view import coverage_map
        chunks = [
            _Chunk("b", "문서.txt", [], char_start=100, char_end=110),
            _Chunk("a", "문서.txt", ["T:공통"], char_start=0, char_end=10),
        ]
        doc = coverage_map(_graph(), chunks)["documents"][0]
        assert [c["chunk_id"] for c in doc["chunks"]] == ["a", "b"]
        assert [c["order"] for c in doc["chunks"]] == [1, 2]
        assert doc["ordered"] is True

    def test_missing_offsets_fall_back_and_declare(self):
        # 오프셋 없는 청크가 섞이면 순서를 신뢰할 수 없다 — ordered=False 로 알린다.
        from ontology.core.document_view import coverage_map
        chunks = [_Chunk("a", "문서.txt", [], char_start=None),
                  _Chunk("b", "문서.txt", [])]
        doc = coverage_map(_graph(), chunks)["documents"][0]
        assert doc["ordered"] is False
        assert [c["chunk_id"] for c in doc["chunks"]] == ["a", "b"]  # 적재 순서 유지

    def test_links_exclude_dangling(self):
        from ontology.core.document_view import coverage_map
        chunks = [_Chunk("c", "문서.txt", ["T:공통", "T:유령"])]
        row = coverage_map(_graph(), chunks)["documents"][0]["chunks"][0]
        assert row["links"] == 1 and row["node_ids"] == ["T:공통"]

    def test_coverage_and_counts(self):
        doc = [d for d in self._map()["documents"] if d["source"] == "요구서.txt"][0]
        assert doc["total"] == 3 and doc["linked"] == 2
        assert abs(doc["coverage"] - 2 / 3) < 1e-9

    def test_text_head_clamped(self):
        from ontology.core.document_view import coverage_map
        chunks = [_Chunk("c", "문서.txt", [], text="가" * 500)]
        row = coverage_map(_graph(), chunks, text_head=7)["documents"][0]["chunks"][0]
        assert row["text_head"] == "가" * 7

    def test_source_filter(self):
        docs = self._map(source="제안서.pdf")["documents"]
        assert [d["source"] for d in docs] == ["제안서.pdf"]

    def test_unknown_source_is_loud(self):
        # 오타 파일명이 빈 문서 목록을 내면 "청크 0 = 전부 커버됨"으로 오독된다.
        r = self._map(source="없는문서.txt")
        assert r["error"] == "source_not_found"
        assert "요구서.txt" in r["detail"]

    def test_never_raises(self):
        from ontology.core.document_view import coverage_map
        class Boom:
            def nodes(self):
                raise RuntimeError("폭발")
        assert coverage_map(Boom(), _chunks()) == {"documents": []}
