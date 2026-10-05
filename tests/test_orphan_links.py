"""고아 노드 근거 회복 — "청크가 없는 노드"를 원문 인용으로 되돌린다.

**커버리지 검사와 대칭인 반쪽이다.** check_coverage 는 "노드가 없는 청크"를
묻는다(추출이 놓친 개체). 아무도 **"청크가 없는 노드"**를 묻지 않았고, 그 사이
고아 노드가 27개(14%) 쌓였다 — 그중 25개는 이름이 원문에 **그대로 있다**
(`InsuranceTerm:보험료` 는 원문 32곳, `계약일` 11곳).

**왜 이게 검색 문제보다 중요한가**: evidence 채점의 hit@5 천장 0.8125 는 어떤
검색 knob 으로도 움직이지 않았다(15개 조합 스윕). 정답 노드에 근거 청크가 0개면
채널 B 가 끌어올 것이 없고 채점이 **구조적으로 불가능**하다. 천장은 검색이 아니라
연결이 정한다.

**shadow 검사가 이 기능의 필수 부품이다.** 단순 부분문자열 매칭은 잘못된 링크를
만든다 — 실측된 오추출 `Disease:상선암`("갑상선암"에서 앞글자가 잘린 노드)이 원문
10곳에서 "발견"되는데 전부 "갑상선암"의 부분문자열이다. 더 긴 노드 이름이 같은
청크에 있으면 건너뛰고 `shadowed` 로 **보고한다** — 오추출을 조용히 덮는 대신
데이터로 드러낸다.
"""
import pytest

from ontology.core.orphan_links import (find_orphan_candidates,
                                        shadowing_names)


# ─── shadow 검사 (순수 함수) ─────────────────────────────────────────

class TestShadowingNames:
    def test_longer_name_containing_it_shadows(self):
        """'상선암' 은 '갑상선암' 안에 있다 — 같은 청크에 있으면 가려진다."""
        assert shadowing_names("상선암", ["갑상선암", "보험료"],
                               "갑상선암 진단 시") == ["갑상선암"]

    def test_not_shadowed_when_longer_name_absent_from_text(self):
        """더 긴 이름이 **그 청크에 없으면** 가리지 못한다 — 독립 출현이다."""
        assert shadowing_names("상선암", ["갑상선암"], "상선암 이라는 표기") == []

    def test_self_is_not_a_shadow(self):
        assert shadowing_names("보험료", ["보험료"], "보험료 납입") == []

    def test_unrelated_longer_name_does_not_shadow(self):
        """포함 관계가 아니면 길이만으로 가리지 않는다."""
        assert shadowing_names("보험료", ["보험료 납입기간 안내문"],
                               "보험료 납입") == []
        assert shadowing_names("보험료", ["계약자적립액"], "보험료 계약자적립액") == []

    def test_substring_relation_is_required_not_mere_overlap(self):
        assert shadowing_names("암진단비", ["진단비"], "암진단비 지급") == []

    def test_whitespace_is_normalized(self):
        """원문의 개행·공백 때문에 가림이 풀리면 잘못된 링크가 새어든다."""
        assert shadowing_names("상선암", ["갑상선암"], "갑상선암\n  진단") == ["갑상선암"]

    def test_empty_inputs_are_safe(self):
        assert shadowing_names("", ["갑상선암"], "갑상선암") == []
        assert shadowing_names("상선암", [], "갑상선암") == []
        assert shadowing_names("상선암", ["갑상선암"], "") == []


# ─── 후보 수집 ───────────────────────────────────────────────────────

class _Chunk:
    def __init__(self, chunk_id, text, section="", source="약관.pdf",
                 node_ids=None):
        self.chunk_id = chunk_id
        self.text = text
        self.section = section
        self.source = source
        self.node_ids = list(node_ids or [])


def _graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_node("T:보험료", type="T", name="보험료")
    g.add_node("T:계약일", type="T", name="계약일")
    g.add_node("D:갑상선암", type="D", name="갑상선암")
    g.add_node("D:상선암", type="D", name="상선암")        # 오추출
    g.add_node("T:없는말", type="T", name="없는말")         # 원문에 없음
    g.add_node("T:연결됨", type="T", name="연결됨")         # 이미 링크 있음
    return g


def _chunks():
    return [
        _Chunk("c1", "보험료 를 납입한 계약일 부터", section="제1조"),
        _Chunk("c2", "갑상선암 진단 시 보험료 를 면제한다", section="제6조"),
        _Chunk("c3", "연결됨 조문", node_ids=["T:연결됨"]),
    ]


class TestFindOrphanCandidates:
    def test_only_orphan_nodes_are_considered(self):
        """이미 근거가 있는 노드는 후보가 아니다 — 고칠 것이 없다."""
        res = find_orphan_candidates(_graph(), _chunks())
        assert "T:연결됨" not in {c["node_id"] for c in res["candidates"]}
        assert res["orphans_total"] == 5      # 연결됨 제외

    def test_quotable_orphan_gets_every_chunk_that_quotes_it(self):
        """근거는 여럿일 수 있다 — 한 청크만 주면 나머지가 조용히 사라진다."""
        res = find_orphan_candidates(_graph(), _chunks())
        by_id = {c["node_id"]: c for c in res["candidates"]}
        assert by_id["T:보험료"]["chunk_ids"] == ["c1", "c2"]

    def test_unquotable_orphan_is_reported_not_dropped(self):
        """원문에 없는 노드는 **추출이 원문을 넘었다**는 신호다. 후보에서
        빼되 개수는 보고한다 — 조용히 사라지면 그 결함을 못 본다."""
        res = find_orphan_candidates(_graph(), _chunks())
        assert "T:없는말" not in {c["node_id"] for c in res["candidates"]}
        assert "T:없는말" in res["unquotable"]

    def test_shadowed_orphan_is_separated(self):
        """`상선암` 은 '갑상선암' 의 부분문자열로만 나타난다 — 링크하면 오답."""
        res = find_orphan_candidates(_graph(), _chunks())
        assert "D:상선암" not in {c["node_id"] for c in res["candidates"]}
        shadowed = {s["node_id"]: s for s in res["shadowed"]}
        assert shadowed["D:상선암"]["shadowed_by"] == ["갑상선암"]

    def test_genuinely_quotable_node_not_shadowed(self):
        res = find_orphan_candidates(_graph(), _chunks())
        assert "D:갑상선암" in {c["node_id"] for c in res["candidates"]}

    def test_candidate_carries_what_a_reviewer_needs(self):
        res = find_orphan_candidates(_graph(), _chunks())
        cand = next(c for c in res["candidates"] if c["node_id"] == "T:계약일")
        assert cand["name"] == "계약일" and cand["type"] == "T"
        assert cand["sections"] == ["제1조"]      # 어느 조문인지 보여야 판단한다

    def test_limit_caps_candidates_but_totals_stay_true(self):
        """상한을 걸면서 총계를 같이 줄이면 "다 처리했다"로 오해한다."""
        res = find_orphan_candidates(_graph(), _chunks(), limit=1)
        assert len(res["candidates"]) == 1
        assert res["orphans_total"] == 5

    def test_empty_graph_is_not_an_error(self):
        import networkx as nx
        res = find_orphan_candidates(nx.MultiDiGraph(), _chunks())
        assert res["candidates"] == [] and res["orphans_total"] == 0

    def test_nodes_without_name_fall_back_to_id_tail(self):
        """name 이 없는 노드도 회복 대상이다 — id 꼬리가 사실상 이름이다."""
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("T:보험료", type="T")          # name 없음
        res = find_orphan_candidates(g, _chunks())
        assert {c["node_id"] for c in res["candidates"]} == {"T:보험료"}

    def test_deterministic_order(self):
        """검수 화면이 새로고침마다 순서가 바뀌면 일괄 승인이 위험해진다."""
        a = find_orphan_candidates(_graph(), _chunks())
        b = find_orphan_candidates(_graph(), _chunks())
        assert [c["node_id"] for c in a["candidates"]] \
            == [c["node_id"] for c in b["candidates"]]
