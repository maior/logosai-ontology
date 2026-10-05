"""중복 검수 지원 — 46개 클러스터를 사람이 빠르게 판정할 수 있게.

**왜 필요한가.** health 의 duplicate_clusters 는 id 목록뿐이고 표본으로 잘린다
(46 중 20). 판정에 필요한 것 — 정의·근거 수·출처 문서·생애주기 — 이 없어서
검수자가 클러스터마다 노드를 따로 조회해야 했다. 그리고 커버리지 회복이 중복을
낳는 패턴이 세 번 재현됐고(3→4, 20→31, 31→46), PROJ-A 골든셋 hit@1 −2건의
원인이기도 하다 — 검수가 밀리면 잡음이 지표를 갉는다.

**분류는 신호이지 판정이 아니다** (C73 교훈: "C73(…) 중 중증 갑상선암을 제외한
갑상선암"은 이름이 비슷하지만 **다른 개념** — 기계가 판정하면 안 되는 실물):
  · variant    — 같은 타입 + 정규화명 동일 → 표기 변형. 병합 후보(강).
                 대조 정준화가 이미 같은 개념으로 취급하는 그 기준이다.
  · cross_type — 정규화명 동일·타입 상이 → 타입이 갈렸을 뿐 같은 말일 수도,
                 진짜 다른 것일 수도. 사람 판단.
  · similar    — 포함 등 그 외 → 사람 판단. C73 경고 동반.
"""
import pytest

from ontology.core.dup_review import propose_duplicate_resolutions


class _Chunk:
    def __init__(self, chunk_id, source, node_ids):
        self.chunk_id = chunk_id
        self.source = source
        self.node_ids = list(node_ids)


def _graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_node("S:통합 저장소", type="S", name="통합 저장소", definition="저장소 정의 A")
    g.add_node("S:통합저장소", type="S", name="통합저장소")
    g.add_node("D:개발DB", type="D", name="개발DB", definition="개발용 DB")
    g.add_node("P:개발DB", type="P", name="개발DB", definition="개발 플랫폼?")
    g.add_node("X:예산설명자료", type="X", name="예산설명자료")
    g.add_node("X:공개된 예산설명자료", type="X", name="공개된 예산설명자료")
    g.add_node("A:운영중", type="A", name="운영중", lifecycle="active")
    g.add_node("A:운영 중", type="A", name="운영 중")
    return g


def _chunks():
    return [
        _Chunk("c1", "요구서.txt", ["S:통합 저장소", "D:개발DB"]),
        _Chunk("c2", "제안서.pdf", ["S:통합저장소"]),
        _Chunk("c3", "제안서.pdf", ["S:통합 저장소"]),
    ]


def _clusters():
    return [["S:통합 저장소", "S:통합저장소"],
            ["D:개발DB", "P:개발DB"],
            ["X:예산설명자료", "X:공개된 예산설명자료"],
            ["A:운영중", "A:운영 중"]]


class TestClassification:
    def test_variant_same_type_equal_normalized(self):
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        by = {tuple(sorted(m["node_id"] for m in c["members"])): c
              for c in res["clusters"]}
        cluster = by[("S:통합 저장소", "S:통합저장소")]
        assert cluster["kind"] == "variant"

    def test_cross_type_equal_name(self):
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        by = {tuple(sorted(m["node_id"] for m in c["members"])): c
              for c in res["clusters"]}
        assert by[("D:개발DB", "P:개발DB")]["kind"] == "cross_type"

    def test_similar_containment_carries_warning(self):
        """포함 관계는 다른 개념일 수 있다 — C73 경고가 붙어야 한다."""
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        by = {tuple(sorted(m["node_id"] for m in c["members"])): c
              for c in res["clusters"]}
        cluster = by[("X:공개된 예산설명자료", "X:예산설명자료")]
        assert cluster["kind"] == "similar"
        assert "다른 개념" in cluster["caution"]


class TestJudgmentSignals:
    def test_members_carry_what_a_reviewer_needs(self):
        """정의·근거 수·출처가 없으면 검수자가 노드마다 따로 조회해야 한다."""
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        cluster = next(c for c in res["clusters"] if c["kind"] == "variant"
                       and c["members"][0]["node_id"].startswith("S:"))
        by_id = {m["node_id"]: m for m in cluster["members"]}
        assert by_id["S:통합 저장소"]["evidence_chunks"] == 2
        assert set(by_id["S:통합 저장소"]["sources"]) == {"요구서.txt", "제안서.pdf"}
        assert by_id["S:통합 저장소"]["definition"].startswith("저장소")
        assert by_id["S:통합저장소"]["evidence_chunks"] == 1

    def test_variant_suggests_winner_by_evidence(self):
        """병합 제안의 승자 = 근거가 많은 쪽 — 검수자가 payload 를 그대로 쓸 수
        있어야 한다. 동률은 사전순(결정론)."""
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        cluster = next(c for c in res["clusters"]
                       if c["members"][0]["node_id"].startswith("S:"))
        assert cluster["suggested"]["winner"] == "S:통합 저장소"     # 근거 2 > 1
        assert cluster["suggested"]["losers"] == ["S:통합저장소"]

    def test_non_variant_has_no_merge_suggestion(self):
        """사람 판단 대상에 병합 payload 를 주면 '기계가 권했다'가 된다."""
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        cross = next(c for c in res["clusters"] if c["kind"] == "cross_type")
        assert "suggested" not in cross

    def test_active_lifecycle_is_flagged(self):
        """active 노드는 병합으로 흡수될 수 없다(생애주기 관문) — 미리 보여야
        검수자가 계획을 다 세우고 나서 거부당하지 않는다."""
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        cluster = next(c for c in res["clusters"]
                       if c["members"][0]["node_id"].startswith("A:"))
        by_id = {m["node_id"]: m for m in cluster["members"]}
        assert by_id["A:운영중"]["lifecycle"] == "active"
        # active 가 패자 후보면 suggested 에서 승자로 올리거나 경고
        if "suggested" in cluster:
            assert "A:운영중" not in cluster["suggested"]["losers"]

    def test_deterministic(self):
        a = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        b = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        assert a == b

    def test_counts_by_kind(self):
        """46개를 종류별로 세 줘야 '어디부터 볼지'가 정해진다."""
        res = propose_duplicate_resolutions(_graph(), _chunks(), _clusters())
        assert res["by_kind"] == {"variant": 2, "cross_type": 1, "similar": 1}

    def test_vanished_member_skipped_never_raises(self):
        res = propose_duplicate_resolutions(_graph(), _chunks(),
                                            [["S:통합 저장소", "S:유령"]])
        assert all(m["node_id"] != "S:유령"
                   for c in res["clusters"] for m in c["members"])

    def test_broken_graph_never_raises(self):
        class Boom:
            def nodes(self, data=False):
                raise RuntimeError("down")

        res = propose_duplicate_resolutions(Boom(), _chunks(), _clusters())
        assert res["clusters"] == []
