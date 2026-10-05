"""근거 단위 채점 (target="evidence") — 청크 랭킹을 노드 라벨로 잰다.

**왜 이 자리가 필요한가.** 확산(PPR) 청크 채널을 켰을 때 지표가 0.9375 → 0.8750
으로 떨어졌는데, 그 숫자는 **청크 품질을 잰 것이 아니다**. 경로가 이렇다:

    propagation_channel=True → hits(청크) 순서 변경
                             → retrieve_result_to_nodes 의 3번 소스 재배열
                             → node 채점 지표 이동

청크 채널을 바꿨는데 노드 자로 쟀다. 그런데 청크 자(target="chunk")는 골든셋에
청크 라벨이 없어 0 케이스다 — 즉 **청크 채널은 자기 능력을 보여줄 자가 없었다.**

노드 채점은 청크 채널에 **구조적으로** 불리하기까지 하다: 1위 청크가 정답 노드의
근거인데 그 청크의 node_ids 5개 중 정답이 5번째면 노드 채점은 rank 5 로 센다
(retrieve_result_to_nodes docstring 의 "구조적으로 0" 기록). 청크 위치로 세면
rank 1 이다. 이 차이가 test_top_chunk_credited_regardless_of_node_position 이다.

**설계**: 라벨을 새로 만들지 않는다. 기존 노드 라벨 16건을 청크 위치에 적용해
"몇 번째 청크가 정답 노드의 근거인가"를 센다 (supporting-passage recall).

**편향을 숨기지 않는다**: 라벨이 근거 링크(노드↔청크)에서 오므로 그래프를 아는
채널에 유리하다. 절대 우열이 아니라 **같은 자로 재는 채널 비교**로만 쓴다.
"""
import pytest

from ontology.core.search_qa import (
    GoldenCase,
    evaluate_cases,
    make_chunk_expander,
)


def _node_cases():
    """노드 라벨만 있는 케이스 — 실제 골든셋(청크 라벨 0개)과 같은 모양."""
    return [
        GoldenCase(case_id="a", query="계약자란?", expected_node_id="P:계약자",
                   status="confirmed", tags=["exact"]),
        GoldenCase(case_id="b", query="유방암 보장?", expected_node_id="D:C50",
                   status="confirmed", tags=["semantic"]),
    ]


# 청크 → 그 청크가 근거인 노드들. 실제로는 ChunkStore 에서 온다.
_LINKS = {
    "c1": ["P:계약자", "T:보험료"],
    "c2": ["D:C50", "T:진단확정"],
    "c3": ["T:해지"],
    "c9": [],                       # 근거 없는 청크 (고아)
}


def _expander(chunk_id):
    return set(_LINKS.get(chunk_id) or [])


# ─── expand_fn — 범용 계약 (타깃과 무관) ─────────────────────────────

class TestExpandFn:
    def test_default_is_identity_node_target_unchanged(self):
        """하위호환 관문: expand_fn 을 안 주면 오늘과 **완전히** 같아야 한다.
        이 테스트가 깨지면 기존 모든 측정치가 비교 불가능해진다."""
        cases = _node_cases()
        ranker = {"ch": lambda q, k: ["P:계약자", "D:C50"]}
        assert (evaluate_cases(cases, ranker, k=5)
                == evaluate_cases(cases, ranker, k=5, expand_fn=None))

    def test_expand_fn_matches_via_intersection(self):
        """랭킹 항목은 청크 id 인데 정답은 노드 id — 교집합으로 판정한다."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1"]},
                                k=5, target="evidence", expand_fn=_expander)
        ranks = {r["case_id"]: r["ch_rank"] for r in result["per_case"]}
        assert ranks["a"] == 1        # c1 이 P:계약자 의 근거
        assert ranks["b"] is None     # c1 은 D:C50 과 무관

    def test_empty_expansion_is_not_a_hit(self):
        """근거 링크가 없는 청크는 어떤 노드도 맞히지 못한다."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c9"]},
                                k=5, target="evidence", expand_fn=_expander)
        assert result["channels"]["ch"]["hit@5"] == 0.0

    def test_position_is_first_matching_chunk(self):
        """순위는 **정답을 근거하는 첫 청크의 위치**다."""
        result = evaluate_cases(_node_cases(),
                                {"ch": lambda q, k: ["c3", "c9", "c2"]},
                                k=5, target="evidence", expand_fn=_expander)
        ranks = {r["case_id"]: r["ch_rank"] for r in result["per_case"]}
        assert ranks["b"] == 3        # c2 가 3위
        assert ranks["a"] is None

    def test_top_chunk_credited_regardless_of_node_position(self):
        """**이 작업의 핵심.** 1위 청크가 정답의 근거면 rank 1 이다 —
        그 청크의 node_ids 안에서 정답이 몇 번째든 상관없다.

        노드 채점(chunk_hits_to_nodes)은 node_ids 를 펴서 세므로 같은 상황에
        rank 2 를 준다. 그 구조적 불리함이 청크 채널을 과소평가하고 있었다."""
        case = [GoldenCase(case_id="a", query="q", expected_node_id="T:보험료",
                           status="confirmed")]
        result = evaluate_cases(case, {"ch": lambda q, k: ["c1"]},
                                k=5, target="evidence", expand_fn=_expander)
        # T:보험료 는 c1.node_ids 의 **2번째**인데 청크는 1위다
        assert result["per_case"][0]["ch_rank"] == 1
        assert result["channels"]["ch"]["hit@1"] == 1.0

    def test_k_truncation_applies_to_chunks(self):
        """k 는 청크 개수 상한이다 — 노드로 펴서 세지 않는다."""
        ranking = ["c9", "c9", "c3", "c2"]      # c2 는 4위
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ranking},
                                k=3, target="evidence", expand_fn=_expander)
        assert result["channels"]["ch"]["hit@3"] == 0.0


# ─── target="evidence" — 채점 위치 ───────────────────────────────────

class TestEvidenceTarget:
    def test_requires_expand_fn(self):
        """expand_fn 없이 evidence 를 요구하면 **터진다**.

        폴백하면 청크 id ∈ 노드 id = 항상 공집합 → 전 채널 0.0 이다. 그건
        지금 고치고 있는 바로 그 거짓말(측정 불가를 실패로 보고)이라, 조용히
        0 을 내는 대신 호출자를 세운다."""
        with pytest.raises(ValueError):
            evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1"]},
                           k=5, target="evidence")

    def test_uses_node_labels_so_nothing_is_skipped(self):
        """청크 라벨이 하나도 없어도 16건이 전부 채점된다 — 이게 목적이다."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1"]},
                                k=5, target="evidence", expand_fn=_expander)
        assert result["target"] == "evidence"
        assert result["cases"] == 2
        assert result["skipped"] == 0

    def test_expected_field_shows_node_id(self):
        """정답은 노드다 — 화면에 청크 id 를 보여주면 헷갈린다."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: []},
                                k=5, target="evidence", expand_fn=_expander)
        expected = {r["case_id"]: r["expected"] for r in result["per_case"]}
        assert expected == {"a": "P:계약자", "b": "D:C50"}

    def test_accepted_synonym_nodes_credited(self):
        """동의어 노드(유방암·C50) 중 아무거나 근거하면 맞다."""
        case = [GoldenCase(case_id="a", query="q", expected_node_id="D:유방암",
                           accepted=["D:C50"], status="confirmed")]
        result = evaluate_cases(case, {"ch": lambda q, k: ["c2"]},
                                k=5, target="evidence", expand_fn=_expander)
        assert result["channels"]["ch"]["hit@1"] == 1.0

    def test_node_label_missing_is_skipped(self):
        """노드 라벨이 없는 케이스는 스킵 — 오답으로 세지 않는다."""
        cases = _node_cases() + [GoldenCase(case_id="c", query="q",
                                            expected_node_id="",
                                            expected_chunk_id="c1",
                                            status="confirmed")]
        result = evaluate_cases(cases, {"ch": lambda q, k: ["c1"]},
                                k=5, target="evidence", expand_fn=_expander)
        assert result["cases"] == 2 and result["skipped"] == 1

    def test_drafts_excluded_by_default(self):
        draft = [GoldenCase(case_id="d", query="q", expected_node_id="P:계약자",
                            status="draft")]
        assert evaluate_cases(draft, {"ch": lambda q, k: ["c1"]}, k=5,
                              target="evidence", expand_fn=_expander)["cases"] == 0
        assert evaluate_cases(draft, {"ch": lambda q, k: ["c1"]}, k=5,
                              target="evidence", expand_fn=_expander,
                              include_drafts=True)["cases"] == 1

    def test_tags_decompose(self):
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1", "c2"]},
                                k=5, target="evidence", expand_fn=_expander)
        assert set(result["by_tag"]) == {"exact", "semantic"}
        assert result["by_tag"]["semantic"]["ch"]["hit@1"] == 0.0   # c2 가 2위

    def test_two_channels_differ(self):
        """청크 순서가 갈리면 지표가 갈린다 — 노드 채점에서는 이게 안 보였다."""
        result = evaluate_cases(_node_cases(), {
            "chunk": lambda q, k: ["c3", "c1", "c2"],
            "retrieve": lambda q, k: ["c1", "c2", "c3"],
        }, k=5, target="evidence", expand_fn=_expander)
        assert result["channels"]["chunk"]["hit@1"] == 0.0
        assert result["channels"]["retrieve"]["hit@1"] == 0.5   # a 만 1위


# ─── 0 건 정직 보고 ──────────────────────────────────────────────────

class TestZeroCaseHonesty:
    """0건을 0.0 으로 내면 "측정 안 됨"이 "다 틀림"으로 읽힌다.

    소비자에게 `cases` 를 같이 보라고 요구하는 건 foot-gun 이다 — graph_health
    에서 이미 같은 부류(0/0 → 0.0)를 고쳤다. None 은 정보가 더 많다.
    """

    def test_no_eligible_cases_reports_none_not_zero(self):
        result = evaluate_cases([], {"ch": lambda q, k: ["c1"]}, k=5)
        assert result["cases"] == 0
        assert result["measured"] is False
        m = result["channels"]["ch"]
        assert m["hit@1"] is None and m["hit@5"] is None and m["mrr"] is None

    def test_all_skipped_reports_none(self):
        """실제로 밟은 경로 — 노드만 라벨된 골든셋에 target=chunk."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1"]},
                                k=5, target="chunk")
        assert result["cases"] == 0 and result["skipped"] == 2
        assert result["measured"] is False
        assert result["channels"]["ch"]["hit@1"] is None

    def test_reason_names_what_is_missing(self):
        """왜 못 쟀는지가 응답에 있어야 사람이 다음 행동을 안다."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1"]},
                                k=5, target="chunk")
        assert "reason" in result and result["reason"]

    def test_measured_true_when_cases_exist(self):
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: ["c1"]},
                                k=5, target="evidence", expand_fn=_expander)
        assert result["measured"] is True
        assert result["channels"]["ch"]["hit@1"] == 0.5
        assert "reason" not in result

    def test_zero_case_still_reports_skipped_count(self):
        """None 으로 바꾸면서 스킵 개수를 잃으면 진단 정보가 사라진다."""
        result = evaluate_cases(_node_cases(), {"ch": lambda q, k: []},
                                k=5, target="chunk")
        assert result["skipped"] == 2
        assert result["channels"]["ch"]["cases"] == 0


# ─── make_chunk_expander — ChunkStore 어댑터 ─────────────────────────

class _FakeStore:
    def __init__(self, links):
        self._links = links

    def get(self, chunk_id):
        node_ids = self._links.get(chunk_id)
        if node_ids is None:
            return None
        return type("C", (), {"node_ids": list(node_ids)})()


class TestMakeChunkExpander:
    def test_returns_linked_nodes(self):
        expand = make_chunk_expander(_FakeStore(_LINKS))
        assert expand("c1") == {"P:계약자", "T:보험료"}

    def test_unknown_chunk_is_empty_not_error(self):
        """청크가 지워졌는데 색인에 남아 있는 것은 **정상**이다 —
        측정 전체가 죽으면 안 된다."""
        expand = make_chunk_expander(_FakeStore(_LINKS))
        assert expand("nope") == set()

    def test_orphan_chunk_is_empty(self):
        expand = make_chunk_expander(_FakeStore(_LINKS))
        assert expand("c9") == set()

    def test_broken_store_degrades_to_empty(self):
        class Boom:
            def get(self, chunk_id):
                raise RuntimeError("store down")

        assert make_chunk_expander(Boom())("c1") == set()

    def test_none_store_degrades_to_empty(self):
        assert make_chunk_expander(None)("c1") == set()
