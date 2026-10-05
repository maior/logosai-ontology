"""검수 트리아지 결합기 — 순수 함수 계약 + 변이 사살 (C1-a · C2).

설계 원문: docs/review-collaboration-architecture.html §5. 고정하는 계약:
- strong_confirm 은 다섯 조건 전부(confirm ∧ 근거≥1 ∧ 중복 비소속 ∧ 일관성 0
  ∧ 구조단위 아님) — 하나라도 빼먹는 mutant 는 여기서 죽는다.
- strong_reject 는 LLM reject + 사유일 때만. unsure/no_evidence/None 은 전부
  borderline (evidence_count==0 단독으로 reject 를 만들지 않는다 — relink
  실패일 수 있다).
- 관계는 2분류 — 기계 기각 없음 (시그니처 미관측 83%의 다수가 정상 실측).
- 일치율 자는 로그 순서에 기대지 않고(at 정렬), 판정 없는 추천은 n 제외
  (pending ≠ disagree).
"""

import random

from ontology.core.review_triage import (
    recommendation_agreement,
    triage_node,
    triage_relation,
)

CONFIRM = {"verdict": "confirm", "rationale": "원문 명시", "evidence_quote": "q"}
REJECT = {"verdict": "reject", "rationale": "원문 모순", "evidence_quote": "q"}
CLEAN = {"dup_member": False, "consistency_findings": 0,
         "structural_candidate": False}
ITEM = {"node_id": "Term:암진단비", "evidence_count": 2}


# ─── 1. triage_node ──────────────────────────────────────────────────

class TestTriageNode:
    def test_strong_confirm_happy_path(self):
        out = triage_node(ITEM, CONFIRM, CLEAN)
        assert out["band"] == "strong_confirm"
        assert out["reasons"]

    def test_unsure_never_strong_confirm(self):
        """변이 사살: unsure → strong_confirm mutant. no_evidence·None 포함
        전부 borderline 이어야 한다."""
        for v in ("unsure", "no_evidence"):
            out = triage_node(ITEM, {"verdict": v, "rationale": "r"}, CLEAN)
            assert out["band"] == "borderline", v
        assert triage_node(ITEM, None, CLEAN)["band"] == "borderline"

    def test_dup_member_demotes_confirm(self):
        """변이 사살: dup_member 무시 mutant — confirm 추천이 있어도 강등,
        사유에 '병합 판단이 먼저'가 명시된다."""
        out = triage_node(ITEM, CONFIRM, {**CLEAN, "dup_member": True})
        assert out["band"] == "borderline"
        assert any("병합" in r for r in out["reasons"])

    def test_structural_candidate_demotes_confirm(self):
        out = triage_node(ITEM, CONFIRM,
                          {**CLEAN, "structural_candidate": True})
        assert out["band"] == "borderline"
        assert any("재분류" in r for r in out["reasons"])

    def test_consistency_findings_demote_confirm(self):
        out = triage_node(ITEM, CONFIRM,
                          {**CLEAN, "consistency_findings": 3})
        assert out["band"] == "borderline"
        assert any("일관성" in r for r in out["reasons"])

    def test_zero_evidence_demotes_confirm(self):
        out = triage_node({"node_id": "T:x", "evidence_count": 0},
                          CONFIRM, CLEAN)
        assert out["band"] == "borderline"

    def test_reject_with_rationale_is_strong_reject(self):
        assert triage_node(ITEM, REJECT, CLEAN)["band"] == "strong_reject"

    def test_reject_without_rationale_is_borderline(self):
        """이유 없는 reject 로는 인간이 판정할 수 없다 (근거대조와 같은 규율)."""
        out = triage_node(ITEM, {"verdict": "reject", "rationale": ""}, CLEAN)
        assert out["band"] == "borderline"

    def test_zero_evidence_alone_never_strong_reject(self):
        """evidence_count==0 단독으로 reject 를 만들지 않는다 — 고아 27 중
        25 가 빌더의 링크 유실이었다 (2026-07-31 실측)."""
        out = triage_node({"node_id": "T:x", "evidence_count": 0}, None, CLEAN)
        assert out["band"] == "borderline"

    def test_reject_survives_dup_member(self):
        """dup 은 confirm 강등 신호다('정리가 먼저') — '이 개체가 틀렸다'는
        reject 판단을 뒤집을 근거가 아니다."""
        out = triage_node(ITEM, REJECT, {**CLEAN, "dup_member": True})
        assert out["band"] == "strong_reject"

    def test_never_raises_on_garbage(self):
        out = triage_node(None, "쓰레기", 123)   # 전부 잘못된 타입
        assert out["band"] == "borderline"
        out2 = triage_node({"evidence_count": "많음"},
                           {"verdict": ["confirm"]}, {"dup_member": object()})
        assert out2["band"] in ("borderline", "strong_confirm", "strong_reject")


# ─── 2. triage_relation ──────────────────────────────────────────────

REL = {"subject": "Disease:암", "predicate": "covers",
       "object": "Contract:계약", "signature_seen": True,
       "names_in_quote": 2, "previously_rejected": None}


class TestTriageRelation:
    def test_strong_confirm(self):
        assert triage_relation(REL)["band"] == "strong_confirm"

    def test_previously_rejected_demotes(self):
        """변이 사살: previously_rejected 무시 mutant — 기각 이력이 있으면
        어떤 인용이 와도 기계가 다시 올리면 안 된다."""
        out = triage_relation(
            {**REL, "previously_rejected": {"reason": "전이를 보장으로 왜곡"}})
        assert out["band"] == "borderline"
        assert any("기각" in r for r in out["reasons"])

    def test_signature_unseen_demotes(self):
        out = triage_relation({**REL, "signature_seen": False})
        assert out["band"] == "borderline"

    def test_names_in_quote_below_two_demotes(self):
        for n in (0, 1):
            assert triage_relation(
                {**REL, "names_in_quote": n})["band"] == "borderline", n

    def test_no_machine_reject_band(self):
        """관계는 2분류 — 최악의 입력도 borderline 이지 기계 기각이 아니다
        (시그니처 위반 30/36 중 다수가 정상 관계라는 실측 유지)."""
        out = triage_relation({"signature_seen": False, "names_in_quote": 0,
                               "previously_rejected": {"reason": "x"}})
        assert out["band"] == "borderline"
        assert len(out["reasons"]) == 3   # 세 신호 전부 사유로 보인다

    def test_never_raises(self):
        assert triage_relation(None)["band"] == "borderline"
        assert triage_relation({"names_in_quote": "둘"})["band"] == "borderline"


# ─── 3. recommendation_agreement ─────────────────────────────────────

def ev(action, node_id, at, verdict=None, actor="triage",
       predicate=None, target=None):
    after = {}
    if verdict is not None:
        after["verdict"] = verdict
    if predicate is not None:
        after["predicate"] = predicate
        after["target"] = target
    return {"action": action, "node_id": node_id, "at": at,
            "actor": actor, "after": after or None}


class TestAgreement:
    def test_basic_agree_disagree(self):
        events = [
            ev("recommend", "N:1", "2026-01-01T00:00:01", verdict="confirm"),
            ev("confirm", "N:1", "2026-01-01T00:00:02"),
            ev("recommend", "N:2", "2026-01-01T00:00:03", verdict="reject"),
            ev("confirm", "N:2", "2026-01-01T00:00:04"),   # 불일치
        ]
        out = recommendation_agreement(events)
        assert out["overall"] == {"n": 2, "agree": 1, "rate": 0.5}
        by = out["per_actor"]["triage"]["by_verdict"]
        assert by["confirm"] == {"n": 1, "agree": 1, "rate": 1.0}
        assert by["reject"] == {"n": 1, "agree": 0, "rate": 0.0}

    def test_pending_recommend_excluded(self):
        """변이 사살: 판정-없는-recommend 를 disagree 로 세는 mutant.
        pending ≠ disagree — n 에서 제외되고 rate 는 None (0.0 오보고 금지)."""
        out = recommendation_agreement(
            [ev("recommend", "N:1", "2026-01-01T00:00:01", verdict="confirm")])
        assert out["overall"]["n"] == 0
        assert out["overall"]["rate"] is None

    def test_order_shuffle_invariance(self):
        """변이 사살: 입력(로그) 순서 의존 mutant — at 이 구별되는 이벤트는
        어떤 순서로 넘어와도(셔플·최신-먼저) 같은 결과여야 한다."""
        events = [
            ev("recommend", "N:1", "2026-01-01T00:00:01", verdict="confirm"),
            ev("confirm", "N:1", "2026-01-01T00:00:02"),
            ev("recommend", "N:2", "2026-01-01T00:00:03", verdict="reject"),
            ev("reject", "N:2", "2026-01-01T00:00:04"),
            ev("recommend", "N:3", "2026-01-01T00:00:05", verdict="confirm"),
            ev("reject", "N:3", "2026-01-01T00:00:06"),
        ]
        expected = recommendation_agreement(list(events))
        shuffled = list(events)
        random.Random(7).shuffle(shuffled)
        assert recommendation_agreement(shuffled) == expected
        # history() 는 최신-먼저를 준다 — 그 모양 그대로도 옳아야 한다
        assert recommendation_agreement(list(reversed(events))) == expected

    def test_judgment_before_recommend_does_not_pair(self):
        """대조는 '다음' 판정과만 — 추천 이전의 판정은 그 추천의 성적이 아니다."""
        out = recommendation_agreement([
            ev("confirm", "N:1", "2026-01-01T00:00:01"),
            ev("recommend", "N:1", "2026-01-01T00:00:02", verdict="confirm"),
        ])
        assert out["overall"]["n"] == 0

    def test_relation_recommend_pairs_only_with_relation_judgments(self):
        rec = ev("recommend", "A:s", "2026-01-01T00:00:01", verdict="confirm",
                 predicate="covers", target="B:o")
        # 노드 confirm 은 관계 추천과 대조되지 않는다
        out = recommendation_agreement(
            [rec, ev("confirm", "A:s", "2026-01-01T00:00:02")])
        assert out["overall"]["n"] == 0
        # relation_approve 가 오면 confirm 으로 정규화되어 agree
        out2 = recommendation_agreement([
            rec, ev("relation_approve", "A:s", "2026-01-01T00:00:03",
                    predicate="covers", target="B:o")])
        assert out2["overall"] == {"n": 1, "agree": 1, "rate": 1.0}

    def test_relation_reject_disagrees_with_confirm_recommend(self):
        out = recommendation_agreement([
            ev("recommend", "A:s", "2026-01-01T00:00:01", verdict="confirm",
               predicate="covers", target="B:o"),
            ev("relation_reject", "A:s", "2026-01-01T00:00:02",
               predicate="covers", target="B:o"),
        ])
        assert out["overall"] == {"n": 1, "agree": 0, "rate": 0.0}

    def test_relation_triple_matching_squashes_whitespace(self):
        """트리플 비교는 공백 정규화 — _relation_key 와 같은 규칙."""
        out = recommendation_agreement([
            ev("recommend", "A:s", "2026-01-01T00:00:01", verdict="confirm",
               predicate="covers", target="B:o( 상세 )"),
            ev("relation_approve", "A:s", "2026-01-01T00:00:02",
               predicate="covers", target="B:o(  상세 )"),
        ])
        assert out["overall"]["n"] == 1
        assert out["overall"]["agree"] == 1

    def test_verdictless_recommend_excluded(self):
        """예측 없는 추천(llm_skipped 기록 등)은 판정이 따라와도 n 제외 —
        없는 예측의 일치율은 셀 수 없다."""
        out = recommendation_agreement([
            ev("recommend", "N:1", "2026-01-01T00:00:01", verdict=""),
            ev("confirm", "N:1", "2026-01-01T00:00:02"),
        ])
        assert out["overall"]["n"] == 0

    def test_unsure_counts_and_never_agrees(self):
        """unsure 는 n 에 들되 절대 agree 가 못 된다 — 보수적 방향.
        렌즈별 실질 적중률은 by_verdict 로 읽는다."""
        out = recommendation_agreement([
            ev("recommend", "N:1", "2026-01-01T00:00:01", verdict="unsure"),
            ev("confirm", "N:1", "2026-01-01T00:00:02"),
        ])
        assert out["overall"] == {"n": 1, "agree": 0, "rate": 0.0}
        assert out["per_actor"]["triage"]["by_verdict"]["unsure"]["n"] == 1

    def test_multiple_recommends_pair_with_next_judgment(self):
        """한 판정 이전의 추천들은 전부 그 판정과 대조된다 (재검사 허용)."""
        out = recommendation_agreement([
            ev("recommend", "N:1", "2026-01-01T00:00:01", verdict="unsure"),
            ev("recommend", "N:1", "2026-01-01T00:00:02", verdict="confirm"),
            ev("confirm", "N:1", "2026-01-01T00:00:03"),
        ])
        assert out["overall"]["n"] == 2
        assert out["overall"]["agree"] == 1

    def test_empty_and_garbage_inputs(self):
        assert recommendation_agreement([])["overall"]["n"] == 0
        assert recommendation_agreement(None)["overall"]["rate"] is None
        out = recommendation_agreement([{"action": "recommend"}, "쓰레기", 42])
        assert out["overall"]["n"] == 0
