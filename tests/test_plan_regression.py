"""계획 회귀 하네스의 판정 로직 — 플래너 이전(P2) 전후로 계획이 같은가.

LLM 을 부르지 않는 순수 부분만 시험한다. 실제 실행은 scripts/plan_regression.py.
"""
import pytest

from ontology.orchestrator.models import AgentTask, ExecutionPlan, ExecutionStage
from ontology.orchestrator.plan_regression import (
    ConfigMismatch, compare, plan_shape, registry_fingerprint, summarize,
)


def _plan(*stages, gap=False):
    plan = ExecutionPlan(query="q", workflow_strategy="hybrid", stages=[
        ExecutionStage(stage_id=i, execution_type=kind,
                       agents=[AgentTask(agent_id=a, sub_query="x") for a in agents])
        for i, (kind, agents) in enumerate(stages, 1)])
    if gap:
        plan.capability_gap = {"detected": True}
    return plan


# ── 계획 모양 ────────────────────────────────────────────────────────

def test_shape_ignores_order_inside_a_parallel_stage():
    a = plan_shape(_plan(("parallel", ["weather_agent", "currency_exchange_agent"]),
                         ("sequential", ["llm_search_agent"])))
    b = plan_shape(_plan(("parallel", ["currency_exchange_agent", "weather_agent"]),
                         ("sequential", ["llm_search_agent"])))
    assert a == b


def test_shape_keeps_stage_order_and_kind():
    """대조군 — 순서·종류가 다르면 다른 모양이다 (위 무시가 전부를 뭉개지 않는다)."""
    seq = plan_shape(_plan(("sequential", ["internet_agent"]), ("sequential", ["summarization_agent"])))
    rev = plan_shape(_plan(("sequential", ["summarization_agent"]), ("sequential", ["internet_agent"])))
    par = plan_shape(_plan(("parallel", ["internet_agent", "summarization_agent"])))
    assert len({seq, rev, par}) == 3


def test_shape_keeps_duplicates():
    """같은 에이전트 두 번(날씨 ×2)과 한 번은 다른 계획이다."""
    two = plan_shape(_plan(("parallel", ["weather_agent", "weather_agent"])))
    one = plan_shape(_plan(("parallel", ["weather_agent"])))
    assert two != one


def test_shape_marks_capability_gap_and_errors():
    assert "gap" in plan_shape(_plan(gap=True))
    assert plan_shape(ValueError("boom")) == "error:ValueError"
    assert "gap" not in plan_shape(_plan(("sequential", ["internet_agent"])))   # 대조군


# ── 분포 요약 ────────────────────────────────────────────────────────

def test_summarize_counts_and_modal():
    s = summarize(["A", "A", "B"])
    assert s == {"n": 3, "counts": {"A": 2, "B": 1}, "modal": "A"}


# ── 비교 판정 ────────────────────────────────────────────────────────
#
# 판정 규칙은 잡음 재실행으로 정했다 (2026-10-04, 코드 무변경):
#   N=3 · 처음 보는 모양=실패        → 회귀 5건 (전부 거짓)
#   N=5 · 안정 ≥80% · 유지 ≥60%      → 회귀 2건씩 두 번, 매번 다른 시나리오 (전부 거짓)
# 그래서: 기준선 N≥8, 안정 ≥7/8, 기준선 최빈이 현재 절반 미만일 때만 drift.
# 진짜 회귀(프롬프트가 달라짐)는 계통적이라 여러 시나리오에서 크게 드러난다.

BASE_CFG = {"registry": "r1", "selector": False, "runs": 8}


def _doc(cfg=BASE_CFG, **scenarios):
    return {"config": dict(cfg), "scenarios": {k: summarize(v) for k, v in scenarios.items()}}


def _status(findings, sid):
    return next(f["status"] for f in findings if f["id"] == sid)


def test_stable_scenario_unchanged_is_same():
    assert _status(compare(_doc(a=["X"] * 8), _doc(a=["X"] * 5)), "a") == "same"


def test_occasional_rare_plan_is_not_a_regression():
    """잡음 재실행 실측 — 안정 시나리오에 드문 계획이 섞이는 건 정상이다."""
    findings = compare(_doc(a=["X"] * 8), _doc(a=["X", "X", "X", "Z", "Y"]))
    assert _status(findings, "a") == "same"
    assert next(f for f in findings if f["id"] == "a")["unseen"] == ["Y", "Z"]   # 정보로는 남는다


def test_stable_scenario_modal_mostly_gone_is_drift():
    assert _status(compare(_doc(a=["X"] * 8), _doc(a=["Y"] * 4 + ["X"])), "a") == "drift"


def test_half_kept_is_not_drift():
    """대조군 — 기준선 최빈이 절반 이상 남아 있으면 drift 가 아니다 (잡음 범위)."""
    findings = compare(_doc(a=["X"] * 8), _doc(a=["X", "X", "X", "Y", "Y"]))
    assert _status(findings, "a") == "same"


def test_seven_of_eight_counts_as_stable():
    assert _status(compare(_doc(a=["X"] * 7 + ["Y"]), _doc(a=["Z"] * 5)), "a") == "drift"


def test_variable_baseline_is_reported_not_judged():
    """기준선 최빈이 7/8 미만이면 판정하지 않는다 — 비교 근거가 약하다."""
    findings = compare(_doc(a=["X"] * 6 + ["Y"] * 2), _doc(a=["Y"] * 5))
    assert _status(findings, "a") == "variable"


def test_variable_baseline_with_never_seen_modal_is_shifted():
    """흔들리던 시나리오라도 기준선에 한 번도 없던 계획이 최빈이 되면 표시한다(실패는 아님)."""
    findings = compare(_doc(a=["X"] * 6 + ["Y"] * 2), _doc(a=["Q"] * 5))
    assert _status(findings, "a") == "shifted"


def test_too_few_baseline_runs_cannot_be_stable():
    """N<8 기준선은 전부 같아도 안정으로 보지 않는다 — 5회 기준선의 오판 원인."""
    findings = compare(_doc(cfg={**BASE_CFG, "runs": 5}, a=["X"] * 5), _doc(a=["Y"] * 5))
    assert _status(findings, "a") == "shifted"


def test_missing_scenario_is_reported():
    findings = compare(_doc(a=["X"] * 8, b=["Y"] * 8), _doc(a=["X"] * 5))
    assert _status(findings, "b") == "missing"


def test_regressions_are_only_drift_and_missing():
    from ontology.orchestrator.plan_regression import FAILING
    assert set(FAILING) == {"drift", "missing"}


@pytest.mark.parametrize("key,value", [("registry", "r2"), ("selector", True)])
def test_config_mismatch_refuses_to_compare(key, value):
    """다른 레지스트리·선택기 설정이면 계획이 달라도 플래너 탓이 아니다 — 비교 자체를 거부."""
    current_cfg = {**BASE_CFG, key: value}
    with pytest.raises(ConfigMismatch):
        compare(_doc(a=["X"] * 8), _doc(cfg=current_cfg, a=["X"] * 5))


def test_different_run_count_is_allowed():
    """대조군 — 반복 횟수는 분포로 비교하므로 달라도 된다."""
    findings = compare(_doc(a=["X"] * 8), _doc(cfg={**BASE_CFG, "runs": 3}, a=["X"] * 3))
    assert _status(findings, "a") == "same"


# ── gap 수준 비교 (FORGE 생성 요청) ───────────────────────────────────

def test_gap_level_keeps_only_the_gap_decision():
    """FORGE 요청에서 라우팅을 정하는 건 gap 선언이다 — 함께 붙는 단계는 잡음이다."""
    from ontology.orchestrator.plan_regression import at_level

    a = "S[task_classifier_agent] → S[llm_search_agent] | gap"
    b = "S[code_generation_agent] | gap"
    assert at_level(a, "gap") == at_level(b, "gap") == "gap"
    assert at_level("S[internet_agent]", "gap") == "no-gap"        # 대조군 — gap 유무는 구별
    assert at_level("error:ValueError", "gap") == "error:ValueError"
    assert at_level(a, "plan") == a                                  # 기본 수준은 그대로


# ── 레지스트리 지문 ──────────────────────────────────────────────────

def test_registry_fingerprint_ignores_order_but_not_content():
    a = [{"agent_id": "x", "description": "d1"}, {"agent_id": "y", "description": "d2"}]
    b = list(reversed(a))
    c = [{"agent_id": "x", "description": "d1 changed"}, {"agent_id": "y", "description": "d2"}]
    assert registry_fingerprint(a) == registry_fingerprint(b)
    assert registry_fingerprint(a) != registry_fingerprint(c)
