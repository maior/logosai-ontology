"""배제 관문(exclusion gate) 테스트 (2026-07-11).

배경: description 에 "점자는 대상이 아닙니다"가 명시돼 있고 프롬프트에 C-3
규칙 + 추천 기각 단서까지 있어도, flash-lite 플래너가 temperature 0 에서
점자→모스 배정을 반복 (라이브 + 오프라인 분리 실험으로 확정 — 추천 주입
없이도 동일). 프롬프트 설득의 한계.

해법 전례: find_equivalent 관문 — "gap 판정이 흔들려도 단순 매칭 질문은
안정적". 계획 파싱 직후, 배제 조건 보유 에이전트가 선택됐을 때만 좁은
단일 판정("이 쿼리가 배제에 해당하나?")을 LLM 에 묻고 위반이면 제거,
대안이 없으면 capability_gap 강제.

계약:
  - enforce_exclusion_gate(query, plan_data, agent_descriptions, llm_invoke)
    → (plan_data, violations)
  - 배제 마커("대상이 아닙니다") 없는 에이전트는 LLM 콜 0회 (비용 가드)
  - 위반 판정 → stages 에서 제거, 전부 제거되면 capability_gap.detected=true
  - '무관' 판정 → 계획 그대로 (과교정 방지)
  - LLM 에러 → 관문 통과 (계획을 막지 않음 — find_equivalent 와 동일 fail-open)
  - 기존 capability_gap 이 이미 detected=true 면 덮어쓰지 않음

실행: .venv/bin/python ontology/orchestrator/test_exclusion_gate.py
"""
import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ontology"))


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


DESCS = {
    "morse_agent": "영문 텍스트를 모스 부호로 변환합니다. 점자(braille) 등 다른 부호 체계는 대상이 아닙니다.",
    "weather_agent": "날씨를 조회합니다.",
}


def _plan(agents):
    return {"stages": [{"stage_id": 1, "execution_type": "sequential",
                        "agents": [{"agent_id": a, "sub_query": "..."} for a in agents]}]}


def test_violation_removes_agent_and_sets_gap():
    """배제 위반 판정 → 에이전트 제거 + (대안 없음) capability_gap 강제."""
    from ontology.orchestrator.query_planner import enforce_exclusion_gate

    calls = []
    async def llm(prompt):
        calls.append(prompt)
        return "해당"  # 배제 조건에 해당 = 위반

    plan, viol = _run(enforce_exclusion_gate(
        "점자로 변환해줘", _plan(["morse_agent"]), DESCS, llm))
    assert viol == ["morse_agent"]
    assert plan["stages"] == [], "위반 에이전트가 stages 에서 제거되지 않음"
    gap = plan.get("capability_gap") or {}
    assert gap.get("detected") is True, "대안 없는 위반인데 capability_gap 미설정"
    assert len(calls) == 1


def test_irrelevant_keeps_plan():
    """'무관' 판정 → 계획 그대로 (과교정 방지)."""
    from ontology.orchestrator.query_planner import enforce_exclusion_gate

    async def llm(prompt):
        return "무관"

    plan, viol = _run(enforce_exclusion_gate(
        "SOS를 모스 부호로 변환해줘", _plan(["morse_agent"]), DESCS, llm))
    assert viol == []
    assert plan["stages"][0]["agents"][0]["agent_id"] == "morse_agent"
    assert not (plan.get("capability_gap") or {}).get("detected")


def test_no_marker_no_llm_call():
    """배제 마커 없는 에이전트만 선택 → LLM 콜 0회 (비용 가드)."""
    from ontology.orchestrator.query_planner import enforce_exclusion_gate

    calls = []
    async def llm(prompt):
        calls.append(prompt)
        return "해당"

    plan, viol = _run(enforce_exclusion_gate(
        "서울 날씨", _plan(["weather_agent"]), DESCS, llm))
    assert calls == [], "마커 없는데 LLM 호출됨"
    assert viol == []


def test_llm_error_fails_open():
    """관문 LLM 에러 → 계획을 막지 않음 (fail-open, find_equivalent 동일)."""
    from ontology.orchestrator.query_planner import enforce_exclusion_gate

    async def llm(prompt):
        raise RuntimeError("LLM down")

    plan, viol = _run(enforce_exclusion_gate(
        "점자로 변환해줘", _plan(["morse_agent"]), DESCS, llm))
    assert viol == []
    assert plan["stages"], "관문 에러가 계획을 삭제함 (fail-open 위반)"


def test_partial_removal_keeps_other_agents():
    """위반 1 + 정상 1 → 위반만 제거, stage 유지, gap 미설정."""
    from ontology.orchestrator.query_planner import enforce_exclusion_gate

    async def llm(prompt):
        return "해당"  # morse 만 검사 대상 (weather 는 마커 없음)

    plan, viol = _run(enforce_exclusion_gate(
        "점자로 변환하고 서울 날씨도", _plan(["morse_agent", "weather_agent"]), DESCS, llm))
    assert viol == ["morse_agent"]
    remaining = [a["agent_id"] for a in plan["stages"][0]["agents"]]
    assert remaining == ["weather_agent"]
    assert not (plan.get("capability_gap") or {}).get("detected")


def test_existing_gap_not_overwritten():
    """플래너가 이미 gap 선언한 경우 관문이 덮어쓰지 않음."""
    from ontology.orchestrator.query_planner import enforce_exclusion_gate

    async def llm(prompt):
        return "해당"

    p = _plan(["morse_agent"])
    p["capability_gap"] = {"detected": True, "reason": "원본", "missing_capabilities": ["x"]}
    plan, _ = _run(enforce_exclusion_gate("점자로 변환해줘", p, DESCS, llm))
    assert plan["capability_gap"]["reason"] == "원본"


def test_backfill_gap_for_empty_plan():
    """빈 stages + gap 미선언 → gap 백필 (LLM 이 무능력을 빈 계획으로만
    표현하는 케이스 — 라이브 실측: 점자 쿼리 PLAN=[] gap=None →
    validation_error 3회 낭비). GREEN: detected=true 백필."""
    from ontology.orchestrator.query_planner import backfill_gap_for_empty_plan

    plan = {"stages": []}
    out = backfill_gap_for_empty_plan("점자로 변환해줘", plan)
    gap = out.get("capability_gap") or {}
    assert gap.get("detected") is True
    assert "점자로 변환해줘" in gap.get("suggested_agent_description", "")


def test_backfill_skips_nonempty_or_declared():
    """stages 가 있으면/이미 선언됐으면 백필하지 않음."""
    from ontology.orchestrator.query_planner import backfill_gap_for_empty_plan

    p1 = {"stages": [{"agents": [{"agent_id": "weather_agent"}]}]}
    assert not (backfill_gap_for_empty_plan("q", p1).get("capability_gap") or {}).get("detected")

    p2 = {"stages": [], "capability_gap": {"detected": True, "reason": "원본"}}
    assert backfill_gap_for_empty_plan("q", p2)["capability_gap"]["reason"] == "원본"


def main():
    fails = []
    for fn in (
        test_violation_removes_agent_and_sets_gap,
        test_irrelevant_keeps_plan,
        test_no_marker_no_llm_call,
        test_llm_error_fails_open,
        test_partial_removal_keeps_other_agents,
        test_existing_gap_not_overwritten,
        test_backfill_gap_for_empty_plan,
        test_backfill_skips_nonempty_or_declared,
    ):
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            print("FAIL", fn.__name__, "→", type(e).__name__, str(e)[:120])
            fails.append(fn.__name__)
    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
