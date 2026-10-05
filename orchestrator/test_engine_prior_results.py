"""ExecutionEngine 전 스테이지 구조화 핸드오프 (Agentic Upgrade Phase 2).

기존: 다음 에이전트는 직전 스테이지의 2000자 절단 문자열만 받음 → viz 가
internet 원문을 못 보는 구조적 결함 (2026-07-14 실측).
변경: 모든 이전 스테이지의 원본 결과를 context.previous_results(무절단,
{agent_id: data})로 상시 탑재 — logosai HandoffEnvelope 가 표준 소비.

직접 실행: .venv/bin/python ontology/orchestrator/test_engine_prior_results.py
"""

import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ontology"))

from orchestrator.execution_engine import ExecutionEngine  # noqa: E402
from orchestrator.models import AgentResult  # noqa: E402


def _engine():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._current_user_query = "원 쿼리"
    eng._agent_results = {}
    return eng


def main():
    fails = []

    def t(name, cond):
        print(("PASS  " if cond else "FAIL  ") + name)
        if not cond:
            fails.append(name)

    captured = {}

    async def fake_exec(aid, q, ctx):
        captured.clear()
        captured.update(ctx)
        return {"ok": 1}

    # 1. 이전 스테이지 결과가 context.previous_results 로 전달 (성공 건만)
    eng = _engine()
    eng.agent_executor = fake_exec
    eng._agent_results["stage_1.internet_agent"] = AgentResult(
        agent_id="internet_agent", stage_id=1, success=True,
        data={"answer": "일별 종가 표 " + "x" * 5000})  # 무절단 확인용 5KB+
    eng._agent_results["stage_1.broken_agent"] = AgentResult(
        agent_id="broken_agent", stage_id=1, success=False, error="died")
    eng._agent_results["stage_2.analysis_agent"] = AgentResult(
        agent_id="analysis_agent", stage_id=2, success=True,
        data={"summary": "추세", "results": {"data_values": [1, 2]}})

    asyncio.run(eng._call_agent("viz", "차트 생성", None, {"email": "x@y"}))
    prev = captured.get("previous_results")
    t("P-1 전 스테이지 성공 결과 전달 ({agent_id: data})",
      isinstance(prev, dict) and set(prev) == {"internet_agent", "analysis_agent"})
    t("P-2 실패 에이전트는 제외", "broken_agent" not in (prev or {}))
    t("P-3 무절단 (5KB 원문 그대로)",
      prev and len(prev["internet_agent"]["answer"]) > 5000)
    t("P-4 기존 context 필드 보존", captured.get("email") == "x@y"
      and captured.get("original_query") == "원 쿼리")

    # 2. NaN 정화 (HTTP JSON 직렬화 안전 — 브라우저/파서 거부 방지)
    eng2 = _engine()
    eng2.agent_executor = fake_exec
    eng2._agent_results["stage_1.analysis_agent"] = AgentResult(
        agent_id="analysis_agent", stage_id=1, success=True,
        data={"normality": {"statistic": float("nan")}, "mean": 2.5})
    asyncio.run(eng2._call_agent("viz", "차트", None, {}))
    prev2 = captured.get("previous_results", {})
    t("P-5 prior 결과 NaN → None 정화",
      prev2.get("analysis_agent", {}).get("normality", {}).get("statistic") is None
      and prev2.get("analysis_agent", {}).get("mean") == 2.5)

    # 3. 호출측이 이미 previous_results 를 넣었으면 존중 (setdefault)
    eng3 = _engine()
    eng3.agent_executor = fake_exec
    eng3._agent_results["stage_1.a"] = AgentResult(agent_id="a", stage_id=1, success=True, data={"x": 1})
    asyncio.run(eng3._call_agent("viz", "차트", None, {"previous_results": {"custom": {"y": 2}}}))
    t("P-6 기존 previous_results 존중", captured.get("previous_results") == {"custom": {"y": 2}})

    # 4. 이전 결과 없으면 키 미추가 (1스테이지 워크플로우 무영향)
    eng4 = _engine()
    eng4.agent_executor = fake_exec
    asyncio.run(eng4._call_agent("solo", "쿼리", None, {}))
    t("P-7 이전 결과 없으면 previous_results 미추가", "previous_results" not in captured)

    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
