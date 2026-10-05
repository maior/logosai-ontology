"""WorkflowOrchestrator 특성화 기록 — SDK 로 옮겨도(orchestrator-unify P3) 동작이 같은가.

리팩터링 **전** 코드로 기록을 떴다. 가짜 LLM(플래너 _call_llm 대체)과 가짜 실행기로
다섯 경로를 돌려 다음을 비교한다:
  · run_streaming 이 내는 이벤트 순서와 내용 (시각·소요·id 는 제외)
  · run 의 WorkflowResult 요약, create_plan_only 의 계획
  · 경로별 예외 종류, 피드백 저장 호출
경로: 단일, 병렬→순차, 검증 실패(미등록 에이전트), 에이전트 실패, 계획 오류(비JSON).

기록 갱신(의도된 변경일 때만): ORCH_CHAR_RECORD=1 pytest tests/test_orchestrator_characterization.py
"""
import json
import os
import re
from pathlib import Path

import pytest

from ontology.orchestrator import AgentRegistry, QueryPlanner, WorkflowOrchestrator
from ontology.orchestrator.models import AgentRegistryEntry, AgentSchema

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "orchestrator_characterization.json"
RECORD = os.environ.get("ORCH_CHAR_RECORD") == "1"

#: 실행마다 달라지는 값 — 비교에서 뺀다
_VOLATILE = ("time", "_ms", "elapsed", "timestamp", "workflow_id", "plan_id", "created_at",
             "duration", "start", "end")

AGENTS = [("weather_agent", "날씨 조회"), ("currency_exchange_agent", "환율 조회"),
          ("llm_search_agent", "종합 답변"), ("broken_agent", "늘 실패한다")]


def _plan(*stages):
    return json.dumps({"workflow_strategy": "hybrid", "reasoning": "r",
                       "final_aggregation": {"type": "combine"},
                       "stages": [{"stage_id": i, "execution_type": kind,
                                   "agents": [{"agent_id": a, "sub_query": f"{a} 할 일",
                                               "input_from": inp} for a in agents]}
                                  for i, (kind, agents, inp) in enumerate(stages, 1)]},
                      ensure_ascii=False)


CRITIQUE_OK = '{"unnecessary_agents": [], "broken_chain": false, "reason": "ok"}'

CASES = {
    "single": _plan(("sequential", ["weather_agent"], None)),
    "parallel_then_seq": _plan(("parallel", ["weather_agent", "currency_exchange_agent"], None),
                               ("sequential", ["llm_search_agent"], ["stage_1"])),
    "validation_fail": _plan(("sequential", ["nonexistent_agent"], None)),
    "agent_fails": _plan(("sequential", ["weather_agent"], None),
                         ("sequential", ["broken_agent"], ["stage_1"])),
    "planning_error": "이건 JSON 이 아니다",
}


def _registry():
    reg = AgentRegistry()
    for aid, desc in AGENTS:
        reg.register_agent(AgentRegistryEntry(
            agent_id=aid, name=aid, description=desc, capabilities=[], tags=[],
            schema=AgentSchema(input_type="query", output_type="text")))
    return reg


async def _executor(agent_id, query, context):
    if agent_id == "broken_agent":
        return {"success": False, "error": "의도된 실패"}
    return {"success": True, "result": {"answer": f"{agent_id} 결과"}}


_DURATION = re.compile(r"\d+(?:\.\d+)?\s?(?:ms|s)\b")


def _scrub(x):
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in sorted(x.items())
                if not any(t in k.lower() for t in _VOLATILE)}
    if isinstance(x, (list, tuple)):
        return [_scrub(v) for v in x]
    if isinstance(x, str):
        x = _DURATION.sub("<dur>", x)          # 문구 안의 소요 시간 ("completed in 1ms")
        return x[:300]
    return x if isinstance(x, (int, float, bool, type(None))) else str(x)


def _install(monkeypatch, plan_text, feedback):
    monkeypatch.setenv("GOOGLE_API_KEY", "characterization-key")
    monkeypatch.setenv("LOGOSAI_ARTIFACT_GATE", "off")
    monkeypatch.setattr(QueryPlanner, "USE_HYBRID_SELECTOR", False)

    async def fake_llm(self, prompt):
        return CRITIQUE_OK if "두 가지만 판정하라" in prompt else plan_text

    async def fake_feedback(self, query, agent_id, success, execution_result=None):
        feedback.append((agent_id, success))

    monkeypatch.setattr(QueryPlanner, "_call_llm", fake_llm)
    monkeypatch.setattr(QueryPlanner, "store_execution_feedback", fake_feedback)


async def _trace(monkeypatch, plan_text):
    out = {}
    feedback = []
    _install(monkeypatch, plan_text, feedback)

    events, err = [], None
    orch = WorkflowOrchestrator(agent_executor=_executor, registry=_registry())
    try:
        async for ev in orch.run_streaming("질문", {"k": "v"}):
            events.append(_scrub({"type": ev.type.value, "stage": ev.stage_id, "agent": ev.agent_id,
                                  "status": getattr(ev.status, "value", ev.status),
                                  "msg": ev.message, "data": ev.data}))
    except Exception as e:      # noqa: BLE001 — 예외도 기록 대상이다
        err = type(e).__name__
    out["stream"] = {"events": events, "error": err, "feedback": list(feedback)}

    feedback.clear()
    try:
        r = await WorkflowOrchestrator(agent_executor=_executor, registry=_registry()).run("질문")
        out["run"] = _scrub({"success": r.success, "final": r.final_output, "error": r.error,
                             "error_stage": r.error_stage, "error_agent": r.error_agent,
                             "executed": r.total_agents_executed, "ok": r.successful_agents,
                             "failed": r.failed_agents})
    except Exception as e:      # noqa: BLE001
        out["run"] = {"raised": type(e).__name__}

    try:
        p = await WorkflowOrchestrator(agent_executor=_executor, registry=_registry()).create_plan_only("질문")
        out["plan_only"] = _scrub({**p.to_dict(), "validation_errors": getattr(p, "validation_errors", None)})
    except Exception as e:      # noqa: BLE001
        out["plan_only"] = {"raised": type(e).__name__}
    return out


async def test_orchestrator_matches_characterization(monkeypatch):
    current = {name: await _trace(monkeypatch, text) for name, text in CASES.items()}
    current = json.loads(json.dumps(current, ensure_ascii=False, default=str))
    if RECORD:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
        pytest.skip(f"특성화 기록 갱신: {FIXTURE}")
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for name in golden:
        for part in ("stream", "run", "plan_only"):
            assert current[name][part] == golden[name][part], f"[{name}] {part} 가 달라졌다"


def test_characterization_covers_distinct_paths():
    """대조군 — 다섯 경로가 서로 다른 결말을 기록했다 (공허한 기록이면 비교도 공허하다)."""
    g = json.loads(FIXTURE.read_text(encoding="utf-8"))
    last = {k: v["stream"]["events"][-1]["type"] if v["stream"]["events"] else None for k, v in g.items()}
    assert last["single"] == "workflow_complete"
    assert g["validation_fail"]["stream"]["error"] == "PlanValidationError"
    assert g["planning_error"]["stream"]["error"] is not None
    assert g["agent_fails"]["run"].get("failed", 0) >= 1
    assert len(g["parallel_then_seq"]["stream"]["events"]) > len(g["single"]["stream"]["events"])
    assert g["single"]["stream"]["feedback"] == [["weather_agent", True]]
