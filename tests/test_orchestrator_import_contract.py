"""`ontology.orchestrator` 계약 — logos_api 가 쓰는 이름과 실행 엔진의 동작.

왜 있나: 오케스트레이터 실행부를 logosai SDK 로 옮기는 작업(orchestrator-unify)
에서 logos_api 를 한 줄도 고치지 않아도 되게 하는 안전망이다.
소비자: logos_api/app/services/orchestrator_service.py (`_ensure_imports`,
`initialize`, `_register_agents_to_registry`, 재계획 분기).

행동 테스트는 LLM 없이 손으로 만든 계획을 실행한다 — 옮길 대상의 핵심인
"병렬 실행"과 "앞 단계 결과를 다음 단계 쿼리에 싣는 핸드오프"를 고정한다.
SDK 의 구 엔진(logosai.workflow)은 이 핸드오프가 없어 순차 2단계가
`agent_query=None` 으로 실패했다(2026-10-04 실측).

출력 문구(`[결과 1]` 같은 라벨)는 단언하지 않는다 — 결과가 '실린다'만 본다.

실패 기록 — 처음엔 '알려진 결함'으로 계약에서 뺐다가(엔진이 실행기의
``{"success": False, "error": ...}`` 를 성공으로 기록했다), 고친 뒤(orchestrator-unify
3-2) 아래 test_agent_failure_is_recorded_as_failure 로 고정했다.
"""
import asyncio
import importlib
import inspect
import time

import pytest

# logos_api orchestrator_service._ensure_imports 가 가져오는 이름 그대로
LOGOS_API_NAMES = [
    "WorkflowOrchestrator", "QueryPlanner", "DataTransformer", "ExecutionEngine",
    "AgentRegistry", "ProgressStreamer", "ProgressEventType", "ExecutionPlan",
]
LOGOS_API_SUBMODULE_NAMES = [
    ("ontology.orchestrator.models", "AgentRegistryEntry"),
    ("ontology.orchestrator.models", "AgentSchema"),
    ("ontology.orchestrator.query_planner", "detect_explicit_capability_gap"),
]


# ── 1. import 경로 ───────────────────────────────────────────────────

@pytest.mark.parametrize("name", LOGOS_API_NAMES)
def test_logos_api_names_importable(name):
    mod = importlib.import_module("ontology.orchestrator")
    assert hasattr(mod, name), f"logos_api 가 쓰는 ontology.orchestrator.{name} 이 사라졌다"


@pytest.mark.parametrize("module,name", LOGOS_API_SUBMODULE_NAMES)
def test_logos_api_submodule_names_importable(module, name):
    assert hasattr(importlib.import_module(module), name)


def test_every_exported_name_resolves():
    mod = importlib.import_module("ontology.orchestrator")
    assert len(mod.__all__) >= 20  # 대조군 — 빈 __all__ 이면 아래가 공허하다
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert missing == []


# ── 2. 호출 모양 (logos_api 가 부르는 그대로) ─────────────────────────

def test_call_shapes_used_by_logos_api():
    from ontology.orchestrator import (
        AgentRegistry, ExecutionEngine, WorkflowOrchestrator,
    )
    from ontology.orchestrator.query_planner import detect_explicit_capability_gap

    inspect.signature(WorkflowOrchestrator).bind(
        agent_executor=lambda *a: None, registry=None,
        enable_validation=True, enable_streaming=True,
    )
    inspect.signature(WorkflowOrchestrator.run_streaming).bind(
        object(), query="q", context={}
    )
    registry = AgentRegistry()
    assert callable(registry.register_agent) and callable(registry.unregister_agent)
    inspect.signature(ExecutionEngine).bind(agent_executor=lambda *a: None)
    inspect.signature(ExecutionEngine.execute).bind(object(), "plan", {})
    assert detect_explicit_capability_gap("날씨 알려줘") is None
    assert detect_explicit_capability_gap("에이전트 만들어줘")["detected"] is True


def test_registry_round_trip():
    from ontology.orchestrator import AgentRegistry
    from ontology.orchestrator.models import AgentRegistryEntry, AgentSchema

    registry = AgentRegistry()
    entry = AgentRegistryEntry(
        agent_id="contract_probe_agent", name="probe", description="계약 확인용",
        capabilities=[], tags=[],
        schema=AgentSchema(input_type="query", output_type="text"),
    )
    assert not registry.has_agent("contract_probe_agent")  # 대조군 — 처음엔 없다
    registry.register_agent(entry)
    assert registry.has_agent("contract_probe_agent")
    assert registry.get_agent("contract_probe_agent").agent_id == "contract_probe_agent"
    registry.unregister_agent("contract_probe_agent")
    assert not registry.has_agent("contract_probe_agent")


# ── 3. 실행 엔진의 행동 ──────────────────────────────────────────────

STEP_SECONDS = 0.3


def _recording_executor(outputs, log, fail=()):
    async def executor(agent_id, sub_query, context):
        start = time.monotonic()
        await asyncio.sleep(STEP_SECONDS)
        log[agent_id] = {
            "start": start, "end": time.monotonic(),
            "query": sub_query, "context": dict(context or {}),
        }
        if agent_id in fail:
            return {"success": False, "error": f"{agent_id} 실패"}
        return {"success": True, "result": {"answer": outputs[agent_id]}}
    return executor


def _plan(query, stages):
    from ontology.orchestrator import ExecutionPlan
    from ontology.orchestrator.models import AgentTask, ExecutionStage

    built = []
    for i, (kind, tasks) in enumerate(stages, 1):
        built.append(ExecutionStage(
            stage_id=i, execution_type=kind,
            agents=[AgentTask(agent_id=a, sub_query=q) for a, q in tasks],
            depends_on=[i - 1] if i > 1 else None,
        ))
    strategy = "hybrid" if len(stages) > 1 and stages[0][0] == "parallel" else "sequential"
    return ExecutionPlan(query=query, workflow_strategy=strategy, stages=built)


async def test_hybrid_plan_runs_parallel_then_hands_off():
    from ontology.orchestrator import ExecutionEngine

    outputs = {"alpha_agent": "ALPHA-7731", "beta_agent": "BETA-4410", "gamma_agent": "GAMMA-FINAL"}
    log = {}
    plan = _plan("서울과 부산 날씨를 비교해줘", [
        ("parallel", [("alpha_agent", "서울 날씨"), ("beta_agent", "부산 날씨")]),
        ("sequential", [("gamma_agent", "두 도시를 비교")]),
    ])

    result = await ExecutionEngine(agent_executor=_recording_executor(outputs, log)).execute(plan)

    a, b, g = log["alpha_agent"], log["beta_agent"], log["gamma_agent"]
    # ⓐ 같은 stage 의 두 에이전트는 겹쳐 돈다
    assert b["start"] < a["end"] and a["start"] < b["end"], "병렬 stage 가 순차로 돌았다"
    # ⓑ 대조군 — 다음 stage 는 앞 stage 가 끝난 뒤 시작한다 (겹치지 않는다)
    assert g["start"] >= max(a["end"], b["end"]) - 0.01, "다음 stage 가 앞 stage 를 기다리지 않았다"
    # ⓒ 핸드오프 — 두 결과와 원래 요청이 다음 단계 쿼리에 실린다
    assert "ALPHA-7731" in g["query"] and "BETA-4410" in g["query"]
    assert "두 도시를 비교" in g["query"]
    # 대조군 — 첫 stage 쿼리에는 아직 아무 결과도 없다
    assert "ALPHA-7731" not in a["query"] and "BETA-4410" not in b["query"]
    # ⓓ 원 사용자 질문이 문맥으로 전달된다
    assert g["context"].get("original_query") == "서울과 부산 날씨를 비교해줘"
    # ⓔ 전체 성공, 마지막 결과가 최종 출력에 들어 있다
    assert result.success is True
    assert "GAMMA-FINAL" in str(result.final_output)


async def test_sequential_two_steps_hand_off():
    """SDK 구 엔진이 깨져 있던 바로 그 모양 — 2단계가 1단계 결과를 받는다."""
    from ontology.orchestrator import ExecutionEngine

    outputs = {"upper_agent": "HELLO WORLD", "reverse_agent": "DLROW OLLEH"}
    log = {}
    plan = _plan("hello world 를 대문자로 바꾼 뒤 뒤집어줘", [
        ("sequential", [("upper_agent", "대문자로 바꿔라: hello world")]),
        ("sequential", [("reverse_agent", "그 결과를 뒤집어라")]),
    ])

    result = await ExecutionEngine(agent_executor=_recording_executor(outputs, log)).execute(plan)

    assert result.success is True
    second = log["reverse_agent"]["query"]
    assert isinstance(second, str) and second, "2단계가 빈 쿼리를 받았다"
    assert "HELLO WORLD" in second and "그 결과를 뒤집어라" in second


async def test_agent_failure_is_recorded_as_failure():
    """logos_api 실행기의 실패 모양이 옛 경로로도 실패로 기록된다.

    핸드오프는 그대로다 — 실패 내용도 다음 단계로 넘어간다. 뺐더니 하류 요약
    에이전트가 무관한 내용을 지어냈다 (2026-10-04 실측). 바뀌는 것은 기록뿐이다.
    """
    from ontology.orchestrator import ExecutionEngine

    outputs = {"ok_agent": "OK-55", "next_agent": "N"}
    log = {}
    plan = _plan("둘 다 해줘", [
        ("parallel", [("ok_agent", "하나"), ("bad_agent", "둘")]),
        ("sequential", [("next_agent", "합쳐라")]),
    ])
    outputs["bad_agent"] = "never"

    result = await ExecutionEngine(
        agent_executor=_recording_executor(outputs, log, fail={"bad_agent"})
    ).execute(plan)

    by_agent = {r.agent_id: r for stage in result.stages for r in stage.results}
    assert by_agent["bad_agent"].success is False, "실패한 에이전트가 성공으로 기록됐다"
    assert by_agent["ok_agent"].success is True  # 대조군
    assert "OK-55" in log["next_agent"]["query"]
    assert "bad_agent 실패" in log["next_agent"]["query"], "실패 내용이 하류에 전달되지 않았다"
    assert result.success is True  # 결말은 그대로 — logos_api 재시도를 부르지 않는다
