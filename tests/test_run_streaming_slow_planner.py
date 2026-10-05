"""계획 수립이 0.5s 를 넘겨도(루프는 막지 않고) run_streaming 이 끝까지 흐른다.

logos_api 가 소비하는 경로 그대로다: WorkflowOrchestrator.run_streaming.
플래너 LLM 호출을 비차단으로 고치자, 그동안 루프 차단에 가려져 있던 스트리머
결함(계획 중 0.5s 무이벤트 → 스트림 종료)이 드러나 모든 쿼리가 폴백으로 빠졌다
(2026-10-04 실측: planning_complete 없이 3회 재시도). 그 회귀를 고정한다.
"""
import asyncio

from ontology.orchestrator import AgentRegistry, QueryPlanner, WorkflowOrchestrator
from ontology.orchestrator.models import (
    AgentRegistryEntry, AgentSchema, AgentTask, ExecutionPlan, ExecutionStage,
)


async def test_slow_nonblocking_planning_still_streams_to_the_end(monkeypatch):
    monkeypatch.setenv("LOGOSAI_ARTIFACT_GATE", "off")
    registry = AgentRegistry()
    registry.register_agent(AgentRegistryEntry(
        agent_id="currency_exchange_agent", name="환율", description="환율 조회",
        capabilities=[], tags=[], schema=AgentSchema(input_type="query", output_type="text")))

    async def executor(agent_id, query, context):
        return {"success": True, "result": {"answer": "1 USD = 1,349 KRW"}}

    orch = WorkflowOrchestrator(agent_executor=executor, registry=registry,
                                enable_validation=True, enable_streaming=True)

    async def slow_plan(self, query, context=None, *a, **k):
        await asyncio.sleep(1.2)                  # 실제 LLM 처럼 시간이 걸리되 루프는 막지 않는다
        return ExecutionPlan(query=query, workflow_strategy="sequential", stages=[
            ExecutionStage(stage_id=1, execution_type="sequential", agents=[
                AgentTask(agent_id="currency_exchange_agent", sub_query=query)])])

    # 플래너는 run_streaming 마다 새로 만들어진다 — 클래스에 끼운다
    monkeypatch.setattr(QueryPlanner, "create_plan", slow_plan)

    types = []
    async for event in orch.run_streaming(query="원달러 환율 알려줘", context={}):
        types.append(getattr(event.type, "value", event.type))

    # planning_* 이벤트는 실제 플래너가 낸다 — 여기선 가짜라 없다. 판별은 '계획 뒤의
    # 이벤트가 끝까지 오는가'다 (고치기 전엔 계획 동안 끊겨 [] 였다).
    assert "validation_start" in types, f"계획 수립 중 스트림이 끊겼다: {types}"
    assert "agent_complete" in types and types[-1] == "workflow_complete"


async def test_planning_failure_after_a_slow_start_does_not_hang(monkeypatch):
    """계획 단계 예외 — 스트림이 열려 있는 채로 멈추지 않고 오류가 호출자에게 올라간다."""
    import pytest

    monkeypatch.setenv("LOGOSAI_ARTIFACT_GATE", "off")

    async def executor(agent_id, query, context):
        return {"success": True}

    orch = WorkflowOrchestrator(agent_executor=executor, registry=AgentRegistry(),
                                enable_validation=True, enable_streaming=True)

    async def failing_plan(self, query, context=None, *a, **k):
        await asyncio.sleep(1.0)
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(QueryPlanner, "create_plan", failing_plan)

    async def consume():
        async for _ in orch.run_streaming(query="q", context={}):
            pass

    with pytest.raises(RuntimeError, match="planner exploded"):
        await asyncio.wait_for(consume(), timeout=5)
