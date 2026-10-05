"""
Workflow Orchestrator — Logos 운영 오케스트레이터.

Pipeline:
1. Query → QueryPlanner (Flash-Lite) → ExecutionPlan
2. ExecutionPlan → PlanValidator → Validated Plan
3. Validated Plan → ExecutionEngine → Stage Results
4. Stage Results → ResultAggregator → Final Output

본문(계획 → 검증 → 실행 → 피드백, 스트리밍)은 logosai.orchestration.workflow_orchestrator
가 정본이다 (2026-10-05, orchestrator-unify P3). 여기서는 플래너만 Logos 플래너
(Gemini 직접 호출, Logos 프롬프트·배제 관문·키워드 안전망, GNN+RL 힌트)로 바꾼다.
옮기기 전후 동작이 같음은 tests/test_orchestrator_characterization.py 가 확인한다.
"""

import asyncio
from typing import Any, Dict, Optional

# 이 모듈에서 import 하던 이름을 그대로 유지한다 (호환)
from .models import (  # noqa: F401
    ExecutionPlan,
    WorkflowResult,
    ProgressEvent,
    ProgressEventType,
)
from .agent_registry import AgentRegistry, get_registry  # noqa: F401
from .query_planner import QueryPlanner
from .plan_validator import PlanValidator  # noqa: F401
from .data_transformer import DataTransformer  # noqa: F401
from .execution_engine import ExecutionEngine, AgentExecutor  # noqa: F401
from .result_aggregator import ResultAggregator  # noqa: F401
from .progress_streamer import ProgressStreamer
from .exceptions import OrchestratorError, PlanValidationError  # noqa: F401

__sdk_base__ = "logosai.orchestration.workflow_orchestrator"
try:
    from logosai.orchestration.workflow_orchestrator import (
        WorkflowOrchestrator as _SDKWorkflowOrchestrator,
    )
except ModuleNotFoundError as _e:
    if _e.name not in ("logosai", "logosai.orchestration",
                       "logosai.orchestration.workflow_orchestrator"):
        raise
    raise ImportError(
        "ontology.orchestrator 의 워크플로 실행은 logosai.orchestration 으로 옮겨졌다 — "
        "`pip install logosai-ontology[logosai]` 로 logosai 를 설치하라."
    ) from _e


class WorkflowOrchestrator(_SDKWorkflowOrchestrator):
    """Logos 플래너를 쓰는 WorkflowOrchestrator.

    Example:
        orchestrator = WorkflowOrchestrator(agent_executor=my_executor, registry=registry)
        async for event in orchestrator.run_streaming("서울과 부산 날씨 비교해줘"):
            print(f"{event.type}: {event.message}")
    """

    def __init__(
        self,
        agent_executor: Optional[AgentExecutor] = None,
        registry: Optional[AgentRegistry] = None,
        enable_validation: bool = True,
        enable_streaming: bool = True,
    ):
        # llm_invoke 는 받지 않는다 — Logos 플래너는 Gemini 를 직접 부르므로, 받아 놓고
        # 안 쓰면 호출자를 속이게 된다. SDK 플래너를 쓰려면 logosai.orchestration 을 쓴다.
        super().__init__(
            agent_executor=agent_executor,
            registry=registry,
            enable_validation=enable_validation,
            enable_streaming=enable_streaming,
        )

    def _make_planner(self, streamer: Optional[ProgressStreamer]) -> QueryPlanner:
        return QueryPlanner(registry=self.registry, streamer=streamer)


# Factory functions

def create_orchestrator(
    agent_executor: Optional[AgentExecutor] = None,
    registry: Optional[AgentRegistry] = None,
) -> WorkflowOrchestrator:
    """Create a WorkflowOrchestrator with default configuration"""
    return WorkflowOrchestrator(
        agent_executor=agent_executor,
        registry=registry,
    )


async def quick_run(
    query: str,
    agent_executor: AgentExecutor,
    context: Optional[Dict[str, Any]] = None,
) -> WorkflowResult:
    """Quick helper to run a query with minimal setup"""
    orchestrator = WorkflowOrchestrator(agent_executor=agent_executor)
    return await orchestrator.run(query, context)


# Convenience for sync usage
def run_sync(
    query: str,
    agent_executor: AgentExecutor,
    context: Optional[Dict[str, Any]] = None,
) -> WorkflowResult:
    """Synchronous wrapper for quick_run"""
    return asyncio.run(quick_run(query, agent_executor, context))
