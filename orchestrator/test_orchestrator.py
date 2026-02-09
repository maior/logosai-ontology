"""
Integration test for Workflow Orchestrator

Tests the complete pipeline:
Query → Plan → Validate → Execute → Aggregate → Result
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ontology.orchestrator import (
    WorkflowOrchestrator,
    AgentRegistry,
    QueryPlanner,
    PlanValidator,
    ProgressStreamer,
    ProgressEventType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock agent executor for testing
async def mock_agent_executor(
    agent_id: str,
    sub_query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Mock agent executor that simulates agent responses"""

    # Simulate processing time
    await asyncio.sleep(0.5)

    # Return mock data based on agent type
    if agent_id == "internet_agent":
        return {
            "source": "web_search",
            "query": sub_query,
            "results": [
                {"date": "2024-01-22", "price": 72500},
                {"date": "2024-01-23", "price": 71200},
                {"date": "2024-01-24", "price": 73100},
                {"date": "2024-01-25", "price": 72800},
                {"date": "2024-01-26", "price": 74200},
            ],
            "raw_text": "삼성전자 최근 5일 종가: 72,500원, 71,200원, 73,100원, 72,800원, 74,200원"
        }

    elif agent_id == "analysis_agent":
        input_data = context.get("input_data", {}) if context else {}
        # input_data가 문자열일 수 있으므로 타입 체크
        if isinstance(input_data, dict):
            data_points = input_data.get("results", [])
        else:
            data_points = []
        return {
            "analysis_type": "stock_price",
            "summary": "최근 5일간 삼성전자 주가는 상승 추세를 보임",
            "data_points": data_points,
            "trend": "upward",
            "change_percent": 2.3,
        }

    elif agent_id == "data_visualization_agent":
        return {
            "chart_type": "line",
            "svg": """<svg width="400" height="200">
                <polyline points="50,150 100,160 150,140 200,145 250,120"
                          style="fill:none;stroke:blue;stroke-width:2"/>
            </svg>""",
            "title": "삼성전자 5일 종가 추이",
        }

    elif agent_id == "llm_search_agent":
        return {
            "answer": f"질문 '{sub_query}'에 대한 답변입니다.",
            "sources": ["Wikipedia", "Investopedia"],
        }

    elif agent_id == "samsung_gateway_agent":
        return {
            "domain": "samsung_semiconductor",
            "data": {"yield": 92.5, "fab": "P3"},
        }

    else:
        return {
            "agent_id": agent_id,
            "response": f"Mock response for {sub_query}",
        }


async def test_plan_only():
    """Test plan creation without execution"""
    print("\n" + "="*60)
    print("TEST: Plan Creation Only")
    print("="*60)

    orchestrator = WorkflowOrchestrator(
        agent_executor=mock_agent_executor,
        enable_validation=True,
    )

    query = "삼성전자 5일 종가 그래프로 그려줘"
    print(f"\n쿼리: {query}")

    plan = await orchestrator.create_plan_only(query)

    print(f"\n✅ 실행 계획 생성 완료!")
    print(f"  - Workflow Strategy: {plan.workflow_strategy}")
    print(f"  - Stages: {plan.get_stage_count()}")
    print(f"  - Total Agents: {plan.get_total_agents()}")
    print(f"  - Reasoning: {plan.reasoning}")

    print(f"\n📋 Stage Details:")
    for stage in plan.stages:
        print(f"  Stage {stage.stage_id} ({stage.execution_type}):")
        for agent in stage.agents:
            print(f"    - {agent.agent_id}: {agent.sub_query[:50]}...")

    return plan


async def test_full_execution():
    """Test complete workflow execution"""
    print("\n" + "="*60)
    print("TEST: Full Workflow Execution")
    print("="*60)

    orchestrator = WorkflowOrchestrator(
        agent_executor=mock_agent_executor,
        enable_validation=True,
        enable_streaming=True,
    )

    query = "삼성전자 5일 종가 그래프로 그려줘"
    print(f"\n쿼리: {query}")

    result = await orchestrator.run(query)

    print(f"\n✅ 워크플로우 실행 완료!")
    print(f"  - Success: {result.success}")
    print(f"  - Total Time: {result.total_time_ms:.0f}ms")
    print(f"  - Agents Executed: {result.total_agents_executed}")
    print(f"  - Successful: {result.successful_agents}")
    print(f"  - Failed: {result.failed_agents}")

    print(f"\n📊 Final Output:")
    if result.final_output:
        print(json.dumps(result.final_output, ensure_ascii=False, indent=2)[:500])

    return result


async def test_streaming():
    """Test streaming progress events"""
    print("\n" + "="*60)
    print("TEST: Streaming Progress Events")
    print("="*60)

    orchestrator = WorkflowOrchestrator(
        agent_executor=mock_agent_executor,
        enable_validation=True,
        enable_streaming=True,
    )

    query = "환율 그래프 보여주고 왜 그런지 알려줘"
    print(f"\n쿼리: {query}")
    print("\n📡 Streaming Events:")

    event_count = 0
    async for event in orchestrator.run_streaming(query):
        event_count += 1
        icon = event.icon or "📌"
        print(f"  {icon} [{event.type.value}] {event.message_ko or event.message}")

        # Show progress if available
        if event.progress_percent is not None:
            print(f"      Progress: {event.progress_percent:.1f}%")

    print(f"\n✅ Total events: {event_count}")

    return event_count


async def test_registry():
    """Test agent registry"""
    print("\n" + "="*60)
    print("TEST: Agent Registry")
    print("="*60)

    registry = AgentRegistry()
    registry.initialize()

    print(f"\n📋 Registered Agents: {len(registry)}")
    for agent in registry:
        print(f"  - {agent.agent_id}: {agent.name}")
        print(f"    Capabilities: {', '.join(agent.capabilities[:3])}...")

    # Test capability search
    search_agents = registry.get_agents_by_capability("web_search")
    print(f"\n🔍 Agents with 'web_search' capability: {len(search_agents)}")
    for agent in search_agents:
        print(f"  - {agent.agent_id}")

    return registry


async def test_parallel_execution():
    """Test parallel agent execution"""
    print("\n" + "="*60)
    print("TEST: Parallel Execution")
    print("="*60)

    orchestrator = WorkflowOrchestrator(
        agent_executor=mock_agent_executor,
        enable_validation=True,
    )

    # Query that should trigger parallel execution
    query = "삼성전자와 애플 실적 비교해줘"
    print(f"\n쿼리: {query}")

    plan = await orchestrator.create_plan_only(query)

    # Check for parallel stages
    parallel_stages = [s for s in plan.stages if s.execution_type == "parallel"]
    print(f"\n📊 Plan Analysis:")
    print(f"  - Workflow Strategy: {plan.workflow_strategy}")
    print(f"  - Parallel Stages: {len(parallel_stages)}")

    for stage in plan.stages:
        print(f"  Stage {stage.stage_id} ({stage.execution_type}): "
              f"{len(stage.agents)} agents")

    return plan


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 WORKFLOW ORCHESTRATOR INTEGRATION TEST")
    print("="*60)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not set. Some tests may fail.")
        print("   Set it with: export GOOGLE_API_KEY=your_key")
        return

    try:
        # Run tests
        await test_registry()
        await test_plan_only()
        await test_parallel_execution()
        await test_full_execution()
        await test_streaming()

        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
