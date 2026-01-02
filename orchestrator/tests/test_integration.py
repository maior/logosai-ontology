"""
Integration tests for Workflow Orchestrator

These tests verify the complete pipeline works correctly.
Run AFTER any modification to the orchestrator.

Run with:
    pytest ontology/orchestrator/tests/test_integration.py -v

Required:
    - GOOGLE_API_KEY environment variable
"""

import asyncio
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Any, Dict, Optional

from ontology.orchestrator import (
    WorkflowOrchestrator,
    AgentRegistry,
    ProgressEventType,
)


# Skip all tests if GOOGLE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set"
)


# Mock agent executor for testing
async def mock_agent_executor(
    agent_id: str,
    sub_query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Mock agent executor that simulates agent responses"""
    await asyncio.sleep(0.1)  # Faster for tests

    if agent_id == "internet_agent":
        return {
            "source": "web_search",
            "query": sub_query,
            "results": [
                {"date": "2024-01-22", "price": 72500},
                {"date": "2024-01-23", "price": 71200},
            ],
            "raw_text": "검색 결과입니다."
        }
    elif agent_id == "analysis_agent":
        return {
            "analysis_type": "data_analysis",
            "summary": "분석 결과입니다.",
            "trend": "upward",
        }
    elif agent_id == "data_visualization_agent":
        return {
            "chart_type": "line",
            "svg": "<svg>...</svg>",
            "title": "차트",
        }
    elif agent_id == "llm_search_agent":
        return {
            "answer": f"'{sub_query}'에 대한 답변입니다.",
            "sources": ["Wikipedia"],
        }
    else:
        return {
            "agent_id": agent_id,
            "response": f"Response for {sub_query}",
        }


class TestAgentRegistry:
    """Tests for AgentRegistry"""

    def test_registry_initialization(self):
        """Registry should initialize with default agents"""
        registry = AgentRegistry()
        registry.initialize()

        assert len(registry) == 8
        assert registry.get_agent("internet_agent") is not None
        assert registry.get_agent("analysis_agent") is not None

    def test_get_agents_by_capability(self):
        """Registry should filter agents by capability"""
        registry = AgentRegistry()
        registry.initialize()

        web_agents = registry.get_agents_by_capability("web_search")
        assert len(web_agents) >= 1
        assert any(a.agent_id == "internet_agent" for a in web_agents)

    def test_get_agents_by_tag(self):
        """Registry should filter agents by tag"""
        registry = AgentRegistry()
        registry.initialize()

        search_agents = registry.get_agents_by_tag("search")
        assert len(search_agents) >= 1


class TestPlanCreation:
    """Tests for query planning"""

    @pytest.mark.asyncio
    async def test_simple_query_plan(self):
        """Should create plan for simple query"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        plan = await orchestrator.create_plan_only("삼성전자 주가 알려줘")

        assert plan is not None
        assert plan.get_stage_count() >= 1
        assert plan.get_total_agents() >= 1
        assert plan.workflow_strategy in ["sequential", "parallel", "hybrid"]

    @pytest.mark.asyncio
    async def test_visualization_query_plan(self):
        """Should create multi-stage plan for visualization query"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        plan = await orchestrator.create_plan_only("삼성전자 5일 종가 그래프로 그려줘")

        assert plan is not None
        # Should have data collection, analysis, visualization stages
        assert plan.get_stage_count() >= 2
        assert plan.get_total_agents() >= 2

    @pytest.mark.asyncio
    async def test_comparison_query_plan(self):
        """Should create parallel plan for comparison query"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        plan = await orchestrator.create_plan_only("삼성전자와 애플 비교해줘")

        assert plan is not None
        # Comparison queries should use parallel or hybrid strategy
        assert plan.workflow_strategy in ["parallel", "hybrid"]


class TestWorkflowExecution:
    """Tests for workflow execution"""

    @pytest.mark.asyncio
    async def test_full_execution(self):
        """Should execute complete workflow"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
            enable_streaming=True,
        )

        result = await orchestrator.run("삼성전자 5일 종가 그래프로 그려줘")

        assert result.success is True
        assert result.total_agents_executed >= 1
        assert result.successful_agents == result.total_agents_executed
        assert result.failed_agents == 0
        assert result.final_output is not None

    @pytest.mark.asyncio
    async def test_execution_with_failure_recovery(self):
        """Should handle agent failures gracefully"""
        call_count = {"count": 0}

        async def failing_executor(agent_id, sub_query, context):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Simulated failure")
            return {"result": "success after retry"}

        orchestrator = WorkflowOrchestrator(
            agent_executor=failing_executor,
            enable_validation=True,
        )

        # Should still work due to retry logic
        result = await orchestrator.run("간단한 질문")

        # At least one retry should have happened
        assert call_count["count"] >= 2


class TestStreaming:
    """Tests for progress streaming"""

    @pytest.mark.asyncio
    async def test_streaming_events(self):
        """Should emit progress events during execution"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
            enable_streaming=True,
        )

        events = []
        async for event in orchestrator.run_streaming("환율 정보 알려줘"):
            events.append(event)

        # Should have key events
        event_types = [e.type for e in events]

        assert ProgressEventType.PLANNING_START in event_types
        assert ProgressEventType.PLANNING_COMPLETE in event_types
        assert ProgressEventType.WORKFLOW_START in event_types
        assert ProgressEventType.WORKFLOW_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_streaming_has_korean_messages(self):
        """Should include Korean messages in events"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
            enable_streaming=True,
        )

        events = []
        async for event in orchestrator.run_streaming("삼성전자 주가"):
            events.append(event)

        # Check Korean messages exist
        korean_messages = [e.message_ko for e in events if e.message_ko]
        assert len(korean_messages) > 0

    @pytest.mark.asyncio
    async def test_streaming_progress_increases(self):
        """Progress percentage should generally increase"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
            enable_streaming=True,
        )

        progress_values = []
        async for event in orchestrator.run_streaming("간단한 검색"):
            if event.progress_percent is not None:
                progress_values.append(event.progress_percent)

        # Should have some progress values
        assert len(progress_values) > 0

        # Final progress should be higher than initial
        if len(progress_values) > 1:
            assert progress_values[-1] >= progress_values[0]


class TestSchemaCompatibility:
    """Tests for agent schema compatibility"""

    @pytest.mark.asyncio
    async def test_text_to_query_chain(self):
        """Should allow text output to query input chain"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        # This query should chain: internet_agent (text) -> llm_search_agent (query)
        result = await orchestrator.run("환율 그래프 보여주고 왜 그런지 알려줘")

        # Should succeed without schema errors
        assert result.success is True

    @pytest.mark.asyncio
    async def test_json_to_structured_data_chain(self):
        """Should allow json output to structured_data input chain"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        # This query should chain: analysis_agent (json) -> data_visualization_agent (json)
        result = await orchestrator.run("데이터 분석하고 차트로 보여줘")

        # Should succeed without schema errors
        assert result.success is True


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Should handle empty query gracefully"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        # Empty query might fail at planning, but shouldn't crash
        try:
            result = await orchestrator.run("")
            # If it succeeds, that's fine
        except Exception as e:
            # Should be a handled error, not a crash
            assert "error" in str(e).lower() or "fail" in str(e).lower()

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Should handle very long query"""
        orchestrator = WorkflowOrchestrator(
            agent_executor=mock_agent_executor,
            enable_validation=True,
        )

        long_query = "삼성전자 주가 " * 100  # Very long query
        plan = await orchestrator.create_plan_only(long_query)

        # Should create a plan even for long queries
        assert plan is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
