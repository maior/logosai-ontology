"""
Unit tests for orchestrator models

Run with:
    cd /Users/maior/Development/skku/Logos
    python -m pytest ontology/orchestrator/tests/test_models.py -v
"""

import pytest
from datetime import datetime

import sys
import os

# Add orchestrator directory to path to import models directly
_orchestrator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _orchestrator_dir not in sys.path:
    sys.path.insert(0, _orchestrator_dir)

# Import models directly without going through package __init__.py
# This avoids circular import issues with logosai
from models import (
    AgentStatus,
    ProgressEventType,
    ProgressEvent,
    AgentSchema,
    AgentRegistryEntry,
    AgentTask,
    ExecutionStage,
    ExecutionPlan,
    AgentResult,
    StageResult,
    WorkflowResult,
)


class TestAgentSchema:
    """Tests for AgentSchema compatibility checking"""

    def test_any_is_compatible_with_everything(self):
        """'any' type should be compatible with all types"""
        any_schema = AgentSchema(input_type="any", output_type="any")
        query_schema = AgentSchema(input_type="query", output_type="text")

        assert any_schema.is_compatible_with(query_schema)
        assert query_schema.is_compatible_with(any_schema)

    def test_direct_type_match(self):
        """Same types should be compatible"""
        text_schema = AgentSchema(input_type="text", output_type="text")
        text_schema2 = AgentSchema(input_type="text", output_type="text")

        assert text_schema.is_compatible_with(text_schema2)

    def test_text_to_query_compatible(self):
        """text output should be compatible with query input"""
        text_output = AgentSchema(input_type="any", output_type="text")
        query_input = AgentSchema(input_type="query", output_type="any")

        assert text_output.is_compatible_with(query_input)

    def test_text_to_structured_data_compatible(self):
        """text output should be compatible with structured_data input"""
        text_output = AgentSchema(input_type="any", output_type="text")
        structured_input = AgentSchema(input_type="structured_data", output_type="any")

        assert text_output.is_compatible_with(structured_input)

    def test_json_to_query_compatible(self):
        """json output should be compatible with query input"""
        json_output = AgentSchema(input_type="any", output_type="json")
        query_input = AgentSchema(input_type="query", output_type="any")

        assert json_output.is_compatible_with(query_input)

    def test_json_to_structured_data_compatible(self):
        """json output should be compatible with structured_data input"""
        json_output = AgentSchema(input_type="any", output_type="json")
        structured_input = AgentSchema(input_type="structured_data", output_type="any")

        assert json_output.is_compatible_with(structured_input)

    def test_structured_data_to_query_compatible(self):
        """structured_data output should be compatible with query input"""
        structured_output = AgentSchema(input_type="any", output_type="structured_data")
        query_input = AgentSchema(input_type="query", output_type="any")

        assert structured_output.is_compatible_with(query_input)

    def test_incompatible_types(self):
        """Incompatible types should return False"""
        html_output = AgentSchema(input_type="any", output_type="html")
        json_input = AgentSchema(input_type="json", output_type="any")

        assert not html_output.is_compatible_with(json_input)


class TestProgressEvent:
    """Tests for ProgressEvent serialization"""

    def test_to_dict(self):
        """ProgressEvent should serialize to dict correctly"""
        event = ProgressEvent(
            type=ProgressEventType.AGENT_START,
            workflow_id="test-123",
            stage_id=1,
            agent_id="internet_agent",
            message="Agent started",
            message_ko="에이전트 시작",
            progress_percent=25.0,
        )

        data = event.to_dict()

        assert data["type"] == "agent_start"
        assert data["workflow_id"] == "test-123"
        assert data["stage_id"] == 1
        assert data["agent_id"] == "internet_agent"
        assert data["message"] == "Agent started"
        assert data["message_ko"] == "에이전트 시작"
        assert data["progress_percent"] == 25.0

    def test_to_json(self):
        """ProgressEvent should serialize to JSON string"""
        event = ProgressEvent(
            type=ProgressEventType.WORKFLOW_START,
            message="Workflow started",
        )

        json_str = event.to_json()

        assert '"type": "workflow_start"' in json_str
        assert '"message": "Workflow started"' in json_str

    def test_to_sse(self):
        """ProgressEvent should serialize to SSE format"""
        event = ProgressEvent(
            type=ProgressEventType.STAGE_COMPLETE,
            message="Stage completed",
        )

        sse_str = event.to_sse()

        assert sse_str.startswith("data: ")
        assert sse_str.endswith("\n\n")
        assert "stage_complete" in sse_str


class TestExecutionPlan:
    """Tests for ExecutionPlan"""

    def test_get_total_agents(self):
        """ExecutionPlan should count all agents correctly"""
        plan = ExecutionPlan(
            query="test query",
            workflow_strategy="sequential",
            stages=[
                ExecutionStage(
                    stage_id=1,
                    execution_type="sequential",
                    agents=[
                        AgentTask(agent_id="agent1", sub_query="q1"),
                        AgentTask(agent_id="agent2", sub_query="q2"),
                    ]
                ),
                ExecutionStage(
                    stage_id=2,
                    execution_type="parallel",
                    agents=[
                        AgentTask(agent_id="agent3", sub_query="q3"),
                    ]
                ),
            ]
        )

        assert plan.get_total_agents() == 3
        assert plan.get_stage_count() == 2

    def test_to_dict(self):
        """ExecutionPlan should serialize correctly"""
        plan = ExecutionPlan(
            query="삼성전자 주가",
            workflow_strategy="sequential",
            reasoning="Simple query",
            stages=[
                ExecutionStage(
                    stage_id=1,
                    execution_type="sequential",
                    agents=[
                        AgentTask(agent_id="internet_agent", sub_query="주가 검색"),
                    ]
                ),
            ]
        )

        data = plan.to_dict()

        assert data["query"] == "삼성전자 주가"
        assert data["workflow_strategy"] == "sequential"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["agents"][0]["agent_id"] == "internet_agent"


class TestStageResult:
    """Tests for StageResult"""

    def test_get_successful_results(self):
        """StageResult should filter successful results"""
        stage = StageResult(
            stage_id=1,
            execution_type="parallel",
            success=False,
            results=[
                AgentResult(agent_id="agent1", stage_id=1, success=True, data="result1"),
                AgentResult(agent_id="agent2", stage_id=1, success=False, error="failed"),
                AgentResult(agent_id="agent3", stage_id=1, success=True, data="result3"),
            ]
        )

        successful = stage.get_successful_results()
        failed = stage.get_failed_results()

        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0].agent_id == "agent2"


class TestWorkflowResult:
    """Tests for WorkflowResult"""

    def test_to_dict(self):
        """WorkflowResult should serialize correctly"""
        result = WorkflowResult(
            success=True,
            workflow_id="test-123",
            query="test query",
            final_output={"answer": "result"},
            total_time_ms=1500.0,
            total_agents_executed=3,
            successful_agents=3,
            failed_agents=0,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["workflow_id"] == "test-123"
        assert data["total_time_ms"] == 1500.0
        assert data["successful_agents"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
