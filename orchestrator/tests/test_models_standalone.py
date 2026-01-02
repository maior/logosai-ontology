#!/usr/bin/env python
"""
Standalone unit tests for orchestrator models

Run with:
    python ontology/orchestrator/tests/test_models_standalone.py
"""

import sys
import os

# Add orchestrator directory to path BEFORE any imports
_orchestrator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _orchestrator_dir)

# Now we can import models directly
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


def test_schema_any_compatible():
    """'any' type should be compatible with all types"""
    any_schema = AgentSchema(input_type="any", output_type="any")
    query_schema = AgentSchema(input_type="query", output_type="text")

    assert any_schema.is_compatible_with(query_schema), "any should be compatible with query"
    assert query_schema.is_compatible_with(any_schema), "query should be compatible with any"
    print("  ✅ test_schema_any_compatible")


def test_schema_text_to_query():
    """text output should be compatible with query input"""
    text_output = AgentSchema(input_type="any", output_type="text")
    query_input = AgentSchema(input_type="query", output_type="any")

    assert text_output.is_compatible_with(query_input), "text should convert to query"
    print("  ✅ test_schema_text_to_query")


def test_schema_text_to_structured_data():
    """text output should be compatible with structured_data input"""
    text_output = AgentSchema(input_type="any", output_type="text")
    structured_input = AgentSchema(input_type="structured_data", output_type="any")

    assert text_output.is_compatible_with(structured_input), "text should convert to structured_data"
    print("  ✅ test_schema_text_to_structured_data")


def test_schema_json_to_query():
    """json output should be compatible with query input"""
    json_output = AgentSchema(input_type="any", output_type="json")
    query_input = AgentSchema(input_type="query", output_type="any")

    assert json_output.is_compatible_with(query_input), "json should convert to query"
    print("  ✅ test_schema_json_to_query")


def test_schema_json_to_structured_data():
    """json output should be compatible with structured_data input"""
    json_output = AgentSchema(input_type="any", output_type="json")
    structured_input = AgentSchema(input_type="structured_data", output_type="any")

    assert json_output.is_compatible_with(structured_input), "json should convert to structured_data"
    print("  ✅ test_schema_json_to_structured_data")


def test_schema_structured_to_query():
    """structured_data output should be compatible with query input"""
    structured_output = AgentSchema(input_type="any", output_type="structured_data")
    query_input = AgentSchema(input_type="query", output_type="any")

    assert structured_output.is_compatible_with(query_input), "structured_data should convert to query"
    print("  ✅ test_schema_structured_to_query")


def test_schema_incompatible():
    """Incompatible types should return False"""
    html_output = AgentSchema(input_type="any", output_type="html")
    json_input = AgentSchema(input_type="json", output_type="any")

    assert not html_output.is_compatible_with(json_input), "html should not be compatible with json"
    print("  ✅ test_schema_incompatible")


def test_progress_event_to_dict():
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
    print("  ✅ test_progress_event_to_dict")


def test_progress_event_to_json():
    """ProgressEvent should serialize to JSON string"""
    event = ProgressEvent(
        type=ProgressEventType.WORKFLOW_START,
        message="Workflow started",
    )

    json_str = event.to_json()

    assert '"type": "workflow_start"' in json_str
    assert '"message": "Workflow started"' in json_str
    print("  ✅ test_progress_event_to_json")


def test_progress_event_to_sse():
    """ProgressEvent should serialize to SSE format"""
    event = ProgressEvent(
        type=ProgressEventType.STAGE_COMPLETE,
        message="Stage completed",
    )

    sse_str = event.to_sse()

    assert sse_str.startswith("data: ")
    assert sse_str.endswith("\n\n")
    assert "stage_complete" in sse_str
    print("  ✅ test_progress_event_to_sse")


def test_execution_plan_counts():
    """ExecutionPlan should count agents correctly"""
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
    print("  ✅ test_execution_plan_counts")


def test_execution_plan_to_dict():
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
    print("  ✅ test_execution_plan_to_dict")


def test_stage_result_filters():
    """StageResult should filter results correctly"""
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
    print("  ✅ test_stage_result_filters")


def test_workflow_result_to_dict():
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
    print("  ✅ test_workflow_result_to_dict")


def main():
    print("\n" + "=" * 60)
    print("🧪 Orchestrator Models Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Schema Compatibility", [
            test_schema_any_compatible,
            test_schema_text_to_query,
            test_schema_text_to_structured_data,
            test_schema_json_to_query,
            test_schema_json_to_structured_data,
            test_schema_structured_to_query,
            test_schema_incompatible,
        ]),
        ("ProgressEvent Serialization", [
            test_progress_event_to_dict,
            test_progress_event_to_json,
            test_progress_event_to_sse,
        ]),
        ("ExecutionPlan", [
            test_execution_plan_counts,
            test_execution_plan_to_dict,
        ]),
        ("StageResult", [
            test_stage_result_filters,
        ]),
        ("WorkflowResult", [
            test_workflow_result_to_dict,
        ]),
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for group_name, test_funcs in tests:
        print(f"📦 {group_name}")
        for test_func in test_funcs:
            total_tests += 1
            try:
                test_func()
                passed_tests += 1
            except AssertionError as e:
                failed_tests.append((test_func.__name__, str(e)))
                print(f"  ❌ {test_func.__name__}: {e}")
            except Exception as e:
                failed_tests.append((test_func.__name__, str(e)))
                print(f"  ❌ {test_func.__name__}: {e}")
        print()

    print("=" * 60)
    if failed_tests:
        print(f"❌ FAILED: {len(failed_tests)}/{total_tests} tests failed")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
    else:
        print(f"✅ PASSED: {passed_tests}/{total_tests} tests passed")
    print("=" * 60 + "\n")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
