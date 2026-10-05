"""ontology WorkflowOrchestrator — SDK 오케스트레이터를 상속한 Logos 인스턴스 (P3).

본문은 logosai.orchestration.workflow_orchestrator 가 정본이고, 여기서는 플래너만
Logos 플래너(Gemini·Logos 프롬프트·관문)로 바꾼다. 동작이 이전과 같음은
test_orchestrator_characterization.py 가 확인한다.
"""
import pytest

import ontology.orchestrator as O
from logosai.orchestration import workflow_orchestrator as sdk


def _orch(monkeypatch, **kw):
    monkeypatch.setenv("GOOGLE_API_KEY", "instance-test-key")
    monkeypatch.setattr(O.QueryPlanner, "USE_HYBRID_SELECTOR", False)

    async def executor(agent_id, query, context):
        return {"success": True}
    return O.WorkflowOrchestrator(agent_executor=executor, registry=O.AgentRegistry(), **kw)


def test_is_the_sdk_orchestrator_with_the_logos_planner(monkeypatch):
    orch = _orch(monkeypatch)
    assert isinstance(orch, sdk.WorkflowOrchestrator) and type(orch) is not sdk.WorkflowOrchestrator
    orch._init_components(None)
    assert type(orch._planner) is O.QueryPlanner                 # Logos 플래너


def test_llm_invoke_is_rejected_not_silently_ignored(monkeypatch):
    """Logos 플래너는 Gemini 를 직접 부른다 — 주입한 LLM 이 안 쓰이는데 받아 주면 거짓이다."""
    with pytest.raises(TypeError, match="llm_invoke"):
        _orch(monkeypatch, llm_invoke=lambda p: None)


def test_factories_build_the_logos_instance(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "instance-test-key")
    from ontology.orchestrator.workflow_orchestrator import create_orchestrator
    assert type(create_orchestrator()) is O.WorkflowOrchestrator
