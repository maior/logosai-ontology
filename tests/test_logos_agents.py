"""Logos 기본 에이전트 목록 — SDK 에서 옮겨 온 인스턴스 데이터.

fixtures/sdk_default_agents_before.json 은 옮기기 **전** SDK DEFAULT_AGENTS 를 전체
필드(dataclasses.asdict)로 뜬 것이다. 12개가 순서까지 한 글자도 다르지 않아야 한다.
유령 rag_search_agent 는 일부러 남겼다 — 이유는 logos_agents 모듈 docstring.
"""
import dataclasses
import json
from pathlib import Path

from ontology.orchestrator import AgentRegistry
from ontology.orchestrator.logos_agents import KNOWN_GHOSTS, logos_default_agents

BEFORE = Path(__file__).resolve().parent / "fixtures" / "sdk_default_agents_before.json"


def _asdict(e):
    return json.loads(json.dumps(dataclasses.asdict(e), ensure_ascii=False, default=str))


def test_moved_verbatim():
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    now = [_asdict(e) for e in logos_default_agents()]
    assert len(before) == 12                                       # 대조군 — 기록이 비어 있지 않다
    assert now == before


def test_known_ghost_is_still_listed_on_purpose():
    """유령을 뺄 때 이 테스트를 뒤집는다 — 정답 기준 평가와 함께 (logos_agents docstring)."""
    assert KNOWN_GHOSTS == ("rag_search_agent",)
    assert "rag_search_agent" in [e.agent_id for e in logos_default_agents()]


def test_each_call_returns_fresh_entries():
    a, b = logos_default_agents(), logos_default_agents()
    assert [x.agent_id for x in a] == [x.agent_id for x in b]
    assert all(x is not y for x, y in zip(a, b))
    a[0].description = "변경"
    assert logos_default_agents()[0].description != "변경"


def test_registry_with_logos_defaults_keeps_their_order_first():
    reg = AgentRegistry(defaults=logos_default_agents())
    ids = reg.get_agent_ids()
    assert ids[:2] == ["internet_agent", "weather_agent"] and len(ids) == 12


def test_plain_registry_has_no_logos_defaults():
    """ontology 경로로 만들어도 SDK 와 같은 클래스다 — 기본값은 명시적으로만 들어간다."""
    assert AgentRegistry().get_agent_ids() == []


def test_workflow_orchestrator_keeps_an_injected_empty_registry(monkeypatch):
    """logos_api 는 빈 AgentRegistry() 를 넘긴 뒤 DB 에이전트로 채운다 — 그 객체가 쓰여야 한다."""
    from ontology.orchestrator import WorkflowOrchestrator

    mine = AgentRegistry()

    async def executor(agent_id, query, context):
        return {"success": True}

    from ontology.orchestrator import QueryPlanner
    monkeypatch.setenv("GOOGLE_API_KEY", "wiring-test-key")
    monkeypatch.setattr(QueryPlanner, "USE_HYBRID_SELECTOR", False)
    orch = WorkflowOrchestrator(agent_executor=executor, registry=mine,
                                enable_validation=True, enable_streaming=True)
    orch._init_components(None)          # 구성 요소는 실행 시점에 만들어진다
    parts = {"orchestrator": orch.registry, "planner": orch._planner.registry,
             "validator": orch._validator.registry, "transformer": orch._transformer.registry,
             "engine": orch._engine.registry}
    wrong = [k for k, v in parts.items() if v is not mine]
    assert wrong == [], f"넘긴 빈 레지스트리를 버리고 전역을 쓴다: {wrong}"
