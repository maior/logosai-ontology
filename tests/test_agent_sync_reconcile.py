"""Agent sync reconciliation — deregistered agents must leave the candidate pool.

Background (2026-07-31): the Knowledge Graph held 168 agent nodes while the registry
of record (acp_server/configs/agents.json) held 100. The 68 extras came from two
defects, both covered here:

1. The sync was add-only. Agents deregistered from agents.json were never removed or
   marked, so they lingered indefinitely.
2. Membership was a *union* of three sources — the config, a scan of
   acp_server/agents/*_agent.py, and the legacy agents/config/agent_metadata.json —
   so unregistered code files and stale metadata entries were synced in as if live.

The fix deactivates (never deletes) stale nodes: they carry learning history
(query_agent_mapping edges, capability/tag links, success rates) that deletion would
sever. `is_available=False` is the record that the agent is gone.
"""

import asyncio
import json
from pathlib import Path

import pytest

from ontology.core.agent_sync_service import AgentSyncService
from ontology.orchestrator.agent_registry import AgentRegistry
from ontology.orchestrator.models import AgentRegistryEntry, AgentSchema


# ---------------------------------------------------------------------------
# Fakes — no LLM, no disk KG, no network
# ---------------------------------------------------------------------------

class FakeGraph:
    """Minimal networkx-shaped stand-in: nodes(data=True) yielding mutable dicts."""

    def __init__(self):
        self._nodes = {}

    def add(self, node_id, **attrs):
        self._nodes[node_id] = dict(attrs)

    def nodes(self, data=False):
        if data:
            return list(self._nodes.items())
        return list(self._nodes)

    def attrs(self, node_id):
        return self._nodes[node_id]


class FakeGraphEngine:
    def __init__(self):
        self.graph = FakeGraph()


class FakeKnowledgeGraph:
    def __init__(self):
        self.graph_engine = FakeGraphEngine()

    async def add_concept(self, concept_id, concept_type, attributes):
        existing = self.graph_engine.graph._nodes.get(concept_id, {})
        existing.update({"type": concept_type, **attributes})
        self.graph_engine.graph._nodes[concept_id] = existing
        return True

    async def add_relationship(self, source, target, relationship):
        return True


def _write_config(tmp_path: Path, agent_ids) -> Path:
    cfg = tmp_path / "agents.json"
    cfg.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "agent_id": a,
                        "name": a,
                        "description": f"{a} description",
                        "capabilities": [],
                        "tags": [],
                    }
                    for a in agent_ids
                ]
            }
        ),
        encoding="utf-8",
    )
    return cfg


def _make_service(tmp_path, config_ids, code_files=(), metadata_ids=()):
    """Build a service whose three sources can be set independently."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(exist_ok=True)
    for name in code_files:
        (agents_dir / f"{name}.py").write_text(
            'class FooAgent:\n    description = "scanned from code"\n', encoding="utf-8"
        )

    meta_file = tmp_path / "agent_metadata.json"
    if metadata_ids:
        meta_file.write_text(
            json.dumps({m: {"name": m, "description": "legacy"} for m in metadata_ids}),
            encoding="utf-8",
        )

    return AgentSyncService(
        agents_dir=agents_dir,
        metadata_file=meta_file,
        acp_config_file=_write_config(tmp_path, config_ids),
        knowledge_graph=FakeKnowledgeGraph(),
        agent_registry=AgentRegistry(),
    )


def _entry(agent_id):
    return AgentRegistryEntry(
        agent_id=agent_id,
        name=agent_id,
        description="",
        schema=AgentSchema(input_type="query", output_type="text"),
    )


# ---------------------------------------------------------------------------
# The headline regression: a deregistered agent must leave the candidate pool
# ---------------------------------------------------------------------------

def test_deregistered_agent_drops_out_of_selection_candidates(tmp_path):
    """The property that matters: after a sync, an agent removed from the registry
    of record is no longer offered to the selector as a candidate.

    `available_agents` handed to HybridAgentSelector.select_agent() comes from
    AgentRegistry.get_available_agents(), which filters on is_available.
    """
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    candidates = {e.agent_id for e in service.agent_registry.get_available_agents()}
    assert candidates == {"alpha_agent", "beta_agent"}

    # beta is deregistered
    _write_config(tmp_path, ["alpha_agent"])
    service.acp_config_file = tmp_path / "agents.json"
    result = asyncio.run(service.full_sync())

    candidates = {e.agent_id for e in service.agent_registry.get_available_agents()}
    assert "beta_agent" not in candidates, "deregistered agent still a selection candidate"
    assert candidates == {"alpha_agent"}
    assert "beta_agent" in result["removed"]


def test_deregistered_agent_is_deactivated_not_deleted(tmp_path):
    """Deactivation preserves the node and its learning history."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    _write_config(tmp_path, ["alpha_agent"])
    asyncio.run(service.full_sync())

    graph = service.knowledge_graph.graph_engine.graph
    assert "beta_agent" in graph.nodes(), "node deleted — learning history severed"
    assert graph.attrs("beta_agent")["is_available"] is False
    assert "deactivated_at" in graph.attrs("beta_agent")

    # registry entry survives too, merely unavailable
    entry = service.agent_registry.get_agent_safe("beta_agent")
    assert entry is not None
    assert entry.is_available is False


def test_returning_agent_is_reactivated(tmp_path):
    """Re-registering an agent must bring it back — deactivation is not a tombstone."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    _write_config(tmp_path, ["alpha_agent"])
    asyncio.run(service.full_sync())
    assert service.knowledge_graph.graph_engine.graph.attrs("beta_agent")["is_available"] is False

    _write_config(tmp_path, ["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    assert service.knowledge_graph.graph_engine.graph.attrs("beta_agent")["is_available"] is True
    candidates = {e.agent_id for e in service.agent_registry.get_available_agents()}
    assert "beta_agent" in candidates


def test_deactivation_does_not_rewrite_existing_timestamp(tmp_path):
    """A node already deactivated must not have its deactivated_at refreshed on every
    sync — that would destroy the record of when the agent actually disappeared."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())
    _write_config(tmp_path, ["alpha_agent"])

    result_first = asyncio.run(service.full_sync())
    stamp = service.knowledge_graph.graph_engine.graph.attrs("beta_agent")["deactivated_at"]
    assert "beta_agent" in result_first["deactivated_kg"]

    result_second = asyncio.run(service.full_sync())
    assert result_second["deactivated_kg"] == [], "re-reported an already-deactivated agent"
    assert (
        service.knowledge_graph.graph_engine.graph.attrs("beta_agent")["deactivated_at"]
        == stamp
    )


# ---------------------------------------------------------------------------
# Root cause 2: membership comes from the registry of record, not a union
# ---------------------------------------------------------------------------

def test_unregistered_code_file_is_not_synced(tmp_path):
    """An *_agent.py file that was never registered must not become an agent.

    This is how 21 of the 68 KG ghosts were being (re)created on every sync.
    """
    service = _make_service(
        tmp_path,
        config_ids=["alpha_agent"],
        code_files=["ghost_agent", "alpha_agent"],
    )
    merged = asyncio.run(service.collect_agents())

    assert set(merged) == {"alpha_agent"}
    assert "ghost_agent" not in merged


def test_legacy_metadata_file_does_not_add_members(tmp_path):
    """The legacy agent_metadata.json contributes fields, never membership."""
    service = _make_service(
        tmp_path,
        config_ids=["alpha_agent"],
        metadata_ids=["rag_search", "supervisor"],
    )
    merged = asyncio.run(service.collect_agents())

    assert set(merged) == {"alpha_agent"}
    assert "rag_search_agent" not in merged
    assert "supervisor_agent" not in merged


def test_code_scan_still_supplies_fields(tmp_path):
    """Narrowing membership must not lose the code-scan metadata for real agents."""
    service = _make_service(
        tmp_path, config_ids=["alpha_agent"], code_files=["alpha_agent"]
    )
    merged = asyncio.run(service.collect_agents())

    assert merged["alpha_agent"]["class_name"] == "FooAgent"
    assert merged["alpha_agent"]["file_path"] is not None


def test_union_fallback_when_config_unavailable(tmp_path):
    """An unreadable config must not wipe the fleet — fall back to the union."""
    service = _make_service(
        tmp_path, config_ids=["alpha_agent"], code_files=["scanned_agent"]
    )
    service.acp_config_file = tmp_path / "does_not_exist.json"

    merged = asyncio.run(service.collect_agents())
    assert "scanned_agent" in merged


def test_empty_sources_do_not_deactivate_everything(tmp_path):
    """A collection failure must never be read as 'all agents are gone'."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    # every source now empty
    service.acp_config_file = tmp_path / "missing.json"
    service.agents_dir = tmp_path / "missing_dir"
    service.metadata_file = tmp_path / "missing_meta.json"

    result = asyncio.run(service.full_sync())

    assert result["removed"] == []
    assert result["deactivated_kg"] == []
    candidates = {e.agent_id for e in service.agent_registry.get_available_agents()}
    assert {"alpha_agent", "beta_agent"} <= candidates


# ---------------------------------------------------------------------------
# The watcher must not resurrect what full_sync just deactivated
# ---------------------------------------------------------------------------

def test_change_detection_ignores_unregistered_code_files(tmp_path):
    """check_for_changes previously scanned the agent directory directly, so every
    unregistered *_agent.py showed up as 'added' and the watcher synced it back in."""
    service = _make_service(
        tmp_path,
        config_ids=["alpha_agent"],
        code_files=["alpha_agent", "ghost_agent"],
    )
    asyncio.run(service.full_sync())

    changes = asyncio.run(service.check_for_changes())
    assert "ghost_agent" not in changes["added"]
    assert changes["added"] == []


def test_removal_is_reported_once(tmp_path):
    """Sync state must advance, or a removal repeats every watcher interval."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    _write_config(tmp_path, ["alpha_agent"])
    first = asyncio.run(service.check_for_changes())
    assert first["removed"] == ["beta_agent"]

    second = asyncio.run(service.check_for_changes())
    assert second["removed"] == []


def test_change_detection_survives_empty_collection(tmp_path):
    """An empty collection must report no removals at all."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    service.acp_config_file = tmp_path / "missing.json"
    service.agents_dir = tmp_path / "missing_dir"
    service.metadata_file = tmp_path / "missing_meta.json"

    changes = asyncio.run(service.check_for_changes())
    assert changes["removed"] == []


# ---------------------------------------------------------------------------
# Reconciliation must leave everything that is not an agent node alone
# ---------------------------------------------------------------------------

def test_only_agent_nodes_are_touched(tmp_path):
    service = _make_service(tmp_path, config_ids=["alpha_agent"])
    graph = service.knowledge_graph.graph_engine.graph
    graph.add("capability_foo", type="capability", name="foo")
    graph.add("mapping_x", type="query_agent_mapping", selected_agent="beta_agent")

    asyncio.run(service.full_sync())

    assert "is_available" not in graph.attrs("capability_foo")
    assert "is_available" not in graph.attrs("mapping_x")
    # the mapping's reference to a now-gone agent is preserved, not scrubbed
    assert graph.attrs("mapping_x")["selected_agent"] == "beta_agent"


def test_registry_entries_are_never_unregistered(tmp_path):
    """Deactivation keeps execution metadata that unregistering would drop."""
    service = _make_service(tmp_path, config_ids=["alpha_agent", "beta_agent"])
    asyncio.run(service.full_sync())

    entry = service.agent_registry.get_agent_safe("beta_agent")
    entry.success_rate = 0.93

    _write_config(tmp_path, ["alpha_agent"])
    asyncio.run(service.full_sync())

    still_there = service.agent_registry.get_agent_safe("beta_agent")
    assert still_there is not None
    assert still_there.success_rate == 0.93
    assert still_there.is_available is False
