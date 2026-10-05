"""
Ontology schema layer tests — TBox validation, edge dedup,
transitive inference, and agents.json-based sync.
"""

import asyncio
import json

import pytest


def run(coro):
    """Run an async engine call from a sync test."""
    return asyncio.run(coro)


@pytest.fixture()
def kg():
    """Fresh (non-singleton) knowledge graph engine."""
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
    return KnowledgeGraphEngine(fast_mode=True)


# ─── 1. Schema (TBox) ───────────────────────────────────────────────

class TestOntologySchema:
    def test_relation_definitions_exist(self):
        from ontology.core.ontology_schema import RELATIONS
        for predicate in ("is_a", "has_capability", "has_tag",
                          "has_mapping", "belongs_to_category"):
            assert predicate in RELATIONS, f"{predicate} missing from schema"

    def test_is_a_is_transitive(self):
        from ontology.core.ontology_schema import RELATIONS
        assert RELATIONS["is_a"].transitive is True
        assert RELATIONS["has_capability"].transitive is False

    def test_validate_relation_accepts_correct_types(self):
        from ontology.core.ontology_schema import validate_relation
        ok, _ = validate_relation("has_capability", "agent", "capability")
        assert ok is True

    def test_validate_relation_rejects_wrong_domain(self):
        from ontology.core.ontology_schema import validate_relation
        ok, msg = validate_relation("has_capability", "tag", "capability")
        assert ok is False
        assert "has_capability" in msg

    def test_validate_relation_unknown_predicate_passes_with_note(self):
        # Unknown predicates must not break the system (warn-only policy).
        from ontology.core.ontology_schema import validate_relation
        ok, msg = validate_relation("totally_new_predicate", "a", "b")
        assert ok is True
        assert msg  # a note is returned

    def test_class_hierarchy_ancestors(self):
        from ontology.core.ontology_schema import get_class_ancestors
        ancestors = get_class_ancestors("agent")
        assert "entity" in ancestors
        # Unknown type has no ancestors, not an error
        assert get_class_ancestors("no_such_type") == []


# ─── 2. Edge deduplication on write ─────────────────────────────────

class TestEdgeDedup:
    def test_same_relation_twice_creates_one_edge(self, kg):
        run(kg.add_concept("a1", "agent", {"name": "a1"}))
        run(kg.add_concept("capability_x", "capability", {"name": "x"}))
        run(kg.add_relationship("a1", "capability_x", "has_capability"))
        run(kg.add_relationship("a1", "capability_x", "has_capability"))

        edges = kg.graph.get_edge_data("a1", "capability_x")
        same = [e for e in edges.values() if e.get("predicate") == "has_capability"]
        assert len(same) == 1, f"expected 1 edge, got {len(same)}"

    def test_duplicate_write_updates_existing_edge(self, kg):
        run(kg.add_concept("a1", "agent", {}))
        run(kg.add_concept("t1", "tag", {}))
        run(kg.add_relationship("a1", "t1", "has_tag", {"weight": 1.0}))
        run(kg.add_relationship("a1", "t1", "has_tag", {"weight": 2.0}))

        edges = kg.graph.get_edge_data("a1", "t1")
        edge = [e for e in edges.values() if e.get("predicate") == "has_tag"][0]
        assert edge["weight"] == 2.0  # attributes refreshed, not duplicated

    def test_different_predicates_keep_separate_edges(self, kg):
        run(kg.add_concept("a1", "agent", {}))
        run(kg.add_concept("m1", "query_agent_mapping", {}))
        run(kg.add_relationship("a1", "m1", "has_mapping"))
        run(kg.add_relationship("a1", "m1", "produces"))

        edges = kg.graph.get_edge_data("a1", "m1")
        predicates = {e.get("predicate") for e in edges.values()}
        assert predicates == {"has_mapping", "produces"}


# ─── 3. Transitive inference (is_a) ────────────────────────────────

class TestInference:
    def _seed_taxonomy(self, kg):
        run(kg.add_concept("weather_query", "query_category", {}))
        run(kg.add_concept("information_query", "query_category", {}))
        run(kg.add_concept("query", "query_category", {}))
        run(kg.add_relationship("weather_query", "information_query", "is_a"))
        run(kg.add_relationship("information_query", "query", "is_a"))

    def test_get_ancestors_transitive(self, kg):
        self._seed_taxonomy(kg)
        assert kg.get_ancestors("weather_query") == ["information_query", "query"]

    def test_get_descendants_transitive(self, kg):
        self._seed_taxonomy(kg)
        assert set(kg.get_descendants("query")) == {"information_query", "weather_query"}

    def test_ancestors_of_unknown_node_is_empty(self, kg):
        assert kg.get_ancestors("nope") == []

    def test_cycle_does_not_hang(self, kg):
        run(kg.add_concept("x", "query_category", {}))
        run(kg.add_concept("y", "query_category", {}))
        run(kg.add_relationship("x", "y", "is_a"))
        run(kg.add_relationship("y", "x", "is_a"))
        ancestors = kg.get_ancestors("x")
        assert "y" in ancestors  # terminates and returns


# ─── 4. Capability / tag queries ───────────────────────────────────

class TestCapabilityQueries:
    def _seed_agents(self, kg):
        run(kg.add_concept("weather_agent", "agent", {"name": "weather"}))
        run(kg.add_concept("news_agent", "agent", {"name": "news"}))
        run(kg.add_concept("capability_web_search", "capability", {"name": "web_search"}))
        run(kg.add_concept("capability_realtime_search", "capability", {"name": "realtime_search"}))
        run(kg.add_relationship("weather_agent", "capability_realtime_search", "has_capability"))
        run(kg.add_relationship("news_agent", "capability_web_search", "has_capability"))
        # realtime_search is a kind of web_search
        run(kg.add_relationship("capability_realtime_search", "capability_web_search", "is_a"))
        run(kg.add_concept("tag_weather", "tag", {"name": "weather"}))
        run(kg.add_relationship("weather_agent", "tag_weather", "has_tag"))

    def test_find_agents_by_capability_direct(self, kg):
        self._seed_agents(kg)
        agents = kg.find_agents_by_capability("realtime_search", include_inherited=False)
        assert agents == ["weather_agent"]

    def test_find_agents_by_capability_inherited(self, kg):
        # Searching the parent capability also finds agents holding the child.
        self._seed_agents(kg)
        agents = kg.find_agents_by_capability("web_search", include_inherited=True)
        assert set(agents) == {"news_agent", "weather_agent"}

    def test_find_agents_by_capability_accepts_prefixed_id(self, kg):
        self._seed_agents(kg)
        agents = kg.find_agents_by_capability("capability_web_search",
                                              include_inherited=False)
        assert agents == ["news_agent"]

    def test_find_agents_by_tag(self, kg):
        self._seed_agents(kg)
        assert kg.find_agents_by_tag("weather") == ["weather_agent"]
        assert kg.find_agents_by_tag("no_such_tag") == []

    def test_get_agent_profile(self, kg):
        self._seed_agents(kg)
        run(kg.add_concept("m1", "query_agent_mapping",
                           {"success_rate": 0.9, "generalization_pattern": "p"}))
        run(kg.add_relationship("weather_agent", "m1", "has_mapping"))

        profile = kg.get_agent_profile("weather_agent")
        assert profile["agent_id"] == "weather_agent"
        assert "realtime_search" in profile["capabilities"]
        assert "weather" in profile["tags"]
        assert len(profile["success_patterns"]) == 1

    def test_get_agent_profile_unknown_agent(self, kg):
        assert kg.get_agent_profile("ghost_agent") == {}


# ─── 5. Cleanup of pre-existing duplicates ──────────────────────────

class TestDeduplicateEdges:
    def test_removes_duplicates_keeps_one_per_predicate(self, kg):
        kg.graph.add_node("a", type="agent")
        kg.graph.add_node("t", type="tag")
        for _ in range(5):
            kg.graph.add_edge("a", "t", predicate="has_tag", weight=1.0)
        kg.graph.add_edge("a", "t", predicate="has_capability")

        removed = kg.deduplicate_edges()
        assert removed == 4
        edges = kg.graph.get_edge_data("a", "t")
        predicates = sorted(e.get("predicate") for e in edges.values())
        assert predicates == ["has_capability", "has_tag"]

    def test_noop_on_clean_graph(self, kg):
        run(kg.add_concept("a", "agent", {}))
        run(kg.add_concept("t", "tag", {}))
        run(kg.add_relationship("a", "t", "has_tag"))
        assert kg.deduplicate_edges() == 0


# ─── 6. Sync reads agents.json (acp config) ────────────────────────

class TestSyncFromAcpConfig:
    @pytest.fixture()
    def sync_env(self, tmp_path):
        """Isolated sync service pointing at a temp agents.json."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        # one code-only agent with no metadata in config
        (agents_dir / "lonely_agent.py").write_text(
            'class LonelyAgent:\n    """Code-only agent"""\n', encoding="utf-8")

        config_file = tmp_path / "agents.json"
        config_file.write_text(json.dumps({
            "agents": [{
                "agent_id": "rich_agent",
                "name": "리치 에이전트",
                "description": "설정 파일 기반 에이전트",
                "capabilities": [
                    {"id": "web_search", "name": "웹 검색", "description": "검색"},
                    {"id": "summarize", "name": "요약", "description": "요약"},
                ],
                "tags": ["검색", "요약"],
            }]
        }, ensure_ascii=False), encoding="utf-8")

        from ontology.core.agent_sync_service import AgentSyncService
        svc = AgentSyncService(
            agents_dir=agents_dir,
            metadata_file=tmp_path / "missing_metadata.json",
            acp_config_file=config_file,
            knowledge_graph=None,
            agent_registry=None,
        )
        return svc

    def test_config_capabilities_extracted_as_ids(self, sync_env):
        config_agents = run(sync_env._load_acp_config())
        assert "rich_agent" in config_agents
        info = config_agents["rich_agent"]
        assert info["capabilities"] == ["web_search", "summarize"]
        assert info["tags"] == ["검색", "요약"]
        assert info["description"] == "설정 파일 기반 에이전트"

    def test_merge_prefers_config_over_code_parse(self, sync_env):
        merged = run(sync_env.collect_agents())
        # config-defined agent keeps its rich metadata
        assert merged["rich_agent"]["capabilities"] == ["web_search", "summarize"]

        # Contract reversed 2026-07-31: a code-only agent is NOT discovered.
        #
        # This previously asserted `"lonely_agent" in merged` — membership was a union
        # across config + code scan + legacy metadata. Measured consequence: the
        # Knowledge Graph held 168 agent nodes against 100 in agents.json, and 21 of
        # those extras were unregistered *_agent.py files re-created on every sync.
        # An unregistered file is not runnable — ACP loads agents.json — so declaring
        # it as an agent is a claim the runtime cannot honour.
        #
        # The code scan remains a *field* source (see the next assertion); it is no
        # longer a membership source. See test_agent_sync_reconcile.py.
        assert "lonely_agent" not in merged

    def test_missing_config_file_is_not_fatal(self, tmp_path):
        from ontology.core.agent_sync_service import AgentSyncService
        svc = AgentSyncService(
            agents_dir=tmp_path,
            metadata_file=tmp_path / "m.json",
            acp_config_file=tmp_path / "no_such.json",
            knowledge_graph=None,
            agent_registry=None,
        )
        assert run(svc._load_acp_config()) == {}
