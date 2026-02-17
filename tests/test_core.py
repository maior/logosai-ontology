"""
Core module tests — models, enums, interfaces, and public API.
"""

import pytest
from datetime import datetime


# ─── Public Import Tests ────────────────────────────────────────────

class TestPublicImports:
    """Verify that the public API is importable."""

    def test_root_package_imports(self):
        from ontology import (
            SemanticQuery,
            ExecutionContext,
            AgentExecutionResult,
            WorkflowPlan,
            ExecutionStrategy,
            QueryAnalyzer,
            ExecutionEngine,
            DataTransformer,
            ResultProcessor,
            SemanticQueryManager,
            AdvancedExecutionEngine,
            SmartWorkflowDesigner,
            KnowledgeGraphEngine,
            OntologySystem,
        )

    def test_core_models_import(self):
        from ontology.core.models import (
            QueryType,
            ExecutionStrategy,
            AgentType,
            ExecutionStatus,
            DataTransformationType,
            WorkflowComplexity,
            OptimizationStrategy,
            LLMProvider,
            OntologyLLMType,
            OntologyLLMConfig,
            SemanticQuery,
            ExecutionContext,
            AgentExecutionResult,
            WorkflowPlan,
        )

    def test_core_interfaces_import(self):
        from ontology.core.interfaces import (
            QueryAnalyzer,
            ExecutionEngine,
            DataTransformer,
            ResultProcessor,
            WorkflowDesigner,
            CacheManager,
            KnowledgeGraph,
            AgentCaller,
            SystemMonitor,
        )

    def test_engines_import(self):
        from ontology.engines import (
            SemanticQueryManager,
            InMemoryCacheManager,
            AdvancedExecutionEngine,
            SmartDataTransformer,
            SmartWorkflowDesigner,
            KnowledgeGraphEngine,
        )

    def test_orchestrator_import(self):
        from ontology.orchestrator import (
            AgentSchema,
            AgentRegistryEntry,
            ExecutionPlan,
            WorkflowResult,
            ProgressEvent,
            ProgressEventType,
            AgentRegistry,
            QueryPlanner,
            WorkflowOrchestrator,
        )

    def test_ml_availability_flag(self):
        from ontology.ml import ML_AVAILABLE
        assert isinstance(ML_AVAILABLE, bool)

    def test_version(self):
        from ontology import __version__
        assert __version__ == "2.0.0"


# ─── Enum Tests ─────────────────────────────────────────────────────

class TestEnums:
    """Verify enum definitions and values."""

    def test_query_type_values(self):
        from ontology.core.models import QueryType
        assert QueryType.SIMPLE.value == "simple"
        assert QueryType.COMPLEX.value == "complex"
        assert QueryType.MULTI_STEP.value == "multi_step"
        assert QueryType.ANALYTICAL.value == "analytical"
        assert QueryType.CREATIVE.value == "creative"

    def test_execution_strategy_values(self):
        from ontology.core.models import ExecutionStrategy
        assert ExecutionStrategy.SINGLE_AGENT.value == "single_agent"
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.HYBRID.value == "hybrid"
        assert ExecutionStrategy.AUTO.value == "auto"

    def test_agent_type_values(self):
        from ontology.core.models import AgentType
        assert AgentType.RESEARCH.value == "research"
        assert AgentType.ANALYSIS.value == "analysis"
        assert AgentType.CREATIVE.value == "creative"
        assert AgentType.TECHNICAL.value == "technical"
        assert AgentType.GENERAL.value == "general"

    def test_execution_status_values(self):
        from ontology.core.models import ExecutionStatus
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"

    def test_llm_provider_values(self):
        from ontology.core.models import LLMProvider
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"


# ─── Dataclass Tests ────────────────────────────────────────────────

class TestSemanticQuery:
    """Verify SemanticQuery dataclass behavior."""

    def test_creation_minimal(self):
        from ontology.core.models import SemanticQuery
        q = SemanticQuery(query_text="test query")
        assert q.query_text == "test query"
        assert q.natural_language == "test query"
        assert q.query_id  # auto-generated UUID
        assert q.complexity_score == 0.0

    def test_creation_with_params(self):
        from ontology.core.models import SemanticQuery, QueryType
        q = SemanticQuery(
            query_text="complex analysis",
            query_type=QueryType.ANALYTICAL,
            complexity_score=0.8,
            entities=["samsung", "stock"],
        )
        assert q.query_type == QueryType.ANALYTICAL
        assert q.complexity_score == 0.8
        assert q.entities == ["samsung", "stock"]

    def test_cache_key_generation(self):
        from ontology.core.models import SemanticQuery
        q = SemanticQuery(query_text="test")
        key = q.get_cache_key()
        assert key.startswith("query_")
        assert "simple" in key

    def test_to_dict(self):
        from ontology.core.models import SemanticQuery
        q = SemanticQuery(query_text="hello")
        d = q.to_dict()
        assert d["query_text"] == "hello"
        assert "query_id" in d
        assert "query_type" in d
        assert "created_at" in d

    def test_hash_consistency(self):
        from ontology.core.models import SemanticQuery
        q1 = SemanticQuery(query_text="test")
        q2 = SemanticQuery(query_text="test")
        assert hash(q1) == hash(q2)

    def test_hash_difference(self):
        from ontology.core.models import SemanticQuery
        q1 = SemanticQuery(query_text="test1")
        q2 = SemanticQuery(query_text="test2")
        assert hash(q1) != hash(q2)

    def test_natural_language_default(self):
        from ontology.core.models import SemanticQuery
        q = SemanticQuery(query_text="hello world")
        assert q.natural_language == "hello world"

    def test_natural_language_explicit(self):
        from ontology.core.models import SemanticQuery
        q = SemanticQuery(query_text="hello", natural_language="custom")
        assert q.natural_language == "custom"


class TestOntologyLLMConfig:
    """Verify OntologyLLMConfig dataclass."""

    def test_creation(self):
        from ontology.core.models import OntologyLLMConfig, LLMProvider
        cfg = OntologyLLMConfig(
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash-lite",
        )
        assert cfg.provider == LLMProvider.GOOGLE
        assert cfg.model == "gemini-2.5-flash-lite"
        assert cfg.temperature == 0.7
        assert cfg.max_retries == 3
        assert cfg.cache_enabled is True

    def test_defaults(self):
        from ontology.core.models import OntologyLLMConfig, LLMProvider
        cfg = OntologyLLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        assert cfg.top_p == 1.0
        assert cfg.reasoning_depth == "standard"
        assert cfg.creativity_level == "balanced"
        assert cfg.precision_level == "high"


class TestExecutionContext:
    """Verify ExecutionContext dataclass."""

    def test_creation(self):
        from ontology.core.models import ExecutionContext
        ctx = ExecutionContext(max_parallel_agents=5, timeout_seconds=30)
        assert ctx.max_parallel_agents == 5
        assert ctx.timeout_seconds == 30


# ─── Orchestrator Model Tests ───────────────────────────────────────

class TestOrchestratorModels:
    """Verify orchestrator data models."""

    def test_agent_schema(self):
        from ontology.orchestrator import AgentSchema
        schema = AgentSchema(input_type="text", output_type="json")
        assert schema.input_type == "text"
        assert schema.output_type == "json"

    def test_agent_schema_compatibility(self):
        from ontology.orchestrator import AgentSchema
        s1 = AgentSchema(input_type="text", output_type="json")
        s2 = AgentSchema(input_type="json", output_type="text")
        assert s1.is_compatible_with(s2)

    def test_agent_registry_entry(self):
        from ontology.orchestrator import AgentRegistryEntry, AgentSchema
        entry = AgentRegistryEntry(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            schema=AgentSchema(input_type="text", output_type="text"),
        )
        assert entry.agent_id == "test_agent"
        assert entry.name == "Test Agent"

    def test_progress_event_type(self):
        from ontology.orchestrator import ProgressEventType
        assert hasattr(ProgressEventType, "WORKFLOW_START")
        assert hasattr(ProgressEventType, "STAGE_START")
        assert hasattr(ProgressEventType, "AGENT_START")

    def test_agent_registry_crud(self):
        from ontology.orchestrator import AgentRegistry, AgentRegistryEntry, AgentSchema
        registry = AgentRegistry()
        entry = AgentRegistryEntry(
            agent_id="test_agent",
            name="Test",
            description="Test agent",
            schema=AgentSchema(input_type="text", output_type="text"),
        )
        registry.register_agent(entry)
        agent = registry.get_agent("test_agent")
        assert agent is not None
        assert agent.agent_id == "test_agent"

    def test_agent_registry_list(self):
        from ontology.orchestrator import AgentRegistry, AgentRegistryEntry, AgentSchema
        registry = AgentRegistry()
        initial_count = len(registry.get_all_agents())
        for i in range(3):
            entry = AgentRegistryEntry(
                agent_id=f"test_agent_{i}",
                name=f"Test Agent {i}",
                description=f"Test agent number {i}",
                schema=AgentSchema(input_type="text", output_type="text"),
            )
            registry.register_agent(entry)
        agents = registry.get_all_agents()
        assert len(agents) == initial_count + 3


# ─── Knowledge Graph Engine Tests ───────────────────────────────────

class TestKnowledgeGraphEngine:
    """Verify basic KnowledgeGraphEngine operations."""

    def test_initialization(self):
        from ontology.engines import KnowledgeGraphEngine
        engine = KnowledgeGraphEngine()
        assert engine is not None

    def test_graph_exists(self):
        from ontology.engines import KnowledgeGraphEngine
        engine = KnowledgeGraphEngine()
        assert hasattr(engine, "graph") or hasattr(engine, "G")


# ─── Semantic Query Manager Tests ───────────────────────────────────

class TestSemanticQueryManager:
    """Verify SemanticQueryManager basic operations."""

    def test_initialization(self):
        from ontology.engines import SemanticQueryManager
        manager = SemanticQueryManager()
        assert manager is not None


# ─── InMemoryCacheManager Tests ─────────────────────────────────────

class TestInMemoryCacheManager:
    """Verify cache manager operations."""

    def test_initialization(self):
        from ontology.engines import InMemoryCacheManager
        cache = InMemoryCacheManager()
        assert cache is not None
