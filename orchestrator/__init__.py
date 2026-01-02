"""
Workflow Orchestrator Package

This package provides intelligent workflow orchestration for multi-agent systems.
Key features:
- Single LLM call for planning (gemini-2.5-flash-lite)
- Sequential, Parallel, and Hybrid execution strategies
- Real-time progress streaming for frontend visualization
- Automatic data transformation between agents
"""

from .models import (
    AgentSchema,
    AgentRegistryEntry,
    AgentTask,
    ExecutionStage,
    ExecutionPlan,
    AgentResult,
    StageResult,
    WorkflowResult,
    ProgressEvent,
    ProgressEventType,
    AgentStatus,
)

from .exceptions import (
    OrchestratorError,
    PlanValidationError,
    ExecutionError,
    TransformationError,
    AgentNotFoundError,
    CircularDependencyError,
    SchemaCompatibilityError,
)

from .agent_registry import AgentRegistry, get_registry
from .progress_streamer import ProgressStreamer
from .query_planner import QueryPlanner
from .plan_validator import PlanValidator
from .data_transformer import DataTransformer
from .execution_engine import ExecutionEngine
from .result_aggregator import ResultAggregator
from .workflow_orchestrator import WorkflowOrchestrator

__all__ = [
    # Models
    "AgentSchema",
    "AgentRegistryEntry",
    "AgentTask",
    "ExecutionStage",
    "ExecutionPlan",
    "AgentResult",
    "StageResult",
    "WorkflowResult",
    "ProgressEvent",
    "ProgressEventType",
    "AgentStatus",
    # Exceptions
    "OrchestratorError",
    "PlanValidationError",
    "ExecutionError",
    "TransformationError",
    "AgentNotFoundError",
    "CircularDependencyError",
    "SchemaCompatibilityError",
    # Components
    "AgentRegistry",
    "ProgressStreamer",
    "QueryPlanner",
    "PlanValidator",
    "DataTransformer",
    "ExecutionEngine",
    "ResultAggregator",
    "WorkflowOrchestrator",
]

__version__ = "1.0.0"
