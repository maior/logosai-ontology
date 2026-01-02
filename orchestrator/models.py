"""
Data Models for Workflow Orchestrator

Defines all data classes used throughout the orchestration system,
including execution plans, results, and progress streaming events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class AgentStatus(str, Enum):
    """Agent execution status for progress tracking"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ProgressEventType(str, Enum):
    """Types of progress events for frontend streaming"""
    # Workflow level events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"

    # Planning phase events
    PLANNING_START = "planning_start"
    PLANNING_COMPLETE = "planning_complete"
    PLANNING_ERROR = "planning_error"

    # Validation phase events
    VALIDATION_START = "validation_start"
    VALIDATION_COMPLETE = "validation_complete"
    VALIDATION_ERROR = "validation_error"

    # Stage level events
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_ERROR = "stage_error"

    # Agent level events
    AGENT_QUEUED = "agent_queued"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"  # For long-running agents
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    AGENT_RETRY = "agent_retry"

    # Data transformation events
    TRANSFORM_START = "transform_start"
    TRANSFORM_COMPLETE = "transform_complete"
    TRANSFORM_ERROR = "transform_error"

    # Aggregation events
    AGGREGATION_START = "aggregation_start"
    AGGREGATION_COMPLETE = "aggregation_complete"

    # General info/debug events
    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"


@dataclass
class ProgressEvent:
    """
    Progress event for real-time frontend streaming.

    Designed to provide comprehensive status updates that the frontend
    can use to display execution progress, agent status, and results.
    """
    type: ProgressEventType
    timestamp: datetime = field(default_factory=datetime.now)

    # Context identifiers
    workflow_id: Optional[str] = None
    stage_id: Optional[int] = None
    agent_id: Optional[str] = None

    # Status and progress
    status: AgentStatus = AgentStatus.PENDING
    progress_percent: Optional[float] = None  # 0.0 - 100.0

    # Human-readable messages
    message: str = ""
    message_ko: str = ""  # Korean message for localization

    # Detailed data
    data: Optional[Dict[str, Any]] = None

    # Timing information
    elapsed_time_ms: Optional[float] = None
    estimated_remaining_ms: Optional[float] = None

    # Error information
    error: Optional[str] = None
    error_code: Optional[str] = None

    # For frontend display
    display_name: Optional[str] = None  # Agent/stage display name
    icon: Optional[str] = None  # Emoji or icon identifier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "stage_id": self.stage_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "message": self.message,
            "message_ko": self.message_ko,
            "data": self.data,
            "elapsed_time_ms": self.elapsed_time_ms,
            "estimated_remaining_ms": self.estimated_remaining_ms,
            "error": self.error,
            "error_code": self.error_code,
            "display_name": self.display_name,
            "icon": self.icon,
        }

    def to_json(self) -> str:
        """Convert to JSON string for streaming"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format"""
        return f"data: {self.to_json()}\n\n"


@dataclass
class AgentSchema:
    """Agent input/output schema definition"""
    input_type: str  # "query" | "structured_data" | "any"
    output_type: str  # "text" | "json" | "html" | "chart_data" | "any"
    input_format: Optional[Dict[str, Any]] = None  # JSON schema if applicable
    output_format: Optional[Dict[str, Any]] = None  # JSON schema if applicable

    def is_compatible_with(self, other: "AgentSchema") -> bool:
        """Check if this schema's output is compatible with other's input"""
        # "any" is compatible with everything
        if self.output_type == "any" or other.input_type == "any":
            return True
        # Direct type match
        if self.output_type == other.input_type:
            return True
        # text can be converted to query (simple passthrough)
        if self.output_type == "text" and other.input_type == "query":
            return True
        # text can be converted to structured_data with transformation
        if self.output_type == "text" and other.input_type == "structured_data":
            return True  # Requires transformation
        # json can be converted to query (stringification)
        if self.output_type == "json" and other.input_type == "query":
            return True
        # json can be converted to structured_data (passthrough)
        if self.output_type == "json" and other.input_type == "structured_data":
            return True
        # structured_data can be converted to query
        if self.output_type == "structured_data" and other.input_type == "query":
            return True
        return False


@dataclass
class AgentRegistryEntry:
    """Agent registry entry with full metadata"""
    agent_id: str
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    schema: AgentSchema = field(default_factory=lambda: AgentSchema("query", "text"))
    priority: int = 0  # Higher priority = preferred for selection

    # Display information for frontend
    display_name: Optional[str] = None
    display_name_ko: Optional[str] = None
    icon: str = "🤖"
    color: str = "#6366f1"  # Default indigo color

    # Execution metadata
    average_execution_time_ms: Optional[float] = None
    success_rate: Optional[float] = None
    is_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt building"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "input_type": self.schema.input_type,
            "output_type": self.schema.output_type,
        }


@dataclass
class AgentTask:
    """Individual agent task within a stage"""
    agent_id: str
    sub_query: str
    input_from: Optional[List[str]] = None  # ["stage_1.internet_agent"]
    output_to: Optional[List[str]] = None   # ["stage_2", "final"]
    expected_output: Optional[str] = None

    # Execution configuration
    timeout_ms: int = 30000  # 30 seconds default
    max_retries: int = 2

    # Runtime data (set during execution)
    task_id: Optional[str] = None


@dataclass
class ExecutionStage:
    """Execution stage containing one or more agent tasks"""
    stage_id: int
    execution_type: str  # "sequential" | "parallel"
    agents: List[AgentTask] = field(default_factory=list)

    # Stage metadata
    name: Optional[str] = None
    description: Optional[str] = None

    # Dependencies
    depends_on: Optional[List[int]] = None  # Previous stage IDs

    def get_agent_count(self) -> int:
        return len(self.agents)


@dataclass
class ExecutionPlan:
    """Complete execution plan generated by Query Planner"""
    query: str
    workflow_strategy: str  # "sequential" | "parallel" | "hybrid"
    stages: List[ExecutionStage] = field(default_factory=list)
    final_aggregation: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    # Plan metadata
    plan_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    # Validation status
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def get_total_agents(self) -> int:
        return sum(stage.get_agent_count() for stage in self.stages)

    def get_stage_count(self) -> int:
        return len(self.stages)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "workflow_strategy": self.workflow_strategy,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "execution_type": stage.execution_type,
                    "agents": [
                        {
                            "agent_id": agent.agent_id,
                            "sub_query": agent.sub_query,
                            "input_from": agent.input_from,
                            "output_to": agent.output_to,
                        }
                        for agent in stage.agents
                    ]
                }
                for stage in self.stages
            ],
            "final_aggregation": self.final_aggregation,
            "reasoning": self.reasoning,
        }


@dataclass
class AgentResult:
    """Result from a single agent execution"""
    agent_id: str
    stage_id: int
    success: bool
    data: Any = None
    error: Optional[str] = None

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0

    # Metadata
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "stage_id": self.stage_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
        }


@dataclass
class StageResult:
    """Result from a complete stage execution"""
    stage_id: int
    execution_type: str
    success: bool
    results: List[AgentResult] = field(default_factory=list)

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_time_ms: float = 0.0

    # Aggregated data for next stage
    aggregated_output: Any = None

    def get_successful_results(self) -> List[AgentResult]:
        return [r for r in self.results if r.success]

    def get_failed_results(self) -> List[AgentResult]:
        return [r for r in self.results if not r.success]


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    success: bool
    workflow_id: str
    query: str
    stages: List[StageResult] = field(default_factory=list)
    final_output: Any = None

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_time_ms: float = 0.0

    # Plan reference
    plan: Optional[ExecutionPlan] = None

    # Error information
    error: Optional[str] = None
    error_stage: Optional[int] = None
    error_agent: Optional[str] = None

    # Statistics
    total_agents_executed: int = 0
    successful_agents: int = 0
    failed_agents: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "workflow_id": self.workflow_id,
            "query": self.query,
            "final_output": self.final_output,
            "total_time_ms": self.total_time_ms,
            "total_agents_executed": self.total_agents_executed,
            "successful_agents": self.successful_agents,
            "failed_agents": self.failed_agents,
            "error": self.error,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "success": stage.success,
                    "total_time_ms": stage.total_time_ms,
                    "results": [r.to_dict() for r in stage.results]
                }
                for stage in self.stages
            ]
        }


# Type aliases for clarity
AgentId = str
StageId = int
WorkflowId = str
