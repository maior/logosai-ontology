"""
Custom Exceptions for Workflow Orchestrator

Provides specific exception types for different error scenarios
during workflow planning, validation, and execution.
"""

from typing import Any, Dict, List, Optional


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors"""

    def __init__(
        self,
        message: str,
        code: str = "ORCHESTRATOR_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class PlanValidationError(OrchestratorError):
    """Raised when execution plan validation fails"""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        code: str = "PLAN_VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)
        self.validation_errors = validation_errors or []

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["validation_errors"] = self.validation_errors
        return result


class AgentNotFoundError(PlanValidationError):
    """Raised when a referenced agent is not found in registry"""

    def __init__(self, agent_id: str, available_agents: Optional[List[str]] = None):
        message = f"Agent '{agent_id}' not found in registry"
        super().__init__(
            message=message,
            code="AGENT_NOT_FOUND",
            details={
                "agent_id": agent_id,
                "available_agents": available_agents or [],
            }
        )
        self.agent_id = agent_id


class CircularDependencyError(PlanValidationError):
    """Raised when circular dependency is detected in execution plan"""

    def __init__(self, cycle: List[str]):
        cycle_str = " → ".join(cycle)
        message = f"Circular dependency detected: {cycle_str}"
        super().__init__(
            message=message,
            code="CIRCULAR_DEPENDENCY",
            details={"cycle": cycle}
        )
        self.cycle = cycle


class SchemaCompatibilityError(PlanValidationError):
    """Raised when agent schemas are incompatible"""

    def __init__(
        self,
        source_agent: str,
        target_agent: str,
        source_output: str,
        target_input: str
    ):
        message = (
            f"Schema incompatibility: {source_agent} outputs '{source_output}' "
            f"but {target_agent} expects '{target_input}'"
        )
        super().__init__(
            message=message,
            code="SCHEMA_INCOMPATIBILITY",
            details={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "source_output": source_output,
                "target_input": target_input,
            }
        )


class ExecutionError(OrchestratorError):
    """Base class for execution-related errors"""

    def __init__(
        self,
        message: str,
        stage_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        code: str = "EXECUTION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)
        self.stage_id = stage_id
        self.agent_id = agent_id

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["stage_id"] = self.stage_id
        result["agent_id"] = self.agent_id
        return result


class AgentExecutionError(ExecutionError):
    """Raised when an agent fails during execution"""

    def __init__(
        self,
        agent_id: str,
        stage_id: int,
        original_error: Optional[Exception] = None,
        retry_count: int = 0
    ):
        message = f"Agent '{agent_id}' failed at stage {stage_id}"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(
            message=message,
            stage_id=stage_id,
            agent_id=agent_id,
            code="AGENT_EXECUTION_ERROR",
            details={
                "original_error": str(original_error) if original_error else None,
                "retry_count": retry_count,
            }
        )
        self.original_error = original_error
        self.retry_count = retry_count


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent execution times out"""

    def __init__(
        self,
        agent_id: str,
        stage_id: int,
        timeout_ms: int
    ):
        super().__init__(
            agent_id=agent_id,
            stage_id=stage_id,
        )
        self.code = "AGENT_TIMEOUT"
        self.message = f"Agent '{agent_id}' timed out after {timeout_ms}ms"
        self.details["timeout_ms"] = timeout_ms
        self.timeout_ms = timeout_ms


class TransformationError(OrchestratorError):
    """Raised when data transformation between agents fails"""

    def __init__(
        self,
        source_agent: str,
        target_agent: str,
        reason: str,
        original_error: Optional[Exception] = None
    ):
        message = (
            f"Failed to transform data from '{source_agent}' to '{target_agent}': {reason}"
        )
        super().__init__(
            message=message,
            code="TRANSFORMATION_ERROR",
            details={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "reason": reason,
                "original_error": str(original_error) if original_error else None,
            }
        )
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.original_error = original_error


class PlanningError(OrchestratorError):
    """Raised when query planning fails"""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        llm_response: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code="PLANNING_ERROR",
            details={
                "query": query,
                "llm_response": llm_response,
            }
        )
        self.query = query
        self.llm_response = llm_response


class NoSuitableAgentError(PlanningError):
    """Raised when no suitable agent is found for the query"""

    def __init__(
        self,
        query: str,
        required_capabilities: Optional[List[str]] = None,
        available_agents: Optional[List[str]] = None
    ):
        message = f"No suitable agent found for query: '{query[:100]}...'"
        super().__init__(
            message=message,
            query=query,
        )
        self.code = "NO_SUITABLE_AGENT"
        self.details["required_capabilities"] = required_capabilities or []
        self.details["available_agents"] = available_agents or []


class StreamingError(OrchestratorError):
    """Raised when streaming progress updates fail"""

    def __init__(self, message: str, event_type: Optional[str] = None):
        super().__init__(
            message=message,
            code="STREAMING_ERROR",
            details={"event_type": event_type}
        )


class AggregationError(OrchestratorError):
    """Raised when result aggregation fails"""

    def __init__(
        self,
        message: str,
        aggregation_type: Optional[str] = None,
        partial_results: Optional[List[Any]] = None
    ):
        super().__init__(
            message=message,
            code="AGGREGATION_ERROR",
            details={
                "aggregation_type": aggregation_type,
                "partial_results_count": len(partial_results) if partial_results else 0,
            }
        )
