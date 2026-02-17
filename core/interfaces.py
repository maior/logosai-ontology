"""
🧠 Core Interfaces
Core interface definitions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator, AsyncIterator
from .models import (
    SemanticQuery, 
    ExecutionContext, 
    AgentExecutionResult, 
    WorkflowPlan, 
    ComplexityAnalysis,
    DataTransformationType,
    AgentType,
    ExecutionStrategy,
    SystemMetrics
)


class QueryAnalyzer(ABC):
    """Query analyzer interface"""
    
    @abstractmethod
    async def analyze_query(self, query_text: str, context: Dict[str, Any] = None) -> SemanticQuery:
        """Analyze query and create SemanticQuery object"""
        pass
    
    @abstractmethod
    def estimate_complexity(self, query: SemanticQuery) -> float:
        """Estimate query complexity (0.0 ~ 1.0)"""
        pass
    
    @abstractmethod
    def suggest_agents(self, query: SemanticQuery) -> List[AgentType]:
        """Recommend agents suitable for the query"""
        pass


class ExecutionEngine(ABC):
    """Execution engine interface"""
    
    @abstractmethod
    async def execute_query(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> List[AgentExecutionResult]:
        """Execute query"""
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[ExecutionStrategy]:
        """List of supported execution strategies"""
        pass
    
    @abstractmethod
    async def estimate_execution_time(
        self, 
        query: SemanticQuery, 
        strategy: ExecutionStrategy
    ) -> float:
        """Estimate execution time"""
        pass


class DataTransformer(ABC):
    """Data transformer interface"""
    
    @abstractmethod
    async def transform_input(
        self, 
        data: Any, 
        source_agent: AgentType, 
        target_agent: AgentType
    ) -> Any:
        """Transform input data"""
        pass
    
    @abstractmethod
    async def transform_output(
        self, 
        result: AgentExecutionResult, 
        target_format: str
    ) -> Any:
        """Transform output data"""
        pass
    
    @abstractmethod
    def get_supported_transformations(self) -> Dict[str, List[str]]:
        """List of supported transformations"""
        pass


class ResultProcessor(ABC):
    """Result processor interface"""
    
    @abstractmethod
    async def process_results(self, 
                            results: List[AgentExecutionResult],
                            context: ExecutionContext) -> Dict[str, Any]:
        """Process results"""
        pass
    
    @abstractmethod
    def validate_result(self, result: AgentExecutionResult) -> bool:
        """Validate result"""
        pass


class WorkflowDesigner(ABC):
    """Workflow designer interface"""
    
    @abstractmethod
    async def design_workflow(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> WorkflowPlan:
        """Design workflow"""
        pass
    
    @abstractmethod
    async def optimize_workflow(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Optimize workflow"""
        pass
    
    @abstractmethod
    def validate_workflow(self, plan: WorkflowPlan) -> bool:
        """Validate workflow"""
        pass


class CacheManager(ABC):
    """Cache manager interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Store value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Cache statistics"""
        pass


class KnowledgeGraph(ABC):
    """Knowledge graph interface"""
    
    @abstractmethod
    async def add_concept(self, concept: str, properties: Dict[str, Any] = None):
        """Add concept"""
        pass
    
    @abstractmethod
    async def add_relation(self, source: str, target: str, relation_type: str, properties: Dict[str, Any] = None):
        """Add relationship"""
        pass
    
    @abstractmethod
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Query the graph"""
        pass
    
    @abstractmethod
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Find related concepts"""
        pass
    
    @abstractmethod
    def visualize_graph(self, output_path: str = None) -> str:
        """Visualize graph"""
        pass


class AgentCaller(ABC):
    """Agent caller interface"""
    
    @abstractmethod
    async def call_agent(
        self, 
        agent_type: AgentType, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Call a single agent"""
        pass
    
    @abstractmethod
    async def call_agents_parallel(
        self, 
        agent_calls: List[Dict[str, Any]]
    ) -> List[AgentExecutionResult]:
        """Call agents in parallel"""
        pass
    
    @abstractmethod
    def get_agent_status(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get agent status"""
        pass


class ProgressCallback(ABC):
    """Progress callback interface"""
    
    @abstractmethod
    async def on_progress(self, message: str, progress: float, metadata: Dict[str, Any] = None):
        """Update progress"""
        pass
    
    @abstractmethod
    async def on_step_complete(self, step_id: str, result: AgentExecutionResult):
        """Notify step completion"""
        pass
    
    @abstractmethod
    async def on_error(self, error_message: str, error_details: Dict[str, Any] = None):
        """Notify error occurrence"""
        pass


class SystemMonitor(ABC):
    """System monitor interface"""
    
    @abstractmethod
    def record_execution(self, 
                        strategy: str, 
                        execution_time: float, 
                        success: bool, 
                        metadata: Dict[str, Any] = None):
        """Record execution"""
        pass
    
    @abstractmethod
    def record_cache_access(self, hit: bool, key: str = None):
        """Record cache access"""
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass


class MetricsCollector(ABC):
    """Metrics collector interface"""
    
    @abstractmethod
    def record_query_execution(
        self, 
        query: SemanticQuery, 
        execution_time: float, 
        success: bool
    ):
        """Record query execution"""
        pass
    
    @abstractmethod
    def record_cache_access(self, hit: bool):
        """Record cache access"""
        pass
    
    @abstractmethod
    def record_duplicate_prevention(self):
        """Record duplicate call prevention"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> SystemMetrics:
        """Get metrics"""
        pass
    
    @abstractmethod
    async def export_metrics(self, format: str = "json") -> str:
        """Export metrics"""
        pass


class EventListener(ABC):
    """Event listener interface"""
    
    @abstractmethod
    async def on_query_start(self, query: SemanticQuery, context: ExecutionContext):
        """Query start event"""
        pass
    
    @abstractmethod
    async def on_query_complete(self, query: SemanticQuery, results: List[AgentExecutionResult]):
        """Query complete event"""
        pass
    
    @abstractmethod
    async def on_agent_call(self, agent_type: AgentType, query: SemanticQuery):
        """Agent call event"""
        pass
    
    @abstractmethod
    async def on_cache_hit(self, key: str):
        """Cache hit event"""
        pass
    
    @abstractmethod
    async def on_error(self, error: Exception, context: Dict[str, Any]):
        """Error event"""
        pass


class ConfigurationManager(ABC):
    """Configuration manager interface"""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """Save configuration value"""
        pass
    
    @abstractmethod
    def load_config_file(self, file_path: str) -> bool:
        """Load configuration file"""
        pass
    
    @abstractmethod
    def save_config_file(self, file_path: str) -> bool:
        """Save configuration file"""
        pass
    
    @abstractmethod
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations"""
        pass


class HealthChecker(ABC):
    """Health checker interface"""
    
    @abstractmethod
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        pass
    
    @abstractmethod
    async def check_agent_health(self, agent_type: AgentType) -> Dict[str, Any]:
        """Check specific agent health"""
        pass
    
    @abstractmethod
    async def check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> str:
        """Overall health status (healthy/degraded/unhealthy)"""
        pass


class ProgressTracker(ABC):
    """Progress tracker interface"""
    
    @abstractmethod
    async def start_tracking(self, session_id: str, total_steps: int):
        """Start tracking"""
        pass
    
    @abstractmethod
    async def update_progress(self, session_id: str, completed_steps: int, message: str = None):
        """Update progress"""
        pass
    
    @abstractmethod
    async def complete_tracking(self, session_id: str, success: bool = True):
        """Complete tracking"""
        pass
    
    @abstractmethod
    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Get progress status"""
        pass
    
    @abstractmethod
    async def subscribe_progress(self, session_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to progress updates (real-time)"""
        pass 