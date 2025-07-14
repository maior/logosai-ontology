"""
🧠 Ontology-Enhanced Multi-Agent System
온톨로지 기반 멀티 에이전트 시스템

새로운 아키텍처로 완전히 재설계된 시스템
"""

from .core.models import (
    SemanticQuery,
    ExecutionContext,
    AgentExecutionResult,
    WorkflowPlan,
    ExecutionStrategy
)

from .core.interfaces import (
    QueryAnalyzer,
    ExecutionEngine,
    DataTransformer,
    ResultProcessor
)

from .engines.semantic_query_manager import SemanticQueryManager
from .engines.execution_engine import AdvancedExecutionEngine
from .engines.workflow_designer import SmartWorkflowDesigner
from .engines.knowledge_graph_clean import KnowledgeGraphEngine

from .system.ontology_system import OntologySystem

__version__ = "2.0.0"
__author__ = "Logos AI Team"

__all__ = [
    # Core Models
    "SemanticQuery",
    "ExecutionContext", 
    "AgentExecutionResult",
    "WorkflowPlan",
    "ExecutionStrategy",
    
    # Core Interfaces
    "QueryAnalyzer",
    "ExecutionEngine",
    "DataTransformer",
    "ResultProcessor",
    
    # Engines
    "SemanticQueryManager",
    "AdvancedExecutionEngine",
    "SmartWorkflowDesigner",
    "SimpleKnowledgeGraphEngine",
    
    # System
    "OntologySystem"
] 