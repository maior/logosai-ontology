"""
LogosAI Ontology System

Knowledge-driven multi-agent orchestration with LLM-powered
query analysis and intelligent agent selection.
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
    "KnowledgeGraphEngine",
    
    # System
    "OntologySystem"
] 