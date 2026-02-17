"""
Ontology System Package — Modular system components.
"""

# Main ontology system
from .ontology_system_clean import CleanOntologySystem, OntologySystem

# Sub-modules
from .query_processing import QueryProcessor
from .result_integration import ResultIntegrator
from .knowledge_management import KnowledgeGraphManager
from .metrics_manager import MetricsManager

# Legacy system (backward compatibility)
try:
    from .ontology_system import OntologySystem as LegacyOntologySystem
except ImportError:
    LegacyOntologySystem = None

__all__ = [
    # Main system
    'CleanOntologySystem',
    'OntologySystem',

    # Sub-modules
    'QueryProcessor',
    'ResultIntegrator',
    'KnowledgeGraphManager',
    'MetricsManager',

    # Backward compatibility
    'LegacyOntologySystem'
]
