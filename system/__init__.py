"""
🧠 Ontology System Package
온톨로지 시스템 패키지

분할된 모듈들을 통한 깔끔한 시스템 구조
"""

# 깔끔한 온톨로지 시스템 (메인)
from .ontology_system_clean import CleanOntologySystem, OntologySystem

# 분할된 모듈들
from .query_processing import QueryProcessor
from .result_integration import ResultIntegrator
from .knowledge_management import KnowledgeGraphManager
from .metrics_manager import MetricsManager

# 기존 시스템 (하위 호환성)
try:
    from .ontology_system import OntologySystem as LegacyOntologySystem
except ImportError:
    LegacyOntologySystem = None

__all__ = [
    # 메인 시스템
    'CleanOntologySystem',
    'OntologySystem',
    
    # 분할된 모듈들
    'QueryProcessor',
    'ResultIntegrator', 
    'KnowledgeGraphManager',
    'MetricsManager',
    
    # 하위 호환성
    'LegacyOntologySystem'
] 