"""
LogosAI Ontology System

Knowledge-driven multi-agent orchestration with LLM-powered
query analysis and intelligent agent selection.

지연 로딩(PEP 562). 이 배럴이 eager 였을 때는 `import ontology.builder` 처럼
커널 한 조각만 쓰려 해도 부모 패키지 초기화가 먼저 돌아 OntologySystem·
execution_engine·workflow_designer(= 에이전트 스택 전량)가 로드됐다.
문서에서 개념을 뽑고 데이터셋을 추출하려는 소비자에게는 전부 불필요한
비용이고, 실제로 이것 때문에 채택이 막혔다
(aicoach backend/app/kg/graph.py:1-10 "without pulling its heavy transitive deps").

공개 이름은 바뀌지 않는다 — `from ontology import OntologySystem` 은 계속
동작하며, 그 순간에 비로소 해당 모듈이 로드된다.
(계약: tests/test_kernel_decoupling.py)
"""

from importlib import import_module

_LAZY = {
    # Core Models
    "SemanticQuery": ".core.models",
    "ExecutionContext": ".core.models",
    "AgentExecutionResult": ".core.models",
    "WorkflowPlan": ".core.models",
    "ExecutionStrategy": ".core.models",

    # Core Interfaces
    "QueryAnalyzer": ".core.interfaces",
    "ExecutionEngine": ".core.interfaces",
    "DataTransformer": ".core.interfaces",
    "ResultProcessor": ".core.interfaces",

    # Engines
    "SemanticQueryManager": ".engines.semantic_query_manager",
    "AdvancedExecutionEngine": ".engines.execution_engine",
    "SmartWorkflowDesigner": ".engines.workflow_designer",
    "KnowledgeGraphEngine": ".engines.knowledge_graph_clean",

    # System
    "OntologySystem": ".system.ontology_system",
}

# 설치 메타데이터를 정본으로 읽는다 — pyproject 만 올리고 여기 하드코딩을
# 놓쳐 pip 은 2.0.1, __version__ 은 2.0.0 을 보고한 사고가 있었다 (2026-07-20).
# 폴백은 소스 트리에서 직접 실행할 때만 쓰이며, pyproject 와 일치해야 한다
# (계약: tests/test_packaging.py::test_version_string_matches_pyproject).
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("logosai-ontology")
except Exception:
    __version__ = "2.0.2"
__author__ = "Logos AI Team"

__all__ = list(_LAZY)


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value  # 두 번째 접근부터는 __getattr__ 을 우회한다
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
