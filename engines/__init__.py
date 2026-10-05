"""
Engines Package — Core execution engines.

지연 로딩(PEP 562). 이 배럴이 eager 였을 때는 `from ontology.engines.
knowledge_graph_clean import ...` 한 줄이 execution_engine·workflow_designer·
semantic_query_manager(= 에이전트 오케스트레이션 전량)까지 끌어왔다. 그래서
KG 만 쓰려던 소비자가 에이전트 스택을 통째로 설치해야 했고, aicoach 는
그 비용 때문에 KG 엔진을 재구현했다.

공개 이름(__all__)은 그대로다 — `from ontology.engines import
AdvancedExecutionEngine` 은 계속 동작하고, 그때 비로소 로드된다.
(계약: tests/test_kernel_decoupling.py)
"""

from importlib import import_module

_LAZY = {
    # Semantic Query Management
    "SemanticQueryManager": ".semantic_query_manager",
    "InMemoryCacheManager": ".semantic_query_manager",

    # Execution Engine
    "AdvancedExecutionEngine": ".execution_engine",
    "SmartDataTransformer": ".execution_engine",
    "MockAgentCaller": ".execution_engine",

    # Workflow Design
    "SmartWorkflowDesigner": ".workflow_designer",

    # Knowledge Graph
    "KnowledgeGraphEngine": ".knowledge_graph_clean",
}

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
