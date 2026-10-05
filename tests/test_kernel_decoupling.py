"""
축 1 — 커널 분리 회귀 테스트.

이 테스트가 존재하는 이유는 실측이다. aicoach 는 Logos KG 엔진의 API 를
그대로 베끼면서도 import 하지 않고 재구현했고, 그 이유를 코드에 남겼다:

    "Mirrors the Logos SimpleKnowledgeGraphEngine API (...) *without pulling
     its heavy transitive deps*, so it can be swapped for the Logos engine
     later."  — aicoach backend/app/kg/graph.py:1-10

즉 채택을 막은 것은 기능이 아니라 **import 비용**이다. 문서에서 개념을
뽑고 싶을 뿐인 소비자가 `ontology.builder` 를 import 하면 에이전트
오케스트레이터(7,643줄) + GNN/RL(1,866줄) 까지 딸려왔다.

여기서 고정하는 계약: 커널(builder / engines.knowledge_graph_clean /
server) 을 import 해도 에이전트 스택과 무거운 ML 런타임은 로드되지 않는다.

서브프로세스로 격리 실행한다 — 같은 프로세스에서는 다른 테스트가 이미
에이전트 스택을 import 했을 수 있어 sys.modules 관찰이 오염된다.
"""

import os
import subprocess
import sys

import pytest

_ONTOLOGY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGOS_ROOT = os.path.dirname(_ONTOLOGY_ROOT)

# 커널이 끌어오면 안 되는 모듈들.
# - orchestrator / ml / system / processors: 에이전트 라우팅 전용. 문서 검색·
#   데이터셋 추출과 무관하다.
# - engines.execution_engine / workflow_designer / semantic_query_manager:
#   engines/__init__.py 가 eager import 하던 것들.
# - logosai: 에이전트 프레임워크. LLM 호출 하나 때문에 딸려오면 안 된다.
# - torch / sentence_transformers: 임베더는 lazy 여야 한다. import 만으로
#   수백 MB 를 로드하면 CLI·테스트 시작이 초 단위로 느려진다.
BANNED_PREFIXES = (
    "ontology.orchestrator",
    "ontology.ml",
    "ontology.system",
    "ontology.processors",
    "ontology.engines.execution_engine",
    "ontology.engines.workflow_designer",
    "ontology.engines.semantic_query_manager",
    "logosai",
    "torch",
    "sentence_transformers",
)

_PROBE = """
import sys
{import_stmt}
banned = {banned!r}
leaked = sorted(m for m in sys.modules
                if any(m == b or m.startswith(b + ".") for b in banned))
print("|".join(leaked))
"""


def _leaked_modules(import_stmt: str):
    """격리 서브프로세스에서 import 하고 유출된 금지 모듈 목록을 돌려준다."""
    env = dict(os.environ, PYTHONPATH=_LOGOS_ROOT)
    code = _PROBE.format(import_stmt=import_stmt, banned=list(BANNED_PREFIXES))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=_LOGOS_ROOT, timeout=120,
    )
    assert result.returncode == 0, (
        f"import 자체가 실패했다 ({import_stmt}):\n{result.stderr}")
    out = result.stdout.strip()
    return out.split("|") if out else []


class TestKernelImportIsolation:
    """커널 import 는 에이전트 스택을 끌어오지 않는다."""

    def test_builder_does_not_pull_agent_stack(self):
        leaked = _leaked_modules("import ontology.builder")
        assert leaked == [], (
            "ontology.builder 가 에이전트 스택을 끌어왔다: " + ", ".join(leaked))

    def test_kg_engine_does_not_pull_agent_stack(self):
        leaked = _leaked_modules(
            "from ontology.engines.knowledge_graph_clean "
            "import get_knowledge_graph_engine")
        assert leaked == [], (
            "KG 엔진이 에이전트 스택을 끌어왔다: " + ", ".join(leaked))

    def test_core_kernel_modules_do_not_pull_agent_stack(self):
        leaked = _leaked_modules(
            "import ontology.core.semantic_index, ontology.core.vector_backend")
        assert leaked == [], (
            "core 커널이 에이전트 스택을 끌어왔다: " + ", ".join(leaked))


class TestPublicApiPreserved:
    """지연 로딩으로 바뀌어도 기존 공개 API 는 그대로 import 된다.

    __init__ 을 lazy 로 만들면서 공개 이름을 잃으면 그건 분리가 아니라
    파괴다. 소비자 코드(`from ontology import OntologySystem`)는 무변경으로
    계속 동작해야 한다.
    """

    def test_top_level_names_still_importable(self):
        code = ("from ontology import KnowledgeGraphEngine, SemanticQuery, "
                "ExecutionStrategy; print('ok')")
        env = dict(os.environ, PYTHONPATH=_LOGOS_ROOT)
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, env=env,
                                cwd=_LOGOS_ROOT, timeout=120)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_bare_package_import_works(self):
        """`import ontology` 자체가 성공한다.

        지연 로딩의 부수 효과로 오히려 **고쳐진** 것. 이전 배럴은
        `from .system.ontology_system import OntologySystem` 을 eager 로
        실행했는데, 그 체인 끝의 system/strategy_manager.py:15 가
        `from core.models import ...` — 즉 ontology/ 자신이 sys.path 에
        있어야만 되는 절대 import 다. 그래서 PYTHONPATH 에 Logos 루트만
        있는 정상 설치에서는 `import ontology` 가 통째로 실패했다.
        (test_ontology_system_is_still_broken 가 그 잔존 결함을 기록한다)
        """
        env = dict(os.environ, PYTHONPATH=_LOGOS_ROOT)
        result = subprocess.run([sys.executable, "-c", "import ontology; print('ok')"],
                                capture_output=True, text=True, env=env,
                                cwd=_LOGOS_ROOT, timeout=120)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_ontology_system_imports(self):
        """축 1 이전부터 있던 절대 import 결함의 회귀 계약 (2026-08-21 수리).

        `system/strategy_manager.py` 가 `sys.path.append` 로 ontology/ 를 밀어
        넣고 `from core.models import ...` 를 썼다 — ontology/ 자신이 sys.path
        에 있어야만 로드되고, import 시점에 프로세스의 sys.path 를 오염시켰다.
        상대 import 로 바꿔 패키지로서 정상 로드된다.
        """
        env = dict(os.environ, PYTHONPATH=_LOGOS_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", "from ontology import OntologySystem"],
            capture_output=True, text=True, env=env, cwd=_LOGOS_ROOT, timeout=120)
        assert result.returncode == 0, result.stderr

    def test_engines_barrel_names_still_importable(self):
        code = ("from ontology.engines import KnowledgeGraphEngine, "
                "AdvancedExecutionEngine, SmartWorkflowDesigner; print('ok')")
        env = dict(os.environ, PYTHONPATH=_LOGOS_ROOT)
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, env=env,
                                cwd=_LOGOS_ROOT, timeout=120)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_dir_still_lists_public_names(self):
        import ontology
        listed = dir(ontology)
        for name in ("KnowledgeGraphEngine", "OntologySystem", "SemanticQuery"):
            assert name in listed, f"dir(ontology) 에서 {name} 이 사라졌다"
