"""`ontology.orchestrator` 의 옛 경로 = `logosai.orchestration` 의 같은 모듈 객체.

실행부 8개가 logosai 로 옮겨졌다(2026-10-04, orchestrator-unify). 옛 경로는
재export 가 아니라 **별칭**이어야 한다 — 사전 실험(스파이크)에서 `import *`
재export 는 세 가지를 깨뜨렸다:
  · 모듈 동일성 — `agent_registry._default_registry` 싱글턴이 둘로 갈라진다
    (logos_api 가 등록한 레지스트리와 엔진이 읽는 레지스트리가 달라진다)
  · 내부 이름 접근 — 밑줄 이름은 `*` 로 넘어오지 않는다
  · 옛 경로로 한 패치가 실제 구현에 반영되지 않는다
"""
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ONTOLOGY = Path(__file__).resolve().parents[1]
MOVED = [
    "models", "exceptions", "agent_registry", "progress_streamer",
    "data_transformer", "plan_validator", "result_aggregator", "execution_engine",
]


@pytest.mark.parametrize("name", MOVED)
def test_old_path_is_the_same_module(name):
    old = importlib.import_module(f"ontology.orchestrator.{name}")
    new = importlib.import_module(f"logosai.orchestration.{name}")
    assert old is new
    assert getattr(importlib.import_module("ontology.orchestrator"), name) is new


def test_registry_singleton_is_shared():
    from logosai.orchestration import agent_registry as new
    from ontology.orchestrator import agent_registry as old
    from ontology.orchestrator.models import AgentRegistryEntry, AgentSchema

    assert old.get_registry() is new.get_registry()
    agent_id = "compat_alias_probe_agent"
    assert not new.get_registry().has_agent(agent_id)  # 대조군 — 처음엔 없다
    old.get_registry().register_agent(AgentRegistryEntry(
        agent_id=agent_id, name="p", description="d", capabilities=[], tags=[],
        schema=AgentSchema(input_type="query", output_type="text"),
    ))
    try:
        assert new.get_registry().has_agent(agent_id)
    finally:
        old.get_registry().unregister_agent(agent_id)


def test_patch_through_old_path_reaches_the_implementation(monkeypatch):
    import logosai.orchestration.agent_registry as new
    import ontology.orchestrator.agent_registry as old

    sentinel = object()
    monkeypatch.setattr(old, "_default_registry", sentinel)
    assert new._default_registry is sentinel


def test_instances_cross_both_paths():
    from logosai.orchestration import ExecutionPlan as New
    from ontology.orchestrator import ExecutionPlan as Old

    assert isinstance(Old(query="q", workflow_strategy="sequential"), New)


def _run(code: str):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          cwd=ONTOLOGY.parent, capture_output=True, text=True)


def test_bare_name_import_gets_the_same_class():
    """orchestrator 폴더를 sys.path 에 넣고 `from models import` 하는 기존 테스트 형태."""
    out = _run(f"""
        import sys; sys.path.insert(0, {str(ONTOLOGY / 'orchestrator')!r})
        from models import ExecutionPlan
        import logosai.orchestration.models as m
        print("SAME" if ExecutionPlan is m.ExecutionPlan else "DIFF")
    """)
    assert out.stdout.strip().endswith("SAME"), out.stdout + out.stderr[-400:]


@pytest.mark.parametrize("name", MOVED)
def test_loading_the_file_by_path_still_exposes_the_api(name):
    """다섯 번째 import 형태 — 파일 경로로 직접 로드 (spec_from_file_location).

    이 형태는 sys.modules 바꿔치기를 반영하지 않고 shim 자신을 돌려준다.
    사전 실험(import 4형태)이 놓쳤고, orchestrator/test_capability_gap_unit.py 가
    정확히 이렇게 로드하다 3건 실패해서 드러났다. 모듈 동일성은 이 형태에서
    지킬 수 없지만, 공개 이름은 정본과 같은 객체여야 한다.
    """
    import importlib.util

    new = importlib.import_module(f"logosai.orchestration.{name}")
    path = ONTOLOGY / "orchestrator" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"path_loaded_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    public = [n for n in vars(new) if not n.startswith("_")]
    assert public, "대조군 — 정본에 공개 이름이 있어야 이 검사가 의미 있다"
    missing = [n for n in public if getattr(mod, n, None) is not getattr(new, n)]
    assert missing == [], f"경로 로드한 {name}.py 에 없거나 다른 이름: {missing[:5]}"


def test_missing_orchestration_explains_itself():
    """logosai 가 orchestration 이전 버전이면 원인과 해결책을 말한다.

    logos_api 는 이 ImportError 를 경고 한 줄로 삼키고 직접 ACP 모드로 돈다 —
    그 한 줄이 원인을 말하지 않으면 워크플로가 꺼진 이유를 찾을 수 없다.
    """
    out = _run("""
        import sys
        sys.modules["logosai.orchestration"] = None   # 이전 버전 흉내
        try:
            import ontology.orchestrator.models
        except ImportError as e:
            print("ERR", type(e).__name__, e)
    """)
    assert "ERR ImportError" in out.stdout, out.stdout + out.stderr[-400:]
    assert "logosai.orchestration" in out.stdout and "logosai" in out.stdout.split("ERR", 1)[1]


def test_unrelated_import_error_is_not_relabelled():
    """대조군 — orchestration 내부의 다른 의존성 누락을 버전 문제로 둔갑시키지 않는다."""
    out = _run("""
        import sys, importlib.abc
        class Boom(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "logosai.orchestration.models":
                    raise ModuleNotFoundError("No module named 'some_missing_dep'", name="some_missing_dep")
        sys.meta_path.insert(0, Boom())
        try:
            import ontology.orchestrator.models
        except ImportError as e:
            print("ERR", type(e).__name__, getattr(e, "name", None))
    """)
    assert "ERR ModuleNotFoundError some_missing_dep" in out.stdout, out.stdout + out.stderr[-400:]
