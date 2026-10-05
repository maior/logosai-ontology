"""ontology/agents/ 경계 계약 (2026-07-23, Phase 3c).

ontology/agents/ 는 acp 가 파일 경로로 로드하는 grounded 에이전트(logosai
의존)가 사는 곳이다. 이들은 acp 프로세스에서만 실행되며 온톨로지 wheel·커널과
분리돼야 한다([[acp-location-independent-loader]]). 이 테스트는 그 분리가
실수로 깨지는 것을 막는다:

  1. `ontology.agents` 는 pyproject packages 에 없어야 한다 — 있으면 소비자가
     `pip install ontology` 시 logosai 의존 에이전트가 wheel 에 딸려온다
     (축 1 이 없앤 "무거운 transitive deps" 재발).
  2. grounded_common 은 순수(logosai import 금지) — 에이전트가 logosai 를
     import 하는 것과 대비되는 경계선. 순수라 단위 테스트가 결정적.

test_kernel_decoupling / test_packaging 은 커널 디렉터리만 본다 — agents/ 는
그들의 스캔 밖이므로 이 파일이 agents/ 쪽 계약을 따로 못박는다.
"""
import os
import tomllib
from pathlib import Path

_ONTOLOGY_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _ONTOLOGY_ROOT / "agents"


def _pyproject():
    with open(_ONTOLOGY_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_agents_not_in_wheel_packages():
    pkgs = _pyproject()["tool"]["setuptools"].get("packages") \
        or _pyproject()["project"].get("packages") or []
    # 명시 목록이어야(자동탐색이면 agents/ 가 빨려들 수 있음) + agents 미포함
    assert "ontology.agents" not in pkgs, \
        "ontology.agents 가 wheel packages 에 있으면 logosai 가 배포에 딸려온다"
    assert not any(p.endswith(".agents") or p == "agents" for p in pkgs), pkgs


def test_agents_dir_exists_with_agents():
    # 이 테스트의 전제(에이전트가 여기 산다)가 유효한지
    assert _AGENTS_DIR.is_dir()
    files = {p.name for p in _AGENTS_DIR.glob("*.py")}
    assert "grounded_common.py" in files
    assert {"grounded_qa_agent.py"} <= files


def test_grounded_common_is_pure_no_logosai():
    src = (_AGENTS_DIR / "grounded_common.py").read_text()
    # 순수 로직 — logosai/aiohttp/torch import 금지 (경계 대비선)
    for banned in ("import logosai", "from logosai", "import aiohttp", "import torch"):
        assert banned not in src, f"grounded_common 이 {banned} 를 import — 순수성 위반"


def test_agents_do_import_logosai():
    # 대비: 에이전트는 logosai 에이전트다(그래서 wheel 에서 제외돼야 하는 것).
    for name in ("grounded_qa_agent",):
        src = (_AGENTS_DIR / f"{name}.py").read_text()
        assert "logosai" in src, f"{name} 가 logosai 를 안 씀(설계상 SimpleAgent 여야)"


if __name__ == "__main__":
    import sys
    failed = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS {n}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL {n}: {e}")
    print("OK" if not failed else f"FAILED {failed}")
    sys.exit(1 if failed else 0)
