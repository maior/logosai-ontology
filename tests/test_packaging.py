"""
축 1 (마지막 구간) — 배포 계약.

커널이 in-repo 에서 import 되는 것과 **소비자가 설치할 수 있는 것**은 다르다.
aicoach 가 `ontology = { path = "...", editable = true }` 로 물어야 축 1 이
완성되는데, pyproject 에 네 가지 결함이 있었다 (전부 실증):

1. langchain-core / langchain-openai 를 하드 의존으로 선언 — 그런데 2026-07-07
   L2/L3 스윕이 langchain 을 이미 제거해 **사용처가 0건**이다. 쓰지도 않는
   패키지를 모든 소비자가 설치하게 만든다. 커널을 가볍게 만들자는 축 1 의
   목적과 정면으로 어긋난다.
2. numpy 미선언 — semantic_index / npy_backend / es_backend 가 **모듈 레벨**에서
   import 한다. 선언 없이 설치하면 첫 import 에서 죽는다.
3. gemini extra 가 google-generativeai(구 SDK)를 가리킴 — 코드는
   `from google import genai`, 즉 **google-genai** 라는 다른 패키지를 쓴다.
   extra 를 설치해도 Gemini 가 안 켜진다.
4. [build-system] 부재 — 빌드 백엔드가 없으면 설치 자체가 불확실하다.

여기서 고정하는 것: **선언된 의존성이 실제 import 를 덮는가**, 그리고 그 반대로
**안 쓰는 것을 선언하지 않는가**. 이 테스트는 pyproject 를 파싱만 하므로
네트워크도 설치도 필요 없다.
"""

import re
import tomllib
from pathlib import Path

import pytest

_ONTOLOGY_ROOT = Path(__file__).resolve().parent.parent

# 라이브러리로 물었을 때 base 의존성만으로 import 돼야 하는 범위.
# orchestrator/ml 은 에이전트 스택이라 여기 없다 (축 1 의 경계).
KERNEL_DIRS = ("core", "builder", "engines")

# server/ 는 선택적이다 — 라이브러리로 쓰는 소비자(aicoach)는 FastAPI 앱이
# 필요 없다. 그래서 fastapi/pydantic 은 base 가 아니라 `server` extra 다.
SERVER_DIRS = ("server",)


@pytest.fixture(scope="module")
def pyproject():
    with open(_ONTOLOGY_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def declared(pyproject):
    """선언된 런타임 의존성 이름 집합 (버전 스펙 제거)."""
    return {re.split(r"[><=!\[~;]", spec)[0].strip().lower()
            for spec in pyproject["project"]["dependencies"]}


def _sources(dirs):
    for name in dirs:
        for path in (_ONTOLOGY_ROOT / name).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _kernel_sources():
    return _sources(KERNEL_DIRS)


def _module_level_imports(path: Path) -> set:
    """모듈 레벨(들여쓰기 0) import 만 — 함수 안 지연 import 는 선택 의존이다."""
    found = set()
    pattern = re.compile(r"^(?:import|from)\s+([a-zA-Z_][\w.]*)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)  # 들여쓰기 없는 줄만 매치된다
        if match:
            found.add(match.group(1).split(".")[0])
    return found


class TestDeclaredDepsCoverModuleLevelImports:
    """설치 즉시 죽지 않으려면, 모듈 레벨 import 는 전부 선언돼야 한다."""

    THIRD_PARTY = {"numpy", "networkx", "loguru", "yaml", "aiohttp",
                   "fastapi", "pydantic"}
    # 이름이 다른 것들: import 이름 → 배포 이름
    DIST_NAME = {"yaml": "pyyaml"}

    def test_every_module_level_third_party_import_is_declared(self, declared):
        missing = {}
        for path in _kernel_sources():
            for name in _module_level_imports(path) & self.THIRD_PARTY:
                dist = self.DIST_NAME.get(name, name)
                if dist not in declared:
                    missing.setdefault(dist, []).append(
                        str(path.relative_to(_ONTOLOGY_ROOT)))
        assert not missing, (
            "모듈 레벨 import 인데 선언되지 않았다 (설치 후 첫 import 에서 죽는다): "
            + "; ".join(f"{d} ← {', '.join(files[:3])}"
                        for d, files in missing.items()))

    def test_numpy_is_declared(self, declared):
        """회귀 고정 — 실제로 빠져 있었고, 커널 3개 파일이 모듈 레벨로 쓴다."""
        assert "numpy" in declared

    def test_server_imports_are_covered_by_the_server_extra(self, pyproject,
                                                            declared):
        """server/ 는 선택적이다 — 라이브러리 소비자(aicoach)는 FastAPI 앱이
        필요 없다. 그러니 fastapi/pydantic 은 base 가 아니라 extra 여야 하고,
        대신 그 extra 가 실제 import 를 덮어야 한다.
        """
        extra = {re.split(r"[><=!\[~;]", s)[0].strip().lower()
                 for s in pyproject["project"]["optional-dependencies"]["server"]}
        available = declared | extra
        needed = set()
        for path in _sources(SERVER_DIRS):
            needed |= _module_level_imports(path) & {"fastapi", "pydantic"}
        assert needed, "전제가 깨졌다: server/ 가 fastapi 를 안 쓴다"
        missing = {n for n in needed if n not in available}
        # pydantic 은 fastapi 가 끌고 오지만, 직접 import 하면 직접 선언이 맞다
        assert not missing or missing == {"pydantic"}, (
            f"server extra 가 덮지 못하는 import: {missing}")


class TestNoUnusedHeavyDeps:
    """안 쓰는 것을 선언하지 않는다 — 커널을 가볍게 하자는 게 축 1 의 목적이다."""

    def test_langchain_is_not_a_dependency(self, declared):
        """2026-07-07 L2/L3 스윕이 langchain 을 제거했다. 선언만 남아 있었다."""
        assert not any("langchain" in d for d in declared), (
            "langchain 은 코드에서 사용처가 0건이다 — 선언에서 빼야 한다")

    def test_no_kernel_source_imports_langchain(self):
        """위 단언의 근거를 코드로 확인한다 (선언과 현실의 drift 방지)."""
        for path in _kernel_sources():
            assert not any(name.startswith("langchain")
                           for name in _module_level_imports(path)), path


class TestOptionalExtras:
    def test_gemini_extra_names_the_sdk_the_code_actually_uses(self, pyproject):
        """코드는 `from google import genai` — 배포명은 google-genai 다.
        google-generativeai 는 **다른(구) 패키지**라 설치해도 안 켜진다.
        """
        extras = pyproject["project"]["optional-dependencies"]
        gemini = " ".join(extras["gemini"]).lower()
        assert "google-genai" in gemini
        assert "google-generativeai" not in gemini

    def test_es_extra_exists(self, pyproject):
        """축 3 의 ES 백엔드는 선택 의존이다 — 없으면 memory 로 degrade 한다."""
        extras = pyproject["project"]["optional-dependencies"]
        assert "elasticsearch" in " ".join(extras["es"]).lower()

    def test_embeddings_are_optional_not_required(self, declared):
        """sentence-transformers 는 필수가 아니다 — 없으면 semantic_index 가
        빈 결과로 degrade 한다(설계). 필수로 선언하면 torch 까지 끌고 온다.
        """
        assert "sentence-transformers" not in declared
        assert "torch" not in declared


class TestBuildable:
    def test_build_system_is_declared(self, pyproject):
        assert "build-system" in pyproject
        assert pyproject["build-system"]["build-backend"]

    def test_package_discovery_is_explicit(self, pyproject):
        """이 저장소는 프로젝트 루트가 곧 패키지(`ontology/`)라는 특이 구조라,
        setuptools 자동 탐색이 엉뚱한 것을 잡는다 — 명시해야 한다."""
        tool = pyproject.get("tool", {}).get("setuptools", {})
        assert tool.get("packages") or tool.get("package-dir")


# ── 런타임 데이터 파일 동봉 (2026-07-20) ──────────────────────────────────
# 격리 venv 에 wheel 만 설치했을 때 llm_config.yaml 이 없어
#   "Configuration file not found: .../site-packages/ontology/config/llm_config.yaml"
#   "⚠️ Using minimal hardcoded fallback configuration"
# 로 degrade 했다. .py 만 담기는 게 기본이라 코드가 참조하는 데이터 파일은
# package-data 로 명시해야 한다. 소비자는 경고를 보고 설치가 깨졌다고 오해한다.

def test_config_yaml_is_declared_as_package_data(pyproject):
    """코드가 런타임에 읽는 config/llm_config.yaml 이 배포 대상으로 선언돼야 한다."""
    pd = pyproject.get("tool", {}).get("setuptools", {}).get("package-data", {})
    patterns = pd.get("ontology", [])
    assert any("config" in p and "yaml" in p for p in patterns), (
        f"config/*.yaml 이 package-data 에 없다 (현재: {patterns}). "
        "llm_config_loader 가 참조하므로 wheel 에 동봉되어야 한다."
    )


def test_referenced_data_files_exist():
    """선언만 하고 파일이 없으면 배포 시점에 조용히 빠진다."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "config" / "llm_config.yaml").is_file(), \
        "llm_config_loader.py:34 가 참조하는 config/llm_config.yaml 이 없다"


# ── logosai 는 선택 의존이다 (2026-07-20) ─────────────────────────────────
# logosai 가 PyPI 에 올라오면서 "그냥 의존으로 걸면 되지 않나" 가 자연스러운
# 유혹이 됐다. 걸면 축 1(커널 분리)이 통째로 무효가 된다 — KG import 시
# logosai 31개 → 0 으로 만든 작업이고, aicoach 가 온톨로지 채택을 포기한
# 이유가 정확히 "heavy transitive deps" 였다.
# 코드는 이미 지연 import + _logosai_available() 프로브로 선택 경로를 만들어
# 놨다. 포장만 그 설계를 따르면 된다.

def test_logosai_is_not_a_core_dependency(pyproject):
    """logosai 를 필수 의존으로 올리면 커널 분리가 무효가 된다."""
    core = pyproject["project"]["dependencies"]
    assert not any(d.split(">=")[0].split("[")[0].strip() == "logosai" for d in core), (
        f"logosai 가 필수 의존에 있다 ({core}). "
        "openai/anthropic 프로바이더 전용 선택 경로이므로 extra 여야 한다."
    )


def test_logosai_is_offered_as_an_extra(pyproject):
    """선택 경로를 쓰려는 소비자가 설치할 방법은 있어야 한다."""
    extras = pyproject["project"]["optional-dependencies"]
    joined = [d for v in extras.values() for d in v]
    assert any("logosai" in d for d in joined), (
        f"어떤 extra 에도 logosai 가 없다. core/llm_provider.py 의 "
        f"LogosAIProvider 를 설치할 방법이 사라진다. (extras: {list(extras)})"
    )


#: 실행부가 logosai.orchestration 으로 옮겨지며 남은 호환 경로 (2026-10-04).
#: 이 파일들은 logosai 모듈을 '가리키는 것'이 전부라 모듈 레벨 import 가 곧 본체다.
#: orchestrator 는 커널이 아니라 에이전트 스택이고(축 1 의 경계), 이 경로를 쓰려면
#: `logosai` extra 가 필요하다. 커널(core·builder·engines)은 여전히 끌어오지 않는다.
COMPAT_ALIASES = {
    "models", "exceptions", "agent_registry", "progress_streamer",
    "data_transformer", "plan_validator", "result_aggregator", "execution_engine",
}
_COMPAT_MARKER = re.compile(r'^__compat_alias__\s*=\s*"logosai\.orchestration\.(\w+)"', re.M)


def _compat_target(path: Path):
    m = _COMPAT_MARKER.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


#: SDK 메커니즘을 상속하는 Logos 인스턴스 (2026-10-04, P2). 별칭과 달리 본문(Logos
#: 프롬프트·관문·키워드)이 여기 있고, 기반 클래스만 SDK 에서 온다. 표식으로 좁게 묶는다.
SDK_SUBCLASSES = {"query_planner": "planner", "workflow_orchestrator": "workflow_orchestrator"}
_SDK_BASE_MARKER = re.compile(r'^__sdk_base__\s*=\s*"logosai\.orchestration\.(\w+)"', re.M)


def _sdk_base(path: Path):
    m = _SDK_BASE_MARKER.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def test_no_module_level_logosai_import():
    """지연 import 계약 — 모듈 상단에서 끌어오면 extra 가 무의미해진다.

    예외는 호환 별칭 파일뿐이다. 그 범위는 아래 테스트가 좁게 묶는다.
    """
    root = Path(__file__).resolve().parent.parent
    dirs = ["core", "builder", "engines", "server", "utils", "ml", "orchestrator", "system"]
    bad = []
    for d in dirs:
        for p in (root / d).rglob("*.py"):
            if _compat_target(p) or _sdk_base(p):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if re.match(r"^(from|import)\s+logosai", line):
                    bad.append(f"{p.relative_to(root)}:{i}")
    assert not bad, f"모듈 레벨 logosai import: {bad}"


def test_compat_alias_exemption_is_narrow():
    """예외가 번지지 않는다 — 표식은 정해진 파일에만, 각자 같은 이름만 가리킨다."""
    root = Path(__file__).resolve().parent.parent
    marked = {}
    for p in root.rglob("*.py"):
        if "tests" in p.parts:
            continue
        target = _compat_target(p)
        if target:
            marked[p.relative_to(root).as_posix()] = target
    expected = {f"orchestrator/{n}.py": n for n in COMPAT_ALIASES}
    assert marked == expected, f"호환 표식 위치가 다르다: {marked}"
    for rel, target in marked.items():
        src = (root / rel).read_text(encoding="utf-8")
        imports = set(re.findall(r"^\s*from\s+(logosai[\w\.]*)\s+import\s+(\w+)", src, re.M))
        assert imports == {("logosai.orchestration", target)}, (
            f"{rel} 가 자기 정본 외의 logosai 를 끌어온다: {imports}")


def test_sdk_subclass_exemption_is_narrow():
    """SDK 상속 예외도 번지지 않는다 — 정해진 파일이 자기 기반 모듈 하나만 끌어온다."""
    root = Path(__file__).resolve().parent.parent
    marked = {p.relative_to(root).as_posix(): _sdk_base(p)
              for p in root.rglob("*.py") if "tests" not in p.parts and _sdk_base(p)}
    assert marked == {f"orchestrator/{k}.py": v for k, v in SDK_SUBCLASSES.items()}, marked
    for rel, base in marked.items():
        src = (root / rel).read_text(encoding="utf-8")
        mods = set(re.findall(r"^\s*(?:from|import)\s+(logosai[\w\.]*)", src, re.M))
        assert mods == {f"logosai.orchestration.{base}"}, f"{rel} 가 기반 외의 logosai 를 끌어온다: {mods}"


def test_compat_marker_detection_is_not_vacuous(tmp_path):
    """대조군 — 표식 판정이 아무 파일이나 통과시키지 않는다."""
    plain = tmp_path / "plain.py"
    plain.write_text("from logosai.orchestration import models\n", encoding="utf-8")
    marked = tmp_path / "marked.py"
    marked.write_text('__compat_alias__ = "logosai.orchestration.models"\n', encoding="utf-8")
    assert _compat_target(plain) is None
    assert _compat_target(marked) == "models"


def test_logosai_extra_excludes_langchain_era_versions(pyproject):
    """0.11.2 이하는 langchain import 가 살아 있다 (실측: llm_client.py, agent.py).

    memory 의 알려진 지뢰 — langchain_google_genai 가 deprecated google.generativeai
    를 끌어 무한 hang. 하한선을 낮추면 소비자가 그 버전을 받는다.
    """
    extras = pyproject["project"]["optional-dependencies"]
    spec = next(d for v in extras.values() for d in v if "logosai" in d)
    m = re.search(r">=\s*([\d.]+)", spec)
    assert m, f"logosai 하한선이 명시되지 않았다: {spec}"
    parts = tuple(int(x) for x in m.group(1).split("."))
    assert parts >= (0, 12, 0), (
        f"logosai 하한이 {m.group(1)} — 0.12.0 미만은 langchain 시대 코드다."
    )


def test_version_string_matches_pyproject(pyproject):
    """__init__.py 의 버전이 pyproject 와 어긋나면 소비자가 잘못된 버전을 신고한다.

    실제로 2.0.1 배포 때 pyproject 만 올리고 __init__.py:45 하드코딩을 놓쳐
    pip 은 2.0.1, ontology.__version__ 은 2.0.0 을 보고했다.
    """
    root = Path(__file__).resolve().parent.parent
    src = (root / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', src, re.M)
    declared = pyproject["project"]["version"]
    if m:  # 하드코딩 폴백이 있으면 pyproject 와 같아야 한다
        assert m.group(1) == declared, (
            f"__init__.py={m.group(1)} vs pyproject={declared} — 함께 올려야 한다"
        )
