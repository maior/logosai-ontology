"""테스트 데이터 격리 — 테스트가 **프로덕션 `data/` 를 오염시키지 못하게**.

**실제로 두 번 밟았다.** 테스트가 만든 네임스페이스 파일이 실 `data/` 에 쌓여
1157개가 됐고, `/admin/system` 이 전 네임스페이스의 store 를 로드하느라
**120초+** 가 걸렸다. 1136개를 지웠는데 **전체 스위트를 한 번 돌리자 다시
생겼다** — 개별 테스트를 고치는 방식이 통하지 않는다는 증거다.

그래서 `conftest.py` 의 autouse fixture 가 데이터 경로를 tmp 로 **전역
리다이렉트**한다. 개별 테스트가 monkeypatch 를 잊어도 오염되지 않는다.

**이 파일의 핵심은 두 번째 클래스다.** 새 모듈이 데이터 경로 상수를 추가하면
격리가 조용히 새는데, 그걸 잡으려면 **소스를 스캔해서 conftest 가 덮는 목록과
대조**해야 한다 — `test_packaging.py` 가 선언 의존성을 검사하는 것과 같은 부류.
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_REAL_DATA = _ROOT / "data"

# conftest 가 격리하는 (모듈 경로, 상수명) 목록. 아래 스캔 테스트가 이 목록이
# 실제 소스와 일치하는지 검사한다.
ISOLATED = (
    ("core/chunk_store.py", "_DEFAULT_DATA_DIR"),
    ("core/review_store.py", "_DEFAULT_DATA_DIR"),
    ("core/search_qa.py", "_DEFAULT_DATA_DIR"),
    ("core/eval_history.py", "_DEFAULT_DATA_DIR"),
    ("core/experiment.py", "_DEFAULT_DATA_DIR"),
    ("core/retrieval_config.py", "_DEFAULT_DATA_DIR"),
    ("core/coverage_expectations.py", "_DEFAULT_DATA_DIR"),
    ("core/npy_backend.py", "_DEFAULT_CACHE_DIR"),
    ("engines/knowledge_graph_clean.py", "_DEFAULT_DATA_DIR"),
    ("server/service.py", "_DEFAULT_DATA_DIR"),   # data/datasets — 업로드 저장소
)


class TestIsolationIsActive:
    """격리가 실제로 걸려 있는가 — fixture 가 조용히 빠지면 여기서 잡힌다."""

    @pytest.mark.parametrize("module_path,const", ISOLATED)
    def test_constant_points_outside_real_data(self, module_path, const):
        import importlib
        mod_name = "ontology." + module_path[:-3].replace("/", ".")
        module = importlib.import_module(mod_name)
        value = Path(getattr(module, const))
        assert _REAL_DATA not in value.parents and value != _REAL_DATA, (
            f"{mod_name}.{const} 가 실 data/ 를 가리킨다 — 테스트가 프로덕션을 "
            f"오염시킨다: {value}")

    def test_chunk_store_writes_to_tmp(self):
        from ontology.core.chunk_store import ChunkStore
        assert _REAL_DATA not in ChunkStore(namespace="iso_probe").path.parents

    def test_kg_checkpoint_writes_to_tmp(self):
        """KG 체크포인트가 가장 큰 오염원이었다 (`kg_apprns_*.json` 수백 개)."""
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        engine = KnowledgeGraphEngine(fast_mode=True, namespace="iso_probe")
        path = Path(engine._checkpoint_path()
                    if hasattr(engine, "_checkpoint_path")
                    else engine.checkpoint_path)
        assert _REAL_DATA not in path.parents

    def test_golden_set_and_eval_history_write_to_tmp(self):
        from ontology.core.eval_history import EvalHistory
        from ontology.core.search_qa import GoldenSet
        assert _REAL_DATA not in GoldenSet(namespace="iso_probe").path.parents
        assert _REAL_DATA not in EvalHistory(namespace="iso_probe").path.parents

    def test_real_data_is_untouched_by_this_run(self):
        """이 테스트 세션이 실 data/ 에 `iso_probe` 를 만들지 않았는가."""
        leaked = list(_REAL_DATA.glob("*iso_probe*"))
        assert leaked == [], f"실 data/ 로 새어나간 파일: {leaked}"


class TestNoUncoveredDataPaths:
    """**새 모듈이 데이터 경로를 추가하면 여기서 실패한다.**

    격리 목록을 손으로 유지하면 반드시 뒤처진다 — 실제로 `eval_history.py` 를
    어제 만들었고 그때 격리 목록이 있었다면 빠뜨렸을 것이다. 그래서 소스를
    스캔해서 대조한다.
    """

    # `parent.parent / "data"` 로 저장 경로를 만드는 모듈-레벨 상수
    # `.resolve()` 유무와 "data" 뒤의 추가 세그먼트("datasets" 등)까지 잡는다 —
    # server/service.py 의 datasets 경로가 정확히 그 모양으로 이 스캔을 피해갔다
    # (첫 버전의 사각지대, 실측으로 발견).
    PATTERN = re.compile(
        r'^(?P<name>_[A-Z0-9_]+)\s*=\s*Path\(__file__\)(?:\.resolve\(\))?'
        r'\.parent\.parent\s*/\s*"data"',
        re.M)

    def _scan(self):
        found = set()
        for sub in ("core", "engines", "builder", "server"):
            for path in sorted((_ROOT / sub).glob("*.py")):
                for match in self.PATTERN.finditer(path.read_text(encoding="utf-8")):
                    rel = f"{sub}/{path.name}"
                    found.add((rel, match.group("name")))
        return found

    def test_every_data_path_constant_is_isolated(self):
        found = self._scan()
        missing = found - set(ISOLATED)
        assert not missing, (
            "격리되지 않은 데이터 경로 상수가 있다 — conftest 의 "
            "_isolate_data_dirs 와 이 파일의 ISOLATED 에 추가하라: "
            f"{sorted(missing)}")

    def test_isolated_list_has_no_stale_entries(self):
        """지워진 모듈이 목록에 남으면 격리가 조용히 실패한다(패치 대상 없음)."""
        found = self._scan()
        stale = set(ISOLATED) - found
        assert not stale, f"ISOLATED 에 실존하지 않는 항목: {sorted(stale)}"

    def test_scan_finds_the_known_ones(self):
        """스캔 정규식이 죽으면 위 두 테스트가 공허하게 통과한다 — 판별력 확인."""
        found = self._scan()
        assert ("core/chunk_store.py", "_DEFAULT_DATA_DIR") in found
        assert len(found) >= 5
