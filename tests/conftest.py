"""Conftest that fixes sys.path for standalone ontology testing.

Logos/__init__.py is an old SDK artifact with broken imports.
We block it via a dummy sys.modules entry, then keep Logos/ in
sys.path so that `from ontology.ml.config import ...` resolves
correctly (Python finds ontology/ under Logos/).
"""
import importlib
import sys
import os
import types

_ontology_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_logos_root = os.path.dirname(_ontology_root)

# 1. Block Logos/__init__.py from being loaded by inserting a dummy module.
#    This prevents the broken `from .agent import LogosAIAgent` inside it.
for mod_name in list(sys.modules.keys()):
    if mod_name == "Logos" or mod_name.startswith("Logos."):
        del sys.modules[mod_name]

_dummy_logos = types.ModuleType("Logos")
_dummy_logos.__path__ = [_logos_root]
_dummy_logos.__file__ = os.path.join(_logos_root, "__init__.py")
_dummy_logos.__package__ = "Logos"
sys.modules["Logos"] = _dummy_logos

# 2. Ensure Logos/ is in sys.path so `from ontology.x import y` works.
#    (ontology/ lives at Logos/ontology/)
if _logos_root not in sys.path:
    sys.path.insert(0, _logos_root)


# ─── 데이터 격리 (autouse) ──────────────────────────────────────────
#
# **실제로 두 번 밟았다.** 테스트가 만든 네임스페이스 파일이 실 `data/` 에 쌓여
# 1157개가 됐고, `/admin/system` 이 전 네임스페이스의 store 를 로드하느라
# **120초+** 가 걸렸다. 1136개를 지웠는데 **전체 스위트를 한 번 돌리자 다시
# 생겼다** — 개별 테스트에서 monkeypatch 하는 방식이 통하지 않는다는 증거다
# (새 테스트가 잊으면 그대로 샌다).
#
# 그래서 전역 리다이렉트다. 개별 테스트의 monkeypatch 는 그대로 유효하고
# (더 좁은 범위가 이긴다), 잊었을 때의 바닥이 실 `data/` 가 아니라 tmp 가 된다.
#
# 격리 목록이 소스와 어긋나면 `test_data_isolation.py` 가 실패한다 — 손으로
# 유지하는 목록은 반드시 뒤처지므로 소스를 스캔해서 대조한다.
import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_data_dirs(tmp_path_factory):
    """모든 테스트의 데이터 경로를 세션 tmp 로 돌린다.

    session 스코프인 이유: 함수마다 새 디렉터리를 주면 같은 네임스페이스를
    두 테스트가 이어서 쓰는 경우(싱글턴 재사용)에 앞 테스트의 상태가 사라져
    원인 파악이 어려운 실패가 난다. 격리의 목적은 **실 data/ 보호**이고
    테스트 간 격리는 각 테스트가 tmp_path·reset_* 으로 이미 한다.
    """
    root = tmp_path_factory.mktemp("ontology_data")
    targets = [
        ("ontology.core.chunk_store", "_DEFAULT_DATA_DIR"),
        ("ontology.core.review_store", "_DEFAULT_DATA_DIR"),
        ("ontology.core.search_qa", "_DEFAULT_DATA_DIR"),
        ("ontology.core.eval_history", "_DEFAULT_DATA_DIR"),
        ("ontology.core.experiment", "_DEFAULT_DATA_DIR"),
        ("ontology.core.retrieval_config", "_DEFAULT_DATA_DIR"),
        ("ontology.core.coverage_expectations", "_DEFAULT_DATA_DIR"),
        ("ontology.core.npy_backend", "_DEFAULT_CACHE_DIR"),
        ("ontology.engines.knowledge_graph_clean", "_DEFAULT_DATA_DIR"),
        ("ontology.server.service", "_DEFAULT_DATA_DIR"),
    ]
    saved = []
    for mod_name, const in targets:
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            # 무거운 의존성이 없는 환경에서도 나머지 격리는 걸려야 한다.
            continue
        saved.append((module, const, getattr(module, const, None)))
        setattr(module, const, root)
    yield root
    for module, const, old in saved:
        if old is not None:
            setattr(module, const, old)
