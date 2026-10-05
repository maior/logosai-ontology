"""루트 Logos/__init__.py 차단 + sys.path 준비 — tests/·agents/ conftest 가 함께 쓴다.

Logos/__init__.py 는 깨진 import 가 남은 옛 SDK 잔재다. 테스트 수집 중 그걸 밟지 않도록
더미 모듈을 sys.modules 에 넣고, Logos/ 를 sys.path 에 둬서 `from ontology.x import y` 가
풀리게 한다. 전엔 이 코드가 tests/conftest.py 에만 있어서, agents/ 테스트를 단독으로 돌리면
수집 단계에서 실패했다 — 전체 실행에선 tests/ 가 먼저 수집되며 이 코드가 돈 덕에 우연히
통과했다 (순서 의존, 2026-10-05). 사본을 두지 않는다 — 한쪽만 고쳐진다.
"""
import os
import sys
import types

_ontology_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_logos_root = os.path.dirname(_ontology_root)

for mod_name in list(sys.modules.keys()):
    if mod_name == "Logos" or mod_name.startswith("Logos."):
        del sys.modules[mod_name]

_dummy_logos = types.ModuleType("Logos")
_dummy_logos.__path__ = [_logos_root]
_dummy_logos.__file__ = os.path.join(_logos_root, "__init__.py")
_dummy_logos.__package__ = "Logos"
sys.modules["Logos"] = _dummy_logos

if _logos_root not in sys.path:
    sys.path.insert(0, _logos_root)
