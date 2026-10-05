"""agents/ 테스트용 conftest — tests/ 와 같은 import 준비를 쓴다.

없으면 agents/ 테스트를 단독으로 돌릴 때 깨진 루트 Logos/__init__.py 를 밟아 수집이 실패한다
(tests/_logos_import_guard.py 의 docstring 참고).
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tests", "_logos_import_guard.py"))
