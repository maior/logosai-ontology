"""ontology llm_manager langchain 탈피 검증 (2026-07-07 L2 스윕).

배경: llm_manager.py 가 모듈 레벨 unguarded `from langchain_openai import
ChatOpenAI` 등을 가져 langchain 미설치 시 ontology 코어(planner 경로)가
통째로 즉사했다. 실행은 GeminiLLMWrapper(native)지만 import 가 하드 의존.

계약:
  - langchain 차단 환경에서 llm_manager 모듈 로드 성공
  - google 경로: 기존 GeminiLLMWrapper 그대로 (프로덕션 불변)
  - openai/anthropic/openai-호환 경로: LLMClient 기반 ainvoke 호환 객체 반환
  - 소스에 langchain import 잔존 0 (재생산 가드)

실행: .venv/bin/python ontology/tests/test_llm_manager_no_langchain.py
"""
import importlib.util
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ONTOLOGY = os.path.join(_ROOT, "ontology")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ONTOLOGY)
sys.path.insert(0, os.path.join(_ROOT, "logosai"))


class _BlockLangchain:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("langchain"):
            raise ImportError(f"blocked for test: {fullname}")
        return None


def main():
    fails = []

    def t(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    path = os.path.join(_ONTOLOGY, "core", "llm_manager.py")
    src = open(path, encoding="utf-8").read()

    # ── 정적 가드 ──
    hits = re.findall(r"^\s*(?:from|import)\s+langchain", src, re.MULTILINE)
    t("S-1 소스에 langchain import 0건", not hits)

    # ── 동적: langchain 차단 로드 ──
    removed = {k: v for k, v in list(sys.modules.items()) if k.startswith("langchain")}
    for k in removed:
        del sys.modules[k]
    blocker = _BlockLangchain()
    sys.meta_path.insert(0, blocker)
    try:
        try:
            # 상대 import 가 있으므로 패키지 컨텍스트로 로드
            import importlib as _il
            for k in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
                del sys.modules[k]
            mod = _il.import_module("core.llm_manager")
            t("D-1 langchain 차단 환경에서 모듈 로드 성공", True)
        except Exception as e:
            t(f"D-1 langchain 차단 환경에서 모듈 로드 성공 ({type(e).__name__}: {str(e)[:60]})", False)
            print("\nRESULT: RED (로드 실패로 이후 검증 불가)")
            return 1

        # ── 동작: openai 경로가 ainvoke 호환 객체 반환 ──
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
        mgr_cls = getattr(mod, "OntologyLLMManager", None)
        t("D-2 OntologyLLMManager 클래스 존재", mgr_cls is not None)
        if mgr_cls is not None:
            mgr = mgr_cls.__new__(mgr_cls)  # 무거운 __init__ 우회
            maker = getattr(mgr, "_create_openai_llm", None) or getattr(mod, "_make_openai_llm", None)
            # 구현 형태는 자유 — 최소 계약: 어떤 경로로든 openai LLM 객체가
            # ainvoke 코루틴 함수를 갖는다 (langchain 없이)
            candidates = []
            if maker:
                try:
                    candidates.append(maker("gpt-test", 0.1))
                except TypeError:
                    try:
                        candidates.append(maker(model="gpt-test", temperature=0.1))
                    except Exception:
                        pass
                except Exception:
                    pass
            ok = any(hasattr(c, "ainvoke") for c in candidates if c is not None)
            t("D-3 openai 경로 LLM 객체가 ainvoke 제공 (langchain 없이)", ok)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(removed)

    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
