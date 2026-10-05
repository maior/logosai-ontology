"""QueryPlanner._call_llm 일시 오류(503/429) 백오프 재시도 검증.

배경: Gemini flash-lite 수요 스파이크 시 503 UNAVAILABLE 이 수십 초 지속 —
기존 _call_llm 은 재시도 없이 즉시 raise → planning_error → "에이전트 사용 불가"
응답 (2026-07-14 실측 3회 연속).

직접 실행: .venv/bin/python ontology/orchestrator/test_planner_503_retry.py
"""

import asyncio
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ontology"))

from orchestrator.query_planner import QueryPlanner  # noqa: E402


class _FakeModels:
    def __init__(self, fail_times, error_msg):
        self.calls = 0
        self.fail_times = fail_times
        self.error_msg = error_msg

    def generate_content(self, model=None, config=None, contents=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.error_msg)

        class R:
            text = '{"ok": true}'
        return R()


def _planner(fail_times, error_msg) -> QueryPlanner:
    p = QueryPlanner.__new__(QueryPlanner)
    class _C:
        pass
    p.client = _C()
    p.client.models = _FakeModels(fail_times, error_msg)
    return p


def main():
    fails = []

    def t(name, cond):
        print(("PASS  " if cond else "FAIL  ") + name)
        if not cond:
            fails.append(name)

    # 재시도 지연 최소화 (0.01 — 실제 sleep 경로도 통과시켜 import 누락 감지)
    QueryPlanner._TRANSIENT_RETRY_DELAYS = (0.01, 0.01, 0.01)

    # 1. 503 2회 후 성공 → 총 3회 호출, 결과 반환
    p = _planner(2, "503 UNAVAILABLE. model overloaded")
    out = asyncio.run(p._call_llm("prompt"))
    t("R-1 503 2회 후 성공: 재시도로 회복", out == '{"ok": true}' and p.client.models.calls == 3)

    # 2. 429 도 일시 오류로 재시도
    p2 = _planner(1, "429 RESOURCE_EXHAUSTED rate limit")
    out2 = asyncio.run(p2._call_llm("prompt"))
    t("R-2 429 도 재시도 대상", out2 == '{"ok": true}' and p2.client.models.calls == 2)

    # 3. 지속 503 → 재시도 소진 후 raise (기존 계약 유지: 예외 전파)
    p3 = _planner(99, "503 UNAVAILABLE forever")
    raised = False
    try:
        asyncio.run(p3._call_llm("prompt"))
    except Exception:
        raised = True
    t("R-3 지속 503: 소진 후 예외 전파 (4회 시도)", raised and p3.client.models.calls == 4)

    # 4. 비일시 오류(400 등)는 즉시 raise — 낭비 재시도 금지
    p4 = _planner(99, "400 INVALID_ARGUMENT bad request")
    raised4 = False
    try:
        asyncio.run(p4._call_llm("prompt"))
    except Exception:
        raised4 = True
    t("R-4 비일시 오류: 즉시 raise (1회만)", raised4 and p4.client.models.calls == 1)

    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
