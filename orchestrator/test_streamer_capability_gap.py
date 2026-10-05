"""planning_complete 이벤트에 capability_gap 포함 검증 (2026-07-11).

배경 (잠복 버그): 플래너가 capability_gap detected=true 인 계획을 만들어도
progress_streamer.planning_complete 가 이벤트 data 에 capability_gap 을 담지
않아, logos_api 의 gap fallback 분기(2026-05-09)가 이 경로로는 **한 번도
발화할 수 없었다** (라이브 실측: 오프라인 create_plan 은 gap 정확, SSE
planning_complete 는 gap 부재).

계약: planning_complete 이벤트 data 에 plan.capability_gap 이 그대로 실린다
(None 이면 None — 소비자가 detected 를 판단).

실행: .venv/bin/python ontology/orchestrator/test_streamer_capability_gap.py
"""
import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ontology"))


class _StubPlan:
    workflow_strategy = "sequential"
    reasoning = "테스트"
    stages = []
    capability_gap = {"detected": True, "missing_capabilities": ["korean_braille"],
                      "suggested_agent_description": "점자 변환 에이전트", "reason": "전문 에이전트 부재"}

    def get_stage_count(self): return 0
    def get_total_agents(self): return 0


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _last_event(streamer):
    """emit 은 내부 버퍼에 항상 적재 — 리스너 등록 방식과 무관한 관찰 지점."""
    assert streamer._event_buffer, "이벤트 미방출"
    return streamer._event_buffer[-1]


def test_planning_complete_event_carries_capability_gap():
    from ontology.orchestrator.progress_streamer import ProgressStreamer

    streamer = ProgressStreamer(workflow_id="t1")
    _run(streamer.planning_complete(_StubPlan()))

    data = _last_event(streamer).data
    assert "capability_gap" in data, "planning_complete data 에 capability_gap 부재 (gap 분기 영구 미발화)"
    assert data["capability_gap"]["detected"] is True


def test_planning_complete_gap_none_passthrough():
    from ontology.orchestrator.progress_streamer import ProgressStreamer

    class _NoGap(_StubPlan):
        capability_gap = None

    streamer = ProgressStreamer(workflow_id="t2")
    _run(streamer.planning_complete(_NoGap()))
    assert _last_event(streamer).data.get("capability_gap") is None


def main():
    fails = []
    for fn in (test_planning_complete_event_carries_capability_gap,
               test_planning_complete_gap_none_passthrough):
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            print("FAIL", fn.__name__, "→", type(e).__name__, str(e)[:120])
            fails.append(fn.__name__)
    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
