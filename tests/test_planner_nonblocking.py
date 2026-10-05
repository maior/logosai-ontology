"""플래너 LLM 호출이 이벤트 루프를 막지 않는다.

`_call_llm` 은 async 함수 안에서 동기 API(`client.models.generate_content`)를 그대로
불렀다. 호출하는 동안 이벤트 루프 전체가 멈춘다 — 실측: 0.92s 호출 동안 50ms 주기
시계가 0.92s 통째로 멈췄다(2026-10-04). logos_api 는 루프 하나로 모든 사용자를
처리하므로, 한 사용자의 계획 수립(쿼리당 1~5회, 5~8s) 동안 다른 사용자의 스트림도
멈출 수 있다. 산출물 관문의 '백그라운드' 판정도 이 호출을 써서 응답을 0.9s 늦췄다.
"""
import asyncio
import time

import pytest

from ontology.orchestrator.query_planner import QueryPlanner

CALL_SECONDS = 0.4


class _SlowSyncModels:
    def __init__(self, fail_first=None):
        self.calls = 0
        self.fail_first = fail_first

    def generate_content(self, model=None, config=None, contents=None):
        self.calls += 1
        time.sleep(CALL_SECONDS)                     # 동기 네트워크 호출 흉내
        if self.fail_first and self.calls == 1:
            raise RuntimeError(self.fail_first)
        return type("R", (), {"text": f"answer to: {contents}"})()


def _planner(models):
    p = QueryPlanner.__new__(QueryPlanner)
    p.client = type("C", (), {"models": models})()
    return p


async def _max_loop_stall(coro):
    """coro 를 도는 동안 50ms 주기 시계가 가장 오래 멈춘 시간."""
    stop, gaps = asyncio.Event(), []

    async def ticker():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.05)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.1)
    result = await coro
    await asyncio.sleep(0.1)
    stop.set()
    await task
    return result, max(gaps)


async def test_llm_call_does_not_stall_the_event_loop():
    models = _SlowSyncModels()
    result, stall = await _max_loop_stall(_planner(models)._call_llm("하늘 색은?"))

    assert result == "answer to: 하늘 색은?"                      # 결과는 그대로
    assert models.calls == 1
    assert stall < CALL_SECONDS / 2, f"LLM 호출 중 이벤트 루프가 {stall:.2f}s 멈췄다"


async def test_stall_measure_detects_a_blocking_call():
    """대조군 — 측정기가 실제 차단을 잡는다 (막는 코드를 넣으면 빨개진다)."""
    async def blocking():
        time.sleep(CALL_SECONDS)
        return "x"

    _, stall = await _max_loop_stall(blocking())
    assert stall >= CALL_SECONDS * 0.8


async def test_transient_retry_still_works_off_the_loop(monkeypatch):
    """일시 오류 재시도 경로도 그대로다 (재시도 대기는 0 으로 줄여서)."""
    planner = _planner(_SlowSyncModels(fail_first="503 UNAVAILABLE"))
    monkeypatch.setattr(QueryPlanner, "_TRANSIENT_RETRY_DELAYS", (0.01,), raising=False)

    result = await planner._call_llm("q")

    assert result == "answer to: q" and planner.client.models.calls == 2


async def test_non_transient_error_still_raises():
    planner = _planner(_SlowSyncModels(fail_first="400 INVALID_ARGUMENT"))
    with pytest.raises(RuntimeError, match="400"):
        await planner._call_llm("q")
