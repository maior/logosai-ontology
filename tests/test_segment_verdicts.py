"""구간 검증 판정이 이벤트에 실려 나가는가 (Phase D).

배경 (2026-08-07 라이브 실측):
  validation_complete → {}      ← ValidationResult 의 errors/warnings 전량 유실
  transform_start     → {"source": "weather_agent", "target": "summarization_agent"}
  transform_complete  → {}      ← start 에 있던 source/target 조차 사라진다

특히 transform 은 5개 전략(specific/wildcard/type-based/LLM/passthrough)이
**전부 같은 인자 없는 호출**을 해서, 정상 변환과 "변환기가 없어 원본 그대로
흘림(passthrough)"이 관측상 구분 불가였다.

계약: verdict 는 3-state — pass | fail | **skipped**(모름·미수행).
      fail-open 으로 pass 위장 금지.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ontology.orchestrator.progress_streamer import ProgressStreamer
from ontology.orchestrator.models import ProgressEventType


def _collect(coro_factory):
    """스트리머 이벤트를 콜백으로 수집."""
    events = []
    streamer = ProgressStreamer(workflow_id="wf-test")
    streamer.on_event(events.append)  # on_event 는 동기 콜백 (async 는 on_event_async)
    asyncio.run(coro_factory(streamer))
    return events


class _FakeValidationResult:
    def __init__(self, is_valid=True, errors=None, warnings=None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


# ── 계획 검증 ────────────────────────────────────────────────────

def test_validation_complete_carries_pass_verdict():
    events = _collect(lambda s: s.validation_complete(
        _FakeValidationResult(is_valid=True)))
    d = events[-1].data or {}
    assert d.get("verdict") == "pass", d
    assert d.get("error_count") == 0
    assert d.get("warning_count") == 0


def test_validation_warnings_survive_even_on_pass():
    """경고를 달고 통과한 계획 — 종전엔 성공해도 경고가 통째로 사라졌다."""
    events = _collect(lambda s: s.validation_complete(
        _FakeValidationResult(is_valid=True,
                              warnings=["stage 2 는 입력이 비어 있을 수 있음"])))
    d = events[-1].data or {}
    assert d.get("verdict") == "pass"
    assert d.get("warning_count") == 1
    assert "stage 2" in " ".join(d.get("warnings") or [])


def test_validation_without_result_is_skipped_not_pass():
    """판정을 못 받았으면 '통과'가 아니라 '모름'이다 (fail-open 금지)."""
    events = _collect(lambda s: s.validation_complete())
    d = events[-1].data or {}
    assert d.get("verdict") == "skipped", d


def test_validation_error_carries_fail_verdict_and_reasons():
    events = _collect(lambda s: s.validation_error(
        ["agent 'ghost_agent' not found", "circular dependency: a→b→a"]))
    d = events[-1].data or {}
    assert d.get("verdict") == "fail", d
    assert d.get("error_count") == 2
    assert any("ghost_agent" in e for e in (d.get("errors") or []))


# ── 핸드오프(변환) ────────────────────────────────────────────────

def test_transform_complete_keeps_source_and_target():
    """start 에는 있고 complete 에는 없던 정보 — 어느 구간인지 알 수 없었다."""
    events = _collect(lambda s: s.transform_complete(
        "weather_agent", "summarization_agent", strategy="type_based"))
    d = events[-1].data or {}
    assert d.get("source") == "weather_agent"
    assert d.get("target") == "summarization_agent"


def test_transform_strategy_is_reported():
    events = _collect(lambda s: s.transform_complete(
        "a", "b", strategy="llm_fallback"))
    d = events[-1].data or {}
    assert d.get("strategy") == "llm_fallback"
    assert d.get("verdict") == "pass"


def test_passthrough_is_skipped_not_pass():
    """변환기가 없어 원본을 그대로 흘린 것은 '성공'이 아니다.

    이 구분이 Phase D 의 핵심 — 종전엔 정상 변환과 완전히 동일하게 보고됐다.
    """
    events = _collect(lambda s: s.transform_complete(
        "a", "b", strategy="passthrough"))
    d = events[-1].data or {}
    assert d.get("verdict") == "skipped", d
    assert d.get("strategy") == "passthrough"
    assert d.get("reason"), "왜 건너뛰었는지 사유가 있어야 한다"


def test_transform_failure_carries_fail_verdict():
    events = _collect(lambda s: s.transform_complete(
        "a", "b", success=False, strategy="llm_fallback", reason="LLM timeout"))
    assert events[-1].type == ProgressEventType.TRANSFORM_ERROR
    d = events[-1].data or {}
    assert d.get("verdict") == "fail"
    assert "timeout" in (d.get("reason") or "")


def test_transform_without_strategy_stays_backward_compatible():
    """기존 호출부(인자 없음)가 깨지지 않아야 한다 — additive 계약."""
    events = _collect(lambda s: s.transform_complete("a", "b"))
    d = events[-1].data or {}
    assert d.get("verdict") == "skipped", "전략을 모르면 pass 로 위장하지 않는다"
    assert d.get("source") == "a"


# ── 호출부가 실제로 판정을 넘기는가 (배선) ─────────────────────────

def test_plan_validator_passes_result_to_streamer():
    """게이트를 만들었다 ≠ 게이트가 돈다 — 호출 경로를 고정한다."""
    import inspect
    from ontology.orchestrator import plan_validator
    src = inspect.getsource(plan_validator)
    assert "validation_complete(result)" in src or \
           "validation_complete(result=result)" in src, \
           "plan_validator 가 ValidationResult 를 넘기지 않는다"


def test_data_transformer_labels_every_branch():
    """5개 전략이 각각 자기 라벨을 넘기는가."""
    import inspect
    from ontology.orchestrator import data_transformer
    src = inspect.getsource(data_transformer)
    for label in ("specific", "wildcard", "type_based", "llm_fallback", "passthrough"):
        assert f'strategy="{label}"' in src, f"{label} 분기에 전략 라벨 없음"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = f = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            p += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
            f += 1
    print(f"\npass={p} fail={f}")
    sys.exit(1 if f else 0)
