"""선택 근거 영속화 (2026-07-31).

`_selection_history` 는 `deque(maxlen=200)` 이다. 즉 온톨로지가 **왜 그 에이전트를
골랐는지**는 최근 200건만 남고 그 이전은 사라진다. 학습 데이터 추출(`/decisions/export`)
과 사후 조사가 모두 이 창(window) 안에서만 가능하다는 뜻이다.

실측: 파일의 가장 오래된 항목이 2026-07-08, 가장 최근이 07-31 — 23일치가 아니라
**200건치**다. 트래픽이 늘면 하루도 안 남는다.

계약:
  · 선택 1건 = `ontology.selection` span 1개 (Pulse 로 fire-and-forget)
  · `selection_id` 는 클라이언트가 발급 — 나중에 도착하는 피드백이 이걸로 되짚는다
    (피드백은 선택보다 **나중에** 온다. 쓰기 순서 역전은 이미 겪은 함정이다)
  · 관측 실패가 선택을 막지 않는다
  · 기존 deque·파일 저장은 그대로 (역호환, 오프라인 분석 계속 가능)

P1~P5 = 신규 계약 (수정 전 RED).

직접 실행: python3 ontology/tests/test_selection_persistence.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ontology.core import selection_recorder as sr


def test_p1_span_body_shape():
    """P1 ★ Pulse SpanRecord 계약 + 되짚기 키."""
    body = sr.build_selection_span(
        selection_id="sel-abc",
        query="이태원 맛집 알려줘",
        selected_agent="restaurant_map_agent",
        method="kg_assisted",
        confidence=0.82,
        elapsed_ms=1234.5,
        reasoning="지도 데이터를 함께 반환하는 유일한 에이전트",
        graph_insights={"entities": ["이태원"], "past_patterns": [{"agent": "x"}],
                        "kg_confidence": 0.7},
    )
    assert body["name"] == "ontology.selection"
    assert body["span_id"] == "sel-abc", "피드백이 되짚을 키가 span id 여야 한다"
    assert body["agent_id"] == "restaurant_map_agent"
    m = body["metadata"]
    assert m["selection_id"] == "sel-abc"
    assert m["method"] == "kg_assisted"
    assert m["confidence"] == 0.82
    assert m["entities"] == ["이태원"]
    assert m["reasoning"].startswith("지도")


def test_p2_long_fields_are_bounded():
    """P2: 근거 문장은 길다 — 저장 전에 자른다 (span 하나가 DB 를 밀어내면 안 된다)."""
    body = sr.build_selection_span(
        selection_id="s", query="q" * 5000, selected_agent="a",
        method="m", confidence=0.0, elapsed_ms=0.0,
        reasoning="r" * 20000, graph_insights={})
    assert len(body["input_text"]) <= 2000
    assert len(body["metadata"]["reasoning"]) <= 4000


def test_p3_missing_insights_is_not_invented():
    """P3 ★ 그래프 근거가 없으면 빈 채로 둔다 — 지어내지 않는다."""
    m = sr.build_selection_span(
        selection_id="s", query="q", selected_agent="a", method="llm_only",
        confidence=0.0, elapsed_ms=1.0, reasoning="", graph_insights=None)["metadata"]
    assert m["entities"] == []
    assert m["past_patterns"] == []
    assert m["kg_confidence"] is None, "0.0 과 '모름'을 섞으면 안 된다"


def test_p4_feedback_references_selection():
    """P4 ★ 피드백은 선택보다 나중에 온다 — selection_id 로 되짚는다."""
    ev = sr.build_feedback_event(selection_id="sel-abc", selected_agent="a",
                                 success=True, ema_success_rate=0.75)
    assert ev["event_type"] == "ontology.selection.feedback"
    assert ev["payload"]["selection_id"] == "sel-abc"
    assert ev["payload"]["success"] is True
    assert ev["event_id"], "재전송 멱등 키가 없으면 이중 계상된다"


def test_p5_emit_never_raises():
    """P5: Pulse 가 죽어 있어도 선택은 계속된다."""
    sr.emit_selection(selection_id="s", query="q", selected_agent="a",
                      method="m", confidence=0.0, elapsed_ms=0.0,
                      reasoning="", graph_insights=None)
    sr.emit_feedback(selection_id="s", selected_agent="a", success=False,
                     ema_success_rate=None)


def test_p6_selection_id_is_bare_uuid():
    """P6 ★ span_id 는 **맨 UUID** 여야 한다 (라이브에서 한 번 잃었다).

    Pulse `trace_spans.id` 는 UUID 컬럼이다. `sel-...` 접두사를 붙였더니
    asyncpg 가 거부했는데, ingest 가 그 실패를 **200 OK** 로 되돌려주고
    발신은 fire-and-forget 이라 전량 유실이 조용히 지나갔다.
    (`feedback_pulse_silent_metric_loss` 와 같은 함정.)
    """
    import uuid as _uuid
    sid = sr.new_selection_id()
    _uuid.UUID(sid)                      # 파싱 안 되면 여기서 터진다
    assert not sid.startswith("sel-"), "접두사를 붙이면 UUID 컬럼이 거부한다"
    assert sr.build_selection_span(
        selection_id=sid, query="q", selected_agent="a", method="m",
        confidence=0.0, elapsed_ms=0.0, reasoning="",
        graph_insights=None)["span_id"] == sid


def test_p7_joins_ambient_trace_instead_of_minting_its_own():
    """P7 ★ 선택 근거가 그 쿼리의 여정에 붙어야 한다.

    실측(2026-08-08): `trace_id` 를 무조건 새로 발급해 30일간 59건 전부
    **고아 trace** 로 떨어졌다. 5.9초짜리 선택이 26초 쿼리의 22% 인데
    여정 화면 어디에도 없었다.
    """
    body = sr.build_selection_span(
        selection_id="s", query="q", selected_agent="a", method="m",
        confidence=0.0, elapsed_ms=0.0, reasoning="", graph_insights=None,
        trace_id="11111111-1111-1111-1111-111111111111",
        parent_id="22222222-2222-2222-2222-222222222222")
    assert body["trace_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["parent_id"] == "22222222-2222-2222-2222-222222222222"


def test_p8_standalone_use_still_gets_a_trace():
    """주변 trace 가 없으면 스스로 발급한다 — 기존 단독 사용 보존."""
    body = sr.build_selection_span(
        selection_id="s", query="q", selected_agent="a", method="m",
        confidence=0.0, elapsed_ms=0.0, reasoning="", graph_insights=None)
    assert body["trace_id"], "trace 가 비면 Pulse 가 거부한다"
    assert body["parent_id"] == "", "부모를 지어내지 않는다"


def test_p9_declares_route_stage():
    """이름 휴리스틱에 걸리지 않아 '미분류'로 위장되던 span."""
    m = sr.build_selection_span(
        selection_id="s", query="q", selected_agent="a", method="m",
        confidence=0.0, elapsed_ms=0.0, reasoning="", graph_insights=None)["metadata"]
    assert m.get("stage") == "route", m


def test_p10_emit_resolves_ambient_trace():
    """emit 이 실제로 주변 trace 를 읽는가 (배선). 전송은 하지 않는다."""
    captured = {}
    original_post, original_probe, original_resolver = (
        sr._post, sr._resolver_probed, sr._trace_resolver)
    try:
        sr._post = lambda path, body: captured.update(path=path, body=body)
        sr._resolver_probed = True
        sr._trace_resolver = (lambda: "trace-x", lambda: "span-y")
        sr.emit_selection(selection_id="s", query="q", selected_agent="a",
                          method="m", confidence=0.0, elapsed_ms=0.0,
                          reasoning="", graph_insights=None)
        assert captured["body"]["trace_id"] == "trace-x", captured["body"]
        assert captured["body"]["parent_id"] == "span-y"
    finally:
        sr._post, sr._resolver_probed, sr._trace_resolver = (
            original_post, original_probe, original_resolver)


def test_p11_resolver_failure_does_not_break_emit():
    """trace 조회가 터져도 선택은 계속된다 (관측이 판단을 막지 않는다)."""
    captured = {}
    original_post, original_probe, original_resolver = (
        sr._post, sr._resolver_probed, sr._trace_resolver)
    try:
        sr._post = lambda path, body: captured.update(body=body)
        sr._resolver_probed = True

        def _boom():
            raise RuntimeError("no context")

        sr._trace_resolver = (_boom, _boom)
        sr.emit_selection(selection_id="s", query="q", selected_agent="a",
                          method="m", confidence=0.0, elapsed_ms=0.0,
                          reasoning="", graph_insights=None)
        assert captured["body"]["trace_id"], "폴백 trace 가 있어야 한다"
    finally:
        sr._post, sr._resolver_probed, sr._trace_resolver = (
            original_post, original_probe, original_resolver)


def test_p12_no_module_level_logosai_import():
    """이 파일의 원칙: logosai 를 모듈 레벨로 끌어오지 않는다 (import 비용)."""
    import inspect
    src = inspect.getsource(sr)
    head = src.split("def ")[0]
    assert "import logosai" not in head and "from logosai" not in head, \
        "모듈 상단에서 logosai 를 import 하고 있다"


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, f in tests:
        try:
            f()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  💥 {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run())
