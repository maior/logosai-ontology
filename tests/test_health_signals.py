"""구성 요소 상태 신호 — 기능이 조용히 꺼지면 소리를 낸다 (2026-10-05).

사고: macOS 27 업그레이드 후 scipy 네이티브 라이브러리가 거부되자 임베더(bge-m3)를 못
불러왔고, 의미 검색이 **전 네임스페이스에서 결과 0건**이었다. degrade 설계라 HTTP 200 에
빈 결과, 남은 것은 WARNING 한 줄 — 이틀 가까이 아무도 몰랐다. 확산 채널(PageRank)도 같이
죽어 있었다.

계약:
  · 상태는 **바뀔 때만** 알린다 (정상 트래픽마다 이벤트를 쏘지 않는다 — 수집 규약)
  · 첫 보고가 정상이면 조용하다 (정상 기동은 소식이 아니다), 첫 보고가 실패면 알린다
  · 리스너가 실패해도 보고하는 쪽(검색)은 영향받지 않는다
  · Pulse 전송은 테스트 중 기본 차단 (테스트가 관측 DB 를 오염시킨 전례)
"""
import sys
import types

import pytest

from ontology.core import health_signals as hs


@pytest.fixture(autouse=True)
def _fresh_signals():
    hs.reset_for_tests()
    yield
    hs.reset_for_tests()


def _capture():
    seen = []
    hs.add_listener(lambda comp, ok, detail, prev: seen.append((comp, ok, prev)))
    return seen


# ── 전이 규칙 ──────────────────────────────────────────────────────────

def test_healthy_start_is_silent_and_recorded():
    seen = _capture()
    hs.report("embedder:m", True, "loaded")
    assert seen == []
    assert hs.snapshot()["embedder:m"]["ok"] is True


def test_first_failure_is_announced():
    seen = _capture()
    hs.report("embedder:m", False, "dlopen failed")
    assert seen == [("embedder:m", False, None)]


def test_repeated_state_is_not_reannounced():
    seen = _capture()
    for _ in range(3):
        hs.report("graph_propagation", False, "scipy")
    hs.report("graph_propagation", True)
    hs.report("graph_propagation", True)
    assert seen == [("graph_propagation", False, None), ("graph_propagation", True, False)]


def test_listener_failure_does_not_reach_the_reporter():
    hs.add_listener(lambda *a: 1 / 0)
    seen = _capture()
    hs.report("embedder:m", False, "x")          # 예외가 올라오면 안 된다
    assert seen == [("embedder:m", False, None)]  # 대조군 — 다른 리스너는 그대로 받는다


def test_degraded_reflects_any_failed_component():
    hs.report("embedder:m", True)
    assert hs.degraded() == []
    hs.report("graph_propagation", False, "scipy")
    assert hs.degraded() == ["graph_propagation"]


# ── 배선: 실제 실패 지점이 보고한다 ──────────────────────────────────

def test_embedder_load_failure_is_reported(monkeypatch):
    from ontology.core import semantic_index as si
    broken = types.ModuleType("sentence_transformers")

    class Boom:
        def __init__(self, *a, **k):
            raise ImportError("dlopen(_spropack.so): zero-fill section")
    broken.SentenceTransformer = Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", broken)

    assert si._build_embed_fn("test/model") is None
    state = hs.snapshot()["embedder:test/model"]
    assert state["ok"] is False and "zero-fill" in state["detail"]


def test_embedder_load_success_is_reported(monkeypatch):
    from ontology.core import semantic_index as si
    ok = types.ModuleType("sentence_transformers")

    class Fine:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, show_progress_bar=False):
            return [[0.0, 1.0] for _ in texts]
    ok.SentenceTransformer = Fine
    monkeypatch.setitem(sys.modules, "sentence_transformers", ok)

    assert si._build_embed_fn("test/model2") is not None
    assert hs.snapshot()["embedder:test/model2"]["ok"] is True


def test_propagation_failure_is_reported(monkeypatch):
    import networkx as nx
    from ontology.core import graph_propagation as gp
    g = nx.Graph()
    g.add_edge("n:a", "c:1")
    g.add_edge("n:b", "c:1")

    def boom(*a, **k):
        raise ImportError("scipy.sparse.linalg unavailable")
    monkeypatch.setattr(nx, "pagerank", boom)        # 함수 안에서 import 한다 — 모듈 자체를 바꾼다
    assert gp.propagate(g, {"n:a": 1.0}) == {}
    assert hs.snapshot()["graph_propagation"]["ok"] is False

    monkeypatch.undo()
    assert gp.propagate(g, {"n:a": 1.0})                     # 복구되면
    assert hs.snapshot()["graph_propagation"]["ok"] is True  # 상태도 돌아온다


# ── 서버: Pulse 이벤트와 /health ──────────────────────────────────────

def test_event_shape_and_severity():
    from ontology.server import pulse_events as pe
    down = pe.build_event("embedder:BAAI/bge-m3", False, "dlopen failed", None)
    up = pe.build_event("embedder:BAAI/bge-m3", True, "loaded", False)
    assert down["event_type"] == "ontology.component.unavailable" and down["severity"] == "critical"
    assert up["event_type"] == "ontology.component.recovered" and up["severity"] == "info"
    assert down["source"] == "ontology" and down["payload"]["component"] == "embedder:BAAI/bge-m3"
    assert down["event_id"] and down["event_id"] != up["event_id"]


def test_sending_is_blocked_under_pytest(monkeypatch):
    from ontology.server import pulse_events as pe
    monkeypatch.delenv("LOGOSAI_PULSE_ALLOW_IN_TESTS", raising=False)
    assert pe.sending_blocked() is True
    monkeypatch.setenv("LOGOSAI_PULSE_ALLOW_IN_TESTS", "1")
    assert pe.sending_blocked() is False                     # 대조군 — 명시하면 열린다


def test_non_200_is_counted_not_raised(monkeypatch):
    from ontology.server import pulse_events as pe
    monkeypatch.setenv("LOGOSAI_PULSE_ALLOW_IN_TESTS", "1")
    pe.reset_stats()
    pe.send(pe.build_event("x", False, "d", None), post=lambda url, body: 422)
    pe.send(pe.build_event("x", True, "d", False), post=lambda url, body: 200)
    assert pe.stats() == {"sent": 1, "failed": 1}


def test_health_reports_degraded_components():
    from fastapi.testclient import TestClient
    from ontology.server.main import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    hs.report("embedder:BAAI/bge-m3", False, "dlopen failed")
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["embedder:BAAI/bge-m3"]["ok"] is False
