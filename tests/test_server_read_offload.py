"""읽기 엔드포인트는 이벤트 루프를 잡지 않는다 — 전용 작업 스레드 1개에서 돈다 (2026-10-05).

실측: `async def` 엔드포인트가 무거운 동기 서비스 함수를 직접 불러, 검색 1회(2.89초)
동안 다른 요청도 2.87초 묶였다(평소 0.001초). 첫 검색은 임베딩 모델(bge-m3 2.2GB)까지
로딩한다 — 9/28 의 2분 무응답이 이 형태. 감독의 헬스체크도 같이 묶여 거짓 '사망'을 낸다.

설계:
  · 읽기(GET + 검색 POST)만 작업 스레드로 — 스레드는 1개라 읽기끼리는 지금처럼 한 번에
    하나 (모델 이중 로딩·읽기 간 경쟁 없음)
  · 쓰기(POST·PUT·DELETE)는 그대로 루프 — 서비스에 잠금이 없어 쓰기 직렬화는 루프의
    암묵적 직렬화에 기댄다. 그걸 깨지 않는다.
"""
import ast
import asyncio
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from ontology.server import router as router_module

ROUTER = Path(router_module.__file__)

#: 읽기 중심 POST — GET 이 아니어도 작업 스레드로 간다. 공유 상태를 바꾸지 않고 자기
#: 전용 저장소(평가 이력·실험 기록)에만 쓴다. 평가는 루프에서 96.6s 를 잡았다(실측).
READ_POSTS = {"search_graph", "evaluate_golden_set", "run_retrieval_experiments"}


def _endpoints():
    tree = ast.parse(ROUTER.read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        meth = [d.func.attr for d in fn.decorator_list
                if isinstance(d, ast.Call) and getattr(d.func, "attr", "") in
                ("get", "post", "put", "delete", "patch")]
        if meth:
            yield meth[0], fn


def _bare_service_calls(fn):
    """await 없이 부르는 service.<m>(...) — 루프를 잡는 동기 호출."""
    awaited = {id(n.value) for n in ast.walk(fn) if isinstance(n, ast.Await)}
    return sorted({n.func.attr for n in ast.walk(fn)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == "service"
                   and id(n) not in awaited})


def test_every_read_endpoint_offloads_its_service_calls():
    """배선 계약 — 읽기 엔드포인트에 루프를 잡는 동기 service 호출이 없다."""
    offenders = {fn.name: _bare_service_calls(fn) for meth, fn in _endpoints()
                 if (meth == "get" or fn.name in READ_POSTS) and _bare_service_calls(fn)}
    assert offenders == {}, f"루프에서 동기 호출하는 읽기 엔드포인트: {offenders}"


def test_write_endpoints_stay_on_the_loop():
    """대조군 — 쓰기 엔드포인트는 바꾸지 않았다 (동기 호출이 그대로 남아 있다)."""
    writes = [fn.name for meth, fn in _endpoints()
              if meth in ("post", "put", "delete", "patch") and fn.name not in READ_POSTS
              and _bare_service_calls(fn)]
    assert len(writes) >= 30


# ── 실제 동작: 무거운 읽기가 루프를 잡지 않는다 ─────────────────────────

class SlowService:
    """읽기 하나가 1초 동안 동기로 일한다 (임베딩 계산·모델 로딩 흉내)."""

    def __init__(self):
        self.spans = []           # (스레드, 시작, 끝)

    def get_namespace_stats(self, namespace):
        start = time.monotonic()
        time.sleep(1.0)
        self.spans.append((threading.current_thread().name, start, time.monotonic()))
        return None if namespace == "ghost" else {"namespace": namespace}


@pytest.fixture
def app_and_service():
    service = SlowService()
    app = FastAPI()
    app.include_router(router_module.router, prefix="/api/v1/ontology")
    app.dependency_overrides[router_module.get_ontology_service] = lambda: service

    @app.get("/health")
    async def health():
        return {"status": "ok"}
    return app, service


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_health_answers_while_a_heavy_read_runs(app_and_service):
    app, _ = app_and_service
    async with await _client(app) as c:
        # 시계는 무거운 읽기를 보내기 **전에** 잰다 — 루프가 막히면 아래 sleep 자체가
        # 멈춰서, 그 뒤부터 재면 막힘이 측정에서 빠진다 (첫 판이 그렇게 헛통과했다)
        t0 = time.monotonic()
        slow = asyncio.create_task(c.get("/api/v1/ontology/graphs/ns/stats"))
        await asyncio.sleep(0.1)
        r = await c.get("/health")
        health_done = time.monotonic() - t0
        assert (await slow).status_code == 200
    assert r.status_code == 200
    assert health_done < 0.6, f"헬스체크가 무거운 읽기(1s)에 묶였다 — {health_done:.2f}s 에 응답"


async def test_reads_run_one_at_a_time_off_the_loop(app_and_service):
    app, service = app_and_service
    async with await _client(app) as c:
        await asyncio.gather(c.get("/api/v1/ontology/graphs/a/stats"),
                             c.get("/api/v1/ontology/graphs/b/stats"))
    (t1, s1, e1), (t2, s2, e2) = sorted(service.spans, key=lambda x: x[1])
    assert threading.main_thread().name not in (t1, t2)        # 루프 스레드가 아니다
    assert s2 >= e1 - 0.01, "읽기 둘이 겹쳤다 — 작업 스레드는 1개여야 한다"


async def test_existing_responses_are_preserved(app_and_service):
    """대조군 — 404 같은 기존 응답은 그대로다."""
    app, _ = app_and_service
    async with await _client(app) as c:
        r = await c.get("/api/v1/ontology/graphs/ghost/stats")
    assert r.status_code == 404
