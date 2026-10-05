"""
Ontology Builder Server — standalone FastAPI app (port 9274).

Run via ontology/scripts/start.sh (sets PYTHONPATH), or:
    PYTHONPATH="$LOGOS_ROOT:$LOGOS_ROOT/ontology" \
    uvicorn ontology.server.main:app --port 9274
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core import health_signals
from .pulse_events import on_component_change
from .router import router

# 구성 요소(임베더·확산)가 조용히 꺼지면 Pulse Events 에 알린다 (2026-10-05)
health_signals.add_listener(on_component_change)

app = FastAPI(
    title="LogosAI Ontology Builder",
    description="데이터 + 온톨로지 프레임워크 + LLM — 임의의 데이터를 온톨로지로 구성",
    version="0.1.0",
)

# frontend (Next.js dev server on 9275)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9275", "http://127.0.0.1:9275"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1/ontology", tags=["Ontology"])


@app.get("/health")
async def health():
    """프로세스가 응답한다 ≠ 기능이 정상이다. 꺼진 구성 요소가 있으면 status=degraded.

    감독(keepalive)은 포트 응답만 보므로 이 값으로 재기동하지 않는다 — 재기동으로 고쳐지는
    종류가 아니다(2026-10-05 scipy 사고는 재기동해도 같았다). 사람과 대시보드가 본다.
    아직 한 번도 시도되지 않은 구성 요소는 나오지 않는다(지연 로딩 — 모름 ≠ 정상)."""
    components = health_signals.snapshot()
    down = health_signals.degraded()
    return {"status": "degraded" if down else "ok", "service": "ontology-builder",
            "port": 9274, "degraded": down, "components": components}
