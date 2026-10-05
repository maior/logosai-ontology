"""
Ontology Builder Server — standalone FastAPI app (port 9274).

Run via ontology/scripts/start.sh (sets PYTHONPATH), or:
    PYTHONPATH="$LOGOS_ROOT:$LOGOS_ROOT/ontology" \
    uvicorn ontology.server.main:app --port 9274
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import router

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
    return {"status": "ok", "service": "ontology-builder", "port": 9274}
