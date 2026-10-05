"""
Ontology Builder API routes.

POST /datasets                    파일 업로드 → dataset_id
GET  /datasets                    데이터셋 목록
POST /build                       빌드 잡 시작 (BackgroundTasks)
GET  /jobs/{job_id}               잡 상태/진행/리포트
GET  /graphs/{namespace}          그래프 통계 + 샘플 노드
GET  /graphs/{namespace}/data     시각화용 nodes+links
POST /graphs/{namespace}/search   의미 검색
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .service import (COVERAGE_MIN_LEN, PROTECTED_NAMESPACES,
                      OntologyBuilderService)
from ..builder.pipeline import PROVIDERS

router = APIRouter()


# ── 읽기 전용 작업 스레드 (2026-10-05) ──────────────────────────────────
# 무거운 동기 서비스 함수(임베딩 계산·첫 모델 로딩·그래프 순회)를 async 엔드포인트에서
# 직접 부르면 그동안 이벤트 루프 전체가 멈춘다 — 실측: 검색 1회(2.89s) 동안 다른 요청도
# 2.87s 묶였다. 읽기(GET + 검색·골든셋 평가·검색 실험 — 자기 전용 기록에만 쓴다)만 여기로 보낸다. 스레드는 1개라 읽기끼리는 지금처럼
# 한 번에 하나다(모델 이중 로딩·읽기 간 경쟁 없음). 쓰기는 루프에 그대로 둔다 — 서비스에
# 잠금이 없어 쓰기 직렬화는 루프의 암묵적 직렬화에 기댄다.
# 계약: tests/test_server_read_offload.py
_READ_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ontology-read")


async def _read(fn, *args, **kwargs):
    """동기 읽기 함수를 작업 스레드에서 실행한다 (예외는 그대로 전파)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_READ_EXECUTOR, functools.partial(fn, *args, **kwargs))

_service: Optional[OntologyBuilderService] = None


def get_ontology_service() -> OntologyBuilderService:
    """Module-level singleton; tests override via dependency_overrides."""
    global _service
    if _service is None:
        _service = OntologyBuilderService()
    return _service


class BuildRequest(BaseModel):
    dataset_id: str
    namespace: str
    schema_mode: str = "document"  # 프리셋 이름(document/generic/…) 또는 custom
    custom_schema: Optional[Dict[str, Any]] = None
    chunk_size: int = Field(default=800, ge=100, le=8000)
    overlap: int = Field(default=120, ge=0, le=2000)
    segment_mode: str = Field(default="auto", pattern="^(auto|heading|window)$")
    llm_model: Optional[str] = None      # 미지정 시 프로바이더별 기본 모델
    llm_provider: str = "google"         # google | openai | anthropic | openai_compatible
    llm_base_url: Optional[str] = None   # openai_compatible(오픈소스 서버)용
    rebuild: bool = False                # true면 네임스페이스 그래프 초기화 후 빌드
    save: bool = True


class DatasetRequest(BaseModel):
    formats: List[str] = Field(default=["triples", "qa", "surface"])
    node_types: Optional[List[str]] = None
    predicates: Optional[List[str]] = None
    limit: int = Field(default=5000, ge=1, le=100000)
    # 관계 qa 행에 근거 원문을 붙인다 (축 2). 기본 False — 기존 행 모양 보존.
    include_evidence: bool = False
    # ⑥ 큐레이션 옵션 (dedup/min_input_chars/max_per_source/exclude_trust/
    # llm_quality/quality_threshold). 옵트인 — 없으면 기존 응답 모양 보존.
    curate: Optional[Dict[str, Any]] = None
    # 코호트 추출 — 구조화 쿼리(/query 와 동일 키)로 추출 대상을 제한. 옵트인.
    cohort: Optional[Dict[str, Any]] = None


class RecordsRequest(BaseModel):
    records: List[Dict[str, Any]]
    mapping: Dict[str, Any]
    source: str = ""
    save: bool = True
    rebuild: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    node_types: Optional[List[str]] = None


@router.post("/datasets")
async def upload_dataset(
    files: List[UploadFile] = File(...),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    contents = [(f.filename or "unnamed", await f.read()) for f in files]
    return service.save_dataset(contents)


@router.get("/datasets")
async def list_datasets(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return {"datasets": (await _read(service.list_datasets))}


class IngestRequest(BaseModel):
    namespace: str
    # analyze 응답의 plan 초안을 사용자가 고쳐 그대로 제출한다.
    # 각 항목: {filename, route, records_path?, mapping?, hierarchy?}
    plan: List[Dict[str, Any]]
    save: bool = True
    schema_mode: str = "auto"          # 텍스트 경로(articled/prose)용
    custom_schema: Optional[Dict[str, Any]] = None
    llm_model: Optional[str] = None
    llm_provider: str = "google"
    llm_base_url: Optional[str] = None
    extraction_mode: str = "exhaustive"   # exhaustive | topic(aicoach식 회수)
    topics: Optional[List[str]] = None    # topic 모드에서 사용자 지정(비우면 LLM 유도)


@router.post("/datasets/{dataset_id}/ingest")
async def ingest_dataset(
    dataset_id: str,
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """확인 게이트 실행 — 업로드 → analyze → 확인 → **ingest**.

    승인된 plan 을 종별로 라우팅한다: records(결정적, +계층 is_a) /
    articled·prose(LLM 추출) / seed_ontology(멱등 upsert) / skip.
    """
    if request.namespace in PROTECTED_NAMESPACES:
        raise HTTPException(
            status_code=400,
            detail=f"namespace '{request.namespace}' is protected")
    if service.dataset_path(dataset_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"dataset not found: {dataset_id}")
    if not request.plan:
        raise HTTPException(status_code=400, detail="plan must not be empty")

    job_id = service.create_job(request.namespace)
    background_tasks.add_task(
        service.run_ingest, job_id, dataset_id, request.namespace,
        request.plan, request.save, request.schema_mode,
        request.custom_schema, request.llm_model, request.llm_provider,
        request.llm_base_url, request.extraction_mode, request.topics)
    return {"job_id": job_id, "status": "queued"}


@router.post("/datasets/{dataset_id}/analyze")
async def analyze_dataset(
    dataset_id: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """감식 — 빌드 전 확인 게이트의 근거 (업로드 → **analyze** → 확인 → build).

    파일별 종(records/articled/prose/seed_ontology) + 카디널리티 기반
    노드/속성 역할 + 매핑 제안(gemini-3.5-flash) + LLM 비용 견적을 돌려준다.
    정형 레코드가 LLM 추출로 흘러 수천 콜을 낭비하기 전에 여기서 보인다.
    """
    try:
        return await service.analyze_dataset(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/build")
async def start_build(
    request: BuildRequest,
    background_tasks: BackgroundTasks,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    if request.namespace in PROTECTED_NAMESPACES:
        raise HTTPException(
            status_code=400,
            detail=f"namespace '{request.namespace}' is protected "
                   "(agent-routing graph) — choose a dedicated namespace")
    if service.dataset_path(request.dataset_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"dataset '{request.dataset_id}' not found")
    if request.schema_mode == "custom" and not request.custom_schema:
        raise HTTPException(status_code=400,
                            detail="schema_mode='custom' requires custom_schema")
    if request.schema_mode != "custom":
        try:
            service._resolve_schema(request.schema_mode, None)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if request.llm_provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown llm_provider (available: {', '.join(PROVIDERS)})")
    if request.llm_provider == "openai_compatible" and not request.llm_base_url:
        raise HTTPException(status_code=400,
                            detail="openai_compatible requires llm_base_url")

    job_id = service.create_job(request.namespace)
    background_tasks.add_task(
        service.run_build,
        job_id=job_id,
        dataset_id=request.dataset_id,
        namespace=request.namespace,
        schema_mode=request.schema_mode,
        custom_schema=request.custom_schema,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        save=request.save,
        segment_mode=request.segment_mode,
        llm_model=request.llm_model,
        llm_provider=request.llm_provider,
        llm_base_url=request.llm_base_url,
        rebuild=request.rebuild,
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/schemas")
async def list_schemas(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return {"presets": (await _read(service.list_schema_presets))}


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """작업 이력 — 최근 인제스트/재색인 잡을 상태·스테이지와 함께(최신 먼저).

    인메모리 레지스트리라 서버 재시작 시 비는 것이 정상이다. 상세(파일별 결과
    등)는 GET /jobs/{job_id} 로."""
    return (await _read(service.list_jobs, limit=limit))


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    job = (await _read(service.get_job, job_id))
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return job


@router.get("/graphs/{namespace}")
async def get_graph(
    namespace: str,
    limit: int = Query(default=10, ge=1, le=200),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return (await _read(service.get_graph_summary, namespace, limit=limit))


@router.get("/graphs/{namespace}/data")
async def get_graph_data(
    namespace: str,
    limit: int = Query(default=300, ge=1, le=2000),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return (await _read(service.get_graph_data, namespace, limit=limit))


@router.get("/namespaces")
async def list_namespaces(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return {"namespaces": (await _read(service.list_namespaces))}


@router.get("/graphs/{namespace}/node")
async def get_node_detail(
    namespace: str,
    id: str = Query(..., min_length=1),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    detail = (await _read(service.get_node_detail, namespace, id))
    if detail is None:
        raise HTTPException(status_code=404, detail=f"node '{id}' not found")
    return detail


@router.get("/graphs/{namespace}/rollup")
async def hierarchy_rollup(
    namespace: str,
    class_id: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=500),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    result = (await _read(service.get_hierarchy_rollup, namespace, class_id, limit=limit))
    if result is None:
        raise HTTPException(status_code=404, detail=f"class '{class_id}' not found")
    return result


@router.get("/graphs/{namespace}/map")
async def get_map(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return (await _read(service.get_map_data, namespace))


@router.get("/graphs/{namespace}/export")
async def export_graph(
    namespace: str,
    format: str = Query(default="turtle", pattern="^(turtle|json)$"),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    if format == "turtle":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            (await _read(service.export_graph, namespace, "turtle")),
            media_type="text/turtle; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="ontology_{namespace}.ttl"'})
    return (await _read(service.export_graph, namespace, "json"))


@router.post("/graphs/{namespace}/records")
async def ingest_records(
    namespace: str,
    request: RecordsRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    if namespace in PROTECTED_NAMESPACES:
        raise HTTPException(status_code=400,
                            detail=f"namespace '{namespace}' is protected")
    if not request.mapping.get("node_type") or not request.mapping.get("name_field"):
        raise HTTPException(status_code=400,
                            detail="mapping requires node_type and name_field")
    from dataclasses import asdict
    from ontology.builder import OntologyBuilder
    if request.rebuild:
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        get_knowledge_graph_engine(namespace).clear()
    builder = OntologyBuilder(schema=None, namespace=namespace,
                              auto_save=request.save)
    report = await builder.build_from_records(
        request.records, request.mapping, source=request.source)
    return asdict(report)


@router.post("/graphs/{namespace}/dataset")
async def build_dataset(
    namespace: str,
    request: DatasetRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    try:
        if request.curate is not None:
            # 큐레이션은 옵트인 — curate 없이는 기존 경로/응답 모양 그대로
            return await service.build_curated_dataset(
                namespace, formats=request.formats,
                node_types=request.node_types, predicates=request.predicates,
                limit=request.limit, include_evidence=request.include_evidence,
                curate=request.curate, cohort=request.cohort)
        return service.build_training_dataset(
            namespace, formats=request.formats,
            node_types=request.node_types, predicates=request.predicates,
            limit=request.limit, include_evidence=request.include_evidence,
            cohort=request.cohort)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/graphs/{namespace}/dataset.jsonl")
async def download_dataset(
    namespace: str,
    formats: str = Query(default="triples,qa,surface"),
    node_types: Optional[str] = Query(default=None),
    predicates: Optional[str] = Query(default=None),
    include_evidence: bool = Query(default=False),
    # 코호트 추출 — /query 와 동일 키를 쿼리스트링으로 (다운로드 URL 이 코호트를 담는다)
    c_type: Optional[str] = Query(default=None),
    c_trust: Optional[str] = Query(default=None),
    c_prop_key: Optional[str] = Query(default=None),
    c_prop_value: Optional[str] = Query(default=None),
    c_prop_op: str = Query(default="eq"),
    c_rel_predicate: Optional[str] = Query(default=None),
    c_rel_target: Optional[str] = Query(default=None),
    c_rel_target_type: Optional[str] = Query(default=None),
    c_rel_direction: str = Query(default="out"),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    import json as _json
    from fastapi.responses import PlainTextResponse
    cohort = None
    if any([c_type, c_trust, c_prop_key, c_rel_predicate, c_rel_target, c_rel_target_type]):
        cohort = {"node_type": c_type, "trust": c_trust, "prop_key": c_prop_key,
                  "prop_value": c_prop_value, "prop_op": c_prop_op,
                  "rel_predicate": c_rel_predicate, "rel_target": c_rel_target,
                  "rel_target_type": c_rel_target_type, "rel_direction": c_rel_direction}
    try:
        result = (await _read(service.build_training_dataset, 
            namespace,
            formats=[f.strip() for f in formats.split(",") if f.strip()],
            node_types=[t.strip() for t in node_types.split(",")] if node_types else None,
            predicates=[p.strip() for p in predicates.split(",")] if predicates else None,
            include_evidence=include_evidence, cohort=cohort))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    body = "\n".join(_json.dumps(r, ensure_ascii=False) for r in result["rows"])
    return PlainTextResponse(
        body, media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="dataset_{namespace}.jsonl"'})


@router.post("/graphs/{namespace}/search")
async def search_graph(
    namespace: str,
    request: SearchRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")
    results = (await _read(service.semantic_search, 
        namespace, request.query, top_k=request.top_k,
        node_types=request.node_types))
    return {"namespace": namespace, "query": request.query, "results": results}


# ─── Chunks (원문 · 근거) ────────────────────────────────────────────
# 그래프가 "무엇을 아는가"라면 이 셋은 "어디서 알았는가"를 답한다 (축 2).

@router.get("/graphs/{namespace}/chunks/{chunk_id}")
async def get_chunk(
    namespace: str,
    chunk_id: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    chunk = (await _read(service.get_chunk, namespace, chunk_id))
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"chunk not found: {chunk_id}")
    return chunk


@router.get("/graphs/{namespace}/node/chunks")
async def get_node_chunks(
    namespace: str,
    node_id: str = Query(...),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """이 노드가 추출된 원문 청크들 — 상세 패널의 '근거' 탭."""
    return (await _read(service.get_node_chunks, namespace, node_id))


@router.get("/graphs/{namespace}/chunks")
async def search_chunks(
    namespace: str,
    query: str = Query(...),
    top_k: int = Query(default=5, ge=1, le=100),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """원문 구절 검색 — 의미(임베딩) 또는 하이브리드(ES)."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")
    return (await _read(service.search_chunks, namespace, query, top_k=top_k))


# ─── Review (검수 루프) ──────────────────────────────────────────────
# LLM 추출은 틀린다. 거절은 삭제가 아니라 묘비(tombstone)다 — 재인제스트에서
# 같은 오추출이 부활하는 것을 막는 영속 기록 (KorAct confirm/reject 패턴).


class ReviewActionRequest(BaseModel):
    node_id: str
    actor: str = ""
    reason: str = ""  # reject 전용 — confirm 은 무시한다


@router.get("/graphs/{namespace}/review/queues")
async def get_review_queues(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """검수 큐 집계 — "오늘 뭘 검수해야 하나". 기존 경로 위임 + 셈 (LLM 0콜)."""
    result = (await _read(service.review_queues, namespace))
    if result.get("error") == "namespace_not_found":
        raise HTTPException(status_code=404, detail=f"namespace not found: {namespace}")
    return result


@router.get("/graphs/{namespace}/review")
async def get_review_queue(
    namespace: str,
    trust: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """검수 대기 노드 — LLM 추출로 들어와 아직 판정 없는 것 + 근거 개수."""
    return (await _read(service.get_review_queue, namespace, trust=trust, limit=limit))


@router.post("/graphs/{namespace}/review/confirm")
async def confirm_node(
    namespace: str,
    request: ReviewActionRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    _guard_protected_write(namespace)
    result = service.confirm_node(namespace, request.node_id,
                                  actor=request.actor)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"node '{request.node_id}' not found")
    return result


@router.post("/graphs/{namespace}/review/reject")
async def reject_node(
    namespace: str,
    request: ReviewActionRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """거절 = 묘비 + 그래프에서 노드·간선 제거. 재인제스트를 살아남는다.

    파괴적 경로이므로 보호 네임스페이스를 막는다 — merge·rename·엣지 삭제는
    막으면서 노드 통삭제만 뚫려 있던 것은 정책 누락이었다 (2026-08-21).
    """
    _guard_protected_write(namespace)
    result = service.reject_node(namespace, request.node_id,
                                 actor=request.actor, reason=request.reason)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"node '{request.node_id}' not found")
    return result


class PrecheckRequest(BaseModel):
    trust: Optional[str] = None            # 낮은 신뢰 출처부터 검수하는 워크플로우용
    limit: int = Field(default=20, ge=1, le=50)  # 노드당 LLM 1콜 — 상한 필수


@router.post("/graphs/{namespace}/review/precheck")
async def precheck_review_queue(
    namespace: str,
    request: PrecheckRequest = PrecheckRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """근거대조 에이전트 사전판정 (aicoach 협업 패턴 — 추천까지만).

    큐의 각 노드를 원문 청크와 대조해 confirm/reject/unsure 추천 + 근거
    인용을 단다. 인용이 원문에 없으면 unsure 로 강등(지어낸 근거 무효),
    판정 권한은 인간에게 남는다. 동기 실행 — limit 상한(50)이 지연을 막는다.
    """
    return await service.precheck_review_queue(
        namespace, trust=request.trust, limit=request.limit)


@router.get("/graphs/{namespace}/review/consistency")
async def lint_consistency(
    namespace: str,
    schema_mode: Optional[str] = Query(default=None),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """일관성 감시 — 그래프 내부 모순 스캔 (LLM 0콜, 결정적, 무저장).

    이름-타입 충돌 · 별칭 충돌 · (schema_mode 지정 시) domain/range 위반.
    GET 인 이유: findings 는 재계산 가능한 파생물이라 서버 상태를 바꾸지
    않는다 — precheck/coverage(POST, LLM 비용 + 기록)와 대비되는 지점.
    """
    try:
        return (await _read(service.lint_consistency, namespace, schema_mode=schema_mode))
    except ValueError as e:
        # custom 은 body 가 필요해 GET 으로 못 받는다 → 프리셋 이름만 허용
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/graphs/{namespace}/health")
async def graph_health(
    namespace: str,
    sample: int = Query(default=20, ge=1, le=200),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """근거 사슬 건강 진단 — 추출 커버리지 · 고아 노드 · 중복 노드 후보.

    GET 인 이유는 consistency 와 같다: 그래프+청크에서 재계산되는 파생물이라
    서버 상태를 바꾸지 않는다. LLM 0콜이므로 상한이 필요 없고, 목록만 sample 로
    자른다 — **개수는 자르지 않는다**(표본을 전부로 오해하면 '다 봤다'가 된다).
    """
    return (await _read(service.graph_health, namespace, sample=sample))


class CoverageRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=30)  # 청크당 LLM 1콜 — 상한 필수
    # 추출이 아무 노드도 내지 못한 청크만 검사한다. 실측(graph_health)에서
    # 청크 92 중 65 가 그 상태였는데, 순서대로 훑는 기본 동작은 상한 30 안에
    # 그 공백에 닿지 못했다. 기본값 False — 기존 호출자의 대상을 바꾸지 않는다.
    only_unlinked: bool = False
    min_len: int = Field(default=COVERAGE_MIN_LEN, ge=1, le=5000)


@router.post("/graphs/{namespace}/review/coverage")
async def check_coverage(
    namespace: str,
    request: CoverageRequest = CoverageRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 에이전트 — 원문에 있는데 그래프에 없는 개체(gap) 탐지.

    원문에 문자 그대로 없는 후보는 버린다(환각 무효) · 이미 아는 개체는
    gap 이 아니다 · 발견은 감사 로그에 남는다(action=coverage_gap, 판정
    아님). 동기 실행 — limit 상한(30)이 지연을 막는다.
    """
    return await service.check_coverage(
        namespace, limit=request.limit,
        only_unlinked=request.only_unlinked, min_len=request.min_len)


class CoverageGap(BaseModel):
    name: str
    type: str
    chunk_id: str
    # 기각된 후보의 명시적 번복 — 없으면 승인이 gap 묘비에서 막힌다.
    override_rejected: bool = False


class RejectCoverageGap(BaseModel):
    name: str
    type: str
    reason: str = ""        # 필수 — 서비스가 사유 없는 기각을 거른다
    chunk_id: str = ""
    scope: str = "entity"   # entity(개체 전체) | evidence(이 청크만)


class RejectCoverageRequest(BaseModel):
    gaps: List[RejectCoverageGap] = Field(default_factory=list)
    actor: str = ""


@router.post("/graphs/{namespace}/review/coverage/reject")
async def reject_coverage_gaps(
    namespace: str,
    request: RejectCoverageRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 gap 기각 — 관계판 묘비의 대칭 (LLM 0콜, 그래프 무변경).

    기각이 남지 않으면 같은 후보가 매 라운드 재출현한다 — 관계 제안에서
    8/10 재출현으로 실측된 결함이고, 여기는 재실행 비용이 청크당 LLM 1콜이라
    조건이 더 나쁘다. dry_run 이 없다: 판정만 기록하므로 미리 볼 것이 없다.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.reject_gaps(
        namespace, [g.model_dump() for g in request.gaps],
        actor=request.actor))


class ApproveCoverageRequest(BaseModel):
    gaps: List[CoverageGap] = Field(default_factory=list)
    actor: str = ""
    # merge 와 같은 계약: 일괄 쓰기의 기본값이 '적용'이면 사고가 조용히 커진다
    dry_run: bool = True


@router.post("/graphs/{namespace}/review/coverage/approve")
async def approve_coverage_gaps(
    namespace: str,
    request: ApproveCoverageRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 gap 승인 — 놓친 개체를 **노드 + 근거 링크**로 되돌린다.

    LLM 0콜(검증은 결정적). 요청 본문은 신뢰하지 않는다 — 이름이 그 청크
    원문에 문자 그대로 있어야 하고(환각 차단), 타입은 그래프 실존 타입만
    (스키마 뒷문 차단), 묘비된 후보는 되살리지 않는다(판정을 조용히 뒤집지
    않는다). 이미 있는 노드는 링크만 한다. dry_run 기본 True.

    노드를 만들어도 시맨틱 색인은 갱신되지 않는다 — reindex_required 가
    True 면 /reindex 를 돌려야 검색에 잡힌다.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.approve_coverage_gaps(
        namespace, gaps=[g.model_dump() for g in request.gaps],
        actor=request.actor, dry_run=request.dry_run))


@router.get("/graphs/{namespace}/coverage-gate")
async def coverage_gate(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 expectation 게이트 — 현재 상태 재평가 (LLM 0콜, B2).

    빌드 시점 스냅샷은 잡 리포트의 coverage_gate 에, 현재 상태는 여기.
    같은 자(graph_health) + 네임스페이스별 임계. 임계 미설정 = unconfigured.
    """
    return _unwrap_write((await _read(service.coverage_gate, namespace)))


@router.get("/graphs/{namespace}/coverage-expectations")
async def get_coverage_expectations_settings(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 임계 — 오버라이드 + 실효값 + 전역 기본값 (retrieval-config 대칭)."""
    return _unwrap_write((await _read(service.get_coverage_expectations_settings, namespace)))


class CoverageExpectationsRequest(BaseModel):
    overrides: Dict[str, Any] = Field(default_factory=dict)
    actor: str = ""


@router.post("/graphs/{namespace}/coverage-expectations")
async def set_coverage_expectations_settings(
    namespace: str,
    request: CoverageExpectationsRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 임계 변경 — 병합, null 은 삭제, 오타 키 소리내는 거부, 감사."""
    _guard_protected_write(namespace)
    return _unwrap_write(service.set_coverage_expectations_settings(
        namespace, request.overrides, actor=request.actor))


@router.get("/graphs/{namespace}/retrieval-config")
async def get_retrieval_settings(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """네임스페이스 검색 설정 — 오버라이드 + 실효값 + 전역 기본값.

    셋을 다 주는 이유: 화면이 "무엇을 내가 정했고, 무엇이 기본값이며, 지금 실제로
    쓰이는 값은 무엇인가"를 구별해 보여줘야 한다.
    """
    return _unwrap_write((await _read(service.get_retrieval_settings, namespace)))


class RetrievalConfigRequest(BaseModel):
    # 부분 오버라이드 — 미설정 키는 전역 기본값. null 은 그 키를 지운다.
    overrides: Dict[str, Any] = Field(default_factory=dict)
    actor: str = ""


@router.post("/graphs/{namespace}/retrieval-config")
async def set_retrieval_settings(
    namespace: str,
    request: RetrievalConfigRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """검색 설정 변경 — **병합**, `null` 은 기본값 복귀. 값·키를 검증한다.

    확산 채널의 가치가 커버리지에 따라 정반대로 측정됐다(ins_cancer_demo 동률 /
    PROJ-A hit@1 +2~3건) — 전역 상수로는 둘을 동시에 맞출 수 없다. 색인은
    건드리지 않는다(질의 시점 파라미터).
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.set_retrieval_settings(
        namespace, request.overrides, actor=request.actor))


class LifecycleRequest(BaseModel):
    node_id: str
    state: str                      # experimental | active | deprecated
    reason: str = ""                # deprecated 필수
    sunset: str = ""                # deprecated 필수 (YYYY-MM-DD)
    superseded_by: str = ""         # 선택 — 실존 노드여야 한다
    actor: str = ""


@router.post("/graphs/{namespace}/nodes/lifecycle")
async def set_node_lifecycle(
    namespace: str,
    request: LifecycleRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """노드 생애주기 전이 — **파괴를 통제하는 선언 지점**.

    `active` 는 삭제·개명을 차단하고(reject·merge 가 거부한다), 그것을 벗어나는
    유일한 길이 `deprecated` 다 — 사유 + 삭제 기한이 필수이며, 그래야
    `active → experimental → 삭제` 우회가 막힌다. 대체 노드는 실존을 확인한다.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.set_node_lifecycle(
        namespace, request.node_id, request.state, reason=request.reason,
        sunset=request.sunset, superseded_by=request.superseded_by,
        actor=request.actor))


class ProposeRelationsRequest(BaseModel):
    limit: int = Field(default=0, ge=0, le=500)
    min_nodes: int = Field(default=2, ge=2, le=20)


@router.post("/graphs/{namespace}/review/relations")
async def propose_relations(
    namespace: str,
    request: ProposeRelationsRequest = ProposeRelationsRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관계 제안 — 청크마다 LLM 1콜, **그래프를 바꾸지 않는다**.

    술어 어휘는 그래프에 이미 있는 것 + `is_a`(코드가 아는 유일한 술어)로
    제한한다. 양끝은 그 청크에 연결된 노드만, 근거는 원문 인용이 필수다.
    """
    return _unwrap_write(await service.propose_relations(
        namespace, limit=request.limit, min_nodes=request.min_nodes))


class RelationItem(BaseModel):
    subject: str
    predicate: str
    object: str
    chunk_id: str
    evidence_quote: str = ""
    # 관계판 묘비의 명시적 번복 — 기각된 트리플은 기본 skip 이고,
    # 이 플래그를 명시해야만 승인이 묘비를 걷는다 (조용한 번복 차단).
    override_rejected: bool = False


class ApproveRelationsRequest(BaseModel):
    relations: List[RelationItem] = Field(default_factory=list)
    actor: str = ""
    dry_run: bool = True


@router.post("/graphs/{namespace}/review/relations/approve")
async def approve_relations(
    namespace: str,
    request: ApproveRelationsRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관계 승인 — LLM 0콜, dry_run 기본 True.

    요청 본문은 신뢰하지 않는다: 양끝이 그 청크의 노드인지, 술어가 허용 어휘인지,
    인용이 그 청크 원문에 있는지를 **다시** 검증한다. 노드를 만들지 않으며
    묘비된 노드로는 잇지 않는다. 엣지 추가는 시맨틱 색인을 무효화하지 않는다.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.approve_relations(
        namespace, relations=[r.model_dump() for r in request.relations],
        actor=request.actor, dry_run=request.dry_run))


class RejectRelationItem(BaseModel):
    subject: str
    predicate: str
    object: str
    reason: str = Field(min_length=1)   # 이유 없는 기각은 감사가 아니다
    chunk_id: str = ""
    scope: str = "triple"               # "triple" | "evidence"


class RejectRelationsRequest(BaseModel):
    relations: List[RejectRelationItem] = Field(min_length=1)
    actor: str = ""


@router.post("/graphs/{namespace}/review/relations/reject")
async def reject_relations(
    namespace: str,
    request: RejectRelationsRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관계 제안 기각 — 관계판 묘비 (A2).

    기각이 기록되지 않아 같은 오제안이 매 라운드 재출현하던 결함(실측 8/10)의
    처방. scope=triple(기본)은 어느 인용에서 와도 거르고, evidence 는 같은
    청크 재제안만 거른다. 번복은 approve 의 override_rejected 로만.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.reject_relations(
        namespace, relations=[r.model_dump() for r in request.relations],
        actor=request.actor))


@router.get("/graphs/{namespace}/review/structural")
async def review_structural(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """구조 단위(조항·문서) 재분류 후보 — 로드맵 4 P-1 (LLM 0콜, 결정적).

    section 라벨 **전체-정규화 동등성**만 후보 (포함 매칭은 오탐 실측으로
    기각). evidence_matched·lifecycle 등 판단 재료 동봉 — 판정은 사람이,
    적용은 P-2 개명 경로가 한다.
    """
    return _unwrap_write((await _read(service.review_structural, namespace)))


class ApproveStructuralRequest(BaseModel):
    items: List[str] = Field(min_length=1)
    new_type: str = Field(min_length=1)   # 서버 기본값 없음 — 검수자가 정한다
    actor: str = ""
    dry_run: bool = True                  # 개명은 되돌릴 수 없다 — merge 와 동일


@router.post("/graphs/{namespace}/review/structural/approve")
async def approve_structural(
    namespace: str,
    request: ApproveStructuralRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """구조 단위 재분류 승인 — P-3 (dry_run 기본).

    본문 불신: 탐지를 재실행해 여전히 후보인 것만 개명한다. 항목별
    실패(active·비후보)는 전체를 막지 않되 미리보기에 미리 나타난다.
    미선언 타입은 적용 시 schema_decl 에 선언 + 감사.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.approve_structural(
        namespace, request.items, request.new_type,
        actor=request.actor, dry_run=request.dry_run))


@router.get("/graphs/{namespace}/review/duplicates")
async def review_duplicates(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """중복 클러스터 검수 제안 — 판단 신호 동봉 (LLM 0콜, 결정적).

    variant(표기 변형·병합 payload 동봉) / cross_type(타입 갈림·사람 판단) /
    similar(포함 등·C73 경고). 분류는 신호이지 판정이 아니다 — 적용은
    /nodes/merge (dry_run 기본) 로, 판단은 사람이 한다.
    """
    return _unwrap_write((await _read(service.review_duplicates, namespace)))


@router.get("/graphs/{namespace}/review/orphans")
async def find_orphan_nodes(
    namespace: str,
    limit: int = Query(0, ge=0, le=500),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """근거 링크가 없는 노드 + 원문 인용 후보 (LLM 0콜, 결정적).

    커버리지 검사의 **대칭**이다 — 그쪽은 "노드가 없는 청크", 이쪽은 "청크가
    없는 노드". 세 갈래로 나눠 준다: candidates(회복 가능) ·
    shadowed(더 긴 이름의 부분문자열 = 오추출 의심) · unquotable(원문에 없음).
    """
    return _unwrap_write((await _read(service.find_orphan_nodes, namespace, limit=limit)))


class OrphanLink(BaseModel):
    node_id: str
    chunk_id: str


class ApproveOrphanLinksRequest(BaseModel):
    links: List[OrphanLink] = Field(default_factory=list)
    actor: str = ""
    dry_run: bool = True


@router.post("/graphs/{namespace}/review/orphans/approve")
async def approve_orphan_links(
    namespace: str,
    request: ApproveOrphanLinksRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """고아 노드에 근거 청크를 잇는다. LLM 0콜, dry_run 기본 True.

    요청 본문은 신뢰하지 않는다 — 노드 이름이 그 청크 원문에 문자 그대로 있어야
    하고(이름은 그래프에서 읽는다), 더 긴 노드 이름에 가려지면 거부한다(오추출
    차단). 묘비된 노드는 잇지 않는다. 노드를 만들지 않으므로 재색인 불필요.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.approve_orphan_links(
        namespace, links=[l.model_dump() for l in request.links],
        actor=request.actor, dry_run=request.dry_run))


@router.get("/graphs/{namespace}/review/history")
async def get_review_history(
    namespace: str,
    node_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """감사 이력(최신 먼저) — 누가 언제 무엇을 어떤 근거로 판정했는가."""
    return (await _read(service.get_review_history, namespace, node_id=node_id, limit=limit))


class TriageRequest(BaseModel):
    limit: int = Field(default=30, ge=1, le=50)   # 노드당 LLM ≤1콜 — 상한 필수
    include_relations: bool = False               # 청크당 LLM 1콜 — 명시 옵트인
    actor: str = "triage"


@router.post("/graphs/{namespace}/review/triage")
async def triage_review(
    namespace: str,
    request: TriageRequest = TriageRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """검수 트리아지 — 렌즈 신호(근거대조·중복·일관성·구조단위)를 밴드로 결합.

    결정적 신호만으로 borderline 이 확정된 항목은 LLM 콜을 생략한다(예산).
    결과는 recommend 이벤트로 남는다 — **판정이 아니다**, 판정은 judge-batch
    경유 인간만 만든다. include_relations 는 propose 비용 때문에 옵트인.
    """
    return _unwrap_write(await service.triage_review(
        namespace, limit=request.limit, actor=request.actor,
        include_relations=request.include_relations))


class JudgeBatchRequest(BaseModel):
    # 항목은 이종(kind=node|relation)이라 자유 dict — 검증은 서비스가 항목별로
    # 한다 (skip 사유로 보고, 한 항목의 오류가 배치를 죽이지 않는다).
    items: List[Dict[str, Any]] = Field(min_length=1)
    actor: str = ""
    dry_run: bool = True    # 일괄 판정은 조용히 적용되면 안 된다 — 미리보기 기본


@router.post("/graphs/{namespace}/review/judge-batch")
async def judge_batch(
    namespace: str,
    request: JudgeBatchRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """트리아지 밴드 일괄 판정 — 본문 불신 + dry_run 미리보기 == 적용.

    노드 항목은 미판정 여부와 **최신 추천의 verdict/band 일치**를 재검증하고
    (낡은 추천 skip), 통과분만 confirm/reject 기존 경로에 위임한다. 관계
    항목은 approve/reject_relations 위임 — 그쪽 관문이 인용을 재검증한다.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.judge_batch(
        namespace, items=request.items, actor=request.actor,
        dry_run=request.dry_run))


@router.get("/graphs/{namespace}/review/recommendation-quality")
async def recommendation_quality(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """일치율 자 — recommend ↔ 이후 사람 판정 대조 (로그 replay, LLM 0콜).

    GET 인 이유: 이벤트 로그에서 재계산되는 파생물이라 서버 상태를 바꾸지
    않는다 (lint_consistency 와 같은 계약). 일괄 승인 UI 의 안전핀 —
    표본 n 이 작으면 일괄 버튼을 열지 않는 근거 데이터가 이것이다.
    """
    return _unwrap_write((await _read(service.recommendation_quality, namespace)))


# ─── 검색 QA (⑤ — 골든셋) ───────────────────────────────────────────

class GoldenCaseRequest(BaseModel):
    query: str
    expected_node_id: str = ""             # 노드 정답 (청크만 라벨할 수도 있다)
    tags: Optional[List[str]] = None       # 시나리오 축(exact/semantic/graph × 의도)
    accepted: Optional[List[str]] = None   # 추가 정답(relevant set) — 동의어/교차연결
    # 청크 단위 정답 — 노드와 별개 축. 그래프 조건화의 효과는 청크 회수에서
    # 드러나므로 이 축이 있어야 RRF·entry_ratio 를 측정으로 정할 수 있다.
    expected_chunk_id: str = ""
    accepted_chunks: Optional[List[str]] = None


class GoldenGenerateRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=30)     # 노드당 LLM 1콜 — 상한 필수
    per_node: int = Field(default=2, ge=1, le=5)


class GoldenVerifyRequest(BaseModel):
    limit: int = Field(default=0, ge=0, le=200)


class GoldenEvaluateRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20)
    include_drafts: bool = False
    # 어느 라벨 집합을 잴 것인가 (기본 None → confirmed 만).
    # ["confirmed","verified"] 로 주면 왕복 검증분까지 포함한다.
    statuses: Optional[List[str]] = None
    # 채점 위치. node(기본, 하위호환) | chunk(청크 라벨 필요) |
    # evidence(청크 랭킹을 기존 노드 라벨로 채점 — 새 라벨 없이 청크 단위 측정)
    target: str = "node"


@router.get("/graphs/{namespace}/qa")
async def list_golden_cases(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return (await _read(service.get_golden_cases, namespace))


@router.post("/graphs/{namespace}/qa/cases")
async def add_golden_case(
    namespace: str,
    request: GoldenCaseRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """수동 케이스 추가 (곧바로 confirmed). 중복 질의는 409."""
    if not (request.expected_node_id or request.expected_chunk_id):
        # 정답이 하나도 없는 케이스는 어느 타깃에서도 채점되지 않는다 —
        # 조용히 스킵될 케이스를 만들어두면 골든셋 수가 거짓으로 늘어난다.
        raise HTTPException(status_code=400,
                            detail="expected_node_id or expected_chunk_id required")
    case_id = service.add_golden_case(namespace, request.query,
                                      request.expected_node_id,
                                      tags=request.tags,
                                      accepted=request.accepted,
                                      expected_chunk_id=request.expected_chunk_id,
                                      accepted_chunks=request.accepted_chunks)
    if case_id is None:
        raise HTTPException(status_code=409, detail="duplicate or blank query")
    return {"case_id": case_id}


@router.post("/graphs/{namespace}/qa/cases/{case_id}/confirm")
async def confirm_golden_case(
    namespace: str,
    case_id: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    if not service.confirm_golden_case(namespace, case_id):
        raise HTTPException(status_code=404, detail=f"case not found: {case_id}")
    return {"case_id": case_id, "status": "confirmed"}


class AcceptAnswerRequest(BaseModel):
    node_id: str


@router.post("/graphs/{namespace}/qa/cases/{case_id}/accept")
async def accept_golden_answer(
    namespace: str,
    case_id: str,
    request: AcceptAnswerRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """정답 집합 확장(accepted 추가) — 라벨 노후화 처방.

    코퍼스가 자라면 같은 개념이 새 노드로 생기고, 그걸 모르는 라벨은 검색 개선을
    회귀로 오보고한다(실측: PROJ-A 회복 후 hit@1 하락의 정체). 판정(status)은
    건드리지 않는다 — 정답 확장은 판정 번복이 아니다. 노드 실존 확인.
    """
    return _unwrap_write(service.accept_golden_answer(
        namespace, case_id, request.node_id))


@router.post("/graphs/{namespace}/qa/generate")
async def generate_golden_cases(
    namespace: str,
    request: GoldenGenerateRequest = GoldenGenerateRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """LLM 패러프레이즈 초안 생성 (gemini, 노드당 1콜) — 전부 draft,
    확정은 인간이. 이름이 그대로 든 질의는 검증에서 버려진다."""
    return await service.generate_golden_cases(
        namespace, limit=request.limit, per_node=request.per_node)


@router.post("/graphs/{namespace}/qa/evaluate")
async def evaluate_golden_set(
    namespace: str,
    request: GoldenEvaluateRequest = GoldenEvaluateRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """골든셋 평가 — 결정적 (hit@1 / hit@k / MRR + 케이스별 순위).
    검색 변경 전후로 이걸 돌리면 회귀가 숫자로 보인다.
    target=chunk 면 청크 단위로 채점한다 — 그래프 조건화의 효과가 드러나는 자리."""
    return (await _read(service.evaluate_golden_set, namespace, k=request.k,
                                       include_drafts=request.include_drafts,
                                       target=request.target,
                                       statuses=request.statuses))


class RunExperimentsRequest(BaseModel):
    # None = 기본 Tier-0 격자 (채널 3종: vector / graph / graph+prop)
    axes: Optional[Dict[str, List[Any]]] = None
    k: int = 5
    target: str = "evidence"
    statuses: Optional[List[str]] = None
    max_combos: int = Field(64, ge=1, le=256)
    actor: str = "manual"


@router.post("/graphs/{namespace}/experiments/run")
async def run_retrieval_experiments(
    namespace: str,
    request: RunExperimentsRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """실험 하네스 Tier 0 러너 — 조합 × 골든셋 → 품질+비용 (LLM 0콜).

    "벡터만 vs +온톨로지 vs +확산" 채널 비교가 기본 격자. 레코드에 지문
    3종(설정·그래프·골든셋)과 표본 경고가 박히고, eval_history 는 우회한다
    (운영 품질 블록 오염 방지). 응답에 파레토 프런티어 동봉.
    """
    return _unwrap_write((await _read(service.run_retrieval_experiments, 
        namespace, axes=request.axes, k=request.k, target=request.target,
        statuses=request.statuses, max_combos=request.max_combos,
        actor=request.actor)))


@router.get("/graphs/{namespace}/experiments")
async def get_experiments(
    namespace: str,
    limit: int = Query(200, ge=1, le=1000),
    layer: Optional[str] = Query(None),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """실험 레코드 조회 (최신 먼저, layer=retrieval|routing 필터)."""
    return _unwrap_write((await _read(service.get_experiments, namespace, limit=limit,
                                                 layer=layer)))


@router.get("/graphs/{namespace}/experiments/recommendation")
async def recommend_retrieval_config(
    namespace: str,
    quality: str = Query("mrr"),
    cost: str = Query("latency_ms_p50"),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """실험 결과 → 설정 제안 (제안까지만 — 적용은 retrieval-config API).

    현재 그래프 지문과 같은 레코드만 후보 (다르면 stale — 재실험 먼저).
    표본 부족이면 제안 대신 경고.
    """
    return _unwrap_write((await _read(service.recommend_retrieval_config, 
        namespace, quality=quality, cost=cost)))


@router.get("/graphs/{namespace}/qa/history")
async def get_eval_history(
    namespace: str,
    limit: int = Query(50, ge=1, le=500),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """평가 이력 — **설정 지문 + 지표**. 관리 콘솔의 비교용.

    점수만 남기면 "그때 그 숫자가 어떤 설정에서 나온 것인가"를 잃는다.
    임베더·entry_k·max_terms·확산 플래그가 전부 지표를 움직인다(각각 실측).
    """
    return _unwrap_write((await _read(service.get_eval_history, namespace, limit=limit)))


@router.post("/graphs/{namespace}/qa/verify")
async def verify_golden_cases(
    namespace: str,
    request: GoldenVerifyRequest = GoldenVerifyRequest(),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """초안 왕복 검증 → verified 승격 (케이스당 LLM 1콜).

    노드 → 질의 → **혼동 후보 중에서 다시 노드**. 후보는 의미 이웃이라 과제가
    쉽지 않고, 거부("none")를 허용해 추측이 통과하지 못한다. verified 는
    confirmed(인간 확정)와 구별되며 기본 평가에는 들어가지 않는다 —
    qa/evaluate 에 statuses=["confirmed","verified"] 로 명시해야 포함된다.
    """
    return _unwrap_write(await service.verify_golden_cases(
        namespace, limit=request.limit))


@router.get("/graphs/{namespace}/documents")
async def list_documents(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """문서 목록 — 문서별 청크·근거 커버리지·노드 수 (LLM 0콜, 결정적).

    문서는 KG 노드가 아니라 `chunk.source` 축의 뷰다 — 노드로 만들면 슈퍼허브가
    되어 채널 B·확산을 오염시킨다 (core/document_view 의 측정 근거 참고).
    """
    return _unwrap_write((await _read(service.list_documents, namespace)))


@router.get("/graphs/{namespace}/coverage-map")
async def get_coverage_map(
    namespace: str,
    source: Optional[str] = Query(default=None,
                                  description="문서(source)로 한정 — 미지정 시 전 문서"),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """커버리지 지도 — 문서를 원문 순서대로 편 청크 스트립 (근거 링크 수 포함).

    관리 콘솔의 "비어 있는 구간이 어느 절인가" 화면 데이터. 없는 source 는
    빈 결과가 아니라 소리내어 거부한다 (오타 = "전부 커버됨" 오독 방지).
    """
    result = (await _read(service.coverage_map, namespace, source=source or ""))
    if result.get("error") == "namespace_not_found":
        raise HTTPException(status_code=404, detail=f"namespace not found: {namespace}")
    if result.get("error") == "source_not_found":
        raise HTTPException(status_code=404, detail=result.get("detail"))
    return result


@router.get("/graphs/{namespace}/documents/compare")
async def compare_documents(
    namespace: str,
    a: str = Query(..., description="기준 문서 source"),
    b: str = Query(..., description="대조 문서 source"),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """문서 간 개체 대조 — "A가 다루는 것 중 B에 없는 것" (LLM 0콜).

    개체의 문서 귀속 = 그 문서의 청크에 근거 링크가 있는가. 근거 링크가 성기면
    대조도 성기다 — 문서별 coverage(/documents)를 함께 볼 것. 없는 문서는
    소리내어 거부한다(오타가 빈 결과를 내면 "전부 커버됨"으로 오독된다).
    """
    return _unwrap_write((await _read(service.compare_documents, namespace, a, b)))


@router.get("/graphs/{namespace}/retrieve")
async def retrieve(
    namespace: str,
    query: str = Query(...),
    top_k: int = Query(default=5, ge=1, le=100),
    source: Optional[str] = Query(default=None,
                                  description="문서 필터 — 근거만 이 문서로 제한"),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """그래프-조건부 검색 (축 4) — 온톨로지로 확장한 원문 검색.

    응답에 expansion(확장 어휘 · 진입 노드)과 히트별 matched_via/channels 가
    함께 실린다 — 왜 이 결과인지 보이지 않으면 신뢰할 근거가 없다.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")
    return (await _read(service.retrieve, namespace, query, top_k=top_k, source=source))


# ─── 관리 콘솔 (frontend /ontology-admin 전용 소비자) ────────────────

@router.get("/admin/overview")
async def admin_overview(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관리 대시보드 한 콜 — 네임스페이스별 노드/청크/검수/trust/디스크.

    깨진 네임스페이스는 그 행만 error 로 degrade 한다 — 한 그래프의 손상이
    대시보드 전체를 죽이면 관리자는 정확히 그 순간에 장님이 된다.
    """
    return (await _read(service.get_admin_overview))


@router.get("/admin/system")
async def admin_system(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """시스템 개요 한 콜 — ES/PG/VectorDB/청크/파일의 실물 지표 + 네임스페이스 요약.

    각 저장 백엔드는 독립적으로 degrade 한다({available:false}) — ES 가 죽어도
    PG·VectorDB 는 살고, 어느 하나의 부재가 500 이 되지 않는다.
    """
    return (await _read(service.system_overview))


@router.get("/admin/health")
async def admin_health(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """연결 헬스 — PG/ES 핑 레이턴시(ms) + 임베딩 모델. system_overview 보다
    가벼운 프로브(SELECT 1 / _cluster/health 왕복만). 각 백엔드 독립 degrade."""
    return (await _read(service.health_check))


@router.get("/graphs/{namespace}/stats")
async def get_namespace_stats(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관리 상세 — 분포(타입/술어/trust) + 검수 현황 + 디스크 파일.

    미지 네임스페이스는 404 — 조회가 빈 엔진을 만들어 등록하면 그 자체가
    유령 네임스페이스 오염이다.
    """
    stats = (await _read(service.get_namespace_stats, namespace))
    if stats is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return stats


@router.get("/graphs/{namespace}/tombstones")
async def list_tombstones(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """묘비 목록 — 거절된 노드의 사유·시점·판정자. 부활 차단의 열람 창구."""
    return (await _read(service.list_tombstones, namespace))


# 쓰기 계열(생성·수정·엣지) 공통 에러 → HTTP 매핑
_WRITE_ERROR_STATUS = {
    "namespace_not_found": 404,
    "node_not_found": 404,
    "winner_not_found": 404,
    "loser_not_found": 404,
    "edge_not_found": 404,
    "view_not_found": 404,
    "type_not_found": 404,
    "duplicate": 409,
    "invalid": 400,
    "pg_unavailable": 503,
}


def _unwrap_write(result: Dict[str, Any]) -> Dict[str, Any]:
    error = result.get("error")
    if error:
        status = _WRITE_ERROR_STATUS.get(error, 400)
        raise HTTPException(status_code=status,
                            detail=result.get("detail") or error)
    return result


def _guard_protected_write(namespace: str) -> None:
    """default(에이전트 라우팅 그래프)는 관리 콘솔에서 읽기 전용이다 —
    수동 편집이 라우팅을 조용히 망가뜨리는 경로를 원천 차단한다."""
    if namespace in PROTECTED_NAMESPACES:
        raise HTTPException(
            status_code=403,
            detail=f"namespace '{namespace}' is protected (read-only)")


@router.get("/graphs/{namespace}/schema")
async def get_schema_overview(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """스키마(TBox) 요약 — 관측된 클래스·프로퍼티 사용·술어 시그니처·
    is_a 계층. 선언이 아니라 관측을 보고한다 (관측이 그래프의 진실이다)."""
    schema = (await _read(service.get_schema_overview, namespace))
    if schema is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return schema


@router.get("/graphs/{namespace}/nodes")
async def list_nodes(
    namespace: str,
    q: Optional[str] = Query(default=None),
    node_type: Optional[str] = Query(default=None),
    trust: Optional[str] = Query(default=None),
    property: Optional[str] = Query(default=None),
    kind: Optional[str] = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """노드 브라우저 — 검색(이름·별칭 부분일치) + 타입/trust/property 필터
    + 페이지네이션. trust=unset 은 등급 미부여 노드를, property=X 는 그
    프로퍼티가 채워진 노드만(각 항목에 prop_value 동봉) 고른다.
    kind=class|instance 는 클래스(타입 정의)만/인스턴스(실제 객체)만 고른다."""
    result = (await _read(service.list_nodes, namespace, q=q, node_type=node_type,
                                trust=trust, prop=property, kind=kind,
                                offset=offset, limit=limit))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


@router.get("/graphs/{namespace}/edges")
async def list_edges(
    namespace: str,
    predicate: Optional[str] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    target_type: Optional[str] = Query(default=None),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """엣지 목록 — 술어(+선택적 시그니처)로 거른 관계들. 양끝 노드의 이름·
    타입이 해석돼 실려온다. 술어를 '나열만' 하던 스키마에 관리 진입점을 준다."""
    result = (await _read(service.list_edges, namespace, predicate=predicate,
                                source_type=source_type, target_type=target_type))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


@router.get("/graphs/{namespace}/neighbors")
async def list_neighbors(
    namespace: str,
    node_id: str = Query(...),
    limit: int = Query(default=60, ge=1, le=300),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """앵커 노드의 이웃 서브그래프 (앵커→이웃 확장 3D 기반). 비용은 그 노드의
    차수에 비례 — 전체 그래프를 읽지 않는다. 이웃이 limit 초과면 truncated."""
    result = (await _read(service.list_neighbors, namespace, node_id, limit=limit))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return _unwrap_write(result)


class NodeCreateRequest(BaseModel):
    node_type: str
    name: str
    definition: str = ""
    aliases: List[str] = Field(default_factory=list)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    actor: str = ""


@router.post("/graphs/{namespace}/nodes")
async def create_node(
    namespace: str,
    request: NodeCreateRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """수동 노드 생성 — 감사 로그(action=create)에 남는다.
    묘비가 있던 id 는 confirm 으로 걷고 만든다 (명시적 번복)."""
    _guard_protected_write(namespace)
    result = _unwrap_write(service.create_node(
        namespace, request.node_type, request.name,
        definition=request.definition, aliases=request.aliases,
        attrs=request.attrs, actor=request.actor))
    return result["node"]


class NodeUpdateRequest(BaseModel):
    updates: Dict[str, Any]
    actor: str = ""


@router.patch("/graphs/{namespace}/node")
async def update_node(
    namespace: str,
    request: NodeUpdateRequest,
    node_id: str = Query(...),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """노드 프로퍼티 편집 — 값 null 은 프로퍼티 삭제. 변경된 키의
    before/after 가 감사 로그(action=edit)에 남는다."""
    _guard_protected_write(namespace)
    result = _unwrap_write(service.update_node(
        namespace, node_id, request.updates, actor=request.actor))
    return result["node"]


class EdgeRequest(BaseModel):
    source: str
    predicate: str
    target: str
    actor: str = ""


@router.post("/graphs/{namespace}/edges")
async def add_edge(
    namespace: str,
    request: EdgeRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관계 추가 — 양끝 노드가 있어야 한다 (유령 노드 auto-create 금지),
    같은 (source, predicate, target) 은 409."""
    _guard_protected_write(namespace)
    return _unwrap_write(service.add_edge(
        namespace, request.source, request.predicate, request.target,
        actor=request.actor))


@router.delete("/graphs/{namespace}/edges")
async def remove_edge(
    namespace: str,
    source: str = Query(...),
    predicate: str = Query(...),
    target: str = Query(...),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """관계 삭제 — 감사 로그(action=edge_removed)에 남는다."""
    _guard_protected_write(namespace)
    return _unwrap_write(service.remove_edge(
        namespace, source, predicate, target))


@router.delete("/graphs/{namespace}")
async def delete_namespace(
    namespace: str,
    actor: str = "",
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """네임스페이스 완전 삭제 — 그래프·청크·검수로그·골든셋·벡터캐시.

    protected(default = 에이전트 라우팅 그래프)는 403 — 어떤 관리 실수도
    이 선을 넘지 못한다. 프론트의 이름 재입력 확인은 UX 안전장치일 뿐,
    최종 방어선은 여기다.
    """
    if namespace in PROTECTED_NAMESPACES:
        raise HTTPException(
            status_code=403,
            detail=f"namespace '{namespace}' is protected and cannot be deleted")
    result = service.delete_namespace(namespace, actor=actor)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


# ─── 저장된 탐색 (SQLite 영속 — 팔란티어 saved explorations) ─────────

class SavedViewRequest(BaseModel):
    name: str
    filter: Dict[str, Any] = Field(default_factory=dict)  # 불투명 필터 blob


@router.get("/graphs/{namespace}/views")
async def list_saved_views(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """이 네임스페이스의 저장된 탐색 목록 (최신 먼저)."""
    result = (await _read(service.list_saved_views, namespace))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


@router.post("/graphs/{namespace}/views")
async def create_saved_view(
    namespace: str,
    request: SavedViewRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """탐색 필터를 이름 붙여 저장 — 같은 이름은 409. 저장은 무해한 운영자
    편의라 protected 네임스페이스에도 허용한다(그래프를 바꾸지 않는다)."""
    return _unwrap_write(service.create_saved_view(
        namespace, request.name, request.filter))


@router.delete("/graphs/{namespace}/views/{view_id}")
async def delete_saved_view(
    namespace: str,
    view_id: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    return _unwrap_write(service.delete_saved_view(namespace, view_id))


# ─── LLM 등록·설정 (전역, env-only 키) ──────────────────────────────

class LLMConfigRequest(BaseModel):
    provider: str
    model: str
    base_url: str = ""


@router.get("/admin/llm")
async def get_llm_config(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """현재 LLM 설정 + 키 존재 여부(마스킹). 원문 키는 반환하지 않는다."""
    return (await _read(service.get_llm_config))


@router.put("/admin/llm")
async def set_llm_config(
    request: LLMConfigRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """provider/model/base_url 저장 (키는 저장하지 않음 — 환경변수에서 읽는다)."""
    return _unwrap_write(service.set_llm_config(
        request.provider, request.model, request.base_url))


@router.post("/admin/llm/test")
async def test_llm(
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """연결 테스트 — 유효 LLM 으로 한 콜. 항상 200 (진단용 {ok, sample|error})."""
    return await service.test_llm()


# ─── 프로젝트 (네임스페이스 = 프로젝트) ─────────────────────────────

class ProjectCreateRequest(BaseModel):
    name: str
    description: str = ""
    domain: str = ""


@router.post("/admin/projects")
async def create_project(
    request: ProjectCreateRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """빈 프로젝트 생성 — 레코드만. 그래프는 첫 빌드/노드에서 태어난다.
    이미 존재(그래프/레코드)면 409, protected·형식오류면 400."""
    return _unwrap_write(service.create_project(
        request.name, description=request.description, domain=request.domain))


@router.post("/graphs/{namespace}/migrate-pg")
async def migrate_namespace_to_pg(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """이 네임스페이스를 PostgreSQL 로 백필(축 5, P2). KG→PG 미러 후 개수 반환.
    읽기 전환은 별도(ONTOLOGY_PG_NAMESPACES). PG 불가 시 503."""
    return _unwrap_write(service.migrate_namespace_to_pg(namespace))


@router.get("/graphs/{namespace}/index-status")
async def index_status(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """검색 인덱스 상태(관측) — ES 노드 인덱스 문서수·동기 여부, 원문 청크 수,
    벡터검색 가용성. 관리 콘솔에서 인덱스 건강/drift 를 본다."""
    result = (await _read(service.index_status, namespace))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


@router.post("/graphs/{namespace}/reindex")
async def reindex_namespace(
    namespace: str,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """검색 인덱스 재구축 — ES 노드 인덱스 + 청크 벡터 인덱스(업로드 자동
    인덱싱과 동일 경로). 과거 네임스페이스·편집 후 drift 복구용. best-effort."""
    if service.index_status(namespace) is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return {"namespace": namespace, "reindexed": True,
            **service.index_namespace(namespace)}


@router.get("/graphs/{namespace}/query")
async def query_nodes(
    namespace: str,
    node_type: Optional[str] = Query(default=None, alias="type"),
    trust: Optional[str] = Query(default=None),
    prop_key: Optional[str] = Query(default=None),
    prop_value: Optional[str] = Query(default=None),
    prop_op: str = Query(default="eq"),
    rel_predicate: Optional[str] = Query(default=None),
    rel_target: Optional[str] = Query(default=None),
    rel_target_type: Optional[str] = Query(default=None),
    rel_direction: str = Query(default="out"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """구조화 검색(축 5) — 프로퍼티 값 조건(prop_key/value/op) + 관계 제약
    (rel_predicate/target/target_type/direction)으로 노드를 찾는다. PG 는
    jsonb + EXISTS 인덱스 질의. 예: '경주에 located_in 된 노드' =
    rel_predicate=located_in&rel_target=Region:경주&rel_direction=in."""
    result = (await _read(service.query_nodes, 
        namespace, node_type=node_type, trust=trust, prop_key=prop_key,
        prop_value=prop_value, prop_op=prop_op, rel_predicate=rel_predicate,
        rel_target=rel_target, rel_target_type=rel_target_type,
        rel_direction=rel_direction, offset=offset, limit=limit))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


@router.get("/graphs/{namespace}/object-search")
async def object_search(
    namespace: str,
    q: Optional[str] = Query(default=None),
    node_type: Optional[str] = Query(default=None, alias="type"),
    trust: Optional[str] = Query(default=None),
    top_k: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """객체 검색(축 5, P3) — ES BM25 관련도 + 타입/신뢰 파셋(aggregation).
    ES 없으면 PG/memory substring 으로 폴백(source 필드로 구분)."""
    result = (await _read(service.object_search, namespace, q=q, node_type=node_type,
                                   trust=trust, top_k=top_k, offset=offset))
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"namespace '{namespace}' not found")
    return result


# ─── 스키마 편집 (TBox — 개명 · 선언) ───────────────────────────────

class RenameTypeRequest(BaseModel):
    old: str
    new: str
    actor: str = ""


class MergeNodesRequest(BaseModel):
    winner: str = Field(min_length=1)
    losers: List[str] = Field(min_length=1)
    actor: str = ""
    # 기본값 True — 병합은 노드를 지워 되돌릴 수 없다. 실수로 body 를 보내
    # 그래프가 파괴되는 경로를 원천 차단한다(명시적으로 꺼야 적용된다).
    dry_run: bool = True


@router.post("/graphs/{namespace}/nodes/merge")
async def merge_nodes(
    namespace: str,
    request: MergeNodesRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """중복 노드 병합 — graph_health 가 제시한 후보를 사람이 승인해 합친다.

    dry_run(기본 True)이면 계획만 돌려준다: 옮겨질 엣지, 흡수될 별칭, 재지정될
    청크·골든셋 참조, **사람이 정해야 할 프로퍼티 충돌**까지. 미리 본 계획과
    적용 결과가 같도록 계산은 순수 함수 하나(core.node_merge)에서만 한다.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.merge_nodes(
        namespace, request.winner, request.losers,
        actor=request.actor, dry_run=request.dry_run))


class RenameNodeRequest(BaseModel):
    node_id: str = Field(min_length=1)
    new_type: str = Field(min_length=1)
    actor: str = ""
    # 기본값 True — 개명은 옛 id 를 지워 되돌릴 수 없다 (merge 와 같은 이유).
    dry_run: bool = True


@router.post("/graphs/{namespace}/nodes/rename")
async def rename_node(
    namespace: str,
    request: RenameNodeRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """노드 타입 재분류 — id 개명 (로드맵 4 P-2, dry_run 기본).

    구조 단위 후보(/review/structural)를 사람이 승인해 타입을 바꾼다.
    타깃 id 가 이미 있으면 개명이 아니라 병합 — /nodes/merge 로 안내.
    묘비를 남기지 않는다 (재분류 ≠ 거절). reindex_required 항상 True.
    """
    _guard_protected_write(namespace)
    return _unwrap_write(service.rename_node(
        namespace, request.node_id, request.new_type,
        actor=request.actor, dry_run=request.dry_run))


@router.post("/graphs/{namespace}/schema/rename-type")
async def rename_type(
    namespace: str,
    request: RenameTypeRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """타입 개명 — 그 타입 노드 전체의 type 속성 변경(node_id 는 불변).
    기존 타입명으로 개명하면 병합된다. 감사 로그(action=rename_type)."""
    _guard_protected_write(namespace)
    return _unwrap_write(service.rename_type(
        namespace, request.old, request.new, actor=request.actor))


class PredicateDeclRequest(BaseModel):
    predicate: str
    domain: str = ""
    range: str = ""
    description: str = ""
    actor: str = ""


@router.put("/graphs/{namespace}/schema/predicate")
async def set_predicate_decl(
    namespace: str,
    request: PredicateDeclRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """술어 domain/range·설명 선언 — 관측 위에 얹는 의도. lint 이 대조한다."""
    _guard_protected_write(namespace)
    return _unwrap_write(service.set_predicate_decl(
        namespace, request.predicate, domain=request.domain,
        range_=request.range, description=request.description,
        actor=request.actor))


class TypeDeclRequest(BaseModel):
    type: str
    description: str = ""
    deprecated: bool = False
    actor: str = ""


@router.put("/graphs/{namespace}/schema/type")
async def set_type_decl(
    namespace: str,
    request: TypeDeclRequest,
    service: OntologyBuilderService = Depends(get_ontology_service),
):
    """타입 설명·deprecated 선언."""
    _guard_protected_write(namespace)
    return _unwrap_write(service.set_type_decl(
        namespace, request.type, description=request.description,
        deprecated=request.deprecated, actor=request.actor))
