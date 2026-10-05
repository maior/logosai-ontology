"""
Ontology Builder Service — dataset storage + build jobs over ontology.builder.

Standalone-server wrapper for the builder framework:
- datasets are folders of uploaded files under data_dir
- builds run as jobs with live progress (in-memory registry)
- llm_fn / data_dir are injectable so tests run with a fake LLM and tmp dirs
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

# provider → API 키를 담는 환경변수. 키 자체는 절대 저장·반환하지 않는다
# (env-only) — admin 은 "어떤 provider/model 인가"만 안다.
_ENV_KEY_BY_PROVIDER = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai_compatible": "OPENAI_COMPAT_API_KEY",
}

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"

# 커버리지 검사에서 '본문'으로 볼 최소 길이. graph_health 의 같은 이름 상수와
# 뜻이 같다(머리말·페이지번호를 추출 실패로 세지 않는다) — 한 곳에서 가져와
# 두 벌이 어긋나지 않게 한다. 측정값이 아니라 관찰에서 고른 값이므로 덮을 수 있다.
from ..core.graph_health import DEFAULT_MIN_CHUNK_LEN as COVERAGE_MIN_LEN  # noqa: E402

# The default namespace holds the agent-routing knowledge graph —
# document builds must never write into it.
PROTECTED_NAMESPACES = {"default"}


class OntologyBuilderService:
    # 근사 카운트 상한 — 목록 스캔이 이 개수에 닿으면 멈추고 total 을 "N+" 로
    # 보고한다. 초대형 그래프에서 전체 노드 스캔을 피하는 장치.
    COUNT_CAP = 10000

    def __init__(self, data_dir=None, llm_fn: Optional[Callable[[str], str]] = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.llm_fn = llm_fn
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._saved_views = None    # 지연 생성 (admin.db, 첫 사용 시)
        self._settings = None       # admin.db 전역 설정 (LLM 등)
        self._projects = None       # admin.db 프로젝트 메타
        self._schema_decl = None    # admin.db 스키마 선언
        self._llm_provider_cache: Dict[Tuple[str, str, str], Any] = {}

    # ─── Datasets ────────────────────────────────────────────────────

    @staticmethod
    def _safe_rel_path(filename: str) -> str:
        """상대 경로 정화 — traversal(`..`)·절대경로·빈 세그먼트를 제거한다.

        폴더 업로드의 상대 경로(예: 브라우저 webkitRelativePath)는 보존할
        가치가 있는 데이터지만, 그대로 파일시스템에 쓰면 공격 표면이다.
        정화된 경로만 매니페스트에도 남긴다 — "원문 보존"이 traversal 문자열
        보존을 뜻하면 매니페스트가 공격 문자열의 저장소가 된다.
        """
        raw = (filename or "").replace("\\", "/")
        parts = [p for p in raw.split("/") if p not in ("", ".", "..")]
        return "/".join(parts) if parts else "unnamed"

    def save_dataset(self, files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        """업로드 저장 — **폴더 경로를 문 앞에서 버리지 않는다.**

        종전(`Path(filename).name`)의 두 가지 손실:
          1. 폴더 계층 소실 — "제안서/본문.pdf" → "본문.pdf"
          2. **이름 충돌 시 조용히 덮어쓰기** — a/보고서.pdf 와 b/보고서.pdf 를
             같이 올리면 뒤가 앞을 지웠다 (데이터 유실인데 아무도 모른다)

        저장 이름은 상대 경로를 평탄화(`/` → `__`)해 한 디렉터리에 담고(기존
        분석·인제스트 경로 무변경), 원 상대 경로는 매니페스트에 남긴다.
        매니페스트는 데이터셋 디렉터리 **밖**에 둔다 — 안에 두면 list/analyze 가
        업로드 파일로 오인해 인제스트한다.
        """
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        dataset_dir = self.data_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        saved: List[str] = []
        manifest: Dict[str, str] = {}
        total_bytes = 0
        for filename, content in files:
            rel = self._safe_rel_path(filename)
            stored = rel.replace("/", "__")
            # 같은 상대 경로가 두 번 오면 접미로 가른다 — 덮어쓰기 금지.
            if stored in manifest:
                stem, dot, ext = stored.rpartition(".")
                counter = 2
                while stored in manifest:
                    stored = (f"{stem} ({counter}).{ext}" if dot
                              else f"{stored} ({counter})")
                    counter += 1
            (dataset_dir / stored).write_bytes(content)
            saved.append(stored)
            manifest[stored] = rel
            total_bytes += len(content)

        try:
            (self.data_dir / f"{dataset_id}.manifest.json").write_text(
                json.dumps({"dataset_id": dataset_id, "files": manifest},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            # 매니페스트는 부가 정보다 — 실패해도 업로드 자체는 성립한다.
            logger.warning(f"⚠️ 업로드 매니페스트 기록 실패 ({dataset_id}): {e}")

        logger.info(f"📦 Ontology dataset saved: {dataset_id} ({len(saved)} files)")
        return {"dataset_id": dataset_id, "files": saved,
                "paths": manifest, "total_bytes": total_bytes}

    def dataset_manifest(self, dataset_id: str) -> Dict[str, str]:
        """저장 이름 → 원 상대 경로. 옛 데이터셋(매니페스트 없음)은 빈 dict."""
        path = self.data_dir / f"{dataset_id}.manifest.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return dict(data.get("files") or {})
        except Exception as e:
            logger.warning(f"⚠️ 업로드 매니페스트 로드 실패 ({dataset_id}): {e}")
            return {}

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets = []
        for entry in sorted(self.data_dir.iterdir()):
            if entry.is_dir():
                files = [f.name for f in sorted(entry.iterdir()) if f.is_file()]
                datasets.append({"dataset_id": entry.name, "files": files})
        return datasets

    def dataset_path(self, dataset_id: str) -> Optional[Path]:
        path = self.data_dir / dataset_id
        return path if path.is_dir() else None

    # ─── Analyze (감식 — 빌드 전 확인 게이트의 근거) ─────────────────

    async def analyze_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """업로드된 데이터셋을 감식한다: 파일별 종(種) + 매핑 제안 + 비용 견적.

        빌드 전에 호출하는 것이 의도된 사용법이다 — 정형 레코드가 LLM 추출
        경로로 흘러 수천 콜을 낭비하는 것을 여기서 미리 보이게 한다.
        LLM 비용: 레코드 파일당 1콜(매핑) + 파일명 배치 1콜. 감식 자체는 결정적.
        """
        from dataclasses import asdict as _asdict

        from ..builder.sniffer import DatasetAnalyzer

        path = self.dataset_path(dataset_id)
        if path is None:
            raise FileNotFoundError(f"dataset not found: {dataset_id}")

        analyzer = DatasetAnalyzer(llm_fn=self._active_llm_fn())
        report = await analyzer.analyze(path)

        files = []
        plan = []
        for profile in report.files:
            entry = _asdict(profile)
            entry.pop("path", None)  # 서버 내부 경로는 응답에 노출하지 않는다
            files.append(entry)
            # plan 초안 — 사용자가 이걸 고쳐서 POST .../ingest 로 그대로 낸다.
            # analyze 가 제안까지 하고 제출물을 안 주면, 클라이언트가 응답을
            # 뒤져 조립해야 한다 (제안과 제출의 스키마 drift 가 생기는 지점).
            draft: Dict[str, Any] = {"filename": profile.filename,
                                     "route": profile.species}
            # 파일명 해석의 신뢰 등급(trust)을 초안에 싣는다 — 사용자가
            # 게이트에서 등급을 보고 고칠 수 있어야 한다 (요약서가 약관을
            # 덮지 못하게 하는 가드의 입력값이다).
            draft["trust"] = (profile.filename_meta or {}).get("trust") or ""
            if profile.species == "records":
                draft["records_path"] = profile.records_path
                draft["mapping"] = profile.mapping_proposal
                draft["hierarchy"] = profile.hierarchy_proposal
            elif profile.species == "seed_ontology":
                draft["records_path"] = profile.records_path
            plan.append(draft)
        return {"dataset_id": dataset_id, "files": files, "plan": plan,
                "total_estimated_llm_calls": report.total_estimated_llm_calls,
                "ingest_config": self._ingest_config()}

    def _ingest_config(self) -> Dict[str, Any]:
        """수집 전 미리보기 설정 — 임베딩 모델·청크·키워드검색(ES)·인덱스명.
        es/embedder 가용성은 cheap 하게 판정(ES ping · 패키지 존재)해 주입한다."""
        import importlib.util
        from ..core.ingest_config import build_ingest_config
        try:
            from ..core.object_index import ObjectIndex
            es_ok = ObjectIndex("_preview").available()
        except Exception:
            es_ok = False
        embedder_ok = importlib.util.find_spec("sentence_transformers") is not None
        return build_ingest_config(es_available=es_ok, embedder_available=embedder_ok)

    # ─── Ingest (확인 게이트 실행 — 승인된 plan 을 종별 라우팅) ──────

    async def run_ingest(self, job_id: str, dataset_id: str, namespace: str,
                         plan: List[Dict[str, Any]], save: bool = True,
                         schema_mode: str = "auto",
                         custom_schema: Optional[Dict] = None,
                         llm_model: Optional[str] = None,
                         llm_provider: str = "google",
                         llm_base_url: Optional[str] = None,
                         extraction_mode: str = "exhaustive",
                         topics: Optional[List[str]] = None) -> None:
        """승인된 plan 실행. 업로드 → analyze → 확인 → **여기**.

        plan 은 클라이언트가 보낸 것이므로 불신한다 — 파일 존재·라우트·매핑을
        재검증하고, 한 항목의 실패는 그 항목에만 기록한다 (빌더의 resilient
        원칙: 나머지 파일은 계속된다). 실행 자체는 결정적/기존 빌더 재사용 —
        승인한 것과 다른 것이 만들어지면 확인 게이트의 의미가 없다.
        """
        if namespace in PROTECTED_NAMESPACES:
            raise ValueError(
                f"namespace '{namespace}' is protected (agent-routing graph)")

        job = self.jobs[job_id]
        job["status"] = "running"
        try:
            from ..builder import OntologyBuilder
            from ..builder.ingestion import (import_seed, ingest_hierarchy,
                                             load_records_from_file)
            from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

            dataset_dir = self.dataset_path(dataset_id)
            if dataset_dir is None:
                raise FileNotFoundError(f"dataset not found: {dataset_id}")
            kg = get_knowledge_graph_engine(namespace)

            # 텍스트 경로용 빌더는 하나를 공유한다 — 스키마(auto 제안 포함)가
            # 파일마다 흔들리면 같은 데이터셋 안에서 타입이 갈라진다.
            text_builder = OntologyBuilder(
                schema=self._resolve_schema_for_ingest(
                    namespace, schema_mode, custom_schema),
                namespace=namespace, kg=kg, llm_fn=self.llm_fn,
                llm_model=llm_model, llm_provider=llm_provider,
                llm_base_url=llm_base_url, auto_save=False,
                extraction_mode=extraction_mode, topics=topics,
                progress_cb=lambda **e: job.update(progress=e))

            outcomes: List[Dict[str, Any]] = []
            for entry in plan:
                outcome = {"filename": entry.get("filename", ""),
                           "route": entry.get("route", ""),
                           "entities_added": 0, "relations_added": 0,
                           "error": ""}
                outcomes.append(outcome)
                try:
                    await self._ingest_one(entry, dataset_dir, kg,
                                           text_builder, outcome,
                                           import_seed, ingest_hierarchy,
                                           load_records_from_file)
                except Exception as e:
                    outcome["error"] = str(e)
                    logger.warning(f"⚠️ Ingest failed for "
                                   f"{outcome['filename']}: {e}")

            if save:
                kg.save_to_disk()
                if text_builder.chunk_store is not None:
                    text_builder.chunk_store.save_to_disk()

            self._mirror_to_pg(namespace)   # 축 5 P2 — best-effort dual-write
            job["progress"] = {"stage": "indexing"}
            idx = self.index_namespace(namespace)   # ES 노드 + 청크 벡터(업로드→검색가능)
            job["report"] = {"namespace": namespace, "files": outcomes,
                             "index": idx}
            # B2: expectation 게이트 — 경고만, 인제스트는 완료 (best-effort)
            job["report"]["coverage_gate"] = await self._coverage_gate_for_job(namespace)
            job["status"] = "completed"
            logger.info(f"✅ Ingest {job_id} completed: "
                        f"{len(outcomes)} plan entries")
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"❌ Ingest {job_id} failed: {e}")
            raise

    async def _ingest_one(self, entry, dataset_dir, kg, text_builder,
                          outcome, import_seed, ingest_hierarchy,
                          load_records_from_file) -> None:
        route = entry.get("route", "")
        if route == "skip":
            return

        filename = entry.get("filename", "")
        file_path = dataset_dir / Path(filename).name  # traversal 방지
        if not file_path.is_file():
            raise FileNotFoundError(f"file not in dataset: {filename}")

        trust = entry.get("trust") or ""

        if route == "records":
            mapping = entry.get("mapping")
            if not mapping or not mapping.get("name_field"):
                raise ValueError(
                    "records route requires a mapping with name_field "
                    "(정체성 없는 인제스트 금지)")
            records = load_records_from_file(
                file_path, entry.get("records_path", ""))
            report = await text_builder.build_from_records(
                records, mapping, source=filename, trust=trust)
            outcome["entities_added"] = report.entities_added
            outcome["relations_added"] = report.relations_added

            hierarchy = entry.get("hierarchy")
            if hierarchy:
                import json as _json
                with open(file_path, encoding="utf-8") as f:
                    data = _json.load(f)
                items = data.get(hierarchy.get("path", "")) \
                    if isinstance(data, dict) else None
                if items:
                    added = await ingest_hierarchy(kg, items, hierarchy,
                                                   source=filename)
                    outcome["entities_added"] += added["nodes"]
                    outcome["relations_added"] += added["edges"]

        elif route in ("articled", "prose"):
            from ..builder import readers
            text = readers.read_file(file_path)
            report = await text_builder.build_from_text(text, source=filename,
                                                        trust=trust)
            outcome["entities_added"] = report.entities_added
            outcome["relations_added"] = report.relations_added
            if report.chunks_failed:
                outcome["error"] = f"{report.chunks_failed} chunk(s) failed"

        elif route == "seed_ontology":
            import json as _json
            with open(file_path, encoding="utf-8") as f:
                data = _json.load(f)
            records_path = entry.get("records_path", "")
            if records_path and isinstance(data, dict):
                items = data.get(records_path) or []
            elif isinstance(data, list):
                items = data
            else:
                items = data.get("@graph") or [] if isinstance(data, dict) else []
            outcome["entities_added"] = await import_seed(
                kg, items, source=filename)

        else:
            raise ValueError(f"unknown route '{route}'")

    # ─── Build jobs ──────────────────────────────────────────────────

    def create_job(self, namespace: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "namespace": namespace,
            "status": "queued",
            "progress": {},
            "report": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> Dict[str, Any]:
        """작업 이력 — 인메모리 잡 레지스트리를 최근순 평탄 목록으로.

        관리 콘솔이 최근 인제스트/재색인 잡의 상태를 한눈에 본다. progress dict
        는 관측용으로 stage 만 평탄화하고, 무거운 report 본문은 싣지 않는다
        (목록이라 요약만 — 상세는 GET /jobs/{job_id}). 재시작하면 비는 것이
        정상이다: 잡은 프로세스 수명의 인메모리 신호다."""
        items = [
            {
                "job_id": j.get("job_id"),
                "namespace": j.get("namespace"),
                "status": j.get("status"),
                "stage": (j.get("progress") or {}).get("stage"),
                "error": j.get("error"),
                "created_at": j.get("created_at"),
            }
            for j in self.jobs.values()
        ]
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return {"jobs": items[:limit], "total": len(self.jobs)}

    def health_check(self) -> Dict[str, Any]:
        """연결 헬스 — PG/ES 핑 레이턴시(ms) + 임베딩 모델/차원.

        연결·상태 패널이 "무엇이 살아있고 얼마나 빠른가"를 본다. 각 백엔드는
        독립적으로 degrade 한다 — PG 가 죽어도 ES 핑은 나오고, 어느 하나의 부재가
        500 이 되지 않는다({available:false}). system_overview 보다 가벼운 프로브
        (SELECT 1 / _cluster/health 왕복 시간만)."""
        import time as _time

        from ..core import health_signals
        result: Dict[str, Any] = {"components": health_signals.snapshot(),
                                  "degraded": health_signals.degraded()}

        # PostgreSQL — SELECT 1 왕복
        try:
            from ..core import pg
            if pg.available():
                t0 = _time.perf_counter()
                with pg.connect() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    cur.fetchone()
                result["pg"] = {
                    "available": True,
                    "latency_ms": round((_time.perf_counter() - t0) * 1000, 1),
                    "schema": pg.get_schema(),
                }
            else:
                result["pg"] = {"available": False, "error": "PG unavailable"}
        except Exception as e:
            result["pg"] = {"available": False, "error": str(e)}

        # Elasticsearch — _cluster/health 왕복
        try:
            import json as _json
            import urllib.request
            try:
                from ..core.es_backend import DEFAULT_ES_URL
            except Exception:
                DEFAULT_ES_URL = os.environ.get(
                    "ONTOLOGY_ES_URL", "http://localhost:9200")
            es_url = os.environ.get("ONTOLOGY_ES_URL", DEFAULT_ES_URL).rstrip("/")
            t0 = _time.perf_counter()
            with urllib.request.urlopen(f"{es_url}/_cluster/health", timeout=4) as r:
                h = _json.loads(r.read().decode("utf-8"))
            result["es"] = {
                "available": True,
                "latency_ms": round((_time.perf_counter() - t0) * 1000, 1),
                "cluster_status": h.get("status"),
                "unassigned_shards": h.get("unassigned_shards"),
                "url": es_url,
            }
        except Exception as e:
            result["es"] = {"available": False, "error": str(e)}

        # 임베딩 모델/차원 (프로세스 로컬 — 프로브 아님)
        try:
            model, dim = self._embedding_model_dim()
            result["embedding"] = {"model": model, "dim": dim}
        except Exception as e:
            result["embedding"] = {"error": str(e)}

        return result

    def _resolve_schema(self, schema_mode: str, custom_schema: Optional[Dict]):
        from ..builder import BuilderSchema
        if schema_mode == "auto":
            return None  # LLM이 샘플을 보고 스키마 제안 (builder가 처리)
        if schema_mode == "custom":
            if not custom_schema:
                raise ValueError("schema_mode='custom' requires custom_schema")
            return BuilderSchema.from_dict(custom_schema)
        return BuilderSchema.preset(schema_mode)  # unknown name → ValueError

    def _resolve_schema_for_ingest(self, namespace: str, schema_mode: str,
                                   custom_schema: Optional[Dict]):
        """인제스트용 스키마 해소 — auto 인데 대상 네임스페이스에 이미 그래프가
        있으면 그 스키마를 **고정 재사용**(다-1, 재현성): 재인제스트마다 LLM 이
        타입을 새로 지어 node_id 가 어긋나는 것을 막는다. 신규 ns 는 그대로 auto."""
        from ..builder import BuilderSchema
        schema = self._resolve_schema(schema_mode, custom_schema)
        if schema is None and schema_mode == "auto":
            from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
            pinned = BuilderSchema.from_graph(
                get_knowledge_graph_engine(namespace).graph)
            if pinned is not None:
                logger.info(f"📌 기존 스키마 재사용(재현성): {namespace} — "
                            f"타입 {len(pinned.node_types)}·술어 {len(pinned.predicates)}")
                return pinned
        return schema

    def list_schema_presets(self) -> Dict[str, Any]:
        from ..builder.models import SCHEMA_PRESETS
        return {
            name: {"node_types": spec["node_types"],
                   "predicates": {k: list(v) for k, v in spec["predicates"].items()}}
            for name, spec in SCHEMA_PRESETS.items()
        }

    async def run_build(self, job_id: str, dataset_id: str, namespace: str,
                        schema_mode: str = "document",
                        custom_schema: Optional[Dict] = None,
                        chunk_size: int = 800, overlap: int = 120,
                        save: bool = True, segment_mode: str = "auto",
                        llm_model: Optional[str] = None,
                        llm_provider: str = "google",
                        llm_base_url: Optional[str] = None,
                        rebuild: bool = False) -> None:
        job = self.jobs[job_id]
        job["status"] = "running"
        try:
            from ..builder import OntologyBuilder
            # rebuild 는 처음부터 다시 → 스키마 고정 안 함(fresh). 아니면 기존
            # 스키마 재사용(다-1). 고정 판정은 clear 이전 그래프 기준.
            schema = (self._resolve_schema(schema_mode, custom_schema) if rebuild
                      else self._resolve_schema_for_ingest(
                          namespace, schema_mode, custom_schema))

            if rebuild:
                from ..core.chunk_index import reset_chunk_indices
                from ..core.chunk_store import get_chunk_store
                from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
                get_knowledge_graph_engine(namespace).clear()
                # 그래프를 비우면 원문도 비운다 — 안 그러면 삭제된 노드를
                # 가리키는 유령 청크가 남는다 (KG.clear 가 시맨틱 인덱스를
                # 함께 버리는 것과 같은 이유).
                get_chunk_store(namespace).clear()
                # 청크 인덱스도 버린다 — 죽은 청크를 가리키는 파생 데이터다.
                # (search 가 잔재를 걸러내긴 하지만, 남겨두면 top_k 를 유령이
                #  차지해 실제 히트가 밀려난다)
                reset_chunk_indices()

            def progress_cb(**event):
                job["progress"] = event

            builder = OntologyBuilder(
                schema=schema,
                namespace=namespace,
                llm_fn=self.llm_fn,
                progress_cb=progress_cb,
                chunk_size=chunk_size,
                overlap=overlap,
                segment_mode=segment_mode,
                llm_model=llm_model,
                llm_provider=llm_provider,
                llm_base_url=llm_base_url,
                auto_save=save,
            )
            report = await builder.build_from_folder(self.dataset_path(dataset_id))
            self._mirror_to_pg(namespace)   # 축 5 P2 — best-effort dual-write
            job["progress"] = {"stage": "indexing"}
            self.index_namespace(namespace)   # ES 노드 + 청크 벡터(업로드→검색가능)
            job["report"] = asdict(report)
            # B2: expectation 게이트 — 경고만, 빌드는 완료 (best-effort)
            job["report"]["coverage_gate"] = await self._coverage_gate_for_job(namespace)
            job["status"] = "completed"
            logger.info(f"✅ Ontology build {job_id} completed: {job['report']}")
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"❌ Ontology build {job_id} failed: {e}")

    # ─── Graph queries ───────────────────────────────────────────────

    def get_graph_summary(self, namespace: str, limit: int = 10) -> Dict[str, Any]:
        from collections import Counter
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        graph = get_knowledge_graph_engine(namespace).graph
        node_types = Counter(
            attrs.get("type", "unknown") for _, attrs in graph.nodes(data=True))
        predicates = Counter(
            attrs.get("predicate", "unknown") for _, _, attrs in graph.edges(data=True))

        sample_nodes = [
            {"id": node_id, "type": attrs.get("type", ""),
             "name": attrs.get("name", node_id), "source": attrs.get("source", "")}
            for node_id, attrs in list(graph.nodes(data=True))[:limit]
        ]
        return {
            "namespace": namespace,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "node_types": dict(node_types),
            "predicates": dict(predicates),
            "sample_nodes": sample_nodes,
        }

    def get_graph_data(self, namespace: str, limit: int = 300) -> Dict[str, Any]:
        """Full nodes+links payload for frontend visualization."""
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        graph = get_knowledge_graph_engine(namespace).graph

        node_ids = list(graph.nodes())[:limit]
        allowed = set(node_ids)
        nodes = [
            {"id": nid, "type": graph.nodes[nid].get("type", ""),
             "name": graph.nodes[nid].get("name", nid),
             "source": graph.nodes[nid].get("source", "")}
            for nid in node_ids
        ]
        links = [
            {"source": s, "target": t, "predicate": attrs.get("predicate", "")}
            for s, t, attrs in graph.edges(data=True)
            if s in allowed and t in allowed
        ]
        return {"namespace": namespace, "nodes": nodes, "links": links}

    def list_namespaces(self) -> List[Dict[str, Any]]:
        """디스크 체크포인트 + 현재 로드된 인스턴스의 네임스페이스 목록."""
        from ..engines.knowledge_graph_clean import _DEFAULT_DATA_DIR, _kg_instances

        names = set(_kg_instances.keys())
        for checkpoint in _DEFAULT_DATA_DIR.glob("kg_*.json"):
            stem = checkpoint.stem  # kg_checkpoint → default, kg_{ns} → ns
            if ".backup" in stem or stem.startswith("kg_testns_"):
                continue  # 백업본·테스트 잔여물은 목록에서 제외
            names.add("default" if stem == "kg_checkpoint" else stem[3:])

        result = []
        for name in sorted(names):
            entry: Dict[str, Any] = {"namespace": name}
            if name in _kg_instances:
                entry["nodes"] = _kg_instances[name].graph.number_of_nodes()
                entry["edges"] = _kg_instances[name].graph.number_of_edges()
            result.append(entry)
        return result

    def get_node_detail(self, namespace: str, node_id: str) -> Optional[Dict[str, Any]]:
        """노드 전체 속성 + 인/아웃 관계 (그래프 탐색기 상세 패널용).

        GraphStore seam 위임(축 5 P4-b): PG 백엔드는 PK 조회 + 인덱스 엣지
        조회라, 노드 하나 열자고 전체 그래프를 로드하던 경로가 사라진다."""
        return self._graph_store(namespace).node_detail(namespace, node_id)

    def export_graph(self, namespace: str, format: str = "turtle"):
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        graph = get_knowledge_graph_engine(namespace).graph
        if format == "turtle":
            from ..builder.export import to_turtle
            return to_turtle(graph, namespace=namespace)
        return self.get_graph_data(namespace, limit=2000)

    @staticmethod
    def _as_number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_map_data(self, namespace: str) -> Dict[str, Any]:
        """데이터맵: 위치(lat/lng) 포인트 + 분류(category/region) 그룹."""
        from collections import defaultdict
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        graph = get_knowledge_graph_engine(namespace).graph

        geo: List[Dict[str, Any]] = []
        category_groups: Dict[str, List[str]] = defaultdict(list)
        for node_id, attrs in graph.nodes(data=True):
            name = attrs.get("name", node_id)
            lat = self._as_number(attrs.get("lat") or attrs.get("latitude") or attrs.get("위도"))
            lng = self._as_number(attrs.get("lng") or attrs.get("lon")
                                  or attrs.get("longitude") or attrs.get("경도"))
            category = attrs.get("category") or attrs.get("분류")
            region = attrs.get("region") or attrs.get("지역")
            if lat is not None and lng is not None:
                geo.append({"id": node_id, "name": name,
                            "type": attrs.get("type", ""), "lat": lat, "lng": lng,
                            "category": category, "region": region})
            group_key = category or region
            if group_key:
                category_groups[str(group_key)].append(name)

        categories = {
            key: {"count": len(items), "items": items[:50]}
            for key, items in sorted(category_groups.items(),
                                     key=lambda kv: -len(kv[1]))
        }
        return {"namespace": namespace, "geo": geo, "categories": categories}

    def get_hierarchy_rollup(self, namespace: str, class_id: str,
                             limit: int = 50) -> Optional[Dict[str, Any]]:
        """계층 롤업: 클래스의 상위 사슬 + 하위 분류 + (하위 포함) 소속 인스턴스.

        온톨로지 추론(is_a 전이 폐포)의 실사용 — "석탑에 해당하는 유산 전체"처럼
        명시적으로 저장되지 않은 소속을 계층을 타고 집계한다.
        """
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        if class_id not in graph:
            return None

        ancestors = engine.get_ancestors(class_id)          # is_a 상위 사슬
        descendants = engine.get_descendants(class_id)      # is_a 하위 폐포

        def instances_of(cid):
            return [(s, graph.nodes[s]) for s, _, a in graph.in_edges(cid, data=True)
                    if a.get("predicate") == "classifiedAs"]

        direct = instances_of(class_id)
        rolled = list(direct)
        seen = {s for s, _ in direct}
        for descendant in descendants:
            for s, attrs in instances_of(descendant):
                if s not in seen:
                    seen.add(s)
                    rolled.append((s, attrs))

        return {
            "class_id": class_id,
            "name": graph.nodes[class_id].get("name", class_id),
            "ancestors": ancestors,
            "descendants": descendants,
            "instances_direct": len(direct),
            "instances_total": len(rolled),
            "instances": [{"id": s, "name": a.get("name", s),
                           "type": a.get("type", "")}
                          for s, a in rolled[:limit]],
        }

    DATASET_FORMATS = ("triples", "qa", "surface", "evidence")

    _COHORT_KEYS = ("node_type", "trust", "prop_key", "prop_value", "prop_op",
                    "rel_predicate", "rel_target", "rel_target_type", "rel_direction")

    def _resolve_cohort_ids(self, namespace: str, graph,
                            cohort: Dict[str, Any]) -> set:
        """구조화 쿼리(cohort)를 추출 대상 node_id 집합으로 해석 — 추출이 읽는
        그 KG 그래프에서 InMemory 질의 로직으로 **전부** 모은다(페이지네이션 없이).
        추출과 코호트가 동일 그래프를 보므로 일관적이다."""
        from ..core.graph_store import InMemoryGraphStore
        params = {k: cohort[k] for k in self._COHORT_KEYS if cohort.get(k) not in (None, "")}
        store = InMemoryGraphStore(graph, count_cap=10 ** 9)
        res = store.query_nodes(namespace, limit=10 ** 9, **params)
        return {i["node_id"] for i in res["items"]}

    def build_training_dataset(self, namespace: str,
                               formats: List[str],
                               node_types: Optional[List[str]] = None,
                               predicates: Optional[List[str]] = None,
                               limit: int = 5000,
                               include_evidence: bool = False,
                               cohort: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """온톨로지 → AI 학습 데이터 추출 (koract multi-condition 이식).

        전부 결정적으로 생성 — LLM 무사용, 모든 행에 출처(source). 환각 0.
        - triples: KG 임베딩/링크예측 학습용 관계 트리플
        - qa: 정의·관계 기반 instruction 페어 (LLM 파인튜닝용)
        - surface: 동의어→정규명 정규화 페어 (표면형 학습용)
        - evidence: **원문 → 개체** 추출 지도학습쌍 (축 2). 입력도 출력도 실제
          데이터라 양쪽 모두 환각이 없다. 축 2 이전에는 원문을 버려서 만들 수
          없었고, 그래서 qa 가 그래프를 되읽는 수준에 머물렀다.

        include_evidence=True 면 관계 qa 행에 근거 원문(evidence/chunk_id)을
        붙인다. 기본 False — 기존 행 모양을 바꾸지 않기 위한 옵트인.
        """
        from ..core.chunk_store import get_chunk_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        graph = get_knowledge_graph_engine(namespace).graph
        chunks = get_chunk_store(namespace)

        unknown = [f for f in formats if f not in self.DATASET_FORMATS]
        if unknown:
            raise ValueError(f"unknown dataset format(s): {', '.join(unknown)} "
                             f"(available: {', '.join(self.DATASET_FORMATS)})")

        # 코호트(구조화 쿼리) — 주어지면 추출을 그 node_id 집합으로 제한한다.
        cohort_ids = (self._resolve_cohort_ids(namespace, graph, cohort)
                      if cohort else None)

        def type_ok(node_id):
            if cohort_ids is not None and node_id not in cohort_ids:
                return False
            if not node_types:
                return True
            return graph.nodes[node_id].get("type") in node_types

        rows: List[Dict[str, Any]] = []

        def first_chunk(node_id):
            """이 노드가 추출된 첫 원문 청크 (없으면 None)."""
            found = chunks.chunks_for_node(node_id)
            return found[0] if found else None

        if "triples" in formats or "qa" in formats:
            for subj, obj, attrs in graph.edges(data=True):
                predicate = attrs.get("predicate", "")
                if predicates and predicate not in predicates:
                    continue
                if not type_ok(subj):
                    continue
                subj_attrs, obj_attrs = graph.nodes[subj], graph.nodes[obj]
                subj_name = subj_attrs.get("name", subj)
                obj_name = obj_attrs.get("name", obj)
                source = attrs.get("source") or subj_attrs.get("source", "")
                if "triples" in formats:
                    rows.append({"format": "triple",
                                 "subject": subj_name, "predicate": predicate,
                                 "object": obj_name,
                                 "subject_type": subj_attrs.get("type", ""),
                                 "object_type": obj_attrs.get("type", ""),
                                 "source": source})
                if "qa" in formats:
                    row = {"format": "qa",
                           "instruction": f"{subj_name}의 {predicate} 관계에 있는 대상은 무엇인가?",
                           "output": obj_name, "source": source}
                    if include_evidence:
                        chunk = first_chunk(subj)
                        if chunk:
                            row["evidence"] = chunk.text
                            row["chunk_id"] = chunk.chunk_id
                    rows.append(row)

        for node_id, attrs in graph.nodes(data=True):
            if not type_ok(node_id):
                continue
            name = attrs.get("name", node_id)
            source = attrs.get("source", "")
            if "qa" in formats and attrs.get("definition"):
                rows.append({"format": "qa",
                             "instruction": f"{name}에 대해 한 문장으로 설명하세요.",
                             "output": str(attrs["definition"]), "source": source})
            if "surface" in formats and isinstance(attrs.get("aliases"), list):
                for alias in attrs["aliases"]:
                    if alias:
                        rows.append({"format": "surface",
                                     "input": str(alias), "output": name,
                                     "source": source})

        if "evidence" in formats:
            # 원문 → 개체 추출쌍. 청크에서 나온 노드들 중 타입 필터를 통과한
            # 것만 정답이 된다.
            for stored in chunks.all():
                entities = []
                for node_id in stored.node_ids:
                    if node_id not in graph or not type_ok(node_id):
                        continue
                    attrs = graph.nodes[node_id]
                    entities.append({"name": attrs.get("name", node_id),
                                     "type": attrs.get("type", "")})
                if not entities:
                    # 개체가 없는 청크는 학습쌍이 아니다 — 빈 정답을 가르치면
                    # 모델이 "아무것도 없다"를 배운다
                    continue
                rows.append({"format": "evidence",
                             "instruction": "다음 원문에서 지식 그래프 개체를 추출하세요.",
                             "input": stored.text, "output": entities,
                             "source": stored.source,
                             "chunk_id": stored.chunk_id,
                             "char_start": stored.char_start,
                             "char_end": stored.char_end,
                             # 청크의 신뢰 등급 — 큐레이터의 exclude_trust
                             # (요약서 행 배제)가 이 필드를 본다
                             "trust": stored.trust})

        rows = rows[:limit]
        # 요청된 포맷은 0건이어도 counts에 나타난다 (필터 결과 확인용)
        row_format = {"triples": "triple", "qa": "qa", "surface": "surface",
                      "evidence": "evidence"}
        counts: Dict[str, int] = {row_format[f]: 0 for f in formats}
        for row in rows:
            counts[row["format"]] = counts.get(row["format"], 0) + 1
        return {"namespace": namespace, "rows": rows, "counts": counts,
                "total": len(rows),
                "cohort_size": (len(cohort_ids) if cohort_ids is not None else None)}

    async def build_curated_dataset(self, namespace: str,
                                    formats: List[str],
                                    node_types: Optional[List[str]] = None,
                                    predicates: Optional[List[str]] = None,
                                    limit: int = 5000,
                                    include_evidence: bool = False,
                                    curate: Optional[Dict[str, Any]] = None,
                                    cohort: Optional[Dict[str, Any]] = None
                                    ) -> Dict[str, Any]:
        """추출 + 자동 품질 게이트 (⑥ 데이터셋 큐레이터).

        aicoach 가 curated=true 수동 게이트로 하던 일을 자동화한다:
        1. 결정적 큐레이션 (LLM 0콜) — 중복/짧은 원문/문서 지배/trust 배제
        2. 선택적 LLM 품질 게이트 (curate["llm_quality"]=true 일 때만)

        응답은 build_training_dataset 과 같은 모양 + "curation" 리포트,
        counts/total 은 남은 행 기준으로 재계산 — 'no silent caps' 원칙:
        무엇이 왜 떨어졌는지 응답에서 읽을 수 있어야 한다.
        """
        from ..core.dataset_curator import QualityScorer, curate_rows

        curate = curate or {}
        base = self.build_training_dataset(
            namespace, formats=formats, node_types=node_types,
            predicates=predicates, limit=limit,
            include_evidence=include_evidence, cohort=cohort)

        kept, report = curate_rows(
            base["rows"],
            dedup=bool(curate.get("dedup", True)),
            min_input_chars=int(curate.get("min_input_chars", 0)),
            max_per_source=curate.get("max_per_source"),
            exclude_trust=curate.get("exclude_trust"))

        if curate.get("llm_quality"):
            scorer = QualityScorer(llm_fn=self._active_llm_fn())
            kept, addendum = await scorer.score_rows(
                kept, threshold=int(curate.get("quality_threshold", 3)))
            # low_quality 는 탈락 사유, quality_skipped 는 탈락이 아니라
            # '판정 못 하고 보존한' 행 수 — dropped 에 섞지 않는다
            report["dropped"]["low_quality"] = addendum["low_quality"]
            report["quality_skipped"] = addendum["quality_skipped"]
            report["kept"] = len(kept)

        row_format = {"triples": "triple", "qa": "qa", "surface": "surface",
                      "evidence": "evidence"}
        counts: Dict[str, int] = {row_format[f]: 0 for f in formats}
        for row in kept:
            counts[row["format"]] = counts.get(row["format"], 0) + 1
        return {**base, "rows": kept, "counts": counts, "total": len(kept),
                "curation": report}

    # ─── 검색 QA (⑤ — 골든셋: 검색 품질을 숫자로) ────────────────────

    def get_golden_cases(self, namespace: str) -> Dict[str, Any]:
        from ..core.search_qa import get_golden_set
        golden = get_golden_set(namespace)
        from dataclasses import asdict as _asdict
        return {"namespace": namespace,
                "cases": [_asdict(c) for c in golden.cases()],
                "total": len(golden)}

    def add_golden_case(self, namespace: str, query: str,
                        expected_node_id: str,
                        status: str = "confirmed",
                        tags: Optional[List[str]] = None,
                        accepted: Optional[List[str]] = None,
                        expected_chunk_id: str = "",
                        accepted_chunks: Optional[List[str]] = None) -> Optional[str]:
        """수동 케이스 추가 — 사람이 직접 쓰는 케이스는 곧바로 confirmed.
        tags 는 시나리오 축(exact/semantic/graph × 의도) — eval 태그별 분해용.
        accepted 는 추가 정답(relevant set) — 동의어/교차연결 개념 credit.
        expected_chunk_id/accepted_chunks 는 청크 단위 채점용(선택) — 노드 정답과
        별개 축이라 한쪽만 있어도 된다."""
        from ..core.search_qa import get_golden_set
        return get_golden_set(namespace).add(
            query, expected_node_id, status=status, source="hand",
            tags=tags, accepted=accepted,
            expected_chunk_id=expected_chunk_id, accepted_chunks=accepted_chunks)

    def confirm_golden_case(self, namespace: str, case_id: str) -> bool:
        from ..core.search_qa import get_golden_set
        return get_golden_set(namespace).confirm(case_id)

    def accept_golden_answer(self, namespace: str, case_id: str,
                             node_id: str) -> Dict[str, Any]:
        """정답 집합 확장 — 라벨 노후화 처방 (코퍼스가 자라면 정답도 자란다).

        노드는 **실존을 확인한다** — 허공을 가리키는 accepted 는 지표를 조용히
        관대하게 만들 수 없지만(어차피 랭킹에 안 나온다) 오타를 영구 보존한다.
        """
        from ..core.search_qa import get_golden_set
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        node_id = (node_id or "").strip()
        if node_id not in get_knowledge_graph_engine(namespace).graph:
            return {"error": "node_not_found", "node_id": node_id}
        if not get_golden_set(namespace).accept(case_id, node_id):
            return {"error": "not_accepted",
                    "detail": "케이스가 없거나 이미 정답 집합에 있다"}
        return {"namespace": namespace, "case_id": case_id,
                "accepted": node_id}

    async def generate_golden_cases(self, namespace: str, limit: int = 10,
                                    per_node: int = 2) -> Dict[str, Any]:
        """LLM 초안 생성 — definition 이 있는 노드부터 (패러프레이즈의 재료가
        있어야 이름을 안 쓰고 물을 수 있다). 노드당 LLM 1콜.
        전부 draft — 확정은 인간이 한다.

        **노드를 청크에 퍼지게 고른다.** 종전에는 삽입 순서 상위 N개를 잘랐고,
        그 결과 PROJ-A 에서 31 케이스가 상위 25 노드에만 몰려 커버리지를
        18.5%→25.2% 로 올려도 **지표가 소수점까지 불변**이었다 — 개선이 없었던
        게 아니라 **자가 그 영역을 안 봤다**. `core/golden_sampling` 참고.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.golden_sampling import sample_nodes_for_generation
        from ..core.search_qa import QAGenerator, get_golden_set
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        graph = get_knowledge_graph_engine(namespace).graph
        node_views = sample_nodes_for_generation(
            graph, get_chunk_store(namespace).all(), limit=limit)

        generator = QAGenerator(llm_fn=self._active_llm_fn())
        golden = get_golden_set(namespace)
        added = await generator.generate_for_nodes(golden, node_views,
                                                   per_node=per_node)
        return {"namespace": namespace, "nodes_used": len(node_views),
                "drafts_added": added, "total": len(golden)}

    async def verify_golden_cases(self, namespace: str,
                                  limit: int = 0) -> Dict[str, Any]:
        """초안을 왕복 검증해 verified 로 승격한다 (케이스당 LLM 1콜).

        후보는 **의미 이웃**으로 만든다 — 무작위 후보면 과제가 너무 쉬워 통과율이
        무의미해진다. 검색기의 *순위*를 쓰는 게 아니라 후보 *집합*만 임베더에서
        뽑고 판정은 LLM 이 노드 목록을 보고 하므로 검색 지표에 대해 순환이 아니다.
        정답 노드는 이웃에 없더라도 항상 넣는다(없으면 통과가 불가능하다).
        """
        from ..core.search_qa import QAGenerator, get_golden_set
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph

        def _view(node_id: str) -> Dict[str, Any]:
            attrs = graph.nodes[node_id] if node_id in graph else {}
            return {"node_id": node_id,
                    "name": attrs.get("name", ""),
                    "definition": attrs.get("definition")
                    or attrs.get("description") or ""}

        def candidate_fn(query: str, expected: str) -> List[Dict[str, Any]]:
            ids: List[str] = []
            try:
                ids = [h["node_id"]
                       for h in engine.semantic_search(query, top_k=10)]
            except Exception:
                ids = []
            if expected in graph and expected not in ids:
                ids.append(expected)
            return [_view(nid) for nid in ids if nid in graph]

        golden = get_golden_set(namespace)
        result = await QAGenerator(llm_fn=self._active_llm_fn()).verify_drafts(
            golden, candidate_fn, limit=limit)
        return {"namespace": namespace, **result, "total": len(golden)}

    def evaluate_golden_set(self, namespace: str, k: int = 5,
                            include_drafts: bool = False,
                            target: str = "node",
                            statuses: Optional[List[str]] = None) -> Dict[str, Any]:
        """골든셋 평가 — 결정적, LLM 0콜 (임베딩 추론만).

        evaluate_cases 는 채널을 함수로 받는 범용 구조라, entry_ratio 같은 상수
        스윕은 설정만 다른 검색기 두 개를 채널로 넣어 나란히 재면 된다
        (tests 의 two_channels 케이스가 그 계약).

        target 이 채점 위치를 고르고, 위치에 맞는 채널을 짝지어 준다:
          · node     — semantic(노드 의미검색) vs retrieve(entry-우선 노드 랭킹).
                       이 둘은 구조상 파생 관계라 지표가 같게 나온다(알려진 한계).
          · chunk    — chunk(원 질의로 청크 인덱스만) vs retrieve(확장 질의 +
                       그래프 채널 + RRF). **그래프 조건화의 효과가 드러나는
                       자리**지만 골든셋에 청크 라벨이 없으면 0 케이스다.
          · evidence — 같은 청크 랭킹을 **기존 노드 라벨**로 채점한다. 새 라벨 0개로
                       청크 단위 측정이 성립한다 (search_qa 의 계약 참고).

        chunk/evidence 는 **세 채널**을 세운다 — 확산 청크 채널(PPR)의 값을
        같은 자로 비교하기 위해서다. 지금까지 그 채널은 노드 자로만 재여
        (0.9375→0.8750) 실제 청크 회수 능력이 보이지 않았다.
        """
        from ..core.search_qa import (chunk_hits_to_ids, evaluate_cases,
                                       get_golden_set, make_chunk_expander,
                                       retrieve_result_to_nodes)
        from ..core.retrieval_config import get_retrieval_config
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        # 라이브와 같은 설정 — 측정이 라이브를 설명해야 한다.
        live_cfg = get_retrieval_config(namespace).effective()
        expand_fn = None
        if target in ("chunk", "evidence"):
            # 싱글턴 헬퍼를 쓴다 — ChunkIndex(store, ...) 는 첫 인자가 ChunkStore 라
            # 네임스페이스 문자열을 넘기면 self.store 가 str 이 되어 refresh 에서
            # 죽는다(실제로 밟았다). search_chunks 도 같은 헬퍼를 쓴다.
            from ..core.chunk_index import get_chunk_index
            from ..core.graph_retrieval import GraphConditionedRetriever

            index = get_chunk_index(namespace)
            retriever = GraphConditionedRetriever(namespace=namespace)

            def chunk_rank(query: str, top_k: int) -> List[str]:
                # baseline — 온톨로지 확장 없이 원 질의로만 청크를 찾는다.
                return chunk_hits_to_ids(index.search(query, top_k=top_k))

            def retrieve_chunk_rank(query: str, top_k: int) -> List[str]:
                # **라이브와 같은 설정으로 잰다.** 그러지 않으면 지표가 라이브를
                # 설명하지 못한다 (이 세션에서 가장 많이 데인 부류).
                return chunk_hits_to_ids(
                    retriever.search(query, top_k=top_k, **live_cfg))

            def retrieve_prop_chunk_rank(query: str, top_k: int) -> List[str]:
                # 확산을 **강제로** 켠 비교 채널 — 설정만 다른 두 채널을 나란히
                # 재는 것이 evaluate_cases 의 설계 목적이다. 네임스페이스 설정이
                # 이미 확산 on 이면 두 채널이 같아진다(그게 정상이고, 화면에서
                # "설정이 이미 그 값"임을 보여준다).
                return chunk_hits_to_ids(
                    retriever.search(query, top_k=top_k,
                                     **{**live_cfg, "use_propagation": True,
                                        "propagation_channel": True}))

            channels = {"chunk": chunk_rank, "retrieve": retrieve_chunk_rank,
                        "retrieve+prop": retrieve_prop_chunk_rank}
            if target == "evidence":
                # 청크 랭킹을 노드 라벨로 채점 — 확장 함수가 그 다리다.
                # retriever.store 를 쓴다: 채널이 검색한 그 저장소로 채점해야
                # 근거 링크가 갈리지 않는다 (둘 다 같은 싱글턴이지만 명시적으로).
                expand_fn = make_chunk_expander(retriever.store)
        else:
            engine = get_knowledge_graph_engine(namespace)

            def semantic_rank(query: str, top_k: int) -> List[str]:
                return [hit["node_id"]
                        for hit in engine.semantic_search(query, top_k=top_k)]

            def retrieve_rank(query: str, top_k: int) -> List[str]:
                # 그래프-조건부 /retrieve → 노드 랭킹(entry-node 우선). 청크만 펴면
                # 정답 노드가 뒤로 밀려 hit@1=0 이 됐다 — 노드 단위 신호를 우선한다.
                res = self.retrieve(namespace, query, top_k=top_k)
                return retrieve_result_to_nodes(res)

            channels = {"semantic": semantic_rank, "retrieve": retrieve_rank}

        result = evaluate_cases(get_golden_set(namespace).cases(), channels,
                                k=k, include_drafts=include_drafts,
                                target=target, expand_fn=expand_fn,
                                statuses=set(statuses) if statuses else None)
        # 설정 지문과 함께 남긴다 — 점수만 남기면 "그때 그 숫자가 어떤 설정에서
        # 나온 것인가"를 잃고 비교가 불가능해진다 (실제로 두 번 데였다).
        from ..core.eval_history import get_eval_history
        get_eval_history(namespace).record(result, actor="evaluate")
        return {"namespace": namespace, **result}

    # ─── 실험 하네스 (Phase 2 — Tier 0 러너) ─────────────────────────

    _EXPERIMENT_CHANNELS = ("vector", "graph", "graph+prop")

    def run_retrieval_experiments(self, namespace: str,
                                  axes: Optional[Dict[str, List[Any]]] = None,
                                  k: int = 5, target: str = "evidence",
                                  statuses: Optional[List[str]] = None,
                                  max_combos: int = 64,
                                  actor: str = "manual") -> Dict[str, Any]:
        """Tier 0 실험 러너 — 조합 × 골든셋 → 품질 + 비용 레코드 (LLM 0콜).

        규율 (docs/experiment-harness-architecture.html):
        - 자는 기존 evaluate_cases 그대로, 채널은 라이브와 같은 부품.
        - **eval_history 를 우회한다** — 실험 레코드가 운영 품질 블록
          (latest_quality → /retrieve.quality)을 오염시키면 안 된다.
        - 레코드에 지문 3종(설정·그래프 상태·골든셋) + 표본 경고 필수.
        - 비용: 지연 p50/p95(워밍업 1질의 제외) + embed 콜. 확장 노드 수는
          타이밍 **밖**에서 표본 질의로 따로 센다 (지연 오염 방지).
          pool_chunks 는 search 내부라 재지 않는다 — 지어내지 않는다.
        """
        import statistics
        import time as _time
        from datetime import datetime as _dt

        from ..core.chunk_index import get_chunk_index
        from ..core.chunk_store import get_chunk_store
        from ..core.eval_history import config_fingerprint
        from ..core.experiment import (enumerate_combos, get_experiment_store,
                                       golden_fingerprint, graph_fingerprint,
                                       pareto_frontier, sample_warnings)
        from ..core.graph_retrieval import GraphConditionedRetriever
        from ..core.retrieval_config import (KEYS as _CFG_KEYS,
                                             get_retrieval_config,
                                             validate_overrides)
        from ..core.search_qa import (chunk_hits_to_ids, evaluate_cases,
                                      get_golden_set, make_chunk_expander)
        from ..core.semantic_index import count_embeds
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if target not in ("evidence", "chunk"):
            return {"error": "invalid",
                    "detail": f"target {target!r} 미지원 — evidence|chunk "
                              "(node 채점은 /qa/evaluate 로)"}

        axes = axes or {"channel": list(self._EXPERIMENT_CHANNELS)}
        # 축 검증 — 소리내는 거부 (오타 키가 조용히 무시되면 그 축을 돌았다고
        # 오독한다). channel/k 는 특수 축, 나머지는 retrieval_config _SPEC.
        for key, values in axes.items():
            if not isinstance(values, list) or not values:
                return {"error": "invalid", "detail": f"axis {key!r}: 비어있지 "
                        "않은 리스트여야 한다"}
            if key == "channel":
                bad = [v for v in values if v not in self._EXPERIMENT_CHANNELS]
                if bad:
                    return {"error": "invalid",
                            "detail": f"channel 값 {bad} — 허용: "
                                      f"{list(self._EXPERIMENT_CHANNELS)}"}
            elif key == "k":
                if not all(isinstance(v, int) and 1 <= v <= 50 for v in values):
                    return {"error": "invalid", "detail": "k 축은 1..50 정수"}
            elif key in _CFG_KEYS:
                for v in values:
                    ok, why = validate_overrides({key: v})
                    if not ok:
                        return {"error": "invalid", "detail": why}
            else:
                return {"error": "invalid",
                        "detail": f"알 수 없는 축 {key!r} — 허용: channel, k, "
                                  f"{sorted(_CFG_KEYS)}"}
        try:
            combos = enumerate_combos(axes, max_combos=max_combos)
        except ValueError as e:
            return {"error": "invalid", "detail": str(e)}

        engine = get_knowledge_graph_engine(namespace)
        store = get_chunk_store(namespace)
        index = get_chunk_index(namespace)
        retriever = GraphConditionedRetriever(namespace=namespace)
        live_cfg = get_retrieval_config(namespace).effective()
        cases = get_golden_set(namespace).cases()
        expand_fn = make_chunk_expander(retriever.store)
        gfp = graph_fingerprint(engine.graph, store.all())
        golden_fp = golden_fingerprint(cases)
        base_fp = config_fingerprint(namespace)

        run_id = f"exp-{_dt.now():%Y%m%d%H%M%S}"
        exp_store = get_experiment_store(namespace)
        records: List[Dict[str, Any]] = []

        # expand() 가 받는 키만 (propagation_weight/channel 은 search 측)
        _EXPAND_KEYS = ("entry_k", "min_entry_score", "entry_ratio",
                        "max_terms", "use_propagation", "propagation_top")

        for combo in combos:
            channel = combo.get("channel", "graph")
            combo_k = int(combo.get("k", k))
            overrides = {kk: vv for kk, vv in combo.items()
                         if kk not in ("channel", "k")}
            cfg = {**live_cfg, **overrides}
            if channel == "graph":
                cfg = {**cfg, "use_propagation": False,
                       "propagation_channel": False}
            elif channel == "graph+prop":
                cfg = {**cfg, "use_propagation": True,
                       "propagation_channel": True}

            latencies: List[float] = []

            if channel == "vector":
                def rank_fn(query: str, top_k: int) -> List[str]:
                    t0 = _time.perf_counter()
                    hits = index.search(query, top_k=top_k)
                    latencies.append((_time.perf_counter() - t0) * 1000)
                    return chunk_hits_to_ids(hits)
            else:
                def rank_fn(query: str, top_k: int,
                            _cfg=cfg) -> List[str]:
                    t0 = _time.perf_counter()
                    hits = retriever.search(query, top_k=top_k, **_cfg)
                    latencies.append((_time.perf_counter() - t0) * 1000)
                    return chunk_hits_to_ids(hits)

            with count_embeds() as embed_calls:
                result = evaluate_cases(
                    cases, {"exp": rank_fn}, k=combo_k, target=target,
                    expand_fn=expand_fn,
                    statuses=set(statuses) if statuses else None)

            # 워밍업 1질의 제외 — 첫 질의가 모델 로드를 짊어진다 (실측 8.5s)
            body = latencies[1:] if len(latencies) > 1 else latencies
            cost: Dict[str, Any] = {
                "queries": len(latencies),
                "latency_ms_p50": round(statistics.median(body), 2) if body else None,
                "latency_ms_p95": round(
                    sorted(body)[max(0, int(len(body) * 0.95) - 1)], 2) if body else None,
                "embed_calls": len(embed_calls),
                "llm_calls": 0,
            }
            # 확장 노드 수 — 타이밍 밖, 표본 질의 최대 10개 (비용 정직: 표본수 명시)
            if channel != "vector":
                sample_qs = [c.query for c in cases][:10]
                sizes, prop_sizes = [], []
                for q in sample_qs:
                    try:
                        exp = retriever.expand(q, **{kk: cfg[kk]
                                                     for kk in _EXPAND_KEYS
                                                     if kk in cfg})
                        entry = list(getattr(exp, "entry_nodes", []) or [])
                        expanded = list(getattr(exp, "expanded_nodes", []) or [])
                        sizes.append(len(entry) + len(expanded))
                        prop_sizes.append(sum(
                            1 for n in expanded
                            if "propagation" in str(
                                getattr(exp, "via", {}).get(n, ""))))
                    except Exception:
                        continue
                if sizes:
                    cost["expanded_nodes_avg"] = round(
                        sum(sizes) / len(sizes), 1)
                    cost["expansion_sampled"] = len(sizes)

            metrics = (result.get("channels") or {}).get("exp") or {}
            entry = {
                "run_id": run_id, "namespace": namespace,
                "layer": "retrieval", "actor": actor,
                "axes": dict(combo),
                "config": {**base_fp, **overrides, "k": combo_k,
                           "channel": channel},
                "graph": gfp,
                "golden": {**golden_fp, "target": target,
                           "cases": result.get("cases"),
                           "skipped": result.get("skipped"),
                           "statuses": sorted(statuses) if statuses
                           else ["confirmed"]},
                "metrics": {**metrics, "measured": result.get("measured")},
                "cost": cost,
                "warnings": sample_warnings(int(result.get("cases") or 0)),
            }
            exp_store.record(entry)
            records.append(entry)

        return {"namespace": namespace, "run_id": run_id,
                "combos": len(records), "records": records,
                "pareto": pareto_frontier(records)}

    def get_experiments(self, namespace: str, limit: int = 200,
                        layer: Optional[str] = None) -> Dict[str, Any]:
        """실험 레코드 조회 (최신 먼저)."""
        from ..core.experiment import get_experiment_store
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        entries = get_experiment_store(namespace).entries(limit=limit,
                                                          layer=layer)
        return {"namespace": namespace, "entries": entries}

    def recommend_retrieval_config(self, namespace: str,
                                   quality: str = "mrr",
                                   cost: str = "latency_ms_p50",
                                   min_cases: int = 50) -> Dict[str, Any]:
        """실험 결과 → 네임스페이스 설정 제안 (Phase 4 — **제안까지만**).

        규율:
        - **현재 그래프 지문과 같은 레코드만** 후보 — knob 결론이 그래프
          상태에 두 번 뒤집힌 역사의 코드화. 없으면 stale ("재실험 먼저").
        - 표본 < min_cases 면 제안 대신 경고 — 47케이스 과적합 교훈.
        - 적용은 사람이 기존 retrieval-config API(검증+감사)로. 이 메서드는
          쓰기 경로가 없다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.experiment import (get_experiment_store, graph_fingerprint,
                                       pareto_frontier)
        from ..core.retrieval_config import KEYS as _CFG_KEYS
        from ..core.retrieval_config import get_retrieval_config
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}

        engine = get_knowledge_graph_engine(namespace)
        gfp = graph_fingerprint(engine.graph,
                                get_chunk_store(namespace).all())
        entries = get_experiment_store(namespace).entries(layer="retrieval")
        candidates = [e for e in entries
                      if (e.get("graph") or {}).get("hash") == gfp["hash"]
                      and (e.get("metrics") or {}).get("measured")]
        base = {"namespace": namespace, "graph_hash": gfp["hash"],
                "candidates": len(candidates)}
        if not candidates:
            return {**base, "stale": True, "suggestion": None,
                    "detail": "현재 그래프 상태로 잰 실험이 없다 — "
                              "/experiments/run 재실행이 먼저다"}

        frontier = pareto_frontier(candidates, quality=quality, cost=cost)

        # 현재 운영 설정과 일치하는 레코드 (채널은 확산 플래그로 유추)
        live = get_retrieval_config(namespace).effective()
        live_channel = ("graph+prop"
                        if live.get("use_propagation")
                        and live.get("propagation_channel") else "graph")

        def _matches_live(rec: Dict[str, Any]) -> bool:
            cfg = rec.get("config") or {}
            if cfg.get("channel") != live_channel:
                return False
            return all(cfg.get(key) == live.get(key) for key in _CFG_KEYS
                       if key in cfg)

        current = next((r for r in candidates if _matches_live(r)), None)

        best = frontier[-1] if frontier else None  # 비용 오름차순 — 최고 품질 쪽
        # 품질 최우선 선택: 프런티어에서 quality 최대 (동률이면 저비용)
        if frontier:
            best = max(frontier,
                       key=lambda r: ((r.get("metrics") or {}).get(quality) or 0,
                                      -((r.get("cost") or {}).get(cost) or 0)))

        warnings: List[str] = []
        if best and int((best.get("golden") or {}).get("cases") or 0) < min_cases:
            warnings.append("small_sample")

        suggestion = None
        if best and "small_sample" not in warnings:
            cur_q = ((current or {}).get("metrics") or {}).get(quality)
            best_q = (best.get("metrics") or {}).get(quality)
            cur_c = ((current or {}).get("cost") or {}).get(cost)
            best_c = (best.get("cost") or {}).get(cost)
            improves = (current is None
                        or (best_q is not None and cur_q is not None
                            and best_q > cur_q)
                        or (best_q == cur_q and best_c is not None
                            and cur_c is not None and best_c < cur_c))
            if improves and best is not current:
                overrides = {kk: vv for kk, vv in (best.get("axes") or {}).items()
                             if kk in _CFG_KEYS and live.get(kk) != vv}
                channel = (best.get("axes") or {}).get("channel")
                if channel and channel != live_channel:
                    prop = channel == "graph+prop"
                    if channel != "vector":
                        overrides.setdefault("use_propagation", prop)
                        overrides.setdefault("propagation_channel", prop)
                if overrides:
                    suggestion = {
                        "overrides": overrides,
                        "based_on_run_id": best.get("run_id"),
                        "delta": {quality: [cur_q, best_q],
                                  cost: [cur_c, best_c]},
                        "apply_via": f"POST /graphs/{namespace}/retrieval-config",
                    }
        if suggestion is None and "small_sample" in warnings:
            base["detail"] = (f"표본 {int((best.get('golden') or {}).get('cases') or 0)}"
                              f" < {min_cases} — knob 확정 금지, 골든셋 확대 먼저")

        return {**base, "stale": False,
                "current": ({"axes": current.get("axes"),
                             "metrics": current.get("metrics"),
                             "cost": current.get("cost")} if current else None),
                "frontier": [{"axes": r.get("axes"),
                              "metrics": r.get("metrics"),
                              "cost": r.get("cost"),
                              "run_id": r.get("run_id")} for r in frontier],
                "suggestion": suggestion, "warnings": warnings}

    def get_eval_history(self, namespace: str,
                         limit: int = 50) -> Dict[str, Any]:
        """평가 이력 — 설정 지문 + 지표. 관리 콘솔의 비교용."""
        from ..core.eval_history import get_eval_history as _hist
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        return {"namespace": namespace,
                "entries": _hist(namespace).entries(limit=limit)}

    def semantic_search(self, namespace: str, query: str, top_k: int = 5,
                        node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        engine = get_knowledge_graph_engine(namespace)
        return engine.semantic_search(query, top_k=top_k, node_types=node_types)

    # ─── Chunks (원문 · 근거) ────────────────────────────────────────

    @staticmethod
    def _chunk_view(stored) -> Dict[str, Any]:
        return {"chunk_id": stored.chunk_id, "text": stored.text,
                "source": stored.source, "index": stored.index,
                "section": stored.section,
                "char_start": stored.char_start, "char_end": stored.char_end,
                "node_ids": list(stored.node_ids),
                "trust": getattr(stored, "trust", ""),
                "meta": dict(getattr(stored, "meta", {}) or {})}

    def get_chunk(self, namespace: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        from ..core.chunk_store import get_chunk_store
        stored = get_chunk_store(namespace).get(chunk_id)
        return self._chunk_view(stored) if stored else None

    def get_node_chunks(self, namespace: str, node_id: str) -> Dict[str, Any]:
        """이 노드가 추출된 원문 청크들 — 상세 패널의 '근거' 탭.

        그래프가 "무엇을 아는가"를 말한다면 이것은 "어디서 알았는가"를
        말한다. 축 2 이전에는 후자를 답할 수 없었다.
        """
        from ..core.chunk_store import get_chunk_store
        chunks = get_chunk_store(namespace).chunks_for_node(node_id)
        return {"namespace": namespace, "node_id": node_id,
                "chunks": [self._chunk_view(c) for c in chunks]}

    def search_chunks(self, namespace: str, query: str,
                      top_k: int = 5) -> Dict[str, Any]:
        """원문 구절 검색 — 의미(임베딩) 또는 하이브리드(ES).

        축 3 에서 부분문자열 → ChunkIndex 로 승격. 백엔드 tier 는 청크 수와
        ONTOLOGY_VECTOR_BACKEND 로 결정된다 (elasticsearch 면 BM25+벡터
        하이브리드). 임베더가 없으면 부분문자열로 degrade 하고, 그때 score 는
        유사도가 아니므로 0.0 으로 나온다.
        """
        from ..core.chunk_index import get_chunk_index
        hits = get_chunk_index(namespace).search(query, top_k=top_k)
        return {"namespace": namespace, "query": query,
                "hits": [{**self._chunk_view(c), "score": score}
                         for c, score in hits]}

    # ─── Review (검수 루프 — 확정 · 거절 · 감사 이력) ────────────────

    def get_review_queue(self, namespace: str, trust: Optional[str] = None,
                         limit: int = 50) -> Dict[str, Any]:
        """검수 대기 노드 목록 — LLM 추출로 들어와(source 有) 아직 판정이
        없는 것들. 근거 청크 개수를 함께 싣는다 — 근거 없이는 검수자가
        판정할 수 없다 (상세 근거는 GET node/chunks 가 답한다).

        trust 필터는 낮은 신뢰 출처(summary)부터 검수하는 워크플로우용이다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        graph = get_knowledge_graph_engine(namespace).graph
        reviews = get_review_store(namespace)
        chunks = get_chunk_store(namespace)

        items: List[Dict[str, Any]] = []
        for node_id, attrs in graph.nodes(data=True):
            if not attrs.get("source"):
                continue  # 인제스트 출처가 없는 노드(수동/시스템)는 검수 대상이 아니다
            if reviews.is_confirmed(node_id) or reviews.is_rejected(node_id):
                continue  # 이미 판정된 노드는 큐를 떠난다
            if trust is not None and attrs.get("trust", "") != trust:
                continue
            items.append({
                "node_id": node_id,
                "name": attrs.get("name", node_id),
                "type": attrs.get("type", ""),
                "trust": attrs.get("trust", ""),
                "source": attrs.get("source", ""),
                "definition": attrs.get("definition", ""),
                "evidence_count": len(chunks.chunks_for_node(node_id)),
                # 근거대조 에이전트의 사전판정 (없으면 None) — 검수자가 추천과
                # 근거 인용을 보고 최종 판정한다
                "recommendation": reviews.recommendation_for(node_id),
            })
            if len(items) >= limit:
                break
        return {"namespace": namespace, "items": items, "total": len(items)}

    async def precheck_review_queue(self, namespace: str,
                                    trust: Optional[str] = None,
                                    limit: int = 20) -> Dict[str, Any]:
        """근거대조 에이전트 — 검수 큐 사전판정 (aicoach 협업 패턴).

        큐의 각 노드에 대해: 노드가 추출된 원문 청크(span)와 대조 →
        confirm/reject/unsure **추천** + 근거 인용을 ReviewStore 에 남긴다
        (action=recommend, 감사 이력에 포함). 판정은 만들지 않는다 — 최종
        권한은 인간에게 있다.

        비용: 근거가 있는 노드당 LLM 1콜 (gemini-3.5-flash 기본), 근거 없는
        노드는 0콜(no_evidence). 이미 판정된 노드는 큐에 없으므로 자동 제외.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.evidence_checker import EvidenceChecker
        from ..core.review_store import get_review_store

        queue = self.get_review_queue(namespace, trust=trust, limit=limit)
        chunks = get_chunk_store(namespace)
        reviews = get_review_store(namespace)
        checker = EvidenceChecker(llm_fn=self._active_llm_fn())

        recommendations: List[Dict[str, Any]] = []
        for item in queue["items"]:
            node_id = item["node_id"]
            chunk_texts = [c.text for c in chunks.chunks_for_node(node_id)]
            rec = await checker.check(item, chunk_texts)
            reviews.recommend(node_id, verdict=rec["verdict"],
                              rationale=rec.get("rationale", ""),
                              quote=rec.get("evidence_quote", ""),
                              actor="evidence_checker")
            recommendations.append({"node_id": node_id, **rec})

        return {"namespace": namespace, "checked": len(recommendations),
                "recommendations": recommendations}

    def lint_consistency(self, namespace: str,
                         schema_mode: Optional[str] = None,
                         custom_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """일관성 감시 에이전트 — 그래프 내부 모순 스캔 (LLM 0콜, 결정적).

        이름-타입 충돌 · 별칭 충돌 · (스키마 지정 시) domain/range 위반.
        schema_mode 없이 부르면 range 검사는 생략된다 — 검사할 선언이 없다.

        findings 는 **저장하지 않는다** (review_store 무기록) — 재계산 가능한
        파생물은 저장하지 않는다. 저장하면 그래프와 어긋난 순간 어느 쪽이
        진실인지 알 수 없다 (review_store 의 "로그가 원본" 원칙의 대우).
        gap 을 기록하는 check_coverage 와 대비되는 지점: 그쪽은 재실행 비용이
        LLM 이라 비싸고 발견 자체가 사건이지만, 여기는 공짜 재계산이다.
        """
        from collections import Counter

        from ..core.consistency_checker import lint_graph
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        schema = None
        if schema_mode:
            schema = self._resolve_schema(schema_mode, custom_schema)
        graph = get_knowledge_graph_engine(namespace).graph
        findings = lint_graph(graph, schema=schema)
        counts = Counter(f["kind"] for f in findings)
        return {"namespace": namespace, "findings": findings,
                "counts": dict(counts)}

    def graph_health(self, namespace: str, sample: int = 20) -> Dict[str, Any]:
        """근거 사슬 건강 진단 — 추출 커버리지 · 고아 노드 · 중복 노드 후보.

        lint_consistency(검수자의 작업 큐)와 대비되는 **관리자의 요약**이다.
        그쪽은 그래프 내부 모순을 건별로 내고, 여기는 그래프와 청크를 **함께**
        보아 비율을 낸다 — 노드↔청크 링크가 성기면 /retrieve 의 그래프 채널이
        코퍼스의 일부만 보고 도는데, 그 사실은 어느 지표에도 나타나지 않았다.

        LLM 0콜, 저장 없음 — 재계산 가능한 파생물이다 (lint_consistency 와 같은
        계약). 그래서 GET 이고, 캐시하지 않는다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.graph_health import health_report
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        from datetime import datetime as _dt

        from ..core.lifecycle import overdue, state_counts

        graph = get_knowledge_graph_engine(namespace).graph
        chunks = list(get_chunk_store(namespace).all())
        today = _dt.now().date().isoformat()
        return {"namespace": namespace,
                **health_report(list(graph.nodes()), chunks, sample=sample),
                # 생애주기 — 기한을 적어두고 아무도 안 보면 "언젠가 정리하자"가
                # 그대로 돌아온다. 그래서 분포와 **기한 초과**를 함께 들춘다.
                "lifecycle": state_counts(graph),
                "lifecycle_overdue": overdue(graph, today)}

    def coverage_gate(self, namespace: str) -> Dict[str, Any]:
        """커버리지 expectation 게이트 — 현재 상태 재평가 (읽기, LLM 0콜, B2).

        빌드 시점 스냅샷은 잡 리포트의 `coverage_gate` 에 남고, 이 GET 은
        언제든 **같은 자(graph_health)** 로 재평가한다 — 게이트와 health 탭이
        다른 셈을 쓰면 둘 중 하나는 거짓이 된다. 임계 미설정이면
        unconfigured 로 지표만 보고 (경고 아님 — 임계는 운영자가 명시적으로
        건다, 팔란티어 expectation 차용).
        """
        from ..core.coverage_expectations import get_coverage_expectations

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        hr = self.graph_health(namespace, sample=0)
        metrics = {
            "extraction_coverage": hr.get("extraction_coverage"),
            "orphan_node_count": hr.get("orphan_node_count"),
            "node_count": hr.get("nodes"),
            "unlinked_chunk_count": hr.get("unlinked_chunk_count"),
            "chunk_count": hr.get("chunks"),
            "dangling_node_ref_count": hr.get("dangling_node_ref_count"),
        }
        cfg = get_coverage_expectations(namespace)
        gate = cfg.evaluate(metrics)
        thresholds = {k: v for k, v in cfg.effective().items()
                      if v is not None}
        return {"namespace": namespace, **gate, "thresholds": thresholds}

    async def _coverage_gate_for_job(self, namespace: str) -> Dict[str, Any]:
        """빌드/인제스트 완료 시 게이트 평가 + (옵트인) 자동 커버리지 검사.

        **best-effort** — 어떤 실패도 빌드를 깨뜨리지 않는다 (_mirror_to_pg
        와 같은 계약; 게이트 장애로 빌드 전체를 잃으면 손해가 훨씬 크다).
        auto_coverage_check 는 기본 off — LLM 비용을 측정 없이 전 사용자에게
        물리지 않는다. 켜져 있고 warn 이면 예산 상한 안에서 check_coverage
        를 돌려 gap 을 감사 로그에 미리 채운다 (사후 발견 제거).
        """
        from ..core.coverage_expectations import get_coverage_expectations

        gate = self.coverage_gate(namespace)
        if "error" in gate:
            return gate
        gate["auto_check"] = None
        try:
            eff = get_coverage_expectations(namespace).effective()
            if gate.get("status") == "warn" and eff.get("auto_coverage_check"):
                limit = int(eff.get("auto_coverage_limit") or 30)
                res = await self.check_coverage(namespace, limit=limit,
                                                only_unlinked=True)
                gate["auto_check"] = {
                    "chunks_checked": res.get("chunks_checked"),
                    "gaps": len(res.get("gaps") or []),
                    "limit": limit,
                }
        except Exception as e:
            logger.warning(f"⚠️ auto coverage check failed ({namespace}): {e}")
            gate["auto_check"] = {"error": str(e)}
        return gate

    def get_coverage_expectations_settings(self, namespace: str) -> Dict[str, Any]:
        """커버리지 임계 — 오버라이드 + 실효값 (retrieval settings 와 대칭)."""
        from ..core.coverage_expectations import (effective_config,
                                                  get_coverage_expectations)
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        cfg = get_coverage_expectations(namespace)
        return {"namespace": namespace,
                "overrides": cfg.overrides(),
                "effective": cfg.effective(),
                "defaults": effective_config({})}

    def set_coverage_expectations_settings(self, namespace: str,
                                           overrides: Dict[str, Any],
                                           actor: str = "") -> Dict[str, Any]:
        """커버리지 임계 변경 — 검증·감사 (retrieval settings 와 같은 규율)."""
        from ..core.coverage_expectations import (get_coverage_expectations,
                                                  validate_overrides)
        from ..core.review_store import get_review_store

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not isinstance(overrides, dict) or not overrides:
            return {"error": "invalid", "detail": "overrides must not be empty"}
        ok, why = validate_overrides(overrides)
        if not ok:
            return {"error": "invalid", "detail": why}

        cfg = get_coverage_expectations(namespace)
        before = cfg.overrides()
        after = cfg.set(overrides, actor=actor or "admin")
        get_review_store(namespace).record(
            action="expectations_change", node_id=f"namespace:{namespace}",
            before=before, after=after, actor=actor or "admin")
        return {"namespace": namespace, "overrides": after,
                "effective": cfg.effective()}

    async def check_coverage(self, namespace: str,
                             limit: int = 10,
                             only_unlinked: bool = False,
                             min_len: int = COVERAGE_MIN_LEN) -> Dict[str, Any]:
        """커버리지 에이전트 — 원문에 있는데 그래프에 없는 개체(gap) 탐지.

        기지 개체 = 그래프의 모든 노드 이름 + 별칭(소문자·공백 정규화),
        허용 타입 = 그래프에 실존하는 타입들 — gap 보고가 스키마를 여는
        뒷문이 되지 않게 한다.

        비용: 원문이 있는 청크당 **LLM 1콜** (gemini-3.5-flash 기본) —
        limit 이 비용 상한이다. 빈 청크는 0콜로 건너뛴다.

        발견된 gap 은 감사 로그에 남긴다 (action=coverage_gap) — ②일관성의
        findings 와 달리 재실행 비용이 LLM 이라 비싸고, "무엇을 놓쳤었나"
        자체가 사건이다. 판정 상태(묘비/확정)에는 영향이 없다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.coverage_checker import CoverageChecker
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        graph = get_knowledge_graph_engine(namespace).graph
        chunks = get_chunk_store(namespace)
        reviews = get_review_store(namespace)

        known_names = set()
        node_types = set()
        for _, attrs in graph.nodes(data=True):
            if attrs.get("type"):
                node_types.add(str(attrs["type"]))
            if attrs.get("name"):
                known_names.add(str(attrs["name"]).lower())
            for alias in attrs.get("aliases") or []:
                if alias:
                    known_names.add(str(alias).lower())

        checker = CoverageChecker(llm_fn=self._active_llm_fn())
        known_sorted = sorted(known_names)
        types_sorted = sorted(node_types)
        gaps: List[Dict[str, Any]] = []
        chunks_checked = 0

        candidates = [c for c in chunks.all() if (c.text or "").strip()]
        if only_unlinked:
            # 예산을 공백으로 돌린다. 짧은 조각(머리말·페이지번호)은 건너뛴다 —
            # 거기서 개체가 안 나온 것은 결함이 아니라 정상이고 콜만 태운다.
            candidates = [c for c in candidates
                          if not c.node_ids and len(c.text or "") >= min_len]
        eligible = len(candidates)

        rejected_filtered = 0
        for stored in candidates:
            if chunks_checked >= limit:
                break
            chunks_checked += 1
            findings = await checker.check_chunk(stored, known_sorted,
                                                 types_sorted)
            for finding in findings:
                # 기각된 gap 은 거른다 — 관계 제안과 같은 처방. 기각을
                # 프롬프트의 known_names 에 섞지 않는 이유: "이미 안다"는
                # 거짓이 되고, 이름이 같은 **다른** 개체까지 억누른다.
                tomb = reviews.gap_rejection(finding.get("type"),
                                             finding.get("name"))
                if tomb and (tomb.get("scope") != "evidence"
                             or tomb.get("chunk_id") == stored.chunk_id):
                    rejected_filtered += 1
                    continue
                reviews.record(
                    action="coverage_gap",
                    node_id=f"{finding['type']}:{finding['name']}",
                    after=dict(finding), actor="coverage_checker",
                    source=stored.source)
                gaps.append(finding)

        # eligible 을 함께 낸다 — 검사한 수만 보고하면 "다 봤다"로 읽힌다.
        # 상한(limit)에 걸려 남은 후보가 몇인지 보이지 않으면 조용한 절단이다.
        # rejected_filtered 를 함께 낸다 — 조용히 거르면 "제안이 없었다"와
        # "이미 기각한 것을 걸렀다"를 구별할 수 없다 (관계 쪽과 같은 규율).
        return {"namespace": namespace, "chunks_checked": chunks_checked,
                "eligible": eligible, "only_unlinked": bool(only_unlinked),
                "rejected_filtered": rejected_filtered,
                "gaps": gaps}

    def reject_gaps(self, namespace: str, gaps: List[Dict[str, Any]],
                    actor: str = "") -> Dict[str, Any]:
        """커버리지 gap 제안 기각 — 묘비를 남긴다 (reject_relations 의 대칭).

        dry_run 이 없다: 그래프를 바꾸지 않고 판정만 기록하므로 미리 볼 것이
        없다. 번복은 승인(override_rejected)이 걷는다.
        """
        # 승인 경로와 **같은 정규화**를 써야 "기각한 그것"과 "승인하려는
        # 그것"이 같은 신원이 된다 (approve_coverage_gaps 와 한 벌).
        from ..core.evidence_checker import _squash_ws
        from ..core.review_store import get_review_store
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not isinstance(gaps, list) or not gaps:
            return {"error": "invalid", "detail": "gaps must not be empty"}

        reviews = get_review_store(namespace)
        rejected: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for gap in gaps:
            if not isinstance(gap, dict):
                skipped.append({"gap": gap, "reason": "invalid"})
                continue
            name = _squash_ws(str(gap.get("name") or ""))
            node_type = str(gap.get("type") or "").strip()
            reason = str(gap.get("reason") or "").strip()
            if not name or not node_type:
                skipped.append({"gap": gap, "reason": "invalid"})
                continue
            # 사유 필수 — 이유 없는 기각은 다음 사람이 재검토할 수 없고,
            # 묘비만 남아 "왜 없는가"에 답하지 못한다 (관계 기각과 같은 계약).
            if not reason:
                skipped.append({"type": node_type, "name": name,
                                "reason": "reason_required"})
                continue
            if reviews.gap_rejection(node_type, name):
                skipped.append({"type": node_type, "name": name,
                                "reason": "already_rejected"})
                continue
            reviews.reject_gap(node_type, name,
                               chunk_id=str(gap.get("chunk_id") or "").strip(),
                               reason=reason,
                               scope=str(gap.get("scope") or "entity"),
                               actor=actor or "admin")
            rejected.append({"type": node_type, "name": name})
        return {"namespace": namespace, "rejected": len(rejected),
                "items": rejected, "skipped": skipped}

    def approve_coverage_gaps(self, namespace: str,
                              gaps: List[Dict[str, Any]],
                              actor: str = "",
                              dry_run: bool = True) -> Dict[str, Any]:
        """커버리지 gap 을 **노드 + 근거 링크**로 승인한다 (회복 쓰기 경로).

        check_coverage 는 gap 을 감사 로그에만 남긴다(발견은 판정이 아니다).
        이 메서드가 그 발견을 그래프로 되돌리는 유일한 경로다.

        **요청 본문을 신뢰하지 않는다.** coverage_checker 의 "환각 0" 은 원문
        대조가 만든 성질이고, 엔드포인트가 받는 {name, type, chunk_id} 는 그
        성질을 물려받지 않는다. 그래서 같은 검증을 다시 통과시킨다:
        - 근거 청크가 실존해야 한다 (없으면 provenance 없는 노드다)
        - 이름이 그 청크 원문에 **문자 그대로** 있어야 한다
          (_quote_in_chunks 재사용 — 규칙을 두 벌 두면 두 경로의 '원문'
          정의가 갈라진다)
        - 타입은 그래프에 실존하는 타입만 — check_coverage 의 허용 타입과
          **정확히 같은 집합**이다. 승인이 스키마를 여는 뒷문이 되면 안 된다
          (결과: 노드 0인 새 네임스페이스에서는 승인할 것이 없다)

        create_node 를 그대로 쓰지 않는 이유: 그쪽은 청크 링크를 만들지
        않는다. 링크 없는 노드는 고아 노드(graph_health 가 세는 그것)를
        늘리는 것이고, gap 회복의 목적인 근거 사슬을 배신한다.

        묘비된 후보는 **되살리지 않는다**(reason=tombstoned) — 34건 일괄
        승인 중 하나가 과거 거절이면 부활은 검수자 모르게 판정을 뒤집는
        것이다. 명시적 번복은 create_node 가 담당한다.

        이미 있는 노드는 오류가 아니라 링크만 한다(outcome=linked) — 노드는
        있고 근거 링크만 없는 것도 실제 결함이고, duplicate 로 막으면 그
        결함을 고칠 길이 없다. 기존 attrs 는 덮어쓰지 않는다.

        definition 은 만들지 않는다 — 원문에 있는 것은 이름뿐이다. 정의를
        지어내면 노드 병합에서 거부한 그 문제(환각 정의)를 다시 만든다.
        """
        from datetime import datetime as _dt

        from ..core.chunk_store import get_chunk_store
        from ..core.evidence_checker import _quote_in_chunks, _squash_ws
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not gaps:
            return {"error": "invalid", "detail": "gaps must not be empty"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        chunks = get_chunk_store(namespace)
        reviews = get_review_store(namespace)

        allowed_types = {str(attrs["type"])
                         for _, attrs in graph.nodes(data=True)
                         if attrs.get("type")}

        created: List[str] = []
        linked: List[str] = []
        skipped: List[Dict[str, Any]] = []
        chunks_linked = 0
        seen: set = set()

        def _skip(gap, reason):
            skipped.append({"name": gap.get("name", ""),
                            "type": gap.get("type", ""),
                            "chunk_id": gap.get("chunk_id", ""),
                            "reason": reason})

        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            # 공백 정규화한 이름을 신원으로 쓴다 — 공백만 다른 중복 노드를
            # 만들면 graph_health 가 세는 중복 클러스터를 스스로 늘린다.
            name = _squash_ws(str(gap.get("name") or ""))
            node_type = str(gap.get("type") or "").strip()
            chunk_id = str(gap.get("chunk_id") or "").strip()
            if not name or not node_type:
                _skip(gap, "invalid")
                continue

            stored = chunks.get(chunk_id) if chunk_id else None
            if stored is None:
                _skip(gap, "chunk_not_found")
                continue
            if node_type not in allowed_types:
                _skip(gap, "type_not_allowed")
                continue
            if not _quote_in_chunks(name, [stored.text or ""]):
                _skip(gap, "quote_not_found")   # 원문에 없다 — 환각
                continue

            node_id = f"{node_type}:{name}"
            # dedup 은 **(노드, 청크) 쌍** 기준이다. node_id 로만 걸면 같은
            # 개체가 여러 조문에 나올 때 첫 청크만 이어지고 나머지 근거가
            # 보고도 없이 사라진다 (실측: gap 123건 중 23건이 그 경우).
            # 노드는 하나지만 근거는 여러 개다.
            if (node_id, stored.chunk_id) in seen:
                continue
            seen.add((node_id, stored.chunk_id))
            if reviews.is_rejected(node_id):
                _skip(gap, "tombstoned")
                continue
            # gap 묘비 — 노드가 된 적 없는 후보의 기각. 명시적 번복만
            # 통과시킨다(조용한 되살림 금지, 관계 승인과 같은 계약).
            gap_tomb = reviews.gap_rejection(node_type, name)
            if gap_tomb and not gap.get("override_rejected"):
                _skip(gap, "gap_rejected")
                continue

            exists = node_id in graph
            outcome = "linked" if exists else "created"
            # 노드 자체는 배치에서 한 번만 보고한다 — created 가 같은 id 를
            # 여러 번 담으면 "몇 개 만들었나"를 셀 수 없다.
            if node_id not in created and node_id not in linked:
                (linked if exists else created).append(node_id)
            needs_link = node_id not in stored.node_ids
            if dry_run:
                # 미리보기도 링크 수를 센다 — 0 으로 보고하면 "노드만 만들고
                # 근거는 안 잇는다"로 읽힌다. 적용 결과와 같은 수여야 한다.
                chunks_linked += 1 if needs_link else 0
                continue

            node_attrs: Optional[Dict[str, Any]] = None
            if not exists:
                now = _dt.now().isoformat()
                # source/trust 는 근거 청크에서 물려받는다. 이 노드의 내용은
                # LLM 이 원문에서 읽어 제안한 것이므로 빌더 추출분과 같은
                # 출처 등급이고, 같은 검증 의무를 진다 — source 가 없으면
                # 검수 큐가 담지 않아 개별 검증을 영구히 못 받는다
                # (손으로 적는 create_node 와 갈라지는 지점).
                node_attrs = {"type": node_type, "name": name,
                              "source": stored.source,
                              "trust": stored.trust,
                              "created_at": now, "last_updated": now}
                graph.add_node(node_id, **node_attrs)
                self._pg_apply(namespace,
                               lambda s, nid=node_id, a=dict(node_attrs):
                               s.upsert_node(namespace, nid, a))
            if chunks.link_node(stored.chunk_id, node_id):
                chunks_linked += 1
            reviews.record(action="coverage_approve", node_id=node_id,
                           after={"outcome": outcome,
                                  "chunk_id": stored.chunk_id,
                                  "type": node_type, "name": name},
                           actor=actor or "admin", source=stored.source)

        if not dry_run and (created or chunks_linked):
            engine.save_to_disk()
            chunks.save_to_disk()

        # 시맨틱 색인은 갱신하지 않는다(임베딩 비용, create_node 와 같은 계약).
        # 그 사실을 숨기면 "만들었는데 검색이 안 된다"가 된다 — /reindex 필요.
        return {"namespace": namespace, "dry_run": bool(dry_run),
                "created": created, "linked": linked, "skipped": skipped,
                "chunks_linked": chunks_linked,
                "reindex_required": bool(created)}

    async def propose_relations(self, namespace: str, limit: int = 0,
                                min_nodes: int = 2) -> Dict[str, Any]:
        """청크마다 관계를 제안한다 (읽기 — 그래프를 바꾸지 않는다). 청크당 LLM 1콜.

        노드가 `min_nodes` 개 미만인 청크는 건너뛴다 — 관계는 둘 이상이 있어야
        성립하므로 LLM 콜이 낭비다.

        이미 그래프에 있는 트리플은 제안에서 뺀다(`existing`). 남겨두면 검수자가
        같은 것을 다시 승인하고, "몇 개가 새로 생기나"를 셀 수 없다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.relation_backfill import (allowed_predicates,
                                              build_relation_prompt,
                                              parse_relations, signature_of,
                                              type_signatures)
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        chunks = get_chunk_store(namespace)
        predicates = allowed_predicates(graph)

        existing = {(s, attrs.get("predicate", ""), t)
                    for s, t, attrs in graph.edges(data=True)}
        signatures = type_signatures(graph)

        from ..core.review_store import get_review_store
        reviews = get_review_store(namespace)   # 관계판 묘비 조회 (A2)

        llm_fn = self._active_llm_fn()
        from ..core.search_qa import QAGenerator   # _call_llm 재사용
        caller = QAGenerator(llm_fn=llm_fn)

        candidates = [c for c in chunks.all()
                      if len({n for n in (c.node_ids or []) if n in graph})
                      >= max(2, min_nodes)]
        candidates.sort(key=lambda c: (c.source, c.index))
        if limit:
            candidates = candidates[:limit]

        proposals: List[Dict[str, Any]] = []
        failed = 0
        rejected_filtered = 0
        for chunk in candidates:
            node_ids = {n for n in (chunk.node_ids or []) if n in graph}
            views = [{"node_id": n, "name": graph.nodes[n].get("name", "")}
                     for n in sorted(node_ids)]
            try:
                raw = await caller._call_llm(
                    build_relation_prompt(chunk.text or "", views, predicates))
            except Exception as e:
                logger.warning(f"⚠️ Relation propose failed ({chunk.chunk_id}): {e}")
                failed += 1
                continue
            for rel in parse_relations(raw, chunk.text or "", node_ids,
                                       predicates):
                if (rel["subject"], rel["predicate"], rel["object"]) in existing:
                    continue
                # 관계판 묘비 (A2): scope=triple 이거나 같은 청크의 재제안은
                # 거른다 — 실측 8/10 재출현의 처방. 다른 인용의 재제안은
                # 과거 기각 요약을 동봉해 사람에게 (주장은 인용에 따라
                # 정당할 수 있다). 거른 건수는 셈으로 보고 — 조용한 절단 금지.
                rejection = reviews.relation_rejection(
                    rel["subject"], rel["predicate"], rel["object"])
                if rejection and (rejection.get("scope") == "triple"
                                  or rejection.get("chunk_id") == chunk.chunk_id):
                    rejected_filtered += 1
                    continue
                sig = signature_of(graph, rel["subject"], rel["predicate"],
                                   rel["object"])
                proposals.append({**rel, "chunk_id": chunk.chunk_id,
                                  "section": getattr(chunk, "section", ""),
                                  # 검수 신호 — 거부하지 않는다(type_signatures
                                  # docstring: 83%가 위반인데 다수는 정상 관계다).
                                  "signature_seen": sig in signatures,
                                  "signature": "→".join(sig),
                                  "previously_rejected": rejection})
        return {"namespace": namespace, "chunks_scanned": len(candidates),
                "rejected_filtered": rejected_filtered,
                "chunks_failed": failed, "predicates": predicates,
                "signatures_known": sorted("→".join(s) for s in signatures),
                "proposals": proposals, "proposals_total": len(proposals)}

    def approve_relations(self, namespace: str,
                          relations: List[Dict[str, Any]],
                          actor: str = "",
                          dry_run: bool = True) -> Dict[str, Any]:
        """제안된 관계를 그래프에 넣는다 (쓰기 경로).

        **요청 본문을 신뢰하지 않는다.** propose_relations 의 "인용이 원문에
        있다"는 성질은 그 함수가 만든 것이고, 엔드포인트가 받는 트리플은 그
        성질을 물려받지 않는다. 같은 관문을 다시 통과시킨다 — 양끝이 그 청크의
        노드인지, 술어가 허용 어휘인지, 인용이 그 청크 원문에 있는지.

        **노드를 만들지 않는다.** 양끝이 그래프에 없으면 건너뛴다 — 관계 승인이
        노드 생성의 뒷문이 되면 커버리지 승인의 검증을 우회한다.

        묘비된 노드로는 잇지 않는다(거절된 개체를 관계로 부활시키지 않는다).
        시맨틱 색인은 노드 텍스트에 의존하므로 엣지 추가로는 무효화되지 않는다 —
        `reindex_required` 는 항상 False 다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.evidence_checker import _quote_in_chunks
        from ..core.relation_backfill import allowed_predicates
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not relations:
            return {"error": "invalid", "detail": "relations must not be empty"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        chunks = get_chunk_store(namespace)
        reviews = get_review_store(namespace)
        predicates = set(allowed_predicates(graph))

        added: List[Dict[str, str]] = []
        skipped: List[Dict[str, Any]] = []
        seen: set = set()

        def _skip(rel, reason):
            skipped.append({"subject": rel.get("subject", ""),
                            "predicate": rel.get("predicate", ""),
                            "object": rel.get("object", ""),
                            "reason": reason})

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subject = str(rel.get("subject") or "").strip()
            obj = str(rel.get("object") or "").strip()
            predicate = str(rel.get("predicate") or "").strip()
            chunk_id = str(rel.get("chunk_id") or "").strip()
            quote = str(rel.get("evidence_quote") or "")
            key = (subject, predicate, obj)
            if not subject or not obj or not predicate:
                _skip(rel, "invalid")
                continue
            if key in seen:
                continue
            seen.add(key)
            if subject == obj:
                _skip(rel, "self_loop")
                continue
            if subject not in graph or obj not in graph:
                _skip(rel, "node_not_found")
                continue
            if predicate not in predicates:
                _skip(rel, "predicate_not_allowed")
                continue
            if reviews.is_rejected(subject) or reviews.is_rejected(obj):
                _skip(rel, "tombstoned")
                continue
            # 관계판 묘비 (A2): 기각된 트리플은 기본 skip. 일괄 승인이 기각을
            # **조용히** 뒤집는 사고를 막는다 — 번복은 명시적 override 로만
            # (승인되면 relation_approve 이벤트가 묘비를 걷는다).
            if (reviews.relation_rejection(subject, predicate, obj)
                    and not rel.get("override_rejected")):
                _skip(rel, "relation_rejected")
                continue
            stored = chunks.get(chunk_id) if chunk_id else None
            if stored is None:
                _skip(rel, "chunk_not_found")
                continue
            node_ids = set(stored.node_ids or [])
            if subject not in node_ids or obj not in node_ids:
                _skip(rel, "not_in_chunk")
                continue
            if not _quote_in_chunks(quote, [stored.text or ""]):
                _skip(rel, "quote_not_found")
                continue
            already = any(a.get("predicate") == predicate
                          for a in (graph.get_edge_data(subject, obj)
                                    or {}).values())
            if already:
                _skip(rel, "already_exists")
                continue

            added.append({"subject": subject, "predicate": predicate,
                          "object": obj})
            if dry_run:
                continue
            graph.add_edge(subject, obj, predicate=predicate,
                           source=stored.source, chunk_id=stored.chunk_id)
            self._pg_apply(namespace,
                           lambda s, a=subject, p=predicate, o=obj:
                           s.add_edge(namespace, a, p, o))
            reviews.record(action="relation_approve", node_id=subject,
                           after={"predicate": predicate, "target": obj,
                                  "chunk_id": stored.chunk_id,
                                  "evidence_quote": quote.strip()[:200]},
                           actor=actor or "admin", source=stored.source)

        if not dry_run and added:
            engine.save_to_disk()

        return {"namespace": namespace, "dry_run": bool(dry_run),
                "added": added, "added_total": len(added),
                "skipped": skipped,
                # 엣지는 노드 텍스트를 바꾸지 않는다 → 색인 그대로 유효
                "reindex_required": False}

    def reject_relations(self, namespace: str,
                         relations: List[Dict[str, Any]],
                         actor: str = "") -> Dict[str, Any]:
        """관계 제안 기각 — 관계판 묘비 기록 (A2).

        종전에는 기각이 응답 JSON 으로만 반환되고 어디에도 남지 않아 같은
        오제안이 매 라운드 재출현했다 (실측 8/10). 규정:
        - **reason 필수** — 이유 없는 기각은 감사가 아니다 (근거대조의
          "이유 없는 reject 강등"과 같은 규율). 없으면 skip.
        - scope: "triple"(기본 — 주장 자체가 거짓) | "evidence"(이 인용만).
        - 이미 기각된 트리플은 skip (already_rejected) — 이중 기록 방지.
        """
        from ..core.review_store import get_review_store

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not relations:
            return {"error": "invalid", "detail": "relations must not be empty"}

        reviews = get_review_store(namespace)
        rejected: List[Dict[str, str]] = []
        skipped: List[Dict[str, Any]] = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subject = str(rel.get("subject") or "").strip()
            predicate = str(rel.get("predicate") or "").strip()
            obj = str(rel.get("object") or "").strip()
            reason = str(rel.get("reason") or "").strip()
            base = {"subject": subject, "predicate": predicate, "object": obj}
            if not subject or not predicate or not obj:
                skipped.append({**base, "reason": "invalid"})
                continue
            if not reason:
                skipped.append({**base, "reason": "reason_required"})
                continue
            if reviews.relation_rejection(subject, predicate, obj):
                skipped.append({**base, "reason": "already_rejected"})
                continue
            reviews.reject_relation(
                subject, predicate, obj,
                chunk_id=str(rel.get("chunk_id") or ""),
                reason=reason,
                scope=str(rel.get("scope") or "triple"),
                actor=actor or "admin")
            rejected.append(base)

        return {"namespace": namespace, "rejected": rejected,
                "rejected_total": len(rejected), "skipped": skipped}

    def list_documents(self, namespace: str) -> Dict[str, Any]:
        """문서 목록 — 문서별 청크·근거 커버리지·노드 수.

        문서는 KG 노드가 아니라 `chunk.source` 축의 **뷰**다 — 노드로 만들면
        청크 수백 개와 링크된 슈퍼허브가 되어 채널 B·확산을 오염시킨다
        (허브 오염은 lift 교정에서 실측된 위험이다). core/document_view 참고.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.document_view import document_views
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        graph = get_knowledge_graph_engine(namespace).graph
        docs = document_views(graph, get_chunk_store(namespace).all())
        return {"namespace": namespace, "documents": docs,
                "documents_total": len(docs)}

    def coverage_map(self, namespace: str, source: str = "") -> Dict[str, Any]:
        """커버리지 지도 — 문서를 원문 순서(char_start)대로 편 청크 스트립.

        list_documents 가 "커버리지 70%"를 주면, 이건 **"빈 30%가 어느 절인가"**
        를 준다 — 커버리지 회복 루프의 타깃 선정 재료. core/document_view 참고.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.document_view import coverage_map as _cov_map
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        graph = get_knowledge_graph_engine(namespace).graph
        result = _cov_map(graph, get_chunk_store(namespace).all(), source=source)
        result["namespace"] = namespace
        return result

    def compare_documents(self, namespace: str, source_a: str,
                          source_b: str) -> Dict[str, Any]:
        """문서 간 개체 대조 — "A가 다루는 것 중 B에 없는 것" (LLM 0콜, 결정적).

        PROJ-A 용례가 이것이다: 제안요청서 ↔ 제안서. 개체의 문서 귀속 = 그 문서의
        청크에 근거 링크가 있는가 — 근거 링크가 성기면 대조도 성기므로, 문서별
        coverage 를 함께 보라(list_documents).
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.document_view import compare_documents as _compare
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        graph = get_knowledge_graph_engine(namespace).graph
        result = _compare(graph, get_chunk_store(namespace).all(),
                          source_a, source_b)
        return {"namespace": namespace, **result}

    def get_retrieval_settings(self, namespace: str) -> Dict[str, Any]:
        """네임스페이스 검색 설정 — 오버라이드 + 실효값 + 전역 기본값.

        셋을 다 주는 이유: 화면이 "무엇을 내가 정했고, 무엇이 기본값이며, 지금
        실제로 쓰이는 값은 무엇인가"를 구별해 보여줘야 한다.
        """
        from ..core.retrieval_config import (effective_config,
                                             get_retrieval_config)
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        config = get_retrieval_config(namespace)
        return {"namespace": namespace,
                "overrides": config.overrides(),
                "effective": config.effective(),
                "defaults": effective_config({})}

    def set_retrieval_settings(self, namespace: str,
                               overrides: Dict[str, Any],
                               actor: str = "") -> Dict[str, Any]:
        """검색 설정 변경 — 병합, `None` 은 그 키를 지운다(기본값 복귀).

        값을 **검증한다**: entry_k=0 이면 진입 노드가 0개가 되는데 조용히 통과하면
        "검색이 안 된다"의 원인을 찾기 어렵다. 오타 키도 소리내어 거부한다.

        감사에 남긴다 — 검색 품질이 바뀌는 변경이므로 "누가 언제 왜"가 필요하다.
        """
        from ..core.retrieval_config import (get_retrieval_config,
                                             validate_overrides)
        from ..core.review_store import get_review_store

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not isinstance(overrides, dict) or not overrides:
            return {"error": "invalid", "detail": "overrides must not be empty"}
        ok, why = validate_overrides(overrides)
        if not ok:
            return {"error": "invalid", "detail": why}

        config = get_retrieval_config(namespace)
        before = config.overrides()
        after = config.set(overrides, actor=actor or "admin")
        get_review_store(namespace).record(
            action="retrieval_config", node_id=f"namespace:{namespace}",
            before=before, after=after, actor=actor or "admin")
        return {"namespace": namespace, "overrides": after,
                "effective": config.effective(),
                # 검색 설정은 색인을 건드리지 않는다 (질의 시점 파라미터).
                "reindex_required": False}

    def set_node_lifecycle(self, namespace: str, node_id: str, state: str,
                           reason: str = "", sunset: str = "",
                           superseded_by: str = "",
                           actor: str = "") -> Dict[str, Any]:
        """노드 생애주기 전이 — 파괴 통제의 선언 지점.

        `active` 는 삭제·개명을 차단하고, 그것을 벗어나는 유일한 길이
        `deprecated`(사유 + 삭제 기한 필수)다 — `active → experimental → 삭제`
        우회를 막기 위해서다. 자세한 근거는 `core/lifecycle` docstring.

        **대체 노드(superseded_by)는 실존을 확인한다.** 허공을 가리키는 대체
        지정은 "이걸 대신 쓰라"는 안내를 거짓으로 만든다.
        """
        from datetime import datetime as _dt

        from ..core.lifecycle import (apply_transition, current_state,
                                      validate_transition)
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        if node_id not in graph:
            return {"error": "node_not_found", "node_id": node_id}

        attrs = graph.nodes[node_id]
        from_state = current_state(attrs)
        state = str(state or "").strip().lower()
        ok, why = validate_transition(from_state, state,
                                      reason=reason, sunset=sunset)
        if not ok:
            return {"error": "invalid_transition", "detail": why,
                    "from": from_state, "to": state}
        if superseded_by and superseded_by not in graph:
            return {"error": "superseder_not_found",
                    "detail": "대체 노드가 그래프에 없다 — 허공을 가리키는 "
                              "대체 지정은 안내를 거짓으로 만든다",
                    "superseded_by": superseded_by}

        now = _dt.now().isoformat(timespec="seconds")
        apply_transition(attrs, state, reason=reason, sunset=sunset,
                         superseded_by=superseded_by, at=now)
        attrs["last_updated"] = now
        self._pg_apply(namespace,
                       lambda s, nid=node_id, a=dict(attrs):
                       s.upsert_node(namespace, nid, a))
        get_review_store(namespace).record(
            action="lifecycle", node_id=node_id,
            before={"lifecycle": from_state},
            after={"lifecycle": state, "reason": reason, "sunset": sunset,
                   "superseded_by": superseded_by},
            actor=actor or "admin")
        engine.save_to_disk()
        return {"namespace": namespace, "node_id": node_id,
                "from": from_state, "to": state,
                # 상태만 바꿨을 뿐 노드 텍스트는 그대로다 → 색인 유효
                "reindex_required": False}

    def review_queues(self, namespace: str) -> Dict[str, Any]:
        """검수 큐 집계 — "오늘 뭘 검수해야 하나"의 한 화면.

        병목이 사람 검수로 넘어온 뒤(재라벨·중복·draft) 큐가 JSON 응답 속에
        흩어져 있어 아무 화면도 이 질문에 답하지 못했다. 전부 기존 검증된
        경로의 **위임 + 셈**이다 (LLM 0콜, 결정적). 재라벨 후보는 여기 없다 —
        평가 실행(케이스당 회수)이 필요해 비싸므로 보드에서 온디맨드로 돈다.
        """
        from ..core.lifecycle import overdue, state_counts
        from ..core.search_qa import get_golden_set
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}

        # 노드 검수 대기 — get_review_queue 의 total 은 limit 에 잘린다(항목
        # 조립용 API 라 옳은 동작). 카드의 셈은 절단 없이 직접 센다 (같은 조건:
        # 인제스트 출처 有 + 미판정).
        from ..core.review_store import get_review_store
        reviews = get_review_store(namespace)
        graph0 = get_knowledge_graph_engine(namespace).graph
        review_n = sum(
            1 for node_id, attrs in graph0.nodes(data=True)
            if attrs.get("source")
            and not reviews.is_confirmed(node_id)
            and not reviews.is_rejected(node_id))

        dup = self.review_duplicates(namespace)
        by_kind: Dict[str, int] = {"variant": 0, "cross_type": 0, "similar": 0}
        for c in (dup.get("clusters") or []):
            kind = str(c.get("kind") or "")
            if kind in by_kind:
                by_kind[kind] += 1

        golden = get_golden_set(namespace)
        by_status: Dict[str, int] = {}
        for case in golden.cases():
            st = str(getattr(case, "status", "") or "draft")
            by_status[st] = by_status.get(st, 0) + 1

        structural = self.review_structural(namespace)

        from datetime import date
        graph = graph0
        return {
            "namespace": namespace,
            "queues": {
                "node_review": review_n,
                "duplicates": by_kind,
                "golden": by_status,
                "lifecycle": state_counts(graph),
                "lifecycle_overdue": len(overdue(graph, date.today().isoformat())),
                # 로드맵 4 P-1: 구조 단위(조항·문서) 재분류 후보
                "structural": int(structural.get("total") or 0),
            },
        }

    def review_structural(self, namespace: str) -> Dict[str, Any]:
        """구조 단위(조항·문서) 후보 검수 제안 — 로드맵 4 P-1.

        조항·문서가 개념과 같은 타입으로 잡힌 것이 관계 정답률 67%·시그니처
        위반 83% 의 근본 원인. 자동 판별은 기각됐다(오탐 `계약자`·놓침
        표기차) — section 라벨 **전체-정규화 동등성**만 후보로 올리고 판정은
        사람이 한다. LLM 0콜, 결정적.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.review_store import get_review_store
        from ..core.structural_units import find_structural_candidates
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        graph = get_knowledge_graph_engine(namespace).graph
        reviews = get_review_store(namespace)
        result = find_structural_candidates(
            graph, get_chunk_store(namespace).all(),
            tombstoned=reviews.rejected_ids(),
            # 승인이 선언한 구조 단위 타입은 재탐지에서 제외 — 큐가 마른다
            exclude_types=reviews.structural_types())
        return {"namespace": namespace, **result}

    def approve_structural(self, namespace: str, items: List[str],
                           new_type: str, actor: str = "",
                           dry_run: bool = True) -> Dict[str, Any]:
        """구조 단위 재분류 승인 — 로드맵 4 P-3.

        **본문을 신뢰하지 않는다**: 탐지를 재실행해 여전히 후보인 것만 개명
        (coverage/orphan approve 와 같은 규율 — 규칙 두 벌 금지). `new_type`
        은 요청당 하나·필수, 서버 기본값 없음 — 타입 이름은 검수자가 정한다
        (하드코딩 금지의 실행 형태). 항목별 실패(active·비후보)는 전체를 막지
        않되 dry_run 미리보기에 미리 나타난다.
        """
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        new_type = (new_type or "").strip()
        if not new_type:
            return {"error": "invalid", "detail": "new_type required — 서버가 "
                    "타입을 정하지 않는다, 검수자가 요청에 담아라"}

        detection = self.review_structural(namespace)
        if "error" in detection:
            return detection
        candidate_ids = {c["node_id"] for c in detection.get("candidates", [])}

        results: List[Dict[str, Any]] = []
        renamed = 0
        seen: set = set()
        for raw in items or []:
            nid = str(raw or "").strip()
            if not nid or nid in seen:
                continue
            seen.add(nid)
            if nid not in candidate_ids:
                results.append({"node_id": nid, "status": "skipped",
                                "reason": "not_a_candidate"})
                continue
            r = self.rename_node(namespace, nid, new_type,
                                 actor=actor, dry_run=dry_run)
            if "error" in r:
                results.append({"node_id": nid, "status": "blocked",
                                "reason": r["error"],
                                "detail": r.get("detail", "")})
            elif dry_run:
                plan = r["plan"]
                results.append({
                    "node_id": nid, "status": "would_rename",
                    "new_id": plan["new_id"],
                    "edges": len(plan["edges_repointed"]),
                    "chunks": len(plan["chunks_rewritten"]),
                    "golden": len(plan["golden_relabels"]),
                })
            else:
                renamed += 1
                results.append({
                    "node_id": nid, "status": "renamed",
                    "new_id": r["new_id"],
                    "edges": r["edges_repointed"],
                    "chunks": len(r["chunks_rewritten"]),
                    "golden": len(r["golden_relabels"]),
                })

        # 미선언 타입은 적용 시 선언한다 — 스키마 개요가 "관측"만 보고하므로,
        # 선언이 없으면 이 타입의 의도(구조 단위)가 어디에도 남지 않는다.
        # 감사 이벤트(type_declared)는 **배치마다** 기록한다: 탐지 제외
        # (structural_types)가 이 이벤트의 replay 에서 파생되는데, 최초
        # 선언에만 묶으면 타입이 다른 경로로 먼저 선언된 경우 이벤트가 영영
        # 없어 큐가 마르지 않는다. 파생 상태는 set 이라 멱등이다.
        if not dry_run and renamed:
            if new_type not in self.schema_decl.types(namespace):
                self.schema_decl.set_type(
                    namespace, new_type,
                    description="구조 단위 — /review/structural 승인으로 도입")
            from ..core.review_store import get_review_store
            get_review_store(namespace).record(
                action="type_declared", node_id=f"type:{new_type}",
                after={"type": new_type, "via": "approve_structural"},
                actor=actor or "admin")

        return {"namespace": namespace, "new_type": new_type,
                "dry_run": bool(dry_run), "results": results,
                "renamed": renamed,
                "reindex_required": bool(renamed)}

    def review_duplicates(self, namespace: str) -> Dict[str, Any]:
        """중복 클러스터 검수 제안 — 판단 신호(정의·근거 수·출처·생애주기) 동봉.

        health 의 duplicate_clusters 는 id 목록뿐이고 표본으로 잘린다 — 여기는
        **전량**에 신호를 붙인다. 분류(variant/cross_type/similar)는 신호이지
        판정이 아니다(C73 교훈). variant 에만 병합 payload 를 준다 — 사람 판단
        대상에 payload 를 주면 "기계가 권했다"가 된다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.dup_review import propose_duplicate_resolutions
        from ..core.graph_health import duplicate_clusters
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        graph = get_knowledge_graph_engine(namespace).graph
        clusters = duplicate_clusters(list(graph.nodes()))
        result = propose_duplicate_resolutions(
            graph, get_chunk_store(namespace).all(), clusters)
        return {"namespace": namespace, **result}

    # ─── Triage (검수 사전판정 결합 — C1·C2) ─────────────────────────

    async def triage_review(self, namespace: str, limit: int = 30,
                            actor: str = "triage",
                            include_relations: bool = False) -> Dict[str, Any]:
        """검수 큐 트리아지 — 렌즈 신호를 밴드로 결합 (설계 §5, C1-b).

        결정적 렌즈(중복·일관성·구조단위)를 먼저 계산하고, 결정적 신호만으로
        이미 borderline 이 확정된 항목(dup_member·structural_candidate — confirm
        추천이 와도 강등되는 신호들)은 **LLM 콜을 생략**한다 (예산 절약,
        reasons 에 "llm_skipped"). 나머지만 근거대조(EvidenceChecker, precheck
        와 같은 부품 재사용 — 노드당 1콜)를 거쳐 triage_node 로 밴드를 정한다.

        결과는 reviews 의 recommend 이벤트로 남는다 — after 에 기존 계약
        (verdict/rationale/evidence_quote)을 유지한 채 band·signals 를 확장.
        **판정이 아니다** — 판정은 judge_batch 경유 인간만 만든다.

        include_relations=True 면 propose_relations(청크당 LLM 1콜 — 그래서
        명시 옵트인)를 돌려 각 제안에 triage_relation 을 적용한다.
        LLM 이 없으면(_active_llm_fn None) verdict 없이 결정적 신호만으로
        트리아지한다 — 전부 borderline 이 될 수 있지만 죽지 않는다 (degrade).
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.consistency_checker import lint_graph
        from ..core.evidence_checker import EvidenceChecker
        from ..core.graph_health import duplicate_clusters
        from ..core.review_store import get_review_store
        from ..core.review_triage import triage_node, triage_relation
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}

        graph = get_knowledge_graph_engine(namespace).graph
        chunks = get_chunk_store(namespace)
        reviews = get_review_store(namespace)
        queue = self.get_review_queue(namespace, limit=limit)

        # 결정적 렌즈 — 큐 항목마다가 아니라 그래프당 한 번씩 계산 (LLM 0콜)
        dup_members: set = set()
        for cluster in duplicate_clusters(list(graph.nodes())):
            dup_members.update(cluster)
        findings_by_node: Dict[str, int] = {}
        for f in lint_graph(graph):
            for nid in f.get("node_ids", []):
                findings_by_node[nid] = findings_by_node.get(nid, 0) + 1
        structural = self.review_structural(namespace)
        structural_ids = {c["node_id"]
                          for c in structural.get("candidates", [])}

        llm_fn = self._active_llm_fn()
        checker = EvidenceChecker(llm_fn=llm_fn) if llm_fn is not None else None

        llm_calls = 0
        nodes_by_band: Dict[str, List[Dict[str, Any]]] = {
            "strong_confirm": [], "strong_reject": [], "borderline": []}
        for item in queue["items"]:
            node_id = item["node_id"]
            signals = {
                "dup_member": node_id in dup_members,
                "consistency_findings": findings_by_node.get(node_id, 0),
                "structural_candidate": node_id in structural_ids,
            }
            verdict = None
            skip_reason = None
            if signals["dup_member"] or signals["structural_candidate"]:
                # 이 두 신호는 confirm 추천이 와도 강등된다(triage_node 규칙)
                # → LLM 판정이 밴드를 못 바꾸므로 콜이 낭비다. 잃는 것은
                # 잠재적 strong_reject 하나뿐인데, 그 항목은 어차피
                # borderline = 사람 검수 대상이다.
                skip_reason = "llm_skipped"
            elif checker is None:
                skip_reason = "llm_unavailable"
            else:
                chunk_texts = [c.text for c in chunks.chunks_for_node(node_id)]
                if chunk_texts:
                    llm_calls += 1   # 빈 근거는 checker 가 0콜(no_evidence)
                verdict = await checker.check(item, chunk_texts)

            band = triage_node(item, verdict, signals)
            reasons = list(band["reasons"])
            if skip_reason:
                reasons.append(skip_reason)
            after = {"verdict": (verdict or {}).get("verdict"),
                     "rationale": (verdict or {}).get("rationale", ""),
                     "evidence_quote": (verdict or {}).get("evidence_quote", ""),
                     "band": band["band"], "signals": signals}
            # 기존 record() 로만 기록 — recommend 계약(verdict 키)을 유지한 채
            # band/signals 를 확장한다. _apply 가 최신 추천으로 반영한다.
            reviews.record(action="recommend", node_id=node_id,
                           reason=str(after["rationale"] or ""),
                           after=after, actor=actor or "triage")
            nodes_by_band[band["band"]].append({
                "node_id": node_id,
                "name": item.get("name", node_id),
                "band": band["band"], "reasons": reasons,
                "verdict": after["verdict"],
                "rationale": after["rationale"],
                "evidence_quote": after["evidence_quote"],
                "signals": signals,
                "evidence_count": item.get("evidence_count", 0),
            })

        relations = None
        if include_relations:
            props = await self.propose_relations(namespace)
            if "error" in props:
                return props   # 방어적 — namespace 는 위에서 이미 확인했다
            llm_calls += int(props.get("chunks_scanned") or 0)
            relations = {"strong_confirm": [], "borderline": []}
            for p in props.get("proposals", []):
                band = triage_relation(p)
                # 관계 추천의 verdict: strong_confirm 은 승인 예측(confirm),
                # borderline 은 예측 없음(unsure) — 일치율 자가 같은 축에서
                # 읽을 수 있어야 한다 (approve↔confirm 정규화).
                verdict = ("confirm" if band["band"] == "strong_confirm"
                           else "unsure")
                reviews.record(
                    action="recommend", node_id=p["subject"],
                    reason="; ".join(band["reasons"]),
                    after={"verdict": verdict, "band": band["band"],
                           "predicate": p["predicate"], "target": p["object"],
                           "chunk_id": p.get("chunk_id", ""),
                           "evidence_quote": p.get("evidence_quote", "")},
                    actor=actor or "triage")
                relations[band["band"]].append(
                    {**p, "band": band["band"], "reasons": band["reasons"]})

        counts: Dict[str, Any] = {
            "nodes": {k: len(v) for k, v in nodes_by_band.items()}}
        if relations is not None:
            counts["relations"] = {k: len(v) for k, v in relations.items()}
        return {"namespace": namespace, "nodes": nodes_by_band,
                "relations": relations, "counts": counts,
                "llm_calls": llm_calls}

    @staticmethod
    def _latest_node_recommendation(reviews, node_id: str
                                    ) -> Optional[Dict[str, Any]]:
        """최신 **노드** 추천 (관계 추천 제외). recommendation_for 를 못 쓰는
        이유: _recommended 는 node_id 당 최신 하나라, 같은 노드가 관계 제안의
        subject 로도 추천을 받으면 관계 추천이 노드 추천을 덮는다 — 판정
        재검증이 엉뚱한 추천을 보게 된다. history 를 뒤져 predicate 없는
        recommend 를 찾는다 (최신 먼저)."""
        for ev in reviews.history(node_id=node_id, limit=500):
            if ev.get("action") != "recommend":
                continue
            after = ev.get("after") or {}
            if after.get("predicate"):
                continue   # 관계 추천 — 노드 판정의 근거가 아니다
            return after
        return None

    def judge_batch(self, namespace: str, items: List[Dict[str, Any]],
                    actor: str = "", dry_run: bool = True) -> Dict[str, Any]:
        """트리아지 밴드의 일괄 판정 — 본문 불신 + dry_run 미리보기 (C1-c).

        **요청 본문을 신뢰하지 않는다** (approve_relations 와 같은 규율):
        노드 항목은 (a) 아직 미판정인지 (b) **최신 노드 추천의 verdict/band 가
        요청 verdict 와 일치**하는지 재검증한다 — 트리아지 이후 재검사로
        추천이 바뀌었으면(낡은 추천) 일괄 버튼이 그것을 조용히 덮으면 안 된다.
        통과분만 confirm_node/reject_node **기존 경로에 위임**한다 (판정 쓰기
        두 벌 금지). 관계 항목은 approve_relations/reject_relations 위임 —
        그쪽 관문이 인용 재검증을 한다.

        dry_run=True 기본 — **미리보기 상태 == 적용 상태** 가 계약이다
        (approve_structural 의 test_preview_equals_apply_statuses 패턴).
        같은 배치 안의 중복 항목도 미리보기가 적용과 같게 예측한다.
        """
        from ..core.lifecycle import can_destroy
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not items:
            return {"error": "invalid", "detail": "items must not be empty"}

        graph = get_knowledge_graph_engine(namespace).graph
        reviews = get_review_store(namespace)

        results: List[Dict[str, Any]] = []
        applied = 0
        skipped = 0
        # 배치 내 중복 예측 — 적용에서는 두 번째 항목이 already_* 로 걸리는데
        # 미리보기가 그걸 모르면 미리보기 ≠ 적용이 된다.
        batch_judged_nodes: set = set()
        batch_rejected_triples: set = set()
        batch_approved_triples: set = set()

        def _skip(base: Dict[str, Any], reason: str, **extra) -> None:
            nonlocal skipped
            skipped += 1
            results.append({**base, "status": "skipped",
                            "reason": reason, **extra})

        for raw in items:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "").strip()

            if kind == "node":
                node_id = str(raw.get("node_id") or "").strip()
                verdict = str(raw.get("verdict") or "").strip().lower()
                base = {"kind": "node", "node_id": node_id, "verdict": verdict}
                if not node_id or verdict not in ("confirm", "reject"):
                    _skip(base, "invalid")
                    continue
                if (node_id in batch_judged_nodes
                        or reviews.is_confirmed(node_id)
                        or reviews.is_rejected(node_id)):
                    _skip(base, "already_judged")
                    continue
                rec = self._latest_node_recommendation(reviews, node_id)
                if rec is None:
                    # 추천 없이 일괄 판정은 없다 — 개별 판정 API 를 쓰라
                    _skip(base, "no_recommendation")
                    continue
                expected_band = ("strong_confirm" if verdict == "confirm"
                                 else "strong_reject")
                if (str(rec.get("verdict") or "").strip().lower() != verdict
                        or rec.get("band") != expected_band):
                    _skip(base, "recommendation_mismatch",
                          recommended={"verdict": rec.get("verdict"),
                                       "band": rec.get("band")})
                    continue
                if node_id not in graph:
                    _skip(base, "node_not_found")
                    continue
                if verdict == "reject":
                    # 생애주기 관문을 미리보기에도 반영 — reject_node 가 적용
                    # 단계에서 거부할 계획을 미리보기가 통과시키면 안 된다.
                    allowed, why = can_destroy(dict(graph.nodes[node_id]))
                    if not allowed:
                        skipped += 1
                        results.append({**base, "status": "blocked",
                                        "reason": "lifecycle_protected",
                                        "detail": why})
                        continue
                batch_judged_nodes.add(node_id)
                if dry_run:
                    results.append({**base, "status": f"would_{verdict}"})
                    continue
                if verdict == "confirm":
                    r = self.confirm_node(namespace, node_id, actor=actor)
                    if r is None:
                        _skip(base, "node_not_found")
                    else:
                        applied += 1
                        results.append({**base, "status": "confirmed"})
                else:
                    reason = (str(raw.get("reason") or "").strip()
                              or str(rec.get("rationale") or "").strip())
                    r = self.reject_node(namespace, node_id, actor=actor,
                                         reason=reason)
                    if r is None or (isinstance(r, dict) and "error" in r):
                        skipped += 1
                        results.append({**base, "status": "blocked",
                                        "reason": (r or {}).get(
                                            "error", "node_not_found")})
                    else:
                        applied += 1
                        results.append({**base, "status": "rejected"})

            elif kind == "relation":
                subject = str(raw.get("subject") or "").strip()
                predicate = str(raw.get("predicate") or "").strip()
                obj = str(raw.get("object") or "").strip()
                verdict = str(raw.get("verdict") or "").strip().lower()
                base = {"kind": "relation", "subject": subject,
                        "predicate": predicate, "object": obj,
                        "verdict": verdict}
                triple = (subject, predicate, obj)

                if verdict == "approve":
                    if triple in batch_approved_triples:
                        _skip(base, "already_exists")
                        continue
                    r = self.approve_relations(
                        namespace,
                        [{"subject": subject, "predicate": predicate,
                          "object": obj,
                          "chunk_id": str(raw.get("chunk_id") or ""),
                          "evidence_quote": str(
                              raw.get("evidence_quote") or "")}],
                        actor=actor, dry_run=dry_run)
                    if "error" in r:
                        _skip(base, r["error"])
                    elif r.get("added_total"):
                        batch_approved_triples.add(triple)
                        if dry_run:
                            results.append({**base, "status": "would_approve"})
                        else:
                            applied += 1
                            results.append({**base, "status": "approved"})
                    else:
                        why = (r.get("skipped") or [{}])[0].get(
                            "reason", "unknown")
                        _skip(base, why)

                elif verdict == "reject":
                    # reject_relations 는 dry_run 이 없다(즉시 쓰기) — 미리보기는
                    # 같은 규칙(invalid/reason_required/already_rejected)으로
                    # 결과를 예측한다. 규칙 원본은 그쪽이다.
                    reason = str(raw.get("reason") or "").strip()
                    if not subject or not predicate or not obj:
                        _skip(base, "invalid")
                        continue
                    if not reason:
                        _skip(base, "reason_required")
                        continue
                    if (triple in batch_rejected_triples
                            or reviews.relation_rejection(subject, predicate,
                                                          obj)):
                        _skip(base, "already_rejected")
                        continue
                    batch_rejected_triples.add(triple)
                    if dry_run:
                        results.append({**base, "status": "would_reject"})
                        continue
                    r = self.reject_relations(
                        namespace,
                        [{"subject": subject, "predicate": predicate,
                          "object": obj, "reason": reason,
                          "chunk_id": str(raw.get("chunk_id") or ""),
                          "scope": str(raw.get("scope") or "triple")}],
                        actor=actor)
                    if r.get("rejected_total"):
                        applied += 1
                        results.append({**base, "status": "rejected"})
                    else:
                        why = (r.get("skipped") or [{}])[0].get(
                            "reason", "unknown")
                        _skip(base, why)
                else:
                    _skip(base, "invalid")

            else:
                skipped += 1
                results.append({"kind": kind or "?", "status": "skipped",
                                "reason": "unknown_kind"})

        return {"namespace": namespace, "dry_run": bool(dry_run),
                "results": results, "applied": applied, "skipped": skipped}

    def recommendation_quality(self, namespace: str) -> Dict[str, Any]:
        """일치율 자 — recommend ↔ 이후 사람 판정 (C2, LLM 0콜, 무저장).

        review_store 의 이벤트 **전량**을 core 순수 함수에 위임한다 —
        history(limit=…) 는 잘리므로 쓰지 않는다 (_events 가 로그 순서 전량,
        같은 초 안의 순서까지 보존된다). 재계산 가능한 파생물이라 저장하지
        않는다 (lint_consistency 와 같은 계약) — 그래서 GET 이다.
        """
        from dataclasses import asdict as _asdict

        from ..core.review_store import get_review_store
        from ..core.review_triage import recommendation_agreement

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        store = get_review_store(namespace)
        events = [_asdict(e) for e in store._events]
        return {"namespace": namespace, "events_total": len(events),
                **recommendation_agreement(events)}

    def find_orphan_nodes(self, namespace: str,
                          limit: int = 0) -> Dict[str, Any]:
        """근거 링크가 없는 노드 + 원문 인용 후보 (읽기, LLM 0콜).

        check_coverage 의 **대칭**이다 — 그쪽은 "노드가 없는 청크"를 묻고 이쪽은
        "청크가 없는 노드"를 묻는다. 후자를 아무도 묻지 않은 사이 고아 노드가
        14% 쌓였고, 그것이 evidence 채점의 hit@5 천장(어떤 검색 knob 으로도
        움직이지 않는 0.8125)을 정하고 있었다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.orphan_links import find_orphan_candidates
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        graph = get_knowledge_graph_engine(namespace).graph
        result = find_orphan_candidates(graph, get_chunk_store(namespace).all(),
                                        limit=limit)
        return {"namespace": namespace, **result}

    def approve_orphan_links(self, namespace: str,
                             links: List[Dict[str, Any]],
                             actor: str = "",
                             dry_run: bool = True) -> Dict[str, Any]:
        """고아 노드에 근거 청크를 잇는다 (회복 쓰기 경로).

        **요청 본문을 신뢰하지 않는다.** find_orphan_candidates 의 "shadow 를
        걸렀다"는 성질은 그 함수가 만든 것이고, 엔드포인트가 받는
        {node_id, chunk_id} 는 그 성질을 물려받지 않는다. 같은 검증을 다시
        통과시킨다 — 규칙을 두 벌 두면 두 경로의 '원문' 정의가 갈라진다:
        - 노드가 그래프에 실존해야 한다
        - 청크가 실존해야 한다
        - 노드 **이름**(그래프의 attrs 에서 읽는다 — 본문이 준 이름이 아니다)이
          그 청크 원문에 문자 그대로 있어야 한다
        - 더 긴 노드 이름에 가려지지 않아야 한다 (오추출 `상선암` 차단)

        approve_coverage_gaps 와 다른 두 가지:
        - **노드를 만들지 않는다.** 여기서 없는 것은 노드가 아니라 근거다.
        - **reindex_required 가 없다.** 노드 텍스트(compose_node_text 의 입력)가
          바뀌지 않으므로 시맨틱 색인은 영향받지 않는다. True 로 보고하면 불필요한
          임베딩 비용을 부른다.

        묘비된 노드는 잇지 않는다 — 링크를 되살리는 것은 그 노드를 검색에
        되돌리는 것이고, 검수자 모르게 판정을 뒤집는 일이다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.orphan_links import node_name, shadowing_names
        from ..core.evidence_checker import _quote_in_chunks
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not links:
            return {"error": "invalid", "detail": "links must not be empty"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        chunks = get_chunk_store(namespace)
        reviews = get_review_store(namespace)

        all_names = [node_name(nid, attrs)
                     for nid, attrs in graph.nodes(data=True)]

        linked: List[str] = []
        skipped: List[Dict[str, Any]] = []
        chunks_linked = 0
        seen: set = set()

        def _skip(node_id, chunk_id, reason):
            skipped.append({"node_id": node_id, "chunk_id": chunk_id,
                            "reason": reason})

        for link in links:
            if not isinstance(link, dict):
                continue
            node_id = str(link.get("node_id") or "").strip()
            chunk_id = str(link.get("chunk_id") or "").strip()
            if not node_id or not chunk_id:
                _skip(node_id, chunk_id, "invalid")
                continue
            # dedup 은 **(노드, 청크) 쌍** — node_id 로만 걸면 한 노드의 여러
            # 근거 중 첫 청크만 이어지고 나머지가 보고도 없이 사라진다
            # (approve_coverage_gaps 에서 실측된 결함: 123건 중 23건).
            if (node_id, chunk_id) in seen:
                continue
            seen.add((node_id, chunk_id))

            if node_id not in graph:
                _skip(node_id, chunk_id, "node_not_found")
                continue
            stored = chunks.get(chunk_id)
            if stored is None:
                _skip(node_id, chunk_id, "chunk_not_found")
                continue
            if reviews.is_rejected(node_id):
                _skip(node_id, chunk_id, "tombstoned")
                continue

            name = node_name(node_id, graph.nodes[node_id])
            text = stored.text or ""
            if not name or not _quote_in_chunks(name, [text]):
                _skip(node_id, chunk_id, "quote_not_found")
                continue
            if shadowing_names(name, all_names, text):
                _skip(node_id, chunk_id, "shadowed")
                continue
            if node_id in stored.node_ids:
                _skip(node_id, chunk_id, "already_linked")
                continue

            if node_id not in linked:
                linked.append(node_id)
            if dry_run:
                # 미리보기는 적용과 **같은 수**여야 한다 — 위 검증을 모두 통과한
                # 뒤에만 센다(already_linked 는 위에서 이미 걸러졌다).
                chunks_linked += 1
                continue
            if chunks.link_node(chunk_id, node_id):
                chunks_linked += 1
            reviews.record(action="orphan_link_approve", node_id=node_id,
                           after={"chunk_id": chunk_id, "name": name,
                                  "section": getattr(stored, "section", "")},
                           actor=actor or "admin", source=stored.source)

        if not dry_run and chunks_linked:
            chunks.save_to_disk()

        return {"namespace": namespace, "dry_run": bool(dry_run),
                "linked": linked, "skipped": skipped,
                "chunks_linked": chunks_linked,
                # 노드 텍스트가 안 바뀌므로 색인은 그대로 유효하다.
                "reindex_required": False}

    def confirm_node(self, namespace: str, node_id: str,
                     actor: str = "") -> Optional[Dict[str, Any]]:
        """확정 — 판정 + 감사 기록(before=판정 시점 attrs 스냅샷).
        없는 노드는 None (라우터가 404 로 옮긴다)."""
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        graph = get_knowledge_graph_engine(namespace).graph
        if node_id not in graph:
            return None
        get_review_store(namespace).confirm(
            node_id, actor=actor, before=dict(graph.nodes[node_id]))
        return {"namespace": namespace, "node_id": node_id,
                "status": "confirmed"}

    def reject_node(self, namespace: str, node_id: str, actor: str = "",
                    reason: str = "") -> Optional[Dict[str, Any]]:
        """거절 = 묘비 + 그래프에서 노드·간선 제거.

        순서가 중요하다: **묘비 먼저, 제거 나중** — 제거 후 묘비 기록 전에
        죽으면 노드는 사라졌지만 재인제스트에서 부활한다. 반대 순서면 최악이
        "묘비는 있는데 노드가 남은" 상태고, 그건 다음 조회에서 무해하다.
        """
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        if node_id not in graph:
            return None

        before = dict(graph.nodes[node_id])
        # 생애주기 관문 — active 노드는 삭제할 수 없다. 거절이 정확성 판정이라
        # 해도, 운영이 물린 노드를 지우는 것은 별개 결정이다(먼저 deprecated).
        from ..core.lifecycle import can_destroy
        allowed, why = can_destroy(before)
        if not allowed:
            return {"error": "lifecycle_protected", "detail": why,
                    "node_id": node_id, "lifecycle": before.get("lifecycle")}

        edges_removed = graph.degree(node_id)
        get_review_store(namespace).reject(
            node_id, reason=reason, actor=actor, before=before)
        graph.remove_node(node_id)
        self._pg_apply(namespace, lambda s: s.delete_node(namespace, node_id))

        # 시맨틱 인덱스에서도 내린다 — 남겨두면 삭제된 노드가 검색 top_k 를
        # 차지한다. 인덱스가 없거나 실패해도 거절 자체는 성립한다.
        try:
            index = getattr(engine, "_semantic_index", None)
            if index is not None and hasattr(index, "remove"):
                index.remove(node_id)
        except Exception as e:
            logger.warning(f"⚠️ Semantic index removal failed ({node_id}): {e}")

        # 제거를 체크포인트에 내린다 — 묘비는 재빌드의 부활을 막지만,
        # 저장 안 된 그래프는 재시작만으로 노드가 되돌아온다.
        engine.save_to_disk()
        return {"namespace": namespace, "node_id": node_id,
                "status": "rejected", "edges_removed": edges_removed}

    def get_review_history(self, namespace: str,
                           node_id: Optional[str] = None,
                           limit: int = 100) -> Dict[str, Any]:
        from ..core.review_store import get_review_store
        return {"namespace": namespace,
                "history": get_review_store(namespace).history(
                    node_id=node_id, limit=limit)}

    def retrieve(self, namespace: str, query: str,
                 top_k: int = 5, source: Optional[str] = None) -> Dict[str, Any]:
        """그래프-조건부 검색 (축 4) — 온톨로지로 조건화된 원문 검색.

        search_chunks 와의 차이: 질의를 온톨로지로 확장하고(별칭·계층·인접),
        노드에 달린 청크를 두 번째 채널로 합친다. 임베딩만으로 못 찾는 것을
        개념을 거쳐 데려온다.

        응답에 expansion 을 함께 실어 **왜 이 결과인지** 드러낸다 — 확장
        어휘와 진입 노드가 보이지 않으면 사용자는 결과를 신뢰할 근거가 없다.

        `source` 는 문서 필터("이 문서에서만") — 근거만 거르고 온톨로지 확장은
        네임스페이스 전체를 쓴다 (graph_retrieval.search 의 계약).
        """
        from ..core.eval_history import latest_quality as _latest_quality
        from ..core.graph_retrieval import GraphConditionedRetriever
        from ..core.retrieval_config import get_retrieval_config
        retriever = GraphConditionedRetriever(namespace=namespace)
        # **네임스페이스별 설정을 적용한다.** 종전에는 전역 상수만 써서 확산 채널을
        # 라이브에서 켤 방법이 없었다 — 확산의 가치가 커버리지에 따라 정반대로
        # 측정됐는데(ins_cancer_demo 동률 / PROJ-A +2~3건) 반영할 길이 없었다.
        cfg = get_retrieval_config(namespace).effective()
        expansion = retriever.expand(query, **{k: v for k, v in cfg.items()
                                               if k != "propagation_channel"
                                               and k != "propagation_weight"})
        hits = retriever.search(query, top_k=top_k, source=source, **cfg)
        return {
            "namespace": namespace,
            "query": query,
            "expansion": {
                "expanded_query": expansion.expanded_query,
                "terms": expansion.terms,
                "entry_nodes": expansion.entry_nodes,
                "expanded_nodes": expansion.expanded_nodes,
            },
            "hits": [{**self._chunk_view(h.chunk), "score": h.score,
                      "matched_via": h.matched_via, "channels": h.channels}
                     for h in hits],
            # 이 검색기의 **측정된** 품질 — 골든셋 지표 + 측정 시점 + stale 여부.
            # 히트별 score(RRF)·best_evidence(코사인)는 보정되지 않은 값이라
            # "신뢰도"로 포장하지 않는다 — 지어내지 않는 것이 계약이다.
            "quality": _latest_quality(namespace),
        }

    # ─── Admin (관리 콘솔 — 통계 · 묘비 · 삭제) ──────────────────────

    # 네임스페이스 하나가 디스크에 남기는 파일들. 전부 같은 data/ 디렉터리에
    # 산다 (review_store 의 "청크·KG 체크포인트와 같은 디렉터리" 규약).
    _NS_FILE_PATTERNS = ("kg_{ns}.json", "chunks_{ns}.jsonl",
                         "reviews_{ns}.jsonl", "goldenset_{ns}.jsonl",
                         "vectors_{ns}.npy", "vectors_{ns}.json")

    @staticmethod
    def _ns_files(namespace: str) -> List[Path]:
        from ..engines.knowledge_graph_clean import _DEFAULT_DATA_DIR
        base = "kg_checkpoint" if namespace == "default" else None
        files = []
        for pattern in OntologyBuilderService._NS_FILE_PATTERNS:
            name = pattern.format(ns=namespace)
            if base and pattern.startswith("kg_"):
                name = f"{base}.json"  # default 만 역사적 예외 파일명
            files.append(_DEFAULT_DATA_DIR / name)
        return files

    def _graph_store(self, namespace: str):
        """인스턴스 읽기 seam(축 5). ONTOLOGY_GRAPH_BACKEND 에 따라
        InMemoryGraphStore(NetworkX, 기본) 또는 PostgresGraphStore 를 돌려준다.
        postgres 선택이나 접속 불가 시 memory 로 degrade(팩토리가 처리)."""
        from ..core.graph_store import create_graph_store
        return create_graph_store(namespace, count_cap=self.COUNT_CAP)

    def _pg_apply(self, namespace: str, fn) -> None:
        """수동 변경(P4-b)을 PG(진실)에도 이중기록. PG-backed 이고 접속 가능할
        때만 — 읽기가 PG 에서 오므로 create/delete/rename 이 PG 에 반영돼야
        일관적이다. PG 불가 시 그 네임스페이스는 memory 로 degrade(읽기도 KG)
        되므로 KG 변경만으로 일관. 실패는 ERROR 로 남긴다(KG↔PG 어긋남 신호)."""
        try:
            from ..core import graph_store, pg
            if graph_store.aicoach_source(namespace):
                # aicoach 라이브 스토어 직접 소비 네임스페이스 → aicoach 테이블에
                # 쓴다(AicoachGraphStore, ONTOLOGY_AICOACH_WRITE gated). ontology
                # 스키마(PostgresGraphStore)가 아니라 create_graph_store 로 라우팅.
                fn(graph_store.create_graph_store(namespace))
            elif graph_store.pg_backed(namespace) and pg.available():
                fn(graph_store.PostgresGraphStore(namespace, pg.get_schema()))
        except Exception as e:
            logger.error(f"❌ PG 이중기록 실패({namespace}) — PG 가 KG 와 어긋날 수 있음: {e}")

    def _mirror_to_pg(self, namespace: str) -> None:
        """빌드/인제스트 성공 후 백엔드 미러(축 5, P2 PG + P3 ES) — **best-effort**.

        PG 대상 네임스페이스(허용목록/전역)이고 접속 가능할 때만 PG(진실)에
        미러하고, 이어서 ES 객체 인덱스(검색·파셋 투영)에도 투영한다. 어느
        미러 실패도 절대 빌드를 깨뜨리지 않는다(WARNING 만)."""
        from ..core import graph_store, pg
        if not graph_store.pg_backed(namespace):
            return
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        graph = get_knowledge_graph_engine(namespace).graph
        try:
            if pg.available():
                pg.ensure_schema()
                graph_store.sync_from_graph(namespace, graph)
        except Exception as e:
            logger.warning(f"⚠️ PG 미러 실패(무시, 빌드는 성공): {namespace} — {e}")
        # ES 투영은 PG 게이트와 무관하다 — _index_namespace 로 이관(신규
        # 네임스페이스도 검색 인덱스를 갖도록). 여기선 PG(진실)만 다룬다.

    def index_namespace(self, namespace: str) -> Dict[str, Any]:
        """검색 인덱스 채우기 — ES 노드 인덱스 + 청크 벡터 인덱스. **best-effort**,
        **PG 게이트와 무관**(신규 네임스페이스도 대상). 업로드/빌드 직후 호출해
        "업로드 → 즉시 검색가능" 을 보장한다. 원문·그래프는 이미 저장됐으므로
        인덱싱 실패가 빌드를 깨지 않는다 — 인덱스는 파생 데이터다.

        aicoach 가 '업로드=벡터DB 적재' 를 1급 단계로 둔 것과 대응 — 여기서는
        온톨로지(KG) 는 이미 만들어졌고, 그 위에 벡터·노드 검색을 얹는다."""
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        out: Dict[str, Any] = {"es_nodes": None, "chunk_vectors": None,
                              "node_vectors": None}
        try:
            from ..core import object_index
            out["es_nodes"] = object_index.project_graph(namespace, graph)
        except Exception as e:
            logger.warning(f"⚠️ ES 노드 투영 실패(무시): {namespace} — {e}")
        try:
            # semantic 채널의 색인. 이게 빠져 있어서 노드 편집·병합 후
            # /reindex 를 돌려도 semantic 이 그래프와 어긋난 채 남았다 —
            # 서버 재시작만이 유일한 복구 경로였다.
            out["node_vectors"] = engine.rebuild_semantic_index()
        except Exception as e:
            logger.warning(f"⚠️ 노드 벡터 재색인 실패(무시): {namespace} — {e}")
        try:
            from ..core.chunk_index import get_chunk_index
            out["chunk_vectors"] = get_chunk_index(namespace).refresh()
        except Exception as e:
            logger.warning(f"⚠️ 청크 벡터 인덱싱 실패(무시): {namespace} — {e}")
        return out

    def index_status(self, namespace: str) -> Optional[Dict[str, Any]]:
        """검색 인덱스 상태 — 관리 콘솔 관측용(원시 ES 노출 아님). ES 노드
        인덱스 문서수·그래프와의 동기 여부 + 원문 청크 수 + 벡터검색 가용성."""
        if not self._namespace_exists(namespace):
            return None
        nodes = self._graph_store(namespace).aggregate(namespace).get("nodes", 0)
        es = {"available": False, "node_docs": None, "index": None}
        try:
            from ..core.object_index import ObjectIndex
            idx = ObjectIndex(namespace)
            es["index"] = idx.index
            if idx.available():
                es["available"] = True
                es["node_docs"] = (idx.client.count(index=idx.index)["count"]
                                   if idx.client.indices.exists(index=idx.index) else 0)
        except Exception as e:
            logger.warning(f"⚠️ 인덱스 상태 조회 실패(ES): {namespace} — {e}")
        chunks = 0
        embedder = False
        try:
            from ..core.chunk_store import get_chunk_store
            from ..core.chunk_index import get_chunk_index
            chunks = len(get_chunk_store(namespace).all())
            embedder = get_chunk_index(namespace)._has_embedder()
        except Exception as e:
            logger.warning(f"⚠️ 인덱스 상태 조회 실패(청크): {namespace} — {e}")
        return {
            "namespace": namespace,
            "nodes": nodes,
            "chunks": chunks,
            "es_available": es["available"],
            "es_index": es["index"],
            "node_docs": es["node_docs"],
            "node_in_sync": (es["node_docs"] == nodes
                             if es["node_docs"] is not None else None),
            "vector_search": embedder,   # 임베더 있으면 청크 의미검색 가능
            # **어떤 모델로 임베딩됐는가** — 이게 없으면 화면의 점수가 어떤
            # 설정에서 나온 것인지 알 수 없다. 노드와 청크는 다른 모델을 쓸 수
            # 있고(resolve_chunk_model), 그 조합이 지표를 크게 움직인다.
            # 네임스페이스를 넘긴다 — 이 화면은 **이 네임스페이스**의 상태다.
            "embedding": self.embedding_info(namespace),
        }

    def embedding_info(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """현재 임베딩 설정 — 관리 콘솔 표시용. 모델을 **로드하지 않는다**
        (설정만 읽는다): 상태 조회가 3.5s 짜리 로드를 유발하면 안 된다.

        `namespace` 를 주면 그 네임스페이스의 **실효 검색 설정**을 보여준다.
        종전에는 인자 없이 불러 전역 기본값을 표시했다 — PROJ-A 은 확산을
        켜 두었는데(`use_propagation: true`) 화면은 false 로 보고했다.
        "지금 무엇이 쓰이는가"를 틀리게 보여주는 것은 이 저장소가 가장
        경계하는 결함류다 (측정과 라이브가 같은 설정이어야 한다).
        """
        from ..core.eval_history import config_fingerprint
        from ..core.semantic_index import (resolve_chunk_model,
                                           resolve_embedding_dim,
                                           resolve_node_model)
        node_model = resolve_node_model()
        chunk_model = resolve_chunk_model()
        # 커널의 추론표를 쓴다 — ml.config 를 부르면 torch 가 딸려 와서 상태
        # 조회 한 번에 수십 초가 든다(실제로 밟았다).
        dim = resolve_embedding_dim(node_model)
        fingerprint = config_fingerprint(namespace) if namespace \
            else config_fingerprint()
        return {
            "node_model": node_model,
            "chunk_model": chunk_model,
            "unified": node_model == chunk_model,
            "dim": dim,
            "retrieval": {k: v for k, v in fingerprint.items()
                          if k not in ("node_model", "chunk_model")},
        }

    def migrate_namespace_to_pg(self, namespace: str) -> Dict[str, Any]:
        """기존 네임스페이스를 PG 로 백필 — 운영자의 명시적 이전 액션.

        허용목록/전역 설정과 무관하게 즉시 미러한다(읽기 전환 전 데이터를 먼저
        채우는 용도). 이후 ONTOLOGY_PG_NAMESPACES 에 추가하면 읽기가 PG 로 간다."""
        from ..core import graph_store, pg
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not pg.available():
            return {"error": "pg_unavailable",
                    "detail": "ONTOLOGY_PG_DSN 미설정/접속 불가"}
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
        graph = get_knowledge_graph_engine(namespace).graph
        pg.ensure_schema()
        counts = graph_store.sync_from_graph(namespace, graph)
        # ES 객체 인덱스 투영(검색·파셋) — best-effort
        es_docs = None
        try:
            from ..core import object_index
            es_docs = object_index.project_graph(namespace, graph)
        except Exception as e:
            logger.warning(f"⚠️ ES 객체 투영 실패(무시): {namespace} — {e}")
        return {"namespace": namespace, "migrated": True,
                "es_docs": es_docs, **counts}

    def object_search(self, namespace: str, q: Optional[str] = None,
                      node_type: Optional[str] = None, trust: Optional[str] = None,
                      top_k: int = 50, offset: int = 0) -> Optional[Dict[str, Any]]:
        """객체 검색(축 5, P3) — ES 객체 인덱스로 BM25 관련도 + 타입/신뢰 파셋.

        ES 인덱스가 있으면 그쪽(source='es', 파셋 포함), 없으면 list_nodes 의
        substring 검색으로 폴백(source='fallback', 파셋 없음). 브라우즈(정확 필터
        +페이지네이션)와 달리 여기는 랭킹 검색이라 payload 가 백엔드마다 다를 수
        있다 — 그게 목적(더 나은 검색)."""
        if not self._namespace_exists(namespace):
            return None
        store = self._graph_store(namespace)
        try:
            from ..core.object_index import ObjectIndex
            idx = ObjectIndex(namespace)
            if idx.available() and idx.client.indices.exists(index=idx.index):
                r = {**idx.search(q=q, node_type=node_type, trust=trust,
                                  top_k=top_k, offset=offset), "source": "es"}
                self._mark_classes(store, namespace, r)
                return r
        except Exception as e:
            logger.warning(f"⚠️ ES 객체 검색 실패, 폴백: {namespace} — {e}")
        # 폴백: PG/memory substring
        base = store.list_nodes(
            namespace, q=q, node_type=node_type, trust=trust,
            offset=offset, limit=top_k)
        r = {"namespace": namespace, "total": base["total"],
             "items": [{"node_id": i["node_id"], "name": i["name"],
                        "type": i["type"], "trust": i["trust"], "score": None}
                       for i in base["items"]],
             "facets": None, "offset": offset, "limit": top_k,
             "source": "fallback"}
        self._mark_classes(store, namespace, r)
        return r

    def _namespace_exists(self, namespace: str) -> bool:
        """알려진 네임스페이스인가 — 로드됐거나 디스크 파일이 있거나.

        stats/delete 는 이 게이트를 먼저 통과해야 한다: 미지 이름으로
        get_knowledge_graph_engine 을 부르면 빈 엔진이 생성·등록돼 조회
        자체가 유령 네임스페이스를 만든다.
        """
        from ..engines.knowledge_graph_clean import _kg_instances
        if namespace in _kg_instances:
            return True
        if any(f.exists() for f in self._ns_files(namespace)):
            return True
        # 프로젝트 레코드만 있고 아직 그래프가 없는 '빈 프로젝트'도 존재한다
        # (생성 직후 온보딩 화면이 열려야 하므로 stats/schema 가 404 면 안 됨).
        return self.projects.get(namespace) is not None

    def _namespace_entry(self, namespace: str) -> Dict[str, Any]:
        """overview 한 행 — 그래프·청크·검수·trust·디스크를 한 번에.

        그래프 집계는 GraphStore seam 에 위임(축 5 P4-b): PG 백엔드는 인덱스
        GROUP BY 로 세므로 **전체 그래프를 RAM 에 로드하지 않는다**(수천만 대비
        핵심 — 개요 한 행이 OOM 을 부르던 경로 제거). memory 백엔드는 기존대로
        그래프를 순회한다."""
        from ..core.chunk_store import get_chunk_store
        from ..core.review_store import get_review_store
        from ..core.search_qa import get_golden_set

        store = self._graph_store(namespace)
        agg = store.aggregate(namespace)
        reviews = get_review_store(namespace)
        review = store.review_counts(namespace, reviews.confirmed_ids(),
                                     reviews.rejected_ids())
        disk_bytes = sum(f.stat().st_size
                         for f in self._ns_files(namespace) if f.exists())
        return {
            "namespace": namespace,
            "protected": namespace in PROTECTED_NAMESPACES,
            "nodes": agg["nodes"],
            "edges": agg["edges"],
            "chunks": len(get_chunk_store(namespace).all()),
            "golden_cases": len(get_golden_set(namespace).cases()),
            "review": review,
            "trust": agg["trust"],
            "disk_bytes": disk_bytes,
        }

    def get_admin_overview(self) -> Dict[str, Any]:
        """관리 대시보드 한 콜 — 모든 네임스페이스의 관리 요약.

        네임스페이스 하나가 깨져도(체크포인트 손상 등) 대시보드 전체가
        죽으면 안 된다 — 그 행만 error 로 degrade 한다.
        """
        graph_names = {m["namespace"] for m in self.list_namespaces()}
        project_recs = {p["namespace"]: p for p in self.projects.list()}
        entries: List[Dict[str, Any]] = []
        for name in sorted(graph_names | set(project_recs)):
            try:
                if name in graph_names:
                    entry = self._namespace_entry(name)
                else:
                    # 프로젝트 레코드만 있는 빈 프로젝트 — 엔진을 만들지 않는다
                    # (유령 방지). 0 으로 채운 empty 엔트리로 목록에만 띄운다.
                    entry = {
                        "namespace": name,
                        "protected": name in PROTECTED_NAMESPACES,
                        "nodes": 0, "edges": 0, "chunks": 0, "golden_cases": 0,
                        "review": {"confirmed": 0, "rejected": 0, "pending": 0},
                        "trust": {}, "disk_bytes": 0, "empty": True,
                    }
                rec = project_recs.get(name)
                if rec:
                    entry["description"] = rec["description"]
                    entry["domain"] = rec["domain"]
                    entry["created_at"] = rec["created_at"]
                entries.append(entry)
            except Exception as e:
                logger.warning(f"⚠️ Admin overview degraded for '{name}': {e}")
                entries.append({"namespace": name,
                                "protected": name in PROTECTED_NAMESPACES,
                                "error": str(e)})
        ok = [e for e in entries if "error" not in e]
        totals = {
            "namespaces": len(entries),
            "nodes": sum(e["nodes"] for e in ok),
            "edges": sum(e["edges"] for e in ok),
            "chunks": sum(e["chunks"] for e in ok),
            "pending_reviews": sum(e["review"]["pending"] for e in ok),
            "disk_bytes": sum(e["disk_bytes"] for e in ok),
        }
        return {"namespaces": entries, "totals": totals}

    def system_overview(self) -> Dict[str, Any]:
        """시스템 개요 — 저장 백엔드(ES/PG/VectorDB)·청크·파일의 실물 지표.

        각 섹션은 독립적으로 degrade 한다: ES 가 죽어도 PG 는 살고, 어느 하나가
        없다고 500 을 내지 않는다({available:false}). 관리자는 "무엇이 살아있고
        무엇이 죽었나"를 한 화면에서 봐야 한다 — 부분 장애가 전체 실명을 낳으면
        안 된다.
        """
        return {
            "es": self._system_es(),
            "pg": self._system_pg(),
            "vector": self._system_vector(),
            "files": self._system_files(),
            "namespaces": self.get_admin_overview(),
        }

    # ─── system_overview 섹션 (각자 독립 degrade) ──────────────────

    def _system_es(self) -> Dict[str, Any]:
        """ES 클러스터 상태 + object 인덱스별 문서 수/용량 + 임베딩 모델."""
        import json as _json
        import urllib.request

        try:
            from ..core.es_backend import DEFAULT_ES_URL, DEFAULT_INDEX_PREFIX
        except Exception:
            DEFAULT_ES_URL = os.environ.get("ONTOLOGY_ES_URL", "http://localhost:9200")
            DEFAULT_INDEX_PREFIX = os.environ.get("ONTOLOGY_ES_INDEX_PREFIX", "ontology")
        es_url = os.environ.get("ONTOLOGY_ES_URL", DEFAULT_ES_URL).rstrip("/")

        def _get(path: str):
            with urllib.request.urlopen(f"{es_url}{path}", timeout=4) as r:
                return _json.loads(r.read().decode("utf-8"))

        try:
            model, dim = self._embedding_model_dim()
            health = _get("/_cluster/health")
            rows = _get(
                f"/_cat/indices/{DEFAULT_INDEX_PREFIX}-obj*"
                "?format=json&h=index,docs.count,store.size,pri.store.size,store.size.bytes"
            )
            indices, total_docs, total_bytes = [], 0, 0
            for row in rows:
                docs = int(row.get("docs.count") or 0)
                # _cat 는 사람이 읽는 크기("629.5kb")를 주므로 우리가 파싱한다
                size_bytes = self._parse_es_size(row.get("store.size"))
                indices.append({
                    "index": row.get("index"),
                    "docs": docs,
                    "size": row.get("store.size"),
                    "size_bytes": size_bytes,
                })
                total_docs += docs
                total_bytes += size_bytes
            indices.sort(key=lambda x: x["size_bytes"], reverse=True)
            return {
                "available": True,
                "url": es_url,
                "cluster_status": health.get("status"),
                "nodes": health.get("number_of_nodes"),
                "active_shards": health.get("active_shards"),
                "unassigned_shards": health.get("unassigned_shards"),
                "embedding_model": model,
                "dim": dim,
                "index_options": "int8_hnsw m=16",
                "indices": indices,
                "total_docs": total_docs,
                "total_size_bytes": total_bytes,
            }
        except Exception as e:
            logger.warning(f"⚠️ system_overview ES degraded: {e}")
            return {"available": False, "error": str(e)}

    @staticmethod
    def _parse_es_size(s) -> int:
        """_cat 인덱스 크기 문자열('629.5kb', '1.2gb') → bytes."""
        if not s:
            return 0
        s = str(s).strip().lower()
        units = {"tb": 1024 ** 4, "gb": 1024 ** 3, "mb": 1024 ** 2, "kb": 1024, "b": 1}
        for u, mul in units.items():
            if s.endswith(u):
                try:
                    return int(float(s[: -len(u)]) * mul)
                except ValueError:
                    return 0
        try:
            return int(float(s))
        except ValueError:
            return 0

    def _embedding_model_dim(self) -> Tuple[str, int]:
        """임베딩 모델명 + 차원. 모델명은 semantic_index, 차원은 실제 캐시 메타
        에서 읽고(모델이 곧 진실), 캐시가 없으면 알려진 기본(768)."""
        try:
            from ..core.semantic_index import DEFAULT_EMBEDDING_MODEL as model
        except Exception:
            model = "jhgan/ko-sroberta-nli"
        dim = 768
        try:
            import json as _json
            for meta in self._vector_cache_dir().glob("vectors_*.json"):
                d = _json.loads(meta.read_text())
                if d.get("dim"):
                    dim = int(d["dim"])
                    break
        except Exception:
            pass
        return model, dim

    def _system_pg(self) -> Dict[str, Any]:
        """PG 노드/엣지 총계 + 테이블 크기 + 일별 성장(성장 라인차트 원천)."""
        try:
            from ..core import pg
        except Exception as e:
            return {"available": False, "error": f"pg module: {e}"}
        if not pg.available():
            return {"available": False, "error": "PG unavailable"}
        try:
            schema = pg.get_schema()
            with pg.connect() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT count(*) FROM {schema}.node")
                total_nodes = cur.fetchone()[0]
                cur.execute(f"SELECT count(*) FROM {schema}.edge")
                total_edges = cur.fetchone()[0]
                cur.execute(
                    "SELECT pg_total_relation_size(%s), pg_total_relation_size(%s)",
                    (f"{schema}.node", f"{schema}.edge"),
                )
                node_sz, edge_sz = cur.fetchone()
                cur.execute(
                    f"SELECT namespace, count(*) FROM {schema}.node "
                    "GROUP BY namespace ORDER BY 2 DESC"
                )
                by_ns = [{"namespace": r[0], "count": r[1]} for r in cur.fetchall()]
                cur.execute(
                    f"SELECT date_trunc('day', updated_at) d, count(*) "
                    f"FROM {schema}.node WHERE updated_at IS NOT NULL "
                    "GROUP BY d ORDER BY d"
                )
                growth = [
                    {"date": r[0].strftime("%Y-%m-%d") if r[0] else None, "count": r[1]}
                    for r in cur.fetchall()
                ]
            return {
                "available": True,
                "schema": schema,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "table_sizes": {"node": int(node_sz or 0), "edge": int(edge_sz or 0)},
                "nodes_by_namespace": by_ns,
                "growth": growth,
            }
        except Exception as e:
            logger.warning(f"⚠️ system_overview PG degraded: {e}")
            return {"available": False, "error": str(e)}

    def _vector_cache_dir(self) -> Path:
        """npy 벡터 캐시 디렉터리 — data_dir(datasets)의 부모(=data/)."""
        return self.data_dir.parent

    def _system_vector(self) -> Dict[str, Any]:
        """디스크 npy 벡터 캐시(용량=capacity) + 청크 설정 + 백엔드 tier."""
        import json as _json

        try:
            from ..core.ingest_config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
        except Exception:
            DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP = 800, 120
        model, dim = self._embedding_model_dim()
        try:
            caches, total_bytes = [], 0
            for npy in sorted(self._vector_cache_dir().glob("vectors_*.npy")):
                size = npy.stat().st_size
                total_bytes += size
                # vectors_chunks_AI-Coach.npy → namespace 표시
                ns = npy.stem[len("vectors_"):]
                vectors = None
                meta = npy.with_suffix(".json")
                if meta.exists():
                    try:
                        vectors = len(_json.loads(meta.read_text()).get("ids", []))
                    except Exception:
                        pass
                caches.append({
                    "namespace": ns,
                    "size_bytes": size,
                    "vectors": vectors,
                })
            caches.sort(key=lambda c: c["size_bytes"], reverse=True)
            return {
                "available": True,
                "embedding_model": model,
                "dim": dim,
                # 노드·청크 채널이 **다른 모델**을 쓸 수 있다. 실측에서 그 조합이
                # 지표를 크게 움직였고(융합 MRR 0.8333 ↔ 0.8732 ↔ 0.8167),
                # 화면에 안 보이면 점수가 어떤 설정에서 나온 것인지 알 수 없다.
                "embedding": self.embedding_info(),
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "overlap": DEFAULT_OVERLAP,
                "tiers": {"memory": "<2K nodes", "npy": "2K+ (disk cache)",
                          "elasticsearch": "explicit (hybrid)"},
                "caches": caches,
                "total_size_bytes": total_bytes,
            }
        except Exception as e:
            logger.warning(f"⚠️ system_overview vector degraded: {e}")
            return {"available": False, "error": str(e)}

    def _system_files(self) -> Dict[str, Any]:
        """업로드 데이터셋/파일 수 (list_datasets 재사용)."""
        try:
            datasets = self.list_datasets()
            return {
                "available": True,
                "datasets": len(datasets),
                "total_files": sum(len(d.get("files", [])) for d in datasets),
            }
        except Exception as e:
            logger.warning(f"⚠️ system_overview files degraded: {e}")
            return {"available": False, "error": str(e)}

    def get_namespace_stats(self, namespace: str) -> Optional[Dict[str, Any]]:
        """관리 상세 — overview 행 + 타입/술어 분포 + 파일 목록.

        미지 네임스페이스는 None (라우터가 404 로 옮긴다) — 조회가 유령을
        만들지 않는다."""
        if not self._namespace_exists(namespace):
            return None
        entry = self._namespace_entry(namespace)
        # 타입/술어 분포도 seam 위임 — PG 는 GROUP BY, 전체 로드 없음(P4-b)
        dists = self._graph_store(namespace).distributions(namespace)
        entry["node_types"] = dists["node_types"]
        entry["predicates"] = dists["predicates"]
        entry["files"] = [
            {"name": f.name, "bytes": f.stat().st_size}
            for f in self._ns_files(namespace) if f.exists()]
        return entry

    def list_tombstones(self, namespace: str) -> Dict[str, Any]:
        """묘비 목록 — 거절된 노드의 사유·시점·판정자.

        부활 차단은 열람 가능해야 신뢰받는다: "왜 이 개체가 없는가"에
        관리자가 답할 수 있어야 한다.
        """
        from ..core.review_store import get_review_store
        return {"namespace": namespace,
                "tombstones": get_review_store(namespace).rejected()}

    def delete_namespace(self, namespace: str,
                         actor: str = "") -> Optional[Dict[str, Any]]:
        """네임스페이스 완전 삭제 — 디스크 파일 + 인메모리 싱글턴.

        protected 는 라우터가 403 으로 먼저 막는다(여기 도달 안 함).
        이름 검증이 첫 게이트다 — 경로 문자가 섞인 이름으로 data/ 밖을
        지우는 사고를 막는다. ES 인덱스(ontology-{ns})는 범위 밖 —
        ES 는 옵션 백엔드고, 남은 인덱스는 무해하다(다음 빌드가 덮는다).

        순서: 파일 먼저, 메모리 나중 — 반대면 파일 삭제 실패 시 "메모리엔
        없는데 디스크에 남아 재시작에 부활"하는 상태가 된다.

        감사는 **네임스페이스-독립 싱크**(ADMIN_AUDIT_NAMESPACE)에 남긴다:
        `reviews_{ns}.jsonl` 이 삭제 대상에 포함되므로 자기 로그에 적으면
        기록과 증거가 함께 사라진다. 지워진 뒤에 "누가 언제 지웠나"에
        답할 수 있어야 한다.
        """
        import re

        from ..core.review_store import ADMIN_AUDIT_NAMESPACE, get_review_store
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", namespace):
            return None
        if not self._namespace_exists(namespace):
            return None
        # 삭제 전에 규모를 읽어 둔다 — 지운 뒤에는 셀 수 없다.
        from ..engines.knowledge_graph_clean import _kg_instances
        _engine = _kg_instances.get(namespace)
        node_count = _engine.graph.number_of_nodes() if _engine is not None else None

        deleted: List[str] = []
        for f in self._ns_files(namespace):
            if f.exists():
                f.unlink()
                deleted.append(f.name)

        from ..core.chunk_index import _indices
        from ..core.chunk_store import _stores as _chunk_stores
        from ..core.review_store import _stores as _review_stores
        from ..core.search_qa import _golden_sets
        for registry in (_kg_instances, _chunk_stores, _indices,
                         _review_stores, _golden_sets):
            registry.pop(namespace, None)

        # 저장된 탐색도 함께 정리 — 삭제된 네임스페이스의 뷰가 유령으로 남으면
        # 같은 이름 재빌드 시 엉뚱한 필터가 되살아난다.
        self.saved_views.delete_namespace(namespace)
        self.schema_decl.delete_namespace(namespace)
        self.projects.delete(namespace)

        get_review_store(ADMIN_AUDIT_NAMESPACE).record(
            action="namespace_deleted", node_id=f"namespace:{namespace}",
            before={"deleted_files": deleted, "nodes": node_count},
            actor=actor or "admin")
        logger.info(f"🗑️ Namespace deleted: {namespace} "
                    f"({len(deleted)} files) by {actor or 'admin'}")
        return {"namespace": namespace, "deleted_files": deleted}

    # ─── Admin: 저장된 탐색 (SQLite 영속) ────────────────────────────

    @property
    def saved_views(self):
        """SavedViewStore 지연 생성 — admin.db 는 data_dir 안에 둔다
        (네임스페이스별 테스트가 tmp_path 로 격리된다)."""
        if self._saved_views is None:
            from .saved_views import SavedViewStore
            self._saved_views = SavedViewStore(self.data_dir / "admin.db")
        return self._saved_views

    def list_saved_views(self, namespace: str) -> Optional[Dict[str, Any]]:
        if not self._namespace_exists(namespace):
            return None
        return {"namespace": namespace,
                "views": self.saved_views.list(namespace)}

    def create_saved_view(self, namespace: str, name: str,
                          filter_: Dict[str, Any]) -> Dict[str, Any]:
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        return self.saved_views.create(namespace, name, filter_)

    def delete_saved_view(self, namespace: str,
                          view_id: str) -> Dict[str, Any]:
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not self.saved_views.delete(namespace, view_id):
            return {"error": "view_not_found", "detail": view_id}
        return {"deleted": view_id}

    # ─── Admin: LLM 등록·설정 (env-only 키) ──────────────────────────

    @property
    def settings(self):
        if self._settings is None:
            from .admin_store import SettingsStore
            self._settings = SettingsStore(self.data_dir / "admin.db")
        return self._settings

    def get_llm_config(self) -> Dict[str, Any]:
        """현재 LLM 설정 + 키 존재 여부(마스킹). **원문 키는 절대 반환 안 함.**

        provider/model/base_url 은 admin.db 에, 키는 환경변수에만. 여기서는
        키가 있는지와 끝 4자 힌트만 노출한다.
        """
        from ..core.llm_provider import DEFAULT_MODELS, DEFAULT_PROVIDER, PROVIDERS

        cfg = self.settings.get("llm", {}) or {}
        provider = cfg.get("provider") or DEFAULT_PROVIDER
        model = cfg.get("model") or DEFAULT_MODELS.get(provider) or ""
        base_url = cfg.get("base_url") or ""
        env_var = _ENV_KEY_BY_PROVIDER.get(provider, "")
        raw = os.getenv(env_var, "") if env_var else ""
        key_hint = ("…" + raw[-4:]) if len(raw) >= 4 else ("●●●●" if raw else "")
        return {
            "provider": provider, "model": model, "base_url": base_url,
            "providers": list(PROVIDERS), "env_var": env_var,
            "default_model": DEFAULT_MODELS.get(provider) or "",
            "key_present": bool(raw) or self.llm_fn is not None,
            "key_hint": key_hint,
            "injected": self.llm_fn is not None,
        }

    def set_llm_config(self, provider: str, model: str,
                       base_url: str = "") -> Dict[str, Any]:
        from ..core.llm_provider import PROVIDERS

        provider = (provider or "").strip()
        model = (model or "").strip()
        base_url = (base_url or "").strip()
        if provider not in PROVIDERS:
            return {"error": "invalid", "detail": f"unknown provider '{provider}'"}
        if not model:
            return {"error": "invalid", "detail": "model is required"}
        if provider == "openai_compatible" and not base_url:
            return {"error": "invalid",
                    "detail": "openai_compatible requires base_url"}
        self.settings.set("llm", {"provider": provider, "model": model,
                                  "base_url": base_url})
        self._llm_provider_cache.clear()   # 설정 바뀌면 캐시 무효
        logger.info(f"🤖 LLM config set: {provider}/{model}")
        return self.get_llm_config()

    def _active_llm_fn(self) -> Optional[Callable[[str], str]]:
        """지금 유효한 LLM 호출 함수. 주입(llm_fn)이 최우선(테스트·커스텀),
        없으면 저장된 설정으로 resolve_provider 에서 만든다. 설정도 없으면
        None → 하위 컴포넌트가 자기 기본값으로 degrade (기존 동작 보존)."""
        if self.llm_fn is not None:
            return self.llm_fn
        # 저장된 설정이 없어도 기본값(google/gemini + env 키)이 유효하면 쓴다
        # — 명시 저장을 강제하지 않는다. 키가 없으면 None (호출 불가).
        cfg = self.get_llm_config()
        if not cfg.get("key_present"):
            return None
        return self._provider_fn(cfg["provider"], cfg["model"],
                                 cfg.get("base_url"))

    def _provider_fn(self, provider, model, base_url) -> Callable[[str], str]:
        import asyncio

        key = (provider or "", model or "", base_url or "")

        def _fn(prompt: str) -> str:
            prov = self._llm_provider_cache.get(key)
            if prov is None:
                from ..core.llm_provider import resolve_provider
                prov = resolve_provider(provider=provider, model=model,
                                        base_url=base_url or None)
                self._llm_provider_cache[key] = prov
            return asyncio.run(prov.complete(prompt))  # to_thread 안에서 호출됨

        return _fn

    async def test_llm(self) -> Dict[str, Any]:
        """연결 테스트 — 유효 LLM 으로 한 콜. 실패해도 500 이 아니라
        {ok:false, error} 로 돌려준다(설정 화면에서 진단용)."""
        from ..core.llm_provider import CallableProvider

        fn = self._active_llm_fn()
        if fn is None:
            return {"ok": False, "error": "no LLM configured or available"}
        try:
            provider = fn if hasattr(fn, "complete") else CallableProvider(fn)
            out = await provider.complete("Reply with one word: pong")
            text = (out or "").strip()
            return {"ok": bool(text), "sample": text[:120]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ─── Admin: 프로젝트 (네임스페이스 = 프로젝트) ───────────────────

    @property
    def projects(self):
        if self._projects is None:
            from .admin_store import ProjectStore
            self._projects = ProjectStore(self.data_dir / "admin.db")
        return self._projects

    def create_project(self, name: str, description: str = "",
                       domain: str = "") -> Dict[str, Any]:
        """빈 프로젝트(네임스페이스) 생성 — 레코드만 만든다. 그래프는 첫
        빌드/노드 추가 때 태어난다. 여기서 인증·테넌시는 만들지 않는다
        (서비스 경계: 소유권은 logos_api 몫). 운영자 편의 레코드일 뿐이다.
        """
        import re

        name = (name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name or ""):
            return {"error": "invalid",
                    "detail": "name must be [A-Za-z0-9_-]+"}
        if name in PROTECTED_NAMESPACES:
            return {"error": "invalid", "detail": f"'{name}' is protected"}
        if self._namespace_exists(name):
            return {"error": "duplicate", "detail": name}
        return self.projects.create(name, description=description, domain=domain)

    # ─── Admin: 스키마 편집 (TBox — 개명 · 선언) ─────────────────────

    @property
    def schema_decl(self):
        if self._schema_decl is None:
            from .admin_store import SchemaDeclStore
            self._schema_decl = SchemaDeclStore(self.data_dir / "admin.db")
        return self._schema_decl

    def rename_type(self, namespace: str, old: str, new: str,
                    actor: str = "") -> Dict[str, Any]:
        """타입 개명 — 그 타입인 모든 노드의 `type` 속성을 바꾼다.

        node_id 는 손대지 않는다: id 는 불투명 키(빌더 규약 "Type:name")일
        뿐이고, 표시·질의에 쓰이는 것은 `type` 속성이다. id 를 재작성하면
        엣지·청크 매핑·시맨틱 인덱스까지 연쇄로 흔들려 provenance 가 깨진다.
        기존 타입명으로 개명하면 자연히 두 타입이 **병합**된다.
        """
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        old = (old or "").strip()
        new = (new or "").strip()
        if not old or not new:
            return {"error": "invalid", "detail": "old/new type required"}
        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        renamed = 0
        for _, attrs in graph.nodes(data=True):
            if attrs.get("type") == old:
                attrs["type"] = new
                renamed += 1
        if renamed == 0:
            return {"error": "type_not_found", "detail": old}
        self.schema_decl.rename_type(namespace, old, new)
        get_review_store(namespace).record(
            action="rename_type", node_id=f"type:{old}",
            before={"type": old}, after={"type": new, "count": renamed},
            actor=actor or "admin")
        self._pg_apply(namespace, lambda s: s.rename_type(namespace, old, new))
        engine.save_to_disk()
        return {"namespace": namespace, "old": old, "new": new,
                "renamed": renamed}

    def merge_nodes(self, namespace: str, winner: str, losers: List[str],
                    actor: str = "", dry_run: bool = False) -> Dict[str, Any]:
        """중복 노드 병합 — graph_health 가 찾은 것을 사람이 승인해 합친다.

        dry_run=True 면 **아무것도 바꾸지 않고** 계획만 돌려준다. 병합은 노드를
        지워 되돌릴 수 없으므로 미리보기가 필수다(팔란티어의 제안-승인 구조와
        같은 이유). 계획 계산은 순수 함수(core.node_merge)라 미리 본 것과
        적용되는 것이 같다.

        적용 순서에 의미가 있다:
          1) 엣지 재지정 — 진 노드를 지우기 **전에** 옮긴다. PG delete_node 는
             그 노드의 엣지를 함께 지우므로 순서를 바꾸면 사실이 사라진다.
          2) 청크·골든셋 참조 재지정 — 노드를 지운 뒤에 하면 그 사이에 읽은
             쪽은 dangling 을 본다.
          3) 진 노드 삭제.

        묘비(tombstone)를 남기지 않는 이유: 병합은 "이 개체는 틀렸다"(reject)가
        아니라 "같은 것이었다"(absorb)다. 묘비를 남기면 재빌드에서 그 이름이
        영구 차단되는데, 그 이름은 이제 이긴 노드의 **별칭**으로 살아 있어야 한다.
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.node_merge import plan_merge
        from ..core.review_store import get_review_store
        from ..core.search_qa import get_golden_set
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        winner = (winner or "").strip()
        if not winner:
            return {"error": "invalid", "detail": "winner required"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        store = get_chunk_store(namespace)
        golden = get_golden_set(namespace)
        plan = plan_merge(graph, winner, losers or [],
                          chunks=store.all(), cases=golden.cases())
        if "error" in plan:
            return plan
        # 생애주기 관문 — active 노드는 흡수(삭제)될 수 없다. **미리보기에서도
        # 알려준다**: 적용 단계에서만 막으면 검수자가 계획을 다 보고 나서
        # 거부당한다. 대조 문서의 지적("그 노드를 누가 쓰는지 아무도 모른다")이
        # 겨눈 자리가 정확히 이곳이다.
        from ..core.lifecycle import can_destroy
        blocked = [{"node_id": lid,
                    "lifecycle": graph.nodes[lid].get("lifecycle"),
                    "detail": can_destroy(graph.nodes[lid])[1]}
                   for lid in plan["losers"]
                   if not can_destroy(graph.nodes[lid])[0]]
        if blocked:
            return {"error": "lifecycle_protected", "blocked": blocked,
                    "detail": "운영 사용 중(active)인 노드는 병합으로 흡수할 수 "
                              "없다. 먼저 deprecated 로 내려라."}
        if dry_run:
            return {"namespace": namespace, "dry_run": True, "plan": plan}

        loser_set = set(plan["losers"])
        before = {lid: dict(graph.nodes[lid]) for lid in plan["losers"]}

        # 1) 엣지 — 계획이 정한 keep 집합에 맞춰 **원래 속성 그대로** 옮긴다.
        #    (weight·confidence 를 버리면 그래프의 신뢰도 정보가 사라진다)
        keep = {(e["from"], e["to"], e["predicate"])
                for e in plan["edges_repointed"]}
        self._repoint_edges(namespace, graph, loser_set, winner, keep)

        # 2) 이긴 노드 보강 — 진 이름을 별칭으로 흡수(질의 확장이 읽는다)
        win_attrs = graph.nodes[winner]
        if plan["aliases_added"]:
            existing = list(win_attrs.get("aliases") or [])
            win_attrs["aliases"] = existing + plan["aliases_added"]
        for key, value in (plan["properties_adopted"] or {}).items():
            win_attrs[key] = value
        self._pg_apply(namespace,
                       lambda s: s.upsert_node(namespace, winner, dict(win_attrs)))

        # 3) 참조 재지정 → 그 다음에 삭제
        chunks_touched: List[str] = []
        cases_touched: List[str] = []
        for lid in plan["losers"]:
            chunks_touched.extend(store.relabel_node(lid, winner))
            cases_touched.extend(golden.relabel_node(lid, winner))
        for lid in plan["losers"]:
            graph.remove_node(lid)
            self._pg_apply(namespace, lambda s, l=lid: s.delete_node(namespace, l))

        get_review_store(namespace).record(
            action="merge", node_id=winner,
            before=before,
            after={"aliases_added": plan["aliases_added"],
                   "edges_repointed": len(plan["edges_repointed"]),
                   "edges_dropped": len(plan["edges_dropped"]),
                   "property_conflicts": plan["property_conflicts"],
                   "chunks_rewritten": sorted(set(chunks_touched)),
                   "golden_relabels": sorted(set(cases_touched))},
            actor=actor or "admin")

        engine.save_to_disk()
        store.save_to_disk()
        return {"namespace": namespace, "winner": winner,
                "losers": plan["losers"],
                "edges_repointed": len(plan["edges_repointed"]),
                "edges_dropped": len(plan["edges_dropped"]),
                "aliases_added": plan["aliases_added"],
                "properties_adopted": plan["properties_adopted"],
                "property_conflicts": plan["property_conflicts"],
                "chunks_rewritten": sorted(set(chunks_touched)),
                "golden_relabels": sorted(set(cases_touched)),
                "node": self.get_node_detail(namespace, winner)}

    def _repoint_edges(self, namespace: str, graph, old_ids, new_id: str,
                       keep) -> int:
        """옛 노드(들)의 엣지를 새 노드로 옮긴다 — merge·rename 공용 (P-2 추출).

        두 벌로 두면 PG 반영 순서 계약이 갈라진다 — 그래서 추출했다.
        keep: 계획이 허용한 (from, to, predicate) 집합 (미리보기 == 적용의
        집행 장치). 같은 키의 평행 엣지는 1개로 접는다 (merge 규칙).
        """
        old_set = set(old_ids)
        added: set = set()
        count = 0
        for old in sorted(old_set):
            edges = ([(old, t, d) for _, t, d in graph.out_edges(old, data=True)]
                     + [(s, old, d) for s, _, d in graph.in_edges(old, data=True)])
            for source, target, data in edges:
                new_source = new_id if source in old_set else source
                new_target = new_id if target in old_set else target
                key = (new_source, new_target, (data or {}).get("predicate", ""))
                if key not in keep or key in added:
                    continue
                graph.add_edge(new_source, new_target, **dict(data or {}))
                added.add(key)
                self._pg_apply(namespace, lambda s, k=key, d=dict(data or {}):
                               s.add_edge(namespace, k[0], k[2], k[1], d))
                count += 1
        return count

    def rename_node(self, namespace: str, node_id: str, new_type: str,
                    actor: str = "", dry_run: bool = True) -> Dict[str, Any]:
        """노드 타입 재분류 — id 개명 (로드맵 4 P-2).

        재분류 = id 개명이다 (`{type}:{name}` 이 id). merge 와 같은 적용 순서
        계약(① 새 노드 upsert ② 엣지 재지정 ③ 청크·골든셋 참조 ④ 옛 노드
        삭제)을 따르되, **묘비를 남기지 않는다** — 재분류는 "타입이 틀렸다"지
        "개체가 틀렸다"가 아니고, 묘비면 재빌드에서 근거가 통째로 버려진다.
        재빌드 부활 통제는 P-4 재지도가 맡는다 (그 전의 부활은 cross_type
        클러스터로 보드에 잡히는 보이는 결함).
        """
        from ..core.chunk_store import get_chunk_store
        from ..core.node_rename import plan_rename
        from ..core.review_store import get_review_store
        from ..core.search_qa import get_golden_set
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        store = get_chunk_store(namespace)
        golden = get_golden_set(namespace)
        plan = plan_rename(graph, node_id, new_type,
                           chunks=store.all(), cases=golden.cases())
        if "error" in plan:
            return plan
        node_id = plan["node_id"]

        # 생애주기 관문 — 개명은 옛 id 의 파괴다. 미리보기에서도 알린다
        # (merge 와 같은 이유: 적용 단계에서만 막으면 계획을 다 보고 거부당한다).
        from ..core.lifecycle import can_destroy
        allowed, why = can_destroy(graph.nodes[node_id])
        if not allowed:
            return {"error": "lifecycle_protected", "detail": why,
                    "node_id": node_id,
                    "lifecycle": graph.nodes[node_id].get("lifecycle")}

        if dry_run:
            return {"namespace": namespace, "dry_run": True, "plan": plan}

        new_id = plan["new_id"]
        before = dict(graph.nodes[node_id])

        # ① 새 노드 upsert — 엣지보다 먼저 (PG 의 엣지가 노드 실존을 전제한다)
        graph.add_node(new_id, **plan["attrs_after"])
        self._pg_apply(namespace, lambda s: s.upsert_node(
            namespace, new_id, dict(plan["attrs_after"])))

        # ② 엣지 재지정 — merge 와 공용 부품, 계획이 keep 을 정한다
        keep = {(e["from"], e["to"], e["predicate"])
                for e in plan["edges_repointed"]}
        edges_moved = self._repoint_edges(namespace, graph, {node_id},
                                          new_id, keep)

        # ③ 참조 재지정 → ④ 그 다음에 삭제 (순서 계약은 merge docstring 참조)
        chunks_touched = store.relabel_node(node_id, new_id)
        cases_touched = golden.relabel_node(node_id, new_id)
        graph.remove_node(node_id)
        self._pg_apply(namespace, lambda s: s.delete_node(namespace, node_id))

        # 색인: 옛 id 를 내린다 — 새 id 는 /reindex 가 채운다 (reindex_required)
        try:
            index = getattr(engine, "_semantic_index", None)
            if index is not None and hasattr(index, "remove"):
                index.remove(node_id)
        except Exception as e:
            logger.warning(f"⚠️ Semantic index removal failed ({node_id}): {e}")

        # 감사 — 묘비 없이 reclassify 기록만. P-4 재지도가 이 기록을 재생한다.
        get_review_store(namespace).record(
            action="reclassify", node_id=new_id,
            before={"old_id": node_id, **before},
            after={"old_id": node_id, "new_id": new_id,
                   "new_type": plan["new_type"],
                   "edges_repointed": edges_moved,
                   "chunks_rewritten": sorted(set(chunks_touched)),
                   "golden_relabels": sorted(set(cases_touched))},
            actor=actor or "admin")

        engine.save_to_disk()
        store.save_to_disk()
        return {"namespace": namespace, "old_id": node_id, "new_id": new_id,
                "new_type": plan["new_type"],
                "edges_repointed": edges_moved,
                "chunks_rewritten": sorted(set(chunks_touched)),
                "golden_relabels": sorted(set(cases_touched)),
                "reindex_required": True,
                "node": self.get_node_detail(namespace, new_id)}

    def set_predicate_decl(self, namespace: str, predicate: str,
                           domain: str = "", range_: str = "",
                           description: str = "", actor: str = "") -> Dict[str, Any]:
        """술어 domain/range 선언 — **감사 대상 쓰기**.

        그래프를 건드리지 않지만 lint(consistency_checker.range_violation)가
        이 선언을 기준으로 판정하므로, 선언 변경은 검수 결과를 뒤집는다.
        누가 무엇에서 무엇으로 바꿨는지 남지 않으면 되돌릴 근거가 없다.
        """
        from ..core.review_store import get_review_store
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        predicate = (predicate or "").strip()
        if not predicate:
            return {"error": "invalid", "detail": "predicate required"}
        before = self.schema_decl.predicates(namespace).get(predicate)
        self.schema_decl.set_predicate(namespace, predicate, domain=domain,
                                       range_=range_, description=description)
        after = {"domain": domain or "", "range": range_ or "",
                 "description": description or ""}
        get_review_store(namespace).record(
            action="schema_predicate", node_id=f"predicate:{predicate}",
            before=before, after=after, actor=actor or "admin")
        return {"predicate": predicate, **after}

    def set_type_decl(self, namespace: str, type_: str, description: str = "",
                      deprecated: bool = False, actor: str = "") -> Dict[str, Any]:
        """타입 설명·deprecated 선언 — 감사 대상 (set_predicate_decl 과 같은 이유)."""
        from ..core.review_store import get_review_store
        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        type_ = (type_ or "").strip()
        if not type_:
            return {"error": "invalid", "detail": "type required"}
        before = self.schema_decl.types(namespace).get(type_)
        self.schema_decl.set_type(namespace, type_, description=description,
                                  deprecated=deprecated)
        after = {"description": description or "", "deprecated": bool(deprecated)}
        get_review_store(namespace).record(
            action="schema_type", node_id=f"type:{type_}",
            before=before, after=after, actor=actor or "admin")
        return {"type": type_, **after}

    # ─── Admin: 스키마 (TBox — 클래스 · 술어 · is_a 계층) ────────────

    # 내부 관리 속성 — 프로퍼티 사용 분포에서 제외 (type 은 클래스 자신)
    _INTERNAL_ATTRS = {"type", "created_at", "last_updated"}

    def get_schema_overview(self, namespace: str) -> Optional[Dict[str, Any]]:
        """스키마 요약 — 그래프에 **관측된** 어휘를 그대로 보고한다.

        선언(빌드 시 BuilderSchema)이 아니라 관측을 보고하는 이유: 선언은
        네임스페이스에 저장되지 않고, 관측이야말로 지금 그래프의 진실이다.
        선언-관측 어긋남은 lint_consistency(domain/range 검사)의 몫.
        """
        from collections import Counter, defaultdict

        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return None
        graph = get_knowledge_graph_engine(namespace).graph

        type_counts: Counter = Counter()
        type_props: Dict[str, Counter] = defaultdict(Counter)
        for _, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            type_counts[node_type] += 1
            for key, value in attrs.items():
                if key in self._INTERNAL_ATTRS:
                    continue
                if value in (None, "", []):
                    continue  # 빈 값은 "채워진 프로퍼티"가 아니다
                type_props[node_type][key] += 1

        pred_counts: Counter = Counter()
        pred_pairs: Dict[str, Counter] = defaultdict(Counter)
        is_a_edges: List[Tuple[str, str]] = []  # (child, parent)
        for s, t, attrs in graph.edges(data=True):
            predicate = attrs.get("predicate", "unknown")
            pred_counts[predicate] += 1
            pred_pairs[predicate][(
                graph.nodes[s].get("type", "unknown"),
                graph.nodes[t].get("type", "unknown"))] += 1
            if predicate == "is_a":
                is_a_edges.append((s, t))

        # 선언(의도) — 관측 위에 얹는다. domain/range·설명·deprecated.
        type_decls = self.schema_decl.types(namespace)
        pred_decls = self.schema_decl.predicates(namespace)

        classes = [
            {"type": node_type, "count": count,
             "properties": dict(type_props[node_type].most_common()),
             "declared": type_decls.get(node_type)}
            for node_type, count in type_counts.most_common()]
        predicates = [
            {"predicate": predicate, "count": count,
             "pairs": [{"source_type": st, "target_type": tt, "count": c}
                       for (st, tt), c in pred_pairs[predicate].most_common()],
             "declared": pred_decls.get(predicate)}
            for predicate, count in pred_counts.most_common()]

        return {"namespace": namespace, "classes": classes,
                "predicates": predicates,
                "hierarchy": self._build_hierarchy(graph, is_a_edges),
                "hierarchy_edges": len(is_a_edges)}

    @staticmethod
    def _build_hierarchy(graph, is_a_edges) -> List[Dict[str, Any]]:
        """is_a 트리 — 루트(부모이면서 자식 아님)부터 재귀 조립.

        사이클은 방문 집합으로 끊는다 — is_a 사이클은 데이터 오류지만,
        오류 때문에 스키마 화면이 무한 재귀로 죽으면 오류를 볼 수도 없다.
        """
        from collections import defaultdict

        children_of: Dict[str, List[str]] = defaultdict(list)
        child_set = set()
        for child, parent in is_a_edges:
            children_of[parent].append(child)
            child_set.add(child)

        def node_view(node_id, visited):
            name = graph.nodes[node_id].get("name", node_id) \
                if node_id in graph else node_id
            kids = [node_view(c, visited | {node_id})
                    for c in sorted(children_of.get(node_id, []))
                    if c not in visited]
            return {"id": node_id, "name": name, "children": kids}

        roots = sorted(p for p in children_of if p not in child_set)
        return [node_view(r, {r}) for r in roots]

    # ─── Admin: 노드 브라우저 · 편집 (ABox) ──────────────────────────

    def list_nodes(self, namespace: str, q: Optional[str] = None,
                   node_type: Optional[str] = None,
                   trust: Optional[str] = None, prop: Optional[str] = None,
                   kind: Optional[str] = None,
                   offset: int = 0, limit: int = 50) -> Optional[Dict[str, Any]]:
        """노드 브라우저 — 부분일치 검색(id/이름/별칭) + 타입/trust/프로퍼티
        필터 + 페이지네이션. total 은 필터 후 전체 개수다 (창 크기가 아니라).

        prop 이 주어지면 그 프로퍼티가 **채워진** 노드만 남기고, 각 항목에
        실제 값(prop_value)을 싣는다 — "이 프로퍼티가 어디서·어떤 값으로
        쓰이나"에 답하는 스키마→인스턴스 드릴다운의 근거다.

        데이터 순회는 GraphStore seam(축 5)에 위임한다 — 백엔드가 memory
        (NetworkX)든 postgres(인덱스 질의)든 반환 형태는 동일하다.
        """
        if not self._namespace_exists(namespace):
            return None
        store = self._graph_store(namespace)
        result = store.list_nodes(
            namespace, q=q, node_type=node_type, trust=trust, prop=prop,
            kind=kind, offset=offset, limit=limit)
        self._mark_classes(store, namespace, result)
        return result

    def _mark_classes(self, store, namespace: str, result) -> None:
        """목록 items 에 is_class 를 붙인다(클래스 vs 인스턴스 분리) — 메타클래스
        타입 집합으로 판정. 실패해도 목록 자체는 살린다(부가 정보)."""
        if not result or not result.get("items"):
            return
        try:
            typeset = set(store.distributions(namespace).get("node_types", {}))
            meta = store.metaclass_types(namespace)
        except Exception as e:
            logger.warning(f"⚠️ 클래스 판정 실패: {namespace} — {e}")
            return
        for it in result["items"]:
            # 클래스 = 이름이 타입으로 쓰인다(사원·삼층석탑) 또는 타입이
            # 메타클래스다(HeritageClass 의 빈 클래스까지 포함). 그 외는 인스턴스.
            it["is_class"] = (it.get("name") in typeset) or (it.get("type") in meta)

    def query_nodes(self, namespace: str, node_type: Optional[str] = None,
                    trust: Optional[str] = None, prop_key: Optional[str] = None,
                    prop_value: Optional[str] = None, prop_op: str = "eq",
                    rel_predicate: Optional[str] = None,
                    rel_target: Optional[str] = None,
                    rel_target_type: Optional[str] = None,
                    rel_direction: str = "out", offset: int = 0,
                    limit: int = 50) -> Optional[Dict[str, Any]]:
        """구조화 검색(프로퍼티 값 + 관계 제약) — seam 위임. 미지 네임스페이스 None."""
        if not self._namespace_exists(namespace):
            return None
        return self._graph_store(namespace).query_nodes(
            namespace, node_type=node_type, trust=trust,
            prop_key=prop_key, prop_value=prop_value, prop_op=prop_op,
            rel_predicate=rel_predicate, rel_target=rel_target,
            rel_target_type=rel_target_type, rel_direction=rel_direction,
            offset=offset, limit=limit)

    def list_edges(self, namespace: str, predicate: Optional[str] = None,
                   source_type: Optional[str] = None,
                   target_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """엣지 목록 — 술어(+선택적 시그니처)로 거른 관계들. 양끝 노드의
        이름·타입을 해석해 싣는다 (클릭 이동·삭제의 진입점).

        술어를 "나열만" 하던 스키마 뷰에 관리 포인트를 준다: 술어를 클릭하면
        그 술어를 쓰는 실제 엣지가 여기서 나오고, 각 엣지를 삭제할 수 있다.

        데이터 순회는 GraphStore seam 에 위임한다(축 5).
        """
        if not self._namespace_exists(namespace):
            return None
        return self._graph_store(namespace).list_edges(
            namespace, predicate=predicate, source_type=source_type,
            target_type=target_type)

    def list_neighbors(self, namespace: str, node_id: str,
                       limit: int = 60) -> Optional[Dict[str, Any]]:
        """앵커 노드의 이웃 서브그래프 — 3D '앵커→이웃 확장'의 기반 원시연산.

        전체 그래프를 절대 읽지 않는다: 앵커의 out/in 엣지만 순회하므로
        비용은 O(그 노드의 차수)이지 O(그래프 크기)가 아니다 — 수천만
        규모에서도 한 노드의 이웃 조회는 싸다. 이웃이 limit 을 넘으면
        truncated=True 로 알리고 앞 limit 개만 싣는다(프론트는 노드를 더
        클릭해 점진 확장). 반환 모양은 Graph3D 가 먹는 {nodes, links}.

        데이터 순회는 GraphStore seam 에 위임한다(축 5) — postgres 백엔드는
        인덱스 스캔(edge_out/edge_in)으로 O(degree) 를 유지한다.
        """
        if not self._namespace_exists(namespace):
            return None
        return self._graph_store(namespace).list_neighbors(
            namespace, node_id, limit=limit)

    def create_node(self, namespace: str, node_type: str, name: str,
                    definition: str = "", aliases: Optional[List[str]] = None,
                    attrs: Optional[Dict[str, Any]] = None,
                    actor: str = "") -> Dict[str, Any]:
        """수동 노드 생성 — 관리자의 명시적 지식 추가.

        - node_id 는 빌더와 같은 규약: "{Type}:{이름}".
        - source 는 비워 둔다 — 인간이 만든 노드는 검수 큐 대상이 아니다
          (큐는 LLM 추출분의 검증 장치다).
        - 묘비가 있던 id 면 confirm 으로 먼저 걷는다 — 수동 재생성은 인간의
          명시적 번복이고, "나중 판정이 이긴다"는 검수 루프 규칙 그대로다.
        - 시맨틱 인덱스는 갱신하지 않는다(임베딩 비용) — 다음 재빌드/재색인
          까지 검색에는 안 잡힐 수 있다.
        """
        from datetime import datetime as _dt

        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        node_type = (node_type or "").strip()
        name = (name or "").strip()
        if not node_type or not name:
            return {"error": "invalid", "detail": "node_type/name required"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        node_id = f"{node_type}:{name}"
        if node_id in graph:
            return {"error": "duplicate", "detail": node_id}

        reviews = get_review_store(namespace)
        if reviews.is_rejected(node_id):
            reviews.confirm(node_id, actor=actor or "admin")

        now = _dt.now().isoformat()
        node_attrs: Dict[str, Any] = {
            "type": node_type, "name": name,
            "created_at": now, "last_updated": now,
            **(attrs or {})}
        if definition:
            node_attrs["definition"] = definition
        if aliases:
            node_attrs["aliases"] = list(aliases)
        graph.add_node(node_id, **node_attrs)

        reviews.record(action="create", node_id=node_id,
                       after=dict(node_attrs), actor=actor or "admin")
        self._pg_apply(namespace, lambda s: s.upsert_node(namespace, node_id, dict(node_attrs)))
        engine.save_to_disk()
        return {"node": self.get_node_detail(namespace, node_id)}

    def update_node(self, namespace: str, node_id: str,
                    updates: Dict[str, Any],
                    actor: str = "") -> Dict[str, Any]:
        """노드 프로퍼티 편집 — 값 None 은 그 프로퍼티 삭제다.

        node_id 자체는 불변("Type:이름" 규약이 곧 주소다). 감사에는 변경된
        키의 before/after 만 싣는다 — 전체 스냅샷은 노이즈다.
        """
        from datetime import datetime as _dt

        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        if not updates:
            return {"error": "invalid", "detail": "updates must not be empty"}
        if "aliases" in updates and updates["aliases"] is not None \
                and not isinstance(updates["aliases"], list):
            return {"error": "invalid", "detail": "aliases must be a list"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        if node_id not in graph:
            return {"error": "node_not_found", "detail": node_id}

        node_attrs = graph.nodes[node_id]
        before: Dict[str, Any] = {}
        after: Dict[str, Any] = {}
        for key, value in updates.items():
            before[key] = node_attrs.get(key)
            if value is None:
                node_attrs.pop(key, None)
                after[key] = None
            else:
                node_attrs[key] = value
                after[key] = value
        node_attrs["last_updated"] = _dt.now().isoformat()

        get_review_store(namespace).record(
            action="edit", node_id=node_id, before=before, after=after,
            actor=actor or "admin")
        self._pg_apply(namespace, lambda s: s.upsert_node(namespace, node_id, dict(node_attrs)))
        engine.save_to_disk()
        return {"node": self.get_node_detail(namespace, node_id)}

    # ─── Admin: 엣지 편집 ────────────────────────────────────────────

    def add_edge(self, namespace: str, source: str, predicate: str,
                 target: str, actor: str = "") -> Dict[str, Any]:
        """관계 추가 — 양끝 노드가 있어야 한다 (auto-create 하지 않는다:
        오타가 조용히 유령 노드를 만드는 것이 빌더 fast_mode 의 실수 경로다).
        같은 (source, predicate, target) 은 중복이다 — MultiDiGraph 라
        그래프는 허용하지만 온톨로지 관점에서 같은 주장 두 번은 무의미하다.
        """
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        predicate = (predicate or "").strip()
        if not predicate:
            return {"error": "invalid", "detail": "predicate required"}

        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph
        for node_id in (source, target):
            if node_id not in graph:
                return {"error": "node_not_found", "detail": node_id}
        existing = graph.get_edge_data(source, target) or {}
        if any(a.get("predicate") == predicate for a in existing.values()):
            return {"error": "duplicate",
                    "detail": f"{source} -{predicate}-> {target}"}

        graph.add_edge(source, target, predicate=predicate)
        get_review_store(namespace).record(
            action="edge_added", node_id=source,
            after={"predicate": predicate, "target": target},
            actor=actor or "admin")
        self._pg_apply(namespace, lambda s: s.add_edge(namespace, source, predicate, target))
        engine.save_to_disk()
        return {"edge": {"source": source, "predicate": predicate,
                         "target": target}}

    def remove_edge(self, namespace: str, source: str, predicate: str,
                    target: str, actor: str = "") -> Dict[str, Any]:
        """관계 삭제 — MultiDiGraph 에서 predicate 가 일치하는 키만 제거."""
        from ..core.review_store import get_review_store
        from ..engines.knowledge_graph_clean import get_knowledge_graph_engine

        if not self._namespace_exists(namespace):
            return {"error": "namespace_not_found"}
        engine = get_knowledge_graph_engine(namespace)
        graph = engine.graph

        edge_data = graph.get_edge_data(source, target) or {}
        keys = [k for k, a in edge_data.items()
                if a.get("predicate") == predicate]
        if not keys:
            return {"error": "edge_not_found",
                    "detail": f"{source} -{predicate}-> {target}"}
        for key in keys:
            graph.remove_edge(source, target, key=key)

        get_review_store(namespace).record(
            action="edge_removed", node_id=source,
            before={"predicate": predicate, "target": target},
            actor=actor or "admin")
        self._pg_apply(namespace, lambda s: s.delete_edge(namespace, source, predicate, target))
        engine.save_to_disk()
        return {"removed": {"source": source, "predicate": predicate,
                            "target": target, "count": len(keys)}}
