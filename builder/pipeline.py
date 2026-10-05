"""
OntologyBuilder — file/folder → segment → extract(LLM) → validate → merge.

Data + framework + LLM 결합의 오케스트레이터:
- 노드 ID는 f"{Type}:{name}" 규약 (aicoach와 동일)
- 모든 노드/관계에 provenance(source, chunk_index) 기록
- 중복 방지는 KnowledgeGraphEngine 쓰기 레이어가 이미 보장
- 한 청크의 LLM 실패는 세지고 넘어간다 — 빌드는 계속된다 (resilient)
"""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from ..core.llm_provider import (DEFAULT_EXTRACTION_MODEL, DEFAULT_MODELS,
                                 PROVIDERS, resolve_provider)
from .extractor import (LLMFn, build_extraction_prompt,
                        build_schema_proposal_prompt, parse_llm_json)
from .models import BuilderSchema, BuildReport, Chunk
from .segmenter import segment
from .validator import clean_extraction
from . import readers

ProgressCb = Callable[..., None]

# ─── 신뢰 등급 (trust layer) ─────────────────────────────────────────
# aicoach 실증: 같은 상품에 통합약관(원본)과 상품요약서(boilerplate — aicoach
# 는 블록리스트로 걸러냈다)가 함께 들어온다. KG add_concept 은 나중 쓰기가
# 이기는 update 라서, 등급 없이는 요약서가 약관의 사실을 조용히 덮는다.
# 미지정("")과 unknown 을 summary 보다 위에 두는 이유: 출처를 모르는 것과
# boilerplate 로 판정된 것은 다르다 — 후자만 명시적으로 강등한다.
TRUST_RANK = {"authoritative": 2, "": 1, "unknown": 1, "summary": 0}


def _trust_rank(trust: Optional[str]) -> int:
    return TRUST_RANK.get(trust or "", 1)


def guard_attrs_by_trust(existing: Dict[str, Any],
                         incoming: Dict[str, Any],
                         trust: str) -> Dict[str, Any]:
    """낮은 신뢰의 쓰기가 높은 신뢰의 기존 값을 덮지 못하게 거른다.

    - 낮은 등급: 기존에 있는 키는 버리고 **빈 자리만 채운다** — 요약서만
      아는 사실은 유효하다. 금지는 덮어쓰기지 기여가 아니다. trust 필드
      자체도 기존 것이 있으면 강등하지 않는다.
    - 같거나 높은 등급: 정상 병합 (trust 는 승격될 수 있다).
    - trust 미사용 경로(""): 동작 무변경 — 기존 호출부 전부 무영향.

    순수 함수 — KG 쓰기 레이어(add_concept)는 건드리지 않는다. 그 레이어는
    에이전트 라우팅 그래프도 쓰므로, 문서 인제스트의 정책을 커널에 넣으면
    안 된다.
    """
    merged = dict(incoming)
    if trust:
        merged["trust"] = trust
    if _trust_rank(trust) < _trust_rank(existing.get("trust", "")):
        merged = {k: v for k, v in merged.items() if k not in existing}
    return merged

# 프로바이더 목록·기본 모델·해석 로직은 core.llm_provider 로 이관 (축 1).
# 여기서 re-export 하는 것은 기존 import 경로를 깨지 않기 위해서다.
__all__ = ["OntologyBuilder", "make_llm_client", "PROVIDERS",
           "DEFAULT_MODELS", "DEFAULT_EXTRACTION_MODEL"]


def make_llm_client(provider: str, model: Optional[str] = None,
                    base_url: Optional[str] = None):
    """프로바이더 인스턴스 생성 — core.llm_provider.resolve_provider 의 얇은 래퍼.

    반환값은 `await provider.complete(prompt) -> str` 를 가진 객체다.
    (이전에는 logosai GoogleLangChainWrapper 를 직접 돌려줬다. 그 하드
    의존이 커널 채택을 막던 지점이라 seam 뒤로 옮겼다 — core/llm_provider.py 참고.)

    openai_compatible은 vLLM/Ollama/Qwen 등 OpenAI 호환 서버용 —
    base_url 필수, API 키는 OPENAI_COMPAT_API_KEY(없으면 더미).
    """
    return resolve_provider(provider, model, base_url=base_url)


class OntologyBuilder:
    def __init__(
        self,
        schema: Optional[BuilderSchema],  # None = auto (LLM이 스키마 제안)
        namespace: str = "default",
        llm_fn: Optional[LLMFn] = None,
        kg=None,
        progress_cb: Optional[ProgressCb] = None,
        chunk_size: int = 800,
        overlap: int = 120,
        segment_mode: str = "auto",
        auto_save: bool = True,
        llm_retries: int = 3,
        retry_base_delay: float = 2.0,
        llm_model: Optional[str] = None,
        llm_provider: str = "google",
        llm_base_url: Optional[str] = None,
        chunk_store=None,
        store_chunks: bool = True,
        review_store=None,
        check_tombstones: bool = True,
        curate_schema: bool = True,
        extraction_mode: str = "exhaustive",  # exhaustive | topic
        topics: Optional[List[str]] = None,
        topic_top_k: int = 6,
        quality_gate: Optional[bool] = None,
        image_assets: Optional[bool] = None,
    ):
        if kg is None:
            from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
            kg = get_knowledge_graph_engine(namespace)
        self.kg = kg
        # 원문 청크 저장소 (축 2). 기본 on — 원문 없이는 인용도, 근거 있는
        # 데이터셋도, 구절 검색도 불가능하다. 대용량 코퍼스처럼 원문 보존이
        # 부담인 경우만 store_chunks=False 로 끈다.
        self.store_chunks = store_chunks
        if chunk_store is None and store_chunks:
            from ..core.chunk_store import get_chunk_store
            chunk_store = get_chunk_store(getattr(kg, "namespace", namespace))
        self.chunk_store = chunk_store
        # 청크 품질 게이트 — 목차·OCR 잡음 청크의 trust 를 강등한다.
        # **기본 off**: 임계값이 데이터 관찰에서 고른 값이고 골든셋으로 측정된
        # 값이 아니다. 측정 없이 기본 동작을 바꾸면 이 저장소가 스스로 금지한
        # "감으로 정한 상수 강제"를 하는 셈이다. 명시 인자 > 환경변수 순.
        if quality_gate is None:
            quality_gate = os.environ.get(
                "ONTOLOGY_CHUNK_QUALITY_GATE", "").strip().lower() in (
                    "1", "true", "yes", "on")
        self.quality_gate = bool(quality_gate)
        # 문서 내 이미지 주소화·부착 — **기본 off**. 추출은 PDF 를 한 번 더 열고
        # 캡션 밴드마다 crop→extract_text 를 돌린다. 인제스트 비용을 측정 없이
        # 전 사용자에게 물리지 않는다(품질 게이트와 같은 원칙).
        if image_assets is None:
            image_assets = os.environ.get(
                "ONTOLOGY_IMAGE_ASSETS", "").strip().lower() in (
                    "1", "true", "yes", "on")
        self.image_assets = bool(image_assets)
        # 검수 묘비 저장소 (검수 루프). 기본 on — 검수자가 거절한 오추출이
        # 재빌드에서 조용히 부활하면 검수의 의미가 없다. 검수를 쓰지 않는
        # 소비자(대량 마이그레이션 등)만 check_tombstones=False 로 끈다.
        self.check_tombstones = check_tombstones
        if review_store is None and check_tombstones:
            from ..core.review_store import get_review_store
            review_store = get_review_store(getattr(kg, "namespace", namespace))
        self.review_store = review_store
        self.schema = schema
        self.llm_fn = llm_fn
        self.progress_cb = progress_cb
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.segment_mode = segment_mode
        # 추출 방식(aicoach식 토픽 회수 옵션). exhaustive=전 청크 LLM 추출(완전,
        # 무거움). topic=전 청크 저장·임베딩 후 토픽별 상위 청크만 추출(대용량
        # 약관 저비용·집중). 벡터DB(청크)는 두 방식 모두 채운다.
        self.extraction_mode = extraction_mode
        self.topics = topics
        self.topic_top_k = topic_top_k
        self.auto_save = auto_save
        self.llm_retries = llm_retries
        self.retry_base_delay = retry_base_delay
        self.llm_provider = llm_provider
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model or DEFAULT_MODELS.get(llm_provider)
        self._default_llm = None       # lazy langchain client
        self._proposed_schema = None   # auto 모드에서 LLM이 제안한 스키마 기록
        # 스키마 큐레이터 (④). 기본 on — auto 제안이 빌드마다 타입명을 새로
        # 지으면 같은 네임스페이스가 조용히 쪼개진다 (일관성 린터가 heritage_kr
        # 에서 실측한 name_type_conflict 10건의 원인 경로). auto 모드에서만
        # 동작한다 — 사용자가 명시한 스키마를 고치는 것은 게이트 철학 위반.
        self.curate_schema = curate_schema
        self._schema_mappings = None   # 큐레이터의 병합 기록

    # ─── LLM 호출 ────────────────────────────────────────────────────

    # transient API failures worth retrying (rate limit / overload)
    _TRANSIENT_MARKERS = ("429", "503", "unavailable", "resource_exhausted",
                          "rate limit", "overloaded", "high demand")

    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Retry transient LLM failures with exponential backoff
        (aicoach _extract_resilient 이식). Permanent errors raise at once."""
        delay = self.retry_base_delay
        for attempt in range(self.llm_retries):
            try:
                return await self._call_llm(prompt)
            except Exception as e:
                message = str(e).lower()
                transient = any(m in message for m in self._TRANSIENT_MARKERS)
                if not transient or attempt == self.llm_retries - 1:
                    raise
                logger.warning(
                    f"⏳ Transient LLM error (attempt {attempt + 1}/"
                    f"{self.llm_retries}), retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
                delay *= 2

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_fn is not None:
            # injected fn is sync — keep the event loop free
            return await asyncio.to_thread(self.llm_fn, prompt)
        # production default: builder-owned provider so the extraction model
        # (llm_model 옵션) is independent from the global planner config.
        if self._default_llm is None:
            self._default_llm = make_llm_client(
                self.llm_provider, self.llm_model, self.llm_base_url)
        return await self._default_llm.complete(prompt)

    def _save(self) -> None:
        """그래프 + 원문 청크를 함께 내린다 — 둘은 한 빌드의 두 산출물이다."""
        self.kg.save_to_disk()
        if self.store_chunks and self.chunk_store is not None:
            self.chunk_store.save_to_disk()

    def _progress(self, **event) -> None:
        if self.progress_cb is not None:
            try:
                self.progress_cb(**event)
            except Exception:
                pass  # progress reporting must never break the build

    # ─── 빌드 ────────────────────────────────────────────────────────

    async def _ensure_schema(self, sample_text: str) -> None:
        """auto 모드(schema=None): LLM에게 샘플로 스키마를 제안받는다.
        제안이 못 쓰는 형태면 document 프리셋으로 폴백 — 빌드는 계속된다.

        제안 직후 스키마 큐레이터(④)가 기존 어휘에 정합시킨다 — 빌드마다
        LLM 이 타입명을 새로 지으면(Clause vs Provision) 같은 네임스페이스가
        조용히 쪼개진다. **auto 경로에서만** 큐레이션한다: 이 메서드는
        schema=None 일 때만 도달하므로 사용자가 명시한 스키마는 구조적으로
        여기 오지 않는다.
        """
        if self.schema is not None:
            return
        existing_vocab = None
        if self.curate_schema:
            # 1차 방어선: 제안 시점에 기존 어휘를 보여준다 (원천 정렬).
            # 아래 reconcile 은 그래도 어긋난 것을 잡는 백스톱이다.
            from ..core.schema_curator import existing_vocabulary
            existing_vocab = existing_vocabulary(self.kg.graph)
        try:
            raw = await self._call_llm_with_retry(
                build_schema_proposal_prompt(sample_text[:2000],
                                             existing_vocab=existing_vocab))
            parsed = parse_llm_json(raw)
            if parsed and parsed.get("node_types"):
                self.schema = BuilderSchema.from_dict(parsed)
                self._proposed_schema = {
                    "node_types": self.schema.node_types,
                    "predicates": {k: list(v) for k, v in self.schema.predicates.items()},
                }
                logger.info(f"🧩 Auto schema proposed: {self.schema.node_types}")
                if self.curate_schema:
                    try:
                        from ..core.schema_curator import SchemaCurator
                        curator = SchemaCurator(llm_fn=self.llm_fn)
                        self.schema, self._schema_mappings = \
                            await curator.reconcile(self.schema, self.kg.graph)
                    except Exception as e:
                        # 큐레이션은 개선이지 관문이 아니다 — 여기서 죽으면
                        # 제안 원안으로 계속한다. 바깥 except 로 새어나가면
                        # 멀쩡한 제안이 프리셋 폴백으로 대체되는 더 나쁜 결과.
                        logger.warning(f"⚠️ Schema curation failed ({e}) — "
                                       f"using the raw proposal")
                return
        except Exception as e:
            logger.warning(f"⚠️ Schema proposal failed ({e}) — falling back to preset")
        self.schema = BuilderSchema.preset("document")

    async def build_from_text(self, text: str, source: str = "",
                              trust: str = "",
                              images: Optional[List[Any]] = None) -> BuildReport:
        report = BuildReport(namespace=getattr(self.kg, "namespace", "default"))
        await self._ensure_schema(text)
        report.proposed_schema = self._proposed_schema
        report.schema_mappings = self._schema_mappings
        await self._build_text_into(report, text, source, trust=trust,
                                   images=images)
        if self.auto_save:
            self._save()
        return report

    async def build_from_file(self, path) -> BuildReport:
        text = readers.read_file(path)
        # 이미지는 **파일 경로를 아는 이 지점**에서만 수집할 수 있다 — 아래 층은
        # 텍스트만 받는다. 게이트 off 면 빈 목록이라 기존 경로와 동일하다.
        report = await self.build_from_text(text, source=str(path),
                                           images=self.collect_images(path))
        report.files_read = 1
        return report

    async def build_from_folder(self, path,
                                patterns: Optional[List[str]] = None) -> BuildReport:
        report = BuildReport(namespace=getattr(self.kg, "namespace", "default"))
        documents = readers.read_folder(path, patterns=patterns)
        if documents:
            await self._ensure_schema(documents[0][1])
            report.proposed_schema = self._proposed_schema
            report.schema_mappings = self._schema_mappings
        for source, text in documents:
            self._progress(stage="read", source=source)
            try:
                await self._build_text_into(report, text, source)
                report.files_read += 1
            except Exception as e:
                report.errors.append(f"{source}: {e}")
                logger.warning(f"⚠️ Build failed for {source}: {e}")
        if self.auto_save:
            self._save()
        return report

    async def build_from_records(self, records: List[Dict[str, Any]],
                                 mapping: Dict[str, Any],
                                 source: str = "",
                                 trust: str = "") -> BuildReport:
        """정형 레코드 → 결정적 인제스트 (LLM 무사용).

        비정형은 LLM 추출, 정형은 결정적 매핑 — 이미 구조화된 데이터에
        LLM을 태우는 것은 비용·오염 양쪽에서 손해다 (aicoach premiums 원칙).

        mapping = {
          "node_type": "HeritageSite", "name_field": "이름",
          "relations": [{"predicate": "...", "target_type": "...", "field": "..."}],
        }
        관계 필드 값은 타깃 노드가 되고, 나머지 필드는 attrs로 보존된다.
        """
        report = BuildReport(namespace=getattr(self.kg, "namespace", "default"))
        fallback_type = mapping["node_type"]
        name_field = mapping["name_field"]
        type_field = mapping.get("type_field")  # 값이 곧 노드 타입 (온톨로지 세분류)
        relations = mapping.get("relations", [])
        relation_fields = {r["field"] for r in relations}
        if type_field:
            relation_fields.add(type_field)
        graph = self.kg.graph

        for i, record in enumerate(records):
            name = str(record.get(name_field) or "").strip()
            if not name:
                continue  # 이름 없는 레코드는 노드가 될 수 없다
            node_type = fallback_type
            if type_field:
                type_value = str(record.get(type_field) or "").strip()
                if type_value:
                    node_type = type_value.replace(" ", "_")
            node_id = f"{node_type}:{name}"
            # 재지도(P-4) → 묘비 순서: 재지도의 종착지가 묘비면 묘비가 이긴다
            remapped = self._remap_reclassified(node_id, report)
            if remapped != node_id:
                node_id = remapped
                node_type = node_id.split(":", 1)[0] if ":" in node_id else node_type
            # 묘비 관문 — 거절된 레코드는 관계까지 통째로 건너뛴다
            # (본 노드 없이 간선만 만들 수는 없다)
            if self._is_tombstoned(node_id):
                report.entities_rejected += 1
                continue
            attrs = {"name": name, "source": source,
                     **{k: v for k, v in record.items()
                        if k != name_field and k not in relation_fields and v is not None}}
            is_new = node_id not in graph
            attrs = guard_attrs_by_trust(
                graph.nodes[node_id] if not is_new else {}, attrs, trust)
            await self.kg.add_concept(node_id, node_type, attrs)
            if is_new:
                report.entities_added += 1

            for relation in relations:
                value = str(record.get(relation["field"]) or "").strip()
                if not value:
                    continue
                target_id = f"{relation['target_type']}:{value}"
                target_type = relation["target_type"]
                remapped = self._remap_reclassified(target_id, report)
                if remapped != target_id:
                    target_id = remapped
                    target_type = target_id.split(":", 1)[0] if ":" in target_id \
                        else target_type
                # 타깃 쪽 묘비도 막는다 — 본 노드만 걸러서는 거절된 개체가
                # 관계 타깃 경로로 부활하는 구멍이 남는다
                if self._is_tombstoned(target_id):
                    report.entities_rejected += 1
                    continue
                if target_id not in graph:
                    target_attrs = guard_attrs_by_trust(
                        {}, {"name": value, "source": source}, trust)
                    await self.kg.add_concept(target_id, target_type,
                                              target_attrs)
                    report.entities_added += 1
                existing = graph.get_edge_data(node_id, target_id) or {}
                already = any(a.get("predicate") == relation["predicate"]
                              for a in existing.values())
                await self.kg.add_relationship(node_id, target_id,
                                               relation["predicate"],
                                               {"source": source})
                if not already:
                    report.relations_added += 1

            if (i + 1) % 200 == 0:
                self._progress(stage="ingest", current=i + 1, total=len(records))

        report.chunks_processed = len(records)
        if self.auto_save:
            self.kg.save_to_disk()
        return report

    async def _build_text_into(self, report: BuildReport,
                               text: str, source: str,
                               trust: str = "",
                               images: Optional[List[Any]] = None) -> None:
        # 소스별 멱등 재적재(Phase 1-3): 이 소스를 (다시) 처리하기 전에 옛 청크를
        # 비운다 — 내용이 바뀐 문서를 재업로드해도 헌 청크가 쌓이지 않고 '교체'된다.
        # 첫 적재면 no-op. 청크는 source 1:1 이라 다른 문서엔 영향 없다.
        if self.store_chunks and self.chunk_store is not None:
            removed = self.chunk_store.delete_by_source(source)
            if removed:
                logger.info(f"♻️ 재적재: '{source}' 옛 청크 {removed}개 교체")
        chunks = segment(text, mode=self.segment_mode,
                         chunk_size=self.chunk_size, overlap=self.overlap,
                         source=source)
        # 이미지 부착은 **세그먼트 직후**에 한다: 청크의 char 오프셋이 있어야
        # 캡션 위치로 담을 청크를 찾을 수 있고, 저장 전이라 meta 가 그대로 실린다
        # (chunk_id 는 text 기준이므로 meta 를 붙여도 벡터 캐시가 안 무효화된다).
        self.apply_images(chunks, text, images, report)
        if self.extraction_mode == "topic":
            await self._extract_topic(report, chunks, source, trust=trust)
        else:
            await self._extract_exhaustive(report, chunks, source, trust=trust)

        # 재적재로 끊긴 근거 링크 복원. 위에서 이 소스의 청크를 전량 지웠는데
        # 그래프 노드는 남긴다(노드는 여러 소스에 걸칠 수 있고 검수 판정도
        # 붙어 있다). LLM 추출이 비결정적이라 이번에 안 뽑힌 노드는 그래프에
        # 남은 채 **근거를 잃는다** — 그렇게 고아 노드가 14%까지 쌓였고 그것이
        # evidence 채점의 천장을 정하고 있었다.
        #
        # 노드 attrs 의 chunk_index 힌트로 **원래 그 청크**만 좁게 복원한다.
        # 이름이 나오는 모든 청크로 넓히는 것은 검수 몫이다(orphan_links) —
        # 무인 실행이 넓게 이으면 provenance 가 느슨해진다.
        if self.store_chunks and self.chunk_store is not None:
            from ..core.relink import relink_by_chunk_index
            report.links_relinked += relink_by_chunk_index(
                self.kg.graph, self.chunk_store, source)

    async def _extract_one(self, report: BuildReport, chunk, source: str,
                           trust: str = "") -> None:
        """한 청크 LLM 추출 → 병합. 실패는 카운트만 하고 넘긴다."""
        try:
            raw_response = await self._call_llm_with_retry(
                build_extraction_prompt(self.schema, chunk.text))
            parsed = parse_llm_json(raw_response)
        except Exception as e:
            report.chunks_failed += 1
            logger.warning(f"⚠️ Extraction failed ({source}): {e}")
            return
        if parsed is None:
            report.chunks_failed += 1
            return
        extraction = clean_extraction(parsed, self.schema)
        await self._merge(report, extraction, chunk, trust=trust)
        report.chunks_processed += 1

    async def _extract_exhaustive(self, report: BuildReport, chunks, source: str,
                                  trust: str = "") -> None:
        """전 청크 추출(기본) — 완전하지만 무겁다."""
        for i, chunk in enumerate(chunks):
            self._progress(stage="extract", current=i + 1, total=len(chunks),
                           source=source)
            await self._extract_one(report, chunk, source, trust=trust)

    async def _extract_topic(self, report: BuildReport, chunks, source: str,
                             trust: str = "") -> None:
        """토픽 회수 추출(aicoach식) — 전 청크를 저장·임베딩(벡터DB 완전)한 뒤,
        토픽별 상위 청크만 LLM 추출. 대용량 약관에서 저비용·집중.

        불변식: 추출 여부와 무관하게 **모든 청크가 저장·색인**된다 — KB/벡터
        검색은 완전해야 하고, 온톨로지는 그 위의 선택적 레이어다.
        폴백: 임베더 없음/회수 0 이면 전체 추출로 degrade(빈 결과 방지)."""
        ns = getattr(self.kg, "namespace", "default")
        # 1) 전 청크 저장(추출 안 돼도) — node_ids 는 나중에 _merge 가 채운다
        by_id = {}
        if self.store_chunks and self.chunk_store is not None:
            for c in chunks:
                try:
                    cid = self.chunk_store.add(
                        c, (), trust=self.resolve_chunk_trust(c, trust, report))
                    by_id[cid] = c
                except Exception as e:
                    logger.warning(f"⚠️ Chunk store write failed ({source}): {e}")
            self.chunk_store.save_to_disk()
        # 2) 벡터 인덱스 refresh(임베딩)
        from ..core.chunk_index import get_chunk_index
        ci = get_chunk_index(ns)
        try:
            ci.refresh()
        except Exception as e:
            logger.warning(f"⚠️ 청크 인덱스 refresh 실패({source}): {e}")
        # 3) 토픽 결정 — 사용자 지정 우선, 없으면 LLM 유도(도메인 하드코딩 금지)
        topics = self.topics or await self._derive_topics(chunks)
        self._progress(stage="topics", source=source, topics=topics)
        # 4) 토픽별 상위 청크 회수(이 소스 것만), 첫 등장 순 중복 제거
        selected, seen = [], set()
        for topic in topics:
            try:
                hits = ci.search(topic, top_k=self.topic_top_k)
            except Exception:
                hits = []
            for stored, _score in hits:
                c = by_id.get(stored.chunk_id)
                if c is None or stored.chunk_id in seen:
                    continue
                seen.add(stored.chunk_id)
                selected.append(c)
        # 5) 폴백 — 회수 0(임베더 없음 등)이면 전체 추출로 degrade
        if not selected:
            logger.warning(f"⚠️ 토픽 회수 0 → 전체 추출 폴백: {source}")
            await self._extract_exhaustive(report, chunks, source, trust=trust)
            return
        for i, chunk in enumerate(selected):
            self._progress(stage="extract", current=i + 1, total=len(selected),
                           source=source)
            await self._extract_one(report, chunk, source, trust=trust)

    async def _derive_topics(self, chunks) -> List[str]:
        """문서 표본에서 핵심 토픽/조항유형 6~10개를 LLM 이 뽑는다(도메인 무관)."""
        sample = "\n\n".join(c.text for c in chunks[:8])[:4000]
        if not sample.strip():
            return []
        prompt = (
            "다음 문서에서 검색 질의로 쓸 핵심 주제/조항 유형을 6~10개 뽑아 "
            "JSON 문자열 배열로만 답하라(간결한 명사구, 설명 없이).\n\n"
            f"문서:\n{sample}")
        try:
            raw = await self._call_llm_with_retry(prompt)
            import json
            import re
            m = re.search(r"\[.*\]", raw, re.S)
            arr = json.loads(m.group(0)) if m else []
            topics = [str(t).strip() for t in arr if str(t).strip()]
            return topics[:10]
        except Exception as e:
            logger.warning(f"⚠️ 토픽 유도 실패(전체 추출로 폴백): {e}")
            return []

    def _store_chunk(self, chunk: Chunk, node_ids, trust: str = "",
                     report: Optional[BuildReport] = None) -> None:
        """원문 청크 + 이 청크에서 나온 노드들을 저장한다 (축 2).

        저장 실패가 빌드를 죽이면 안 된다 — 추출은 이미 끝났고 LLM 비용도
        이미 지불했다. 그래프는 살리고 청크만 잃는 쪽이 낫다.

        exhaustive 모드는 청크를 **이 경로로만** 저장하므로(topic 모드처럼 미리
        전량 저장하지 않는다) 품질 게이트도 여기 걸려야 기본 경로에서 작동한다.
        report 가 없으면 강등을 집계할 곳이 없으므로 게이트를 건너뛴다 —
        조용한 강등을 만들지 않는다.
        """
        if not self.store_chunks or self.chunk_store is None:
            return
        if report is not None:
            trust = self.resolve_chunk_trust(chunk, trust, report)
        try:
            self.chunk_store.add(chunk, node_ids, trust=trust)
        except Exception as e:
            logger.warning(f"⚠️ Chunk store write failed ({chunk.source}): {e}")

    def _remap_reclassified(self, node_id: str, report=None) -> str:
        """재분류된 id 를 새 id 로 재지도한다 — 묘비의 자매 (P-4, 2026-08-03).

        재분류(rename)는 묘비를 남기지 않으므로, 다음 재빌드에서 LLM 이 같은
        개체를 옛 타입으로 다시 추출한다. 차단하면 근거가 버려지고, 방치하면
        cross_type 중복이 부활한다 — 재지도하면 새 id 의 **보강**(근거
        합집합)이 된다. fail-open (묘비 관문과 같은 이유: 저장소 장애로
        빌드가 죽는 것보다 한 번의 부활이 싸다 — 부활은 보드에 잡힌다).
        """
        if not self.check_tombstones or self.review_store is None:
            return node_id
        try:
            target = self.review_store.reclassify_target(node_id)
        except Exception:
            return node_id
        if not target or target == node_id:
            return node_id
        if report is not None:
            report.entities_remapped += 1
        return target

    def _is_tombstoned(self, node_id: str) -> bool:
        """검수자가 거절한 노드인가 — 병합 전 관문.

        조회 실패는 False 로 넘긴다(fail-open): 묘비 저장소 장애로 빌드
        전체가 죽는 것보다 한 번의 부활이 싸다 — 다음 검수에서 다시 거절된다.
        """
        if not self.check_tombstones or self.review_store is None:
            return False
        try:
            return self.review_store.is_rejected(node_id)
        except Exception:
            return False

    def collect_images(self, path) -> List[Any]:
        """PDF 안 이미지를 ImageAsset 으로 수집한다. 게이트 off 면 빈 목록.

        추출 실패를 삼키는 이유: 이미지는 부가 정보다. 손상 PDF 하나로 문서
        인제스트(LLM 추출 비용을 이미 지불한)를 잃으면 손해가 훨씬 크다.
        """
        if not self.image_assets or not path:
            return []
        try:
            return readers.extract_pdf_images(path)
        except Exception as e:
            logger.warning(f"⚠️ 이미지 수집 실패 ({path}): {e}")
            return []

    def apply_images(self, chunks, text: str, assets, report: BuildReport) -> None:
        """이미지를 **캡션이 든 청크**에 meta 로 붙인다 (새 청크 만들지 않음).

        새 청크를 안 만드는 이유는 image_assets.attach_images_to_chunks 의
        규정과 같다: 합성 대리 텍스트는 원문의 부분 문자열이 아니라
        `original[char_start:char_end] == chunk.text` 불변식을 깬다.

        캡션 없음과 원문 미발견을 합쳐 unanchored 로 보고한다 — 사용자 입장에서
        둘 다 "검색되지 않는 이미지"라는 같은 사실이다(세부 갈래는 로그에 남는다).
        """
        if not self.image_assets or not assets:
            return
        try:
            from .image_assets import attach_images_to_chunks
            stats = attach_images_to_chunks(chunks, text, assets)
        except Exception as e:
            logger.warning(f"⚠️ 이미지 부착 실패: {e}")
            return
        report.images_attached += stats["attached"]
        report.images_unanchored += stats["unanchored"] + stats["no_caption"]
        if stats["attached"] or stats["unanchored"] or stats["no_caption"]:
            logger.info(
                f"🖼️ 이미지 부착 {stats['attached']}개 · 앵커없음 "
                f"{stats['unanchored']}개 · 캡션없음 {stats['no_caption']}개")

    def resolve_chunk_trust(self, chunk, trust: str,
                            report: BuildReport) -> str:
        """청크의 실효 trust — 게이트가 켜져 있으면 근거 아닌 조각을 강등한다.

        삭제하지 않는 이유는 guard_attrs_by_trust 의 규정과 같다: 요약서만 아는
        사실은 유효하고, 금지 대상은 덮어쓰기지 기여가 아니다. 목차·잡음 청크도
        저장되고 등급만 내려가 사실을 덮지 못한다.

        판정 실패는 삼킨다 — 한 청크의 품질 판정 오류로 인제스트가 멈추면
        (LLM 추출 비용을 이미 지불한) 빌드 전체를 잃는다.
        """
        if not self.quality_gate:
            return trust
        try:
            from .chunk_quality import DEMOTED_TRUST, assess_chunk
            verdict, reason, _ = assess_chunk(chunk.text)
        except Exception as e:
            logger.warning(f"⚠️ Chunk quality assess failed: {e}")
            return trust
        if verdict == trust or verdict == "ok":
            return trust
        # 들어온 등급이 이미 같거나 더 낮으면 강등이 아니다 — 집계가 부풀면
        # "몇 개를 실제로 강등했나"가 거짓이 된다.
        if _trust_rank(trust) <= _trust_rank(DEMOTED_TRUST):
            return trust
        report.chunks_demoted += 1
        report.demote_reasons[reason] = report.demote_reasons.get(reason, 0) + 1
        logger.debug(f"↓ chunk demoted ({reason}): {chunk.source}#{chunk.index}")
        return DEMOTED_TRUST

    async def _merge(self, report: BuildReport, extraction, chunk: Chunk,
                     trust: str = "") -> None:
        graph = self.kg.graph
        name_to_id: Dict[str, str] = {}

        for entity in extraction.entities:
            node_id = f"{entity['type']}:{entity['name']}"
            node_type = entity["type"]
            # 재지도(P-4) → 묘비 순서 — 재분류된 개체의 재추출은 새 id 의
            # 보강이 되고, 종착지가 묘비면 묘비가 이긴다.
            remapped = self._remap_reclassified(node_id, report)
            if remapped != node_id:
                node_id = remapped
                node_type = node_id.split(":", 1)[0] if ":" in node_id else node_type
            # 묘비 관문 — 거절된 개체는 재빌드에서 부활하지 않는다.
            # name_to_id 에 넣지 않아 이 개체를 참조하는 관계도 함께 버려진다.
            if self._is_tombstoned(node_id):
                report.entities_rejected += 1
                continue
            name_to_id[entity["name"]] = node_id
            is_new = node_id not in graph
            attributes: Dict[str, Any] = {
                "name": entity["name"],
                "source": chunk.source,
                "chunk_index": chunk.index,
                **entity.get("attrs", {}),
            }
            # 신뢰 가드 — add_concept 은 나중 쓰기가 이기므로 그 전에 거른다
            attributes = guard_attrs_by_trust(
                graph.nodes[node_id] if not is_new else {}, attributes, trust)
            await self.kg.add_concept(node_id, node_type, attributes)
            if is_new:
                report.entities_added += 1

        for relation in extraction.relations:
            # validator 가 관계 양끝이 추출 개체임을 보장하므로, 여기서
            # name_to_id 에 없다는 것은 정확히 "묘비로 건너뛰어졌다"는 뜻이다
            # — 유령 노드로의 간선을 만들지 않고 관계를 통째로 버린다.
            subject_id = name_to_id.get(relation["subject"])
            object_id = name_to_id.get(relation["object"])
            if subject_id is None or object_id is None:
                continue
            existing = self.kg.graph.get_edge_data(subject_id, object_id) or {}
            already = any(attrs.get("predicate") == relation["predicate"]
                          for attrs in existing.values())
            await self.kg.add_relationship(
                subject_id, object_id, relation["predicate"],
                {"source": chunk.source})
            if not already:
                report.relations_added += 1

        self._store_chunk(chunk, list(name_to_id.values()), trust=trust,
                          report=report)

        self._progress(stage="merge", source=chunk.source,
                       entities=len(extraction.entities),
                       relations=len(extraction.relations))
