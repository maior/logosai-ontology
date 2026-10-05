"""
신뢰 등급(trust layer) — 출처의 급이 사실의 병합 순서를 결정한다.

aicoach 가 실증한 필요다: 같은 상품에 통합약관(원본·권위)과 상품요약서
(boilerplate — aicoach 는 블록리스트로 걸러냈다)가 함께 들어온다. KG 의
add_concept 은 나중 쓰기가 이기는 update 라서, 요약서를 나중에 빌드하면
약관에서 뽑은 사실이 조용히 요약서 값으로 덮인다.

고정하는 계약:
1. trust 는 provenance 다 — 노드 attrs 와 청크에 기록된다.
2. **낮은 신뢰는 높은 신뢰의 기존 값을 덮지 못한다.** 빈 자리는 채울 수 있다
   (요약서만 아는 사실은 유효하다 — 금지는 덮어쓰기지 기여가 아니다).
3. 같은/높은 신뢰는 정상 병합 (기존 동작 보존 — trust 미사용 코드는 무영향).
4. 게이트: analyze 의 파일명 해석(trust)이 plan 초안에 실리고, ingest 가
   빌더까지 전달한다.
"""

import asyncio
import json

import pytest

from ontology.builder import BuilderSchema, OntologyBuilder
from ontology.builder.pipeline import TRUST_RANK, guard_attrs_by_trust
from ontology.core.chunk_store import ChunkStore, StoredChunk


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def kg():
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
    return KnowledgeGraphEngine(fast_mode=True, namespace="trust_test")


SCHEMA = BuilderSchema(node_types=["Clause", "Coverage"], predicates={})


def make_llm(definition: str):
    """definition 만 다른 같은 개체를 추출하는 가짜 LLM."""
    def llm(prompt: str) -> str:
        return json.dumps({"entities": [
            {"name": "청약철회", "type": "Clause",
             "attrs": {"definition": definition}}], "relations": []},
            ensure_ascii=False)
    return llm


# ─── 1. 순수 가드 함수 ───────────────────────────────────────────────

class TestGuardAttrsByTrust:
    def test_rank_ordering(self):
        assert TRUST_RANK["authoritative"] > TRUST_RANK[""]
        assert TRUST_RANK[""] > TRUST_RANK["summary"]
        assert TRUST_RANK["unknown"] == TRUST_RANK[""]

    def test_lower_trust_cannot_overwrite(self):
        existing = {"definition": "약관의 정의", "trust": "authoritative"}
        incoming = {"definition": "요약서의 정의", "extra": "새 사실"}
        guarded = guard_attrs_by_trust(existing, incoming, "summary")
        assert "definition" not in guarded          # 덮어쓰기 차단
        assert guarded["extra"] == "새 사실"         # 빈 자리는 채운다

    def test_equal_trust_merges_normally(self):
        existing = {"definition": "이전", "trust": "authoritative"}
        incoming = {"definition": "갱신"}
        guarded = guard_attrs_by_trust(existing, incoming, "authoritative")
        assert guarded["definition"] == "갱신"

    def test_higher_trust_overwrites(self):
        existing = {"definition": "요약서 정의", "trust": "summary"}
        incoming = {"definition": "약관 정의"}
        guarded = guard_attrs_by_trust(existing, incoming, "authoritative")
        assert guarded["definition"] == "약관 정의"
        assert guarded["trust"] == "authoritative"   # 급 자체는 승격

    def test_lower_trust_does_not_downgrade_the_trust_field(self):
        existing = {"trust": "authoritative"}
        guarded = guard_attrs_by_trust(existing, {"a": 1}, "summary")
        assert guarded.get("trust") != "summary"

    def test_no_trust_anywhere_is_a_plain_merge(self):
        """trust 를 쓰지 않는 기존 코드 경로는 동작이 변하지 않는다."""
        guarded = guard_attrs_by_trust({"a": "old"}, {"a": "new"}, "")
        assert guarded["a"] == "new"


# ─── 2. 빌더 관통 (text 경로) ────────────────────────────────────────

class TestBuilderTextPath:
    def test_trust_is_stamped_on_nodes_and_chunks(self, kg, tmp_path):
        store = ChunkStore(namespace="t", path=tmp_path / "c.jsonl")
        builder = OntologyBuilder(schema=SCHEMA, kg=kg,
                                  llm_fn=make_llm("약관 정의"),
                                  chunk_store=store, auto_save=False)
        run(builder.build_from_text("청약철회 조항.", source="약관.pdf",
                                    trust="authoritative"))
        assert kg.graph.nodes["Clause:청약철회"]["trust"] == "authoritative"
        assert store.all()[0].trust == "authoritative"

    def test_summary_cannot_overwrite_authoritative_fact(self, kg, tmp_path):
        """aicoach 시나리오 그대로: 약관 먼저, 요약서 나중."""
        store = ChunkStore(namespace="t", path=tmp_path / "c.jsonl")
        first = OntologyBuilder(schema=SCHEMA, kg=kg,
                                llm_fn=make_llm("약관에 근거한 정의"),
                                chunk_store=store, auto_save=False)
        run(first.build_from_text("청약철회 조항.", source="약관.pdf",
                                  trust="authoritative"))

        second = OntologyBuilder(schema=SCHEMA, kg=kg,
                                 llm_fn=make_llm("요약서의 느슨한 정의"),
                                 chunk_store=store, auto_save=False)
        run(second.build_from_text("청약철회 요약.", source="요약서.pdf",
                                   trust="summary"))

        node = kg.graph.nodes["Clause:청약철회"]
        assert node["definition"] == "약관에 근거한 정의"
        assert node["trust"] == "authoritative"

    def test_authoritative_overwrites_summary(self, kg, tmp_path):
        """역순 업로드 — 요약서 먼저 와도 약관이 오면 바로잡힌다."""
        store = ChunkStore(namespace="t", path=tmp_path / "c.jsonl")
        first = OntologyBuilder(schema=SCHEMA, kg=kg,
                                llm_fn=make_llm("요약서의 느슨한 정의"),
                                chunk_store=store, auto_save=False)
        run(first.build_from_text("요약.", source="요약서.pdf", trust="summary"))

        second = OntologyBuilder(schema=SCHEMA, kg=kg,
                                 llm_fn=make_llm("약관에 근거한 정의"),
                                 chunk_store=store, auto_save=False)
        run(second.build_from_text("조항.", source="약관.pdf",
                                   trust="authoritative"))
        assert kg.graph.nodes["Clause:청약철회"]["definition"] == "약관에 근거한 정의"

    def test_no_trust_build_is_unchanged(self, kg, tmp_path):
        """trust 미지정 빌드(기존 호출부 전부)는 attrs 에 trust 를 남기지 않는다."""
        store = ChunkStore(namespace="t", path=tmp_path / "c.jsonl")
        builder = OntologyBuilder(schema=SCHEMA, kg=kg, llm_fn=make_llm("정의"),
                                  chunk_store=store, auto_save=False)
        run(builder.build_from_text("조항.", source="d.md"))
        assert "trust" not in kg.graph.nodes["Clause:청약철회"]


# ─── 3. records 경로 ─────────────────────────────────────────────────

class TestRecordsPath:
    def test_trust_is_stamped_on_record_nodes(self, kg):
        builder = OntologyBuilder(schema=None, kg=kg, llm_fn=lambda p: "{}",
                                  store_chunks=False, auto_save=False)
        run(builder.build_from_records(
            [{"이름": "숭례문", "지정": "국보"}],
            {"node_type": "Site", "name_field": "이름",
             "relations": [{"field": "지정", "predicate": "hasDesignation",
                            "target_type": "Designation"}]},
            source="wikidata", trust="authoritative"))
        assert kg.graph.nodes["Site:숭례문"]["trust"] == "authoritative"
        assert kg.graph.nodes["Designation:국보"]["trust"] == "authoritative"


# ─── 4. 청크 저장소 왕복 ─────────────────────────────────────────────

class TestChunkTrustPersistence:
    def test_roundtrip_preserves_trust(self, tmp_path):
        from ontology.builder.models import Chunk
        path = tmp_path / "c.jsonl"
        store = ChunkStore(namespace="p", path=path)
        store.add(Chunk(text="본문", source="약관.pdf", index=0), ["N:1"],
                  trust="authoritative")
        store.save_to_disk()

        restored = ChunkStore(namespace="p", path=path)
        restored.load_from_disk()
        assert restored.all()[0].trust == "authoritative"

    def test_old_jsonl_without_trust_still_loads(self, tmp_path):
        """축 2 에서 만든 기존 청크 파일(trust 필드 없음)이 깨지면 안 된다."""
        path = tmp_path / "c.jsonl"
        path.write_text('{"chunk_id": "a", "text": "본문", "source": "d",'
                        ' "index": 0, "section": "", "char_start": 0,'
                        ' "char_end": 2, "node_ids": []}\n', encoding="utf-8")
        store = ChunkStore(namespace="p", path=path)
        store.load_from_disk()
        assert store.all()[0].trust == ""


# ─── 5. 게이트 관통 ──────────────────────────────────────────────────

class TestGateThreading:
    def test_plan_draft_carries_trust_from_filename_meta(self, tmp_path):
        """analyze 의 Gemini 파일명 해석(trust)이 plan 초안에 실린다 —
        사용자가 게이트에서 등급을 보고 고칠 수 있어야 한다."""
        from ontology.server.service import OntologyBuilderService

        def llm(prompt):
            if "파일명" in prompt:
                return json.dumps({"files": [
                    {"filename": "통합약관_x.md", "doc_kind": "약관",
                     "entity": "x", "version": None,
                     "trust": "authoritative"}]}, ensure_ascii=False)
            return "{}"

        service = OntologyBuilderService(data_dir=tmp_path, llm_fn=llm)
        saved = service.save_dataset([
            ("통합약관_x.md", "## 제1조 목적\n본문.\n\n## 제2조 정의\n본문.\n".encode())])
        report = run(service.analyze_dataset(saved["dataset_id"]))
        assert report["plan"][0]["trust"] == "authoritative"

    def test_ingest_threads_trust_to_nodes(self, tmp_path):
        from ontology.engines.knowledge_graph_clean import (
            KnowledgeGraphEngine, _kg_instances)
        from ontology.server.service import OntologyBuilderService

        ns = "trust_gate"
        _kg_instances[ns] = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
        try:
            service = OntologyBuilderService(
                data_dir=tmp_path, llm_fn=make_llm("정의"))
            saved = service.save_dataset([("약관.md", "청약철회 조항.".encode())])
            job_id = service.create_job(ns)
            run(service.run_ingest(
                job_id, saved["dataset_id"], ns,
                [{"filename": "약관.md", "route": "prose",
                  "trust": "authoritative"}],
                save=False, schema_mode="custom",
                custom_schema={"node_types": ["Clause"], "predicates": {}}))
            assert service.get_job(job_id)["status"] == "completed"
            node = _kg_instances[ns].graph.nodes["Clause:청약철회"]
            assert node["trust"] == "authoritative"
        finally:
            _kg_instances.pop(ns, None)
