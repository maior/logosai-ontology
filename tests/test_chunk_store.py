"""
축 2 — 청크 저장소 + span provenance.

빌더는 추출이 끝나면 원문을 버렸다. 노드 attrs 와 source 경로, chunk_index
만 남았고 그래서:
- 검색 결과로 보여줄 문장이 없다 → "비정형 데이터를 찾는다"가 성립 안 함
- 인용이 불가능하다 → aicoach 가 rag/legal.py 를 따로 만든 이유가 이것이다
  (인용이 p.47 이 아니라 제21조를 가리켜야 한다)
- 데이터셋 추출이 그래프를 되읽는 수준에 머문다 → 근거 문장이 없다

여기서 고정하는 계약:
1. 오프셋은 **원본 텍스트 기준으로 정확하다** — text[start:end] == chunk.text.
   이게 깨지면 인용이 엉뚱한 곳을 가리키므로 span provenance 전체가 무의미해진다.
2. chunk_id 는 내용 해시 — 같은 문서를 다시 인제스트해도 중복이 쌓이지 않는다.
3. 노드 ↔ 청크는 양방향으로 조회된다.
4. 저장/로드 왕복에서 정보가 손실되지 않는다.
"""

import pytest

from ontology.builder.models import Chunk
from ontology.builder.segmenter import segment
from ontology.core.chunk_store import (
    ChunkStore,
    StoredChunk,
    chunk_id_for,
    get_chunk_store,
    reset_chunk_stores,
)


@pytest.fixture(autouse=True)
def _clean_singletons():
    reset_chunk_stores()
    yield
    reset_chunk_stores()


@pytest.fixture
def store(tmp_path):
    return ChunkStore(namespace="testns", path=tmp_path / "chunks_testns.jsonl")


# ─── 1. span provenance — 오프셋 정확성 ──────────────────────────────

class TestSpanProvenance:
    """오프셋은 원본 기준으로 정확해야 한다. 이게 축 2 의 전부다."""

    def test_window_offsets_reproduce_text_exactly(self):
        text = ("사과는 빨갛다. " * 60).strip()
        chunks = segment(text, mode="window", chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_heading_offsets_reproduce_text_exactly(self):
        text = (
            "# 서론\n본문 하나입니다.\n\n"
            "제1조 (목적)\n이 약관은 목적을 정한다.\n\n"
            "제2조 (정의)\n용어의 뜻은 다음과 같다.\n"
        )
        chunks = segment(text, mode="heading")
        assert len(chunks) >= 3
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_offsets_survive_leading_whitespace(self):
        """segment() 는 앞뒤 공백을 strip 한다. 오프셋은 strip 전 원본 기준이어야
        한다 — 아니면 인용이 통째로 밀린다."""
        text = "\n\n\n   첫 문단입니다.\n\n둘째 문단입니다.\n\n"
        chunks = segment(text, mode="window", chunk_size=50, overlap=0)
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text
        assert chunks[0].char_start > 0  # 선행 공백만큼 밀려 있어야 정상

    def test_single_short_text_offsets(self):
        text = "  짧은 문서.  "
        chunks = segment(text, mode="auto", chunk_size=800)
        assert len(chunks) == 1
        assert text[chunks[0].char_start:chunks[0].char_end] == chunks[0].text

    def test_oversized_heading_section_offsets(self):
        """긴 절은 heading 모드에서도 다시 윈도우로 쪼개진다 — 그때도 오프셋은
        원본 기준이어야 한다 (조각의 조각이라 이중 보정이 필요한 경로)."""
        body = "가나다라마바사아자차. " * 200
        text = f"# 큰 절\n{body}\n\n# 작은 절\n짧다.\n"
        chunks = segment(text, mode="heading", chunk_size=200, overlap=40)
        assert len(chunks) > 2
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text


# ─── 2. chunk_id — 내용 해시, 멱등 ───────────────────────────────────

class TestChunkId:
    def test_same_content_same_id(self):
        a = chunk_id_for("doc.txt", 0, "같은 내용")
        b = chunk_id_for("doc.txt", 0, "같은 내용")
        assert a == b

    def test_different_source_different_id(self):
        a = chunk_id_for("a.txt", 0, "같은 내용")
        b = chunk_id_for("b.txt", 0, "같은 내용")
        assert a != b

    def test_different_index_different_id(self):
        assert chunk_id_for("d.txt", 0, "t") != chunk_id_for("d.txt", 1, "t")

    def test_different_text_different_id(self):
        assert chunk_id_for("d.txt", 0, "하나") != chunk_id_for("d.txt", 0, "둘")


# ─── 3. 저장소 기본 동작 ─────────────────────────────────────────────

class TestChunkStoreBasics:
    def test_add_returns_id_and_stores(self, store):
        chunk = Chunk(text="본문", source="d.txt", index=0,
                      char_start=0, char_end=2)
        cid = store.add(chunk, ["Concept:본문"])
        assert len(store) == 1
        stored = store.get(cid)
        assert isinstance(stored, StoredChunk)
        assert stored.text == "본문"
        assert stored.source == "d.txt"
        assert stored.char_start == 0 and stored.char_end == 2
        assert stored.node_ids == ["Concept:본문"]

    def test_re_adding_same_chunk_is_idempotent(self, store):
        """같은 문서를 다시 빌드해도 청크가 두 배가 되면 안 된다."""
        chunk = Chunk(text="본문", source="d.txt", index=0)
        first = store.add(chunk, ["Concept:A"])
        second = store.add(chunk, ["Concept:A"])
        assert first == second
        assert len(store) == 1

    def test_re_adding_merges_new_node_ids(self, store):
        """재빌드에서 다른 개체가 추출되면 노드 링크는 합집합으로 늘어난다."""
        chunk = Chunk(text="본문", source="d.txt", index=0)
        cid = store.add(chunk, ["Concept:A"])
        store.add(chunk, ["Concept:B", "Concept:A"])
        assert sorted(store.get(cid).node_ids) == ["Concept:A", "Concept:B"]

    def test_get_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_clear_empties_store(self, store):
        store.add(Chunk(text="t", source="d", index=0), ["N:1"])
        store.clear()
        assert len(store) == 0
        assert store.chunks_for_node("N:1") == []


# ─── 4. 양방향 조회 ──────────────────────────────────────────────────

class TestBidirectionalLookup:
    def test_chunks_for_node(self, store):
        store.add(Chunk(text="첫째", source="d.txt", index=0), ["Concept:A"])
        store.add(Chunk(text="둘째", source="d.txt", index=1), ["Concept:A"])
        store.add(Chunk(text="셋째", source="d.txt", index=2), ["Concept:B"])

        found = store.chunks_for_node("Concept:A")
        assert {c.text for c in found} == {"첫째", "둘째"}

    def test_chunks_for_unknown_node(self, store):
        assert store.chunks_for_node("Concept:없음") == []

    def test_nodes_for_chunk(self, store):
        cid = store.add(Chunk(text="t", source="d", index=0),
                        ["Concept:A", "Concept:B"])
        assert sorted(store.nodes_for_chunk(cid)) == ["Concept:A", "Concept:B"]

    def test_reverse_index_updated_on_merge(self, store):
        chunk = Chunk(text="t", source="d", index=0)
        store.add(chunk, ["Concept:A"])
        store.add(chunk, ["Concept:B"])
        assert len(store.chunks_for_node("Concept:B")) == 1


# ─── 5. 영속성 ───────────────────────────────────────────────────────

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        store = ChunkStore(namespace="ns", path=path)
        store.add(Chunk(text="본문 하나", source="d.txt", index=0,
                        section="제1조", char_start=5, char_end=10),
                  ["Concept:A", "Concept:B"])
        store.add(Chunk(text="본문 둘", source="d.txt", index=1), ["Concept:C"])
        assert store.save_to_disk() is True

        restored = ChunkStore(namespace="ns", path=path)
        restored.load_from_disk()
        assert len(restored) == 2
        found = restored.chunks_for_node("Concept:A")
        assert len(found) == 1
        assert found[0].text == "본문 하나"
        assert found[0].section == "제1조"
        assert found[0].char_start == 5 and found[0].char_end == 10
        assert sorted(found[0].node_ids) == ["Concept:A", "Concept:B"]

    def test_load_missing_file_is_not_an_error(self, tmp_path):
        store = ChunkStore(namespace="ns", path=tmp_path / "absent.jsonl")
        assert store.load_from_disk() is False
        assert len(store) == 0

    def test_load_skips_corrupt_lines(self, tmp_path):
        """한 줄이 깨졌다고 저장소 전체를 잃으면 안 된다 — JSONL 을 고른 이유."""
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            '{"chunk_id": "a", "text": "정상", "source": "d", "index": 0,'
            ' "section": "", "char_start": 0, "char_end": 2, "node_ids": []}\n'
            'this is not json\n'
            '{"chunk_id": "b", "text": "정상2", "source": "d", "index": 1,'
            ' "section": "", "char_start": 0, "char_end": 3, "node_ids": []}\n',
            encoding="utf-8")
        store = ChunkStore(namespace="ns", path=path)
        store.load_from_disk()
        assert len(store) == 2


# ─── 6. 네임스페이스 싱글턴 ──────────────────────────────────────────

class TestNamespaceSingleton:
    def test_same_namespace_same_instance(self):
        assert get_chunk_store("nsA") is get_chunk_store("nsA")

    def test_different_namespace_different_instance(self):
        assert get_chunk_store("nsA") is not get_chunk_store("nsB")

    def test_namespace_isolation(self):
        get_chunk_store("nsA").add(Chunk(text="A 것", source="a", index=0),
                                   ["Concept:A"])
        assert len(get_chunk_store("nsB")) == 0


# ─── 7. 키워드 폴백 검색 ─────────────────────────────────────────────

class TestKeywordSearch:
    """임베딩·ES 가 없어도 원문이 있으면 최소한의 구절 검색은 된다.
    (축 3 에서 ES 백엔드가 이 자리를 대체한다)"""

    def test_finds_chunk_by_substring(self, store):
        store.add(Chunk(text="청약철회권은 15일 이내 행사한다.", source="d", index=0), [])
        store.add(Chunk(text="보험금 지급 사유는 다음과 같다.", source="d", index=1), [])
        hits = store.search_text("청약철회")
        assert len(hits) == 1
        assert "청약철회권" in hits[0].text

    def test_empty_query_returns_nothing(self, store):
        store.add(Chunk(text="아무 본문", source="d", index=0), [])
        assert store.search_text("") == []

    def test_respects_top_k(self, store):
        for i in range(5):
            store.add(Chunk(text=f"보험금 관련 문장 {i}", source="d", index=i), [])
        assert len(store.search_text("보험금", top_k=3)) == 3


# ─── 8. 파이프라인 연결 (축 2 의 성과) ───────────────────────────────

SAMPLE_MD = """# 보험 상품 안내

## 제1조 청약철회
계약자는 보험증권을 받은 날부터 15일 이내에 청약을 철회할 수 있습니다.
금융소비자 보호에 관한 법률을 따릅니다.

## 제2조 보장내용
암진단비는 최초 1회에 한하여 지급합니다.
"""


def fake_llm(prompt: str) -> str:
    """결정적 가짜 추출기 — 프롬프트에 든 청크 본문으로 분기한다."""
    import json
    if "청약철회" in prompt:
        return json.dumps({
            "entities": [
                {"name": "청약철회", "type": "Clause", "attrs": {}},
                {"name": "금융소비자 보호에 관한 법률", "type": "Regulation", "attrs": {}},
            ],
            "relations": [
                {"subject": "청약철회", "predicate": "citesRegulation",
                 "object": "금융소비자 보호에 관한 법률"},
            ],
        }, ensure_ascii=False)
    if "암진단비" in prompt:
        return json.dumps({
            "entities": [{"name": "암진단비", "type": "Coverage", "attrs": {}}],
            "relations": [],
        }, ensure_ascii=False)
    return json.dumps({"entities": [], "relations": []})


_SCHEMA_TYPES = ["Clause", "Coverage", "Regulation"]
_SCHEMA_PREDICATES = {"citesRegulation": ("Clause", "Regulation")}


@pytest.fixture
def builder_parts(tmp_path):
    import asyncio

    from ontology.builder import BuilderSchema, OntologyBuilder
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

    kg = KnowledgeGraphEngine(fast_mode=True)
    store = ChunkStore(namespace="pipe", path=tmp_path / "chunks.jsonl")
    schema = BuilderSchema(node_types=_SCHEMA_TYPES,
                           predicates=_SCHEMA_PREDICATES)
    builder = OntologyBuilder(schema=schema, kg=kg, llm_fn=fake_llm,
                              chunk_store=store, auto_save=False)
    return builder, kg, store, asyncio.run


class TestPipelineWiring:
    """빌드가 끝나면 원문이 남아 있어야 한다 — 이전에는 버려졌다."""

    def test_build_stores_chunks(self, builder_parts):
        builder, _kg, store, run = builder_parts
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert len(store) > 0

    def test_stored_chunk_text_matches_source_offsets(self, builder_parts):
        """저장된 청크의 오프셋으로 원문을 다시 잘라내면 그 청크가 나온다.
        이 사슬이 인용의 근거다."""
        builder, _kg, store, run = builder_parts
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        for stored in store.all():
            assert SAMPLE_MD[stored.char_start:stored.char_end] == stored.text

    def test_extracted_node_links_back_to_its_chunk(self, builder_parts):
        builder, _kg, store, run = builder_parts
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))

        chunks = store.chunks_for_node("Clause:청약철회")
        assert len(chunks) == 1
        assert "청약철회" in chunks[0].text
        assert chunks[0].source == "약관.md"

    def test_each_node_links_to_the_chunk_it_came_from(self, builder_parts):
        builder, _kg, store, run = builder_parts
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))

        cancer = store.chunks_for_node("Coverage:암진단비")
        assert len(cancer) == 1
        assert "암진단비" in cancer[0].text
        # 청약철회 청크와는 다른 청크여야 한다
        withdrawal = store.chunks_for_node("Clause:청약철회")
        assert cancer[0].chunk_id != withdrawal[0].chunk_id

    def test_rebuild_is_idempotent(self, builder_parts):
        """같은 문서를 두 번 빌드해도 청크가 두 배가 되지 않는다."""
        builder, _kg, store, run = builder_parts
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        count = len(store)
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert len(store) == count

    def test_store_chunks_can_be_disabled(self, tmp_path):
        """원문 보존이 부담스러운 소비자를 위한 옵트아웃 (대용량 코퍼스)."""
        import asyncio

        from ontology.builder import BuilderSchema, OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        store = ChunkStore(namespace="off", path=tmp_path / "c.jsonl")
        builder = OntologyBuilder(
            schema=BuilderSchema(node_types=_SCHEMA_TYPES,
                                 predicates=_SCHEMA_PREDICATES),
            kg=KnowledgeGraphEngine(fast_mode=True), llm_fn=fake_llm,
            chunk_store=store, store_chunks=False, auto_save=False)
        asyncio.run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert len(store) == 0

    def test_chunks_persist_with_the_graph(self, tmp_path):
        """auto_save 는 그래프와 원문을 함께 내린다 — 둘은 한 빌드의 두 산출물."""
        import asyncio

        from ontology.builder import BuilderSchema, OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        kg = KnowledgeGraphEngine(fast_mode=True)
        kg_path = tmp_path / "kg.json"
        kg.save_to_disk = lambda: kg_path.write_text("{}", encoding="utf-8")
        chunk_path = tmp_path / "chunks.jsonl"
        store = ChunkStore(namespace="save", path=chunk_path)
        builder = OntologyBuilder(
            schema=BuilderSchema(node_types=_SCHEMA_TYPES,
                                 predicates=_SCHEMA_PREDICATES),
            kg=kg, llm_fn=fake_llm, chunk_store=store, auto_save=True)
        asyncio.run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert chunk_path.exists(), "auto_save 인데 청크가 디스크에 없다"

    def test_structured_records_do_not_create_chunks(self, tmp_path):
        """정형 레코드 인제스트는 원문 청크가 없다 — 청크는 비정형 전용."""
        import asyncio

        from ontology.builder import OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        store = ChunkStore(namespace="rec", path=tmp_path / "c.jsonl")
        builder = OntologyBuilder(schema=None, kg=KnowledgeGraphEngine(fast_mode=True),
                                  llm_fn=fake_llm, chunk_store=store,
                                  auto_save=False)
        asyncio.run(builder.build_from_records(
            [{"이름": "불국사"}], {"node_type": "Site", "name_field": "이름"},
            source="wikidata"))
        assert len(store) == 0


# ─── 9. 근거 있는 데이터셋 추출 (축 2 의 목적) ───────────────────────

class TestGroundedDataset:
    """축 2 이전의 qa 포맷은 그래프를 되읽는 수준이었다:
    "{subj}의 {pred} 관계에 있는 대상은?" → obj. 원문이 없으니 근거도 없었다.
    이제 원문이 남으므로 **추출 지도학습쌍**(원문 → 개체)을 만들 수 있고,
    양쪽 모두 실제 데이터라 환각이 0 이다.
    """

    @pytest.fixture
    def built(self, tmp_path, monkeypatch):
        import asyncio

        from ontology.builder import BuilderSchema, OntologyBuilder
        from ontology.engines.knowledge_graph_clean import (
            KnowledgeGraphEngine, _kg_instances)
        from ontology.server.service import OntologyBuilderService

        ns = "evidence_ns"
        kg = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
        _kg_instances[ns] = kg
        store = ChunkStore(namespace=ns, path=tmp_path / "chunks.jsonl")
        import ontology.core.chunk_store as cs_mod
        cs_mod._stores[ns] = store

        builder = OntologyBuilder(
            schema=BuilderSchema(node_types=_SCHEMA_TYPES,
                                 predicates=_SCHEMA_PREDICATES),
            kg=kg, llm_fn=fake_llm, chunk_store=store, auto_save=False)
        asyncio.run(builder.build_from_text(SAMPLE_MD, source="약관.md"))

        yield OntologyBuilderService(data_dir=tmp_path / "ds"), ns
        _kg_instances.pop(ns, None)

    def test_evidence_format_pairs_real_text_with_real_entities(self, built):
        service, ns = built
        result = service.build_training_dataset(ns, formats=["evidence"])
        rows = [r for r in result["rows"] if r["format"] == "evidence"]
        assert rows, "evidence 행이 없다"

        row = next(r for r in rows if "청약철회" in r["input"])
        assert row["input"] in SAMPLE_MD          # 입력은 실제 원문 그대로
        assert {"name": "청약철회", "type": "Clause"} in row["output"]
        assert row["source"] == "약관.md"
        assert row["chunk_id"]

    def test_evidence_rows_carry_exact_span(self, built):
        service, ns = built
        result = service.build_training_dataset(ns, formats=["evidence"])
        for row in result["rows"]:
            assert SAMPLE_MD[row["char_start"]:row["char_end"]] == row["input"]

    def test_evidence_skips_chunks_with_no_entities(self, built):
        """개체가 안 나온 청크는 학습쌍이 아니다 — 빈 output 을 정답으로
        가르치면 모델이 '아무것도 없다'를 배운다."""
        service, ns = built
        result = service.build_training_dataset(ns, formats=["evidence"])
        assert all(r["output"] for r in result["rows"])

    def test_qa_rows_can_be_grounded_with_evidence(self, built):
        """include_evidence=True → 관계 QA 에 근거 원문이 붙는다."""
        service, ns = built
        result = service.build_training_dataset(
            ns, formats=["qa"], include_evidence=True)
        grounded = [r for r in result["rows"] if r.get("evidence")]
        assert grounded, "근거가 붙은 qa 행이 없다"
        for row in grounded:
            assert row["evidence"] in SAMPLE_MD

    def test_qa_without_include_evidence_is_unchanged(self, built):
        """기존 계약 보존 — 옵트인하지 않으면 행 모양이 그대로다."""
        service, ns = built
        result = service.build_training_dataset(ns, formats=["qa"])
        assert all("evidence" not in r for r in result["rows"])

    def test_evidence_is_a_known_format(self, built):
        service, ns = built
        assert "evidence" in service.DATASET_FORMATS
        with pytest.raises(ValueError, match="unknown dataset format"):
            service.build_training_dataset(ns, formats=["nope"])


# ─── 소스별 멱등 재적재 (Phase 1-3) ─────────────────────────────────

def test_delete_by_source_removes_only_that_source(store):
    store.add(Chunk(text="문서A 첫 조각", source="a.pdf", index=0))
    store.add(Chunk(text="문서A 둘째 조각", source="a.pdf", index=1))
    store.add(Chunk(text="문서B 조각", source="b.pdf", index=0))
    removed = store.delete_by_source("a.pdf")
    assert removed == 2
    sources = {c.source for c in store.all()}
    assert sources == {"b.pdf"}


def test_delete_by_source_cleans_node_index(store):
    store.add(Chunk(text="조각", source="a.pdf", index=0), node_ids=["Clause:x"])
    assert store.chunks_for_node("Clause:x")          # 링크 존재
    store.delete_by_source("a.pdf")
    assert store.chunks_for_node("Clause:x") == []    # 역인덱스 정리됨


def test_reingest_changed_source_replaces_not_accumulates(store):
    # 같은 소스, 내용이 바뀐 재적재 → 옛 조각은 사라지고 새 조각만 남는다
    store.add(Chunk(text="구버전 본문", source="약관.pdf", index=0))
    store.delete_by_source("약관.pdf")                # 재적재 전 purge
    store.add(Chunk(text="신버전 본문", source="약관.pdf", index=0))
    texts = [c.text for c in store.all() if c.source == "약관.pdf"]
    assert texts == ["신버전 본문"]


# ─── 사후 링크 (커버리지 gap 승인의 청크 쪽 절반) ────────────────────

class TestLinkNode:
    """추출이 놓친 노드↔청크 링크를 나중에 복구한다.

    add() 로도 합집합 병합이 되지만 그건 StoredChunk 가 Chunk 와 같은 속성
    이름을 가진 **우연**에 의존한다. 링크만 필요한 호출자에게 저장 경로를
    쓰게 하면, 나중에 add() 가 trust/meta 규칙을 바꿀 때 조용히 끌려간다.
    """

    def test_links_existing_chunk(self, store):
        cid = store.add(Chunk(text="제6조 암진단비를 지급한다", source="약관.pdf",
                              index=0))
        assert store.link_node(cid, "InsuranceTerm:암진단비") is True
        assert store.nodes_for_chunk(cid) == ["InsuranceTerm:암진단비"]

    def test_updates_reverse_index(self, store):
        """역색인을 안 고치면 chunks_for_node 가 근거를 못 찾는다 —
        링크를 만든 목적 자체가 사라진다."""
        cid = store.add(Chunk(text="본문", source="a.pdf", index=0))
        store.link_node(cid, "Disease:암")
        found = store.chunks_for_node("Disease:암")
        assert [c.chunk_id for c in found] == [cid]

    def test_unknown_chunk_is_false_and_creates_nothing(self, store):
        """없는 청크에 링크를 만들면 허공을 가리키는 참조가 된다
        (graph_health 의 dangling_node_refs)."""
        assert store.link_node("nosuchchunk", "Disease:암") is False
        assert store.chunks_for_node("Disease:암") == []

    def test_duplicate_link_is_idempotent(self, store):
        cid = store.add(Chunk(text="본문", source="a.pdf", index=0))
        assert store.link_node(cid, "Disease:암") is True
        assert store.link_node(cid, "Disease:암") is False   # 이미 있다
        assert store.nodes_for_chunk(cid) == ["Disease:암"]
        assert len(store.chunks_for_node("Disease:암")) == 1

    def test_blank_arguments_rejected(self, store):
        cid = store.add(Chunk(text="본문", source="a.pdf", index=0))
        assert store.link_node(cid, "") is False
        assert store.link_node("", "Disease:암") is False

    def test_link_preserves_existing_links(self, store):
        cid = store.add(Chunk(text="본문", source="a.pdf", index=0),
                        node_ids=["Clause:제6조"])
        store.link_node(cid, "Disease:암")
        assert store.nodes_for_chunk(cid) == ["Clause:제6조", "Disease:암"]

    def test_link_survives_roundtrip(self, store, tmp_path):
        cid = store.add(Chunk(text="본문", source="a.pdf", index=0))
        store.link_node(cid, "Disease:암")
        store.save_to_disk()
        fresh = ChunkStore(namespace="testns", path=store.path)
        fresh.load_from_disk()
        assert [c.chunk_id for c in fresh.chunks_for_node("Disease:암")] == [cid]
