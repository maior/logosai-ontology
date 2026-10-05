"""
Ontology Builder Framework tests — arbitrary data → ontology pipeline.

All LLM calls are injected fakes (deterministic). Namespace isolation,
readers, segmentation, closed-schema extraction, validation (aicoach
clean_extraction/noise-filter port), and end-to-end pipeline.
"""

import asyncio
import json

import pytest


def run(coro):
    return asyncio.run(coro)


# ─── Fixtures ───────────────────────────────────────────────────────

SAMPLE_MD = """# 보험 상품 안내

## 제1조 청약철회
계약자는 보험증권을 받은 날부터 15일 이내에 청약을 철회할 수 있습니다.
금융소비자 보호에 관한 법률을 따릅니다.

## 제2조 보장내용
암진단비는 최초 1회에 한하여 지급합니다.
"""


def fake_llm(prompt: str) -> str:
    """Deterministic fake extractor: returns entities/relations based on
    which chunk text is inside the prompt."""
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
        return "```json\n" + json.dumps({
            "entities": [{"name": "암진단비", "type": "Coverage", "attrs": {}}],
            "relations": [],
        }, ensure_ascii=False) + "\n```"
    return json.dumps({"entities": [], "relations": []})


@pytest.fixture()
def schema():
    from ontology.builder import BuilderSchema
    return BuilderSchema(
        node_types=["Document", "Clause", "Coverage", "Regulation"],
        predicates={
            "hasProvision": ("Document", "Clause"),
            "citesRegulation": ("Clause", "Regulation"),
            "hasCoverage": ("Document", "Coverage"),
        },
    )


@pytest.fixture()
def kg():
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
    return KnowledgeGraphEngine(fast_mode=True)


# ─── 1. Namespace support ───────────────────────────────────────────

class TestNamespace:
    def test_default_singleton_unchanged(self):
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        a = get_knowledge_graph_engine()
        b = get_knowledge_graph_engine("default")
        assert a is b

    def test_namespace_gets_separate_instance_and_graph(self):
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        default = get_knowledge_graph_engine()
        ns = get_knowledge_graph_engine("test_builder_ns")
        assert ns is not default
        assert get_knowledge_graph_engine("test_builder_ns") is ns  # cached

        run(ns.add_concept("only_in_ns", "Clause", {}))
        assert "only_in_ns" in ns.graph
        assert "only_in_ns" not in default.graph

    def test_checkpoint_filename_per_namespace(self):
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        default_engine = KnowledgeGraphEngine(fast_mode=True)
        ns_engine = KnowledgeGraphEngine(fast_mode=True, namespace="insurance")
        assert default_engine.checkpoint_path.name == "kg_checkpoint.json"
        assert ns_engine.checkpoint_path.name == "kg_insurance.json"

    def test_namespace_save_load_roundtrip(self, tmp_path):
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        engine = KnowledgeGraphEngine(fast_mode=True, namespace="rt")
        run(engine.add_concept("n1", "Clause", {"name": "n1"}))
        path = tmp_path / "kg_rt.json"
        assert engine.save_to_disk(str(path)) is True

        restored = KnowledgeGraphEngine(fast_mode=True, namespace="rt")
        assert restored.load_from_disk(str(path)) is True
        assert "n1" in restored.graph


# ─── 2. Readers ─────────────────────────────────────────────────────

class TestReaders:
    def test_read_txt_and_md(self, tmp_path):
        from ontology.builder.readers import read_file
        f = tmp_path / "a.txt"
        f.write_text("텍스트 내용", encoding="utf-8")
        assert read_file(f) == "텍스트 내용"
        m = tmp_path / "b.md"
        m.write_text(SAMPLE_MD, encoding="utf-8")
        assert "청약철회" in read_file(m)

    def test_read_json_flattens_to_text(self, tmp_path):
        from ontology.builder.readers import read_file
        f = tmp_path / "d.json"
        f.write_text(json.dumps({"상품명": "암보험", "보장": ["진단", "수술"]},
                                ensure_ascii=False), encoding="utf-8")
        text = read_file(f)
        assert "상품명" in text and "암보험" in text

    def test_read_csv_as_labeled_rows(self, tmp_path):
        from ontology.builder.readers import read_file
        f = tmp_path / "t.csv"
        f.write_text("이름,보장\n암보험,진단비\n", encoding="utf-8")
        text = read_file(f)
        assert "이름" in text and "암보험" in text

    def test_unsupported_extension_raises(self, tmp_path):
        from ontology.builder.readers import read_file, UnsupportedFormatError
        f = tmp_path / "x.exe"
        f.write_text("binary", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            read_file(f)

    def test_missing_file_raises(self, tmp_path):
        from ontology.builder.readers import read_file
        with pytest.raises(FileNotFoundError):
            read_file(tmp_path / "nope.txt")

    def test_read_folder_collects_supported_files(self, tmp_path):
        from ontology.builder.readers import read_folder
        (tmp_path / "a.txt").write_text("하나", encoding="utf-8")
        (tmp_path / "b.md").write_text("둘", encoding="utf-8")
        (tmp_path / "c.exe").write_text("skip", encoding="utf-8")
        docs = read_folder(tmp_path)
        sources = sorted(s for s, _ in docs)
        assert len(docs) == 2
        assert sources[0].endswith("a.txt") and sources[1].endswith("b.md")


# ─── 3. Segmenter ───────────────────────────────────────────────────

class TestSegmenter:
    def test_heading_mode_splits_on_headings(self):
        from ontology.builder.segmenter import segment
        chunks = segment(SAMPLE_MD, mode="heading", source="s.md")
        texts = [c.text for c in chunks]
        assert len(chunks) >= 2
        assert any("청약철회" in t for t in texts)
        assert any("암진단비" in t for t in texts)
        # 청약철회 조항과 암진단비 조항이 서로 다른 청크
        assert not any("청약철회" in t and "암진단비" in t for t in texts)

    def test_window_mode_respects_size(self):
        from ontology.builder.segmenter import segment
        long_text = "문장입니다. " * 300
        chunks = segment(long_text, mode="window", chunk_size=200, overlap=50)
        assert len(chunks) > 1
        assert all(len(c.text) <= 260 for c in chunks)  # size + boundary slack

    def test_short_text_single_chunk(self):
        from ontology.builder.segmenter import segment
        chunks = segment("짧은 글", mode="auto")
        assert len(chunks) == 1
        assert chunks[0].text == "짧은 글"

    def test_empty_text_returns_nothing(self):
        from ontology.builder.segmenter import segment
        assert segment("", mode="auto") == []
        assert segment("   \n  ", mode="auto") == []

    def test_chunks_carry_source_and_index(self):
        from ontology.builder.segmenter import segment
        chunks = segment(SAMPLE_MD, mode="heading", source="doc.md")
        assert all(c.source == "doc.md" for c in chunks)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    # ── 구조-인지 조문 세그먼테이션(약관) — aicoach legal.py 이식 ──
    def test_article_headings_get_citation_labels(self):
        from ontology.builder.segmenter import segment
        doc = "제1조(목적) 이 계약은 보장을 목적으로 한다.\n제2조(정의) 용어의 뜻은 다음과 같다.\n"
        labels = [c.section for c in segment(doc, mode="heading") if c.section]
        assert "제1조(목적)" in labels and "제2조(정의)" in labels

    def test_inline_article_reference_does_not_create_heading(self):
        from ontology.builder.segmenter import segment
        # 본문 속 "제3조(...)의 …" 참조는 새 조항 헤딩이 되면 안 된다(조사 직접부착)
        doc = ("제1조(목적) 이 계약은 제3조(보험금의 지급)의 기준에 따른다. 이어지는 본문.\n"
               "제2조(정의) 용어 정의.\n")
        labels = [c.section for c in segment(doc, mode="heading") if c.section]
        assert "제1조(목적)" in labels and "제2조(정의)" in labels
        assert not any(l.startswith("제3조") for l in labels)   # 참조일 뿐

    def test_toc_dot_leaders_are_dropped(self):
        from ontology.builder.segmenter import segment
        toc = ("제1관 목적 " + "." * 30 + " 1\n"
               + "제2관 계약 " + "." * 30 + " 5\n"
               + "".join("세부항목 " + "." * 20 + f" {i}\n" for i in range(6)))
        body = "제1조(목적) 실제 조문 본문입니다. 충분히 긴 내용을 담고 있습니다."
        texts = [c.text for c in segment(toc + "\n" + body, mode="heading")]
        assert any("실제 조문 본문" in t for t in texts)      # 진짜 조문은 남고
        assert not any("." * 20 in t for t in texts)          # 목차 점선은 버려짐


# ─── 4. Extractor (prompt / lenient parse) ─────────────────────────

class TestExtractor:
    def test_prompt_injects_closed_schema_and_text(self, schema):
        from ontology.builder.extractor import build_extraction_prompt
        prompt = build_extraction_prompt(schema, "청약철회는 15일 이내 가능")
        for t in schema.node_types:
            assert t in prompt
        for p in schema.predicates:
            assert p in prompt
        assert "청약철회는 15일 이내 가능" in prompt
        # aicoach grounding 원칙이 프롬프트에 명시
        assert "없는" in prompt  # "텍스트에 없는 사실 금지" 계열 문구

    def test_parse_llm_json_handles_codefence(self):
        from ontology.builder.extractor import parse_llm_json
        raw = '```json\n{"entities": [], "relations": []}\n```'
        assert parse_llm_json(raw) == {"entities": [], "relations": []}

    def test_parse_llm_json_handles_surrounding_prose(self):
        from ontology.builder.extractor import parse_llm_json
        raw = '추출 결과입니다:\n{"entities": [{"name": "a", "type": "Clause"}], "relations": []}\n이상입니다.'
        parsed = parse_llm_json(raw)
        assert parsed["entities"][0]["name"] == "a"

    def test_parse_llm_json_garbage_returns_none(self):
        from ontology.builder.extractor import parse_llm_json
        assert parse_llm_json("완전히 망가진 응답") is None


# ─── 5. Validator (clean_extraction port) ──────────────────────────

class TestValidator:
    def test_drops_unknown_types_and_predicates(self, schema):
        from ontology.builder.validator import clean_extraction
        raw = {
            "entities": [
                {"name": "청약철회", "type": "Clause"},
                {"name": "이상한것", "type": "Alien"},          # unknown type
            ],
            "relations": [
                {"subject": "청약철회", "predicate": "citesRegulation", "object": "금소법"},
                {"subject": "청약철회", "predicate": "unknownPred", "object": "금소법"},
            ],
        }
        result = clean_extraction(raw, schema)
        assert [e["name"] for e in result.entities] == ["청약철회"]
        # citesRegulation의 object '금소법'은 entities에 없으므로 dangling → drop
        assert result.relations == []

    def test_keeps_relation_when_endpoints_extracted(self, schema):
        from ontology.builder.validator import clean_extraction
        raw = {
            "entities": [
                {"name": "청약철회", "type": "Clause"},
                {"name": "금소법", "type": "Regulation"},
            ],
            "relations": [
                {"subject": "청약철회", "predicate": "citesRegulation", "object": "금소법"},
            ],
        }
        result = clean_extraction(raw, schema)
        assert len(result.relations) == 1

    def test_noise_names_filtered(self, schema):
        # aicoach 노이즈 필터: 지시어 stub, 너무 짧은 이름
        from ontology.builder.validator import clean_extraction
        raw = {
            "entities": [
                {"name": "이 특약", "type": "Clause"},   # 지시어 stub
                {"name": "그 보장", "type": "Coverage"}, # 지시어 stub
                {"name": "암", "type": "Coverage"},      # 2자 미만? '암'은 1자 — drop
                {"name": "암진단비", "type": "Coverage"},
            ],
            "relations": [],
        }
        result = clean_extraction(raw, schema)
        assert [e["name"] for e in result.entities] == ["암진단비"]

    def test_empty_or_none_input(self, schema):
        from ontology.builder.validator import clean_extraction
        assert clean_extraction(None, schema).entities == []
        assert clean_extraction({}, schema).entities == []


# ─── 6. BuilderSchema ───────────────────────────────────────────────

class TestBuilderSchema:
    def test_from_dict_and_preset(self):
        from ontology.builder import BuilderSchema
        s = BuilderSchema.from_dict({
            "node_types": ["A", "B"],
            "predicates": {"rel": ["A", "B"]},
        })
        assert s.node_types == ["A", "B"]
        assert s.predicates["rel"] == ("A", "B")

        preset = BuilderSchema.preset_document()
        assert preset.node_types  # non-empty
        assert preset.predicates


# ─── 7. Pipeline end-to-end (fake LLM) ─────────────────────────────

class TestPipeline:
    def _builder(self, schema, kg, **kw):
        from ontology.builder import OntologyBuilder
        return OntologyBuilder(schema=schema, llm_fn=fake_llm, kg=kg,
                               auto_save=False, **kw)

    def test_build_from_text_creates_typed_nodes_with_provenance(self, schema, kg):
        builder = self._builder(schema, kg)
        report = run(builder.build_from_text(SAMPLE_MD, source="sample.md"))

        assert "Clause:청약철회" in kg.graph
        assert "Coverage:암진단비" in kg.graph
        node = kg.graph.nodes["Clause:청약철회"]
        assert node["type"] == "Clause"
        assert node["source"] == "sample.md"        # provenance 필수
        assert report.entities_added >= 3
        assert report.relations_added >= 1

    def test_relations_merged_with_predicate(self, schema, kg):
        builder = self._builder(schema, kg)
        run(builder.build_from_text(SAMPLE_MD, source="s.md"))
        edges = kg.graph.get_edge_data(
            "Clause:청약철회", "Regulation:금융소비자 보호에 관한 법률")
        assert edges and any(
            e.get("predicate") == "citesRegulation" for e in edges.values())

    def test_rebuild_does_not_duplicate(self, schema, kg):
        builder = self._builder(schema, kg)
        run(builder.build_from_text(SAMPLE_MD, source="s.md"))
        edges_before = kg.graph.number_of_edges()
        run(builder.build_from_text(SAMPLE_MD, source="s.md"))  # 같은 데이터 재실행
        assert kg.graph.number_of_edges() == edges_before

    def test_progress_callback_invoked(self, schema, kg):
        events = []
        builder = self._builder(schema, kg,
                                progress_cb=lambda **e: events.append(e))
        run(builder.build_from_text(SAMPLE_MD, source="s.md"))
        assert events
        stages = {e.get("stage") for e in events}
        assert "extract" in stages

    def test_build_from_file_and_folder(self, schema, kg, tmp_path):
        (tmp_path / "one.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "two.txt").write_text("암진단비 관련 문서", encoding="utf-8")
        (tmp_path / "skip.exe").write_text("x", encoding="utf-8")

        builder = self._builder(schema, kg)
        report = run(builder.build_from_folder(tmp_path))
        assert report.files_read == 2
        assert "Coverage:암진단비" in kg.graph
        assert report.errors == []

    def test_llm_failure_on_one_chunk_does_not_abort(self, schema, kg):
        calls = {"n": 0}

        def flaky_llm(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("LLM down")
            return fake_llm(prompt)

        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=schema, llm_fn=flaky_llm, kg=kg,
                                  auto_save=False)
        report = run(builder.build_from_text(SAMPLE_MD, source="s.md"))
        assert report.chunks_failed >= 1
        assert report.entities_added >= 1  # 나머지 청크는 계속 처리됨

    def test_transient_llm_error_retried(self, schema, kg):
        # 503/429 같은 일시 오류는 재시도로 살아나야 한다 (aicoach resilient 패턴)
        calls = {"n": 0}

        def transient_llm(prompt):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("Gemini API call failed: 503 UNAVAILABLE")
            return fake_llm(prompt)

        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=schema, llm_fn=transient_llm, kg=kg,
                                  auto_save=False, retry_base_delay=0.01)
        report = run(builder.build_from_text(SAMPLE_MD, source="s.md"))
        assert report.chunks_failed == 0          # 재시도로 전부 성공
        assert calls["n"] >= 4                    # 실패 2회 + 재시도 포함

    def test_permanent_llm_error_not_retried_forever(self, schema, kg):
        calls = {"n": 0}

        def broken_llm(prompt):
            calls["n"] += 1
            raise RuntimeError("invalid api key")  # 영구 오류 — 즉시 포기

        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=schema, llm_fn=broken_llm, kg=kg,
                                  auto_save=False, retry_base_delay=0.01)
        report = run(builder.build_from_text(SAMPLE_MD, source="s.md"))
        assert report.chunks_failed >= 1
        assert calls["n"] == report.chunks_failed  # 청크당 1회 — 재시도 없음


# ─── 8. Build options (모델 선택·프리셋·초기화) ─────────────────────

class TestBuildOptions:
    def test_llm_model_default_is_gemini_35_flash(self, schema, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=schema, kg=kg, auto_save=False)
        assert builder.llm_model == "gemini-3.5-flash"

    def test_llm_model_override(self, schema, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=schema, kg=kg, auto_save=False,
                                  llm_model="gemini-2.5-pro")
        assert builder.llm_model == "gemini-2.5-pro"

    def test_schema_preset_registry(self):
        from ontology.builder import BuilderSchema
        generic = BuilderSchema.preset("generic")
        assert generic.node_types and generic.predicates
        document = BuilderSchema.preset("document")
        assert document.node_types == BuilderSchema.preset_document().node_types
        with pytest.raises(ValueError):
            BuilderSchema.preset("no_such_preset")

    def test_preset_names_listed(self):
        from ontology.builder import BuilderSchema
        names = BuilderSchema.preset_names()
        assert "document" in names and "generic" in names

    def test_engine_clear_empties_graph(self, kg):
        run(kg.add_concept("n1", "Clause", {}))
        run(kg.add_concept("n2", "Clause", {}))
        assert kg.graph.number_of_nodes() == 2
        kg.clear()
        assert kg.graph.number_of_nodes() == 0
        assert kg.graph.number_of_edges() == 0


# ─── 9. LLM 프로바이더 · auto 스키마 · OWL 내보내기 ─────────────────

class TestLLMProviders:
    def test_provider_classes(self):
        # 2026-07-07 L3 스윕: langchain 래퍼 → LLMClient 기반 ainvoke 호환 래퍼
        # 2026-07-17 축 1(커널 분리): logosai 구상 클래스 반환 → core.llm_provider
        #   프로토콜(`await .complete(prompt) -> str`)로 계약 변경. 커널이
        #   에이전트 프레임워크에 하드 의존하지 않게 하는 것이 목적이므로,
        #   여기서 logosai 타입을 단언하면 분리 자체를 되돌리게 된다.
        from ontology.builder.pipeline import make_llm_client
        from ontology.core.llm_provider import GeminiProvider, LogosAIProvider

        google = make_llm_client("google", "gemini-3.5-flash")
        assert isinstance(google, GeminiProvider)
        assert google.model == "gemini-3.5-flash"

        for prov, model in [("openai", "gpt-4o-mini"),
                            ("anthropic", "claude-haiku-4-5-20251001")]:
            llm = make_llm_client(prov, model)
            assert isinstance(llm, LogosAIProvider)
            assert llm._client.provider == prov
            assert llm._client.model == model

        compat = make_llm_client("openai_compatible", "qwen2.5-7b",
                                 base_url="http://localhost:9825/v1")
        assert isinstance(compat, LogosAIProvider)
        assert compat._client.base_url == "http://localhost:9825/v1"

    def test_every_provider_exposes_the_complete_protocol(self):
        """커널이 LLM 에 요구하는 전부: complete(prompt) -> str."""
        from ontology.builder.pipeline import make_llm_client

        for prov, model, kw in [("google", "gemini-3.5-flash", {}),
                                ("openai", "gpt-4o-mini", {}),
                                ("anthropic", "claude-haiku-4-5-20251001", {}),
                                ("openai_compatible", "qwen2.5-7b",
                                 {"base_url": "http://localhost:9825/v1"})]:
            llm = make_llm_client(prov, model, **kw)
            assert callable(getattr(llm, "complete", None)), prov

    def test_default_models_per_provider(self):
        from ontology.builder.pipeline import DEFAULT_MODELS
        assert DEFAULT_MODELS["google"] == "gemini-3.5-flash"
        assert "openai" in DEFAULT_MODELS and "anthropic" in DEFAULT_MODELS

    def test_unknown_provider_rejected(self):
        from ontology.builder.pipeline import make_llm_client
        with pytest.raises(ValueError):
            make_llm_client("no_such", "m")

    def test_openai_compatible_requires_base_url(self):
        from ontology.builder.pipeline import make_llm_client
        with pytest.raises(ValueError):
            make_llm_client("openai_compatible", "qwen2.5-7b")


def schema_proposing_llm(prompt: str) -> str:
    """스키마 제안 요청이면 스키마를, 추출 요청이면 개체를 반환하는 가짜 LLM."""
    if "스키마를 제안" in prompt:
        return json.dumps({
            "node_types": ["HeritageSite", "Region"],
            "predicates": {"locatedIn": ["HeritageSite", "Region"]},
        }, ensure_ascii=False)
    return json.dumps({
        "entities": [
            {"name": "석굴암", "type": "HeritageSite",
             "attrs": {"lat": 35.795, "lng": 129.349, "category": "문화"}},
            {"name": "경주", "type": "Region", "attrs": {}},
        ],
        "relations": [
            {"subject": "석굴암", "predicate": "locatedIn", "object": "경주"},
        ],
    }, ensure_ascii=False)


class TestAutoSchema:
    def test_auto_mode_proposes_and_uses_schema(self, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, llm_fn=schema_proposing_llm,
                                  kg=kg, auto_save=False)
        report = run(builder.build_from_text("석굴암은 경주에 있다", source="s.txt"))

        assert report.proposed_schema is not None
        assert "HeritageSite" in report.proposed_schema["node_types"]
        assert "HeritageSite:석굴암" in kg.graph          # 제안 스키마로 추출됨
        node = kg.graph.nodes["HeritageSite:석굴암"]
        assert node["lat"] == 35.795                      # 위치 attrs 보존

    def test_auto_mode_falls_back_on_garbage_proposal(self, kg):
        def garbage_then_nothing(prompt):
            if "스키마를 제안" in prompt:
                return "이건 JSON이 아님"
            return json.dumps({"entities": [], "relations": []})

        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, llm_fn=garbage_then_nothing,
                                  kg=kg, auto_save=False)
        report = run(builder.build_from_text("아무 텍스트", source="s.txt"))
        # 제안 실패 → document 프리셋 폴백, 빌드는 계속
        assert report.proposed_schema is None
        assert builder.schema is not None


class TestOwlExport:
    def test_turtle_contains_classes_properties_individuals(self, kg):
        run(kg.add_concept("HeritageSite:석굴암", "HeritageSite",
                           {"name": "석굴암", "source": "s.txt"}))
        run(kg.add_concept("Region:경주", "Region", {"name": "경주"}))
        run(kg.add_relationship("HeritageSite:석굴암", "Region:경주", "locatedIn"))

        from ontology.builder.export import to_turtle
        ttl = to_turtle(kg.graph, namespace="heritage")
        assert "@prefix owl:" in ttl
        assert "owl:Class" in ttl and "owl:ObjectProperty" in ttl
        assert 'rdfs:label "석굴암"' in ttl
        assert "locatedIn" in ttl

    def test_turtle_empty_graph(self, kg):
        from ontology.builder.export import to_turtle
        ttl = to_turtle(kg.graph, namespace="empty")
        assert "@prefix" in ttl  # 빈 그래프도 유효한 문서


# ─── 10. 의미론적 추출 (definition · aliases) ───────────────────────

class TestSemanticExtraction:
    def test_prompt_requests_definition_and_aliases(self, schema):
        from ontology.builder.extractor import build_extraction_prompt
        prompt = build_extraction_prompt(schema, "아무 텍스트")
        assert "definition" in prompt and "aliases" in prompt
        assert "동의어" in prompt          # 의미론 규칙이 한국어로 명시됨

    def test_prompt_allows_common_synonym_aliases(self, schema):
        # 가1: 별칭에 한해 텍스트에 없어도 널리 통용되는 통칭 허용(리콜↑).
        # 정의·사실은 여전히 근거 필요(rule 4 불변).
        from ontology.builder.extractor import build_extraction_prompt
        prompt = build_extraction_prompt(schema, "아무 텍스트")
        assert "통칭" in prompt              # 통용 통칭 허용 지시

    def test_compose_node_text_includes_aliases_and_definition(self):
        from ontology.core.semantic_index import compose_node_text
        text = compose_node_text("HeritageSite:숭례문", {
            "type": "HeritageSite", "name": "숭례문",
            "definition": "조선 한양도성의 남대문",
            "aliases": ["남대문", "국보 1호"],
        })
        assert "남대문" in text            # 동의어가 임베딩 대상에 포함
        assert "조선 한양도성의 남대문" in text

    def test_aliases_survive_pipeline_and_boost_search(self, kg):
        def alias_llm(prompt):
            if "숭례문" in prompt:
                return json.dumps({
                    "entities": [{"name": "숭례문", "type": "Coverage",
                                  "attrs": {"definition": "서울 도성의 남쪽 정문",
                                            "aliases": ["남대문"]}}],
                    "relations": [],
                }, ensure_ascii=False)
            return json.dumps({"entities": [], "relations": []})

        import zlib
        import numpy as np

        def token_embed(texts):
            vectors = np.zeros((len(texts), 64), dtype=np.float32)
            for i, t in enumerate(texts):
                for tok in str(t).lower().split():
                    vectors[i, zlib.crc32(tok.encode()) % 64] += 1.0
            return vectors

        from ontology.builder import OntologyBuilder, BuilderSchema
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        # 빌더의 실사용 조건: 비-default 네임스페이스 (전 타입 인덱싱)
        kg = KnowledgeGraphEngine(fast_mode=True, namespace="alias_test")
        schema = BuilderSchema(node_types=["Coverage"], predicates={})
        builder = OntologyBuilder(schema=schema, llm_fn=alias_llm, kg=kg,
                                  auto_save=False)
        run(builder.build_from_text("숭례문은 남대문이라고도 불린다", source="s.txt"))

        node = kg.graph.nodes["Coverage:숭례문"]
        assert node["aliases"] == ["남대문"]           # attrs로 영속화

        kg.init_semantic_index(embed_fn=token_embed)
        results = kg.semantic_search("남대문", top_k=1)  # 동의어로 검색됨
        assert results and results[0]["node_id"] == "Coverage:숭례문"


# ─── 11. 정형 레코드 결정적 인제스트 (LLM 무사용) ───────────────────

HERITAGE_MAPPING = {
    "node_type": "HeritageSite",
    "name_field": "이름",
    "relations": [
        {"predicate": "hasDesignation", "target_type": "Designation", "field": "지정"},
        {"predicate": "locatedInRegion", "target_type": "Region", "field": "지역"},
    ],
}


class TestRecordsIngest:
    def test_records_create_nodes_relations_provenance(self, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, kg=kg, auto_save=False)
        records = [
            {"이름": "숭례문", "지정": "국보", "지역": "서울", "lat": 37.56, "lng": 126.97},
            {"이름": "첨성대", "지정": "국보", "지역": "경주"},
        ]
        report = run(builder.build_from_records(records, HERITAGE_MAPPING,
                                                source="wikidata"))
        # 유산 2 + 국보 1 + 지역 2 = 5 노드, 관계 4
        assert report.entities_added == 5
        assert report.relations_added == 4

        node = kg.graph.nodes["HeritageSite:숭례문"]
        assert node["type"] == "HeritageSite"
        assert node["source"] == "wikidata"          # provenance
        assert node["lat"] == 37.56                  # 매핑 외 필드는 attrs
        edges = kg.graph.get_edge_data("HeritageSite:숭례문", "Designation:국보")
        assert any(e.get("predicate") == "hasDesignation" for e in edges.values())

    def test_nameless_records_skipped(self, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, kg=kg, auto_save=False)
        records = [{"이름": "숭례문", "지정": "국보"}, {"지정": "보물"}, {"이름": "  "}]
        report = run(builder.build_from_records(records, HERITAGE_MAPPING))
        assert "HeritageSite:숭례문" in kg.graph
        assert report.entities_added == 2            # 숭례문 + 국보만

    def test_records_rerun_no_duplicates(self, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, kg=kg, auto_save=False)
        records = [{"이름": "숭례문", "지정": "국보"}]
        run(builder.build_from_records(records, HERITAGE_MAPPING))
        edges_before = kg.graph.number_of_edges()
        report = run(builder.build_from_records(records, HERITAGE_MAPPING))
        assert kg.graph.number_of_edges() == edges_before
        assert report.entities_added == 0            # 전부 기존

    def test_type_field_uses_record_value_as_node_type(self, kg):
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, kg=kg, auto_save=False)
        mapping = {"node_type": "HeritageSite", "name_field": "이름",
                   "type_field": "유형"}
        records = [
            {"이름": "불국사 삼층석탑", "유형": "삼층석탑"},
            {"이름": "숭례문", "유형": "성문"},
            {"이름": "무명유산"},                      # 유형 없음 → 폴백
        ]
        run(builder.build_from_records(records, mapping))
        assert kg.graph.nodes["삼층석탑:불국사 삼층석탑"]["type"] == "삼층석탑"
        assert kg.graph.nodes["성문:숭례문"]["type"] == "성문"
        assert kg.graph.nodes["HeritageSite:무명유산"]["type"] == "HeritageSite"

    def test_is_a_hierarchy_via_records_and_inference(self, kg):
        # 클래스 계층을 records 모드로 인제스트 → 전이 추론 동작
        from ontology.builder import OntologyBuilder
        builder = OntologyBuilder(schema=None, kg=kg, auto_save=False)
        mapping = {"node_type": "HeritageClass", "name_field": "이름",
                   "relations": [{"predicate": "is_a",
                                  "target_type": "HeritageClass", "field": "상위"}]}
        records = [
            {"이름": "삼층석탑", "상위": "석탑"},
            {"이름": "석탑", "상위": "탑"},
        ]
        run(builder.build_from_records(records, mapping))
        ancestors = kg.get_ancestors("HeritageClass:삼층석탑")
        assert ancestors == ["HeritageClass:석탑", "HeritageClass:탑"]  # 전이 폐포


# ─── 12. 토픽 회수 추출(aicoach식 옵션) ──────────────────────────────

class TestTopicExtraction:
    """extraction_mode='topic' — 전 청크 저장·임베딩 후 토픽별 상위 청크만
    LLM 추출. 임베더 없으면 전체 추출로 degrade."""

    def _token_embed(self):
        import zlib
        import numpy as np

        def embed(texts):
            v = np.zeros((len(texts), 64), dtype="float32")
            for i, t in enumerate(texts):
                for tok in str(t).lower().replace("\n", " ").split():
                    v[i, zlib.crc32(tok.encode()) % 64] += 1.0
            return v
        return embed

    def test_topic_mode_indexes_all_but_extracts_only_retrieved(self, schema, monkeypatch):
        from ontology.core import semantic_index
        from ontology.core.chunk_index import reset_chunk_indices
        from ontology.builder import OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        monkeypatch.setattr(semantic_index, "_load_default_embed_fn",
                            lambda: self._token_embed())
        reset_chunk_indices()
        kg = KnowledgeGraphEngine(fast_mode=True, namespace="topic_sel")
        kg.clear()
        builder = OntologyBuilder(schema=schema, llm_fn=fake_llm, kg=kg,
                                  auto_save=False, segment_mode="heading",
                                  extraction_mode="topic", topics=["청약철회"],
                                  topic_top_k=1)
        builder.chunk_store.clear()
        report = run(builder.build_from_text(SAMPLE_MD, source="s.txt"))

        # 모든 청크가 저장된다(추출 여부와 무관) — 벡터DB/KB 완전성
        assert len(builder.chunk_store.all()) >= 2
        # 청약철회 토픽으로 그 조항만 추출됨 → 암진단비(Coverage)는 미추출
        assert "Clause:청약철회" in kg.graph
        assert "Coverage:암진단비" not in kg.graph
        assert report.chunks_processed >= 1

    def test_topic_mode_falls_back_to_exhaustive_when_retrieval_empty(self, schema, monkeypatch):
        # 임베더 없음 → search 는 substring 폴백. 어떤 토픽도 매칭 못하면
        # 회수 0 → 전체 추출 폴백(빈 온톨로지 방지).
        from ontology.core import semantic_index
        from ontology.core.chunk_index import reset_chunk_indices
        from ontology.builder import OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        monkeypatch.setattr(semantic_index, "_load_default_embed_fn", lambda: None)
        reset_chunk_indices()
        kg = KnowledgeGraphEngine(fast_mode=True, namespace="topic_fb")
        kg.clear()
        builder = OntologyBuilder(schema=schema, llm_fn=fake_llm, kg=kg,
                                  auto_save=False, segment_mode="heading",
                                  extraction_mode="topic", topics=["존재하지않는토픽ZZZ"])
        builder.chunk_store.clear()
        run(builder.build_from_text(SAMPLE_MD, source="s.txt"))
        # 회수 0 → 전체 추출 폴백 → 두 조항 모두 추출
        assert "Clause:청약철회" in kg.graph
        assert "Coverage:암진단비" in kg.graph

    def test_derive_topics_parses_llm_json_array(self, schema):
        from ontology.builder import OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        def topic_llm(prompt):
            return '설명 ["청약철회", "보장내용", "면책"] 끝'
        kg = KnowledgeGraphEngine(fast_mode=True, namespace="topic_derive")
        builder = OntologyBuilder(schema=schema, llm_fn=topic_llm, kg=kg,
                                  auto_save=False)

        class C:
            text = "제1조 청약철회 관련 내용"
        topics = run(builder._derive_topics([C()]))
        assert topics == ["청약철회", "보장내용", "면책"]


# ─── 13. PDF 리더 정규화 (Phase 1-2) ────────────────────────────────

class TestPdfCleaning:
    """clean_pdf_pages — 러닝헤더·쪽번호·우측정렬 꼬리말 제거 + 공백 정리.
    순수 함수라 실제 PDF 없이 검증한다. 본문은 보존한다."""

    def test_repeating_header_and_pagenumbers_removed(self):
        from ontology.builder.readers import clean_pdf_pages
        pages = [
            "무배당 약관\n12 / 313\n제1조(목적) 본문 하나입니다.",
            "무배당 약관\n13 / 313\n제2조(정의) 본문 둘입니다.",
            "무배당 약관\n14 / 313\n제3조(효력) 본문 셋입니다.",
        ]
        out = clean_pdf_pages(pages)
        assert "무배당 약관" not in out          # 반복 헤더 제거
        assert "/ 313" not in out                # 쪽번호 제거
        assert "제1조(목적) 본문 하나입니다." in out   # 본문 보존

    def test_leading_footer_prefix_stripped_body_kept(self):
        from ontology.builder.readers import clean_pdf_pages
        pages = [
            "5    ACME CORP    실제 본문 내용 가.",
            "6    ACME CORP    실제 본문 내용 나.",
            "7    ACME CORP    실제 본문 내용 다.",
        ]
        out = clean_pdf_pages(pages)
        assert "ACME CORP" not in out             # 반복 꼬리말 접두 제거
        assert "실제 본문 내용 가." in out          # 본문 보존

    def test_non_repeating_content_survives(self):
        from ontology.builder.readers import clean_pdf_pages
        pages = ["고유한 첫 페이지 내용.", "완전히 다른 둘째 페이지."]
        out = clean_pdf_pages(pages)
        assert "고유한 첫 페이지 내용." in out
        assert "완전히 다른 둘째 페이지." in out


# ─── 14. 소스별 멱등 재적재 (Phase 1-3) ─────────────────────────────

class TestReingest:
    def test_reingest_changed_content_replaces_source_chunks(self, schema, kg):
        from ontology.builder import OntologyBuilder
        from ontology.core.chunk_store import reset_chunk_stores
        reset_chunk_stores()
        b = OntologyBuilder(schema=schema, llm_fn=fake_llm, kg=kg, auto_save=False,
                            segment_mode="window", chunk_size=200)
        b.chunk_store.clear()
        run(b.build_from_text("청약철회 관련 조항입니다. 계약자는 철회할 수 있습니다.",
                              source="doc.txt"))
        assert any("청약철회" in c.text for c in b.chunk_store.all())
        # 같은 소스, 내용 교체 재적재
        run(b.build_from_text("암진단비 지급 관련 내용입니다. 최초 1회 지급합니다.",
                              source="doc.txt"))
        texts = [c.text for c in b.chunk_store.all() if c.source == "doc.txt"]
        assert any("암진단비" in t for t in texts)
        assert not any("청약철회" in t for t in texts)   # 옛 청크 교체됨(누적 아님)


# ─── 15. 스키마 고정 (재인제스트 재현성, 다-1) ──────────────────────

class TestSchemaFromGraph:
    def test_from_graph_derives_types_and_predicates(self):
        import networkx as nx
        from ontology.builder import BuilderSchema
        g = nx.MultiDiGraph()
        g.add_node("Disease:암", type="Disease", name="암")
        g.add_node("Term:보장", type="Term", name="보장")
        g.add_edge("Term:보장", "Disease:암", predicate="covers")
        s = BuilderSchema.from_graph(g)
        assert set(s.node_types) == {"Disease", "Term"}
        assert s.predicates["covers"] == ("Term", "Disease")

    def test_from_graph_empty_returns_none(self):
        import networkx as nx
        from ontology.builder import BuilderSchema
        assert BuilderSchema.from_graph(nx.MultiDiGraph()) is None
