"""
감식기(Sniffer) — 업로드 파일의 종(種) 판별 + 매핑 제안.

실데이터 조사(2026-07-17)에서 나온 결론: 업로드 파일은 4종이고 종마다 올바른
처리 경로가 다르다. 지금은 전부 LLM 추출로 직행해서, wikidata 1,999 레코드
같은 정형 데이터가 ~2,000 LLM 콜을 낭비하며 환각 위험까지 진다 (올바른 경로는
매핑 제안 1콜 + 결정적 인제스트).

    records        정형 레코드 (wikidata JSON, CSV)   → 매핑 제안 + build_from_records
    articled       조문형 문서 (약관, 법령, 규정)      → heading 분할 + LLM 추출
    prose          자유 산문                          → window 분할 + LLM 추출
    seed_ontology  이미 온톨로지인 파일 (JSON-LD)      → 멱등 upsert (추출 아님)

역할 분담 (고정 계약):
- 종 판별·카디널리티 통계 = **결정적, LLM 0콜**. JSON 구조·heading 밀도·
  distinct/total 은 계산이지 판단이 아니다.
- 노드 vs 속성 = **통계가 제안** — 공유값(distinct 낮음)은 노드 후보,
  리터럴(숫자·고유값)은 속성. "찾을 것인가 읽을 것인가"를 데이터로 측정.
- LLM(gemini-3.5-flash) = 이름 짓기와 의미 해석만: 술어명, 파일명 메타데이터
  (분야가 다양하므로 문서종류를 하드코딩할 수 없다 — 프로젝트 절대 원칙).
- LLM 출력은 전부 불신 — 실제 필드와 대조 검증 (aicoach clean_extraction 원칙).
"""

import asyncio
import json

import pytest

from ontology.builder.sniffer import (
    DatasetAnalyzer,
    FileProfile,
    analyze_json_structure,
    detect_text_species,
    field_stats,
    parse_mapping_proposal,
    suggest_roles,
)


def run(coro):
    return asyncio.run(coro)


# ─── 실데이터를 본뜬 fixture ─────────────────────────────────────────

WIKIDATA_STYLE = {
    "records": [
        {"이름": "합천 해인사 대장경판", "유형": "불경", "지정": "국보",
         "지역": "경상남도", "lat": 35.8, "lng": 128.1},
        {"이름": "숭례문", "유형": "성문", "지정": "국보",
         "지역": "서울", "lat": 37.6, "lng": 127.0},
        {"이름": "불국사", "유형": "사찰", "지정": "사적",
         "지역": "경상북도", "lat": 35.8, "lng": 129.3},
    ],
    "hierarchy": [{"이름": "성문", "상위": "구조물"}],
}

JSONLD_SEED = {
    "@context": {"ko": "https://koract.ai/ontology/v0.1#"},
    "@graph": [
        {"@id": "ko:intent/pay", "@type": "ko:IntentSlot", "prefLabel": "결제",
         "ko:examples": ["결제하기", "주문하기"]},
        {"@id": "ko:entity/button", "@type": "ko:EntityClass", "prefLabel": "버튼"},
    ],
}

ARTICLED_TEXT = """제1조 (목적)
이 약관은 보험계약의 목적을 정한다.

제2조 (정의)
용어의 뜻은 다음과 같다.

제3조 (청약철회)
계약자는 15일 이내에 청약을 철회할 수 있다.
"""

PROSE_TEXT = ("해인사는 신라 시대에 창건된 사찰로, 팔만대장경을 보관하고 있는 "
              "장경판전으로 유명하다. 가야산 자락에 자리잡고 있으며 한국 불교의 "
              "중심지 중 하나로 오랜 역사를 이어왔다.")


# ─── 1. JSON 구조 분석 (결정적) ──────────────────────────────────────

class TestJsonStructure:
    def test_wikidata_style_is_records(self):
        species, records, path = analyze_json_structure(WIKIDATA_STYLE)
        assert species == "records"
        assert len(records) == 3
        assert path == "records"  # 어느 키 아래에 레코드가 있었는지

    def test_bare_list_of_dicts_is_records(self):
        species, records, path = analyze_json_structure(WIKIDATA_STYLE["records"])
        assert species == "records"
        assert path == ""

    def test_jsonld_is_seed_ontology(self):
        species, records, path = analyze_json_structure(JSONLD_SEED)
        assert species == "seed_ontology"
        assert len(records) == 2  # @graph 노드들

    def test_preflabel_items_without_at_graph_is_seed(self):
        """@graph 없이도 prefLabel/definition 필드가 지배적이면 시드다."""
        data = [{"prefLabel": "결제", "definition": "금전 거래"},
                {"prefLabel": "인증", "definition": "본인 확인"}]
        species, _, _ = analyze_json_structure(data)
        assert species == "seed_ontology"

    def test_heterogeneous_json_is_prose(self):
        """키가 제각각인 JSON 은 레코드 배열이 아니다 — 텍스트 경로로."""
        data = {"제목": "보고서", "본문": "긴 글", "메타": {"저자": "김"}}
        species, records, _ = analyze_json_structure(data)
        assert species == "prose"
        assert records == []

    def test_nested_records_are_found(self):
        """korean_heritage.json 꼴: {dataset, 설명, items:[...]} — 항목이
        최상위가 아니라 안쪽 키에 있다."""
        data = {"dataset": "컬렉션", "설명": "...",
                "items": [{"명칭": "석굴암", "시대": "신라"},
                          {"명칭": "첨성대", "시대": "신라"}]}
        species, records, path = analyze_json_structure(data)
        assert species == "records"
        assert path == "items"

    def test_low_key_overlap_is_not_records(self):
        data = [{"a": 1, "b": 2}, {"x": 1, "y": 2}, {"p": 1, "q": 2}]
        species, _, _ = analyze_json_structure(data)
        assert species != "records"


# ─── 2. 텍스트 종 판별 (결정적 — segmenter 의 heading 신호 재사용) ───

class TestTextSpecies:
    def test_articled_text(self):
        assert detect_text_species(ARTICLED_TEXT) == "articled"

    def test_prose_text(self):
        assert detect_text_species(PROSE_TEXT) == "prose"

    def test_markdown_headings_count_as_articled(self):
        md = "# 개요\n본문.\n\n# 상세\n본문.\n\n# 결론\n본문."
        assert detect_text_species(md) == "articled"


# ─── 3. 카디널리티 통계 → 노드/속성 제안 (결정적) ────────────────────

class TestFieldStats:
    def test_stats_shape(self):
        stats = field_stats(WIKIDATA_STYLE["records"])
        assert stats["지정"]["distinct"] == 2       # 국보, 사적
        assert stats["지정"]["total"] == 3
        assert stats["lat"]["numeric_ratio"] == 1.0
        assert stats["이름"]["distinct"] == 3

    def test_missing_values_are_counted(self):
        records = [{"a": "x"}, {"a": None}, {"b": "y"}]
        stats = field_stats(records)
        assert stats["a"]["total"] == 3
        assert stats["a"]["present"] == 1  # None 과 부재는 둘 다 결측

    def test_suggest_roles_from_real_shape(self):
        """실측 원리: 공유값(카디널리티 낮음)은 질의 축 → 노드 후보.
        숫자·고유값은 리터럴 → 속성. 전부 고유한 문자열 → 정체성(name)."""
        # 실데이터 비율을 본뜬 30 레코드 (지정 3종, 지역 5종, lat 전부 고유)
        records = [{"이름": f"유산{i}", "지정": ["국보", "보물", "사적"][i % 3],
                    "지역": f"지역{i % 5}", "lat": 30.0 + i * 0.7}
                   for i in range(30)]
        roles = suggest_roles(field_stats(records))
        assert roles["이름"] == "identity"
        assert roles["지정"] == "node_candidate"
        assert roles["지역"] == "node_candidate"
        assert roles["lat"] == "attribute"

    def test_roles_carry_no_domain_hardcoding(self):
        """판정은 통계로만 — 필드명이 영어든 다른 도메인이든 같은 규칙."""
        records = [{"sku": f"P{i:04}", "brand": ["A", "B"][i % 2],
                    "price": 1000 + i} for i in range(20)]
        roles = suggest_roles(field_stats(records))
        assert roles["sku"] == "identity"
        assert roles["brand"] == "node_candidate"
        assert roles["price"] == "attribute"


# ─── 4. 매핑 제안 검증 (LLM 출력은 불신) ─────────────────────────────

class TestMappingValidation:
    FIELDS = ["이름", "유형", "지정", "지역", "lat", "lng"]

    def test_valid_proposal_passes(self):
        raw = {"node_type": "HeritageSite", "name_field": "이름",
               "type_field": "유형",
               "relations": [
                   {"field": "지정", "predicate": "hasDesignation",
                    "target_type": "Designation"},
                   {"field": "지역", "predicate": "locatedInRegion",
                    "target_type": "Region"}]}
        mapping = parse_mapping_proposal(json.dumps(raw), self.FIELDS)
        assert mapping["name_field"] == "이름"
        assert len(mapping["relations"]) == 2

    def test_hallucinated_field_is_dropped(self):
        """LLM 이 없는 필드를 지어내면 그 관계만 버린다 — 전체를 죽이지 않는다."""
        raw = {"node_type": "T", "name_field": "이름",
               "relations": [
                   {"field": "없는필드", "predicate": "p", "target_type": "X"},
                   {"field": "지정", "predicate": "hasDesignation",
                    "target_type": "Designation"}]}
        mapping = parse_mapping_proposal(json.dumps(raw), self.FIELDS)
        assert len(mapping["relations"]) == 1
        assert mapping["relations"][0]["field"] == "지정"

    def test_hallucinated_name_field_fails(self):
        """name_field 가 실제 필드가 아니면 매핑 전체가 무효다 — 정체성 없이
        인제스트하면 안 된다."""
        raw = {"node_type": "T", "name_field": "존재안함", "relations": []}
        assert parse_mapping_proposal(json.dumps(raw), self.FIELDS) is None

    def test_garbage_returns_none(self):
        assert parse_mapping_proposal("not json at all", self.FIELDS) is None

    def test_code_fenced_json_is_accepted(self):
        raw = ('```json\n{"node_type": "T", "name_field": "이름", '
               '"relations": []}\n```')
        assert parse_mapping_proposal(raw, self.FIELDS) is not None


# ─── 5. DatasetAnalyzer 통합 (가짜 LLM 주입) ─────────────────────────

def fake_llm(prompt: str) -> str:
    """결정적 가짜 — 프롬프트 종류로 분기."""
    if "매핑" in prompt or "mapping" in prompt.lower():
        return json.dumps({
            "node_type": "HeritageSite", "name_field": "이름",
            "type_field": "유형",
            "relations": [{"field": "지정", "predicate": "hasDesignation",
                           "target_type": "Designation"}],
        }, ensure_ascii=False)
    if "파일명" in prompt:
        return json.dumps({"files": [
            {"filename": "통합약관_치매간병보험_20260101.pdf",
             "doc_kind": "약관", "entity": "치매간병보험",
             "version": "2026-01-01", "trust": "authoritative"}]},
            ensure_ascii=False)
    return "{}"


# 실데이터 비율을 본뜬 크기 — wikidata 실물은 1,999 레코드에 지정 7종이다.
# 3레코드짜리 미니 fixture 로는 카디널리티 규칙(distinct/present ≤ 0.5)이
# 통계적으로 성립하지 않아 아무것도 검증하지 못한다.
REALISTIC_RECORDS = {
    "records": [
        {"이름": f"유산{i:02}", "유형": ["사찰", "석탑", "성문"][i % 3],
         "지정": ["국보", "보물", "사적"][i % 3],
         "지역": f"지역{i % 4}", "lat": 33.0 + i * 0.31, "lng": 126.0 + i * 0.17}
        for i in range(12)
    ],
    "hierarchy": [{"이름": "성문", "상위": "구조물"}],
}


@pytest.fixture
def dataset_dir(tmp_path):
    (tmp_path / "wikidata.json").write_text(
        json.dumps(REALISTIC_RECORDS, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "약관.md").write_text(ARTICLED_TEXT, encoding="utf-8")
    (tmp_path / "소개.txt").write_text(PROSE_TEXT, encoding="utf-8")
    (tmp_path / "seed.jsonld").write_text(
        json.dumps(JSONLD_SEED, ensure_ascii=False), encoding="utf-8")
    return tmp_path


class TestDatasetAnalyzer:
    def test_analyzes_every_file_with_its_species(self, dataset_dir):
        analyzer = DatasetAnalyzer(llm_fn=fake_llm)
        report = run(analyzer.analyze(dataset_dir))
        by_name = {p.filename: p for p in report.files}
        assert by_name["wikidata.json"].species == "records"
        assert by_name["약관.md"].species == "articled"
        assert by_name["소개.txt"].species == "prose"
        assert by_name["seed.jsonld"].species == "seed_ontology"

    def test_records_file_gets_mapping_proposal(self, dataset_dir):
        analyzer = DatasetAnalyzer(llm_fn=fake_llm)
        report = run(analyzer.analyze(dataset_dir))
        wikidata = next(p for p in report.files if p.species == "records")
        assert wikidata.mapping_proposal["name_field"] == "이름"
        assert wikidata.field_roles["지정"] == "node_candidate"

    def test_cost_estimate_reflects_routing(self, dataset_dir):
        """종별 비용: records=매핑 1콜, seed=0콜, 텍스트=청크 수.
        이 견적이 확인 게이트에 표시된다 — 2,000콜 낭비를 사전에 보이게."""
        analyzer = DatasetAnalyzer(llm_fn=fake_llm)
        report = run(analyzer.analyze(dataset_dir))
        by_name = {p.filename: p for p in report.files}
        assert by_name["wikidata.json"].estimated_llm_calls == 1
        assert by_name["seed.jsonld"].estimated_llm_calls == 0
        assert by_name["약관.md"].estimated_llm_calls >= 1  # 청크 수
        assert report.total_estimated_llm_calls == sum(
            p.estimated_llm_calls for p in report.files)

    def test_records_species_avoids_per_record_llm(self, dataset_dir):
        """핵심 계약: 정형 3레코드가 3콜이 아니라 1콜(매핑 제안)이어야 한다.
        1,999 레코드면 1,999콜 vs 1콜 차이다."""
        calls = {"n": 0}

        def counting(prompt):
            calls["n"] += 1
            return fake_llm(prompt)

        analyzer = DatasetAnalyzer(llm_fn=counting)
        run(analyzer.analyze(dataset_dir))
        # 매핑 1 + 파일명 배치 1 = 2 (레코드 수·청크 수와 무관)
        assert calls["n"] <= 2

    def test_llm_failure_degrades_to_stats_only(self, dataset_dir):
        """LLM 이 죽어도 분석은 나온다 — 종·통계·역할 제안은 결정적이므로.
        매핑 제안만 비고 이유가 남는다."""
        def broken(prompt):
            raise RuntimeError("LLM down")

        analyzer = DatasetAnalyzer(llm_fn=broken)
        report = run(analyzer.analyze(dataset_dir))
        wikidata = next(p for p in report.files if p.species == "records")
        assert wikidata.mapping_proposal is None
        assert wikidata.field_roles["지정"] == "node_candidate"  # 통계는 산다

    def test_hierarchy_sidecar_is_detected(self, dataset_dir):
        """wikidata 꼴의 {records, hierarchy} — is_a 간선 재료가 함께 온 것을
        알아본다 (놓치면 계층 롤업 추론이 통째로 빠진다)."""
        analyzer = DatasetAnalyzer(llm_fn=fake_llm)
        report = run(analyzer.analyze(dataset_dir))
        wikidata = next(p for p in report.files if p.species == "records")
        assert wikidata.hierarchy_count == 1
