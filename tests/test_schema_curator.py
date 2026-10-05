"""
스키마 큐레이터 (④) — auto 스키마 제안을 기존 어휘에 정합시킨다.

문제는 실측돼 있다: 일관성 린터가 heritage_kr 에서 name_type_conflict 10건을
찾았다 ("국보"가 Designation/HeritageClass 두 타입에 중복). 원인 경로는
auto 스키마다 — 빌드마다 LLM 이 2000자 샘플을 보고 타입을 새로 제안하므로,
같은 네임스페이스에 데이터셋을 쌓을수록 같은 개념이 다른 타입명으로 쪼개진다
(Clause vs Provision). 쪼개지면 롤업 추론·축 4 확장·데이터셋 품질이 함께
나빠진다. 린터는 사후 탐지, 큐레이터는 **사전 예방**이다.

고정하는 계약:
1. **결정적 우선**: 대소문자·공백만 다른 타입은 LLM 없이 병합한다.
2. LLM 은 남은 것(의미 동치 판단)만 — 1콜. 매핑 타깃이 기존 어휘에 실존하지
   않으면 그 매핑만 버린다 (출력 불신).
3. **auto 모드에서만** 동작한다 — 사용자가 명시한 스키마(preset/custom)를
   고치는 것은 확인 게이트 철학 위반이다.
4. 무엇을 어디로 매핑했는지 리포트에 남는다 (조용한 개명 금지).
5. 큐레이션 실패는 빌드를 막지 않는다 — 제안 원안으로 계속한다.
"""

import asyncio
import json

import pytest

from ontology.builder.models import BuilderSchema
from ontology.core.schema_curator import (
    SchemaCurator,
    apply_mapping,
    existing_vocabulary,
    match_deterministic,
    parse_reconcile_mapping,
)


def run(coro):
    return asyncio.run(coro)


def make_graph(nodes, edges=()):
    import networkx as nx
    graph = nx.MultiDiGraph()
    for node_id, node_type in nodes:
        graph.add_node(node_id, type=node_type, name=node_id.split(":")[-1])
    for source, target, predicate in edges:
        graph.add_edge(source, target, predicate=predicate)
    return graph


# ─── 1. 기존 어휘 추출 (결정적) ──────────────────────────────────────

class TestExistingVocabulary:
    def test_collects_types_and_predicates(self):
        graph = make_graph(
            [("Clause:청약철회", "Clause"), ("Regulation:금소법", "Regulation")],
            [("Clause:청약철회", "Regulation:금소법", "citesRegulation")])
        vocab = existing_vocabulary(graph)
        assert set(vocab["types"]) == {"Clause", "Regulation"}
        assert vocab["predicates"] == ["citesRegulation"]

    def test_empty_graph(self):
        vocab = existing_vocabulary(make_graph([]))
        assert vocab["types"] == [] and vocab["predicates"] == []


# ─── 2. 결정적 매칭 (LLM 0콜) ────────────────────────────────────────

class TestDeterministicMatch:
    def test_exact_match_is_identity(self):
        mapping, unmatched = match_deterministic(["Clause"], ["Clause"])
        assert mapping == {} and unmatched == []

    def test_case_difference_is_merged_without_llm(self):
        mapping, unmatched = match_deterministic(["clause"], ["Clause"])
        assert mapping == {"clause": "Clause"}
        assert unmatched == []

    def test_whitespace_difference_is_merged(self):
        mapping, unmatched = match_deterministic(["Heritage Class"],
                                                 ["HeritageClass"])
        assert mapping == {"Heritage Class": "HeritageClass"}

    def test_genuinely_new_type_stays_unmatched(self):
        mapping, unmatched = match_deterministic(["Provision"], ["Clause"])
        assert mapping == {} and unmatched == ["Provision"]

    def test_no_existing_vocab_means_nothing_to_match(self):
        mapping, unmatched = match_deterministic(["A", "B"], [])
        assert mapping == {} and unmatched == ["A", "B"]


# ─── 3. LLM 매핑 검증 (출력 불신) ────────────────────────────────────

class TestParseReconcileMapping:
    EXISTING = ["Clause", "Regulation"]

    def test_valid_mapping_passes(self):
        raw = json.dumps({"mapping": {"Provision": "Clause", "Rule": "Regulation"}})
        assert parse_reconcile_mapping(raw, ["Provision", "Rule"],
                                       self.EXISTING) == \
            {"Provision": "Clause", "Rule": "Regulation"}

    def test_hallucinated_target_is_dropped(self):
        """타깃이 기존 어휘에 없으면 그 매핑만 버린다 — 지어낸 타입으로의
        개명은 병합이 아니라 새 파편이다."""
        raw = json.dumps({"mapping": {"Provision": "없는타입", "Rule": "Regulation"}})
        assert parse_reconcile_mapping(raw, ["Provision", "Rule"],
                                       self.EXISTING) == {"Rule": "Regulation"}

    def test_unknown_source_key_is_dropped(self):
        """묻지 않은 타입을 매핑하면 버린다 — 요청 밖 개명 금지."""
        raw = json.dumps({"mapping": {"엉뚱한소스": "Clause"}})
        assert parse_reconcile_mapping(raw, ["Provision"], self.EXISTING) == {}

    def test_null_means_keep_as_new(self):
        raw = json.dumps({"mapping": {"Provision": None}})
        assert parse_reconcile_mapping(raw, ["Provision"], self.EXISTING) == {}

    def test_garbage_returns_empty(self):
        assert parse_reconcile_mapping("json 아님", ["A"], self.EXISTING) == {}


# ─── 4. 매핑 적용 ────────────────────────────────────────────────────

class TestApplyMapping:
    def test_types_and_predicate_ranges_are_renamed(self):
        schema = BuilderSchema(
            node_types=["Provision", "Document"],
            predicates={"hasProvision": ("Document", "Provision")})
        merged = apply_mapping(schema, {"Provision": "Clause"}, {})
        assert "Clause" in merged.node_types
        assert "Provision" not in merged.node_types
        assert merged.predicates["hasProvision"] == ("Document", "Clause")

    def test_predicate_mapping_renames_predicate(self):
        schema = BuilderSchema(node_types=["A", "B"],
                               predicates={"locatedIn": ("A", "B")})
        merged = apply_mapping(schema, {}, {"locatedIn": "locatedInRegion"})
        assert "locatedInRegion" in merged.predicates
        assert "locatedIn" not in merged.predicates

    def test_no_mapping_is_identity(self):
        schema = BuilderSchema(node_types=["A"], predicates={})
        merged = apply_mapping(schema, {}, {})
        assert merged.node_types == ["A"]

    def test_merged_duplicate_types_are_deduped(self):
        """제안에 Clause 와 Provision 이 둘 다 있고 Provision→Clause 로
        매핑되면 결과에 Clause 가 두 번 있으면 안 된다."""
        schema = BuilderSchema(node_types=["Clause", "Provision"], predicates={})
        merged = apply_mapping(schema, {"Provision": "Clause"}, {})
        assert merged.node_types.count("Clause") == 1


# ─── 5. 빌더 통합 (auto 모드에서만) ──────────────────────────────────

def schema_proposing_llm(prompt: str) -> str:
    """스키마 제안 요청 → 기존 어휘와 어긋나는 타입명 제안 (대소문자 상이 +
    의미 동치). 추출 요청 → 빈 결과.

    분기 표지는 프롬프트의 **역할 문장**("온톨로지 설계자" vs "스키마 관리자")
    — 상호 배타적이라 안전하다. 내용 낱말('스키마', '같은 개념')로 분기하면
    안 된다: 어휘 힌트 등 다른 블록에도 등장해 분기가 오염된다 (실제로 두 번
    당했다).
    """
    if "스키마 관리자" in prompt:
        return json.dumps({"mapping": {"Provision": "Clause"}}, ensure_ascii=False)
    if "온톨로지 설계자" in prompt:
        return json.dumps({"node_types": ["clause", "Provision"],
                           "predicates": {"cites": ["clause", "Provision"]}},
                          ensure_ascii=False)
    return json.dumps({"entities": [], "relations": []}, ensure_ascii=False)


@pytest.fixture
def kg_with_vocab():
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
    kg = KnowledgeGraphEngine(fast_mode=True, namespace="curator_test")

    async def seed():
        await kg.add_concept("Clause:청약철회", "Clause", {"name": "청약철회"})
    run(seed())
    return kg


class TestBuilderIntegration:
    def test_auto_schema_is_reconciled_against_graph(self, kg_with_vocab):
        """제안 'clause'(대소문자)·'Provision'(의미 동치)이 기존 'Clause' 로
        수렴한다 — 타입 파편화가 생기기 전에 막는다."""
        from ontology.builder import OntologyBuilder

        builder = OntologyBuilder(schema=None, kg=kg_with_vocab,
                                  llm_fn=schema_proposing_llm,
                                  store_chunks=False, auto_save=False)
        report = run(builder.build_from_text("아무 본문.", source="d.md"))
        assert "Clause" in builder.schema.node_types
        assert "clause" not in builder.schema.node_types
        assert "Provision" not in builder.schema.node_types
        # 무엇을 어디로 병합했는지 리포트에 남는다 (조용한 개명 금지)
        assert report.schema_mappings["types"]["clause"] == "Clause"
        assert report.schema_mappings["types"]["Provision"] == "Clause"

    def test_empty_graph_skips_curation(self):
        """기존 어휘가 없으면 정합할 대상이 없다 — LLM 매핑 콜도 없어야 한다."""
        from ontology.builder import OntologyBuilder
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        calls = []

        def counting(prompt):
            calls.append(prompt)
            return schema_proposing_llm(prompt)

        kg = KnowledgeGraphEngine(fast_mode=True, namespace="empty_cur")
        builder = OntologyBuilder(schema=None, kg=kg, llm_fn=counting,
                                  store_chunks=False, auto_save=False)
        report = run(builder.build_from_text("본문.", source="d.md"))
        assert report.schema_mappings is None
        # 스키마 제안 1 + 추출 1 = 2콜뿐 (매핑 콜 없음)
        assert len(calls) == 2

    def test_explicit_schema_is_never_curated(self, kg_with_vocab):
        """계약 3 — 사용자가 명시한 스키마를 고치는 것은 게이트 철학 위반."""
        from ontology.builder import OntologyBuilder

        schema = BuilderSchema(node_types=["clause"], predicates={})
        builder = OntologyBuilder(schema=schema, kg=kg_with_vocab,
                                  llm_fn=schema_proposing_llm,
                                  store_chunks=False, auto_save=False)
        run(builder.build_from_text("본문.", source="d.md"))
        assert builder.schema.node_types == ["clause"]  # 그대로

    def test_curation_failure_does_not_kill_the_build(self, kg_with_vocab):
        """계약 5 — 큐레이션은 개선이지 관문이 아니다. 죽으면 원안으로 계속."""
        from ontology.builder import OntologyBuilder

        def flaky(prompt):
            if "스키마 관리자" in prompt:
                raise RuntimeError("reconcile down")
            return schema_proposing_llm(prompt)

        builder = OntologyBuilder(schema=None, kg=kg_with_vocab,
                                  llm_fn=flaky, store_chunks=False,
                                  auto_save=False)
        report = run(builder.build_from_text("본문.", source="d.md"))
        # 결정적 병합(clause→Clause)은 LLM 없이 이미 적용됐다
        assert "Clause" in builder.schema.node_types
        # LLM 판단분(Provision)은 원안 유지 — 빌드는 계속됐다
        assert "Provision" in builder.schema.node_types
        assert report.chunks_processed + report.chunks_failed >= 1

    def test_proposal_prompt_carries_existing_vocab(self, kg_with_vocab):
        """1차 방어선 — 제안 프롬프트에 기존 어휘가 보인다 (원천 정렬).

        라이브 실측 근거: 힌트 없이 제안받자 실제 Gemini 가 기존
        Clause/Regulation 을 두고 LegalProvision/Law 를 새로 지었고,
        보수적 reconcile 은 병합을 거부했다. 사후 병합만으로는 부족하다.
        """
        from ontology.builder import OntologyBuilder

        prompts = []

        def capturing(prompt):
            prompts.append(prompt)
            return schema_proposing_llm(prompt)

        builder = OntologyBuilder(schema=None, kg=kg_with_vocab,
                                  llm_fn=capturing, store_chunks=False,
                                  auto_save=False)
        run(builder.build_from_text("본문.", source="d.md"))
        proposal_prompt = next(p for p in prompts if "온톨로지 설계자" in p)
        assert "Clause" in proposal_prompt          # 기존 타입이 보인다
        assert "재사용" in proposal_prompt           # 재사용 지시가 있다

    def test_first_build_has_no_vocab_hint(self):
        """첫 빌드(빈 그래프)에는 힌트 블록 자체가 없다 — 빈 목록을 보여주며
        재사용을 지시하면 LLM 을 혼란시킬 뿐이다."""
        from ontology.builder.extractor import build_schema_proposal_prompt

        prompt = build_schema_proposal_prompt("샘플", existing_vocab={
            "types": [], "predicates": []})
        assert "기존 어휘" not in prompt
        assert build_schema_proposal_prompt("샘플") == prompt

    def test_curate_schema_opt_out(self, kg_with_vocab):
        from ontology.builder import OntologyBuilder

        builder = OntologyBuilder(schema=None, kg=kg_with_vocab,
                                  llm_fn=schema_proposing_llm,
                                  store_chunks=False, auto_save=False,
                                  curate_schema=False)
        report = run(builder.build_from_text("본문.", source="d.md"))
        assert "clause" in builder.schema.node_types  # 원안 그대로
        assert report.schema_mappings is None
