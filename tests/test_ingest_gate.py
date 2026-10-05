"""
확인 게이트 — analyze 의 제안을 받아 종별로 라우팅해 실행한다.

흐름: 업로드 → analyze(제안) → **사용자 확인/수정** → ingest(이 파일이 검증하는 것).

고정하는 계약:
1. plan 은 데이터다 — analyze 응답의 plan 초안을 사용자가 고쳐서 그대로
   제출한다. 서버는 plan 을 재검증한다 (클라이언트가 보낸 것은 불신).
2. 종별 라우팅: records → 결정적 인제스트(+계층 sidecar → is_a),
   articled/prose → LLM 추출, seed_ontology → 결정적 upsert, skip → 건너뜀.
3. 한 파일의 실패가 나머지를 죽이지 않는다 (빌더의 resilient 원칙과 동일).
4. protected namespace(default) 거부 — 에이전트 라우팅 그래프 보호.
5. 인제스트가 끝나면 **롤업 추론이 실제로 동작**해야 한다 — is_a 재료를
   버리지 않았다는 최종 증거.
"""

import asyncio
import json

import pytest

from ontology.builder.ingestion import (
    import_seed,
    ingest_hierarchy,
    load_records_from_file,
)
from ontology.builder.sniffer import parse_hierarchy_proposal


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def kg():
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
    return KnowledgeGraphEngine(fast_mode=True, namespace="gate_test")


# ─── 1. 레코드 재로딩 (analyze 가 찾은 경로로) ───────────────────────

class TestLoadRecords:
    def test_loads_via_records_path(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"meta": 1, "items": [
            {"이름": "석굴암"}, {"이름": "첨성대"}]}, ensure_ascii=False),
            encoding="utf-8")
        records = load_records_from_file(path, "items")
        assert [r["이름"] for r in records] == ["석굴암", "첨성대"]

    def test_top_level_list(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
        assert len(load_records_from_file(path, "")) == 2

    def test_csv_records(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("이름,지정\n숭례문,국보\n불국사,사적\n", encoding="utf-8")
        records = load_records_from_file(path, "")
        assert records[0] == {"이름": "숭례문", "지정": "국보"}

    def test_wrong_path_raises(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"items": [{"a": 1}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="records"):
            load_records_from_file(path, "없는키")


# ─── 2. 계층 sidecar → is_a (롤업 추론의 재료) ───────────────────────

class TestIngestHierarchy:
    SPEC = {"path": "hierarchy", "child_field": "이름",
            "parent_field": "상위", "node_type": "HeritageClass"}

    def test_creates_class_nodes_and_is_a_edges(self, kg):
        items = [{"이름": "성문", "상위": "구조물"},
                 {"이름": "석탑", "상위": "구조물"}]
        added = run(ingest_hierarchy(kg, items, self.SPEC, source="wikidata"))
        graph = kg.graph
        assert "HeritageClass:성문" in graph
        assert "HeritageClass:구조물" in graph
        assert added["edges"] == 2
        # is_a 전이 폐포가 실제로 동작한다
        assert "HeritageClass:구조물" in kg.get_ancestors("HeritageClass:성문")

    def test_skips_incomplete_items(self, kg):
        items = [{"이름": "성문"}, {"상위": "구조물"},
                 {"이름": "석탑", "상위": "구조물"}]
        added = run(ingest_hierarchy(kg, items, self.SPEC))
        assert added["edges"] == 1

    def test_idempotent(self, kg):
        items = [{"이름": "성문", "상위": "구조물"}]
        run(ingest_hierarchy(kg, items, self.SPEC))
        before = kg.graph.number_of_edges()
        run(ingest_hierarchy(kg, items, self.SPEC))
        assert kg.graph.number_of_edges() == before


# ─── 3. hierarchy 제안 검증 (LLM 출력 불신) ──────────────────────────

class TestHierarchyProposalValidation:
    SIDECARS = {"hierarchy": [{"이름": "성문", "상위": "구조물"}]}

    def test_valid_proposal(self):
        raw = json.dumps({"hierarchy": {
            "path": "hierarchy", "child_field": "이름",
            "parent_field": "상위", "node_type": "HeritageClass"}})
        spec = parse_hierarchy_proposal(raw, self.SIDECARS)
        assert spec["child_field"] == "이름"

    def test_hallucinated_path_is_rejected(self):
        raw = json.dumps({"hierarchy": {
            "path": "없는목록", "child_field": "이름",
            "parent_field": "상위", "node_type": "T"}})
        assert parse_hierarchy_proposal(raw, self.SIDECARS) is None

    def test_hallucinated_field_is_rejected(self):
        raw = json.dumps({"hierarchy": {
            "path": "hierarchy", "child_field": "없는필드",
            "parent_field": "상위", "node_type": "T"}})
        assert parse_hierarchy_proposal(raw, self.SIDECARS) is None

    def test_missing_hierarchy_key_is_none(self):
        assert parse_hierarchy_proposal(
            json.dumps({"node_type": "T"}), self.SIDECARS) is None


# ─── 4. 시드 온톨로지 결정적 import ─────────────────────────────────

class TestImportSeed:
    ITEMS = [
        {"@id": "ko:intent/pay", "@type": "ko:IntentSlot", "prefLabel": "결제",
         "definition": "금전 거래 완료 행동",
         "ko:examples": ["결제하기", "주문하기", "송금"]},
        {"@id": "ko:entity/button", "@type": "ko:EntityClass",
         "prefLabel": "버튼"},
    ]

    def test_nodes_with_name_type_definition(self, kg):
        added = run(import_seed(kg, self.ITEMS, source="seed.jsonld"))
        assert added == 2
        graph = kg.graph
        node = graph.nodes["IntentSlot:결제"]        # @type 접두사 제거
        assert node["type"] == "IntentSlot"
        assert node["definition"] == "금전 거래 완료 행동"
        assert node["source"] == "seed.jsonld"

    def test_examples_become_aliases(self, kg):
        """KorAct 의 ko:examples(표면형)가 우리 aliases 가 된다 — 축 4 의
        확장 재료. 이게 시드 import 의 존재 이유다."""
        run(import_seed(kg, self.ITEMS))
        assert "결제하기" in kg.graph.nodes["IntentSlot:결제"]["aliases"]

    def test_id_tail_fallback_when_no_preflabel(self, kg):
        items = [{"@id": "ko:intent/search", "@type": "ko:IntentSlot"}]
        run(import_seed(kg, items))
        assert "IntentSlot:search" in kg.graph

    def test_nameless_items_are_skipped(self, kg):
        assert run(import_seed(kg, [{"@type": "ko:X"}, {}])) == 0

    def test_idempotent_upsert(self, kg):
        run(import_seed(kg, self.ITEMS))
        before = kg.graph.number_of_nodes()
        run(import_seed(kg, self.ITEMS))
        assert kg.graph.number_of_nodes() == before


# ─── 5. 게이트 전체 흐름 (서비스 레벨) ───────────────────────────────

WIKIDATA_MINI = {
    "records": [
        {"이름": f"유산{i:02}", "유형클래스": ["성문", "석탑"][i % 2],
         "지정": ["국보", "보물", "사적"][i % 3], "lat": 35.0 + i}
        for i in range(6)
    ],
    "hierarchy": [{"이름": "성문", "상위": "구조물"},
                  {"이름": "석탑", "상위": "구조물"}],
}

MAPPING = {"node_type": "HeritageSite", "name_field": "이름",
           "type_field": None,
           "relations": [
               {"field": "유형클래스", "predicate": "classifiedAs",
                "target_type": "HeritageClass"},
               {"field": "지정", "predicate": "hasDesignation",
                "target_type": "Designation"}]}

HIERARCHY = {"path": "hierarchy", "child_field": "이름",
             "parent_field": "상위", "node_type": "HeritageClass"}


def gate_fake_llm(prompt: str) -> str:
    if "지식을 추출" in prompt:  # 텍스트 추출 경로
        return json.dumps({"entities": [
            {"name": "청약철회", "type": "Concept", "attrs": {}}],
            "relations": []}, ensure_ascii=False)
    return "{}"


@pytest.fixture
def gate(tmp_path):
    from ontology.engines.knowledge_graph_clean import (KnowledgeGraphEngine,
                                                        _kg_instances)
    from ontology.server.service import OntologyBuilderService

    ns = "gate_e2e"
    _kg_instances[ns] = KnowledgeGraphEngine(fast_mode=True, namespace=ns)

    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=gate_fake_llm)
    saved = service.save_dataset([
        ("wikidata.json",
         json.dumps(WIKIDATA_MINI, ensure_ascii=False).encode()),
        ("약관.md", "## 제1조 목적\n청약철회 조항.\n\n## 제2조 정의\n용어.\n".encode()),
        ("seed.jsonld", json.dumps({"@graph": [
            {"@id": "ko:intent/pay", "@type": "ko:IntentSlot",
             "prefLabel": "결제", "ko:examples": ["결제하기"]}]},
            ensure_ascii=False).encode()),
    ])
    yield service, saved["dataset_id"], ns, _kg_instances[ns]
    _kg_instances.pop(ns, None)


PLAN = [
    {"filename": "wikidata.json", "route": "records",
     "records_path": "records", "mapping": MAPPING, "hierarchy": HIERARCHY},
    {"filename": "약관.md", "route": "articled"},
    {"filename": "seed.jsonld", "route": "seed_ontology",
     "records_path": "@graph"},
]


class TestGateEndToEnd:
    def test_all_routes_execute(self, gate):
        service, dataset_id, ns, kg = gate
        job_id = service.create_job(ns)
        run(service.run_ingest(job_id, dataset_id, ns, PLAN, save=False))
        job = service.get_job(job_id)
        assert job["status"] == "completed", job
        outcomes = {r["filename"]: r for r in job["report"]["files"]}
        assert outcomes["wikidata.json"]["route"] == "records"
        assert outcomes["wikidata.json"]["entities_added"] >= 6
        assert outcomes["약관.md"]["entities_added"] >= 1     # LLM 추출
        assert outcomes["seed.jsonld"]["entities_added"] == 1  # upsert

    def test_rollup_inference_works_after_ingest(self, gate):
        """최종 증거 — 인제스트된 그래프에서 계층 롤업이 실제로 동작한다.
        '구조물'에 직접 달린 인스턴스는 0이지만 성문·석탑을 타고 6개가 나온다."""
        service, dataset_id, ns, kg = gate
        job_id = service.create_job(ns)
        run(service.run_ingest(job_id, dataset_id, ns, PLAN, save=False))

        rollup = service.get_hierarchy_rollup(ns, "HeritageClass:구조물")
        assert rollup["instances_direct"] == 0
        assert rollup["instances_total"] == 6

    def test_seed_aliases_feed_retrieval(self, gate):
        service, dataset_id, ns, kg = gate
        job_id = service.create_job(ns)
        run(service.run_ingest(job_id, dataset_id, ns, PLAN, save=False))
        assert "결제하기" in kg.graph.nodes["IntentSlot:결제"]["aliases"]

    def test_skip_route_is_skipped(self, gate):
        service, dataset_id, ns, kg = gate
        plan = [dict(PLAN[0], route="skip"), PLAN[1], PLAN[2]]
        job_id = service.create_job(ns)
        run(service.run_ingest(job_id, dataset_id, ns, plan, save=False))
        outcomes = {r["filename"]: r for r in
                    service.get_job(job_id)["report"]["files"]}
        assert outcomes["wikidata.json"]["route"] == "skip"
        assert outcomes["wikidata.json"]["entities_added"] == 0

    def test_one_bad_entry_does_not_kill_the_rest(self, gate):
        service, dataset_id, ns, kg = gate
        plan = [{"filename": "없는파일.json", "route": "records",
                 "mapping": MAPPING}] + PLAN[1:]
        job_id = service.create_job(ns)
        run(service.run_ingest(job_id, dataset_id, ns, plan, save=False))
        job = service.get_job(job_id)
        assert job["status"] == "completed"
        outcomes = {r["filename"]: r for r in job["report"]["files"]}
        assert outcomes["없는파일.json"]["error"]
        assert outcomes["seed.jsonld"]["entities_added"] == 1  # 나머지는 산다

    def test_records_route_without_mapping_is_an_error(self, gate):
        """매핑 없는 records 인제스트는 정체성 없는 노드를 만든다 — 거부."""
        service, dataset_id, ns, kg = gate
        plan = [{"filename": "wikidata.json", "route": "records"}]
        job_id = service.create_job(ns)
        run(service.run_ingest(job_id, dataset_id, ns, plan, save=False))
        outcomes = service.get_job(job_id)["report"]["files"]
        assert outcomes[0]["error"]

    def test_protected_namespace_is_rejected(self, gate):
        service, dataset_id, ns, kg = gate
        job_id = service.create_job("default")
        with pytest.raises(ValueError, match="protected"):
            run(service.run_ingest(job_id, dataset_id, "default", PLAN,
                                   save=False))


# ─── 6. analyze 응답에 plan 초안이 실린다 ────────────────────────────

class TestAnalyzeProducesPlanDraft:
    def test_plan_draft_matches_species(self, gate):
        service, dataset_id, ns, kg = gate

        def llm_with_mapping(prompt):
            if "매핑" in prompt:
                return json.dumps({**MAPPING, "hierarchy": HIERARCHY},
                                  ensure_ascii=False)
            return gate_fake_llm(prompt)

        service.llm_fn = llm_with_mapping
        report = run(service.analyze_dataset(dataset_id))
        plan = {p["filename"]: p for p in report["plan"]}
        assert plan["wikidata.json"]["route"] == "records"
        assert plan["wikidata.json"]["mapping"]["name_field"] == "이름"
        assert plan["wikidata.json"]["hierarchy"]["child_field"] == "이름"
        assert plan["약관.md"]["route"] == "articled"
        assert plan["seed.jsonld"]["route"] == "seed_ontology"
