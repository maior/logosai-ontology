"""
일관성 감시 에이전트 (Consistency Linter) — 그래프 내부 모순 탐지.

②번 검수 보조 에이전트. ①근거대조(evidence_checker)와 의도적으로 대비된다:
근거대조는 "원문에 있는가"라는 **판단**이 필요해 LLM 을 쓰지만, 일관성
검사(이름-타입 충돌 · 별칭 충돌 · domain/range 위반)는 전부 그래프 스캔으로
**결정 가능**하다. 프로젝트 절대 원칙: 결정 가능한 곳에 LLM 을 쓰지 않는다.

고정하는 계약:
1. **LLM 0콜** — lint_graph 는 llm 파라미터 자체가 없고, 모듈은 LLM
   프로바이더를 import 하지 않는다 (소스 스캔으로 고정 — test_kernel_decoupling
   이 import 유출을 서브프로세스로 고정하는 것과 같은 취지의 회귀 장치).
2. 발견(findings)은 재계산 가능한 파생물이다 — 저장하지 않는다. 저장하면
   그래프와 어긋난 순간 어느 쪽이 진실인지 알 수 없다 (review_store 의
   "로그가 원본" 원칙의 대우).
3. 출력은 결정적(정렬)이다 — 같은 그래프면 같은 순서. UI diff 와 테스트가
   순서에 기대도 안전하다. error 가 warn 보다 앞선다.
4. 미선언 술어는 range 검사를 건너뛴다 — 열린 어휘 허용 (계층 술어 is_a 등은
   설계상 미선언이다). 스키마에 없다고 위반이 아니다.
"""

import inspect
import json
import uuid

import networkx as nx
import pytest

from ontology.core import consistency_checker
from ontology.core.consistency_checker import lint_graph


@pytest.fixture(autouse=True)
def _clean():
    from ontology.core.chunk_store import reset_chunk_stores
    from ontology.core.review_store import reset_review_stores
    reset_chunk_stores()
    reset_review_stores()
    yield
    reset_chunk_stores()
    reset_review_stores()


def make_graph():
    """검사 대상의 최소 그래프 — KG 엔진의 .graph 와 같은 MultiDiGraph."""
    g = nx.MultiDiGraph()
    g.add_node("Clause:청약철회", name="청약철회", type="Clause",
               source="약관.md")
    g.add_node("Coverage:암진단비", name="암진단비", type="Coverage",
               source="약관.md")
    return g


# ─── 1. name_type_conflict — 같은 이름, 다른 타입 ────────────────────

class TestNameTypeConflict:
    def test_same_name_under_two_types_is_flagged(self):
        """빌드 간 스키마 drift 의 전형: Concept:청약철회 와 Clause:청약철회 가
        공존하면 사실상 같은 개체가 두 노드로 갈라진 것이다."""
        g = make_graph()
        g.add_node("Concept:청약철회", name="청약철회", type="Concept",
                   source="요약서.md")
        findings = [f for f in lint_graph(g) if f["kind"] == "name_type_conflict"]
        assert len(findings) == 1
        f = findings[0]
        assert f["severity"] == "warn"
        assert set(f["node_ids"]) == {"Clause:청약철회", "Concept:청약철회"}
        assert "병합" in f["suggestion"]

    def test_unique_names_are_not_flagged(self):
        assert [f for f in lint_graph(make_graph())
                if f["kind"] == "name_type_conflict"] == []


# ─── 2. alias_collision — 별칭이 다른 노드를 가리킨다 ─────────────────

class TestAliasCollision:
    def test_alias_equal_to_another_nodes_name_collides(self):
        """검색 확장(축 4)은 별칭을 확장어로 쓴다 — 별칭이 남의 이름과 같으면
        확장이 두 노드를 함께 끌어온다."""
        g = make_graph()
        g.add_node("Concept:계약해지", name="계약해지", type="Concept",
                   aliases=["암진단비"])  # 남의 이름과 충돌
        findings = [f for f in lint_graph(g) if f["kind"] == "alias_collision"]
        assert len(findings) == 1
        f = findings[0]
        assert f["severity"] == "warn"
        assert "암진단비" in f["detail"]  # 어느 별칭이 충돌했는지 보인다
        assert set(f["node_ids"]) == {"Concept:계약해지", "Coverage:암진단비"}

    def test_alias_equal_to_another_nodes_alias_collides(self):
        g = make_graph()
        g.add_node("Concept:취소", name="취소", type="Concept",
                   aliases=["계약 취소"])
        g.add_node("Concept:철회", name="철회", type="Concept",
                   aliases=["계약 취소"])
        findings = [f for f in lint_graph(g) if f["kind"] == "alias_collision"]
        assert len(findings) == 1
        assert "계약 취소" in findings[0]["detail"]

    def test_own_name_as_alias_is_not_a_collision(self):
        """자기 이름을 별칭으로 가진 것은 중복일 뿐 충돌이 아니다 —
        확장이 같은 노드로 돌아올 뿐이다."""
        g = make_graph()
        g.add_node("Concept:철회", name="철회", type="Concept",
                   aliases=["철회"])
        assert [f for f in lint_graph(g)
                if f["kind"] == "alias_collision"] == []


# ─── 3. range_violation — 선언된 술어의 endpoint 타입 위반 ────────────

SCHEMA_DICT = {"node_types": ["Clause", "Regulation", "Coverage"],
               "predicates": {"citesRegulation": ("Clause", "Regulation")}}


class TestRangeViolation:
    def test_declared_predicate_with_wrong_range_is_error(self):
        g = make_graph()
        # Clause → Coverage 인데 citesRegulation 은 (Clause, Regulation)
        g.add_edge("Clause:청약철회", "Coverage:암진단비",
                   predicate="citesRegulation")
        findings = [f for f in lint_graph(g, schema=SCHEMA_DICT)
                    if f["kind"] == "range_violation"]
        assert len(findings) == 1
        f = findings[0]
        assert f["severity"] == "error"
        assert "citesRegulation" in f["detail"]
        assert f["node_ids"] == ["Clause:청약철회", "Coverage:암진단비"]

    def test_matching_domain_and_range_pass(self):
        """dict 대신 BuilderSchema 객체로도 같은 결과 — 스키마는 데이터다."""
        from ontology.builder.models import BuilderSchema
        g = make_graph()
        g.add_node("Regulation:금소법", name="금소법", type="Regulation")
        g.add_edge("Clause:청약철회", "Regulation:금소법",
                   predicate="citesRegulation")
        schema = BuilderSchema.from_dict(SCHEMA_DICT)
        assert [f for f in lint_graph(g, schema=schema)
                if f["kind"] == "range_violation"] == []

    def test_undeclared_predicate_is_skipped(self):
        """열린 어휘 허용 — 계층 술어(is_a 등)는 설계상 미선언이다."""
        g = make_graph()
        g.add_edge("Clause:청약철회", "Coverage:암진단비", predicate="is_a")
        assert [f for f in lint_graph(g, schema=SCHEMA_DICT)
                if f["kind"] == "range_violation"] == []

    def test_no_schema_means_no_range_check(self):
        g = make_graph()
        g.add_edge("Clause:청약철회", "Coverage:암진단비",
                   predicate="citesRegulation")
        assert [f for f in lint_graph(g)
                if f["kind"] == "range_violation"] == []


# ─── 4. 출력 계약 — 결정적 순서 + 무저장 + LLM 0콜 ───────────────────

class TestOutputContract:
    def _messy_graph(self):
        g = make_graph()
        g.add_node("Concept:청약철회", name="청약철회", type="Concept")
        g.add_edge("Clause:청약철회", "Coverage:암진단비",
                   predicate="citesRegulation")
        return g

    def test_clean_graph_has_no_findings(self):
        assert lint_graph(make_graph()) == []

    def test_output_is_deterministic_and_errors_come_first(self):
        """같은 그래프 → 같은 결과 같은 순서. severity 는 error 가 앞선다 —
        UI 가 위에서부터 읽는 순서가 곧 심각도 순서다."""
        g = self._messy_graph()
        first = lint_graph(g, schema=SCHEMA_DICT)
        second = lint_graph(g, schema=SCHEMA_DICT)
        assert first == second
        severities = [f["severity"] for f in first]
        assert severities == sorted(severities,
                                    key=lambda s: 0 if s == "error" else 1)
        assert first[0]["severity"] == "error"

    def test_lint_graph_signature_has_no_llm(self):
        """계약 1 — 결정 가능한 곳에 LLM 을 쓰지 않는다. 시그니처 수준에서
        llm 주입 지점 자체가 없다."""
        params = list(inspect.signature(lint_graph).parameters)
        assert params == ["graph", "schema"]

    def test_module_source_never_touches_llm(self):
        """모듈 소스 스캔 — LLM 프로바이더/주입 코드가 한 줄도 없다."""
        src = inspect.getsource(consistency_checker)
        for banned in ("llm_provider", "resolve_provider", "llm_fn"):
            assert banned not in src, f"일관성 감시에 LLM 코드 유입: {banned}"


# ─── 5. 서비스 + API 관통 ────────────────────────────────────────────

@pytest.fixture
def lint_ns(tmp_path):
    from ontology.engines.knowledge_graph_clean import (KnowledgeGraphEngine,
                                                        _kg_instances)
    ns = f"lintns_{uuid.uuid4().hex[:8]}"
    engine = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
    _kg_instances[ns] = engine
    g = engine.graph
    g.add_node("Clause:청약철회", name="청약철회", type="Clause",
               source="약관.md")
    g.add_node("Concept:청약철회", name="청약철회", type="Concept",
               source="요약서.md")
    yield ns
    _kg_instances.pop(ns, None)


class TestServiceAndRoute:
    def test_service_lints_and_counts_without_persisting(self, lint_ns, tmp_path):
        """findings 는 재계산 가능한 파생물 — review_store 에 아무것도
        남기지 않는다 (저장하면 그래프와 어긋난 순간 진실이 둘이 된다)."""
        from ontology.core.review_store import get_review_store
        from ontology.server.service import OntologyBuilderService

        service = OntologyBuilderService(data_dir=tmp_path)
        result = service.lint_consistency(lint_ns)
        assert result["namespace"] == lint_ns
        assert result["counts"] == {"name_type_conflict": 1}
        assert result["findings"][0]["kind"] == "name_type_conflict"
        assert get_review_store(lint_ns).history() == []

    def test_route_happy_path(self, lint_ns, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ontology.server import router as server_router
        from ontology.server.service import OntologyBuilderService

        service = OntologyBuilderService(data_dir=tmp_path)
        app = FastAPI()
        app.include_router(server_router.router, prefix="/api/v1/ontology")
        app.dependency_overrides[server_router.get_ontology_service] = \
            lambda: service
        client = TestClient(app)

        response = client.get(
            f"/api/v1/ontology/graphs/{lint_ns}/review/consistency")
        assert response.status_code == 200
        body = response.json()
        assert body["counts"]["name_type_conflict"] == 1
        assert body["findings"][0]["severity"] == "warn"
