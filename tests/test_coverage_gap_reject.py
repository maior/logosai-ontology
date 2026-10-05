"""커버리지 gap 기각의 영속성 — 관계판 묘비의 대칭 (2026-08-21).

관계 제안은 기각이 어디에도 남지 않아 **같은 오제안이 매 라운드 재출현**했다
(실측 8/10). 그 처방이 `relation_reject` 묘비였다. 커버리지 gap 에는 그
장치가 없어 같은 결함이 그대로 남아 있었다 — 게다가 조건이 더 나쁘다:

  - 재실행 비용이 **청크당 LLM 1콜**이고,
  - `auto_coverage_check` 로 빌드마다 자동 실행될 수 있으며,
  - `reject_node` 는 노드를 그래프에서 **제거**하므로 다음 라운드의
    `known_names` 에서도 사라져 LLM 이 반드시 다시 제안한다.

계약:
  1. 기각은 append-only 로그에 남고 replay 로 복원된다 (프로세스 재시작 생존).
  2. `check_coverage` 는 기각된 후보를 거르되 **몇 건 걸렀는지 보고한다** —
     조용한 절단 금지.
  3. scope="entity"(기본)는 어느 청크에서 와도 거르고, "evidence" 는 그
     청크의 재제안만 거른다 (관계 쪽 triple/evidence 와 같은 갈래).
  4. 승인이 묘비를 걷는다 — 나중 판정이 이긴다.
  5. 기각된 후보는 승인 경로에서도 막힌다 (명시적 override 만 통과).
  6. 사유(reason)는 필수 — 이유 없는 기각은 다음 사람이 재검토할 수 없다.
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.core.chunk_index import reset_chunk_indices
from ontology.core.chunk_store import reset_chunk_stores
from ontology.core.review_store import get_review_store, reset_review_stores
from ontology.core.search_qa import reset_golden_sets

API = "/api/v1/ontology"
BODY = "제6조 【보험금의 지급사유】 회사는 피보험자에게 암진단비를 지급합니다. " * 6


@pytest.fixture(autouse=True)
def _clean():
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()
    yield
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()


class _StubLLM:
    """언제나 같은 gap 하나를 낸다 — 재출현을 재현하는 최소 조건."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return ('{"candidates": [{"name": "암진단비", '
                '"type": "InsuranceTerm"}]}')


@pytest.fixture()
def env(tmp_path):
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import get_chunk_store
    from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    llm = _StubLLM()
    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=llm)
    app = FastAPI()
    app.include_router(server_router.router, prefix=API)
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    ns = f"gapns_{uuid.uuid4().hex[:8]}"

    engine = get_knowledge_graph_engine(ns)
    engine.graph.clear()
    # 허용 타입이 그래프에 실존해야 승인이 통과한다 (스키마 뒷문 금지)
    engine.graph.add_node("InsuranceTerm:해지", type="InsuranceTerm", name="해지")

    store = get_chunk_store(ns)
    c1 = store.add(Chunk(text=BODY, source="약관.pdf", index=0,
                         char_start=0, char_end=len(BODY)))
    c2 = store.add(Chunk(text=BODY, source="약관.pdf", index=1,
                         char_start=len(BODY), char_end=len(BODY) * 2))
    return TestClient(app), ns, service, llm, c1, c2


def _check(client, ns, **body):
    return client.post(f"{API}/graphs/{ns}/review/coverage", json=body).json()


def _reject(client, ns, chunk_id="", scope="entity", reason="개념이 아니라 상품명"):
    return client.post(f"{API}/graphs/{ns}/review/coverage/reject", json={
        "gaps": [{"type": "InsuranceTerm", "name": "암진단비",
                  "chunk_id": chunk_id, "scope": scope, "reason": reason}],
        "actor": "kenneth"}).json()


class TestRejectionPersists:
    def test_rejection_is_recorded_and_survives_reload(self, env):
        client, ns, service, llm, c1, _ = env
        out = _reject(client, ns)
        assert out.get("rejected") == 1, out

        store = get_review_store(ns)
        assert store.gap_rejection("InsuranceTerm", "암진단비") is not None

        # replay 로 복원되는가 — 프로세스 재시작 생존이 묘비의 존재 이유
        reset_review_stores()
        assert get_review_store(ns).gap_rejection(
            "InsuranceTerm", "암진단비") is not None

    def test_reason_is_required(self, env):
        client, ns, *_ = env
        out = client.post(f"{API}/graphs/{ns}/review/coverage/reject", json={
            "gaps": [{"type": "InsuranceTerm", "name": "암진단비"}],
            "actor": "k"}).json()
        assert out.get("rejected") == 0
        assert any(s.get("reason") == "reason_required"
                   for s in out.get("skipped") or []), out


class TestFilteringIsLoud:
    def test_rejected_gap_is_filtered_and_counted(self, env):
        client, ns, service, llm, c1, c2 = env
        before = _check(client, ns, limit=2)
        assert len(before["gaps"]) == 2          # 두 청크 모두에서 제안
        assert before.get("rejected_filtered", 0) == 0

        _reject(client, ns)
        after = _check(client, ns, limit=2)
        assert after["gaps"] == []
        # 조용한 절단 금지 — 몇 건을 걸렀는지 보고해야 한다
        assert after["rejected_filtered"] == 2, after

    def test_evidence_scope_only_filters_that_chunk(self, env):
        client, ns, service, llm, c1, c2 = env
        _reject(client, ns, chunk_id=c1, scope="evidence")
        after = _check(client, ns, limit=2)
        assert after["rejected_filtered"] == 1
        assert [g["chunk_id"] for g in after["gaps"]] == [c2]


class TestApprovalInteraction:
    def test_rejected_gap_is_blocked_on_approve(self, env):
        client, ns, service, llm, c1, _ = env
        _reject(client, ns)
        out = client.post(f"{API}/graphs/{ns}/review/coverage/approve", json={
            "gaps": [{"type": "InsuranceTerm", "name": "암진단비",
                      "chunk_id": c1}],
            "actor": "k", "dry_run": False}).json()
        assert out["created"] == []
        assert any(s.get("reason") == "gap_rejected"
                   for s in out.get("skipped") or []), out

    def test_explicit_override_passes_and_lifts_the_tombstone(self, env):
        """나중 판정이 이긴다 — 명시적 번복만 통과하고 묘비는 걷힌다."""
        client, ns, service, llm, c1, _ = env
        _reject(client, ns)
        out = client.post(f"{API}/graphs/{ns}/review/coverage/approve", json={
            "gaps": [{"type": "InsuranceTerm", "name": "암진단비",
                      "chunk_id": c1, "override_rejected": True}],
            "actor": "k", "dry_run": False}).json()
        assert out["created"] == ["InsuranceTerm:암진단비"], out
        assert get_review_store(ns).gap_rejection(
            "InsuranceTerm", "암진단비") is None


class TestGuard:
    def test_protected_namespace_is_blocked(self, env):
        client, *_ = env
        resp = client.post(f"{API}/graphs/default/review/coverage/reject", json={
            "gaps": [{"type": "T", "name": "x", "reason": "r"}], "actor": "k"})
        assert resp.status_code == 403
