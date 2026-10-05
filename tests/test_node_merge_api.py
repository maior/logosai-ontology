"""노드 병합 API — 계획 미리보기(dry-run)와 적용이 같은 결과를 내는가.

plan_merge 는 순수 함수로 따로 검증했다(test_node_merge.py). 여기서 고정하는 것은
**배선**이다: 계획만 세우고 아무것도 안 바꾸는가, 적용하면 그래프·청크·골든셋이
함께 움직이는가, 그리고 기본값이 안전한가.

**dry_run 기본 True 가 계약이다.** 병합은 노드를 지워 되돌릴 수 없다 — body 를
실수로 보냈을 때 그래프가 파괴되는 경로가 있으면 안 된다.
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.core.chunk_index import reset_chunk_indices
from ontology.core.chunk_store import reset_chunk_stores
from ontology.core.review_store import reset_review_stores
from ontology.core.search_qa import reset_golden_sets

API = "/api/v1/ontology"


@pytest.fixture(autouse=True)
def _clean_singletons():
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()
    yield
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()


@pytest.fixture()
def client_and_service(tmp_path):
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    service = OntologyBuilderService(data_dir=tmp_path)
    app = FastAPI()
    app.include_router(server_router.router, prefix=API)
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app), service


@pytest.fixture()
def dup_namespace(client_and_service):
    """실측 사례를 그대로 재현한 네임스페이스.

    `Disease:C50( 유방의 악성 신생물 )` ↔ `Disease:유방의 악성 신생물` —
    ins_cancer_demo 에서 골든셋 정답이 근거를 못 찾게 만든 바로 그 쌍이다.
    """
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import get_chunk_store
    from ontology.core.search_qa import get_golden_set
    from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine

    client, service = client_and_service
    ns = f"mergens_{uuid.uuid4().hex[:8]}"
    winner = "Disease:C50( 유방의 악성 신생물 )"
    loser = "Disease:유방의 악성 신생물"

    engine = get_knowledge_graph_engine(ns)
    graph = engine.graph
    graph.add_node(winner, type="Disease", name="C50( 유방의 악성 신생물 )")
    graph.add_node(loser, type="Disease", name="유방의 악성 신생물",
                   definition="여성 유방에 생기는 악성 신생물")
    graph.add_node("Coverage:암진단비", type="Coverage", name="암진단비")
    graph.add_edge(loser, "Coverage:암진단비", predicate="hasCoverage",
                   weight=1.0, confidence=0.8)

    store = get_chunk_store(ns)
    store.add(Chunk(text="유방의 악성 신생물은 암진단비 지급 대상입니다.",
                    source="약관.pdf", index=0, char_start=0, char_end=28),
              node_ids=[loser])

    golden = get_golden_set(ns)
    case_id = golden.add("유방암 진단 시 보장되나?", loser, source="hand")
    golden.confirm(case_id)
    return client, service, ns, winner, loser, case_id


class TestDryRun:
    def test_default_is_dry_run(self, dup_namespace):
        """dry_run 을 명시하지 않으면 **바뀌지 않아야** 한다."""
        client, _, ns, winner, loser, _ = dup_namespace
        res = client.post(f"{API}/graphs/{ns}/nodes/merge",
                          json={"winner": winner, "losers": [loser]})
        assert res.status_code == 200 and res.json()["dry_run"] is True

    def test_dry_run_leaves_graph_untouched(self, dup_namespace):
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        client, _, ns, winner, loser, _ = dup_namespace
        client.post(f"{API}/graphs/{ns}/nodes/merge",
                    json={"winner": winner, "losers": [loser]})
        assert loser in get_knowledge_graph_engine(ns).graph

    def test_plan_shows_what_would_happen(self, dup_namespace):
        client, _, ns, winner, loser, case_id = dup_namespace
        plan = client.post(f"{API}/graphs/{ns}/nodes/merge",
                           json={"winner": winner, "losers": [loser]}
                           ).json()["plan"]
        assert plan["aliases_added"] == ["유방의 악성 신생물"]
        assert len(plan["chunks_rewritten"]) == 1
        assert plan["golden_relabels"] == [case_id]
        # 이긴 쪽에 정의문이 없으니 충돌이 아니라 흡수다
        assert plan["property_conflicts"] == []
        assert "definition" in plan["properties_adopted"]


class TestApply:
    def _apply(self, client, ns, winner, loser):
        return client.post(f"{API}/graphs/{ns}/nodes/merge",
                           json={"winner": winner, "losers": [loser],
                                 "dry_run": False})

    def test_loser_is_removed_and_alias_absorbed(self, dup_namespace):
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        client, _, ns, winner, loser, _ = dup_namespace
        assert self._apply(client, ns, winner, loser).status_code == 200
        graph = get_knowledge_graph_engine(ns).graph
        assert loser not in graph
        assert "유방의 악성 신생물" in graph.nodes[winner]["aliases"]

    def test_edge_moves_with_its_attributes(self, dup_namespace):
        """weight·confidence 를 잃으면 그래프의 신뢰도 정보가 사라진다."""
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        client, _, ns, winner, loser, _ = dup_namespace
        self._apply(client, ns, winner, loser)
        graph = get_knowledge_graph_engine(ns).graph
        edges = [d for _, t, d in graph.out_edges(winner, data=True)
                 if t == "Coverage:암진단비"]
        assert len(edges) == 1
        assert edges[0]["predicate"] == "hasCoverage"
        assert edges[0]["confidence"] == 0.8

    def test_chunk_evidence_follows_the_winner(self, dup_namespace):
        """이것이 병합의 목적이다 — 근거가 이긴 노드에 모여야 인용이 산다."""
        from ontology.core.chunk_store import get_chunk_store
        client, _, ns, winner, loser, _ = dup_namespace
        self._apply(client, ns, winner, loser)
        store = get_chunk_store(ns)
        assert len(store.chunks_for_node(winner)) == 1
        assert store.chunks_for_node(loser) == []

    def test_golden_label_follows_the_winner(self, dup_namespace):
        from ontology.core.search_qa import get_golden_set
        client, _, ns, winner, loser, case_id = dup_namespace
        self._apply(client, ns, winner, loser)
        case = {c.case_id: c for c in get_golden_set(ns).cases()}[case_id]
        assert case.expected_node_id == winner
        assert case.status == "confirmed"      # 사람의 확정 판단은 유지된다

    def test_merge_is_audited(self, dup_namespace):
        """되돌릴 수 없는 변경은 흔적을 남겨야 한다."""
        from ontology.core.review_store import get_review_store
        client, _, ns, winner, loser, _ = dup_namespace
        self._apply(client, ns, winner, loser)
        actions = [e.get("action") for e in get_review_store(ns).history()]
        assert "merge" in actions

    def test_no_tombstone_so_the_name_can_live_as_alias(self, dup_namespace):
        """묘비를 남기면 그 이름이 재빌드에서 영구 차단된다 — 별칭이 죽는다."""
        from ontology.core.review_store import get_review_store
        client, _, ns, winner, loser, _ = dup_namespace
        self._apply(client, ns, winner, loser)
        assert not get_review_store(ns).is_rejected(loser)


class TestErrors:
    def test_unknown_winner_is_404(self, dup_namespace):
        client, _, ns, _, loser, _ = dup_namespace
        res = client.post(f"{API}/graphs/{ns}/nodes/merge",
                          json={"winner": "Nope:없음", "losers": [loser]})
        assert res.status_code == 404

    def test_unknown_loser_is_404(self, dup_namespace):
        client, _, ns, winner, _, _ = dup_namespace
        res = client.post(f"{API}/graphs/{ns}/nodes/merge",
                          json={"winner": winner, "losers": ["Nope:없음"]})
        assert res.status_code == 404

    def test_self_merge_is_400(self, dup_namespace):
        client, _, ns, winner, _, _ = dup_namespace
        res = client.post(f"{API}/graphs/{ns}/nodes/merge",
                          json={"winner": winner, "losers": [winner]})
        assert res.status_code == 400

    def test_empty_losers_is_rejected(self, dup_namespace):
        client, _, ns, winner, _, _ = dup_namespace
        res = client.post(f"{API}/graphs/{ns}/nodes/merge",
                          json={"winner": winner, "losers": []})
        assert res.status_code in (400, 422)   # 스키마 또는 계획 단계에서 거절

    def test_protected_namespace_is_403(self, client_and_service):
        client, _ = client_and_service
        res = client.post(f"{API}/graphs/default/nodes/merge",
                          json={"winner": "A:a", "losers": ["A:b"]})
        assert res.status_code == 403
