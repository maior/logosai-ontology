"""
관리 콘솔 API — /admin/overview · /graphs/{ns}/stats · tombstones · DELETE.

관리페이지(9275 /ontology-admin)가 소비하는 네 능력을 고정한다:
1. **overview 는 진실을 말한다** — 로드 여부와 무관하게 노드/청크/검수
   현황을 싣고, protected 플래그로 삭제 불가 네임스페이스를 표시한다.
2. **stats 는 유령을 만들지 않는다** — 미지 네임스페이스 조회가 빈 엔진을
   생성·등록하면 그 자체가 네임스페이스 오염이다 → 404.
3. **묘비는 조회 가능하다** — 거절 이력(사유·시점)은 관리자가 열람할 수
   있어야 부활 차단이 신뢰받는다.
4. **삭제는 완전하고, protected 는 절대 못 지운다** — 파일(kg/chunks/
   reviews)과 인메모리 싱글턴이 함께 사라지고, default 는 403 이다.
"""

import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.core.chunk_index import reset_chunk_indices
from ontology.core.chunk_store import reset_chunk_stores
from ontology.core.review_store import reset_review_stores
from ontology.core.search_qa import reset_golden_sets

SAMPLE_MD = """## 제1조 청약철회
계약자는 15일 이내에 청약을 철회할 수 있습니다. 금융소비자 보호에 관한 법률을 따릅니다.

## 제2조 보장내용
암진단비는 최초 1회에 한하여 지급합니다.
"""


def fake_llm(prompt: str) -> str:
    if "청약철회" in prompt:
        return json.dumps({
            "entities": [
                {"name": "청약철회", "type": "Clause", "attrs": {}},
                {"name": "금융소비자 보호에 관한 법률", "type": "Regulation",
                 "attrs": {}},
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


@pytest.fixture(autouse=True)
def _clean_singletons():
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()
    yield
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()


@pytest.fixture()
def client(tmp_path):
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=fake_llm)
    app = FastAPI()
    app.include_router(server_router.router, prefix="/api/v1/ontology")
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app)


@pytest.fixture()
def client_and_service(tmp_path):
    """client + 그 service 인스턴스 (내부 상수 조정이 필요한 테스트용)."""
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=fake_llm)
    app = FastAPI()
    app.include_router(server_router.router, prefix="/api/v1/ontology")
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app), service


def unique_ns():
    return f"testns_{uuid.uuid4().hex[:8]}"


def build_namespace(client, namespace, save=False):
    """샘플 문서 업로드 → 빌드 완료까지 (TestClient 는 BackgroundTasks 동기)."""
    upload = client.post(
        "/api/v1/ontology/datasets",
        files=[("files", ("terms.md", SAMPLE_MD.encode("utf-8"),
                          "text/markdown"))])
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()["dataset_id"]

    response = client.post("/api/v1/ontology/build", json={
        "dataset_id": dataset_id, "namespace": namespace, "save": save})
    assert response.status_code == 200, response.text
    job = client.get(
        f"/api/v1/ontology/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "completed", job
    return namespace


def delete_ns(client, namespace):
    """테스트 잔여물 정리 — 실패해도 무시 (정리는 보조다)."""
    client.delete(f"/api/v1/ontology/graphs/{namespace}")


# ─── 1. Overview (대시보드 한 콜) ───────────────────────────────────

class TestAdminOverview:
    def test_overview_lists_built_namespace_with_stats(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            response = client.get("/api/v1/ontology/admin/overview")
            assert response.status_code == 200
            body = response.json()
            entries = {e["namespace"]: e for e in body["namespaces"]}
            assert ns in entries
            entry = entries[ns]
            assert entry["nodes"] >= 3          # 청약철회·법률·암진단비
            assert entry["edges"] >= 1
            assert entry["chunks"] >= 1         # 축 2 — 원문이 보존됐다
            assert entry["protected"] is False
            # 검수 현황: LLM 추출 노드는 판정 전이므로 전부 pending
            assert entry["review"]["pending"] >= 3
            assert entry["review"]["confirmed"] == 0
            assert entry["review"]["rejected"] == 0
        finally:
            delete_ns(client, ns)

    def test_overview_has_totals_and_protected_flags(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get("/api/v1/ontology/admin/overview").json()
            assert body["totals"]["namespaces"] == len(body["namespaces"])
            assert body["totals"]["nodes"] >= 3
            for entry in body["namespaces"]:
                assert "protected" in entry
                if entry["namespace"] == "default":
                    assert entry["protected"] is True
        finally:
            delete_ns(client, ns)


# ─── 2. Namespace stats (상세) ──────────────────────────────────────

class TestNamespaceStats:
    def test_stats_shape_for_built_namespace(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            response = client.get(f"/api/v1/ontology/graphs/{ns}/stats")
            assert response.status_code == 200
            body = response.json()
            assert body["namespace"] == ns
            assert body["nodes"] >= 3
            assert body["node_types"].get("Clause", 0) >= 1
            assert isinstance(body["trust"], dict)
            assert isinstance(body["predicates"], dict)
            review = body["review"]
            assert review["pending"] >= 3
            assert review["confirmed"] == 0 and review["rejected"] == 0
            assert body["chunks"] >= 1
            assert body["golden_cases"] == 0
        finally:
            delete_ns(client, ns)

    def test_stats_unknown_namespace_404_and_no_ghost(self, client):
        ghost = unique_ns()  # 어디에도 없는 이름
        response = client.get(f"/api/v1/ontology/graphs/{ghost}/stats")
        assert response.status_code == 404
        # 조회가 유령 네임스페이스를 만들지 않았다 (핵심 계약)
        listing = client.get("/api/v1/ontology/namespaces").json()
        assert ghost not in [n["namespace"] for n in listing["namespaces"]]


# ─── 3. Tombstones (묘비 열람) ──────────────────────────────────────

class TestTombstones:
    def test_rejected_node_appears_with_reason(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            queue = client.get(
                f"/api/v1/ontology/graphs/{ns}/review?limit=5").json()
            node_id = queue["items"][0]["node_id"]
            rejected = client.post(
                f"/api/v1/ontology/graphs/{ns}/review/reject",
                json={"node_id": node_id, "reason": "오추출",
                      "actor": "admin_test"})
            assert rejected.status_code == 200

            response = client.get(f"/api/v1/ontology/graphs/{ns}/tombstones")
            assert response.status_code == 200
            stones = response.json()["tombstones"]
            assert len(stones) == 1
            assert stones[0]["node_id"] == node_id
            assert stones[0]["reason"] == "오추출"
            assert stones[0]["actor"] == "admin_test"
            assert stones[0]["at"]
        finally:
            delete_ns(client, ns)

    def test_tombstones_empty_for_fresh_namespace(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(
                f"/api/v1/ontology/graphs/{ns}/tombstones").json()
            assert body["tombstones"] == []
        finally:
            delete_ns(client, ns)


# ─── 4. Delete (완전 삭제 · protected 불가침) ───────────────────────

class TestDeleteNamespace:
    def test_delete_protected_namespace_403(self, client):
        response = client.delete("/api/v1/ontology/graphs/default")
        assert response.status_code == 403

    def test_delete_unknown_namespace_404(self, client):
        response = client.delete(f"/api/v1/ontology/graphs/{unique_ns()}")
        assert response.status_code == 404

    def test_delete_rejects_path_traversal(self, client):
        response = client.delete("/api/v1/ontology/graphs/..%2F..%2Fetc")
        assert response.status_code in (403, 404)

    def test_delete_removes_files_memory_and_listing(self, client):
        from ontology.engines.knowledge_graph_clean import (
            _DEFAULT_DATA_DIR, _kg_instances)

        ns = build_namespace(client, unique_ns(), save=True)
        kg_file = _DEFAULT_DATA_DIR / f"kg_{ns}.json"
        assert kg_file.exists()          # save=True 가 체크포인트를 남겼다
        assert ns in _kg_instances

        response = client.delete(f"/api/v1/ontology/graphs/{ns}")
        assert response.status_code == 200
        body = response.json()
        assert body["namespace"] == ns
        assert any(f"kg_{ns}.json" in f for f in body["deleted_files"])

        assert not kg_file.exists()      # 파일이 실제로 사라졌다
        assert ns not in _kg_instances   # 인메모리도 내려갔다
        listing = client.get("/api/v1/ontology/namespaces").json()
        assert ns not in [n["namespace"] for n in listing["namespaces"]]
        # 삭제 후 stats 는 404 — 유령 생성 없이
        assert client.get(
            f"/api/v1/ontology/graphs/{ns}/stats").status_code == 404

    def test_delete_removes_chunk_and_review_sidecars(self, client):
        from ontology.engines.knowledge_graph_clean import _DEFAULT_DATA_DIR

        ns = build_namespace(client, unique_ns(), save=True)
        # 거절 한 건 → reviews_{ns}.jsonl 이 디스크에 생긴다
        queue = client.get(
            f"/api/v1/ontology/graphs/{ns}/review?limit=1").json()
        client.post(f"/api/v1/ontology/graphs/{ns}/review/reject",
                    json={"node_id": queue["items"][0]["node_id"],
                          "reason": "cleanup", "actor": "t"})
        review_file = _DEFAULT_DATA_DIR / f"reviews_{ns}.jsonl"
        assert review_file.exists()

        response = client.delete(f"/api/v1/ontology/graphs/{ns}")
        assert response.status_code == 200
        assert not review_file.exists()
        assert not (_DEFAULT_DATA_DIR / f"chunks_{ns}.jsonl").exists()


# ─── 5. Schema overview (TBox — 클래스 · 술어 · is_a 계층) ──────────

class TestSchemaOverview:
    def test_schema_lists_classes_and_predicate_signatures(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            response = client.get(f"/api/v1/ontology/graphs/{ns}/schema")
            assert response.status_code == 200
            body = response.json()

            classes = {c["type"]: c for c in body["classes"]}
            assert classes["Clause"]["count"] == 1
            assert classes["Regulation"]["count"] == 1
            # 프로퍼티 사용 분포 — 어떤 속성이 얼마나 채워졌는가
            assert "name" in classes["Clause"]["properties"]

            preds = {p["predicate"]: p for p in body["predicates"]}
            assert preds["citesRegulation"]["count"] == 1
            pairs = preds["citesRegulation"]["pairs"]
            assert {"source_type": "Clause", "target_type": "Regulation",
                    "count": 1} in pairs
            # is_a 없음 → 빈 계층 (있다고 지어내지 않는다)
            assert body["hierarchy"] == []
            assert body["hierarchy_edges"] == 0
        finally:
            delete_ns(client, ns)

    def test_schema_hierarchy_reflects_is_a_edges(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            # 수동으로 is_a 한 줄: 청약철회 --is_a--> 암진단비 (내용은 무의미,
            # 계층 조립만 검증)
            child, parent = "Clause:청약철회", "Coverage:암진단비"
            added = client.post(f"/api/v1/ontology/graphs/{ns}/edges", json={
                "source": child, "predicate": "is_a", "target": parent})
            assert added.status_code == 200

            body = client.get(f"/api/v1/ontology/graphs/{ns}/schema").json()
            assert body["hierarchy_edges"] == 1
            roots = {t["id"]: t for t in body["hierarchy"]}
            assert parent in roots
            assert [c["id"] for c in roots[parent]["children"]] == [child]
        finally:
            delete_ns(client, ns)

    def test_schema_unknown_namespace_404(self, client):
        assert client.get(
            f"/api/v1/ontology/graphs/{unique_ns()}/schema").status_code == 404


# ─── 6. Node browser (검색 · 필터 · 페이지네이션) ───────────────────

class TestNodeBrowser:
    def test_list_nodes_with_total_and_shape(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(f"/api/v1/ontology/graphs/{ns}/nodes").json()
            assert body["total"] == 3
            item = body["items"][0]
            for key in ("node_id", "name", "type", "trust", "source",
                        "aliases", "out_degree", "in_degree"):
                assert key in item
        finally:
            delete_ns(client, ns)

    def test_list_nodes_query_and_type_filters(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            base = f"/api/v1/ontology/graphs/{ns}/nodes"
            hits = client.get(f"{base}?q=청약").json()
            assert hits["total"] == 1
            assert hits["items"][0]["name"] == "청약철회"

            typed = client.get(f"{base}?node_type=Regulation").json()
            assert typed["total"] == 1
            assert typed["items"][0]["type"] == "Regulation"
        finally:
            delete_ns(client, ns)

    def test_list_nodes_pagination_keeps_total(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            base = f"/api/v1/ontology/graphs/{ns}/nodes"
            page = client.get(f"{base}?limit=1&offset=1").json()
            assert page["total"] == 3       # total 은 전체, items 는 창
            assert len(page["items"]) == 1
        finally:
            delete_ns(client, ns)

    def test_list_nodes_unknown_namespace_404(self, client):
        assert client.get(
            f"/api/v1/ontology/graphs/{unique_ns()}/nodes").status_code == 404

    def test_list_nodes_count_not_capped_when_small(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(f"/api/v1/ontology/graphs/{ns}/nodes").json()
            assert body["total"] == 3
            assert body["capped"] is False   # 상한 훨씬 아래
        finally:
            delete_ns(client, ns)

    def test_list_nodes_count_capped_stops_full_scan(self, client_and_service):
        # 상한을 2로 낮춰, 3노드 그래프에서 스캔이 멈추고 capped 로 표시되는지
        client, service = client_and_service
        service.COUNT_CAP = 2
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(f"/api/v1/ontology/graphs/{ns}/nodes").json()
            assert body["total"] == 2          # 전체(3)가 아니라 상한에서 멈춤
            assert body["capped"] is True       # "N+" 로 표기하라는 신호
        finally:
            client.delete(f"/api/v1/ontology/graphs/{ns}")


# ─── 7. Node editing (생성 · 수정 · 프로퍼티 삭제) ──────────────────

class TestNodeEditing:
    def test_create_node_appears_in_graph_and_audit(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            response = client.post(f"/api/v1/ontology/graphs/{ns}/nodes", json={
                "node_type": "Coverage", "name": "수술비",
                "definition": "수술 1회당 지급", "aliases": ["수술급여금"],
                "actor": "admin"})
            assert response.status_code == 200, response.text
            node_id = response.json()["id"]
            assert node_id == "Coverage:수술비"

            detail = client.get(
                f"/api/v1/ontology/graphs/{ns}/node?id={node_id}").json()
            assert detail["attrs"]["definition"] == "수술 1회당 지급"
            assert detail["attrs"]["aliases"] == ["수술급여금"]

            history = client.get(
                f"/api/v1/ontology/graphs/{ns}/review/history").json()["history"]
            assert any(h["action"] == "create" and h["node_id"] == node_id
                       for h in history)
        finally:
            delete_ns(client, ns)

    def test_create_duplicate_409_blank_400(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            dup = client.post(f"/api/v1/ontology/graphs/{ns}/nodes", json={
                "node_type": "Clause", "name": "청약철회"})
            assert dup.status_code == 409
            blank = client.post(f"/api/v1/ontology/graphs/{ns}/nodes", json={
                "node_type": "Clause", "name": "   "})
            assert blank.status_code in (400, 422)
        finally:
            delete_ns(client, ns)

    def test_create_lifts_tombstone(self, client):
        """수동 재생성은 인간의 명시적 번복 — confirm 이벤트로 묘비를 걷는다
        (나중 판정이 이긴다는 검수 루프 계약과 같은 규칙)."""
        ns = build_namespace(client, unique_ns())
        try:
            node_id = "Clause:청약철회"
            client.post(f"/api/v1/ontology/graphs/{ns}/review/reject",
                        json={"node_id": node_id, "reason": "test"})
            assert client.get(f"/api/v1/ontology/graphs/{ns}/tombstones"
                              ).json()["tombstones"] != []

            recreated = client.post(
                f"/api/v1/ontology/graphs/{ns}/nodes",
                json={"node_type": "Clause", "name": "청약철회",
                      "actor": "admin"})
            assert recreated.status_code == 200
            assert client.get(f"/api/v1/ontology/graphs/{ns}/tombstones"
                              ).json()["tombstones"] == []
        finally:
            delete_ns(client, ns)

    def test_update_node_fields_and_property_removal(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            node_id = "Clause:청약철회"
            response = client.patch(
                f"/api/v1/ontology/graphs/{ns}/node?node_id={node_id}",
                json={"updates": {"definition": "15일 이내 철회 가능",
                                  "trust": "authoritative",
                                  "custom_tag": "핵심조항"},
                      "actor": "admin"})
            assert response.status_code == 200, response.text
            attrs = response.json()["attrs"]
            assert attrs["definition"] == "15일 이내 철회 가능"
            assert attrs["trust"] == "authoritative"
            assert attrs["custom_tag"] == "핵심조항"

            # null 은 프로퍼티 삭제다
            removed = client.patch(
                f"/api/v1/ontology/graphs/{ns}/node?node_id={node_id}",
                json={"updates": {"custom_tag": None}})
            assert "custom_tag" not in removed.json()["attrs"]

            # 감사: edit 이벤트에 before/after 가 남는다
            history = client.get(
                f"/api/v1/ontology/graphs/{ns}/review/history").json()["history"]
            edits = [h for h in history if h["action"] == "edit"]
            assert len(edits) == 2
            assert edits[-1]["after"]["trust"] == "authoritative"
        finally:
            delete_ns(client, ns)

    def test_update_missing_node_404_empty_400(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            missing = client.patch(
                f"/api/v1/ontology/graphs/{ns}/node?node_id=no:such",
                json={"updates": {"name": "x"}})
            assert missing.status_code == 404
            empty = client.patch(
                f"/api/v1/ontology/graphs/{ns}/node?node_id=Clause:청약철회",
                json={"updates": {}})
            assert empty.status_code in (400, 422)
        finally:
            delete_ns(client, ns)

    def test_writes_to_protected_namespace_403(self, client):
        create = client.post("/api/v1/ontology/graphs/default/nodes", json={
            "node_type": "X", "name": "y"})
        assert create.status_code == 403
        patch = client.patch(
            "/api/v1/ontology/graphs/default/node?node_id=x",
            json={"updates": {"name": "y"}})
        assert patch.status_code == 403
        edge = client.post("/api/v1/ontology/graphs/default/edges", json={
            "source": "a", "predicate": "p", "target": "b"})
        assert edge.status_code == 403


# ─── 8. Edge editing (관계 추가 · 삭제) ─────────────────────────────

class TestEdgeEditing:
    def test_add_edge_appears_in_node_detail_and_audit(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            response = client.post(f"/api/v1/ontology/graphs/{ns}/edges", json={
                "source": "Coverage:암진단비", "predicate": "citesRegulation",
                "target": "Regulation:금융소비자 보호에 관한 법률",
                "actor": "admin"})
            assert response.status_code == 200, response.text

            detail = client.get(
                f"/api/v1/ontology/graphs/{ns}/node?id=Coverage:암진단비"
            ).json()
            assert any(e["predicate"] == "citesRegulation"
                       for e in detail["out_edges"])

            history = client.get(
                f"/api/v1/ontology/graphs/{ns}/review/history").json()["history"]
            assert any(h["action"] == "edge_added" for h in history)
        finally:
            delete_ns(client, ns)

    def test_add_edge_duplicate_409_missing_404(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            edge = {"source": "Clause:청약철회", "predicate": "citesRegulation",
                    "target": "Regulation:금융소비자 보호에 관한 법률"}
            dup = client.post(f"/api/v1/ontology/graphs/{ns}/edges", json=edge)
            assert dup.status_code == 409      # 빌드가 이미 만든 엣지
            ghost = client.post(f"/api/v1/ontology/graphs/{ns}/edges", json={
                "source": "no:such", "predicate": "p",
                "target": "Clause:청약철회"})
            assert ghost.status_code == 404
        finally:
            delete_ns(client, ns)

    def test_remove_edge_and_404_when_absent(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            params = ("source=Clause:청약철회&predicate=citesRegulation"
                      "&target=Regulation:금융소비자 보호에 관한 법률")
            removed = client.delete(
                f"/api/v1/ontology/graphs/{ns}/edges?{params}")
            assert removed.status_code == 200

            detail = client.get(
                f"/api/v1/ontology/graphs/{ns}/node?id=Clause:청약철회"
            ).json()
            assert detail["out_edges"] == []

            again = client.delete(
                f"/api/v1/ontology/graphs/{ns}/edges?{params}")
            assert again.status_code == 404

            history = client.get(
                f"/api/v1/ontology/graphs/{ns}/review/history").json()["history"]
            assert any(h["action"] == "edge_removed" for h in history)
        finally:
            delete_ns(client, ns)


# ─── 9. Property usage (프로퍼티가 어디서·어떤 값으로 쓰이나) ────────

class TestPropertyUsage:
    def test_property_filter_returns_only_nodes_with_value(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            # 한 노드에만 definition 을 채운다
            client.patch(
                f"/api/v1/ontology/graphs/{ns}/node?node_id=Clause:청약철회",
                json={"updates": {"definition": "15일 이내 철회"}})
            body = client.get(
                f"/api/v1/ontology/graphs/{ns}/nodes?property=definition").json()
            assert body["total"] == 1
            item = body["items"][0]
            assert item["node_id"] == "Clause:청약철회"
            # 그 프로퍼티의 실제 값이 실려온다 (어떻게 쓰이나)
            assert item["prop_value"] == "15일 이내 철회"
        finally:
            delete_ns(client, ns)

    def test_property_filter_combines_with_type(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            # name 은 모든 노드에 있다 → type 으로 좁히면 그 타입만
            body = client.get(
                f"/api/v1/ontology/graphs/{ns}/nodes"
                f"?property=name&node_type=Regulation").json()
            assert body["total"] == 1
            assert body["items"][0]["type"] == "Regulation"
            assert body["items"][0]["prop_value"]
        finally:
            delete_ns(client, ns)

    def test_property_filter_empty_when_unused(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(
                f"/api/v1/ontology/graphs/{ns}/nodes?property=no_such_prop").json()
            assert body["total"] == 0
        finally:
            delete_ns(client, ns)


# ─── 10. Edge listing (술어별 엣지 — 술어 관리 진입점) ──────────────

class TestEdgeListing:
    def test_list_edges_by_predicate_resolves_endpoints(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(
                f"/api/v1/ontology/graphs/{ns}/edges"
                f"?predicate=citesRegulation").json()
            assert body["predicate"] == "citesRegulation"
            assert body["total"] == 1
            e = body["edges"][0]
            assert e["source"] == "Clause:청약철회"
            assert e["target"] == "Regulation:금융소비자 보호에 관한 법률"
            # 양끝 노드의 이름·타입이 해석돼 실려온다 (클릭 이동용)
            assert e["source_type"] == "Clause"
            assert e["target_type"] == "Regulation"
            assert e["source_name"] and e["target_name"]
        finally:
            delete_ns(client, ns)

    def test_list_edges_signature_filter(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            base = f"/api/v1/ontology/graphs/{ns}/edges?predicate=citesRegulation"
            hit = client.get(
                f"{base}&source_type=Clause&target_type=Regulation").json()
            assert hit["total"] == 1
            miss = client.get(f"{base}&source_type=Coverage").json()
            assert miss["total"] == 0
        finally:
            delete_ns(client, ns)

    def test_list_edges_all_when_no_predicate(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            body = client.get(f"/api/v1/ontology/graphs/{ns}/edges").json()
            assert body["total"] == 1   # 빌드가 만든 엣지 1개
        finally:
            delete_ns(client, ns)

    def test_list_edges_unknown_namespace_404(self, client):
        assert client.get(
            f"/api/v1/ontology/graphs/{unique_ns()}/edges").status_code == 404


# ─── 11. Saved explorations (SQLite 영속 — 팔란티어 saved views) ─────

class TestSavedViews:
    def test_create_list_delete_roundtrip(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            base = f"/api/v1/ontology/graphs/{ns}/views"
            assert client.get(base).json()["views"] == []

            created = client.post(base, json={
                "name": "Clause 인스턴스",
                "filter": {"node_type": "Clause", "q": "청약"}})
            assert created.status_code == 200, created.text
            vid = created.json()["id"]

            views = client.get(base).json()["views"]
            assert len(views) == 1
            assert views[0]["name"] == "Clause 인스턴스"
            # 필터가 그대로 복원된다 (클릭 한 번으로 탐색 재현)
            assert views[0]["filter"]["node_type"] == "Clause"
            assert views[0]["filter"]["q"] == "청약"
            assert views[0]["created_at"]

            assert client.delete(f"{base}/{vid}").status_code == 200
            assert client.get(base).json()["views"] == []
        finally:
            delete_ns(client, ns)

    def test_duplicate_name_409(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            base = f"/api/v1/ontology/graphs/{ns}/views"
            payload = {"name": "내 탐색", "filter": {}}
            assert client.post(base, json=payload).status_code == 200
            assert client.post(base, json=payload).status_code == 409
        finally:
            delete_ns(client, ns)

    def test_views_are_namespace_scoped(self, client):
        ns_a = build_namespace(client, unique_ns())
        ns_b = build_namespace(client, unique_ns())
        try:
            client.post(f"/api/v1/ontology/graphs/{ns_a}/views",
                        json={"name": "A뷰", "filter": {}})
            assert len(client.get(
                f"/api/v1/ontology/graphs/{ns_a}/views").json()["views"]) == 1
            # 다른 네임스페이스에는 새어들지 않는다
            assert client.get(
                f"/api/v1/ontology/graphs/{ns_b}/views").json()["views"] == []
        finally:
            delete_ns(client, ns_a); delete_ns(client, ns_b)

    def test_delete_namespace_purges_views(self, client):
        ns = unique_ns()
        build_namespace(client, ns, save=True)
        client.post(f"/api/v1/ontology/graphs/{ns}/views",
                    json={"name": "곧사라질뷰", "filter": {}})
        assert client.delete(f"/api/v1/ontology/graphs/{ns}").status_code == 200
        # 같은 이름으로 다시 빌드 → 저장된 뷰가 유령으로 남지 않는다
        build_namespace(client, ns, save=True)
        try:
            assert client.get(
                f"/api/v1/ontology/graphs/{ns}/views").json()["views"] == []
        finally:
            delete_ns(client, ns)

    def test_delete_missing_view_404(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            assert client.delete(
                f"/api/v1/ontology/graphs/{ns}/views/sv_nope").status_code == 404
        finally:
            delete_ns(client, ns)

    def test_views_unknown_namespace_404(self, client):
        assert client.get(
            f"/api/v1/ontology/graphs/{unique_ns()}/views").status_code == 404


# ─── 12. LLM 등록·설정 (env-only 키 — 키는 저장하지 않는다) ──────────

class TestLLMConfig:
    def test_get_config_shape_with_defaults(self, client):
        body = client.get("/api/v1/ontology/admin/llm").json()
        assert body["provider"] in (
            "google", "openai", "anthropic", "openai_compatible")
        assert "model" in body and "base_url" in body
        assert "key_present" in body and "providers" in body
        assert "google" in body["providers"]

    def test_set_config_persists(self, client):
        r = client.put("/api/v1/ontology/admin/llm", json={
            "provider": "google", "model": "gemini-3.5-flash"})
        assert r.status_code == 200, r.text
        got = client.get("/api/v1/ontology/admin/llm").json()
        assert got["provider"] == "google"
        assert got["model"] == "gemini-3.5-flash"

    def test_invalid_provider_and_missing_model_400(self, client):
        assert client.put("/api/v1/ontology/admin/llm", json={
            "provider": "bogus", "model": "x"}).status_code == 400
        assert client.put("/api/v1/ontology/admin/llm", json={
            "provider": "google", "model": "  "}).status_code == 400
        # openai_compatible 은 base_url 필수
        assert client.put("/api/v1/ontology/admin/llm", json={
            "provider": "openai_compatible", "model": "local"}).status_code == 400

    def test_key_presence_masked_never_raw(self, client, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "supersecretkey_abcd1234")
        client.put("/api/v1/ontology/admin/llm", json={
            "provider": "google", "model": "gemini-3.5-flash"})
        res = client.get("/api/v1/ontology/admin/llm")
        body = res.json()
        assert body["key_present"] is True
        # 힌트는 끝 4자만, 원문 키는 응답 어디에도 없어야 한다
        assert body["key_hint"].endswith("1234")
        assert "supersecretkey" not in res.text

    def test_test_connection_uses_active_llm(self, client):
        # fixture 의 fake_llm 이 effective LLM — 실제 API 호출 없이 응답 확인
        res = client.post("/api/v1/ontology/admin/llm/test")
        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert "sample" in body


# ─── 13. 프로젝트 (네임스페이스 = 프로젝트, 빈 프로젝트 생성) ────────

class TestProjects:
    def test_create_project_shows_empty_in_overview_and_stats(self, client):
        name = unique_ns()
        r = client.post("/api/v1/ontology/admin/projects", json={
            "name": name, "description": "내 프로젝트", "domain": "heritage"})
        assert r.status_code == 200, r.text
        try:
            ov = client.get("/api/v1/ontology/admin/overview").json()
            entry = {e["namespace"]: e for e in ov["namespaces"]}[name]
            assert entry["nodes"] == 0
            assert entry.get("empty") is True
            assert entry["description"] == "내 프로젝트"
            assert entry["domain"] == "heritage"
            # 빈 프로젝트도 stats 가 열린다(404 아님) → 온보딩 화면 가능
            st = client.get(f"/api/v1/ontology/graphs/{name}/stats")
            assert st.status_code == 200
            assert st.json()["nodes"] == 0
        finally:
            client.delete(f"/api/v1/ontology/graphs/{name}")

    def test_create_duplicate_invalid_protected(self, client):
        name = unique_ns()
        assert client.post("/api/v1/ontology/admin/projects",
                           json={"name": name}).status_code == 200
        try:
            assert client.post("/api/v1/ontology/admin/projects",
                               json={"name": name}).status_code == 409
        finally:
            client.delete(f"/api/v1/ontology/graphs/{name}")
        # protected · 형식 오류
        assert client.post("/api/v1/ontology/admin/projects",
                           json={"name": "default"}).status_code in (400, 409)
        assert client.post("/api/v1/ontology/admin/projects",
                           json={"name": "bad name!"}).status_code == 400
        assert client.post("/api/v1/ontology/admin/projects",
                           json={"name": "   "}).status_code == 400

    def test_create_over_existing_graph_409(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            assert client.post("/api/v1/ontology/admin/projects",
                               json={"name": ns}).status_code == 409
        finally:
            delete_ns(client, ns)

    def test_delete_removes_project_record(self, client):
        name = unique_ns()
        client.post("/api/v1/ontology/admin/projects", json={"name": name})
        assert client.delete(
            f"/api/v1/ontology/graphs/{name}").status_code == 200
        ov = client.get("/api/v1/ontology/admin/overview").json()
        assert name not in [e["namespace"] for e in ov["namespaces"]]


# ─── 14. 스키마 편집 (타입 개명 · 술어/타입 선언) ───────────────────

class TestSchemaEditing:
    def test_rename_type_updates_all_nodes(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            r = client.post(f"/api/v1/ontology/graphs/{ns}/schema/rename-type",
                            json={"old": "Clause", "new": "Article"})
            assert r.status_code == 200, r.text
            assert r.json()["renamed"] >= 1
            sch = client.get(f"/api/v1/ontology/graphs/{ns}/schema").json()
            types = [c["type"] for c in sch["classes"]]
            assert "Article" in types and "Clause" not in types
            # 감사 로그에 rename_type 남는다
            hist = client.get(
                f"/api/v1/ontology/graphs/{ns}/review/history").json()["history"]
            assert any(h["action"] == "rename_type" for h in hist)
        finally:
            delete_ns(client, ns)

    def test_rename_missing_type_404(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            r = client.post(f"/api/v1/ontology/graphs/{ns}/schema/rename-type",
                            json={"old": "NoSuchType", "new": "X"})
            assert r.status_code == 404
        finally:
            delete_ns(client, ns)

    def test_predicate_decl_persists_and_shows_in_schema(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            r = client.put(f"/api/v1/ontology/graphs/{ns}/schema/predicate",
                           json={"predicate": "citesRegulation",
                                 "domain": "Clause", "range": "Regulation",
                                 "description": "조문이 인용하는 규정"})
            assert r.status_code == 200
            sch = client.get(f"/api/v1/ontology/graphs/{ns}/schema").json()
            p = {x["predicate"]: x for x in sch["predicates"]}["citesRegulation"]
            assert p["declared"]["domain"] == "Clause"
            assert p["declared"]["range"] == "Regulation"
        finally:
            delete_ns(client, ns)

    def test_type_decl_deprecated_shows(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            client.put(f"/api/v1/ontology/graphs/{ns}/schema/type",
                       json={"type": "Coverage", "description": "보장 항목",
                             "deprecated": True})
            sch = client.get(f"/api/v1/ontology/graphs/{ns}/schema").json()
            c = {x["type"]: x for x in sch["classes"]}["Coverage"]
            assert c["declared"]["deprecated"] is True
            assert c["declared"]["description"] == "보장 항목"
        finally:
            delete_ns(client, ns)

    def test_schema_writes_protected_403(self, client):
        assert client.post(
            "/api/v1/ontology/graphs/default/schema/rename-type",
            json={"old": "a", "new": "b"}).status_code == 403
        assert client.put(
            "/api/v1/ontology/graphs/default/schema/predicate",
            json={"predicate": "p"}).status_code == 403


# ─── 15. 이웃 서브그래프 (앵커→이웃 확장 3D 기반 — 전체 로드 안 함) ──

class TestNeighbors:
    def test_neighbors_returns_anchor_ego_subgraph(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            r = client.get(f"/api/v1/ontology/graphs/{ns}/neighbors"
                           f"?node_id=Clause:청약철회")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["anchor"] == "Clause:청약철회"
            ids = {n["id"] for n in body["nodes"]}
            # 앵커 + 인용 규정 이웃만 (전체 그래프가 아니다)
            assert "Clause:청약철회" in ids
            assert "Regulation:금융소비자 보호에 관한 법률" in ids
            assert "Coverage:암진단비" not in ids   # 연결 없는 노드는 안 온다
            assert any(l["predicate"] == "citesRegulation"
                       for l in body["links"])
            assert body["truncated"] is False
            # 노드마다 degree 동봉 (어느 이웃이 더 펼칠 게 있나)
            assert all("degree" in n for n in body["nodes"])
        finally:
            delete_ns(client, ns)

    def test_neighbors_limit_truncates(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            # 허브 하나에 이웃 3개 연결
            client.post(f"/api/v1/ontology/graphs/{ns}/nodes",
                        json={"node_type": "Hub", "name": "허브"})
            for i in range(3):
                client.post(f"/api/v1/ontology/graphs/{ns}/nodes",
                            json={"node_type": "Leaf", "name": f"잎{i}"})
                client.post(f"/api/v1/ontology/graphs/{ns}/edges",
                            json={"source": "Hub:허브", "predicate": "has",
                                  "target": f"Leaf:잎{i}"})
            body = client.get(f"/api/v1/ontology/graphs/{ns}/neighbors"
                              f"?node_id=Hub:허브&limit=2").json()
            assert body["total_neighbors"] == 3
            assert body["truncated"] is True
            # 앵커 + 이웃 2개 = 3 노드만
            assert len(body["nodes"]) == 3
        finally:
            delete_ns(client, ns)

    def test_neighbors_missing_node_404(self, client):
        ns = build_namespace(client, unique_ns())
        try:
            assert client.get(f"/api/v1/ontology/graphs/{ns}/neighbors"
                              f"?node_id=no:such").status_code == 404
        finally:
            delete_ns(client, ns)

    def test_neighbors_unknown_namespace_404(self, client):
        assert client.get(
            f"/api/v1/ontology/graphs/{unique_ns()}/neighbors"
            f"?node_id=x").status_code == 404
