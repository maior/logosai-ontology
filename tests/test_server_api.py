"""
Ontology Builder 독립 서버(9274) API tests — upload / build job / graph / search.

Minimal FastAPI app + TestClient with a fake LLM and tmp data dir injected
via dependency_overrides. BackgroundTasks complete before TestClient
returns, so job polling is deterministic.
"""

import json
import uuid
import zlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    if "커스텀개념" in prompt:
        return json.dumps({
            "entities": [{"name": "커스텀개념", "type": "MyType", "attrs": {}}],
            "relations": [],
        }, ensure_ascii=False)
    return json.dumps({"entities": [], "relations": []})


def fake_embed(texts):
    dim = 64
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in str(text).lower().split():
            vectors[i, zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    return vectors


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
def default_checkpoint(tmp_path, monkeypatch):
    """'default'(에이전트 그래프)를 이 테스트가 직접 만든다.

    list_namespaces 는 서비스 data_dir 이 아니라 **전역** 체크포인트 폴더
    (knowledge_graph_clean._DEFAULT_DATA_DIR)와 프로세스 전역 엔진 목록(_kg_instances)에서
    'default' 를 찾는다. conftest 가 그 폴더를 빈 세션 tmp 로 돌리므로, 전엔 앞서 돈 다른
    테스트가 'default' 엔진을 우연히 만들어 둔 경우에만 통과했다 — 이 파일을 단독으로 돌리면
    실패했다(2026-10-05). 앞 테스트에 기대지 않도록 폴더와 엔진 목록을 이 테스트 것으로 바꾼다.
    """
    from ontology.engines import knowledge_graph_clean as kgc
    root = tmp_path / "kg_global"
    root.mkdir()
    (root / "kg_checkpoint.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(kgc, "_DEFAULT_DATA_DIR", root)
    monkeypatch.setattr(kgc, "_kg_instances", {})
    return root


def unique_ns():
    return f"testns_{uuid.uuid4().hex[:8]}"


def upload_sample(client) -> str:
    response = client.post(
        "/api/v1/ontology/datasets",
        files=[
            ("files", ("terms.md", SAMPLE_MD.encode("utf-8"), "text/markdown")),
            ("files", ("note.txt", "암진단비 관련 메모".encode("utf-8"), "text/plain")),
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()["dataset_id"]


def build(client, dataset_id, namespace, **options):
    payload = {"dataset_id": dataset_id, "namespace": namespace,
               "save": False, **options}
    return client.post("/api/v1/ontology/build", json=payload)


# ─── 1. Dataset upload ──────────────────────────────────────────────

class TestDatasets:
    def test_upload_returns_dataset_with_files(self, client):
        response = client.post(
            "/api/v1/ontology/datasets",
            files=[("files", ("a.md", b"# doc", "text/markdown"))],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["dataset_id"]
        assert body["files"] == ["a.md"]
        assert body["total_bytes"] > 0

    def test_upload_without_files_rejected(self, client):
        response = client.post("/api/v1/ontology/datasets", files=[])
        assert response.status_code in (400, 422)

    def test_list_datasets_shows_uploaded(self, client):
        dataset_id = upload_sample(client)
        response = client.get("/api/v1/ontology/datasets")
        assert response.status_code == 200
        ids = [d["dataset_id"] for d in response.json()["datasets"]]
        assert dataset_id in ids


# ─── 2. Build job ───────────────────────────────────────────────────

class TestBuildJob:
    def test_build_unknown_dataset_404(self, client):
        response = build(client, "no_such_dataset", unique_ns())
        assert response.status_code == 404

    def test_build_into_default_namespace_rejected(self, client):
        # 'default'는 에이전트 라우팅 그래프 — 문서 빌드로 오염 금지
        dataset_id = upload_sample(client)
        response = build(client, dataset_id, "default")
        assert response.status_code == 400

    def test_build_happy_path_completes_with_report(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        response = build(client, dataset_id, namespace)
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        job = client.get(f"/api/v1/ontology/jobs/{job_id}").json()
        assert job["status"] == "completed"
        report = job["report"]
        assert report["files_read"] == 2
        assert report["entities_added"] >= 3
        assert report["relations_added"] >= 1
        assert report["namespace"] == namespace

    def test_job_progress_recorded(self, client):
        dataset_id = upload_sample(client)
        response = build(client, dataset_id, unique_ns())
        job = client.get(f"/api/v1/ontology/jobs/{response.json()['job_id']}").json()
        assert job["progress"].get("stage")  # 마지막 진행 단계가 남아 있음

    def test_unknown_job_404(self, client):
        assert client.get("/api/v1/ontology/jobs/nope").status_code == 404

    def test_custom_schema_mode(self, client):
        response = client.post(
            "/api/v1/ontology/datasets",
            files=[("files", ("c.txt", "커스텀개념에 대한 문서".encode("utf-8"), "text/plain"))],
        )
        dataset_id = response.json()["dataset_id"]
        namespace = unique_ns()
        response = build(client, dataset_id, namespace,
                         schema_mode="custom",
                         custom_schema={"node_types": ["MyType"],
                                        "predicates": {"rel": ["MyType", "MyType"]}})
        assert response.status_code == 200
        job = client.get(f"/api/v1/ontology/jobs/{response.json()['job_id']}").json()
        assert job["status"] == "completed"

        graph = client.get(f"/api/v1/ontology/graphs/{namespace}").json()
        assert graph["node_types"].get("MyType", 0) >= 1

    def test_custom_mode_without_schema_rejected(self, client):
        dataset_id = upload_sample(client)
        response = build(client, dataset_id, unique_ns(), schema_mode="custom")
        assert response.status_code == 400


# ─── 3. Graph query ─────────────────────────────────────────────────

class TestGraphQuery:
    def test_graph_stats_after_build(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        response = client.get(f"/api/v1/ontology/graphs/{namespace}")
        assert response.status_code == 200
        body = response.json()
        assert body["namespace"] == namespace
        assert body["nodes"] >= 3
        assert body["edges"] >= 1
        assert body["node_types"].get("Clause", 0) >= 1
        assert "citesRegulation" in body["predicates"]

    def test_graph_sample_nodes_included(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        response = client.get(f"/api/v1/ontology/graphs/{namespace}?limit=2")
        body = response.json()
        assert len(body["sample_nodes"]) == 2
        node = body["sample_nodes"][0]
        assert "id" in node and "type" in node

    def test_graph_data_for_visualization(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        response = client.get(f"/api/v1/ontology/graphs/{namespace}/data")
        assert response.status_code == 200
        body = response.json()
        assert body["nodes"] and body["links"]
        assert {"source", "target", "predicate"} <= set(body["links"][0].keys())


# ─── 4. Semantic search ─────────────────────────────────────────────

class TestSemanticSearch:
    def test_search_returns_relevant_nodes(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        # 테스트용 가짜 임베더를 해당 네임스페이스 엔진에 주입
        from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
        get_knowledge_graph_engine(namespace).init_semantic_index(embed_fn=fake_embed)

        response = client.post(
            f"/api/v1/ontology/graphs/{namespace}/search",
            json={"query": "암진단비 보장", "top_k": 3},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert results
        assert any("암진단비" in r["node_id"] for r in results)

    def test_search_empty_query_rejected(self, client):
        response = client.post(
            f"/api/v1/ontology/graphs/{unique_ns()}/search",
            json={"query": "   "},
        )
        assert response.status_code in (400, 422)


# ─── 5. Build options API ───────────────────────────────────────────

class TestBuildOptionsAPI:
    def test_schemas_endpoint_lists_presets(self, client):
        response = client.get("/api/v1/ontology/schemas")
        assert response.status_code == 200
        presets = response.json()["presets"]
        assert "document" in presets and "generic" in presets
        assert presets["document"]["node_types"]

    def test_build_accepts_segment_and_model_options(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        response = build(client, dataset_id, namespace,
                         segment_mode="window", overlap=50,
                         llm_model="gemini-3.5-flash")
        assert response.status_code == 200
        job = client.get(f"/api/v1/ontology/jobs/{response.json()['job_id']}").json()
        assert job["status"] == "completed"

    def test_generic_preset_via_schema_mode(self, client):
        dataset_id = upload_sample(client)
        response = build(client, dataset_id, unique_ns(), schema_mode="generic")
        assert response.status_code == 200
        job = client.get(f"/api/v1/ontology/jobs/{response.json()['job_id']}").json()
        assert job["status"] == "completed"

    def test_invalid_schema_mode_rejected(self, client):
        dataset_id = upload_sample(client)
        response = build(client, dataset_id, unique_ns(), schema_mode="nope")
        assert response.status_code in (400, 422)

    def test_rebuild_clears_previous_namespace_graph(self, client):
        namespace = unique_ns()
        first = upload_sample(client)
        build(client, first, namespace)
        before = client.get(f"/api/v1/ontology/graphs/{namespace}").json()
        assert before["nodes"] >= 3

        # 전혀 다른 데이터로 rebuild → 이전 노드는 사라져야 함
        response = client.post(
            "/api/v1/ontology/datasets",
            files=[("files", ("other.txt", "커스텀개념에 대한 문서".encode("utf-8"), "text/plain"))],
        )
        second = response.json()["dataset_id"]
        build(client, second, namespace, rebuild=True,
              schema_mode="custom",
              custom_schema={"node_types": ["MyType"], "predicates": {}})
        after = client.get(f"/api/v1/ontology/graphs/{namespace}").json()
        assert "Clause" not in after["node_types"]          # 이전 그래프 제거됨
        assert after["node_types"].get("MyType", 0) >= 1    # 새 그래프만 존재


# ─── 6. Export · Map · Provider API ────────────────────────────────

def geo_llm(prompt: str) -> str:
    if "스키마를 제안" in prompt:
        return json.dumps({"node_types": ["HeritageSite", "Region"],
                           "predicates": {"locatedIn": ["HeritageSite", "Region"]}},
                          ensure_ascii=False)
    if "석굴암" in prompt:
        return json.dumps({
            "entities": [
                {"name": "석굴암", "type": "HeritageSite",
                 "attrs": {"lat": 35.795, "lng": 129.349, "category": "문화", "region": "경주"}},
                {"name": "성산일출봉", "type": "HeritageSite",
                 "attrs": {"lat": 33.458, "lng": 126.942, "category": "자연", "region": "제주"}},
            ],
            "relations": [],
        }, ensure_ascii=False)
    return json.dumps({"entities": [], "relations": []})


@pytest.fixture()
def geo_client(tmp_path):
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService
    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=geo_llm)
    app = FastAPI()
    app.include_router(server_router.router, prefix="/api/v1/ontology")
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app)


class TestExportMapAPI:
    def _build_geo(self, client):
        res = client.post("/api/v1/ontology/datasets",
                          files=[("files", ("h.txt", "석굴암과 성산일출봉".encode("utf-8"), "text/plain"))])
        dataset_id = res.json()["dataset_id"]
        namespace = unique_ns()
        res = client.post("/api/v1/ontology/build", json={
            "dataset_id": dataset_id, "namespace": namespace,
            "schema_mode": "auto", "save": False})
        assert res.status_code == 200, res.text
        job = client.get(f"/api/v1/ontology/jobs/{res.json()['job_id']}").json()
        assert job["status"] == "completed", job
        return namespace, job

    def test_auto_schema_recorded_in_job(self, geo_client):
        _, job = self._build_geo(geo_client)
        assert job["report"]["proposed_schema"]["node_types"]

    def test_map_endpoint_returns_geo_and_categories(self, geo_client):
        namespace, _ = self._build_geo(geo_client)
        res = geo_client.get(f"/api/v1/ontology/graphs/{namespace}/map")
        assert res.status_code == 200
        body = res.json()
        assert len(body["geo"]) == 2
        point = body["geo"][0]
        assert {"id", "name", "lat", "lng"} <= set(point.keys())
        assert body["categories"].get("문화") and body["categories"].get("자연")

    def test_map_endpoint_empty_namespace(self, geo_client):
        res = geo_client.get(f"/api/v1/ontology/graphs/{unique_ns()}/map")
        assert res.status_code == 200
        assert res.json()["geo"] == []

    def test_export_turtle(self, geo_client):
        namespace, _ = self._build_geo(geo_client)
        res = geo_client.get(f"/api/v1/ontology/graphs/{namespace}/export?format=turtle")
        assert res.status_code == 200
        assert "owl:Class" in res.text
        assert "text/turtle" in res.headers["content-type"]

    def test_export_json(self, geo_client):
        namespace, _ = self._build_geo(geo_client)
        res = geo_client.get(f"/api/v1/ontology/graphs/{namespace}/export?format=json")
        assert res.status_code == 200
        assert "nodes" in res.json()

    def test_export_unknown_format_rejected(self, geo_client):
        res = geo_client.get(f"/api/v1/ontology/graphs/{unique_ns()}/export?format=xml")
        assert res.status_code in (400, 422)

    def test_build_accepts_provider_options(self, geo_client):
        res = geo_client.post("/api/v1/ontology/datasets",
                              files=[("files", ("a.txt", b"text", "text/plain"))])
        dataset_id = res.json()["dataset_id"]
        res = geo_client.post("/api/v1/ontology/build", json={
            "dataset_id": dataset_id, "namespace": unique_ns(), "save": False,
            "llm_provider": "anthropic", "llm_model": "claude-haiku-4-5-20251001"})
        assert res.status_code == 200
        job = geo_client.get(f"/api/v1/ontology/jobs/{res.json()['job_id']}").json()
        assert job["status"] == "completed"  # llm_fn 주입이 우선하므로 실호출 없음

    def test_invalid_provider_rejected(self, geo_client):
        res = geo_client.post("/api/v1/ontology/datasets",
                              files=[("files", ("a.txt", b"text", "text/plain"))])
        res = geo_client.post("/api/v1/ontology/build", json={
            "dataset_id": res.json()["dataset_id"], "namespace": unique_ns(),
            "llm_provider": "midjourney"})
        assert res.status_code in (400, 422)


# ─── 7. Namespaces · Node detail ────────────────────────────────────

class TestNamespacesAndNodeDetail:
    def test_list_namespaces_includes_built(self, client, default_checkpoint):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        res = client.get("/api/v1/ontology/namespaces")
        assert res.status_code == 200
        names = [n["namespace"] for n in res.json()["namespaces"]]
        assert namespace in names          # 방금 빌드한 ns (로드 인스턴스)
        assert "default" in names          # 에이전트 그래프 체크포인트도 목록에 포함

    def test_node_detail_returns_attrs_and_neighbors(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        res = client.get(f"/api/v1/ontology/graphs/{namespace}/node",
                         params={"id": "Clause:청약철회"})
        assert res.status_code == 200
        body = res.json()
        assert body["id"] == "Clause:청약철회"
        assert body["attrs"]["type"] == "Clause"
        assert body["attrs"]["source"]                    # provenance 포함
        out = body["out_edges"]
        assert any(e["predicate"] == "citesRegulation" and
                   "금융소비자" in e["target_name"] for e in out)

    def test_node_detail_in_edges(self, client):
        dataset_id = upload_sample(client)
        namespace = unique_ns()
        build(client, dataset_id, namespace)

        res = client.get(f"/api/v1/ontology/graphs/{namespace}/node",
                         params={"id": "Regulation:금융소비자 보호에 관한 법률"})
        assert res.status_code == 200
        assert any(e["predicate"] == "citesRegulation"
                   for e in res.json()["in_edges"])

    def test_node_detail_unknown_404(self, client):
        res = client.get(f"/api/v1/ontology/graphs/{unique_ns()}/node",
                         params={"id": "없는노드"})
        assert res.status_code == 404


# ─── 8. AI 학습 데이터 추출 (koract multi-condition 이식) ──────────

def heritage_llm(prompt: str) -> str:
    if "스키마를 제안" in prompt:
        return json.dumps({"node_types": ["Heritage", "Designation"],
                           "predicates": {"hasDesignation": ["Heritage", "Designation"]}},
                          ensure_ascii=False)
    if "숭례문" in prompt:
        return json.dumps({
            "entities": [
                {"name": "숭례문", "type": "Heritage",
                 "attrs": {"definition": "조선 도성의 남쪽 정문",
                           "aliases": ["남대문"]}},
                {"name": "국보", "type": "Designation", "attrs": {}},
            ],
            "relations": [
                {"subject": "숭례문", "predicate": "hasDesignation", "object": "국보"},
            ],
        }, ensure_ascii=False)
    return json.dumps({"entities": [], "relations": []})


@pytest.fixture()
def dataset_client(tmp_path):
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService
    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=heritage_llm)
    app = FastAPI()
    app.include_router(server_router.router, prefix="/api/v1/ontology")
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app)


class TestTrainingDataset:
    def _build(self, client):
        res = client.post("/api/v1/ontology/datasets",
                          files=[("files", ("h.txt", "숭례문은 국보다".encode("utf-8"), "text/plain"))])
        namespace = unique_ns()
        res = client.post("/api/v1/ontology/build", json={
            "dataset_id": res.json()["dataset_id"], "namespace": namespace,
            "schema_mode": "auto", "save": False})
        job = client.get(f"/api/v1/ontology/jobs/{res.json()['job_id']}").json()
        assert job["status"] == "completed", job
        return namespace

    def test_triples_format_with_provenance(self, dataset_client):
        namespace = self._build(dataset_client)
        res = dataset_client.post(f"/api/v1/ontology/graphs/{namespace}/dataset",
                                  json={"formats": ["triples"]})
        assert res.status_code == 200
        rows = res.json()["rows"]
        triple = next(r for r in rows if r["format"] == "triple")
        assert triple["subject"] == "숭례문"
        assert triple["predicate"] == "hasDesignation"
        assert triple["object"] == "국보"
        assert triple["source"]                       # 출처 필수 (환각 0)

    def test_qa_format_from_definition_and_edges(self, dataset_client):
        namespace = self._build(dataset_client)
        res = dataset_client.post(f"/api/v1/ontology/graphs/{namespace}/dataset",
                                  json={"formats": ["qa"]})
        rows = res.json()["rows"]
        # 정의 기반 QA
        assert any("숭례문" in r["instruction"] and r["output"] == "조선 도성의 남쪽 정문"
                   for r in rows)
        # 관계 기반 QA
        assert any("hasDesignation" in r["instruction"] and r["output"] == "국보"
                   for r in rows)

    def test_surface_format_alias_pairs(self, dataset_client):
        namespace = self._build(dataset_client)
        res = dataset_client.post(f"/api/v1/ontology/graphs/{namespace}/dataset",
                                  json={"formats": ["surface"]})
        rows = res.json()["rows"]
        assert any(r["input"] == "남대문" and r["output"] == "숭례문" for r in rows)

    def test_predicate_filter_and_counts(self, dataset_client):
        namespace = self._build(dataset_client)
        res = dataset_client.post(f"/api/v1/ontology/graphs/{namespace}/dataset",
                                  json={"formats": ["triples", "qa", "surface"],
                                        "predicates": ["no_such_predicate"]})
        body = res.json()
        assert body["counts"]["triple"] == 0          # 관계 필터 적용
        assert body["counts"]["surface"] >= 1         # surface는 관계 무관

    def test_jsonl_download(self, dataset_client):
        namespace = self._build(dataset_client)
        res = dataset_client.get(
            f"/api/v1/ontology/graphs/{namespace}/dataset.jsonl?formats=triples,qa,surface")
        assert res.status_code == 200
        assert "jsonl" in res.headers.get("content-disposition", "")
        lines = [l for l in res.text.strip().split("\n") if l]
        assert len(lines) >= 3
        json.loads(lines[0])                          # 각 줄이 유효한 JSON

    def test_unknown_format_rejected(self, dataset_client):
        namespace = self._build(dataset_client)
        res = dataset_client.post(f"/api/v1/ontology/graphs/{namespace}/dataset",
                                  json={"formats": ["bogus"]})
        assert res.status_code in (400, 422)


# ─── 9. 정형 레코드 인제스트 API ────────────────────────────────────

class TestRecordsAPI:
    def test_records_ingest_endpoint(self, client):
        namespace = unique_ns()
        res = client.post(f"/api/v1/ontology/graphs/{namespace}/records", json={
            "records": [{"이름": "숭례문", "지정": "국보", "lat": 37.56, "lng": 126.97}],
            "mapping": {"node_type": "HeritageSite", "name_field": "이름",
                        "relations": [{"predicate": "hasDesignation",
                                       "target_type": "Designation", "field": "지정"}]},
            "source": "wikidata", "save": False,
        })
        assert res.status_code == 200, res.text
        report = res.json()
        assert report["entities_added"] == 2 and report["relations_added"] == 1

        graph = client.get(f"/api/v1/ontology/graphs/{namespace}").json()
        assert graph["node_types"].get("HeritageSite") == 1

    def test_records_protected_namespace_rejected(self, client):
        res = client.post("/api/v1/ontology/graphs/default/records", json={
            "records": [{"이름": "x"}],
            "mapping": {"node_type": "T", "name_field": "이름"},
        })
        assert res.status_code == 400


# ─── 10. 계층 롤업 질의 (온톨로지 추론 활용) ────────────────────────

class TestHierarchyRollup:
    def _seed(self, client):
        ns = unique_ns()
        # 클래스 계층: 삼층석탑 is_a 석탑 is_a 탑
        client.post(f"/api/v1/ontology/graphs/{ns}/records", json={
            "records": [{"이름": "삼층석탑", "상위": "석탑"}, {"이름": "석탑", "상위": "탑"}],
            "mapping": {"node_type": "HeritageClass", "name_field": "이름",
                        "relations": [{"predicate": "is_a",
                                       "target_type": "HeritageClass", "field": "상위"}]},
            "save": False})
        # 인스턴스: 삼층석탑 2개 + 석탑(직접) 1개
        client.post(f"/api/v1/ontology/graphs/{ns}/records", json={
            "records": [
                {"이름": "불국사 삼층석탑", "분류": "삼층석탑"},
                {"이름": "고선사지 삼층석탑", "분류": "삼층석탑"},
                {"이름": "정림사지 오층석탑", "분류": "석탑"},
            ],
            "mapping": {"node_type": "문화유산", "name_field": "이름",
                        "relations": [{"predicate": "classifiedAs",
                                       "target_type": "HeritageClass", "field": "분류"}]},
            "save": False})
        return ns

    def test_rollup_aggregates_descendant_instances(self, client):
        ns = self._seed(client)
        res = client.get(f"/api/v1/ontology/graphs/{ns}/rollup",
                         params={"class_id": "HeritageClass:석탑"})
        assert res.status_code == 200
        body = res.json()
        assert body["ancestors"] == ["HeritageClass:탑"]              # 상위 사슬
        assert "HeritageClass:삼층석탑" in body["descendants"]         # 하위 분류
        assert body["instances_direct"] == 1                          # 석탑 직접
        assert body["instances_total"] == 3                           # 하위 포함 롤업
        names = [i["name"] for i in body["instances"]]
        assert "불국사 삼층석탑" in names and "정림사지 오층석탑" in names

    def test_rollup_unknown_class_404(self, client):
        res = client.get(f"/api/v1/ontology/graphs/{unique_ns()}/rollup",
                         params={"class_id": "HeritageClass:없음"})
        assert res.status_code == 404


# ─── 11. 네임스페이스 목록에서 시스템 그래프 제외 ───────────────────

class TestNamespaceListingFilters:
    def test_default_included_in_listing(self, client, default_checkpoint):
        # default(에이전트 그래프)도 목록에 노출 — 사용자가 열람 가능
        dataset_id = upload_sample(client)
        build(client, dataset_id, unique_ns())
        names = [n["namespace"] for n in
                 client.get("/api/v1/ontology/namespaces").json()["namespaces"]]
        assert "default" in names

    def test_default_absent_without_checkpoint(self, client, tmp_path, monkeypatch):
        """대조군 — 체크포인트도 'default' 엔진도 없으면 목록에 없다. 위 두 테스트가
        다른 테스트의 흔적이 아니라 자기가 만든 체크포인트 덕에 통과한다는 증거."""
        from ontology.engines import knowledge_graph_clean as kgc
        empty = tmp_path / "kg_empty"
        empty.mkdir()
        monkeypatch.setattr(kgc, "_DEFAULT_DATA_DIR", empty)
        monkeypatch.setattr(kgc, "_kg_instances", {})
        ns = unique_ns()
        build(client, upload_sample(client), ns)
        names = [n["namespace"] for n in
                 client.get("/api/v1/ontology/namespaces").json()["namespaces"]]
        assert ns in names and "default" not in names

    def test_builder_namespaces_still_listed(self, client):
        ns = unique_ns()
        build(client, upload_sample(client), ns)
        names = [n["namespace"] for n in
                 client.get("/api/v1/ontology/namespaces").json()["namespaces"]]
        assert ns in names  # 빌더가 만든 네임스페이스는 그대로 노출

    def test_default_graph_still_accessible_directly(self, client):
        # 목록에서만 뺄 뿐, 직접 조회는 여전히 가능해야 함
        res = client.get("/api/v1/ontology/graphs/default")
        assert res.status_code == 200
