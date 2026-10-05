"""
검수 루프(review loop) — 거절은 묘비(tombstone)다.

KorAct(gov/desktopgui)가 실증한 필요다: LLM 추출은 틀린다. 검수자가 거절한
노드를 그래프에서 지우는 것만으로는 부족하다 — 같은 문서를 다시 인제스트하면
**같은 오추출이 부활한다**. 거절은 재빌드를 살아남아야 한다.

고정하는 계약:
1. **묘비는 부활을 막는다** — is_rejected 인 노드는 재빌드에서 다시 추가되지
   않고, 그 노드를 참조하는 관계도 함께 버려진다. 다른 개체의 병합은
   깨지지 않는다 (한 개체의 거절이 빌드를 죽이면 안 된다).
2. **나중 판정이 이긴다** — confirm 이 이전 reject 를 뒤집는다 (검수자의
   번복은 정상 워크플로우다).
3. **로그가 원본이다** — 상태(묘비/확정)는 append-only 이벤트 로그를
   재생(replay)해서 복원된다 (KorAct ontology_versions 패턴). 감사 항목은
   절대 다시 쓰이지 않는다.
4. reject 는 그래프에서 노드+간선을 제거하고, 그 제거가 재인제스트를
   살아남는다.
"""

import asyncio
import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.builder import BuilderSchema, OntologyBuilder
from ontology.core.chunk_store import ChunkStore, reset_chunk_stores
from ontology.core.review_store import (
    ReviewStore,
    get_review_store,
    reset_review_stores,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_singletons():
    reset_chunk_stores()
    reset_review_stores()
    yield
    reset_chunk_stores()
    reset_review_stores()


SAMPLE_MD = """## 제1조 청약철회
계약자는 15일 이내에 청약을 철회할 수 있습니다. 금융소비자 보호에 관한 법률을 따릅니다.

## 제2조 보장내용
암진단비는 최초 1회에 한하여 지급합니다.
"""

SCHEMA = BuilderSchema(
    node_types=["Clause", "Coverage", "Regulation"],
    predicates={"citesRegulation": ("Clause", "Regulation")})


def fake_llm(prompt: str) -> str:
    """결정적 가짜 추출기 — 프롬프트에 든 청크 본문으로 분기한다."""
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


# ─── 1. ReviewStore — 묘비 + 판정 번복 ───────────────────────────────

class TestReviewStoreBasics:
    @pytest.fixture
    def store(self, tmp_path):
        return ReviewStore(namespace="rv", path=tmp_path / "reviews_rv.jsonl")

    def test_reject_creates_tombstone(self, store):
        store.reject("Clause:오추출", reason="원문에 없는 개체", actor="ken")
        assert store.is_rejected("Clause:오추출") is True
        assert store.is_confirmed("Clause:오추출") is False

    def test_unreviewed_node_is_neither(self, store):
        assert store.is_rejected("Clause:미검수") is False
        assert store.is_confirmed("Clause:미검수") is False

    def test_confirm_overrides_earlier_reject(self, store):
        """검수자의 번복은 정상 워크플로우다 — 나중 판정이 이긴다.
        confirm 이 묘비를 걷어내지 못하면 잘못 거절한 노드를 영영 못 살린다."""
        store.reject("Clause:A", reason="실수")
        store.confirm("Clause:A", actor="ken")
        assert store.is_rejected("Clause:A") is False
        assert store.is_confirmed("Clause:A") is True

    def test_reject_overrides_earlier_confirm(self, store):
        store.confirm("Clause:A")
        store.reject("Clause:A", reason="재검토 결과 오추출")
        assert store.is_rejected("Clause:A") is True
        assert store.is_confirmed("Clause:A") is False

    def test_rejected_lists_only_active_tombstones(self, store):
        store.reject("Clause:A")
        store.reject("Clause:B")
        store.confirm("Clause:B")  # 번복 — 묘비 목록에서 빠져야 한다
        ids = [t["node_id"] for t in store.rejected()]
        assert ids == ["Clause:A"]


class TestAuditHistory:
    @pytest.fixture
    def store(self, tmp_path):
        return ReviewStore(namespace="rv", path=tmp_path / "reviews_rv.jsonl")

    def test_history_is_newest_first(self, store):
        """검수 UI 는 최근 판정부터 본다. 같은 초 안의 이벤트는 타임스탬프로
        구별되지 않으므로 정렬이 아니라 **로그 순서**가 근거여야 한다."""
        store.reject("Clause:A")
        store.confirm("Clause:B")
        store.reject("Clause:C")
        actions = [(e["action"], e["node_id"]) for e in store.history()]
        assert actions == [("reject", "Clause:C"), ("confirm", "Clause:B"),
                           ("reject", "Clause:A")]

    def test_history_filters_by_node_id(self, store):
        store.reject("Clause:A", reason="첫 거절")
        store.confirm("Clause:B")
        store.confirm("Clause:A")
        entries = store.history(node_id="Clause:A")
        assert len(entries) == 2
        assert all(e["node_id"] == "Clause:A" for e in entries)
        assert entries[0]["action"] == "confirm"  # 최신 먼저

    def test_history_respects_limit(self, store):
        for i in range(5):
            store.record("note", f"Clause:{i}")
        assert len(store.history(limit=3)) == 3

    def test_record_carries_before_after_and_timestamp(self, store):
        """감사 항목은 {before, after} diff 를 싣는다 (KorAct
        ontology_versions 패턴) — 무엇이 어떻게 바뀌었는지 없이는 감사가 아니다."""
        store.record("confirm", "Clause:A", before={"definition": "이전"},
                     after={"definition": "이후"}, actor="ken", source="api")
        entry = store.history()[0]
        assert entry["before"] == {"definition": "이전"}
        assert entry["after"] == {"definition": "이후"}
        assert entry["actor"] == "ken"
        assert entry["at"]  # ISO 타임스탬프


# ─── 2. 영속성 — 로그가 원본이다 ─────────────────────────────────────

class TestPersistence:
    def test_state_is_rebuilt_by_replaying_the_log(self, tmp_path):
        """상태 파일을 따로 두지 않는다 — 이벤트 재생만으로 묘비/확정이
        복원되어야 로그와 상태가 어긋날 수 없다."""
        path = tmp_path / "reviews.jsonl"
        store = ReviewStore(namespace="p", path=path)
        store.reject("Clause:A", reason="오추출", actor="ken")
        store.confirm("Clause:B")
        store.reject("Clause:B")   # 번복 — replay 가 순서를 지켜야 한다

        restored = ReviewStore(namespace="p", path=path)
        restored.load_from_disk()
        assert restored.is_rejected("Clause:A") is True
        assert restored.is_rejected("Clause:B") is True
        assert restored.is_confirmed("Clause:B") is False
        assert len(restored.history()) == 3

    def test_events_are_appended_not_rewritten(self, tmp_path):
        """감사 로그는 append-only 다 — 두 번째 record 가 첫 줄을 다시 쓰면
        과거 감사 기록의 무결성을 보장할 수 없다."""
        path = tmp_path / "reviews.jsonl"
        store = ReviewStore(namespace="p", path=path)
        store.reject("Clause:A")
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        store.confirm("Clause:B")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert lines[0] == first_line  # 첫 이벤트는 그대로

    def test_load_skips_corrupt_lines(self, tmp_path):
        """한 줄이 깨졌다고 묘비 전체를 잃으면 오추출이 일제히 부활한다 —
        chunk_store 와 같은 이유로 깨진 줄만 버린다."""
        path = tmp_path / "reviews.jsonl"
        store = ReviewStore(namespace="p", path=path)
        store.reject("Clause:A")
        store.reject("Clause:B")
        content = path.read_text(encoding="utf-8").splitlines()
        path.write_text(content[0] + "\nthis is not json\n" + content[1] + "\n",
                        encoding="utf-8")

        restored = ReviewStore(namespace="p", path=path)
        restored.load_from_disk()
        assert restored.is_rejected("Clause:A") is True
        assert restored.is_rejected("Clause:B") is True

    def test_load_missing_file_is_not_an_error(self, tmp_path):
        store = ReviewStore(namespace="p", path=tmp_path / "absent.jsonl")
        assert store.load_from_disk() is False
        assert store.history() == []


# ─── 3. 네임스페이스 싱글턴 ──────────────────────────────────────────

class TestNamespaceSingleton:
    @pytest.fixture(autouse=True)
    def _tmp_data_dir(self, tmp_path, monkeypatch):
        # 싱글턴 기본 경로가 실제 data/ 를 오염시키지 않게 격리한다
        import ontology.core.review_store as rs_mod
        monkeypatch.setattr(rs_mod, "_DEFAULT_DATA_DIR", tmp_path)

    def test_same_namespace_same_instance(self):
        assert get_review_store("nsA") is get_review_store("nsA")

    def test_namespace_isolation(self):
        """nsA 의 거절이 nsB 의 빌드를 막으면 안 된다 — 네임스페이스는
        독립 온톨로지다 (KG·chunk store 와 같은 경계)."""
        get_review_store("nsA").reject("Clause:A")
        assert get_review_store("nsB").is_rejected("Clause:A") is False

    def test_reset_gives_fresh_instance(self):
        before = get_review_store("nsA")
        reset_review_stores()
        assert get_review_store("nsA") is not before


# ─── 4. 빌더 관통 — 묘비가 부활을 막는다 ────────────────────────────

@pytest.fixture
def build_env(tmp_path):
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

    kg = KnowledgeGraphEngine(fast_mode=True, namespace="review_build")
    review = ReviewStore(namespace="review_build",
                         path=tmp_path / "reviews.jsonl")
    chunks = ChunkStore(namespace="review_build", path=tmp_path / "chunks.jsonl")
    builder = OntologyBuilder(schema=SCHEMA, kg=kg, llm_fn=fake_llm,
                              chunk_store=chunks, review_store=review,
                              auto_save=False)
    return builder, kg, review


class TestTombstoneBlocksResurrection:
    def test_rejected_node_does_not_come_back_on_rebuild(self, build_env):
        """검수 루프의 존재 이유: 거절 후 같은 문서를 다시 인제스트해도
        같은 오추출이 되살아나면 안 된다."""
        builder, kg, review = build_env
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Clause:청약철회" in kg.graph

        review.reject("Clause:청약철회", reason="오추출")
        kg.graph.remove_node("Clause:청약철회")  # service reject 가 하는 제거

        report = run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Clause:청약철회" not in kg.graph
        assert report.entities_rejected >= 1

    def test_relations_referencing_tombstoned_entity_are_dropped(self, build_env):
        """묘비 노드를 참조하는 관계는 함께 버려진다 — 없는 노드로의 간선은
        유령 참조다. KeyError 로 빌드가 죽어서도 안 된다."""
        builder, kg, review = build_env
        review.reject("Clause:청약철회")

        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        # citesRegulation 의 주어가 묘비 → 간선 0
        assert kg.graph.number_of_edges() == 0

    def test_other_entities_still_merge(self, build_env):
        """한 개체의 거절이 같은 청크의 다른 개체 병합을 깨면 안 된다 —
        빌더의 resilient 원칙 그대로다."""
        builder, kg, review = build_env
        review.reject("Clause:청약철회")

        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Regulation:금융소비자 보호에 관한 법률" in kg.graph
        assert "Coverage:암진단비" in kg.graph

    def test_records_path_respects_tombstones(self, build_env):
        builder, kg, review = build_env
        review.reject("Site:숭례문")
        report = run(builder.build_from_records(
            [{"이름": "숭례문"}, {"이름": "불국사"}],
            {"node_type": "Site", "name_field": "이름"}, source="w"))
        assert "Site:숭례문" not in kg.graph
        assert "Site:불국사" in kg.graph
        assert report.entities_rejected == 1

    def test_check_tombstones_can_be_disabled(self, tmp_path):
        """옵트아웃 — 검수를 쓰지 않는 소비자(대량 마이그레이션 등)는
        묘비 조회 비용 없이 기존 동작 그대로 빌드한다."""
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        kg = KnowledgeGraphEngine(fast_mode=True, namespace="review_off")
        review = ReviewStore(namespace="review_off",
                             path=tmp_path / "reviews.jsonl")
        review.reject("Clause:청약철회")
        builder = OntologyBuilder(schema=SCHEMA, kg=kg, llm_fn=fake_llm,
                                  review_store=review, check_tombstones=False,
                                  store_chunks=False, auto_save=False)
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Clause:청약철회" in kg.graph


# ─── 5. 서비스 — 검수 큐 · 확정 · 거절 · 이력 ────────────────────────

@pytest.fixture
def svc_env(tmp_path):
    """서비스가 보는 네임스페이스 싱글턴(KG·chunk·review)을 tmp 로 격리하고
    샘플 온톨로지를 빌드해 둔다."""
    import ontology.core.chunk_store as cs_mod
    import ontology.core.review_store as rs_mod
    from ontology.engines.knowledge_graph_clean import (
        KnowledgeGraphEngine, _kg_instances)
    from ontology.server.service import OntologyBuilderService

    ns = f"review_svc_{uuid.uuid4().hex[:8]}"
    kg = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
    kg.save_to_disk = lambda *a, **k: True  # 테스트가 data/ 를 오염시키지 않게
    _kg_instances[ns] = kg
    chunks = ChunkStore(namespace=ns, path=tmp_path / "chunks.jsonl")
    cs_mod._stores[ns] = chunks
    review = ReviewStore(namespace=ns, path=tmp_path / "reviews.jsonl")
    rs_mod._stores[ns] = review

    builder = OntologyBuilder(schema=SCHEMA, kg=kg, llm_fn=fake_llm,
                              chunk_store=chunks, review_store=review,
                              auto_save=False)
    run(builder.build_from_text(SAMPLE_MD, source="약관.md"))

    service = OntologyBuilderService(data_dir=tmp_path / "ds", llm_fn=fake_llm)
    yield service, ns, kg, builder
    _kg_instances.pop(ns, None)


class TestReviewService:
    def test_queue_lists_unreviewed_llm_nodes_with_evidence(self, svc_env):
        """검수 큐는 '어디서 왔는지(source)' 있는 미검수 노드를 근거 청크
        개수와 함께 보여준다 — 근거 없이는 검수자가 판정할 수 없다."""
        service, ns, _kg, _b = svc_env
        queue = service.get_review_queue(ns)
        ids = {item["node_id"] for item in queue["items"]}
        assert "Clause:청약철회" in ids
        assert "Coverage:암진단비" in ids
        item = next(i for i in queue["items"]
                    if i["node_id"] == "Clause:청약철회")
        assert item["source"] == "약관.md"
        assert item["evidence_count"] >= 1

    def test_confirmed_node_leaves_the_queue(self, svc_env):
        service, ns, _kg, _b = svc_env
        result = service.confirm_node(ns, "Clause:청약철회", actor="ken")
        assert result is not None
        ids = {i["node_id"] for i in service.get_review_queue(ns)["items"]}
        assert "Clause:청약철회" not in ids

    def test_reject_removes_node_and_edges_and_survives_reingest(self, svc_env):
        """reject = 묘비 + 그래프 제거. 그리고 그 제거가 재인제스트를
        살아남는다 — 이것이 검수 루프의 핵심 계약이다."""
        service, ns, kg, builder = svc_env
        assert kg.graph.number_of_edges() == 1  # citesRegulation

        result = service.reject_node(ns, "Clause:청약철회",
                                     actor="ken", reason="오추출")
        assert result is not None
        assert "Clause:청약철회" not in kg.graph
        assert kg.graph.number_of_edges() == 0  # 노드 제거 = 간선도 제거

        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))  # 재인제스트
        assert "Clause:청약철회" not in kg.graph

    def test_confirm_and_reject_unknown_node_return_none(self, svc_env):
        service, ns, _kg, _b = svc_env
        assert service.confirm_node(ns, "Clause:없음") is None
        assert service.reject_node(ns, "Clause:없음") is None

    def test_history_returns_audit_entries(self, svc_env):
        service, ns, _kg, _b = svc_env
        service.confirm_node(ns, "Coverage:암진단비", actor="ken")
        service.reject_node(ns, "Clause:청약철회", reason="오추출")
        history = service.get_review_history(ns)["history"]
        assert [e["action"] for e in history[:2]] == ["reject", "confirm"]
        # 감사 항목은 판정 전 attrs 스냅샷(before)을 싣는다
        assert history[0]["before"].get("name") == "청약철회"

    def test_queue_trust_filter(self, svc_env):
        """trust 필터 — 요약서(summary)에서 온 노드만 골라 검수하는
        워크플로우 (낮은 신뢰 출처가 검수 1순위다)."""
        service, ns, kg, _b = svc_env
        kg.graph.nodes["Coverage:암진단비"]["trust"] = "summary"
        queue = service.get_review_queue(ns, trust="summary")
        ids = {i["node_id"] for i in queue["items"]}
        assert ids == {"Coverage:암진단비"}


# ─── 6. API — 4 라우트 happy-path + 404 ─────────────────────────────

@pytest.fixture
def api_env(svc_env):
    from ontology.server import router as server_router

    service, ns, kg, builder = svc_env
    app = FastAPI()
    app.include_router(server_router.router, prefix="/api/v1/ontology")
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app), ns, kg


class TestReviewAPI:
    def test_get_review_queue(self, api_env):
        client, ns, _kg = api_env
        response = client.get(f"/api/v1/ontology/graphs/{ns}/review")
        assert response.status_code == 200
        ids = {i["node_id"] for i in response.json()["items"]}
        assert "Clause:청약철회" in ids

    def test_confirm_endpoint(self, api_env):
        client, ns, _kg = api_env
        response = client.post(
            f"/api/v1/ontology/graphs/{ns}/review/confirm",
            json={"node_id": "Clause:청약철회", "actor": "ken"})
        assert response.status_code == 200
        queue = client.get(f"/api/v1/ontology/graphs/{ns}/review").json()
        assert "Clause:청약철회" not in {i["node_id"] for i in queue["items"]}

    def test_reject_endpoint_removes_node(self, api_env):
        client, ns, kg = api_env
        response = client.post(
            f"/api/v1/ontology/graphs/{ns}/review/reject",
            json={"node_id": "Clause:청약철회", "reason": "오추출"})
        assert response.status_code == 200
        assert "Clause:청약철회" not in kg.graph

    def test_history_endpoint(self, api_env):
        client, ns, _kg = api_env
        client.post(f"/api/v1/ontology/graphs/{ns}/review/confirm",
                    json={"node_id": "Coverage:암진단비"})
        response = client.get(f"/api/v1/ontology/graphs/{ns}/review/history",
                              params={"node_id": "Coverage:암진단비"})
        assert response.status_code == 200
        history = response.json()["history"]
        assert history and history[0]["action"] == "confirm"

    def test_confirm_and_reject_unknown_node_404(self, api_env):
        client, ns, _kg = api_env
        assert client.post(
            f"/api/v1/ontology/graphs/{ns}/review/confirm",
            json={"node_id": "Clause:없음"}).status_code == 404
        assert client.post(
            f"/api/v1/ontology/graphs/{ns}/review/reject",
            json={"node_id": "Clause:없음"}).status_code == 404
