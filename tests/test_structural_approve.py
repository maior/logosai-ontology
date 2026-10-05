"""
구조 단위 재분류 승인 — 로드맵 4 P-3 회귀 계약.

- **본문을 신뢰하지 않는다**: 승인 시 탐지(find_structural_candidates)를
  재실행해 여전히 후보인 것만 개명한다 — coverage/orphan approve 와 같은
  규율 (규칙을 두 벌 두면 두 경로의 '후보' 정의가 갈라진다).
- `new_type` 은 요청당 하나·필수 — 서버 기본값 없음 (타입 이름 하드코딩
  금지의 실행 형태: PROJ-A 엔 '조항'이 없다, 검수자가 정한다).
- 항목별 실패(active·비후보)는 전체를 막지 않되 **dry_run 미리보기에 미리
  나타난다** (미리보기 == 적용).
- 미선언 타입은 적용 시 schema_decl 에 선언된다.
"""

import pytest

OLD = "InsuranceTerm:제6조(보험금의 지급사유)"
OLD2 = "InsuranceTerm:제1조 【목적 】"
CONCEPT = "Disease:암"


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    import ontology.core.search_qa as sq
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import reset_chunk_stores
    from ontology.core.review_store import reset_review_stores
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(sq, "_DEFAULT_DATA_DIR", tmp_path)
    reset_chunk_stores()
    reset_review_stores()
    kgc._kg_instances.pop("structns", None)

    NS = "structns"
    service = OntologyBuilderService()
    service._schema_decl = None
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node(OLD, type="InsuranceTerm",
                          name="제6조(보험금의 지급사유)")
    engine.graph.add_node(OLD2, type="InsuranceTerm", name="제1조 【목적 】")
    engine.graph.add_node(CONCEPT, type="Disease", name="암")  # 비후보

    store = cs.get_chunk_store(NS)
    store.clear()
    # 표기차: 라벨은 【】 — normalize 동등성이 잡는다
    c1 = store.add(Chunk(text="보험금의 지급사유는 …", source="약관.pdf",
                         index=0, section="제6조 【보험금의 지급사유】",
                         char_start=0, char_end=20))
    c2 = store.add(Chunk(text="이 약관의 목적은 …", source="약관.pdf",
                         index=1, section="제1조 【목적】",
                         char_start=20, char_end=40))
    store.link_node(c1, OLD)

    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, NS, store, engine


class TestApproveStructural:
    def test_new_type_required(self, svc):
        service, NS, *_ = svc
        res = service.approve_structural(NS, items=[OLD], new_type="")
        assert res["error"] == "invalid"

    def test_body_not_trusted_non_candidate_skipped(self, svc):
        """실존 노드라도 탐지 후보가 아니면 skipped — 본문 불신."""
        service, NS, *_ = svc
        res = service.approve_structural(NS, items=[CONCEPT], new_type="Clause",
                                         dry_run=False)
        assert res["results"][0]["status"] == "skipped"
        assert res["results"][0]["reason"] == "not_a_candidate"
        assert res["renamed"] == 0

    def test_dry_run_previews_without_mutation(self, svc):
        service, NS, store, engine = svc
        res = service.approve_structural(NS, items=[OLD], new_type="Clause")
        assert res["dry_run"] is True
        assert res["results"][0]["status"] == "would_rename"
        assert res["results"][0]["new_id"] == "Clause:제6조(보험금의 지급사유)"
        assert OLD in engine.graph
        assert "Clause" not in service.schema_decl.types(NS)  # 선언도 안 한다

    def test_apply_renames_and_declares_type(self, svc):
        service, NS, store, engine = svc
        res = service.approve_structural(NS, items=[OLD, OLD2],
                                         new_type="Clause", dry_run=False,
                                         actor="tester")
        assert res["renamed"] == 2
        assert "Clause:제6조(보험금의 지급사유)" in engine.graph
        assert OLD not in engine.graph
        assert "Clause" in service.schema_decl.types(NS)   # 미선언 타입 선언
        assert res["reindex_required"] is True

    def test_active_blocks_item_not_batch(self, svc):
        service, NS, store, engine = svc
        engine.graph.nodes[OLD]["lifecycle"] = "active"
        res = service.approve_structural(NS, items=[OLD, OLD2],
                                         new_type="Clause", dry_run=False)
        by_id = {r["node_id"]: r for r in res["results"]}
        assert by_id[OLD]["status"] == "blocked"
        assert by_id[OLD]["reason"] == "lifecycle_protected"
        assert by_id[OLD2]["status"] == "renamed"
        assert res["renamed"] == 1

    def test_queue_drains_after_approve(self, svc):
        """승인된 타입(Clause)의 노드는 재탐지에서 빠진다 — 안 빠지면 큐가
        영원히 마르지 않는다 (라이브에서 실측된 결함). 제외 근거는 승인이
        남긴 type_declared 감사 이벤트 — 타입명 하드코딩 없음."""
        service, NS, store, engine = svc
        before = service.review_structural(NS)
        assert before["total"] == 2  # OLD, OLD2

        service.approve_structural(NS, items=[OLD, OLD2], new_type="Clause",
                                   dry_run=False)
        after = service.review_structural(NS)
        ids = {c["node_id"] for c in after["candidates"]}
        assert "Clause:제6조(보험금의 지급사유)" not in ids
        assert after["total"] == 0

    def test_structural_types_survive_replay(self, tmp_path, svc):
        """선언은 감사 로그에서 파생된다 — 재기동(replay) 후에도 제외 유지."""
        from ontology.core.review_store import ReviewStore
        service, NS, store, engine = svc
        service.approve_structural(NS, items=[OLD], new_type="Clause",
                                   dry_run=False)
        from ontology.core.review_store import get_review_store
        path = get_review_store(NS).path
        fresh = ReviewStore(namespace=NS, path=path)
        fresh.load_from_disk()
        assert "Clause" in fresh.structural_types()

    def test_preview_equals_apply_statuses(self, svc):
        """미리보기의 항목별 판정(would_rename/blocked/skipped)이 적용과 같다."""
        service, NS, store, engine = svc
        engine.graph.nodes[OLD]["lifecycle"] = "active"
        items = [OLD, OLD2, CONCEPT]
        prev = service.approve_structural(NS, items=items, new_type="Clause")
        appl = service.approve_structural(NS, items=items, new_type="Clause",
                                          dry_run=False)
        norm = {"would_rename": "renamed"}
        prev_map = {r["node_id"]: norm.get(r["status"], r["status"])
                    for r in prev["results"]}
        appl_map = {r["node_id"]: r["status"] for r in appl["results"]}
        assert prev_map == appl_map
