"""고아 노드 링크 승인 — 쓰기 경로. approve_coverage_gaps 와 같은 계약.

**요청 본문을 신뢰하지 않는다.** find_orphan_candidates 의 "shadow 를 걸렀다"는
성질은 그 함수가 만든 것이고, 엔드포인트가 받는 {node_id, chunk_id} 는 그 성질을
물려받지 않는다. 같은 검증(원문 인용 + shadow)을 **다시** 통과시킨다 — 규칙을 두
벌 두면 두 경로의 '원문' 정의가 갈라진다.

노드를 만들지 않는다는 점이 커버리지 승인과 다르다. 여기서 노드는 이미 있고
없는 것은 **근거**다. 그래서 `reindex_required` 도 없다 — 노드 텍스트가 바뀌지
않으므로 시맨틱 색인은 영향받지 않는다(있다고 하면 불필요한 임베딩 비용을 부른다).
"""
import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_store import ChunkStore, reset_chunk_stores
from ontology.core.review_store import get_review_store, reset_review_stores


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    """실제 서비스 + tmp 저장소. 네임스페이스 하나."""
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    reset_chunk_stores()
    reset_review_stores()
    kgc._kg_instances.pop("orphanns", None)

    NS = "orphanns"
    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node("T:보험료", type="T", name="보험료")
    engine.graph.add_node("D:갑상선암", type="D", name="갑상선암")
    engine.graph.add_node("D:상선암", type="D", name="상선암")     # 오추출
    engine.graph.add_node("T:없는말", type="T", name="없는말")

    store = cs.get_chunk_store(NS)
    store.clear()
    c1 = store.add(Chunk(text="보험료 를 납입한 계약", source="약관.pdf",
                         index=0, char_start=0, char_end=20))
    c2 = store.add(Chunk(text="갑상선암 진단 시 보험료 면제", source="약관.pdf",
                         index=1, char_start=20, char_end=50))

    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, NS, store, engine, c1, c2


class TestApproveOrphanLinks:
    def test_links_node_to_chunk(self, svc):
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}],
            dry_run=False)
        assert res["chunks_linked"] == 1
        assert "T:보험료" in store.get(c1).node_ids
        assert store.chunks_for_node("T:보험료") == [store.get(c1)]

    def test_one_node_many_chunks(self, svc):
        """근거는 여럿이다 — node_id 기준 dedup 이면 나머지가 사라진다."""
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1},
                       {"node_id": "T:보험료", "chunk_id": c2}],
            dry_run=False)
        assert res["chunks_linked"] == 2
        assert len(store.chunks_for_node("T:보험료")) == 2

    def test_duplicate_pair_counted_once(self, svc):
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1},
                       {"node_id": "T:보험료", "chunk_id": c1}],
            dry_run=False)
        assert res["chunks_linked"] == 1

    # ─── 요청 본문 불신 ──────────────────────────────────────────────

    def test_name_absent_from_chunk_is_rejected(self, svc):
        """이름이 그 청크 원문에 없으면 근거가 아니다 — 거짓 provenance."""
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:없는말", "chunk_id": c1}], dry_run=False)
        assert res["chunks_linked"] == 0
        assert res["skipped"][0]["reason"] == "quote_not_found"
        assert "T:없는말" not in store.get(c1).node_ids

    def test_shadowed_link_is_rejected_even_if_requested(self, svc):
        """**본문을 신뢰하지 않는다.** 조회 경로가 걸러도 승인 경로가 또 걸러야
        한다 — 오추출 `상선암` 을 '갑상선암' 청크에 잇는 요청은 거부된다."""
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "D:상선암", "chunk_id": c2}], dry_run=False)
        assert res["chunks_linked"] == 0
        assert res["skipped"][0]["reason"] == "shadowed"

    def test_unknown_node_is_rejected(self, svc):
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:유령", "chunk_id": c1}], dry_run=False)
        assert res["skipped"][0]["reason"] == "node_not_found"

    def test_unknown_chunk_is_rejected(self, svc):
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": "nope"}],
            dry_run=False)
        assert res["skipped"][0]["reason"] == "chunk_not_found"

    def test_tombstoned_node_is_not_relinked(self, svc):
        """묘비를 무시하면 검수자 모르게 판정을 뒤집는다."""
        service, NS, store, engine, c1, c2 = svc
        get_review_store(NS).reject("T:보험료", actor="t", reason="x")
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}], dry_run=False)
        assert res["skipped"][0]["reason"] == "tombstoned"

    def test_already_linked_is_reported_not_double_counted(self, svc):
        service, NS, store, engine, c1, c2 = svc
        service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}], dry_run=False)
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}], dry_run=False)
        assert res["chunks_linked"] == 0
        assert res["skipped"][0]["reason"] == "already_linked"

    # ─── dry_run ────────────────────────────────────────────────────

    def test_dry_run_is_the_default(self, svc):
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}])
        assert res["dry_run"] is True
        assert "T:보험료" not in store.get(c1).node_ids

    def test_preview_count_matches_apply(self, svc):
        """미리보기가 적용과 다르면 검수자가 결정을 못 한다."""
        service, NS, store, engine, c1, c2 = svc
        links = [{"node_id": "T:보험료", "chunk_id": c1},
                 {"node_id": "T:보험료", "chunk_id": c2},
                 {"node_id": "D:갑상선암", "chunk_id": c2},
                 {"node_id": "D:상선암", "chunk_id": c2}]      # shadowed
        preview = service.approve_orphan_links(NS, links=links, dry_run=True)
        applied = service.approve_orphan_links(NS, links=links, dry_run=False)
        assert preview["chunks_linked"] == applied["chunks_linked"] == 3
        assert ([s["reason"] for s in preview["skipped"]]
                == [s["reason"] for s in applied["skipped"]])

    # ─── 계약 ───────────────────────────────────────────────────────

    def test_no_reindex_required(self, svc):
        """노드 텍스트가 안 바뀌므로 시맨틱 색인은 영향 없다 — True 로 보고하면
        불필요한 임베딩 비용을 부른다."""
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}], dry_run=False)
        assert res["reindex_required"] is False

    def test_no_new_nodes_created(self, svc):
        service, NS, store, engine, c1, c2 = svc
        before = engine.graph.number_of_nodes()
        service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}], dry_run=False)
        assert engine.graph.number_of_nodes() == before

    def test_audit_records_the_link(self, svc):
        """감사 없이 링크가 생기면 나중에 왜 이어졌는지 알 수 없다."""
        service, NS, store, engine, c1, c2 = svc
        service.approve_orphan_links(
            NS, links=[{"node_id": "T:보험료", "chunk_id": c1}], dry_run=False)
        events = get_review_store(NS).history()
        rec = [e for e in events if e.get("action") == "orphan_link_approve"]
        assert rec and rec[0]["after"]["chunk_id"] == c1

    def test_empty_links_is_an_error_not_a_silent_noop(self, svc):
        service, NS, store, engine, c1, c2 = svc
        assert service.approve_orphan_links(NS, links=[])["error"] == "invalid"

    def test_unknown_namespace(self, svc):
        service, NS, store, engine, c1, c2 = svc
        res = service.approve_orphan_links(
            "nope", links=[{"node_id": "T:보험료", "chunk_id": c1}])
        assert res["error"] == "namespace_not_found"
