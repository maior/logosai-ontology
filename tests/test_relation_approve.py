"""관계 승인 — 쓰기 경로. approve_coverage_gaps / approve_orphan_links 와 같은 계약.

**요청 본문을 신뢰하지 않는다.** propose_relations 의 "인용이 원문에 있다"는 성질은
그 함수가 만든 것이고, 엔드포인트가 받는 트리플은 그 성질을 물려받지 않는다.
같은 관문을 다시 통과시킨다.

**노드를 만들지 않는다.** 관계 승인이 노드 생성의 뒷문이 되면 커버리지 승인이
지키는 검증(원문 대조·타입 관문·묘비)을 우회한다.
"""
import pytest

from ontology.builder.models import Chunk
from ontology.core.review_store import get_review_store, reset_review_stores


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    cs.reset_chunk_stores()
    reset_review_stores()
    kgc._kg_instances.pop("relns", None)

    NS = "relns"
    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    for nid, name in (("D:암", "암"), ("D:갑상선암", "갑상선암"),
                      ("T:암진단비", "암진단비"), ("P:회사", "회사")):
        engine.graph.add_node(nid, type=nid.split(":")[0], name=name)
    # 허용 어휘가 데이터에서 오므로 기존 엣지가 하나는 있어야 한다
    engine.graph.add_edge("P:회사", "T:암진단비", predicate="hasCondition")

    store = cs.get_chunk_store(NS)
    store.clear()
    cid = store.add(Chunk(text='“갑상선암”은 암의 일종이다. 회사는 암진단비를 지급한다.',
                          source="약관.pdf", index=0,
                          char_start=0, char_end=60),
                    node_ids=["D:암", "D:갑상선암", "T:암진단비", "P:회사"])
    other = store.add(Chunk(text="관계 없는 조문", source="약관.pdf", index=1,
                            char_start=60, char_end=80),
                      node_ids=["D:암"])

    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, NS, engine, store, cid, other


def _rel(**kw):
    base = {"subject": "D:갑상선암", "predicate": "is_a", "object": "D:암",
            "evidence_quote": "“갑상선암”은 암의 일종이다"}
    base.update(kw)
    return base


class TestApproveRelations:
    def test_adds_the_edge(self, svc):
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(NS, [_rel(chunk_id=cid)], dry_run=False)
        assert res["added_total"] == 1
        data = engine.graph.get_edge_data("D:갑상선암", "D:암") or {}
        assert any(a.get("predicate") == "is_a" for a in data.values())

    def test_is_a_is_allowed_even_though_absent_from_graph(self, svc):
        """`is_a` 는 코드가 아는 유일한 술어다 — 그래프에 0개여도 허용해야
        폐포 확장(HIERARCHY_PREDICATE)이 살아난다."""
        service, NS, engine, store, cid, other = svc
        assert service.approve_relations(NS, [_rel(chunk_id=cid)],
                                         dry_run=False)["added_total"] == 1

    def test_unknown_predicate_rejected(self, svc):
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(
            NS, [_rel(chunk_id=cid, predicate="그냥관련")], dry_run=False)
        assert res["skipped"][0]["reason"] == "predicate_not_allowed"

    def test_quote_not_in_chunk_rejected(self, svc):
        """**본문 불신의 핵심 관문** — 지어낸 근거로 관계가 들어가면 안 된다."""
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(
            NS, [_rel(chunk_id=cid, evidence_quote="갑상선암은 암이 아니다")],
            dry_run=False)
        assert res["skipped"][0]["reason"] == "quote_not_found"
        assert engine.graph.get_edge_data("D:갑상선암", "D:암") in (None, {})

    def test_endpoints_must_be_in_that_chunk(self, svc):
        """다른 청크의 인용으로 아무 노드나 잇지 못한다."""
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(
            NS, [_rel(chunk_id=other, evidence_quote="관계 없는 조문")],
            dry_run=False)
        assert res["skipped"][0]["reason"] == "not_in_chunk"

    def test_unknown_node_rejected_no_node_created(self, svc):
        service, NS, engine, store, cid, other = svc
        before = engine.graph.number_of_nodes()
        res = service.approve_relations(
            NS, [_rel(chunk_id=cid, object="D:없는병")], dry_run=False)
        assert res["skipped"][0]["reason"] == "node_not_found"
        assert engine.graph.number_of_nodes() == before

    def test_self_loop_rejected(self, svc):
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(
            NS, [_rel(chunk_id=cid, object="D:갑상선암")], dry_run=False)
        assert res["skipped"][0]["reason"] == "self_loop"

    def test_tombstoned_endpoint_rejected(self, svc):
        """거절된 개체를 관계로 부활시키지 않는다."""
        service, NS, engine, store, cid, other = svc
        get_review_store(NS).reject("D:암", actor="t", reason="x")
        res = service.approve_relations(NS, [_rel(chunk_id=cid)], dry_run=False)
        assert res["skipped"][0]["reason"] == "tombstoned"

    def test_existing_edge_not_duplicated(self, svc):
        service, NS, engine, store, cid, other = svc
        service.approve_relations(NS, [_rel(chunk_id=cid)], dry_run=False)
        res = service.approve_relations(NS, [_rel(chunk_id=cid)], dry_run=False)
        assert res["skipped"][0]["reason"] == "already_exists"

    def test_duplicate_in_one_batch_counted_once(self, svc):
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(
            NS, [_rel(chunk_id=cid), _rel(chunk_id=cid)], dry_run=False)
        assert res["added_total"] == 1

    def test_dry_run_default_and_no_write(self, svc):
        service, NS, engine, store, cid, other = svc
        res = service.approve_relations(NS, [_rel(chunk_id=cid)])
        assert res["dry_run"] is True and res["added_total"] == 1
        assert engine.graph.get_edge_data("D:갑상선암", "D:암") in (None, {})

    def test_preview_matches_apply(self, svc):
        service, NS, engine, store, cid, other = svc
        rels = [_rel(chunk_id=cid),
                _rel(chunk_id=cid, predicate="그냥관련"),
                _rel(chunk_id=cid, object="D:갑상선암")]
        preview = service.approve_relations(NS, rels, dry_run=True)
        applied = service.approve_relations(NS, rels, dry_run=False)
        assert preview["added_total"] == applied["added_total"]
        assert ([s["reason"] for s in preview["skipped"]]
                == [s["reason"] for s in applied["skipped"]])

    def test_no_reindex_required(self, svc):
        """엣지는 노드 텍스트를 바꾸지 않는다 — True 면 불필요한 임베딩 비용."""
        service, NS, engine, store, cid, other = svc
        assert service.approve_relations(
            NS, [_rel(chunk_id=cid)], dry_run=False)["reindex_required"] is False

    def test_audit_keeps_the_quote(self, svc):
        """왜 이 관계가 생겼는지 인용 없이는 설명할 수 없다."""
        service, NS, engine, store, cid, other = svc
        service.approve_relations(NS, [_rel(chunk_id=cid)], dry_run=False)
        rec = [e for e in get_review_store(NS).history()
               if e.get("action") == "relation_approve"]
        assert rec and "갑상선암" in rec[0]["after"]["evidence_quote"]

    def test_empty_is_an_error(self, svc):
        service, NS, engine, store, cid, other = svc
        assert service.approve_relations(NS, [])["error"] == "invalid"

    def test_unknown_namespace(self, svc):
        service, NS, engine, store, cid, other = svc
        assert service.approve_relations(
            "nope", [_rel(chunk_id=cid)])["error"] == "namespace_not_found"
