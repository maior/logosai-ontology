"""생애주기 관문 — 파괴 경로가 실제로 막히는가 (서비스 레벨).

순수 함수 테스트(`test_lifecycle.py`)는 규칙을 고정한다. 여기서는 **배선**을
검사한다 — 규칙만 있고 관문이 안 걸리면 `active` 선언이 장식이 된다.

막아야 할 곳 셋: `reject_node`(묘비+제거) · `merge_nodes`(진 노드 삭제) ·
그리고 Phase 2 의 타입 재분류(= `{type}:{name}` id 개명).
"""
import pytest

from ontology.core.lifecycle import ACTIVE, DEPRECATED, EXPERIMENTAL


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    cs.reset_chunk_stores()
    rs.reset_review_stores()
    kgc._kg_instances.pop("lifens", None)

    NS = "lifens"
    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node("T:운영중", type="T", name="운영중", lifecycle=ACTIVE)
    engine.graph.add_node("T:실험", type="T", name="실험")
    engine.graph.add_node("T:폐기예정", type="T", name="폐기예정",
                          lifecycle=DEPRECATED, sunset="2026-12-31",
                          deprecated_reason="대체됨")
    engine.graph.add_node("T:승자", type="T", name="승자")

    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, NS, engine


class TestRejectGuard:
    def test_active_node_cannot_be_rejected(self, svc):
        service, NS, engine = svc
        res = service.reject_node(NS, "T:운영중", actor="t", reason="x")
        assert res["error"] == "lifecycle_protected"
        assert "T:운영중" in engine.graph        # 여전히 살아 있다

    def test_experimental_node_can_be_rejected(self, svc):
        service, NS, engine = svc
        res = service.reject_node(NS, "T:실험", actor="t", reason="x")
        assert res.get("status") == "rejected"
        assert "T:실험" not in engine.graph

    def test_deprecated_node_can_be_rejected(self, svc):
        """이미 선언된 경로다 — 여기서 또 막으면 영구히 못 지우는 노드가 된다."""
        service, NS, engine = svc
        res = service.reject_node(NS, "T:폐기예정", actor="t", reason="기한 도래")
        assert res.get("status") == "rejected"


class TestMergeGuard:
    def test_active_loser_is_blocked(self, svc):
        service, NS, engine = svc
        res = service.merge_nodes(NS, winner="T:승자", losers=["T:운영중"],
                                  dry_run=False)
        assert res["error"] == "lifecycle_protected"
        assert [b["node_id"] for b in res["blocked"]] == ["T:운영중"]
        assert "T:운영중" in engine.graph

    def test_dry_run_also_reports_the_block(self, svc):
        """적용 단계에서만 막으면 검수자가 계획을 다 보고 나서 거부당한다."""
        service, NS, engine = svc
        res = service.merge_nodes(NS, winner="T:승자", losers=["T:운영중"],
                                  dry_run=True)
        assert res["error"] == "lifecycle_protected"

    def test_experimental_loser_merges(self, svc):
        service, NS, engine = svc
        res = service.merge_nodes(NS, winner="T:승자", losers=["T:실험"],
                                  dry_run=False)
        assert "error" not in res
        assert "T:실험" not in engine.graph


class TestTransitionApi:
    def test_promote_to_active(self, svc):
        service, NS, engine = svc
        res = service.set_node_lifecycle(NS, "T:실험", ACTIVE, actor="t")
        assert res["to"] == ACTIVE
        assert engine.graph.nodes["T:실험"]["lifecycle"] == ACTIVE

    def test_active_to_experimental_is_refused(self, svc):
        """우회 차단 — 이게 뚫리면 active 선언이 무의미해진다."""
        service, NS, engine = svc
        res = service.set_node_lifecycle(NS, "T:운영중", EXPERIMENTAL, actor="t")
        assert res["error"] == "invalid_transition"
        assert engine.graph.nodes["T:운영중"]["lifecycle"] == ACTIVE

    def test_deprecate_requires_reason_and_sunset(self, svc):
        service, NS, engine = svc
        assert service.set_node_lifecycle(NS, "T:운영중", DEPRECATED,
                                          actor="t")["error"] == "invalid_transition"
        res = service.set_node_lifecycle(NS, "T:운영중", DEPRECATED,
                                         reason="C50 으로 통합",
                                         sunset="2026-12-31", actor="t")
        assert res["to"] == DEPRECATED
        assert engine.graph.nodes["T:운영중"]["sunset"] == "2026-12-31"

    def test_deprecate_then_reject_works(self, svc):
        """**선언된 경로 전체**가 통해야 한다 — 관문이 영구 차단이면 안 된다."""
        service, NS, engine = svc
        service.set_node_lifecycle(NS, "T:운영중", DEPRECATED, reason="r",
                                   sunset="2026-12-31", actor="t")
        assert service.reject_node(NS, "T:운영중", actor="t",
                                   reason="기한")["status"] == "rejected"

    def test_superseder_must_exist(self, svc):
        """허공을 가리키는 대체 지정은 '이걸 대신 쓰라'는 안내를 거짓으로 만든다."""
        service, NS, engine = svc
        res = service.set_node_lifecycle(NS, "T:운영중", DEPRECATED, reason="r",
                                         sunset="2026-12-31",
                                         superseded_by="T:없는것", actor="t")
        assert res["error"] == "superseder_not_found"

    def test_unknown_node(self, svc):
        service, NS, engine = svc
        assert service.set_node_lifecycle(NS, "T:유령",
                                          ACTIVE)["error"] == "node_not_found"

    def test_audit_records_the_transition(self, svc):
        """왜 이 노드가 폐기됐는지 감사 없이는 설명할 수 없다."""
        from ontology.core.review_store import get_review_store
        service, NS, engine = svc
        service.set_node_lifecycle(NS, "T:실험", ACTIVE, actor="tester")
        rec = [e for e in get_review_store(NS).history()
               if e.get("action") == "lifecycle"]
        assert rec and rec[0]["after"]["lifecycle"] == ACTIVE

    def test_no_reindex_required(self, svc):
        """상태만 바꿨을 뿐 노드 텍스트는 그대로다 → 색인 유효."""
        service, NS, engine = svc
        assert service.set_node_lifecycle(
            NS, "T:실험", ACTIVE)["reindex_required"] is False


class TestHealthReporting:
    def test_health_reports_distribution_and_overdue(self, svc, monkeypatch):
        service, NS, engine = svc
        engine.graph.add_node("T:기한지남", type="T", name="기한지남",
                              lifecycle=DEPRECATED, sunset="2000-01-01",
                              deprecated_reason="옛것")
        res = service.graph_health(NS)
        assert res["lifecycle"][ACTIVE] == 1
        assert "T:기한지남" in [r["node_id"] for r in res["lifecycle_overdue"]]
