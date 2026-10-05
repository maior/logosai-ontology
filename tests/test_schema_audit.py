"""스키마 선언·네임스페이스 삭제의 감사 계약 (2026-08-21).

종전 결함(실측): `set_predicate_decl` / `set_type_decl` 은 감사 이벤트를
남기지 않았고 `actor` 인자조차 없었다 — 술어의 domain/range 를 누가 언제
바꿨는지 추적할 방법이 없었다. lint(`consistency_checker.range_violation`)가
이 선언을 기준으로 오류를 내므로, 선언 변경은 그래프를 건드리지 않고도
**검수 결과를 뒤집는다**. 추적 없는 쓰기로 둘 이유가 없다.

`delete_namespace` 는 한 겹 더 있다: 그 네임스페이스의 감사 로그
(`reviews_{ns}.jsonl`)가 삭제 대상에 **포함**되므로, 거기에 기록하면 기록과
증거가 함께 사라진다. 그래서 네임스페이스-독립 싱크(`_admin`)에 남긴다.

계약:
  1. 선언 쓰기는 actor 와 before/after 를 남긴다 (before 는 직전 선언).
  2. 삭제는 `_admin` 싱크에 남고, 그 싱크는 네임스페이스 목록에 나타나지
     않는다 (유령 네임스페이스를 만들면 안 된다).
  3. 감사 action 이름은 파생 상태(묘비·확정)를 건드리지 않는다 —
     스키마 선언은 개체 판정이 아니다.
"""

import pytest

from ontology.core.review_store import (ADMIN_AUDIT_NAMESPACE, get_review_store,
                                        reset_review_stores)
from ontology.engines import knowledge_graph_clean as kgc
from ontology.server.service import OntologyBuilderService

NS = "auditns"


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.review_store as rs
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(kgc, "_DEFAULT_DATA_DIR", tmp_path)
    reset_review_stores()
    kgc._kg_instances.pop(NS, None)

    service = OntologyBuilderService()
    service.data_dir = tmp_path            # admin.db 를 tmp 로
    service._schema_decl = None
    service._saved_views = None
    service._projects = None
    engine = kgc.get_knowledge_graph_engine(NS)   # 네임스페이스 실존화
    engine.graph.clear()
    engine.graph.add_node("T:암", type="T", name="암")
    yield service
    reset_review_stores()
    kgc._kg_instances.pop(NS, None)


def _events(namespace, action):
    """해당 action 의 이벤트를 **시간 순**으로. history() 는 최신 먼저라
    (검수 화면용) 그대로 쓰면 before/after 의 순서 검사가 뒤집힌다."""
    rows = [e for e in get_review_store(namespace).history()
            if e.get("action") == action]
    return list(reversed(rows))


class TestSchemaDeclAudit:
    def test_predicate_decl_is_audited_with_actor(self, svc):
        out = svc.set_predicate_decl(NS, "coversDisease", domain="Contract",
                                     range_="Disease", description="보장",
                                     actor="alice")
        assert not out.get("error")
        events = _events(NS, "schema_predicate")
        assert len(events) == 1
        ev = events[0]
        assert ev["actor"] == "alice"
        assert ev["node_id"] == "predicate:coversDisease"
        assert ev["after"]["domain"] == "Contract"
        assert ev["after"]["range"] == "Disease"

    def test_predicate_decl_before_carries_prior_declaration(self, svc):
        svc.set_predicate_decl(NS, "p", domain="A", range_="B", actor="a1")
        svc.set_predicate_decl(NS, "p", domain="C", range_="D", actor="a2")
        events = _events(NS, "schema_predicate")
        assert len(events) == 2
        # 두 번째 이벤트의 before 가 첫 선언이어야 "무엇에서 무엇으로"가 남는다
        assert (events[1]["before"] or {}).get("domain") == "A"
        assert events[1]["after"]["domain"] == "C"

    def test_type_decl_is_audited(self, svc):
        svc.set_type_decl(NS, "Clause", description="조항", actor="bob")
        events = _events(NS, "schema_type")
        assert len(events) == 1
        assert events[0]["actor"] == "bob"
        assert events[0]["node_id"] == "type:Clause"
        assert events[0]["after"]["description"] == "조항"

    def test_invalid_declaration_is_not_audited(self, svc):
        """거부된 쓰기가 이력에 남으면 '했다'와 '하려다 막혔다'가 섞인다."""
        assert svc.set_predicate_decl(NS, "  ", actor="x").get("error")
        assert svc.set_type_decl(NS, "", actor="x").get("error")
        assert _events(NS, "schema_predicate") == []
        assert _events(NS, "schema_type") == []

    def test_decl_audit_does_not_touch_verdicts(self, svc):
        """스키마 선언은 개체 판정이 아니다 — 묘비·확정을 건드리면 안 된다."""
        store = get_review_store(NS)
        store.reject("T:암", reason="오추출", actor="r")
        svc.set_type_decl(NS, "T", description="타입", actor="bob")
        assert "T:암" in store.rejected_ids()
        assert not store.confirmed_ids()


class TestNamespaceDeletionAudit:
    def test_deletion_is_recorded_in_admin_sink(self, svc):
        svc.set_type_decl(NS, "T", actor="bob")          # ns 자체 로그 생성
        out = svc.delete_namespace(NS, actor="carol")
        assert out is not None

        # 그 네임스페이스의 로그는 삭제와 함께 사라진다 — 그래서 싱크가 필요하다
        admin = _events(ADMIN_AUDIT_NAMESPACE, "namespace_deleted")
        assert len(admin) == 1
        ev = admin[0]
        assert ev["actor"] == "carol"
        assert ev["node_id"] == f"namespace:{NS}"
        assert ev["before"]["deleted_files"]      # 무엇이 지워졌는지 남는다

    def test_admin_sink_is_not_a_visible_namespace(self, svc):
        svc.delete_namespace(NS, actor="carol")
        names = {row["namespace"] for row in svc.list_namespaces()}
        assert ADMIN_AUDIT_NAMESPACE not in names

    def test_failed_deletion_is_not_recorded(self, svc):
        """존재하지 않는/이름이 잘못된 삭제는 이력을 만들지 않는다."""
        assert svc.delete_namespace("없는네임스페이스", actor="carol") is None
        assert svc.delete_namespace("../escape", actor="carol") is None
        assert _events(ADMIN_AUDIT_NAMESPACE, "namespace_deleted") == []
