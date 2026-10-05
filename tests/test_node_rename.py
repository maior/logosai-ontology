"""
노드 타입 재분류(개명) 경로 — 로드맵 4 P-2 회귀 계약.

재분류 = id 개명이다 (`{type}:{name}` 이 id). 계약:
- 계획은 순수 함수(plan_rename) — **미리보기 == 적용** (merge 와 같은 구조).
- 적용 순서는 merge 계약: ① 새 노드 upsert ② 엣지 재지정 ③ 청크·골든셋
  참조 재지정 ④ 옛 노드 삭제. 순서가 바뀌면 PG 가 사실을 잃는다.
- **묘비를 남기지 않는다** — 재분류는 "타입이 틀렸다"지 "개체가 틀렸다"가
  아니다. 묘비면 재빌드에서 근거가 통째로 버려진다.
- 타깃이 이미 존재하면 그것은 개명이 아니라 **병합**이다 — /nodes/merge 로
  안내하고 거부한다.
- `reindex_required: True` 고정 — compose_node_text 가 id·타입을 포함한다.
- active 노드는 미리보기부터 차단 (생애주기 관문 — merge 와 동일).
"""

import networkx as nx
import pytest

from ontology.core.node_rename import plan_rename


def _graph():
    g = nx.MultiDiGraph()
    g.add_node("InsuranceTerm:제6조(보험금의 지급사유)",
               type="InsuranceTerm", name="제6조(보험금의 지급사유)",
               definition="지급사유 조항")
    g.add_node("Disease:암", type="Disease", name="암")
    g.add_node("ContractParty:계약자", type="ContractParty", name="계약자")
    g.add_edge("InsuranceTerm:제6조(보험금의 지급사유)", "Disease:암",
               predicate="coversDisease", weight=0.9)
    g.add_edge("ContractParty:계약자", "InsuranceTerm:제6조(보험금의 지급사유)",
               predicate="hasCondition")
    return g


class FakeChunk:
    def __init__(self, chunk_id, node_ids):
        self.chunk_id = chunk_id
        self.node_ids = node_ids


class FakeCase:
    def __init__(self, case_id, ids):
        self.case_id = case_id
        self._ids = set(ids)

    def accepted_ids(self):
        return set(self._ids)


OLD = "InsuranceTerm:제6조(보험금의 지급사유)"


# ─── plan_rename (순수) ──────────────────────────────────────────────


def test_plan_builds_new_id_and_attrs():
    plan = plan_rename(_graph(), OLD, "Clause")
    assert plan["new_id"] == "Clause:제6조(보험금의 지급사유)"
    assert plan["attrs_after"]["type"] == "Clause"
    assert plan["attrs_after"]["name"] == "제6조(보험금의 지급사유)"
    assert plan["attrs_after"]["definition"] == "지급사유 조항"  # 나머지 보존
    assert plan["reindex_required"] is True


def test_plan_repoints_both_directions():
    plan = plan_rename(_graph(), OLD, "Clause")
    edges = {(e["from"], e["to"], e["predicate"])
             for e in plan["edges_repointed"]}
    assert ("Clause:제6조(보험금의 지급사유)", "Disease:암", "coversDisease") in edges
    assert ("ContractParty:계약자", "Clause:제6조(보험금의 지급사유)", "hasCondition") in edges


def test_plan_preserves_self_loop_and_dedups_parallel():
    g = _graph()
    g.add_edge(OLD, OLD, predicate="ref")                 # 자기 참조 — 보존
    g.add_edge(OLD, "Disease:암", predicate="coversDisease")  # 평행 중복 — 1개로
    plan = plan_rename(g, OLD, "Clause")
    edges = [(e["from"], e["to"], e["predicate"]) for e in plan["edges_repointed"]]
    new = "Clause:제6조(보험금의 지급사유)"
    assert (new, new, "ref") in edges
    assert edges.count((new, "Disease:암", "coversDisease")) == 1


def test_plan_counts_chunk_and_golden_references():
    chunks = [FakeChunk("c1", [OLD, "Disease:암"]), FakeChunk("c2", ["Disease:암"])]
    cases = [FakeCase("g1", {OLD}), FakeCase("g2", {"Disease:암"})]
    plan = plan_rename(_graph(), OLD, "Clause", chunks=chunks, cases=cases)
    assert plan["chunks_rewritten"] == ["c1"]
    assert plan["chunk_node_ids"]["c1"] == ["Clause:제6조(보험금의 지급사유)", "Disease:암"]
    assert plan["golden_relabels"] == ["g1"]


def test_plan_rejects_bad_inputs():
    g = _graph()
    assert plan_rename(g, "T:없다", "Clause")["error"] == "node_not_found"
    assert plan_rename(g, OLD, "")["error"] == "invalid_type"
    assert plan_rename(g, OLD, "Cla:use")["error"] == "invalid_type"
    assert plan_rename(g, OLD, "Cla use")["error"] == "invalid_type"
    assert plan_rename(g, OLD, "InsuranceTerm")["error"] == "same_type"


def test_plan_target_exists_points_to_merge():
    g = _graph()
    g.add_node("Clause:제6조(보험금의 지급사유)", type="Clause")
    plan = plan_rename(g, OLD, "Clause")
    assert plan["error"] == "target_exists"
    assert "merge" in plan["detail"]  # 개명이 아니라 병합이다 — 경로 안내


# ─── service.rename_node (배선) ──────────────────────────────────────


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
    kgc._kg_instances.pop("renamens", None)

    NS = "renamens"
    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node(OLD, type="InsuranceTerm",
                          name="제6조(보험금의 지급사유)", definition="지급사유 조항")
    engine.graph.add_node("Disease:암", type="Disease", name="암")
    engine.graph.add_edge(OLD, "Disease:암", predicate="coversDisease", weight=0.9)

    store = cs.get_chunk_store(NS)
    store.clear()
    cid = store.add(Chunk(text="제6조 보험금의 지급사유 …", source="약관.pdf",
                          index=0, char_start=0, char_end=20))
    store.link_node(cid, OLD)

    pg_calls = []
    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply",
                        lambda ns, fn: pg_calls.append(fn))
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, NS, store, engine, cid, pg_calls


NEW = "Clause:제6조(보험금의 지급사유)"


class TestRenameNode:
    def test_dry_run_is_default_and_mutates_nothing(self, svc):
        service, NS, store, engine, cid, _ = svc
        res = service.rename_node(NS, OLD, "Clause")
        assert res["dry_run"] is True
        assert res["plan"]["new_id"] == NEW
        assert OLD in engine.graph and NEW not in engine.graph
        assert OLD in store.get(cid).node_ids

    def test_apply_follows_merge_order_and_repoints_all(self, svc):
        service, NS, store, engine, cid, pg_calls = svc
        res = service.rename_node(NS, OLD, "Clause", actor="tester",
                                  dry_run=False)
        assert res.get("error") is None
        # 그래프: 새 노드가 살고 옛 노드가 죽었다 — 엣지·속성 보존
        assert NEW in engine.graph and OLD not in engine.graph
        assert engine.graph.nodes[NEW]["type"] == "Clause"
        assert engine.graph.nodes[NEW]["definition"] == "지급사유 조항"
        edges = [(s, t, d.get("predicate"), d.get("weight"))
                 for s, t, d in engine.graph.out_edges(NEW, data=True)]
        assert edges == [(NEW, "Disease:암", "coversDisease", 0.9)]
        # 청크 참조 재지정 + 역색인
        assert store.get(cid).node_ids == [NEW]
        assert [c.chunk_id for c in store.chunks_for_node(NEW)] == [cid]
        assert store.chunks_for_node(OLD) == []
        assert res["reindex_required"] is True
        assert len(pg_calls) >= 3  # upsert + edge + delete

    def test_no_tombstone_left(self, svc):
        """재분류는 거절이 아니다 — 묘비면 재빌드에서 근거가 통째로 버려진다."""
        from ontology.core.review_store import get_review_store
        service, NS, *_ = svc
        service.rename_node(NS, OLD, "Clause", dry_run=False)
        assert OLD not in get_review_store(NS).rejected_ids()
        history = get_review_store(NS).history()
        assert any(h.get("action") == "reclassify" for h in history)

    def test_active_blocked_even_in_preview(self, svc):
        service, NS, store, engine, cid, _ = svc
        engine.graph.nodes[OLD]["lifecycle"] = "active"
        res = service.rename_node(NS, OLD, "Clause")   # dry_run 기본
        assert res["error"] == "lifecycle_protected"

    def test_preview_equals_apply(self, svc):
        """미리보기 = 적용 불변식 — 계획의 수치가 적용 결과와 일치한다."""
        service, NS, store, engine, cid, _ = svc
        preview = service.rename_node(NS, OLD, "Clause")["plan"]
        applied = service.rename_node(NS, OLD, "Clause", dry_run=False)
        assert applied["edges_repointed"] == len(preview["edges_repointed"])
        assert applied["chunks_rewritten"] == preview["chunks_rewritten"]
