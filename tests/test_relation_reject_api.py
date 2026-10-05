"""
관계 기각 API + propose/approve 배선 — A2 회귀 계약.

- propose: scope=triple 또는 같은 청크의 기각 트리플은 걸러지고
  (`rejected_filtered` 로 셈 보고 — 조용한 절단 금지), 다른 인용의 재제안은
  `previously_rejected` 를 동봉해 사람에게.
- approve: 기각 트리플은 기본 skip(relation_rejected), 명시적
  override_rejected 만 통과 — 통과하면 relation_approve 가 묘비를 걷는다.
- reject API: reason 필수, 이중 기각 skip.
"""

import json

import pytest

from ontology.builder.models import Chunk

S, P, O = "Disease:암", "coversDisease", "InsuranceContract:계약"


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    from ontology.core.chunk_store import reset_chunk_stores
    from ontology.core.review_store import reset_review_stores
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    reset_chunk_stores()
    reset_review_stores()
    kgc._kg_instances.pop("relrejns", None)

    NS = "relrejns"
    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node(S, type="Disease", name="암")
    engine.graph.add_node(O, type="InsuranceContract", name="계약")
    # 허용 술어 어휘를 데이터에서 만들기 위한 기존 엣지 (다른 노드 쌍)
    engine.graph.add_node("D:x", type="D")
    engine.graph.add_node("C:y", type="C")
    engine.graph.add_edge("D:x", "C:y", predicate=P)

    store = cs.get_chunk_store(NS)
    store.clear()
    c1 = store.add(Chunk(text="암은 계약 이 보장한다", source="약관.pdf",
                         index=0, char_start=0, char_end=20),
                   node_ids=[S, O])
    c2 = store.add(Chunk(text="계약 은 암 을 보장하지 않는다", source="약관.pdf",
                         index=1, char_start=20, char_end=45),
                   node_ids=[S, O])

    # propose 의 LLM 을 결정적 가짜로 — 항상 (S,P,O) + c 청크 인용을 낸다
    async def _fake_propose(namespace, limit=0, min_nodes=2):
        raise AssertionError("not used")

    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, NS, store, engine, c1, c2


def _llm_for(chunk_text_marker: str):
    """청크 본문에 marker 가 있으면 (S,P,O) 관계 하나를 제안하는 가짜 LLM."""
    def fake_llm(prompt: str) -> str:
        if chunk_text_marker in prompt:
            quote = "암은 계약 이 보장한다" if "보장한다" in prompt else prompt
        return json.dumps({"relations": [{
            "subject": S, "predicate": P, "object": O,
            "evidence_quote": ("암은 계약 이 보장한다"
                               if "암은 계약 이 보장한다" in prompt
                               else "계약 은 암 을 보장하지 않는다"),
        }]})
    return fake_llm


class TestRejectAPI:
    def test_reason_required_and_duplicate_skipped(self, svc):
        service, NS, *_ = svc
        res = service.reject_relations(NS, [
            {"subject": S, "predicate": P, "object": O},              # reason 없음
            {"subject": S, "predicate": P, "object": O, "reason": "왜곡"},
            {"subject": S, "predicate": P, "object": O, "reason": "중복"},
        ])
        reasons = [x["reason"] for x in res["skipped"]]
        assert "reason_required" in reasons
        assert "already_rejected" in reasons
        assert res["rejected_total"] == 1


class TestApproveRespectsTombstone:
    def _approve(self, service, NS, c1, override=False):
        rel = {"subject": S, "predicate": P, "object": O, "chunk_id": c1,
               "evidence_quote": "암은 계약 이 보장한다"}
        if override:
            rel["override_rejected"] = True
        return service.approve_relations(NS, [rel], dry_run=False)

    def test_rejected_triple_skipped_then_override_lifts(self, svc):
        service, NS, store, engine, c1, c2 = svc
        service.reject_relations(NS, [{"subject": S, "predicate": P,
                                       "object": O, "reason": "왜곡"}])
        res = self._approve(service, NS, c1)
        assert res["skipped"][0]["reason"] == "relation_rejected"
        assert res["added_total"] == 0

        res2 = self._approve(service, NS, c1, override=True)
        assert res2["added_total"] == 1
        # 승인이 묘비를 걷었다 — 나중 판정이 이긴다
        from ontology.core.review_store import get_review_store
        assert get_review_store(NS).relation_rejection(S, P, O) is None


class TestProposeFiltersAndMarks:
    def _propose(self, service, NS):
        import asyncio
        return asyncio.run(service.propose_relations(NS))

    def test_triple_scope_filters_everywhere(self, svc, monkeypatch):
        service, NS, store, engine, c1, c2 = svc
        monkeypatch.setattr(service, "_active_llm_fn",
                            lambda: _llm_for("보장"))
        service.reject_relations(NS, [{"subject": S, "predicate": P,
                                       "object": O, "reason": "왜곡",
                                       "scope": "triple"}])
        res = self._propose(service, NS)
        assert res["proposals"] == []          # 두 청크 모두 걸러짐
        assert res["rejected_filtered"] == 2   # 조용한 절단 금지 — 셈 보고

    def test_evidence_scope_filters_same_chunk_marks_other(self, svc, monkeypatch):
        service, NS, store, engine, c1, c2 = svc
        monkeypatch.setattr(service, "_active_llm_fn",
                            lambda: _llm_for("보장"))
        service.reject_relations(NS, [{"subject": S, "predicate": P,
                                       "object": O, "reason": "이 인용만",
                                       "chunk_id": c1, "scope": "evidence"}])
        res = self._propose(service, NS)
        # c1 유래는 걸러지고, c2 유래는 과거 기각 요약을 달고 나온다
        assert res["rejected_filtered"] == 1
        assert len(res["proposals"]) == 1
        p = res["proposals"][0]
        assert p["chunk_id"] == c2
        assert p["previously_rejected"]["reason"] == "이 인용만"

    def test_no_rejection_no_mark(self, svc, monkeypatch):
        service, NS, store, engine, c1, c2 = svc
        monkeypatch.setattr(service, "_active_llm_fn",
                            lambda: _llm_for("보장"))
        res = self._propose(service, NS)
        assert res["rejected_filtered"] == 0
        assert all(p["previously_rejected"] is None for p in res["proposals"])
