"""트리아지 파이프라인 서비스 배선 — triage_review · judge_batch ·
recommendation_quality (C1-b · C1-c · C2 회귀 계약).

고정하는 계약:
- 결정적 신호(dup·structural)만으로 borderline 이 확정된 항목은 **LLM 콜을
  생략**한다 (예산 — reasons 에 llm_skipped). LLM 이 없으면 죽지 않고
  결정적 신호만으로 degrade.
- 트리아지 결과는 recommend 이벤트로 남고(기존 record 경유, verdict 계약
  유지 + band/signals 확장) **판정을 만들지 않는다**.
- judge_batch 는 본문 불신: 미판정 + 최신 노드 추천의 verdict/band 일치를
  재검증하고, 통과분만 confirm_node/reject_node **기존 경로 위임** (판정
  쓰기 두 벌 금지). 관계는 approve/reject_relations 위임.
- dry_run 미리보기 상태 == 적용 상태 (approve_structural 의 계약과 동형).
"""

import asyncio
import json

import pytest

from ontology.builder.models import Chunk

NS = "triagens"
A = "Term:암진단비"        # LLM confirm + 근거 → strong_confirm
B = "Term:화성약관"        # LLM reject + 사유 → strong_reject
D1 = "TypeA:계약"          # 중복 쌍 — llm_skipped, borderline
D2 = "TypeB:계약"
E1, E2, E3 = "Disease:암", "Contract:계약서", "Contract:특약"   # 관계용 (source 없음 — 큐 밖)


def run(coro):
    return asyncio.run(coro)


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
    kgc._kg_instances.pop(NS, None)

    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        if "개체 후보" in prompt:               # 관계 제안 프롬프트
            if E1 in prompt:
                return json.dumps({"relations": [
                    {"subject": E1, "predicate": "covers", "object": E2,
                     "evidence_quote": "암 은 계약서 가 보장한다"},
                    {"subject": E1, "predicate": "covers", "object": E3,
                     "evidence_quote": "특약 도 있다"},   # 이름 1/2 → borderline
                ]}, ensure_ascii=False)
            return json.dumps({"relations": []})
        # 근거대조 프롬프트 — fact JSON 의 이름으로 분기
        if '"화성약관"' in prompt:
            return json.dumps({"verdict": "reject",
                               "rationale": "원문 어디에도 없는 사실",
                               "evidence_quote":
                                   "암진단비는 최초 1회에 한하여 지급한다"},
                              ensure_ascii=False)
        return json.dumps({"verdict": "confirm", "rationale": "원문 명시",
                           "evidence_quote":
                               "암진단비는 최초 1회에 한하여 지급한다"},
                          ensure_ascii=False)

    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node(A, type="Term", name="암진단비", source="약관.pdf")
    engine.graph.add_node(B, type="Term", name="화성약관", source="약관.pdf")
    engine.graph.add_node(D1, type="TypeA", name="계약", source="약관.pdf")
    engine.graph.add_node(D2, type="TypeB", name="계약", source="약관.pdf")
    # 관계 트리아지용 — source 없음(검수 큐 밖), 시그니처 선례 포함
    engine.graph.add_node(E1, type="Disease", name="암")
    engine.graph.add_node(E2, type="Contract", name="계약서")
    engine.graph.add_node(E3, type="Contract", name="특약")
    engine.graph.add_node("Disease:x2", type="Disease", name="x2")
    engine.graph.add_node("Contract:y2", type="Contract", name="y2")
    engine.graph.add_edge("Disease:x2", "Contract:y2", predicate="covers")

    store = cs.get_chunk_store(NS)
    store.clear()
    c1 = store.add(Chunk(text="암진단비는 최초 1회에 한하여 지급한다",
                         source="약관.pdf", index=0,
                         char_start=0, char_end=30),
                   node_ids=[A, B])
    c2 = store.add(Chunk(text="암 은 계약서 가 보장한다. 특약 도 있다",
                         source="약관.pdf", index=1,
                         char_start=30, char_end=60),
                   node_ids=[E1, E2, E3])

    service = OntologyBuilderService(llm_fn=fake_llm)
    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    monkeypatch.setattr(service, "_pg_apply", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "save_to_disk", lambda *a, **kw: True)
    return service, engine, store, calls, c1, c2


# ─── 1. triage_review ────────────────────────────────────────────────

class TestTriageReview:
    def test_bands_and_llm_budget(self, svc):
        service, engine, store, calls, *_ = svc
        res = run(service.triage_review(NS))
        bands = res["nodes"]
        assert [x["node_id"] for x in bands["strong_confirm"]] == [A]
        assert [x["node_id"] for x in bands["strong_reject"]] == [B]
        assert {x["node_id"] for x in bands["borderline"]} == {D1, D2}
        # dup 항목은 LLM 을 부르지 않는다 — A·B 두 콜뿐 (예산 계약)
        assert calls["n"] == 2
        assert res["llm_calls"] == 2
        for x in bands["borderline"]:
            assert "llm_skipped" in x["reasons"]
            assert x["signals"]["dup_member"] is True
        assert res["relations"] is None            # 옵트인 전엔 안 돈다
        assert res["counts"]["nodes"] == {"strong_confirm": 1,
                                          "strong_reject": 1, "borderline": 2}

    def test_recommend_recorded_with_band_and_no_judgment(self, svc):
        """추천은 기록되고(기존 verdict 계약 + band/signals 확장) 판정은
        만들지 않는다 — 최종 권한은 인간."""
        service, *_ = svc
        run(service.triage_review(NS))
        from ontology.core.review_store import get_review_store
        reviews = get_review_store(NS)
        assert not reviews.is_confirmed(A) and not reviews.is_rejected(B)
        queue = service.get_review_queue(NS)
        by = {i["node_id"]: i for i in queue["items"]}
        rec = by[A]["recommendation"]
        assert rec["verdict"] == "confirm"
        assert rec["band"] == "strong_confirm"
        assert rec["signals"]["dup_member"] is False

    def test_no_llm_degrades_to_borderline(self, svc, monkeypatch):
        """LLM 없음 → verdict 없이 결정적 신호만 — 죽지 않는다 (degrade)."""
        service, engine, store, calls, *_ = svc
        monkeypatch.setattr(service, "_active_llm_fn", lambda: None)
        res = run(service.triage_review(NS))
        assert res["llm_calls"] == 0 and calls["n"] == 0
        assert not res["nodes"]["strong_confirm"]
        assert not res["nodes"]["strong_reject"]
        assert len(res["nodes"]["borderline"]) == 4

    def test_relations_optin_banding_and_recording(self, svc):
        service, *_ , c1, c2 = svc
        res = run(service.triage_review(NS, include_relations=True))
        rels = res["relations"]
        strong = rels["strong_confirm"]
        assert [(r["subject"], r["object"]) for r in strong] == [(E1, E2)]
        assert strong[0]["chunk_id"] == c2
        border = rels["borderline"]
        assert [(r["subject"], r["object"]) for r in border] == [(E1, E3)]
        assert any("이름" in reason for reason in border[0]["reasons"])
        # recommend 이벤트에 predicate/target/band/chunk_id 가 실린다
        from ontology.core.review_store import get_review_store
        recs = [h for h in get_review_store(NS).history(node_id=E1)
                if h["action"] == "recommend"]
        by_target = {h["after"]["target"]: h["after"] for h in recs}
        assert by_target[E2]["band"] == "strong_confirm"
        assert by_target[E2]["verdict"] == "confirm"
        assert by_target[E3]["band"] == "borderline"
        assert by_target[E2]["chunk_id"] == c2

    def test_namespace_not_found(self, svc):
        service, *_ = svc
        assert run(service.triage_review("없는ns"))["error"] == \
            "namespace_not_found"


# ─── 2. judge_batch ──────────────────────────────────────────────────

class TestJudgeBatch:
    def _items(self):
        return [
            {"kind": "node", "node_id": A, "verdict": "confirm"},
            {"kind": "node", "node_id": B, "verdict": "reject"},
            {"kind": "node", "node_id": D1, "verdict": "confirm"},   # borderline
        ]

    def test_preview_equals_apply_statuses(self, svc):
        """미리보기의 항목별 상태가 적용과 같다 — 일괄 판정의 핵심 계약."""
        service, *_ = svc
        run(service.triage_review(NS))
        items = self._items()
        prev = service.judge_batch(NS, items)              # dry_run 기본
        assert prev["dry_run"] is True and prev["applied"] == 0
        appl = service.judge_batch(NS, items, dry_run=False)
        norm = {"would_confirm": "confirmed", "would_reject": "rejected"}
        prev_map = {r["node_id"]: norm.get(r["status"], r["status"])
                    for r in prev["results"]}
        appl_map = {r["node_id"]: r["status"] for r in appl["results"]}
        assert prev_map == appl_map
        assert appl["applied"] == 2

    def test_borderline_never_batch_judged(self, svc):
        """band 가 strong 이 아니면 요청 verdict 가 무엇이든 skip —
        borderline 은 사람 몫이다 (recommendation_mismatch)."""
        service, *_ = svc
        run(service.triage_review(NS))
        res = service.judge_batch(
            NS, [{"kind": "node", "node_id": D1, "verdict": "confirm"}])
        assert res["results"][0]["status"] == "skipped"
        assert res["results"][0]["reason"] == "recommendation_mismatch"

    def test_mismatched_verdict_skipped(self, svc):
        """추천 confirm 에 요청 reject — 낡은/뒤집힌 추천을 조용히 덮지 않는다."""
        service, *_ = svc
        run(service.triage_review(NS))
        res = service.judge_batch(
            NS, [{"kind": "node", "node_id": A, "verdict": "reject"}],
            dry_run=False)
        assert res["results"][0]["reason"] == "recommendation_mismatch"
        assert res["applied"] == 0

    def test_no_recommendation_and_already_judged(self, svc):
        service, engine, *_ = svc
        run(service.triage_review(NS))
        # 트리아지 이후 들어온 노드 — 추천 없음
        engine.graph.add_node("Term:신규", type="Term", name="신규",
                              source="약관.pdf")
        res = service.judge_batch(
            NS, [{"kind": "node", "node_id": "Term:신규",
                  "verdict": "confirm"}])
        assert res["results"][0]["reason"] == "no_recommendation"
        # 이미 판정된 노드
        service.judge_batch(
            NS, [{"kind": "node", "node_id": A, "verdict": "confirm"}],
            dry_run=False)
        res2 = service.judge_batch(
            NS, [{"kind": "node", "node_id": A, "verdict": "confirm"}],
            dry_run=False)
        assert res2["results"][0]["reason"] == "already_judged"

    def test_apply_delegates_to_existing_paths(self, svc):
        """confirm_node/reject_node 위임 — 묘비·그래프 제거·감사가 그 경로의
        계약대로 일어난다. reject 사유 미기재 시 추천 rationale 로 폴백."""
        service, engine, *_ = svc
        run(service.triage_review(NS))
        res = service.judge_batch(
            NS, [{"kind": "node", "node_id": A, "verdict": "confirm"},
                 {"kind": "node", "node_id": B, "verdict": "reject"}],
            actor="human", dry_run=False)
        assert res["applied"] == 2
        from ontology.core.review_store import get_review_store
        reviews = get_review_store(NS)
        assert reviews.is_confirmed(A)
        assert reviews.is_rejected(B)
        assert B not in engine.graph          # reject_node 가 노드를 제거했다
        reject_ev = [h for h in reviews.history(node_id=B)
                     if h["action"] == "reject"][0]
        assert reject_ev["reason"] == "원문 어디에도 없는 사실"   # rationale 폴백

    def test_lifecycle_blocked_in_preview_and_apply(self, svc):
        """active 노드 거절은 미리보기에서도 blocked — 적용 단계에서야
        거부되면 검수 계획이 헛돈다 (생애주기 관문의 사전 반영)."""
        service, engine, *_ = svc
        run(service.triage_review(NS))
        engine.graph.nodes[B]["lifecycle"] = "active"
        item = [{"kind": "node", "node_id": B, "verdict": "reject"}]
        prev = service.judge_batch(NS, item)
        appl = service.judge_batch(NS, item, dry_run=False)
        assert prev["results"][0]["status"] == "blocked"
        assert appl["results"][0]["status"] == "blocked"
        assert prev["results"][0]["reason"] == "lifecycle_protected"

    def test_duplicate_items_in_batch_previewed_like_apply(self, svc):
        """같은 배치의 중복 항목 — 적용에선 둘째가 already_judged 로 걸리므로
        미리보기도 같게 예측해야 미리보기 == 적용이 성립한다."""
        service, *_ = svc
        run(service.triage_review(NS))
        items = [{"kind": "node", "node_id": A, "verdict": "confirm"},
                 {"kind": "node", "node_id": A, "verdict": "confirm"}]
        prev = service.judge_batch(NS, items)
        assert prev["results"][0]["status"] == "would_confirm"
        assert prev["results"][1]["reason"] == "already_judged"
        appl = service.judge_batch(NS, items, dry_run=False)
        assert appl["results"][0]["status"] == "confirmed"
        assert appl["results"][1]["reason"] == "already_judged"

    def test_relation_judge_roundtrip(self, svc):
        """관계 항목은 approve/reject_relations 위임 — 그쪽 관문(인용 재검증·
        묘비)이 그대로 작동한다. 미리보기 == 적용."""
        service, engine, *_, c1, c2 = svc
        run(service.triage_review(NS, include_relations=True))
        approve = {"kind": "relation", "subject": E1, "predicate": "covers",
                   "object": E2, "chunk_id": c2,
                   "evidence_quote": "암 은 계약서 가 보장한다",
                   "verdict": "approve"}
        reject = {"kind": "relation", "subject": E1, "predicate": "covers",
                  "object": E3, "verdict": "reject", "reason": "약한 근거"}
        no_reason = {"kind": "relation", "subject": E1, "predicate": "covers",
                     "object": E3, "verdict": "reject"}
        prev = service.judge_batch(NS, [approve, reject])
        assert [r["status"] for r in prev["results"]] == \
            ["would_approve", "would_reject"]
        res = service.judge_batch(NS, [approve, reject, no_reason],
                                  dry_run=False)
        statuses = [r["status"] for r in res["results"]]
        assert statuses[:2] == ["approved", "rejected"]
        assert res["applied"] == 2
        # 이유 없는 관계 기각은 skip — reject_relations 의 규칙 그대로
        assert res["results"][2]["status"] == "skipped"
        assert res["results"][2]["reason"] == "reason_required"
        # 승인은 실제 엣지가 됐고, 기각은 묘비가 됐다 (위임 확인)
        assert any(a.get("predicate") == "covers"
                   for a in (engine.graph.get_edge_data(E1, E2) or {}).values())
        from ontology.core.review_store import get_review_store
        assert get_review_store(NS).relation_rejection(E1, "covers", E3)
        # 묘비된 트리플의 재승인은 위임된 관문이 막는다
        res2 = service.judge_batch(
            NS, [{**reject, "verdict": "approve", "chunk_id": c2,
                  "evidence_quote": "특약 도 있다"}], dry_run=False)
        assert res2["results"][0]["reason"] == "relation_rejected"

    def test_quote_revalidated_through_delegation(self, svc):
        """지어낸 인용은 위임된 approve_relations 관문이 걸러낸다 — 본문 불신."""
        service, *_, c1, c2 = svc
        res = service.judge_batch(
            NS, [{"kind": "relation", "subject": E1, "predicate": "covers",
                  "object": E2, "chunk_id": c2,
                  "evidence_quote": "원문에 없는 문장", "verdict": "approve"}],
            dry_run=False)
        assert res["results"][0]["status"] == "skipped"
        assert res["results"][0]["reason"] == "quote_not_found"

    def test_invalid_inputs(self, svc):
        service, *_ = svc
        assert service.judge_batch(NS, [])["error"] == "invalid"
        res = service.judge_batch(NS, [
            {"kind": "node", "node_id": "", "verdict": "confirm"},
            {"kind": "node", "node_id": A, "verdict": "maybe"},
            {"kind": "외계", "x": 1},
        ])
        reasons = [r["reason"] for r in res["results"]]
        assert reasons == ["invalid", "invalid", "unknown_kind"]
        assert service.judge_batch("없는ns", [{"kind": "node"}])["error"] == \
            "namespace_not_found"


# ─── 3. recommendation_quality (C2) ──────────────────────────────────

class TestRecommendationQuality:
    def test_agreement_from_full_log(self, svc):
        """트리아지 추천 → 사람 판정 → 일치율. 판정 없는 추천(dup 2건 —
        verdict 없음)은 n 에 들지 않는다."""
        service, *_ = svc
        run(service.triage_review(NS))
        service.judge_batch(
            NS, [{"kind": "node", "node_id": A, "verdict": "confirm"},
                 {"kind": "node", "node_id": B, "verdict": "reject"}],
            dry_run=False)
        res = service.recommendation_quality(NS)
        assert res["namespace"] == NS
        assert res["overall"] == {"n": 2, "agree": 2, "rate": 1.0}
        by = res["per_actor"]["triage"]["by_verdict"]
        assert by["confirm"]["agree"] == 1 and by["reject"]["agree"] == 1
        assert res["events_total"] >= 6   # recommend 4 + confirm/reject 2

    def test_empty_log_reports_none_not_zero(self, svc):
        service, *_ = svc
        res = service.recommendation_quality(NS)
        assert res["overall"]["n"] == 0
        assert res["overall"]["rate"] is None   # 0건은 None — 오보고 금지

    def test_namespace_not_found(self, svc):
        service, *_ = svc
        assert service.recommendation_quality("없는ns")["error"] == \
            "namespace_not_found"
