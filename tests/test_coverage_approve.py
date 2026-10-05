"""커버리지 gap 을 **노드 + 근거 링크**로 승인한다 (회복 쓰기 경로).

동기(실측): ins_cancer_demo 의 추출 커버리지는 29.3% 였고, only_unlinked 로
겨냥해 보니 청크 10개에서 gap 34건이 나왔다 — `InsuranceTerm:암진단비`,
`Disease:암` 처럼 암보험 약관의 **가장 중심 개체**가 빠져 있었다. 즉 낮은
커버리지는 "형식적 문구라서"가 아니라 진짜 추출 실패였다. 그런데 gap 은
감사 로그(action=coverage_gap)에만 남고 그래프로 돌아오는 길이 없었다.

이 테스트가 고정하는 계약 두 가지:

1. **승인은 요청 본문을 신뢰하지 않는다.** coverage_checker 의 "환각 0" 은
   원문 대조가 만든 성질이고, 엔드포인트가 받는 {name, type, chunk_id} 는
   그 성질을 물려받지 않는다. 승인 경로가 같은 검증(_quote_in_chunks·허용
   타입)을 다시 통과시켜야 한다 — 안 하면 승인이 임의 노드 주입구가 된다.

2. **노드만 만들면 안 된다.** 링크 없는 노드는 고아 노드(graph_health 가
   세는 그것)를 늘리는 것이고, gap 회복의 목적인 근거 사슬을 배신한다.
   create_node 를 그대로 못 쓰는 이유가 이것이다.
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.core.chunk_index import reset_chunk_indices
from ontology.core.chunk_store import reset_chunk_stores
from ontology.core.review_store import reset_review_stores
from ontology.core.search_qa import reset_golden_sets

API = "/api/v1/ontology"
BODY = "제6조 【암진단비의 지급】 회사는 암진단비를 지급합니다. " + "본문 " * 40


@pytest.fixture(autouse=True)
def _clean():
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()
    yield
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()


@pytest.fixture()
def env(tmp_path):
    """노드 1개(타입 확보) + 본문 청크 1개인 최소 네임스페이스."""
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import get_chunk_store
    from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    service = OntologyBuilderService(data_dir=tmp_path)
    app = FastAPI()
    app.include_router(server_router.router, prefix=API)
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    ns = f"apprns_{uuid.uuid4().hex[:8]}"

    engine = get_knowledge_graph_engine(ns)
    # 허용 타입은 그래프에 실존하는 타입에서만 온다 — 이 노드가 그 근거다
    engine.graph.add_node("InsuranceTerm:해지", type="InsuranceTerm", name="해지")
    engine.save_to_disk()

    store = get_chunk_store(ns)
    cid = store.add(Chunk(text=BODY, source="약관.pdf", index=0,
                          char_start=0, char_end=len(BODY)))
    return TestClient(app), ns, cid, engine, store


def _approve(client, ns, gaps, **kw):
    return client.post(f"{API}/graphs/{ns}/review/coverage/approve",
                       json={"gaps": gaps, **kw})


def _gap(cid, name="암진단비", type_="InsuranceTerm"):
    return {"name": name, "type": type_, "chunk_id": cid}


# ─── 1. 요청 본문 불신 — 검사기의 검증을 다시 통과시킨다 ──────────────

class TestUntrustedRequest:
    def test_name_absent_from_source_is_rejected(self, env):
        """원문에 없는 이름을 승인하면 승인이 곧 환각 주입구다."""
        client, ns, cid, engine, _ = env
        body = _approve(client, ns, [_gap(cid, name="심근경색진단비")],
                        dry_run=False).json()
        assert body["created"] == []
        assert body["skipped"][0]["reason"] == "quote_not_found"
        assert "InsuranceTerm:심근경색진단비" not in engine.graph

    def test_type_outside_graph_is_rejected(self, env):
        """승인이 스키마를 여는 뒷문이 되면 안 된다 —
        검사기가 제안할 수 있는 것과 정확히 같은 집합만 받는다."""
        client, ns, cid, engine, _ = env
        body = _approve(client, ns, [_gap(cid, type_="Weapon")],
                        dry_run=False).json()
        assert body["created"] == []
        assert body["skipped"][0]["reason"] == "type_not_allowed"
        assert "Weapon:암진단비" not in engine.graph

    def test_unknown_chunk_is_rejected(self, env):
        """근거 청크가 없으면 provenance 가 없다 — 그건 gap 승인이 아니다."""
        client, ns, cid, engine, _ = env
        body = _approve(client, ns, [_gap("nosuchchunk")],
                        dry_run=False).json()
        assert body["created"] == []
        assert body["skipped"][0]["reason"] == "chunk_not_found"

    def test_quote_match_tolerates_whitespace(self, env):
        """검사기와 같은 정규화 규칙을 써야 한다 (두 벌 두면 정의가 갈라진다).
        _squash_ws 는 공백을 **하나로 줄일 뿐** 제거하지 않는다."""
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid, name="암진단비의  지급")],
                        dry_run=False).json()
        # 노드 이름도 정규화된 형태다 — 공백만 다른 중복 노드를 만들면
        # graph_health 가 세는 중복 클러스터를 스스로 늘리는 것이다
        assert body["created"] == ["InsuranceTerm:암진단비의 지급"]

    def test_blank_candidate_is_rejected(self, env):
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [{"name": "  ", "type": "InsuranceTerm",
                                      "chunk_id": cid}],
                        dry_run=False).json()
        assert body["created"] == []
        assert body["skipped"][0]["reason"] == "invalid"


# ─── 2. 판정을 조용히 뒤집지 않는다 ──────────────────────────────────

class TestTombstoneRespected:
    def test_rejected_node_is_not_resurrected(self, env):
        """34건 일괄 승인 중 하나가 과거 거절이면, 부활은 검수자 모르게
        판정을 뒤집는 것이다. 되살리려면 명시적 create 를 쓰면 된다."""
        from ontology.core.review_store import get_review_store
        client, ns, cid, engine, _ = env
        get_review_store(ns).reject("InsuranceTerm:암진단비", reason="오추출",
                                    actor="reviewer")
        body = _approve(client, ns, [_gap(cid)], dry_run=False).json()
        assert body["created"] == []
        assert body["skipped"][0]["reason"] == "tombstoned"
        assert "InsuranceTerm:암진단비" not in engine.graph


# ─── 3. 노드 + 링크가 함께 생긴다 ────────────────────────────────────

class TestCreatesNodeAndLink:
    def test_creates_node_with_id_convention(self, env):
        client, ns, cid, engine, _ = env
        body = _approve(client, ns, [_gap(cid)], dry_run=False).json()
        assert body["created"] == ["InsuranceTerm:암진단비"]
        attrs = engine.graph.nodes["InsuranceTerm:암진단비"]
        assert attrs["type"] == "InsuranceTerm"
        assert attrs["name"] == "암진단비"

    def test_links_the_evidence_chunk(self, env):
        """이게 create_node 를 그대로 쓰지 않는 이유다 — 링크 없는 노드는 고아다."""
        client, ns, cid, _, store = env
        _approve(client, ns, [_gap(cid)], dry_run=False)
        linked = store.chunks_for_node("InsuranceTerm:암진단비")
        assert [c.chunk_id for c in linked] == [cid]

    def test_no_definition_is_fabricated(self, env):
        """원문에 있는 것은 이름뿐이다. 정의를 지어내면 병합에서 거부한 그 문제."""
        client, ns, cid, engine, _ = env
        _approve(client, ns, [_gap(cid)], dry_run=False)
        assert "definition" not in engine.graph.nodes["InsuranceTerm:암진단비"]

    def test_created_node_enters_the_review_queue(self, env):
        """이 노드의 내용은 LLM 이 원문에서 읽어 제안한 것이다 — 빌더 추출분과
        같은 출처 등급이므로 같은 검증 의무를 진다. 큐는 source 있는 노드만
        담으므로, source 를 안 넣으면 100개가 큐를 통째로 우회해 개별 검증을
        영구히 못 받는다 (수동 create_node 와 갈라지는 지점)."""
        client, ns, cid, engine, _ = env
        _approve(client, ns, [_gap(cid)], dry_run=False)
        assert engine.graph.nodes["InsuranceTerm:암진단비"]["source"] == "약관.pdf"
        queue = client.get(f"{API}/graphs/{ns}/review").json()
        assert "InsuranceTerm:암진단비" in [i["node_id"] for i in queue["items"]]

    def test_created_node_inherits_chunk_trust(self, env):
        """trust 는 청크(출처)의 성질이다 — 노드가 제 등급을 스스로 정하면
        trust 필터 워크플로우가 무의미해진다."""
        from ontology.builder.models import Chunk
        client, ns, _, engine, store = env
        cid = store.add(Chunk(text=BODY, source="요약서.pdf", index=9,
                              char_start=0, char_end=len(BODY)),
                        trust="summary")
        _approve(client, ns, [_gap(cid)], dry_run=False)
        assert engine.graph.nodes["InsuranceTerm:암진단비"]["trust"] == "summary"

    def test_reports_chunks_linked_count(self, env):
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid)], dry_run=False).json()
        assert body["chunks_linked"] == 1

    def test_flags_that_search_needs_reindex(self, env):
        """노드를 만들어도 시맨틱 색인은 갱신하지 않는다(임베딩 비용).
        그 사실을 숨기면 '만들었는데 검색이 안 된다'가 된다."""
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid)], dry_run=False).json()
        assert body["reindex_required"] is True

    def test_no_reindex_flag_when_nothing_created(self, env):
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid, name="없는이름")],
                        dry_run=False).json()
        assert body["reindex_required"] is False


# ─── 4. 이미 있는 노드 — 오류가 아니라 링크 ──────────────────────────

class TestExistingNode:
    def test_links_instead_of_failing(self, env):
        """노드는 있고 근거 링크만 없는 것도 실제 결함이다.
        duplicate 오류로 막으면 그 결함을 고칠 길이 없다."""
        client, ns, cid, engine, store = env
        engine.graph.add_node("InsuranceTerm:암진단비", type="InsuranceTerm",
                              name="암진단비", definition="원래 있던 정의")
        body = _approve(client, ns, [_gap(cid)], dry_run=False).json()
        assert body["created"] == []
        assert body["linked"] == ["InsuranceTerm:암진단비"]
        assert [c.chunk_id
                for c in store.chunks_for_node("InsuranceTerm:암진단비")] == [cid]

    def test_does_not_overwrite_existing_attrs(self, env):
        client, ns, cid, engine, _ = env
        engine.graph.add_node("InsuranceTerm:암진단비", type="InsuranceTerm",
                              name="암진단비", definition="원래 있던 정의")
        _approve(client, ns, [_gap(cid)], dry_run=False)
        assert engine.graph.nodes["InsuranceTerm:암진단비"]["definition"] \
            == "원래 있던 정의"

    def test_relink_is_idempotent(self, env):
        client, ns, cid, _, store = env
        _approve(client, ns, [_gap(cid)], dry_run=False)
        again = _approve(client, ns, [_gap(cid)], dry_run=False).json()
        assert again["chunks_linked"] == 0        # 이미 이어져 있다
        assert len(store.chunks_for_node("InsuranceTerm:암진단비")) == 1


# ─── 5. dry_run — 미리보기가 기본값 ──────────────────────────────────

class TestDryRun:
    def test_default_writes_nothing(self, env):
        """상한 없는 일괄 쓰기의 기본값이 '적용'이면 사고가 조용히 커진다
        (merge 와 같은 계약)."""
        client, ns, cid, engine, store = env
        body = _approve(client, ns, [_gap(cid)]).json()
        assert body["dry_run"] is True
        assert "InsuranceTerm:암진단비" not in engine.graph
        assert store.chunks_for_node("InsuranceTerm:암진단비") == []

    def test_preview_reports_what_would_happen(self, env):
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid)]).json()
        assert body["created"] == ["InsuranceTerm:암진단비"]

    def test_preview_counts_links_it_would_make(self, env):
        """0 으로 보고하면 '노드만 만들고 근거는 안 잇는다'로 읽힌다."""
        client, ns, cid, _, _ = env
        assert _approve(client, ns, [_gap(cid)]).json()["chunks_linked"] == 1

    def test_preview_matches_apply(self, env):
        """미리보기와 적용이 다른 수를 말하면 미리보기가 쓸모없다."""
        from ontology.builder.models import Chunk
        client, ns, cid, _, store = env
        other = store.add(Chunk(text="제7조 암진단비 지급 제한 " + "본문 " * 30,
                                source="약관.pdf", index=1,
                                char_start=200, char_end=400))
        gaps = [_gap(cid), _gap(other), _gap(cid, name="없는이름")]
        preview = _approve(client, ns, gaps).json()
        applied = _approve(client, ns, gaps, dry_run=False).json()
        for key in ("created", "linked", "chunks_linked", "reindex_required"):
            assert preview[key] == applied[key], key
        assert [s["reason"] for s in preview["skipped"]] \
            == [s["reason"] for s in applied["skipped"]]

    def test_preview_still_validates(self, env):
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid, name="심근경색")]).json()
        assert body["skipped"][0]["reason"] == "quote_not_found"


# ─── 6. 배치 위생 ────────────────────────────────────────────────────

class TestBatchHygiene:
    def test_duplicate_gaps_collapse(self, env):
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid), _gap(cid)],
                        dry_run=False).json()
        assert body["created"] == ["InsuranceTerm:암진단비"]

    def test_same_node_in_two_chunks_links_both(self, env):
        """실측(ins_cancer_demo): gap 123건 중 고유 노드는 100개였다 — 같은
        개체가 여러 조문에 걸쳐 나온다. node_id 로만 dedup 하면 첫 청크만
        이어지고 나머지 근거(23건, 19%)가 **보고도 없이** 버려진다.
        노드는 하나지만 근거는 여러 개다."""
        from ontology.builder.models import Chunk
        client, ns, cid, _, store = env
        other = store.add(Chunk(text="제7조 암진단비 지급 제한 " + "본문 " * 30,
                                source="약관.pdf", index=1,
                                char_start=200, char_end=400))
        body = _approve(client, ns, [_gap(cid), _gap(other)],
                        dry_run=False).json()
        assert body["created"] == ["InsuranceTerm:암진단비"]   # 노드는 한 번만
        assert body["chunks_linked"] == 2                     # 근거는 둘 다
        assert {c.chunk_id
                for c in store.chunks_for_node("InsuranceTerm:암진단비")} \
            == {cid, other}

    def test_identical_gap_twice_still_collapses(self, env):
        """같은 (노드, 청크) 쌍의 중복 제시는 여전히 한 번이다."""
        client, ns, cid, _, _ = env
        body = _approve(client, ns, [_gap(cid), _gap(cid)],
                        dry_run=False).json()
        assert body["chunks_linked"] == 1

    def test_one_bad_candidate_does_not_kill_the_batch(self, env):
        """후보 단위 드롭 — 검사기와 같은 규정이다."""
        client, ns, cid, _, _ = env
        body = _approve(client, ns,
                        [_gap(cid, name="없는이름"), _gap(cid)],
                        dry_run=False).json()
        assert body["created"] == ["InsuranceTerm:암진단비"]
        assert len(body["skipped"]) == 1

    def test_empty_batch_is_a_client_error(self, env):
        client, ns, _, _, _ = env
        assert _approve(client, ns, []).status_code == 400


# ─── 7. 감사 이력 ────────────────────────────────────────────────────

class TestAudit:
    def test_records_approval_with_provenance(self, env):
        """gap 발견(coverage_gap)과 승인(coverage_approve)이 짝을 이뤄야
        '무엇을 놓쳤고 무엇을 되찾았나'를 추적할 수 있다."""
        client, ns, cid, _, _ = env
        _approve(client, ns, [_gap(cid)], dry_run=False)
        history = client.get(f"{API}/graphs/{ns}/review/history").json()
        events = [e for e in history["history"]
                  if e["action"] == "coverage_approve"]
        assert len(events) == 1
        assert events[0]["node_id"] == "InsuranceTerm:암진단비"
        assert events[0]["after"]["chunk_id"] == cid
        assert events[0]["after"]["outcome"] == "created"

    def test_dry_run_leaves_no_audit_trace(self, env):
        client, ns, cid, _, _ = env
        _approve(client, ns, [_gap(cid)])
        history = client.get(f"{API}/graphs/{ns}/review/history").json()
        assert not [e for e in history["history"]
                    if e["action"] == "coverage_approve"]


# ─── 8. 경계 ─────────────────────────────────────────────────────────

class TestGuards:
    def test_protected_namespace_is_read_only(self, env):
        client, _, cid, _, _ = env
        resp = client.post(f"{API}/graphs/default/review/coverage/approve",
                           json={"gaps": [_gap(cid)], "dry_run": False})
        assert resp.status_code == 403

    def test_missing_namespace_is_404(self, env):
        client, _, cid, _, _ = env
        resp = _approve(client, "nosuchns_xyz", [_gap(cid)], dry_run=False)
        assert resp.status_code == 404
