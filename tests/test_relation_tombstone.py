"""
관계판 묘비 (기각 관계 registry) — A1 회귀 계약.

근거: docs/review-collaboration-architecture.html §3 + 실측 (관계 백필 기각
10건 중 8건이 이전 라운드에서 이미 기각된 패턴의 재출현 — 기각이 응답 JSON
으로만 반환되고 어디에도 기록되지 않았기 때문).

계약:
- 로그가 원본 — relation_reject 이벤트를 replay 하면 묘비가 복원된다.
- 나중 판정이 이긴다 — 같은 트리플의 relation_approve 가 묘비를 걷는다.
- 키는 트리플 (subject, predicate, object) — 방향이 있고, 술어가 다르면
  다른 주장이다. 정규화는 인용 검증과 같은 자(_squash_ws) 를 쓴다.
- scope: "triple"(기본 — 주장 자체가 거짓) | "evidence"(이 인용만).
"""

import pytest

from ontology.core.review_store import ReviewStore


@pytest.fixture
def store(tmp_path):
    return ReviewStore(namespace="rel", path=tmp_path / "r.jsonl")


S, P, O = "Disease:C50", "coversDisease", "Disease:C78.0"


class TestRelationTombstone:
    def test_reject_registers_and_lookup(self, store):
        store.reject_relation(S, P, O, chunk_id="c1",
                              reason="전이를 보장으로 왜곡", scope="triple")
        rej = store.relation_rejection(S, P, O)
        assert rej is not None
        assert rej["reason"] == "전이를 보장으로 왜곡"
        assert rej["scope"] == "triple"
        assert rej["chunk_id"] == "c1"
        assert len(store.rejected_relations()) == 1

    def test_direction_matters(self, store):
        """(A,p,B) 기각이 (B,p,A) 를 잡으면 안 된다 — frozenset 키 mutant 사살."""
        store.reject_relation(S, P, O, chunk_id="c1", reason="r")
        assert store.relation_rejection(O, P, S) is None

    def test_predicate_matters(self, store):
        """같은 두 노드라도 술어가 다르면 다른 주장 — 쌍-키 mutant 사살."""
        store.reject_relation(S, P, O, chunk_id="c1", reason="r")
        assert store.relation_rejection(S, "hasCondition", O) is None

    def test_whitespace_variant_is_same_triple(self, store):
        """공백 변형 재제안은 걸려야 한다 — raw 문자열 키 mutant 사살.
        '같다'의 정의는 인용 검증(_squash_ws)과 한 벌이다."""
        store.reject_relation("Disease:C50( 유방의 악성 신생물 )", P, O,
                              chunk_id="c1", reason="r")
        assert store.relation_rejection(
            "Disease:C50(  유방의   악성 신생물 )", P, O) is not None

    def test_later_approve_lifts_tombstone(self, store):
        """나중 판정이 이긴다 — approve 이벤트가 같은 트리플 묘비를 걷는다.
        (service.approve_relations 가 남기는 relation_approve 와 같은 모양)"""
        store.reject_relation(S, P, O, chunk_id="c1", reason="r")
        store.record(action="relation_approve", node_id=S,
                     after={"predicate": P, "target": O, "chunk_id": "c2",
                            "evidence_quote": "…"})
        assert store.relation_rejection(S, P, O) is None

    def test_replay_restores_final_state(self, tmp_path):
        """reject→approve→reject 순서 재생 — 로그 순서가 곧 시간이다."""
        path = tmp_path / "r.jsonl"
        s1 = ReviewStore(namespace="rel", path=path)
        s1.reject_relation(S, P, O, chunk_id="c1", reason="1차 기각")
        s1.record(action="relation_approve", node_id=S,
                  after={"predicate": P, "target": O})
        s1.reject_relation(S, P, O, chunk_id="c3", reason="재기각")

        s2 = ReviewStore(namespace="rel", path=path)
        s2.load_from_disk()
        rej = s2.relation_rejection(S, P, O)
        assert rej is not None and rej["reason"] == "재기각"

    def test_malformed_events_ignored(self, store):
        store.record(action="relation_reject", node_id=S, after={})  # 필드 없음
        store.record(action="relation_approve", node_id=S, after=None)
        assert store.rejected_relations() == []

    def test_default_scope_is_triple(self, store):
        store.reject_relation(S, P, O, chunk_id="c1", reason="r")
        assert store.relation_rejection(S, P, O)["scope"] == "triple"


class TestLoadClearsDerivedState:
    """이번 조사가 발견한 기존 결함의 회귀 계약: load_from_disk 가
    _reclassified·_structural_types 를 clear 하지 않아 재로드 시 중복
    누적됐다. 신규 _rejected_relations 포함 전 파생 상태가 clear 대상."""

    def test_double_load_does_not_duplicate(self, tmp_path):
        path = tmp_path / "r.jsonl"
        s = ReviewStore(namespace="rel", path=path)
        s.record(action="reclassify", node_id="B:x",
                 after={"old_id": "A:x", "new_id": "B:x"})
        s.record(action="type_declared", node_id="type:Clause",
                 after={"type": "Clause"})
        s.reject_relation(S, P, O, chunk_id="c1", reason="r")

        s.load_from_disk()
        s.load_from_disk()  # 두 번 로드 — 상태는 로그 1회분과 같아야 한다
        assert s.reclassified_map() == {"A:x": "B:x"}
        assert s.structural_types() == {"Clause"}
        assert len(s.rejected_relations()) == 1

    def test_stale_state_gone_after_reload_of_other_log(self, tmp_path):
        """다른(짧은) 로그를 로드하면 이전 로그의 파생 상태가 남지 않는다."""
        s = ReviewStore(namespace="rel", path=tmp_path / "a.jsonl")
        s.record(action="reclassify", node_id="B:x",
                 after={"old_id": "A:x", "new_id": "B:x"})
        (tmp_path / "b.jsonl").write_text("")
        s.load_from_disk(tmp_path / "b.jsonl")
        assert s.reclassified_map() == {}
