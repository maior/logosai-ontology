"""노드 생애주기 — 소멸을 **통제**하는 장치. 팔란티어 대조가 지목한 결손.

**방향을 오해하지 말 것**: 생애주기는 삭제를 가능하게 하는 게 아니라 막는다.
`active` 는 삭제도 개명도 불가이고, `deprecated` 는 사유 + 기한을 요구한다.

우리에게 필요한 이유는 구체적이다 — 파괴 경로가 이미 있는데 통제가 없다:
`reject_node`(묘비+제거) · `merge_nodes`(진 노드 삭제) · 그리고 `Clause` 재분류는
`{type}:{name}` id 를 바꾸므로 그 자체가 개명이다.
"""
import pytest

from ontology.core.lifecycle import (ACTIVE, DEPRECATED, EXPERIMENTAL,
                                     apply_transition, can_destroy,
                                     current_state, overdue, state_counts,
                                     validate_transition)


class TestCurrentState:
    def test_unset_is_experimental(self):
        """미지정 노드가 191개다 — 기본값이 관문을 통과해야 소급 적용이 불필요하다."""
        assert current_state({}) == EXPERIMENTAL
        assert current_state(None) == EXPERIMENTAL

    def test_reads_the_attr(self):
        assert current_state({"lifecycle": "active"}) == ACTIVE

    def test_case_and_space_tolerant(self):
        assert current_state({"lifecycle": " Active "}) == ACTIVE

    def test_unknown_value_falls_back(self):
        """알 수 없는 값을 그대로 흘리면 관문이 '차단도 허용도 아닌' 상태가 된다."""
        assert current_state({"lifecycle": "promoted"}) == EXPERIMENTAL


class TestCanDestroy:
    def test_active_is_protected(self):
        ok, why = can_destroy({"lifecycle": ACTIVE})
        assert ok is False and "active" in why

    def test_experimental_is_free(self):
        assert can_destroy({})[0] is True

    def test_deprecated_is_allowed(self):
        """**이미 선언된 경로**다 — 사유·기한을 밝히는 것이 deprecate 의 대가이고,
        그 뒤 삭제를 다시 막으면 영구히 못 지우는 노드가 된다."""
        assert can_destroy({"lifecycle": DEPRECATED})[0] is True


class TestTransitions:
    def test_experimental_to_active(self):
        assert validate_transition(EXPERIMENTAL, ACTIVE)[0] is True

    def test_active_to_deprecated_needs_reason_and_sunset(self):
        assert validate_transition(ACTIVE, DEPRECATED)[0] is False
        assert validate_transition(ACTIVE, DEPRECATED, reason="대체됨")[0] is False
        ok, _ = validate_transition(ACTIVE, DEPRECATED, reason="대체됨",
                                    sunset="2026-12-31")
        assert ok is True

    def test_active_cannot_go_back_to_experimental(self):
        """**이 표의 요점.** 허용하면 active → experimental → 삭제 우회가 열린다."""
        ok, why = validate_transition(ACTIVE, EXPERIMENTAL)
        assert ok is False and "deprecated" in why

    def test_deprecated_can_be_reverted(self):
        """번복은 명시적 행위이므로 허용한다 — 잘못 내린 것을 되돌릴 길이 필요하다."""
        assert validate_transition(DEPRECATED, ACTIVE)[0] is True
        assert validate_transition(DEPRECATED, EXPERIMENTAL)[0] is True

    def test_same_state_is_rejected(self):
        assert validate_transition(ACTIVE, ACTIVE)[0] is False

    def test_unknown_target_state(self):
        ok, why = validate_transition(EXPERIMENTAL, "promoted")
        assert ok is False and "promoted" in why

    def test_sunset_must_look_like_a_date(self):
        """'나중에' 같은 문자열을 통과시키면 기한 요구가 형식만 남는다."""
        assert validate_transition(ACTIVE, DEPRECATED, reason="r",
                                    sunset="나중에")[0] is False
        assert validate_transition(ACTIVE, DEPRECATED, reason="r",
                                    sunset="2026-1-1")[0] is False

    def test_unknown_from_state_is_treated_as_default(self):
        assert validate_transition("promoted", ACTIVE)[0] is True


class TestApplyTransition:
    def test_records_reason_sunset_and_superseder(self):
        attrs = {}
        apply_transition(attrs, DEPRECATED, reason="C50 으로 통합",
                         sunset="2026-12-31", superseded_by="Disease:C50",
                         at="2026-08-02T10:00:00")
        assert attrs["lifecycle"] == DEPRECATED
        assert attrs["sunset"] == "2026-12-31"
        assert attrs["superseded_by"] == "Disease:C50"
        assert attrs["lifecycle_at"] == "2026-08-02T10:00:00"

    def test_leaving_deprecated_clears_the_sunset(self):
        """남겨두면 active 노드가 옛 삭제 기한을 들고 다니고 화면이 거짓을 보여준다."""
        attrs = {}
        apply_transition(attrs, DEPRECATED, reason="r", sunset="2026-12-31",
                         superseded_by="X:1")
        apply_transition(attrs, ACTIVE)
        assert attrs["lifecycle"] == ACTIVE
        for key in ("sunset", "deprecated_reason", "superseded_by"):
            assert key not in attrs

    def test_superseder_is_optional_and_not_left_stale(self):
        attrs = {"superseded_by": "X:old"}
        apply_transition(attrs, DEPRECATED, reason="r", sunset="2026-12-31")
        assert "superseded_by" not in attrs


def _graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_node("A:1", type="A")                                  # 미지정
    g.add_node("A:2", type="A", lifecycle=ACTIVE)
    g.add_node("A:3", type="A", lifecycle=DEPRECATED,
               sunset="2026-01-01", deprecated_reason="옛것")
    g.add_node("A:4", type="A", lifecycle=DEPRECATED,
               sunset="2099-01-01", deprecated_reason="아직")
    g.add_node("A:5", type="A", lifecycle=DEPRECATED)             # 기한 없음
    return g


class TestReporting:
    def test_state_counts_includes_unset_as_experimental(self):
        """미지정을 별도 칸으로 나누면 '미지정 = 안전'으로 오해된다."""
        counts = state_counts(_graph())
        assert counts == {EXPERIMENTAL: 1, ACTIVE: 1, DEPRECATED: 3}

    def test_overdue_finds_past_sunsets(self):
        """기한을 적어두고 아무도 안 보면 '언젠가 정리하자'가 그대로 돌아온다."""
        rows = overdue(_graph(), today="2026-08-02")
        assert [r["node_id"] for r in rows] == ["A:3"]
        assert rows[0]["reason"] == "옛것"

    def test_future_sunset_is_not_overdue(self):
        assert "A:4" not in [r["node_id"]
                             for r in overdue(_graph(), today="2026-08-02")]

    def test_missing_sunset_is_not_reported_as_overdue(self):
        """기한 없는 deprecated 는 별개 결함이다 — 여기서 섞으면 원인이 흐려진다."""
        assert "A:5" not in [r["node_id"]
                             for r in overdue(_graph(), today="2026-08-02")]

    def test_deterministic_order(self):
        g = _graph()
        assert overdue(g, "2099-12-31") == overdue(g, "2099-12-31")

    def test_broken_graph_never_raises(self):
        class Boom:
            def nodes(self, data=False):
                raise RuntimeError("down")

        assert state_counts(Boom())[ACTIVE] == 0
        assert overdue(Boom(), "2026-08-02") == []
