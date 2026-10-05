"""노드 병합 계획 — 되돌릴 수 없는 변경 전에 **무엇이 일어날지** 계산한다.

graph_health 가 찾아낸 중복(실측 9군집: `Disease:C50( 유방의 악성 신생물 )` ↔
`Disease:유방의 악성 신생물`, `납입최고` 6형제)을 사람이 합칠 때 쓴다.

**왜 계획을 따로 두는가**: 병합은 노드를 지운다 — 되돌릴 수 없다. 팔란티어가
온톨로지 변경을 브랜치→제안→승인으로 감싸는 이유와 같다. 계획은 순수 함수라
미리 보여줄 수 있고(dry-run), 사람이 승인한 뒤에야 적용한다.

**왜 별칭으로 흡수하는가**: 진 노드의 이름을 이긴 노드의 별칭으로 옮기면 축 4 의
질의 확장이 그 이름으로도 찾아온다 — 중복이 **동의어로 승격**되어 회수율이 오른다.
그냥 지우면 그 표기로 검색하던 질의를 잃는다.
"""

import networkx as nx
import pytest

from ontology.core.node_merge import plan_merge

WINNER = "Disease:C50( 유방의 악성 신생물 )"
LOSER = "Disease:유방의 악성 신생물"


class _Chunk:
    def __init__(self, chunk_id, node_ids):
        self.chunk_id = chunk_id
        self.node_ids = list(node_ids)


class _Case:
    def __init__(self, case_id, expected, accepted=None):
        self.case_id = case_id
        self.expected_node_id = expected
        self.accepted_nodes = list(accepted or [])

    def accepted_ids(self):
        return {self.expected_node_id, *self.accepted_nodes} - {""}


def _graph(*nodes):
    g = nx.MultiDiGraph()
    for node in nodes:
        name = node.split(":", 1)[-1]
        g.add_node(node, type=node.split(":", 1)[0], name=name)
    return g


# ── 입력 검증 ────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_winner_is_error(self):
        plan = plan_merge(_graph(LOSER), WINNER, [LOSER])
        assert plan["error"] == "winner_not_found"

    def test_missing_loser_is_error(self):
        plan = plan_merge(_graph(WINNER), WINNER, [LOSER])
        assert plan["error"] == "loser_not_found"

    def test_winner_in_losers_is_error(self):
        """자기 자신과 병합하면 노드가 사라진다 — 반드시 막아야 한다."""
        plan = plan_merge(_graph(WINNER), WINNER, [WINNER])
        assert plan["error"] == "winner_in_losers"

    def test_no_losers_is_error(self):
        assert plan_merge(_graph(WINNER), WINNER, [])["error"] == "no_losers"

    def test_valid_plan_has_no_error(self):
        assert "error" not in plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER])


# ── 엣지 재지정 ──────────────────────────────────────────────────────

class TestEdgeRepointing:
    def test_outgoing_edge_moves_to_winner(self):
        g = _graph(WINNER, LOSER, "X:x")
        g.add_edge(LOSER, "X:x", predicate="mentions")
        plan = plan_merge(g, WINNER, [LOSER])
        assert {"from": WINNER, "to": "X:x", "predicate": "mentions"} \
            in plan["edges_repointed"]

    def test_incoming_edge_moves_to_winner(self):
        g = _graph(WINNER, LOSER, "X:x")
        g.add_edge("X:x", LOSER, predicate="mentions")
        plan = plan_merge(g, WINNER, [LOSER])
        assert {"from": "X:x", "to": WINNER, "predicate": "mentions"} \
            in plan["edges_repointed"]

    def test_edge_between_winner_and_loser_becomes_self_loop_and_is_dropped(self):
        """진 노드가 이긴 노드를 가리키고 있었다면 병합 후 자기 참조다."""
        g = _graph(WINNER, LOSER)
        g.add_edge(LOSER, WINNER, predicate="relatedTo")
        plan = plan_merge(g, WINNER, [LOSER])
        assert plan["edges_repointed"] == []
        assert plan["edges_dropped"][0]["reason"] == "self_loop"

    def test_duplicate_edge_is_dropped_not_doubled(self):
        g = _graph(WINNER, LOSER, "X:x")
        g.add_edge(WINNER, "X:x", predicate="mentions")
        g.add_edge(LOSER, "X:x", predicate="mentions")
        plan = plan_merge(g, WINNER, [LOSER])
        assert plan["edges_repointed"] == []
        assert plan["edges_dropped"][0]["reason"] == "duplicate"

    def test_same_pair_different_predicate_is_not_duplicate(self):
        """MultiDiGraph 다 — (from,to) 가 같아도 술어가 다르면 다른 사실이다."""
        g = _graph(WINNER, LOSER, "X:x")
        g.add_edge(WINNER, "X:x", predicate="mentions")
        g.add_edge(LOSER, "X:x", predicate="hasCoverage")
        plan = plan_merge(g, WINNER, [LOSER])
        assert len(plan["edges_repointed"]) == 1
        assert plan["edges_repointed"][0]["predicate"] == "hasCoverage"

    def test_plan_does_not_mutate_the_graph(self):
        """계획은 순수해야 한다 — 미리보기가 데이터를 바꾸면 dry-run 이 아니다."""
        g = _graph(WINNER, LOSER, "X:x")
        g.add_edge(LOSER, "X:x", predicate="mentions")
        before = (g.number_of_nodes(), g.number_of_edges())
        plan_merge(g, WINNER, [LOSER])
        assert (g.number_of_nodes(), g.number_of_edges()) == before


# ── 별칭 흡수 ────────────────────────────────────────────────────────

class TestAliasAbsorption:
    def test_loser_name_becomes_alias(self):
        plan = plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER])
        assert "유방의 악성 신생물" in plan["aliases_added"]

    def test_loser_aliases_are_carried_over(self):
        g = _graph(WINNER, LOSER)
        g.nodes[LOSER]["aliases"] = ["유방암"]
        plan = plan_merge(g, WINNER, [LOSER])
        assert "유방암" in plan["aliases_added"]

    def test_winner_own_name_is_not_added_as_alias(self):
        g = _graph(WINNER, "Disease:C50(유방의 악성 신생물)")
        plan = plan_merge(g, WINNER, ["Disease:C50(유방의 악성 신생물)"])
        assert plan["aliases_added"] == []

    def test_alias_already_present_is_not_re_added(self):
        g = _graph(WINNER, LOSER)
        g.nodes[WINNER]["aliases"] = ["유방의 악성 신생물"]
        plan = plan_merge(g, WINNER, [LOSER])
        assert plan["aliases_added"] == []

    def test_near_duplicate_aliases_collapse(self):
        """`납입최고 (독촉 )` 과 `납입최고(독촉)` 을 둘 다 넣으면 별칭이 쓰레기가 된다."""
        g = _graph("T:납입최고", "T:납입최고 (독촉 )", "T:납입최고(독촉)")
        plan = plan_merge(g, "T:납입최고",
                          ["T:납입최고 (독촉 )", "T:납입최고(독촉)"])
        assert len(plan["aliases_added"]) == 1


# ── 청크 · 골든셋 재지정 ─────────────────────────────────────────────

class TestReferenceRewrites:
    def test_chunk_node_ids_are_rewritten(self):
        chunks = [_Chunk("c1", [LOSER, "X:x"]), _Chunk("c2", ["X:x"])]
        plan = plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER], chunks=chunks)
        assert plan["chunks_rewritten"] == ["c1"]

    def test_chunk_holding_both_does_not_duplicate_winner(self):
        """둘 다 가리키던 청크는 병합 후 이긴 노드를 두 번 갖게 된다."""
        chunks = [_Chunk("c1", [WINNER, LOSER])]
        plan = plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER], chunks=chunks)
        assert plan["chunks_rewritten"] == ["c1"]
        assert plan["chunk_node_ids"]["c1"] == [WINNER]

    def test_golden_case_expectation_is_relabeled(self):
        """라벨이 지워진 노드를 가리키면 골든셋 46건이 조용히 무효가 된다."""
        cases = [_Case("g1", LOSER), _Case("g2", "X:x")]
        plan = plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER], cases=cases)
        assert plan["golden_relabels"] == ["g1"]

    def test_golden_accepted_list_also_counts(self):
        cases = [_Case("g1", "X:x", accepted=[LOSER])]
        plan = plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER], cases=cases)
        assert plan["golden_relabels"] == ["g1"]

    def test_no_references_gives_empty_lists(self):
        plan = plan_merge(_graph(WINNER, LOSER), WINNER, [LOSER])
        assert plan["chunks_rewritten"] == [] and plan["golden_relabels"] == []


# ── 프로퍼티 충돌 ────────────────────────────────────────────────────

class TestPropertyConflicts:
    def test_differing_definition_is_reported(self):
        g = _graph(WINNER, LOSER)
        g.nodes[WINNER]["definition"] = "유방에 생긴 악성 신생물"
        g.nodes[LOSER]["definition"] = "여성 유방의 암"
        plan = plan_merge(g, WINNER, [LOSER])
        conflicts = plan["property_conflicts"]
        assert len(conflicts) == 1 and conflicts[0]["key"] == "definition"

    def test_identical_values_are_not_conflicts(self):
        g = _graph(WINNER, LOSER)
        g.nodes[WINNER]["definition"] = g.nodes[LOSER]["definition"] = "같음"
        assert plan_merge(g, WINNER, [LOSER])["property_conflicts"] == []

    def test_value_only_on_loser_is_adopted_not_a_conflict(self):
        """이긴 쪽이 비어 있으면 잃을 것이 없다 — 가져오는 게 정보 보존이다."""
        g = _graph(WINNER, LOSER)
        g.nodes[LOSER]["definition"] = "여성 유방의 암"
        plan = plan_merge(g, WINNER, [LOSER])
        assert plan["property_conflicts"] == []
        assert plan["properties_adopted"]["definition"] == "여성 유방의 암"

    def test_bookkeeping_keys_are_never_conflicts(self):
        """created_at·last_updated 가 다른 건 당연하다 — 사람에게 물을 일이 아니다."""
        g = _graph(WINNER, LOSER)
        g.nodes[WINNER]["created_at"] = "2026-01-01"
        g.nodes[LOSER]["created_at"] = "2026-02-02"
        assert plan_merge(g, WINNER, [LOSER])["property_conflicts"] == []


# ── 결정론 · 직렬화 ──────────────────────────────────────────────────

class TestContract:
    def _multi(self):
        g = _graph("T:a", "T:a ", "T:a  ", "X:x")
        g.add_edge("T:a ", "X:x", predicate="p")
        g.add_edge("T:a  ", "X:x", predicate="q")
        return g

    def test_deterministic_regardless_of_loser_order(self):
        g = self._multi()
        one = plan_merge(g, "T:a", ["T:a ", "T:a  "])
        two = plan_merge(g, "T:a", ["T:a  ", "T:a "])
        assert one == two

    def test_plan_is_json_serializable(self):
        import json
        json.dumps(plan_merge(self._multi(), "T:a", ["T:a ", "T:a  "]),
                   ensure_ascii=False)

    def test_multiple_losers_all_reported(self):
        plan = plan_merge(self._multi(), "T:a", ["T:a ", "T:a  "])
        assert plan["losers"] == sorted(["T:a ", "T:a  "])
        assert len(plan["edges_repointed"]) == 2


# ── 저장소 재지정 계약 ───────────────────────────────────────────────
# 계획만으로는 병합이 끝나지 않는다. 청크의 node_ids 와 골든셋 라벨이 지워진
# 노드를 계속 가리키면, 청크는 dangling 이 되고 골든셋 46건은 조용히 무효가 된다.

class TestChunkStoreRelabel:
    def _store(self, tmp_path):
        from ontology.builder.models import Chunk
        from ontology.core.chunk_store import ChunkStore
        store = ChunkStore(namespace="merge_t", path=tmp_path / "c.jsonl")
        store.add(Chunk(text="본문 하나", source="d.pdf", index=0,
                        char_start=0, char_end=5), node_ids=[LOSER, "X:x"])
        store.add(Chunk(text="본문 둘", source="d.pdf", index=1,
                        char_start=5, char_end=9), node_ids=["X:x"])
        return store

    def test_rewrites_node_ids(self, tmp_path):
        store = self._store(tmp_path)
        touched = store.relabel_node(LOSER, WINNER)
        assert len(touched) == 1
        assert WINNER in store.nodes_for_chunk(touched[0])
        assert LOSER not in store.nodes_for_chunk(touched[0])

    def test_updates_the_reverse_index(self, tmp_path):
        """_by_node 를 갱신하지 않으면 지워진 노드가 계속 근거를 돌려준다."""
        store = self._store(tmp_path)
        store.relabel_node(LOSER, WINNER)
        assert store.chunks_for_node(LOSER) == []
        assert len(store.chunks_for_node(WINNER)) == 1

    def test_merging_into_existing_ref_does_not_duplicate(self, tmp_path):
        from ontology.builder.models import Chunk
        from ontology.core.chunk_store import ChunkStore
        store = ChunkStore(namespace="merge_t2", path=tmp_path / "c2.jsonl")
        cid = store.add(Chunk(text="둘 다", source="d.pdf", index=0,
                              char_start=0, char_end=4),
                        node_ids=[WINNER, LOSER])
        store.relabel_node(LOSER, WINNER)
        assert store.nodes_for_chunk(cid) == [WINNER]
        assert len(store.chunks_for_node(WINNER)) == 1

    def test_unknown_node_is_a_noop(self, tmp_path):
        store = self._store(tmp_path)
        assert store.relabel_node("Nope:없음", WINNER) == []

    def test_survives_disk_round_trip(self, tmp_path):
        from ontology.core.chunk_store import ChunkStore
        store = self._store(tmp_path)
        store.relabel_node(LOSER, WINNER)
        store.save_to_disk()
        fresh = ChunkStore(namespace="merge_t", path=tmp_path / "c.jsonl")
        fresh.load_from_disk()
        assert len(fresh.chunks_for_node(WINNER)) == 1
        assert fresh.chunks_for_node(LOSER) == []


class TestGoldenSetRelabel:
    def _gs(self, tmp_path):
        from ontology.core.search_qa import GoldenSet
        gs = GoldenSet(namespace="merge_t", path=tmp_path / "g.jsonl")
        gs.add("유방암 보장되나?", LOSER, source="hand")
        gs.add("위암 보장되나?", "Disease:위암", source="hand")
        return gs

    def test_rewrites_expected_node_id(self, tmp_path):
        gs = self._gs(tmp_path)
        touched = gs.relabel_node(LOSER, WINNER)
        assert len(touched) == 1
        assert {c.expected_node_id for c in gs.cases()} == {WINNER, "Disease:위암"}

    def test_rewrites_accepted_list(self, tmp_path):
        from ontology.core.search_qa import GoldenSet
        gs = GoldenSet(namespace="merge_t3", path=tmp_path / "g3.jsonl")
        gs.add("q", "Disease:위암", accepted=[LOSER], source="hand")
        gs.relabel_node(LOSER, WINNER)
        assert gs.cases()[0].accepted == [WINNER]

    def test_does_not_duplicate_when_already_present(self, tmp_path):
        from ontology.core.search_qa import GoldenSet
        gs = GoldenSet(namespace="merge_t4", path=tmp_path / "g4.jsonl")
        gs.add("q", WINNER, accepted=[LOSER], source="hand")
        gs.relabel_node(LOSER, WINNER)
        assert gs.cases()[0].accepted == []       # expected 와 같아졌으므로 흡수
        assert gs.cases()[0].accepted_ids() == {WINNER}

    def test_unknown_node_is_a_noop(self, tmp_path):
        assert self._gs(tmp_path).relabel_node("Nope:없음", WINNER) == []

    def test_survives_disk_round_trip(self, tmp_path):
        """추가전용 로그라 relabel 도 **이벤트**여야 한다 — 파일을 고쳐 쓰면
        '로그가 원본'이라는 골든셋의 계약이 깨진다."""
        from ontology.core.search_qa import GoldenSet
        gs = self._gs(tmp_path)
        gs.relabel_node(LOSER, WINNER)
        fresh = GoldenSet(namespace="merge_t", path=tmp_path / "g.jsonl")
        fresh.load_from_disk()
        assert {c.expected_node_id for c in fresh.cases()} == \
               {WINNER, "Disease:위암"}

    def test_confirmed_status_survives_relabel(self, tmp_path):
        gs = self._gs(tmp_path)
        case_id = [c.case_id for c in gs.cases() if c.expected_node_id == LOSER][0]
        gs.confirm(case_id)
        gs.relabel_node(LOSER, WINNER)
        assert gs.cases(status="confirmed")[0].expected_node_id == WINNER
