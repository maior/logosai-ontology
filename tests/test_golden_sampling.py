"""골든셋 노드 샘플링 — **자가 코퍼스를 대표해야 한다.**

**실측이 이 기능을 요구했다.** PROJ-A 에서 커버리지를 18.5% → 25.2% 로 올리고
(노드 +202, 근거 링크 +218) 재측정했는데 **지표가 소수점까지 불변**이었다.

원인: `generate_golden_cases` 가 노드를 **삽입 순서 상위 N개**로 골랐다
(`for node_id, attrs in graph.nodes(data=True)` → `break`). 그래서 31 케이스가
상위 25 노드에서 나왔고, 새로 만든 202개와 회복한 청크 30개는 그 영역이 아니었다.

  네임스페이스        청크    케이스   대표성
  ─────────────────────────────────────────────────────
  ins_cancer_demo      92      46     사실상 전 조문
  PROJ-A               432      31     상위 25 노드에 국소

즉 자가 좁은 곳만 보고 있어서 **개선이 보이지 않았다.** 지표 불변은 "개선이
없었다"가 아니라 "재지 못했다"였다.

**해결: 서로 다른 청크에서 고른다.** 노드를 근거 청크로 그룹화해 라운드로빈으로
뽑으면 N 케이스가 N 개 청크를 대표한다. 무작위가 아닌 이유는 결정론성 —
같은 그래프에서 두 번 뽑으면 같아야 재현 가능한 측정이 된다.
"""
import pytest

from ontology.core.golden_sampling import sample_nodes_for_generation


class _Chunk:
    def __init__(self, chunk_id, node_ids):
        self.chunk_id = chunk_id
        self.node_ids = list(node_ids)


def _graph(n=9):
    import networkx as nx
    g = nx.MultiDiGraph()
    for i in range(1, n + 1):
        g.add_node(f"T:{i}", type="T", name=f"이름{i}",
                   definition=f"정의 {i}")
    return g


def _chunks():
    # c1 에 3개, c2 에 3개, c3 에 3개 — 삽입 순서로 자르면 c1 만 덮는다
    return [_Chunk("c1", ["T:1", "T:2", "T:3"]),
            _Chunk("c2", ["T:4", "T:5", "T:6"]),
            _Chunk("c3", ["T:7", "T:8", "T:9"])]


class TestChunkSpread:
    def test_spreads_across_chunks(self):
        """**이 함수의 존재 이유.** 3개를 뽑으면 3개 청크에서 하나씩이어야 한다 —
        삽입 순서로 자르면 c1 의 T:1·T:2·T:3 만 나온다."""
        picked = sample_nodes_for_generation(_graph(), _chunks(), limit=3)
        ids = [p["node_id"] for p in picked]
        by_chunk = {c.chunk_id: set(c.node_ids) for c in _chunks()}
        covered = {cid for cid, ns in by_chunk.items()
                   if ns & set(ids)}
        assert len(covered) == 3, f"청크 {len(covered)}개만 덮었다: {ids}"

    def test_round_robin_before_second_pass(self):
        """청크 수보다 많이 뽑으면 한 바퀴 돈 뒤 두 번째를 채운다."""
        picked = sample_nodes_for_generation(_graph(), _chunks(), limit=6)
        assert len(picked) == 6
        by_chunk = {c.chunk_id: set(c.node_ids) for c in _chunks()}
        for cid, ns in by_chunk.items():
            assert len(ns & {p["node_id"] for p in picked}) == 2

    def test_deterministic(self):
        """무작위면 두 번 측정이 달라져 비교가 불가능해진다."""
        a = sample_nodes_for_generation(_graph(), _chunks(), limit=5)
        b = sample_nodes_for_generation(_graph(), _chunks(), limit=5)
        assert [x["node_id"] for x in a] == [x["node_id"] for x in b]

    def test_limit_zero_means_all(self):
        assert len(sample_nodes_for_generation(_graph(), _chunks(), limit=0)) == 9


class TestEligibility:
    def test_requires_definition(self):
        """패러프레이즈의 재료가 없으면 이름을 안 쓰고 물을 수 없다 (기존 계약)."""
        g = _graph()
        del g.nodes["T:1"]["definition"]
        picked = sample_nodes_for_generation(g, _chunks(), limit=9)
        assert "T:1" not in [p["node_id"] for p in picked]

    def test_orphan_nodes_are_included_last(self):
        """근거 청크가 없는 노드도 노드 채점의 대상이다 — 빼면 그 영역을
        영원히 못 잰다. 다만 청크 분산의 뒤에 둔다."""
        g = _graph()
        g.add_node("T:고아", type="T", name="고아", definition="정의")
        picked = sample_nodes_for_generation(g, _chunks(), limit=10)
        ids = [p["node_id"] for p in picked]
        assert "T:고아" in ids and ids.index("T:고아") >= 9

    def test_view_shape_matches_generator_contract(self):
        picked = sample_nodes_for_generation(_graph(), _chunks(), limit=1)
        assert set(picked[0]) >= {"node_id", "name", "type", "definition"}

    def test_missing_chunks_falls_back_to_all_nodes(self):
        """청크가 없는 네임스페이스에서도 생성은 되어야 한다."""
        picked = sample_nodes_for_generation(_graph(), [], limit=4)
        assert len(picked) == 4

    def test_dangling_chunk_refs_ignored(self):
        """청크가 지워진 노드를 가리켜도 죽지 않는다."""
        chunks = [_Chunk("c1", ["T:1", "T:없는것"])]
        picked = sample_nodes_for_generation(_graph(), chunks, limit=3)
        assert "T:없는것" not in [p["node_id"] for p in picked]

    def test_broken_graph_never_raises(self):
        class Boom:
            def nodes(self, data=False):
                raise RuntimeError("down")

        assert sample_nodes_for_generation(Boom(), _chunks(), limit=3) == []
