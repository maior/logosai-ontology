"""시맨틱 색인의 유령 제거 — 그래프에서 사라진 노드는 검색에도 없어야 한다.

**실측으로 확인한 결함**: `build_from_graph` 는 upsert 만 하고, 그래프에서 사라진
노드의 벡터를 걷어내지 않았다. `remove()` 는 있는데 아무도 부르지 않았다.
그래서 노드를 지운 뒤 검색하면 **지워진 노드가 그대로 반환**된다.

병합(node_merge)뿐 아니라 **검수 거절(reject_node)도 노드를 지운다** — 즉 이 결함은
병합 기능보다 오래됐고, 거절된 개체가 계속 검색되고 있었다는 뜻이다.

발견 경로가 중요하다: 병합 직후 지표에서 semantic 채널만 나빠졌고(hit@1 0.3125→0.25)
retrieve 는 좋아져, 하마터면 "두 채널이 처음으로 갈라졌다"는 의미 있는 신호로
오보고할 뻔했다. `/reindex` 를 돌려도 그대로였고 **서버 재시작 후에야** 지표가
뒤집혔다 — 그게 staleness 라는 증거였다.

tier 별 상태(이 테스트가 고정하는 계약):
  · tier 0 memory (SemanticIndex) — 정리 안 했다 → **이 커밋에서 수정**
  · tier 1 npy (NpyBackend)       — 이미 정리하고 있었다 → 회귀 방지만
  · elasticsearch                 — 정리 안 한다(문서화된 잔존 결함, ES 없이 검증 불가)
"""

import networkx as nx
import numpy as np
import pytest

from ontology.core.semantic_index import SemanticIndex


def embed(texts):
    """결정적 가짜 임베더 — 첫 글자로 벡터를 만든다(모델 불필요)."""
    return np.array([[float(ord(t[0]) % 17), 1.0, 0.5] for t in texts],
                    dtype=np.float32)


def _graph(*specs):
    g = nx.MultiDiGraph()
    for node_id, node_type in specs:
        g.add_node(node_id, type=node_type, name=node_id.split(":", 1)[-1])
    return g


def _index(graph, node_types=None, embed_fn=embed):
    idx = SemanticIndex(embed_fn=embed_fn, auto_default=False)
    idx.build_from_graph(graph, node_types=node_types)
    return idx


class TestPruning:
    def test_removed_node_is_dropped(self):
        g = _graph(("Disease:유방암", "Disease"), ("Disease:위암", "Disease"))
        idx = _index(g)
        g.remove_node("Disease:유방암")
        idx.build_from_graph(g, node_types=None)
        assert len(idx) == 1

    def test_removed_node_is_not_searchable(self):
        """길이만 맞아도 소용없다 — 검색이 유령을 돌려주면 인용이 빈 화면이 된다."""
        g = _graph(("Disease:유방암", "Disease"), ("Disease:위암", "Disease"))
        idx = _index(g)
        g.remove_node("Disease:유방암")
        idx.build_from_graph(g, node_types=None)
        found = {h["node_id"] for h in idx.search("유방암", top_k=5)}
        assert "Disease:유방암" not in found

    def test_live_nodes_survive(self):
        g = _graph(("Disease:유방암", "Disease"), ("Disease:위암", "Disease"))
        idx = _index(g)
        g.remove_node("Disease:유방암")
        idx.build_from_graph(g, node_types=None)
        assert {h["node_id"] for h in idx.search("위암", top_k=5)} == {"Disease:위암"}

    def test_type_moved_out_of_scope_is_dropped(self):
        """타입 필터 밖으로 나간 노드도 그 색인에서는 사라진 것이다."""
        g = _graph(("A:x", "Disease"), ("A:y", "Disease"))
        idx = _index(g, node_types=["Disease"])
        g.nodes["A:x"]["type"] = "Other"
        idx.build_from_graph(g, node_types=["Disease"])
        assert {h["node_id"] for h in idx.search("x", top_k=5)} == {"A:y"}

    def test_prunes_even_without_an_embedder(self):
        """제거는 임베딩이 필요 없다 — 임베더가 없다고 유령을 남기면 안 된다."""
        g = _graph(("A:x", "Disease"), ("A:y", "Disease"))
        idx = _index(g)
        idx._embed_fn = None                      # 임베더 소실 상황 재현
        g.remove_node("A:x")
        idx.build_from_graph(g, node_types=None)
        assert len(idx) == 1

    def test_row_alignment_survives_pruning_a_middle_row(self):
        """remove() 는 뒤 행을 앞으로 당긴다 — 어긋나면 **다른 노드의 벡터**를 쓴다.

        검색 순위로 검사하지 않는다: 가짜 임베더가 판별력을 가지도록 만들기가
        어렵고(방향이 비슷하면 순위가 임의), 그러면 정렬이 깨져도 통과하는
        무의미한 테스트가 된다. 불변식 자체를 본다 — 각 노드의 행에 그 노드의
        벡터가 있는가.
        """
        from ontology.core.semantic_index import _normalize, compose_node_text

        g = _graph(("A:a", "T"), ("B:b", "T"), ("C:c", "T"))
        idx = _index(g)
        g.remove_node("B:b")                      # 가운데 행
        idx.build_from_graph(g, node_types=None)

        assert sorted(idx._ids) == ["A:a", "C:c"]
        for node_id in ("A:a", "C:c"):
            want = _normalize(np.atleast_2d(
                embed([compose_node_text(node_id, g.nodes[node_id])])))[0]
            got = idx._vectors[idx._row_of[node_id]]
            assert np.allclose(got, want), f"{node_id} 의 행이 다른 벡터를 가리킨다"


class TestContracts:
    def test_return_value_is_still_embedded_count(self):
        """기존 계약 유지 — 반환값은 '(재)임베딩한 노드 수'다(제거 수가 아니다)."""
        g = _graph(("A:x", "T"), ("A:y", "T"))
        idx = _index(g)
        g.remove_node("A:x")
        assert idx.build_from_graph(g, node_types=None) == 0

    def test_prune_to_graph_reports_removed_count(self):
        g = _graph(("A:x", "T"), ("A:y", "T"))
        idx = _index(g)
        g.remove_node("A:x")
        assert idx.prune_to_graph(g, node_types=None) == 1

    def test_prune_is_idempotent(self):
        g = _graph(("A:x", "T"))
        idx = _index(g)
        assert idx.prune_to_graph(g, node_types=None) == 0

    def test_empty_graph_prunes_everything(self):
        g = _graph(("A:x", "T"))
        idx = _index(g)
        assert idx.prune_to_graph(nx.MultiDiGraph(), node_types=None) == 1
        assert len(idx) == 0


class TestNpyStillPrunes:
    """tier 1 은 이미 정리하고 있었다 — 기반 클래스를 고치며 깨뜨리지 않는다."""

    def test_npy_drops_removed_node(self, tmp_path):
        from ontology.core.npy_backend import NpyBackend
        g = _graph(("A:x", "T"), ("A:y", "T"))
        idx = NpyBackend(embed_fn=embed, namespace="prune_t",
                         cache_dir=tmp_path)
        idx.build_from_graph(g, node_types=None)
        g.remove_node("A:x")
        idx.build_from_graph(g, node_types=None)
        assert len(idx) == 1
        assert {h["node_id"] for h in idx.search("y", top_k=5)} == {"A:y"}


class TestEngineRebuild:
    """엔진 레벨 — /reindex 가 이걸 부르지 않아서 결함이 사용자에게 보였다."""

    def _engine(self):
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        engine = KnowledgeGraphEngine(fast_mode=True)
        engine.graph.add_node("Disease:유방암", type="Disease", name="유방암")
        engine.graph.add_node("Disease:위암", type="Disease", name="위암")
        return engine

    def test_rebuild_initializes_when_absent(self):
        """색인이 아직 없으면 만들어야 한다 — refresh 는 None 일 때 0 을 돌려준다."""
        engine = self._engine()
        out = engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        assert out["embedded"] == 2 and out["total"] == 2

    def test_rebuild_prunes_after_node_removal(self):
        engine = self._engine()
        engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        engine.graph.remove_node("Disease:유방암")
        out = engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        assert out["pruned"] == 1 and out["total"] == 1

    def test_rebuild_removes_the_ghost_from_search(self):
        engine = self._engine()
        engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        engine.graph.remove_node("Disease:유방암")
        engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        found = {h["node_id"] for h in engine.semantic_search("유방암", top_k=5)}
        assert "Disease:유방암" not in found

    def test_rebuild_is_idempotent(self):
        engine = self._engine()
        engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        out = engine.rebuild_semantic_index(embed_fn=embed, node_types=None)
        assert out["embedded"] == 0 and out["pruned"] == 0


class TestServiceReportsNodeVectors:
    def test_index_namespace_includes_node_vectors(self, tmp_path):
        """관측 가능해야 한다 — 조용히 성공하면 다음에 또 같은 오진을 한다."""
        from ontology.server.service import OntologyBuilderService
        service = OntologyBuilderService(data_dir=tmp_path)
        out = service.index_namespace("prune_svc_t")
        assert "node_vectors" in out
