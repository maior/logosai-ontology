"""source 필터 — "이 문서에서만" 검색. 문서 일급화의 검색 절반.

**설계 규정: 필터는 근거(청크)만 거르고 온톨로지(노드 확장)는 거르지 않는다.**
그래프는 네임스페이스 전체의 지식이고, "이 문서에서만"은 **증거의 출처 제한**이지
지식의 제한이 아니다 — 다른 문서에서 배운 별칭·계층이 이 문서의 청크를 찾는 데
쓰이는 것이 그래프-조건부 검색의 요점이다.

**세 채널 전부** 걸러야 한다. 하나라도 새면 필터가 "대체로 그 문서"가 되고,
사용자는 결과에 섞인 남의 문서 청크를 그 문서 것으로 오독한다.
"""
import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_index import ChunkIndex
from ontology.core.chunk_store import ChunkStore
from ontology.core.graph_retrieval import GraphConditionedRetriever


@pytest.fixture()
def parts(tmp_path):
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_node("T:암진단비", type="T", name="암진단비")
    g.add_node("T:보장개시일", type="T", name="보장개시일")
    g.add_edge("T:암진단비", "T:보장개시일", predicate="hasCondition")

    store = ChunkStore(namespace="srcns", path=tmp_path / "c.jsonl")
    a1 = store.add(Chunk(text="암진단비 를 지급한다 " + "본문 " * 10,
                         source="요구서.txt", index=0, char_start=0, char_end=60),
                   node_ids=["T:암진단비"])
    a2 = store.add(Chunk(text="보장개시일 규정 " + "본문 " * 10,
                         source="요구서.txt", index=1, char_start=60, char_end=120),
                   node_ids=["T:보장개시일"])
    b1 = store.add(Chunk(text="암진단비 산정 방식 " + "본문 " * 10,
                         source="제안서.pdf", index=0, char_start=0, char_end=60),
                   node_ids=["T:암진단비"])

    class _KG:
        graph = g

        def semantic_search(self, query, top_k=5):
            return [{"node_id": "T:암진단비", "score": 0.9}]

        def get_ancestors(self, node_id, predicate=None):
            return []

        def get_descendants(self, node_id, predicate=None):
            return []

    def fake_embed(texts):
        # "암" 포함 여부만 구별하는 결정적 가짜 임베더 (2차원)
        return [[1.0, 0.0] if "암" in t else [0.0, 1.0] for t in texts]

    index = ChunkIndex(store, embed_fn=fake_embed)
    retriever = GraphConditionedRetriever(namespace="srcns", kg=_KG(),
                                          store=store, index=index)
    return retriever, index, store, a1, a2, b1


class TestChunkIndexFilter:
    def test_only_that_source(self, parts):
        retriever, index, store, a1, a2, b1 = parts
        hits = index.search("암진단비", top_k=5, source="요구서.txt")
        assert {c.source for c, _ in hits} == {"요구서.txt"}

    def test_no_starvation(self, parts):
        """전역 top_k 를 먼저 자르고 거르면 필터 문서의 정답이 잘려 나간다 —
        필터 시에는 전량 채점 후 거른다 (브루트포스 규모라 정확·저렴)."""
        retriever, index, store, a1, a2, b1 = parts
        hits = index.search("암진단비", top_k=1, source="요구서.txt")
        assert len(hits) == 1 and hits[0][0].chunk_id == a1

    def test_unknown_source_returns_empty(self, parts):
        retriever, index, store, a1, a2, b1 = parts
        assert index.search("암진단비", top_k=5, source="없는파일") == []

    def test_fallback_path_also_filters(self, parts, monkeypatch):
        """임베더 없는 폴백(부분문자열)도 같은 계약이다 — 한 경로만 거르면
        환경에 따라 필터가 새는 검색이 된다."""
        retriever, index, store, a1, a2, b1 = parts
        monkeypatch.setattr(index, "_has_embedder", lambda: False)
        hits = index.search("암진단비", top_k=5, source="제안서.pdf")
        assert hits and {c.source for c, _ in hits} == {"제안서.pdf"}


class TestRetrieverFilter:
    def test_all_channels_respect_the_filter(self, parts):
        """채널 A(임베딩)·B(노드 경유)·C(확산) 전부 — 하나라도 새면
        '대체로 그 문서'가 된다."""
        retriever, index, store, a1, a2, b1 = parts
        hits = retriever.search("암진단비", top_k=10, source="요구서.txt",
                                use_propagation=True, propagation_channel=True)
        assert hits and {h.chunk.source for h in hits} == {"요구서.txt"}

    def test_channel_b_would_leak_without_filter(self, parts):
        """판별력 확인 — 필터 없이는 남의 문서 청크(b1)가 실제로 섞인다.
        이게 안 섞이면 위 테스트는 공허하게 통과한다."""
        retriever, index, store, a1, a2, b1 = parts
        hits = retriever.search("암진단비", top_k=10)
        assert "제안서.pdf" in {h.chunk.source for h in hits}

    def test_expansion_is_not_filtered(self, parts):
        """온톨로지 확장은 문서 필터와 무관하다 — 지식은 네임스페이스 전체다."""
        retriever, index, store, a1, a2, b1 = parts
        exp_plain = retriever.expand("암진단비")
        # expand 는 source 를 받지 않는다 — 시그니처 자체가 계약이다
        import inspect
        assert "source" not in inspect.signature(retriever.expand).parameters
        assert exp_plain.entry_nodes

    def test_provenance_survives_filtering(self, parts):
        """필터가 matched_via/channels 를 지우면 안 된다 — 설명 가능성 유지."""
        retriever, index, store, a1, a2, b1 = parts
        hits = retriever.search("암진단비", top_k=10, source="요구서.txt")
        assert all(h.channels for h in hits)

    def test_none_source_unchanged(self, parts):
        """필터 미지정이면 종전과 완전히 같아야 한다 — 하위호환 관문."""
        retriever, index, store, a1, a2, b1 = parts
        before = [(h.chunk.chunk_id, h.score) for h in retriever.search("암진단비", top_k=10)]
        after = [(h.chunk.chunk_id, h.score) for h in retriever.search("암진단비", top_k=10, source=None)]
        assert before == after
