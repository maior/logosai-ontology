"""
축 4 — 그래프-조건부 검색 (graph-conditioned retrieval).

앞선 분석에서 아무도 만들지 않은 것으로 지목된 지점이다. aicoach 는 온톨로지와
ES 를 두 개의 평행한 검색 시스템으로 두고 `source` 키워드 하나로만 이었다:
동의어 테이블(kg/match.py:21-26)은 ES 에 도달하지 않고, 그래프 순회로 쿼리를
확장하는 코드도 없고, ES aggregation 은 0건이다. 결국 상품명으로 검색하면
요약서가 잡히는 문제를 `f"{label} 보장 보험금 지급사유"` 라는 **하드코딩
리터럴**로 편향시켜 막았다 (rag/search.py:126) — 온톨로지에 그 지식이 있는데도.

Logos 는 그래프와 임베딩을 한 곳에 가진 유일한 시스템이라 이걸 할 수 있다.

축 3 에서 실측된 실패가 이 축의 존재 이유다 (실제 ko-sroberta):
    질의 "계약을 무를 수 있나요?"
    기대  "청약철회권은 ... 15일 이내에 행사할 수 있다."
    실제  "보험료 납입이 연체되면 계약은 실효된다."   ← "계약" 글자에 끌려감
임베딩 단독으로 안 되는 것을 그래프가 고친다.

고정하는 계약:
1. 확장은 온톨로지에서 나온다 — 하드코딩 어휘가 아니다 (프로젝트 절대 원칙).
2. 두 채널(청크 임베딩 · 그래프 경유)을 RRF 로 융합한다.
3. 모든 히트는 **왜 걸렸는지**(matched_via)를 들고 나온다 — 설명 가능해야 한다.
4. 그래프가 비어도 청크 검색으로 degrade 한다 — 그래프 없는 코퍼스도 있다.
"""

import numpy as np
import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_index import ChunkIndex
from ontology.core.chunk_store import ChunkStore
from ontology.core.graph_retrieval import (
    RRF_K,
    GraphConditionedRetriever,
    RetrievalHit,
)


def fake_embed(texts):
    """문자 bigram 해시 임베더 (test_chunk_index.py 와 동일한 이유).

    한국어는 교착어라 공백 토큰 기준으로는 어미 변화를 못 넘는다.
    """
    import zlib
    dim = 256
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        source = str(text)
        for j in range(len(source) - 1):
            bigram = source[j:j + 2]
            if bigram.strip():
                vectors[i, zlib.crc32(bigram.encode("utf-8")) % dim] += 1.0
    return vectors


@pytest.fixture
def parts(tmp_path):
    """축 3 실패 시나리오를 그대로 재현하는 그래프 + 청크."""
    import asyncio

    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

    kg = KnowledgeGraphEngine(fast_mode=True, namespace="gc")

    async def build():
        # 온톨로지: 개념 + 별칭 + is_a 계층 — 확장의 재료
        await kg.add_concept("Clause:청약철회", "Clause", {
            "name": "청약철회",
            "definition": "청약을 물러 계약을 없던 것으로 하는 권리",
            "aliases": ["계약 취소", "청약 철회권"]})
        await kg.add_concept("Clause:실효", "Clause", {
            "name": "실효", "definition": "보험료 미납으로 계약 효력이 사라짐"})
        await kg.add_concept("Coverage:암진단비", "Coverage", {"name": "암진단비"})
        await kg.add_concept("Clause:소비자권리", "Clause", {"name": "소비자권리"})
        # 청약철회 is_a 소비자권리 — 상위 개념으로도 닿을 수 있게
        await kg.add_relationship("Clause:청약철회", "Clause:소비자권리", "is_a", {})

    asyncio.run(build())
    kg.init_semantic_index(embed_fn=fake_embed, node_types=None)

    store = ChunkStore(namespace="gc", path=tmp_path / "chunks.jsonl")
    store.add(Chunk(text="청약철회권은 보험증권을 받은 날부터 15일 이내에 행사할 수 있다.",
                    source="약관.md", index=0), ["Clause:청약철회"])
    store.add(Chunk(text="보험료 납입이 연체되면 계약은 실효된다.",
                    source="약관.md", index=1), ["Clause:실효"])
    store.add(Chunk(text="암진단비는 최초 1회에 한하여 지급한다.",
                    source="약관.md", index=2), ["Coverage:암진단비"])

    index = ChunkIndex(store=store, embed_fn=fake_embed)
    retriever = GraphConditionedRetriever(namespace="gc", kg=kg, store=store,
                                          index=index)
    return retriever, kg, store, index


# ─── 1. 쿼리 확장은 온톨로지에서 나온다 ─────────────────────────────

class TestExpansionComesFromTheOntology:
    """하드코딩 어휘 금지 — 확장어는 전부 그래프에서 읽어온다."""

    def test_entry_nodes_are_found_semantically(self, parts):
        retriever = parts[0]
        expansion = retriever.expand("계약을 무를 수 있나요?")
        assert any(e["node_id"] == "Clause:청약철회"
                   for e in expansion.entry_nodes)

    def test_aliases_become_expansion_terms(self, parts):
        """별칭은 온톨로지에 저장된 데이터다 — aicoach 의 _SYN 하드코딩
        테이블이 하려던 일을, 데이터로 한다."""
        retriever = parts[0]
        expansion = retriever.expand("계약을 무를 수 있나요?")
        assert "계약 취소" in expansion.terms

    def test_node_names_become_expansion_terms(self, parts):
        retriever = parts[0]
        expansion = retriever.expand("계약을 무를 수 있나요?")
        assert "청약철회" in expansion.terms

    def test_is_a_ancestors_are_expanded(self, parts):
        """청약철회 is_a 소비자권리 — 상위 개념도 확장에 들어온다."""
        retriever = parts[0]
        expansion = retriever.expand("계약을 무를 수 있나요?")
        assert any(n["node_id"] == "Clause:소비자권리"
                   for n in expansion.expanded_nodes)

    def test_expanded_query_contains_original(self, parts):
        retriever = parts[0]
        expansion = retriever.expand("계약을 무를 수 있나요?")
        assert expansion.expanded_query.startswith("계약을 무를 수 있나요?")

    def test_expansion_is_capped(self, parts):
        """확장어가 무한정 늘면 원 질의가 희석된다 — 상한이 있어야 한다."""
        retriever = parts[0]
        expansion = retriever.expand("계약을 무를 수 있나요?", max_terms=2)
        assert len(expansion.terms) <= 2

    def test_no_entry_means_no_expansion(self, parts):
        retriever = parts[0]
        expansion = retriever.expand("양자역학 초전도 현상", min_entry_score=0.99)
        assert expansion.terms == []
        assert expansion.expanded_query == "양자역학 초전도 현상"


# ─── 2. 축 3 실패를 고친다 (이 축의 성공 기준) ──────────────────────

class TestOntologyBridgesTheVocabularyGap:
    """축 4 의 본질: **원문에 없는 지식이 온톨로지에는 있다.**

    "물러달라" 는 세 청크 어디에도 안 나온다 — 원문만 보는 청크 채널은 전부
    0.0 을 준다(실측). 그러나 온톨로지의 definition("청약을 **물러** 계약을
    없던 것으로 하는 권리")에는 그 말이 있다. 임베딩이 원문에서 못 찾는 것을
    그래프가 개념을 거쳐 데려온다.

    이것이 aicoach 가 `f"{label} 보장 보험금 지급사유"` 라는 하드코딩 리터럴로
    막던 문제의 정공법이다 — 그 지식은 온톨로지에 이미 있었다.
    """

    QUERY = "물러달라"

    def test_chunk_channel_alone_cannot_find_it(self, parts):
        """전제 확인 — 이게 깨지면 아래 테스트는 아무것도 증명하지 못한다."""
        index = parts[3]
        hits = index.search(self.QUERY, top_k=3)
        assert all(score == 0.0 for _chunk, score in hits), (
            "전제가 깨졌다: 청크 채널이 이미 이 질의를 처리한다")

    def test_graph_conditioning_finds_the_right_chunk(self, parts):
        retriever = parts[0]
        hits = retriever.search(self.QUERY, top_k=1)
        assert hits
        assert "청약철회권" in hits[0].chunk.text

    def test_hit_explains_why_it_matched(self, parts):
        """설명 가능성 — 왜 이게 나왔는지 말할 수 있어야 한다."""
        retriever = parts[0]
        hit = retriever.search(self.QUERY, top_k=1)[0]
        assert "Clause:청약철회" in hit.matched_via
        assert "graph" in hit.channels

    def test_expansion_pulled_the_term_from_the_definition(self, parts):
        """definition 이 진입을 만들고, name/alias 가 확장어가 된다."""
        retriever = parts[0]
        expansion = retriever.expand(self.QUERY)
        assert any(e["node_id"] == "Clause:청약철회" for e in expansion.entry_nodes)
        assert "청약철회" in expansion.terms


class TestEntryGateIsScaleFree:
    """진입 게이트에 절대 임계값을 쓰지 않는다.

    처음엔 min_entry_score=0.25 로 짰다가 두 번 연달아 정답 노드를 잘라먹었다
    (정답이 1위인데 점수가 0.18, 0.07). 코사인은 모델·코퍼스마다 스케일이 달라
    보정되지 않은 값이라, 절대 임계값은 정확히 이렇게 조용히 실패한다.
    """

    def test_low_absolute_scores_still_produce_entries(self, parts):
        """점수가 0.07 이어도 1위면 진입 노드다 — 절대값이 아니라 순위·비율이
        판단 기준이다."""
        retriever = parts[0]
        expansion = retriever.expand("물러달라")
        assert expansion.entry_nodes
        assert expansion.entry_nodes[0]["score"] < 0.25  # 절대값은 낮다

    def test_zero_similarity_is_rejected(self, parts):
        """유사도 0 은 지어낸 임계값이 아니라 코사인의 정의상 무관이다."""
        retriever = parts[0]
        expansion = retriever.expand("물러달라")
        assert all(e["score"] > 0 for e in expansion.entry_nodes)

    def test_relative_ratio_drops_much_weaker_entries(self, parts):
        retriever = parts[0]
        strict = retriever.expand("계약을 무를 수 있나요?", entry_ratio=0.99)
        loose = retriever.expand("계약을 무를 수 있나요?", entry_ratio=0.1)
        assert len(strict.entry_nodes) <= len(loose.entry_nodes)


class TestExpansionCapIsBounded:
    """확장어 상한 — 이 값은 **그래프 상태에 종속적**이라는 것이 교훈이다.

    고아 노드를 안고 재면 2 가 8 을 이기고(hit@1 0.6250 vs 0.5000), 근거 링크
    66개를 이은 뒤에는 8 이 2 를 이긴다(0.8125 vs 0.7500). 연결이 성길 때
    확장어는 잡음을 나르고 촘촘해지면 신호를 나른다. 상수 주석의 표 참고.

    그래서 이 테스트는 "최적값"을 주장하지 않는다 — 배선이 실제로 상한을
    지키는지, 스윕이 가능한지만 고정한다.
    """

    def test_default_cap_exists_and_is_bounded(self):
        from ontology.core.graph_retrieval import DEFAULT_MAX_TERMS
        assert 0 < DEFAULT_MAX_TERMS <= 12

    def test_entry_cap_leaves_room_for_expansion(self):
        """진입 상한이 k(=5) 를 다 먹으면 확장 노드가 **한 칸도** 보이지 않는다.

        실측: entry_k=5 일 때 손 시드 케이스의 정답이 확장으로 들어오는데도
        6위여서 k=5 미검출이었다. 3 으로 줄이니 hit@5 0.7609 → 0.8478.
        상수 주석의 표 참고 — 이 값도 그래프 상태에 종속적이다."""
        from ontology.core.graph_retrieval import DEFAULT_ENTRY_K
        assert 0 < DEFAULT_ENTRY_K < 5

    def test_default_entry_cap_is_wired(self, parts):
        """상수만 바꾸고 배선이 빠지면 측정과 라이브가 갈린다."""
        from ontology.core.graph_retrieval import DEFAULT_ENTRY_K
        retriever = parts[0]
        assert len(retriever.expand("계약을 무를 수 있나요?").entry_nodes) \
            <= DEFAULT_ENTRY_K

    def test_default_expansion_is_capped(self, parts):
        """기본 경로가 상한을 실제로 지키는지 — 상수만 바꾸고 배선이 빠지면
        측정과 라이브가 갈린다."""
        retriever = parts[0]
        from ontology.core.graph_retrieval import DEFAULT_MAX_TERMS
        assert len(retriever.expand("계약을 무를 수 있나요?").terms) \
            <= DEFAULT_MAX_TERMS

    def test_explicit_argument_still_wins(self, parts):
        """스윕이 가능해야 한다 — 상한은 인자로 열려 있다."""
        retriever = parts[0]
        assert len(retriever.expand("계약을 무를 수 있나요?",
                                    max_terms=0).terms) == 0


# ─── 3. 채널 융합 (RRF) ──────────────────────────────────────────────

class TestChannelFusion:
    def test_rrf_k_is_the_standard_constant(self):
        assert RRF_K == 60

    def test_hits_are_retrieval_hit_objects(self, parts):
        retriever = parts[0]
        hits = retriever.search("청약철회", top_k=3)
        assert all(isinstance(h, RetrievalHit) for h in hits)

    def test_chunk_channel_alone_still_works(self, parts):
        """그래프에 안 걸려도 청크 임베딩만으로 걸린 히트는 살아남는다."""
        retriever = parts[0]
        hits = retriever.search("암진단비 지급", top_k=3)
        assert any("암진단비" in h.chunk.text for h in hits)

    def test_hit_found_by_both_channels_lists_both(self, parts):
        retriever = parts[0]
        hits = retriever.search("청약철회권 15일", top_k=3)
        top = hits[0]
        assert set(top.channels) == {"chunk", "graph"}

    def test_scores_are_descending(self, parts):
        retriever = parts[0]
        hits = retriever.search("계약", top_k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, parts):
        retriever = parts[0]
        assert len(retriever.search("계약", top_k=2)) <= 2

    def test_rrf_ties_are_broken_deterministically(self, parts):
        """RRF 는 순위만 쓰므로 채널이 둘일 때 1·2위가 맞바뀌면 수학적으로
        반드시 동점이다 (1/61+1/62 == 1/62+1/61). 실측에서 실제로 났고, 그때
        순서가 dict 삽입 순서로 갈렸다 — 임의적이라 그 자체로 버그다.
        같은 입력이면 항상 같은 순서가 나와야 한다.
        """
        retriever = parts[0]
        first = [h.chunk.chunk_id for h in retriever.search("계약", top_k=5)]
        for _ in range(3):
            assert [h.chunk.chunk_id
                    for h in retriever.search("계약", top_k=5)] == first

    def test_hits_carry_best_evidence(self, parts):
        retriever = parts[0]
        hits = retriever.search("청약철회권 15일", top_k=3)
        assert all(h.best_evidence >= 0.0 for h in hits)
        assert hits[0].best_evidence > 0.0

    def test_no_duplicate_chunks(self, parts):
        """두 채널에 다 걸린 청크가 두 번 나오면 안 된다."""
        retriever = parts[0]
        hits = retriever.search("청약철회권 15일", top_k=5)
        ids = [h.chunk.chunk_id for h in hits]
        assert len(ids) == len(set(ids))


# ─── 4. Degradation ─────────────────────────────────────────────────

class TestDegradation:
    def test_empty_graph_falls_back_to_chunk_search(self, tmp_path):
        """그래프 없는 코퍼스도 있다 — 청크 검색만으로 동작해야 한다."""
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        kg = KnowledgeGraphEngine(fast_mode=True, namespace="empty")
        kg.init_semantic_index(embed_fn=fake_embed, node_types=None)
        store = ChunkStore(namespace="empty", path=tmp_path / "c.jsonl")
        store.add(Chunk(text="암진단비는 최초 1회 지급한다.", source="d", index=0), [])
        index = ChunkIndex(store=store, embed_fn=fake_embed)

        retriever = GraphConditionedRetriever(namespace="empty", kg=kg,
                                              store=store, index=index)
        hits = retriever.search("암진단비 지급", top_k=1)
        assert len(hits) == 1
        assert hits[0].channels == ["chunk"]

    def test_empty_store_returns_empty(self, tmp_path):
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

        kg = KnowledgeGraphEngine(fast_mode=True, namespace="nostore")
        kg.init_semantic_index(embed_fn=fake_embed, node_types=None)
        store = ChunkStore(namespace="nostore", path=tmp_path / "c.jsonl")
        retriever = GraphConditionedRetriever(
            namespace="nostore", kg=kg, store=store,
            index=ChunkIndex(store=store, embed_fn=fake_embed))
        assert retriever.search("무엇이든", top_k=3) == []

    def test_blank_query_returns_empty(self, parts):
        assert parts[0].search("   ", top_k=3) == []
