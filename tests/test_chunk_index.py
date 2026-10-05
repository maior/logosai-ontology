"""
축 3 — 청크 레벨 의미/하이브리드 검색.

축 2 가 원문을 남겼지만 검색은 부분문자열(store.search_text)뿐이었다. "청약을
무를 수 있나?" 로는 "청약철회권" 청크를 못 찾는다 — 글자가 안 겹친다.

ChunkIndex 는 VectorBackend 계약이 (id, text, type) 이라는 성질을 그대로
재사용한다. 노드용으로 만든 백엔드가 청크에도 그대로 쓰인다:
- memory/npy → 임베딩 의미 검색
- elasticsearch → BM25 + 벡터 하이브리드 (조문번호 같은 리터럴 질의까지)

고정하는 계약:
1. 히트는 StoredChunk 로 해석되어 나온다 — 호출부가 id 를 다시 조회하지 않는다.
2. 저장소에서 사라진 청크의 인덱스 잔재는 결과에서 조용히 걸러진다.
3. 원문이 바뀌지 않으면 재임베딩하지 않는다.
4. 임베더가 없으면 부분문자열 폴백으로 degrade — raise 하지 않는다.
"""

import numpy as np
import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_index import ChunkIndex
from ontology.core.chunk_store import ChunkStore


def fake_embed(texts):
    """문자 bigram 해시 임베더 — 겹치는 bigram 이 많을수록 코사인이 커진다.

    공백 토큰이 아니라 문자 bigram 을 쓰는 이유: 한국어는 교착어라 "암진단비"와
    "암진단비는"이 공백 기준으로는 서로 다른 토큰이다. 공백 토크나이저 가짜
    임베더로는 의미 검색을 흉내조차 낼 수 없어 테스트가 구현이 아니라 fixture
    를 검사하게 된다. 문자 bigram 은 ES cjk analyzer 가 쓰는 방식과 같고,
    어미 변화를 넘어 겹침을 만든다.
    """
    import zlib
    dim = 128
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        source = str(text)
        for j in range(len(source) - 1):
            bigram = source[j:j + 2]
            if bigram.strip():
                vectors[i, zlib.crc32(bigram.encode("utf-8")) % dim] += 1.0
    return vectors


@pytest.fixture
def store(tmp_path):
    store = ChunkStore(namespace="idx", path=tmp_path / "chunks.jsonl")
    store.add(Chunk(text="청약철회권은 보험증권을 받은 날부터 15일 이내 행사한다.",
                    source="약관.md", index=0), ["Clause:청약철회"])
    store.add(Chunk(text="암진단비는 최초 1회에 한하여 지급한다.",
                    source="약관.md", index=1), ["Coverage:암진단비"])
    store.add(Chunk(text="보험료 납입이 연체되면 계약은 실효된다.",
                    source="약관.md", index=2), ["Clause:실효"])
    return store


@pytest.fixture
def index(store):
    return ChunkIndex(store=store, embed_fn=fake_embed)


class TestSearchReturnsChunks:
    def test_hits_are_resolved_to_stored_chunks(self, index):
        hits = index.search("청약철회권 15일", top_k=1)
        assert len(hits) == 1
        chunk, score = hits[0]
        assert "청약철회권" in chunk.text
        assert chunk.source == "약관.md"
        assert isinstance(score, float)

    def test_ranks_by_similarity(self, index):
        hits = index.search("암진단비 지급", top_k=3)
        assert "암진단비" in hits[0][0].text

    def test_respects_top_k(self, index):
        assert len(index.search("보험", top_k=2)) <= 2

    def test_blank_query_returns_empty(self, index):
        assert index.search("  ", top_k=3) == []

    def test_empty_store_returns_empty(self, tmp_path):
        empty = ChunkIndex(store=ChunkStore(namespace="e",
                                            path=tmp_path / "c.jsonl"),
                           embed_fn=fake_embed)
        assert empty.search("무엇이든", top_k=3) == []


class TestNodeLinkage:
    """청크 히트는 노드로 이어진다 — 축 4 의 그래프-조건부 검색이 여기서 출발한다."""

    def test_hit_carries_its_node_ids(self, index):
        chunk, _score = index.search("청약철회권 15일", top_k=1)[0]
        assert "Clause:청약철회" in chunk.node_ids


class TestStaleEntries:
    def test_chunk_removed_from_store_is_filtered_out(self, index, store):
        index.refresh()
        store.clear()
        # 인덱스에는 아직 남아 있지만 저장소에 없다 → 결과에서 빠져야 한다
        assert index.search("청약철회권", top_k=3) == []


class TestIncrementalRefresh:
    def test_unchanged_chunks_are_not_reembedded(self, store):
        calls = {"n": 0}

        def counting(texts):
            calls["n"] += len(texts)
            return fake_embed(texts)

        index = ChunkIndex(store=store, embed_fn=counting)
        index.refresh()
        first = calls["n"]
        assert first == 3
        index.refresh()
        assert calls["n"] == first, "안 바뀐 청크를 다시 임베딩했다"

    def test_new_chunk_is_indexed_on_refresh(self, store, index):
        index.refresh()
        store.add(Chunk(text="해지환급금은 표에 따라 지급한다.",
                        source="약관.md", index=3), ["Clause:해지환급금"])
        index.refresh()
        hits = index.search("해지환급금 표", top_k=1)
        assert "해지환급금" in hits[0][0].text


class TestDegradation:
    """임베더가 없어도 원문이 있으면 최소한의 검색은 된다."""

    def test_no_embedder_falls_back_to_substring(self, store):
        index = ChunkIndex(store=store, embed_fn=None, auto_default=False)
        hits = index.search("청약철회", top_k=3)
        assert len(hits) == 1
        assert "청약철회권" in hits[0][0].text

    def test_fallback_reports_zero_score(self, store):
        """폴백 점수는 유사도가 아니다 — 0.0 으로 정직하게 표시한다."""
        index = ChunkIndex(store=store, embed_fn=None, auto_default=False)
        assert hits_scores(index.search("청약철회", top_k=3)) == [0.0]


def hits_scores(hits):
    return [score for _chunk, score in hits]
