"""재적재에서 끊긴 근거 링크 복원 — 고아 노드가 **쌓이는 원인**을 막는다.

**근본 원인 (실측)**: `_build_text_into` 는 재적재 시 `delete_by_source` 로 그
소스의 청크를 전량 삭제하지만 **그래프 노드는 남긴다**(의도적 — 노드는 여러 소스에
걸칠 수 있고 검수 판정도 붙어 있다). 그런데 LLM 추출은 비결정적이라 재적재에서
이번엔 안 뽑힌 노드가 생기고, 그 노드는 그래프에 남은 채 **근거를 잃는다**.

그렇게 ins_cancer_demo 에 고아 노드가 27개(14%) 쌓였고, 그것이 evidence 채점의
hit@5 천장(검색 knob 15개 조합으로도 못 넘던 0.8125)을 정하고 있었다.

**복원의 열쇠는 노드가 이미 갖고 있다**: `_merge` 가 노드 attrs 에
`source` + `chunk_index` 를 남긴다 — "어느 청크에서 나왔는지"의 기록이다.
실제로 고아 2건이 `chunk_index=21/91` 을 들고 있으면서 링크가 없었다.

**좁게 자동, 넓게 검수** — 이 분리가 설계의 핵심이다:
  · 빌더(자동): `chunk_index` 힌트로 **원래 그 청크**만 복원한다. 정확하고 좁다.
  · 검수(사람): 이름이 나오는 **모든** 청크로 확장한다(approve_orphan_links).
자동 경로가 넓게 이으면 무인 실행에서 근거 사슬이 느슨해진다.

`chunk_index` 는 힌트일 뿐이므로 **원문 대조를 반드시 통과시킨다** — 문서 내용이
바뀌면 같은 index 가 다른 텍스트다. 이름이 그 청크에 문자 그대로 없으면 잇지 않는다.
"""
import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_store import ChunkStore
from ontology.core.relink import relink_by_chunk_index


def _graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    # 재적재에서 살아남았지만 링크를 잃은 노드 (source + chunk_index 를 기억한다)
    g.add_node("T:보험료", type="T", name="보험료",
               source="약관.pdf", chunk_index=0)
    g.add_node("T:계약일", type="T", name="계약일",
               source="약관.pdf", chunk_index=1)
    # 힌트가 가리키는 청크에 이름이 없다 (문서 내용이 바뀐 경우)
    g.add_node("T:없는말", type="T", name="없는말",
               source="약관.pdf", chunk_index=0)
    # 힌트가 없는 노드 (수동 생성 등)
    g.add_node("T:힌트없음", type="T", name="보험료")
    # 다른 소스의 노드
    g.add_node("T:타소스", type="T", name="보험료",
               source="다른문서.pdf", chunk_index=0)
    return g


def _store(tmp_path):
    store = ChunkStore(namespace="relinkns", path=tmp_path / "c.jsonl")
    store.add(Chunk(text="보험료 를 납입한다", source="약관.pdf", index=0,
                    char_start=0, char_end=20))
    store.add(Chunk(text="계약일 은 청약일이다", source="약관.pdf", index=1,
                    char_start=20, char_end=40))
    return store


class TestRelinkByChunkIndex:
    def test_restores_the_original_chunk(self, tmp_path):
        store = _store(tmp_path)
        n = relink_by_chunk_index(_graph(), store, source="약관.pdf")
        assert n == 2
        assert store.chunks_for_node("T:보험료")[0].index == 0
        assert store.chunks_for_node("T:계약일")[0].index == 1

    def test_quote_check_blocks_a_stale_hint(self, tmp_path):
        """문서가 바뀌면 같은 index 가 다른 텍스트다 — 이름이 없으면 잇지 않는다.
        이 관문이 없으면 근거가 엉뚱한 조문을 가리켜 축 2 의 계약이 깨진다."""
        store = _store(tmp_path)
        relink_by_chunk_index(_graph(), store, source="약관.pdf")
        assert store.chunks_for_node("T:없는말") == []

    def test_nodes_without_hint_are_left_alone(self, tmp_path):
        """힌트 없는 노드는 빌더가 만든 것이 아니다 — 자동으로 추측하지 않는다
        (이름 기반 확장은 검수 몫이다)."""
        store = _store(tmp_path)
        relink_by_chunk_index(_graph(), store, source="약관.pdf")
        assert store.chunks_for_node("T:힌트없음") == []

    def test_other_sources_untouched(self, tmp_path):
        """이 소스를 재적재한 것이므로 다른 소스의 노드를 건드리면 안 된다."""
        store = _store(tmp_path)
        relink_by_chunk_index(_graph(), store, source="약관.pdf")
        assert store.chunks_for_node("T:타소스") == []

    def test_already_linked_is_not_double_counted(self, tmp_path):
        store = _store(tmp_path)
        first = relink_by_chunk_index(_graph(), store, source="약관.pdf")
        second = relink_by_chunk_index(_graph(), store, source="약관.pdf")
        assert first == 2 and second == 0

    def test_missing_index_is_skipped_not_an_error(self, tmp_path):
        """힌트가 가리키는 청크가 사라졌을 수 있다 (문서가 짧아진 경우)."""
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("T:보험료", type="T", name="보험료",
                   source="약관.pdf", chunk_index=999)
        assert relink_by_chunk_index(g, _store(tmp_path),
                                     source="약관.pdf") == 0

    def test_name_falls_back_to_id_tail(self, tmp_path):
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node("T:보험료", type="T", source="약관.pdf", chunk_index=0)
        assert relink_by_chunk_index(g, _store(tmp_path),
                                     source="약관.pdf") == 1

    def test_empty_source_does_nothing(self, tmp_path):
        """소스를 안 주면 전 그래프를 훑어 다른 문서까지 건드릴 위험이 있다."""
        assert relink_by_chunk_index(_graph(), _store(tmp_path),
                                     source="") == 0

    def test_never_raises_on_broken_store(self):
        class Boom:
            def all(self):
                raise RuntimeError("down")

        import networkx as nx
        assert relink_by_chunk_index(nx.MultiDiGraph(), Boom(),
                                     source="약관.pdf") == 0

    def test_none_inputs_are_safe(self, tmp_path):
        assert relink_by_chunk_index(None, _store(tmp_path), source="s") == 0
        assert relink_by_chunk_index(_graph(), None, source="s") == 0
