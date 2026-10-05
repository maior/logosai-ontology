"""
축 3 — tier 1 (npy) 벡터 백엔드: 디스크 영속화.

이전에는 5-tier 중 tier 0(memory)만 실물이고 npy/parallel/faiss/distributed
는 전부 스텁이었다 — create_backend 가 조용히 memory 를 돌려줬다
(core/vector_backend.py:145-149, 자기 docstring 도 인정). 게다가 인덱스가
디스크에 없어서 **프로세스마다 전 노드를 재임베딩**했다.

tier 1 을 실물로 만든다. 고정하는 계약:
1. 저장/로드 왕복에서 검색 결과가 동일하다.
2. 캐시가 있으면 재임베딩하지 않는다 — 콜드스타트 비용이 tier 1 의 존재 이유다.
3. 텍스트가 바뀐 노드만 다시 임베딩한다.
4. 캐시가 깨졌거나 차원이 안 맞으면 무시하고 새로 만든다 — 절대 죽지 않는다.
"""

import numpy as np
import pytest

from ontology.core.npy_backend import NpyBackend
from ontology.core.vector_backend import TIER_NPY


def fake_embed(texts):
    """결정적 가짜 임베더 — 토큰 해시 기반 (test_vector_backend.py 와 동일)."""
    import zlib
    dim = 64
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in str(text).lower().split():
            vectors[i, zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    return vectors


class CountingEmbed:
    """임베딩 호출 횟수를 센다 — '재임베딩 안 한다'를 증명하려면 필요하다."""

    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return fake_embed(texts)


def make_graph(pairs):
    import networkx as nx
    graph = nx.MultiDiGraph()
    for node_id, description in pairs:
        graph.add_node(node_id, type="agent", name=node_id,
                       description=description)
    return graph


@pytest.fixture
def graph():
    return make_graph([
        ("weather_agent", "도시 날씨 기온 예보"),
        ("shopping_agent", "쇼핑 상품 가격 비교"),
        ("calc_agent", "수학 계산 사칙연산"),
    ])


class TestTierIdentity:
    def test_reports_npy_tier(self, tmp_path):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns",
                             embed_fn=fake_embed)
        assert backend.tier_name == TIER_NPY


class TestPersistence:
    def test_save_load_roundtrip_preserves_search(self, tmp_path, graph):
        first = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        first.build_from_graph(graph, node_types=None)
        expected = first.search("날씨 기온", top_k=3)
        assert first.save_to_disk() is True

        second = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        assert second.load_from_disk() is True
        assert len(second) == len(first)
        assert second.search("날씨 기온", top_k=3) == expected

    def test_load_without_cache_is_not_an_error(self, tmp_path):
        backend = NpyBackend(cache_dir=tmp_path, namespace="absent",
                             embed_fn=fake_embed)
        assert backend.load_from_disk() is False
        assert len(backend) == 0

    def test_namespaces_have_separate_caches(self, tmp_path, graph):
        a = NpyBackend(cache_dir=tmp_path, namespace="nsA", embed_fn=fake_embed)
        a.build_from_graph(graph, node_types=None)
        a.save_to_disk()

        b = NpyBackend(cache_dir=tmp_path, namespace="nsB", embed_fn=fake_embed)
        assert b.load_from_disk() is False


class TestColdStartAvoidance:
    """tier 1 의 존재 이유 — 캐시가 있으면 재임베딩하지 않는다."""

    def test_cached_build_does_not_reembed(self, tmp_path, graph):
        counter = CountingEmbed()
        first = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=counter)
        first.build_from_graph(graph, node_types=None)
        first.save_to_disk()
        assert counter.texts_embedded == 3

        counter2 = CountingEmbed()
        second = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=counter2)
        second.load_from_disk()
        second.build_from_graph(graph, node_types=None)
        assert counter2.texts_embedded == 0, "캐시가 있는데 다시 임베딩했다"
        assert len(second) == 3

    def test_only_changed_nodes_are_reembedded(self, tmp_path, graph):
        counter = CountingEmbed()
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=counter)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()

        graph.nodes["weather_agent"]["description"] = "완전히 다른 설명"
        counter2 = CountingEmbed()
        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=counter2)
        restored.load_from_disk()
        restored.build_from_graph(graph, node_types=None)
        assert counter2.texts_embedded == 1, "바뀐 노드 1개만 재임베딩해야 한다"

    def test_complete_cache_does_not_load_the_embedder(self, tmp_path, graph,
                                                       monkeypatch):
        """캐시가 완전하면 임베더 **모델 자체를** 로드하지 않는다.

        tier 1 의 진짜 값어치가 여기다. 재임베딩을 건너뛰어도 모델을 로드하면
        sentence-transformers 초기화에만 수 초가 든다 — 실측 결과 캐시 경로가
        2.7s 였고, 그 대부분이 쓰지도 않을 모델 로딩이었다. 변경분 계산은
        텍스트 비교라 임베더가 필요 없으므로, 먼저 확인하고 필요할 때만 로드한다.
        """
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()

        import ontology.core.semantic_index as si

        def boom():
            raise AssertionError("캐시가 완전한데 임베더 모델을 로드했다")

        monkeypatch.setattr(si, "_load_default_embed_fn", boom)

        restored = NpyBackend(cache_dir=tmp_path, namespace="ns")  # embed_fn 미주입
        restored.load_from_disk()
        assert restored.build_from_graph(graph, node_types=None) == 0
        assert len(restored) == 3

    def test_new_nodes_are_embedded_after_load(self, tmp_path, graph):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()

        graph.add_node("new_agent", type="agent", name="new_agent",
                       description="새로 생긴 에이전트")
        counter = CountingEmbed()
        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=counter)
        restored.load_from_disk()
        restored.build_from_graph(graph, node_types=None)
        assert counter.texts_embedded == 1
        assert len(restored) == 4


class TestCorruptCacheNeverRaises:
    """캐시는 파생 데이터다. 깨지면 버리고 다시 만들면 되지, 죽으면 안 된다."""

    def test_corrupt_meta_is_ignored(self, tmp_path, graph):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()
        backend.meta_path.write_text("not json at all", encoding="utf-8")

        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        assert restored.load_from_disk() is False
        assert len(restored) == 0
        restored.build_from_graph(graph, node_types=None)  # 재구축은 정상
        assert len(restored) == 3

    def test_corrupt_vectors_are_ignored(self, tmp_path, graph):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()
        backend.vectors_path.write_bytes(b"\x00\x01 garbage")

        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        assert restored.load_from_disk() is False
        assert len(restored) == 0

    def test_row_count_mismatch_is_ignored(self, tmp_path, graph):
        """메타의 id 개수와 벡터 행 수가 어긋나면 캐시 전체가 신뢰 불가다."""
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()
        np.save(backend.vectors_path, np.zeros((99, 64), dtype=np.float32))

        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        assert restored.load_from_disk() is False

    def test_model_change_invalidates_cache(self, tmp_path, graph):
        """임베딩 모델을 갈아끼우면 옛 벡터는 새 쿼리 벡터와 같은 공간에 있지
        않다 — 캐시를 버려야 한다.

        신원(model_id)으로 잡는 이유: **차원 비교로는 못 잡는다**. 서로 다른
        모델이 같은 차원을 쓰는 일이 흔하다 (ko-sroberta 768 vs
        gemini-embedding 768 — aicoach 는 실제로 둘 다 쓴다).
        """
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns",
                             embed_fn=fake_embed, model_id="model-a")
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()

        same_dim_other_model = NpyBackend(cache_dir=tmp_path, namespace="ns",
                                          embed_fn=fake_embed, model_id="model-b")
        assert same_dim_other_model.load_from_disk() is False

    def test_same_model_keeps_cache(self, tmp_path, graph):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns",
                             embed_fn=fake_embed, model_id="model-a")
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()

        again = NpyBackend(cache_dir=tmp_path, namespace="ns",
                           embed_fn=fake_embed, model_id="model-a")
        assert again.load_from_disk() is True

    def test_meta_dim_mismatch_is_ignored(self, tmp_path, graph):
        """메타와 .npy 를 따로 쓰는 대가 — 둘이 어긋나면 캐시를 버린다."""
        import json

        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.save_to_disk()
        meta = json.loads(backend.meta_path.read_text(encoding="utf-8"))
        meta["dim"] = 999
        backend.meta_path.write_text(json.dumps(meta), encoding="utf-8")

        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        assert restored.load_from_disk() is False


class TestSearchStillWorks:
    """tier 1 은 tier 0 의 정확도를 그대로 물려받는다 (brute-force 동일)."""

    def test_search_finds_semantic_match(self, tmp_path, graph):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        hits = backend.search("날씨 기온", top_k=1)
        assert hits[0]["node_id"] == "weather_agent"

    def test_remove_then_save_load(self, tmp_path, graph):
        backend = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        backend.build_from_graph(graph, node_types=None)
        backend.remove("shopping_agent")
        backend.save_to_disk()

        restored = NpyBackend(cache_dir=tmp_path, namespace="ns", embed_fn=fake_embed)
        restored.load_from_disk()
        assert len(restored) == 2
        assert all(h["node_id"] != "shopping_agent"
                   for h in restored.search("쇼핑 상품", top_k=5))
