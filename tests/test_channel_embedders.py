"""채널별 임베더 분리 — 노드와 청크는 다른 모델이 이긴다.

**실측이 이 기능을 요구했다.** 같은 골든셋(46 케이스)으로 두 채널을 재면 모든
대안 모델이 **노드에서 이기고 청크에서 진다**:

    모델           dim    노드 hit@1   노드 MRR   청크 hit@1   청크 MRR
    ────────────────────────────────────────────────────────────────────
    ko-sroberta    768    0.5217       0.6232     0.5217       0.6764
    bge-m3        1024    0.6304       0.7007     0.4783       0.6322
    KURE-v1       1024    0.5652       0.6511     0.4565       0.6065
    e5-base(+pfx)  768    0.5435       0.6308     0.4348       0.6076

원인은 텍스트 길이로 보인다 — 노드 텍스트는 짧은 개체명+정의이고 청크는 조문
전체(긴 문단)다. 검색 특화 모델(bge-m3/KURE/e5)은 짧은 질의↔짧은 passage 에
최적화됐고, ko-sroberta 는 NLI(문장 유사도)로 학습돼 문장↔긴문단에 강하다.

**접두사 지원은 만들지 않는다.** 최적 조합(노드=bge-m3, 청크=ko-sroberta)에
접두사가 한 번도 등장하지 않는다 — bge-m3 도 ko-sroberta 도 접두사가 없다.
쓰지 않을 인프라를 미리 만들지 않는다(YAGNI). e5 계열을 쓸 근거가 생기면 그때
만든다.

**기본값은 바꾸지 않는다.** 채널 분리 *능력*만 넣고, 실제 모델 선택은 융합(RRF)
지표로 재고 나서 정한다 — 노드 임베더 개선은 진입 노드를 통해 채널 B 에도
기여하므로 채널 하나만 봐서는 최종 이득을 알 수 없다(이 세션에서 두 번 데였다).
"""
import os

import pytest

from ontology.core import semantic_index as si


class TestChunkModelSetting:
    def test_defaults_to_the_node_model(self):
        """미설정이면 오늘과 **완전히** 같아야 한다 — 하위호환 관문."""
        assert si.DEFAULT_CHUNK_EMBEDDING_MODEL == si.DEFAULT_EMBEDDING_MODEL

    def test_resolver_reads_env(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_CHUNK_EMBEDDING_MODEL", "BAAI/bge-m3")
        assert si.resolve_chunk_model() == "BAAI/bge-m3"

    def test_resolver_falls_back_to_node_model(self, monkeypatch):
        monkeypatch.delenv("ONTOLOGY_CHUNK_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "some/node-model")
        assert si.resolve_chunk_model() == "some/node-model"

    def test_blank_env_is_not_a_model_name(self, monkeypatch):
        """빈 문자열을 모델명으로 쓰면 로드가 조용히 실패해 검색이 죽는다."""
        monkeypatch.setenv("ONTOLOGY_CHUNK_EMBEDDING_MODEL", "   ")
        assert si.resolve_chunk_model() == si.resolve_node_model()


class TestEmbedderCache:
    """임베더는 **모델별로 프로세스당 하나**다.

    종전에는 `_load_default_embed_fn()` 이 호출마다 SentenceTransformer 를 새로
    만들었다 — SemanticIndex · ChunkIndex · ES 백엔드가 각자 부르므로 같은 모델을
    3번 로드했다(각 3.5s + 메모리). 채널을 둘로 나누면 그 낭비가 곱절이 된다.
    """

    def test_same_model_returns_the_same_object(self, monkeypatch):
        made = []

        def _fake(model_id):
            made.append(model_id)
            return lambda texts: [[0.0]] * len(texts)

        monkeypatch.setattr(si, "_build_embed_fn", _fake)
        si.reset_embedders()
        a = si.get_embed_fn("m1")
        b = si.get_embed_fn("m1")
        assert a is b and made == ["m1"]

    def test_different_models_are_separate(self, monkeypatch):
        monkeypatch.setattr(si, "_build_embed_fn",
                            lambda mid: (lambda texts: [[0.0]] * len(texts)))
        si.reset_embedders()
        assert si.get_embed_fn("m1") is not si.get_embed_fn("m2")

    def test_failure_is_none_and_not_retried_forever(self, monkeypatch):
        """로드 실패를 매번 재시도하면 질의마다 수 초를 잃는다."""
        calls = []

        def _boom(model_id):
            calls.append(model_id)
            return None

        monkeypatch.setattr(si, "_build_embed_fn", _boom)
        si.reset_embedders()
        assert si.get_embed_fn("bad") is None
        assert si.get_embed_fn("bad") is None
        assert calls == ["bad"]          # 한 번만 시도

    def test_reset_clears(self, monkeypatch):
        monkeypatch.setattr(si, "_build_embed_fn",
                            lambda mid: (lambda texts: [[0.0]] * len(texts)))
        si.reset_embedders()
        first = si.get_embed_fn("m1")
        si.reset_embedders()
        assert si.get_embed_fn("m1") is not first


class TestChunkIndexUsesChunkModel:
    def test_chunk_index_resolves_the_chunk_model(self, monkeypatch, tmp_path):
        """배선 관문 — 설정만 있고 청크 색인이 안 쓰면 측정과 라이브가 갈린다."""
        from ontology.builder.models import Chunk
        from ontology.core.chunk_index import ChunkIndex
        from ontology.core.chunk_store import ChunkStore

        asked = []
        monkeypatch.setattr(si, "get_embed_fn",
                            lambda mid: asked.append(mid) or
                            (lambda texts: [[0.1]] * len(texts)))
        monkeypatch.setenv("ONTOLOGY_CHUNK_EMBEDDING_MODEL", "chunk/model")
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "node/model")

        store = ChunkStore(namespace="cens", path=tmp_path / "c.jsonl")
        store.add(Chunk(text="본문", source="s.pdf", index=0,
                        char_start=0, char_end=2))
        index = ChunkIndex(store)
        index._resolve_embed_fn()
        assert asked == ["chunk/model"]

    def test_semantic_index_resolves_the_node_model(self, monkeypatch):
        asked = []
        monkeypatch.setattr(si, "get_embed_fn",
                            lambda mid: asked.append(mid) or
                            (lambda texts: [[0.1]] * len(texts)))
        monkeypatch.setenv("ONTOLOGY_CHUNK_EMBEDDING_MODEL", "chunk/model")
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "node/model")
        idx = si.SemanticIndex()
        assert idx.embed_fn is not None
        assert asked == ["node/model"]


class TestQueryDimResolution:
    """`_resolve_query_dim` 은 문자열 추론이라 1024 모델에 768 을 돌려줬다.

    그러면 RLConfig.state_dim 이 896 으로 계산되는데 실제 임베딩은 1024 라
    **정책망 입력층에서 크래시**한다 — config 주석이 경고한 바로 그 상황이다.
    """

    def test_known_1024_models(self, monkeypatch):
        from ontology.ml.config import _resolve_query_dim
        for model in ("BAAI/bge-m3", "nlpai-lab/KURE-v1",
                      "dragonkue/BGE-m3-ko"):
            monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", model)
            monkeypatch.delenv("ONTOLOGY_ML_EMBEDDING_MODEL", raising=False)
            monkeypatch.delenv("ONTOLOGY_ML_QUERY_DIM", raising=False)
            assert _resolve_query_dim() == 1024, model

    def test_ml_follows_the_ontology_embedder_setting(self, monkeypatch):
        """**정렬 관문.** 종전에는 ML 전용 변수만 읽어서, 온톨로지 임베더를 바꿔도
        여기는 모르는 채 ko-sroberta 로 남았다 — 차원이 어긋나 정책망이 크래시한다.
        """
        from ontology.ml.config import _resolve_embedding_model
        monkeypatch.delenv("ONTOLOGY_ML_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "BAAI/bge-m3")
        assert _resolve_embedding_model() == "BAAI/bge-m3"

    def test_ml_override_still_wins(self, monkeypatch):
        """롤백 경로는 유지 — ML 전용 변수가 더 강하다."""
        from ontology.ml.config import _resolve_embedding_model
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "BAAI/bge-m3")
        monkeypatch.setenv("ONTOLOGY_ML_EMBEDDING_MODEL", "jhgan/ko-sroberta-nli")
        assert _resolve_embedding_model() == "jhgan/ko-sroberta-nli"

    def test_known_384_models(self, monkeypatch):
        from ontology.ml.config import _resolve_query_dim
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL",
                           "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        monkeypatch.delenv("ONTOLOGY_ML_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("ONTOLOGY_ML_QUERY_DIM", raising=False)
        assert _resolve_query_dim() == 384

    def test_default_768_unchanged(self, monkeypatch):
        from ontology.ml.config import _resolve_query_dim
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "jhgan/ko-sroberta-nli")
        monkeypatch.delenv("ONTOLOGY_ML_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("ONTOLOGY_ML_QUERY_DIM", raising=False)
        assert _resolve_query_dim() == 768

    def test_env_override_still_wins(self, monkeypatch):
        """알려지지 않은 모델을 쓸 때의 탈출구 — 추론보다 명시가 강하다."""
        from ontology.ml.config import _resolve_query_dim
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "BAAI/bge-m3")
        monkeypatch.delenv("ONTOLOGY_ML_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("ONTOLOGY_ML_QUERY_DIM", "512")
        assert _resolve_query_dim() == 512
