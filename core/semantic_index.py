"""
Semantic Index — embedding-based similarity search over graph nodes.

Fills the gap that pure graph traversal cannot: queries phrased with words
that do not literally appear in the graph ("날씨 알려주는 애") still find
their entry nodes via embedding similarity. Graph traversal then expands
from those entry points (see KnowledgeGraphEngine.find_agents_semantic).

Design notes:
- The embedding function is injectable (tests pass a deterministic fake;
  production lazily loads sentence-transformers). If no embedder is
  available the index degrades to empty results — never raises.
- Vectors are L2-normalized so dot product == cosine similarity.
- Scale is ~1e3 nodes: a numpy matrix product is milliseconds, so no
  vector-DB dependency is warranted.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

from .vector_backend import VectorBackend

# NOT the ml/config.py model (paraphrase-multilingual-MiniLM-L12-v2): that
# model was measured to produce degenerate Korean embeddings in this
# environment — unrelated Korean sentences at 0.96 cosine (English pairs are
# fine). ko-sroberta separates them correctly (unrelated ≈0.3, related ≈0.7).
# ─── 임베더 교체 측정 (2026-07-31) ─────────────────────────────────
#
# `compose_node_text` 의 A1 진단("긴 질문 ↔ 짧은 개체명" 비대칭)이 지목한 해법을
# 실제로 측정했다. 46 케이스, target=node, 실제 모델:
#
#   모델                       dim   hit@1    hit@5    hit@10   MRR      진단질의
#   ─────────────────────────────────────────────────────────────────────────────
#   ko-sroberta-nli (현재)     768   0.5217   0.7609   0.8478   0.6232   C50 top5 밖
#   e5-small +접두사           384   0.5652   0.7609   0.7826   0.6330   C50 top5 밖
#   e5-base  +접두사           768   0.5435   0.7826   0.8696   0.6308   **C50 5위**
#   e5-small 접두사 없음       384   0.4565   0.7391   0.7826   0.5743   C50 top5 밖
#
# 읽어야 할 둘:
#  1) **A1 가설이 검증됐다.** e5-base 가 "유방암 진단 시 보장되나?" 에서 정답
#     Disease:C50 을 처음으로 top5 에 넣는다. 노드 채널의 전 지표에서 이긴다.
#  2) **접두사가 결정적이다.** e5 를 접두사 없이 쓰면 hit@1 0.5652 → 0.4565 로
#     현재 모델보다 **나빠진다**. 환경변수만 바꾸는 교체는 실패한다.
#  (e5-small 은 hit@1 이 최고지만 리콜을 잃고 진단질의를 못 고친다. 그리고 384 라
#   GNN+RL 의 state_dim 이 896→512 로 바뀌어 학습된 정책을 못 쓴다.)
#
# ─── 그런데 청크 채널을 재니 결론이 뒤집혔다 ─────────────────────────
#
# ⚠️ **위 표는 노드 채널만 잰 것이다.** 임베더는 세 곳을 구동한다: 진입 노드 선택 ·
# **청크 색인 검색(채널 A)** · 그 둘의 RRF 융합. 청크 쪽을 같은 골든셋으로
# 재니(target=evidence, 92 청크) e5-base 가 **크게 진다**:
#
#   채널            모델              hit@1    hit@5    hit@10   MRR
#   ────────────────────────────────────────────────────────────────────
#   노드            ko-sroberta       0.5217   0.7609   0.8478   0.6232
#   노드            e5-base +prefix   0.5435   0.7826   0.8696   0.6308  (+1건씩)
#   **청크**        ko-sroberta       0.5217   0.8913   0.9783   0.6764
#   **청크**        e5-base +prefix   0.4348   0.8261   0.8913   0.6076  (**−3~4건**)
#
# 청크 쪽 손실(hit@1 −4건, MRR −0.0688)이 노드 쪽 이득(+1건, 잡음 경계)보다 크다.
# 원인은 텍스트 길이로 보인다 — 청크는 조문 전체(긴 문단)이고 ko-sroberta 는
# NLI(문장 유사도)로 학습돼 문장↔긴문단에 강하다. e5 는 짧은 질의↔짧은 passage
# 검색에 최적화됐다. 색인 비용도 1.45s → 5.25s(3.6배)로 벌어진다.
#
# e5 는 교체하지 않는다 — 청크 손실이 노드 이득보다 크다. bge-m3 계열도 격리
# 측정으로는 같은 모양이었다 (46 케이스):
#
#   모델           dim    노드 hit@1   노드 MRR   청크 hit@1   청크 MRR
#   ────────────────────────────────────────────────────────────────────
#   ko-sroberta    768    0.5217       0.6232     0.5217       0.6764
#   bge-m3        1024    0.6304       0.7007     0.4783       0.6322
#   KURE-v1       1024    0.5652       0.6511     0.4565       0.6065
#
# ─── 그런데 융합(RRF)을 재니 또 뒤집혔다 ─────────────────────────────
#
# ⚠️ **격리 측정과 통합 측정이 다르다.** 위 표는 두 채널을 **따로** 잰 것이다.
# 실제 파이프라인은 채널 A(청크 임베딩) + 채널 B(진입 노드 경유 청크)를 RRF 로
# 융합하고, 노드 임베더는 **진입 노드를 통해 채널 B 에도 기여**한다:
#
#   구성                            노드 h@1  노드 MRR │ 융합청크 h@1  h@5     MRR
#   ──────────────────────────────────────────────────────────────────────────────
#   둘 다 ko-sroberta (현재)        0.5217    0.6428   │ 0.6957      1.0000  0.8333
#   노드=bge-m3, 청크=ko-sroberta   0.6304    0.7018   │ 0.7174      0.9565  0.8167 ←최악
#   **둘 다 bge-m3**                0.6304    0.7018   │ 0.7609      1.0000  0.8732
#
# 격리 측정에서 bge-m3 는 청크에 **나빴는데**(0.4783 vs 0.5217) 융합에서는 오히려
# **좋다**(0.7609 vs 0.6957) — 좋은 진입 노드가 채널 B 를 개선해 채널 A 손실을
# 넘는다. 그리고 **하이브리드가 가장 나쁘다**: 두 채널이 다른 임베딩 공간이라
# 융합이 어긋난다(h@5 만 0.9565 로 떨어진다).
#
# **교훈 둘.** ① 자가 하나면 부족하다 — 노드만 재고 "전 지표에서 이긴다"고 적었다가
# 청크에서 뒤집혔다. ② **채널을 따로 재는 것과 융합을 재는 것이 또 다르다** —
# 격리 측정은 하이브리드를 가리켰지만 융합은 통일을 가리킨다.
#
# 그래서 채널 분리(`resolve_chunk_model`)는 **측정 도구로 남긴다**. 기본값이 통일이라
# 무해하고, 이 결론을 낼 수 있게 해준 것이 그 기능이다. 최적 설정에서는 쓰이지 않는다.
#
# **bge-m3 통일이 순이득이지만 기본값은 아직 바꾸지 않았다** — 1024 차원의 대가가
# 있다: ES 백엔드 `dim=768` 재설정 + 인덱스 재생성, RL 정책 재학습(state_dim
# 896→1152), 색인 8배(2.5s→20s), 모델 2.2GB. 켜려면:
#   ONTOLOGY_EMBEDDING_MODEL=BAAI/bge-m3   (ml/config 가 이 값을 따라간다)
# 접두사 지원은 만들지 않았다 — bge-m3 도 ko-sroberta 도 접두사가 없어서 최적
# 조합에 한 번도 등장하지 않는다(YAGNI). e5 계열을 쓸 근거가 생기면 그때 만든다.
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ONTOLOGY_EMBEDDING_MODEL", "jhgan/ko-sroberta-nli")

# Node types indexed by default — the "semantic surface" of the graph
DEFAULT_NODE_TYPES = ("agent", "capability", "tag", "query_category")

EmbedFn = Callable[[List[str]], np.ndarray]


# 청크 채널 임베더 — **노드와 다른 모델이 이긴다** (46 케이스 실측, 위 표 참고).
# 모든 검색 특화 모델이 노드에서 이기고 청크에서 진다. 미설정이면 노드 모델과
# 같으므로 종전 동작과 완전히 같다.
DEFAULT_CHUNK_EMBEDDING_MODEL = os.environ.get(
    "ONTOLOGY_CHUNK_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL


def resolve_node_model() -> str:
    """노드 채널 모델. env 를 **매번 읽는다** — 모듈 상수는 import 시점에
    고정되므로 테스트·런타임 변경이 반영되지 않는다."""
    return (os.environ.get("ONTOLOGY_EMBEDDING_MODEL", "").strip()
            or DEFAULT_EMBEDDING_MODEL)


# 모델 계열별 출력 차원. 하드코딩이지만 **어휘가 아니라 모델의 인터페이스 사실**이고,
# 틀리면 조용히 실패하지 않고 정책망 입력층에서 크래시한다.
#
# 이 표가 **커널에 있는 이유**: ml/config 에 두면 상태 조회(관리 콘솔)가 차원을
# 알려고 ml 패키지를 import 하고 그게 torch 를 끌고 온다 — 조회 한 번에 수십 초다.
# ml/config 가 여기를 import 한다(에이전트 스택 → 커널 방향은 정상).
_DIM_HINTS = (
    ("bge-m3", 1024),
    ("kure", 1024),          # nlpai-lab/KURE-v1 — bge-m3 기반
    ("e5-large", 1024),
    ("minilm", 384),
    ("e5-small", 384),
)
DEFAULT_EMBEDDING_DIM = 768   # sroberta/roberta-base/e5-base 계열


def resolve_embedding_dim(model_id: Optional[str] = None) -> int:
    """모델명으로 출력 차원을 추론한다 (모델을 **로드하지 않는다**).

    알려지지 않은 모델은 768 로 가정한다 — 틀리면 정책망이 크래시하므로
    `ONTOLOGY_ML_QUERY_DIM` 이 탈출구다(ml/config 참고).
    """
    model = (model_id or resolve_node_model()).lower()
    for needle, dim in _DIM_HINTS:
        if needle in model:
            return dim
    return DEFAULT_EMBEDDING_DIM


def resolve_chunk_model() -> str:
    """청크 채널 모델. 미설정·공백이면 노드 모델을 쓴다 —
    빈 문자열을 모델명으로 넘기면 로드가 조용히 실패해 검색이 통째로 죽는다."""
    return (os.environ.get("ONTOLOGY_CHUNK_EMBEDDING_MODEL", "").strip()
            or resolve_node_model())


def _looks_natural(text: str) -> bool:
    """Natural-language text (spaces or non-ASCII) vs identifier-like."""
    return " " in text or not text.isascii()


def compose_node_text(node_id: str, attrs: Dict[str, Any]) -> str:
    """Build the text a node is embedded under.

    Semantic fields come FIRST and identifiers go last in parentheses:
    measured on the real model, id tokens placed up front poison the
    sentence embedding and can rank an unrelated node higher
    ("weather agent 도시 날씨..." lost to shopping on weather queries;
    "도시 날씨... (weather agent)" wins). Nodes with no semantic fields
    fall back to their id so they stay findable.

    ⚠️ **이름 반복을 줄이려는 가설은 측정에서 기각됐다 — 다시 시도하지 말 것.**
    `Disease:C50( 유방의 악성 신생물 )` 의 텍스트는 "유방의 악성 신생물" 을 3번,
    "C50" 을 3번 담는다(name + aliases + node_id). 그 중복이 질의어를 묻는
    잡음이라고 보고 두 가지를 시험했다 (43 케이스, target=node, 실제 임베더):

      버전                                   hit@1    hit@5    MRR
      ────────────────────────────────────────────────────────────────
      현재                                   0.5349   0.7674   0.6376
      node_id 중복 제거(괄호에 타입만)       0.5116   0.7442   0.6027
      + 이름의 부분문자열인 별칭 제거        0.4884   0.7442   0.5853

    단조적으로 나빠진다. **반복은 잡음이 아니라 가중치**다 — 같은 토큰이 여러 번
    들어가면 문장 벡터에서 비중이 커져 노드 신원이 강화된다.

    그 가설을 부른 관찰("유방암 진단 시 보장되나?" 의 진입 top5 에 C50 이 없다)의
    진짜 원인은 노드 쪽이 아니라 **질의 쪽 비대칭**이었다:
      "유방암"                    → C50 **1위** (0.635)
      "유방암 진단 시 보장되나?"  → C50 top5 밖 (피보험자·암진단비가 올라옴)
    서술부가 문장 임베딩을 지배해 핵심 명사가 희석된다. ko-sroberta 는 NLI(문장
    유사도)로 학습돼 문장↔문장에 강하고 **긴 질문 ↔ 짧은 개체명**에는 약하다.
    해법은 노드 텍스트가 아니라 비대칭 학습 임베더(e5 계열의 query:/passage:
    접두사)이며, 그건 별도 결정이다(캐시 무효화 + 전 네임스페이스 재색인).
    """
    semantic: List[str] = []
    for field in ("display_name", "description", "definition", "category"):
        value = attrs.get(field)
        if value and str(value) not in semantic:
            semantic.append(str(value))
    aliases = attrs.get("aliases")
    if isinstance(aliases, (list, tuple)):
        for alias in aliases:
            if alias and str(alias) not in semantic:
                semantic.append(str(alias))

    name = str(attrs.get("name", "") or "")
    identifiers = [str(node_id)]
    spaced = str(node_id).replace("_", " ")
    if spaced != str(node_id):
        identifiers.append(spaced)
    if name and name not in identifiers:
        # a natural-language name ("기상 날씨") is semantic content;
        # an id-like name ("weather_agent") is just another identifier
        if _looks_natural(name) and name not in semantic:
            semantic.insert(0, name)
        else:
            identifiers.append(name)

    if semantic:
        return f"{' '.join(semantic)} ({' '.join(identifiers)})"
    return " ".join(identifiers)


def _build_embed_fn(model_id: str) -> Optional[EmbedFn]:
    """실제 임베더 하나를 만든다 (없으면 None). 캐시는 get_embed_fn 이 한다."""
    from .health_signals import report
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, device="cpu")

        def embed(texts: List[str]) -> np.ndarray:
            return np.asarray(model.encode(texts, show_progress_bar=False),
                              dtype=np.float32)

        logger.info(f"🔎 Semantic index embedder loaded: {model_id}")
        report(f"embedder:{model_id}", True, "loaded")
        return embed
    except Exception as e:
        logger.warning(f"⚠️ Semantic embedder unavailable ({e}) — semantic search disabled")
        # 의미 검색이 통째로 꺼지는 지점이다 — 경고 한 줄로 끝내면 안 된다 (2026-10-05:
        # scipy 가 깨져 전 네임스페이스 검색이 0건이었는데 이틀 가까이 아무도 몰랐다)
        report(f"embedder:{model_id}", False, f"{type(e).__name__}: {e}")
        return None


# 모델별 임베더 — **프로세스당 하나**. 종전에는 호출마다 SentenceTransformer 를
# 새로 만들었고, SemanticIndex · ChunkIndex · ES 백엔드가 각자 부르므로 같은 모델을
# 3번 로드했다(각 3.5s + 메모리). 채널을 둘로 나누면 그 낭비가 곱절이 된다.
# 실패(None)도 캐시한다 — 매 질의마다 수 초짜리 로드를 재시도하면 안 된다.
_embedders: Dict[str, Optional[EmbedFn]] = {}

# ─── embed 콜 카운터 (실험 하네스 Phase 1) ──────────────────────────
#
# 비용 축의 분모("이 조합이 임베딩을 몇 번 불렀나")를 러너가 잴 수 있게 한다.
# **라이브 오버헤드 0 이 계약**: 카운터가 없으면(기본) 호출당 ContextVar.get()
# 체크 1회 후 원 함수로 위임한다 — 임베딩 추론(ms~s) 대비 무시 가능.
_embed_counter: ContextVar[Optional[list]] = ContextVar(
    "embed_counter", default=None)


@contextmanager
def count_embeds():
    """`with count_embeds() as calls:` — 구간 안의 embed 호출 1회당 원소
    1개(배치 크기)가 calls 에 append 된다. 중첩하면 안쪽 구간은 안쪽
    리스트에만 잡힌다 (ContextVar set/reset 의 자연스러운 의미)."""
    calls: list = []
    token = _embed_counter.set(calls)
    try:
        yield calls
    finally:
        _embed_counter.reset(token)


def _make_counted(fn: EmbedFn) -> EmbedFn:
    """카운터를 **호출 시점**에 평가하는 얇은 래퍼. 카운터 리스트를 래핑
    시점에 캡처하면 컨텍스트가 끝난 뒤의 호출까지 세거나, 다른 실험의
    호출이 섞인다 — 그래서 ContextVar 를 매 호출 읽는다."""
    def counted(texts: List[str]):
        calls = _embed_counter.get()
        if calls is not None:
            calls.append(len(texts))
        return fn(texts)

    counted.__wrapped__ = fn      # 원본 추적 — 캐시 항목 교체 감지에 쓴다
    return counted


# 래퍼 메모 — 원본 캐시(_embedders)와 **분리**해 둔다. 원본 캐시에 감싼
# 함수를 저장하면 안 되고(카운터는 호출 시점 평가·원본 계약 보존), 그렇다고
# 호출마다 새 래퍼를 만들면 "같은 모델 = 같은 객체" 캐시 계약이 깨진다
# (test_channel_embedders 가 계약으로 고정). 원본이 교체되면(테스트가 캐시에
# 직접 심는 경우) __wrapped__ 대조로 래퍼를 다시 만든다.
_wrapped: Dict[str, EmbedFn] = {}


def get_embed_fn(model_id: Optional[str] = None) -> Optional[EmbedFn]:
    """모델별 공유 임베더. model_id 미지정이면 노드 채널 모델."""
    key = (model_id or resolve_node_model()).strip() or resolve_node_model()
    if key not in _embedders:
        _embedders[key] = _build_embed_fn(key)
    raw = _embedders[key]
    if raw is None:
        return None
    wrapper = _wrapped.get(key)
    if wrapper is None or getattr(wrapper, "__wrapped__", None) is not raw:
        wrapper = _make_counted(raw)
        _wrapped[key] = wrapper
    return wrapper


def reset_embedders() -> None:
    """테스트용 — 캐시를 비운다."""
    _embedders.clear()
    _wrapped.clear()


def _load_default_embed_fn(model_id: Optional[str] = None) -> Optional[EmbedFn]:
    """하위호환 별칭 (기존 호출부가 인자 없이 부른다)."""
    return get_embed_fn(model_id)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SemanticIndex(VectorBackend):
    """In-memory embedding index over graph nodes — the tier-0 (memory)
    VectorBackend: numpy brute-force, no persistence. Inherits tier_name
    == 'memory' from VectorBackend."""

    def __init__(self, embed_fn: Optional[EmbedFn] = None, auto_default: bool = True):
        self._embed_fn = embed_fn
        self._auto_default = auto_default
        self._default_attempted = False

        self._ids: List[str] = []
        self._row_of: Dict[str, int] = {}
        self._vectors: Optional[np.ndarray] = None  # (n, dim), L2-normalized
        self._texts: Dict[str, str] = {}
        self._types: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def embed_fn(self) -> Optional[EmbedFn]:
        # 노드 채널 — 청크와 다른 모델을 쓸 수 있다 (resolve_chunk_model 참고).
        if self._embed_fn is None and self._auto_default and not self._default_attempted:
            self._default_attempted = True
            self._embed_fn = get_embed_fn(resolve_node_model())
        return self._embed_fn

    def upsert(self, node_id: str, text: str, node_type: str) -> bool:
        """Index (or re-index) a node. Returns False when nothing changed
        or no embedder is available — re-embedding is the expensive part,
        so unchanged text is skipped."""
        if self.embed_fn is None:
            return False
        if self._texts.get(node_id) == text:
            return False

        vector = _normalize(np.atleast_2d(self.embed_fn([text])).astype(np.float32))

        if node_id in self._row_of:
            self._vectors[self._row_of[node_id]] = vector[0]
        else:
            self._row_of[node_id] = len(self._ids)
            self._ids.append(node_id)
            self._vectors = vector if self._vectors is None \
                else np.vstack([self._vectors, vector])

        self._texts[node_id] = text
        self._types[node_id] = node_type
        return True

    def remove(self, node_id: str) -> bool:
        if node_id not in self._row_of:
            return False
        row = self._row_of.pop(node_id)
        self._ids.pop(row)
        self._vectors = np.delete(self._vectors, row, axis=0)
        self._texts.pop(node_id, None)
        self._types.pop(node_id, None)
        # rows after the removed one shift up by one
        for nid, r in self._row_of.items():
            if r > row:
                self._row_of[nid] = r - 1
        return True

    def _live_nodes(self, graph, node_types):
        """그래프에서 이 색인이 담당하는 노드들. tier 1(npy)도 이걸 쓴다 —
        '무엇이 살아 있는가'의 정의가 두 벌이면 tier 마다 다르게 정리한다."""
        for node_id, attrs in graph.nodes(data=True):
            if node_types is None or attrs.get("type", "") in node_types:
                yield node_id, attrs

    def prune_to_graph(self, graph, node_types=DEFAULT_NODE_TYPES) -> int:
        """그래프에서 사라진 노드의 벡터를 걷어낸다. 제거한 수를 돌려준다.

        **없으면 검색이 유령을 돌려준다.** upsert 만 하는 색인은 노드 삭제를
        모른다 — 병합(node_merge)뿐 아니라 검수 거절(reject_node)도 노드를
        지우므로, 거절된 개체가 계속 검색되고 있었다. 인용을 클릭하면 빈 화면이 된다.

        타입 필터 밖으로 나간 노드도 제거 대상이다: 이 색인 기준으로는 사라진 것이다.

        임베딩이 필요 없으므로 임베더가 없어도 동작한다 — 모델이 없다고 유령을
        남기면 degrade 가 아니라 오답이다.
        """
        live = {node_id for node_id, _ in self._live_nodes(graph, node_types)}
        stale = [node_id for node_id in list(self._ids) if node_id not in live]
        for node_id in stale:
            self.remove(node_id)
        if stale:
            logger.info(f"🔎 Semantic index: dropped {len(stale)} stale node(s)")
        return len(stale)

    def build_from_graph(self, graph, node_types=DEFAULT_NODE_TYPES) -> int:
        """(Re)index every graph node of the given types (None = all types).
        Returns the number of nodes (re-)embedded. Batches all texts into
        one encode call.

        정리(prune)를 **임베더 확인보다 먼저** 한다: 제거는 임베딩이 필요 없고,
        embed_fn 프로퍼티는 만지는 순간 sentence-transformers 를 로드한다.
        """
        self.prune_to_graph(graph, node_types=node_types)
        if self.embed_fn is None:
            return 0

        pending_ids: List[str] = []
        pending_texts: List[str] = []
        pending_types: List[str] = []
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "")
            if node_types is not None and node_type not in node_types:
                continue
            text = compose_node_text(node_id, attrs)
            if self._texts.get(node_id) == text:
                continue
            pending_ids.append(node_id)
            pending_texts.append(text)
            pending_types.append(node_type)

        if not pending_ids:
            return 0

        vectors = _normalize(np.atleast_2d(
            self.embed_fn(pending_texts)).astype(np.float32))
        for node_id, text, node_type, vector in zip(
                pending_ids, pending_texts, pending_types, vectors):
            if node_id in self._row_of:
                self._vectors[self._row_of[node_id]] = vector
            else:
                self._row_of[node_id] = len(self._ids)
                self._ids.append(node_id)
                self._vectors = np.atleast_2d(vector) if self._vectors is None \
                    else np.vstack([self._vectors, vector])
            self._texts[node_id] = text
            self._types[node_id] = node_type

        logger.info(f"🔎 Semantic index built: +{len(pending_ids)} nodes (total {len(self)})")
        return len(pending_ids)

    def search(self, query: str, top_k: int = 5,
               node_types: Optional[List[str]] = None,
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Cosine-similarity search. Returns [{node_id, node_type, score}]
        sorted by score descending. Empty list when the index is empty,
        the query is blank, or no embedder is available."""
        if not query or not query.strip() or not self._ids or self.embed_fn is None:
            return []

        query_vector = _normalize(
            np.atleast_2d(self.embed_fn([query])).astype(np.float32))[0]
        scores = self._vectors @ query_vector

        order = np.argsort(-scores)
        results: List[Dict[str, Any]] = []
        for row in order:
            node_id = self._ids[row]
            if node_types and self._types.get(node_id) not in node_types:
                continue
            score = float(scores[row])
            if score < min_score:
                break  # scores are sorted — nothing better follows
            results.append({
                "node_id": node_id,
                "node_type": self._types.get(node_id, ""),
                "score": round(score, 4),
            })
            if len(results) >= top_k:
                break
        return results
