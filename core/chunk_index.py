"""
Chunk index — 원문 청크에 대한 의미/하이브리드 검색.

축 3. 축 2 가 원문을 남겼지만 검색은 부분문자열(ChunkStore.search_text)뿐이었다.
"청약을 무를 수 있나?" 로는 "청약철회권" 청크를 못 찾는다 — 글자가 안 겹친다.

핵심 재사용: VectorBackend 계약이 (id, text, type) 이라 **노드용으로 만든
백엔드가 청크에도 그대로 쓰인다**. 그래서 tier 선택이 그대로 따라온다:
- memory / npy      → 임베딩 의미 검색 (npy 는 디스크 캐시로 콜드스타트 제거)
- elasticsearch     → BM25 + 벡터 하이브리드

청크에서 하이브리드가 특히 중요한 이유: 구절에는 조문번호("제21조")·고유명사·
금액처럼 **문자 그대로 맞아야 하는** 토큰이 섞인다. 임베딩만으로는 이런 걸
놓치고, 그래서 aicoach 도 청크 검색에는 BM25 를 섞었다 (recall@5 0.944).
노드 라벨 검색과 달리 청크 검색은 하이브리드가 기본값이어야 한다.

임베더가 없으면 부분문자열 폴백으로 degrade — raise 하지 않는다. 인덱스는
파생 데이터고, 원문은 저장소에 이미 있다.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from .chunk_store import StoredChunk

# 청크의 "타입" — VectorBackend 계약이 node_type 을 요구하지만 청크에는
# 타입이 없다. 단일 값을 넣어 계약만 만족시킨다.
CHUNK_TYPE = "chunk"

Hit = Tuple[StoredChunk, float]


class ChunkIndex:
    """ChunkStore 위의 검색 인덱스. 백엔드는 주입 가능(tier 선택 그대로)."""

    def __init__(self, store, backend=None, embed_fn: Optional[Callable] = None,
                 auto_default: bool = True, namespace: Optional[str] = None,
                 backend_override: Optional[str] = None):
        self.store = store
        self._embed_fn = embed_fn
        self._auto_default = auto_default
        self._namespace = namespace or getattr(store, "namespace", "default")
        self._backend_override = backend_override
        self._backend = backend
        self._indexed: Dict[str, str] = {}  # chunk_id → 인덱싱된 본문

    # ─── 백엔드 ──────────────────────────────────────────────────────

    @property
    def backend(self):
        """첫 사용 때 tier 를 고른다. 청크 수 기준 — 노드보다 훨씬 많으므로
        같은 임계값이 다른 tier 를 뽑는다(그게 맞다)."""
        if self._backend is None:
            from .vector_backend import select_backend
            self._backend = select_backend(
                len(self.store), embed_fn=self._resolve_embed_fn(),
                override=self._backend_override,
                namespace=f"chunks_{self._namespace}")
            # 디스크 캐시 로드 — 대형 청크 네임스페이스(예: AI-Coach 7,643)가
            # 프로세스마다 전량 재임베딩하지 않도록(느림·OOM 로 서버가 죽었다).
            # npy tier 만 해당(load_from_disk 보유). 로드되면 _indexed 를 캐시된
            # 본문으로 복원해 refresh 가 재임베딩을 건너뛴다.
            if (hasattr(self._backend, "load_from_disk")
                    and self._backend.load_from_disk()):
                self._indexed = dict(getattr(self._backend, "_texts", {}) or {})
        return self._backend

    def _resolve_embed_fn(self):
        if self._embed_fn is None and self._auto_default:
            # 청크 채널은 노드와 **다른 모델**을 쓸 수 있다. 실측에서 모든 검색
            # 특화 모델이 노드에서 이기고 청크에서 졌다 — 청크는 조문 전체(긴
            # 문단)라 문장 유사도로 학습된 모델이 강하다 (semantic_index 상단 표).
            from . import semantic_index as _si
            self._embed_fn = _si.get_embed_fn(_si.resolve_chunk_model())
        return self._embed_fn

    def _has_embedder(self) -> bool:
        return self._resolve_embed_fn() is not None

    # ─── 인덱싱 ──────────────────────────────────────────────────────

    def refresh(self) -> int:
        """저장소의 청크를 인덱스에 반영한다. 본문이 안 바뀐 것은 건너뛴다.

        재임베딩이 비싼 부분이므로 스킵이 핵심이다 (SemanticIndex 가 노드에
        대해 하는 것과 같은 규약).
        """
        if not self._has_embedder():
            return 0
        indexed = 0
        for stored in self.store.all():
            if self._indexed.get(stored.chunk_id) == stored.text:
                continue
            if self.backend.upsert(stored.chunk_id, stored.text, CHUNK_TYPE):
                self._indexed[stored.chunk_id] = stored.text
                indexed += 1
        if indexed:
            logger.info(f"🔎 Chunk index refreshed: +{indexed} chunks "
                        f"(total {len(self._indexed)})")
            # 새로 임베딩한 게 있으면 디스크에 내린다 — 다음 프로세스는 로드만
            # 하고 재임베딩하지 않는다(build-once). npy tier 만 save_to_disk 보유.
            if hasattr(self.backend, "save_to_disk"):
                self.backend.save_to_disk()
        return indexed

    # ─── 검색 ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0,
               source: Optional[str] = None) -> List[Hit]:
        """청크 검색. [(StoredChunk, score)] 를 점수 내림차순으로 돌려준다.

        히트를 StoredChunk 로 해석해서 내보내는 것은 호출부가 id 를 들고 다시
        저장소를 조회하지 않게 하려는 것이다 — 인용은 항상 원문과 함께 온다.

        `source` 는 문서 필터("이 문서에서만"). 필터 시에는 **전량 채점 후 거른다**
        — 전역 top_k 를 먼저 자르고 거르면 필터 문서의 정답이 잘려 나간다
        (starvation). 백엔드는 브루트포스 규모(수백 청크)라 전량이 정확·저렴하다.
        """
        if not query or not query.strip():
            return []
        if not len(self.store):
            return []
        fetch_k = len(self.store) if source else top_k

        if not self._has_embedder():
            # 임베더 없음 → 부분문자열 폴백. 점수는 유사도가 아니므로 0.0 으로
            # 정직하게 표시한다 (없는 정밀도를 지어내지 않는다).
            # 필터는 이 경로에도 같은 계약으로 건다 — 한 경로만 거르면 환경에
            # 따라 필터가 새는 검색이 된다.
            hits = [(c, 0.0) for c in self.store.search_text(query, top_k=fetch_k)
                    if source is None or c.source == source]
            return hits[:top_k]

        self.refresh()
        raw = self.backend.search(query, top_k=fetch_k, min_score=min_score)

        hits: List[Hit] = []
        for entry in raw:
            stored = self.store.get(entry["node_id"])
            if stored is None:
                # 저장소에서 사라진 청크의 인덱스 잔재 (rebuild 후 등) —
                # 조용히 거른다. 인덱스는 파생 데이터, 저장소가 진실이다.
                continue
            if source is not None and stored.source != source:
                continue
            hits.append((stored, float(entry["score"])))
            if len(hits) >= top_k:
                break
        return hits


# ─── 네임스페이스 싱글턴 ────────────────────────────────────────────

_indices: Dict[str, ChunkIndex] = {}


def get_chunk_index(namespace: str = "default") -> ChunkIndex:
    """네임스페이스별 공유 ChunkIndex (ChunkStore 와 같은 수명·경계)."""
    if namespace not in _indices:
        from .chunk_store import get_chunk_store
        _indices[namespace] = ChunkIndex(store=get_chunk_store(namespace),
                                         namespace=namespace)
        logger.info(f"🔎 Chunk index initialized (namespace={namespace})")
    return _indices[namespace]


def reset_chunk_indices() -> None:
    """싱글턴 초기화 — 테스트 격리용."""
    _indices.clear()
