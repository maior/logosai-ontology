"""
Tier 1 (npy) 벡터 백엔드 — brute-force + 디스크 캐시.

축 3. 이전까지 5-tier 중 실물은 tier 0(memory)뿐이었고 npy/parallel/faiss/
distributed 는 create_backend 가 조용히 memory 를 돌려주는 스텁이었다. 게다가
인덱스가 디스크에 없어 **프로세스마다 전 노드를 재임베딩**했다 — heritage_us
3,623 노드면 매 기동마다 수십 초다.

tier 1 이 고치는 것은 딱 하나: 콜드스타트. 검색은 tier 0 과 동일한 정확
brute-force 이고(정확도 손실 0), 달라지는 건 벡터가 디스크에 남는다는 것뿐이다.

캐시 무효화가 이 파일의 대부분인 이유: 캐시는 파생 데이터라 틀리면 조용히
틀린 검색 결과를 낸다. 그래서 조금이라도 의심스러우면 버리고 다시 만든다
(깨진 파일 · 행 수 불일치 · 임베딩 차원 변경). 절대 raise 하지 않는다 —
캐시 때문에 서비스가 죽는 것은 본말전도다.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .semantic_index import (DEFAULT_EMBEDDING_MODEL, DEFAULT_NODE_TYPES,
                            SemanticIndex, compose_node_text)
from .vector_backend import TIER_NPY

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data"

CACHE_VERSION = 1


class NpyBackend(SemanticIndex):
    """SemanticIndex + 디스크 영속화. tier 0 의 검색 로직을 그대로 상속한다."""

    def __init__(self, embed_fn=None, auto_default: bool = True,
                 namespace: str = "default", cache_dir=None,
                 model_id: Optional[str] = None):
        super().__init__(embed_fn=embed_fn, auto_default=auto_default)
        self.namespace = namespace
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        # 캐시 유효성의 핵심 키. **차원이 아니라 모델 신원**으로 잡는다:
        # 서로 다른 모델이 같은 차원을 쓰는 일이 흔하다 (ko-sroberta 768 vs
        # gemini-embedding 768 — aicoach 는 실제로 둘 다 쓴다). 차원만 비교하면
        # 모델을 갈아끼워도 캐시가 살아남아 조용히 틀린 검색 결과를 낸다.
        # 게다가 모델명 비교는 임베딩 호출이 0회라 콜드스타트를 해치지 않는다.
        self.model_id = model_id or DEFAULT_EMBEDDING_MODEL

    @property
    def tier_name(self) -> str:
        return TIER_NPY

    # ─── 경로 ────────────────────────────────────────────────────────

    @property
    def vectors_path(self) -> Path:
        return self._cache_dir / f"vectors_{self.namespace}.npy"

    @property
    def meta_path(self) -> Path:
        return self._cache_dir / f"vectors_{self.namespace}.json"

    # ─── 영속화 ──────────────────────────────────────────────────────

    def save_to_disk(self) -> bool:
        """벡터(.npy) + 메타(.json)를 원자적으로 내린다 (KG 와 같은 규약).

        둘을 따로 쓰는 것은 numpy 배열을 JSON 에 넣으면 크기가 3배가 되고
        로드가 느려지기 때문이다. 대신 둘이 어긋날 수 있으므로 로드할 때
        행 수를 대조한다.
        """
        if self._vectors is None or not self._ids:
            return False
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            tmp_vectors = self.vectors_path.with_suffix(".npy.tmp")
            # 경로가 아니라 파일 핸들로 저장한다 — np.save 는 경로가 .npy 로
            # 끝나지 않으면 확장자를 덧붙여서 tmp 이름이 어긋난다.
            with open(tmp_vectors, "wb") as f:
                np.save(f, self._vectors)
            tmp_vectors.replace(self.vectors_path)

            meta = {
                "version": CACHE_VERSION,
                "namespace": self.namespace,
                "model_id": self.model_id,
                "dim": int(self._vectors.shape[1]),
                "ids": self._ids,
                "texts": self._texts,
                "types": self._types,
            }
            tmp_meta = self.meta_path.with_suffix(".json.tmp")
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            tmp_meta.replace(self.meta_path)

            logger.info(f"💾 Vector cache saved: {len(self._ids)} vectors "
                        f"→ {self.vectors_path}")
            return True
        except Exception as e:
            logger.error(f"Vector cache save failed: {e}")
            return False

    def load_from_disk(self) -> bool:
        """캐시를 읽는다. 조금이라도 의심스러우면 False 를 돌려주고 비운 채
        시작한다 — 틀린 캐시는 없는 캐시보다 나쁘다."""
        if not (self.vectors_path.exists() and self.meta_path.exists()):
            return False
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("version") != CACHE_VERSION:
                logger.info("🔎 Vector cache version mismatch — rebuilding")
                return False

            if meta.get("model_id") != self.model_id:
                # 모델을 갈아끼웠다. 옛 벡터는 새 쿼리 벡터와 같은 공간에
                # 있지 않으므로 내적 결과가 무의미하다.
                logger.info(f"🔎 Vector cache model '{meta.get('model_id')}' != "
                            f"'{self.model_id}' — rebuilding")
                return False

            vectors = np.load(self.vectors_path)
            ids = list(meta.get("ids") or [])

            if vectors.ndim != 2 or vectors.shape[0] != len(ids):
                logger.warning("⚠️ Vector cache row-count mismatch — discarding")
                return False

            if meta.get("dim") is not None and vectors.shape[1] != meta["dim"]:
                # 메타와 .npy 가 어긋났다 — 둘을 따로 쓰는 대가다
                logger.warning("⚠️ Vector cache dim mismatch — discarding")
                return False

            self._ids = ids
            self._row_of = {node_id: i for i, node_id in enumerate(ids)}
            self._vectors = vectors.astype(np.float32)
            self._texts = dict(meta.get("texts") or {})
            self._types = dict(meta.get("types") or {})
            logger.info(f"📂 Vector cache loaded: {len(ids)} vectors "
                        f"← {self.vectors_path}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Vector cache load failed ({e}) — rebuilding")
            return False

    # ─── 빌드 ────────────────────────────────────────────────────────

    # _live_nodes 는 기반 클래스(SemanticIndex)에 있다 — '무엇이 살아 있는가'의
    # 정의가 두 벌이면 tier 마다 다르게 정리하게 된다.

    def _has_pending(self, graph, node_types) -> bool:
        """재임베딩할 노드가 있는가 — **임베더 없이** 판정한다.

        텍스트 비교뿐이라 모델이 필요 없다. 이 판정을 먼저 하는 것이 tier 1 의
        핵심이다: 상위 build_from_graph 는 첫 줄에서 self.embed_fn 을 만지는데,
        그 프로퍼티가 sentence-transformers 를 로드한다(실측 2.7s). 캐시가
        완전하면 그 모델은 끝내 쓰이지 않으므로 통째로 낭비다.
        """
        return any(self._texts.get(node_id) != compose_node_text(node_id, attrs)
                   for node_id, attrs in self._live_nodes(graph, node_types))

    def build_from_graph(self, graph, node_types=DEFAULT_NODE_TYPES) -> int:
        """상위와 동일하되, 캐시에 있고 텍스트가 그대로인 노드는 건너뛴다.

        상위 build_from_graph 가 이미 `self._texts.get(node_id) == text` 로
        스킵하므로, load_from_disk 로 _texts 를 채워두기만 하면 재임베딩이
        자동으로 사라진다. 여기서는 (1) 바뀐 게 아예 없으면 임베더 로딩까지
        건너뛰고, (2) 그래프에서 없어진 노드를 캐시에서 걷어낸다.
        """
        added = (super().build_from_graph(graph, node_types=node_types)
                 if self._has_pending(graph, node_types) else 0)

        live = {node_id for node_id, _ in self._live_nodes(graph, node_types)}
        stale = [node_id for node_id in list(self._ids) if node_id not in live]
        for node_id in stale:
            self.remove(node_id)
        if stale:
            logger.info(f"🔎 Vector cache: dropped {len(stale)} stale node(s)")
        return added
