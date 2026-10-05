"""평가 스냅샷 이력 — "이 설정에서 이 점수"를 기록해 비교 가능하게 한다.

**왜 필요한가 (경험)**: 검색 knob 과 임베더를 바꿔가며 지표를 쟀는데 비교표를
전부 손으로 만들었고, 그 사이 결론이 **두 번 뒤집혔다**:
  · `max_terms` 8→2 로 바꿨다가 되돌렸다 — 고아 노드를 고치자 순서가 뒤집혔다.
  · 임베더도 노드 채널만 재고 결론냈다가 청크·융합에서 뒤집혔다.
점수만 남기고 설정을 안 남기면 **"그때 그 숫자가 어떤 설정에서 나온 것인가"**를
잃는다. 그러면 비교가 불가능하고, 같은 실수를 반복한다.

설계 규정:
- **추가전용 JSONL** — 골든셋·검수 로그와 같은 규약. 로그가 원본이다.
- **설정 지문을 같이 남긴다** — 임베더·entry_k·max_terms·entry_ratio·확산
  플래그·케이스 수가 전부 지표를 움직인다(각각 실측됐다).
- **평가할 때 자동으로 남긴다.** 사람이 기억해서 기록하는 절차는 지켜지지 않는다.
- **per_case 는 저장하지 않는다** — 이력의 목적이 아니고 파일만 불린다.
- 기록 실패가 평가를 죽이지 않는다 — 측정이 본체고 이력은 부수물이다.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def _retrieval_defaults(namespace: str = "") -> Dict[str, Any]:
    """검색 설정 — **네임스페이스 오버라이드를 반영한다**.

    전역 기본값만 기록하면 이력이 거짓이 된다: 이 모듈의 목적이 "그때 그 숫자가
    어떤 설정에서 나왔나"인데, 네임스페이스별 knob(확산 on/off 등)을 무시하면
    그 목적을 정면으로 배신한다. 지연 import (커널 경계: 가볍게 유지)."""
    from .retrieval_config import effective_config
    if namespace:
        from .retrieval_config import get_retrieval_config
        return dict(get_retrieval_config(namespace).effective())
    return dict(effective_config({}))


def config_fingerprint(namespace: str = "") -> Dict[str, Any]:
    """지금 지표를 만들고 있는 설정. 실패해도 빈 dict 가 아니라 아는 만큼 돌려준다 —
    이력이 없는 것보다 부분 이력이 낫다.

    namespace 를 주면 그 네임스페이스의 오버라이드를 반영한다."""
    out: Dict[str, Any] = {}
    try:
        from .semantic_index import resolve_chunk_model, resolve_node_model
        out["node_model"] = resolve_node_model()
        out["chunk_model"] = resolve_chunk_model()
    except Exception as e:
        logger.warning(f"⚠️ 임베더 지문 조회 실패: {e}")
        out.setdefault("node_model", "")
        out.setdefault("chunk_model", "")
    try:
        out.update(_retrieval_defaults(namespace))
    except Exception as e:
        logger.warning(f"⚠️ 검색 지문 조회 실패: {e}")
        for key in ("entry_k", "entry_ratio", "max_terms", "use_propagation",
                    "propagation_channel"):
            out.setdefault(key, None)
    out["vector_backend"] = os.environ.get("ONTOLOGY_VECTOR_BACKEND", "")
    return out


class EvalHistory:
    """네임스페이스 하나의 평가 이력 (추가전용 JSONL)."""

    def __init__(self, namespace: str = "default", path=None):
        self.namespace = namespace
        self._path = Path(path) if path else None
        self._entries: List[Dict[str, Any]] = []

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        return _DEFAULT_DATA_DIR / f"evalhistory_{self.namespace}.jsonl"

    def record(self, result: Dict[str, Any], actor: str = "",
               note: str = "") -> Dict[str, Any]:
        """평가 결과 한 건을 남긴다. per_case 는 뺀다(이력의 목적이 아니다)."""
        entry = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "actor": actor or "system",
            "note": note,
            "target": result.get("target", ""),
            "k": result.get("k"),
            "cases": result.get("cases"),
            "skipped": result.get("skipped"),
            "measured": result.get("measured"),
            "statuses": list(result.get("statuses") or []),
            "channels": result.get("channels") or {},
            # 이 네임스페이스의 오버라이드를 반영한다 — 전역 기본값만 남기면
            # "그때 그 설정" 이 거짓이 된다.
            "config": config_fingerprint(self.namespace),
        }
        self._entries.append(entry)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # 측정이 본체다 — 기록 실패로 평가를 죽이지 않는다.
            logger.warning(f"⚠️ 평가 이력 기록 실패 ({self.path}): {e}")
        return entry

    def entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """최신 먼저. 추가전용이라 **로그 순서가 곧 시간 순서**다 — 같은 초 안의
        기록은 타임스탬프로 갈리지 않으므로 정렬이 아니라 역순을 쓴다."""
        rows = list(reversed(self._entries))
        return rows[:limit] if limit else rows

    def load_from_disk(self, path=None) -> bool:
        load_path = Path(path) if path else self.path
        if not load_path.exists():
            return False
        try:
            self._entries.clear()
            skipped = 0
            with open(load_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._entries.append(json.loads(line))
                    except Exception:
                        skipped += 1
            if skipped:
                logger.warning(f"⚠️ 평가 이력: 깨진 줄 {skipped}개 건너뜀")
            return True
        except Exception as e:
            logger.error(f"평가 이력 로드 실패: {e}")
            return False


# 지문 중 지표를 움직이는 키 — stale 판정 기준. vector_backend 는 품질이 아니라
# 저장 방식이므로 뺀다 (memory ↔ npy 는 같은 벡터의 캐시 차이일 뿐).
_QUALITY_KEYS = ("node_model", "chunk_model", "entry_k", "entry_ratio",
                 "min_entry_score", "max_terms", "use_propagation",
                 "propagation_channel", "propagation_top",
                 "propagation_weight")


def latest_quality(namespace: str, target: str = "evidence") -> Dict[str, Any]:
    """검색 응답에 동봉할 품질 블록 — **지어내지 않는다**.

    RRF 점수(0.016…)나 코사인은 보정되지 않은 값이라 사용자에게 "신뢰도"로
    보여주면 없는 정밀도를 지어내는 것이다. 정직한 답은 셋뿐이다:
      · **측정된 골든셋 지표** (retrieve 채널의 hit@1/hit@k/MRR)
      · **측정 시점과 케이스 수** — 얼마나 믿을 자인지
      · **stale 여부** — 측정 후 설정이 바뀌었으면 그 숫자는 현재를 대변하지
        않는다. 어떤 키가 바뀌었는지(stale_keys)까지 보여준다.

    측정한 적 없으면 measured:false + 이유 — 0 점이나 빈 dict 가 아니라
    (evaluate 의 0건 정직 보고와 같은 계약). cases=0 기록은 품질 근거가 아니므로
    건너뛴다. 실패는 measured:false (never raise — 품질 블록이 검색을 죽이면
    안 된다).
    """
    try:
        for entry in get_eval_history(namespace).entries(limit=0):
            if entry.get("target") != target or not entry.get("measured"):
                continue
            retrieve = (entry.get("channels") or {}).get("retrieve")
            if not retrieve:
                continue
            recorded = entry.get("config") or {}
            current = config_fingerprint(namespace)
            stale_keys = sorted(k for k in _QUALITY_KEYS
                                if recorded.get(k) != current.get(k))
            return {"measured": True, "target": target,
                    "measured_at": entry.get("at", ""),
                    "cases": entry.get("cases"), "k": entry.get("k"),
                    "retrieve": {key: retrieve.get(key)
                                 for key in ("hit@1", f"hit@{entry.get('k')}",
                                             "mrr") if key in retrieve},
                    "stale": bool(stale_keys), "stale_keys": stale_keys}
        return {"measured": False,
                "reason": f"이 네임스페이스에 target={target} 평가 기록이 없다 — "
                          "POST /qa/evaluate 로 측정하라"}
    except Exception as e:
        logger.warning(f"⚠️ 품질 블록 조회 실패 ({namespace}): {e}")
        return {"measured": False, "reason": "품질 이력 조회 실패"}


_histories: Dict[str, EvalHistory] = {}


def get_eval_history(namespace: str = "default") -> EvalHistory:
    if namespace not in _histories:
        history = EvalHistory(namespace=namespace)
        history.load_from_disk()
        _histories[namespace] = history
    return _histories[namespace]


def reset_eval_histories() -> None:
    _histories.clear()
