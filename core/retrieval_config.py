"""네임스페이스별 검색 설정 — knob 이 전역 상수여서 못 하던 것.

**실측이 요구했다.** 확산 채널의 가치가 두 네임스페이스에서 정반대였다:

    네임스페이스        커버리지   evidence hit@1: 확산 off → on
    ────────────────────────────────────────────────────────────
    ins_cancer_demo     100%       동률
    PROJ-A                25%       0.3871→0.4516 · 0.6154→0.6615

두 자 모두에서 재현됐다 — **확산의 가치는 그래프가 성길 때 나타난다**(1-hop 이
못 닿는 곳을 청크를 허브로 건너는 설계 의도와 일치). 그런데
`DEFAULT_PROPAGATION_CHANNEL` 이 전역 상수라 한쪽에 맞추면 다른 쪽이 틀리고,
`service.retrieve(ns, query, top_k)` 가 플래그를 받지 않아 **라이브에서는 아예 켤
수 없었다** — 측정만 되고 실사용에 못 쓰는 상태였다.

같은 이유가 `max_terms`·`entry_k` 에도 있다. 이 세션에서 셋 다 "그래프 상태에
종속적"임이 측정됐다(상수 주석의 스윕 표들).

설계 규정:
- **부분 오버라이드.** 미설정 키는 전역 기본값을 쓴다 — 전부 명시하도록 요구하면
  기본값이 개선될 때 네임스페이스마다 낡은 값이 굳는다.
- **값을 검증한다.** entry_k=0 이면 진입 노드가 0개가 되는데, 조용히 통과하면
  "검색이 안 된다"의 원인을 찾기 어렵다.
- **`None` 은 그 키를 지운다** — 전역 기본값으로 되돌리는 길. 없으면 한 번 설정한
  값이 영구히 굳는다.
- **측정과 라이브가 같은 설정을 쓴다** (service 쪽 계약). 그러지 않으면 지표가
  라이브를 설명하지 못한다 — 이 세션에서 가장 많이 데인 부류.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# (키, 타입, 최소, 최대). None 은 상한 없음.
_SPEC = (
    ("entry_k", int, 1, 50),
    ("entry_ratio", float, 0.0, 1.0),
    ("min_entry_score", float, 0.0, 1.0),
    ("max_terms", int, 0, 64),
    ("use_propagation", bool, None, None),
    ("propagation_channel", bool, None, None),
    ("propagation_top", int, 1, 100),
    ("propagation_weight", float, 0.0, 10.0),
)
KEYS = tuple(name for name, *_ in _SPEC)


def _global_defaults() -> Dict[str, Any]:
    """전역 기본값 — 지연 import (커널 안이지만 순환을 피한다)."""
    from .graph_retrieval import (DEFAULT_ENTRY_K, DEFAULT_ENTRY_RATIO,
                                  DEFAULT_MAX_TERMS, DEFAULT_MIN_ENTRY_SCORE,
                                  DEFAULT_PROPAGATION_CHANNEL,
                                  DEFAULT_PROPAGATION_TOP,
                                  DEFAULT_PROPAGATION_WEIGHT,
                                  DEFAULT_USE_PROPAGATION)
    return {"entry_k": DEFAULT_ENTRY_K,
            "entry_ratio": DEFAULT_ENTRY_RATIO,
            "min_entry_score": DEFAULT_MIN_ENTRY_SCORE,
            "max_terms": DEFAULT_MAX_TERMS,
            "use_propagation": DEFAULT_USE_PROPAGATION,
            "propagation_channel": DEFAULT_PROPAGATION_CHANNEL,
            "propagation_top": DEFAULT_PROPAGATION_TOP,
            "propagation_weight": DEFAULT_PROPAGATION_WEIGHT}


def effective_config(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """전역 기본값 + 오버라이드 → `search()`/`expand()` 에 그대로 넘길 kwargs.

    알 수 없는 키는 **버린다** — 오타가 통과하면 `search()` 가 TypeError 로 죽는다.
    (쓰기 API 는 버리지 않고 알려준다: `validate_overrides`.)
    """
    cfg = _global_defaults()
    for key, value in (overrides or {}).items():
        if key in cfg and value is not None:
            cfg[key] = value
    return cfg


def validate_overrides(overrides: Dict[str, Any]) -> Tuple[bool, str]:
    """쓰기 전 검증 → (유효, 이유). 오타 키를 **소리내어** 거부한다."""
    spec = {name: (typ, lo, hi) for name, typ, lo, hi in _SPEC}
    for key, value in (overrides or {}).items():
        if key not in spec:
            return False, (f"알 수 없는 설정 키 '{key}' "
                           f"(가능: {', '.join(KEYS)})")
        if value is None:
            continue                      # 지우기 — 타입 검사 대상 아님
        typ, lo, hi = spec[key]
        if typ is bool:
            if not isinstance(value, bool):
                return False, f"'{key}' 는 true/false 여야 한다"
            continue
        # bool 은 int 의 하위형이므로 명시적으로 배제한다 (True 가 1 로 통과).
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"'{key}' 는 숫자여야 한다"
        if typ is int and not float(value).is_integer():
            return False, f"'{key}' 는 정수여야 한다"
        if lo is not None and value < lo:
            return False, f"'{key}' 는 {lo} 이상이어야 한다 (받음: {value})"
        if hi is not None and value > hi:
            return False, f"'{key}' 는 {hi} 이하여야 한다 (받음: {value})"
    return True, ""


class RetrievalConfig:
    """네임스페이스 하나의 검색 설정 (현재 상태 JSON).

    추가전용 로그가 아닌 이유: 설정은 **현재 값**이 본체다. "누가 언제 왜 바꿨나"는
    review_store 감사에 남긴다(service 쪽) — 두 곳에 이력을 두면 갈라진다.
    """

    def __init__(self, namespace: str = "default", path=None):
        self.namespace = namespace
        self._path = Path(path) if path else None
        self._overrides: Dict[str, Any] = {}

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        return _DEFAULT_DATA_DIR / f"retrievalconfig_{self.namespace}.json"

    def overrides(self) -> Dict[str, Any]:
        return dict(self._overrides)

    def effective(self) -> Dict[str, Any]:
        return effective_config(self._overrides)

    def set(self, overrides: Dict[str, Any], actor: str = "") -> Dict[str, Any]:
        """**병합**한다(교체 아님). `None` 값은 그 키를 지운다.

        한 키만 바꾸려고 부르는 게 정상 사용이므로 교체면 나머지가 날아간다.
        """
        for key, value in (overrides or {}).items():
            if key not in KEYS:
                continue
            if value is None:
                self._overrides.pop(key, None)
            else:
                self._overrides[key] = value
        try:
            self.path.write_text(
                json.dumps({"namespace": self.namespace,
                            "overrides": self._overrides,
                            "actor": actor}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            # 메모리 상태는 유지한다 — 디스크 오류로 설정 변경이 통째로
            # 실패하면 운영자가 원인을 못 본다(다음 조회에서 옛 값이 보인다).
            logger.warning(f"⚠️ 검색 설정 저장 실패 ({self.path}): {e}")
        return self.overrides()

    def load_from_disk(self, path=None) -> bool:
        load_path = Path(path) if path else self.path
        if not load_path.exists():
            return False
        try:
            data = json.loads(load_path.read_text(encoding="utf-8"))
            raw = data.get("overrides") if isinstance(data, dict) else {}
            self._overrides = {k: v for k, v in (raw or {}).items()
                               if k in KEYS and v is not None}
            return True
        except Exception as e:
            # 깨진 설정으로 검색을 죽이지 않는다 — 전역 기본값으로 degrade.
            logger.warning(f"⚠️ 검색 설정 로드 실패 ({load_path}): {e}")
            self._overrides = {}
            return False


_configs: Dict[str, RetrievalConfig] = {}


def get_retrieval_config(namespace: str = "default") -> RetrievalConfig:
    if namespace not in _configs:
        config = RetrievalConfig(namespace=namespace)
        config.load_from_disk()
        _configs[namespace] = config
    return _configs[namespace]


def reset_retrieval_configs() -> None:
    _configs.clear()
