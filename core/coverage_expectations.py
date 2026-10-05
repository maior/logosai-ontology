"""커버리지 expectation 게이트 — 네임스페이스별 데이터 품질 기대치 (팔란티어 expectation 차용).

/health 가 들추는 데이터 품질 지표(추출 커버리지·고아 노드·미연결 청크·dangling
참조)는 검색 품질의 **상한**을 정한다 — 천장 실측 2회가 그 근거다. 그런데
"얼마면 나쁜가"의 기준이 없어 사람이 매번 눈으로 판독해야 했다. 이 모듈은 그
기준(expectation)을 네임스페이스별 설정으로 두고, 지표를 기준과 대조해
ok / warn 을 낸다 (설계 문서 docs/review-collaboration-architecture.html §4).

설계 규정 (retrieval_config 과 같은 골격):
- **전역 기본값은 전부 None (unconfigured).** 측정 없이 감으로 정한 임계를
  강제하지 않는다 — 기준은 운영자가 자기 네임스페이스의 실측을 보고 정한다.
- **부분 오버라이드.** 미설정 키는 기본값(None)을 쓴다.
- **값을 검증한다.** 오타 키·범위 밖 값은 소리내어 거부한다 (`validate_overrides`).
- **`None` 은 그 키를 지운다** — unconfigured 로 되돌리는 길.
- **0/0 정직.** 분모가 0이면 그 비율은 None 이고, "측정 안 됨"(measured:false)과
  "미달"(warn)을 구별해 보고한다. 0건을 0.0 으로 보고하는 오보고를 반복하지
  않는다 (graph_health 에서 고친 그 결함류).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# (키, 타입, 최소, 최대). None 은 상한 없음.
_SPEC = (
    ("min_extraction_coverage", float, 0.0, 1.0),
    ("max_orphan_ratio", float, 0.0, 1.0),
    ("max_unlinked_ratio", float, 0.0, 1.0),
    ("max_dangling_refs", int, 0, None),
    ("auto_coverage_check", bool, None, None),
    ("auto_coverage_limit", int, 1, 200),
)
KEYS = tuple(name for name, *_ in _SPEC)

# evaluate 가 대조하는 임계 키 (auto_* 는 동작 설정이지 임계가 아니다).
THRESHOLD_KEYS = ("min_extraction_coverage", "max_orphan_ratio",
                  "max_unlinked_ratio", "max_dangling_refs")


def _global_defaults() -> Dict[str, Any]:
    """전역 기본값 — **전부 None (unconfigured)**.

    retrieval_config 과 달리 물려받을 전역 상수가 없다: "얼마면 나쁜가"는
    측정 없이 정할 수 없고, 지어낸 임계는 강제하는 순간 거짓 경보가 된다.
    """
    return {name: None for name in KEYS}


def effective_config(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """기본값(전부 None) + 오버라이드 → evaluate 에 넘길 thresholds.

    알 수 없는 키는 **버린다** — 오타가 통과하면 없는 임계로 평가하는 척하게
    된다. (쓰기 API 는 버리지 않고 알려준다: `validate_overrides`.)
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
            return False, f"'{key}' 는 {hi} 이하이어야 한다 (받음: {value})"
    return True, ""


def _num(value: Any) -> Optional[float]:
    """숫자면 float, 아니면 None. bool 은 숫자가 아니다 (int 하위형 함정)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    """분자/분모 → 비율. **분모 0 이면 None** — 0/0 을 값으로 지어내지 않는다."""
    num = _num(numerator)
    den = _num(denominator)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def evaluate_expectations(metrics: Dict[str, Any],
                          thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """지표를 기대치와 대조 → {"status", "warnings", "metrics"}.

    - status: "ok"(전 임계 통과) | "warn"(측정된 위반 존재) |
      "unconfigured"(임계가 하나도 설정되지 않음 — 또는 입력이 쓰레기).
    - **경계값은 통과다**: min 임계는 `actual >= expected`, max 임계는
      `actual <= expected` 가 통과 — 임계와 정확히 같으면 위반이 아니다.
    - **0/0 정직**: 지표가 None(분모 0·미제공)이면 경고를 내지 않고
      {"key", "measured": false, "reason": "no_data"} 로 보고하되 status 를
      warn 으로 만들지 않는다. "측정 안 됨"과 "미달"의 구별이 이 함수의
      존재 이유다.
    - 절대 던지지 않는다 — 쓰레기 입력은 unconfigured + 빈 결과.

    metrics 입력 키: extraction_coverage(float|None), orphan_node_count(int),
    node_count(int), unlinked_chunk_count(int), chunk_count(int),
    dangling_node_ref_count(int). 비율은 이 함수 안에서 계산한다.
    """
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict):
        return {"status": "unconfigured", "warnings": [], "metrics": {}}

    computed: Dict[str, Any] = {
        "extraction_coverage": _num(metrics.get("extraction_coverage")),
        # orphan_ratio = 고아 노드 / 전체 노드 (노드 0 → None)
        "orphan_ratio": _ratio(metrics.get("orphan_node_count"),
                               metrics.get("node_count")),
        # unlinked_ratio = 미연결 청크 / 전체 청크 (청크 0 → None)
        "unlinked_ratio": _ratio(metrics.get("unlinked_chunk_count"),
                                 metrics.get("chunk_count")),
        "dangling_refs": _num(metrics.get("dangling_node_ref_count")),
    }

    # (임계 키, 지표 키, 방향, 위반 메시지). direction "min" = actual >= expected
    # 통과, "max" = actual <= expected 통과.
    checks = (
        ("min_extraction_coverage", "extraction_coverage", "min",
         "추출 커버리지 {actual} 가 최소 기준 {expected} 에 미달한다"),
        ("max_orphan_ratio", "orphan_ratio", "max",
         "고아 노드 비율 {actual} 이 최대 기준 {expected} 를 초과한다"),
        ("max_unlinked_ratio", "unlinked_ratio", "max",
         "미연결 청크 비율 {actual} 이 최대 기준 {expected} 를 초과한다"),
        ("max_dangling_refs", "dangling_refs", "max",
         "dangling 노드 참조 {actual} 건이 최대 기준 {expected} 건을 초과한다"),
    )

    warnings = []
    configured = False
    violated = False
    for key, metric_key, direction, template in checks:
        expected = _num(thresholds.get(key))
        if expected is None:
            continue                      # 미설정 임계 — 평가 대상 아님
        configured = True
        actual = computed.get(metric_key)
        if actual is None:
            # 측정 불가 ≠ 미달. 보고는 하되 warn 으로 만들지 않는다.
            warnings.append({"key": key, "measured": False,
                             "reason": "no_data"})
            continue
        passed = actual >= expected if direction == "min" else actual <= expected
        if not passed:
            violated = True
            warnings.append({
                "key": key,
                "expected": expected,
                "actual": actual,
                "measured": True,
                "message": template.format(actual=round(actual, 4),
                                           expected=expected),
            })

    if not configured:
        return {"status": "unconfigured", "warnings": [], "metrics": computed}
    return {"status": "warn" if violated else "ok",
            "warnings": warnings,
            "metrics": computed}


class CoverageExpectations:
    """네임스페이스 하나의 커버리지 기대치 (현재 상태 JSON).

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
        return _DEFAULT_DATA_DIR / f"coverageexpectations_{self.namespace}.json"

    def overrides(self) -> Dict[str, Any]:
        return dict(self._overrides)

    def effective(self) -> Dict[str, Any]:
        return effective_config(self._overrides)

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_expectations(metrics, self.effective())

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
            logger.warning(f"⚠️ 커버리지 기대치 저장 실패 ({self.path}): {e}")
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
            # 깨진 설정으로 평가를 죽이지 않는다 — 기본값(unconfigured)으로 degrade.
            logger.warning(f"⚠️ 커버리지 기대치 로드 실패 ({load_path}): {e}")
            self._overrides = {}
            return False


_expectations: Dict[str, CoverageExpectations] = {}


def get_coverage_expectations(namespace: str = "default") -> CoverageExpectations:
    if namespace not in _expectations:
        exp = CoverageExpectations(namespace=namespace)
        exp.load_from_disk()
        _expectations[namespace] = exp
    return _expectations[namespace]


def reset_coverage_expectations() -> None:
    _expectations.clear()
