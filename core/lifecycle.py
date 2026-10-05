"""노드 생애주기 — **소멸을 가능하게 하는 게 아니라 통제하는 장치다.**

팔란티어 대조(`docs/palantir-ontology-comparison.html`)가 결손으로 지목한 축이다.
그쪽은 모든 타입에 상태를 붙이고 **`Active` 리소스는 삭제도 개명도 불가**,
`Deprecated` 는 사유 + 삭제 기한(+ 선택적 대체)을 요구한다 — "언젠가 정리하자"가
구조적으로 불가능하다.

**우리에게 필요한 구체적 이유**: 파괴 경로가 이미 있는데 통제가 없다.
  · `reject_node` → 묘비 + 노드·엣지 제거
  · `merge_nodes` → 진 노드 삭제. 대조 문서의 지적: "그 노드를 누가 쓰고 있는지
    아무도 모른다"
  · 그리고 `Clause` 재분류는 `{type}:{name}` id 를 바꾸므로 **그 자체가 개명**이다.
    그래서 관문이 먼저다.

**검수 상태와 다른 축이다** (섞으면 표현력을 잃는다):
  · 검수(`confirmed`/`rejected`, review_store) = "이 노드가 **맞는가**" — 정확성
  · 생애주기(여기) = "이 노드가 **운영에서 쓰이는가**" — 역할
확정됐지만 아직 실험인 노드도, 검수 전인데 운영이 물린 노드도 있을 수 있다.

**기본값은 미지정(=experimental)이고 소급 적용하지 않는다.** 191 노드에 자동으로
`active` 를 붙이면 관문이 무의미해진다(전부 차단). `active` 는 사람이 선언한다.

**전이 규칙의 핵심**: `active` 에서 벗어나는 유일한 길이 `deprecated` 다.
`active → experimental` 을 허용하면 `active → experimental → 삭제` 우회가 열린다.
"""

import re
from typing import Any, Dict, Optional, Tuple

from loguru import logger

EXPERIMENTAL = "experimental"
ACTIVE = "active"
DEPRECATED = "deprecated"

STATES = (EXPERIMENTAL, ACTIVE, DEPRECATED)

# 미지정 노드의 상태. 명시적 상수로 두는 이유: 호출부가 "" 과 experimental 을
# 다르게 다루기 시작하면 관문이 두 갈래로 갈린다.
DEFAULT_STATE = EXPERIMENTAL

# 허용 전이. active → experimental 이 **없는 것**이 이 표의 요점이다.
_TRANSITIONS = {
    EXPERIMENTAL: {ACTIVE, DEPRECATED},
    ACTIVE: {DEPRECATED},
    DEPRECATED: {ACTIVE, EXPERIMENTAL},   # 번복은 명시적 행위이므로 허용
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def current_state(attrs: Optional[Dict[str, Any]]) -> str:
    """노드의 현재 상태. 미지정·오타는 기본값으로 읽는다 — 알 수 없는 값을
    그대로 흘리면 관문이 "차단도 허용도 아닌" 상태에 빠진다."""
    value = str((attrs or {}).get("lifecycle") or "").strip().lower()
    return value if value in STATES else DEFAULT_STATE


def can_destroy(attrs: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """이 노드를 지우거나 개명해도 되는가 → (가능, 이유).

    `active` 만 막는다. `deprecated` 는 **이미 선언된 경로**이므로 허용한다 —
    사유와 기한을 밝히는 것이 deprecate 의 대가이고, 그 뒤 삭제를 다시 막으면
    영구히 못 지우는 노드가 된다.
    """
    state = current_state(attrs)
    if state == ACTIVE:
        return False, ("운영 사용 중(active)인 노드는 삭제·개명할 수 없다. "
                       "먼저 deprecated 로 내리고 사유와 기한을 밝혀라.")
    return True, ""


def validate_transition(from_state: str, to_state: str, *,
                        reason: str = "", sunset: str = "") -> Tuple[bool, str]:
    """상태 전이가 허용되는가 → (허용, 이유).

    `deprecated` 로 갈 때는 **사유와 기한이 필수**다. 그게 없으면 deprecate 가
    그냥 "삭제 허용 스위치"가 되고, 이 모듈의 목적(통제)이 사라진다.
    """
    from_state = from_state if from_state in STATES else DEFAULT_STATE
    if to_state not in STATES:
        return False, f"알 수 없는 상태 '{to_state}' (가능: {', '.join(STATES)})"
    if to_state == from_state:
        return False, f"이미 {from_state} 다"
    if to_state not in _TRANSITIONS[from_state]:
        extra = (" — active 에서 벗어나는 길은 deprecated 뿐이다"
                 if from_state == ACTIVE else "")
        return False, f"{from_state} → {to_state} 전이는 허용되지 않는다{extra}"
    if to_state == DEPRECATED:
        if not (reason or "").strip():
            return False, "deprecated 는 사유(reason)가 필수다"
        if not _DATE_RE.match((sunset or "").strip()):
            return False, ("deprecated 는 삭제 기한(sunset, YYYY-MM-DD)이 "
                           "필수다 — 기한 없는 deprecate 는 '언젠가 정리하자'다")
    return True, ""


def apply_transition(attrs: Dict[str, Any], to_state: str, *,
                     reason: str = "", sunset: str = "",
                     superseded_by: str = "", at: str = "") -> Dict[str, Any]:
    """전이를 attrs 에 적용한다 (검증은 호출부가 validate_transition 으로).

    deprecated 를 벗어날 때 사유·기한·대체를 **지운다** — 남겨두면 active 노드가
    옛 삭제 기한을 들고 다니고, 화면이 그걸 보여주면 거짓이 된다.
    """
    attrs["lifecycle"] = to_state
    attrs["lifecycle_at"] = at
    if to_state == DEPRECATED:
        attrs["deprecated_reason"] = reason
        attrs["sunset"] = sunset
        if superseded_by:
            attrs["superseded_by"] = superseded_by
        else:
            attrs.pop("superseded_by", None)
    else:
        for key in ("deprecated_reason", "sunset", "superseded_by"):
            attrs.pop(key, None)
    return attrs


def state_counts(graph) -> Dict[str, int]:
    """상태 분포 — `/health` 가 보고한다. 미지정도 experimental 로 센다
    (별도 칸으로 나누면 "미지정 = 안전"으로 오해된다)."""
    out = {state: 0 for state in STATES}
    try:
        for _, attrs in graph.nodes(data=True):
            out[current_state(attrs)] += 1
    except Exception as e:
        logger.warning(f"⚠️ 생애주기 분포 조회 실패: {e}")
    return out


def overdue(graph, today: str) -> list:
    """삭제 기한이 지난 deprecated 노드 — 기한을 적어두고 아무도 안 보면
    "언젠가 정리하자"가 그대로 돌아온다. `/health` 가 들춘다.

    today 는 호출부가 넘긴다(이 모듈이 시계를 읽지 않는다 — 테스트 가능성)."""
    out = []
    try:
        for node_id, attrs in graph.nodes(data=True):
            if current_state(attrs) != DEPRECATED:
                continue
            sunset = str(attrs.get("sunset") or "").strip()
            if sunset and sunset < today:
                out.append({"node_id": node_id, "sunset": sunset,
                            "reason": attrs.get("deprecated_reason", ""),
                            "superseded_by": attrs.get("superseded_by", "")})
    except Exception as e:
        logger.warning(f"⚠️ 기한 초과 조회 실패: {e}")
    return sorted(out, key=lambda r: (r["sunset"], r["node_id"]))
