"""구성 요소 상태 신호 — 기능이 조용히 꺼지면 소리를 낸다.

이 커널은 degrade 하도록 설계돼 있다: 임베더·확산·ES 가 없어도 죽지 않고 품질이 내려간다.
그런데 degrade 는 "오류 없음"을 만들 뿐 "정상"을 만들지 않는다 — 2026-10-05, macOS 27
업그레이드 후 scipy 네이티브 라이브러리가 거부되자 임베더를 못 불러왔고, 의미 검색이 전
네임스페이스에서 결과 0건이었다. 남은 것은 WARNING 한 줄이었다.

그래서 꺼질 수 있는 구성 요소는 성공·실패를 여기에 보고한다. 상태가 **바뀔 때만**
리스너를 부른다 — 정상 트래픽마다 신호를 내면 안 된다. 첫 보고가 정상이면 조용하다
(정상 기동은 소식이 아니다). 전송(Pulse 등)은 서버 계층이 리스너로 붙인다 — 커널은
네트워크를 모른다 (커널 경계, 표준 라이브러리만).
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: (component, ok, detail, previous_ok) — previous_ok 는 첫 보고면 None
Listener = Callable[[str, bool, str, Optional[bool]], None]

_lock = threading.Lock()
_states: Dict[str, Dict[str, Any]] = {}
_listeners: List[Listener] = []


def report(component: str, ok: bool, detail: str = "") -> None:
    """구성 요소의 현재 상태를 보고한다. 바뀌었을 때만 리스너를 부른다."""
    now = time.time()
    with _lock:
        prev = _states.get(component)
        prev_ok = prev["ok"] if prev else None
        changed = prev_ok is None and not ok or prev_ok is not None and prev_ok != ok
        if prev is None or prev_ok != ok:
            _states[component] = {"ok": ok, "detail": str(detail)[:500], "since": now, "checked": now}
        else:
            prev["checked"] = now
        listeners = list(_listeners) if changed else []
    for fn in listeners:          # 잠금 밖에서 — 리스너가 느려도 보고자는 막히지 않는다
        try:
            fn(component, ok, str(detail)[:500], prev_ok)
        except Exception as e:     # 신호 전송이 검색을 깨뜨리면 안 된다
            logger.warning(f"health signal listener failed ({component}): {e}")


def add_listener(fn: Listener) -> None:
    with _lock:
        if fn not in _listeners:
            _listeners.append(fn)


def snapshot() -> Dict[str, Dict[str, Any]]:
    """구성 요소별 상태 사본 (헬스 엔드포인트가 쓴다)."""
    with _lock:
        return {k: dict(v) for k, v in _states.items()}


def degraded() -> List[str]:
    """지금 실패 상태인 구성 요소."""
    with _lock:
        return sorted(k for k, v in _states.items() if not v["ok"])


def reset_for_tests() -> None:
    with _lock:
        _states.clear()
        _listeners.clear()
