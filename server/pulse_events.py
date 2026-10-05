"""구성 요소 상태 변화를 LogosPulse 런타임 이벤트로 보낸다 (Events 탭).

core.health_signals 의 리스너다 — 커널은 네트워크를 모르고, 전송은 서버가 붙인다.
표준 라이브러리만 쓴다 (logos_api 와 같은 이유: POST 한 번을 위해 logosai 를 끌어오지 않는다).

수집 규약(루트 CLAUDE.md "Pulse 수집 규약")을 따른다:
  · fire-and-forget 이되 **상태 코드를 확인**하고 실패를 센다 (422 를 3주간 성공으로 셌던 전례)
  · 클라이언트 발급 event_id — 재전송이 멱등하다
  · 테스트 중엔 기본 차단 — 테스트가 프로덕션 관측 DB 를 오염시킨 전례.
    해제는 LOGOSAI_PULSE_ALLOW_IN_TESTS=1, 전체 끄기는 LOGOS_PULSE_DISABLED=1
"""

import json
import logging
import os
import sys
import threading
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_stats = {"sent": 0, "failed": 0}
_stats_lock = threading.Lock()


def _event_url() -> str:
    base = os.environ.get("LOGOS_PULSE_URL", "http://localhost:8095").rstrip("/")
    return f"{base}/api/v1/ingest/event"


def sending_blocked() -> bool:
    if os.environ.get("LOGOS_PULSE_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if "pytest" in sys.modules:
        return os.environ.get("LOGOSAI_PULSE_ALLOW_IN_TESTS", "").strip() != "1"
    return False


def build_event(component: str, ok: bool, detail: str, previous_ok: Optional[bool]) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "ontology.component.recovered" if ok else "ontology.component.unavailable",
        "source": "ontology",
        "agent_id": "ontology",
        "severity": "info" if ok else "critical",
        "payload": {"component": component, "ok": ok, "detail": detail,
                    "previous_ok": previous_ok, "service": "ontology:9274"},
    }


def _http_post(url: str, body: bytes) -> int:
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status


def send(event: Dict[str, Any], post: Callable[[str, bytes], int] = _http_post) -> None:
    """동기 전송 1회 — 실패는 세고 남기되 올리지 않는다."""
    if sending_blocked():
        return
    try:
        code = post(_event_url(), json.dumps(event, ensure_ascii=False).encode("utf-8"))
    except Exception as e:
        code = f"{type(e).__name__}: {e}"
    with _stats_lock:
        if code == 200:
            _stats["sent"] += 1
        else:
            _stats["failed"] += 1
    if code != 200:
        logger.warning(f"Pulse 이벤트 전송 실패 ({event['event_type']} {event['payload']['component']}): {code}")


def on_component_change(component: str, ok: bool, detail: str, previous_ok: Optional[bool]) -> None:
    """health_signals 리스너 — 보고자(검색 경로)를 막지 않도록 데몬 스레드로 보낸다."""
    event = build_event(component, ok, detail, previous_ok)
    logger.warning(f"구성 요소 상태 변화: {component} → {'정상' if ok else '불가'} ({detail})")
    threading.Thread(target=send, args=(event,), daemon=True, name="pulse-event").start()


def stats() -> Dict[str, int]:
    with _stats_lock:
        return dict(_stats)


def reset_stats() -> None:
    with _stats_lock:
        _stats.update(sent=0, failed=0)
