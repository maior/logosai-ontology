"""온톨로지 선택 근거를 밖으로 내보낸다 (2026-07-31).

문제: `HybridAgentSelector._selection_history` 는 `deque(maxlen=200)` 이다.
"왜 그 에이전트를 골랐는가"는 최근 200건만 남는다. 학습 데이터 추출과 사후
조사가 모두 그 창 안에서만 가능하다 — 트래픽이 늘면 하루도 안 남는다.

여기서는 선택 1건을 Pulse span 하나로 내보낸다. 기존 deque·파일 저장은
그대로 둔다 (오프라인 분석 경로를 끊지 않는다).

설계 메모 두 가지:

1. **피드백은 선택보다 나중에 온다.** 그래서 클라이언트가 `selection_id` 를
   먼저 발급하고, 피드백 이벤트가 그 id 를 참조한다. 수신 측에서 시각으로
   짝을 맞추려 하면 동시 요청에서 어긋난다 (쓰기 순서 역전은 이미 겪었다).

2. **`logosai.utils.pulse_client` 를 쓰지 않는다.** 그 패키지 __init__ 이
   프레임워크 전체를 끌고 온다. 필요한 건 POST 한 번뿐이라 aiohttp 로 직접 보낸다
   (logos_api `interaction_engine` 과 같은 규칙).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

PULSE_URL = os.getenv("LOGOS_PULSE_URL", "http://localhost:8095")

_MAX_QUERY = 2000
_MAX_REASONING = 4000


def new_selection_id() -> str:
    """선택 1건의 식별자. 피드백이 나중에 이걸로 되짚는다.

    ⚠️ **맨 UUID 여야 한다.** Pulse `trace_spans.id` 는 UUID 컬럼이라
    `sel-...` 같은 접두사를 붙이면 asyncpg 가 거부한다. 그런데 ingest 는
    그 실패를 **200 OK 로 되돌려주고**(status:"error" 는 본문에만 있다),
    발신은 fire-and-forget 이라 아무도 모른 채 전량 유실된다.
    실제로 이 조합으로 한 번 잃었다 (2026-07-31, 즉시 발견).
    """
    return str(uuid.uuid4())


def _clip(text: Any, limit: int) -> str:
    return str(text or "")[:limit]


# ── 주변 trace 합류 (2026-08-08) ──────────────────────────────────────
#
# 종전엔 `trace_id` 를 무조건 새로 발급해, 선택 근거가 그 쿼리의 여정과
# **끊긴 고아 trace** 로 떨어졌다 (30일 59건 전부). 5.9초짜리 선택이 26초
# 쿼리의 22% 를 차지하는데 여정 화면 어디에도 없었다.
#
# 위 모듈 원칙대로 `logosai` 를 모듈 레벨로 import 하지 않는다. 대신 **한 번만**
# 지연 조회하고 결과(성공이든 실패든)를 기억한다 — 커널 단독 소비자(aicoach 등)
# 에서 매 선택마다 무거운 import 를 재시도하지 않기 위해서다.
_trace_resolver = None      # (get_trace_id, get_span_id) or None
_resolver_probed = False


def _ambient_trace() -> tuple:
    """실행 중인 요청의 (trace_id, span_id). 못 찾으면 (None, None)."""
    global _trace_resolver, _resolver_probed
    if not _resolver_probed:
        _resolver_probed = True
        try:
            from logosai.utils.trace_span import (
                get_current_trace_id, get_current_span_id)
            _trace_resolver = (get_current_trace_id, get_current_span_id)
        except Exception:
            _trace_resolver = None
    if not _trace_resolver:
        return (None, None)
    try:
        return (_trace_resolver[0]() or None, _trace_resolver[1]() or None)
    except Exception:
        return (None, None)   # 관측 조회 실패가 선택을 막지 않는다


def build_selection_span(selection_id: str, query: str, selected_agent: str,
                         method: str, confidence: float, elapsed_ms: float,
                         reasoning: str,
                         graph_insights: Optional[Dict[str, Any]] = None,
                         value_estimate: Optional[float] = None,
                         trace_id: Optional[str] = None,
                         parent_id: Optional[str] = None,
                         ) -> Dict[str, Any]:
    """Pulse SpanRecord 본문. 순수 함수 — 전송과 분리해 검증 가능하게 둔다.

    trace_id/parent_id 를 주면 그 여정에 합류한다. 없으면 스스로 발급 —
    단독 사용(오프라인 분석·직접 호출)을 깨지 않는다. **부모는 지어내지 않는다.**
    """
    gi = graph_insights or {}

    # 그래프 근거가 없으면 빈 채로 둔다. 0.0 과 '모름'을 섞으면 나중에
    # "신뢰도 0 인 선택이 이만큼"이라는 잘못된 통계가 나온다.
    kg_conf = gi.get("kg_confidence", gi.get("confidence"))

    return {
        "span_id": selection_id,          # 클라이언트 발급 → 재전송 멱등 + 되짚기 키
        "trace_id": trace_id or uuid.uuid4().hex,
        "parent_id": parent_id or "",
        "name": "ontology.selection",
        "agent_id": selected_agent or "",
        "status": "success",
        "input_text": _clip(query, _MAX_QUERY),
        "output_text": selected_agent or "",
        "duration_ms": float(elapsed_ms or 0.0),
        "metadata": {
            # 여정 구간: 라우팅 판단. 이름 휴리스틱에 안 걸려 '미분류'였다.
            "stage": "route",
            "selection_id": selection_id,
            "method": method or "",
            "confidence": float(confidence or 0.0),
            "value_estimate": value_estimate,
            "reasoning": _clip(reasoning, _MAX_REASONING),
            "entities": list(gi.get("entities") or [])[:10],
            "related_concepts": list(gi.get("related_concepts") or [])[:10],
            "past_patterns": list(gi.get("past_patterns") or [])[:3],
            "recommended_agents": list(gi.get("recommended_agents") or [])[:3],
            "kg_confidence": float(kg_conf) if kg_conf is not None else None,
            "has_insights": bool(gi.get("has_insights", bool(gi))),
        },
    }


def build_feedback_event(selection_id: str, selected_agent: str, success: bool,
                         ema_success_rate: Optional[float] = None,
                         query_semantics: Optional[Dict[str, Any]] = None,
                         ) -> Dict[str, Any]:
    """피드백 이벤트. 선택 span 을 `selection_id` 로 참조한다."""
    return {
        "event_id": str(uuid.uuid4()),     # 재전송 멱등
        "event_type": "ontology.selection.feedback",
        "source": "ontology.hybrid_selector",
        "severity": "info" if success else "warning",
        "payload": {
            "selection_id": selection_id,
            "selected_agent": selected_agent or "",
            "success": bool(success),
            "ema_success_rate": ema_success_rate,
            "query_semantics": query_semantics or {},
        },
    }


def _warn(msg: str) -> None:
    try:
        from loguru import logger
        logger.warning(f"[selection_recorder] {msg}")
    except Exception:
        pass


def _post(path: str, body: Dict[str, Any]) -> None:
    """fire-and-forget. 관측 실패가 선택을 막지 않는다."""
    async def _send() -> None:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as http:
                async with http.post(f"{PULSE_URL}{path}", json=body,
                                     timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    # 200 OK ≠ 저장됨. ingest 는 내부 예외를 200 + status:"error"
                    # 로 되돌려준다 — 확인하지 않으면 전량 유실이 조용히 지나간다.
                    if resp.status >= 400:
                        _warn(f"pulse {path} HTTP {resp.status}")
                        return
                    try:
                        data = await resp.json()
                    except Exception:
                        return
                    if isinstance(data, dict) and data.get("status") == "error":
                        _warn(f"pulse {path} rejected: "
                              f"{str(data.get('error'))[:200]}")
        except Exception:
            pass   # 관측 실패가 선택을 막지 않는다

    try:
        asyncio.ensure_future(_send())
    except Exception:
        pass   # 실행 중인 루프가 없으면 조용히 포기 (관측일 뿐이다)


def emit_selection(selection_id: str, query: str, selected_agent: str,
                   method: str, confidence: float, elapsed_ms: float,
                   reasoning: str,
                   graph_insights: Optional[Dict[str, Any]] = None,
                   value_estimate: Optional[float] = None) -> None:
    _trace, _parent = _ambient_trace()   # 그 쿼리의 여정에 합류
    _post("/api/v1/ingest/span",
          build_selection_span(selection_id, query, selected_agent, method,
                               confidence, elapsed_ms, reasoning,
                               graph_insights, value_estimate,
                               trace_id=_trace, parent_id=_parent))


def emit_feedback(selection_id: str, selected_agent: str, success: bool,
                  ema_success_rate: Optional[float] = None,
                  query_semantics: Optional[Dict[str, Any]] = None) -> None:
    _post("/api/v1/ingest/event",
          build_feedback_event(selection_id, selected_agent, success,
                               ema_success_rate, query_semantics))
