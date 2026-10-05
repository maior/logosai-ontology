"""청크 품질 판정 — 근거가 아닌 조각을 **어휘 없이 구조로** 가려낸다.

aicoach 는 블록리스트(`참조순보험요율`·`예정사업비`…)로 요약서를 걸렀다. 그쪽
코퍼스엔 상품요약서가 별도 문서로 들어오기 때문에 통했다. 우리 데이터로 같은
마커를 돌리면 걸리는 것이 정당한 조항(`제18조 【보험계약의 성립】`)이라 이식하면
해악이다(2026-07-27 실측). 그래서 신호를 구조에서 찾는다 — 도메인·언어 무관.

삭제하지 않고 강등한다: TRUST_RANK 의 summary=0 을 쓰고, 판정은 pipeline 이
게이트(기본 off) 아래에서만 적용한다. 측정 없이 기본 동작을 바꾸지 않는다.
"""

import re
from typing import Any, Dict, Optional, Tuple

# 강등 등급 — pipeline.TRUST_RANK 에 이미 있는 값을 쓴다. 새 어휘를 만들면
# guard_attrs_by_trust 가 그것을 모르고 등급 비교가 조용히 무력해진다.
DEMOTED_TRUST = "summary"

# 표본이 얇으면 비율이 쉽게 튄다 — 짧은 조각은 판정을 보류한다(무죄 추정).
MIN_ASSESS_LEN = 40

# 목차의 보편적 형태: 제목과 쪽번호를 잇는 leader. 어휘가 아니라 **구두점**이다.
_LEADER = re.compile(r"[.·‥…∙•]{3,}|\.{3,}")
TOC_RATIO_THRESHOLD = 0.03

# OCR 잡음의 보편적 형태: 고립된 한 글자 토큰이 이어진다("a) 은 olnt zl 오 Pal").
# 숫자 단독 토큰은 정당한 표(연도·금액)에 흔하므로 잡음으로 세지 않는다.
_TOKEN = re.compile(r"\S+")
_SINGLE_CHAR_WORD = re.compile(r"^[^\W\d_]$", re.UNICODE)
GARBLED_RATIO_THRESHOLD = 0.35


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def toc_ratio(text: Optional[str]) -> float:
    """점선 leader 가 차지하는 문자 비율 — 목차 페이지 지표.

    밀도로 재는 이유: 정상 조문도 가운뎃점을 쓴다(`제3호·제4호`). 존재 유무로
    판정하면 조항이 목차로 오분류된다 — 비율이어야 구분된다.
    """
    body = _text(text)
    if not body:
        return 0.0
    leader_chars = sum(len(m.group(0)) for m in _LEADER.finditer(body))
    return leader_chars / len(body)


def garbled_ratio(text: Optional[str]) -> float:
    """고립 단문자 토큰의 비율 — OCR 잡음 지표.

    숫자를 제외하는 것이 핵심이다: 표의 `120 145 183` 은 정당한 데이터이고
    잡음이 아니다. 글자 한 개짜리 토큰이 많은 것만 잡는다.
    """
    body = _text(text)
    if not body:
        return 0.0
    tokens = _TOKEN.findall(body)
    if not tokens:
        return 0.0
    lone = sum(1 for tok in tokens if _SINGLE_CHAR_WORD.match(tok))
    return lone / len(tokens)


def assess_chunk(text: Optional[str]) -> Tuple[str, str, Dict[str, float]]:
    """청크 하나의 품질 판정 → (verdict, reason, metrics).

    verdict 는 "ok" 또는 DEMOTED_TRUST. reason 은 "" | "toc" | "garbled" —
    집계에 쓰이므로 이유를 반드시 남긴다(no silent caps).

    던지지 않는다: 인제스트 도중 한 청크의 판정 실패가 빌드를 멈추면 안 된다.
    """
    body = _text(text)
    metrics = {"toc_ratio": toc_ratio(body),
               "garbled_ratio": garbled_ratio(body)}
    if len(body) < MIN_ASSESS_LEN:
        return "ok", "", metrics      # 표본 부족 — 무죄 추정
    if metrics["toc_ratio"] >= TOC_RATIO_THRESHOLD:
        return DEMOTED_TRUST, "toc", metrics
    if metrics["garbled_ratio"] >= GARBLED_RATIO_THRESHOLD:
        return DEMOTED_TRUST, "garbled", metrics
    return "ok", "", metrics
