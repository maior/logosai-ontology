"""
근거대조 에이전트 (Evidence Checker) — 검수 큐 사전판정.

aicoach ACP 협업 6에이전트(recommendation_orchestrator 계열)에서 확인된
패턴을 검수에 적용한다: **역할 하나 + grounded 도구 하나, 추천까지만.**
- 역할: 추출된 사실이 원문 청크(span provenance, 축 2)에 근거하는지 판단
- grounded 도구: ChunkStore — 이 노드가 추출된 바로 그 원문
- 권한 경계: 검수 큐에 confirm/reject **추천**을 다는 것까지. 최종 판정
  (묘비/확정)은 인간만 만든다. aicoach 오케스트레이터가 "사실 생성 안 함"을
  docstring 에 못박은 것과 같은 급의 경계다.

신뢰의 근거 — LLM 출력은 전부 불신하고 검증 가능한 것만 채택한다:
- **인용이 가짜면 추천도 가짜다**: evidence_quote 가 원문 청크의 실제 부분
  문자열이 아니면 unsure 로 강등. 근거를 지어내는 추천은 검수를 돕는 게
  아니라 오염시킨다 (aicoach Advisor 의 카탈로그 화이트리스트와 같은 원리).
- 근거 청크가 없으면 LLM 을 부르지 않는다 — 원문 없이 물으면 지어낸 판단이
  돌아온다 (no_evidence, 0콜).
- 이유 없는 reject 는 unsure 로 강등 — 이유 없이는 인간이 판정할 수 없다.

LLM 기본은 gemini-3.5-flash (core.llm_provider 경유, 빌더 추출 모델과 동일
기본 공유). 테스트는 llm_fn 주입.
"""

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

VERDICTS = ("confirm", "reject", "unsure")

_CHECK_PROMPT = """당신은 지식그래프 검수 보조자입니다. 아래 '추출된 사실'이
'원문'에 근거하는지만 판단하세요.

## 규칙 (반드시 지킬 것)
1. 원문에 적힌 내용만 근거로 인정합니다. 원문 밖의 지식·추측 사용 금지.
2. verdict 판단 기준:
   - confirm: 사실의 핵심(이름·정의)이 원문에서 확인된다
   - reject: 원문이 사실과 모순되거나, 사실이 원문 어디에도 없다
   - unsure: 원문만으로는 판단할 수 없다
3. evidence_quote: 판단의 근거가 된 원문 구절을 **원문 그대로** 인용하세요
   (confirm/reject 필수). 지어내면 추천 전체가 무효 처리됩니다.
4. rationale: 판단 이유 한 문장 (reject 는 필수).
5. JSON 외의 다른 텍스트를 출력하지 마세요.

## 추출된 사실 (검수 대상)
{fact}

## 원문 (이 사실이 추출된 청크 {chunk_count}개)
{chunks}

## 출력 형식
{{"verdict": "confirm|reject|unsure", "rationale": "...", "evidence_quote": "..."}}
"""


def build_check_prompt(node_view: Dict[str, Any],
                       chunk_texts: List[str]) -> str:
    """grounded 프롬프트 조립 — 순수 함수."""
    fact = {k: v for k, v in node_view.items()
            if k in ("node_id", "name", "type", "definition", "trust", "source")
            and v}
    chunks_block = "\n".join(f"[{i + 1}] {text}"
                             for i, text in enumerate(chunk_texts))
    return _CHECK_PROMPT.format(
        fact=json.dumps(fact, ensure_ascii=False, indent=1),
        chunk_count=len(chunk_texts),
        chunks=chunks_block)


def _squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _quote_in_chunks(quote: str, chunk_texts: List[str]) -> bool:
    """인용이 원문의 실제 부분문자열인가 — 공백 정규화 비교.

    LLM 은 공백·개행을 바꿔 인용하는 일이 잦다. 내용이 같은데 공백 때문에
    강등되면 추천이 무의미하게 비어지므로, 공백만 정규화하고 글자는
    그대로 비교한다 (글자가 다르면 지어낸 것이다).
    """
    needle = _squash_ws(quote)
    if not needle:
        return False
    return any(needle in _squash_ws(text) for text in chunk_texts)


def _unsure(rationale: str) -> Dict[str, Any]:
    return {"verdict": "unsure", "rationale": rationale, "evidence_quote": ""}


def parse_check_verdict(raw: str,
                        chunk_texts: List[str]) -> Dict[str, Any]:
    """LLM 판정 검증 — 출력은 불신한다. 절대 raise 하지 않는다.

    강등 규칙(전부 unsure 로, 이유를 rationale 에 남긴다 — 사람이 왜
    강등됐는지 읽을 수 있어야 재검수가 가능하다):
    - JSON 이 아니거나 verdict 가 미지의 값
    - confirm/reject 인데 인용이 원문에 없다 (지어낸 근거)
    - reject 인데 이유가 없다
    """
    from ..builder.extractor import parse_llm_json

    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return _unsure("LLM 응답이 JSON 이 아님")

    verdict = str(parsed.get("verdict") or "").strip().lower()
    rationale = str(parsed.get("rationale") or "").strip()
    quote = str(parsed.get("evidence_quote") or "").strip()

    if verdict not in VERDICTS:
        return _unsure(f"미지의 verdict '{verdict}'")

    if verdict in ("confirm", "reject"):
        if not _quote_in_chunks(quote, chunk_texts):
            return _unsure("인용(evidence_quote)이 원문에 없음 — 지어낸 근거로 판정 불가")
        if verdict == "reject" and not rationale:
            return _unsure("이유 없는 reject 추천 — 인간이 판정할 근거가 없음")

    return {"verdict": verdict, "rationale": rationale,
            "evidence_quote": quote if verdict != "unsure" else quote}


class EvidenceChecker:
    """검수 큐 항목 하나를 사전판정한다. LLM 은 주입 가능."""

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None,
                 llm_provider: str = "google",
                 llm_model: Optional[str] = None,
                 llm_base_url: Optional[str] = None):
        self.llm_fn = llm_fn
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self._provider = None

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_fn is not None:
            return await asyncio.to_thread(self.llm_fn, prompt)
        if self._provider is None:
            from .llm_provider import resolve_provider
            self._provider = resolve_provider(
                self.llm_provider, self.llm_model, base_url=self.llm_base_url)
        return await self._provider.complete(prompt)

    async def check(self, node_view: Dict[str, Any],
                    chunk_texts: List[str]) -> Dict[str, Any]:
        """사실 vs 원문 대조 → {verdict, rationale, evidence_quote}.

        근거가 없으면 LLM 을 부르지 않고 no_evidence — 원문 없이 물으면
        지어낸 판단이 돌아온다. LLM 실패는 unsure — 사전판정이 죽는다고
        검수 자체가 막히면 안 된다.
        """
        if not chunk_texts:
            return {"verdict": "no_evidence",
                    "rationale": "이 노드가 추출된 원문 청크가 없음 — 대조 불가",
                    "evidence_quote": ""}
        try:
            raw = await self._call_llm(build_check_prompt(node_view, chunk_texts))
        except Exception as e:
            logger.warning(f"⚠️ Evidence check LLM failed "
                           f"({node_view.get('node_id')}): {e}")
            return _unsure(f"LLM 호출 실패: {e}")
        return parse_check_verdict(raw, chunk_texts)
