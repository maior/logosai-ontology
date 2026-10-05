"""
커버리지 에이전트 (Coverage Checker) — 추출이 놓친 개체 탐지.

③번 검수 보조. ①근거대조가 "그래프에 들어온 것이 옳은가"(precision)를,
②일관성 감시가 "그래프 안이 모순 없는가"(결정 가능 → LLM 0콜)를 본다면,
커버리지는 "원문에는 있는데 그래프에 없는 것"(recall)을 본다. 무엇이
개체인가는 원문을 읽는 **판단**이라 LLM 이 필요하다 — evidence_checker 와
같은 주입 패턴, gemini-3.5-flash 기본 (core.llm_provider 경유).

신뢰의 근거 — LLM 출력은 전부 불신하고 검증 가능한 것만 채택한다:
- **원문에 문자 그대로 없는 후보는 버린다** — 공백 정규화 부분문자열 비교
  (evidence_checker._quote_in_chunks 재사용 — 같은 규칙을 두 벌 두면
  두 에이전트의 '원문' 정의가 갈라진다). 원문 밖 개체를 "놓쳤다"고
  보고하면 커버리지가 아니라 환각 주입이다.
- 이미 아는 개체(이름·별칭, 대소문자/공백 무시)는 버린다 — 있는 것을
  "놓쳤다"고 하면 검수자가 도구를 불신하게 된다.
- 허용 타입 밖의 후보는 버린다 — gap 보고가 스키마를 여는 뒷문이 되면 안 된다.
- 위반은 후보 단위로 조용히 버린다 — 한 후보 때문에 배치 전체를 잃지 않는다.
- 빈 원문이면 LLM 을 부르지 않는다 (0콜) / LLM 실패는 [] (never raise).

발견(gap)의 기록은 호출자(service.check_coverage)가 review_store 감사
로그에 남긴다 — ②의 findings 와 달리 재실행 비용이 LLM 이라 비싸고,
"무엇을 놓쳤었나" 자체가 사건이기 때문이다.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# 인용 검증 규칙 재사용 — "원문에 있다"의 정의는 에이전트마다 달라선 안 된다
from .evidence_checker import _quote_in_chunks, _squash_ws

_COVERAGE_PROMPT = """당신은 지식그래프 검수 보조자입니다. 아래 원문에서
그래프 추출이 놓친 **새 개체 후보**만 찾으세요.

## 규칙 (반드시 지킬 것)
1. 원문에 **문자 그대로 등장하는** 표현만 후보로 제시합니다.
   원문 밖 지식·의역·요약 금지 — 지어낸 후보는 전부 무효 처리됩니다.
2. '이미 알려진 개체' 목록에 있는 것은 제외합니다 — 이미 그래프에 있습니다.
   목록에 **없는** 새 개체만 제시하세요.
3. type 은 아래 '허용 타입' 목록에서만 고릅니다.
4. JSON 외의 다른 텍스트를 출력하지 마세요.

## 원문
{chunk}

## 이미 알려진 개체 ({known_count}개)
{known}

## 허용 타입
{types}

## 출력 형식
{{"candidates": [{{"name": "...", "type": "..."}}]}}
"""


def build_coverage_prompt(chunk_text: str, known_names: List[str],
                          node_types: List[str]) -> str:
    """grounded 프롬프트 조립 — 순수 함수."""
    return _COVERAGE_PROMPT.format(
        chunk=chunk_text,
        known_count=len(known_names),
        known=json.dumps(list(known_names), ensure_ascii=False),
        types=json.dumps(list(node_types), ensure_ascii=False))


def parse_coverage(raw: str, chunk_text: str, known_names: List[str],
                   node_types: List[str]) -> List[Dict[str, Any]]:
    """LLM 후보 검증 — 출력은 불신한다. 절대 raise 하지 않는다.

    후보 단위 드롭 규칙 (배치 전체를 죽이지 않는다):
    - 원문에 문자 그대로 없다 (공백 정규화 비교) → 지어낸 개체
    - 이미 아는 개체다 (대소문자/공백 무시) → gap 이 아니다
    - 허용 타입 밖이다 → 스키마 열지 않기
    """
    from ..builder.extractor import parse_llm_json

    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return []
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        return []

    known = {_squash_ws(str(k)).casefold() for k in known_names}
    allowed = set(node_types)
    accepted: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = str(candidate.get("name") or "").strip()
        node_type = str(candidate.get("type") or "").strip()
        if not name or node_type not in allowed:
            continue
        if not _quote_in_chunks(name, [chunk_text]):
            continue  # 원문에 없는 개체 — 환각
        norm = _squash_ws(name).casefold()
        if norm in known:
            continue  # 이미 그래프에 있다 — 놓친 게 아니다
        key = (norm, node_type)
        if key in seen:
            continue  # 같은 후보 중복 제시 방지
        seen.add(key)
        accepted.append({"name": name, "type": node_type})
    return accepted


class CoverageChecker:
    """청크 하나에서 놓친 개체 후보를 찾는다. LLM 은 주입 가능."""

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

    async def check_chunk(self, stored_chunk, known_names: List[str],
                          node_types: List[str]) -> List[Dict[str, Any]]:
        """청크 원문 vs 기지 개체 → 놓친 개체 후보 목록.

        각 후보에 provenance(chunk_id/source/quote)를 싣는다 — 어디서
        놓쳤는지 없이는 검수자가 확인할 수 없다. 빈 원문은 0콜 — 원문 없이
        물으면 지어낸 후보가 돌아온다. LLM 실패는 [] — 커버리지 점검이
        죽는다고 검수 자체가 막히면 안 된다.
        """
        text = (stored_chunk.text or "").strip()
        if not text:
            return []
        try:
            raw = await self._call_llm(
                build_coverage_prompt(text, known_names, node_types))
        except Exception as e:
            logger.warning(f"⚠️ Coverage check LLM failed "
                           f"({stored_chunk.chunk_id}): {e}")
            return []
        return [{"name": found["name"], "type": found["type"],
                 "chunk_id": stored_chunk.chunk_id,
                 "source": stored_chunk.source,
                 # quote == name — "원문에 문자 그대로 있다"가 채택 조건이므로
                 # 이름 자체가 곧 검증된 인용이다
                 "quote": found["name"]}
                for found in parse_coverage(raw, text, known_names, node_types)]
