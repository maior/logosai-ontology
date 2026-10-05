"""
LLM extractor — natural language chunk → structured entities/relations.

The LLM function is injected (tests pass a deterministic fake; production
resolves the ontology LLM manager lazily). The prompt injects the CLOSED
schema vocabulary and forbids facts not present in the text — both are
aicoach extract.py patterns that measurably suppress hallucination.
"""

import json
import re
from typing import Any, Callable, Dict, Optional

from .models import BuilderSchema, Chunk

LLMFn = Callable[[str], str]

_CODEFENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_PROMPT_TEMPLATE = """당신은 문서에서 지식을 추출하는 분석기입니다.
아래 텍스트에서 개체(entity)와 관계(relation)를 JSON으로만 추출하세요.

## 규칙 (반드시 지킬 것)
1. entity의 type은 다음 목록에서만 선택: {node_types}
2. relation의 predicate는 다음 목록에서만 선택: {predicates}
3. relation의 subject/object는 반드시 entities에 있는 name과 정확히 일치
4. 텍스트에 없는 사실을 만들지 마세요. 확실하지 않으면 제외하세요.
5. 수치·금액은 추출하지 말고 텍스트 그대로의 명칭만 추출하세요.
6. 단, 위치 정보(위도 lat, 경도 lng — 숫자)와 지역(region), 분류(category)가
   텍스트에 명시되어 있으면 entity의 attrs에 보존하세요.
   예: "attrs": {{"lat": 35.79, "lng": 129.35, "region": "경주", "category": "문화"}}
7. 의미론적 정보: 각 entity의 attrs에 definition(텍스트에 근거한 한 줄 정의)과
   aliases(동의어·별칭·다른 표기, 배열, 최대 4개)를 포함하세요. definition은 반드시
   텍스트에 근거해야 합니다. aliases는 텍스트의 표현에 더해, 정식 명칭의 **널리
   통용되는 통칭·약칭**도 포함하세요(예: "유방의 악성 신생물"→"유방암",
   "위의 악성 신생물"→"위암"). 단 확실히 같은 대상일 때만 — 불확실하면 제외하세요.
   예: "attrs": {{"definition": "조선 도성의 남쪽 정문", "aliases": ["남대문"]}}
8. JSON 외의 다른 텍스트를 출력하지 마세요.

## 출력 형식
{{"entities": [{{"name": "...", "type": "...", "attrs": {{}}}}], "relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}

## 텍스트
{text}
"""


_SCHEMA_PROPOSAL_TEMPLATE = """당신은 온톨로지 설계자입니다.
아래 샘플 데이터를 보고, 이 데이터를 지식그래프로 만들기에 적합한
스키마를 제안하세요. JSON으로만 답하세요.

## 규칙
1. node_types: 이 데이터에 등장하는 개체의 종류 (3~8개, 영문 PascalCase)
2. predicates: 관계 이름과 [domain, range] (2~8개, 영문 camelCase)
3. JSON 외의 다른 텍스트를 출력하지 마세요.
{vocab_block}
## 출력 형식
{{"node_types": ["TypeA", "TypeB"], "predicates": {{"relName": ["TypeA", "TypeB"]}}}}

## 샘플 데이터
{text}
"""

# 기존 어휘 힌트 — 스키마 파편화의 원천 수정 (④ 큐레이터의 1차 방어선).
# 라이브 실측: 기존에 Clause/Regulation 이 있는 네임스페이스에서 이 힌트 없이
# 제안을 받자 Gemini 가 LegalProvision/Law 를 새로 지었고, 사후 reconcile
# ("확실히 같은 개념일 때만")도 보수적으로 병합을 거부했다. 제안 시점에
# 어휘를 보여주는 것이 사후 병합보다 근본적이다 — reconcile 은 백스톱이다.
_VOCAB_HINT = """
## 기존 어휘 (이 네임스페이스에 이미 있는 이름)
타입: {types}
술어: {predicates}
4. 같은 개념에는 반드시 위 기존 이름을 재사용하세요. 새 이름은 기존 어휘에
   정말 없는 개념에만 만드세요 — 같은 개념이 다른 이름으로 쪼개지면
   그래프 추론과 검색이 함께 나빠집니다.
"""


def build_schema_proposal_prompt(sample_text: str,
                                 existing_vocab=None) -> str:
    """auto 모드: LLM에게 데이터에 맞는 스키마를 제안받는 프롬프트.

    existing_vocab={"types": [...], "predicates": [...]} 가 있으면 재사용을
    지시한다 (없거나 비어 있으면 힌트 블록 자체가 빠진다 — 첫 빌드).
    """
    vocab_block = ""
    if existing_vocab and (existing_vocab.get("types")
                           or existing_vocab.get("predicates")):
        vocab_block = _VOCAB_HINT.format(
            types=json.dumps(existing_vocab.get("types", []),
                             ensure_ascii=False),
            predicates=json.dumps(existing_vocab.get("predicates", []),
                                  ensure_ascii=False))
    return _SCHEMA_PROPOSAL_TEMPLATE.format(text=sample_text,
                                            vocab_block=vocab_block)


def build_extraction_prompt(schema: BuilderSchema, text: str) -> str:
    predicate_lines = ", ".join(
        f"{name}({dom}→{rng})" for name, (dom, rng) in schema.predicates.items())
    return _PROMPT_TEMPLATE.format(
        node_types=", ".join(schema.node_types),
        predicates=predicate_lines,
        text=text,
    )


def parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Lenient JSON parse: strips code fences and surrounding prose.
    Returns None (never raises) when no JSON object can be recovered."""
    if not raw:
        return None
    fenced = _CODEFENCE.search(raw)
    candidate = fenced.group(1) if fenced else raw

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def extract_chunk(chunk: Chunk, schema: BuilderSchema,
                  llm_fn: LLMFn) -> Optional[Dict[str, Any]]:
    """Run the LLM on one chunk. Returns the parsed raw dict (pre-validation)
    or None when the response is unusable."""
    prompt = build_extraction_prompt(schema, chunk.text)
    return parse_llm_json(llm_fn(prompt))
