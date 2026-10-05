"""
스키마 큐레이터 (④) — auto 스키마 제안을 네임스페이스의 기존 어휘에 정합.

문제는 실측돼 있다: 일관성 린터가 heritage_kr 에서 name_type_conflict 10건을
찾았다 ("국보"가 Designation/HeritageClass 두 타입에 중복). 원인 경로는 auto
스키마다 — 빌드마다 LLM 이 2000자 샘플만 보고 타입을 새로 제안하므로, 같은
네임스페이스에 데이터셋을 쌓을수록 같은 개념이 다른 타입명으로 쪼개진다
(Clause vs Provision). 쪼개지면 롤업 추론(is_a 사슬 단절)·축 4 확장·데이터셋
품질이 함께 나빠진다. 린터는 **사후 탐지**, 이 모듈은 **사전 예방**이다.

역할 분담 (프로젝트 핵심 원칙 그대로):
- 결정적 우선: 대소문자·공백만 다른 타입/술어는 LLM 없이 병합한다 —
  이건 계산이지 판단이 아니다.
- LLM(gemini-3.5-flash)은 남은 것만: 의미 동치 판단 (Provision ≟ Clause).
  타입·술어를 한 번에 물어 **1콜**. 출력은 불신 — 타깃이 기존 어휘에
  실존하지 않으면 그 매핑만 버린다 (지어낸 타입으로의 개명은 병합이 아니라
  새 파편이다).
- **auto 모드에서만**: 사용자가 명시한 스키마(preset/custom)를 고치는 것은
  확인 게이트 철학("승인한 것과 다른 것을 만들지 않는다") 위반이다.
- 무엇을 어디로 병합했는지 BuildReport.schema_mappings 에 남는다 —
  조용한 개명 금지.
- 큐레이션 실패는 빌드를 막지 않는다 — 개선이지 관문이 아니다.
"""

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


def _norm(name: str) -> str:
    """대소문자·공백·언더스코어 무시 정규화 — 이 차이는 표기이지 개념이 아니다."""
    return re.sub(r"[\s_]+", "", str(name)).casefold()


def existing_vocabulary(graph) -> Dict[str, List[str]]:
    """그래프의 현재 어휘 (타입·술어) — 결정적 스캔."""
    types = sorted({attrs.get("type", "") for _, attrs in graph.nodes(data=True)
                    if attrs.get("type")})
    predicates = sorted({attrs.get("predicate", "")
                         for _, _, attrs in graph.edges(data=True)
                         if attrs.get("predicate")})
    return {"types": types, "predicates": predicates}


def match_deterministic(proposed: List[str],
                        existing: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """표기만 다른 이름을 LLM 없이 병합한다.

    반환: (mapping {제안명 → 기존명}, 미해결 목록). 정확히 같은 이름은
    매핑이 아니라 항등이므로 mapping 에 넣지 않는다.
    """
    by_norm = {_norm(name): name for name in existing}
    mapping: Dict[str, str] = {}
    unmatched: List[str] = []
    for name in proposed:
        canonical = by_norm.get(_norm(name))
        if canonical is None:
            unmatched.append(name)
        elif canonical != name:
            mapping[name] = canonical
    return mapping, unmatched


_RECONCILE_PROMPT = """당신은 지식그래프 스키마 관리자입니다. 새 빌드가 제안한
타입/술어 이름이 기존 어휘의 어떤 이름과 **같은 개념**을 가리키면 매핑하세요.

## 규칙 (반드시 지킬 것)
1. 확실히 같은 개념일 때만 매핑하세요. 애매하면 null (새 이름으로 유지).
2. 매핑 대상은 반드시 아래 '기존 어휘' 목록의 이름이어야 합니다.
3. JSON 외의 다른 텍스트를 출력하지 마세요.

## 기존 어휘 (이 네임스페이스에 이미 있는 이름들)
타입: {existing_types}
술어: {existing_predicates}

## 새 제안 (판단 대상)
타입: {proposed_types}
술어: {proposed_predicates}

## 출력 형식
{{"mapping": {{"제안이름": "기존이름 또는 null", ...}}}}
"""


def build_reconcile_prompt(proposed_types: List[str],
                           proposed_predicates: List[str],
                           existing_types: List[str],
                           existing_predicates: List[str]) -> str:
    return _RECONCILE_PROMPT.format(
        existing_types=json.dumps(existing_types, ensure_ascii=False),
        existing_predicates=json.dumps(existing_predicates, ensure_ascii=False),
        proposed_types=json.dumps(proposed_types, ensure_ascii=False),
        proposed_predicates=json.dumps(proposed_predicates, ensure_ascii=False))


def parse_reconcile_mapping(raw: str, asked: List[str],
                            existing: List[str]) -> Dict[str, str]:
    """LLM 매핑 검증 — 출력은 불신한다.

    - 타깃이 기존 어휘에 없으면 그 매핑만 버린다.
    - 묻지 않은 소스 키는 버린다 (요청 밖 개명 금지).
    - null/자기자신 매핑은 '새 이름 유지' — 매핑 아님.
    절대 raise 하지 않는다.
    """
    from ..builder.extractor import parse_llm_json

    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return {}
    raw_mapping = parsed.get("mapping")
    if not isinstance(raw_mapping, dict):
        return {}

    asked_set = set(asked)
    existing_set = set(existing)
    mapping: Dict[str, str] = {}
    for source, target in raw_mapping.items():
        if source not in asked_set:
            continue
        if not isinstance(target, str) or target not in existing_set:
            continue
        if target == source:
            continue
        mapping[source] = target
    return mapping


def apply_mapping(schema, type_mapping: Dict[str, str],
                  predicate_mapping: Dict[str, str]):
    """매핑을 적용한 새 BuilderSchema — 원본은 건드리지 않는다.

    타입 개명은 술어의 domain/range 에도 전파된다 (안 하면 스키마가
    자기모순이 된다). 병합으로 생긴 중복 타입은 하나로 준다.
    """
    from ..builder.models import BuilderSchema

    def rename(name: str) -> str:
        return type_mapping.get(name, name)

    node_types: List[str] = []
    for name in schema.node_types:
        renamed = rename(name)
        if renamed not in node_types:
            node_types.append(renamed)

    predicates: Dict[str, Tuple[str, str]] = {}
    for pred, (domain, range_) in schema.predicates.items():
        new_pred = predicate_mapping.get(pred, pred)
        predicates[new_pred] = (rename(domain), rename(range_))

    return BuilderSchema(node_types=node_types, predicates=predicates)


class SchemaCurator:
    """auto 스키마 제안 ↔ 기존 어휘 정합기. LLM 은 주입 가능."""

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self.llm_fn = llm_fn

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_fn is not None:
            return await asyncio.to_thread(self.llm_fn, prompt)
        from .llm_provider import resolve_provider
        provider = resolve_provider("google")
        return await provider.complete(prompt)

    async def reconcile(self, schema, graph):
        """제안 스키마를 그래프의 기존 어휘에 정합시킨다.

        반환: (정합된 schema, mappings dict 또는 None).
        기존 어휘가 비어 있으면 정합할 대상이 없다 — 즉시 (원안, None).
        LLM 단계 실패 시 결정적 병합분만 적용하고 계속한다 (fail-open).
        """
        vocab = existing_vocabulary(graph)
        if not vocab["types"] and not vocab["predicates"]:
            return schema, None

        type_map, unmatched_types = match_deterministic(
            schema.node_types, vocab["types"])
        pred_map, unmatched_preds = match_deterministic(
            list(schema.predicates), vocab["predicates"])

        # LLM 은 결정적으로 못 가른 것만 — 타입·술어를 한 프롬프트에 담아 1콜
        if unmatched_types or unmatched_preds:
            try:
                raw = await self._call_llm(build_reconcile_prompt(
                    unmatched_types, unmatched_preds,
                    vocab["types"], vocab["predicates"]))
                type_map.update(parse_reconcile_mapping(
                    raw, unmatched_types, vocab["types"]))
                pred_map.update(parse_reconcile_mapping(
                    raw, unmatched_preds, vocab["predicates"]))
            except Exception as e:
                # 큐레이션은 개선이지 관문이 아니다 — 결정적 병합분만 안고 계속
                logger.warning(f"⚠️ Schema reconcile LLM failed ({e}) — "
                               f"deterministic merges only")

        if not type_map and not pred_map:
            return schema, None

        merged = apply_mapping(schema, type_map, pred_map)
        mappings = {"types": type_map, "predicates": pred_map}
        logger.info(f"🧭 Schema curated: {len(type_map)} type(s), "
                    f"{len(pred_map)} predicate(s) merged into existing vocabulary")
        return merged, mappings
