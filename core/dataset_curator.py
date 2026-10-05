"""
데이터셋 큐레이터 (⑥) — build_training_dataset 산출물의 자동 품질 게이트.

프로젝트의 원래 목표는 파인튜닝용 학습 데이터 추출이다. 추출 행은 환각이
없지만(전부 결정적 생성 + 출처) 품질은 고르지 않다: 중복, 너무 짧은 원문,
한 문서의 지배, summary 신뢰 등급 행의 혼입. aicoach 는 이 게이트를
수동(curated=true)으로 운영했고, 이 모듈은 그것을 자동화한다.

두 층으로 나눈 이유 — 비용과 신뢰가 다르다:
1. **결정적 큐레이션 (LLM 0콜)** — curate_rows. 중복·길이·지배·trust 는
   LLM 없이 판정 가능하다. 순수 함수, 순서 보존, 입력 불변.
2. **선택적 LLM 품질 게이트** — QualityScorer. "input 만으로 output 을
   재현할 수 있는가"는 결정적으로 잴 수 없어 LLM 에 묻되, 출력은 전부
   불신한다 (evidence_checker 와 같은 원리): 판정 불가는 **보존**이고
   배치 실패도 **보존**이다. 품질 게이트가 죽는다고 데이터가 사라지면
   안 된다.

리포트는 'no silent caps' 원칙 — 무엇이 왜 떨어졌는지 반드시 센다.
kept + sum(dropped) == 입력 행 수.

LLM 기본은 evidence_checker 와 동일 (core.llm_provider 경유). 테스트는
llm_fn 주입.
"""

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

DROP_REASONS = ("duplicate", "too_short", "source_capped", "trust_excluded")


def _squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _dedup_key(row: Dict[str, Any]) -> Tuple:
    """행의 **의미 페이로드**로 중복 키를 만든다 — source 나 chunk_id 같은
    provenance 가 달라도 학습 신호가 같으면 같은 행이다.

    - evidence: 원문(input)이 학습 신호의 전부다. 공백만 다른 원문은
      같은 원문이므로 whitespace-squash 후 비교한다.
    - qa: instruction+output 쌍
    - triple: subject+predicate+object
    - surface: input+output 쌍
    - 미지 포맷: 행 전체 내용 (보수적 — 완전 동일할 때만 중복)
    """
    fmt = row.get("format", "")
    if fmt == "evidence":
        return ("evidence", _squash_ws(str(row.get("input") or "")))
    if fmt == "qa":
        return ("qa", str(row.get("instruction") or ""),
                _stable_json(row.get("output")))
    if fmt == "triple":
        return ("triple", str(row.get("subject") or ""),
                str(row.get("predicate") or ""), str(row.get("object") or ""))
    if fmt == "surface":
        return ("surface", str(row.get("input") or ""),
                str(row.get("output") or ""))
    return ("other", _stable_json(row))


def curate_rows(rows: List[Dict[str, Any]],
                dedup: bool = True,
                min_input_chars: int = 0,
                max_per_source: Optional[int] = None,
                exclude_trust: Optional[List[str]] = None,
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """결정적 큐레이션 — LLM 0콜, 순수 함수, 순서 보존, 입력 불변.

    - dedup: 의미 페이로드가 같은 행은 첫 등장만 남긴다 (_dedup_key).
    - min_input_chars: input 필드가 있는 행(evidence/surface)만 대상 —
      공백 정규화 후 길이가 미달이면 탈락 (너무 짧은 원문은 학습 신호가
      약하다). input 이 없는 행(qa/triple)은 이 필터를 통과한다.
    - max_per_source: source 별 앞 N개만 유지 (한 문서가 데이터셋을
      지배하면 모델이 그 문서를 외운다).
    - exclude_trust: trust 필드가 목록에 있는 행 탈락 (예: summary —
      boilerplate 에서 뽑은 쌍을 원본 쌍과 섞지 않는다, aicoach layer 원칙).

    반환 report 는 'no silent caps': 모든 탈락이 이유별로 집계된다.
    """
    excluded = set(exclude_trust or ())
    kept: List[Dict[str, Any]] = []
    dropped = {reason: 0 for reason in DROP_REASONS}
    seen_keys: set = set()
    per_source: Dict[str, int] = {}

    for row in rows:
        if excluded and row.get("trust") in excluded:
            dropped["trust_excluded"] += 1
            continue

        if min_input_chars > 0 and "input" in row:
            if len(_squash_ws(str(row.get("input") or ""))) < min_input_chars:
                dropped["too_short"] += 1
                continue

        if dedup:
            key = _dedup_key(row)
            if key in seen_keys:
                dropped["duplicate"] += 1
                continue
            seen_keys.add(key)

        if max_per_source is not None:
            source = str(row.get("source") or "")
            if per_source.get(source, 0) >= max_per_source:
                dropped["source_capped"] += 1
                continue
            per_source[source] = per_source.get(source, 0) + 1

        kept.append(row)

    return kept, {"kept": len(kept), "dropped": dropped}


# ─── LLM 품질 게이트 (선택적) ────────────────────────────────────────

_SCORE_PROMPT = """당신은 AI 학습 데이터 품질 심사자입니다. 아래 학습쌍
각각의 품질을 1~5로 채점하세요.

## 채점 기준
- input(또는 instruction)만으로 output 을 재현할 수 있는가
- output 이 input 에 근거하는가
- 1 = 학습에 해롭다, 3 = 보통, 5 = 훌륭한 학습쌍

## 학습쌍 (index 0..{last})
{rows}

## 출력 형식 (JSON 외 다른 텍스트 금지)
{{"scores": [{{"index": 0, "score": 5, "reason": "..."}}]}}
"""


def build_score_prompt(rows: List[Dict[str, Any]]) -> str:
    """채점 프롬프트 조립 — 순수 함수. 행마다 index 를 붙여 배치로 묻는다
    (행마다 1콜이면 비용이 폭발한다)."""
    numbered = [{"index": i, **{k: v for k, v in row.items()
                                if k in ("format", "instruction", "input",
                                         "output", "subject", "predicate",
                                         "object")}}
                for i, row in enumerate(rows)]
    return _SCORE_PROMPT.format(
        last=len(rows) - 1,
        rows=json.dumps(numbered, ensure_ascii=False, indent=1))


def parse_scores(raw: str, batch_len: int) -> Optional[Dict[int, int]]:
    """LLM 채점 검증 — 출력은 불신한다. 절대 raise 하지 않는다.

    - 응답 전체가 JSON 이 아니거나 scores 가 리스트가 아니면 None
      (배치 단위 판정 불가 → 호출측이 배치 전체를 보존한다)
    - 항목 단위 검증: index 는 배치 범위 내 int, score 는 int 1..5 —
      아니면 그 항목만 버린다 (해당 행은 채점 누락 → 보존).
    """
    from ..builder.extractor import parse_llm_json

    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed.get("scores"), list):
        return None

    scores: Dict[int, int] = {}
    for entry in parsed["scores"]:
        if not isinstance(entry, dict):
            continue
        index, score = entry.get("index"), entry.get("score")
        # bool 은 int 의 서브클래스 — True/False 를 index/score 로 받지 않는다
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        if not isinstance(score, int) or isinstance(score, bool):
            continue
        if not (0 <= index < batch_len) or not (1 <= score <= 5):
            continue
        scores[index] = score
    return scores


class QualityScorer:
    """학습쌍 품질을 LLM 으로 배치 채점한다. LLM 은 주입 가능.

    권한 경계: threshold 미달 행을 떨어뜨리는 것까지. 판정 불가(잘못된
    index/score, 항목 누락)는 **보존**이다 — 판정 불가로 데이터를 버리지
    않는다."""

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

    async def score_rows(self, rows: List[Dict[str, Any]],
                         threshold: int = 3,
                         batch_size: int = 10,
                         ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """배치 채점 → (kept_rows, {"low_quality": x, "quality_skipped": y}).

        - score < threshold → 탈락 (low_quality 집계)
        - 항목 누락/비정상 → 그 행은 보존 (판정 불가 ≠ 저품질)
        - 배치 LLM 실패 → 배치 전체 보존 + quality_skipped 집계 —
          품질 게이트가 죽는다고 데이터가 사라지면 안 된다.
        절대 raise 하지 않는다.
        """
        kept: List[Dict[str, Any]] = []
        report = {"low_quality": 0, "quality_skipped": 0}

        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            scores: Optional[Dict[int, int]] = None
            try:
                raw = await self._call_llm(build_score_prompt(batch))
                scores = parse_scores(raw, len(batch))
            except Exception as e:
                logger.warning(f"⚠️ Quality scoring LLM failed "
                               f"(batch @{start}): {e}")

            if scores is None:
                kept.extend(batch)
                report["quality_skipped"] += len(batch)
                continue

            for i, row in enumerate(batch):
                score = scores.get(i)
                if score is not None and score < threshold:
                    report["low_quality"] += 1
                else:
                    kept.append(row)

        return kept, report
