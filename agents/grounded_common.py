"""grounded_common — ontology/agents/ grounded 에이전트 공유 순수 로직.

온톨로지 `/retrieve` 응답을 다루는 순수 함수들. 네트워크·LLM·logosai 를
import 하지 않는다 (stdlib 만) — 에이전트 handle 이 call_tool_http + LLM 과
조립할 때 이 함수들을 근거 파싱·필터·프롬프트·인용에 쓴다. 순수라 단위
테스트가 결정적(test_grounded_common.py).

설계 원칙: 사실은 온톨로지에서만(환각 0). 프롬프트가 "근거 발췌만 사용,
없으면 근거 없음, 각 문장에 [n] 인용"을 강제한다 — aicoach clause_evidence
가 약관 조항을 인용하던 원리를 도메인-무관 검색으로 옮긴 것.
"""

from typing import Any, Dict, List, Optional

__all__ = [
    "parse_hits", "filter_hits_by_source",
    "build_grounding_prompt", "citations_from_hits",
    "quality_note", "pick_source",
    "retrieval_error", "no_evidence_message",
]


def quality_note(retrieve_json: Any) -> str:
    """답변 꼬리에 붙일 측정 품질 문장 — /retrieve 의 `quality` 블록 소비.

    원칙은 온톨로지 쪽과 같다: **지어내지 않는다**. 측정됐으면 수치+시점+케이스
    수, 측정 후 설정이 바뀌었으면(stale) 경고, 측정이 없으면 **빈 문자열** —
    없는 신뢰도를 문장으로 만들지 않는다.
    """
    if not isinstance(retrieve_json, dict):
        return ""
    quality = retrieve_json.get("quality")
    if not isinstance(quality, dict) or not quality.get("measured"):
        return ""
    retrieve = quality.get("retrieve") or {}
    k = quality.get("k")
    hit_k = retrieve.get(f"hit@{k}")
    mrr = retrieve.get("mrr")
    if hit_k is None or mrr is None:
        return ""
    when = str(quality.get("measured_at") or "")[:10]
    note = (f"측정 품질: hit@{k} {hit_k:.2f} · MRR {mrr:.2f} "
            f"(골든셋 {quality.get('cases')}건, {when})")
    if quality.get("stale"):
        keys = ", ".join(quality.get("stale_keys") or [])
        note += f" ⚠ 측정 후 설정 변경됨({keys}) — 재측정 전 참고용"
    return note


def pick_source(documents_json: Any, markers: tuple) -> Optional[str]:
    """`/documents` 실목록에서 마커 힌트로 문서 하나를 고른다.

    종전 클라이언트 부분문자열 필터(`filter_hits_by_source`)의 대체다: 임베딩
    편향으로 한 문서가 top_k 를 독식하면 다른 문서 근거가 **기아**했다(실측:
    필터 없이 5/5 가 제안서). 실목록에서 정확한 source 이름을 골라 서버측
    `retrieve?source=` 로 넘기면 기아·누수 둘 다 서버가 해결한다.

    마커 순서가 우선순위다(호출자가 의도를 표현). 같은 마커에 여럿 걸리면
    청크 많은 쪽(내용이 많은 문서). **못 찾으면 None** — 아무 문서나 고르면
    필터가 조용히 엉뚱해진다.
    """
    if not isinstance(documents_json, dict):
        return None
    docs = documents_json.get("documents")
    if not isinstance(docs, list):
        return None
    rows = [(str(d.get("source") or ""), int(d.get("chunks") or 0))
            for d in docs if isinstance(d, dict) and d.get("source")]
    for marker in markers or ():
        matched = [(src, chunks) for src, chunks in rows if marker and marker in src]
        if matched:
            matched.sort(key=lambda r: (-r[1], r[0]))
            return matched[0][0]
    return None


def parse_hits(retrieve_json: Any) -> List[Dict[str, Any]]:
    """/retrieve 응답에서 hits 리스트 추출. 형식 이상/error 면 빈 리스트.

    call_tool_http 는 실패 시 {"error":..., "_status":N} 를 돌려주므로
    (예외 없이) 여기서 흡수한다 — relay 는 graceful 해야 한다.
    """
    if not isinstance(retrieve_json, dict):
        return []
    if retrieve_json.get("error"):
        return []
    hits = retrieve_json.get("hits")
    if not isinstance(hits, list):
        return []
    return [h for h in hits if isinstance(h, dict)]


def retrieval_error(retrieve_json: Any) -> Optional[str]:
    """검색이 **실패했는가** — 실패면 사람이 읽을 사유, 아니면 None.

    `parse_hits` 의 짝이다. call_tool_http 는 예외를 던지지 않고
    `{"error":..., "_status":N}` 를 돌려주므로(graceful relay), parse_hits 가
    그걸 빈 리스트로 흡수하면 **"코퍼스에 근거가 없다"와 "온톨로지에 닿지
    못했다"가 같은 문장이 된다**. 실측(2026-08-21): 9274 콜드스타트 중 첫
    질의가 "근거를 찾지 못했습니다"를 반환했고 워밍업 후 정상 답을 냈다 —
    사용자는 둘을 구별할 수 없었다.

    히트 0개는 실패가 아니다(정상 응답). 그 구별이 이 함수의 전부다.
    dict 가 아니면 실패로 본다 — 서버는 언제나 dict 를 돌려준다.
    """
    if not isinstance(retrieve_json, dict):
        return f"검색 응답 형식 이상 ({type(retrieve_json).__name__})"
    err = retrieve_json.get("error")
    if not err:
        return None
    status = retrieve_json.get("_status")
    return f"{err} (status {status})" if status else str(err)


def no_evidence_message(namespace: str, error: Optional[str]) -> str:
    """근거 없이 끝날 때의 문장 — 실패와 부재를 **다르게** 말한다.

    부재는 사실 보고("이 코퍼스엔 없다"), 실패는 장애 보고("확인하지 못했다").
    둘을 같은 문장으로 내면 재시도해야 할 상황이 "답이 없다"로 종결된다.
    """
    if not error:
        return f"'{namespace}' 문서에서 근거를 찾지 못했습니다."
    return (f"온톨로지 검색에 닿지 못해 '{namespace}' 근거를 확인하지 "
            f"못했습니다 — {error}")


def filter_hits_by_source(hits: List[Dict[str, Any]], *substrs: str) -> List[Dict[str, Any]]:
    """source 에 substr 중 하나라도 포함된 히트만. 문서 구분용
    (예: 제안요청서 vs 제안서 취합본)."""
    if not substrs:
        return list(hits)
    out = []
    for h in hits:
        src = str(h.get("source") or "")
        if any(s and s in src for s in substrs):
            out.append(h)
    return out


def _evidence_block(hits: List[Dict[str, Any]]) -> str:
    """히트를 번호 근거 블록으로. [n] {source} §{section}: {text}"""
    lines = []
    for i, h in enumerate(hits, 1):
        src = str(h.get("source") or "?")
        sec = str(h.get("section") or "").strip()
        sec_label = f" §{sec}" if sec else ""
        text = str(h.get("text") or "").strip()
        lines.append(f"[{i}] {src}{sec_label}\n{text}")
    return "\n\n".join(lines)


_DEFAULT_INSTRUCTION = "사용자 질의에 대해 근거만으로 정확히 답하라."

_PROMPT_TEMPLATE = """당신은 문서 근거에 기반해서만 답하는 분석기입니다.

## 규칙 (반드시)
1. 아래 [근거] 발췌에 있는 내용만 사용하세요. 근거에 없는 사실을 지어내지 마세요.
2. 근거로 답할 수 없으면 "제공된 문서에서 근거를 찾지 못했습니다."라고만 답하세요.
3. 각 문장·항목 끝에 사용한 근거 번호를 [n] 형식으로 표기하세요 (여러 개면 [1][3]).
4. 한국어로, 간결하고 사실 위주로 답하세요.

## 지시
{instruction}

## 질의
{query}

## 근거
{evidence}
"""


def build_grounding_prompt(query: str, hits: List[Dict[str, Any]],
                           instruction: Optional[str] = None) -> str:
    """LLM 조립 프롬프트. 근거 발췌 + 환각 억제 규칙 + 번호 인용 강제.
    히트가 비어도 프롬프트를 만든다(규칙 2 로 '근거 없음' 유도)."""
    evidence = _evidence_block(hits) if hits else "(근거 없음)"
    return _PROMPT_TEMPLATE.format(
        instruction=(instruction or _DEFAULT_INSTRUCTION),
        query=query,
        evidence=evidence,
    )


def citations_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """히트 → 구조화 인용 (원문 추적 가능: source + char offset)."""
    cites = []
    for i, h in enumerate(hits, 1):
        cites.append({
            "n": i,
            "chunk_id": h.get("chunk_id"),
            "source": h.get("source"),
            "section": h.get("section") or "",
            "char_start": h.get("char_start"),
            "char_end": h.get("char_end"),
            "score": h.get("score"),
            "matched_via": h.get("matched_via") or [],
        })
    return cites
