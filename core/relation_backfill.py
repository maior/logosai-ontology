"""관계 백필 — 온톨로지의 뼈대를 채운다 (프롬프트 + 검증, 순수 함수).

**동기(실측)**: 노드 191개에 엣지 97개, 술어는 definesTerm 58 · hasParty 15 ·
coversDisease 15 · hasCondition 8 뿐이고 **`is_a` 는 0개**다. `graph_retrieval`
의 is_a 폐포 확장(`HIERARCHY_PREDICATE`)은 이 데이터에서 죽은 코드다. 커버리지·
고아 회복으로 되살린 노드 123개는 관계가 하나도 없다 — 근거 링크는 촘촘해졌지만
(고아 2개) 노드끼리는 여전히 성기다.

**청크 단위로 묻는다.** 같은 청크를 공유하는 노드 쌍이 652개다. 쌍마다 물으면
652콜이고 각 호출이 문맥을 잃는다. 청크마다 한 번이면 92콜이고 LLM 이 조문 전체를
본다 — 빌더 `_merge` 추출과 같은 모양이다.

**새 노드를 만들지 않는다.** subject/object 가 그 청크에 연결된 노드가 아니면
버린다. 관계 백필이 노드 생성의 뒷문이 되면 커버리지 승인이 지키는 검증(원문
대조·타입 관문·묘비)을 우회한다.

**환각 차단은 인용이 한다.** 관계는 문자열이 아니라 주장이므로 `(subject,
predicate, object)` 자체를 원문에서 찾을 수 없다. 그래서 `evidence_quote` 를
요구하고 그것이 청크 원문의 부분문자열인지 검증한다 — `evidence_checker` 의
`parse_check_verdict` 와 같은 규정이다.

**술어 어휘는 데이터에서 온다** (하드코딩 금지). 그래프에 이미 있는 술어 +
`is_a` — 후자는 **코드가 아는 유일한 술어**이기 때문이다
(`graph_retrieval.HIERARCHY_PREDICATE`, KG 엔진의 폐포 API). 빈 어휘는 "전부
허용"이 아니라 "아무것도 허용 안 함"으로 읽는다: 그렇지 않으면 첫 백필이 술어를
무한정 늘린다.
"""

import json
from typing import Any, Dict, Iterable, List

from loguru import logger

from .evidence_checker import _quote_in_chunks, _squash_ws

_PROMPT = """다음 조문에서 **주어진 개체들 사이의 관계**만 뽑으세요.

## 조문 원문
{chunk}

## 개체 후보 (이 목록의 node_id 만 사용)
{nodes}

## 사용 가능한 술어 (이 목록만)
{predicates}

## 규칙 (반드시 지킬 것)
1. subject 와 object 는 위 **개체 후보 목록**의 node_id 여야 합니다.
   목록에 없는 개체나 새 술어를 만들지 마세요.
2. `evidence_quote` 에는 그 관계를 뒷받침하는 **조문 원문의 문장을 그대로**
   옮기세요. 요약하거나 고쳐 쓰면 검증에서 탈락합니다.
3. 조문이 명시하지 않은 관계는 **추측하지 마세요**. 관계가 없으면
   빈 배열을 답하세요 — 대부분의 조문은 관계가 적습니다.
4. JSON 외의 다른 텍스트를 출력하지 마세요.

## 출력 형식
{{"relations": [
  {{"subject": "...", "predicate": "...", "object": "...", "evidence_quote": "..."}}
]}}
"""


def build_relation_prompt(chunk_text: str,
                          node_views: Iterable[Dict[str, Any]],
                          predicates: Iterable[str]) -> str:
    """관계 추출 프롬프트 (순수 함수).

    후보와 술어를 **정렬**한다 — 호출 순서가 프롬프트를 바꾸면 같은 청크가
    회차마다 다른 답을 내고, 그러면 재현할 수 없다.
    """
    nodes = sorted({str(v.get("node_id", "")): v
                    for v in (node_views or [])
                    if v.get("node_id")}.items())
    node_lines = []
    for node_id, view in nodes:
        name = _squash_ws(str(view.get("name") or ""))
        node_lines.append(f"- {node_id}" + (f' ("{name}")' if name else ""))
    return _PROMPT.format(
        chunk=chunk_text or "",
        nodes="\n".join(node_lines) if node_lines else "(없음)",
        predicates=", ".join(sorted({str(p) for p in (predicates or []) if p})
                             ) or "(없음)")


def parse_relations(raw: str, chunk_text: str, node_ids: set,
                    predicates: Iterable[str]) -> List[Dict[str, Any]]:
    """LLM 출력 검증 — 통과한 관계만 (순수 함수, 절대 raise 안 함).

    관문 다섯 개. 하나라도 빼면 그래프에 거짓이 들어간다:
      · subject/object 가 그 청크의 노드여야 한다 (새 노드 뒷문 차단)
      · 술어가 허용 어휘여야 한다 (스키마 무한 팽창 차단)
      · subject != object (자기 루프는 관계가 아니다)
      · evidence_quote 가 청크 원문의 부분문자열이어야 한다 (환각 차단)
      · 같은 트리플 중복 제거 — 단, **쌍이 아니라 트리플** 기준이다.
        쌍으로 묶으면 같은 두 개체 사이의 다른 관계가 사라진다.

    `names_in_quote` 는 인용 안에 양끝 이름이 몇 개 들어 있는지다 —
    **판정에 쓰지 않는다**(대명사로 지시하는 문장을 버리지 않기 위해). 검수자가
    강한 근거와 약한 근거를 가리는 데 쓴다.
    """
    allowed = {str(p) for p in (predicates or []) if p}
    if not allowed or not node_ids:
        # 빈 어휘를 "전부 허용"으로 읽으면 첫 백필이 술어를 무한정 늘린다.
        return []
    try:
        from ..builder.extractor import parse_llm_json
        parsed = parse_llm_json(raw)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    items = parsed.get("relations")
    if not isinstance(items, list):
        return []

    haystack = _squash_ws(chunk_text or "")
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        subject = _squash_ws(str(item.get("subject") or ""))
        obj = _squash_ws(str(item.get("object") or ""))
        predicate = _squash_ws(str(item.get("predicate") or ""))
        quote = str(item.get("evidence_quote") or "")
        if subject not in node_ids or obj not in node_ids:
            continue
        if predicate not in allowed or subject == obj:
            continue
        if not _quote_in_chunks(quote, [chunk_text or ""]):
            continue
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        needle = _squash_ws(quote)
        names = 0
        for nid in (subject, obj):
            tail = nid.split(":", 1)[-1].strip()
            if tail and _squash_ws(tail) in needle:
                names += 1
        out.append({"subject": subject, "predicate": predicate,
                    "object": obj, "evidence_quote": quote.strip(),
                    "names_in_quote": names})
    return out


def type_signatures(graph) -> set:
    """그래프에 실제로 쓰인 `(주어 타입, 술어, 목적어 타입)` 집합.

    **거부 관문이 아니라 검수 신호다.** 실측에서 제안 36건 중 **30건(83%)이
    기존 시그니처를 벗어났는데 그중 다수는 정상 관계**였다 — 조항
    (`제3조 【…】`)과 문서(`사업방법서`)가 `InsuranceTerm` 으로 잡혀 있어서
    "조항이 용어를 정의한다"는 옳은 관계가 타입상 새 패턴으로 보인다. 관문으로
    쓰면 그 정상 관계까지 막힌다.

    그런데 **진짜 술어 오용도 여기서 드러난다**: `Disease -coversDisease->
    Disease` 3건은 전이 관계("C50 이 폐로 전이되어 C78.0 로 진단확정된 경우에도")를
    보장 관계로 왜곡한 것이다. 기존 coversDisease 는 예외 없이
    `InsuranceContract → Disease` 였다.

    그래서 플래그로 보고하고 판단은 사람이 한다 — 새 패턴인가 오용인가는
    타입 체계가 부실한 이 그래프에서 결정적으로 가릴 수 없다.
    """
    try:
        out = set()
        for source, target, attrs in graph.edges(data=True):
            predicate = str(attrs.get("predicate", ""))
            if not predicate:
                continue
            out.add((str(graph.nodes[source].get("type", "")), predicate,
                     str(graph.nodes[target].get("type", ""))))
        return out
    except Exception as e:
        logger.warning(f"⚠️ Type signature scan failed ({e})")
        return set()


def signature_of(graph, subject: str, predicate: str, obj: str) -> tuple:
    """한 트리플의 타입 시그니처. 없는 노드는 빈 타입이 된다."""
    def _t(node_id):
        try:
            return str(graph.nodes[node_id].get("type", "")) \
                if node_id in graph else ""
        except Exception:
            return ""
    return (_t(subject), str(predicate), _t(obj))


def allowed_predicates(graph,
                       extra: Iterable[str] = ("is_a", "sameAs")) -> List[str]:
    """허용 술어 = 그래프에 이미 있는 것 + `extra`.

    하드코딩 어휘를 만들지 않는다 — 술어는 데이터에서 읽는다. 예외는 **코드가
    아는 두 술어**뿐이다(검색 확장이 이 둘에만 특별한 의미를 준다):
      · `is_a`   — graph_retrieval.HIERARCHY_PREDICATE, KG 의 전이 폐포 API.
                   실측 그래프에 0개라 그 확장 경로가 죽어 있었다.
      · `sameAs` — graph_retrieval.SYNONYM_PREDICATE, 대칭 1-hop 확장.
                   **어휘에 없어서 생긴 오류가 측정됐다**: is_a 제안 11건 중 3건이
                   동일시("…와 같습니다")를 가장 가까운 is_a 로 왜곡했다.
                   어휘가 부족하면 LLM 은 거부하지 않고 **왜곡한다**.
    """
    try:
        found = {str(attrs.get("predicate", ""))
                 for _, _, attrs in graph.edges(data=True)
                 if attrs.get("predicate")}
    except Exception as e:
        logger.warning(f"⚠️ Predicate scan failed ({e})")
        found = set()
    found.update(str(p) for p in (extra or []) if p)
    return sorted(found)
