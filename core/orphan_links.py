"""고아 노드 근거 회복 — "청크가 없는 노드"를 원문 인용으로 되돌린다.

**커버리지 검사와 대칭인 반쪽이다.** `coverage_checker` 는 "노드가 없는 청크"를
묻는다(추출이 놓친 개체). 아무도 **"청크가 없는 노드"**를 묻지 않았고, 그 사이
ins_cancer_demo 에 고아 노드가 27개(14%) 쌓였다. 그중 **25개는 이름이 원문에
그대로 있다** — `InsuranceTerm:보험료` 는 원문 **32곳**, `계약일` 11곳,
`계약 전 알릴 의무` 10곳. 빌더가 노드를 만들고 링크를 흘린 것이다.

**왜 검색 튜닝보다 먼저인가**: evidence 채점의 hit@5 천장 0.8125(=13/16)는 검색
knob 15개 조합 어디서도 움직이지 않았다. 정답 노드에 근거 청크가 0개면 채널 B 가
끌어올 것이 없고, "정답 노드에 연결된 청크가 top-k 에 있는가"라는 채점이
**구조적으로 불가능**하다. 천장은 검색이 아니라 연결이 정한다.

**shadow 검사가 필수 부품이다.** 단순 부분문자열 매칭은 잘못된 링크를 만든다 —
실측된 오추출 `Disease:상선암`("갑상선암"에서 앞글자가 잘린 노드)이 원문 10곳에서
"발견"되는데 전부 "갑상선암"의 부분문자열이다. 한국어는 어절 경계가 없어 정규식
단어경계(\\b)가 무의미하므로, **그래프 자신의 노드 이름**을 사전으로 쓴다: 더 긴
노드 이름이 이 이름을 포함하면서 같은 청크에 있으면 가려진 것으로 본다.
하드코딩 어휘가 아니라 데이터에서 읽는다.

가려진 후보는 버리지 않고 `shadowed` 로 **보고한다** — 오추출을 조용히 덮으면
그 결함(잘린 노드)을 영원히 못 본다.

LLM 0콜, 결정적. 이 모듈은 순수 함수만 담는다(그래프·청크를 인자로 받는다) —
쓰기는 server/service 가 한다.
"""

from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from .evidence_checker import _quote_in_chunks, _squash_ws


def node_name(node_id: str, attrs: Optional[Dict[str, Any]] = None) -> str:
    """노드의 표시 이름. name 이 없으면 id 꼬리를 쓴다 —  name 누락은 흔하고,
    그 노드를 회복 대상에서 빼면 고아로 영구히 남는다."""
    name = _squash_ws(str((attrs or {}).get("name") or ""))
    if name:
        return name
    return _squash_ws(str(node_id).split(":", 1)[-1])


def shadowing_names(name: str, all_names: Iterable[str],
                    chunk_text: str) -> List[str]:
    """`name` 의 출현을 가리는 더 긴 노드 이름들 (같은 청크 안에서).

    가림의 정의: 다른 노드 이름 M 이 `name` 을 **부분문자열로 포함**하고
    (M != name), M 이 그 청크 원문에 있다. 그러면 `name` 의 출현은 M 의 일부일
    개연성이 높다 — `상선암` ⊂ `갑상선암` 이 실측된 사례다.

    보수적으로 판단한다(가려지면 건너뛴다): 잘못된 근거 링크는 인용이 엉뚱한
    곳을 가리키게 만들어 축 2 의 계약을 깨는데, 놓친 링크는 검수자가 손으로
    이을 수 있다. 비대칭한 비용이다.

    돌려주는 것은 가린 이름들 — 검수 화면이 "왜 건너뛰었나"를 보여줘야 한다.
    """
    needle = _squash_ws(name)
    if not needle or not chunk_text:
        return []
    haystack = _squash_ws(chunk_text)
    out: List[str] = []
    for other in all_names or []:
        longer = _squash_ws(str(other or ""))
        if not longer or longer == needle:
            continue
        if needle in longer and longer in haystack:
            out.append(longer)
    return sorted(set(out))


def find_orphan_candidates(graph, chunks: Iterable[Any],
                           limit: int = 0) -> Dict[str, Any]:
    """근거 링크가 없는 노드 → 그 이름을 원문에 담은 청크 후보.

    돌려주는 세 갈래는 각각 다른 결함을 가리킨다:
      · candidates  — 이름이 원문에 있고 가려지지 않음 → **회복 가능**
      · shadowed    — 더 긴 이름의 부분문자열로만 나타남 → **오추출 의심**
      · unquotable  — 원문에 아예 없음 → **추출이 원문을 넘었다**(환각)

    `limit` 은 candidates 만 자른다. 총계를 같이 줄이면 "다 처리했다"로
    오해된다 (조용한 절단 금지 — 이 저장소의 계약).

    실패는 빈 결과 (never raise) — 진단 경로가 그래프를 죽이면 안 된다.
    """
    try:
        stored = [c for c in (chunks or []) if getattr(c, "chunk_id", None)]
        # 노드 → 근거 청크. store 를 인자로 받지 않으므로 청크에서 되짚는다.
        linked: set = set()
        for chunk in stored:
            for nid in (getattr(chunk, "node_ids", None) or []):
                linked.add(nid)

        all_names: List[str] = [node_name(nid, attrs)
                                for nid, attrs in graph.nodes(data=True)]

        candidates: List[Dict[str, Any]] = []
        shadowed: List[Dict[str, Any]] = []
        unquotable: List[str] = []
        orphans_total = 0

        # id 순으로 돈다 — 검수 화면이 새로고침마다 순서가 바뀌면 일괄 승인이
        # 위험해진다(무엇을 승인했는지 재현할 수 없다).
        for node_id in sorted(graph.nodes()):
            if node_id in linked:
                continue
            orphans_total += 1
            attrs = graph.nodes[node_id]
            name = node_name(node_id, attrs)
            if not name:
                unquotable.append(node_id)
                continue

            hits: List[Any] = []
            blockers: set = set()
            for chunk in stored:
                text = getattr(chunk, "text", "") or ""
                if not _quote_in_chunks(name, [text]):
                    continue
                blocked = shadowing_names(name, all_names, text)
                if blocked:
                    blockers.update(blocked)
                    continue
                hits.append(chunk)

            if hits:
                candidates.append({
                    "node_id": node_id,
                    "name": name,
                    "type": str(attrs.get("type") or ""),
                    "chunk_ids": [c.chunk_id for c in hits],
                    "sections": sorted({str(getattr(c, "section", "") or "")
                                        for c in hits} - {""}),
                    "occurrences": len(hits),
                })
            elif blockers:
                shadowed.append({"node_id": node_id, "name": name,
                                 "shadowed_by": sorted(blockers)})
            else:
                unquotable.append(node_id)

        return {"orphans_total": orphans_total,
                "candidates": candidates[:limit] if limit else candidates,
                "candidates_total": len(candidates),
                "shadowed": shadowed,
                "unquotable": unquotable}
    except Exception as e:                    # 진단이 그래프를 죽이면 안 된다
        logger.warning(f"⚠️ Orphan candidate scan failed ({e})")
        return {"orphans_total": 0, "candidates": [], "candidates_total": 0,
                "shadowed": [], "unquotable": []}
