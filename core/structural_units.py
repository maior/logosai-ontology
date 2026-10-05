"""구조 단위(조항·문서) 후보 탐지 — 로드맵 4 P-1 (2026-08-03).

**왜**: 조항(`제24조(계약의 소멸)`)·문서(`사업방법서`)가 개념과 같은 타입
(`InsuranceTerm`)으로 잡혀 있어 관계 추출이 구조적으로 헷갈린다 — 관계 백필
정답률 67%, 타입 시그니처 위반 83% 의 근본 원인 (2026-07-31 실측). 재분류는
id 개명이라 통제된 경로(P-2)가 필요하고, 그 전에 "어떤 노드가 구조 단위인가"
를 **제안**하는 것이 이 모듈이다.

**자동 판별은 측정으로 기각됐다** — 판별 규칙의 실패가 양방향으로 실측됐다:
- 포함 매칭 → 오탐: `계약자` 가 조문 제목들 속에 부분문자열로 흔하다.
- 원문 그대로 비교 → 놓침: 노드 `제24조(계약의 소멸)` vs 라벨
  `제24조 【계약의 소멸】` (괄호·공백 표기차).

그래서 규칙은 하나다: **section 라벨 전체와의 normalize 동등성만** 후보다.
부분문자열 매칭을 버리는 것이 오탐 차단의 본체이고, normalize(graph_health 의
그 함수 — '같다'의 정의를 두 벌 두지 않는다)가 놓침을 해결한다.

라벨 사전은 청크 스토어의 section 전량이다 — 조문 정규식을 여기 하드코딩하지
않는다 (segmenter 가 이미 도메인-일반으로 라벨을 만들었다). 계층 라벨(` > `)은
구획별로 사전에 들되, 여전히 구획 **전체** 동등만 인정한다.

분류는 신호이지 판정이 아니다 (dup_review 의 규율): evidence_matched·lifecycle
·술어 분포를 동봉하되 거르지 않는다. 판정은 사람이 한다 (P-3 approve 는 이
함수를 재실행해 여전히 후보인 것만 승인한다 — 규칙 두 벌 금지).

LLM 0콜, 결정적, 순수 함수 (그래프·청크를 인자로 받는다 — 쓰기 없음).
"""

from typing import Any, Collection, Dict, Iterable, List

from .graph_health import normalize_name

# 계층 section 라벨의 구획 구분자 (segmenter·coverage 지도와 같은 규약)
_SEGMENT_SEP = " > "


def _norm(text: Any) -> str:
    """라벨·이름의 비교용 형태. normalize_name 은 첫 ':' 앞을 타입 접두사로
    떼므로 가짜 접두사로 본문 속 ':' 를 보호한다 (dup_review 와 같은 트릭)."""
    return normalize_name(f"_:{text}")


def _source_title(source: Any) -> str:
    """`a/b/약관_본문.pdf` → `약관_본문`. 경로·확장자는 제목이 아니다."""
    name = str(source or "").replace("\\", "/").rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def _node_display_name(node_id: str, attrs: Dict[str, Any]) -> str:
    name = str((attrs or {}).get("name") or "").strip()
    if name:
        return name
    return str(node_id).split(":", 1)[-1]


def find_structural_candidates(
    graph,
    chunks: Iterable[Any],
    tombstoned: Collection[str] = (),
    exclude_types: Collection[str] = (),
) -> Dict[str, Any]:
    """구조 단위(조항·문서) 후보 노드 + 판단 재료.

    Returns:
        {"total", "by_kind": {"section", "document"}, "candidates": [...]}
        candidate = {node_id, name, kind, type, definition, lifecycle, caution,
                     evidence_matched, evidence_count, matched_sections,
                     matched_chunk_ids, sources, out_predicates}
    """
    from .lifecycle import ACTIVE, current_state

    chunk_list = list(chunks or [])

    # 사전 1: 정규화 section 라벨(구획 포함) → 그 라벨을 단 청크들
    sec_map: Dict[str, Dict[str, Any]] = {}
    # 사전 2: 정규화 source 제목 → 그 문서의 청크들
    src_map: Dict[str, Dict[str, Any]] = {}
    # 노드별 근거 청크 수 (매칭 여부와 무관한 전체 근거 — 판단 재료)
    evidence_of: Dict[str, List[str]] = {}

    for chunk in chunk_list:
        cid = str(getattr(chunk, "chunk_id", "") or "")
        section = str(getattr(chunk, "section", "") or "")
        source = getattr(chunk, "source", "")

        for nid in (getattr(chunk, "node_ids", None) or []):
            evidence_of.setdefault(str(nid), []).append(cid)

        labels = [section] if section else []
        if section and _SEGMENT_SEP in section:
            labels.extend(seg for seg in section.split(_SEGMENT_SEP) if seg.strip())
        for label in labels:
            key = _norm(label)
            if not key:
                continue
            entry = sec_map.setdefault(key, {"labels": set(), "chunks": []})
            entry["labels"].add(label.strip())
            entry["chunks"].append(chunk)

        title = _source_title(source)
        if title:
            key = _norm(title)
            if key:
                entry = src_map.setdefault(key, {"labels": set(), "chunks": []})
                entry["labels"].add(title)
                entry["chunks"].append(chunk)

    dead = set(tombstoned or ())
    # 이미 구조 단위 타입으로 재분류된 노드는 후보가 아니다 — 이름이 여전히
    # 라벨과 동등하므로(그게 재분류의 근거였다) 빼지 않으면 큐가 영원히
    # 마르지 않는다 (라이브 실측). 타입 집합은 호출자가 데이터(선언 감사
    # 이벤트)에서 읽어 넘긴다 — 여기 하드코딩하지 않는다.
    done_types = {str(t) for t in (exclude_types or ()) if str(t)}
    candidates: List[Dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        if node_id in dead:
            continue
        if done_types and ":" in str(node_id) \
                and str(node_id).split(":", 1)[0] in done_types:
            continue
        attrs = attrs or {}
        name = _node_display_name(node_id, attrs)
        key = _norm(name)
        if not key:
            continue

        # section 이 document 보다 구체적이다 — 양쪽 다 맞으면 section (결정적)
        if key in sec_map:
            kind, entry = "section", sec_map[key]
        elif key in src_map:
            kind, entry = "document", src_map[key]
        else:
            continue

        matched = entry["chunks"]
        matched_ids = []
        seen_ids = set()
        for c in matched:
            cid = str(getattr(c, "chunk_id", "") or "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                matched_ids.append(cid)

        linked = evidence_of.get(str(node_id), [])
        evidence_matched = any(
            str(node_id) in (getattr(c, "node_ids", None) or []) for c in matched
        )

        out_predicates: Dict[str, int] = {}
        try:
            for _, _, edata in graph.out_edges(node_id, data=True):
                pred = str((edata or {}).get("predicate") or "")
                if pred:
                    out_predicates[pred] = out_predicates.get(pred, 0) + 1
        except Exception:
            pass  # 방향 그래프가 아니어도 후보 자체는 성립한다

        state = current_state(attrs)
        candidates.append({
            "node_id": node_id,
            "name": name,
            "kind": kind,
            "type": str(node_id).split(":", 1)[0] if ":" in str(node_id) else "",
            "definition": str(attrs.get("definition") or ""),
            "lifecycle": state,
            # active 는 개명이 차단된다(생애주기 관문) — 후보에서 빼지 않고
            # 미리 알린다: 승인 단계에서야 거부되면 검수 계획이 헛돈다.
            "caution": state == ACTIVE,
            "evidence_matched": evidence_matched,
            "evidence_count": len(linked),
            "matched_sections": sorted(entry["labels"]),
            "matched_chunk_ids": matched_ids,
            "sources": sorted({
                str(getattr(c, "source", "") or "") for c in matched
                if getattr(c, "source", "")
            }),
            "out_predicates": out_predicates,
        })

    candidates.sort(key=lambda c: (c["kind"], str(c["node_id"])))
    by_kind = {"section": 0, "document": 0}
    for c in candidates:
        by_kind[c["kind"]] += 1
    return {"total": len(candidates), "by_kind": by_kind, "candidates": candidates}
