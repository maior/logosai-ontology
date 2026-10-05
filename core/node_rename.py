"""노드 타입 재분류(개명) 계획 — 로드맵 4 P-2 (2026-08-03, 순수 함수).

**왜 별도 경로인가**: 이 시스템의 노드 id 는 `{type}:{name}` 이라 재분류는 곧
**id 개명**이다. `rename_type`(스키마 일괄 개명)은 id 를 바꾸지 않으므로 노드
단위 재분류에 못 쓴다. 그리고 id 가 바뀌면 엣지·청크 근거 링크·골든셋 라벨이
전부 허공을 가리키게 되므로, merge 와 같은 3단 구조(제시 → 미리보기 → 적용)와
같은 적용 순서 계약이 필요하다.

merge 와 다른 결정 하나 — **묘비를 남기지 않는다**. 재분류는 "타입이 틀렸다"지
"개체가 틀렸다"가 아니다. 옛 id 에 묘비를 남기면 재빌드에서 같은 이름의 재추출이
영구 차단되어 근거가 통째로 버려진다. 재빌드의 부활 통제는 P-4 재지도(reclassify
registry)가 맡는다 — 그 전의 퇴화는 안전하다: 부활한 옛 타입 노드는 cross_type
중복 클러스터로 보드에 잡힌다 (보이는 결함).

타깃 id 가 이미 존재하면 그것은 개명이 아니라 **병합**이다 — 조용히 합치면
merge 의 관문(별칭 흡수·충돌 보고·생애주기)을 전부 우회하게 되므로 거부하고
/nodes/merge 로 안내한다.

계산은 그래프를 바꾸지 않는다. 반환은 JSON 직렬화 가능 — dry-run 응답으로
그대로 나가고 적용도 같은 구조를 쓴다 (미리보기 == 적용).
"""

import re
from typing import Any, Dict, Iterable, List, Optional

# 타입 이름: 한 단어 (":" 는 id 구분자, 공백은 id 를 깨뜨린다)
_TYPE_OK = re.compile(r"^[A-Za-z0-9_가-힣]+$")


def plan_rename(graph,
                node_id: str,
                new_type: str,
                chunks: Optional[Iterable[Any]] = None,
                cases: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """재분류 결과를 계산한다 — 그래프를 **바꾸지 않는다**."""
    node_id = str(node_id or "").strip()
    new_type = str(new_type or "").strip()

    if node_id not in graph:
        return {"error": "node_not_found", "detail": node_id}
    if not new_type or not _TYPE_OK.match(new_type):
        return {"error": "invalid_type",
                "detail": "타입은 한 단어여야 한다 (':'·공백 불가) — id 가 "
                          f"'{{type}}:{{name}}' 형식이기 때문: {new_type!r}"}

    old_type = node_id.split(":", 1)[0] if ":" in node_id else ""
    name_part = node_id.split(":", 1)[1] if ":" in node_id else node_id
    if new_type == old_type:
        return {"error": "same_type", "detail": f"{node_id} 는 이미 {new_type}"}

    new_id = f"{new_type}:{name_part}"
    if new_id in graph:
        return {"error": "target_exists",
                "detail": f"{new_id} 가 이미 있다 — 이것은 개명이 아니라 "
                          "병합이다. /nodes/merge 를 쓰라 (별칭 흡수·충돌 "
                          "보고·생애주기 관문이 그쪽에 있다)."}

    attrs_after = dict(graph.nodes[node_id])
    attrs_after["type"] = new_type

    # ── 엣지: 전량 재지정. (from,to,predicate) 동일 평행 엣지는 merge 와
    #    같은 규칙으로 1개로 접는다 — 적용부(_repoint_edges)와 같은 dedup.
    repointed: List[Dict[str, str]] = []
    seen = set()
    edges = ([(node_id, t, d) for _, t, d in graph.out_edges(node_id, data=True)]
             + [(s, node_id, d) for s, _, d in graph.in_edges(node_id, data=True)])
    for source, target, data in edges:
        new_source = new_id if source == node_id else source
        new_target = new_id if target == node_id else target
        key = (new_source, new_target, (data or {}).get("predicate", ""))
        if key in seen:
            continue
        seen.add(key)
        repointed.append({"from": key[0], "to": key[1], "predicate": key[2]})

    # ── 참조 (청크 · 골든셋) — plan_merge 와 같은 셈법
    chunk_node_ids: Dict[str, List[str]] = {}
    chunks_rewritten: List[str] = []
    for chunk in chunks or ():
        refs = [str(r) for r in (getattr(chunk, "node_ids", None) or ())]
        if node_id not in refs:
            continue
        rewritten: List[str] = []
        for ref in refs:
            new_ref = new_id if ref == node_id else ref
            if new_ref not in rewritten:
                rewritten.append(new_ref)
        cid = str(getattr(chunk, "chunk_id", "") or "")
        chunk_node_ids[cid] = rewritten
        chunks_rewritten.append(cid)

    golden_relabels: List[str] = []
    for case in cases or ():
        ids = case.accepted_ids() if hasattr(case, "accepted_ids") else {
            getattr(case, "expected_node_id", "")}
        if node_id in ids:
            golden_relabels.append(str(getattr(case, "case_id", "") or ""))

    return {
        "node_id": node_id,
        "new_id": new_id,
        "new_type": new_type,
        "old_type": old_type,
        "name": name_part,
        "attrs_after": attrs_after,
        "edges_repointed": sorted(repointed,
                                  key=lambda e: (e["from"], e["to"],
                                                 e["predicate"])),
        "chunks_rewritten": sorted(chunks_rewritten),
        "chunk_node_ids": chunk_node_ids,
        "golden_relabels": sorted(golden_relabels),
        # compose_node_text 가 id·타입을 포함하므로 색인이 반드시 낡는다
        "reindex_required": True,
    }
