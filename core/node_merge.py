"""노드 병합 계획 — 되돌릴 수 없는 변경 **전에** 결과를 계산한다 (순수 함수).

graph_health 가 중복 후보를 **제시**하고, 여기가 그 병합의 **결과를 미리 보여주며**,
service.merge_nodes 가 사람이 승인한 뒤 **적용**한다. 세 단계로 나눈 이유는
병합이 노드를 지우기 때문이다 — 팔란티어가 온톨로지 변경을 브랜치→제안→승인으로
감싸는 것과 같은 이유고, 여기서는 dry-run 이 그 제안 역할을 한다.

**설계 결정 4가지와 근거**:

1. **진 노드의 이름은 별칭으로 흡수한다.** 지우면 그 표기로 검색하던 질의를 잃는다.
   별칭은 축 4 질의 확장이 읽으므로, 중복이 **동의어로 승격**되며 회수율이 오른다.

2. **엣지는 술어까지 보고 중복 판정한다.** 그래프가 MultiDiGraph 라 (from,to) 가
   같아도 술어가 다르면 다른 사실이다 — (from,to)만 보면 사실이 조용히 사라진다.

3. **프로퍼티 충돌은 합치지 않고 보고한다.** 이긴 쪽 값을 남기되 진 쪽 값을
   충돌로 올린다(감사에도 남는다). 정의문 두 개를 이어 붙이면 원문에 없는 문장이
   생겨 환각이 된다 — 어느 쪽이 옳은지는 사람이 안다.

4. **청크·골든셋 참조도 함께 센다.** 노드만 지우면 청크의 node_ids 가 허공을
   가리키고(dangling), 골든셋 라벨은 조용히 무효가 된다 — 그러면 "지표가 나빠진
   것"과 "라벨이 깨진 것"을 구별할 수 없다.

**병합이 유일한 선택지는 아니다**: GoldenCase.accepted 는 같은 개념이 여러 노드로
갈린 경우를 정답 집합으로 허용한다(그 필드 주석이 바로 이 사례를 든다). 그래프
채널의 근거가 쪼개지는 문제까지 고치려면 병합이 맞고, 채점만 살리려면 accepted 가
가볍다 — 판단은 사람 몫이다.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .graph_health import normalize_name

# 병합 시 사람에게 물을 일이 아닌 키 — 다른 게 당연하다.
BOOKKEEPING_KEYS = frozenset({
    "created_at", "last_updated", "update_count", "chunk_index",
})
# 구조 키 — 별칭·엣지 규칙이 따로 다루므로 충돌 검사에서 뺀다.
STRUCTURAL_KEYS = frozenset({"name", "type", "aliases"})


def _alias_key(text: str) -> str:
    """별칭 비교용 키 — `납입최고 (독촉 )` 과 `납입최고(독촉)` 을 같게 본다.

    표기만 다른 별칭을 둘 다 넣으면 별칭 목록이 쓰레기가 되고, 질의 확장이
    같은 말을 두 번 확장한다.
    """
    return normalize_name(text)


def _existing_aliases(attrs: Dict[str, Any]) -> List[str]:
    raw = attrs.get("aliases") or []
    return [str(a) for a in raw if str(a).strip()]


def _edge_key(source: str, target: str, predicate: str) -> tuple:
    return (source, target, predicate)


def plan_merge(graph,
               winner: str,
               losers: Sequence[str],
               chunks: Optional[Iterable[Any]] = None,
               cases: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """병합 결과를 계산한다 — 그래프를 **바꾸지 않는다**.

    반환은 JSON 직렬화 가능하다: dry-run 응답으로 그대로 나가고, 승인 후
    적용에도 같은 구조를 쓴다 — 미리 본 것과 적용된 것이 어긋나지 않게.
    """
    if winner not in graph:
        return {"error": "winner_not_found", "detail": winner}
    loser_list = sorted({str(x) for x in (losers or ()) if str(x).strip()})
    if not loser_list:
        return {"error": "no_losers", "detail": "at least one loser required"}
    if winner in loser_list:
        return {"error": "winner_in_losers", "detail": winner}
    missing = [x for x in loser_list if x not in graph]
    if missing:
        return {"error": "loser_not_found", "detail": missing[0]}

    loser_set = set(loser_list)
    win_attrs = dict(graph.nodes[winner])

    # ── 엣지 ─────────────────────────────────────────────────────────
    # 이미 존재하는 엣지 + 이 계획에서 추가될 엣지를 함께 보아야 한다.
    # 진 노드가 둘일 때 같은 엣지를 각각 들고 있으면 두 번 추가된다.
    existing = set()
    for _, target, data in graph.out_edges(winner, data=True):
        if target not in loser_set:
            existing.add(_edge_key(winner, target, data.get("predicate", "")))
    for source, _, data in graph.in_edges(winner, data=True):
        if source not in loser_set:
            existing.add(_edge_key(source, winner, data.get("predicate", "")))

    repointed: List[Dict[str, str]] = []
    dropped: List[Dict[str, str]] = []
    planned = set(existing)

    def consider(source: str, target: str, predicate: str,
                 origin: str) -> None:
        new_source = winner if source in loser_set else source
        new_target = winner if target in loser_set else target
        record = {"from": new_source, "to": new_target,
                  "predicate": predicate, "origin": origin}
        if new_source == new_target:
            dropped.append({**record, "reason": "self_loop"})
            return
        key = _edge_key(new_source, new_target, predicate)
        if key in planned:
            dropped.append({**record, "reason": "duplicate"})
            return
        planned.add(key)
        repointed.append({"from": new_source, "to": new_target,
                          "predicate": predicate})

    for loser in loser_list:
        for _, target, data in graph.out_edges(loser, data=True):
            consider(loser, target, data.get("predicate", ""), loser)
        for source, _, data in graph.in_edges(loser, data=True):
            consider(source, loser, data.get("predicate", ""), loser)

    # ── 별칭 ─────────────────────────────────────────────────────────
    seen_keys = {_alias_key(win_attrs.get("name", "") or winner)}
    seen_keys.update(_alias_key(a) for a in _existing_aliases(win_attrs))
    seen_keys.discard("")
    aliases_added: List[str] = []
    for loser in loser_list:
        attrs = graph.nodes[loser]
        candidates = [attrs.get("name", "") or loser, *_existing_aliases(attrs)]
        for candidate in candidates:
            key = _alias_key(candidate)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            aliases_added.append(str(candidate).strip())

    # ── 프로퍼티 ─────────────────────────────────────────────────────
    conflicts: List[Dict[str, Any]] = []
    adopted: Dict[str, Any] = {}
    for loser in loser_list:
        for key, value in graph.nodes[loser].items():
            if key in BOOKKEEPING_KEYS or key in STRUCTURAL_KEYS:
                continue
            if value in (None, "", [], {}):
                continue
            current = win_attrs.get(key)
            if current in (None, "", [], {}) and key not in adopted:
                adopted[key] = value           # 이긴 쪽이 비었으면 정보 보존
            elif current not in (None, "", [], {}) and current != value:
                conflicts.append({"key": key, "winner": current,
                                  "loser": value, "loser_id": loser})

    # ── 참조 (청크 · 골든셋) ─────────────────────────────────────────
    chunk_node_ids: Dict[str, List[str]] = {}
    chunks_rewritten: List[str] = []
    for chunk in chunks or ():
        refs = [str(r) for r in (getattr(chunk, "node_ids", None) or ())]
        if not any(r in loser_set for r in refs):
            continue
        rewritten: List[str] = []
        for ref in refs:                       # 순서 보존 dedup — 병합으로
            new_ref = winner if ref in loser_set else ref   # 같은 id 가 둘 될 수 있다
            if new_ref not in rewritten:
                rewritten.append(new_ref)
        cid = str(getattr(chunk, "chunk_id", "") or "")
        chunk_node_ids[cid] = rewritten
        chunks_rewritten.append(cid)

    golden_relabels: List[str] = []
    for case in cases or ():
        ids = case.accepted_ids() if hasattr(case, "accepted_ids") else {
            getattr(case, "expected_node_id", "")}
        if ids & loser_set:
            golden_relabels.append(str(getattr(case, "case_id", "") or ""))

    return {
        "winner": winner,
        "losers": loser_list,
        "edges_repointed": sorted(repointed,
                                  key=lambda e: (e["from"], e["to"],
                                                 e["predicate"])),
        "edges_dropped": sorted(dropped,
                                key=lambda e: (e["from"], e["to"],
                                               e["predicate"], e["reason"])),
        "aliases_added": aliases_added,
        "properties_adopted": adopted,
        "property_conflicts": sorted(conflicts,
                                     key=lambda c: (c["key"], c["loser_id"])),
        "chunks_rewritten": sorted(chunks_rewritten),
        "chunk_node_ids": chunk_node_ids,
        "golden_relabels": sorted(golden_relabels),
    }
