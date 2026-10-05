"""중복 검수 지원 — 클러스터를 사람이 빠르게 판정할 수 있게 (순수 함수).

**왜 필요한가.** `graph_health.duplicate_clusters` 는 id 목록뿐이라 검수자가
클러스터마다 노드를 따로 조회해야 했다. 커버리지 회복이 중복을 낳는 패턴이
세 번 재현됐고(3→4, 20→31, 31→46), PROJ-A 골든셋 hit@1 −2건의 원인이기도 하다 —
검수가 밀리면 잡음이 지표를 갉는다.

**분류는 신호이지 판정이 아니다.** C73 교훈: "C73(…) 중 중증 갑상선암을 제외한
갑상선암"은 이름이 비슷하지만 **다른 개념**이었다 — 기계가 판정하면 안 되는 실물.

  · variant    — 같은 타입 + 정규화명 동일 → 표기 변형. 병합 후보(강).
                 대조 정준화(document_view)가 이미 같은 개념으로 취급하는
                 **그 기준**이다 — 병합은 그래프를 보고서와 일치시키는 일.
  · cross_type — 정규화명 동일·타입 상이 → 타입이 갈렸을 뿐 같은 말일 수도,
                 진짜 다른 것일 수도. 사람 판단.
  · similar    — 포함 등 그 외 → 사람 판단. C73 경고 동반.

variant 에만 병합 payload(suggested)를 준다 — 사람 판단 대상에 payload 를 주면
"기계가 권했다"가 된다. 승자 = 근거 많은 쪽(동률은 사전순 — 결정론). `active`
노드는 패자가 될 수 없다(생애주기 관문이 거부한다) — 미리 반영해 검수자가 계획을
다 세우고 나서 거부당하지 않게 한다.
"""

from typing import Any, Dict, Iterable, List

from loguru import logger

from .graph_health import normalize_name
from .lifecycle import ACTIVE, current_state


def _norm_name_of(attrs: Dict[str, Any], node_id: str) -> str:
    name = attrs.get("name") or node_id.split(":", 1)[-1]
    # normalize_name 은 node_id 를 받아 첫 ":" 앞을 타입으로 떼므로 가짜 접두사로
    # 이름 속 ":" 를 보호한다 (document_view 와 같은 트릭).
    return normalize_name(f"_:{name}")


def propose_duplicate_resolutions(graph, chunks: Iterable[Any],
                                  clusters: Iterable[List[str]]
                                  ) -> Dict[str, Any]:
    """클러스터 목록 → 판단 신호가 붙은 검수 제안. 실패는 빈 결과 (never raise)."""
    try:
        node_ids = set(graph.nodes())

        by_node_chunks: Dict[str, List[Any]] = {}
        for chunk in (chunks or []):
            for nid in (getattr(chunk, "node_ids", None) or []):
                by_node_chunks.setdefault(nid, []).append(chunk)

        out: List[Dict[str, Any]] = []
        by_kind: Dict[str, int] = {}
        for raw in (clusters or []):
            members_ids = sorted(nid for nid in raw if nid in node_ids)
            if len(members_ids) < 2:
                continue          # 유령 제거 후 홀로 남으면 중복이 아니다

            members: List[Dict[str, Any]] = []
            for nid in members_ids:
                attrs = graph.nodes[nid]
                evidence = by_node_chunks.get(nid, [])
                members.append({
                    "node_id": nid,
                    "type": str(attrs.get("type") or ""),
                    "name": attrs.get("name", nid.split(":", 1)[-1]),
                    "definition": str(attrs.get("definition") or "")[:160],
                    "evidence_chunks": len(evidence),
                    "sources": sorted({str(getattr(c, "source", "") or "")
                                       for c in evidence} - {""}),
                    "lifecycle": current_state(attrs),
                    "norm": _norm_name_of(attrs, nid),
                })

            types = {m["type"] for m in members}
            norms = {m["norm"] for m in members}
            if len(norms) == 1 and len(types) == 1:
                kind = "variant"
            elif len(norms) == 1:
                kind = "cross_type"
            else:
                kind = "similar"

            cluster: Dict[str, Any] = {"kind": kind, "members": [
                {k: v for k, v in m.items() if k != "norm"} for m in members]}

            if kind == "similar":
                cluster["caution"] = (
                    "이름이 비슷해도 다른 개념일 수 있다 — C73 사례"
                    "(\"…을 제외한 갑상선암\"은 제외 집합이었다). 병합 전 정의와 "
                    "원문을 대조할 것.")
            if kind == "cross_type":
                cluster["caution"] = (
                    "타입이 갈렸을 뿐 같은 말일 수도, 진짜 다른 것일 수도 있다. "
                    "어느 타입이 옳은지는 사람이 정한다.")
            if kind == "variant":
                # 승자 = 근거 많은 쪽 · active 우선(패자가 될 수 없다) ·
                # 동률은 사전순. 결정론 — 실행마다 제안이 바뀌면 검수를 재현할
                # 수 없다.
                ranked = sorted(
                    members,
                    key=lambda m: (m["lifecycle"] != ACTIVE,
                                   -m["evidence_chunks"], m["node_id"]))
                winner = ranked[0]["node_id"]
                losers = [m["node_id"] for m in ranked[1:]]
                if any(m["lifecycle"] == ACTIVE for m in ranked[1:]):
                    # 패자 중 active 가 있으면 payload 를 주지 않는다 — 관문이
                    # 거부할 계획을 제안하는 것은 검수자 시간 낭비다.
                    cluster["caution"] = ("active 노드가 둘 이상 — 병합 전에 "
                                          "생애주기 정리가 필요하다.")
                else:
                    cluster["suggested"] = {"winner": winner, "losers": losers}
            out.append(cluster)
            by_kind[kind] = by_kind.get(kind, 0) + 1

        # variant 먼저(즉시 처리 가능) → cross_type → similar. 종류 안에서는
        # 근거 합 내림차순 — 지표에 많이 걸리는 것부터.
        order = {"variant": 0, "cross_type": 1, "similar": 2}
        out.sort(key=lambda c: (order[c["kind"]],
                                -sum(m["evidence_chunks"] for m in c["members"]),
                                c["members"][0]["node_id"]))
        return {"clusters": out, "clusters_total": len(out), "by_kind": by_kind}
    except Exception as e:
        logger.warning(f"⚠️ Duplicate review proposal failed ({e})")
        return {"clusters": [], "clusters_total": 0, "by_kind": {}}
