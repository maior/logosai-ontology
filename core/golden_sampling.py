"""골든셋 노드 샘플링 — **자가 코퍼스를 대표하게** 만든다 (순수 함수).

**실측이 이 모듈을 요구했다.** PROJ-A 에서 커버리지를 18.5% → 25.2% 로 올리고
(노드 +202, 근거 링크 +218) 재측정했는데 **지표가 소수점까지 불변**이었다.

원인은 검색이 아니라 **자**였다. `generate_golden_cases` 가 노드를 삽입 순서
상위 N개로 골랐다(`for node_id, attrs in graph.nodes(data=True)` → `break`).
그래서 31 케이스가 상위 25 노드에서 나왔고, 새로 만든 202개와 회복한 청크 30개는
그 영역이 아니었다.

    네임스페이스        청크    케이스   대표성
    ─────────────────────────────────────────────
    ins_cancer_demo      92      46     사실상 전 조문
    PROJ-A               432      31     상위 25 노드에 국소

지표 불변은 "개선이 없었다"가 아니라 **"재지 못했다"** 였다. 이 세션에서 반복된
교훈("자가 없으면 결론이 뒤집힌다")의 한 겹 깊은 형태다 — **자가 있어도 좁으면
못 잰다.**

설계 규정:
- **서로 다른 청크에서 라운드로빈**으로 뽑는다 → N 케이스가 N 개 청크를 덮는다.
- **결정적이다**(무작위 아님). 같은 그래프에서 두 번 뽑으면 같아야 재현 가능한
  측정이 된다 — 이 저장소가 확산 순위·RRF 동점에서 지켜온 것과 같은 계약.
- `definition` 필수 (기존 계약) — 패러프레이즈의 재료가 없으면 이름을 안 쓰고
  물을 수 없다.
- **고아 노드도 포함하되 뒤에 둔다** — 근거 청크가 없어도 노드 채점의 대상이다.
  빼면 그 영역을 영원히 못 잰다.
"""

from typing import Any, Dict, Iterable, List

from loguru import logger


def _view(node_id: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {"node_id": node_id,
            "name": attrs.get("name", node_id),
            "type": attrs.get("type", ""),
            "definition": attrs.get("definition", "")}


def sample_nodes_for_generation(graph, chunks: Iterable[Any],
                                limit: int = 0) -> List[Dict[str, Any]]:
    """생성 대상 노드를 **청크에 퍼지게** 고른다.

    `limit=0` 은 전부. 실패는 빈 목록 (never raise) — 샘플링이 생성을 죽이면
    안 된다.
    """
    try:
        eligible = {node_id: attrs for node_id, attrs in graph.nodes(data=True)
                    if attrs.get("definition")}
        if not eligible:
            return []

        # 청크 → 그 청크가 근거인 (자격 있는) 노드. 청크 id 순으로 고정해
        # 결정론성을 보장한다.
        buckets: List[List[str]] = []
        seen_nodes: set = set()
        for chunk in sorted((c for c in (chunks or [])
                             if getattr(c, "chunk_id", None)),
                            key=lambda c: c.chunk_id):
            members = [n for n in (getattr(chunk, "node_ids", None) or [])
                       if n in eligible]
            if members:
                buckets.append(sorted(members))

        picked: List[str] = []
        # 라운드로빈 — 한 바퀴에 청크당 하나씩. 청크 수보다 많이 뽑아야 하면
        # 두 번째 바퀴로 넘어간다.
        depth = 0
        while buckets:
            progressed = False
            for bucket in buckets:
                if depth >= len(bucket):
                    continue
                node_id = bucket[depth]
                progressed = True
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                picked.append(node_id)
                if limit and len(picked) >= limit:
                    return [_view(n, eligible[n]) for n in picked]
            if not progressed:
                break
            depth += 1

        # 근거 청크가 없는 노드 — 뒤에 붙인다. 노드 채점의 대상이므로 빼면
        # 그 영역을 영원히 못 잰다.
        for node_id in sorted(eligible):
            if node_id in seen_nodes:
                continue
            picked.append(node_id)
            if limit and len(picked) >= limit:
                break
        return [_view(n, eligible[n]) for n in picked]
    except Exception as e:
        logger.warning(f"⚠️ 골든셋 샘플링 실패 ({e})")
        return []
