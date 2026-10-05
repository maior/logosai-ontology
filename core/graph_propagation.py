"""이분 그래프 확산 (Personalized PageRank) — 그래프-조건부 검색의 확장 규칙.

**실측 진단이 이 모듈의 존재 이유다.** ins_cancer_demo 를 개체-only 그래프로
보면 엣지 97, 평균 차수 1.00, 고립 노드 122/194(62.9%), `is_a` 엣지 0개다.
`graph_retrieval.expand` 의 확장(`is_a` 폐포 + 1-hop 인접)은 이 데이터에서
거의 아무 일도 하지 못한다 — 폐포는 술어가 없어 죽은 코드이고, 인접은 63%가
고립이라 닿지 않는다.

같은 그래프를 **노드 + 청크 이분 그래프**로 보면 고립이 7/194(3.6%), 최대
연결요소가 70 → 103 이 된다. 그래프가 성긴 것이 아니라 우리가 잘못 보고
있었다: 연결은 근거 링크(노드↔청크)가 이미 나르고 있다. 커버리지 회복이
만든 링크 123개가 그대로 다리가 된다.

문헌의 전제와 같다:
- **HippoRAG 2** (ICML'25, arXiv:2502.14802) — phrase 노드 + passage 노드를
  `contains` 엣지로 이은 이중 노드 KG 에 PPR. passage 노드의 reset 확률에
  가중치(기본 0.05)를 곱해 두 노드 종류의 영향을 조절한다.
- **LinearRAG** (2025) — 문장-개체 이분 그래프에 PPR 로 개체·passage 관련도
  합산.
- **KET-RAG** — 골격 KG + 키워드-청크 이분 그래프로 추출 비용 절감.

우리 쪽 차이: 우리는 청크를 **이미** 원문 오프셋까지 들고 있어(축 2) 확산
결과가 곧 인용 가능한 근거다. 별도 passage 스토어가 필요 없다.

설계 규정:
- **LLM 0콜, 결정적.** 검색 경로의 계약이다.
- **방향을 버린다.** `definesTerm` 이 한쪽으로만 걸려 있어도 관련성은
  양방향이다. 방향을 지키면 술어를 어느 쪽으로 적었는지라는 우연이 도달성을
  좌우한다.
- **엣지가 없으면 빈 결과.** PPR 은 엣지 없는 그래프에서 균등 분포를 낸다.
  그건 정보가 아니라 잡음이고, RRF 에 넣으면 무작위 순위가 진짜 채널을
  밀어낸다 (조용한 품질 저하).
- **캐시하지 않는다.** 286 노드에서 PPR 은 마이크로초다. 캐시는 무효화 키를
  요구하고, 개수 기반 키는 내용 변경을 놓친다 — 시맨틱 색인이 삭제된 노드를
  계속 반환했던 결함과 같은 부류다. 규모가 문제가 되면 그때 **측정하고**
  넣는다.

**미측정 상수**: `DEFAULT_DAMPING` 0.85 는 PageRank 관행값이고, 감쇠가 곧
"몇 홉까지 볼 것인가"다. 도메인 골든셋으로 측정해야 하는 값이며, 아직
측정하지 않았다 (`entry_ratio`·`RRF_K` 와 같은 처지).
"""

from typing import Any, Dict, Iterable, Mapping, Tuple

from loguru import logger

# 청크 노드 접두사. 개체 id 는 "{Type}:{이름}" 규약이라 "chunk::" 와 충돌하지
# 않는다 (타입에 콜론이 들어갈 수 없다).
CHUNK_PREFIX = "chunk::"

DEFAULT_DAMPING = 0.85
DEFAULT_MAX_ITER = 100


def build_bipartite(graph, chunks: Iterable[Any]):
    """개체 그래프 + 근거 링크 → 무방향 이분 그래프.

    개체-개체 엣지는 그대로 살리고(온톨로지가 아는 관계), 개체-청크 엣지를
    더한다(원문이 아는 공존). 청크를 지운 뒤 남은 참조(dangling)는 무시한다 —
    되살리면 그래프에 없는 개체에 질량을 준다.
    """
    import networkx as nx

    bipartite = nx.Graph()
    bipartite.add_nodes_from(graph.nodes())
    for source, target in graph.edges():
        if source != target:            # 자기 루프는 확산에 기여하지 않는다
            bipartite.add_edge(source, target)

    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None)
        if not chunk_id:
            continue
        node_ids = [nid for nid in (getattr(chunk, "node_ids", None) or [])
                    if nid in graph]
        if not node_ids:
            continue                    # 근거가 없는 청크는 다리가 아니다
        chunk_node = f"{CHUNK_PREFIX}{chunk_id}"
        for node_id in node_ids:
            bipartite.add_edge(node_id, chunk_node)
    return bipartite


def propagate(bipartite, seeds: Mapping[str, float],
              damping: float = DEFAULT_DAMPING,
              max_iter: int = DEFAULT_MAX_ITER) -> Dict[str, float]:
    """진입 노드에서 출발한 Personalized PageRank 질량.

    seeds 는 {node_id: 가중치} — 보통 진입 노드의 의미검색 점수다. 음수·0
    가중치는 버린다: 코사인은 음수가 될 수 있고, 음수 reset 확률은 PPR 의
    정의를 깬다. 그래프에 없는 seed 도 버린다 — networkx 는 예외를 던지고,
    진입 노드가 그 사이 지워졌다고 검색 전체가 죽으면 안 된다.

    실패는 빈 dict 다 (never raise). 확산은 보조 채널이고, 이것 때문에 검색이
    막히면 안 된다.
    """
    import networkx as nx

    if bipartite is None or bipartite.number_of_nodes() == 0:
        return {}
    # 엣지가 없으면 PPR 은 균등 분포 — 잡음이다. 위 docstring 참고.
    if bipartite.number_of_edges() == 0:
        return {}

    personalization = {node: float(weight)
                       for node, weight in (seeds or {}).items()
                       if node in bipartite and float(weight) > 0.0}
    if not personalization:
        return {}

    try:
        return nx.pagerank(bipartite, alpha=damping,
                           personalization=personalization,
                           max_iter=max_iter)
    except Exception as e:            # 수렴 실패 등 — 채널 하나를 잃을 뿐이다
        logger.warning(f"⚠️ Graph propagation failed ({e}) — 확산 채널 생략")
        return {}


def background_masses(bipartite, damping: float = DEFAULT_DAMPING,
                      max_iter: int = DEFAULT_MAX_ITER) -> Dict[str, float]:
    """질의-무관 배경 중요도 (personalization 없는 PageRank).

    lift 의 분모다. 질의마다 다시 계산한다 — 캐시하지 않는 이유는 모듈
    docstring 과 같다(무효화 키가 내용 변경을 놓친다). 286 노드에서 한 번 더
    도는 비용은 마이크로초이고, 규모가 문제가 되면 그때 측정하고 넣는다.
    """
    import networkx as nx

    if bipartite is None or bipartite.number_of_edges() == 0:
        return {}
    try:
        return nx.pagerank(bipartite, alpha=damping, max_iter=max_iter)
    except Exception as e:
        logger.warning(f"⚠️ Background PageRank failed ({e}) — lift 생략")
        return {}


def lift(masses: Mapping[str, float],
         background: Mapping[str, float]) -> Dict[str, float]:
    """배경 대비 배율 — PageRank 의 차수 편향 교정.

    **실측이 요구한 교정이다.** 원 질량으로 정렬하면 상위가 전부 허브였다:
    "중증 갑상선암이란?" 의 1위가 `InsuranceContract:보험계약` 이었고, 갑상선
    관련 노드는 4위 밖이었다. PPR 질량은 질의와 무관하게 고차수 노드에 쏠린다.

    배경으로 나누면 "이 질의 때문에 얼마나 올랐는가"만 남는다 — TF-IDF 의 IDF 와
    같은 발상이다(전역적으로 인기 있는 것을 할인). 실측 교정 결과:
      중증 갑상선암 → 진단확정 4.79 · C50 3.0 · 갑상선암 2.42
      유방암 보장   → C50( 유방의 악성 신생물 ) 3.95   ← 이 질의의 정답 노드

    배경에 없거나 0 인 키는 버린다 — 1.0 으로 가정하면 그 노드만 배율이
    폭등해 순위가 뒤집힌다(0 나눗셈을 상수로 가리는 것과 같다).
    """
    out: Dict[str, float] = {}
    for key, mass in (masses or {}).items():
        base = (background or {}).get(key)
        if not base or base <= 0.0:
            continue
        out[key] = mass / base
    return out


def split_masses(masses: Mapping[str, float]) -> Tuple[Dict[str, float],
                                                       Dict[str, float]]:
    """확산 질량을 (개체, 청크)로 나눈다. 청크 키는 접두사를 뗀 chunk_id 다 —
    호출자가 그걸로 store 를 조회하므로 접두사가 새면 조회가 실패한다."""
    entities: Dict[str, float] = {}
    chunks: Dict[str, float] = {}
    for key, mass in (masses or {}).items():
        if key.startswith(CHUNK_PREFIX):
            chunks[key[len(CHUNK_PREFIX):]] = mass
        else:
            entities[key] = mass
    return entities, chunks
