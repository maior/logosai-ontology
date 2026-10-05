"""재적재에서 끊긴 근거 링크 복원 — 고아 노드가 **쌓이는 원인**을 막는다.

**근본 원인 (실측)**: `builder.pipeline._build_text_into` 는 재적재 시
`delete_by_source` 로 그 소스의 청크를 전량 삭제하지만 **그래프 노드는 남긴다**.
그 선택 자체는 옳다 — 노드는 여러 소스에 걸칠 수 있고 검수 판정(확정/묘비)도
붙어 있어서 문서 한 번 재업로드로 지울 수 없다. 문제는 LLM 추출이 비결정적이라
재적재에서 **이번엔 안 뽑힌 노드**가 생기고, 그 노드가 그래프에 남은 채 근거를
잃는다는 것이다.

그렇게 ins_cancer_demo 에 고아 노드가 27개(14%) 쌓였고, 그것이 evidence 채점의
hit@5 천장(검색 knob 15개 조합으로도 못 넘던 0.8125)을 정하고 있었다.

**복원의 열쇠는 노드가 이미 갖고 있다**: `_merge` 가 노드 attrs 에
`source` + `chunk_index` 를 남긴다 — "어느 청크에서 나왔는지"의 기록이다.
실제로 고아 2건이 `chunk_index=21/91` 을 들고 있으면서 링크가 없었다.

**좁게 자동, 넓게 검수** — 이 분리가 설계의 핵심이다:
  · 빌더(자동): `chunk_index` 힌트로 **원래 그 청크**만 복원한다. 정확하고 좁다.
  · 검수(사람): 이름이 나오는 **모든** 청크로 확장한다(`orphan_links`).
자동 경로가 넓게 이으면 무인 실행에서 근거 사슬이 느슨해진다 — `보험료` 를 그것이
언급된 32개 조문에 다 잇는 것은 검색에는 도움이 되지만 "어디서 추출됐나"라는
provenance 는 아니다. 그 판단은 사람이 한다.

LLM 0콜, 결정적.
"""

from typing import Any, Optional

from loguru import logger

from .evidence_checker import _quote_in_chunks
from .orphan_links import node_name


def relink_by_chunk_index(graph, store, source: str) -> int:
    """`source` 를 재적재한 뒤, 링크를 잃은 그 소스의 노드를 원래 청크에 다시 잇는다.

    이은 링크 수를 돌려준다.

    `chunk_index` 는 **힌트일 뿐**이므로 원문 대조를 반드시 통과시킨다 — 문서
    내용이 바뀌면 같은 index 가 다른 텍스트이고, 그때 이으면 근거가 엉뚱한 조문을
    가리켜 축 2 의 계약(인용 가능성)이 깨진다.

    `source` 가 비면 아무것도 하지 않는다: 전 그래프를 훑으면 이번 재적재와
    무관한 문서의 노드까지 건드린다.

    실패는 0 (never raise) — 복원이 빌드를 죽이면 안 된다. 추출은 이미 끝났고
    LLM 비용도 지불했다.
    """
    if graph is None or store is None or not (source or "").strip():
        return 0
    try:
        by_index = {}
        for chunk in store.all():
            if getattr(chunk, "source", "") == source:
                by_index[getattr(chunk, "index", None)] = chunk

        relinked = 0
        for node_id, attrs in list(graph.nodes(data=True)):
            if attrs.get("source") != source:
                continue
            index = attrs.get("chunk_index")
            if index is None:
                continue          # 빌더가 만든 노드가 아니다 — 추측하지 않는다
            chunk = by_index.get(index)
            if chunk is None:
                continue          # 문서가 짧아져 그 청크가 사라졌다
            if node_id in (getattr(chunk, "node_ids", None) or []):
                continue          # 이미 이어져 있다
            name = node_name(node_id, attrs)
            if not name or not _quote_in_chunks(name, [chunk.text or ""]):
                continue          # 내용이 바뀌었다 — 힌트를 신뢰하지 않는다
            if store.link_node(chunk.chunk_id, node_id):
                relinked += 1
        if relinked:
            logger.info(f"🔗 재적재 복원: 근거 링크 {relinked}개 되살림 ({source})")
        return relinked
    except Exception as e:
        logger.warning(f"⚠️ Relink after re-ingest failed ({source}): {e}")
        return 0
