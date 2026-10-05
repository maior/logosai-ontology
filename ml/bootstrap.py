"""P0-5: kg_checkpoint 의 query_agent_mapping 을 정본 학습 데이터로 (2026-08-03).

진단(docs/rl-adoption-zero-diagnosis.md)이 확정한 사실: 정책은 randn 잡음
1000건으로 학습됐는데, 라벨은 처음부터 있었다 — kg_checkpoint.json 그래프의
`query_agent_mapping` 노드 596개. 각 노드가 **원 질의 텍스트**(query_sample /
query_samples) + selected_agent + success_rate 를 들고 있다.

여기는 파서만 산다 (순수 함수, torch 불필요). 버퍼 적재·학습은
IntelligentAgentSelector.bootstrap_from_mappings 가 한다.

규율:
- **지어내지 않는다** — success_rate 없는 레코드는 기본값을 만들지 않고 버린다.
- (query, agent) 쌍으로 dedup — 같은 라벨의 중복 적재는 가중치 조작이다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from loguru import logger

# 기본 경로: ontology/data/kg_checkpoint.json (KG 엔진의 체크포인트)
_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "kg_checkpoint.json",
)


def load_kg_mappings(path: str | None = None) -> List[Dict[str, Any]]:
    """kg_checkpoint.json → [{query, agent, success_rate}] 라벨 목록.

    query_samples 가 여러 개면 각각을 한 레코드로 편다 — 라벨의 단위는
    "이 질의에 이 에이전트가 맞았다" 한 쌍이다.
    """
    path = path or _DEFAULT_CHECKPOINT
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning(f"kg_checkpoint 없음: {path}")
        return []
    except Exception as e:
        logger.warning(f"kg_checkpoint 파싱 실패 ({type(e).__name__}): {e}")
        return []

    nodes = ((data.get("graph") or {}).get("nodes")) or []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    skipped_no_rate = 0

    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "query_agent_mapping":
            continue
        agent = str(node.get("selected_agent") or "").strip()
        if not agent:
            continue
        rate = node.get("success_rate")
        if rate is None:
            skipped_no_rate += 1  # 지어내지 않는다
            continue

        samples = node.get("query_samples") or []
        if not samples and node.get("query_sample"):
            samples = [node["query_sample"]]

        for q in samples:
            q = str(q or "").strip()
            if not q:
                continue
            key = (q, agent)
            if key in seen:
                continue
            seen.add(key)
            out.append({"query": q, "agent": agent, "success_rate": float(rate)})

    if skipped_no_rate:
        logger.info(f"load_kg_mappings: success_rate 없는 노드 {skipped_no_rate}건 제외")
    logger.info(f"load_kg_mappings: {len(out)} labels from {path}")
    return out
