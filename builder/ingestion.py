"""
종별 인제스트 실행기 — 확인 게이트가 승인된 plan 을 실행할 때 쓰는 조각들.

전부 결정적(LLM 무사용)이다. LLM 의 역할은 감식(sniffer)에서 제안까지고,
사용자가 확인한 뒤의 실행은 기계적이어야 한다 — 승인한 것과 다른 것이
만들어지면 확인 게이트의 의미가 없다.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


def load_records_from_file(path, records_path: str = "") -> List[Dict[str, Any]]:
    """analyze 가 찾은 경로(records_path)로 레코드 배열을 다시 꺼낸다.

    감식 시점과 실행 시점 사이에 파일이 바뀌었거나 경로가 틀리면 조용히 빈
    배열을 돌려주지 않고 ValueError — 정체성 없는 인제스트가 최악이다.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    target = data
    if records_path:
        if not isinstance(data, dict) or records_path not in data:
            raise ValueError(f"records path '{records_path}' not in {path.name}")
        target = data[records_path]
    if not isinstance(target, list) or not all(
            isinstance(item, dict) for item in target[:50]):
        raise ValueError(f"no records found at '{records_path or '<root>'}' "
                         f"in {path.name}")
    return target


async def ingest_hierarchy(kg, items: List[Dict[str, Any]],
                           spec: Dict[str, Any],
                           source: str = "") -> Dict[str, int]:
    """계층 sidecar → 클래스 노드 + is_a 간선.

    wikidata 의 {이름: 성문, 상위: 구조물} 같은 부속 배열이 재료다. 이걸
    버리면 계층 롤업 추론(실측 +554 발견)이 통째로 빠진 온톨로지가 된다.
    간선 dedup 은 KG 쓰기 레이어가 보장하므로 여기서는 세기만 한다.
    """
    child_field = spec["child_field"]
    parent_field = spec["parent_field"]
    node_type = spec["node_type"]
    graph = kg.graph

    nodes_added = 0
    edges_added = 0
    for item in items:
        child = str(item.get(child_field) or "").strip()
        parent = str(item.get(parent_field) or "").strip()
        if not child or not parent:
            continue  # 불완전 항목은 간선이 될 수 없다
        child_id = f"{node_type}:{child}"
        parent_id = f"{node_type}:{parent}"
        for node_id, name in ((child_id, child), (parent_id, parent)):
            if node_id not in graph:
                await kg.add_concept(node_id, node_type,
                                     {"name": name, "source": source})
                nodes_added += 1
        existing = graph.get_edge_data(child_id, parent_id) or {}
        already = any(a.get("predicate") == "is_a" for a in existing.values())
        await kg.add_relationship(child_id, parent_id, "is_a",
                                  {"source": source})
        if not already:
            edges_added += 1

    if edges_added:
        logger.info(f"🌳 Hierarchy ingested: +{nodes_added} class nodes, "
                    f"+{edges_added} is_a edges")
    return {"nodes": nodes_added, "edges": edges_added}


# ─── 시드 온톨로지 (JSON-LD/SKOS 계열) ──────────────────────────────

def _tail(value: Optional[str]) -> str:
    """'ko:IntentSlot' → 'IntentSlot', 'https://…#EntityClass' → 'EntityClass'.
    접두사는 네임스페이스 표기일 뿐 타입 정보는 마지막 조각에 있다."""
    if not value or not isinstance(value, str):
        return ""
    for sep in ("#", "/", ":"):
        if sep in value:
            value = value.rsplit(sep, 1)[1]
    return value.strip()


def _collect_aliases(item: Dict[str, Any]) -> List[str]:
    """examples/altLabel/alias 류 키의 문자열 배열 → aliases.

    KorAct 의 ko:examples(표면형: "결제하기", "주문하기")가 우리 aliases 가
    된다 — semantic_index 가 임베딩에 흡수하고 축 4 가 확장어로 쓴다.
    시드 import 의 존재 이유가 이 연결이다.
    """
    aliases: List[str] = []
    for key, value in item.items():
        if not isinstance(value, list):
            continue
        if not all(isinstance(x, str) for x in value):
            continue
        key_tail = _tail(key).lower()
        if any(marker in key_tail for marker in ("example", "altlabel", "alias")):
            aliases.extend(v for v in value if v and v not in aliases)
    return aliases[:8]


async def import_seed(kg, items: List[Dict[str, Any]],
                      source: str = "") -> int:
    """이미 온톨로지인 항목들(JSON-LD)을 멱등 upsert 한다 — 추출이 아니다.

    결정적 규칙만 쓴다:
    - name: prefLabel > label > @id 꼬리
    - type: @type 꼬리 (없으면 Concept)
    - definition 보존, examples/altLabel → aliases
    이름 없는 항목은 노드가 될 수 없으므로 건너뛴다.
    """
    added = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (str(item.get("prefLabel") or "").strip()
                or str(item.get("label") or "").strip()
                or _tail(item.get("@id")))
        if not name:
            continue
        node_type = _tail(item.get("@type")) or "Concept"

        attrs: Dict[str, Any] = {"name": name, "source": source}
        definition = item.get("definition")
        if definition:
            attrs["definition"] = str(definition)
        aliases = _collect_aliases(item)
        if aliases:
            attrs["aliases"] = aliases

        node_id = f"{node_type}:{name}"
        is_new = node_id not in kg.graph
        await kg.add_concept(node_id, node_type, attrs)
        if is_new:
            added += 1
    if added:
        logger.info(f"🌱 Seed ontology imported: +{added} nodes")
    return added
