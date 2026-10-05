"""GraphStore — 인스턴스 읽기의 단일 seam (축 5, P1).

service 가 KG 엔진의 `.graph`(NetworkX)를 직접 순회하던 것을 이 인터페이스
뒤로 숨긴다. 두 구현이 **동일 계약**을 만족한다:

  · InMemoryGraphStore  — 현재 동작(NetworkX 스캔). 기본이자 폴백.
  · PostgresGraphStore  — 인덱스 질의(ontology.node/edge). 진실 원본.

선택: env ONTOLOGY_GRAPH_BACKEND=memory|postgres. postgres 인데 불가하면
memory 로 degrade(ES/PG 백엔드와 동일한 계약 — 조용한 실패 금지, WARNING 남김).

**계약**: 세 읽기 메서드(list_nodes/list_edges/list_neighbors)의 반환 dict 는
두 구현이 키·형태 동일해야 한다 — 관리 콘솔·Graph3D 는 백엔드를 모른 채 같은
payload 를 먹는다. tests/test_graph_store.py 가 이를 지킨다.

커널 경계: 이 모듈은 stdlib + loguru 만 모듈 레벨 import. psycopg(pg.py 경유)와
engines(NetworkX)는 팩토리 안에서 lazy import 한다.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_COUNT_CAP = 10000


def _backend_choice() -> str:
    choice = (os.environ.get("ONTOLOGY_GRAPH_BACKEND", "memory") or "memory").strip().lower()
    return "postgres" if choice == "postgres" else "memory"


def _pg_namespaces() -> set:
    """PG 로 읽는 네임스페이스 허용목록(env ONTOLOGY_PG_NAMESPACES, 쉼표구분).

    전역 플래그를 켜지 않고 **네임스페이스 단위로 점진 이전**하기 위한 통제 —
    demo_ko 하나만 PG 로 돌리고 나머지는 memory 로 두는 식."""
    raw = os.environ.get("ONTOLOGY_PG_NAMESPACES", "") or ""
    return {n.strip() for n in raw.split(",") if n.strip()}


def pg_backed(namespace: str) -> bool:
    """이 네임스페이스가 PG 로 서빙되는가 — 전역 postgres 이거나 허용목록에 포함."""
    return _backend_choice() == "postgres" or namespace in _pg_namespaces()


# 컬럼으로 승격되는 노드/엣지 attrs — 나머지는 properties(jsonb)로 간다.
_NODE_COLS = ("type", "name", "trust")
_EDGE_COLS = ("predicate", "weight")


def _content_hash(*parts) -> str:
    import hashlib
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def sync_from_graph(namespace: str, graph, schema: Optional[str] = None) -> Dict[str, int]:
    """KG(NetworkX) 그래프를 PG 로 미러 — **증분**(P4-b): content_hash 로 diff 를
    내어 변경/신규만 upsert, 사라진 것만 delete 한다. 미변경 행은 쓰기 0 —
    수백만 규모에서 매 빌드 전체 재기록을 피한다(이전엔 delete-all + insert-all).

    node attrs 는 type/name/trust=컬럼, 나머지=properties(jsonb). content_hash 는
    (type,name,trust,properties) / (weight,properties) 로 계산 — 저장 시 기록하고
    다음 동기화에서 같은 방식으로 재계산해 비교하므로 jsonb 재파싱이 불필요하다.

    반환: {"nodes": n, "edges": m} (최종 총계 — 기존 계약 유지). diff 는 로그로."""
    import json
    from . import pg
    schema = schema or pg.get_schema()

    # 원하는 상태(그래프) — id → (type,name,trust,props_json,hash)
    g_nodes: Dict[str, tuple] = {}
    for nid, attrs in graph.nodes(data=True):
        ntype = attrs.get("type", "") or "unknown"
        name = attrs.get("name")
        trust = attrs.get("trust") or "unset"
        pj = json.dumps({k: v for k, v in attrs.items() if k not in _NODE_COLS},
                        ensure_ascii=False, sort_keys=True)
        g_nodes[nid] = (ntype, name, trust, pj, _content_hash(ntype, name, trust, pj))

    g_edges: Dict[tuple, tuple] = {}   # (src,pred,tgt) → (weight,props_json,hash)
    for s, t, attrs in graph.edges(data=True):
        key = (s, attrs.get("predicate", "") or "", t)
        if key in g_edges:
            continue
        w = float(attrs.get("weight", 1.0) or 1.0)
        pj = json.dumps({k: v for k, v in attrs.items() if k not in _EDGE_COLS},
                        ensure_ascii=False, sort_keys=True)
        g_edges[key] = (w, pj, _content_hash(w, pj))

    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT node_id, content_hash FROM {schema}.node WHERE namespace=%s",
                        (namespace,))
            pg_nodes = dict(cur.fetchall())
            cur.execute(f"SELECT source_id, predicate, target_id, content_hash "
                        f"FROM {schema}.edge WHERE namespace=%s", (namespace,))
            pg_edges = {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}

            node_del = [nid for nid in pg_nodes if nid not in g_nodes]
            node_ups = [(nid, *v) for nid, v in g_nodes.items()
                        if pg_nodes.get(nid) != v[4]]
            edge_del = [k for k in pg_edges if k not in g_edges]
            edge_ups = [(k, v) for k, v in g_edges.items() if pg_edges.get(k) != v[2]]

            if node_del:
                cur.executemany(f"DELETE FROM {schema}.node WHERE namespace=%s AND node_id=%s",
                                [(namespace, nid) for nid in node_del])
                # 삭제 노드에 달린 잔여 엣지 정리(그래프에서 이미 빠졌으면 edge_del
                # 이 잡지만, dangling 방지의 안전망)
                cur.executemany(f"DELETE FROM {schema}.edge WHERE namespace=%s "
                                f"AND (source_id=%s OR target_id=%s)",
                                [(namespace, nid, nid) for nid in node_del])
            if node_ups:
                cur.executemany(
                    f"INSERT INTO {schema}.node"
                    f"(namespace,node_id,type,name,trust,properties,content_hash,updated_at)"
                    f" VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,now()) "
                    f"ON CONFLICT (namespace,node_id) DO UPDATE SET "
                    f"type=EXCLUDED.type,name=EXCLUDED.name,trust=EXCLUDED.trust,"
                    f"properties=EXCLUDED.properties,content_hash=EXCLUDED.content_hash,"
                    f"updated_at=now()",
                    [(namespace, nid, ntype, name, trust, pj, h)
                     for (nid, ntype, name, trust, pj, h) in node_ups])
            if edge_del:
                cur.executemany(f"DELETE FROM {schema}.edge WHERE namespace=%s AND "
                                f"source_id=%s AND predicate=%s AND target_id=%s",
                                [(namespace, *k) for k in edge_del])
            if edge_ups:
                cur.executemany(
                    f"INSERT INTO {schema}.edge"
                    f"(namespace,source_id,predicate,target_id,weight,properties,content_hash)"
                    f" VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s) "
                    f"ON CONFLICT (namespace,source_id,predicate,target_id) DO UPDATE SET "
                    f"weight=EXCLUDED.weight,properties=EXCLUDED.properties,"
                    f"content_hash=EXCLUDED.content_hash",
                    [(namespace, k[0], k[1], k[2], v[0], v[1], v[2]) for (k, v) in edge_ups])
        conn.commit()

    logger.info(f"🗄️ PG 증분 미러: {namespace} — 노드 ~{len(node_ups)}/-{len(node_del)}, "
                f"엣지 ~{len(edge_ups)}/-{len(edge_del)} (총 {len(g_nodes)}/{len(g_edges)})")
    return {"nodes": len(g_nodes), "edges": len(g_edges)}


def aicoach_source(namespace: str) -> Optional[str]:
    """이 네임스페이스가 외부 aicoach 스키마의 라이브 KG 를 직접 소비하는가.

    env `ONTOLOGY_AICOACH_NAMESPACE`(기본 없음) 로 매핑되는 단일 네임스페이스를
    aicoach 스키마(`ONTOLOGY_AICOACH_SCHEMA`, 기본 aicoach)의 kg_node/kg_edge 에
    직접 붙인다 — 복사본이 아니라 aicoach 라이브. 설정 없으면 None(비활성).
    같은 PG(ONTOLOGY_PG_DSN)이므로 별도 접속 불필요."""
    mapped = (os.environ.get("ONTOLOGY_AICOACH_NAMESPACE", "") or "").strip()
    if mapped and namespace == mapped:
        return (os.environ.get("ONTOLOGY_AICOACH_SCHEMA", "") or "aicoach").strip()
    return None


def hydrate_graph_aicoach(namespace: str, graph, schema: str = "aicoach") -> Dict[str, int]:
    """aicoach 스키마(kg_node/kg_edge) → NetworkX 하이드레이트 — 라이브 직접 소비.

    aicoach 컬럼 규약 매핑: kg_node(id,type,label,attrs,source) →
    node(id, type, name=label, **attrs, source), kg_edge(subject,predicate,object)
    → edge. namespace 컬럼 없음(스키마 전체가 한 KG). ONTOLOGY_PG_DSN 커넥션으로
    schema-qualified 읽기."""
    import json
    from . import pg
    if not schema.isidentifier():
        raise ValueError(f"invalid aicoach schema: {schema!r}")

    def _props(p):
        if isinstance(p, str):
            try:
                return json.loads(p)
            except Exception:
                return {}
        return dict(p or {})

    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id,type,label,attrs,source FROM {schema}.kg_node")
            for nid, ntype, label, attrs, source in cur.fetchall():
                a = _props(attrs)
                a["type"] = ntype or "unknown"
                if label is not None:
                    a["name"] = label
                if source:
                    a.setdefault("source", source)
                graph.add_node(nid, **a)
            cur.execute(f"SELECT subject,predicate,object FROM {schema}.kg_edge")
            for subj, pred, obj in cur.fetchall():
                graph.add_edge(subj, obj, predicate=pred or "")
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}


def hydrate_graph(namespace: str, graph, schema: Optional[str] = None) -> Dict[str, int]:
    """PG → NetworkX 그래프 하이드레이트 (sync_from_graph 의 역).

    pg_backed 네임스페이스의 엔진 로드 경로에서 JSON 대신 이걸 쓴다 — PG 가
    진실이므로. node attrs = {type,name,trust} 컬럼 + properties(jsonb),
    edge attrs = {predicate,weight} + properties. properties 는 psycopg 가
    dict 로 주면 그대로, str 이면 파싱. graph 는 비어 있는 상태로 넘어온다."""
    import json
    from . import pg
    schema = schema or pg.get_schema()

    def _props(p):
        if isinstance(p, str):
            try:
                return json.loads(p)
            except Exception:
                return {}
        return dict(p or {})

    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT node_id,type,name,trust,properties "
                        f"FROM {schema}.node WHERE namespace=%s", (namespace,))
            for node_id, ntype, name, trust, props in cur.fetchall():
                attrs = _props(props)
                attrs["type"] = ntype or "unknown"
                if name is not None:
                    attrs["name"] = name
                if trust and trust != "unset":
                    attrs["trust"] = trust
                graph.add_node(node_id, **attrs)
            cur.execute(f"SELECT source_id,predicate,target_id,weight,properties "
                        f"FROM {schema}.edge WHERE namespace=%s", (namespace,))
            for s, pred, t, w, props in cur.fetchall():
                attrs = _props(props)
                attrs["predicate"] = pred or ""
                if w is not None:
                    attrs["weight"] = float(w)
                graph.add_edge(s, t, **attrs)
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}


class GraphStore:
    """읽기 계약. 하위 구현이 세 메서드를 채운다."""

    def list_nodes(self, namespace, q=None, node_type=None, trust=None,
                   prop=None, kind=None, offset=0, limit=50) -> Dict[str, Any]:
        raise NotImplementedError

    def list_edges(self, namespace, predicate=None, source_type=None,
                   target_type=None) -> Dict[str, Any]:
        raise NotImplementedError

    def list_neighbors(self, namespace, node_id, limit=60) -> Dict[str, Any]:
        raise NotImplementedError

    def node_detail(self, namespace, node_id) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def aggregate(self, namespace) -> Dict[str, Any]:
        """개요/상세용 집계 — {nodes, edges, trust:{}}. 전체 그래프 로드 없이."""
        raise NotImplementedError

    def distributions(self, namespace) -> Dict[str, Any]:
        """타입/술어 분포 — {node_types:{}, predicates:{}} (스키마 뷰용)."""
        raise NotImplementedError

    def metaclass_types(self, namespace) -> set:
        """'컨테이너 타입'(=메타클래스) 집합 — **데이터 기반**(하드코딩 없음).
        정의: 타입 T 의 멤버 중 **과반(>50%)**의 이름이 다른 노드의 `type` 으로도
        쓰이면 T 는 메타클래스다. 예) HeritageClass 멤버 대부분(사원·삼층석탑…)이
        타입명이므로 → HeritageClass 가 메타클래스 → type==HeritageClass 노드는
        전부 '클래스'(빈 클래스 '3차 자료' 포함).

        기준: 타입 T 의 멤버 이름들이 **전체 타입집합의 과반(>50%)을 이름 짓는가**
        (=클래스들의 클래스). HeritageClass 멤버 이름이 거의 모든 타입명(사원·
        삼층석탑…)이므로 → 메타클래스. 빈 클래스가 많아도 성립(비율이 아니라 '얼마나
        많은 타입을 이름 짓나'로 보므로). 반대로 "삼층석탑" 타입은 자기참조 하나만
        타입명이라 커버리지가 미미 → 아님 → '거돈사지 삼층석탑' 인스턴스가 클래스로
        오분류되지 않는다. (단일 노드 판정은 name∈타입집합 을 함께 본다 —
        service._mark_classes.)"""
        raise NotImplementedError

    def review_counts(self, namespace, confirmed_ids, rejected_ids) -> Dict[str, int]:
        """검수 현황 — {confirmed, rejected, pending}. confirmed/rejected 집합은
        review_store 가 준다(작다). pending = source 있는 미판정 노드 수."""
        raise NotImplementedError

    def query_nodes(self, namespace, node_type=None, trust=None,
                    prop_key=None, prop_value=None, prop_op="eq",
                    rel_predicate=None, rel_target=None, rel_target_type=None,
                    rel_direction="out", offset=0, limit=50) -> Dict[str, Any]:
        """구조화 검색 — 프로퍼티 값 조건 + 관계 제약으로 노드를 찾는다.

        · prop_op: eq(값 일치) | contains(부분일치) | exists(값 있음)
        · rel_*: rel_direction(out/in) 방향으로 rel_predicate 술어의 엣지가
          rel_target(특정 노드) 또는 rel_target_type(그 타입 노드)로 존재하는 노드.
          예) "N 이 경주에 located_in" = N 의 out 엣지 → rel_predicate=located_in,
              rel_target=Region:경주, rel_direction=out. (반대로 "경주로 들어오는
              엣지를 가진 노드"를 원하면 rel_direction=in.)
        반환 모양은 list_nodes 와 동일(items: node_id/name/type/trust/degree)."""
        raise NotImplementedError


# ─── 인메모리(NetworkX) — 현재 동작을 그대로 옮긴다 ────────────────────────
class InMemoryGraphStore(GraphStore):
    """NetworkX MultiDiGraph 위의 읽기. service 에 있던 로직을 **동작 보존**으로
    이관한 것 — 근사 카운트(COUNT_CAP)도 그대로다(인메모리에선 진짜 카운트가
    전체 스캔이므로)."""

    def __init__(self, graph, count_cap: int = DEFAULT_COUNT_CAP):
        self.graph = graph
        self.count_cap = count_cap

    def list_nodes(self, namespace, q=None, node_type=None, trust=None,
                   prop=None, kind=None, offset=0, limit=50) -> Dict[str, Any]:
        graph = self.graph
        needle = (q or "").strip().lower()
        # kind(class|instance) 필터용 클래스 판정 준비 — service._mark_classes 와
        # 같은 규칙(이름이 타입으로 쓰임 OR 타입이 메타클래스).
        want_kind = kind in ("class", "instance")
        types = meta = None
        if want_kind:
            types = {a.get("type", "unknown") for _, a in graph.nodes(data=True)}
            meta = self.metaclass_types(namespace)
        matched = []
        capped = False
        for node_id, attrs in graph.nodes(data=True):
            if node_type and attrs.get("type", "") != node_type:
                continue
            if want_kind:
                is_cls = ((attrs.get("name") in types)
                          or (attrs.get("type", "unknown") in meta))
                if (kind == "class") != is_cls:
                    continue
            node_trust = attrs.get("trust", "") or "unset"
            if trust and node_trust != trust:
                continue
            if prop is not None and attrs.get(prop) in (None, "", []):
                continue
            if needle:
                # 이름·별칭만(node_id 제외) — node_id 의 "{type}:" 접두가 타입명
                # 검색을 인스턴스 전체 매칭으로 새게 하던 것을 막고 ES 와 통일.
                haystack = " ".join(
                    [str(attrs.get("name", ""))]
                    + [str(a) for a in attrs.get("aliases") or []]).lower()
                if needle not in haystack:
                    continue
            matched.append((node_id, attrs))
            if len(matched) >= self.count_cap:
                capped = True
                break

        matched.sort(key=lambda pair: str(pair[1].get("name", pair[0])))
        window = matched[offset:offset + limit]
        items = []
        for node_id, attrs in window:
            item = {
                "node_id": node_id,
                "name": attrs.get("name", node_id),
                "type": attrs.get("type", ""),
                "trust": attrs.get("trust", "") or "unset",
                "source": attrs.get("source", ""),
                "definition": attrs.get("definition", ""),
                "aliases": attrs.get("aliases") or [],
                "out_degree": graph.out_degree(node_id),
                "in_degree": graph.in_degree(node_id)}
            if prop is not None:
                item["prop_value"] = attrs.get(prop)
            items.append(item)
        return {"namespace": namespace, "total": len(matched),
                "capped": capped, "offset": offset, "limit": limit,
                "items": items}

    def list_edges(self, namespace, predicate=None, source_type=None,
                   target_type=None) -> Dict[str, Any]:
        graph = self.graph
        edges: List[Dict[str, Any]] = []
        for s, t, attrs in graph.edges(data=True):
            pred = attrs.get("predicate", "")
            if predicate and pred != predicate:
                continue
            st = graph.nodes[s].get("type", "")
            tt = graph.nodes[t].get("type", "")
            if source_type and st != source_type:
                continue
            if target_type and tt != target_type:
                continue
            edges.append({
                "source": s, "predicate": pred, "target": t,
                "source_name": graph.nodes[s].get("name", s), "source_type": st,
                "target_name": graph.nodes[t].get("name", t), "target_type": tt})
        edges.sort(key=lambda e: (e["source_name"], e["predicate"],
                                  e["target_name"]))
        return {"namespace": namespace, "predicate": predicate,
                "total": len(edges), "edges": edges}

    def list_neighbors(self, namespace, node_id, limit=60) -> Dict[str, Any]:
        graph = self.graph
        if node_id not in graph:
            return {"error": "node_not_found", "detail": node_id}

        neighbor_ids: List[str] = []
        seen = set()
        raw_links: List[Dict[str, Any]] = []
        for _, t, a in graph.out_edges(node_id, data=True):
            raw_links.append({"source": node_id, "target": t,
                              "predicate": a.get("predicate", "")})
            if t != node_id and t not in seen:
                seen.add(t); neighbor_ids.append(t)
        for s, _, a in graph.in_edges(node_id, data=True):
            raw_links.append({"source": s, "target": node_id,
                              "predicate": a.get("predicate", "")})
            if s != node_id and s not in seen:
                seen.add(s); neighbor_ids.append(s)

        total = len(neighbor_ids)
        kept = set(neighbor_ids[:limit]) | {node_id}
        links = [l for l in raw_links
                 if l["source"] in kept and l["target"] in kept]

        def _node_view(nid: str) -> Dict[str, Any]:
            attrs = graph.nodes[nid]
            return {"id": nid, "name": attrs.get("name", nid),
                    "type": attrs.get("type", ""),
                    "trust": attrs.get("trust", "") or "unset",
                    "degree": graph.out_degree(nid) + graph.in_degree(nid)}

        return {
            "namespace": namespace, "anchor": node_id,
            "nodes": [_node_view(nid) for nid in kept],
            "links": links,
            "total_neighbors": total, "limit": limit,
            "truncated": total > limit,
        }

    def node_detail(self, namespace, node_id):
        g = self.graph
        if node_id not in g:
            return None

        def ev(neighbor, a):
            return {"predicate": a.get("predicate", ""), "target": neighbor,
                    "target_name": g.nodes[neighbor].get("name", neighbor),
                    "target_type": g.nodes[neighbor].get("type", "")}
        return {
            "id": node_id, "attrs": dict(g.nodes[node_id]),
            "out_edges": [ev(t, a) for _, t, a in g.out_edges(node_id, data=True)],
            "in_edges": [ev(s, a) for s, _, a in g.in_edges(node_id, data=True)]}

    def aggregate(self, namespace):
        from collections import Counter
        g = self.graph
        trust = Counter(a.get("trust", "") or "unset" for _, a in g.nodes(data=True))
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                "trust": dict(trust)}

    def distributions(self, namespace):
        from collections import Counter
        g = self.graph
        return {"node_types": dict(Counter(a.get("type", "unknown")
                                           for _, a in g.nodes(data=True))),
                "predicates": dict(Counter(a.get("predicate", "unknown")
                                           for _, _, a in g.edges(data=True)))}

    def metaclass_types(self, namespace) -> set:
        g = self.graph
        types = {a.get("type", "unknown") for _, a in g.nodes(data=True)}
        n = len(types)
        if not n:
            return set()
        covered = {}   # 타입 T -> T 멤버들이 이름 짓는 타입집합 값들
        for _, a in g.nodes(data=True):
            nm = a.get("name") or ""
            if nm in types:
                covered.setdefault(a.get("type", "unknown"), set()).add(nm)
        return {t for t, s in covered.items() if len(s) / n > 0.5}

    def review_counts(self, namespace, confirmed_ids, rejected_ids):
        g = self.graph
        confirmed = sum(1 for n in g.nodes if n in confirmed_ids)
        pending = sum(1 for nid, a in g.nodes(data=True)
                      if a.get("source")
                      and nid not in confirmed_ids and nid not in rejected_ids)
        return {"confirmed": confirmed, "rejected": len(rejected_ids),
                "pending": pending}

    def query_nodes(self, namespace, node_type=None, trust=None,
                    prop_key=None, prop_value=None, prop_op="eq",
                    rel_predicate=None, rel_target=None, rel_target_type=None,
                    rel_direction="out", offset=0, limit=50):
        g = self.graph

        def prop_ok(attrs):
            if not prop_key:
                return True
            v = attrs.get(prop_key)
            if prop_op == "exists":
                return v not in (None, "", [])
            if v is None:
                return False
            if prop_op == "contains":
                return str(prop_value).lower() in str(v).lower()
            return str(v) == str(prop_value)

        want_rel = bool(rel_predicate or rel_target or rel_target_type)

        def rel_ok(nid):
            if not want_rel:
                return True
            edges = (g.out_edges(nid, data=True) if rel_direction == "out"
                     else g.in_edges(nid, data=True))
            for e in edges:
                other = e[1] if rel_direction == "out" else e[0]
                attrs = e[2]
                if rel_predicate and attrs.get("predicate") != rel_predicate:
                    continue
                if rel_target and other != rel_target:
                    continue
                if rel_target_type and g.nodes[other].get("type", "") != rel_target_type:
                    continue
                return True
            return False

        matched = []
        capped = False
        for nid, attrs in g.nodes(data=True):
            if node_type and attrs.get("type", "") != node_type:
                continue
            if trust and (attrs.get("trust", "") or "unset") != trust:
                continue
            if not prop_ok(attrs) or not rel_ok(nid):
                continue
            matched.append((nid, attrs))
            if len(matched) >= self.count_cap:
                capped = True
                break
        matched.sort(key=lambda p: str(p[1].get("name", p[0])))
        window = matched[offset:offset + limit]
        items = [{"node_id": nid, "name": attrs.get("name", nid),
                  "type": attrs.get("type", ""),
                  "trust": attrs.get("trust", "") or "unset",
                  "out_degree": g.out_degree(nid), "in_degree": g.in_degree(nid)}
                 for nid, attrs in window]
        return {"namespace": namespace, "total": len(matched), "capped": capped,
                "offset": offset, "limit": limit, "items": items}


# ─── PostgreSQL — 인덱스 질의(진실 원본) ──────────────────────────────────
class PostgresGraphStore(GraphStore):
    """ontology.node/edge 위의 인덱스 질의. 반환 형태는 InMemoryGraphStore 와
    동일해야 한다. type/name/trust 는 컬럼, 그 밖(source/definition/aliases/임의)
    은 properties(jsonb)에 산다 — KG attrs 를 이 둘로 갈라 저장하는 규약."""

    def __init__(self, namespace: str, schema: str):
        from . import pg
        self._pg = pg
        self.schema = schema

    # -- 내부 헬퍼 --
    def _degrees(self, cur, namespace: str, ids: List[str]):
        """주어진 노드들의 out/in 차수를 인덱스 GROUP BY 로 한 번에."""
        s = self.schema
        if not ids:
            return {}, {}
        cur.execute(f"SELECT source_id, count(*) FROM {s}.edge "
                    f"WHERE namespace=%s AND source_id = ANY(%s) GROUP BY source_id",
                    (namespace, ids))
        out = dict(cur.fetchall())
        cur.execute(f"SELECT target_id, count(*) FROM {s}.edge "
                    f"WHERE namespace=%s AND target_id = ANY(%s) GROUP BY target_id",
                    (namespace, ids))
        inn = dict(cur.fetchall())
        return out, inn

    def list_nodes(self, namespace, q=None, node_type=None, trust=None,
                   prop=None, kind=None, offset=0, limit=50) -> Dict[str, Any]:
        s = self.schema
        where = ["namespace = %s"]
        params: List[Any] = [namespace]
        if node_type:
            where.append("type = %s"); params.append(node_type)
        if kind in ("class", "instance"):
            # 클래스 = 이름이 타입으로 쓰임 OR 타입이 메타클래스(service 와 동일 규칙).
            meta = list(self.metaclass_types(namespace))
            cls = (f"(COALESCE(name,'') IN (SELECT DISTINCT "
                   f"   COALESCE(NULLIF(type,''),'unknown') FROM {s}.node "
                   f"   WHERE namespace=%s) "
                   f" OR COALESCE(NULLIF(type,''),'unknown') = ANY(%s))")
            where.append(cls if kind == "class" else f"NOT {cls}")
            params += [namespace, meta]
        if trust:
            # 인메모리는 trust 빈값을 'unset' 으로 본다 → PG 는 기본이 'unset'
            where.append("(COALESCE(NULLIF(trust,''),'unset')) = %s"); params.append(trust)
        if prop is not None:
            # "이 프로퍼티가 채워진 노드만" — 컬럼(type/name/trust)이면 컬럼을,
            # 아니면 properties(jsonb)를 본다.
            if prop in ("type", "name", "trust"):
                where.append(f"COALESCE({prop},'') <> ''")
            else:
                where.append("(properties ? %s AND COALESCE(properties->>%s,'') <> '' "
                             "AND properties->>%s <> '[]')")
                params += [prop, prop, prop]
        if (q or "").strip():
            # 이름·별칭만 본다. node_id 는 "{type}:{name}" 이라 이를 검색하면
            # "사원" 이 사원 타입의 모든 인스턴스(불국사…)를 끌어와 ES(이름 기반)와
            # 어긋났다 — 두 경로를 이름 기반으로 통일.
            needle = f"%{q.strip()}%"
            where.append("(COALESCE(name,'') ILIKE %s "
                         "OR COALESCE(properties->>'aliases','') ILIKE %s)")
            params += [needle, needle]
        wsql = " AND ".join(where)

        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {s}.node WHERE {wsql}", tuple(params))
            total = cur.fetchone()[0]
            # 이름순 정렬(인메모리와 동일: name 없으면 node_id). COLLATE "C" 로
            # 바이트(=UTF-8 코드포인트) 순서 → 파이썬 str 정렬과 일치시켜, DB 로캘
            # 콜레이션이 페이지네이션 창을 memory 와 어긋나게 하지 않도록 한다.
            cur.execute(
                f"SELECT node_id, type, COALESCE(NULLIF(trust,''),'unset'), properties, "
                f"COALESCE(name, node_id) AS disp FROM {s}.node WHERE {wsql} "
                f'ORDER BY COALESCE(name, node_id) COLLATE "C" OFFSET %s LIMIT %s',
                tuple(params) + (offset, limit))
            rows = cur.fetchall()
            ids = [r[0] for r in rows]
            out_deg, in_deg = self._degrees(cur, namespace, ids)

        items = []
        for node_id, ntype, ntrust, props, disp in rows:
            props = props or {}
            item = {
                "node_id": node_id,
                "name": props.get("name") or disp,
                "type": ntype or "",
                "trust": ntrust,
                "source": props.get("source", ""),
                "definition": props.get("definition", ""),
                "aliases": props.get("aliases") or [],
                "out_degree": int(out_deg.get(node_id, 0)),
                "in_degree": int(in_deg.get(node_id, 0))}
            if prop is not None:
                item["prop_value"] = (
                    {"type": ntype, "name": props.get("name") or disp,
                     "trust": ntrust}.get(prop) if prop in ("type", "name", "trust")
                    else props.get(prop))
            items.append(item)
        # PG 는 진짜 카운트 → 근사 상한 불필요
        return {"namespace": namespace, "total": int(total),
                "capped": False, "offset": offset, "limit": limit,
                "items": items}

    def list_edges(self, namespace, predicate=None, source_type=None,
                   target_type=None) -> Dict[str, Any]:
        s = self.schema
        where = ["e.namespace = %s"]
        params: List[Any] = [namespace]
        if predicate:
            where.append("e.predicate = %s"); params.append(predicate)
        if source_type:
            where.append("COALESCE(sn.type,'') = %s"); params.append(source_type)
        if target_type:
            where.append("COALESCE(tn.type,'') = %s"); params.append(target_type)
        wsql = " AND ".join(where)
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT e.source_id, e.predicate, e.target_id, "
                f"       COALESCE(sn.name, e.source_id), COALESCE(sn.type,''), "
                f"       COALESCE(tn.name, e.target_id), COALESCE(tn.type,'') "
                f"FROM {s}.edge e "
                f"LEFT JOIN {s}.node sn ON sn.namespace=e.namespace AND sn.node_id=e.source_id "
                f"LEFT JOIN {s}.node tn ON tn.namespace=e.namespace AND tn.node_id=e.target_id "
                f"WHERE {wsql} "
                # COLLATE "C" — memory(파이썬 코드포인트) 정렬과 일치 (source_name, predicate, target_name)
                f'ORDER BY COALESCE(sn.name, e.source_id) COLLATE "C", '
                f'e.predicate COLLATE "C", COALESCE(tn.name, e.target_id) COLLATE "C"',
                tuple(params))
            rows = cur.fetchall()
        edges = [{"source": r[0], "predicate": r[1], "target": r[2],
                  "source_name": r[3], "source_type": r[4],
                  "target_name": r[5], "target_type": r[6]} for r in rows]
        return {"namespace": namespace, "predicate": predicate,
                "total": len(edges), "edges": edges}

    def list_neighbors(self, namespace, node_id, limit=60) -> Dict[str, Any]:
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {s}.node WHERE namespace=%s AND node_id=%s",
                        (namespace, node_id))
            if cur.fetchone() is None:
                return {"error": "node_not_found", "detail": node_id}

            # out 먼저, 그다음 in — 인메모리의 삽입 순서·중복 제거를 재현
            cur.execute(f"SELECT target_id, predicate FROM {s}.edge "
                        f"WHERE namespace=%s AND source_id=%s", (namespace, node_id))
            out_rows = cur.fetchall()
            cur.execute(f"SELECT source_id, predicate FROM {s}.edge "
                        f"WHERE namespace=%s AND target_id=%s", (namespace, node_id))
            in_rows = cur.fetchall()

            neighbor_ids: List[str] = []
            seen = set()
            raw_links: List[Dict[str, Any]] = []
            for t, pred in out_rows:
                raw_links.append({"source": node_id, "target": t, "predicate": pred or ""})
                if t != node_id and t not in seen:
                    seen.add(t); neighbor_ids.append(t)
            for src, pred in in_rows:
                raw_links.append({"source": src, "target": node_id, "predicate": pred or ""})
                if src != node_id and src not in seen:
                    seen.add(src); neighbor_ids.append(src)

            total = len(neighbor_ids)
            kept_ids = neighbor_ids[:limit]
            kept = set(kept_ids) | {node_id}
            links = [l for l in raw_links if l["source"] in kept and l["target"] in kept]

            all_kept = list(kept)
            cur.execute(f"SELECT node_id, type, COALESCE(NULLIF(trust,''),'unset'), "
                        f"COALESCE(name, node_id) FROM {s}.node "
                        f"WHERE namespace=%s AND node_id = ANY(%s)",
                        (namespace, all_kept))
            meta = {r[0]: {"type": r[1] or "", "trust": r[2], "name": r[3]}
                    for r in cur.fetchall()}
            out_deg, in_deg = self._degrees(cur, namespace, all_kept)

        def _node_view(nid: str) -> Dict[str, Any]:
            m = meta.get(nid, {"type": "", "trust": "unset", "name": nid})
            return {"id": nid, "name": m["name"], "type": m["type"],
                    "trust": m["trust"],
                    "degree": int(out_deg.get(nid, 0)) + int(in_deg.get(nid, 0))}

        return {
            "namespace": namespace, "anchor": node_id,
            "nodes": [_node_view(nid) for nid in kept],
            "links": links,
            "total_neighbors": total, "limit": limit,
            "truncated": total > limit,
        }

    def node_detail(self, namespace, node_id):
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT type, name, trust, properties FROM {s}.node "
                        f"WHERE namespace=%s AND node_id=%s", (namespace, node_id))
            row = cur.fetchone()
            if row is None:
                return None
            ntype, name, trust, props = row
            cur.execute(
                f"SELECT e.predicate, e.target_id, COALESCE(n.name, e.target_id), "
                f"COALESCE(n.type,'') FROM {s}.edge e "
                f"LEFT JOIN {s}.node n ON n.namespace=e.namespace AND n.node_id=e.target_id "
                f"WHERE e.namespace=%s AND e.source_id=%s", (namespace, node_id))
            out = [{"predicate": r[0], "target": r[1], "target_name": r[2],
                    "target_type": r[3]} for r in cur.fetchall()]
            cur.execute(
                f"SELECT e.predicate, e.source_id, COALESCE(n.name, e.source_id), "
                f"COALESCE(n.type,'') FROM {s}.edge e "
                f"LEFT JOIN {s}.node n ON n.namespace=e.namespace AND n.node_id=e.source_id "
                f"WHERE e.namespace=%s AND e.target_id=%s", (namespace, node_id))
            inn = [{"predicate": r[0], "target": r[1], "target_name": r[2],
                    "target_type": r[3]} for r in cur.fetchall()]
        # attrs 재구성: 컬럼(type/name/trust) + properties(나머지) — sync 의 역
        attrs = dict(props or {})
        attrs["type"] = ntype or ""
        attrs["name"] = name if name is not None else node_id
        if trust:
            attrs["trust"] = trust
        return {"id": node_id, "attrs": attrs, "out_edges": out, "in_edges": inn}

    def aggregate(self, namespace):
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {s}.node WHERE namespace=%s", (namespace,))
            nodes = cur.fetchone()[0]
            cur.execute(f"SELECT count(*) FROM {s}.edge WHERE namespace=%s", (namespace,))
            edges = cur.fetchone()[0]
            cur.execute(f"SELECT COALESCE(NULLIF(trust,''),'unset'), count(*) "
                        f"FROM {s}.node WHERE namespace=%s GROUP BY 1", (namespace,))
            trust = {k: v for k, v in cur.fetchall()}
        return {"nodes": int(nodes), "edges": int(edges), "trust": trust}

    def distributions(self, namespace):
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COALESCE(NULLIF(type,''),'unknown'), count(*) "
                        f"FROM {s}.node WHERE namespace=%s GROUP BY 1", (namespace,))
            types = {k: v for k, v in cur.fetchall()}
            cur.execute(f"SELECT COALESCE(NULLIF(predicate,''),'unknown'), count(*) "
                        f"FROM {s}.edge WHERE namespace=%s GROUP BY 1", (namespace,))
            preds = {k: v for k, v in cur.fetchall()}
        return {"node_types": types, "predicates": preds}

    def metaclass_types(self, namespace) -> set:
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"WITH t AS (SELECT DISTINCT COALESCE(NULLIF(type,''),'unknown') v "
                f"           FROM {s}.node WHERE namespace=%s) "
                f"SELECT ty, cov, (SELECT count(*) FROM t) ntypes FROM ("
                f"  SELECT COALESCE(NULLIF(type,''),'unknown') ty, "
                f"    count(DISTINCT name) FILTER (WHERE name IN (SELECT v FROM t)) cov "
                f"  FROM {s}.node WHERE namespace=%s GROUP BY 1) x",
                (namespace, namespace))
            return {ty for ty, cov, ntypes in cur.fetchall()
                    if ntypes and cov / ntypes > 0.5}

    def review_counts(self, namespace, confirmed_ids, rejected_ids):
        s = self.schema
        judged = list(set(confirmed_ids) | set(rejected_ids))
        conf = list(confirmed_ids)
        with self._pg.connect() as conn, conn.cursor() as cur:
            # 확정 = PG 노드 중 confirmed 집합에 든 것 (인메모리: 그래프 노드 기준)
            cur.execute(f"SELECT count(*) FROM {s}.node "
                        f"WHERE namespace=%s AND node_id = ANY(%s)", (namespace, conf))
            confirmed = cur.fetchone()[0]
            # 대기 = source 있는 미판정 노드
            cur.execute(
                f"SELECT count(*) FROM {s}.node WHERE namespace=%s "
                f"AND COALESCE(properties->>'source','') <> '' "
                f"AND node_id <> ALL(%s)", (namespace, judged))
            pending = cur.fetchone()[0]
        return {"confirmed": int(confirmed), "rejected": len(rejected_ids),
                "pending": int(pending)}

    def query_nodes(self, namespace, node_type=None, trust=None,
                    prop_key=None, prop_value=None, prop_op="eq",
                    rel_predicate=None, rel_target=None, rel_target_type=None,
                    rel_direction="out", offset=0, limit=50):
        s = self.schema
        where = ["n.namespace = %s"]
        params = [namespace]
        if node_type:
            where.append("n.type = %s"); params.append(node_type)
        if trust:
            where.append("COALESCE(NULLIF(n.trust,''),'unset') = %s"); params.append(trust)
        if prop_key:
            if prop_op == "exists":
                where.append("(n.properties ? %s AND COALESCE(n.properties->>%s,'') <> '')")
                params += [prop_key, prop_key]
            elif prop_op == "contains":
                where.append("n.properties->>%s ILIKE %s"); params += [prop_key, f"%{prop_value}%"]
            else:  # eq
                where.append("n.properties->>%s = %s"); params += [prop_key, str(prop_value)]
        if rel_predicate or rel_target or rel_target_type:
            # out: n 이 source → 반대편은 target_id / in: n 이 target → 반대편은 source_id
            ecol = "e.source_id" if rel_direction == "out" else "e.target_id"
            ncol = "e.target_id" if rel_direction == "out" else "e.source_id"
            sub = ["e.namespace = n.namespace", f"{ecol} = n.node_id"]
            subp = []
            ex = f"EXISTS(SELECT 1 FROM {s}.edge e "
            if rel_target_type:
                ex += f"JOIN {s}.node tn ON tn.namespace = e.namespace AND tn.node_id = {ncol} "
            if rel_predicate:
                sub.append("e.predicate = %s"); subp.append(rel_predicate)
            if rel_target:
                sub.append(f"{ncol} = %s"); subp.append(rel_target)
            if rel_target_type:
                sub.append("tn.type = %s"); subp.append(rel_target_type)
            ex += "WHERE " + " AND ".join(sub) + ")"
            where.append(ex); params += subp
        wsql = " AND ".join(where)

        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {s}.node n WHERE {wsql}", tuple(params))
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT n.node_id, n.type, COALESCE(NULLIF(n.trust,''),'unset'), "
                f"COALESCE(n.name, n.node_id) FROM {s}.node n WHERE {wsql} "
                f'ORDER BY COALESCE(n.name, n.node_id) COLLATE "C" OFFSET %s LIMIT %s',
                tuple(params) + (offset, limit))
            rows = cur.fetchall()
            ids = [r[0] for r in rows]
            out_deg, in_deg = self._degrees(cur, namespace, ids)
        items = [{"node_id": r[0], "name": r[3], "type": r[1] or "", "trust": r[2],
                  "out_degree": int(out_deg.get(r[0], 0)),
                  "in_degree": int(in_deg.get(r[0], 0))} for r in rows]
        return {"namespace": namespace, "total": int(total), "capped": False,
                "offset": offset, "limit": limit, "items": items}

    # ── 쓰기 프리미티브 (P4-b) — 수동 변경을 PG(진실)에 이중기록 ──
    # KG attrs 를 컬럼(type/name/trust)+properties(나머지)로 가르는 규약은
    # sync_from_graph 와 동일해야 읽기 parity 가 유지된다.
    def upsert_node(self, namespace, node_id, attrs):
        import json
        s = self.schema
        props = {k: v for k, v in attrs.items() if k not in _NODE_COLS}
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {s}.node(namespace,node_id,type,name,trust,properties,updated_at)"
                f" VALUES(%s,%s,%s,%s,%s,%s::jsonb, now()) "
                f"ON CONFLICT (namespace,node_id) DO UPDATE SET "
                f"type=EXCLUDED.type, name=EXCLUDED.name, trust=EXCLUDED.trust, "
                f"properties=EXCLUDED.properties, updated_at=now()",
                (namespace, node_id, attrs.get("type", "") or "unknown",
                 attrs.get("name"), (attrs.get("trust") or "unset"),
                 json.dumps(props, ensure_ascii=False)))
            conn.commit()

    def delete_node(self, namespace, node_id):
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {s}.edge WHERE namespace=%s AND "
                        f"(source_id=%s OR target_id=%s)", (namespace, node_id, node_id))
            cur.execute(f"DELETE FROM {s}.node WHERE namespace=%s AND node_id=%s",
                        (namespace, node_id))
            conn.commit()

    def add_edge(self, namespace, source, predicate, target, attrs=None):
        import json
        s = self.schema
        attrs = attrs or {}
        props = {k: v for k, v in attrs.items() if k not in _EDGE_COLS}
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {s}.edge(namespace,source_id,predicate,target_id,weight,properties)"
                f" VALUES(%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                (namespace, source, predicate, target,
                 float(attrs.get("weight", 1.0) or 1.0),
                 json.dumps(props, ensure_ascii=False)))
            conn.commit()

    def delete_edge(self, namespace, source, predicate, target):
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {s}.edge WHERE namespace=%s AND source_id=%s "
                        f"AND predicate=%s AND target_id=%s",
                        (namespace, source, predicate, target))
            conn.commit()

    def rename_type(self, namespace, old, new):
        s = self.schema
        with self._pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {s}.node SET type=%s, updated_at=now() "
                        f"WHERE namespace=%s AND type=%s", (new, namespace, old))
            conn.commit()


class AicoachGraphStore(InMemoryGraphStore):
    """aicoach 스키마(kg_node/kg_edge) 라이브 스토어 (Track 1 phase 2).

    읽기는 InMemoryGraphStore 상속 — aicoach-하이드레이트된 엔진 그래프를 감싼다.
    쓰기만 aicoach 테이블에 직접(컬럼 매핑: node_id→id, name→label, 나머지→attrs;
    kg_edge subject/predicate/object, namespace 컬럼 없음). **쓰기는
    ONTOLOGY_AICOACH_WRITE=true 일 때만** — aicoach 라이브 프로덕션 테이블이므로
    기본 읽기전용(오프트인). 차단 시 WARNING + no-op(엔진 in-memory 는 이미 변경됨)."""

    def __init__(self, graph, schema: str, count_cap: int = DEFAULT_COUNT_CAP):
        super().__init__(graph, count_cap)
        if not schema.isidentifier():
            raise ValueError(f"invalid aicoach schema: {schema!r}")
        self.schema = schema

    def _write_ok(self, op: str) -> bool:
        if (os.environ.get("ONTOLOGY_AICOACH_WRITE", "") or "").strip().lower() in ("1", "true", "yes"):
            return True
        logger.warning(f"aicoach 쓰기 차단({op}) — ONTOLOGY_AICOACH_WRITE 미설정(읽기전용 안전장치)")
        return False

    def upsert_node(self, namespace, node_id, attrs):
        if not self._write_ok("upsert_node"):
            return
        import json
        from . import pg
        payload = {k: v for k, v in attrs.items() if k not in ("type", "name", "label", "source")}
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.schema}.kg_node(id,type,label,attrs,source) "
                f"VALUES(%s,%s,%s,%s::jsonb,%s) ON CONFLICT (id) DO UPDATE SET "
                f"type=EXCLUDED.type,label=EXCLUDED.label,attrs=EXCLUDED.attrs,source=EXCLUDED.source",
                (node_id, attrs.get("type") or "unknown",
                 attrs.get("name") or attrs.get("label"),
                 json.dumps(payload, ensure_ascii=False), attrs.get("source")))
            conn.commit()

    def delete_node(self, namespace, node_id):
        if not self._write_ok("delete_node"):
            return
        from . import pg
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.schema}.kg_edge WHERE subject=%s OR object=%s",
                        (node_id, node_id))
            cur.execute(f"DELETE FROM {self.schema}.kg_node WHERE id=%s", (node_id,))
            conn.commit()

    def add_edge(self, namespace, source, predicate, target, attrs=None):
        if not self._write_ok("add_edge"):
            return
        from . import pg
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.schema}.kg_edge(subject,predicate,object,source) "
                f"VALUES(%s,%s,%s,%s) ON CONFLICT (subject,predicate,object) DO NOTHING",
                (source, predicate, target, (attrs or {}).get("source")))
            conn.commit()

    def delete_edge(self, namespace, source, predicate, target):
        if not self._write_ok("delete_edge"):
            return
        from . import pg
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.schema}.kg_edge WHERE subject=%s AND "
                        f"predicate=%s AND object=%s", (source, predicate, target))
            conn.commit()

    def rename_type(self, namespace, old, new):
        if not self._write_ok("rename_type"):
            return
        from . import pg
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {self.schema}.kg_node SET type=%s WHERE type=%s", (new, old))
            conn.commit()


def create_graph_store(namespace: str,
                       count_cap: int = DEFAULT_COUNT_CAP) -> GraphStore:
    """백엔드 선택 팩토리. 이 네임스페이스가 PG 대상(전역 postgres 또는 허용목록)
    이고 접속 가능하면 PostgresGraphStore, 아니면 InMemoryGraphStore(NetworkX).
    degrade 는 WARNING 을 남긴다."""
    from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
    # aicoach 라이브 소비 네임스페이스: 엔진이 aicoach 스키마에서 하이드레이트되므로
    # (get_knowledge_graph_engine) 그 in-memory 그래프를 감싸면 seam(목록·통계·이웃)도
    # aicoach 라이브를 읽는다 — PostgresGraphStore(ontology 스키마) 우회.
    aschema = aicoach_source(namespace)
    if aschema:
        # 읽기는 aicoach-하이드레이트 엔진 그래프(InMemory 상속), 쓰기는 aicoach
        # 테이블 직접(ONTOLOGY_AICOACH_WRITE gated). Track 1 phase 2.
        return AicoachGraphStore(get_knowledge_graph_engine(namespace).graph, aschema, count_cap)
    if pg_backed(namespace):
        from . import pg
        if pg.available():
            return PostgresGraphStore(namespace, pg.get_schema())
        logger.warning(f"'{namespace}' PG 대상이나 접속 불가 "
                       "— InMemoryGraphStore 로 degrade")
    return InMemoryGraphStore(get_knowledge_graph_engine(namespace).graph, count_cap)
