"""GraphStore 계약 테스트 (축 5, P1).

InMemoryGraphStore(NetworkX)와 PostgresGraphStore 가 **동일 데이터에 대해
동일 payload** 를 돌려주는지 검증한다 — 관리 콘솔·Graph3D 가 백엔드를 모른 채
같은 결과를 먹는 것이 계약이다.

PG 통합 파트는 ONTOLOGY_PG_DSN 이 있고 접속 가능할 때만 돈다(없으면 skip) —
ES 통합 테스트와 같은 관례. 정렬은 PG 콜레이션 vs 파이썬 코드포인트 차이가
있을 수 있어, 비교 전 양쪽을 파이썬 키로 재정렬한다(집합·필드 동등성이 계약,
백엔드별 내부 정렬은 각자 "이름순"이면 족하다).
"""
import networkx as nx
import pytest

from ontology.core import pg
from ontology.core.graph_store import InMemoryGraphStore, PostgresGraphStore

NS = "test_graphstore_parity"

# (node_id, type, name, trust, {props})  — props 에 source/definition/aliases
NODES = [
    ("temple:불국사", "temple", "불국사", "authoritative",
     {"source": "wiki", "definition": "신라 사찰", "aliases": ["Bulguksa"], "era": "신라"}),
    ("heritage:석굴암", "heritage", "석굴암", "authoritative",
     {"source": "gov", "definition": "석굴 사원", "aliases": []}),
    ("place:경주", "place", "경주", "unknown", {}),
    ("place:서울", "place", "서울", "unknown", {"source": "wiki"}),
]
# (source, predicate, target)
EDGES = [
    ("temple:불국사", "located_in", "place:경주"),
    ("heritage:석굴암", "part_of", "temple:불국사"),
    ("heritage:석굴암", "located_in", "place:경주"),
]


def _build_networkx():
    g = nx.MultiDiGraph()
    for nid, ntype, name, trust, props in NODES:
        g.add_node(nid, type=ntype, name=name, trust=trust, **props)
    for s, p, t in EDGES:
        g.add_edge(s, t, predicate=p)
    return g


def _norm_nodes(payload):
    """비교용 정규화 — items 를 node_id 로 재정렬(콜레이션 차이 무시)."""
    return {
        "total": payload["total"],
        "capped": payload["capped"],
        "items": sorted(payload["items"], key=lambda x: x["node_id"]),
    }


def _norm_edges(payload):
    return {
        "total": payload["total"],
        "edges": sorted(payload["edges"],
                        key=lambda e: (e["source"], e["predicate"], e["target"])),
    }


def _norm_neighbors(payload):
    return {
        "anchor": payload["anchor"],
        "total_neighbors": payload["total_neighbors"],
        "truncated": payload["truncated"],
        "nodes": sorted(payload["nodes"], key=lambda n: n["id"]),
        "links": sorted(payload["links"],
                        key=lambda l: (l["source"], l["predicate"], l["target"])),
    }


# ─── 인메모리 단독(무 PG) — 형태·정확성 ───────────────────────────────────
def test_inmemory_list_nodes_shape():
    store = InMemoryGraphStore(_build_networkx())
    r = store.list_nodes(NS)
    assert r["total"] == 4 and r["capped"] is False
    seok = next(i for i in r["items"] if i["node_id"] == "heritage:석굴암")
    assert seok["type"] == "heritage" and seok["trust"] == "authoritative"
    assert seok["out_degree"] == 2 and seok["in_degree"] == 0  # part_of, located_in 둘 다 out


def test_inmemory_neighbors_anchor():
    store = InMemoryGraphStore(_build_networkx())
    r = store.list_neighbors(NS, "place:경주")
    assert r["anchor"] == "place:경주"
    assert r["total_neighbors"] == 2          # 불국사, 석굴암 (둘 다 in)
    assert {n["id"] for n in r["nodes"]} == {"place:경주", "temple:불국사", "heritage:석굴암"}


def test_inmemory_count_cap():
    store = InMemoryGraphStore(_build_networkx(), count_cap=2)
    r = store.list_nodes(NS)
    assert r["capped"] is True and r["total"] == 2


def test_inmemory_metaclass_detection():
    """클래스/인스턴스 판정 — 메타클래스는 '타입들을 이름 짓는 타입'이고,
    자기참조 클래스 노드가 있어도 그 타입의 인스턴스를 클래스로 오분류하지 않는다."""
    g = nx.MultiDiGraph()
    # 메타클래스 Klass 의 멤버들(사원·탑)이 곧 다른 노드의 type → Klass 가 메타클래스
    g.add_node("Klass:사원", type="Klass", name="사원")
    g.add_node("Klass:탑", type="Klass", name="탑")
    g.add_node("Klass:빈클래스", type="Klass", name="빈클래스")   # 인스턴스 0 인 클래스
    g.add_node("탑:탑", type="탑", name="탑")                    # 자기참조 클래스 노드
    g.add_node("사원:불국사", type="사원", name="불국사")          # 인스턴스
    g.add_node("탑:거돈사지탑", type="탑", name="거돈사지탑")        # 인스턴스
    store = InMemoryGraphStore(g)
    meta = store.metaclass_types(NS)
    # Klass 멤버 이름(사원·탑)이 타입집합의 과반을 이름 지음 → 메타클래스
    assert "Klass" in meta
    # '탑' 타입은 자기참조 하나만 타입명 → 메타클래스 아님(인스턴스 보호)
    assert "탑" not in meta and "사원" not in meta


# ─── PG 통합 + 두 백엔드 동등성 (접속 가능 시만) ─────────────────────────
pg_required = pytest.mark.skipif(
    not pg.available(), reason="ONTOLOGY_PG_DSN 미설정/접속 불가 — PG 통합 skip")


@pytest.fixture
def seeded_pg():
    """throwaway 네임스페이스에 동일 데이터를 심고, 끝나면 지운다."""
    import json
    schema = pg.get_schema()
    pg.ensure_schema(schema)
    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS,))
            for nid, ntype, name, trust, props in NODES:
                cur.execute(
                    f"INSERT INTO {schema}.node(namespace,node_id,type,name,trust,properties)"
                    f" VALUES(%s,%s,%s,%s,%s,%s::jsonb)",
                    (NS, nid, ntype, name, trust, json.dumps(props)))
            for s, p, t in EDGES:
                cur.execute(
                    f"INSERT INTO {schema}.edge(namespace,source_id,predicate,target_id)"
                    f" VALUES(%s,%s,%s,%s)", (NS, s, p, t))
        conn.commit()
    yield PostgresGraphStore(NS, schema)
    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS,))
        conn.commit()


@pg_required
def test_parity_list_nodes(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    assert _norm_nodes(mem.list_nodes(NS)) == _norm_nodes(seeded_pg.list_nodes(NS))


@pg_required
def test_parity_list_nodes_filtered(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    for kw in [dict(node_type="place"), dict(trust="authoritative"),
               dict(q="경주"), dict(prop="source")]:
        assert _norm_nodes(mem.list_nodes(NS, **kw)) == \
               _norm_nodes(seeded_pg.list_nodes(NS, **kw)), f"mismatch on {kw}"


@pg_required
def test_parity_ordering_collation(seeded_pg):
    """콜레이션 정합 — COLLATE "C" 로 PG 정렬이 memory(파이썬 코드포인트)와
    **순서까지** 일치. 재정렬 없이 그대로 비교 + 페이지네이션 창도 일치."""
    mem = InMemoryGraphStore(_build_networkx())
    m = mem.list_nodes(NS)["items"]
    p = seeded_pg.list_nodes(NS)["items"]
    assert [x["node_id"] for x in m] == [x["node_id"] for x in p]  # 순서 동일
    # 창(offset/limit)도 같은 항목
    mw = mem.list_nodes(NS, offset=1, limit=2)["items"]
    pw = seeded_pg.list_nodes(NS, offset=1, limit=2)["items"]
    assert [x["node_id"] for x in mw] == [x["node_id"] for x in pw]
    # 엣지 순서도 동일
    me = mem.list_edges(NS)["edges"]
    pe = seeded_pg.list_edges(NS)["edges"]
    assert [(e["source"], e["predicate"], e["target"]) for e in me] == \
           [(e["source"], e["predicate"], e["target"]) for e in pe]


@pg_required
def test_parity_list_edges(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    assert _norm_edges(mem.list_edges(NS)) == _norm_edges(seeded_pg.list_edges(NS))
    assert _norm_edges(mem.list_edges(NS, predicate="located_in")) == \
           _norm_edges(seeded_pg.list_edges(NS, predicate="located_in"))


@pg_required
def test_parity_neighbors(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    for anchor in ["place:경주", "temple:불국사", "heritage:석굴암"]:
        assert _norm_neighbors(mem.list_neighbors(NS, anchor)) == \
               _norm_neighbors(seeded_pg.list_neighbors(NS, anchor)), anchor


@pg_required
def test_parity_neighbors_missing(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    assert mem.list_neighbors(NS, "nope")["error"] == \
           seeded_pg.list_neighbors(NS, "nope")["error"] == "node_not_found"


# ─── P4-b: stats/detail 강등 — 4개 신규 메서드 parity ─────────────────────
@pg_required
def test_parity_aggregate(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    assert mem.aggregate(NS) == seeded_pg.aggregate(NS)  # nodes,edges,trust 동일


@pg_required
def test_parity_distributions(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    assert mem.distributions(NS) == seeded_pg.distributions(NS)


@pg_required
def test_parity_review_counts(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    # 불국사=confirmed, 서울=rejected, 나머지 source 있는 노드는 pending
    conf, rej = {"temple:불국사"}, {"place:서울"}
    assert mem.review_counts(NS, conf, rej) == seeded_pg.review_counts(NS, conf, rej)


@pg_required
def test_parity_query_prop_value(seeded_pg):
    """프로퍼티 값 검색 parity — era=신라 (불국사만)."""
    mem = InMemoryGraphStore(_build_networkx())
    for kw in [dict(prop_key="era", prop_value="신라", prop_op="eq"),
               dict(prop_key="source", prop_value="wiki", prop_op="eq"),
               dict(prop_key="source", prop_op="exists"),
               dict(prop_key="definition", prop_value="사찰", prop_op="contains")]:
        m = mem.query_nodes(NS, **kw); p = seeded_pg.query_nodes(NS, **kw)
        assert m["total"] == p["total"], kw
        assert sorted(i["node_id"] for i in m["items"]) == \
               sorted(i["node_id"] for i in p["items"]), kw


@pg_required
def test_parity_query_relationship(seeded_pg):
    """관계 검색 parity — 'N 이 경주에 located_in' = N 의 out 엣지."""
    mem = InMemoryGraphStore(_build_networkx())
    kw = dict(rel_predicate="located_in", rel_target="place:경주", rel_direction="out")
    m = mem.query_nodes(NS, **kw); p = seeded_pg.query_nodes(NS, **kw)
    assert {i["node_id"] for i in m["items"]} == {i["node_id"] for i in p["items"]} \
           == {"temple:불국사", "heritage:석굴암"}
    # 타입 제약: place 타입 노드로 가는 엣지를 가진 노드(out)
    kw2 = dict(rel_target_type="place", rel_direction="out")
    m2 = mem.query_nodes(NS, **kw2); p2 = seeded_pg.query_nodes(NS, **kw2)
    assert {i["node_id"] for i in m2["items"]} == {i["node_id"] for i in p2["items"]} \
           == {"temple:불국사", "heritage:석굴암"}
    # 프로퍼티 + 관계 결합: era 있음 + 경주로 located_in → 불국사만(era 보유)
    kw3 = dict(prop_key="era", prop_op="exists", rel_predicate="located_in",
               rel_target="place:경주", rel_direction="out")
    m3 = mem.query_nodes(NS, **kw3); p3 = seeded_pg.query_nodes(NS, **kw3)
    assert {i["node_id"] for i in m3["items"]} == {i["node_id"] for i in p3["items"]} \
           == {"temple:불국사"}


@pg_required
def test_parity_node_detail(seeded_pg):
    mem = InMemoryGraphStore(_build_networkx())
    md = mem.node_detail(NS, "heritage:석굴암")
    pd = seeded_pg.node_detail(NS, "heritage:석굴암")
    assert md["id"] == pd["id"] == "heritage:석굴암"
    # 엣지 집합 동일(순서 무관)
    def es(x): return sorted((e["predicate"], e["target"]) for e in x)
    assert es(md["out_edges"]) == es(pd["out_edges"])
    assert es(md["in_edges"]) == es(pd["in_edges"])
    # 핵심 attrs 일치
    assert pd["attrs"]["type"] == md["attrs"]["type"] == "heritage"
    assert pd["attrs"]["name"] == "석굴암"
    assert seeded_pg.node_detail(NS, "nope") is None


# ─── P2: dual-write (sync_from_graph) 정확성 ─────────────────────────────
@pg_required
def test_sync_from_graph_then_parity():
    """빌드 결과(NetworkX)를 sync_from_graph 로 PG 에 미러한 뒤, PG 읽기가
    같은 그래프의 InMemory 읽기와 동일한지 — write 경로가 PG 를 충실히 채우는가."""
    from ontology.core import graph_store as gs
    NS2 = "test_graphstore_sync"
    g = _build_networkx()
    schema = pg.get_schema()
    pg.ensure_schema(schema)
    try:
        counts = gs.sync_from_graph(NS2, g, schema)
        assert counts == {"nodes": 4, "edges": 3}
        mem = InMemoryGraphStore(g)
        pgs = PostgresGraphStore(NS2, schema)
        assert _norm_nodes(mem.list_nodes(NS2)) == _norm_nodes(pgs.list_nodes(NS2))
        assert _norm_edges(mem.list_edges(NS2)) == _norm_edges(pgs.list_edges(NS2))
        for anchor in ["place:경주", "temple:불국사", "heritage:석굴암"]:
            assert _norm_neighbors(mem.list_neighbors(NS2, anchor)) == \
                   _norm_neighbors(pgs.list_neighbors(NS2, anchor)), anchor
        # 재실행 멱등 — 개수 그대로 (delete+insert)
        assert gs.sync_from_graph(NS2, g, schema) == {"nodes": 4, "edges": 3}
    finally:
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS2,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS2,))
            conn.commit()


@pg_required
def test_incremental_sync():
    """증분 미러(P4-b) — 미변경 노드는 재기록 안 함(updated_at 불변),
    변경/추가/삭제만 반영. 대용량 전체 재기록 회피의 핵심."""
    import networkx as nx
    from ontology.core import graph_store as gs
    from ontology.core.graph_store import PostgresGraphStore
    NS3 = "test_incr_sync"
    schema = pg.get_schema(); pg.ensure_schema(schema)

    def mk(changed=False, extra=False, drop_b=False):
        g = nx.MultiDiGraph()
        g.add_node("a", type="t", name="A!" if changed else "A", trust="unset")
        if not drop_b:
            g.add_node("b", type="t", name="B")
        g.add_node("c", type="t", name="C")
        if extra:
            g.add_node("d", type="t", name="D")
        g.add_edge("a", "c", predicate="rel")
        return g

    def ua(nid):
        with pg.connect() as c, c.cursor() as cur:
            cur.execute(f"SELECT updated_at FROM {schema}.node "
                        f"WHERE namespace=%s AND node_id=%s", (NS3, nid))
            row = cur.fetchone()
            return row[0] if row else None
    try:
        with pg.connect() as c, c.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS3,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS3,))
            c.commit()
        pgs = PostgresGraphStore(NS3, schema)
        assert gs.sync_from_graph(NS3, mk(), schema) == {"nodes": 3, "edges": 1}
        assert pgs.list_nodes(NS3)["total"] == 3
        t_b = ua("b")
        # 미변경 재동기화 → b 재기록 안 함
        gs.sync_from_graph(NS3, mk(), schema)
        assert ua("b") == t_b, "증분이 미변경 노드를 재기록했다"
        # a 변경 + d 추가 + b 삭제
        assert gs.sync_from_graph(NS3, mk(changed=True, extra=True, drop_b=True), schema) \
               == {"nodes": 3, "edges": 1}
        ids = {i["node_id"] for i in pgs.list_nodes(NS3, limit=100)["items"]}
        assert ids == {"a", "c", "d"}          # b 삭제, d 추가
        assert pgs.node_detail(NS3, "a")["attrs"]["name"] == "A!"  # a 변경 반영
        assert ua("b") is None                 # b 사라짐
    finally:
        with pg.connect() as c, c.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS3,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS3,))
            c.commit()


@pg_required
def test_pg_write_primitives():
    """P4-b 쓰기 프리미티브 — upsert/delete node, add/delete edge, rename_type
    가 PG 에 반영되고 읽기에 나타나는지 (수동 변경 이중기록의 토대)."""
    import networkx as nx
    from ontology.core.graph_store import PostgresGraphStore
    NS2 = "test_pg_write"
    schema = pg.get_schema()
    pg.ensure_schema(schema)
    s = PostgresGraphStore(NS2, schema)
    try:
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS2,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS2,))
            conn.commit()
        # upsert 2 노드
        s.upsert_node(NS2, "t:A", {"type": "temple", "name": "A", "trust": "authoritative", "era": "x"})
        s.upsert_node(NS2, "p:B", {"type": "place", "name": "B"})
        assert s.list_nodes(NS2)["total"] == 2
        a = s.node_detail(NS2, "t:A")
        assert a["attrs"]["type"] == "temple" and a["attrs"]["era"] == "x"
        # upsert 업데이트(같은 id → 갱신)
        s.upsert_node(NS2, "t:A", {"type": "temple", "name": "A", "trust": "summary"})
        assert s.list_nodes(NS2, trust="summary")["total"] == 1
        # add_edge → neighbors
        s.add_edge(NS2, "t:A", "located_in", "p:B")
        nb = s.list_neighbors(NS2, "t:A")
        assert nb["total_neighbors"] == 1 and nb["links"][0]["predicate"] == "located_in"
        # rename_type
        s.rename_type(NS2, "temple", "shrine")
        assert s.list_nodes(NS2, node_type="shrine")["total"] == 1
        # delete_edge
        s.delete_edge(NS2, "t:A", "located_in", "p:B")
        assert s.list_neighbors(NS2, "t:A")["total_neighbors"] == 0
        # delete_node (엣지 동반 삭제)
        s.add_edge(NS2, "t:A", "located_in", "p:B")
        s.delete_node(NS2, "p:B")
        assert s.node_detail(NS2, "p:B") is None
        assert s.list_neighbors(NS2, "t:A")["total_neighbors"] == 0  # 엣지도 사라짐
        assert s.list_nodes(NS2)["total"] == 1
    finally:
        with pg.connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.edge WHERE namespace=%s", (NS2,))
            cur.execute(f"DELETE FROM {schema}.node WHERE namespace=%s", (NS2,))
            conn.commit()


def test_cohort_resolution():
    """코호트 추출(3+) — 구조화 쿼리를 추출 대상 node_id 집합으로 해석.
    build_training_dataset 이 이 집합으로 추출을 제한한다."""
    from ontology.server.service import OntologyBuilderService
    svc = OntologyBuilderService.__new__(OntologyBuilderService)  # 순수 메서드만
    g = _build_networkx()
    assert svc._resolve_cohort_ids(NS, g, {"node_type": "place"}) == \
           {"place:경주", "place:서울"}
    assert svc._resolve_cohort_ids(NS, g, {"prop_key": "era", "prop_op": "exists"}) == \
           {"temple:불국사"}
    # 관계: 경주로 가는(out) 엣지를 가진 노드
    assert svc._resolve_cohort_ids(
        NS, g, {"rel_predicate": "located_in", "rel_target": "place:경주",
                "rel_direction": "out"}) == {"temple:불국사", "heritage:석굴암"}
    # 빈 코호트 키는 무시(전부)
    assert svc._resolve_cohort_ids(NS, g, {"node_type": ""}) == set(g.nodes())


def test_pg_backed_namespace_selection(monkeypatch):
    """네임스페이스별 PG 선택 — 허용목록/전역 플래그."""
    from ontology.core import graph_store as gs
    monkeypatch.setenv("ONTOLOGY_GRAPH_BACKEND", "memory")
    monkeypatch.setenv("ONTOLOGY_PG_NAMESPACES", "demo_ko, heritage_kr")
    assert gs.pg_backed("demo_ko") is True
    assert gs.pg_backed("heritage_kr") is True
    assert gs.pg_backed("other") is False
    monkeypatch.setenv("ONTOLOGY_GRAPH_BACKEND", "postgres")
    assert gs.pg_backed("anything") is True   # 전역이면 전부
