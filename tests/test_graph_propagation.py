"""이분 그래프 확산 검색 (Personalized PageRank) — 축 4 의 확장 규칙 교체.

**왜 필요한가 (실측 진단)**: ins_cancer_demo 를 개체-only 그래프로 보면
엣지 97, 평균 차수 1.00, **고립 노드 122/194 (62.9%)**, `is_a` 엣지 **0개**다.
현재 확장(`is_a` 폐포 + 1-hop 인접)은 이 데이터에서 거의 아무 일도 하지 않는다 —
폐포는 술어가 없어 죽은 코드이고, 인접은 63%가 고립이라 닿지 않는다.

그런데 같은 그래프를 **노드 + 청크 이분 그래프**로 보면 고립이 **7/194 (3.6%)**,
최대 연결요소가 70 → 103 으로 커진다. 그래프가 성긴 게 아니라 우리가 잘못 보고
있었다 — 연결은 근거 링크(노드↔청크)가 이미 나르고 있다.

이것이 문헌의 전제와 같다:
- HippoRAG 2 (ICML'25, arXiv:2502.14802): phrase 노드 + passage 노드를
  `contains` 엣지로 이은 **이중 노드 KG** 에 Personalized PageRank. passage
  노드의 reset 확률에 가중치(기본 0.05)를 곱해 두 노드 종류의 영향을 조절.
- LinearRAG (2025): 문장-개체 이분 그래프에 PPR 로 개체·passage 관련도를 합산.
- KET-RAG: 골격 KG + 키워드-청크 이분 그래프로 추출 비용 절감.

우리는 이미 그 구조를 갖고 있고, 커버리지 회복으로 링크 123개를 더 만들었다.
PPR 은 LLM 0콜이고 `networkx` 는 이미 커널 의존이다.

**이 테스트가 고정하는 것**: 확산이 (a) 청크를 허브로 다중 홉을 열고,
(b) 무관한 곳에 질량을 뿌리지 않고, (c) 결정적이라는 것.
"""

import pytest

from ontology.core.graph_propagation import (
    CHUNK_PREFIX,
    build_bipartite,
    propagate,
    split_masses,
)


class _Chunk:
    """StoredChunk 의 확산에 필요한 최소 표면 — 실물은 dataclass 라 무겁다."""

    def __init__(self, chunk_id, node_ids):
        self.chunk_id = chunk_id
        self.node_ids = list(node_ids)


@pytest.fixture()
def graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    for nid in ("T:암진단비", "T:암보장개시일", "T:해지", "T:고립"):
        g.add_node(nid, type="T", name=nid.split(":", 1)[1])
    return g


# ─── 1. 이분 그래프 구성 ─────────────────────────────────────────────

class TestBuildBipartite:
    def test_chunk_becomes_a_node(self, graph):
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"])])
        assert f"{CHUNK_PREFIX}c1" in b

    def test_entity_chunk_edge_is_created(self, graph):
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"])])
        assert b.has_edge("T:암진단비", f"{CHUNK_PREFIX}c1")

    def test_entity_entity_edges_are_kept(self, graph):
        graph.add_edge("T:암진단비", "T:해지", predicate="relatesTo")
        b = build_bipartite(graph, [])
        assert b.has_edge("T:암진단비", "T:해지")

    def test_direction_is_dropped(self, graph):
        """확산은 방향이 없다 — definesTerm 이 한쪽으로만 걸려 있어도 관련성은
        양방향이다. 방향을 지키면 술어 방향이라는 우연이 도달성을 좌우한다."""
        graph.add_edge("T:암진단비", "T:해지", predicate="definesTerm")
        b = build_bipartite(graph, [])
        assert b.has_edge("T:해지", "T:암진단비")

    def test_unknown_node_reference_is_ignored(self, graph):
        """청크가 지워진 노드를 가리킬 수 있다(dangling). 그걸 노드로 되살리면
        확산이 그래프에 없는 개체에 질량을 준다."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:없는노드"])])
        assert "T:없는노드" not in b

    def test_isolated_entity_is_still_a_node(self, graph):
        """고립 노드도 그래프의 일원이다 — 없애면 질량 정규화가 달라진다."""
        b = build_bipartite(graph, [])
        assert "T:고립" in b and b.degree("T:고립") == 0


# ─── 2. 확산 — 청크를 허브로 다중 홉 ────────────────────────────────

class TestPropagate:
    def test_reaches_sibling_through_shared_chunk(self, graph):
        """이 테스트가 전체 설계의 이유다. 두 개체 사이에 **개체-개체 엣지가
        없어도** 같은 청크에 근거가 있으면 서로에게 닿는다 — 개체-only 그래프의
        1-hop 인접으로는 영원히 닿지 못하는 관계다."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비", "T:암보장개시일"])])
        masses = propagate(b, {"T:암진단비": 1.0})
        assert masses.get("T:암보장개시일", 0.0) > 0.0

    def test_unrelated_node_gets_no_mass(self, graph):
        """확산이 무관한 곳까지 적시면 확장이 아니라 잡음이다."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비", "T:암보장개시일"])])
        masses = propagate(b, {"T:암진단비": 1.0})
        assert masses.get("T:고립", 0.0) == 0.0

    def test_seed_outranks_what_it_reaches(self, graph):
        """직접 걸린 개념이 파생 개념보다 앞이어야 한다 (기존 채널 B 의 규정)."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비", "T:암보장개시일"])])
        masses = propagate(b, {"T:암진단비": 1.0})
        assert masses["T:암진단비"] > masses["T:암보장개시일"]

    def test_two_hops_through_two_chunks(self, graph):
        """청크1: A,B · 청크2: B,C → A 에서 C 까지 2홉. 연상 검색의 핵심."""
        b = build_bipartite(graph, [
            _Chunk("c1", ["T:암진단비", "T:암보장개시일"]),
            _Chunk("c2", ["T:암보장개시일", "T:해지"]),
        ])
        masses = propagate(b, {"T:암진단비": 1.0})
        assert masses.get("T:해지", 0.0) > 0.0
        # 거리에 따라 감쇠해야 한다 — 아니면 그래프가 순위 정보를 주지 못한다
        assert masses["T:암보장개시일"] > masses["T:해지"]

    def test_chunks_receive_mass_too(self, graph):
        """청크 질량이 곧 검색 결과다 (LinearRAG: PPR 이 개체와 passage 관련도를
        함께 합산)."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"])])
        masses = propagate(b, {"T:암진단비": 1.0})
        assert masses.get(f"{CHUNK_PREFIX}c1", 0.0) > 0.0

    def test_seed_weights_are_respected(self, graph):
        b = build_bipartite(graph, [
            _Chunk("c1", ["T:암진단비"]), _Chunk("c2", ["T:해지"])])
        strong = propagate(b, {"T:암진단비": 10.0, "T:해지": 1.0})
        assert strong[f"{CHUNK_PREFIX}c1"] > strong[f"{CHUNK_PREFIX}c2"]

    def test_deterministic(self, graph):
        """같은 입력 → 같은 출력. 부동소수 합산 순서에 흔들리면 순위가 임의로
        갈린다 (RRF 동점 사건과 같은 부류)."""
        chunks = [_Chunk("c1", ["T:암진단비", "T:암보장개시일"])]
        a = propagate(build_bipartite(graph, chunks), {"T:암진단비": 1.0})
        c = propagate(build_bipartite(graph, chunks), {"T:암진단비": 1.0})
        assert a == c


# ─── 3. 실패 모드 — 죽지 않고 조용히 비운다 ──────────────────────────

class TestFailureModes:
    def test_no_seeds_returns_empty(self, graph):
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"])])
        assert propagate(b, {}) == {}

    def test_seeds_outside_graph_are_dropped(self, graph):
        """networkx 는 그래프에 없는 personalization 키에 예외를 던진다 —
        진입 노드가 그 사이 지워졌을 때 검색 전체가 죽으면 안 된다."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"])])
        masses = propagate(b, {"T:유령": 1.0, "T:암진단비": 1.0})
        assert masses.get(f"{CHUNK_PREFIX}c1", 0.0) > 0.0

    def test_all_seeds_outside_graph_returns_empty(self, graph):
        b = build_bipartite(graph, [])
        assert propagate(b, {"T:유령": 1.0}) == {}

    def test_edgeless_graph_returns_empty(self, graph):
        """엣지가 없으면 PPR 은 균등 분포를 낸다 — 그건 정보가 아니라 잡음이고,
        RRF 에 넣으면 무작위 순위가 진짜 채널을 밀어낸다."""
        b = build_bipartite(graph, [])
        assert propagate(b, {"T:암진단비": 1.0}) == {}

    def test_empty_graph_returns_empty(self):
        import networkx as nx
        assert propagate(build_bipartite(nx.MultiDiGraph(), []), {"x": 1.0}) == {}

    def test_zero_weight_seeds_return_empty(self, graph):
        """가중치 합이 0 이면 정규화가 0 나눗셈이다."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"])])
        assert propagate(b, {"T:암진단비": 0.0}) == {}

    def test_negative_seed_weights_are_dropped(self, graph):
        """코사인은 음수가 될 수 있다. 음수 reset 확률은 PPR 의 정의를 깬다."""
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비"]),
                                    _Chunk("c2", ["T:해지"])])
        masses = propagate(b, {"T:암진단비": 1.0, "T:해지": -0.5})
        assert masses.get(f"{CHUNK_PREFIX}c1", 0.0) > 0.0
        assert masses.get(f"{CHUNK_PREFIX}c2", 0.0) < masses[f"{CHUNK_PREFIX}c1"]


# ─── 4. 개체/청크 분리 ───────────────────────────────────────────────

class TestSplitMasses:
    def test_splits_by_prefix(self):
        entities, chunks = split_masses(
            {"T:암": 0.4, f"{CHUNK_PREFIX}c1": 0.6})
        assert entities == {"T:암": 0.4}
        assert chunks == {"c1": 0.6}

    def test_chunk_ids_lose_the_prefix(self):
        """호출자는 chunk_id 로 store 를 조회한다 — 접두사가 새면 조회가 실패한다."""
        _, chunks = split_masses({f"{CHUNK_PREFIX}abc123": 1.0})
        assert list(chunks) == ["abc123"]

    def test_empty_input(self):
        assert split_masses({}) == ({}, {})


# ─── 5. 검색 통합 — 채널 C ───────────────────────────────────────────

class TestRetrieverIntegration:
    """확산이 실제 검색 경로에 어떻게 붙는가. 기본값을 바꾸지 않는 것이 계약."""

    @pytest.fixture()
    def retriever(self, tmp_path):
        import networkx as nx
        from ontology.builder.models import Chunk
        from ontology.core.chunk_store import ChunkStore
        from ontology.core.graph_retrieval import GraphConditionedRetriever

        g = nx.MultiDiGraph()
        for nid, nm in (("T:암진단비", "암진단비"), ("T:암보장개시일", "암보장개시일")):
            g.add_node(nid, type="T", name=nm)

        store = ChunkStore(namespace="propns", path=tmp_path / "c.jsonl")
        # 두 개체가 같은 청크에 근거를 갖는다 — 개체-개체 엣지는 **없다**
        shared = store.add(Chunk(text="제6조 암진단비와 암보장개시일 " + "본문 " * 20,
                                 source="약관.pdf", index=0,
                                 char_start=0, char_end=100),
                           node_ids=["T:암진단비", "T:암보장개시일"])
        far = store.add(Chunk(text="제30조 암보장개시일 만 언급 " + "본문 " * 20,
                              source="약관.pdf", index=1,
                              char_start=100, char_end=200),
                        node_ids=["T:암보장개시일"])

        class _KG:
            graph = g

            def semantic_search(self, query, top_k=5):
                return [{"node_id": "T:암진단비", "score": 0.9}]

            def get_ancestors(self, node_id, predicate=None):
                return []

            def get_descendants(self, node_id, predicate=None):
                return []

        class _Index:
            def search(self, query, top_k=5):
                return []          # 청크 채널을 비워 확산의 기여만 본다

        r = GraphConditionedRetriever(namespace="propns", kg=_KG(),
                                      store=store, index=_Index())
        return r, shared, far

    def test_both_flags_default_off(self, retriever):
        """측정 전에 기본 경로를 바꾸지 않는다."""
        r, shared, far = retriever
        hits = r.search("암진단비")
        assert all("propagation" not in h.channels for h in hits)
        assert all(n["via"] != "propagation"
                   for n in r.expand("암진단비").expanded_nodes)

    def test_channel_reaches_a_chunk_the_1hop_channel_cannot(self, retriever):
        """far 청크는 진입 노드(암진단비)에 달려 있지 **않다**. 개체-개체 엣지도
        없으므로 기존 두 채널로는 영원히 못 찾는다 — 확산만이 닿는다."""
        r, shared, far = retriever
        assert far not in [h.chunk.chunk_id for h in r.search("암진단비")]
        found = r.search("암진단비", propagation_channel=True)
        assert far in [h.chunk.chunk_id for h in found]

    def test_expansion_half_does_not_add_chunk_channel(self, retriever):
        """두 절반은 따로 켠다 — 실측에서 청크 채널만 노드 채점을 퇴화시켰다
        (hit@10 0.9375→0.8750). 한 플래그로 묶으면 안전한 절반을 쓸 수 없다."""
        r, shared, far = retriever
        hits = r.search("암진단비", use_propagation=True)
        assert all("propagation" not in h.channels for h in hits)

    def test_expansion_half_adds_node_candidates(self, retriever):
        r, shared, far = retriever
        prop = [n for n in r.expand("암진단비", use_propagation=True).expanded_nodes
                if n["via"] == "propagation"]
        assert prop and "lift" in prop[0]

    def test_expansion_half_does_not_touch_terms(self, retriever):
        """확장어는 원 질의에 직접 섞이는 자리다. 실측: 확산 이름을 넣으면
        질의가 희석돼 청크 채널 결과가 바뀌고 정답이 꼬리 밖으로 나갔다."""
        r, shared, far = retriever
        assert (r.expand("암진단비", use_propagation=True).terms
                == r.expand("암진단비").terms)

    def test_records_the_channel_and_via(self, retriever):
        r, shared, far = retriever
        hits = {h.chunk.chunk_id: h
                for h in r.search("암진단비", propagation_channel=True)}
        assert "propagation" in hits[far].channels
        assert "propagation" in hits[far].matched_via

    def test_mass_does_not_pollute_best_evidence(self, retriever):
        """PPR 질량은 코사인과 비교 불가능한 척도다. 동점 해소 비교에 섞으면
        이 파일이 RRF 를 쓰는 이유(선형 결합 불가)를 스스로 위반한다."""
        r, shared, far = retriever
        hits = {h.chunk.chunk_id: h
                for h in r.search("암진단비", propagation_channel=True)}
        assert hits[far].best_evidence == 0.0

    def test_seedless_query_is_harmless(self, retriever):
        r, shared, far = retriever

        class _Empty:
            graph = r.kg.graph
            def semantic_search(self, query, top_k=5): return []
            def get_ancestors(self, n, predicate=None): return []
            def get_descendants(self, n, predicate=None): return []

        r._kg = _Empty()
        assert r.search("암진단비", propagation_channel=True) == []


class TestChunkChannelLift:
    """청크 랭킹도 lift 로 교정한다 — **실측이 요구한 비대칭 해소**.

    노드 절반(`_propagated_entities`)은 처음부터 lift 를 썼지만 청크 절반
    (`propagate_chunks`)은 raw mass 로 정렬하고 있었다. 같은 차수 편향이 청크
    쪽에 그대로 남아 있었고, evidence 채점(target="evidence", 16 케이스)이 그
    대가를 드러냈다:

        채널             hit@1    hit@5    MRR
        ───────────────────────────────────────
        retrieve         0.5000   0.8125   0.6271
        retrieve+prop    0.4375   0.7500   0.5573   ← 퇴화

    by_tag 가 어디서 무너지는지 지목했다: `graph` 0.67 → 0.33,
    `procedure` 0.50 → 0.25. 즉 여러 홉을 건너야 하는 절차 질의에서 허브 청크
    (보험계약·계약자가 달린 총칙 조문)가 정답 청크를 밀어냈다.
    """

    @pytest.fixture()
    def hub_retriever(self, tmp_path):
        """허브 청크 vs 특정 청크 — raw 와 lift 가 **반대로** 말하는 구성.

        실측 확인: raw{hub 0.323, specific 0.136} / lift{specific 1.48, hub 0.83}.
        hub 의 lift 가 1.0 미만이라는 것은 이 질의가 hub 를 배경보다 **덜**
        띄웠다는 뜻이다 — raw 정렬은 그걸 1위로 올리고 있었다.
        """
        import networkx as nx
        from ontology.builder.models import Chunk
        from ontology.core.chunk_store import ChunkStore
        from ontology.core.graph_retrieval import GraphConditionedRetriever

        g = nx.MultiDiGraph()
        for nid in ("T:암진단비", "T:계약", "T:보험료", "T:해지", "T:부활"):
            g.add_node(nid, type="T", name=nid.split(":")[1])

        store = ChunkStore(namespace="liftns", path=tmp_path / "c.jsonl")
        specific = store.add(Chunk(text="제6조 암진단비 " + "본문 " * 20,
                                   source="약관.pdf", index=0,
                                   char_start=0, char_end=100),
                             node_ids=["T:암진단비"])
        hub = store.add(Chunk(text="제1조 총칙 " + "본문 " * 20,
                              source="약관.pdf", index=1,
                              char_start=100, char_end=200),
                        node_ids=["T:암진단비", "T:계약", "T:보험료",
                                  "T:해지", "T:부활"])

        class _KG:
            graph = g

            def semantic_search(self, query, top_k=5):
                return [{"node_id": "T:암진단비", "score": 0.9}]

            def get_ancestors(self, node_id, predicate=None):
                return []

            def get_descendants(self, node_id, predicate=None):
                return []

        class _Index:
            def search(self, query, top_k=5):
                return []

        return (GraphConditionedRetriever(namespace="liftns", kg=_KG(),
                                          store=store, index=_Index()),
                specific, hub)

    def test_specific_chunk_outranks_hub(self, hub_retriever):
        """허브 청크가 1위를 차지하면 확산은 총칙만 되풀이한다."""
        r, specific, hub = hub_retriever
        ranked = [c.chunk_id for c, _ in r.propagate_chunks(r.expand("암진단비"))]
        assert ranked.index(specific) < ranked.index(hub)

    def test_ranking_equals_independent_lift_order(self, hub_retriever):
        """정의로 검증한다 — 특정 예시의 순서를 외우는 대신 lift 계약 자체를 본다."""
        from ontology.core.graph_propagation import (background_masses,
                                                     build_bipartite, lift,
                                                     propagate, split_masses)
        r, specific, hub = hub_retriever
        expansion = r.expand("암진단비")
        seeds = {e["node_id"]: e.get("score", 0.0)
                 for e in expansion.entry_nodes}
        bipartite = build_bipartite(r.kg.graph, r.store.all())
        _, chunk_masses = split_masses(propagate(bipartite, seeds))
        _, chunk_bg = split_masses(background_masses(bipartite))
        want = [cid for cid, _ in sorted(lift(chunk_masses, chunk_bg).items(),
                                         key=lambda kv: (-kv[1], kv[0]))]
        got = [c.chunk_id for c, _ in r.propagate_chunks(expansion)]
        assert got == want

    def test_score_returned_is_the_lift(self, hub_retriever):
        """호출자가 raw 질량인 줄 알고 임계값을 걸면 조용히 틀린다."""
        r, specific, hub = hub_retriever
        scored = dict((c.chunk_id, v)
                      for c, v in r.propagate_chunks(r.expand("암진단비")))
        assert scored[specific] > 1.0        # 질의가 배경 대비 띄운 청크
        assert scored[hub] < 1.0             # 허브는 오히려 내려간다

    def test_background_failure_omits_channel(self, hub_retriever,
                                              monkeypatch):
        """배경 계산이 실패하면 **생략한다** — raw 로 되돌아가면 조용히 허브
        편향 순위가 된다. 노드 절반과 같은 규정이다."""
        import ontology.core.graph_propagation as gp
        r, specific, hub = hub_retriever
        monkeypatch.setattr(gp, "background_masses", lambda *a, **kw: {})
        assert r.propagate_chunks(r.expand("암진단비")) == []

    def test_vanished_chunk_skipped(self, hub_retriever):
        """그래프를 만든 뒤 사라진 청크는 건너뛴다 (기존 계약 유지).

        ChunkStore 에 단건 삭제 API 가 없어 그 틈을 래퍼로 낸다 — all() 에는
        있는데 get() 이 None 인 상태. 걸러지지 않으면 호출자가 None.chunk_id 로
        터진다."""
        r, specific, hub = hub_retriever
        real = r.store

        class _Vanishing:
            def __getattr__(self, name):
                return getattr(real, name)

            def get(self, chunk_id):
                return None if chunk_id == hub else real.get(chunk_id)

        r._store = _Vanishing()
        ranked = [c.chunk_id for c, _ in r.propagate_chunks(r.expand("암진단비"))]
        assert hub not in ranked and specific in ranked

    def test_still_reaches_far_chunks(self, hub_retriever):
        """lift 를 넣어도 확산의 존재 이유(1-hop 이 못 닿는 곳)는 유지된다."""
        r, specific, hub = hub_retriever
        assert {specific, hub} <= {c.chunk_id
                                   for c, _ in r.propagate_chunks(
                                       r.expand("암진단비"))}


class TestPropagationWeight:
    """확산 채널의 RRF 가중 — 세 번째 채널이 직접 매칭을 밀어내지 않게.

    lift 교정으로 hit@5 는 retrieve 와 동률(0.8125)이 됐지만 hit@1 은 0.4375 vs
    0.5000 으로 남았다. per_case 가 원인을 보여줬다 — 여러 케이스가 **1 계단씩**
    밀린다(2→3, 3→4) 그리고 "언제 보험금을 주나?"가 1위→2위. RRF 가 가중 없이
    `1/(K+rank)` 를 더하므로 확산이 청크·그래프 채널과 **동등한 표**를 갖고,
    확산 단독 청크가 한 채널만 찾은 정답 앞에 끼어든다.

    HippoRAG 2 도 같은 문제를 감쇠로 다룬다 (passage 노드 reset 확률 0.05).
    """

    @pytest.fixture()
    def retriever(self, tmp_path):
        import networkx as nx
        from ontology.builder.models import Chunk
        from ontology.core.chunk_store import ChunkStore
        from ontology.core.graph_retrieval import GraphConditionedRetriever

        g = nx.MultiDiGraph()
        for nid in ("T:암진단비", "T:암보장개시일"):
            g.add_node(nid, type="T", name=nid.split(":")[1])

        store = ChunkStore(namespace="wns", path=tmp_path / "c.jsonl")
        near = store.add(Chunk(text="제6조 암진단비 " + "본문 " * 20,
                               source="약관.pdf", index=0,
                               char_start=0, char_end=100),
                         node_ids=["T:암진단비", "T:암보장개시일"])
        far = store.add(Chunk(text="제30조 암보장개시일 " + "본문 " * 20,
                              source="약관.pdf", index=1,
                              char_start=100, char_end=200),
                        node_ids=["T:암보장개시일"])

        class _KG:
            graph = g

            def semantic_search(self, query, top_k=5):
                return [{"node_id": "T:암진단비", "score": 0.9}]

            def get_ancestors(self, node_id, predicate=None):
                return []

            def get_descendants(self, node_id, predicate=None):
                return []

        class _Index:
            def search(self, query, top_k=5):
                return []          # far 는 확산 단독으로만 걸린다

        return (GraphConditionedRetriever(namespace="wns", kg=_KG(),
                                          store=store, index=_Index()),
                near, far)

    def _far_score(self, r, far, **kw):
        hits = {h.chunk.chunk_id: h
                for h in r.search("암진단비", propagation_channel=True, **kw)}
        return hits[far].score

    def test_weight_scales_contribution_linearly(self, retriever):
        """확산 단독 청크의 점수는 전부 채널 C 에서 온다 — 가중에 비례해야 한다."""
        r, near, far = retriever
        assert (self._far_score(r, far, propagation_weight=1.0)
                == pytest.approx(4 * self._far_score(
                    r, far, propagation_weight=0.25), rel=1e-3))

    def test_default_weight_is_damped(self, retriever):
        """기본값이 1.0 이면 이 클래스의 진단이 그대로 남는다.

        16 케이스 스윕(target="evidence")으로 정한 값이며, 코드 상수의 주석에
        스윕 표를 남긴다. ⚠️ 16 케이스는 작다 — 과적합 위험을 감춰선 안 된다."""
        from ontology.core.graph_retrieval import DEFAULT_PROPAGATION_WEIGHT
        assert 0.0 < DEFAULT_PROPAGATION_WEIGHT < 1.0

    def test_zero_weight_keeps_provenance_without_ranking(self, retriever):
        """가중 0 = "찾았다고 기록하되 순위에 표는 주지 않는다".
        provenance 를 잃으면 왜 걸렸는지 설명할 수 없다 (이 파일의 계약)."""
        r, near, far = retriever
        hits = {h.chunk.chunk_id: h
                for h in r.search("암진단비", propagation_channel=True,
                                  propagation_weight=0.0)}
        assert hits[far].score == 0.0
        assert "propagation" in hits[far].channels
        assert "propagation" in hits[far].matched_via

    def test_zero_weight_scores_identical_to_channel_off(self, retriever):
        """가중 0 은 순위 기여가 채널 off 와 **완전히** 같아야 한다.

        ⚠️ 처음엔 "near 의 점수는 가중에 불변"이라고 썼는데 틀렸다 — near 도
        진입 노드에 달려 있어 확산 질량을 받으므로 채널 C 에 **포함된다**.
        가중이 채널 C 항만 건드린다는 것을 검사하려면 그 항이 0 인 지점을
        채널 off 와 비교하는 것이 정확하다."""
        r, near, far = retriever

        def scores(**kw):
            return {h.chunk.chunk_id: h.score
                    for h in r.search("암진단비", **kw)}

        off = scores(propagation_channel=False)
        zero = scores(propagation_channel=True, propagation_weight=0.0)
        assert {cid: zero[cid] for cid in off} == off

    def test_weight_adds_only_the_channel_c_term(self, retriever):
        """가중을 4배 하면 채널 C 기여분만 4배 — 나머지 항은 상수다."""
        r, near, far = retriever

        def near_score(w):
            return {h.chunk.chunk_id: h.score
                    for h in r.search("암진단비", propagation_channel=True,
                                      propagation_weight=w)}[near]

        base = near_score(0.0)
        assert (near_score(1.0) - base
                == pytest.approx(4 * (near_score(0.25) - base), rel=1e-3))

    def test_damping_cannot_reorder_within_channel_c(self, retriever):
        """감쇠는 채널 C 내부 순위를 바꾸지 않는다 — 스칼라 곱이므로.
        (바뀐다면 lift 순위가 가중에 따라 흔들린다는 뜻이고 그건 결함이다.)"""
        r, near, far = retriever

        def order(w):
            return [h.chunk.chunk_id
                    for h in r.search("암진단비", propagation_channel=True,
                                      propagation_weight=w)
                    if "propagation" in h.channels]

        assert order(1.0) == order(0.1)


# ─── 6. 배경 정규화 (lift) — PageRank 차수 편향 교정 ─────────────────

class TestLift:
    """**실측이 요구한 교정**. 원 질량으로 정렬하면 상위가 전부 허브다:
    "중증 갑상선암이란?" 의 1위가 `InsuranceContract:보험계약` 이었다. PPR 질량은
    질의와 무관하게 고차수 노드에 쏠린다(PageRank 의 알려진 성질).

    질의-무관 배경 PageRank 로 나누면 **배경 대비 얼마나 올랐는가**가 남는다 —
    TF-IDF 의 IDF 와 같은 발상(전역적으로 인기 있는 것을 할인). 실측:
      중증 갑상선암 → 진단확정 4.79 · C50 3.0 · 갑상선암 2.42
      유방암 보장   → C50(유방의 악성 신생물) 3.95   ← 이 질의의 정답 노드
    """

    def test_background_is_query_independent(self, graph):
        from ontology.core.graph_propagation import background_masses
        b = build_bipartite(graph, [_Chunk("c1", ["T:암진단비", "T:해지"])])
        assert background_masses(b) == background_masses(b)

    def test_background_empty_for_edgeless(self, graph):
        from ontology.core.graph_propagation import background_masses
        assert background_masses(build_bipartite(graph, [])) == {}

    def test_equal_mass_higher_background_ranks_lower(self):
        """lift 의 정의. 같은 질량이라면 전역적으로 이미 인기 있던 노드가 뒤로
        가야 한다 — 그게 '질의 때문에 올랐다'의 의미다.

        (합성 토폴로지로 허브 역전을 주장하지 않는다: 허브가 씨드에 더 가까우면
        높은 질량이 **정당**하고, lift 가 그걸 뒤집을 이유도 없다. 실제 교정
        효과는 아래 실데이터 검증과 모듈 docstring 의 실측값에 있다.)
        """
        from ontology.core.graph_propagation import lift
        lifted = lift({"인기": 0.10, "희귀": 0.10},
                      {"인기": 0.10, "희귀": 0.02})
        assert lifted["희귀"] > lifted["인기"]

    def test_lift_is_a_ratio_not_a_difference(self):
        """차이(mass - background)로 하면 절대 질량이 큰 허브가 계속 이긴다."""
        from ontology.core.graph_propagation import lift
        lifted = lift({"큰허브": 0.50, "작은특정": 0.05},
                      {"큰허브": 0.45, "작은특정": 0.01})
        assert lifted["작은특정"] > lifted["큰허브"]

    def test_missing_background_key_is_dropped(self):
        """배경에 없는 키를 1.0 으로 가정하면 그 노드만 배율이 폭등한다."""
        from ontology.core.graph_propagation import lift
        assert lift({"a": 0.5}, {}) == {}

    def test_zero_background_is_dropped(self):
        from ontology.core.graph_propagation import lift
        assert lift({"a": 0.5}, {"a": 0.0}) == {}

    def test_empty_inputs(self):
        from ontology.core.graph_propagation import lift
        assert lift({}, {"a": 1.0}) == {} and lift({"a": 1.0}, {}) == {}
