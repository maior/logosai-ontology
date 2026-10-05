"""
실험 하네스 Phase 2 — 러너 배선 회귀 계약.

- 기본 격자 = 채널 3종 (vector / graph / graph+prop) — "온톨로지 vs 벡터"가
  1급 축.
- **eval_history 우회** — 실험 레코드가 운영 품질 블록을 오염시키지 않는다.
- 레코드에 지문 3종 + 표본 경고 + 비용(지연 p50/p95·embed·llm=0) 필수.
- 축 검증은 소리내는 거부 (오타 축이 조용히 무시되면 "돌았다"로 오독).
- combo 오버라이드가 retriever.search kwargs 에 실제로 닿는다 (배선 변이).
"""

from types import SimpleNamespace

import pytest

from ontology.builder.models import Chunk

NS = "expns"


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.chunk_index as ci
    import ontology.core.chunk_store as cs
    import ontology.core.experiment as ex
    import ontology.core.graph_retrieval as gr
    import ontology.core.search_qa as sq
    from ontology.core.chunk_store import reset_chunk_stores
    from ontology.core.experiment import reset_experiment_stores
    from ontology.core.search_qa import reset_golden_sets
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(sq, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(ex, "_DEFAULT_DATA_DIR", tmp_path)
    import ontology.core.eval_history as eh
    monkeypatch.setattr(eh, "_DEFAULT_DATA_DIR", tmp_path)
    reset_chunk_stores()
    reset_golden_sets()
    reset_experiment_stores()
    kgc._kg_instances.pop(NS, None)

    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node("T:암", type="T", name="암")
    engine.graph.add_node("T:계약", type="T", name="계약")

    store = cs.get_chunk_store(NS)
    store.clear()
    c1 = store.add(Chunk(text="암 보장 조항", source="w.pdf", index=0,
                         char_start=0, char_end=10), node_ids=["T:암"])
    c2 = store.add(Chunk(text="계약 해지 조항", source="w.pdf", index=1,
                         char_start=10, char_end=20), node_ids=["T:계약"])

    golden = sq.get_golden_set(NS)
    golden.add("암은 보장되나?", "T:암", status="confirmed")
    golden.add("계약 해지는?", "T:계약", status="confirmed")

    # 가짜 검색기들 — 결정적: c1 을 항상 1위로
    class FakeIndex:
        def search(self, query, top_k=5):
            return [{"chunk_id": c1}, {"chunk_id": c2}][:top_k]

    captured_cfg = []

    class FakeRetriever:
        def __init__(self, namespace=None):
            self.store = store

        def search(self, query, top_k=5, **cfg):
            captured_cfg.append(cfg)
            return [{"chunk_id": c1}, {"chunk_id": c2}][:top_k]

        def expand(self, query, **kw):
            return SimpleNamespace(entry_nodes=["T:암"],
                                   expanded_nodes=["T:계약"], via={})

    monkeypatch.setattr(ci, "get_chunk_index", lambda ns: FakeIndex())
    monkeypatch.setattr(gr, "GraphConditionedRetriever", FakeRetriever)
    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    return service, captured_cfg, tmp_path


class TestRunner:
    def test_default_grid_runs_three_channels(self, svc):
        service, captured, tmp = svc
        res = service.run_retrieval_experiments(NS)
        assert res["combos"] == 3
        channels = {r["axes"]["channel"] for r in res["records"]}
        assert channels == {"vector", "graph", "graph+prop"}
        for r in res["records"]:
            assert r["cost"]["llm_calls"] == 0            # 계약
            assert r["cost"]["queries"] == 2
            assert r["graph"]["hash"] and r["golden"]["hash"]
            assert "small_sample" in r["warnings"]        # 2케이스 — 정직 경고
            assert r["metrics"]["measured"] is True
        assert res["pareto"]                              # 프런티어 동봉

    def test_eval_history_is_bypassed(self, svc):
        """실험이 운영 품질 블록(latest_quality)을 오염시키면 안 된다."""
        service, captured, tmp = svc
        service.run_retrieval_experiments(NS)
        assert not (tmp / f"evalhistory_{NS}.jsonl").exists()
        assert (tmp / f"experiments_{NS}.jsonl").exists()

    def test_overrides_reach_search_and_channels_force_propagation(self, svc):
        service, captured, tmp = svc
        res = service.run_retrieval_experiments(
            NS, axes={"channel": ["graph", "graph+prop"], "max_terms": [2]})
        assert res["combos"] == 2
        assert all(c.get("max_terms") == 2 for c in captured)
        props = {c["use_propagation"] for c in captured}
        assert props == {True, False}                     # 채널이 확산을 강제

    def test_axis_validation_is_loud(self, svc):
        service, *_ = svc
        assert service.run_retrieval_experiments(
            NS, axes={"오타축": [1]})["error"] == "invalid"
        assert service.run_retrieval_experiments(
            NS, axes={"channel": ["warp"]})["error"] == "invalid"
        assert service.run_retrieval_experiments(
            NS, axes={"entry_k": [0]})["error"] == "invalid"   # _SPEC 범위 밖
        assert service.run_retrieval_experiments(
            NS, axes={"k": [99]})["error"] == "invalid"

    def test_max_combos_exceeded_is_loud(self, svc):
        service, *_ = svc
        res = service.run_retrieval_experiments(
            NS, axes={"entry_k": [1, 2, 3], "max_terms": [1, 2, 3]},
            max_combos=4)
        assert res["error"] == "invalid"

    def test_recommendation_stale_without_matching_graph(self, svc):
        """그래프 지문 불일치 → 추천 거부 (knob 번복 이력의 코드화)."""
        service, captured, tmp = svc
        service.run_retrieval_experiments(NS)
        from ontology.engines import knowledge_graph_clean as kgc
        kgc.get_knowledge_graph_engine(NS).graph.add_node("T:신규", type="T")
        rec = service.recommend_retrieval_config(NS)
        assert rec["stale"] is True
        assert rec["suggestion"] is None

    def test_recommendation_withheld_on_small_sample(self, svc):
        """표본 2케이스 — 제안 대신 경고 (47케이스 과적합 교훈)."""
        service, captured, tmp = svc
        service.run_retrieval_experiments(NS)
        rec = service.recommend_retrieval_config(NS)
        assert rec["stale"] is False
        assert rec["suggestion"] is None
        assert "small_sample" in rec["warnings"]
        assert rec["frontier"]                      # 프런티어는 그래도 보여준다

    def test_recommendation_suggests_dominating_combo(self, svc):
        """표본이 충분하고 지배 조합이 있으면 overrides diff 를 제안한다."""
        service, captured, tmp = svc
        from ontology.core.chunk_store import get_chunk_store
        from ontology.core.experiment import (get_experiment_store,
                                              graph_fingerprint)
        from ontology.engines import knowledge_graph_clean as kgc
        gfp = graph_fingerprint(kgc.get_knowledge_graph_engine(NS).graph,
                                get_chunk_store(NS).all())
        store = get_experiment_store(NS)

        def _rec(channel, mrr, p50, axes_extra=None, cfg_extra=None):
            store.record({
                "run_id": "exp-x", "namespace": NS, "layer": "retrieval",
                "axes": {"channel": channel, **(axes_extra or {})},
                "config": {"channel": channel, **(cfg_extra or {})},
                "graph": gfp, "golden": {"cases": 65, "hash": "g"},
                "metrics": {"mrr": mrr, "measured": True},
                "cost": {"latency_ms_p50": p50}, "warnings": []})

        # 현재 운영(graph, 오버라이드 없음)과 일치하는 레코드 + 지배 조합
        _rec("graph", 0.70, 50.0)
        _rec("graph+prop", 0.80, 55.0,
             axes_extra={"propagation_weight": 0.25},
             cfg_extra={"propagation_weight": 0.25})
        rec = service.recommend_retrieval_config(NS)
        assert rec["stale"] is False and not rec["warnings"]
        s = rec["suggestion"]
        assert s is not None
        assert s["overrides"]["propagation_weight"] == 0.25
        assert s["overrides"]["use_propagation"] is True
        assert s["delta"]["mrr"] == [0.70, 0.80]
        assert "retrieval-config" in s["apply_via"]   # 적용은 기존 경로

    def test_records_persist_and_are_queryable(self, svc):
        service, captured, tmp = svc
        run = service.run_retrieval_experiments(NS)
        got = service.get_experiments(NS)
        assert len(got["entries"]) == 3
        assert got["entries"][0]["run_id"] == run["run_id"]
        assert service.get_experiments(NS, layer="routing")["entries"] == []


class TestIndexStatusShowsNamespaceConfig:
    """index_status 의 검색 knob 은 **그 네임스페이스의 실효 설정**이어야
    한다. 종전에는 config_fingerprint() 를 인자 없이 불러 전역 기본값을
    보여줬다 — PROJ-A 은 확산 on 인데 화면은 false 로 보고했다(실측).
    "지금 무엇이 쓰이는가"를 틀리게 보여주는 것은 이 저장소의 핵심 결함류.
    """

    def test_embedding_info_reflects_namespace_overrides(self, svc, monkeypatch):
        import ontology.core.retrieval_config as rc
        from ontology.core.retrieval_config import (get_retrieval_config,
                                                    reset_retrieval_configs)
        service, captured, tmp = svc
        monkeypatch.setattr(rc, "_DEFAULT_DATA_DIR", tmp)
        reset_retrieval_configs()
        get_retrieval_config(NS).set({"use_propagation": True,
                                      "max_terms": 3}, actor="t")

        default_info = service.embedding_info()
        ns_info = service.embedding_info(NS)
        assert default_info["retrieval"]["use_propagation"] is False  # 전역
        assert ns_info["retrieval"]["use_propagation"] is True        # 실효
        assert ns_info["retrieval"]["max_terms"] == 3
        reset_retrieval_configs()
