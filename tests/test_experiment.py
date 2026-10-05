"""실험 하네스 커널 (`core/experiment.py`) — 열거·지문·파레토·스토어·카운터.

변이 사살이 목적인 테스트가 셋 있다:
- 파레토의 **저품질·저비용 비지배점 생존** — 가중합으로 구현하면 죽는다.
- 골든셋 지문의 **accepted 민감성** — 라벨 세대(accept 로 정답이 자람)가
  hash 에 박제되지 않으면 "같은 자로 쟀는가"가 거짓이 된다.
- 그래프 지문의 **엣지 1개 민감성** — 노드 수만 해시하면 통과해버린다.
"""
import json
from types import SimpleNamespace

import pytest

from ontology.core import experiment as ex
from ontology.core import semantic_index as si


# ─── enumerate_combos ────────────────────────────────────────────────

class TestEnumerateCombos:
    def test_deterministic_and_order_preserving(self):
        """키는 정렬, 값은 입력 순서 보존 — 두 번 불러 같은 순서."""
        axes = {"b": [1, 2], "a": ["x", "y"]}
        first = ex.enumerate_combos(axes)
        second = ex.enumerate_combos(axes)
        assert first == second
        assert first == [
            {"a": "x", "b": 1}, {"a": "x", "b": 2},
            {"a": "y", "b": 1}, {"a": "y", "b": 2},
        ]

    def test_empty_axes_is_single_baseline(self):
        assert ex.enumerate_combos({}) == [{}]

    def test_over_limit_raises_with_total(self):
        """조용한 절단 금지 — 총수가 메시지에 나와야 한다."""
        axes = {"a": [1, 2, 3], "b": [1, 2, 3]}
        with pytest.raises(ValueError, match="9"):
            ex.enumerate_combos(axes, max_combos=8)

    def test_exactly_at_limit_is_allowed(self):
        axes = {"a": [1, 2], "b": [1, 2]}
        assert len(ex.enumerate_combos(axes, max_combos=4)) == 4

    def test_non_list_value_rejected(self):
        with pytest.raises(ValueError):
            ex.enumerate_combos({"a": "not-a-list"})

    def test_empty_list_value_rejected(self):
        """빈 축은 조합 0개 — 조용히 통과하면 '실험이 다 돌았다'로 오독된다."""
        with pytest.raises(ValueError):
            ex.enumerate_combos({"a": []})


# ─── graph_fingerprint ───────────────────────────────────────────────

def _chunk(chunk_id, node_ids=(), text=None):
    if text is None:
        text = "충분히 긴 본문 텍스트다. " * 20   # > DEFAULT_MIN_CHUNK_LEN
    return SimpleNamespace(chunk_id=chunk_id, text=text,
                           node_ids=list(node_ids))


def _graph(nodes=("A", "B"), edges=(("A", "B", "is_a"),)):
    import networkx as nx
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n)
    for s, t, p in edges:
        g.add_edge(s, t, predicate=p)
    return g


class TestGraphFingerprint:
    def test_same_graph_twice_same_hash(self):
        chunks = [_chunk("c1", node_ids=["A"])]
        fp1 = ex.graph_fingerprint(_graph(), chunks)
        fp2 = ex.graph_fingerprint(_graph(), chunks)
        assert fp1["hash"] == fp2["hash"]
        assert len(fp1["hash"]) == 12

    def test_one_extra_edge_changes_hash(self):
        """노드 수만 해시하는 mutant 를 죽인다."""
        base = ex.graph_fingerprint(
            _graph(nodes=("A", "B", "C")), [])
        more = ex.graph_fingerprint(
            _graph(nodes=("A", "B", "C"),
                   edges=(("A", "B", "is_a"), ("B", "C", "is_a"))), [])
        assert base["hash"] != more["hash"]
        assert more["edges"] == base["edges"] + 1

    def test_renamed_node_changes_hash(self):
        a = ex.graph_fingerprint(_graph(nodes=("A", "B"), edges=()), [])
        b = ex.graph_fingerprint(_graph(nodes=("A", "B2"), edges=()), [])
        assert a["hash"] != b["hash"]

    def test_counts_reuse_evidence_gaps(self):
        """linked/orphan 수치가 graph_health 와 같은 정의여야 한다."""
        chunks = [_chunk("c1", node_ids=["A"]), _chunk("c2")]
        fp = ex.graph_fingerprint(_graph(nodes=("A", "B"), edges=()), chunks)
        assert fp["nodes"] == 2
        assert fp["chunks"] == 2
        assert fp["linked_chunks"] == 1
        assert fp["coverage"] == 0.5
        assert fp["orphan_nodes"] == 1        # B 는 근거 청크 0

    def test_zero_chunks_coverage_is_none(self):
        """0/0 을 1.0 이나 0.0 으로 보고하면 빈 NS 가 '건강'으로 보인다."""
        fp = ex.graph_fingerprint(_graph(), [])
        assert fp["coverage"] is None
        assert fp["chunks"] == 0


# ─── golden_fingerprint ──────────────────────────────────────────────

def _case(case_id, status="draft", expected="Type:이름", accepted=()):
    return SimpleNamespace(case_id=case_id, status=status,
                           expected_node_id=expected,
                           accepted=list(accepted))


class TestGoldenFingerprint:
    def test_counts_and_statuses(self):
        fp = ex.golden_fingerprint([
            _case("c1"), _case("c2", status="verified"),
            _case("c3", status="verified")])
        assert fp["n"] == 3
        assert fp["statuses"] == {"draft": 1, "verified": 2}
        assert len(fp["hash"]) == 12

    def test_accept_changes_hash(self):
        """라벨 노후화 법칙의 박제 — 정답 집합이 자라면 자가 달라진 것이다."""
        before = ex.golden_fingerprint([_case("c1")])
        after = ex.golden_fingerprint(
            [_case("c1", accepted=["Type:추가정답"])])
        assert before["hash"] != after["hash"]
        assert before["n"] == after["n"]      # n 만으로는 못 잡는다

    def test_case_order_does_not_matter(self):
        a, b = _case("c1"), _case("c2", status="verified")
        assert ex.golden_fingerprint([a, b])["hash"] == \
            ex.golden_fingerprint([b, a])["hash"]

    def test_dict_cases_supported(self):
        """스토어 밖(JSON 직렬화 이후)에서도 지문을 낼 수 있어야 한다."""
        fp_obj = ex.golden_fingerprint([_case("c1", accepted=["X"])])
        fp_dict = ex.golden_fingerprint([{
            "case_id": "c1", "status": "draft",
            "expected_node_id": "Type:이름", "accepted": ["X"]}])
        assert fp_obj["hash"] == fp_dict["hash"]

    def test_empty_cases(self):
        fp = ex.golden_fingerprint([])
        assert fp["n"] == 0 and fp["statuses"] == {}


# ─── sample_warnings ─────────────────────────────────────────────────

class TestSampleWarnings:
    def test_zero_cases(self):
        assert ex.sample_warnings(0) == ["no_cases", "small_sample"]

    def test_below_minimum(self):
        assert ex.sample_warnings(47) == ["small_sample"]

    def test_at_minimum_is_clean(self):
        assert ex.sample_warnings(50) == []

    def test_custom_minimum(self):
        assert ex.sample_warnings(10, min_cases=10) == []
        assert ex.sample_warnings(9, min_cases=10) == ["small_sample"]


# ─── pareto_frontier ─────────────────────────────────────────────────

def _rec(name, mrr, latency, measured=True):
    return {"name": name, "measured": measured,
            "metrics": {"mrr": mrr}, "cost": {"latency_ms_p50": latency}}


class TestParetoFrontier:
    def test_hand_computed_three_points(self):
        """A(0.9, 100) · B(0.5, 10) · C(0.6, 120) — C 만 지배당한다."""
        a, b, c = _rec("A", 0.9, 100), _rec("B", 0.5, 10), _rec("C", 0.6, 120)
        frontier = ex.pareto_frontier([a, b, c])
        names = [r["name"] for r in frontier]
        assert names == ["B", "A"]            # 비용 오름차순

    def test_cheap_low_quality_point_survives(self):
        """**가중합 mutant 사살** — 품질이 낮아도 비용이 압도적으로 싸면
        프런티어에 남는다. 단일 스칼라로 합치면 B 가 떨어진다."""
        frontier = ex.pareto_frontier([_rec("A", 0.95, 500),
                                       _rec("B", 0.10, 1)])
        assert {r["name"] for r in frontier} == {"A", "B"}

    def test_duplicate_points_both_stay(self):
        """동률(같은 품질·같은 비용)은 서로 지배하지 않는다 — strict 조건."""
        frontier = ex.pareto_frontier([_rec("A", 0.5, 10), _rec("B", 0.5, 10)])
        assert {r["name"] for r in frontier} == {"A", "B"}

    def test_cost_tie_quality_differs(self):
        """비용 동률이면 품질 높은 쪽이 낮은 쪽을 지배한다 (≤ 비용, > 품질)."""
        frontier = ex.pareto_frontier([_rec("A", 0.9, 10), _rec("B", 0.5, 10)])
        assert [r["name"] for r in frontier] == ["A"]

    def test_unmeasured_excluded_even_if_dominant(self):
        """measured=False 는 지배적이어도 프런티어에 못 든다 — 재지 않은
        결과가 추천 후보가 되면 안 된다."""
        frontier = ex.pareto_frontier([_rec("A", 0.99, 1, measured=False),
                                       _rec("B", 0.5, 10)])
        assert [r["name"] for r in frontier] == ["B"]

    def test_none_metric_excluded(self):
        frontier = ex.pareto_frontier([_rec("A", None, 1), _rec("B", 0.5, 10)])
        assert [r["name"] for r in frontier] == ["B"]

    def test_missing_cost_excluded(self):
        rec = {"name": "A", "measured": True, "metrics": {"mrr": 0.9},
               "cost": {}}
        frontier = ex.pareto_frontier([rec, _rec("B", 0.5, 10)])
        assert [r["name"] for r in frontier] == ["B"]

    def test_custom_keys(self):
        recs = [{"name": "A", "measured": True,
                 "metrics": {"hit@1": 0.7}, "cost": {"embed_calls": 3}},
                {"name": "B", "measured": True,
                 "metrics": {"hit@1": 0.6}, "cost": {"embed_calls": 9}}]
        frontier = ex.pareto_frontier(recs, quality="hit@1",
                                      cost="embed_calls")
        assert [r["name"] for r in frontier] == ["A"]

    def test_empty_records(self):
        assert ex.pareto_frontier([]) == []


# ─── ExperimentStore ─────────────────────────────────────────────────

class TestExperimentStore:
    def test_roundtrip_and_newest_first(self, tmp_path):
        path = tmp_path / "exp.jsonl"
        store = ex.ExperimentStore(namespace="t", path=path)
        store.record({"layer": "tier0", "name": "first"})
        store.record({"layer": "tier0", "name": "second"})

        reloaded = ex.ExperimentStore(namespace="t", path=path)
        assert reloaded.load_from_disk() is True
        rows = reloaded.entries()
        assert [r["name"] for r in rows] == ["second", "first"]
        assert all(r["at"] for r in rows)     # 타임스탬프 자동

    def test_append_only(self, tmp_path):
        path = tmp_path / "exp.jsonl"
        store = ex.ExperimentStore(namespace="t", path=path)
        store.record({"name": "a"})
        store.record({"name": "b"})
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2
        assert len(store.entries()) == 2

    def test_broken_line_skipped(self, tmp_path):
        path = tmp_path / "exp.jsonl"
        path.write_text(json.dumps({"name": "ok"}) + "\n"
                        + "{broken json\n"
                        + json.dumps({"name": "ok2"}) + "\n",
                        encoding="utf-8")
        store = ex.ExperimentStore(namespace="t", path=path)
        assert store.load_from_disk() is True
        assert [r["name"] for r in store.entries()] == ["ok2", "ok"]

    def test_layer_filter_and_limit(self, tmp_path):
        store = ex.ExperimentStore(namespace="t", path=tmp_path / "e.jsonl")
        store.record({"layer": "tier0", "name": "a"})
        store.record({"layer": "tier1", "name": "b"})
        store.record({"layer": "tier0", "name": "c"})
        assert [r["name"] for r in store.entries(layer="tier0")] == ["c", "a"]
        assert [r["name"] for r in store.entries(limit=1)] == ["c"]
        assert len(store.entries(limit=0)) == 3

    def test_write_failure_is_harmless(self, tmp_path):
        """기록 실패가 실험을 죽이지 않는다 — 측정이 본체다."""
        store = ex.ExperimentStore(namespace="t",
                                   path=tmp_path)   # 디렉터리 → open 실패
        entry = store.record({"name": "a"})
        assert entry["name"] == "a"
        assert [r["name"] for r in store.entries()] == ["a"]  # 메모리엔 남는다

    def test_singleton_and_reset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ex, "_DEFAULT_DATA_DIR", tmp_path)
        ex.reset_experiment_stores()
        a = ex.get_experiment_store("ns1")
        assert ex.get_experiment_store("ns1") is a
        assert ex.get_experiment_store("ns2") is not a
        ex.reset_experiment_stores()
        assert ex.get_experiment_store("ns1") is not a

    def test_default_path_shape(self):
        store = ex.ExperimentStore(namespace="probe")
        assert store.path.name == "experiments_probe.jsonl"


# ─── count_embeds (semantic_index 카운터) ────────────────────────────

class TestCountEmbeds:
    """실모델 금지 — 임베더 캐시에 가짜를 심어 검증한다."""

    @pytest.fixture(autouse=True)
    def _fake_embedder(self, monkeypatch):
        self.raw_calls = []

        def _fake_build(model_id):
            def fake_embed(texts):
                self.raw_calls.append(list(texts))
                return [[0.1, 0.2]] * len(texts)
            return fake_embed

        monkeypatch.setattr(si, "_build_embed_fn", _fake_build)
        si.reset_embedders()
        yield
        si.reset_embedders()

    def test_counts_one_element_per_call(self):
        fn = si.get_embed_fn("fake/model")
        with si.count_embeds() as calls:
            fn(["a"])
            fn(["b", "c"])
        assert len(calls) == 2

    def test_outside_context_is_not_counted(self):
        fn = si.get_embed_fn("fake/model")
        with si.count_embeds() as calls:
            fn(["a"])
        fn(["b"])                             # 컨텍스트 밖
        assert len(calls) == 1
        assert len(self.raw_calls) == 2       # 원 함수에는 둘 다 도달

    def test_no_context_passthrough(self):
        """카운터 없이(라이브 기본) 결과가 원 함수와 같아야 한다."""
        fn = si.get_embed_fn("fake/model")
        assert fn(["x", "y"]) == [[0.1, 0.2], [0.1, 0.2]]
        assert self.raw_calls == [["x", "y"]]

    def test_nested_contexts_are_independent(self):
        fn = si.get_embed_fn("fake/model")
        with si.count_embeds() as outer:
            fn(["a"])
            with si.count_embeds() as inner:
                fn(["b"])
                fn(["c"])
            fn(["d"])
        assert len(inner) == 2
        assert len(outer) == 2                # 안쪽 구간은 안쪽 것

    def test_identity_contract_preserved(self):
        """캐시 계약(같은 모델 = 같은 객체)이 래핑 후에도 유지돼야 한다."""
        assert si.get_embed_fn("fake/model") is si.get_embed_fn("fake/model")

    def test_raw_cache_is_not_wrapped(self):
        """프로세스 캐시에는 원 함수만 — 감싼 함수가 저장되면 카운터 참조가
        캐시에 박제된다."""
        si.get_embed_fn("fake/model")
        raw = si._embedders["fake/model"]
        assert not hasattr(raw, "__wrapped__")

    def test_none_embedder_stays_none(self, monkeypatch):
        monkeypatch.setattr(si, "_build_embed_fn", lambda mid: None)
        si.reset_embedders()
        assert si.get_embed_fn("gone") is None
