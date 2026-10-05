"""
실험 하네스 Phase 5 — Tier 2 라우팅 실험 회귀 계약.

계약:
- **실험은 정책을 바꾸지 않는다** — 조합마다·종료 시 스냅샷 복원 (변이 사살:
  복원을 빼면 실패해야 한다). Tier 0 가 라이브 retrieval_config 를 안 바꾸는
  것과 같은 규율.
- 라벨 소스 축이 실제로 거른다 + 걸러낸 수를 **센다** (조용한 절단 금지).
- 레코드는 Tier 0 와 같은 뼈대 (layer/axes/config/golden/metrics/cost/warnings),
  layer="routing" 으로 층이 갈린다.
- 라벨 부족 조합은 **레코드 없이** skipped — 없는 수를 지어내지 않는다.
- 홀드아웃 표본 0 → measured=False (0.0 오보고 금지).
- collect_labels 하위 호환 (인자 없이 = 현행 동작).
"""

import pytest
import torch

from ontology.core.experiment import (get_experiment_store,
                                      reset_experiment_stores)
from ontology.ml.config import GNNConfig, RLConfig, SelectorConfig
from ontology.ml.experience_buffer import Experience
from ontology.ml.experiment_routing import (DEFAULT_AXES,
                                            run_routing_experiments)
from ontology.ml.training_loop import collect_labels

NS = "routens"
AGENTS = ["a0", "a1", "a2"]
TOPICS = ["날씨", "계산", "검색"]


@pytest.fixture(autouse=True)
def _clean_store():
    """스토어는 추가전용 JSONL 이라 in-memory 리셋만으로는 격리되지 않는다
    (get_experiment_store 가 디스크에서 되읽는다) — 파일도 지운다."""
    import ontology.core.experiment as ex

    def _wipe():
        reset_experiment_stores()
        path = ex._DEFAULT_DATA_DIR / f"experiments_{NS}.jsonl"
        if path.exists():
            path.unlink()

    _wipe()
    yield
    _wipe()


def _make_selector(tmp_path):
    """test_training_loop 과 같은 픽스처 — MockEmbedder 의 주제 마커가 임베딩
    방향을 정해 학습이 홀드아웃으로 일반화될 수 있는 구조."""
    from ontology.ml.intelligent_selector import IntelligentAgentSelector

    cfg = SelectorConfig(
        gnn=GNNConfig(hidden_dim=32, output_dim=16, heads=2),
        rl=RLConfig(query_embedding_dim=16, graph_embedding_dim=16,
                    history_dim=16, hidden_dim=32, max_agents=10,
                    epochs_per_update=2),
        models_dir=str(tmp_path / "models"),   # ml/models 오염 방지
        enable_background_training=False,
    )
    sel = IntelligentAgentSelector(config=cfg, auto_load=False, device="cpu")
    sel.rl_policy.register_agents(AGENTS)

    class MockEmbedder:
        def encode(self, text, convert_to_tensor=False):
            base = torch.zeros(16)
            idx = (0 if "날씨" in str(text)
                   else 1 if "계산" in str(text) else 2)
            base[idx * 4:(idx + 1) * 4] = 1.0
            torch.manual_seed(hash(text) % 2**31)
            return base + torch.randn(16) * 0.05

    sel._embedding_model = MockEmbedder()
    sel._knowledge_graph = object()
    return sel


def _exp(query, agent, reward=1.0, source="runtime", log_prob=-1.0,
         action=0, off_policy=None):
    return Experience(
        state=torch.zeros(48), action=action, reward=reward,
        next_state=torch.zeros(48), done=True,
        info={"query": query, "agent_id": agent, "source": source,
              "log_prob": log_prob, "off_policy": off_policy})


def _fill(sel, n=60, source="runtime", prefix="", off_policy=None):
    for i in range(n):
        topic = TOPICS[i % 3]
        agent = AGENTS[i % 3]
        sel.experience_buffer.add(_exp(
            f"{prefix}{topic} 관련 질의 {i}", agent, 1.0, source=source,
            log_prob=-1.1, action=sel.rl_policy._agent_to_idx[agent],
            off_policy=off_policy))


# ─── 라벨 소스 필터 ──────────────────────────────────────────────────

class TestSourceFilter:
    def test_backward_compatible_without_args(self, tmp_path):
        """인자 없이 부르면 현행 동작 — 전 소스, off_policy 포함."""
        sel = _make_selector(tmp_path)
        _fill(sel, n=6, source="runtime", prefix="R")
        _fill(sel, n=6, source="kg_checkpoint", prefix="K")
        out = collect_labels(sel.experience_buffer)
        assert len(out["labels"]) == 12
        assert out["skipped_source"] == 0
        assert out["skipped_off_policy"] == 0

    def test_source_whitelist_filters_and_counts(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=6, source="runtime", prefix="R")
        _fill(sel, n=9, source="kg_checkpoint", prefix="K")

        boot = collect_labels(sel.experience_buffer,
                              sources=["kg_checkpoint"])
        assert len(boot["labels"]) == 9
        assert boot["skipped_source"] == 6          # 뺀 수를 센다
        assert {l["source"] for l in boot["labels"]} == {"kg_checkpoint"}

        run = collect_labels(sel.experience_buffer, sources={"runtime"})
        assert len(run["labels"]) == 6
        assert run["skipped_source"] == 9

        both = collect_labels(sel.experience_buffer,
                              sources=["runtime", "kg_checkpoint"])
        assert len(both["labels"]) == 15 and both["skipped_source"] == 0

    def test_off_policy_axis_filters_and_counts(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=6, prefix="on")
        _fill(sel, n=4, prefix="off", off_policy="executed_fallback")

        with_off = collect_labels(sel.experience_buffer)
        assert len(with_off["labels"]) == 10

        without = collect_labels(sel.experience_buffer,
                                 include_off_policy=False)
        assert len(without["labels"]) == 6
        assert without["skipped_off_policy"] == 4
        assert without["skipped_source"] == 0       # 축이 섞이지 않는다

    def test_axis_counts_exclude_failures_and_no_query(self, tmp_path):
        """skipped_source 는 '쓸 수 있었는데 이 축이 뺀 라벨 수' —
        실패·질의없음이 섞이면 축의 효과 크기가 아니게 된다."""
        sel = _make_selector(tmp_path)
        sel.experience_buffer.add(_exp("q ok", "a0", 1.0, source="kg_checkpoint"))
        sel.experience_buffer.add(_exp("q bad", "a0", -0.5, source="kg_checkpoint"))
        sel.experience_buffer.add(_exp(None, "a0", 1.0, source="kg_checkpoint"))
        out = collect_labels(sel.experience_buffer, sources=["runtime"])
        assert out["labels"] == []
        assert out["skipped_source"] == 1           # 성공 라벨 1건만
        assert out["skipped_failure"] == 1
        assert out["skipped_no_query"] == 1


# ─── 러너 ────────────────────────────────────────────────────────────

class TestRunner:
    def test_policy_is_unchanged_by_experiments(self, tmp_path):
        """변이 사살 — 스냅샷 복원을 빼면 실패한다 (학습이 라이브로 샌다)."""
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        _fill(sel, n=60, source="kg_checkpoint", prefix="K")

        before = {k: v.clone()
                  for k, v in sel.rl_policy.actor.state_dict().items()}
        before_idx = dict(sel.rl_policy._agent_to_idx)

        res = run_routing_experiments(sel, NS, epochs=30)
        assert res["combos"] >= 1                    # 실제로 학습이 돌았다

        after = sel.rl_policy.actor.state_dict()
        for key, val in before.items():
            assert torch.equal(val, after[key]), f"actor.{key} 가 바뀌었다"
        assert dict(sel.rl_policy._agent_to_idx) == before_idx

    def test_default_grid_and_record_contract(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        _fill(sel, n=60, source="kg_checkpoint", prefix="K")

        res = run_routing_experiments(sel, NS, epochs=20)
        assert res["combos"] == 6                    # 소스 3 × 가중 2
        assert {r["axes"]["sources"] for r in res["records"]} == {
            "all", "kg_checkpoint", "runtime"}
        assert {r["axes"]["weighted"] for r in res["records"]} == {True, False}

        for r in res["records"]:
            assert r["layer"] == "routing"           # 층이 갈린다
            assert r["run_id"] == res["run_id"]
            assert r["namespace"] == NS
            assert set(r["metrics"]) == {"holdout_acc", "snips",
                                         "train_acc", "measured"}
            assert r["metrics"]["measured"] is True
            assert 0.0 <= r["metrics"]["holdout_acc"] <= 1.0
            assert r["cost"]["epochs"] == 20
            assert r["cost"]["train_seconds"] >= 0
            assert r["cost"]["labels"] == r["golden"]["train"]
            assert r["golden"]["labels"] == (r["golden"]["train"]
                                             + r["golden"]["holdout"])
            assert r["config"]["state_dim"] == sel.config.rl.state_dim
            assert r["config"]["max_agents"] == sel.config.rl.max_agents
            # 홀드아웃 ~24건 — 20 하한 부근, 경고는 정직하게
            assert "no_cases" not in r["warnings"]
        assert res["pareto"]
        assert all(p in res["records"] for p in res["pareto"])

    def test_small_holdout_warns(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["runtime"]], "weighted": [True]},
            holdout_ratio=0.1, epochs=10)
        assert res["combos"] == 1
        rec = res["records"][0]
        assert rec["golden"]["holdout"] < 20
        assert "small_sample" in rec["warnings"]

    def test_label_shortage_is_skipped_without_record(self, tmp_path):
        """없는 수를 지어내지 않는다 — 레코드 0, skipped 에 사유."""
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")   # kg_checkpoint 0건
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["kg_checkpoint"], ["runtime"]],
                           "weighted": [True]},
            epochs=10, min_labels=30)
        assert res["combos"] == 1
        assert [r["axes"]["sources"] for r in res["records"]] == ["runtime"]
        assert len(res["skipped"]) == 1
        sk = res["skipped"][0]
        assert sk["axes"]["sources"] == "kg_checkpoint"
        assert sk["labels"] == 0
        assert "min 30" in sk["reason"]
        # 스토어에도 레코드가 없어야 한다
        assert len(get_experiment_store(NS).entries(layer="routing")) == 1

    def test_unmeasured_holdout_reports_false(self, tmp_path, monkeypatch):
        """표본 0 → measured=False (0.0 오보고 금지) + 파레토 진입 불가."""
        import ontology.ml.experiment_routing as er

        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        monkeypatch.setattr(er, "holdout_accuracy", lambda *a, **kw: None)
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["runtime"]], "weighted": [True]},
            epochs=10)
        rec = res["records"][0]
        assert rec["metrics"]["holdout_acc"] is None
        assert rec["metrics"]["measured"] is False
        assert res["pareto"] == []                   # 못 잰 점은 프런티어 밖

    def test_records_persist_in_store(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["runtime"]], "weighted": [True, False]},
            epochs=10)
        entries = get_experiment_store(NS).entries(layer="routing")
        assert len(entries) == 2
        assert all(e["run_id"] == res["run_id"] for e in entries)
        assert all("at" in e for e in entries)       # 스토어가 자동 부여
        # 층 필터가 실제로 갈린다
        assert get_experiment_store(NS).entries(layer="retrieval") == []

    def test_axis_validation_is_loud(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=40)
        assert run_routing_experiments(
            sel, NS, axes={"오타축": [1]})["error"] == "invalid"
        assert run_routing_experiments(
            sel, NS, axes={"sources": ["runtime"]})["error"] == "invalid"
        assert run_routing_experiments(
            sel, NS, axes={"sources": [[]]})["error"] == "invalid"
        assert run_routing_experiments(
            sel, NS, axes={"weighted": ["yes"]})["error"] == "invalid"
        assert run_routing_experiments(
            sel, NS, axes={"sources": []})["error"] == "invalid"

    def test_max_combos_exceeded_is_loud(self, tmp_path):
        sel = _make_selector(tmp_path)
        _fill(sel, n=40)
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["runtime"], None],
                           "weighted": [True, False],
                           "include_off_policy": [True, False]},
            max_combos=4)
        assert res["error"] == "invalid"

    def test_off_policy_axis_reaches_labels(self, tmp_path):
        """배선 변이 — include_off_policy 축이 실제 라벨 수를 바꾼다."""
        sel = _make_selector(tmp_path)
        _fill(sel, n=45, source="runtime", prefix="on")
        _fill(sel, n=15, source="runtime", prefix="off",
              off_policy="executed_fallback")
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["runtime"]], "weighted": [True],
                           "include_off_policy": [True, False]},
            epochs=10, min_labels=10)
        assert res["combos"] == 2
        by_axis = {r["axes"]["include_off_policy"]: r for r in res["records"]}
        assert by_axis[True]["golden"]["labels"] == 60
        assert by_axis[False]["golden"]["labels"] == 45
        assert by_axis[False]["golden"]["skipped_off_policy"] == 15

    def test_holdout_split_is_shared_across_combos(self, tmp_path):
        """같은 질의는 어느 조합에서도 같은 쪽 — 조합 비교의 성립 조건."""
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        res = run_routing_experiments(
            sel, NS, axes={"sources": [["runtime"]], "weighted": [True, False]},
            epochs=10)
        holdouts = {r["golden"]["holdout"] for r in res["records"]}
        assert len(holdouts) == 1

    def test_default_axes_are_not_mutated(self, tmp_path):
        """기본 격자는 모듈 상수 — 호출이 그것을 갈아엎으면 안 된다."""
        sel = _make_selector(tmp_path)
        _fill(sel, n=60, source="runtime", prefix="R")
        snapshot = {k: list(v) for k, v in DEFAULT_AXES.items()}
        run_routing_experiments(sel, NS, epochs=5, min_labels=10)
        assert {k: list(v) for k, v in DEFAULT_AXES.items()} == snapshot
