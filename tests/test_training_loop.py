"""
RL 학습 루프 — 회귀 계약 (contextual bandit 재정식화).

계약:
- 라벨 수집: 성공 경험만, 실패·질의없음은 **세어** 버린다 (조용한 절단 금지).
- 분할: 질의 해시 기반 결정적 — 같은 데이터면 같은 홀드아웃 (재현 가능 측정).
- SNIPS: 표본 0 → None (0.0 오보고 금지). propensity 0 항목 제외.
- retrain 게이트: 홀드아웃 정확도·SNIPS 퇴화 시 **롤백** (저장 안 함),
  자 없으면 학습하지 않는다.
"""

import pytest
import torch

from ontology.ml.config import GNNConfig, RLConfig, SelectorConfig
from ontology.ml.experience_buffer import Experience
from ontology.ml.training_loop import (collect_labels, holdout_accuracy,
                                       retrain, snips_estimate, split_holdout)

AGENTS = ["a0", "a1", "a2"]


def _make_selector(tmp_path):
    from ontology.ml.intelligent_selector import IntelligentAgentSelector

    cfg = SelectorConfig(
        gnn=GNNConfig(hidden_dim=32, output_dim=16, heads=2),
        rl=RLConfig(query_embedding_dim=16, graph_embedding_dim=16,
                    history_dim=16, hidden_dim=32, max_agents=10,
                    epochs_per_update=2),
        models_dir=str(tmp_path / "models"),
        enable_background_training=False,
    )
    sel = IntelligentAgentSelector(config=cfg, auto_load=False, device="cpu")
    sel.rl_policy.register_agents(AGENTS)

    class MockEmbedder:
        """주제 마커가 임베딩 방향을 정한다 — 학습이 홀드아웃으로 일반화될
        수 있는 구조 (마커 없는 무작위 임베딩이면 라벨에 신호가 없어서
        게이트가 옳게 롤백한다 — 첫 버전 테스트가 실측으로 확인)."""

        def encode(self, text, convert_to_tensor=False):
            base = torch.zeros(16)
            idx = 0 if "날씨" in str(text) else 1 if "계산" in str(text) else 2
            base[idx * 4:(idx + 1) * 4] = 1.0
            torch.manual_seed(hash(text) % 2**31)
            return base + torch.randn(16) * 0.05

    sel._embedding_model = MockEmbedder()
    sel._knowledge_graph = object()
    return sel


def _exp(query, agent, reward, source="runtime", log_prob=-1.0, action=0,
         off_policy=None):
    return Experience(
        state=torch.zeros(48), action=action, reward=reward,
        next_state=torch.zeros(48), done=True,
        info={"query": query, "agent_id": agent, "source": source,
              "log_prob": log_prob, "off_policy": off_policy})


class TestCollectAndSplit:
    def test_collect_keeps_success_counts_rest(self, tmp_path):
        sel = _make_selector(tmp_path)
        sel.experience_buffer.add(_exp("q1", "a0", 1.0))
        sel.experience_buffer.add(_exp("q2", "a1", -0.5))          # 실패
        sel.experience_buffer.add(_exp(None, "a2", 1.0))           # 질의 없음
        sel.experience_buffer.add(_exp("q1", "a0", 1.0))           # 중복 dedup
        out = collect_labels(sel.experience_buffer)
        assert [l["query"] for l in out["labels"]] == ["q1"]
        assert out["skipped_failure"] == 1
        assert out["skipped_no_query"] == 1
        # 라벨 축 계수는 기본 호출에서 0 (전 소스·off_policy 포함 = 현행 동작)
        assert out["skipped_source"] == 0
        assert out["skipped_off_policy"] == 0

    def test_split_is_deterministic_and_leak_free(self):
        labels = [{"query": f"질의 유형 {i}", "agent": "a0", "reward": 1.0}
                  for i in range(50)]
        t1, h1 = split_holdout(labels, 0.3)
        t2, h2 = split_holdout(list(reversed(labels)), 0.3)
        assert {l["query"] for l in h1} == {l["query"] for l in h2}  # 순서 무관
        assert not ({l["query"] for l in t1} & {l["query"] for l in h1})


class TestSnips:
    def test_hand_computed(self):
        logged = [
            {"reward": 1.0, "logged_prob": 0.5, "new_prob": 1.0},  # w=2
            {"reward": 0.0, "logged_prob": 0.5, "new_prob": 0.25}, # w=0.5
        ]
        # (2*1 + 0.5*0) / (2 + 0.5) = 0.8
        assert snips_estimate(logged) == pytest.approx(0.8)

    def test_empty_and_zero_propensity_are_none(self):
        assert snips_estimate([]) is None
        assert snips_estimate([{"reward": 1.0, "logged_prob": 0.0,
                                "new_prob": 0.5}]) is None  # 0.0 오보고 금지


class TestRetrainGate:
    TOPICS = ["날씨", "계산", "검색"]

    def _fill(self, sel, n=60):
        # 주제 → 에이전트의 일관된 대응 (임베딩 방향이 주제를 담으므로
        # 홀드아웃의 새 질의에도 일반화된다)
        for i in range(n):
            topic = self.TOPICS[i % 3]
            agent = AGENTS[i % 3]
            sel.experience_buffer.add(
                _exp(f"{topic} 관련 질의 {i}", agent, 1.0,
                     source="runtime", log_prob=-1.1,
                     action=sel.rl_policy._agent_to_idx[agent]))

    def test_accept_path_learns_and_saves(self, tmp_path):
        sel = _make_selector(tmp_path)
        self._fill(sel)
        report = retrain(sel, epochs=200, min_labels=30)
        assert report["status"] == "accepted"
        assert report["holdout_acc_after"] >= (report["holdout_acc_before"] or 0)
        assert report["fit"]["train_acc"] >= 0.9        # 배웠는가
        assert (tmp_path / "models" / "policy.pt").exists()  # 저장됐는가

    def test_too_few_labels_skips(self, tmp_path):
        sel = _make_selector(tmp_path)
        self._fill(sel, n=5)
        report = retrain(sel, min_labels=30)
        assert report["status"] == "skipped"

    def test_regression_rolls_back(self, tmp_path, monkeypatch):
        """게이트 mutant 사살 — 학습이 정확도를 망가뜨리면 이전 가중치 복원."""
        sel = _make_selector(tmp_path)
        self._fill(sel)
        good = retrain(sel, epochs=200, min_labels=30)
        assert good["status"] == "accepted"
        acc_good = good["holdout_acc_after"]

        # 파괴적 '학습' — imitate 가 정책을 잡음으로 초기화한다고 가정
        def _sabotage(**kw):
            with torch.no_grad():
                for p in sel.rl_policy.actor.parameters():
                    p.copy_(torch.randn_like(p) * 3)
                for p in sel.rl_policy.feature_extractor.parameters():
                    p.copy_(torch.randn_like(p) * 3)
            return {"epochs": 0, "final_loss": 9.9, "train_acc": 0.0}

        monkeypatch.setattr(sel.rl_policy, "imitate", _sabotage)
        bad = retrain(sel, epochs=1, min_labels=30)
        assert bad["status"] == "rolled_back"
        # 롤백 후 홀드아웃 정확도가 채택본 수준으로 복원돼 있어야 한다
        labels = collect_labels(sel.experience_buffer)["labels"]
        _, hold = split_holdout(labels, 0.2)
        mask = sel.rl_policy.build_available_mask(AGENTS)
        assert holdout_accuracy(sel, hold, mask) == pytest.approx(acc_good)


class TestRetrainLabelAxes:
    """Tier 2 실험이 고른 라벨 구성을 **운영 재학습에도 같은 인자로** 걸 수
    있어야 실험 결과가 배포로 이어진다 (기본값은 현행 동작 — 회귀 0)."""

    def _mixed(self, sel, n=40):
        for i in range(n):
            topic = TestRetrainGate.TOPICS[i % 3]
            agent = AGENTS[i % 3]
            idx = sel.rl_policy._agent_to_idx[agent]
            sel.experience_buffer.add(_exp(f"R {topic} 질의 {i}", agent, 1.0,
                                           source="runtime", log_prob=-1.1,
                                           action=idx))
            sel.experience_buffer.add(_exp(f"K {topic} 질의 {i}", agent, 1.0,
                                           source="kg_checkpoint",
                                           log_prob=-1.1, action=idx))

    def test_source_axis_reaches_retrain(self, tmp_path):
        sel = _make_selector(tmp_path)
        self._mixed(sel)
        report = retrain(sel, epochs=20, min_labels=30, sources=["runtime"])
        assert report["labels"] == 40
        assert report["skipped_source"] == 40      # kg_checkpoint 를 뺀 수
        assert report["status"] in ("accepted", "rolled_back")

    def test_off_policy_axis_reaches_retrain(self, tmp_path):
        sel = _make_selector(tmp_path)
        self._mixed(sel)
        for i in range(10):
            agent = AGENTS[i % 3]
            sel.experience_buffer.add(_exp(
                f"F 검색 질의 {i}", agent, 1.0, source="runtime",
                log_prob=-1.1, action=sel.rl_policy._agent_to_idx[agent],
                off_policy="executed_fallback"))
        report = retrain(sel, epochs=20, min_labels=30,
                         include_off_policy=False)
        assert report["skipped_off_policy"] == 10
        assert report["labels"] == 80              # fallback 10건 제외
