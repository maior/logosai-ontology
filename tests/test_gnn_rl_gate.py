"""
P0-4: GNN+RL 게이트 자 재정의 — confidence 를 1/N 병리에서 꺼낸다.

근거: docs/rl-adoption-zero-diagnosis.md.

종전 자: "샘플링된 액션의 softmax 확률". 액션이 N 개면 균등분포에서 ≈1/N
(실측 평균 0.0137 ≈ 1/73) — 게이트 0.7 과 50배 갭, 원리적 도달 불가.

새 자: 최댓값의 상대 마진 p1/(p1+p2).
- 균등분포 = 0.5, **N 무관** (자에서 N 을 제거하는 것이 수리의 본체)
- 한 액션 집중 → 1.0
- kg_confidence(0..1 루브릭)와 같은 축

냉시동 감쇠도 함께 재정의: 0 이 아니라 무정보점 0.5 로 당긴다. 빈 버퍼의
상한이 0.65 < 게이트 0.7 — **실데이터 없이는 채택 불가가 수식으로 보장**된다.
(임계를 낮춰 채택률을 만드는 것 — 진단의 "하지 말 것" — 과 정반대 방향.)
"""

import pytest
import torch

from ontology.ml.config import GNNConfig, RLConfig, SelectorConfig
from ontology.ml.rl_policy import action_confidence


# ─── 자(action_confidence) 자체 ──────────────────────────────────────


def test_uniform_is_half_regardless_of_n():
    """핵심 성질: 무정보(균등)면 N 이 몇이든 0.5 — 1/N 병리의 제거."""
    for n in (2, 10, 78, 118):
        probs = torch.full((n,), 1.0 / n)
        assert action_confidence(probs) == pytest.approx(0.5)


def test_concentrated_approaches_one():
    probs = torch.tensor([0.9] + [0.1 / 9] * 9)
    conf = action_confidence(probs)
    assert conf == pytest.approx(0.9 / (0.9 + 0.1 / 9))
    assert conf > 0.9


def test_gate_meaning_p1_over_2_33x_p2():
    """게이트 0.7 = 1위가 2위의 2.33배 이상 — 학습된 정책이 도달 가능한 값."""
    probs = torch.tensor([0.7, 0.3])
    assert action_confidence(probs) == pytest.approx(0.7)


def test_single_candidate_is_one():
    # 마스킹으로 후보가 하나면 나머지 확률 ≈ 0 → 비교 대상 없음 → 1.0
    probs = torch.tensor([1.0])
    assert action_confidence(probs) == pytest.approx(1.0)


def test_degenerate_inputs():
    assert action_confidence(torch.tensor([])) == 0.0
    assert action_confidence(torch.zeros(5)) == 0.0


# ─── 냉시동 감쇠 (0.5 로 당김) ───────────────────────────────────────


def _make_selector(tmp_path, sample_agents):
    from ontology.ml.intelligent_selector import IntelligentAgentSelector

    cfg = SelectorConfig(
        gnn=GNNConfig(hidden_dim=32, output_dim=16, heads=2),
        rl=RLConfig(
            query_embedding_dim=16, graph_embedding_dim=16, history_dim=16,
            hidden_dim=32, max_agents=10, epochs_per_update=2,
        ),
        models_dir=str(tmp_path / "models"),
        cold_start_buffer_size=50,
        enable_background_training=False,
    )
    sel = IntelligentAgentSelector(config=cfg, auto_load=False, device="cpu")
    sel.rl_policy.register_agents(sample_agents)

    class MockEmbedder:
        def encode(self, text, convert_to_tensor=False):
            torch.manual_seed(hash(text) % 2**31)
            return torch.randn(16)

    sel._embedding_model = MockEmbedder()
    sel._knowledge_graph = object()  # graph 속성 없음 → zero 벡터 경로
    return sel


AGENTS = ["a0", "a1", "a2", "a3", "a4"]


def test_cold_start_pulls_toward_half_not_zero(tmp_path):
    sel = _make_selector(tmp_path, AGENTS)
    assert sel.experience_buffer.size == 0
    # 완전 확신(1.0)도 빈 버퍼에서는 0.65 — 게이트 0.7 미달이 수식으로 보장
    assert sel._calibrate_confidence(1.0) == pytest.approx(0.65)
    # 무정보(0.5)는 감쇠해도 0.5 — 0 으로 곱하던 종전 방식과의 차이
    assert sel._calibrate_confidence(0.5) == pytest.approx(0.5)


def test_warm_buffer_is_identity(tmp_path):
    from ontology.ml.experience_buffer import Experience

    sel = _make_selector(tmp_path, AGENTS)
    exp = Experience(
        state=torch.zeros(48), action=0, reward=1.0,
        next_state=torch.zeros(48), done=True, info={},
    )
    for _ in range(50):  # cold_start_buffer_size
        sel.experience_buffer.add(exp)
    assert sel._calibrate_confidence(0.9) == pytest.approx(0.9)


# ─── select_agent 배선 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fresh_policy_never_passes_gate(tmp_path):
    """미학습(≈균등) 정책은 새 자에서도 게이트를 못 넘는다 — 잡음 정책의
    랜덤 채택이 생기지 않음을 보장 (진단의 '하지 말 것' 준수 증명)."""
    sel = _make_selector(tmp_path, AGENTS)
    for i in range(10):
        agent, meta = await sel.select_agent(f"질의 {i}", AGENTS)
        assert agent in AGENTS
        assert 0.0 <= meta["confidence"] <= 1.0
        assert meta["confidence"] < sel.config.confidence_threshold


@pytest.mark.asyncio
async def test_confident_policy_adopts_argmax(tmp_path):
    """확신이 게이트를 넘으면 탐욕(argmax) — 채택할 거면 최선을 채택한다."""
    from ontology.ml.experience_buffer import Experience

    sel = _make_selector(tmp_path, AGENTS)
    # 냉시동 탈출 (버퍼 채움)
    exp = Experience(
        state=torch.zeros(48), action=0, reward=1.0,
        next_state=torch.zeros(48), done=True, info={},
    )
    for _ in range(50):
        sel.experience_buffer.add(exp)
    # actor 를 idx 2 집중으로 조작 (weight 0 + bias → logits = bias)
    with torch.no_grad():
        sel.rl_policy.actor.logits.weight.zero_()
        sel.rl_policy.actor.logits.bias.zero_()
        sel.rl_policy.actor.logits.bias[2] = 10.0

    for i in range(5):  # 샘플링이었다면 5회 연속 동일 선택은 우연이 아님
        agent, meta = await sel.select_agent(f"질의 {i}", AGENTS)
        assert agent == "a2"
        assert meta["confidence"] >= sel.config.confidence_threshold


@pytest.mark.asyncio
async def test_metadata_carries_both_rulers(tmp_path):
    """confidence(새 자)와 raw_confidence(선택 액션의 확률)를 함께 보고 —
    운영 중 두 자를 대조할 수 있어야 자 교체의 효과가 측정된다."""
    sel = _make_selector(tmp_path, AGENTS)
    _, meta = await sel.select_agent("질의", AGENTS)
    assert "margin_confidence" in meta
    assert "raw_confidence" in meta
    assert 0.0 <= meta["raw_confidence"] <= 1.0
