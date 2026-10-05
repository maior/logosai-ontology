"""
P0-5·P0-6: 정본 데이터 복구 + stats 왕복.

근거: docs/rl-adoption-zero-diagnosis.md.

P0-5 — 정책은 randn 잡음 1000건으로 학습됐는데(KG 패턴 폴백이 조용히 발화),
라벨은 처음부터 있었다: kg_checkpoint.json 의 query_agent_mapping 596 노드
(query_sample 원 질의 텍스트 + selected_agent + success_rate). 여기서 고정하는
계약:
- 파서(load_kg_mappings)는 원 질의 텍스트를 보존한다 — **버퍼 info 에 질의
  텍스트가 들어가야 임베더가 바뀌어도 재임베딩으로 재사용 가능** (종전 스키마는
  임베더 교체마다 학습 데이터를 잃었다).
- randn 폴백은 기본 거부 — 잡음 학습의 재발 방지 (구 계약 의도적 교체).

P0-6 — save 는 stats 를 쓰는데 load 가 복원하지 않아 training_steps 0 이라는
거짓 지표가 진단을 지연시켰다. 왕복이 계약.
"""

import json

import networkx as nx
import pytest
import torch

from ontology.ml.bootstrap import load_kg_mappings
from ontology.ml.config import GNNConfig, RLConfig, SelectorConfig
from ontology.ml.experience_buffer import Experience, SyntheticDataGenerator


# ─── 파서: load_kg_mappings ──────────────────────────────────────────


def _checkpoint(nodes) -> dict:
    return {"graph": {"nodes": nodes, "links": []}, "metadata": {}}


def test_parser_extracts_and_expands_samples(tmp_path):
    p = tmp_path / "kg_checkpoint.json"
    p.write_text(json.dumps(_checkpoint([
        {"type": "query_agent_mapping", "query_sample": "날씨 알려줘",
         "query_samples": ["날씨 알려줘", "오늘 날씨 어때"],
         "selected_agent": "weather_agent", "success_rate": 1.0},
        {"type": "query_agent_mapping", "query_sample": "1+1은?",
         "selected_agent": "calculator_agent", "success_rate": 0.5},
        {"type": "agent", "query_sample": "무관", "selected_agent": "x"},  # 타입 다름
        {"type": "query_agent_mapping", "query_sample": "",
         "selected_agent": "weather_agent", "success_rate": 1.0},          # 빈 질의
        {"type": "query_agent_mapping", "query_sample": "요율 없음",
         "selected_agent": "weather_agent"},                               # rate 없음 — 지어내지 않는다
    ]), ensure_ascii=False))

    ms = load_kg_mappings(str(p))
    assert len(ms) == 3  # 2(samples 확장) + 1
    assert {m["query"] for m in ms} == {"날씨 알려줘", "오늘 날씨 어때", "1+1은?"}
    by_q = {m["query"]: m for m in ms}
    assert by_q["오늘 날씨 어때"]["agent"] == "weather_agent"
    assert by_q["1+1은?"]["success_rate"] == 0.5


def test_parser_dedups_query_agent_pairs(tmp_path):
    p = tmp_path / "kg.json"
    node = {"type": "query_agent_mapping", "query_sample": "같은 질의",
            "selected_agent": "a", "success_rate": 1.0}
    p.write_text(json.dumps(_checkpoint([node, dict(node)])))
    assert len(load_kg_mappings(str(p))) == 1


def test_parser_missing_file_returns_empty(tmp_path):
    assert load_kg_mappings(str(tmp_path / "없다.json")) == []


# ─── bootstrap_from_mappings ─────────────────────────────────────────


AGENTS = ["a0", "a1", "a2"]


def _make_selector(tmp_path):
    from ontology.ml.intelligent_selector import IntelligentAgentSelector

    cfg = SelectorConfig(
        gnn=GNNConfig(hidden_dim=32, output_dim=16, heads=2),
        rl=RLConfig(
            query_embedding_dim=16, graph_embedding_dim=16, history_dim=16,
            hidden_dim=32, max_agents=10, epochs_per_update=2,
        ),
        models_dir=str(tmp_path / "models"),
        min_buffer_size_for_training=4,
        training_batch_size=4,
        cold_start_buffer_size=10,
        enable_background_training=False,
    )
    sel = IntelligentAgentSelector(config=cfg, auto_load=False, device="cpu")
    sel.rl_policy.register_agents(AGENTS)

    class MockEmbedder:
        def encode(self, text, convert_to_tensor=False):
            torch.manual_seed(hash(text) % 2**31)
            return torch.randn(16)

    sel._embedding_model = MockEmbedder()
    sel._knowledge_graph = object()
    return sel


def _mappings(n=3, rate=1.0):
    return [
        {"query": f"질의 {i}", "agent": AGENTS[i % len(AGENTS)], "success_rate": rate}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_bootstrap_fills_buffer_with_query_text(tmp_path):
    sel = _make_selector(tmp_path)
    result = await sel.bootstrap_from_mappings(_mappings(3))
    assert result["loaded"] == 3
    assert sel.experience_buffer.size == 3

    exps, _, _ = sel.experience_buffer.sample(3)
    for e in exps:
        # 재임베딩 가능 계약: 질의 텍스트가 버퍼에 남는다
        assert e.info.get("query", "").startswith("질의")
        assert e.info.get("source") == "kg_checkpoint"
        assert "available_mask" in e.info


@pytest.mark.asyncio
async def test_bootstrap_reward_maps_success_rate(tmp_path):
    sel = _make_selector(tmp_path)
    await sel.bootstrap_from_mappings([
        {"query": "성공", "agent": "a0", "success_rate": 1.0},
        {"query": "실패", "agent": "a1", "success_rate": 0.0},
    ])
    exps, _, _ = sel.experience_buffer.sample(2)
    rewards = {e.info["query"]: e.reward for e in exps}
    assert rewards["성공"] == pytest.approx(sel.config.reward_success)   # 1.0
    assert rewards["실패"] == pytest.approx(sel.config.reward_failure)   # -0.5


@pytest.mark.asyncio
async def test_bootstrap_discards_noise_buffer(tmp_path):
    """P0-5c: 잡음 1000건 폐기 — clear_buffer 기본 True."""
    sel = _make_selector(tmp_path)
    noise = Experience(
        state=torch.zeros(48), action=0, reward=0.3,
        next_state=torch.zeros(48), done=True, info={"random": True},
    )
    for _ in range(5):
        sel.experience_buffer.add(noise)
    result = await sel.bootstrap_from_mappings(_mappings(2))
    assert result["discarded"] == 5
    assert sel.experience_buffer.size == 2


@pytest.mark.asyncio
async def test_bootstrap_registers_new_agents_and_trains(tmp_path):
    sel = _make_selector(tmp_path)
    ms = _mappings(8) + [{"query": "새 에이전트", "agent": "brand_new", "success_rate": 1.0}]
    result = await sel.bootstrap_from_mappings(ms, train_iterations=2)
    assert "brand_new" in sel.rl_policy._agent_to_idx
    assert result["training"]["iterations"] == 2
    assert sel.stats["training_steps"] == 2


@pytest.mark.asyncio
async def test_bootstrap_empty_is_loud(tmp_path):
    sel = _make_selector(tmp_path)
    result = await sel.bootstrap_from_mappings([])
    assert result.get("error") == "no_mappings"


@pytest.mark.asyncio
async def test_imitation_actually_learns_labels(tmp_path):
    """"배웠는가"의 최소 검증 — PPO 첫 시도의 실패(top-1 3%, confidence 0.5
    균등)를 잡았어야 할 테스트. 부트스트랩 후 정책이 학습 라벨을 재현하고
    게이트(0.7)를 실제로 넘어야 한다."""
    sel = _make_selector(tmp_path)
    ms = [
        {"query": f"고유한 질의 유형 {i}번", "agent": AGENTS[i % 3], "success_rate": 1.0}
        for i in range(20)
    ]
    result = await sel.bootstrap_from_mappings(ms, imitate_epochs=400)
    assert result["imitation"]["train_acc"] >= 0.9

    hits = 0
    passed = 0
    for m in ms:
        a, meta = await sel.select_agent(m["query"], AGENTS, deterministic=True)
        sel._pending_state = None  # 평가 루프 — pending 오염 방지
        hits += int(a == m["agent"])
        passed += int(meta["confidence"] >= sel.config.confidence_threshold)
    assert hits >= 18   # 라벨 재현
    assert passed >= 15  # 게이트가 원리적으로 도달 가능해졌다는 실증


@pytest.mark.asyncio
async def test_runtime_feedback_preserves_query_text(tmp_path):
    """P0-5 스키마 계약의 런타임 판 — select→feedback 경로의 경험에도
    질의 텍스트가 남아야 임베더 교체 시 버퍼를 잃지 않는다."""
    sel = _make_selector(tmp_path)
    await sel.select_agent("환율 알려줘", AGENTS)
    await sel.store_feedback(success=True)
    exps, _, _ = sel.experience_buffer.sample(1)
    assert exps[0].info.get("query") == "환율 알려줘"


# ─── randn 폴백 기본 거부 (구 계약 의도적 교체) ──────────────────────


def test_synthetic_random_fallback_refused_by_default():
    """잡음 1000건의 기원 — KG 패턴 0 이면 randn 을 조용히 만들었다.
    기본은 거부, allow_random=True 로만 허용 (명시적 선택)."""
    gen = SyntheticDataGenerator(state_dim=64, query_dim=32)
    empty = nx.MultiDiGraph()
    assert gen.generate(empty, AGENTS, num_samples=10) == []
    assert len(gen.generate(empty, AGENTS, num_samples=10, allow_random=True)) == 10


def test_synthetic_uses_real_patterns_from_nx_graph():
    """더 깊은 근본 원인의 회귀 계약: nx 그래프 자신의 `.graph` 속성(dict)에
    언래핑이 걸려 **진짜 그래프를 줘도 패턴 0 → 항상 randn** 이었다.
    패턴이 있으면 거부되지 않고, 패턴 유래 보상(0.8)이 실제로 나와야 한다."""
    g = nx.MultiDiGraph()
    for a in AGENTS:
        g.add_node(a, type="agent", properties={})
    g.add_node("m0", type="query_agent_mapping",
               properties={"success_rate": 0.8, "usage_count": 5, "category": "c"})
    g.add_edge("m0", AGENTS[0], predicate="selected_for")

    gen = SyntheticDataGenerator(state_dim=64, query_dim=32)
    exps = gen.generate(g, AGENTS, num_samples=30)  # allow_random 없이
    assert len(exps) > 0
    assert any(e.reward == pytest.approx(0.8) for e in exps)  # KG 패턴 유래


# ─── P0-6: stats 왕복 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_roundtrip_save_load(tmp_path):
    """training_steps 0 거짓 지표가 진단을 지연시켰다 — 왕복이 계약."""
    from ontology.ml.intelligent_selector import IntelligentAgentSelector

    sel = _make_selector(tmp_path)
    sel.stats["training_steps"] = 7
    sel.stats["selections"] = 42
    sel.save_models()

    sel2 = IntelligentAgentSelector(config=sel.config, auto_load=True, device="cpu")
    assert sel2.stats["training_steps"] == 7
    assert sel2.stats["selections"] == 42
