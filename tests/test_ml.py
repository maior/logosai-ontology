"""
ML Module Tests — GNN+RL agent selection system.

Tests: config, SumTree, ExperienceBuffer, KGTensorConverter, GNNEncoder,
       HistoryEncoder, RLPolicy, IntelligentAgentSelector, SyntheticDataGenerator.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import networkx as nx

from ontology.ml.config import (
    BufferConfig,
    GNNConfig,
    RLConfig,
    SelectorConfig,
)
from ontology.ml.experience_buffer import (
    Experience,
    ExperienceBuffer,
    SumTree,
    SyntheticDataGenerator,
)
from ontology.ml.gnn_encoder import GNNEncoder, KGTensorConverter
from ontology.ml.rl_policy import HistoryEncoder, RLPolicy


# ─── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_AGENTS = [
    "internet_agent",
    "weather_agent",
    "shopping_agent",
    "calculator_agent",
    "scheduler_agent",
]


@pytest.fixture
def sample_agents():
    return SAMPLE_AGENTS


@pytest.fixture
def mock_knowledge_graph():
    """Small KG with 5 agents, 3 mappings, capabilities, and categories."""
    g = nx.MultiDiGraph()

    # Agent nodes
    for i, aid in enumerate(SAMPLE_AGENTS):
        g.add_node(
            aid,
            type="agent",
            properties={
                "name": aid.replace("_", " ").title(),
                "description": f"Agent {i}",
                "capabilities": [f"cap_{i}"],
                "tags": [f"tag_{i}"],
                "is_available": True,
                "success_rate": 0.7 + i * 0.05,
                "usage_count": 10 + i * 5,
                "created_at": "2026-01-01T00:00:00",
            },
        )

    # Query-agent mapping nodes
    for i in range(3):
        mapping_id = f"mapping_{i}"
        g.add_node(
            mapping_id,
            type="query_agent_mapping",
            properties={
                "category": ["weather", "shopping", "math"][i],
                "success_rate": 0.8,
                "usage_count": 5,
                "created_at": "2026-01-15T00:00:00",
            },
        )
        # Connect mapping → agent
        g.add_edge(mapping_id, SAMPLE_AGENTS[i + 1], predicate="selected_for")
        g.add_edge(SAMPLE_AGENTS[i + 1], mapping_id, predicate="handles")

    # Capability nodes
    for i, aid in enumerate(SAMPLE_AGENTS):
        cap_id = f"cap_{aid}"
        g.add_node(cap_id, type="capability", properties={"name": f"capability_{i}"})
        g.add_edge(aid, cap_id, predicate="has_capability")

    return g


@pytest.fixture
def small_gnn_config():
    """Small GNN config for fast testing."""
    return GNNConfig(hidden_dim=32, output_dim=16, heads=2)


@pytest.fixture
def small_rl_config():
    """Small RL config for fast testing."""
    return RLConfig(
        state_dim=48,  # 16 (query) + 16 (graph) + 16 (history)
        query_embedding_dim=16,
        graph_embedding_dim=16,
        history_dim=16,
        hidden_dim=32,
        max_agents=10,
        epochs_per_update=2,
    )


@pytest.fixture
def small_selector_config(small_gnn_config, small_rl_config, tmp_path):
    """Small config for IntelligentAgentSelector testing."""
    return SelectorConfig(
        gnn=small_gnn_config,
        rl=small_rl_config,
        buffer=BufferConfig(capacity=1000),
        models_dir=str(tmp_path / "models"),
        min_buffer_size_for_training=8,
        training_batch_size=8,
        cold_start_buffer_size=50,
        enable_background_training=False,
    )


# ─── Config Tests ─────────────────────────────────────────────────────

class TestMLConfig:
    def test_gnn_config_defaults(self):
        c = GNNConfig()
        assert c.node_feature_dim == 14
        assert c.hidden_dim == 128
        assert c.output_dim == 64
        assert c.num_layers == 3
        assert c.heads == 4

    def test_rl_config_defaults(self):
        c = RLConfig()
        assert c.clip_ratio == 0.2
        assert c.epochs_per_update == 4

    def test_state_dim_follows_the_embedder(self):
        """state_dim 은 자유 파라미터가 아니라 **불변식**이다:
        state = query + graph + history (config.py:67-70, __post_init__).

        이전 이 테스트는 `state_dim == 512`, `query_embedding_dim == 384` 을
        고정값으로 박아뒀다가 c39cf23(MiniLM 384 → ko-sroberta 768)에서 깨진
        채 방치됐다. 숫자를 896 으로 갱신만 하면 다음 임베더 교체에서 또 썩는다
        — 관계를 검사한다.
        """
        c = RLConfig()
        assert c.state_dim == (c.query_embedding_dim + c.graph_embedding_dim
                               + c.history_dim)

    def test_explicit_state_dim_is_overridden_by_the_invariant(self):
        """명시값을 줘도 계산이 이긴다 — 차원 불일치는 정책망 입력층 크래시로
        이어지므로 계산이 정본이다 (config.py:68-69)."""
        c = RLConfig(state_dim=999)
        assert c.state_dim != 999
        assert c.state_dim == (c.query_embedding_dim + c.graph_embedding_dim
                               + c.history_dim)

    def test_query_dim_tracks_the_configured_model(self, monkeypatch):
        """임베더를 바꾸면 차원이 따라온다 — MiniLM 384 / sroberta 계열 768."""
        import importlib

        import ontology.ml.config as cfg

        monkeypatch.setenv("ONTOLOGY_ML_EMBEDDING_MODEL",
                           "paraphrase-multilingual-MiniLM-L12-v2")
        importlib.reload(cfg)
        assert cfg.RLConfig().query_embedding_dim == 384

        monkeypatch.setenv("ONTOLOGY_ML_EMBEDDING_MODEL", "jhgan/ko-sroberta-nli")
        importlib.reload(cfg)
        assert cfg.RLConfig().query_embedding_dim == 768
        assert cfg.RLConfig().state_dim == 896  # 768 + 64 + 64

        monkeypatch.delenv("ONTOLOGY_ML_EMBEDDING_MODEL", raising=False)
        importlib.reload(cfg)

    def test_buffer_config_defaults(self):
        c = BufferConfig()
        assert c.capacity == 100_000
        assert c.alpha == 0.6

    def test_selector_config_defaults(self):
        c = SelectorConfig()
        assert c.confidence_threshold == 0.7
        assert c.cold_start_threshold == 0.85
        assert c.device == "cpu"


# ─── SumTree Tests ────────────────────────────────────────────────────

class TestSumTree:
    def test_add_and_total(self):
        tree = SumTree(4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        assert abs(tree.total - 6.0) < 1e-6
        assert tree.size == 3

    def test_get_sampling(self):
        tree = SumTree(4)
        tree.add(1.0, "a")
        tree.add(3.0, "b")
        # s=0.5 should hit "a" (priority 1.0)
        _, _, data = tree.get(0.5)
        assert data == "a"
        # s=2.0 should hit "b" (priority 3.0, cumsum starts at 1.0)
        _, _, data = tree.get(2.0)
        assert data == "b"

    def test_update_priority(self):
        tree = SumTree(4)
        tree.add(1.0, "a")
        tree.add(1.0, "b")
        assert abs(tree.total - 2.0) < 1e-6

        # Update first item to priority 5
        tree.update(tree.capacity - 1, 5.0)
        assert abs(tree.total - 6.0) < 1e-6

    def test_capacity_overflow(self):
        tree = SumTree(3)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        tree.add(4.0, "d")  # should overwrite "a"
        assert tree.size == 3
        assert abs(tree.total - 9.0) < 1e-6  # 2+3+4


# ─── ExperienceBuffer Tests ──────────────────────────────────────────

class TestExperienceBuffer:
    def _make_experience(self, reward: float = 1.0) -> Experience:
        return Experience(
            state=torch.randn(32),
            action=0,
            reward=reward,
            next_state=torch.randn(32),
            done=True,
            info={"test": True},
        )

    def test_add_and_size(self):
        buf = ExperienceBuffer(BufferConfig(capacity=100))
        for _ in range(10):
            buf.add(self._make_experience())
        assert buf.size == 10

    def test_sample(self):
        buf = ExperienceBuffer(BufferConfig(capacity=100))
        for i in range(20):
            buf.add(self._make_experience(reward=float(i)))

        exps, weights, indices = buf.sample(5)
        assert len(exps) == 5
        assert len(weights) == 5
        assert len(indices) == 5
        assert all(w >= 0 for w in weights)

    def test_update_priorities(self):
        buf = ExperienceBuffer(BufferConfig(capacity=100))
        for _ in range(10):
            buf.add(self._make_experience())

        _, _, indices = buf.sample(5)
        new_priorities = np.array([10.0] * 5)
        buf.update_priorities(indices, new_priorities)

    def test_save_and_load(self, tmp_path):
        buf = ExperienceBuffer(BufferConfig(capacity=100))
        for i in range(15):
            buf.add(self._make_experience(reward=float(i)))

        path = str(tmp_path / "test_buffer.pkl")
        buf.save(path)

        buf2 = ExperienceBuffer(BufferConfig(capacity=100))
        assert buf2.load(path)
        assert buf2.size == 15

    def test_load_nonexistent(self):
        buf = ExperienceBuffer()
        assert not buf.load("/nonexistent/path.pkl")


# ─── KGTensorConverter Tests ─────────────────────────────────────────

class TestKGTensorConverter:
    def test_convert_basic(self, mock_knowledge_graph):
        converter = KGTensorConverter()
        data, node_map = converter.convert(mock_knowledge_graph)

        assert data.x.shape[1] == 14  # 14-dim features
        assert data.x.shape[0] == len(mock_knowledge_graph.nodes)
        assert data.edge_index.shape[0] == 2

    def test_node_map_contains_agents(self, mock_knowledge_graph):
        converter = KGTensorConverter()
        _, node_map = converter.convert(mock_knowledge_graph)

        for aid in SAMPLE_AGENTS:
            assert aid in node_map

    def test_caching(self, mock_knowledge_graph):
        converter = KGTensorConverter()
        data1, _ = converter.convert(mock_knowledge_graph)
        data2, _ = converter.convert(mock_knowledge_graph)
        assert data1 is data2  # same object from cache

    def test_empty_graph(self):
        converter = KGTensorConverter()
        g = nx.MultiDiGraph()
        data, node_map = converter.convert(g)
        assert data.x.shape == (1, 14)
        assert len(node_map) == 0

    def test_agent_node_indices(self, mock_knowledge_graph):
        converter = KGTensorConverter()
        indices = converter.get_agent_node_indices(mock_knowledge_graph, SAMPLE_AGENTS)
        assert len(indices) == len(SAMPLE_AGENTS)


# ─── GNNEncoder Tests ────────────────────────────────────────────────

class TestGNNEncoder:
    def test_forward_shape(self, small_gnn_config, mock_knowledge_graph):
        encoder = GNNEncoder(small_gnn_config)
        converter = KGTensorConverter()
        data, _ = converter.convert(mock_knowledge_graph)

        output = encoder(data)
        assert output.shape == (data.x.shape[0], small_gnn_config.output_dim)

    def test_encode_single_vector(self, small_gnn_config, mock_knowledge_graph):
        encoder = GNNEncoder(small_gnn_config)
        converter = KGTensorConverter()
        data, _ = converter.convert(mock_knowledge_graph)

        vec = encoder.encode(data)
        assert vec.shape == (small_gnn_config.output_dim,)

    def test_encode_for_agents(self, small_gnn_config, mock_knowledge_graph):
        encoder = GNNEncoder(small_gnn_config)
        converter = KGTensorConverter()
        data, node_map = converter.convert(mock_knowledge_graph)

        agent_indices = [node_map[aid] for aid in SAMPLE_AGENTS if aid in node_map]
        embs = encoder.encode_for_agents(data, agent_indices)
        assert embs.shape == (len(agent_indices), small_gnn_config.output_dim)

    def test_gradient_flow(self, small_gnn_config, mock_knowledge_graph):
        encoder = GNNEncoder(small_gnn_config)
        converter = KGTensorConverter()
        data, _ = converter.convert(mock_knowledge_graph)

        out = encoder.encode(data)
        loss = out.sum()
        loss.backward()

        for param in encoder.parameters():
            if param.grad is not None:
                assert not torch.all(param.grad == 0)
                break


# ─── HistoryEncoder Tests ────────────────────────────────────────────

class TestHistoryEncoder:
    def test_encode_empty(self):
        enc = HistoryEncoder(hidden_dim=16)
        vec = enc.encode([])
        assert vec.shape == (16,)
        assert torch.all(vec == 0)

    def test_encode_with_data(self):
        enc = HistoryEncoder(hidden_dim=16)
        history = [(0, 1.0), (2, 0.5), (1, -0.5)]
        vec = enc.encode(history, max_agents=10)
        assert vec.shape == (16,)
        assert not torch.all(vec == 0)

    def test_reset(self):
        enc = HistoryEncoder(hidden_dim=16)
        enc.encode([(0, 1.0)], max_agents=5)
        enc.reset()
        assert enc._hidden is None


# ─── RLPolicy Tests ──────────────────────────────────────────────────

class TestRLPolicy:
    def test_select_action(self, small_rl_config, sample_agents):
        policy = RLPolicy(small_rl_config, device="cpu")
        policy.register_agents(sample_agents)

        state = torch.randn(small_rl_config.state_dim)
        mask = policy.build_available_mask(sample_agents[:3])

        action, log_prob, value = policy.select_action(state, mask)
        assert 0 <= action < small_rl_config.max_agents
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_action_masking(self, small_rl_config, sample_agents):
        policy = RLPolicy(small_rl_config, device="cpu")
        policy.register_agents(sample_agents)

        state = torch.randn(small_rl_config.state_dim)
        # Only allow agent at index 0
        mask = torch.zeros(small_rl_config.max_agents)
        mask[0] = 1.0

        action, _, _ = policy.select_action(state, mask, deterministic=True)
        assert action == 0

    def test_evaluate_actions(self, small_rl_config, sample_agents):
        policy = RLPolicy(small_rl_config, device="cpu")
        policy.register_agents(sample_agents)

        batch_size = 4
        states = torch.randn(batch_size, small_rl_config.state_dim)
        actions = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        masks = torch.ones(batch_size, small_rl_config.max_agents)

        log_probs, values, entropy = policy.evaluate_actions(states, actions, masks)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_compute_gae(self, small_rl_config):
        policy = RLPolicy(small_rl_config, device="cpu")

        rewards = torch.tensor([1.0, 0.5, -1.0])
        values = torch.tensor([0.5, 0.3, 0.1])
        dones = torch.tensor([0.0, 0.0, 1.0])
        next_values = torch.tensor([0.3, 0.1, 0.0])

        advantages, returns = policy.compute_gae(rewards, values, dones, next_values)
        assert advantages.shape == (3,)
        assert returns.shape == (3,)

    def test_update(self, small_rl_config, sample_agents):
        policy = RLPolicy(small_rl_config, device="cpu")
        policy.register_agents(sample_agents)

        batch_size = 8
        batch = {
            "states": torch.randn(batch_size, small_rl_config.state_dim),
            "actions": torch.randint(0, len(sample_agents), (batch_size,)),
            "old_log_probs": torch.randn(batch_size),
            "returns": torch.randn(batch_size),
            "advantages": torch.randn(batch_size),
            "available_masks": torch.ones(batch_size, small_rl_config.max_agents),
        }

        result = policy.update(batch)
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "entropy" in result
        assert result["update_count"] == 1

    def test_agent_registration(self, small_rl_config, sample_agents):
        policy = RLPolicy(small_rl_config, device="cpu")
        policy.register_agents(sample_agents)

        assert policy.num_registered_agents == len(sample_agents)
        assert policy.agent_to_idx("weather_agent") == 1
        assert policy.idx_to_agent(1) == "weather_agent"

    def test_state_dict_roundtrip(self, small_rl_config, sample_agents):
        policy = RLPolicy(small_rl_config, device="cpu")
        policy.register_agents(sample_agents)

        state = policy.state_dict_all()
        policy2 = RLPolicy(small_rl_config, device="cpu")
        policy2.load_state_dict_all(state)

        assert policy2.num_registered_agents == len(sample_agents)


# ─── IntelligentAgentSelector Tests ──────────────────────────────────

class TestIntelligentAgentSelector:
    """Tests for the integration wrapper. Uses mocked embedding model."""

    @pytest.fixture
    def selector(self, small_selector_config, sample_agents):
        from ontology.ml.intelligent_selector import IntelligentAgentSelector

        sel = IntelligentAgentSelector(
            config=small_selector_config,
            auto_load=False,
            device="cpu",
        )
        sel.rl_policy.register_agents(sample_agents)

        # Mock the embedding model to avoid downloading sentence-transformers
        class MockEmbedder:
            def encode(self, text, convert_to_tensor=False):
                # Return a deterministic embedding based on text hash
                torch.manual_seed(hash(text) % 2**31)
                emb = torch.randn(small_selector_config.rl.query_embedding_dim)
                return emb

        sel._embedding_model = MockEmbedder()
        return sel

    @pytest.mark.asyncio
    async def test_select_agent(self, selector, sample_agents):
        agent_id, meta = await selector.select_agent("test query", sample_agents)
        assert agent_id in sample_agents
        assert "confidence" in meta
        assert "value_estimate" in meta
        assert meta["confidence"] >= 0

    @pytest.mark.asyncio
    async def test_store_feedback(self, selector, sample_agents):
        await selector.select_agent("test query", sample_agents)
        await selector.store_feedback(success=True)
        assert selector.experience_buffer.size == 1
        assert selector.stats["feedbacks"] == 1

    @pytest.mark.asyncio
    async def test_cold_start_confidence_dampening(self, selector, sample_agents):
        """빈 버퍼에서는 게이트를 못 넘는다 (P0-4 로 계약 교체).

        구 계약(confidence ≤ raw_confidence)은 1/N 자에 묶인 것이었다 —
        새 자는 무정보점 0.5 로 당기므로 raw(선택 액션 확률)와의 대소가
        아니라 **게이트 미달**이 냉시동의 본질이다 (상한 0.65 < 0.7).
        """
        _, meta = await selector.select_agent("test", sample_agents)
        assert meta["confidence"] < selector.config.confidence_threshold
        assert meta["confidence"] <= 0.65 + 1e-9

    @pytest.mark.asyncio
    async def test_train_step(self, selector, sample_agents):
        # Fill buffer with enough samples
        for i in range(20):
            await selector.select_agent(f"query {i}", sample_agents)
            await selector.store_feedback(success=i % 2 == 0)

        result = await selector.train_step(batch_size=8)
        assert result is not None
        assert "policy_loss" in result

    @pytest.mark.asyncio
    async def test_save_and_load(self, selector, sample_agents, small_selector_config):
        from ontology.ml.intelligent_selector import IntelligentAgentSelector

        # Make some selections
        for i in range(5):
            await selector.select_agent(f"query {i}", sample_agents)
            await selector.store_feedback(success=True)

        selector.save_models()

        # Load into new selector
        sel2 = IntelligentAgentSelector(
            config=small_selector_config,
            auto_load=True,
            device="cpu",
        )
        sel2.rl_policy.register_agents(sample_agents)
        assert sel2.experience_buffer.size == 5

    @pytest.mark.asyncio
    async def test_generate_synthetic_data(self, selector, mock_knowledge_graph, sample_agents):
        # Provide a mock KG
        class MockKG:
            def __init__(self, g):
                self.graph = g

        selector._knowledge_graph = MockKG(mock_knowledge_graph)

        result = await selector.generate_synthetic_data(
            num_samples=50, train_immediately=True
        )
        assert result["data_generated"] == 50
        assert result["buffer_size"] >= 50

    @pytest.mark.asyncio
    async def test_auto_register_new_agents(self, selector):
        """New agents in available_agents should be auto-registered."""
        new_agents = ["internet_agent", "new_agent_xyz"]
        await selector.select_agent("test", new_agents)
        assert "new_agent_xyz" in selector.rl_policy._agent_to_idx


# ─── SyntheticDataGenerator Tests ────────────────────────────────────

class TestSyntheticDataGenerator:
    def test_generate_with_kg(self, mock_knowledge_graph, sample_agents):
        gen = SyntheticDataGenerator(state_dim=64, query_dim=32)
        experiences = gen.generate(mock_knowledge_graph, sample_agents, num_samples=20)
        assert len(experiences) == 20
        assert all(isinstance(e, Experience) for e in experiences)

    def test_generate_random_fallback(self, sample_agents):
        """P0-5 로 계약 교체: randn 폴백은 기본 거부.

        구 계약(빈 그래프 → 잡음 10건)은 잡음 1000건 학습 사고의 기원을
        정상으로 박아둔 것이었다. 이제 명시적 allow_random 으로만 허용.
        """
        gen = SyntheticDataGenerator(state_dim=64, query_dim=32)
        empty_graph = nx.MultiDiGraph()
        assert gen.generate(empty_graph, sample_agents, num_samples=10) == []
        experiences = gen.generate(
            empty_graph, sample_agents, num_samples=10, allow_random=True
        )
        assert len(experiences) == 10
