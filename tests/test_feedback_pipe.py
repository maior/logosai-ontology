"""
피드백 파이프 수리 — 회귀 계약 (진단 #4 의 처방).

종전: `_pending_state` 전역 단일 슬롯 → 동시 요청이 서로를 덮어써 피드백
95% 유실(889→48) + 보상 오귀속. 계약:
- pending 은 selection_id 키 맵 — 동시 선택이 서로를 덮지 않는다.
- 실행 에이전트 ≠ 샘플이면 보상은 **실행 쪽**에 귀속 (fallback 경로의
  성공/실패가 학습 라벨이 된다 — P0-5 imitation 과 같은 철학, 균등 prior).
- 유실은 전부 세어진다 (no_pending / unmapped / evicted).
"""

import asyncio

import pytest
import torch

from ontology.core.hybrid_agent_selector import HybridAgentSelector
from ontology.ml.config import GNNConfig, RLConfig, SelectorConfig

AGENTS = ["a0", "a1", "a2", "a3", "a4"]


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
        def encode(self, text, convert_to_tensor=False):
            torch.manual_seed(hash(text) % 2**31)
            return torch.randn(16)

    sel._embedding_model = MockEmbedder()
    sel._knowledge_graph = object()
    return sel


def _exps(sel, n):
    # sample() 은 우선순위 **복원 추출**이라 같은 경험이 두 번 나올 수 있다
    # (실제로 flaky 를 만들었다) — 전량 조회로 결정적으로 검사한다.
    exps = sel.experience_buffer.all_experiences()
    assert len(exps) == n
    return exps


class TestPendingMap:
    @pytest.mark.asyncio
    async def test_concurrent_selections_do_not_overwrite(self, tmp_path):
        """단일 슬롯 mutant 사살 — 두 선택이 살아 있고 각자의 질의로 귀속된다."""
        sel = _make_selector(tmp_path)
        a1, m1 = await sel.select_agent("질의 하나", AGENTS)
        a2, m2 = await sel.select_agent("질의 둘", AGENTS)
        assert m1["selection_id"] != m2["selection_id"]
        assert len(sel._pending) == 2

        await sel.store_feedback(success=True, selection_id=m1["selection_id"],
                                 executed_agent=a1)
        await sel.store_feedback(success=False, selection_id=m2["selection_id"],
                                 executed_agent=a2)
        assert sel.experience_buffer.size == 2
        by_query = {e.info["query"]: e for e in _exps(sel, 2)}
        assert by_query["질의 하나"].info["success"] is True
        assert by_query["질의 둘"].info["success"] is False

    @pytest.mark.asyncio
    async def test_reward_attaches_to_executed_agent(self, tmp_path):
        """오귀속 mutant 사살 — 실행된 에이전트가 샘플과 다르면 실행 쪽으로."""
        sel = _make_selector(tmp_path)
        sampled, meta = await sel.select_agent("질의", AGENTS)
        executed = "a2" if sampled != "a2" else "a3"

        await sel.store_feedback(success=True,
                                 selection_id=meta["selection_id"],
                                 executed_agent=executed)
        exp = _exps(sel, 1)[0]
        assert exp.action == sel.rl_policy._agent_to_idx[executed]
        assert exp.info["agent_id"] == executed
        assert exp.info["off_policy"] == "executed_fallback"
        assert exp.info["sampled_agent_id"] == sampled
        assert sel.stats["feedback_off_policy"] == 1

    @pytest.mark.asyncio
    async def test_matched_agent_stays_on_policy(self, tmp_path):
        sel = _make_selector(tmp_path)
        agent, meta = await sel.select_agent("질의", AGENTS)
        await sel.store_feedback(success=True,
                                 selection_id=meta["selection_id"],
                                 executed_agent=agent)
        exp = _exps(sel, 1)[0]
        assert exp.info["off_policy"] is None
        assert sel.stats["feedback_off_policy"] == 0

    @pytest.mark.asyncio
    async def test_unknown_id_and_unmapped_agent_are_counted(self, tmp_path):
        sel = _make_selector(tmp_path)
        await sel.store_feedback(success=True, selection_id="없는id")
        assert sel.stats["feedback_no_pending"] == 1
        assert sel.experience_buffer.size == 0

        _, meta = await sel.select_agent("질의", AGENTS)
        await sel.store_feedback(success=True,
                                 selection_id=meta["selection_id"],
                                 executed_agent="유령에이전트")
        assert sel.stats["feedback_unmapped_agent"] == 1
        assert sel.experience_buffer.size == 0     # 폐기 — 단, 세어졌다
        assert len(sel._pending) == 0              # pending 은 소진 (재사용 금지)

    @pytest.mark.asyncio
    async def test_no_id_backcompat_pops_latest(self, tmp_path):
        """구 호출부(id 없이) 하위 호환 — 최신 pending 으로 폴백."""
        sel = _make_selector(tmp_path)
        await sel.select_agent("질의", AGENTS)
        await sel.store_feedback(success=True)
        assert sel.experience_buffer.size == 1
        assert sel.stats["feedbacks"] == 1

    @pytest.mark.asyncio
    async def test_eviction_cap_is_counted(self, tmp_path):
        sel = _make_selector(tmp_path)
        sel._pending_cap = 2
        metas = []
        for i in range(3):
            _, m = await sel.select_agent(f"질의 {i}", AGENTS)
            metas.append(m)
        assert len(sel._pending) == 2
        assert sel.stats["pending_evicted"] == 1
        # 밀려난 선택의 피드백은 no_pending 으로 세어진다
        await sel.store_feedback(success=True,
                                 selection_id=metas[0]["selection_id"])
        assert sel.stats["feedback_no_pending"] == 1


class TestHybridWiring:
    """hybrid.store_feedback 이 id + 실행 에이전트를 넘기는지 — 배선 계약."""

    def _hybrid_with_fake(self):
        import networkx as nx
        from types import SimpleNamespace

        captured = {}

        class FakeML:
            _embedding_model = object()
            stats = {}
            experience_buffer = SimpleNamespace(size=0)

            async def select_agent(self, query, available_agents,
                                   deterministic=False):
                return "a1", {"confidence": 0.01, "value_estimate": 0.0,
                              "selection_id": "ml-sid-42"}

            async def store_feedback(self, **kw):
                captured.update(kw)

        sel = HybridAgentSelector(auto_sync=False, use_gnn_rl=True)
        sel._intelligent_selector = FakeML()
        g = nx.MultiDiGraph()

        async def _noop(*a, **kw):
            return True

        sel._knowledge_graph = SimpleNamespace(
            graph_engine=SimpleNamespace(graph=g),
            add_concept=_noop, add_relationship=_noop,
            save_to_disk=lambda: True)

        async def _kg(query, agents):
            return {"has_insights": False}

        async def _llm(query, agents, info, insights):
            return "a1", "stub"

        async def _sem(query):
            return {"generalization_pattern": "p", "category": "c"}

        sel._analyze_with_knowledge_graph = _kg
        sel._select_with_llm = _llm
        sel._analyze_query_semantics = _sem
        sel.save_stats = lambda: True
        return sel, captured

    @pytest.mark.asyncio
    async def test_feedback_carries_id_and_executed_agent(self):
        sel, captured = self._hybrid_with_fake()
        agent, meta = await sel.select_agent("환율 알려줘", ["a1"], {"a1": {}})
        assert agent == "a1"
        entry = list(sel._selection_history)[-1]
        assert entry["ml_selection_id"] == "ml-sid-42"

        ok = await sel.store_feedback("환율 알려줘", "a1", success=True)
        assert ok is True
        assert captured["selection_id"] == "ml-sid-42"
        assert captured["executed_agent"] == "a1"
