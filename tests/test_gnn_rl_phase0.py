"""
GNN+RL Phase 0 수리 회귀 계약 — P0-1 관측 / P0-2 예열 재시도 / P0-3 max_agents 지뢰.

근거: docs/rl-adoption-zero-diagnosis.md (2026-08-03).

채택률 0% 진단이 확정한 결함들을 계약으로 고정한다:
- P0-1: 미채택 사유가 하나의 카운터(gnn_rl_fallback)에 뭉쳐 있었고,
  예외 분기는 아예 무계측이었다. 사유별 분리 + 예외 계측이 계약.
- P0-2: 임베더 예열이 1회성이라 첫 실패 = 프로세스 수명 동안 영구 skip.
  재시도 가능(간격 가드 포함)이 계약.
- P0-3: 등록 118 > max_agents 100 → build_available_mask IndexError.
  경계 검사 + 여유 있는 기본값이 계약.
"""

import asyncio

import pytest
import torch

from ontology.core.hybrid_agent_selector import HybridAgentSelector
from ontology.ml.config import RLConfig
from ontology.ml.rl_policy import RLPolicy


# ─── 헬퍼: 가짜 IntelligentAgentSelector ─────────────────────────────


class _FakeBuffer:
    size = 0


class _FakeSelector:
    """토치 없이 Phase 0 경로만 검증하기 위한 대역.

    embedding_model property 는 실물과 같은 계약 — 성공 시 _embedding_model
    을 채우고, 실패 시 raise.
    """

    def __init__(self, fail: bool = False, ready: bool = False):
        self._embedding_model = object() if ready else None
        self._fail = fail
        self.warm_calls = 0
        self.stats = {}
        self.experience_buffer = _FakeBuffer()
        self.select_result = ("a1", {"confidence": 0.01, "value_estimate": 0.0})
        self.select_error = None

    @property
    def embedding_model(self):
        self.warm_calls += 1
        if self._fail:
            raise RuntimeError("warm boom")
        self._embedding_model = object()
        return self._embedding_model

    async def select_agent(self, query, available_agents, deterministic=False):
        if self.select_error is not None:
            raise self.select_error
        return self.select_result


def _make_hybrid(fake: _FakeSelector) -> HybridAgentSelector:
    sel = HybridAgentSelector(auto_sync=False, use_gnn_rl=True)
    sel._intelligent_selector = fake
    return sel


async def _stub_phase12(sel: HybridAgentSelector, answer: str = "a1"):
    """Phase 1(KG)·Phase 2(LLM)를 스텁으로 — GNN+RL 분기만 검증 대상."""

    async def _kg(query, agents):
        return {"has_insights": False}

    async def _llm(query, agents, info, insights):
        return answer, "stub"

    sel._analyze_with_knowledge_graph = _kg
    sel._select_with_llm = _llm


# ─── P0-1: 분리 계측 ─────────────────────────────────────────────────


def test_stats_has_separated_counters():
    sel = HybridAgentSelector(auto_sync=False, use_gnn_rl=False)
    for key in (
        "gnn_rl_skip_not_ready",
        "gnn_rl_timeout",
        "gnn_rl_error",
        "gnn_rl_low_confidence",
        "gnn_rl_unavailable_agent",
    ):
        assert sel.stats.get(key) == 0, key


def test_count_helper_increments_fine_and_aggregate():
    sel = HybridAgentSelector(auto_sync=False, use_gnn_rl=False)
    sel._count_gnn_rl_failure("timeout")
    sel._count_gnn_rl_failure("timeout")
    sel._count_gnn_rl_failure("error")
    assert sel.stats["gnn_rl_timeout"] == 2
    assert sel.stats["gnn_rl_error"] == 1
    # 총계는 유지 — 기존 대시보드가 이 키를 읽는다.
    assert sel.stats["gnn_rl_fallback"] == 3


@pytest.mark.asyncio
async def test_exception_branch_is_metered():
    """진단의 무계측 결함: 예외 분기가 fallback 카운터도 안 올렸다."""
    fake = _FakeSelector(ready=True)
    fake.select_error = RuntimeError("policy boom")
    sel = _make_hybrid(fake)
    await _stub_phase12(sel)

    agent, meta = await sel.select_agent("q", ["a1"], {"a1": {}})
    assert agent == "a1"  # Phase 2 로 정상 폴백
    assert sel.stats["gnn_rl_error"] == 1
    assert sel.stats["gnn_rl_fallback"] == 1


@pytest.mark.asyncio
async def test_timeout_branch_is_metered_separately():
    fake = _FakeSelector(ready=True)
    fake.select_error = asyncio.TimeoutError()
    sel = _make_hybrid(fake)
    await _stub_phase12(sel)

    await sel.select_agent("q", ["a1"], {"a1": {}})
    assert sel.stats["gnn_rl_timeout"] == 1
    assert sel.stats["gnn_rl_error"] == 0


@pytest.mark.asyncio
async def test_low_confidence_branch_is_metered():
    fake = _FakeSelector(ready=True)  # confidence 0.01 < 0.7
    sel = _make_hybrid(fake)
    await _stub_phase12(sel)

    await sel.select_agent("q", ["a1"], {"a1": {}})
    assert sel.stats["gnn_rl_low_confidence"] == 1


@pytest.mark.asyncio
async def test_unavailable_agent_branch_is_metered():
    """확신은 넘었는데 선택 에이전트가 목록 밖 — low_confidence 로 오분류되던 분기."""
    fake = _FakeSelector(ready=True)
    fake.select_result = ("ghost", {"confidence": 0.99, "value_estimate": 0.0})
    sel = _make_hybrid(fake)
    await _stub_phase12(sel)

    await sel.select_agent("q", ["a1"], {"a1": {}})
    assert sel.stats["gnn_rl_unavailable_agent"] == 1
    assert sel.stats["gnn_rl_low_confidence"] == 0


def test_get_stats_exposes_embedding_readiness():
    fake = _FakeSelector(ready=False)
    sel = _make_hybrid(fake)
    assert sel.get_stats()["embedding_ready"] is False

    fake._embedding_model = object()
    assert sel.get_stats()["embedding_ready"] is True


# ─── P0-2: 예열 재시도 ───────────────────────────────────────────────


def _join_warm(sel: HybridAgentSelector):
    t = getattr(sel, "_embedding_warm_thread", None)
    if t is not None:
        t.join(timeout=5)


def test_warmup_failure_is_recorded_and_retryable():
    fake = _FakeSelector(fail=True)
    sel = _make_hybrid(fake)

    assert sel._start_embedding_warmup() is True
    _join_warm(sel)
    assert sel._embedding_warm_error is not None  # 실패가 보인다
    assert sel.get_stats()["embedding_warm_error"] is not None

    # 간격 가드: 직후 재시도는 거부 (폭주 방지)
    assert sel._start_embedding_warmup() is False

    # 간격 경과 후에는 재시도 가능 — 1회 실패 ≠ 영구 skip
    sel._embedding_warm_last_attempt -= (sel.EMBEDDING_WARM_RETRY_SECONDS + 1)
    fake._fail = False
    assert sel._start_embedding_warmup() is True
    _join_warm(sel)
    assert fake._embedding_model is not None
    assert sel._embedding_warm_error is None  # 성공이 오류 기록을 지운다


def test_warmup_noop_when_already_ready():
    fake = _FakeSelector(ready=True)
    sel = _make_hybrid(fake)
    assert sel._start_embedding_warmup() is False
    assert fake.warm_calls == 0


@pytest.mark.asyncio
async def test_skip_branch_counts_and_retries_warmup():
    """모델 미로드 → skip 계측 + 예열 재기동 (종전엔 카운트만 하고 방치)."""
    fake = _FakeSelector(fail=False, ready=False)
    sel = _make_hybrid(fake)
    await _stub_phase12(sel)

    await sel.select_agent("q", ["a1"], {"a1": {}})
    assert sel.stats["gnn_rl_skip_not_ready"] == 1
    _join_warm(sel)
    assert fake._embedding_model is not None  # 재예열이 실제로 돌았다


def test_buffer_load_survives_foreign_pickle(tmp_path):
    """buffer.pkl 이 다른 모듈 경로로 피클된 경우(ModuleNotFoundError) —
    종전엔 catch 목록 밖 예외가 새어나가 IntelligentAgentSelector.__init__
    이 통째로 죽고, hybrid selector 가 GNN+RL 을 영구 비활성화했다
    (예열 게이트와 별개의 blackout 원 — 이번 수리 중 실측으로 재현).
    """
    from ontology.ml.config import BufferConfig
    from ontology.ml.experience_buffer import ExperienceBuffer

    p = tmp_path / "buffer.pkl"
    p.write_bytes(b"cno_such_module_xyz\nKlass\n.")  # 존재하지 않는 모듈 참조 피클
    buf = ExperienceBuffer(BufferConfig())
    assert buf.load(str(p)) is False  # 던지지 않고 False
    assert buf.size == 0


# ─── P0-3: max_agents 지뢰 ───────────────────────────────────────────


def test_max_agents_default_has_headroom():
    # 등록 118 이 이미 100 을 넘었다 — 기본값은 여유가 있어야 한다.
    assert RLConfig().max_agents >= 256


def test_build_available_mask_beyond_max_is_safe():
    cfg = RLConfig(max_agents=4)
    policy = RLPolicy(cfg)
    agents = [f"agent_{i}" for i in range(6)]  # 6 > max 4
    policy.register_agents(agents)

    mask = policy.build_available_mask(agents)  # 종전: IndexError
    assert mask.shape[0] == 4
    assert mask.sum().item() == 4  # 범위 안 4개만 켜진다


def test_select_action_with_over_capacity_registration():
    """지뢰 재현 조건(등록 > max) 그대로 — 선택이 죽지 않고 범위 안에서 고른다."""
    cfg = RLConfig(max_agents=8)
    policy = RLPolicy(cfg)
    policy.register_agents([f"a{i}" for i in range(12)])

    state = torch.zeros(cfg.state_dim)
    mask = policy.build_available_mask([f"a{i}" for i in range(12)])
    action, log_prob, value = policy.select_action(state, mask)
    assert 0 <= action < 8
