"""
ML Configuration — Dataclass-based configuration for GNN+RL agent selection.

All hyperparameters are centralized here for easy tuning.
"""

import os
from dataclasses import dataclass, field


def _resolve_embedding_model() -> str:
    """쿼리 임베더 선택 (Task #11, 2026-07-15).

    기본을 MiniLM → ko-sroberta 로 교체: MiniLM 은 이 환경에서 한국어 퇴화
    (무관 문장 cosine 0.96, 2026-07-02 실측) → GNN+RL 신뢰도 항상 저조 → 채택 0%.
    semantic_index 와 동일 모델로 정렬 (프로세스 내 모델 공유).
    롤백: ONTOLOGY_ML_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

    ⚠️ **`ONTOLOGY_EMBEDDING_MODEL` 도 따라간다.** 종전에는 ML 전용 변수만 읽어서,
    온톨로지 임베더를 바꿔도(`ONTOLOGY_EMBEDDING_MODEL=BAAI/bge-m3`) 여기는
    모르는 채 ko-sroberta 로 남았다 — "동일 모델로 정렬"이라는 이 주석의 약속이
    깨지고, 차원이 어긋나 정책망 입력층에서 크래시한다. ML 전용 오버라이드는
    롤백 경로로 남기되(우선순위 유지) 없으면 온톨로지 설정을 따른다.
    """
    return (os.environ.get("ONTOLOGY_ML_EMBEDDING_MODEL", "").strip()
            or os.environ.get("ONTOLOGY_EMBEDDING_MODEL", "").strip()
            or "jhgan/ko-sroberta-nli")


def _resolve_query_dim() -> int:
    """임베더 출력 차원 — env 우선, 아니면 모델명으로 추론.

    ⚠️ 종전에는 여기서 "minilm 이면 384, 나머지는 768" 로 추론했다. bge-m3 계열은
    **1024** 라 768 을 돌려주면 state_dim 이 896 으로 계산되는데 실제 임베딩은
    1024 다 — RLConfig.__post_init__ 주석이 경고한 바로 그 크래시다.

    추론 표는 **커널(core.semantic_index)에 산다.** 여기 두면 관리 콘솔의 상태
    조회가 차원을 알려고 ml 패키지를 import 하고 그게 torch 를 끌고 온다.
    알려지지 않은 모델은 768 로 가정하므로 `ONTOLOGY_ML_QUERY_DIM` 이 탈출구다.
    """
    env = os.environ.get("ONTOLOGY_ML_QUERY_DIM")
    if env:
        return int(env)
    from ..core.semantic_index import resolve_embedding_dim
    return resolve_embedding_dim(_resolve_embedding_model())


@dataclass
class GNNConfig:
    """Graph Neural Network encoder configuration."""

    node_feature_dim: int = 14  # type(3) + degree(3) + perf(3) + temporal(3) + special(2)
    hidden_dim: int = 128
    output_dim: int = 64  # graph embedding dimension
    num_layers: int = 3
    heads: int = 4  # GAT attention heads
    dropout: float = 0.1


@dataclass
class RLConfig:
    """Reinforcement Learning (PPO) policy configuration."""

    # 차원은 임베더에 종속 — __post_init__ 에서 state_dim 을 자동 정합
    # (구 512 = MiniLM 384 + 64 + 64. ko-sroberta 는 768 + 64 + 64 = 896)
    state_dim: int = 0  # 자동 계산 (query + graph + history)
    query_embedding_dim: int = field(default_factory=_resolve_query_dim)
    graph_embedding_dim: int = 64  # GNN output
    history_dim: int = 64  # HistoryEncoder output
    hidden_dim: int = 256  # shared feature extractor hidden
    # P0-3 (2026-08-03): 100 → 256. 등록 에이전트가 이미 118 이라 mask[idx] 가
    # IndexError 로 죽고 있었다 (예외는 삼켜져 무계측 — 진단 문서 참고).
    # 값 변경은 actor 출력층 크기를 바꾸므로 구 policy.pt 는 로드 실패 →
    # 신규 초기화된다. 구 정책은 randn 잡음 1000건 학습이라 폐기가 처방(P0-5).
    max_agents: int = 256  # maximum number of agents supported
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2  # PPO clip ratio
    entropy_coeff: float = 0.01  # entropy bonus coefficient
    value_loss_coeff: float = 0.5  # value loss weight
    max_grad_norm: float = 0.5  # gradient clipping
    epochs_per_update: int = 4  # PPO epochs per batch
    history_length: int = 10  # number of recent selections to encode

    def __post_init__(self):
        # state = query + graph + history 불변식 (명시값 무시하고 항상 정합 —
        # 차원 불일치는 정책망 입력층 크래시로 이어지므로 계산이 정본)
        self.state_dim = self.query_embedding_dim + self.graph_embedding_dim + self.history_dim


@dataclass
class BufferConfig:
    """Prioritized experience replay buffer configuration."""

    capacity: int = 100_000
    alpha: float = 0.6  # prioritization exponent (0=uniform, 1=full priority)
    beta_start: float = 0.4  # importance sampling start
    beta_end: float = 1.0  # importance sampling end
    beta_frames: int = 100_000  # frames to anneal beta
    min_priority: float = 1e-6  # minimum priority to prevent zero sampling


@dataclass
class SelectorConfig:
    """Top-level configuration for IntelligentAgentSelector."""

    gnn: GNNConfig = field(default_factory=GNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)

    # Sentence-transformers model for query embedding (env 오버라이드 가능)
    embedding_model: str = field(default_factory=_resolve_embedding_model)

    # Device: 'cpu', 'cuda', or 'mps'
    device: str = "cpu"

    # Model persistence
    models_dir: str = os.path.join(os.path.dirname(__file__), "models")

    # Training thresholds
    min_buffer_size_for_training: int = 64
    training_batch_size: int = 64

    # Confidence thresholds
    confidence_threshold: float = 0.7  # matches HybridAgentSelector.GNN_RL_CONFIDENCE_THRESHOLD
    cold_start_threshold: float = 0.85  # higher threshold with little data
    cold_start_buffer_size: int = 500  # buffer size to exit cold start phase

    # Auto-save interval
    save_interval: int = 100  # save models every N training updates

    # Background training
    enable_background_training: bool = True

    # Reward shaping
    reward_success: float = 1.0
    reward_failure: float = -0.5
    reward_partial: float = 0.3
