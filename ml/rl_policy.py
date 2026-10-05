"""
RL Policy — PPO Actor-Critic with action masking and GAE.

Components:
    - HistoryEncoder: GRU-based encoding of recent selection history
    - SharedFeatureExtractor: Shared MLP for Actor and Critic
    - ActorHead: Action logits with masking for variable agent sets
    - CriticHead: State value estimation
    - RLPolicy: Full PPO pipeline (select, evaluate, update)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from loguru import logger

from .config import RLConfig


def action_confidence(probs: torch.Tensor) -> float:
    """P0-4 (2026-08-03): 정책 확신의 자 — 최댓값의 상대 마진 p1/(p1+p2).

    종전 자("샘플링된 액션의 softmax 확률")는 액션 N 개 균등분포에서 ≈1/N
    (실측 평균 0.0137 ≈ 1/73) — 게이트 0.7 과 50배 갭으로 원리적 도달 불가였다.
    이 자는:
    - 균등분포 = 0.5 (**N 무관** — 병리의 제거가 본체)
    - 한 액션 집중 → 1.0 / 게이트 0.7 = "1위가 2위의 2.33배"
    - 후보가 하나뿐이면 1.0 (비교 대상 없음)
    - kg_confidence(0..1 루브릭)와 같은 축에서 비교 가능

    마스킹된 액션은 softmax 후 확률 ≈0 이라 자동으로 비교에서 밀려난다.
    """
    flat = probs.flatten()
    if flat.numel() == 0:
        return 0.0
    top2 = torch.topk(flat, k=min(2, flat.numel())).values
    p1 = float(top2[0])
    p2 = float(top2[1]) if top2.numel() > 1 else 0.0
    if p1 <= 0.0:
        return 0.0
    return p1 / (p1 + p2)


class HistoryEncoder(nn.Module):
    """
    GRU-based encoder for recent agent selection history.

    Encodes the last K (agent_index, reward) pairs into a fixed-size vector.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Input: (agent_idx_normalized, reward) → 2-dim
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self._hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset hidden state for a new episode."""
        self._hidden = None

    def encode(self, history: List[Tuple[int, float]], max_agents: int = 100) -> torch.Tensor:
        """
        Encode selection history into a vector.

        Args:
            history: List of (agent_index, reward) tuples
            max_agents: Max agent count for normalization

        Returns:
            [hidden_dim] tensor
        """
        if not history:
            return torch.zeros(self.hidden_dim)

        # Build sequence tensor [1, seq_len, 2]
        seq = torch.tensor(
            [[idx / max(max_agents, 1), reward] for idx, reward in history],
            dtype=torch.float32,
        ).unsqueeze(0)

        output, self._hidden = self.gru(seq, self._hidden)
        return output[0, -1, :]  # Last hidden state [hidden_dim]


class SharedFeatureExtractor(nn.Module):
    """
    Shared MLP between Actor and Critic.

    Architecture: Linear(state_dim, 256) → LayerNorm → ReLU → Linear(256, 128) → ReLU
    """

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActorHead(nn.Module):
    """
    Actor network head with action masking.

    Outputs action probabilities over agents, with unavailable agents masked.
    """

    def __init__(self, feature_dim: int = 128, max_agents: int = 100):
        super().__init__()
        self.logits = nn.Linear(feature_dim, max_agents)

    def forward(self, features: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked action probabilities.

        Args:
            features: [batch, feature_dim] or [feature_dim]
            available_mask: [batch, max_agents] or [max_agents] — 1=available, 0=unavailable

        Returns:
            Action probabilities [batch, max_agents] or [max_agents]
        """
        raw_logits = self.logits(features)
        # Mask unavailable agents with large negative value
        masked_logits = raw_logits + (1.0 - available_mask) * (-1e8)
        return F.softmax(masked_logits, dim=-1)


class CriticHead(nn.Module):
    """Critic network head — estimates state value."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class RLPolicy:
    """
    PPO-based agent selection policy.

    Combines HistoryEncoder + SharedFeatureExtractor + Actor/Critic heads.
    Supports variable action spaces via masking.
    """

    def __init__(self, config: Optional[RLConfig] = None, device: str = "cpu"):
        self.config = config or RLConfig()
        self.device = torch.device(device)
        c = self.config

        # Components
        self.history_encoder = HistoryEncoder(c.history_dim).to(self.device)
        self.feature_extractor = SharedFeatureExtractor(c.state_dim, c.hidden_dim).to(self.device)
        feature_out_dim = c.hidden_dim // 2  # 128
        self.actor = ActorHead(feature_out_dim, c.max_agents).to(self.device)
        self.critic = CriticHead(feature_out_dim).to(self.device)

        # Optimizers
        actor_params = list(self.feature_extractor.parameters()) + list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        # History encoder params shared between both
        shared_params = list(self.history_encoder.parameters())

        self.actor_optimizer = torch.optim.Adam(
            actor_params + shared_params, lr=c.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params + shared_params, lr=c.lr_critic
        )

        # Agent ID ↔ index mapping
        self._agent_to_idx: Dict[str, int] = {}
        self._idx_to_agent: Dict[int, str] = {}

        # Selection history for HistoryEncoder
        self._selection_history: List[Tuple[int, float]] = []

        # Training step counter
        self._update_count = 0

    def register_agents(self, agent_ids: List[str]) -> None:
        """Register agent IDs and build index mapping."""
        self._agent_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        self._idx_to_agent = {i: aid for i, aid in enumerate(agent_ids)}
        # P0-3: 넘치면 소리를 낸다 — 초과분은 액션 공간(max_agents) 밖이라
        # 선택될 수 없다. 조용히 두면 "일부 에이전트만 학습되는" 편향이 숨는다.
        if len(agent_ids) > self.config.max_agents:
            logger.warning(
                f"register_agents: {len(agent_ids)} agents > max_agents="
                f"{self.config.max_agents} — 초과 인덱스는 선택 불가 (config 상향 필요)"
            )

    def agent_to_idx(self, agent_id: str) -> int:
        return self._agent_to_idx.get(agent_id, 0)

    def idx_to_agent(self, idx: int) -> str:
        return self._idx_to_agent.get(idx, "unknown")

    @property
    def num_registered_agents(self) -> int:
        return len(self._agent_to_idx)

    def build_available_mask(self, available_agents: List[str]) -> torch.Tensor:
        """Build a binary mask tensor for available agents.

        P0-3: 인덱스가 max_agents 를 넘으면 건너뛴다 — 종전에는 여기서
        IndexError 가 났고(등록 118 > max 100), 호출부가 예외를 삼켜
        무계측 fallback 이 됐다. 경계 위반은 skip + warning 이 계약.
        """
        mask = torch.zeros(self.config.max_agents, device=self.device)
        skipped = 0
        for aid in available_agents:
            idx = self._agent_to_idx.get(aid)
            if idx is None:
                continue
            if idx >= self.config.max_agents:
                skipped += 1
                continue
            mask[idx] = 1.0
        if skipped:
            logger.warning(
                f"build_available_mask: {skipped} agents beyond "
                f"max_agents={self.config.max_agents} — 액션 공간 밖, 선택 불가"
            )
        return mask

    def select_action(
        self,
        state: torch.Tensor,
        available_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action (agent index) given state and mask.

        Args:
            state: [state_dim] tensor
            available_mask: [max_agents] binary mask
            deterministic: If True, select argmax instead of sampling

        Returns:
            (action_index, log_probability, state_value)
        """
        with torch.no_grad():
            state = state.to(self.device)
            available_mask = available_mask.to(self.device)

            features = self.feature_extractor(state.unsqueeze(0))
            action_probs = self.actor(features, available_mask.unsqueeze(0)).squeeze(0)
            value = self.critic(features).squeeze(0)

            if deterministic:
                action = action_probs.argmax().item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()

            log_prob = torch.log(action_probs[action] + 1e-10)

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        available_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs, values, and entropy for a batch of transitions.

        Args:
            states: [batch, state_dim]
            actions: [batch] (long)
            available_masks: [batch, max_agents]

        Returns:
            (log_probs [batch], values [batch], entropy [batch])
        """
        features = self.feature_extractor(states)
        action_probs = self.actor(features, available_masks)
        values = self.critic(features)

        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: [batch]
            values: [batch]
            dones: [batch] (0 or 1)
            next_values: [batch]

        Returns:
            (advantages [batch], returns [batch])
        """
        c = self.config
        batch_size = rewards.size(0)

        advantages = torch.zeros(batch_size, device=self.device)
        last_gae = 0.0

        for t in reversed(range(batch_size)):
            delta = rewards[t] + c.gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_gae = delta + c.gamma * c.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO update step.

        Args:
            batch: Dict with keys: states, actions, old_log_probs, returns,
                   advantages, available_masks

        Returns:
            Dict with policy_loss, value_loss, entropy, total_loss
        """
        c = self.config
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        returns = batch["returns"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        available_masks = batch["available_masks"].to(self.device)

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(c.epochs_per_update):
            log_probs, values, entropy = self.evaluate_actions(states, actions, available_masks)

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - c.clip_ratio, 1 + c.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_mean = entropy.mean()

            # Combined loss
            loss = policy_loss + c.value_loss_coeff * value_loss - c.entropy_coeff * entropy_mean

            # Update actor
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            all_params = (
                list(self.feature_extractor.parameters())
                + list(self.actor.parameters())
                + list(self.critic.parameters())
                + list(self.history_encoder.parameters())
            )
            nn.utils.clip_grad_norm_(all_params, c.max_grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()

        epochs = c.epochs_per_update
        self._update_count += 1

        return {
            "policy_loss": total_policy_loss / epochs,
            "value_loss": total_value_loss / epochs,
            "entropy": total_entropy / epochs,
            "total_loss": (total_policy_loss + total_value_loss) / epochs,
            "update_count": self._update_count,
        }

    def imitate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        epochs: int = 20,
        batch_size: int = 64,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """P0-5: 행동 복제(behavior cloning) — 정본 라벨의 지도학습.

        PPO 로 라벨을 배우려던 첫 시도는 실패했다(실측: 100스텝 후 top-1 3%,
        confidence 0.5 균등 그대로). 원인: 라벨의 success_rate 가 거의 전부
        1.0 → 보상 균일 → critic 이 수렴하는 순간 advantage ≈ 0 → 정책
        갱신 소멸. **라벨은 보상 신호가 아니라 지도학습 대상이다** —
        cross-entropy 로 직접 흉내낸다. PPO 는 이후 실피드백(대비가 있는
        보상)의 미세조정에 쓴다.

        rewards 를 주면 critic 도 함께 맞춘다 (이후 PPO 미세조정의 baseline).
        """
        c = self.config
        n = states.size(0)
        if n == 0:
            return {"epochs": 0, "final_loss": 0.0, "train_acc": 0.0}

        states = states.to(self.device)
        actions = actions.to(self.device)
        masks = masks.to(self.device)
        if rewards is not None:
            rewards = rewards.to(self.device)

        final_loss = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                features = self.feature_extractor(states[idx])
                probs = self.actor(features, masks[idx])
                # weights: 보상 가중 행동복제 (contextual bandit 학습의
                # 가장 강건한 형태) — 성공이 클수록 그 라벨을 세게 흉내낸다
                nll = F.nll_loss(torch.log(probs + 1e-10), actions[idx],
                                 reduction="none")
                if weights is not None:
                    w = weights[idx]
                    loss = (w * nll).sum() / (w.sum() + 1e-10)
                else:
                    loss = nll.mean()
                if rewards is not None:
                    values = self.critic(features)
                    loss = loss + c.value_loss_coeff * F.mse_loss(values, rewards[idx])

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(self.feature_extractor.parameters())
                    + list(self.actor.parameters())
                    + list(self.critic.parameters())
                    + list(self.history_encoder.parameters())
                )
                nn.utils.clip_grad_norm_(all_params, c.max_grad_norm)
                self.actor_optimizer.step()
                if rewards is not None:
                    self.critic_optimizer.step()

                epoch_loss += loss.item()
                batches += 1
            final_loss = epoch_loss / max(1, batches)

        # 학습 라벨 재현율 (과적합 여부가 아니라 "배웠는가"의 최소 검증)
        with torch.no_grad():
            probs = self.actor(self.feature_extractor(states), masks)
            train_acc = float((probs.argmax(dim=-1) == actions).float().mean())

        self._update_count += 1
        return {"epochs": epochs, "final_loss": final_loss, "train_acc": train_acc}

    def add_to_history(self, agent_idx: int, reward: float) -> None:
        """Record a selection in the history for HistoryEncoder."""
        self._selection_history.append((agent_idx, reward))
        if len(self._selection_history) > self.config.history_length:
            self._selection_history = self._selection_history[-self.config.history_length :]

    def get_history_encoding(self) -> torch.Tensor:
        """Encode current selection history."""
        return self.history_encoder.encode(self._selection_history, self.config.max_agents)

    def reset_history(self) -> None:
        """Reset selection history and encoder hidden state."""
        self._selection_history.clear()
        self.history_encoder.reset()

    def state_dict_all(self) -> Dict[str, any]:
        """Get combined state dict for all components."""
        return {
            "feature_extractor": self.feature_extractor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "history_encoder": self.history_encoder.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "agent_to_idx": self._agent_to_idx,
            "idx_to_agent": self._idx_to_agent,
            "update_count": self._update_count,
        }

    def load_state_dict_all(self, state: Dict[str, any]) -> None:
        """Load combined state dict for all components."""
        self.feature_extractor.load_state_dict(state["feature_extractor"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.history_encoder.load_state_dict(state["history_encoder"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._agent_to_idx = state.get("agent_to_idx", {})
        self._idx_to_agent = state.get("idx_to_agent", {})
        self._update_count = state.get("update_count", 0)
