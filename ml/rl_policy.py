"""
🎯 RL Policy for Agent Selection

PPO (Proximal Policy Optimization) 기반 에이전트 선택 정책

State:
- 쿼리 임베딩 (sentence transformer)
- 그래프 컨텍스트 임베딩 (GNN)
- 이전 선택 히스토리

Action:
- 에이전트 선택 (discrete action space)

Reward:
- +1.0: 성공적인 실행
- +0.5: 부분 성공
- -0.5: 실패
- -1.0: 타임아웃/에러
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class PolicyOutput:
    """정책 출력"""
    action: int                    # 선택된 에이전트 인덱스
    action_prob: float             # 선택 확률
    log_prob: torch.Tensor         # 로그 확률
    value: torch.Tensor            # 상태 가치
    entropy: torch.Tensor          # 엔트로피
    action_probs: torch.Tensor     # 전체 에이전트 확률 분포


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 네트워크

    Actor: 에이전트 선택 확률 분포
    Critic: 상태 가치 추정
    """

    def __init__(
        self,
        state_dim: int = 192,       # query(64) + graph(64) + history(64)
        hidden_dim: int = 256,
        num_agents: int = 50,       # 최대 에이전트 수
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_agents)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        state: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: 상태 벡터 (batch, state_dim)
            available_mask: 사용 가능한 에이전트 마스크 (batch, num_agents)

        Returns:
            action_logits: 에이전트별 로짓 (batch, num_agents)
            value: 상태 가치 (batch, 1)
        """
        # Shared features
        features = self.shared(state)

        # Actor (action logits)
        action_logits = self.actor(features)

        # 사용 불가능한 에이전트 마스킹
        if available_mask is not None:
            action_logits = action_logits.masked_fill(~available_mask, float('-inf'))

        # Critic (state value)
        value = self.critic(features)

        return action_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> PolicyOutput:
        """
        행동 선택

        Args:
            state: 상태 벡터
            available_mask: 사용 가능한 에이전트 마스크
            deterministic: True면 greedy 선택

        Returns:
            PolicyOutput
        """
        action_logits, value = self.forward(state, available_mask)
        action_probs = F.softmax(action_logits, dim=-1)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()

        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return PolicyOutput(
            action=action.item() if action.dim() == 0 else action[0].item(),
            action_prob=action_probs[0, action].item() if action.dim() == 0 else action_probs[0, action[0]].item(),
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            action_probs=action_probs
        )

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        행동 평가 (학습용)

        Returns:
            log_prob, value, entropy
        """
        action_logits, value = self.forward(state, available_mask)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, value.squeeze(-1), entropy


class AgentSelectionPolicy:
    """
    🎯 에이전트 선택 RL 정책

    PPO 알고리즘 기반 정책 학습
    """

    def __init__(
        self,
        state_dim: int = 192,
        hidden_dim: int = 256,
        num_agents: int = 50,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Actor-Critic 네트워크
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents
        ).to(self.device)

        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # 에이전트 ID 매핑
        self.agent_to_idx: Dict[str, int] = {}
        self.idx_to_agent: Dict[int, str] = {}

        # 통계
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_updates': 0
        }

        logger.info(f"🎯 RL Policy 초기화: state_dim={state_dim}, num_agents={num_agents}")

    def register_agents(self, agent_ids: List[str]):
        """에이전트 ID 등록"""
        for idx, agent_id in enumerate(agent_ids):
            self.agent_to_idx[agent_id] = idx
            self.idx_to_agent[idx] = agent_id

    def select_action(
        self,
        state: torch.Tensor,
        available_agents: List[str],
        deterministic: bool = False
    ) -> Tuple[str, PolicyOutput]:
        """
        에이전트 선택

        Args:
            state: 상태 벡터 (state_dim,)
            available_agents: 사용 가능한 에이전트 목록
            deterministic: greedy 선택 여부

        Returns:
            selected_agent_id, policy_output
        """
        self.network.eval()

        # 등록되지 않은 에이전트 먼저 처리
        for agent_id in available_agents:
            if agent_id not in self.agent_to_idx:
                new_idx = len(self.agent_to_idx)
                if new_idx < self.network.num_agents:
                    self.agent_to_idx[agent_id] = new_idx
                    self.idx_to_agent[new_idx] = agent_id

        # 사용 가능한 에이전트 마스크 생성 (전체 num_agents 크기)
        available_mask = torch.zeros(1, self.network.num_agents, dtype=torch.bool, device=self.device)
        for agent_id in available_agents:
            if agent_id in self.agent_to_idx:
                idx = self.agent_to_idx[agent_id]
                if idx < self.network.num_agents:
                    available_mask[0, idx] = True

        # 마스크가 모두 False면 첫 번째 에이전트 활성화
        if not available_mask.any():
            available_mask[0, 0] = True

        # 상태 전처리
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        # 행동 선택
        with torch.no_grad():
            output = self.network.get_action(state, available_mask, deterministic)

        # 에이전트 ID 반환
        selected_agent = self.idx_to_agent.get(output.action, available_agents[0])

        return selected_agent, output

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
        available_masks: Optional[torch.Tensor] = None,
        num_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        PPO 업데이트

        Args:
            states: (T, state_dim)
            actions: (T,)
            rewards: (T,)
            dones: (T,)
            old_log_probs: (T,)
            available_masks: (T, num_agents)
            num_epochs: 에폭 수
            batch_size: 배치 크기

        Returns:
            학습 통계
        """
        self.network.train()

        # 디바이스 이동
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        if available_masks is not None:
            available_masks = available_masks.to(self.device)

        # GAE (Generalized Advantage Estimation) 계산
        with torch.no_grad():
            _, values = self.network(states, available_masks)
            values = values.squeeze(-1)

        advantages, returns = self._compute_gae(rewards, values, dones)

        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 학습
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        dataset_size = len(states)

        for _ in range(num_epochs):
            # 랜덤 셔플
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = available_masks[batch_indices] if available_masks is not None else None

                # 현재 정책 평가
                log_probs, values, entropy = self.network.evaluate_action(
                    batch_states, batch_actions, batch_masks
                )

                # Policy Loss (PPO Clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy Bonus
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # 통계 업데이트
        avg_policy_loss = total_policy_loss / max(num_updates, 1)
        avg_value_loss = total_value_loss / max(num_updates, 1)
        avg_entropy = total_entropy / max(num_updates, 1)

        self.stats['policy_loss'].append(avg_policy_loss)
        self.stats['value_loss'].append(avg_value_loss)
        self.stats['entropy'].append(avg_entropy)
        self.stats['total_updates'] += num_updates

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'num_updates': num_updates
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE 계산"""
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        last_gae = 0
        last_value = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            mask = 1 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def save(self, path: str):
        """정책 저장"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'agent_to_idx': self.agent_to_idx,
            'idx_to_agent': self.idx_to_agent,
            'stats': self.stats
        }, path)
        logger.info(f"💾 RL Policy 저장: {path}")

    def load(self, path: str):
        """정책 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent_to_idx = checkpoint['agent_to_idx']
        self.idx_to_agent = checkpoint['idx_to_agent']
        self.stats = checkpoint['stats']
        logger.info(f"📂 RL Policy 로드: {path}")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            'total_updates': self.stats['total_updates'],
            'avg_policy_loss': np.mean(self.stats['policy_loss'][-100:]) if self.stats['policy_loss'] else 0,
            'avg_value_loss': np.mean(self.stats['value_loss'][-100:]) if self.stats['value_loss'] else 0,
            'avg_entropy': np.mean(self.stats['entropy'][-100:]) if self.stats['entropy'] else 0,
            'registered_agents': len(self.agent_to_idx)
        }


if __name__ == "__main__":
    print("🧪 RL Policy 테스트")

    # 정책 생성
    policy = AgentSelectionPolicy(state_dim=128, num_agents=10)

    # 에이전트 등록
    agents = ['internet_agent', 'weather_agent', 'shopping_agent', 'code_agent', 'analysis_agent']
    policy.register_agents(agents)

    # 행동 선택 테스트
    state = torch.randn(128)
    selected, output = policy.select_action(state, agents[:3])

    print(f"   선택된 에이전트: {selected}")
    print(f"   선택 확률: {output.action_prob:.3f}")
    print(f"   상태 가치: {output.value.item():.3f}")

    # 간단한 학습 테스트
    batch_size = 16
    states = torch.randn(batch_size, 128)
    actions = torch.randint(0, 3, (batch_size,))
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)
    old_log_probs = torch.randn(batch_size)

    stats = policy.update(states, actions, rewards, dones, old_log_probs, num_epochs=2)
    print(f"   학습 결과: policy_loss={stats['policy_loss']:.4f}, value_loss={stats['value_loss']:.4f}")

    print("✅ RL Policy 테스트 완료!")
