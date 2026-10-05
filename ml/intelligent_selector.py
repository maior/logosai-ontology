"""
Intelligent Agent Selector — GNN+RL integration wrapper.

This is the main entry point called by HybridAgentSelector:
    - select_agent(query, available_agents, deterministic) → (agent_id, metadata)
    - store_feedback(success, execution_result) → None

Features:
    - Lazy-loads sentence-transformers (~470MB) on first call
    - Background async training (never blocks requests)
    - Automatic model save/load with atomic writes
    - Confidence calibration for cold-start safety
    - Synthetic data generation for bootstrapping
"""

import asyncio
import json
import os
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .config import SelectorConfig
from .experience_buffer import Experience, ExperienceBuffer, SyntheticDataGenerator
from .gnn_encoder import GNNEncoder, KGTensorConverter
from .rl_policy import RLPolicy, action_confidence


class IntelligentAgentSelector:
    """
    GNN+RL intelligent agent selector.

    Integrates GNN graph encoding, PPO policy, and experience replay
    to learn optimal agent selection from interaction feedback.
    """

    def __init__(
        self,
        config: Optional[SelectorConfig] = None,
        auto_load: bool = True,
        device: str = "cpu",
    ):
        self.config = config or SelectorConfig(device=device)
        self.config.device = device
        self.device = torch.device(device)

        # Core components
        self.gnn_encoder = GNNEncoder(self.config.gnn).to(self.device)
        self.rl_policy = RLPolicy(self.config.rl, device=device)
        self.experience_buffer = ExperienceBuffer(self.config.buffer)
        self.kg_converter = KGTensorConverter()
        self.synthetic_generator = SyntheticDataGenerator(
            state_dim=self.config.rl.state_dim,
            query_dim=self.config.rl.query_embedding_dim,
        )

        # Lazy-loaded components
        self._embedding_model = None  # sentence-transformers (loaded on first use)
        self._knowledge_graph = None  # KG engine (loaded on first use)

        # Training state
        self._training_in_progress = False
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Pending transitions — **selection_id 키 맵** (피드백 파이프 수리,
        # 2026-08-03). 종전의 전역 단일 슬롯은 동시 요청이 서로를 덮어써
        # 피드백 95% 유실(889→48) + 보상 오귀속의 원인이었다 (진단 #4).
        # 피드백은 선택보다 나중에 오므로 id 로 되짚는다 (selection_recorder
        # 와 같은 규칙 — 시각으로 짝을 맞추면 동시 요청에서 어긋난다).
        self._pending: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._pending_cap = 512  # 피드백이 영영 안 오는 선택의 누수 방지

        # Statistics
        self.stats = {
            "selections": 0,
            "feedbacks": 0,
            "training_steps": 0,
            "last_train_loss": None,
            # 피드백 파이프 관측 (P0-1 과 같은 규율 — 유실은 세어져야 보인다)
            "feedback_no_pending": 0,     # id 미존재 (evict 됐거나 잘못된 id)
            "feedback_off_policy": 0,     # 실행 에이전트 ≠ 샘플 — 실행 쪽으로 귀속
            "feedback_unmapped_agent": 0, # 실행 에이전트가 미등록 — 폐기
            "pending_evicted": 0,         # cap 초과로 밀려난 선택 수
        }

        # Auto-load saved models
        if auto_load:
            self.load_models()

    # ─── Lazy Properties ──────────────────────────────────────────────

    @property
    def embedding_model(self):
        """Lazy-load sentence-transformers model (~470MB, ~3s first load)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.embedding_model
                logger.info(f"Loading embedding model: {model_name}...")
                self._embedding_model = SentenceTransformer(model_name, device=str(self.device))
                logger.info(f"Embedding model loaded: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model

    @property
    def knowledge_graph(self):
        """Lazy-load Knowledge Graph engine."""
        if self._knowledge_graph is None:
            try:
                from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
                self._knowledge_graph = get_knowledge_graph_engine()
                logger.debug("Knowledge graph engine loaded for ML selector")
            except Exception as e:
                logger.warning(f"Could not load KG engine: {e}")
        return self._knowledge_graph

    # ─── Core API ─────────────────────────────────────────────────────

    async def select_agent(
        self,
        query: str,
        available_agents: List[str],
        deterministic: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best agent for the given query.

        Called by HybridAgentSelector.select_agent() at line 206.

        Args:
            query: User query string
            available_agents: List of available agent IDs
            deterministic: If True, always pick highest probability

        Returns:
            (agent_id, metadata) where metadata includes confidence and value_estimate
        """
        start_time = time.time()
        self.stats["selections"] += 1

        # Ensure agents are registered
        self._ensure_agents_registered(available_agents)

        # 1. Embed query → [384]
        query_embedding = self._embed_query(query)

        # 2. Encode graph context → [64]
        graph_embedding = self._encode_graph(available_agents)

        # 3. Encode selection history → [64]
        history_embedding = self.rl_policy.get_history_encoding().to(self.device)

        # 4. Compose state → [512]
        state = torch.cat([query_embedding, graph_embedding, history_embedding], dim=0)

        # 5. Build available mask
        available_mask = self.rl_policy.build_available_mask(available_agents)

        # 6~8. 정책 분포 1회 계산 → 확신(자 = 최댓값 상대 마진) → 행동 선택.
        # P0-4 (2026-08-03): 종전에는 샘플링 먼저, 그 액션의 확률을 confidence
        # 로 썼다 — 구조적으로 ≈1/N 이라 게이트 0.7 도달 불가 (진단 문서).
        # 이제 확신은 분포에서 재고(action_confidence), 게이트 통과가 예상되면
        # 탐욕(argmax) — **채택할 거면 최선을 채택한다**. 미달이면 샘플링으로
        # 탐색을 유지해 학습 신호를 잃지 않는다.
        with torch.no_grad():
            features = self.rl_policy.feature_extractor(state.unsqueeze(0))
            probs = self.rl_policy.actor(features, available_mask.unsqueeze(0)).squeeze(0)
            value = self.rl_policy.critic(features).squeeze(0)

            margin_conf = action_confidence(probs)
            confidence = self._calibrate_confidence(margin_conf)

            if deterministic or confidence >= self.config.confidence_threshold:
                action = int(probs.argmax().item())
            else:
                action = int(torch.distributions.Categorical(probs).sample().item())
            log_prob = torch.log(probs[action] + 1e-10)
            raw_confidence = float(probs[action])

        # Map to agent ID
        agent_id = self.rl_policy.idx_to_agent(action)

        # 9. Store pending transition (reward filled on store_feedback)
        #    Detach to prevent grad graph from leaking into experience buffer
        selection_id = uuid.uuid4().hex
        self._pending[selection_id] = {
            "state": state.detach(),
            "action": action,
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "mask": available_mask.detach(),
            "agent_id": agent_id,
            "query": query,  # P0-5: 임베더 교체 시 재임베딩 가능하게
        }
        while len(self._pending) > self._pending_cap:
            evicted_id, _ = self._pending.popitem(last=False)
            self.stats["pending_evicted"] += 1
            logger.warning(f"pending eviction: {evicted_id} — 피드백이 오기 전에 "
                           f"cap({self._pending_cap}) 초과로 밀려남")

        elapsed_ms = (time.time() - start_time) * 1000

        metadata = {
            "selection_id": selection_id,        # 피드백이 이 선택을 되짚는 키
            "confidence": confidence,            # 게이트와 비교되는 값 (보정된 마진)
            "margin_confidence": margin_conf,    # 보정 전 마진 — 자 교체 효과 대조용
            "raw_confidence": raw_confidence,    # 선택 액션의 확률 (구 자, 관측용)
            "value_estimate": value.item(),
            "action_index": action,
            "elapsed_ms": elapsed_ms,
            "buffer_size": self.experience_buffer.size,
        }

        logger.debug(
            f"ML select: {agent_id} (conf={confidence:.1%}, raw={raw_confidence:.1%}, "
            f"value={value.item():.2f}, {elapsed_ms:.0f}ms)"
        )

        return agent_id, metadata

    async def store_feedback(
        self,
        success: bool,
        execution_result: Optional[Dict[str, Any]] = None,
        selection_id: Optional[str] = None,
        executed_agent: Optional[str] = None,
    ) -> None:
        """실행 피드백 저장 (+ 배경 학습 트리거).

        피드백 파이프 수리 (2026-08-03, 진단 #4):
        - **selection_id 로 되짚는다** — 전역 단일 슬롯은 동시 요청에서 서로를
          덮어써 유실·오귀속을 만들었다. id 없이 부르면 최신 pending 으로
          폴백한다 (직렬 호출 하위 호환 — 단 동시성 하에서는 부정확).
        - **executed_agent 가 샘플과 다르면 실행 쪽으로 귀속한다** — 관측된
          보상은 실제로 실행된 에이전트의 것이다. 채택률이 낮은 동안 fallback
          (KG+LLM) 경로의 성공/실패가 그대로 학습 라벨이 된다 (P0-5 imitation
          과 같은 철학: 행동 정책은 균등 prior). 미등록 에이전트면 폐기하되
          센다 — 조용한 유실 금지.
        """
        pending: Optional[Dict[str, Any]] = None
        if selection_id is not None:
            pending = self._pending.pop(selection_id, None)
            if pending is None:
                self.stats["feedback_no_pending"] += 1
                logger.warning(f"store_feedback: unknown selection_id "
                               f"{selection_id} (evicted or stale)")
                return
        else:
            if not self._pending:
                self.stats["feedback_no_pending"] += 1
                logger.warning("store_feedback called without pending selection")
                return
            _, pending = self._pending.popitem(last=True)

        # 귀속 대상 액션 결정 — 샘플 vs 실제 실행
        action = pending["action"]
        log_prob_value = float(pending["log_prob"].item())
        agent_id = pending["agent_id"]
        off_policy = False
        if executed_agent and executed_agent != pending["agent_id"]:
            idx = self.rl_policy._agent_to_idx.get(executed_agent)
            if idx is None or idx >= self.config.rl.max_agents:
                self.stats["feedback_unmapped_agent"] += 1
                logger.warning(f"store_feedback: executed agent "
                               f"{executed_agent} 미등록 — 경험 폐기(계수됨)")
                return
            import math
            action = idx
            agent_id = executed_agent
            # 행동 정책(KG+LLM)의 log_prob 은 기록에 없다 — 균등 prior
            # (bootstrap_from_mappings 와 같은 결정)
            log_prob_value = math.log(
                1.0 / max(1, self.rl_policy.num_registered_agents))
            off_policy = True
            self.stats["feedback_off_policy"] += 1

        # Compute reward
        reward = self._compute_reward(success, execution_result)

        # Record in history
        self.rl_policy.add_to_history(action, reward)

        # Create next_state (use current state as approximation for episodic tasks)
        next_state = torch.zeros_like(pending["state"])

        experience = Experience(
            state=pending["state"],
            action=action,
            reward=reward,
            next_state=next_state,
            done=True,
            info={
                "agent_id": agent_id,
                "query": pending.get("query"),  # P0-5: 재임베딩 가능 계약
                "success": success,
                "source": "runtime",
                "off_policy": "executed_fallback" if off_policy else None,
                "sampled_agent_id": pending["agent_id"],
                "log_prob": log_prob_value,
                "value": float(pending["value"].item()),
                "available_mask": pending["mask"],
            },
        )

        # Compute priority from TD error
        td_error = abs(reward - float(pending["value"].item()))
        priority = (td_error + self.config.buffer.min_priority) ** self.config.buffer.alpha

        self.experience_buffer.add(experience, priority)
        self.stats["feedbacks"] += 1

        # Maybe train in background
        if self.config.enable_background_training:
            await self._maybe_train_background()

        logger.debug(f"Feedback stored: agent={agent_id} reward={reward:.2f}, "
                     f"off_policy={off_policy}, buffer={self.experience_buffer.size}")

    # ─── Training ─────────────────────────────────────────────────────

    async def train_step(self, batch_size: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Run a single PPO training step from the replay buffer."""
        batch_size = batch_size or self.config.training_batch_size

        if self.experience_buffer.size < batch_size:
            return None

        # Sample batch
        experiences, weights, indices = self.experience_buffer.sample(batch_size)

        # Prepare batch tensors
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([1.0 if e.done else 0.0 for e in experiences], dtype=torch.float32, device=self.device)

        # Build masks (use full mask for training since we don't store per-experience masks)
        masks = torch.ones(batch_size, self.config.rl.max_agents, device=self.device)
        for i, e in enumerate(experiences):
            if "available_mask" in e.info:
                masks[i] = e.info["available_mask"]

        # Compute old log probs and values
        old_log_probs = torch.tensor(
            [e.info.get("log_prob", 0.0) for e in experiences],
            dtype=torch.float32,
            device=self.device,
        )

        # Compute values and next values for GAE
        with torch.no_grad():
            features = self.rl_policy.feature_extractor(states)
            values = self.rl_policy.critic(features)
            next_features = self.rl_policy.feature_extractor(next_states)
            next_values = self.rl_policy.critic(next_features)

        # Compute GAE
        advantages, returns = self.rl_policy.compute_gae(rewards, values, dones, next_values)

        # PPO update
        batch_data = {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
            "available_masks": masks,
        }

        result = self.rl_policy.update(batch_data)
        self.stats["training_steps"] += 1
        self.stats["last_train_loss"] = result["total_loss"]

        # Update priorities based on new TD errors
        with torch.no_grad():
            new_features = self.rl_policy.feature_extractor(states)
            new_values = self.rl_policy.critic(new_features)
        td_errors = torch.abs(returns - new_values).cpu().numpy()
        new_priorities = (td_errors + self.config.buffer.min_priority) ** self.config.buffer.alpha
        self.experience_buffer.update_priorities(indices, new_priorities)

        # Periodic save
        if self.stats["training_steps"] % self.config.save_interval == 0:
            self.save_models()

        return result

    async def train_offline(
        self,
        num_iterations: int = 100,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple training iterations (for cold-start or batch training).
        Runs synchronously (call via run_in_executor for async contexts).
        """
        batch_size = batch_size or self.config.training_batch_size
        results = []

        for i in range(num_iterations):
            result = await self.train_step(batch_size)
            if result:
                results.append(result)

        if results:
            avg_loss = np.mean([r["total_loss"] for r in results])
            avg_policy = np.mean([r["policy_loss"] for r in results])
            avg_value = np.mean([r["value_loss"] for r in results])
            logger.info(
                f"Offline training done: {len(results)} steps, "
                f"loss={avg_loss:.4f} (policy={avg_policy:.4f}, value={avg_value:.4f})"
            )
            return {
                "iterations": len(results),
                "avg_loss": avg_loss,
                "avg_policy_loss": avg_policy,
                "avg_value_loss": avg_value,
            }

        return {"iterations": 0, "avg_loss": 0.0}

    async def _maybe_train_background(self) -> None:
        """Trigger a training step in the background if conditions are met."""
        if self._training_in_progress:
            return
        if self.experience_buffer.size < self.config.min_buffer_size_for_training:
            return

        self._training_in_progress = True

        async def _train():
            try:
                await self.train_step()
            except Exception as e:
                logger.warning(f"Background training error: {e}")
            finally:
                self._training_in_progress = False

        asyncio.ensure_future(_train())

    # ─── Bootstrap from canonical labels (P0-5) ──────────────────────

    async def bootstrap_from_mappings(
        self,
        mappings: List[Dict[str, Any]],
        clear_buffer: bool = True,
        imitate_epochs: int = 20,
        train_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """kg_checkpoint 정본 라벨 → 버퍼 적재 (+ 선택적 오프라인 학습).

        P0-5 (2026-08-03): 정책은 randn 잡음 1000건으로 학습돼 있었다.
        정본은 query_agent_mapping (원 질의 텍스트 + selected_agent +
        success_rate). 여기서 지키는 계약:
        - info 에 **질의 텍스트 보존** — 임베더가 바뀌어도 재임베딩으로
          재사용 가능 (종전 스키마는 임베더 교체마다 데이터를 잃었다)
        - clear_buffer 기본 True — 잡음 폐기가 처방이다
        - 그래프·이력 컨텍스트는 0 벡터 — 기록 시점의 컨텍스트는 남아 있지
          않고, 지어낸 컨텍스트는 잡음이다
        """
        if not mappings:
            logger.warning("bootstrap_from_mappings: 라벨 0건 — 아무것도 하지 않음")
            return {"loaded": 0, "error": "no_mappings"}

        import math

        # 등록: 기존 + 라벨의 에이전트 합집합 (순서 보존 dedup)
        label_agents = sorted({m["agent"] for m in mappings})
        all_agents = list(dict.fromkeys(
            list(self.rl_policy._agent_to_idx.keys()) + label_agents
        ))
        self.rl_policy.register_agents(all_agents)

        discarded = 0
        if clear_buffer:
            discarded = self.experience_buffer.size
            self.experience_buffer = ExperienceBuffer(self.config.buffer)
            if discarded:
                logger.info(f"bootstrap: 기존 버퍼 {discarded}건 폐기 (잡음 학습 데이터)")

        zeros_g = torch.zeros(self.config.rl.graph_embedding_dim)
        zeros_h = torch.zeros(self.config.rl.history_dim)
        avail_mask = self.rl_policy.build_available_mask(all_agents).cpu()
        # 행동 정책의 log_prob 은 기록에 없다 — 균등 정책 가정 (0.0 기본값은
        # ratio=p_new 로 항상 <1 이 되는 편향)
        behavior_log_prob = math.log(1.0 / max(1, len(all_agents)))

        loaded = 0
        skipped = 0
        bc_states: List[torch.Tensor] = []
        bc_actions: List[int] = []
        bc_rewards: List[float] = []
        for m in mappings:
            idx = self.rl_policy._agent_to_idx.get(m["agent"])
            if idx is None or idx >= self.config.rl.max_agents:
                skipped += 1
                continue
            q_emb = self._embed_query(m["query"]).cpu().float()
            state = torch.cat([q_emb, zeros_g, zeros_h], dim=0)
            rate = float(m["success_rate"])
            # rate 1.0 → reward_success, 0.0 → reward_failure (선형)
            reward = (self.config.reward_failure
                      + rate * (self.config.reward_success - self.config.reward_failure))
            bc_states.append(state)
            bc_actions.append(idx)
            bc_rewards.append(reward)
            self.experience_buffer.add(Experience(
                state=state, action=idx, reward=reward,
                next_state=torch.zeros_like(state), done=True,
                info={
                    "query": m["query"],            # 재임베딩 가능 계약
                    "agent_id": m["agent"],
                    "source": "kg_checkpoint",
                    "success_rate": rate,
                    "log_prob": behavior_log_prob,
                    "available_mask": avail_mask,
                },
            ))
            loaded += 1

        if skipped:
            logger.warning(f"bootstrap: {skipped}건 skip (미등록 또는 max_agents 초과)")

        result: Dict[str, Any] = {
            "loaded": loaded, "skipped": skipped, "discarded": discarded,
            "buffer_size": self.experience_buffer.size,
            "registered_agents": len(all_agents),
        }
        # 라벨 학습의 본체는 행동 복제(CE) — PPO 는 균일 보상에서 advantage
        # 가 소멸해 라벨을 못 배운다 (RLPolicy.imitate docstring 의 실측).
        if imitate_epochs and bc_states:
            result["imitation"] = self.rl_policy.imitate(
                states=torch.stack(bc_states),
                actions=torch.tensor(bc_actions, dtype=torch.long),
                masks=avail_mask.unsqueeze(0).expand(len(bc_states), -1),
                rewards=torch.tensor(bc_rewards, dtype=torch.float32),
                epochs=imitate_epochs,
            )
            self.save_models()
        if train_iterations:
            result["training"] = await self.train_offline(num_iterations=train_iterations)
            self.save_models()
        return result

    # ─── Synthetic Data ───────────────────────────────────────────────

    async def generate_synthetic_data(
        self,
        num_samples: int = 1000,
        train_immediately: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate synthetic training data from KG and optionally train.

        Args:
            num_samples: Number of synthetic experiences to generate
            train_immediately: Whether to run offline training after generation
        """
        agent_ids = list(self.rl_policy._agent_to_idx.keys())
        if not agent_ids:
            return {"data_generated": 0, "error": "No agents registered"}

        kg = self.knowledge_graph
        graph = kg.graph if kg and hasattr(kg, "graph") else None

        if graph is None:
            # Generate random data if no KG available
            import networkx as nx
            graph = nx.MultiDiGraph()

        experiences = self.synthetic_generator.generate(graph, agent_ids, num_samples)

        for exp in experiences:
            self.experience_buffer.add(exp)

        result = {"data_generated": len(experiences), "buffer_size": self.experience_buffer.size}

        if train_immediately and len(experiences) >= self.config.min_buffer_size_for_training:
            num_iters = min(100, len(experiences) // self.config.training_batch_size)
            train_result = await self.train_offline(num_iterations=max(1, num_iters))
            result["training"] = train_result

        return result

    # ─── Model Persistence ────────────────────────────────────────────

    def save_models(self) -> None:
        """Save all model components to disk."""
        models_dir = Path(self.config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            # GNN encoder
            gnn_path = models_dir / "gnn.pt"
            torch.save(self.gnn_encoder.state_dict(), str(gnn_path))

            # RL policy
            policy_path = models_dir / "policy.pt"
            torch.save(self.rl_policy.state_dict_all(), str(policy_path))

            # Experience buffer
            buffer_path = models_dir / "buffer.pkl"
            self.experience_buffer.save(str(buffer_path))

            # Config
            config_path = models_dir / "config.json"
            config_data = {
                "device": self.config.device,
                "embedding_model": self.config.embedding_model,
                "stats": self.stats,
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Models saved to {models_dir}")
        except Exception as e:
            logger.error(f"Model save failed: {e}")

    def load_models(self) -> bool:
        """Load all model components from disk. Returns True on success."""
        models_dir = Path(self.config.models_dir)

        if not models_dir.exists():
            logger.debug(f"No saved models at {models_dir}")
            return False

        loaded = False

        # GNN encoder
        gnn_path = models_dir / "gnn.pt"
        if gnn_path.exists():
            try:
                state = torch.load(str(gnn_path), map_location=self.device, weights_only=True)
                self.gnn_encoder.load_state_dict(state)
                loaded = True
                logger.debug("GNN encoder loaded")
            except Exception as e:
                logger.warning(f"GNN load failed: {e}")

        # RL policy
        policy_path = models_dir / "policy.pt"
        if policy_path.exists():
            try:
                state = torch.load(str(policy_path), map_location=self.device, weights_only=False)
                self.rl_policy.load_state_dict_all(state)
                loaded = True
                logger.debug("RL policy loaded")
            except Exception as e:
                logger.warning(f"Policy load failed: {e}")

        # Experience buffer
        buffer_path = models_dir / "buffer.pkl"
        if buffer_path.exists():
            self.experience_buffer.load(str(buffer_path))

        # P0-6 (2026-08-03): stats 왕복. save 는 config.json 에 stats 를 쓰는데
        # load 가 복원하지 않아 재기동마다 0 으로 리셋 → 다음 save 가 0 을
        # 덮었다. "training_steps 0" 거짓 지표가 진단을 지연시킨 결함.
        config_path = models_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    saved = json.load(f)
                saved_stats = saved.get("stats") or {}
                if saved_stats:
                    self.stats.update(saved_stats)
                    logger.debug(f"Selector stats restored: {saved_stats}")
            except Exception as e:
                logger.warning(f"Stats restore failed: {e}")

        if loaded:
            logger.info(f"Models loaded from {models_dir}")
        return loaded

    # ─── Helpers ──────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> torch.Tensor:
        """Embed a query string into a vector using sentence-transformers."""
        embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        return embedding.to(self.device).float()

    def _encode_graph(self, available_agents: List[str]) -> torch.Tensor:
        """Encode knowledge graph context via GNN."""
        kg = self.knowledge_graph
        if kg is None or not hasattr(kg, "graph"):
            return torch.zeros(self.config.gnn.output_dim, device=self.device)

        try:
            self.gnn_encoder.eval()
            with torch.no_grad():
                data, node_map = self.kg_converter.convert(kg.graph)
                data = data.to(self.device)
                agent_indices = [node_map[aid] for aid in available_agents if aid in node_map]
                return self.gnn_encoder.encode_graph_context(data, agent_indices)
        except Exception as e:
            logger.warning(f"Graph encoding failed, using zero vector: {e}")
            return torch.zeros(self.config.gnn.output_dim, device=self.device)

    def _calibrate_confidence(self, margin_confidence: float) -> float:
        """냉시동 감쇠 — 0 이 아니라 무정보점 0.5 로 당긴다 (P0-4).

        종전에는 raw × (0.3+0.7·ratio) 로 0 을 향해 곱했다 — 1/N 자와 결합해
        어떤 정책도 게이트를 못 넘는 이중 벽이었다. 새 자(마진)의 무정보 값은
        0.5 이므로 감쇠도 0.5 를 향한다. 빈 버퍼의 상한은 0.5+0.5×0.3 = 0.65
        < 게이트 0.7 — **실데이터 없이는 채택 불가가 수식으로 보장**된다.
        """
        buffer_ratio = min(1.0, self.experience_buffer.size / max(1, self.config.cold_start_buffer_size))
        w = 0.3 + 0.7 * buffer_ratio
        return 0.5 + (margin_confidence - 0.5) * w

    def _compute_reward(self, success: bool, execution_result: Optional[Dict[str, Any]] = None) -> float:
        """Compute reward from execution outcome."""
        if success:
            reward = self.config.reward_success
            # Bonus for fast execution
            if execution_result and "execution_time" in execution_result:
                exec_time = execution_result["execution_time"]
                if exec_time < 5.0:
                    reward += 0.1
        else:
            reward = self.config.reward_failure

        return reward

    def _ensure_agents_registered(self, available_agents: List[str]) -> None:
        """Ensure all available agents are registered in the policy."""
        missing = [a for a in available_agents if a not in self.rl_policy._agent_to_idx]
        if missing:
            # Re-register with all known + new agents
            all_agents = list(self.rl_policy._agent_to_idx.keys()) + missing
            self.rl_policy.register_agents(all_agents)
            logger.debug(f"Registered {len(missing)} new agents: {missing}")
