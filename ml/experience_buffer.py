"""
Experience Buffer — Prioritized experience replay with Sum Tree.

Supports:
- O(log N) prioritized sampling via Sum Tree
- Importance sampling weight correction
- Disk persistence (pickle-based save/load)
- Synthetic data generation from Knowledge Graph
"""

import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .config import BufferConfig


class Experience(NamedTuple):
    """Single experience tuple stored in the replay buffer."""

    state: torch.Tensor  # [state_dim]
    action: int  # agent index
    reward: float
    next_state: torch.Tensor  # [state_dim]
    done: bool
    info: Dict[str, Any]  # metadata (agent_id, query, timestamp, etc.)


class SumTree:
    """
    Binary sum tree for O(log N) prioritized sampling.

    Each leaf stores a priority value. Internal nodes store the sum of children.
    Sampling proportional to priority: pick random s in [0, total], traverse tree.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    @property
    def total(self) -> float:
        """Total priority sum (root node)."""
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """Add a new experience with given priority."""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self._update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Sample by cumulative priority.

        Args:
            s: Random value in [0, total)

        Returns:
            (tree_index, priority, data)
        """
        idx = 0  # start at root
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def update(self, tree_idx: int, priority: float) -> None:
        """Update the priority of an existing entry."""
        self._update(tree_idx, priority)

    def _update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change


class ExperienceBuffer:
    """
    Prioritized experience replay buffer.

    Uses a Sum Tree for O(log N) proportional priority sampling with
    importance sampling weight correction.
    """

    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        self.tree = SumTree(self.config.capacity)
        self._beta_step = 0
        self._max_priority = 1.0

    @property
    def size(self) -> int:
        return self.tree.size

    def add(self, experience: Experience, priority: Optional[float] = None) -> None:
        """
        Add experience with optional explicit priority.
        If priority is not given, uses max observed priority (ensures new
        experiences are sampled at least once).
        """
        if priority is None:
            priority = self._max_priority
        priority = max(priority, self.config.min_priority)
        self.tree.add(priority, experience)

    def all_experiences(self) -> List[Experience]:
        """저장 순서대로 전량 (학습 루프의 라벨 수집·OPE 로그용 사본 뷰).

        sample() 은 우선순위 편향 표집이라 "버퍼에 무엇이 있나"의 답이 아니다.
        """
        return [e for e in self.tree.data if e is not None]

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """
        Sample a prioritized batch.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            (experiences, importance_weights, tree_indices)
        """
        if self.size < batch_size:
            batch_size = self.size

        beta = self._current_beta()
        experiences = []
        weights = np.zeros(batch_size, dtype=np.float32)
        indices = []

        segment = self.tree.total / batch_size
        min_prob = self.config.min_priority / self.tree.total if self.tree.total > 0 else 1e-6

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = random.uniform(low, high)
            tree_idx, priority, data = self.tree.get(s)

            if data is None:
                # Fallback: sample again from full range
                s = random.uniform(0, self.tree.total - 1e-8)
                tree_idx, priority, data = self.tree.get(s)

            prob = priority / self.tree.total if self.tree.total > 0 else 1.0
            weight = (prob * self.size) ** (-beta)
            experiences.append(data)
            weights[i] = weight
            indices.append(tree_idx)

        # Normalize weights
        max_weight = weights.max()
        if max_weight > 0:
            weights /= max_weight

        self._beta_step += 1
        return experiences, weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            priority = max(float(priority), self.config.min_priority)
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)

    def _current_beta(self) -> float:
        """Anneal beta from beta_start to beta_end over beta_frames."""
        fraction = min(1.0, self._beta_step / max(1, self.config.beta_frames))
        return self.config.beta_start + fraction * (self.config.beta_end - self.config.beta_start)

    def save(self, path: str) -> None:
        """Save buffer to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiences": [self.tree.data[i] for i in range(self.size)],
            "priorities": [float(self.tree.tree[i + self.tree.capacity - 1]) for i in range(self.size)],
            "beta_step": self._beta_step,
            "max_priority": self._max_priority,
        }

        tmp_path = save_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
        tmp_path.rename(save_path)
        logger.debug(f"Buffer saved: {self.size} experiences to {path}")

    def load(self, path: str) -> bool:
        """Load buffer from disk. Returns True on success."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.tree = SumTree(self.config.capacity)
            self._beta_step = data.get("beta_step", 0)
            self._max_priority = data.get("max_priority", 1.0)

            for exp, priority in zip(data["experiences"], data["priorities"]):
                if exp is not None:
                    self.tree.add(priority, exp)

            logger.info(f"Buffer loaded: {self.size} experiences from {path}")
            return True
        except Exception as e:
            # 좁은 catch 는 지뢰였다: 다른 프로세스 루트에서 피클된 파일은
            # ModuleNotFoundError("ml") 를 내는데 그게 새어나가 셀렉터
            # 생성자를 통째로 죽였다 (2026-08-03 실측). 로드 실패는 어떤
            # 이유든 "빈 버퍼로 degrade + warning" 이 계약이다.
            logger.warning(f"Buffer load failed ({type(e).__name__}): {e}")
            self.tree = SumTree(self.config.capacity)
            return False


class SyntheticDataGenerator:
    """
    Generates synthetic training data from Knowledge Graph structure.

    Extracts agent-query mappings from the KG and creates synthetic
    experiences for cold-start bootstrapping.
    """

    def __init__(self, state_dim: int = 512, query_dim: int = 384):
        self.state_dim = state_dim
        self.query_dim = query_dim

    def generate(
        self,
        knowledge_graph,  # nx.MultiDiGraph
        agent_ids: List[str],
        num_samples: int = 1000,
        allow_random: bool = False,
    ) -> List[Experience]:
        """
        Generate synthetic experiences from KG patterns.

        Extracts successful query→agent mappings from the KG and creates
        synthetic state vectors with those patterns.
        """
        # 엔진 래퍼(kg.graph)만 언래핑한다. **nx 그래프 자신도 `.graph` 속성
        # (그래프-레벨 속성 dict)을 갖는다** — 조건이 hasattr 뿐이면 진짜
        # 그래프가 빈 dict 로 강등돼 패턴이 항상 0 → randn 폴백이 항상 발화.
        # 잡음 1000건의 실제 메커니즘이 이 줄이었다 (2026-08-03 실측:
        # KG 를 제대로 줘도 잡음이 나왔다 — 진단의 "knowledge_graph=None
        # 이라서"보다 한 겹 깊다).
        graph = knowledge_graph
        if hasattr(knowledge_graph, "graph") and not hasattr(knowledge_graph, "nodes"):
            graph = knowledge_graph.graph

        # Extract agent→success patterns from KG
        agent_patterns = self._extract_patterns(graph, agent_ids)

        if not agent_patterns:
            # P0-5 (2026-08-03): 이 폴백이 조용히 발화해 randn 잡음 1000건이
            # 정책을 학습시켰다 (채택률 0% 진단). 잡음 생성은 명시적 선택
            # (allow_random=True)으로만 — 기본은 빈 목록으로 거부한다.
            if not allow_random:
                logger.warning(
                    "SyntheticDataGenerator: KG 패턴 0건 — randn 잡음 생성 거부 "
                    "(allow_random=True 로만 허용). 정본 라벨은 ml/bootstrap.py 참고"
                )
                return []
            logger.warning("No patterns found in KG, generating random data (explicitly allowed)")
            return self._generate_random(agent_ids, num_samples)

        experiences = []
        agent_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

        for _ in range(num_samples):
            # 70% from KG patterns, 30% random exploration
            if random.random() < 0.7 and agent_patterns:
                agent_id = random.choice(list(agent_patterns.keys()))
                pattern = random.choice(agent_patterns[agent_id])
                reward = pattern.get("success_rate", 0.8)
            else:
                agent_id = random.choice(agent_ids)
                reward = 0.2  # low reward for random

            if agent_id not in agent_to_idx:
                continue

            action = agent_to_idx[agent_id]
            state = torch.randn(self.state_dim)
            next_state = torch.randn(self.state_dim)

            exp = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=True,
                info={"agent_id": agent_id, "synthetic": True},
            )
            experiences.append(exp)

        logger.info(f"Generated {len(experiences)} synthetic experiences ({len(agent_patterns)} agent patterns)")
        return experiences

    def _extract_patterns(self, graph, agent_ids: List[str]) -> Dict[str, List[Dict]]:
        """Extract query→agent success patterns from KG."""
        patterns: Dict[str, List[Dict]] = {}

        if not hasattr(graph, "nodes"):
            return patterns

        agent_id_set = set(agent_ids)

        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "")

            # Find query_agent_mapping nodes
            if node_type == "query_agent_mapping":
                success_rate = float(attrs.get("properties", {}).get("success_rate", attrs.get("success_rate", 0.5)))
                usage_count = int(attrs.get("properties", {}).get("usage_count", attrs.get("usage_count", 0)))
                category = attrs.get("properties", {}).get("category", attrs.get("category", "unknown"))

                # Find connected agent via edges
                for _, target, edge_data in graph.out_edges(node_id, data=True):
                    target_attrs = graph.nodes.get(target, {})
                    if target_attrs.get("type") == "agent" and target in agent_id_set:
                        if target not in patterns:
                            patterns[target] = []
                        patterns[target].append({
                            "category": category,
                            "success_rate": success_rate,
                            "usage_count": usage_count,
                        })

                # Also check incoming edges from agents
                for source, _, edge_data in graph.in_edges(node_id, data=True):
                    source_attrs = graph.nodes.get(source, {})
                    if source_attrs.get("type") == "agent" and source in agent_id_set:
                        if source not in patterns:
                            patterns[source] = []
                        patterns[source].append({
                            "category": category,
                            "success_rate": success_rate,
                            "usage_count": usage_count,
                        })

        return patterns

    def _generate_random(self, agent_ids: List[str], num_samples: int) -> List[Experience]:
        """Fallback: generate random experiences."""
        agent_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        experiences = []

        for _ in range(num_samples):
            agent_id = random.choice(agent_ids)
            experiences.append(
                Experience(
                    state=torch.randn(self.state_dim),
                    action=agent_to_idx[agent_id],
                    reward=random.uniform(0.0, 0.5),
                    next_state=torch.randn(self.state_dim),
                    done=True,
                    info={"agent_id": agent_id, "synthetic": True, "random": True},
                )
            )

        return experiences
