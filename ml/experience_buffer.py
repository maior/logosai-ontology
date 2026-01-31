"""
📦 Experience Buffer for RL Training

학습 데이터 관리를 위한 경험 버퍼

Features:
- 순환 버퍼로 메모리 효율적 관리
- 우선순위 기반 샘플링 (PER) 지원
- 배치 생성 및 데이터 로더 제공
- 디스크 저장/로드 지원
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from pathlib import Path
from datetime import datetime
from loguru import logger


@dataclass
class Experience:
    """단일 경험 데이터"""
    query: str                          # 원본 쿼리
    query_embedding: np.ndarray         # 쿼리 임베딩
    graph_context: np.ndarray           # 그래프 컨텍스트 임베딩
    state: np.ndarray                   # 전체 상태 (query + graph + history)
    available_agents: List[str]         # 사용 가능한 에이전트 목록
    selected_agent: str                 # 선택된 에이전트
    action_idx: int                     # 선택된 에이전트 인덱스
    log_prob: float                     # 선택 로그 확률
    reward: float                       # 보상
    done: bool                          # 에피소드 종료 여부
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'query': self.query,
            'query_embedding': self.query_embedding.tolist(),
            'graph_context': self.graph_context.tolist(),
            'state': self.state.tolist(),
            'available_agents': self.available_agents,
            'selected_agent': self.selected_agent,
            'action_idx': self.action_idx,
            'log_prob': self.log_prob,
            'reward': self.reward,
            'done': self.done,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """딕셔너리에서 생성"""
        return cls(
            query=data['query'],
            query_embedding=np.array(data['query_embedding']),
            graph_context=np.array(data['graph_context']),
            state=np.array(data['state']),
            available_agents=data['available_agents'],
            selected_agent=data['selected_agent'],
            action_idx=data['action_idx'],
            log_prob=data['log_prob'],
            reward=data['reward'],
            done=data['done'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            metadata=data.get('metadata', {})
        )


class ExperienceBuffer:
    """
    📦 경험 버퍼

    RL 학습을 위한 경험 데이터 저장 및 관리
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 512,
        num_agents: int = 50,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # 버퍼 초기화
        self.buffer: deque = deque(maxlen=capacity)

        # 우선순위 기반 샘플링용
        if prioritized:
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.max_priority = 1.0

        # 통계
        self.stats = {
            'total_added': 0,
            'total_sampled': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'avg_reward': 0
        }

        logger.info(f"📦 Experience Buffer 초기화: capacity={capacity}, prioritized={prioritized}")

    def add(self, experience: Experience, priority: Optional[float] = None):
        """
        경험 추가

        Args:
            experience: 경험 데이터
            priority: 우선순위 (PER 사용 시)
        """
        self.buffer.append(experience)

        if self.prioritized:
            idx = len(self.buffer) - 1
            self.priorities[idx] = priority if priority is not None else self.max_priority

        # 통계 업데이트
        self.stats['total_added'] += 1
        if experience.reward > 0:
            self.stats['positive_rewards'] += 1
        elif experience.reward < 0:
            self.stats['negative_rewards'] += 1

        # 이동 평균 보상
        n = self.stats['total_added']
        self.stats['avg_reward'] = (
            self.stats['avg_reward'] * (n - 1) + experience.reward
        ) / n

    def sample(
        self,
        batch_size: int,
        as_tensors: bool = True
    ) -> Tuple[Any, ...]:
        """
        배치 샘플링

        Args:
            batch_size: 배치 크기
            as_tensors: PyTorch 텐서로 반환 여부

        Returns:
            states, actions, rewards, dones, log_probs, available_masks, (weights, indices if prioritized)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if self.prioritized:
            return self._prioritized_sample(batch_size, as_tensors)
        else:
            return self._uniform_sample(batch_size, as_tensors)

    def _uniform_sample(
        self,
        batch_size: int,
        as_tensors: bool
    ) -> Tuple[Any, ...]:
        """균일 샘플링"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]

        return self._batch_to_tensors(experiences) if as_tensors else experiences

    def _prioritized_sample(
        self,
        batch_size: int,
        as_tensors: bool
    ) -> Tuple[Any, ...]:
        """우선순위 기반 샘플링"""
        # 우선순위 확률 계산
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        experiences = [self.buffer[i] for i in indices]

        # 중요도 가중치 계산
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        self.stats['total_sampled'] += batch_size

        if as_tensors:
            batch = self._batch_to_tensors(experiences)
            return batch + (torch.tensor(weights, dtype=torch.float32), indices)
        else:
            return experiences, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """우선순위 업데이트 (PER용)"""
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority + 1e-6
                self.max_priority = max(self.max_priority, priority)

    def _batch_to_tensors(
        self,
        experiences: List[Experience]
    ) -> Tuple[torch.Tensor, ...]:
        """경험 배치를 텐서로 변환"""
        states = torch.tensor(
            np.array([e.state for e in experiences]),
            dtype=torch.float32
        )
        actions = torch.tensor(
            [e.action_idx for e in experiences],
            dtype=torch.long
        )
        rewards = torch.tensor(
            [e.reward for e in experiences],
            dtype=torch.float32
        )
        dones = torch.tensor(
            [e.done for e in experiences],
            dtype=torch.float32
        )
        log_probs = torch.tensor(
            [e.log_prob for e in experiences],
            dtype=torch.float32
        )

        # 에이전트 마스크 생성
        available_masks = torch.zeros(len(experiences), self.num_agents, dtype=torch.bool)
        # 실제 구현에서는 에이전트 ID to index 매핑 필요

        return states, actions, rewards, dones, log_probs, available_masks

    def get_recent(self, n: int) -> List[Experience]:
        """최근 n개 경험 반환"""
        return list(self.buffer)[-n:]

    def get_episode(self, start_idx: int) -> List[Experience]:
        """에피소드 단위로 경험 반환"""
        episode = []
        for i in range(start_idx, len(self.buffer)):
            exp = self.buffer[i]
            episode.append(exp)
            if exp.done:
                break
        return episode

    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()
        if self.prioritized:
            self.priorities = np.zeros(self.capacity, dtype=np.float32)
            self.max_priority = 1.0
        logger.info("🧹 Experience Buffer 초기화됨")

    def save(self, path: str):
        """버퍼 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'experiences': [e.to_dict() for e in self.buffer],
            'stats': self.stats,
            'config': {
                'capacity': self.capacity,
                'state_dim': self.state_dim,
                'num_agents': self.num_agents,
                'prioritized': self.prioritized
            }
        }

        if self.prioritized:
            data['priorities'] = self.priorities[:len(self.buffer)].tolist()
            data['max_priority'] = self.max_priority
            data['beta'] = self.beta

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"💾 Experience Buffer 저장: {path} ({len(self.buffer)} 경험)")

    def load(self, path: str):
        """버퍼 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.buffer = deque(
            [Experience.from_dict(e) for e in data['experiences']],
            maxlen=self.capacity
        )
        self.stats = data['stats']

        if self.prioritized and 'priorities' in data:
            self.priorities[:len(data['priorities'])] = np.array(data['priorities'])
            self.max_priority = data.get('max_priority', 1.0)
            self.beta = data.get('beta', self.beta)

        logger.info(f"📂 Experience Buffer 로드: {path} ({len(self.buffer)} 경험)")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'capacity': self.capacity,
            'fill_ratio': len(self.buffer) / self.capacity
        }

    def __len__(self) -> int:
        return len(self.buffer)


class SyntheticDataGenerator:
    """
    🔧 합성 학습 데이터 생성기

    LLM을 사용하여 다양한 쿼리-에이전트 매핑 데이터 생성
    """

    def __init__(
        self,
        agents: List[Dict[str, Any]],
        state_dim: int = 512,
        query_embedding_dim: int = 384,
        graph_embedding_dim: int = 64
    ):
        self.agents = agents
        self.agent_ids = [a['id'] for a in agents]
        self.state_dim = state_dim
        self.query_embedding_dim = query_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim

        # 쿼리 템플릿
        self.query_templates = {
            'finance': [
                "{company} 주가 알려줘",
                "{currency} 환율 확인해줘",
                "{stock} 실시간 시세 보여줘",
                "{company} 주식 분석해줘"
            ],
            'weather': [
                "{city} 날씨 어때?",
                "{location} 오늘 기온 알려줘",
                "{city} 주간 예보 보여줘",
                "{location} 미세먼지 확인해줘"
            ],
            'shopping': [
                "{product} 가격 비교해줘",
                "{item} 최저가 찾아줘",
                "{product} 리뷰 모아줘",
                "{brand} {product} 구매하고 싶어"
            ],
            'coding': [
                "{language} {function} 구현해줘",
                "{algorithm} 알고리즘 설명해줘",
                "{error} 버그 수정해줘",
                "{code} 코드 리뷰해줘"
            ],
            'research': [
                "{topic} 논문 찾아줘",
                "{subject} 분석해줘",
                "{data} 통계 분석해줘",
                "{field} 최신 연구 동향 알려줘"
            ]
        }

        # 도메인별 에이전트 매핑
        self.domain_agent_mapping = {
            'finance': ['internet_agent', 'analysis_agent'],
            'weather': ['weather_agent', 'internet_agent'],
            'shopping': ['shopping_agent', 'internet_agent'],
            'coding': ['code_agent', 'analysis_agent'],
            'research': ['analysis_agent', 'internet_agent']
        }

        # 템플릿 변수
        self.variables = {
            'company': ['삼성전자', '애플', '테슬라', 'LG전자', 'SK하이닉스'],
            'currency': ['달러', '엔화', '유로', '위안화'],
            'stock': ['코스피', '나스닥', '코스닥'],
            'city': ['서울', '부산', '뉴욕', '도쿄'],
            'location': ['강남', '판교', '제주도'],
            'product': ['아이폰', '갤럭시', '노트북', '에어팟'],
            'item': ['운동화', '가방', '시계'],
            'brand': ['나이키', '애플', '삼성'],
            'language': ['파이썬', '자바스크립트', '자바'],
            'function': ['정렬', 'API 호출', '데이터 처리'],
            'algorithm': ['퀵소트', 'BFS', 'DFS'],
            'error': ['TypeError', 'IndexError', 'ValueError'],
            'code': ['함수', '클래스', '모듈'],
            'topic': ['AI', '머신러닝', '딥러닝'],
            'subject': ['시장 동향', '사용자 행동', '성능'],
            'data': ['매출', '사용자', '트래픽'],
            'field': ['NLP', '컴퓨터 비전', '강화학습']
        }

    def generate_experience(
        self,
        domain: Optional[str] = None
    ) -> Experience:
        """단일 합성 경험 생성"""
        # 도메인 선택
        if domain is None:
            domain = np.random.choice(list(self.query_templates.keys()))

        # 쿼리 생성
        template = np.random.choice(self.query_templates[domain])
        query = self._fill_template(template)

        # 에이전트 선택 (도메인 기반)
        preferred_agents = self.domain_agent_mapping.get(domain, self.agent_ids[:2])
        available_agents = [a for a in self.agent_ids if a in preferred_agents or np.random.random() > 0.7]
        if not available_agents:
            available_agents = preferred_agents[:1] if preferred_agents else self.agent_ids[:1]

        # 올바른 에이전트가 선택될 확률
        is_correct = np.random.random() > 0.3
        if is_correct:
            selected_agent = np.random.choice(preferred_agents) if any(a in available_agents for a in preferred_agents) else available_agents[0]
        else:
            selected_agent = np.random.choice(available_agents)

        # 보상 계산
        if selected_agent in preferred_agents:
            reward = np.random.choice([1.0, 0.8, 0.5], p=[0.7, 0.2, 0.1])
        else:
            reward = np.random.choice([0.0, -0.5, -1.0], p=[0.3, 0.4, 0.3])

        # 임베딩 생성 (실제로는 Sentence Transformer 사용)
        query_embedding = np.random.randn(self.query_embedding_dim).astype(np.float32)
        graph_context = np.random.randn(self.graph_embedding_dim).astype(np.float32)
        history = np.zeros(self.graph_embedding_dim, dtype=np.float32)  # 첫 쿼리는 히스토리 없음

        state = np.concatenate([query_embedding, graph_context, history])

        action_idx = self.agent_ids.index(selected_agent) if selected_agent in self.agent_ids else 0

        return Experience(
            query=query,
            query_embedding=query_embedding,
            graph_context=graph_context,
            state=state,
            available_agents=available_agents,
            selected_agent=selected_agent,
            action_idx=action_idx,
            log_prob=np.log(0.3),  # 임시 값
            reward=reward,
            done=True,
            metadata={'domain': domain, 'is_correct': is_correct}
        )

    def _fill_template(self, template: str) -> str:
        """템플릿 변수 채우기"""
        import re
        pattern = r'\{(\w+)\}'

        def replace(match):
            var = match.group(1)
            if var in self.variables:
                return np.random.choice(self.variables[var])
            return match.group(0)

        return re.sub(pattern, replace, template)

    def generate_batch(
        self,
        size: int,
        domain_distribution: Optional[Dict[str, float]] = None
    ) -> List[Experience]:
        """배치 생성"""
        experiences = []

        if domain_distribution is None:
            domain_distribution = {d: 1.0 / len(self.query_templates) for d in self.query_templates}

        domains = list(domain_distribution.keys())
        probs = [domain_distribution[d] for d in domains]
        probs = [p / sum(probs) for p in probs]

        for _ in range(size):
            domain = np.random.choice(domains, p=probs)
            exp = self.generate_experience(domain)
            experiences.append(exp)

        return experiences


if __name__ == "__main__":
    print("🧪 Experience Buffer 테스트")

    # 버퍼 생성
    buffer = ExperienceBuffer(capacity=1000)

    # 합성 데이터 생성기
    agents = [
        {'id': 'internet_agent', 'domain': 'general'},
        {'id': 'weather_agent', 'domain': 'weather'},
        {'id': 'shopping_agent', 'domain': 'shopping'},
        {'id': 'code_agent', 'domain': 'coding'},
        {'id': 'analysis_agent', 'domain': 'research'}
    ]
    generator = SyntheticDataGenerator(agents)

    # 합성 데이터 생성 및 추가
    for _ in range(100):
        exp = generator.generate_experience()
        buffer.add(exp)

    print(f"   버퍼 크기: {len(buffer)}")
    print(f"   통계: {buffer.get_stats()}")

    # 샘플링 테스트
    states, actions, rewards, dones, log_probs, masks = buffer.sample(16)
    print(f"   샘플 크기: states={states.shape}, actions={actions.shape}")

    # 저장/로드 테스트
    buffer.save('/tmp/test_buffer.pkl')
    buffer2 = ExperienceBuffer()
    buffer2.load('/tmp/test_buffer.pkl')
    print(f"   로드된 버퍼 크기: {len(buffer2)}")

    print("✅ Experience Buffer 테스트 완료!")
