"""
🧠 Intelligent Agent Selector

GNN + RL 기반 지능형 에이전트 선택기
Knowledge Graph와 통합되어 학습하는 선택 시스템

Features:
- GNN으로 Knowledge Graph 구조 학습
- RL로 에이전트 선택 정책 최적화
- 기존 HybridAgentSelector v2.0과 통합
- 온라인/오프라인 학습 지원
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import asyncio
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False
    logger.warning("⚠️ sentence-transformers 미설치. 기본 임베딩 사용")

from .gnn_encoder import KnowledgeGraphEncoder, create_test_graph
from .rl_policy import AgentSelectionPolicy, PolicyOutput
from .experience_buffer import ExperienceBuffer, Experience, SyntheticDataGenerator


class QueryEmbedder:
    """쿼리 임베딩 생성기"""

    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self._model = None
        self.embedding_dim = 384  # MiniLM 기본 차원

    @property
    def model(self):
        if self._model is None and HAS_SENTENCE_TRANSFORMER:
            try:
                self._model = SentenceTransformer(self.model_name)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"📝 Query Embedder 로드: {self.model_name} (dim={self.embedding_dim})")
            except Exception as e:
                logger.warning(f"⚠️ SentenceTransformer 로드 실패: {e}")
        return self._model

    def embed(self, query: str) -> np.ndarray:
        """쿼리를 임베딩 벡터로 변환"""
        if self.model is not None:
            embedding = self.model.encode(query, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            # 폴백: 랜덤 임베딩
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def embed_batch(self, queries: List[str]) -> np.ndarray:
        """배치 임베딩"""
        if self.model is not None:
            embeddings = self.model.encode(queries, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            return np.random.randn(len(queries), self.embedding_dim).astype(np.float32)


class IntelligentAgentSelector:
    """
    🧠 지능형 에이전트 선택기

    GNN + RL 기반 Knowledge Graph 통합 에이전트 선택
    """

    def __init__(
        self,
        knowledge_graph=None,
        query_embedding_dim: int = 384,
        graph_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_agents: int = 50,
        learning_rate: float = 3e-4,
        buffer_capacity: int = 100000,
        device: str = 'cpu',
        model_dir: str = 'ontology/ml/models',
        auto_load: bool = True
    ):
        self.device = torch.device(device)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Knowledge Graph
        self._knowledge_graph = knowledge_graph

        # 차원 설정
        self.query_embedding_dim = query_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim
        state_dim = query_embedding_dim + graph_embedding_dim + graph_embedding_dim  # query + graph + history

        # 쿼리 임베더
        self.query_embedder = QueryEmbedder()
        if self.query_embedder.model is not None:
            self.query_embedding_dim = self.query_embedder.embedding_dim
            state_dim = self.query_embedding_dim + graph_embedding_dim + graph_embedding_dim

        # GNN 인코더
        self.gnn_encoder = KnowledgeGraphEncoder(
            input_dim=14,
            hidden_dim=128,
            output_dim=graph_embedding_dim,
            num_layers=3
        ).to(self.device)

        # RL 정책
        self.rl_policy = AgentSelectionPolicy(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            learning_rate=learning_rate,
            device=device
        )

        # 경험 버퍼
        self.experience_buffer = ExperienceBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            num_agents=num_agents,
            prioritized=True
        )

        # 히스토리 관리
        self.history_embedding = np.zeros(graph_embedding_dim, dtype=np.float32)
        self.history_decay = 0.9

        # 상태
        self.is_training = False
        self.selection_count = 0
        self.last_experience: Optional[Experience] = None

        # 통계
        self.stats = {
            'total_selections': 0,
            'total_training_steps': 0,
            'avg_reward': 0,
            'success_rate': 0,
            'mode': 'inference'
        }

        # 모델 로드
        if auto_load:
            self._try_load_models()

        logger.info(
            f"🧠 Intelligent Agent Selector 초기화 완료\n"
            f"   - State dim: {state_dim}\n"
            f"   - Graph dim: {graph_embedding_dim}\n"
            f"   - Device: {device}"
        )

    @property
    def knowledge_graph(self):
        """Knowledge Graph 지연 로딩"""
        if self._knowledge_graph is None:
            try:
                from ..engines.knowledge_graph_clean import KnowledgeGraphEngine
                self._knowledge_graph = KnowledgeGraphEngine(fast_mode=True)
                logger.info("📊 Knowledge Graph 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge Graph 로드 실패: {e}")
        return self._knowledge_graph

    async def select_agent(
        self,
        query: str,
        available_agents: List[str],
        agents_info: Optional[Dict[str, Any]] = None,
        deterministic: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        에이전트 선택

        Args:
            query: 사용자 쿼리
            available_agents: 사용 가능한 에이전트 목록
            agents_info: 에이전트 메타데이터
            deterministic: 결정적 선택 여부

        Returns:
            (selected_agent_id, metadata)
        """
        start_time = datetime.now()
        self.selection_count += 1

        # 1. 쿼리 임베딩
        query_embedding = self.query_embedder.embed(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)

        # 2. 그래프 컨텍스트 임베딩
        graph_context = await self._get_graph_context(query_tensor)

        # 3. 상태 구성 (query + graph + history)
        state = np.concatenate([
            query_embedding,
            graph_context.numpy(),
            self.history_embedding
        ])
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # 4. RL 정책으로 에이전트 선택
        selected_agent, policy_output = self.rl_policy.select_action(
            state_tensor,
            available_agents,
            deterministic=deterministic
        )

        # 5. 히스토리 업데이트
        self.history_embedding = (
            self.history_decay * self.history_embedding +
            (1 - self.history_decay) * graph_context.numpy()
        )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        # 6. 마지막 경험 저장 (피드백 대기)
        self.last_experience = Experience(
            query=query,
            query_embedding=query_embedding,
            graph_context=graph_context.numpy(),
            state=state,
            available_agents=available_agents,
            selected_agent=selected_agent,
            action_idx=policy_output.action,
            log_prob=policy_output.log_prob.item(),
            reward=0.0,  # 나중에 피드백으로 업데이트
            done=False
        )

        # 통계 업데이트
        self.stats['total_selections'] += 1

        metadata = {
            'selected_agent': selected_agent,
            'confidence': policy_output.action_prob,
            'action_probs': policy_output.action_probs.detach().cpu().numpy().tolist(),
            'value_estimate': policy_output.value.item(),
            'elapsed_ms': elapsed_ms,
            'method': 'gnn_rl',
            'selection_count': self.selection_count
        }

        logger.info(
            f"🧠 GNN+RL 선택: {selected_agent} "
            f"(신뢰도: {policy_output.action_prob:.1%}, {elapsed_ms:.0f}ms)"
        )

        return selected_agent, metadata

    async def _get_graph_context(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """그래프 컨텍스트 임베딩 계산"""
        try:
            if self.knowledge_graph and hasattr(self.knowledge_graph, 'graph_engine'):
                graph = self.knowledge_graph.graph_engine.graph

                if graph.number_of_nodes() > 0:
                    # GNN으로 그래프 인코딩
                    context = self.gnn_encoder.get_query_context_embedding(
                        graph,
                        query_embedding[:self.graph_embedding_dim],  # 차원 맞춤
                        top_k=5
                    )
                    return context

        except Exception as e:
            logger.warning(f"⚠️ 그래프 컨텍스트 계산 실패: {e}")

        # 폴백: 제로 벡터
        return torch.zeros(self.graph_embedding_dim, dtype=torch.float32)

    async def store_feedback(
        self,
        success: bool,
        reward: Optional[float] = None,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        피드백 저장 및 학습

        Args:
            success: 성공 여부
            reward: 보상 (없으면 자동 계산)
            execution_result: 실행 결과

        Returns:
            저장 성공 여부
        """
        if self.last_experience is None:
            logger.warning("⚠️ 저장할 경험이 없습니다")
            return False

        # 보상 계산
        if reward is None:
            if success:
                reward = 1.0
            else:
                reward = -0.5

        # 경험 업데이트
        self.last_experience.reward = reward
        self.last_experience.done = True
        self.last_experience.metadata.update({
            'success': success,
            'execution_result': execution_result
        })

        # 버퍼에 추가
        self.experience_buffer.add(self.last_experience, priority=abs(reward) + 0.1)

        # 통계 업데이트
        n = self.stats['total_selections']
        self.stats['avg_reward'] = (
            self.stats['avg_reward'] * (n - 1) + reward
        ) / n if n > 0 else reward

        if success:
            self.stats['success_rate'] = (
                self.stats['success_rate'] * (n - 1) + 1
            ) / n if n > 0 else 1.0
        else:
            self.stats['success_rate'] = (
                self.stats['success_rate'] * (n - 1)
            ) / n if n > 0 else 0.0

        # Knowledge Graph에도 피드백 저장 (기존 시스템 연동)
        if self.knowledge_graph:
            try:
                from ..core.hybrid_agent_selector import get_hybrid_selector
                selector = get_hybrid_selector()
                await selector.store_feedback(
                    self.last_experience.query,
                    self.last_experience.selected_agent,
                    success,
                    execution_result
                )
            except Exception as e:
                logger.warning(f"⚠️ KG 피드백 저장 실패: {e}")

        logger.info(f"📝 피드백 저장: reward={reward:.2f}, success={success}")

        # 온라인 학습 (버퍼가 충분히 쌓이면)
        if self.is_training and len(self.experience_buffer) >= 64:
            await self.train_step(batch_size=32)

        self.last_experience = None
        return True

    async def train_step(
        self,
        batch_size: int = 64,
        num_epochs: int = 4
    ) -> Dict[str, float]:
        """
        학습 스텝 실행

        Args:
            batch_size: 배치 크기
            num_epochs: 에폭 수

        Returns:
            학습 통계
        """
        if len(self.experience_buffer) < batch_size:
            return {'status': 'insufficient_data'}

        # 샘플링 (우선순위 버퍼는 weights와 indices도 반환)
        sample_result = self.experience_buffer.sample(batch_size)
        if len(sample_result) == 8:  # prioritized
            states, actions, rewards, dones, log_probs, masks, weights, indices = sample_result
        else:
            states, actions, rewards, dones, log_probs, masks = sample_result

        # PPO 업데이트
        train_stats = self.rl_policy.update(
            states, actions, rewards, dones, log_probs,
            num_epochs=num_epochs,
            batch_size=min(batch_size, 32)
        )

        self.stats['total_training_steps'] += 1

        logger.debug(
            f"📚 학습 스텝: policy_loss={train_stats['policy_loss']:.4f}, "
            f"value_loss={train_stats['value_loss']:.4f}"
        )

        return train_stats

    async def train_offline(
        self,
        num_iterations: int = 1000,
        batch_size: int = 64,
        num_epochs: int = 4,
        save_every: int = 100
    ) -> Dict[str, Any]:
        """
        오프라인 학습

        버퍼에 저장된 데이터로 배치 학습
        """
        self.stats['mode'] = 'training'
        training_stats = []

        for i in range(num_iterations):
            stats = await self.train_step(batch_size, num_epochs)
            training_stats.append(stats)

            if (i + 1) % save_every == 0:
                self.save_models()
                logger.info(f"📊 학습 진행: {i + 1}/{num_iterations}")

        self.stats['mode'] = 'inference'
        self.save_models()

        return {
            'iterations': num_iterations,
            'final_policy_loss': training_stats[-1].get('policy_loss', 0) if training_stats else 0,
            'final_value_loss': training_stats[-1].get('value_loss', 0) if training_stats else 0
        }

    def enable_training(self, enable: bool = True):
        """온라인 학습 활성화/비활성화"""
        self.is_training = enable
        self.stats['mode'] = 'online_training' if enable else 'inference'
        logger.info(f"🎓 학습 모드: {'ON' if enable else 'OFF'}")

    def save_models(self, prefix: str = 'intelligent_selector'):
        """모델 저장"""
        self.gnn_encoder.save(self.model_dir / f'{prefix}_gnn.pt')
        self.rl_policy.save(self.model_dir / f'{prefix}_policy.pt')
        self.experience_buffer.save(self.model_dir / f'{prefix}_buffer.pkl')
        logger.info(f"💾 모델 저장 완료: {self.model_dir}")

    def load_models(self, prefix: str = 'intelligent_selector'):
        """모델 로드"""
        gnn_path = self.model_dir / f'{prefix}_gnn.pt'
        policy_path = self.model_dir / f'{prefix}_policy.pt'
        buffer_path = self.model_dir / f'{prefix}_buffer.pkl'

        if gnn_path.exists():
            self.gnn_encoder = KnowledgeGraphEncoder.load(gnn_path)
        if policy_path.exists():
            self.rl_policy.load(policy_path)
        if buffer_path.exists():
            self.experience_buffer.load(buffer_path)

        logger.info(f"📂 모델 로드 완료: {self.model_dir}")

    def _try_load_models(self):
        """모델 로드 시도"""
        try:
            self.load_models()
        except Exception as e:
            logger.debug(f"ℹ️ 저장된 모델 없음 (새로 시작): {e}")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            'buffer_stats': self.experience_buffer.get_stats(),
            'policy_stats': self.rl_policy.get_stats()
        }

    async def generate_synthetic_data(
        self,
        num_samples: int = 1000,
        train_immediately: bool = True
    ) -> Dict[str, Any]:
        """
        합성 데이터 생성 및 학습

        초기 학습을 위한 합성 데이터 생성
        """
        agents = [
            {'id': 'internet_agent', 'domain': 'general'},
            {'id': 'weather_agent', 'domain': 'weather'},
            {'id': 'shopping_agent', 'domain': 'shopping'},
            {'id': 'code_agent', 'domain': 'coding'},
            {'id': 'analysis_agent', 'domain': 'research'}
        ]

        generator = SyntheticDataGenerator(agents)

        logger.info(f"🔧 합성 데이터 생성 중: {num_samples}개")

        for i in range(num_samples):
            exp = generator.generate_experience()
            self.experience_buffer.add(exp, priority=abs(exp.reward) + 0.1)

            if (i + 1) % 100 == 0:
                logger.debug(f"   생성 진행: {i + 1}/{num_samples}")

        logger.info(f"✅ 합성 데이터 생성 완료: {len(self.experience_buffer)}개")

        if train_immediately:
            logger.info("📚 초기 학습 시작...")
            train_result = await self.train_offline(
                num_iterations=min(num_samples // 64, 500),
                batch_size=64,
                num_epochs=4
            )
            return {'data_generated': num_samples, 'training': train_result}

        return {'data_generated': num_samples}


# 싱글톤 인스턴스
_intelligent_selector_instance = None


def get_intelligent_selector() -> IntelligentAgentSelector:
    """Intelligent Selector 싱글톤 인스턴스 반환"""
    global _intelligent_selector_instance
    if _intelligent_selector_instance is None:
        _intelligent_selector_instance = IntelligentAgentSelector()
    return _intelligent_selector_instance


async def test_intelligent_selector():
    """테스트 함수"""
    print("🧪 Intelligent Agent Selector 테스트")

    # 선택기 생성
    selector = IntelligentAgentSelector(auto_load=False)

    # 에이전트 등록
    agents = ['internet_agent', 'weather_agent', 'shopping_agent', 'code_agent', 'analysis_agent']
    selector.rl_policy.register_agents(agents)

    # 합성 데이터 생성
    await selector.generate_synthetic_data(num_samples=200, train_immediately=True)

    # 선택 테스트
    test_queries = [
        "삼성전자 주가 알려줘",
        "오늘 서울 날씨 어때?",
        "아이폰 16 가격 비교해줘"
    ]

    print("\n=== 선택 테스트 ===")
    for query in test_queries:
        selected, metadata = await selector.select_agent(query, agents[:3])
        print(f"쿼리: {query}")
        print(f"  선택: {selected} (신뢰도: {metadata['confidence']:.1%})")

        # 피드백 저장
        success = np.random.random() > 0.3
        await selector.store_feedback(success)

    print(f"\n통계: {selector.get_stats()}")
    print("✅ 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_intelligent_selector())
