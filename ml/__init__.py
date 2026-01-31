"""
🧠 LogosAI Ontology ML Module

GNN + RL 기반 지능형 에이전트 선택 시스템

Components:
- GNN Encoder: Knowledge Graph를 임베딩으로 변환
- RL Policy: 에이전트 선택 정책 학습
- Experience Buffer: 학습 데이터 관리
- Trainer: 통합 학습 파이프라인
"""

from .gnn_encoder import KnowledgeGraphEncoder
from .rl_policy import AgentSelectionPolicy
from .experience_buffer import ExperienceBuffer
from .intelligent_selector import IntelligentAgentSelector

__all__ = [
    'KnowledgeGraphEncoder',
    'AgentSelectionPolicy',
    'ExperienceBuffer',
    'IntelligentAgentSelector'
]

__version__ = '0.1.0'
