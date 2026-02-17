"""
ML Module — GNN + Reinforcement Learning for intelligent agent selection.

This module provides machine learning-based agent selection using:
- GraphSAGE + GAT (Graph Neural Networks) for knowledge graph encoding
- PPO (Proximal Policy Optimization) for agent selection policy
- Prioritized experience replay for training

Requirements:
    pip install torch sentence-transformers

Note:
    This module is optional. The ontology system works without it,
    falling back to the HybridAgentSelector (Knowledge Graph + LLM).
"""

try:
    from .intelligent_selector import IntelligentAgentSelector
    from .gnn_encoder import GNNEncoder
    from .rl_policy import RLPolicy
    from .experience_buffer import ExperienceBuffer

    __all__ = [
        "IntelligentAgentSelector",
        "GNNEncoder",
        "RLPolicy",
        "ExperienceBuffer",
    ]
    ML_AVAILABLE = True

except ImportError:
    __all__ = []
    ML_AVAILABLE = False
