"""
🧠 GNN Encoder for Knowledge Graph

Knowledge Graph의 구조를 학습하여 노드 임베딩을 생성합니다.

Architecture:
- Input: NetworkX Graph → PyTorch Geometric Data
- Layers: GraphSAGE → GAT → Linear
- Output: Node Embeddings (128-dim)

Features:
- 에이전트 노드의 성공률, 사용 빈도 학습
- 쿼리 패턴과 에이전트 간의 관계 학습
- 도메인/카테고리 계층 구조 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger

try:
    import networkx as nx
except ImportError:
    nx = None


class NodeFeatureExtractor:
    """
    Knowledge Graph 노드에서 특징 벡터 추출

    노드 타입별 특징:
    - agent: success_rate, usage_count, capability_count, recency
    - query_agent_mapping: success_rate, usage_count, time_decay
    - category: query_count, agent_count
    - domain: specificity, coverage
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.node_type_embedding = {
            'agent': 0,
            'query_agent_mapping': 1,
            'query_category': 2,
            'domain': 3,
            'concept': 4,
            'unknown': 5
        }
        self.num_node_types = len(self.node_type_embedding)

    def extract_features(self, graph: 'nx.Graph') -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        그래프의 모든 노드에서 특징 추출

        Returns:
            features: (num_nodes, feature_dim) 텐서
            node_mapping: {node_id: index} 매핑
        """
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        num_nodes = len(node_mapping)

        # 특징 차원: node_type(6) + numeric_features(8) = 14
        feature_dim = self.num_node_types + 8
        features = torch.zeros(num_nodes, feature_dim)

        for node_id, attrs in graph.nodes(data=True):
            idx = node_mapping[node_id]

            # 1. 노드 타입 원-핫 인코딩
            node_type = attrs.get('type', 'unknown')
            type_idx = self.node_type_embedding.get(node_type, 5)
            features[idx, type_idx] = 1.0

            # 2. 수치 특징 (정규화됨)
            features[idx, self.num_node_types + 0] = attrs.get('success_rate', 0.5)
            features[idx, self.num_node_types + 1] = min(attrs.get('usage_count', 0) / 100, 1.0)
            features[idx, self.num_node_types + 2] = self._calculate_recency(attrs.get('last_used'))
            features[idx, self.num_node_types + 3] = len(attrs.get('capabilities', [])) / 10
            features[idx, self.num_node_types + 4] = len(attrs.get('keywords', [])) / 10
            features[idx, self.num_node_types + 5] = attrs.get('confidence', 0.5)
            features[idx, self.num_node_types + 6] = 1.0 if attrs.get('is_active', True) else 0.0
            features[idx, self.num_node_types + 7] = self._calculate_importance(graph, node_id)

        return features, node_mapping

    def _calculate_recency(self, last_used_str: Optional[str]) -> float:
        """최근 사용 정도 계산 (0~1)"""
        if not last_used_str:
            return 0.0
        try:
            last_used = datetime.fromisoformat(last_used_str.replace('Z', '+00:00'))
            if last_used.tzinfo:
                last_used = last_used.replace(tzinfo=None)
            days_ago = (datetime.now() - last_used).days
            return max(0, 1 - days_ago / 365)  # 1년 기준 감쇠
        except:
            return 0.0

    def _calculate_importance(self, graph: 'nx.Graph', node_id: str) -> float:
        """노드 중요도 계산 (연결 수 기반)"""
        try:
            degree = graph.degree(node_id)
            max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 1
            return degree / max(max_degree, 1)
        except:
            return 0.0


class KnowledgeGraphEncoder(nn.Module):
    """
    🧠 Knowledge Graph GNN Encoder

    GraphSAGE + GAT 기반 그래프 인코더
    Knowledge Graph의 구조적 정보를 학습하여 노드 임베딩 생성
    """

    def __init__(
        self,
        input_dim: int = 14,      # 노드 특징 차원
        hidden_dim: int = 128,     # 은닉층 차원
        output_dim: int = 64,      # 출력 임베딩 차원
        num_layers: int = 3,       # GNN 레이어 수
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Feature extractor
        self.feature_extractor = NodeFeatureExtractor()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim

            if use_attention and i == num_layers - 1:
                # 마지막 레이어는 GAT 사용
                self.convs.append(GATConv(in_channels, out_channels, heads=4, concat=False, dropout=dropout))
            else:
                # GraphSAGE 레이어
                self.convs.append(SAGEConv(in_channels, out_channels))

            self.norms.append(nn.LayerNorm(out_channels))

        # Output projection for query embedding alignment
        self.query_proj = nn.Linear(output_dim, output_dim)

        logger.info(f"🧠 GNN Encoder 초기화: {num_layers} layers, {hidden_dim}→{output_dim} dim")

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data 객체
                - x: 노드 특징 (num_nodes, input_dim)
                - edge_index: 엣지 인덱스 (2, num_edges)

        Returns:
            node_embeddings: (num_nodes, output_dim)
        """
        x, edge_index = data.x, data.edge_index

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN Layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                # Residual connection (if dimensions match)
                if x.shape == x_res.shape:
                    x = x + x_res

        return x

    def encode_graph(self, graph: 'nx.Graph') -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        NetworkX 그래프를 인코딩

        Args:
            graph: NetworkX Graph

        Returns:
            embeddings: (num_nodes, output_dim) 노드 임베딩
            node_mapping: {node_id: index} 매핑
        """
        # 특징 추출
        features, node_mapping = self.feature_extractor.extract_features(graph)

        # PyTorch Geometric Data로 변환
        data = self._nx_to_pyg(graph, features, node_mapping)

        # Forward pass
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(data)

        return embeddings, node_mapping

    def get_agent_embeddings(
        self,
        graph: 'nx.Graph',
        agent_ids: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        특정 에이전트들의 임베딩 추출

        Args:
            graph: NetworkX Graph
            agent_ids: 에이전트 ID 목록

        Returns:
            agent_embeddings: (num_agents, output_dim)
            valid_agent_ids: 유효한 에이전트 ID 목록
        """
        embeddings, node_mapping = self.encode_graph(graph)

        valid_agents = []
        agent_embeddings = []

        for agent_id in agent_ids:
            if agent_id in node_mapping:
                idx = node_mapping[agent_id]
                agent_embeddings.append(embeddings[idx])
                valid_agents.append(agent_id)

        if agent_embeddings:
            return torch.stack(agent_embeddings), valid_agents
        else:
            # 에이전트가 그래프에 없으면 랜덤 임베딩 반환
            return torch.randn(len(agent_ids), self.output_dim), agent_ids

    def get_query_context_embedding(
        self,
        graph: 'nx.Graph',
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> torch.Tensor:
        """
        쿼리와 관련된 그래프 컨텍스트 임베딩 생성

        Args:
            graph: NetworkX Graph
            query_embedding: 쿼리 임베딩 (query_dim,)
            top_k: 상위 k개 관련 노드 사용

        Returns:
            context_embedding: (output_dim,)
        """
        embeddings, node_mapping = self.encode_graph(graph)

        # 쿼리 임베딩을 그래프 임베딩 공간으로 투영
        query_proj = self.query_proj(query_embedding.unsqueeze(0))  # (1, output_dim)

        # 유사도 계산
        similarities = F.cosine_similarity(query_proj, embeddings, dim=1)  # (num_nodes,)

        # Top-k 노드 선택
        top_k = min(top_k, len(similarities))
        top_indices = torch.topk(similarities, top_k).indices

        # 가중 평균 컨텍스트
        top_embeddings = embeddings[top_indices]
        top_weights = F.softmax(similarities[top_indices], dim=0)
        context = torch.sum(top_embeddings * top_weights.unsqueeze(1), dim=0)

        return context

    def _nx_to_pyg(
        self,
        graph: 'nx.Graph',
        features: torch.Tensor,
        node_mapping: Dict[str, int]
    ) -> Data:
        """NetworkX 그래프를 PyTorch Geometric Data로 변환"""
        # 엣지 인덱스 생성
        edge_list = []
        for src, dst in graph.edges():
            if src in node_mapping and dst in node_mapping:
                edge_list.append([node_mapping[src], node_mapping[dst]])
                # 양방향 엣지 (undirected)
                edge_list.append([node_mapping[dst], node_mapping[src]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=features, edge_index=edge_index)

    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, path)
        logger.info(f"💾 GNN Encoder 저장: {path}")

    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraphEncoder':
        """모델 로드"""
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"📂 GNN Encoder 로드: {path}")
        return model


# 테스트용 더미 그래프 생성
def create_test_graph() -> 'nx.Graph':
    """테스트용 Knowledge Graph 생성"""
    if nx is None:
        raise ImportError("networkx is required")

    G = nx.Graph()

    # 에이전트 노드
    agents = ['internet_agent', 'weather_agent', 'shopping_agent', 'code_agent', 'analysis_agent']
    for agent in agents:
        G.add_node(agent, type='agent', success_rate=np.random.uniform(0.6, 0.95),
                   usage_count=np.random.randint(10, 200), capabilities=['search', 'analyze'],
                   last_used=datetime.now().isoformat())

    # 카테고리 노드
    categories = ['finance', 'weather', 'shopping', 'coding', 'research']
    for cat in categories:
        G.add_node(f'category_{cat}', type='query_category', name=cat)

    # 쿼리 패턴 노드
    patterns = [
        ('pattern_stock', 'internet_agent', 'finance'),
        ('pattern_weather', 'weather_agent', 'weather'),
        ('pattern_price', 'shopping_agent', 'shopping'),
        ('pattern_code', 'code_agent', 'coding'),
        ('pattern_analysis', 'analysis_agent', 'research')
    ]

    for pattern_id, agent, category in patterns:
        G.add_node(pattern_id, type='query_agent_mapping',
                   success_rate=np.random.uniform(0.7, 0.95),
                   usage_count=np.random.randint(5, 50),
                   selected_agent=agent, category=category,
                   last_used=datetime.now().isoformat())

        # 엣지 연결
        G.add_edge(pattern_id, agent)
        G.add_edge(pattern_id, f'category_{category}')

    return G


if __name__ == "__main__":
    # 테스트
    print("🧪 GNN Encoder 테스트")

    # 1. 테스트 그래프 생성
    graph = create_test_graph()
    print(f"   그래프: {graph.number_of_nodes()} 노드, {graph.number_of_edges()} 엣지")

    # 2. 인코더 생성
    encoder = KnowledgeGraphEncoder()

    # 3. 그래프 인코딩
    embeddings, node_mapping = encoder.encode_graph(graph)
    print(f"   임베딩: {embeddings.shape}")

    # 4. 에이전트 임베딩 추출
    agents = ['internet_agent', 'weather_agent', 'shopping_agent']
    agent_emb, valid_agents = encoder.get_agent_embeddings(graph, agents)
    print(f"   에이전트 임베딩: {agent_emb.shape}")

    print("✅ GNN Encoder 테스트 완료!")
