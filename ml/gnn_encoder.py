"""
GNN Encoder — Graph Neural Network for Knowledge Graph encoding.

Architecture:
    Layer 1: SAGEConv(14, 128) + BatchNorm + ReLU + Dropout
    Layer 2: GATConv(128, 128, heads=4) + BatchNorm + ReLU + Dropout
    Layer 3: SAGEConv(128, 64) + BatchNorm + ReLU
    Readout: Mean pooling over agent nodes → 64-dim vector

Includes KGTensorConverter for NetworkX MultiDiGraph → PyG Data conversion.
"""

import hashlib
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not available, GNN encoder will not work")

from .config import GNNConfig


# Node type indices for one-hot encoding
NODE_TYPE_MAP = {
    "agent": 0,
    "query_agent_mapping": 1,
}
# All other types map to index 2 (other)
NUM_NODE_TYPES = 3

# Known edge predicates
EDGE_PREDICATE_MAP = {
    "selected_for": 0,
    "has_capability": 1,
    "has_tag": 2,
    "belongs_to_category": 3,
    "similar_to": 4,
    "related_to": 5,
}


class KGTensorConverter:
    """
    Converts a NetworkX MultiDiGraph (Knowledge Graph) to a PyG Data object.

    Node features (14-dim):
        - type_onehot[3]: agent / query_mapping / other
        - degree_features[3]: in_degree, out_degree, total (log-normalized)
        - performance[3]: success_rate, usage_count(log), avg_exec_time(log)
        - temporal[3]: age_days(log), recency(1/age), update_frequency
        - special[2]: is_available(0/1), has_capabilities(0/1)

    Uses hash-based caching to avoid redundant conversions.
    """

    def __init__(self):
        self._cached_data: Optional[Data] = None
        self._cached_hash: Optional[str] = None
        self._node_id_to_idx: Dict[str, int] = {}

    def convert(self, graph, agent_ids: Optional[List[str]] = None) -> Tuple[Data, Dict[str, int]]:
        """
        Convert KG to PyG Data.

        Args:
            graph: NetworkX MultiDiGraph (or object with .graph attribute)
            agent_ids: Optional list of agent IDs to track

        Returns:
            (PyG Data object, node_id→index mapping)
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN encoder")

        # Unwrap if needed (e.g. KG engine wrapper with .graph attribute)
        g = graph
        if hasattr(graph, "graph") and isinstance(graph.graph, nx.Graph):
            g = graph.graph

        # Check cache
        graph_hash = self._compute_hash(g)
        if graph_hash == self._cached_hash and self._cached_data is not None:
            return self._cached_data, self._node_id_to_idx

        nodes = list(g.nodes(data=True))
        if not nodes:
            # Empty graph fallback
            data = Data(
                x=torch.zeros(1, 14),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
            self._node_id_to_idx = {}
            self._cached_data = data
            self._cached_hash = graph_hash
            return data, self._node_id_to_idx

        # Build node index mapping
        node_id_to_idx = {}
        for i, (node_id, _) in enumerate(nodes):
            node_id_to_idx[node_id] = i

        num_nodes = len(nodes)
        features = np.zeros((num_nodes, 14), dtype=np.float32)
        now = datetime.now()

        for i, (node_id, attrs) in enumerate(nodes):
            props = attrs.get("properties", {})
            node_type = attrs.get("type", "unknown")

            # [0:3] Type one-hot
            type_idx = NODE_TYPE_MAP.get(node_type, 2)
            features[i, type_idx] = 1.0

            # [3:6] Degree features (log-normalized)
            in_deg = g.in_degree(node_id)
            out_deg = g.out_degree(node_id)
            total_deg = in_deg + out_deg
            features[i, 3] = math.log1p(in_deg)
            features[i, 4] = math.log1p(out_deg)
            features[i, 5] = math.log1p(total_deg)

            # [6:9] Performance features
            success_rate = float(props.get("success_rate", attrs.get("success_rate", 0.0)))
            usage_count = float(props.get("usage_count", attrs.get("usage_count", 0)))
            exec_time = float(props.get("avg_execution_time", attrs.get("execution_time", 0.0)))
            features[i, 6] = success_rate
            features[i, 7] = math.log1p(usage_count)
            features[i, 8] = math.log1p(exec_time)

            # [9:12] Temporal features
            created_str = props.get("created_at", attrs.get("created_at", ""))
            age_days = self._compute_age_days(created_str, now)
            features[i, 9] = math.log1p(age_days)
            features[i, 10] = 1.0 / (1.0 + age_days)  # recency
            last_updated_str = props.get("last_updated", attrs.get("last_updated", ""))
            update_age = self._compute_age_days(last_updated_str, now)
            features[i, 11] = 1.0 / (1.0 + update_age)  # update frequency proxy

            # [12:14] Special features
            is_available = 1.0 if props.get("is_available", attrs.get("is_available", True)) else 0.0
            capabilities = props.get("capabilities", attrs.get("capabilities", []))
            has_capabilities = 1.0 if capabilities else 0.0
            features[i, 12] = is_available
            features[i, 13] = has_capabilities

        # Build edge index
        edges = list(g.edges(data=True))
        if edges:
            src_indices = []
            dst_indices = []
            for src, dst, _ in edges:
                if src in node_id_to_idx and dst in node_id_to_idx:
                    src_indices.append(node_id_to_idx[src])
                    dst_indices.append(node_id_to_idx[dst])
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        x = torch.from_numpy(features)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        # Cache
        self._cached_data = data
        self._cached_hash = graph_hash
        self._node_id_to_idx = node_id_to_idx

        logger.debug(f"KG converted: {num_nodes} nodes, {edge_index.size(1)} edges")
        return data, node_id_to_idx

    def get_agent_node_indices(self, graph, agent_ids: List[str]) -> List[int]:
        """Get node indices for specified agent IDs."""
        _, node_map = self.convert(graph)
        return [node_map[aid] for aid in agent_ids if aid in node_map]

    def _compute_hash(self, g) -> str:
        """Compute a lightweight hash for change detection."""
        n_nodes = g.number_of_nodes() if hasattr(g, "number_of_nodes") else 0
        n_edges = g.number_of_edges() if hasattr(g, "number_of_edges") else 0
        raw = f"{n_nodes}:{n_edges}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _compute_age_days(date_str: str, now: datetime) -> float:
        """Parse a date string and return age in days."""
        if not date_str:
            return 30.0  # default 30 days for unknown
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00").replace("+00:00", ""))
            delta = now - dt
            return max(0.0, delta.total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 30.0


class GNNEncoder(nn.Module):
    """
    3-layer GNN encoder: SAGEConv → GATConv → SAGEConv.

    Encodes a Knowledge Graph into a fixed-size vector representation.
    Uses mean pooling over agent nodes for the graph-level readout.
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNNEncoder")

        self.config = config or GNNConfig()
        c = self.config

        # Layer 1: SAGEConv
        self.conv1 = SAGEConv(c.node_feature_dim, c.hidden_dim)
        self.bn1 = nn.BatchNorm1d(c.hidden_dim)

        # Layer 2: GATConv (multi-head attention, concat=False for same dim)
        self.conv2 = GATConv(c.hidden_dim, c.hidden_dim, heads=c.heads, concat=False, dropout=c.dropout)
        self.bn2 = nn.BatchNorm1d(c.hidden_dim)

        # Layer 3: SAGEConv → output_dim
        self.conv3 = SAGEConv(c.hidden_dim, c.output_dim)
        self.bn3 = nn.BatchNorm1d(c.output_dim)

        self.dropout = nn.Dropout(c.dropout)

    def forward(self, data: "Data") -> torch.Tensor:
        """
        Forward pass through all GNN layers.

        Args:
            data: PyG Data with x [N, 14] and edge_index [2, E]

        Returns:
            Node embeddings [N, output_dim]
        """
        x, edge_index = data.x, data.edge_index

        # Handle single-node case (BatchNorm needs >1)
        use_bn = x.size(0) > 1

        # Layer 1
        x = self.conv1(x, edge_index)
        if use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        if use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        if use_bn:
            x = self.bn3(x)
        x = F.relu(x)

        return x  # [N, output_dim]

    def encode(self, data: "Data") -> torch.Tensor:
        """
        Encode entire graph into a single vector via mean pooling over all nodes.

        Returns:
            Graph embedding [output_dim]
        """
        node_embeddings = self.forward(data)
        return node_embeddings.mean(dim=0)  # [output_dim]

    def encode_for_agents(self, data: "Data", agent_node_indices: List[int]) -> torch.Tensor:
        """
        Encode graph and return embeddings for specific agent nodes.

        Args:
            data: PyG Data object
            agent_node_indices: List of node indices for agents

        Returns:
            Agent embeddings [len(agent_node_indices), output_dim]
        """
        node_embeddings = self.forward(data)

        if not agent_node_indices:
            return node_embeddings.mean(dim=0).unsqueeze(0)

        valid_indices = [i for i in agent_node_indices if i < node_embeddings.size(0)]
        if not valid_indices:
            return node_embeddings.mean(dim=0).unsqueeze(0)

        return node_embeddings[valid_indices]  # [K, output_dim]

    def encode_graph_context(self, data: "Data", agent_node_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Produce a single graph context vector for state composition.

        If agent indices are provided, pools over agent nodes.
        Otherwise, pools over all nodes.

        Returns:
            [output_dim] tensor
        """
        if agent_node_indices:
            agent_embs = self.encode_for_agents(data, agent_node_indices)
            return agent_embs.mean(dim=0)
        return self.encode(data)
