"""
🎨 Visualization Engine - Graph Visualization Specialist

Generates visualization data and rich metadata for the knowledge graph.
"""

import networkx as nx
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from loguru import logger

from ...core.llm_manager import get_ontology_llm_manager, OntologyLLMType


class VisualizationEngine:
    """🎨 Visualization engine"""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.llm_manager = get_ontology_llm_manager()

        # Color palette
        self.node_colors = {
            "agent": "#fd79a8",
            "workflow": "#fdcb6e", 
            "task": "#74b9ff",
            "capability": "#00cec9",
            "domain": "#a29bfe",
            "query": "#e17055",
            "result": "#00b894",
            "concept": "#4ECDC4",
            "entity": "#45B7D1",
            "relation": "#96CEB4",
            "unknown": "#b2bec3"
        }
        
        self.edge_colors = {
            "triggers": "#fd79a8",
            "executes_with": "#74b9ff",
            "produces": "#00b894",
            "related_to": "#b2bec3",
            "is_a": "#e17055",
            "part_of": "#a29bfe"
        }
        
        logger.info("🎨 Visualization engine initialized")

    async def generate_visualization(self, max_nodes: int = 100) -> Dict[str, Any]:
        """Generate rich visualization data"""
        try:
            logger.info(f"🎨 Visualization generation started - max nodes: {max_nodes}")

            # Limit node count
            subgraph = self._create_limited_subgraph(max_nodes)

            # Generate node/edge data
            nodes = await self._generate_nodes(subgraph)
            edges = await self._generate_edges(subgraph)

            # Generate metadata
            metadata = await self._generate_metadata(nodes, edges, subgraph)

            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"nodes": [], "edges": [], "metadata": {"error": str(e)}}

    def _create_limited_subgraph(self, max_nodes: int) -> nx.MultiDiGraph:
        """Create subgraph with node count limit"""
        if self.graph.number_of_nodes() <= max_nodes:
            return self.graph

        # Select by importance
        node_degrees = dict(self.graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        return self.graph.subgraph([node for node, _ in top_nodes])

    async def _generate_nodes(self, subgraph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """Generate node data"""
        nodes = []

        for node_id, attrs in subgraph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')

            node_data = {
                "id": node_id,
                "label": self._get_display_label(node_id, attrs),
                "type": node_type,
                "size": self._calculate_node_size(node_id, subgraph),
                "color": self.node_colors.get(node_type, "#b2bec3"),
                "properties": attrs
            }

            nodes.append(node_data)

        return nodes

    async def _generate_edges(self, subgraph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """Generate edge data"""
        edges = []

        for i, (source, target, attrs) in enumerate(subgraph.edges(data=True)):
            relationship_type = attrs.get('relationship_type', attrs.get('predicate', 'related_to'))

            edge_data = {
                "id": f"edge_{i}",
                "source": source,
                "target": target,
                "label": relationship_type,
                "type": relationship_type,
                "color": self.edge_colors.get(relationship_type, "#b2bec3"),
                "weight": attrs.get('weight', 1.0)
            }

            edges.append(edge_data)

        return edges

    async def _generate_metadata(self, nodes: List[Dict], edges: List[Dict],
                                  subgraph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Generate metadata"""
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": self._get_type_distribution(nodes, "type"),
            "edge_types": self._get_type_distribution(edges, "type"),
            "generated_at": datetime.now().isoformat(),
            "version": "2.0"
        }

    def _get_display_label(self, node_id: str, attrs: Dict[str, Any]) -> str:
        """Generate display label"""
        if "agent_id" in attrs:
            return f"🤖 {attrs['agent_id']}"
        return str(node_id)[:20]

    def _calculate_node_size(self, node_id: str, subgraph: nx.MultiDiGraph) -> int:
        """Calculate node size"""
        degree = subgraph.degree(node_id)
        return min(10 + degree * 3, 40)

    def _get_type_distribution(self, items: List[Dict], type_key: str) -> Dict[str, int]:
        """Calculate type distribution"""
        distribution = {}
        for item in items:
            item_type = item.get(type_key, "unknown")
            distribution[item_type] = distribution.get(item_type, 0) + 1
        return distribution


logger.info("🎨 Visualization engine loaded!")