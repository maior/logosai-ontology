"""
🧠 Knowledge Graph Engine

Ontology-based knowledge graph management and visualization.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
import time

from ..core.models import SemanticQuery, AgentExecutionResult
from ..core.interfaces import KnowledgeGraph


class SimpleKnowledgeGraphEngine(KnowledgeGraph):
    """🧠 Simple knowledge graph engine - cleanly restructured"""

    def __init__(self):
        # Knowledge graph using NetworkX
        self.graph = nx.MultiDiGraph()
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Node and relation counters
        self.node_counter = 0
        self.relation_counter = 0

        logger.info("🧠 Simple knowledge graph engine initialized")

    async def add_concept(self, concept: str, concept_type: str, properties: Dict[str, Any]) -> bool:
        """Add concept"""
        try:
            # Generate node ID
            node_id = f"{concept_type}_{concept}_{self.node_counter}"
            self.node_counter += 1

            # Build node attributes
            node_attrs = {
                "concept": concept,
                "type": concept_type,
                "created_at": datetime.now(),
                **properties
            }

            # Add node to graph
            self.graph.add_node(node_id, **node_attrs)
            self.last_updated = datetime.now()

            logger.debug(f"Concept added: {concept} ({concept_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to add concept: {e}")
            return False

    async def add_relation(self, subject: str, predicate: str, object: str, properties: Dict[str, Any]) -> bool:
        """Add relation"""
        try:
            # Generate relation ID
            relation_id = f"rel_{self.relation_counter}"
            self.relation_counter += 1

            # Build edge attributes
            edge_attrs = {
                "predicate": predicate,
                "relation_id": relation_id,
                "created_at": datetime.now(),
                "weight": properties.get("weight", 1.0),
                "confidence": properties.get("confidence", 0.8),
                **properties
            }

            # Add edge to graph
            self.graph.add_edge(subject, object, key=relation_id, **edge_attrs)
            self.last_updated = datetime.now()

            logger.debug(f"Relation added: {subject} --[{predicate}]--> {object}")
            return True

        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            return False

    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Graph query"""
        try:
            results = []
            query_lower = query.lower()

            # Node search
            for node_id, attrs in self.graph.nodes(data=True):
                if query_lower in str(attrs.get("concept", "")).lower():
                    results.append({
                        "type": "node",
                        "id": node_id,
                        "concept": attrs.get("concept"),
                        "node_type": attrs.get("type"),
                        "attributes": attrs
                    })

            # Relation search
            for source, target, key, attrs in self.graph.edges(keys=True, data=True):
                if query_lower in str(attrs.get("predicate", "")).lower():
                    results.append({
                        "type": "edge",
                        "source": source,
                        "target": target,
                        "predicate": attrs.get("predicate"),
                        "attributes": attrs
                    })

            logger.info(f"Query '{query}' results: {len(results)}")
            return results

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
    
    def generate_visualization(self, max_nodes: int = 100) -> Dict[str, Any]:
        """Generate visualization data - compatible with advanced_multi_agent_manager"""
        try:
            # Limit nodes if too many
            all_nodes = list(self.graph.nodes(data=True))
            if len(all_nodes) > max_nodes:
                # Prioritize high-degree nodes
                node_degrees = [(node, self.graph.degree(node)) for node, _ in all_nodes]
                node_degrees.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = [node for node, _ in node_degrees[:max_nodes]]
                subgraph = self.graph.subgraph(selected_nodes)
            else:
                subgraph = self.graph

            # Build node data
            nodes = []
            for node_id, attrs in subgraph.nodes(data=True):
                nodes.append({
                    "id": node_id,
                    "label": self._get_display_label(node_id, attrs),
                    "type": attrs.get("type", "unknown"),
                    "color": self._get_node_color_by_type(attrs.get("type", "unknown")),
                    "size": self._calculate_node_size(node_id, subgraph),
                    "degree": subgraph.degree(node_id),
                    "attributes": attrs
                })
            
            # Build edge data
            edges = []
            for source, target, key, attrs in subgraph.edges(keys=True, data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "label": self._get_relation_display_name(attrs.get("predicate", "")),
                    "predicate": attrs.get("predicate", ""),
                    "color": self._get_edge_color_by_type(attrs.get("predicate", "")),
                    "weight": attrs.get("weight", 1.0),
                    "confidence": attrs.get("confidence", 0.8),
                    "attributes": attrs
                })
            
            # Statistics info
            stats = self._generate_graph_stats(nodes, edges)

            # Color legend
            legend = self._get_color_legend()
            
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": stats,
                "legend": legend,
                "layout_recommendation": self._recommend_layout(nodes, edges),
                "generated_at": datetime.now().isoformat(),
                "total_nodes_in_graph": self.graph.number_of_nodes(),
                "total_edges_in_graph": self.graph.number_of_edges(),
                "nodes_shown": len(nodes),
                "edges_shown": len(edges)
            }
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {
                "nodes": [],
                "edges": [],
                "stats": {},
                "legend": {},
                "error": str(e)
            }

    async def add_semantic_query_concepts(self, semantic_query: SemanticQuery) -> bool:
        """Add concepts from semantic query to the graph"""
        try:
            query_id = semantic_query.query_id

            # Add query node
            await self.add_concept(
                concept=query_id,
                concept_type="query",
                properties={
                "natural_language": semantic_query.natural_language,
                    "query_text": semantic_query.natural_language,
                    "created_at": semantic_query.created_at,
                    "complexity": getattr(semantic_query, 'complexity', 'unknown')
                }
            )
            
            logger.info(f"Semantic query concept addition complete: {query_id}")
            return True

        except Exception as e:
            logger.error(f"Semantic query concept addition failed: {e}")
            return False

    async def add_execution_results(self, results: List[AgentExecutionResult], workflow_id: str) -> bool:
        """Add execution results to the graph"""
        try:
            # Add workflow node
            await self.add_concept(
                concept=workflow_id,
                concept_type="workflow",
                properties={
                "workflow_id": workflow_id,
                    "total_results": len(results),
                    "created_at": datetime.now()
                }
            )
            
            for result in results:
                # Add result node
                result_id = f"result_{result.agent_id}_{result.execution_id}"
                await self.add_concept(
                    concept=result_id,
                    concept_type="result",
                    properties={
                        "agent_id": result.agent_id,
                        "execution_id": result.execution_id,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "result_data": str(result.result_data)[:200] if result.result_data else None
                    }
                )

                # Add agent node (if not already present)
                agent_concept_id = f"agent_{result.agent_id}"
                await self.add_concept(
                    concept=agent_concept_id,
                    concept_type="agent",
                    properties={
                        "agent_id": result.agent_id,
                        "agent_name": getattr(result, 'agent_name', result.agent_id)
                    }
                )
            
            logger.info(f"Execution result addition complete: {len(results)}")
            return True

        except Exception as e:
            logger.error(f"Execution result addition failed: {e}")
            return False

    async def add_agent_collaboration(self, agent1: str, agent2: str, collaboration_type: str, properties: Dict[str, Any]) -> bool:
        """Add agent collaboration relationship"""
        try:
            await self.add_relation(
                subject=f"agent_{agent1}",
                predicate=collaboration_type,
                object=f"agent_{agent2}",
                properties={
                    "collaboration_type": collaboration_type,
                    "bidirectional": True,
                    **properties
                }
            )

            logger.debug(f"Agent collaboration added: {agent1} --[{collaboration_type}]--> {agent2}")
            return True

        except Exception as e:
            logger.error(f"Failed to add agent collaboration: {e}")
            return False
    
    def _get_display_label(self, node_id: str, attrs: Dict[str, Any]) -> str:
        """Generate node display label"""
        try:
            # Use special attributes first
            if "agent_id" in attrs:
                return f"🤖 {attrs['agent_id']}"
            elif "natural_language" in attrs:
                query = attrs["natural_language"]
                return f"💬 {query[:30]}..." if len(query) > 30 else f"�� {query}"
            elif "concept_name" in attrs:
                return f"💡 {attrs['concept_name']}"
            elif "workflow_id" in attrs:
                return f"⚙️ {attrs['workflow_id']}"
            elif "concept" in attrs:
                concept = attrs["concept"]
                node_type = attrs.get("type", "")
                
                if node_type == "agent":
                    return f"🤖 {concept}"
                elif node_type == "workflow":
                    return f"⚙️ {concept}"
                elif node_type == "concept":
                    return f"💡 {concept}"
                elif node_type == "entity":
                    return f"🏷️ {concept}"
                elif node_type == "result":
                    return f"📊 {concept}"
                else:
                    return str(concept)
            else:
                return str(node_id)[:20] + "..." if len(str(node_id)) > 20 else str(node_id)
                
        except Exception as e:
            logger.warning(f"Label generation failed: {e}")
            return str(node_id)[:15] + "..." if len(str(node_id)) > 15 else str(node_id)

    def _get_node_color_by_type(self, node_type: str) -> str:
        """Return color by node type"""
        color_map = {
            "agent": "#FF6B6B",      # red
            "concept": "#4ECDC4",    # teal
            "entity": "#45B7D1",     # blue
            "relation": "#96CEB4",   # green
            "workflow": "#FFEAA7",   # yellow
            "result": "#DDA0DD",     # purple
            "query": "#98D8C8",      # mint
            "domain": "#00cec9",     # dark teal
            "task": "#a29bfe",       # light purple
            "capability": "#74b9ff"  # sky blue
        }
        return color_map.get(node_type, "#CCCCCC")

    def _calculate_node_size(self, node_id: str, subgraph: nx.MultiDiGraph) -> int:
        """Calculate node size based on degree"""
        degree = subgraph.degree(node_id)
        return min(10 + degree * 3, 40)

    def _get_edge_color_by_type(self, predicate: str) -> str:
        """Return color by edge type"""
        edge_colors = {
            "contains_entity": "#FF6B6B",
            "requires_concept": "#4ECDC4",
            "includes_result": "#45B7D1",
            "produced_result": "#96CEB4",
            "failed_result": "#E17055",
            "collaborated_with": "#FFEAA7",
            "succeeded_in": "#00B894",
            "failed_in": "#E17055",
            "involves_relation": "#A29BFE"
        }
        return edge_colors.get(predicate, "#999999")
    
    def _get_relation_display_name(self, predicate: str) -> str:
        """Return relation display name"""
        display_names = {
            "contains_entity": "contains",
            "requires_concept": "requires",
            "includes_result": "includes_result",
            "produced_result": "produced",
            "failed_result": "failed",
            "collaborated_with": "collaborated",
            "succeeded_in": "succeeded",
            "failed_in": "failed",
            "involves_relation": "involves"
        }
        return display_names.get(predicate, predicate)

    def _generate_graph_stats(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """Generate graph statistics"""
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {},
            "edge_types": {},
            "avg_degree": 0,
            "max_degree": 0,
            "density": 0
        }

        # Statistics by node type
        degrees = []
        for node in nodes:
            node_type = node["type"]
            degree = node.get("degree", 0)
            degrees.append(degree)
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

        # Statistics by edge type
        for edge in edges:
            edge_type = edge.get("predicate", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

        # Degree statistics
        if degrees:
            stats["avg_degree"] = sum(degrees) / len(degrees)
            stats["max_degree"] = max(degrees)

        # Density calculation
        if len(nodes) > 1:
            max_edges = len(nodes) * (len(nodes) - 1)
            stats["density"] = len(edges) / max_edges if max_edges > 0 else 0

        return stats

    def _get_color_legend(self) -> Dict[str, Any]:
        """Return color legend"""
        return {
            "node_colors": {
                "agent": {"color": "#FF6B6B", "label": "Agent"},
                "concept": {"color": "#4ECDC4", "label": "Concept"},
                "entity": {"color": "#45B7D1", "label": "Entity"},
                "relation": {"color": "#96CEB4", "label": "Relation"},
                "workflow": {"color": "#FFEAA7", "label": "Workflow"},
                "result": {"color": "#DDA0DD", "label": "Result"},
                "query": {"color": "#98D8C8", "label": "Query"}
            },
            "edge_colors": {
                "contains_entity": {"color": "#FF6B6B", "label": "Contains"},
                "requires_concept": {"color": "#4ECDC4", "label": "Requires"},
                "includes_result": {"color": "#45B7D1", "label": "Includes Result"},
                "produced_result": {"color": "#96CEB4", "label": "Produced"},
                "collaborated_with": {"color": "#FFEAA7", "label": "Collaborated"}
            }
        }

    def _recommend_layout(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Recommend optimal layout based on graph structure"""
        node_count = len(nodes)
        edge_count = len(edges)

        # Analyze node types
        type_counts = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        has_hierarchy = any(node_type in ["workflow", "task", "domain"] for node_type in type_counts)
        
        if node_count <= 10:
            return "circular"
        elif has_hierarchy and node_count <= 30:
            return "hierarchical"
        elif edge_count / node_count > 2:
            return "force"
        else:
            return "grid"
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Return full graph statistics"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "node_counter": self.node_counter,
            "relation_counter": self.relation_counter
        }
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data"""
        return {
            "nodes": [
                {"id": node_id, **attrs} 
                for node_id, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, "key": key, **attrs}
                for source, target, key, attrs in self.graph.edges(keys=True, data=True)
            ],
            "stats": self.get_graph_stats()
        }


# Global instance
_knowledge_graph_engine = None

def get_knowledge_graph_engine() -> SimpleKnowledgeGraphEngine:
    """Return global knowledge graph engine instance"""
    global _knowledge_graph_engine
    if _knowledge_graph_engine is None:
        _knowledge_graph_engine = SimpleKnowledgeGraphEngine()
    return _knowledge_graph_engine


logger.info("🧠 Cleanly restructured knowledge graph engine loaded!")
