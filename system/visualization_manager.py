"""
🎨 Visualization Manager
Visualization Manager

Creates and manages ontology knowledge graph visualization
"""

from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

from ..engines.knowledge_graph_clean import KnowledgeGraphEngine


class VisualizationManager:
    """🎨 Visualization Manager"""
    
    def __init__(self, knowledge_graph: KnowledgeGraphEngine):
        self.knowledge_graph = knowledge_graph
    
    def get_knowledge_graph_visualization(self, max_nodes: int = 50) -> Dict[str, Any]:
        """Get knowledge graph visualization data"""
        try:
            logger.info(f"🎨 Ontology knowledge graph visualization requested - max nodes: {max_nodes}")
            
            # Return default visualization data if graph is empty
            current_nodes = self.knowledge_graph.graph.number_of_nodes()
            if current_nodes == 0:
                logger.info("📦 Empty graph detected, returning hardcoded visualization data")
                return self._create_hardcoded_visualization_data()
            
            logger.info(f"📊 Current graph node count: {current_nodes}")
            
            # Generate visualization data from knowledge graph engine
            knowledge_graph_visualization = self.knowledge_graph.generate_visualization(max_nodes=max_nodes)
            
            # Return hardcoded visualization data if graph is still empty
            if not knowledge_graph_visualization.get("nodes") and not knowledge_graph_visualization.get("edges"):
                logger.warning("⚠️ Visualization data is empty, returning hardcoded data")
                return self._create_hardcoded_visualization_data()
            
            logger.info(f"✅ Ontology graph visualization complete")
            return knowledge_graph_visualization
            
        except Exception as e:
            logger.error(f"Knowledge graph visualization generation failed: {e}")
            return self._create_fallback_visualization(str(e))
    
    def _create_hardcoded_visualization_data(self) -> Dict[str, Any]:
        """Generate hardcoded visualization data (for fallback)"""
        logger.info("🎨 Generating hardcoded visualization data")
        
        # Sample nodes
        nodes = [
            {
                "id": "workflow_main",
                "label": "🔄 Main Workflow",
                "type": "workflow",
                "size": 25,
                "color": "#fdcb6e",
                "special_type": "workflow",
                "properties": {"workflow_type": "main", "optimization_strategy": "balanced"},
                "confidence": 0.9,
                "relevance_score": 1.0,
                "attributes": {"domain": "system", "complexity": "medium"}
            },
            {
                "id": "agent_internet",
                "label": "🌐 Internet Agent",
                "type": "agent",
                "size": 20,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "internet_agent", "capabilities": ["search", "web"]},
                "confidence": 0.85,
                "relevance_score": 0.9,
                "attributes": {"domain": "information", "execution_time": 2.5}
            },
            {
                "id": "agent_calculator",
                "label": "🔢 Calculator Agent",
                "type": "agent",
                "size": 18,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "calculator_agent", "capabilities": ["math", "calculation"]},
                "confidence": 0.88,
                "relevance_score": 0.8,
                "attributes": {"domain": "calculation", "execution_time": 1.2}
            },
            {
                "id": "agent_analysis",
                "label": "📊 Analysis Agent",
                "type": "agent",
                "size": 19,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "analysis_agent", "capabilities": ["analysis", "insights"]},
                "confidence": 0.82,
                "relevance_score": 0.85,
                "attributes": {"domain": "analysis", "execution_time": 3.1}
            },
            {
                "id": "domain_information",
                "label": "📚 Information Domain",
                "type": "domain",
                "size": 16,
                "color": "#a29bfe",
                "special_type": "domain",
                "properties": {"domain_name": "information"},
                "confidence": 0.9,
                "relevance_score": 0.7,
                "attributes": {"domain_type": "knowledge"}
            },
            {
                "id": "capability_search",
                "label": "🔍 Search Capability",
                "type": "capability",
                "size": 14,
                "color": "#00cec9",
                "special_type": "capability",
                "properties": {"capability_name": "search"},
                "confidence": 0.85,
                "relevance_score": 0.75,
                "attributes": {"capability_type": "core"}
            },
            {
                "id": "task_data_collection",
                "label": "📋 Data Collection",
                "type": "task",
                "size": 17,
                "color": "#74b9ff",
                "special_type": "task",
                "properties": {"task_type": "data_collection"},
                "confidence": 0.87,
                "relevance_score": 0.8,
                "attributes": {"complexity": "medium"}
            },
            {
                "id": "result_analysis",
                "label": "📈 Analysis Result",
                "type": "result",
                "size": 15,
                "color": "#00b894",
                "special_type": "result",
                "properties": {"result_type": "analysis"},
                "confidence": 0.83,
                "relevance_score": 0.78,
                "attributes": {"quality": "high"}
            }
        ]
        
        # Sample edges
        edges = [
            {
                "id": "edge_1",
                "source": "workflow_main",
                "target": "agent_internet",
                "label": "Execute",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#5fd2c9",
                "size": 3,
                "weight": 1.0,
                "confidence": 0.9,
                "properties": {"execution_order": 1},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_2",
                "source": "workflow_main",
                "target": "agent_calculator",
                "label": "Execute",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#74b9ff",
                "size": 3,
                "weight": 0.9,
                "confidence": 0.85,
                "properties": {"execution_order": 2},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_3",
                "source": "workflow_main",
                "target": "agent_analysis",
                "label": "Execute",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#fd79a8",
                "size": 3,
                "weight": 0.85,
                "confidence": 0.87,
                "properties": {"execution_order": 3},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_4",
                "source": "agent_internet",
                "target": "domain_information",
                "label": "Specialization",
                "type": "specialization",
                "relationship_type": "specializes_in",
                "color": "#96CEB4",
                "size": 2,
                "weight": 0.8,
                "confidence": 0.9,
                "properties": {"specialization_level": "high"},
                "attributes": {"bidirectional": False, "strength": "medium"}
            },
            {
                "id": "edge_5",
                "source": "agent_internet",
                "target": "capability_search",
                "label": "Capability",
                "type": "capability",
                "relationship_type": "has_capability",
                "color": "#00cec9",
                "size": 2,
                "weight": 0.9,
                "confidence": 0.88,
                "properties": {"proficiency": "expert"},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_6",
                "source": "agent_internet",
                "target": "task_data_collection",
                "label": "Produce",
                "type": "production",
                "relationship_type": "produces",
                "color": "#e17055",
                "size": 2,
                "weight": 0.75,
                "confidence": 0.82,
                "properties": {"output_quality": "high"},
                "attributes": {"bidirectional": False, "strength": "medium"}
            },
            {
                "id": "edge_7",
                "source": "agent_analysis",
                "target": "result_analysis",
                "label": "Produce",
                "type": "production",
                "relationship_type": "produces",
                "color": "#00b894",
                "size": 2,
                "weight": 0.88,
                "confidence": 0.85,
                "properties": {"output_quality": "high"},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_8",
                "source": "agent_internet",
                "target": "agent_analysis",
                "label": "Collaborate",
                "type": "collaboration",
                "relationship_type": "collaborated_with",
                "color": "#a29bfe",
                "size": 2,
                "weight": 0.7,
                "confidence": 0.8,
                "properties": {"collaboration_type": "sequential"},
                "attributes": {"bidirectional": True, "strength": "medium"}
            }
        ]
        
        # Metadata
        metadata = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {
                "workflow": 1,
                "agent": 3,
                "domain": 1,
                "capability": 1,
                "task": 1,
                "result": 1
            },
            "edge_types": {
                "execution": 3,
                "specialization": 1,
                "capability": 1,
                "production": 2,
                "collaboration": 1
            },
            "graph_metrics": {
                "density": 0.25,
                "average_degree": 2.0,
                "clustering_coefficient": 0.15,
                "connected_components": 1
            },
            "semantic_layers": {
                "workflow_layer": ["workflow_main"],
                "agent_layer": ["agent_internet", "agent_calculator", "agent_analysis"],
                "domain_layer": ["domain_information"],
                "capability_layer": ["capability_search"],
                "task_layer": ["task_data_collection"]
            },
            "layout_suggestions": {
                "recommended": "hierarchical",
                "type_centers": {
                    "workflow": (0.5, 0.1),
                    "agent": (0.3, 0.5),
                    "domain": (0.7, 0.3),
                    "capability": (0.7, 0.7),
                    "task": (0.3, 0.8)
                },
                "force_settings": {
                    "node_repulsion": 800,
                    "link_strength": 0.6,
                    "charge_strength": -250
                }
            },
            "styling": {
                "node_colors": {
                    "workflow": "#fdcb6e",
                    "agent": "#fd79a8",
                    "domain": "#a29bfe",
                    "capability": "#00cec9",
                    "task": "#74b9ff",
                    "result": "#00b894"
                },
                "edge_colors": {
                    "execution": "#5fd2c9",
                    "specialization": "#96CEB4",
                    "capability": "#00cec9",
                    "production": "#e17055",
                    "collaboration": "#a29bfe"
                },
                "node_size_range": {"min": 10, "max": 30},
                "edge_size_range": {"min": 1, "max": 5}
            },
            "generated_at": datetime.now().isoformat(),
            "version": "2.0",
            "graph_type": "hardcoded_demo",
            "description": "Ontology system demo graph"
        }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata
        }

    def _create_fallback_visualization(self, error_message: str) -> Dict[str, Any]:
        """Generate fallback visualization"""
        return {
            "nodes": [
                {
                    "id": "error_node",
                    "label": "⚠️ Error Occurred",
                    "type": "error",
                    "size": 20,
                    "color": "#E74C3C",
                    "properties": {"error_message": error_message}
                }
            ],
            "edges": [],
            "metadata": {
                "description": f"Ontology graph generation error: {error_message}",
                "error": True,
                "generated_at": datetime.now().isoformat(),
                "graph_type": "error_fallback",
                "total_nodes": 1,
                "total_edges": 0
            }
        } 