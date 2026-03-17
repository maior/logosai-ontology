"""
Ontology Knowledge Graph Engine - Clean Integrated Version

Integrated knowledge graph engine leveraging the new split engines.
"""
import json
import os
import networkx as nx
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult
from ..core.interfaces import KnowledgeGraph
from ..core.llm_manager import get_ontology_llm_manager, OntologyLLMType
from .graph.graph_engine import GraphEngine
from .graph.visualization_engine import VisualizationEngine

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


class KnowledgeGraphEngine(KnowledgeGraph):
    """🧠 Ontology Knowledge Graph Engine - Clean Integrated Version"""

    def __init__(self, max_nodes: int = 1000, fast_mode: bool = True):
        self.max_nodes = max_nodes

        # Core graph engine (CRUD operations) - fast mode applied
        self.graph_engine = GraphEngine(fast_mode=fast_mode)

        # Visualization engine
        self.visualization_engine = VisualizationEngine(self.graph_engine.graph)

        # LLM manager
        self.llm_manager = get_ontology_llm_manager()

        # Metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "2.0",
            "engine_type": "integrated_clean",
            "fast_mode": fast_mode
        }
        
        logger.info(f"🧠 Integrated ontology knowledge graph engine initialized (fast mode: {'ON' if fast_mode else 'OFF'})")

    @property
    def graph(self):
        """Direct graph access for backward compatibility"""
        return self.graph_engine.graph

    # 🔗 KnowledgeGraph interface implementation
    async def add_concept(self, concept_id: str, concept_type: str, attributes: Dict[str, Any]) -> bool:
        """Add concept - delegates to GraphEngine"""
        result = await self.graph_engine.add_concept(concept_id, concept_type, attributes)
        if result:
            self._update_metadata()
            # Notify visualization engine of graph update
            self.visualization_engine.graph = self.graph_engine.graph
        return result

    async def add_relationship(self, source: str, target: str, relationship: str,
                               attributes: Dict[str, Any] = None) -> bool:
        """Add relationship - delegates to GraphEngine"""
        if attributes is None:
            attributes = {}

        result = await self.graph_engine.add_relationship(source, target, relationship, attributes)
        if result:
            self._update_metadata()
            # Notify visualization engine of graph update
            self.visualization_engine.graph = self.graph_engine.graph
        return result

    async def add_relation(self, subject: str, predicate: str, object: str,
                           properties: Dict[str, Any] = None) -> bool:
        """Add relation (alias for add_relationship)"""
        if properties is None:
            properties = {}
        return await self.add_relationship(subject, object, predicate, properties)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Retrieve graph statistics - delegates to GraphEngine with metadata added"""
        stats = self.graph_engine.get_graph_stats()
        stats["metadata"].update(self.metadata)
        return stats

    def generate_visualization(self, max_nodes: int = 100) -> Dict[str, Any]:
        """Generate visualization data - delegates to VisualizationEngine"""
        try:
            logger.info(f"🎨 Visualization data generation started - max nodes: {max_nodes}")

            # Safe way to call async method synchronously
            import asyncio

            try:
                # Check if there is a running event loop
                loop = asyncio.get_running_loop()
                # If loop is already running, use synchronous alternative
                logger.info("Existing event loop detected, using synchronous visualization generation")
                visualization_data = self._generate_visualization_sync(max_nodes)
            except RuntimeError:
                # If no running loop, create a new one
                logger.info("Creating new event loop for visualization generation")
                visualization_data = asyncio.run(
                    self.visualization_engine.generate_visualization(max_nodes)
                )

            logger.info("✅ Visualization data generation complete")
            return visualization_data

        except Exception as e:
            logger.error(f"Visualization data generation failed: {e}")
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "error": str(e),
                    "generated_at": datetime.now().isoformat(),
                    "version": "2.0"
                }
            }
    
    # 🔍 Additional convenience methods
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Graph query - delegates to GraphEngine"""
        return await self.graph_engine.query_graph(query)

    async def semantic_query_analysis(self, natural_query: str) -> SemanticQuery:
        """Semantic query analysis - delegates to GraphEngine"""
        return await self.graph_engine.semantic_query_analysis(natural_query)

    async def add_semantic_query_concepts(self, semantic_query: SemanticQuery) -> bool:
        """Add concepts from semantic query"""
        try:
            success_count = 0

            # Add the query itself as a concept
            query_result = await self.add_concept(
                f"query_{semantic_query.query_id}",
                "query",
                {
                    "natural_language": semantic_query.natural_language,
                    "intent": semantic_query.intent,
                    "query_type": semantic_query.query_type.value if hasattr(semantic_query.query_type, 'value') else str(semantic_query.query_type),
                    "complexity_score": semantic_query.complexity_score,
                    "created_at": semantic_query.created_at.isoformat() if hasattr(semantic_query.created_at, 'isoformat') else str(semantic_query.created_at),
                    "metadata": semantic_query.metadata
                }
            )
            if query_result:
                success_count += 1

            # Add entities as concepts
            for entity in semantic_query.entities:
                entity_result = await self.add_concept(
                    entity,
                    "entity",
                    {
                        "source_query": semantic_query.query_id,
                        "extracted_from": semantic_query.natural_language
                    }
                )
                if entity_result:
                    success_count += 1
                    # Add query-entity relationship
                    await self.add_relationship(
                        f"query_{semantic_query.query_id}",
                        entity,
                        "contains_entity"
                    )

            # Add concepts
            for concept in semantic_query.concepts:
                concept_result = await self.add_concept(
                    concept,
                    "concept",
                    {
                        "source_query": semantic_query.query_id,
                        "extracted_from": semantic_query.natural_language
                    }
                )
                if concept_result:
                    success_count += 1
                    # Add query-concept relationship
                    await self.add_relationship(
                        f"query_{semantic_query.query_id}",
                        concept,
                        "involves_concept"
                    )

            # Process relations
            for relation in semantic_query.relations:
                # Add relation as a concept
                relation_result = await self.add_concept(
                    relation,
                    "relation",
                    {
                        "source_query": semantic_query.query_id,
                        "extracted_from": semantic_query.natural_language
                    }
                )
                if relation_result:
                    success_count += 1
                    # Add query-relation relationship
                    await self.add_relationship(
                        f"query_{semantic_query.query_id}",
                        relation,
                        "uses_relation"
                    )

            logger.info(f"Semantic query concept addition complete: {success_count} succeeded")
            return success_count > 0

        except Exception as e:
            logger.error(f"Semantic query concept addition failed: {e}")
            return False
    
    async def add_execution_results(self, results: List[AgentExecutionResult], workflow_id: str) -> bool:
        """Add execution results to the graph"""
        try:
            success_count = 0

            # Add workflow concept (if not already present)
            workflow_result = await self.add_concept(
                workflow_id,
                "workflow",
                {
                    "execution_time": datetime.now().isoformat(),
                    "total_results": len(results)
                }
            )
            if workflow_result:
                success_count += 1

            # Add each execution result as a concept
            for i, result in enumerate(results):
                result_id = f"{workflow_id}_result_{i}"
                
                result_success = await self.add_concept(
                    result_id,
                    "execution_result",
                    {
                        "agent_type": result.agent_type.value if hasattr(result.agent_type, 'value') else str(result.agent_type),
                        "agent_id": result.agent_id,
                        "execution_time": result.execution_time,
                        "success": result.success,
                        "confidence": result.confidence,
                        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                        "error_message": result.error_message,
                        "created_at": result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at),
                        "metadata": result.metadata
                    }
                )
                
                if result_success:
                    success_count += 1

                    # Add workflow-result relationship
                    await self.add_relationship(
                        workflow_id,
                        result_id,
                        "produces",
                        {
                            "execution_order": i,
                            "agent_type": result.agent_type.value if hasattr(result.agent_type, 'value') else str(result.agent_type)
                        }
                    )

                    # Add agent concept (if not already present)
                    agent_success = await self.add_concept(
                        result.agent_id,
                        "agent",
                        {
                            "agent_type": result.agent_type.value if hasattr(result.agent_type, 'value') else str(result.agent_type),
                            "last_execution": result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at)
                        }
                    )

                    # Add agent-result relationship
                    if agent_success:
                        await self.add_relationship(
                            result.agent_id,
                            result_id,
                            "executes",
                            {
                                "execution_time": result.execution_time,
                                "success": result.success
                            }
                        )

            logger.info(f"Execution result addition complete: {success_count} succeeded")
            return success_count > 0

        except Exception as e:
            logger.error(f"Execution result addition failed: {e}")
            return False
    
    async def enhance_with_llm_insights(self, query: str) -> Dict[str, Any]:
        """Generate graph insights using LLM"""
        try:
            # Analyze current graph state
            stats = self.get_graph_stats()

            # Generate insights via LLM
            context = f"""
            현재 온톨로지 그래프 상태:
            - 총 노드 수: {stats.get('total_nodes', 0)}
            - 총 엣지 수: {stats.get('total_edges', 0)}
            - 노드 타입 분포: {stats.get('node_types', {})}
            - 평균 연결도: {stats.get('average_degree', 0)}

            사용자 쿼리: {query}

            이 그래프의 현재 상태와 패턴을 분석하고, 사용자 쿼리와 관련된 인사이트를 제공해주세요.
            """

            insights = await self.llm_manager.invoke_llm(
                OntologyLLMType.KNOWLEDGE_REASONER,
                {"reasoning_context": context}
            )

            return {
                "insights": insights,
                "graph_stats": stats,
                "generated_at": datetime.now().isoformat(),
                "query": query
            }

        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return {"error": str(e)}

    def export_graph_data(self) -> Dict[str, Any]:
        """Export full graph data"""
        return self.graph_engine.export_graph_data()

    # 🔍 Implement missing abstract methods from the KnowledgeGraph interface
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Find related concepts - delegates to GraphEngine"""
        try:
            # Use GraphEngine's graph to find related concepts
            if not hasattr(self.graph_engine, 'graph') or concept not in self.graph_engine.graph:
                logger.warning(f"Concept '{concept}' not found in graph")
                return []

            related_concepts = []
            visited = set()

            def _find_neighbors(node: str, current_depth: int):
                """Recursively find neighboring nodes"""
                if current_depth >= max_depth or node in visited:
                    return

                visited.add(node)

                # Find directly connected nodes
                if node in self.graph_engine.graph:
                    neighbors = list(self.graph_engine.graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in related_concepts and neighbor != concept:
                            related_concepts.append(neighbor)

                        # Recursive call for next depth
                        if current_depth + 1 < max_depth:
                            _find_neighbors(neighbor, current_depth + 1)

            # Start finding related concepts
            _find_neighbors(concept, 0)

            logger.info(f"Found {len(related_concepts)} related concepts for '{concept}' (depth: {max_depth})")
            return related_concepts[:50]  # Limit to 50

        except Exception as e:
            logger.error(f"Finding related concepts failed: {e}")
            return []
    
    def visualize_graph(self, output_path: str = None) -> str:
        """Visualize graph - delegates to VisualizationEngine"""
        try:
            logger.info("🎨 Graph visualization generation started")

            # Generate visualization data
            visualization_data = self.generate_visualization()

            if not visualization_data or not visualization_data.get('nodes'):
                logger.warning("No data to visualize")
                return "No data to visualize"

            # Set output path
            if output_path is None:
                from pathlib import Path
                output_path = Path("graph_visualization.html")

            # Generate HTML visualization
            html_content = self._generate_html_visualization(visualization_data)

            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"✅ Graph visualization saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            return f"Visualization failed: {str(e)}"

    def _generate_html_visualization(self, visualization_data: Dict[str, Any]) -> str:
        """Generate HTML visualization"""
        nodes = visualization_data.get('nodes', [])
        edges = visualization_data.get('edges', [])
        metadata = visualization_data.get('metadata', {})

        # Simple HTML + D3.js visualization
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ontology Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .node {{ fill: #69b3a2; stroke: #fff; stroke-width: 2px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .node-label {{ font-size: 12px; text-anchor: middle; }}
        .info {{ margin-bottom: 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🧠 Ontology Knowledge Graph</h1>

    <div class="info">
        <h3>📊 Graph Statistics</h3>
        <p><strong>Nodes:</strong> {len(nodes)}</p>
        <p><strong>Edges:</strong> {len(edges)}</p>
        <p><strong>Generated at:</strong> {metadata.get('generated_at', 'Unknown')}</p>
        <p><strong>Version:</strong> {metadata.get('version', 'Unknown')}</p>
    </div>
    
    <svg width="800" height="600"></svg>
    
    <script>
        const nodes = {nodes};
        const links = {edges};
        
        const svg = d3.select("svg");
        const width = +svg.attr("width");
        const height = +svg.attr("height");
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link");
        
        const node = svg.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.max(5, Math.min(20, (d.size || 10))))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        const label = svg.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.label || d.id);
        
        node.append("title")
            .text(d => `${{d.label || d.id}}\\nType: ${{d.type || 'Unknown'}}`);
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y + 4);
        }});
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _generate_visualization_sync(self, max_nodes: int = 100) -> Dict[str, Any]:
        """Synchronous visualization data generation (to prevent event loop conflicts)"""
        try:
            logger.info(f"🎨 Synchronous visualization generation started - max nodes: {max_nodes}")

            # Create subgraph with node count limit
            if self.graph_engine.graph.number_of_nodes() <= max_nodes:
                subgraph = self.graph_engine.graph
            else:
                # Select by importance
                node_degrees = dict(self.graph_engine.graph.degree())
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                subgraph = self.graph_engine.graph.subgraph([node for node, _ in top_nodes])

            # Generate node data
            nodes = []
            for node_id, attrs in subgraph.nodes(data=True):
                node_type = attrs.get('type', 'unknown')
                
                node_data = {
                    "id": node_id,
                    "label": self._get_sync_display_label(node_id, attrs),
                    "type": node_type,
                    "size": self._get_sync_node_size(node_id, subgraph),
                    "color": self.visualization_engine.node_colors.get(node_type, "#b2bec3"),
                    "properties": attrs
                }
                nodes.append(node_data)
            
            # Generate edge data
            edges = []
            for i, (source, target, attrs) in enumerate(subgraph.edges(data=True)):
                relationship_type = attrs.get('relationship_type', attrs.get('predicate', 'related_to'))
                
                edge_data = {
                    "id": f"edge_{i}",
                    "source": source,
                    "target": target,
                    "label": relationship_type,
                    "type": relationship_type,
                    "color": self.visualization_engine.edge_colors.get(relationship_type, "#b2bec3"),
                    "weight": attrs.get('weight', 1.0)
                }
                edges.append(edge_data)
            
            # Generate metadata
            metadata = {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": self._get_sync_type_distribution(nodes, "type"),
                "edge_types": self._get_sync_type_distribution(edges, "type"),
                "generated_at": datetime.now().isoformat(),
                "version": "2.0",
                "generation_method": "synchronous"
            }
            
            logger.info(f"✅ Synchronous visualization generation complete: {len(nodes)} nodes, {len(edges)} edges")
            
            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Synchronous visualization generation failed: {e}")
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "error": str(e),
                    "generated_at": datetime.now().isoformat(),
                    "version": "2.0",
                    "generation_method": "synchronous_fallback"
                }
            }

    def _get_sync_display_label(self, node_id: str, attrs: Dict[str, Any]) -> str:
        """Generate synchronous display label"""
        if "agent_id" in attrs:
            return f"🤖 {attrs['agent_id']}"
        return str(node_id)[:20]

    def _get_sync_node_size(self, node_id: str, subgraph: nx.MultiDiGraph) -> int:
        """Calculate synchronous node size"""
        degree = subgraph.degree(node_id)
        return min(10 + degree * 3, 40)

    def _get_sync_type_distribution(self, items: List[Dict], type_key: str) -> Dict[str, int]:
        """Calculate synchronous type distribution"""
        distribution = {}
        for item in items:
            item_type = item.get(type_key, "unknown")
            distribution[item_type] = distribution.get(item_type, 0) + 1
        return distribution

    def _update_metadata(self):
        """Update metadata"""
        self.metadata["last_updated"] = datetime.now().isoformat()

    # ─── Persistence ────────────────────────────────────────────────

    def save_to_disk(self, path: Optional[str] = None) -> bool:
        """Save the knowledge graph to disk as JSON.

        Uses nx.node_link_data() for serialization and atomic write
        (.tmp → rename) for crash safety.
        """
        try:
            save_path = Path(path) if path else _DEFAULT_DATA_DIR / "kg_checkpoint.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            graph_data = nx.node_link_data(self.graph_engine.graph)

            checkpoint = {
                "graph": graph_data,
                "metadata": self.metadata,
                "saved_at": datetime.now().isoformat(),
                "version": "2.0",
            }

            tmp_path = save_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, default=str)
            os.replace(str(tmp_path), str(save_path))

            node_count = self.graph_engine.graph.number_of_nodes()
            edge_count = self.graph_engine.graph.number_of_edges()
            logger.info(f"💾 KG checkpoint saved: {node_count} nodes, {edge_count} edges → {save_path}")
            return True
        except Exception as e:
            logger.error(f"KG checkpoint save failed: {e}")
            return False

    def load_from_disk(self, path: Optional[str] = None) -> bool:
        """Load the knowledge graph from a JSON checkpoint.

        Uses nx.node_link_graph() and re-syncs the visualization engine.
        """
        try:
            load_path = Path(path) if path else _DEFAULT_DATA_DIR / "kg_checkpoint.json"
            if not load_path.exists():
                logger.info(f"No KG checkpoint found at {load_path} — starting fresh")
                return False

            with open(load_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            graph_data = checkpoint.get("graph")
            if not graph_data:
                logger.warning("KG checkpoint has no graph data")
                return False

            restored_graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
            self.graph_engine.graph = restored_graph
            self.visualization_engine.graph = restored_graph

            saved_metadata = checkpoint.get("metadata", {})
            if saved_metadata:
                self.metadata.update(saved_metadata)
            self.metadata["last_loaded"] = datetime.now().isoformat()

            node_count = restored_graph.number_of_nodes()
            edge_count = restored_graph.number_of_edges()
            logger.info(f"📂 KG checkpoint loaded: {node_count} nodes, {edge_count} edges ← {load_path}")
            return True
        except Exception as e:
            logger.error(f"KG checkpoint load failed: {e}")
            return False


# ─── Module-level singleton ─────────────────────────────────────────

_knowledge_graph_engine_instance: Optional[KnowledgeGraphEngine] = None


def get_knowledge_graph_engine() -> KnowledgeGraphEngine:
    """Return the shared KnowledgeGraphEngine singleton.

    On first call, creates the instance and loads the latest checkpoint
    from disk (if any). All components should use this instead of
    creating their own KnowledgeGraphEngine.
    """
    global _knowledge_graph_engine_instance
    if _knowledge_graph_engine_instance is None:
        _knowledge_graph_engine_instance = KnowledgeGraphEngine(fast_mode=True)
        _knowledge_graph_engine_instance.load_from_disk()
        logger.info("🧠 KG singleton initialized (with checkpoint load attempt)")
    return _knowledge_graph_engine_instance


logger.info("🧠 Integrated ontology knowledge graph engine loaded!")