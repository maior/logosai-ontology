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
from ..core.semantic_index import SemanticIndex, DEFAULT_NODE_TYPES
from ..core.vector_backend import VectorBackend, select_backend
from .graph.graph_engine import GraphEngine
from .graph.visualization_engine import VisualizationEngine

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


class _LazyLLMType:
    """OntologyLLMType 지연 프록시 — graph_engine._LazyLLMType 과 같은 이유.

    KG 엔진은 문서 검색·데이터셋 추출 소비자의 주 진입점이다. 여기서
    llm_manager 를 모듈 레벨로 import 하면 그 소비자 전원이 logosai(에이전트
    프레임워크)를 설치해야 한다. LLM 은 analyze_graph_patterns 같은 일부
    경로에서만 쓰이므로 그때 로드한다.
    """

    def __getattr__(self, name):
        from ..core.llm_manager import OntologyLLMType as _Real
        return getattr(_Real, name)


OntologyLLMType = _LazyLLMType()


class KnowledgeGraphEngine(KnowledgeGraph):
    """🧠 Ontology Knowledge Graph Engine - Clean Integrated Version"""

    def __init__(self, max_nodes: int = 1000, fast_mode: bool = True,
                 namespace: str = "default"):
        self.max_nodes = max_nodes
        # Namespace separates independent ontologies (agent routing vs
        # domain knowledge built by ontology.builder) — each gets its own
        # graph instance and checkpoint file.
        self.namespace = namespace

        # Core graph engine (CRUD operations) - fast mode applied
        self.graph_engine = GraphEngine(fast_mode=fast_mode)

        # Visualization engine
        self.visualization_engine = VisualizationEngine(self.graph_engine.graph)

        # LLM manager — 지연 생성 (llm_manager 프로퍼티 참고). 그래프를 읽기만
        # 하는 소비자(문서 검색·데이터셋 추출)가 대다수인데, 생성자에서
        # 만들면 그 전원이 LLM 설정 로딩 비용을 낸다.
        self._llm_manager = None

        # Metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "2.0",
            "engine_type": "integrated_clean",
            "fast_mode": fast_mode
        }

        # Semantic (embedding) index — created lazily on first search,
        # or explicitly via init_semantic_index() (tests inject embed_fn)
        self._semantic_index: Optional[SemanticIndex] = None
        
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

    # ─── Ontology inference (data-perspective queries) ──────────────
    # Pure graph traversal — synchronous, deterministic, no LLM.

    def get_ancestors(self, node_id: str, predicate: str = "is_a") -> List[str]:
        """Transitive closure upward: all nodes reachable via `predicate`
        out-edges (nearest first, BFS order)."""
        graph = self.graph
        if node_id not in graph:
            return []
        ancestors: List[str] = []
        visited = {node_id}
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            for _, target, attrs in graph.out_edges(current, data=True):
                if attrs.get("predicate") == predicate and target not in visited:
                    visited.add(target)
                    ancestors.append(target)
                    queue.append(target)
        return ancestors

    def get_descendants(self, node_id: str, predicate: str = "is_a") -> List[str]:
        """Transitive closure downward: all nodes that reach `node_id`
        via `predicate` edges."""
        graph = self.graph
        if node_id not in graph:
            return []
        descendants: List[str] = []
        visited = {node_id}
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            for source, _, attrs in graph.in_edges(current, data=True):
                if attrs.get("predicate") == predicate and source not in visited:
                    visited.add(source)
                    descendants.append(source)
                    queue.append(source)
        return descendants

    def _resolve_node_id(self, name: str, prefix: str) -> Optional[str]:
        """Accept both bare names ('web_search') and prefixed node ids
        ('capability_web_search')."""
        if name in self.graph:
            return name
        prefixed = f"{prefix}{name}"
        return prefixed if prefixed in self.graph else None

    def find_agents_by_capability(self, capability: str,
                                  include_inherited: bool = True) -> List[str]:
        """Agents holding a capability. With include_inherited=True, agents
        holding a *sub-capability* (is_a descendant) also match — e.g. a
        'realtime_search' holder matches a 'web_search' query."""
        cap_id = self._resolve_node_id(capability, "capability_")
        if cap_id is None:
            return []
        capability_ids = [cap_id]
        if include_inherited:
            capability_ids += self.get_descendants(cap_id)

        graph = self.graph
        agents: List[str] = []
        for cid in capability_ids:
            for source, _, attrs in graph.in_edges(cid, data=True):
                if (attrs.get("predicate") == "has_capability"
                        and graph.nodes[source].get("type") == "agent"
                        and source not in agents):
                    agents.append(source)
        return sorted(agents)

    def find_agents_by_tag(self, tag: str) -> List[str]:
        """Agents annotated with a tag."""
        tag_id = self._resolve_node_id(tag, "tag_")
        if tag_id is None:
            return []
        graph = self.graph
        agents = [
            source
            for source, _, attrs in graph.in_edges(tag_id, data=True)
            if attrs.get("predicate") == "has_tag"
            and graph.nodes[source].get("type") == "agent"
        ]
        return sorted(set(agents))

    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """Unified data view of one agent: properties + capabilities + tags
        + learned success patterns, assembled from the graph."""
        graph = self.graph
        if agent_id not in graph or graph.nodes[agent_id].get("type") != "agent":
            return {}
        props = dict(graph.nodes[agent_id])

        capabilities: List[str] = []
        tags: List[str] = []
        patterns: List[Dict[str, Any]] = []
        for _, target, attrs in graph.out_edges(agent_id, data=True):
            predicate = attrs.get("predicate")
            node = graph.nodes.get(target, {})
            if predicate == "has_capability":
                capabilities.append(node.get("name") or target.replace("capability_", "", 1))
            elif predicate == "has_tag":
                tags.append(node.get("name") or target.replace("tag_", "", 1))
            elif predicate == "has_mapping":
                patterns.append({
                    "pattern": node.get("generalization_pattern"),
                    "category": node.get("category"),
                    "success_rate": node.get("success_rate"),
                    "usage_count": node.get("usage_count"),
                })
        patterns.sort(key=lambda p: p.get("success_rate") or 0, reverse=True)

        return {
            "agent_id": agent_id,
            "name": props.get("name", agent_id),
            "description": props.get("description", ""),
            "capabilities": sorted(set(capabilities)),
            "tags": sorted(set(tags)),
            "success_patterns": patterns[:20],
            "is_available": props.get("is_available", True),
        }

    # ─── Semantic search (embedding entry + graph expansion) ────────

    def _semantic_node_types(self):
        """The default namespace indexes only the agent-semantic surface
        (skipping hundreds of query_agent_mapping records); builder
        namespaces hold arbitrary document schemas, so index every type."""
        return DEFAULT_NODE_TYPES if self.namespace == "default" else None

    def init_semantic_index(self, embed_fn=None, node_types=...) -> VectorBackend:
        """Create (or replace) the semantic index and build it from the
        current graph. The backend tier is auto-selected by node count
        (see vector_backend.select_backend) — callers configure nothing.
        Tests inject a deterministic embed_fn; production omits it to use
        the real sentence-transformers model."""
        if node_types is ...:
            node_types = self._semantic_node_types()
        # namespace 를 넘겨야 tier 1(npy) 캐시가 네임스페이스별로 갈린다 —
        # 안 넘기면 heritage_kr 벡터를 heritage_us 가 물려받는다.
        self._semantic_index = select_backend(
            self.graph.number_of_nodes(), embed_fn=embed_fn,
            namespace=self.namespace)
        self._semantic_index.build_from_graph(self.graph, node_types=node_types)
        return self._semantic_index

    def refresh_semantic_index(self) -> int:
        """Re-index nodes whose text changed and index new nodes.
        Call after sync/feedback batches. Returns nodes (re-)embedded."""
        if self._semantic_index is None:
            return 0
        return self._semantic_index.build_from_graph(
            self.graph, node_types=self._semantic_node_types())

    def rebuild_semantic_index(self, embed_fn=None, node_types=...) -> dict:
        """색인을 그래프와 강제로 맞춘다 — 관리자의 명시적 재색인용.

        refresh_semantic_index 와 다른 점: 색인이 아직 없으면 **만든다**.
        refresh 가 None 일 때 0 을 돌려주는 것은 의도된 게으름이지만(아무도
        검색하지 않았으면 임베딩 비용을 내지 않는다), /reindex 는 사용자가
        "지금 맞춰라"라고 말한 것이므로 그 게으름이 곧 버그가 된다.

        노드 편집·삭제·병합 뒤 이걸 부르지 않으면 semantic 채널이 그래프와
        어긋난 채 남는다(실측: 병합 후 hit@1 0.3125→0.25 로 나빠졌고, 서버를
        재시작해야 회복됐다 — 하마터면 병합의 회귀로 오진할 상황이었다).

        {embedded, pruned, total} 을 돌려준다 — 조용히 성공하면 다음에 또 같은
        오진을 한다.
        """
        if node_types is ...:
            node_types = self._semantic_node_types()
        if self._semantic_index is None:
            self.init_semantic_index(embed_fn=embed_fn, node_types=node_types)
            return {"embedded": len(self._semantic_index), "pruned": 0,
                    "total": len(self._semantic_index)}
        pruned = self._semantic_index.prune_to_graph(
            self.graph, node_types=node_types)
        embedded = self._semantic_index.build_from_graph(
            self.graph, node_types=node_types)
        return {"embedded": embedded, "pruned": pruned,
                "total": len(self._semantic_index)}

    def semantic_search(self, query: str, top_k: int = 5,
                        node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Embedding-similarity search over graph nodes.
        Finds nodes even when the query shares no literal tokens with them.
        Returns [] (never raises) when no embedder is available."""
        if self._semantic_index is None:
            self.init_semantic_index()
        return self._semantic_index.search(query, top_k=top_k, node_types=node_types)

    def find_agents_semantic(self, query: str, top_k: int = 5,
                             min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Semantic entry + graph expansion:

        1. Similarity search finds entry nodes (agent / capability / tag).
        2. Capability and tag hits expand to holding agents via in-edges;
           capability hits also include holders of is_a sub-capabilities.
        3. Each agent keeps its best score. Returns
           [{agent_id, score, matched_via}] sorted by score.
        """
        # Wider entry net than top_k: several entries can map to one agent
        entries = self.semantic_search(query, top_k=max(top_k * 3, 10),
                                       node_types=["agent", "capability", "tag"])
        graph = self.graph
        best: Dict[str, Dict[str, Any]] = {}

        def consider(agent_id: str, score: float, via: str):
            if score < min_score:
                return
            if agent_id not in best or score > best[agent_id]["score"]:
                best[agent_id] = {"agent_id": agent_id,
                                  "score": round(score, 4),
                                  "matched_via": via}

        for entry in entries:
            node_id, node_type, score = entry["node_id"], entry["node_type"], entry["score"]

            if node_type == "agent":
                consider(node_id, score, node_id)
                continue

            predicate = "has_capability" if node_type == "capability" else "has_tag"
            expanded = [node_id]
            if node_type == "capability":
                # holders of a more specific capability also qualify
                expanded += self.get_descendants(node_id, predicate="is_a")

            for target in expanded:
                if target not in graph:
                    continue
                for source, _, attrs in graph.in_edges(target, data=True):
                    if (attrs.get("predicate") == predicate
                            and graph.nodes[source].get("type") == "agent"):
                        consider(source, score, node_id)

        return sorted(best.values(), key=lambda a: a["score"], reverse=True)[:top_k]

    def clear(self) -> None:
        """Reset this namespace's graph to empty (rebuild option).
        The semantic index is dropped too — it would point at dead nodes."""
        import networkx as nx
        self.graph_engine.graph = nx.MultiDiGraph()
        self.visualization_engine.graph = self.graph_engine.graph
        self._semantic_index = None
        self._update_metadata()
        logger.info(f"🧹 KG cleared (namespace={self.namespace})")

    def deduplicate_edges(self) -> int:
        """Remove parallel duplicate edges: keep one edge per
        (source, target, predicate) triple. Returns the number removed.

        One-time cleanup for graphs written before write-time dedup existed.
        """
        graph = self.graph
        to_remove: List[Tuple[str, str, Any]] = []
        for source, target in set(graph.edges()):
            edge_data = graph.get_edge_data(source, target) or {}
            seen_predicates: Set[str] = set()
            for edge_key in sorted(edge_data.keys(), key=str):
                predicate = edge_data[edge_key].get("predicate")
                if predicate in seen_predicates:
                    to_remove.append((source, target, edge_key))
                else:
                    seen_predicates.add(predicate)
        for source, target, edge_key in to_remove:
            graph.remove_edge(source, target, key=edge_key)
        if to_remove:
            self._update_metadata()
            logger.info(f"🧹 Removed {len(to_remove)} duplicate edges")
        return len(to_remove)

    # ─── Persistence ────────────────────────────────────────────────

    @property
    def llm_manager(self):
        """LLM manager — 첫 접근 시 생성 (import 도 이때 일어난다)."""
        if self._llm_manager is None:
            from ..core.llm_manager import get_ontology_llm_manager
            self._llm_manager = get_ontology_llm_manager()
        return self._llm_manager

    @property
    def checkpoint_path(self) -> Path:
        """Namespace-specific checkpoint file. The default namespace keeps
        the historical filename for backward compatibility."""
        if self.namespace == "default":
            return _DEFAULT_DATA_DIR / "kg_checkpoint.json"
        return _DEFAULT_DATA_DIR / f"kg_{self.namespace}.json"

    def save_to_disk(self, path: Optional[str] = None) -> bool:
        """Save the knowledge graph to disk as JSON.

        Uses nx.node_link_data() for serialization and atomic write
        (.tmp → rename) for crash safety.
        """
        try:
            save_path = Path(path) if path else self.checkpoint_path
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
            # 쓰기의 PG 반영은 여기서 하지 않는다 — 수동 변경은 service._pg_apply
            # (surgical upsert/delete), 인제스트/빌드는 _mirror_to_pg(증분)가 이미
            # 담당한다. 여기서 sync_from_graph 를 부르면 full-graph diff 라 중복이고,
            # in-memory 그래프가 부분일 때 PG 를 잘못 삭제할 위험이 있다(축 5).
            return True
        except Exception as e:
            logger.error(f"KG checkpoint save failed: {e}")
            return False

    def load_from_disk(self, path: Optional[str] = None) -> bool:
        """Load the knowledge graph from a JSON checkpoint.

        Uses nx.node_link_graph() and re-syncs the visualization engine.
        """
        try:
            load_path = Path(path) if path else self.checkpoint_path
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

_kg_instances: Dict[str, KnowledgeGraphEngine] = {}


def get_knowledge_graph_engine(namespace: str = "default") -> KnowledgeGraphEngine:
    """Return the shared KnowledgeGraphEngine for a namespace.

    Each namespace is an independent ontology with its own graph and
    checkpoint file (default → kg_checkpoint.json, others → kg_{ns}.json).
    On first call per namespace, creates the instance and loads its
    checkpoint from disk (if any).
    """
    if namespace not in _kg_instances:
        engine = KnowledgeGraphEngine(fast_mode=True, namespace=namespace)
        # PG 가 진실 — pg_backed 네임스페이스는 JSON 대신 PG 에서 하이드레이트한다
        # (축 5). PG 미가용/실패 시 JSON 체크포인트로 degrade(조용한 손실 방지).
        loaded = False
        try:
            from ..core import graph_store, pg
            aschema = graph_store.aicoach_source(namespace)
            if aschema and pg.available():
                # aicoach 라이브 스토어 직접 소비 — 복사본 아님(축 5 위, 서비스 연동)
                counts = graph_store.hydrate_graph_aicoach(namespace, engine.graph, aschema)
                engine.visualization_engine.graph = engine.graph
                logger.info(f"🗄️ KG hydrated from aicoach live ({aschema}): "
                            f"{counts['nodes']} nodes, {counts['edges']} edges "
                            f"(namespace={namespace})")
                loaded = True
            elif graph_store.pg_backed(namespace) and pg.available():
                counts = graph_store.hydrate_graph(namespace, engine.graph)
                engine.visualization_engine.graph = engine.graph
                logger.info(f"🗄️ KG hydrated from PG: {counts['nodes']} nodes, "
                            f"{counts['edges']} edges (namespace={namespace})")
                loaded = True
        except Exception as ex:
            logger.warning(f"⚠️ PG hydrate 실패 → JSON fallback "
                           f"(namespace={namespace}): {ex}")
        if not loaded:
            engine.load_from_disk()
        _kg_instances[namespace] = engine
        logger.info(f"🧠 KG singleton initialized (namespace={namespace})")
    return _kg_instances[namespace]


logger.info("🧠 Integrated ontology knowledge graph engine loaded!")