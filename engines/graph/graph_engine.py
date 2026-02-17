"""
🧠 Graph Engine - Core CRUD Operations

Handles basic create, read, update, and delete operations on the knowledge graph.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from ...core.models import SemanticQuery, AgentExecutionResult
from ...core.interfaces import KnowledgeGraph
from ...core.llm_manager import get_ontology_llm_manager, OntologyLLMType


class GraphEngine(KnowledgeGraph):
    """🧠 Core graph engine"""

    def __init__(self, fast_mode: bool = True):
        # Knowledge graph using NetworkX
        self.graph = nx.MultiDiGraph()

        # LLM manager (lazy loading)
        self._llm_manager = None

        # Performance mode setting
        self.fast_mode = fast_mode  # When True, skip LLM calls

        # Metadata storage
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_concepts": 0,
            "total_relations": 0,
            "version": "2.0",
            "fast_mode": fast_mode
        }
        
        logger.info(f"🧠 Graph engine initialized (fast mode: {'ON' if fast_mode else 'OFF'})")

    async def add_concept(self, concept: str, concept_type: str, properties: Dict[str, Any]) -> bool:
        """Add concept - supports fast mode"""
        try:
            # 🚀 Fast mode: skip LLM calls
            if self.fast_mode:
                enhanced_properties = properties
            else:
                # Validate concept and enrich properties via LLM
                enhanced_properties = await self._enhance_concept_properties(
                    concept, concept_type, properties
                )
            
            # Set node attributes
            node_attrs = {
                "type": concept_type,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                **enhanced_properties
            }

            # Add or update node in graph
            if concept in self.graph:
                # Update existing node
                existing_attrs = self.graph.nodes[concept]
                existing_attrs.update(node_attrs)
                existing_attrs['last_updated'] = datetime.now().isoformat()
                logger.debug(f"Concept updated: {concept} ({concept_type})")
            else:
                # Add new node
                self.graph.add_node(concept, **node_attrs)
                logger.debug(f"Concept added: {concept} ({concept_type})")

            # Update metadata
            self._update_metadata()

            return True

        except Exception as e:
            logger.error(f"Failed to add concept: {concept} - {e}")
            return False
    
    async def add_relation(self, subject: str, predicate: str, object: str,
                          properties: Dict[str, Any]) -> bool:
        """Add relation - supports fast mode"""
        try:
            # 🚀 Fast mode: add missing concepts simply
            if self.fast_mode:
                # Add subject and object as basic concepts if not in graph
                if subject not in self.graph:
                    await self.add_concept(subject, "auto_created", {"created_by": "fast_mode"})

                if object not in self.graph:
                    await self.add_concept(object, "auto_created", {"created_by": "fast_mode"})

                enhanced_properties = properties
            else:
                # Infer and add missing subject/object concepts via LLM
                if subject not in self.graph:
                    await self._infer_and_add_missing_concept(subject)
                
                if object not in self.graph:
                    await self._infer_and_add_missing_concept(object)
                
                # Validate relation and enrich properties via LLM
                enhanced_properties = await self._enhance_relation_properties(
                    subject, predicate, object, properties
                )

            # Set edge attributes
            edge_attrs = {
                "predicate": predicate,
                "created_at": datetime.now().isoformat(),
                "weight": enhanced_properties.get("weight", 1.0),
                "confidence": enhanced_properties.get("confidence", 0.8),
                **enhanced_properties
            }

            # Add edge to graph
            self.graph.add_edge(subject, object, **edge_attrs)

            # Update metadata
            self._update_metadata()

            logger.debug(f"Relation added: {subject} --{predicate}--> {object}")
            return True

        except Exception as e:
            logger.error(f"Failed to add relation: {subject} -> {object} - {e}")
            return False

    async def add_relationship(self, source: str, target: str, relationship: str,
                               attributes: Dict[str, Any] = None) -> bool:
        """Add relationship (alias for add_relation)"""
        if attributes is None:
            attributes = {}
        return await self.add_relation(source, relationship, target, attributes)
    
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """LLM-based graph query"""
        try:
            # Analyze query via LLM
            query_analysis = await self.llm_manager.invoke_llm(
                OntologyLLMType.QUERY_PROCESSOR,
                f"다음 그래프 쿼리를 분석하여 실행 계획을 수립해주세요: {query}"
            )

            # Execute basic query
            results = await self._execute_basic_query(query)

            # Enrich results via LLM
            if results:
                enhanced_results = await self._enhance_query_results(query, results)
                return enhanced_results

            return results

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
    
    async def _enhance_concept_properties(self, concept: str, concept_type: str,
                                          properties: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich concept properties via LLM"""
        try:
            context = f"""
            개념: {concept}
            타입: {concept_type}
            기존 속성: {properties}

            이 개념의 속성을 보강하고 누락된 중요한 메타데이터를 추가해주세요.
            """

            enhanced_data = await self.llm_manager.invoke_llm(
                OntologyLLMType.KNOWLEDGE_REASONER,
                {"reasoning_context": context}
            )

            # Parse LLM response and merge into properties
            # By default return original properties; if parsed successfully, return enriched properties
            return {
                **properties,
                "llm_enhanced": True,
                "enhancement_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Concept property enrichment failed: {e}")
            return properties
    
    async def _enhance_relation_properties(self, subject: str, predicate: str,
                                           object: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich relation properties via LLM"""
        try:
            context = f"""
            관계: {subject} --{predicate}--> {object}
            기존 속성: {properties}

            이 관계의 신뢰도, 가중치, 유형을 분석하고 적절한 속성을 제안해주세요.
            """

            enhanced_data = await self.llm_manager.invoke_llm(
                OntologyLLMType.KNOWLEDGE_REASONER,
                {"reasoning_context": context}
            )

            # Basic enrichment
            return {
                **properties,
                "weight": properties.get("weight", self._calculate_relation_weight(subject, predicate, object)),
                "confidence": properties.get("confidence", 0.8),
                "llm_enhanced": True,
                "enhancement_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Relation property enrichment failed: {e}")
            return properties
    
    async def _infer_and_add_missing_concept(self, concept: str):
        """Infer missing concept via LLM and add it"""
        try:
            context = f"""
            개념: {concept}

            이 개념의 타입과 기본 속성을 추론해주세요.
            """

            inference_result = await self.llm_manager.invoke_llm(
                OntologyLLMType.KNOWLEDGE_REASONER,
                {"reasoning_context": context}
            )

            # Add concept using basic inference result
            await self.add_concept(concept, "inferred", {
                "auto_created": True,
                "inference_source": "llm",
                "inference_timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.warning(f"Concept inference failed: {concept} - {e}")
            # Fallback: add with default type
            await self.add_concept(concept, "unknown", {"auto_created": True})
    
    async def _execute_basic_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute basic query"""
        try:
            results = []
            query_lower = query.lower()

            # Simple keyword-based search
            if "find" in query_lower or "search" in query_lower:
                # Node search
                for node, attrs in self.graph.nodes(data=True):
                    if any(keyword in str(node).lower() or keyword in str(attrs).lower()
                           for keyword in query_lower.split()):
                        results.append({
                            "type": "node",
                            "id": node,
                            "attributes": attrs
                        })

            elif "relation" in query_lower or "edge" in query_lower:
                # Relation search
                for source, target, attrs in self.graph.edges(data=True):
                    if any(keyword in str(attrs).lower()
                           for keyword in query_lower.split()):
                        results.append({
                            "type": "edge",
                            "source": source,
                            "target": target,
                            "attributes": attrs
                        })

            elif "path" in query_lower:
                # Path search
                results.extend(await self._find_paths_in_query(query))

            else:
                # Full graph info
                results.append({
                    "type": "graph_info",
                    "nodes": self.graph.number_of_nodes(),
                    "edges": self.graph.number_of_edges(),
                    "metadata": self.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Basic query execution failed: {e}")
            return []
    
    async def _find_paths_in_query(self, query: str) -> List[Dict[str, Any]]:
        """Find paths from query"""
        try:
            results = []
            words = query.lower().split()

            if len(words) >= 3:
                try:
                    source = words[words.index("from") + 1] if "from" in words else None
                    target = words[words.index("to") + 1] if "to" in words else None

                    if source and target and source in self.graph and target in self.graph:
                        try:
                            path = nx.shortest_path(self.graph, source, target)
                            results.append({
                                "type": "path",
                                "path": path,
                                "length": len(path) - 1
                            })
                        except nx.NetworkXNoPath:
                            results.append({
                                "type": "path",
                                "message": f"No path found from {source} to {target}"
                            })
                except (ValueError, IndexError):
                    pass

            return results

        except Exception as e:
            logger.error(f"Path search failed: {e}")
            return []
    
    async def _enhance_query_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich query results via LLM"""
        try:
            if not results:
                return results

            # Summarize results and generate insights
            context = f"""
            쿼리: {query}
            결과 개수: {len(results)}
            결과 유형들: {[r.get('type', 'unknown') for r in results]}

            이 쿼리 결과에 대한 인사이트를 제공해주세요.
            """

            insights = await self.llm_manager.invoke_llm(
                OntologyLLMType.KNOWLEDGE_REASONER,
                {"reasoning_context": context}
            )

            # Add insights to original results
            enhanced_results = results.copy()
            enhanced_results.append({
                "type": "llm_insights",
                "insights": insights,
                "generated_at": datetime.now().isoformat()
            })

            return enhanced_results

        except Exception as e:
            logger.warning(f"Query result enrichment failed: {e}")
            return results
    
    def _calculate_relation_weight(self, subject: str, predicate: str, object: str) -> float:
        """Calculate relation weight"""
        try:
            # Base weight calculation logic
            base_weight = 1.0
            
            # predicate 기반 가중치 조정
            predicate_weights = {
                "is_a": 0.9,
                "part_of": 0.8,
                "related_to": 0.5,
                "depends_on": 0.7,
                "collaborates_with": 0.6
            }
            
            return predicate_weights.get(predicate.lower(), base_weight)

        except Exception as e:
            logger.warning(f"Weight calculation failed: {e}")
            return 1.0

    def _update_metadata(self):
        """Update metadata"""
        try:
            self.metadata.update({
                "total_concepts": self.graph.number_of_nodes(),
                "total_relations": self.graph.number_of_edges(),
                "last_updated": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Metadata update failed: {e}")

    @property
    def llm_manager(self):
        """LLM manager with lazy loading"""
        if self._llm_manager is None:
            self._llm_manager = get_ontology_llm_manager()
        return self._llm_manager
    
    def enable_fast_mode(self):
        """Enable fast mode"""
        self.fast_mode = True
        self.metadata["fast_mode"] = True
        logger.info("🚀 Fast mode enabled - LLM calls disabled")

    def disable_fast_mode(self):
        """Disable fast mode"""
        self.fast_mode = False
        self.metadata["fast_mode"] = False
        logger.info("🐌 Precision mode enabled - LLM calls enabled")

    def get_graph_stats(self) -> Dict[str, Any]:
        """Retrieve graph statistics"""
        try:
            # Statistics by node type
            node_types = {}
            for node, attrs in self.graph.nodes(data=True):
                node_type = attrs.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1

            # Connectivity statistics
            degrees = dict(self.graph.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            
            return {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "node_types": node_types,
                "average_degree": round(avg_degree, 2),
                "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
                "metadata": self.metadata
            }
            
        except Exception as e:
            logger.error(f"Graph statistics retrieval failed: {e}")
            return {"error": str(e)}

    def export_graph_data(self) -> Dict[str, Any]:
        """Export full graph data"""
        try:
            return {
                "nodes": [
                    {"id": node, **attrs}
                    for node, attrs in self.graph.nodes(data=True)
                ],
                "edges": [
                    {"source": source, "target": target, **attrs}
                    for source, target, attrs in self.graph.edges(data=True)
                ],
                "metadata": self.metadata,
                "stats": self.get_graph_stats()
            }
        except Exception as e:
            logger.error(f"Graph data export failed: {e}")
            return {}
    
    async def semantic_query_analysis(self, natural_query: str) -> SemanticQuery:
        """Semantic query analysis (compatibility method)"""
        try:
            # Analyze query via LLM
            analysis_result = await self.llm_manager.invoke_llm(
                OntologyLLMType.SEMANTIC_ANALYZER,
                natural_query
            )

            # Create SemanticQuery object with default values
            return SemanticQuery(
                query_text=natural_query,
                intent="information_retrieval",
                entities=[],
                concepts=[],
                relations=[],
                metadata={"llm_analysis": analysis_result}
            )

        except Exception as e:
            logger.error(f"Semantic query analysis failed: {e}")
            # Fallback: return default SemanticQuery
            return SemanticQuery(
                query_text=natural_query,
                intent="information_retrieval",
                entities=[],
                concepts=[],
                relations=[]
            )

    # 🔍 Implement missing abstract methods from the KnowledgeGraph interface
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Find related concepts"""
        try:
            if concept not in self.graph:
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
                if node in self.graph:
                    neighbors = list(self.graph.neighbors(node))
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
        """Visualize graph"""
        try:
            logger.info("🎨 Graph visualization generation started")

            if self.graph.number_of_nodes() == 0:
                logger.warning("No nodes to visualize")
                return "No data to visualize"

            # Set output path
            if output_path is None:
                from pathlib import Path
                output_path = Path("graph_visualization.html")

            # Generate simple HTML visualization
            html_content = self._generate_simple_html_visualization()

            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"✅ Graph visualization saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            return f"Visualization failed: {str(e)}"

    def _generate_simple_html_visualization(self) -> str:
        """Generate simple HTML visualization"""
        nodes = []
        edges = []

        # Generate node data
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "type": attrs.get("type", "unknown"),
                "size": 10
            })

        # Generate edge data
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "label": attrs.get("predicate", "related")
            })

        # Simple HTML template
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Graph Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .info {{ margin-bottom: 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
        .node-list {{ margin-top: 20px; }}
        .node-item {{ margin: 5px 0; padding: 5px; background: #e8f4f8; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>🧠 Graph Visualization</h1>

    <div class="info">
        <h3>📊 Graph Statistics</h3>
        <p><strong>Nodes:</strong> {len(nodes)}</p>
        <p><strong>Edges:</strong> {len(edges)}</p>
        <p><strong>Generated at:</strong> {datetime.now().isoformat()}</p>
    </div>

    <div class="node-list">
        <h3>📋 Node List</h3>
        {''.join([f'<div class="node-item"><strong>{node["id"]}</strong> ({node["type"]})</div>' for node in nodes])}
    </div>

    <div class="node-list">
        <h3>🔗 Relation List</h3>
        {''.join([f'<div class="node-item">{edge["source"]} --{edge["label"]}--> {edge["target"]}</div>' for edge in edges])}
    </div>
</body>
</html>
        """

        return html_template


logger.info("🧠 Graph engine loaded!")