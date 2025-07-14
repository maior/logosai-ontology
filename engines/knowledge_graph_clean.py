"""
온톨로지 지식 그래프 엔진 - 깔끔한 통합 버전
Ontology Knowledge Graph Engine - Clean Integrated Version

새로운 분할된 엔진들을 활용한 통합 지식 그래프 엔진
"""
import networkx as nx
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult
from ..core.interfaces import KnowledgeGraph
from ..core.llm_manager import get_ontology_llm_manager, OntologyLLMType
from .graph.graph_engine import GraphEngine
from .graph.visualization_engine import VisualizationEngine


class KnowledgeGraphEngine(KnowledgeGraph):
    """🧠 온톨로지 지식 그래프 엔진 - 깔끔한 통합 버전"""
    
    def __init__(self, max_nodes: int = 1000, fast_mode: bool = True):
        self.max_nodes = max_nodes
        
        # 핵심 그래프 엔진 (CRUD 작업) - 고속 모드 적용
        self.graph_engine = GraphEngine(fast_mode=fast_mode)
        
        # 시각화 엔진
        self.visualization_engine = VisualizationEngine(self.graph_engine.graph)
        
        # LLM 관리자
        self.llm_manager = get_ontology_llm_manager()
        
        # 메타데이터
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "2.0",
            "engine_type": "integrated_clean",
            "fast_mode": fast_mode
        }
        
        logger.info(f"🧠 통합 온톨로지 지식 그래프 엔진 초기화 완료 (고속 모드: {'ON' if fast_mode else 'OFF'})")
    
    @property
    def graph(self):
        """하위 호환성을 위한 그래프 직접 접근"""
        return self.graph_engine.graph
    
    # 🔗 KnowledgeGraph 인터페이스 구현
    async def add_concept(self, concept_id: str, concept_type: str, attributes: Dict[str, Any]) -> bool:
        """개념 추가 - GraphEngine에 위임"""
        result = await self.graph_engine.add_concept(concept_id, concept_type, attributes)
        if result:
            self._update_metadata()
            # 시각화 엔진에 그래프 업데이트 알림
            self.visualization_engine.graph = self.graph_engine.graph
        return result
    
    async def add_relationship(self, source: str, target: str, relationship: str, 
                             attributes: Dict[str, Any] = None) -> bool:
        """관계 추가 - GraphEngine에 위임"""
        if attributes is None:
            attributes = {}
        
        result = await self.graph_engine.add_relationship(source, target, relationship, attributes)
        if result:
            self._update_metadata()
            # 시각화 엔진에 그래프 업데이트 알림
            self.visualization_engine.graph = self.graph_engine.graph
        return result
    
    async def add_relation(self, subject: str, predicate: str, object: str, 
                          properties: Dict[str, Any] = None) -> bool:
        """관계 추가 (add_relationship의 별칭)"""
        if properties is None:
            properties = {}
        return await self.add_relationship(subject, object, predicate, properties)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """그래프 통계 조회 - GraphEngine에 위임하되 메타데이터 추가"""
        stats = self.graph_engine.get_graph_stats()
        stats["metadata"].update(self.metadata)
        return stats
    
    def generate_visualization(self, max_nodes: int = 100) -> Dict[str, Any]:
        """시각화 데이터 생성 - VisualizationEngine에 위임"""
        try:
            logger.info(f"🎨 시각화 데이터 생성 시작 - 최대 노드: {max_nodes}")
            
            # 비동기 메서드를 동기적으로 호출하는 안전한 방법
            import asyncio
            
            try:
                # 현재 실행 중인 이벤트 루프가 있는지 확인
                loop = asyncio.get_running_loop()
                # 이미 루프가 실행 중이면 동기적 대안 사용
                logger.info("기존 이벤트 루프 감지, 동기적 시각화 생성 사용")
                visualization_data = self._generate_visualization_sync(max_nodes)
            except RuntimeError:
                # 실행 중인 루프가 없으면 새로운 루프 생성
                logger.info("새로운 이벤트 루프 생성하여 시각화 생성")
                visualization_data = asyncio.run(
                    self.visualization_engine.generate_visualization(max_nodes)
                )
            
            logger.info(f"✅ 시각화 데이터 생성 완료")
            return visualization_data
            
        except Exception as e:
            logger.error(f"시각화 데이터 생성 실패: {e}")
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "error": str(e),
                    "generated_at": datetime.now().isoformat(),
                    "version": "2.0"
                }
            }
    
    # 🔍 추가 편의 메서드들
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """그래프 쿼리 - GraphEngine에 위임"""
        return await self.graph_engine.query_graph(query)
    
    async def semantic_query_analysis(self, natural_query: str) -> SemanticQuery:
        """의미론적 쿼리 분석 - GraphEngine에 위임"""
        return await self.graph_engine.semantic_query_analysis(natural_query)
    
    async def add_semantic_query_concepts(self, semantic_query: SemanticQuery) -> bool:
        """의미론적 쿼리로부터 개념 추가"""
        try:
            success_count = 0
            
            # 쿼리 자체를 개념으로 추가
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
            
            # 엔티티들을 개념으로 추가
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
                    # 쿼리와 엔티티 관계 추가
                    await self.add_relationship(
                        f"query_{semantic_query.query_id}",
                        entity,
                        "contains_entity"
                    )
            
            # 개념들을 추가
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
                    # 쿼리와 개념 관계 추가
                    await self.add_relationship(
                        f"query_{semantic_query.query_id}",
                        concept,
                        "involves_concept"
                    )
            
            # 관계들을 처리
            for relation in semantic_query.relations:
                # 관계는 개념으로 추가
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
                    # 쿼리와 관계 관계 추가
                    await self.add_relationship(
                        f"query_{semantic_query.query_id}",
                        relation,
                        "uses_relation"
                    )
            
            logger.info(f"의미론적 쿼리 개념 추가 완료: {success_count}개 성공")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"의미론적 쿼리 개념 추가 실패: {e}")
            return False
    
    async def add_execution_results(self, results: List[AgentExecutionResult], workflow_id: str) -> bool:
        """실행 결과들을 그래프에 추가"""
        try:
            success_count = 0
            
            # 워크플로우 개념 추가 (아직 없다면)
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
            
            # 각 실행 결과를 개념으로 추가
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
                    
                    # 워크플로우와 결과 관계 추가
                    await self.add_relationship(
                        workflow_id,
                        result_id,
                        "produces",
                        {
                            "execution_order": i,
                            "agent_type": result.agent_type.value if hasattr(result.agent_type, 'value') else str(result.agent_type)
                        }
                    )
                    
                    # 에이전트 개념 추가 (아직 없다면)
                    agent_success = await self.add_concept(
                        result.agent_id,
                        "agent",
                        {
                            "agent_type": result.agent_type.value if hasattr(result.agent_type, 'value') else str(result.agent_type),
                            "last_execution": result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at)
                        }
                    )
                    
                    # 에이전트와 결과 관계 추가
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
            
            logger.info(f"실행 결과 추가 완료: {success_count}개 성공")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"실행 결과 추가 실패: {e}")
            return False
    
    async def enhance_with_llm_insights(self, query: str) -> Dict[str, Any]:
        """LLM을 활용한 그래프 인사이트 생성"""
        try:
            # 현재 그래프 상태 분석
            stats = self.get_graph_stats()
            
            # LLM을 통한 인사이트 생성
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
            logger.error(f"LLM 인사이트 생성 실패: {e}")
            return {"error": str(e)}
    
    def export_graph_data(self) -> Dict[str, Any]:
        """그래프 데이터 전체 내보내기"""
        return self.graph_engine.export_graph_data()
    
    # 🔍 KnowledgeGraph 인터페이스의 누락된 추상 메서드들 구현
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """관련 개념 찾기 - GraphEngine에 위임"""
        try:
            # GraphEngine의 그래프를 사용하여 관련 개념 찾기
            if not hasattr(self.graph_engine, 'graph') or concept not in self.graph_engine.graph:
                logger.warning(f"개념 '{concept}'을 그래프에서 찾을 수 없습니다")
                return []
            
            related_concepts = []
            visited = set()
            
            def _find_neighbors(node: str, current_depth: int):
                """재귀적으로 이웃 노드 찾기"""
                if current_depth >= max_depth or node in visited:
                    return
                
                visited.add(node)
                
                # 직접 연결된 노드들 찾기
                if node in self.graph_engine.graph:
                    neighbors = list(self.graph_engine.graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in related_concepts and neighbor != concept:
                            related_concepts.append(neighbor)
                        
                        # 다음 깊이로 재귀 호출
                        if current_depth + 1 < max_depth:
                            _find_neighbors(neighbor, current_depth + 1)
            
            # 관련 개념 찾기 시작
            _find_neighbors(concept, 0)
            
            logger.info(f"개념 '{concept}'의 관련 개념 {len(related_concepts)}개 발견 (깊이: {max_depth})")
            return related_concepts[:50]  # 최대 50개로 제한
            
        except Exception as e:
            logger.error(f"관련 개념 찾기 실패: {e}")
            return []
    
    def visualize_graph(self, output_path: str = None) -> str:
        """그래프 시각화 - VisualizationEngine에 위임"""
        try:
            logger.info("🎨 그래프 시각화 생성 시작")
            
            # 시각화 데이터 생성
            visualization_data = self.generate_visualization()
            
            if not visualization_data or not visualization_data.get('nodes'):
                logger.warning("시각화할 데이터가 없습니다")
                return "시각화할 데이터가 없습니다"
            
            # 출력 경로 설정
            if output_path is None:
                from pathlib import Path
                output_path = Path("graph_visualization.html")
            
            # HTML 시각화 생성
            html_content = self._generate_html_visualization(visualization_data)
            
            # 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ 그래프 시각화 저장 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"그래프 시각화 실패: {e}")
            return f"시각화 실패: {str(e)}"
    
    def _generate_html_visualization(self, visualization_data: Dict[str, Any]) -> str:
        """HTML 시각화 생성"""
        nodes = visualization_data.get('nodes', [])
        edges = visualization_data.get('edges', [])
        metadata = visualization_data.get('metadata', {})
        
        # 간단한 HTML + D3.js 시각화
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>온톨로지 지식 그래프 시각화</title>
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
    <h1>🧠 온톨로지 지식 그래프</h1>
    
    <div class="info">
        <h3>📊 그래프 통계</h3>
        <p><strong>노드 수:</strong> {len(nodes)}</p>
        <p><strong>엣지 수:</strong> {len(edges)}</p>
        <p><strong>생성 시간:</strong> {metadata.get('generated_at', 'Unknown')}</p>
        <p><strong>버전:</strong> {metadata.get('version', 'Unknown')}</p>
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
        """동기적 시각화 데이터 생성 (이벤트 루프 충돌 방지용)"""
        try:
            logger.info(f"🎨 동기적 시각화 생성 시작 - 최대 노드: {max_nodes}")
            
            # 노드 수 제한 서브그래프 생성
            if self.graph_engine.graph.number_of_nodes() <= max_nodes:
                subgraph = self.graph_engine.graph
            else:
                # 중요도 기반 선택
                node_degrees = dict(self.graph_engine.graph.degree())
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                subgraph = self.graph_engine.graph.subgraph([node for node, _ in top_nodes])
            
            # 노드 데이터 생성
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
            
            # 엣지 데이터 생성
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
            
            # 메타데이터 생성
            metadata = {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": self._get_sync_type_distribution(nodes, "type"),
                "edge_types": self._get_sync_type_distribution(edges, "type"),
                "generated_at": datetime.now().isoformat(),
                "version": "2.0",
                "generation_method": "synchronous"
            }
            
            logger.info(f"✅ 동기적 시각화 생성 완료: {len(nodes)}개 노드, {len(edges)}개 엣지")
            
            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"동기적 시각화 생성 실패: {e}")
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
        """동기적 표시 레이블 생성"""
        if "agent_id" in attrs:
            return f"🤖 {attrs['agent_id']}"
        return str(node_id)[:20]
    
    def _get_sync_node_size(self, node_id: str, subgraph: nx.MultiDiGraph) -> int:
        """동기적 노드 크기 계산"""
        degree = subgraph.degree(node_id)
        return min(10 + degree * 3, 40)
    
    def _get_sync_type_distribution(self, items: List[Dict], type_key: str) -> Dict[str, int]:
        """동기적 타입별 분포 계산"""
        distribution = {}
        for item in items:
            item_type = item.get(type_key, "unknown")
            distribution[item_type] = distribution.get(item_type, 0) + 1
        return distribution
    
    def _update_metadata(self):
        """메타데이터 업데이트"""
        self.metadata["last_updated"] = datetime.now().isoformat()


logger.info("🧠 통합 온톨로지 지식 그래프 엔진 로드 완료!") 