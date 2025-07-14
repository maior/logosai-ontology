"""
🧠 Knowledge Graph Engine
지식 그래프 엔진

온톨로지 기반 지식 그래프 관리 및 시각화
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
import time

from ..core.models import SemanticQuery, AgentExecutionResult
from ..core.interfaces import KnowledgeGraph


class SimpleKnowledgeGraphEngine(KnowledgeGraph):
    """🧠 간단한 지식 그래프 엔진 - 깔끔하게 재구성됨"""
    
    def __init__(self):
        # NetworkX를 사용한 지식 그래프
        self.graph = nx.MultiDiGraph()
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # 노드 및 관계 카운터
        self.node_counter = 0
        self.relation_counter = 0
        
        logger.info("🧠 간단한 지식 그래프 엔진 초기화 완료")
    
    async def add_concept(self, concept: str, concept_type: str, properties: Dict[str, Any]) -> bool:
        """개념 추가"""
        try:
            # 노드 ID 생성
            node_id = f"{concept_type}_{concept}_{self.node_counter}"
            self.node_counter += 1
            
            # 노드 속성 구성
            node_attrs = {
                "concept": concept,
                "type": concept_type,
                "created_at": datetime.now(),
                **properties
            }
            
            # 그래프에 노드 추가
            self.graph.add_node(node_id, **node_attrs)
            self.last_updated = datetime.now()
            
            logger.debug(f"개념 추가: {concept} ({concept_type})")
            return True
            
        except Exception as e:
            logger.error(f"개념 추가 실패: {e}")
            return False
    
    async def add_relation(self, subject: str, predicate: str, object: str, properties: Dict[str, Any]) -> bool:
        """관계 추가"""
        try:
            # 관계 ID 생성
            relation_id = f"rel_{self.relation_counter}"
            self.relation_counter += 1
            
            # 관계 속성 구성
            edge_attrs = {
                "predicate": predicate,
                "relation_id": relation_id,
                "created_at": datetime.now(),
                "weight": properties.get("weight", 1.0),
                "confidence": properties.get("confidence", 0.8),
                **properties
            }
            
            # 그래프에 엣지 추가
            self.graph.add_edge(subject, object, key=relation_id, **edge_attrs)
            self.last_updated = datetime.now()
            
            logger.debug(f"관계 추가: {subject} --[{predicate}]--> {object}")
            return True
            
        except Exception as e:
            logger.error(f"관계 추가 실패: {e}")
            return False
    
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """그래프 쿼리"""
        try:
            results = []
            query_lower = query.lower()
            
                # 노드 검색
            for node_id, attrs in self.graph.nodes(data=True):
                if query_lower in str(attrs.get("concept", "")).lower():
                        results.append({
                            "type": "node",
                        "id": node_id,
                        "concept": attrs.get("concept"),
                        "node_type": attrs.get("type"),
                            "attributes": attrs
                        })
            
                # 관계 검색
            for source, target, key, attrs in self.graph.edges(keys=True, data=True):
                if query_lower in str(attrs.get("predicate", "")).lower():
                        results.append({
                            "type": "edge",
                            "source": source,
                            "target": target,
                        "predicate": attrs.get("predicate"),
                            "attributes": attrs
                        })
            
            logger.info(f"쿼리 '{query}' 결과: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"그래프 쿼리 실패: {e}")
            return []
    
    def generate_visualization(self, max_nodes: int = 100) -> Dict[str, Any]:
        """시각화 데이터 생성 - advanced_multi_agent_manager와 호환"""
        try:
            # 노드가 너무 많으면 제한
            all_nodes = list(self.graph.nodes(data=True))
            if len(all_nodes) > max_nodes:
                # 연결도가 높은 노드 우선 선택
                node_degrees = [(node, self.graph.degree(node)) for node, _ in all_nodes]
                node_degrees.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = [node for node, _ in node_degrees[:max_nodes]]
                subgraph = self.graph.subgraph(selected_nodes)
            else:
                subgraph = self.graph
            
            # 노드 데이터 구성
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
            
            # 엣지 데이터 구성
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
            
            # 통계 정보
            stats = self._generate_graph_stats(nodes, edges)
            
            # 색상 범례
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
            logger.error(f"시각화 생성 실패: {e}")
            return {
                "nodes": [],
                "edges": [],
                "stats": {},
                "legend": {},
                "error": str(e)
            }
    
    async def add_semantic_query_concepts(self, semantic_query: SemanticQuery) -> bool:
        """의미론적 쿼리의 개념들을 그래프에 추가"""
        try:
            query_id = semantic_query.query_id
            
            # 쿼리 노드 추가
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
            
            logger.info(f"의미론적 쿼리 개념 추가 완료: {query_id}")
            return True
            
        except Exception as e:
            logger.error(f"의미론적 쿼리 개념 추가 실패: {e}")
            return False
    
    async def add_execution_results(self, results: List[AgentExecutionResult], workflow_id: str) -> bool:
        """실행 결과를 그래프에 추가"""
        try:
            # 워크플로우 노드 추가
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
                # 결과 노드 추가
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
                
                # 에이전트 노드 추가 (존재하지 않는 경우)
                agent_concept_id = f"agent_{result.agent_id}"
                await self.add_concept(
                    concept=agent_concept_id,
                    concept_type="agent",
                    properties={
                        "agent_id": result.agent_id,
                        "agent_name": getattr(result, 'agent_name', result.agent_id)
                    }
                )
            
            logger.info(f"실행 결과 추가 완료: {len(results)}개")
            return True
            
        except Exception as e:
            logger.error(f"실행 결과 추가 실패: {e}")
            return False
    
    async def add_agent_collaboration(self, agent1: str, agent2: str, collaboration_type: str, properties: Dict[str, Any]) -> bool:
        """에이전트 협력 관계 추가"""
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
            
            logger.debug(f"에이전트 협력 관계 추가: {agent1} --[{collaboration_type}]--> {agent2}")
            return True
            
        except Exception as e:
            logger.error(f"에이전트 협력 관계 추가 실패: {e}")
            return False
    
    def _get_display_label(self, node_id: str, attrs: Dict[str, Any]) -> str:
        """노드 표시 레이블 생성"""
        try:
            # 특별한 속성 우선 사용
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
            logger.warning(f"레이블 생성 실패: {e}")
            return str(node_id)[:15] + "..." if len(str(node_id)) > 15 else str(node_id)
    
    def _get_node_color_by_type(self, node_type: str) -> str:
        """노드 타입별 색상 반환"""
        color_map = {
            "agent": "#FF6B6B",      # 빨간색
            "concept": "#4ECDC4",    # 청록색
            "entity": "#45B7D1",     # 파란색
            "relation": "#96CEB4",   # 초록색
            "workflow": "#FFEAA7",   # 노란색
            "result": "#DDA0DD",     # 보라색
            "query": "#98D8C8",      # 민트색
            "domain": "#00cec9",     # 짙은 청록색
            "task": "#a29bfe",       # 연보라색
            "capability": "#74b9ff"  # 하늘색
        }
        return color_map.get(node_type, "#CCCCCC")
    
    def _calculate_node_size(self, node_id: str, subgraph: nx.MultiDiGraph) -> int:
        """연결 수에 따른 노드 크기 계산"""
        degree = subgraph.degree(node_id)
        return min(10 + degree * 3, 40)
    
    def _get_edge_color_by_type(self, predicate: str) -> str:
        """엣지 타입별 색상 반환"""
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
        """관계 표시명 반환"""
        display_names = {
            "contains_entity": "포함",
            "requires_concept": "필요",
            "includes_result": "결과포함",
            "produced_result": "생성",
            "failed_result": "실패",
            "collaborated_with": "협력",
            "succeeded_in": "성공",
            "failed_in": "실패",
            "involves_relation": "관련"
        }
        return display_names.get(predicate, predicate)
    
    def _generate_graph_stats(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """그래프 통계 생성"""
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {},
            "edge_types": {},
            "avg_degree": 0,
            "max_degree": 0,
            "density": 0
        }
        
        # 노드 타입별 통계
        degrees = []
        for node in nodes:
            node_type = node["type"]
            degree = node.get("degree", 0)
            degrees.append(degree)
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        # 엣지 타입별 통계
        for edge in edges:
            edge_type = edge.get("predicate", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1
        
        # 연결도 통계
        if degrees:
            stats["avg_degree"] = sum(degrees) / len(degrees)
            stats["max_degree"] = max(degrees)
        
        # 밀도 계산
        if len(nodes) > 1:
            max_edges = len(nodes) * (len(nodes) - 1)
            stats["density"] = len(edges) / max_edges if max_edges > 0 else 0
        
        return stats
    
    def _get_color_legend(self) -> Dict[str, Any]:
        """색상 범례 반환"""
        return {
            "node_colors": {
                "agent": {"color": "#FF6B6B", "label": "에이전트"},
                "concept": {"color": "#4ECDC4", "label": "개념"},
                "entity": {"color": "#45B7D1", "label": "엔티티"},
                "relation": {"color": "#96CEB4", "label": "관계"},
                "workflow": {"color": "#FFEAA7", "label": "워크플로우"},
                "result": {"color": "#DDA0DD", "label": "결과"},
                "query": {"color": "#98D8C8", "label": "쿼리"}
            },
            "edge_colors": {
                "contains_entity": {"color": "#FF6B6B", "label": "포함"},
                "requires_concept": {"color": "#4ECDC4", "label": "필요"},
                "includes_result": {"color": "#45B7D1", "label": "결과포함"},
                "produced_result": {"color": "#96CEB4", "label": "생성"},
                "collaborated_with": {"color": "#FFEAA7", "label": "협력"}
            }
        }
    
    def _recommend_layout(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """그래프 구조에 따른 최적 레이아웃 추천"""
        node_count = len(nodes)
        edge_count = len(edges)
        
        # 노드 타입 분석
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
        """전체 그래프 통계 반환"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "node_counter": self.node_counter,
            "relation_counter": self.relation_counter
        }
    
    def export_graph_data(self) -> Dict[str, Any]:
        """그래프 데이터 내보내기"""
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


# 전역 인스턴스
_knowledge_graph_engine = None

def get_knowledge_graph_engine() -> SimpleKnowledgeGraphEngine:
    """전역 지식 그래프 엔진 인스턴스 반환"""
    global _knowledge_graph_engine
    if _knowledge_graph_engine is None:
        _knowledge_graph_engine = SimpleKnowledgeGraphEngine()
    return _knowledge_graph_engine


logger.info("🧠 깔끔하게 재구성된 지식 그래프 엔진 로드 완료!")
