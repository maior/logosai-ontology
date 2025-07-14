"""
🎨 시각화 엔진 - 그래프 시각화 전문
Visualization Engine - Graph Visualization Specialist

지식 그래프의 시각화 데이터 생성 및 풍부한 메타데이터 제공
"""

import networkx as nx
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from loguru import logger

from ...core.llm_manager import get_ontology_llm_manager, OntologyLLMType


class VisualizationEngine:
    """🎨 시각화 엔진"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.llm_manager = get_ontology_llm_manager()
        
        # 색상 팔레트
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
        
        logger.info("🎨 시각화 엔진 초기화 완료")
    
    async def generate_visualization(self, max_nodes: int = 100) -> Dict[str, Any]:
        """풍부한 시각화 데이터 생성"""
        try:
            logger.info(f"🎨 시각화 생성 시작 - 최대 노드: {max_nodes}")
            
            # 노드 수 제한
            subgraph = self._create_limited_subgraph(max_nodes)
            
            # 노드/엣지 데이터 생성
            nodes = await self._generate_nodes(subgraph)
            edges = await self._generate_edges(subgraph)
            
            # 메타데이터 생성
            metadata = await self._generate_metadata(nodes, edges, subgraph)
            
            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
            return {"nodes": [], "edges": [], "metadata": {"error": str(e)}}
    
    def _create_limited_subgraph(self, max_nodes: int) -> nx.MultiDiGraph:
        """노드 수 제한 서브그래프"""
        if self.graph.number_of_nodes() <= max_nodes:
            return self.graph
        
        # 중요도 기반 선택
        node_degrees = dict(self.graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        return self.graph.subgraph([node for node, _ in top_nodes])
    
    async def _generate_nodes(self, subgraph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """노드 데이터 생성"""
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
        """엣지 데이터 생성"""
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
        """메타데이터 생성"""
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": self._get_type_distribution(nodes, "type"),
            "edge_types": self._get_type_distribution(edges, "type"),
            "generated_at": datetime.now().isoformat(),
            "version": "2.0"
        }
    
    def _get_display_label(self, node_id: str, attrs: Dict[str, Any]) -> str:
        """표시 레이블 생성"""
        if "agent_id" in attrs:
            return f"🤖 {attrs['agent_id']}"
        return str(node_id)[:20]
    
    def _calculate_node_size(self, node_id: str, subgraph: nx.MultiDiGraph) -> int:
        """노드 크기 계산"""
        degree = subgraph.degree(node_id)
        return min(10 + degree * 3, 40)
    
    def _get_type_distribution(self, items: List[Dict], type_key: str) -> Dict[str, int]:
        """타입별 분포 계산"""
        distribution = {}
        for item in items:
            item_type = item.get(type_key, "unknown")
            distribution[item_type] = distribution.get(item_type, 0) + 1
        return distribution


logger.info("🎨 시각화 엔진 로드 완료!") 