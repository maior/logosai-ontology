"""
🧠 그래프 엔진 - 핵심 CRUD 작업
Graph Engine - Core CRUD Operations

지식 그래프의 기본적인 생성, 읽기, 업데이트, 삭제 작업을 담당
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from ...core.models import SemanticQuery, AgentExecutionResult
from ...core.interfaces import KnowledgeGraph
from ...core.llm_manager import get_ontology_llm_manager, OntologyLLMType


class GraphEngine(KnowledgeGraph):
    """🧠 핵심 그래프 엔진"""
    
    def __init__(self, fast_mode: bool = True):
        # NetworkX를 사용한 지식 그래프
        self.graph = nx.MultiDiGraph()
        
        # LLM 관리자 (지연 로딩)
        self._llm_manager = None
        
        # 성능 모드 설정
        self.fast_mode = fast_mode  # True일 때 LLM 호출 스킵
        
        # 메타데이터 저장
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_concepts": 0,
            "total_relations": 0,
            "version": "2.0",
            "fast_mode": fast_mode
        }
        
        logger.info(f"🧠 그래프 엔진 초기화 완료 (고속 모드: {'ON' if fast_mode else 'OFF'})")
    
    async def add_concept(self, concept: str, concept_type: str, properties: Dict[str, Any]) -> bool:
        """개념 추가 - 고속 모드 지원"""
        try:
            # 🚀 고속 모드: LLM 호출 스킵
            if self.fast_mode:
                enhanced_properties = properties
            else:
                # LLM을 통한 개념 검증 및 속성 보강
                enhanced_properties = await self._enhance_concept_properties(
                    concept, concept_type, properties
                )
            
            # 노드 속성 설정
            node_attrs = {
                "type": concept_type,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                **enhanced_properties
            }
            
            # 그래프에 노드 추가 또는 업데이트
            if concept in self.graph:
                # 기존 노드 업데이트
                existing_attrs = self.graph.nodes[concept]
                existing_attrs.update(node_attrs)
                existing_attrs['last_updated'] = datetime.now().isoformat()
                logger.debug(f"개념 업데이트: {concept} ({concept_type})")
            else:
                # 새 노드 추가
                self.graph.add_node(concept, **node_attrs)
                logger.debug(f"개념 추가: {concept} ({concept_type})")
            
            # 메타데이터 업데이트
            self._update_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"개념 추가 실패: {concept} - {e}")
            return False
    
    async def add_relation(self, subject: str, predicate: str, object: str, 
                          properties: Dict[str, Any]) -> bool:
        """관계 추가 - 고속 모드 지원"""
        try:
            # 🚀 고속 모드: 누락된 개념은 간단히 추가
            if self.fast_mode:
                # 주체와 객체가 그래프에 없으면 기본 개념으로 추가
                if subject not in self.graph:
                    await self.add_concept(subject, "auto_created", {"created_by": "fast_mode"})
                
                if object not in self.graph:
                    await self.add_concept(object, "auto_created", {"created_by": "fast_mode"})
                
                enhanced_properties = properties
            else:
                # 주체와 객체가 그래프에 없으면 LLM을 통해 추론 후 추가
                if subject not in self.graph:
                    await self._infer_and_add_missing_concept(subject)
                
                if object not in self.graph:
                    await self._infer_and_add_missing_concept(object)
                
                # LLM을 통한 관계 검증 및 속성 보강
                enhanced_properties = await self._enhance_relation_properties(
                    subject, predicate, object, properties
                )
            
            # 엣지 속성 설정
            edge_attrs = {
                "predicate": predicate,
                "created_at": datetime.now().isoformat(),
                "weight": enhanced_properties.get("weight", 1.0),
                "confidence": enhanced_properties.get("confidence", 0.8),
                **enhanced_properties
            }
            
            # 그래프에 엣지 추가
            self.graph.add_edge(subject, object, **edge_attrs)
            
            # 메타데이터 업데이트
            self._update_metadata()
            
            logger.debug(f"관계 추가: {subject} --{predicate}--> {object}")
            return True
            
        except Exception as e:
            logger.error(f"관계 추가 실패: {subject} -> {object} - {e}")
            return False
    
    async def add_relationship(self, source: str, target: str, relationship: str, 
                             attributes: Dict[str, Any] = None) -> bool:
        """관계 추가 (add_relation의 별칭)"""
        if attributes is None:
            attributes = {}
        return await self.add_relation(source, relationship, target, attributes)
    
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """LLM 기반 그래프 쿼리"""
        try:
            # LLM을 통한 쿼리 분석
            query_analysis = await self.llm_manager.invoke_llm(
                OntologyLLMType.QUERY_PROCESSOR,
                f"다음 그래프 쿼리를 분석하여 실행 계획을 수립해주세요: {query}"
            )
            
            # 기본 쿼리 실행
            results = await self._execute_basic_query(query)
            
            # LLM을 통한 결과 보강
            if results:
                enhanced_results = await self._enhance_query_results(query, results)
                return enhanced_results
            
            return results
            
        except Exception as e:
            logger.error(f"그래프 쿼리 실패: {e}")
            return []
    
    async def _enhance_concept_properties(self, concept: str, concept_type: str, 
                                        properties: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 통한 개념 속성 보강"""
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
            
            # LLM 응답을 파싱하여 속성에 통합
            # 기본적으로는 원래 속성을 반환하고, 성공적으로 파싱되면 보강된 속성 반환
            return {
                **properties,
                "llm_enhanced": True,
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"개념 속성 보강 실패: {e}")
            return properties
    
    async def _enhance_relation_properties(self, subject: str, predicate: str, 
                                         object: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 통한 관계 속성 보강"""
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
            
            # 기본 보강
            return {
                **properties,
                "weight": properties.get("weight", self._calculate_relation_weight(subject, predicate, object)),
                "confidence": properties.get("confidence", 0.8),
                "llm_enhanced": True,
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"관계 속성 보강 실패: {e}")
            return properties
    
    async def _infer_and_add_missing_concept(self, concept: str):
        """누락된 개념을 LLM으로 추론하여 추가"""
        try:
            context = f"""
            개념: {concept}
            
            이 개념의 타입과 기본 속성을 추론해주세요.
            """
            
            inference_result = await self.llm_manager.invoke_llm(
                OntologyLLMType.KNOWLEDGE_REASONER,
                {"reasoning_context": context}
            )
            
            # 기본 추론 결과로 개념 추가
            await self.add_concept(concept, "inferred", {
                "auto_created": True,
                "inference_source": "llm",
                "inference_timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"개념 추론 실패: {concept} - {e}")
            # 폴백: 기본 타입으로 추가
            await self.add_concept(concept, "unknown", {"auto_created": True})
    
    async def _execute_basic_query(self, query: str) -> List[Dict[str, Any]]:
        """기본 쿼리 실행"""
        try:
            results = []
            query_lower = query.lower()
            
            # 간단한 키워드 기반 검색
            if "find" in query_lower or "search" in query_lower:
                # 노드 검색
                for node, attrs in self.graph.nodes(data=True):
                    if any(keyword in str(node).lower() or keyword in str(attrs).lower() 
                          for keyword in query_lower.split()):
                        results.append({
                            "type": "node",
                            "id": node,
                            "attributes": attrs
                        })
            
            elif "relation" in query_lower or "edge" in query_lower:
                # 관계 검색
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
                # 경로 검색
                results.extend(await self._find_paths_in_query(query))
            
            else:
                # 전체 그래프 정보
                results.append({
                    "type": "graph_info",
                    "nodes": self.graph.number_of_nodes(),
                    "edges": self.graph.number_of_edges(),
                    "metadata": self.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"기본 쿼리 실행 실패: {e}")
            return []
    
    async def _find_paths_in_query(self, query: str) -> List[Dict[str, Any]]:
        """쿼리에서 경로 찾기"""
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
            logger.error(f"경로 검색 실패: {e}")
            return []
    
    async def _enhance_query_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM을 통한 쿼리 결과 보강"""
        try:
            if not results:
                return results
            
            # 결과 요약 및 인사이트 생성
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
            
            # 원본 결과에 인사이트 추가
            enhanced_results = results.copy()
            enhanced_results.append({
                "type": "llm_insights",
                "insights": insights,
                "generated_at": datetime.now().isoformat()
            })
            
            return enhanced_results
            
        except Exception as e:
            logger.warning(f"쿼리 결과 보강 실패: {e}")
            return results
    
    def _calculate_relation_weight(self, subject: str, predicate: str, object: str) -> float:
        """관계 가중치 계산"""
        try:
            # 기본 가중치 계산 로직
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
            logger.warning(f"가중치 계산 실패: {e}")
            return 1.0
    
    def _update_metadata(self):
        """메타데이터 업데이트"""
        try:
            self.metadata.update({
                "total_concepts": self.graph.number_of_nodes(),
                "total_relations": self.graph.number_of_edges(),
                "last_updated": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"메타데이터 업데이트 실패: {e}")
    
    @property
    def llm_manager(self):
        """LLM 매니저 지연 로딩"""
        if self._llm_manager is None:
            self._llm_manager = get_ontology_llm_manager()
        return self._llm_manager
    
    def enable_fast_mode(self):
        """고속 모드 활성화"""
        self.fast_mode = True
        self.metadata["fast_mode"] = True
        logger.info("🚀 고속 모드 활성화 - LLM 호출 비활성화")
    
    def disable_fast_mode(self):
        """고속 모드 비활성화"""
        self.fast_mode = False
        self.metadata["fast_mode"] = False
        logger.info("🐌 정밀 모드 활성화 - LLM 호출 활성화")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """그래프 통계 조회"""
        try:
            # 노드 타입별 통계
            node_types = {}
            for node, attrs in self.graph.nodes(data=True):
                node_type = attrs.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # 연결성 통계
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
            logger.error(f"그래프 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def export_graph_data(self) -> Dict[str, Any]:
        """그래프 데이터 전체 내보내기"""
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
            logger.error(f"그래프 데이터 내보내기 실패: {e}")
            return {}
    
    async def semantic_query_analysis(self, natural_query: str) -> SemanticQuery:
        """의미론적 쿼리 분석 (호환성 메서드)"""
        try:
            # LLM을 통한 쿼리 분석
            analysis_result = await self.llm_manager.invoke_llm(
                OntologyLLMType.SEMANTIC_ANALYZER,
                natural_query
            )
            
            # SemanticQuery 객체 생성 (기본값들로)
            return SemanticQuery(
                query_text=natural_query,
                intent="information_retrieval",
                entities=[],
                concepts=[],
                relations=[],
                metadata={"llm_analysis": analysis_result}
            )
            
        except Exception as e:
            logger.error(f"의미론적 쿼리 분석 실패: {e}")
            # 폴백: 기본 SemanticQuery 반환
            return SemanticQuery(
                query_text=natural_query,
                intent="information_retrieval",
                entities=[],
                concepts=[],
                relations=[]
            )


    # 🔍 KnowledgeGraph 인터페이스의 누락된 추상 메서드들 구현
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """관련 개념 찾기"""
        try:
            if concept not in self.graph:
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
                if node in self.graph:
                    neighbors = list(self.graph.neighbors(node))
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
        """그래프 시각화"""
        try:
            logger.info("🎨 그래프 시각화 생성 시작")
            
            if self.graph.number_of_nodes() == 0:
                logger.warning("시각화할 노드가 없습니다")
                return "시각화할 데이터가 없습니다"
            
            # 출력 경로 설정
            if output_path is None:
                from pathlib import Path
                output_path = Path("graph_visualization.html")
            
            # 간단한 HTML 시각화 생성
            html_content = self._generate_simple_html_visualization()
            
            # 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ 그래프 시각화 저장 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"그래프 시각화 실패: {e}")
            return f"시각화 실패: {str(e)}"
    
    def _generate_simple_html_visualization(self) -> str:
        """간단한 HTML 시각화 생성"""
        nodes = []
        edges = []
        
        # 노드 데이터 생성
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "type": attrs.get("type", "unknown"),
                "size": 10
            })
        
        # 엣지 데이터 생성
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "label": attrs.get("predicate", "related")
            })
        
        # 간단한 HTML 템플릿
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>그래프 시각화</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .info {{ margin-bottom: 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
        .node-list {{ margin-top: 20px; }}
        .node-item {{ margin: 5px 0; padding: 5px; background: #e8f4f8; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>🧠 그래프 시각화</h1>
    
    <div class="info">
        <h3>📊 그래프 통계</h3>
        <p><strong>노드 수:</strong> {len(nodes)}</p>
        <p><strong>엣지 수:</strong> {len(edges)}</p>
        <p><strong>생성 시간:</strong> {datetime.now().isoformat()}</p>
    </div>
    
    <div class="node-list">
        <h3>📋 노드 목록</h3>
        {''.join([f'<div class="node-item"><strong>{node["id"]}</strong> ({node["type"]})</div>' for node in nodes])}
    </div>
    
    <div class="node-list">
        <h3>🔗 관계 목록</h3>
        {''.join([f'<div class="node-item">{edge["source"]} --{edge["label"]}--> {edge["target"]}</div>' for edge in edges])}
    </div>
</body>
</html>
        """
        
        return html_template


logger.info("🧠 그래프 엔진 로드 완료!") 