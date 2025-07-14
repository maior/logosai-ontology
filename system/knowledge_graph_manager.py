"""
🧠 Knowledge Graph Manager
지식 그래프 관리자

온톨로지 지식 그래프 업데이트 및 시각화 관리
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult, WorkflowPlan
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine


class KnowledgeGraphManager:
    """🧠 지식 그래프 관리자"""
    
    def __init__(self, knowledge_graph: KnowledgeGraphEngine, session_id: str = "default"):
        self.knowledge_graph = knowledge_graph
        self.session_id = session_id
    
    async def update_knowledge_graph(self, 
                                   semantic_query: SemanticQuery,
                                   workflow_plan: WorkflowPlan,
                                   execution_results: List[AgentExecutionResult],
                                   integrated_result: Dict[str, Any]):
        """온톨로지 지식 그래프 업데이트 - 이미지 참고하여 풍부한 그래프 생성"""
        try:
            logger.info("🔗 풍부한 온톨로지 지식 그래프 업데이트 시작")
            
            workflow_id = workflow_plan.plan_id
            
            # 1. 핵심 쿼리 노드 생성 (중앙 허브)
            query_id = f"query_{semantic_query.query_id}"
            await self.knowledge_graph.add_concept(query_id, "query", {
                "natural_language": semantic_query.natural_language,
                "intent": semantic_query.intent,
                "complexity": getattr(semantic_query, 'complexity_score', 0.5),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            })
            
            # 2. 워크플로우 계획 노드 생성
            workflow_node = f"workflow_{workflow_id}"
            await self.knowledge_graph.add_concept(workflow_node, "workflow", {
                "plan_id": workflow_id,
                "optimization_strategy": str(workflow_plan.optimization_strategy),
                "estimated_quality": workflow_plan.estimated_quality,
                "estimated_time": workflow_plan.estimated_time,
                "total_steps": len(workflow_plan.steps),
                "session_id": self.session_id,
                "reasoning_chain": workflow_plan.reasoning_chain[:3] if workflow_plan.reasoning_chain else []
            })
            
            # 3. 쿼리 → 워크플로우 관계
            await self.knowledge_graph.add_relation(query_id, "triggers", workflow_node, {
                "trigger_type": "user_request",
                "confidence": 0.9
            })
            
            # 4. 에이전트 노드들과 실행 결과 생성
            for i, result in enumerate(execution_results):
                agent_node = f"agent_{result.agent_id}"
                result_node = f"result_{workflow_id}_{i}"
                
                # 에이전트 노드
                await self.knowledge_graph.add_concept(agent_node, "agent", {
                    "agent_id": result.agent_id,
                    "agent_type": str(result.agent_type) if hasattr(result, 'agent_type') else "unknown",
                    "capabilities": self._infer_agent_capabilities(result.agent_id),
                    "performance_history": {
                        "last_execution_time": result.execution_time,
                        "last_confidence": result.confidence,
                        "last_success": result.is_successful()
                    }
                })
                
                # 실행 결과 노드
                await self.knowledge_graph.add_concept(result_node, "execution_result", {
                    "agent_id": result.agent_id,
                    "execution_time": result.execution_time,
                    "confidence": result.confidence,
                    "success": result.is_successful(),
                    "workflow_step": i + 1,
                    "result_type": self._classify_result_type(result.result_data)
                })
                
                # 관계들 생성
                await self.knowledge_graph.add_relation(workflow_node, "executes_with", agent_node, {
                    "execution_order": i + 1,
                    "step_purpose": f"Step {i+1} execution"
                })
                
                await self.knowledge_graph.add_relation(agent_node, "produces", result_node, {
                    "production_time": result.execution_time,
                    "quality_score": result.confidence
                })
                
                await self.knowledge_graph.add_relation(result_node, "contributes_to", workflow_node, {
                    "contribution_weight": result.confidence,
                    "step_number": i + 1
                })
            
            # 5. 추가 노드 및 관계 생성
            await self._create_domain_and_concept_nodes(semantic_query, execution_results, workflow_id)
            await self._create_task_and_capability_nodes(semantic_query, execution_results, workflow_id)
            await self._create_performance_metric_nodes(execution_results, workflow_id)
            await self._create_agent_collaboration_network(execution_results, workflow_id)
            await self._create_temporal_sequence_relations(execution_results, workflow_id)
            await self._create_knowledge_pattern_nodes(semantic_query, execution_results, integrated_result, workflow_id)
            await self._create_context_environment_nodes(semantic_query, workflow_id)
            
            logger.info(f"✅ 풍부한 온톨로지 지식 그래프 업데이트 완료 - 워크플로우: {workflow_id}")
            
        except Exception as e:
            logger.error(f"지식 그래프 업데이트 실패: {e}")
    
    def _infer_agent_capabilities(self, agent_id: str) -> List[str]:
        """에이전트 ID로부터 능력 추론"""
        capabilities = []
        agent_lower = agent_id.lower()
        
        if 'internet' in agent_lower or 'search' in agent_lower:
            capabilities.extend(['web_search', 'information_retrieval', 'real_time_data'])
        if 'calculator' in agent_lower or 'math' in agent_lower:
            capabilities.extend(['calculation', 'mathematical_operations', 'numerical_analysis'])
        if 'weather' in agent_lower:
            capabilities.extend(['weather_data', 'meteorological_info', 'location_based_data'])
        if 'memo' in agent_lower:
            capabilities.extend(['text_storage', 'note_management', 'information_organization'])
        if 'calendar' in agent_lower:
            capabilities.extend(['schedule_management', 'time_organization', 'event_planning'])
        
        return capabilities if capabilities else ['general_processing']
    
    def _classify_result_type(self, result_data: Any) -> str:
        """결과 데이터 타입 분류"""
        if result_data is None:
            return "empty"
        elif isinstance(result_data, dict):
            if 'search_results' in str(result_data):
                return "search_results"
            elif 'calculation' in str(result_data):
                return "calculation_result"
            elif 'weather' in str(result_data):
                return "weather_data"
            else:
                return "structured_data"
        elif isinstance(result_data, str):
            return "text_response"
        elif isinstance(result_data, (int, float)):
            return "numerical_result"
        else:
            return "unknown"
    
    async def _create_domain_and_concept_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """도메인 및 개념 노드들 생성"""
        try:
            # 쿼리에서 도메인 추출
            query_text = semantic_query.natural_language.lower()
            domains = []
            
            if any(word in query_text for word in ['날씨', 'weather', '기상', '온도']):
                domains.append('weather')
            if any(word in query_text for word in ['환율', 'exchange', '달러', '유로']):
                domains.append('finance')
            if any(word in query_text for word in ['계산', 'calculate', '수학', 'math']):
                domains.append('calculation')
            if any(word in query_text for word in ['검색', 'search', '찾아', 'find']):
                domains.append('information')
            if any(word in query_text for word in ['메모', 'memo', '기록', 'note']):
                domains.append('productivity')
            
            if not domains:
                domains = ['general']
            
            # 도메인 노드들 생성
            for domain in domains:
                domain_node = f"domain_{domain}"
                await self.knowledge_graph.add_concept(domain_node, "domain", {
                    "domain_name": domain,
                    "query_relevance": 0.8,
                    "last_accessed": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                })
                
                # 쿼리-도메인 관계
                query_id = f"query_{semantic_query.query_id}"
                await self.knowledge_graph.add_relation(query_id, "belongs_to_domain", domain_node, {
                    "relevance_score": 0.8
                })
                
                # 에이전트-도메인 관계
                for result in execution_results:
                    if result.is_successful():
                        agent_node = f"agent_{result.agent_id}"
                        await self.knowledge_graph.add_relation(agent_node, "operates_in_domain", domain_node, {
                            "performance_score": result.confidence,
                            "execution_time": result.execution_time
                        })
            
            # 개념 엔티티들 생성
            entities = getattr(semantic_query, 'entities', [])
            for entity in entities[:5]:  # 최대 5개
                entity_node = f"entity_{entity}"
                await self.knowledge_graph.add_concept(entity_node, "entity", {
                    "entity_name": entity,
                    "source_query": semantic_query.query_id,
                    "extraction_confidence": 0.7
                })
                
                # 쿼리-엔티티 관계
                await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "contains_entity", entity_node, {
                    "entity_importance": 0.7
                })
                
        except Exception as e:
            logger.warning(f"도메인 및 개념 노드 생성 실패: {e}")
    
    async def _create_task_and_capability_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """태스크 및 능력 노드들 생성"""
        try:
            # 태스크 노드 생성
            task_node = f"task_{workflow_id}"
            await self.knowledge_graph.add_concept(task_node, "task", {
                "task_description": semantic_query.natural_language,
                "task_intent": semantic_query.intent,
                "complexity_level": getattr(semantic_query, 'complexity_score', 0.5),
                "workflow_id": workflow_id
            })
            
            # 쿼리-태스크 관계
            await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "defines_task", task_node, {
                "task_clarity": 0.8
            })
            
            # 각 에이전트의 능력 노드들 생성
            for result in execution_results:
                capabilities = self._infer_agent_capabilities(result.agent_id)
                
                for capability in capabilities:
                    capability_node = f"capability_{capability}"
                    await self.knowledge_graph.add_concept(capability_node, "capability", {
                        "capability_name": capability,
                        "capability_type": "agent_skill",
                        "last_used": datetime.now().isoformat()
                    })
                    
                    # 에이전트-능력 관계
                    agent_node = f"agent_{result.agent_id}"
                    await self.knowledge_graph.add_relation(agent_node, "has_capability", capability_node, {
                        "proficiency_level": result.confidence,
                        "usage_frequency": 1
                    })
                    
                    # 태스크-능력 관계
                    await self.knowledge_graph.add_relation(task_node, "requires_capability", capability_node, {
                        "requirement_strength": 0.6
                    })
                    
        except Exception as e:
            logger.warning(f"태스크 및 능력 노드 생성 실패: {e}")
    
    async def _create_performance_metric_nodes(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """성능 및 품질 메트릭 노드들 생성"""
        try:
            # 전체 워크플로우 성능 메트릭
            total_time = sum(r.execution_time for r in execution_results)
            avg_confidence = sum(r.confidence for r in execution_results) / len(execution_results) if execution_results else 0
            success_rate = sum(1 for r in execution_results if r.is_successful()) / len(execution_results) if execution_results else 0
            
            workflow_performance_node = f"performance_workflow_{workflow_id}"
            await self.knowledge_graph.add_concept(workflow_performance_node, "performance_metric", {
                "metric_type": "workflow_performance",
                "total_execution_time": total_time,
                "average_confidence": avg_confidence,
                "success_rate": success_rate,
                "agent_count": len(execution_results),
                "workflow_id": workflow_id
            })
            
            # 개별 에이전트 성능 메트릭
            for i, result in enumerate(execution_results):
                agent_performance_node = f"performance_{result.agent_id}_{workflow_id}"
                await self.knowledge_graph.add_concept(agent_performance_node, "performance_metric", {
                    "metric_type": "agent_performance",
                    "agent_id": result.agent_id,
                    "execution_time": result.execution_time,
                    "confidence": result.confidence,
                    "success": result.is_successful(),
                    "performance_tier": "high" if result.confidence > 0.8 else "medium" if result.confidence > 0.5 else "low"
                })
                
                # 에이전트-성능 관계
                agent_node = f"agent_{result.agent_id}"
                await self.knowledge_graph.add_relation(agent_node, "has_performance", agent_performance_node, {
                    "measurement_timestamp": datetime.now().isoformat()
                })
                
                # 워크플로우-성능 관계
                workflow_node = f"workflow_{workflow_id}"
                await self.knowledge_graph.add_relation(workflow_node, "measured_by", agent_performance_node, {
                    "step_number": i + 1
                })
                
        except Exception as e:
            logger.warning(f"성능 메트릭 노드 생성 실패: {e}")
    
    async def _create_agent_collaboration_network(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """에이전트 간 협업 관계 생성"""
        try:
            successful_results = [r for r in execution_results if r.is_successful()]
            
            # 성공한 에이전트들 간의 협업 관계
            for i, result1 in enumerate(successful_results):
                for result2 in successful_results[i+1:]:
                    agent1_node = f"agent_{result1.agent_id}"
                    agent2_node = f"agent_{result2.agent_id}"
                    
                    # 협업 관계
                    await self.knowledge_graph.add_relation(agent1_node, "collaborated_with", agent2_node, {
                        "workflow_id": workflow_id,
                        "collaboration_success": True,
                        "combined_confidence": (result1.confidence + result2.confidence) / 2,
                        "collaboration_type": "sequential" if abs(i - successful_results.index(result2)) == 1 else "parallel"
                    })
                    
                    # 상호 보완 관계 (능력이 다른 경우)
                    cap1 = self._infer_agent_capabilities(result1.agent_id)
                    cap2 = self._infer_agent_capabilities(result2.agent_id)
                    
                    if set(cap1) != set(cap2):  # 다른 능력을 가진 경우
                        await self.knowledge_graph.add_relation(agent1_node, "complements", agent2_node, {
                            "complementarity_score": 0.8,
                            "capability_overlap": len(set(cap1) & set(cap2)) / max(len(cap1), len(cap2))
                        })
                        
        except Exception as e:
            logger.warning(f"에이전트 협업 네트워크 생성 실패: {e}")
    
    async def _create_temporal_sequence_relations(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """시간적 순서 관계 생성"""
        try:
            # 실행 순서에 따른 선후 관계
            for i in range(len(execution_results) - 1):
                current_result = execution_results[i]
                next_result = execution_results[i + 1]
                
                current_agent = f"agent_{current_result.agent_id}"
                next_agent = f"agent_{next_result.agent_id}"
                
                # 선행 관계
                await self.knowledge_graph.add_relation(current_agent, "precedes", next_agent, {
                    "sequence_order": i + 1,
                    "time_gap": 0.1,  # 가정된 시간 간격
                    "workflow_id": workflow_id
                })
                
                # 결과 간 의존성
                current_result_node = f"result_{workflow_id}_{i}"
                next_result_node = f"result_{workflow_id}_{i+1}"
                
                await self.knowledge_graph.add_relation(current_result_node, "influences", next_result_node, {
                    "influence_strength": 0.6,
                    "dependency_type": "sequential"
                })
                
        except Exception as e:
            logger.warning(f"시간적 순서 관계 생성 실패: {e}")
    
    async def _create_knowledge_pattern_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], integrated_result: Dict[str, Any], workflow_id: str):
        """지식 패턴 및 학습 노드 생성"""
        try:
            # 학습 패턴 노드
            pattern_node = f"pattern_{workflow_id}"
            await self.knowledge_graph.add_concept(pattern_node, "knowledge_pattern", {
                "pattern_type": "workflow_execution",
                "query_type": semantic_query.intent,
                "agent_combination": [r.agent_id for r in execution_results],
                "success_pattern": [r.is_successful() for r in execution_results],
                "performance_pattern": [r.confidence for r in execution_results],
                "learned_at": datetime.now().isoformat()
            })
            
            # 쿼리-패턴 관계
            await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "generates_pattern", pattern_node, {
                "pattern_strength": integrated_result.get('confidence', 0.7)
            })
            
            # 인사이트 노드 (성공적인 실행에서)
            successful_results = [r for r in execution_results if r.is_successful()]
            if successful_results:
                insight_node = f"insight_{workflow_id}"
                await self.knowledge_graph.add_concept(insight_node, "insight", {
                    "insight_type": "execution_success",
                    "key_factors": [r.agent_id for r in successful_results],
                    "success_rate": len(successful_results) / len(execution_results),
                    "confidence_level": sum(r.confidence for r in successful_results) / len(successful_results),
                    "discovered_at": datetime.now().isoformat()
                })
                
                # 패턴-인사이트 관계
                await self.knowledge_graph.add_relation(pattern_node, "reveals", insight_node, {
                    "revelation_confidence": 0.8
                })
                
        except Exception as e:
            logger.warning(f"지식 패턴 노드 생성 실패: {e}")
    
    async def _create_context_environment_nodes(self, semantic_query: SemanticQuery, workflow_id: str):
        """컨텍스트 및 환경 노드 생성"""
        try:
            # 세션 컨텍스트 노드
            session_node = f"session_{self.session_id}"
            await self.knowledge_graph.add_concept(session_node, "session_context", {
                "session_id": self.session_id,
                "session_start": datetime.now().isoformat(),
                "query_count": 1  # 현재 쿼리 기준
            })
            
            # 쿼리-세션 관계
            await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "occurs_in_session", session_node, {
                "query_sequence": 1
            })
            
            # 환경 노드 (시스템 환경)
            environment_node = f"environment_{workflow_id}"
            await self.knowledge_graph.add_concept(environment_node, "execution_environment", {
                "system_type": "ontology_multi_agent",
                "execution_mode": "production",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id
            })
            
            # 워크플로우-환경 관계
            await self.knowledge_graph.add_relation(f"workflow_{workflow_id}", "executes_in", environment_node, {
                "environment_suitability": 0.9
            })
            
        except Exception as e:
            logger.warning(f"컨텍스트 및 환경 노드 생성 실패: {e}")
    
    async def create_default_ontology_data(self):
        """기본 온톨로지 데이터 생성"""
        try:
            logger.info("🏗️ 기본 온톨로지 데이터 생성 시작")
            
            # 1. 기본 에이전트들 추가
            agents = [
                ("internet_agent", "인터넷 검색 에이전트", ["search", "information_retrieval"]),
                ("calculator_agent", "계산 에이전트", ["calculation", "mathematical_operations"]),
                ("weather_agent", "날씨 에이전트", ["weather_data", "location_services"]),
                ("memo_agent", "메모 에이전트", ["text_storage", "note_management"]),
                ("analysis_agent", "분석 에이전트", ["data_analysis", "pattern_recognition"]),
                ("chart_agent", "차트 에이전트", ["data_visualization", "chart_generation"])
            ]
            
            for agent_id, description, capabilities in agents:
                await self.knowledge_graph.add_concept(f"agent_{agent_id}", "agent", {
                    "agent_id": agent_id,
                    "description": description,
                    "capabilities": capabilities,
                    "agent_type": "system_agent",
                    "confidence": 0.9,
                    "success": True
                })
            
            # 2. 기본 도메인들 추가
            domains = [
                ("information", "정보 검색 도메인"),
                ("calculation", "계산 처리 도메인"),
                ("weather", "날씨 정보 도메인"),
                ("productivity", "생산성 도구 도메인"),
                ("analysis", "데이터 분석 도메인"),
                ("visualization", "시각화 도메인")
            ]
            
            for domain_id, description in domains:
                await self.knowledge_graph.add_concept(f"domain_{domain_id}", "domain", {
                    "domain_name": domain_id,
                    "description": description,
                    "domain_type": "system_domain"
                })
            
            # 3. 기본 능력들 추가
            capabilities = [
                ("search", "검색 능력"),
                ("calculate", "계산 능력"),
                ("analyze", "분석 능력"),
                ("generate", "생성 능력"),
                ("visualize", "시각화 능력"),
                ("store", "저장 능력")
            ]
            
            for capability_id, description in capabilities:
                await self.knowledge_graph.add_concept(f"capability_{capability_id}", "capability", {
                    "capability_name": capability_id,
                    "description": description,
                    "capability_type": "system_capability"
                })
            
            # 4. 샘플 워크플로우 추가
            await self.knowledge_graph.add_concept("workflow_sample", "workflow", {
                "workflow_id": "sample_workflow",
                "description": "샘플 워크플로우",
                "workflow_type": "demo",
                "optimization_strategy": "balanced"
            })
            
            # 5. 관계들 추가
            relationships = [
                ("agent_internet_agent", "specializes_in_domain", "domain_information"),
                ("agent_calculator_agent", "specializes_in_domain", "domain_calculation"),
                ("agent_weather_agent", "specializes_in_domain", "domain_weather"),
                ("agent_memo_agent", "specializes_in_domain", "domain_productivity"),
                ("agent_analysis_agent", "specializes_in_domain", "domain_analysis"),
                ("agent_chart_agent", "specializes_in_domain", "domain_visualization"),
                
                ("agent_internet_agent", "has_capability", "capability_search"),
                ("agent_calculator_agent", "has_capability", "capability_calculate"),
                ("agent_analysis_agent", "has_capability", "capability_analyze"),
                ("agent_chart_agent", "has_capability", "capability_visualize"),
                ("agent_memo_agent", "has_capability", "capability_store"),
                
                ("workflow_sample", "executes_with", "agent_internet_agent"),
                ("workflow_sample", "executes_with", "agent_analysis_agent"),
                ("workflow_sample", "executes_with", "agent_chart_agent")
            ]
            
            for source, relationship, target in relationships:
                await self.knowledge_graph.add_relationship(source, target, relationship, {
                    "relationship_type": relationship,
                    "confidence": 0.8,
                    "context": "default_ontology"
                })
            
            logger.info(f"✅ 기본 온톨로지 데이터 생성 완료 - 노드: {self.knowledge_graph.graph.number_of_nodes()}개")
            
        except Exception as e:
            logger.error(f"기본 온톨로지 데이터 생성 실패: {e}") 