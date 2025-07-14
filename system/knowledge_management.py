"""
🧠 Knowledge Graph Management Module
지식 그래프 관리 모듈

온톨로지 지식 그래프 업데이트, 개념 추가, 관계 생성을 담당
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import SemanticQuery, WorkflowPlan, AgentExecutionResult
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine


class KnowledgeGraphManager:
    """🧠 지식 그래프 관리자 - 온톨로지 업데이트 및 개념 관리"""
    
    def __init__(self, knowledge_graph: KnowledgeGraphEngine):
        self.knowledge_graph = knowledge_graph
        
        logger.info("🧠 지식 그래프 관리자 초기화 완료")
    
    async def update_knowledge_graph(self, 
                                   semantic_query: SemanticQuery,
                                   workflow_plan: WorkflowPlan,
                                   execution_results: List[AgentExecutionResult],
                                   integrated_result: Dict[str, Any]):
        """메인 지식 그래프 업데이트 함수"""
        try:
            logger.info("🧠 지식 그래프 업데이트 시작")
            
            # 워크플로우 ID 생성
            workflow_id = f"workflow_{workflow_plan.plan_id}"
            
            # 1. 기본 온톨로지 개념 확보
            await self._ensure_basic_ontology_concepts()
            
            # 2. 워크플로우 개념 생성
            await self._create_workflow_concept(workflow_plan, workflow_id)
            
            # 3. 도메인 및 개념 노드 생성
            await self._create_domain_and_concept_nodes(semantic_query, execution_results, workflow_id)
            
            # 4. 에이전트 협업 네트워크 생성
            await self._create_agent_collaboration_network(execution_results, workflow_id)
            
            # 5. 지식 패턴 노드 생성
            await self._create_knowledge_pattern_nodes(semantic_query, execution_results, integrated_result, workflow_id)
            
            # 6. 관계 추가
            await self._add_relationships(semantic_query, execution_results, workflow_id)
            
            logger.info("✅ 지식 그래프 업데이트 완료")
            
        except Exception as e:
            logger.error(f"지식 그래프 업데이트 실패: {e}")
    
    async def _ensure_basic_ontology_concepts(self):
        """기본 온톨로지 개념 확보"""
        try:
            basic_concepts = [
                ("concept", "concept", {"name": "개념", "description": "추상적 아이디어나 생각"}),
                ("entity", "concept", {"name": "개체", "description": "구체적인 사물이나 객체"}),
                ("agent", "concept", {"name": "에이전트", "description": "작업을 수행하는 주체"}),
                ("workflow", "concept", {"name": "워크플로우", "description": "작업 흐름과 절차"}),
                ("domain", "concept", {"name": "도메인", "description": "특정 지식 영역"}),
                ("knowledge", "concept", {"name": "지식", "description": "축적된 정보와 경험"})
            ]
            
            for concept_id, concept_type, attributes in basic_concepts:
                await self.knowledge_graph.add_concept(concept_id, concept_type, attributes)
            
            logger.info("✅ 기본 온톨로지 개념 확보 완료")
            
        except Exception as e:
            logger.error(f"기본 온톨로지 개념 확보 실패: {e}")
    
    async def _create_workflow_concept(self, workflow_plan: WorkflowPlan, workflow_id: str):
        """워크플로우 개념 생성"""
        try:
            workflow_attributes = {
                "optimization_strategy": getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy)),
                "estimated_time": workflow_plan.estimated_time,
                "estimated_quality": workflow_plan.estimated_quality,
                "step_count": len(workflow_plan.steps),
                "created_at": workflow_plan.created_at.isoformat() if hasattr(workflow_plan.created_at, 'isoformat') else str(workflow_plan.created_at),
                "query_intent": workflow_plan.query.intent if workflow_plan.query else "unknown"
            }
            
            await self.knowledge_graph.add_concept(workflow_id, "workflow", workflow_attributes)
            
            # 기본 워크플로우 개념과 연결
            await self.knowledge_graph.add_relationship(workflow_id, "workflow", "is_instance_of")
            
            logger.info(f"✅ 워크플로우 개념 생성 완료: {workflow_id}")
            
        except Exception as e:
            logger.error(f"워크플로우 개념 생성 실패: {e}")
    
    async def _create_domain_and_concept_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """도메인 및 개념 노드 생성"""
        try:
            # 쿼리 의도를 기반으로 도메인 추론
            domain_mapping = {
                "information_retrieval": "information_domain",
                "data_analysis": "analysis_domain", 
                "calculation": "computation_domain",
                "creative_generation": "creative_domain",
                "task_automation": "automation_domain"
            }
            
            primary_domain = domain_mapping.get(semantic_query.intent, "general_domain")
            
            # 도메인 노드 생성
            domain_attributes = {
                "domain_name": primary_domain.replace("_", " ").title(),
                "query_intent": semantic_query.intent,
                "complexity_level": "high" if semantic_query.complexity_score > 0.7 else "medium" if semantic_query.complexity_score > 0.4 else "low"
            }
            
            await self.knowledge_graph.add_concept(primary_domain, "domain", domain_attributes)
            
            # 워크플로우와 도메인 연결
            await self.knowledge_graph.add_relationship(workflow_id, primary_domain, "operates_in_domain")
            
            # 엔티티들을 개념으로 추가
            for entity in semantic_query.entities:
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                entity_attributes = {
                    "entity_name": entity,
                    "source_query": semantic_query.query_id,
                    "domain": primary_domain
                }
                
                await self.knowledge_graph.add_concept(entity_id, "entity", entity_attributes)
                await self.knowledge_graph.add_relationship(entity_id, primary_domain, "belongs_to_domain")
                await self.knowledge_graph.add_relationship(workflow_id, entity_id, "processes_entity")
            
            # 개념들을 추가
            for concept in semantic_query.concepts:
                concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                concept_attributes = {
                    "concept_name": concept,
                    "source_query": semantic_query.query_id,
                    "domain": primary_domain
                }
                
                await self.knowledge_graph.add_concept(concept_id, "concept", concept_attributes)
                await self.knowledge_graph.add_relationship(concept_id, primary_domain, "belongs_to_domain")
                await self.knowledge_graph.add_relationship(workflow_id, concept_id, "involves_concept")
            
            logger.info(f"✅ 도메인 및 개념 노드 생성 완료: {primary_domain}")
            
        except Exception as e:
            logger.error(f"도메인 및 개념 노드 생성 실패: {e}")
    
    async def _create_agent_collaboration_network(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """에이전트 협업 네트워크 생성"""
        try:
            # 에이전트 노드들 생성
            agent_ids = []
            for result in execution_results:
                agent_attributes = {
                    "agent_type": getattr(result.agent_type, 'value', str(result.agent_type)),
                    "last_execution_time": result.execution_time,
                    "last_success": result.success,
                    "last_confidence": getattr(result, 'confidence', 0.8)
                }
                
                await self.knowledge_graph.add_concept(result.agent_id, "agent", agent_attributes)
                await self.knowledge_graph.add_relationship(workflow_id, result.agent_id, "utilizes_agent")
                agent_ids.append(result.agent_id)
            
            # 에이전트 간 협업 관계 생성
            for i in range(len(agent_ids) - 1):
                current_agent = agent_ids[i]
                next_agent = agent_ids[i + 1]
                
                await self.knowledge_graph.add_relationship(
                    current_agent, next_agent, "collaborates_with",
                    {"collaboration_type": "sequential", "workflow_id": workflow_id}
                )
            
            logger.info(f"✅ 에이전트 협업 네트워크 생성 완료")
            
        except Exception as e:
            logger.error(f"에이전트 협업 네트워크 생성 실패: {e}")
    
    async def _create_knowledge_pattern_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], integrated_result: Dict[str, Any], workflow_id: str):
        """지식 패턴 노드 생성"""
        try:
            # 실행 패턴 분석
            agent_types = [getattr(r.agent_type, 'value', str(r.agent_type)) for r in execution_results]
            unique_agent_types = list(set(agent_types))
            
            pattern_id = f"pattern_{workflow_id}"
            pattern_attributes = {
                "pattern_name": f"{semantic_query.intent}_pattern",
                "agent_types_used": unique_agent_types,
                "total_agents": len(execution_results),
                "unique_agent_types": len(unique_agent_types),
                "complexity_level": semantic_query.complexity_score,
                "success_pattern": integrated_result.get('success', False),
                "execution_strategy": "multi_agent" if len(execution_results) > 1 else "single_agent"
            }
            
            await self.knowledge_graph.add_concept(pattern_id, "knowledge", pattern_attributes)
            await self.knowledge_graph.add_relationship(workflow_id, pattern_id, "generates_knowledge")
            
            # 결과 유형별 지식 패턴
            if integrated_result.get('components'):
                for component_type, component_data in integrated_result['components'].items():
                    knowledge_id = f"knowledge_{workflow_id}_{component_type}"
                    knowledge_attributes = {
                        "knowledge_type": component_type,
                        "data_size": len(str(component_data)) if component_data else 0,
                        "generated_by": workflow_id
                    }
                    
                    await self.knowledge_graph.add_concept(knowledge_id, "knowledge", knowledge_attributes)
                    await self.knowledge_graph.add_relationship(pattern_id, knowledge_id, "contains_knowledge")
            
            logger.info(f"✅ 지식 패턴 노드 생성 완료")
            
        except Exception as e:
            logger.error(f"지식 패턴 노드 생성 실패: {e}")
    
    async def _add_relationships(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """추가 관계들 생성"""
        try:
            # 기본 관계들
            basic_relations = [
                ("agent", "capability", "has"),
                ("workflow", "task", "contains"),
                ("concept", "domain", "belongs_to"),
                ("entity", "concept", "is_instance_of")
            ]
            
            for subject, object_concept, predicate in basic_relations:
                await self.knowledge_graph.add_relationship(subject, object_concept, predicate)
            
            logger.info(f"✅ 추가 관계 생성 완료")
            
        except Exception as e:
            logger.error(f"추가 관계 생성 실패: {e}") 