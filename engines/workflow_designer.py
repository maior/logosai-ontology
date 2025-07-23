"""
🎯 Workflow Designer
워크플로우 설계자

SemanticQuery를 기반으로 최적의 워크플로우를 설계합니다.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import time
import uuid

from ..core.models import (
    SemanticQuery, WorkflowPlan, WorkflowStep, 
    ExecutionStrategy, OptimizationStrategy, WorkflowComplexity
)
from ..core.interfaces import WorkflowDesigner as IWorkflowDesigner


class SmartWorkflowDesigner(IWorkflowDesigner):
    """🎯 스마트 워크플로우 설계자"""
    
    def __init__(self, installed_agents_info: List[Dict[str, Any]] = None):
        # 실제 설치된 에이전트 정보 저장
        self.installed_agents_info = installed_agents_info or []
        self.agents_capabilities_cache = {}  # 에이전트 능력 캐시
        
        # 기본 에이전트 능력 템플릿 (폴백용)
        self.agent_capability_templates = {
            "internet_agent": {
                "domains": ["web", "search", "information"],
                "capabilities": ["search", "web_scraping", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 15.0
            },
            "finance_agent": {
                "domains": ["finance", "stock", "currency"],
                "capabilities": ["financial_data", "market_analysis"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0
            },
            "weather_agent": {
                "domains": ["weather", "climate"],
                "capabilities": ["weather_forecast", "climate_data"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 10.0
            },
            "calculate_agent": {
                "domains": ["math", "calculation"],
                "capabilities": ["arithmetic", "computation"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 5.0
            },
            "chart_agent": {
                "domains": ["visualization", "chart"],
                "capabilities": ["chart_creation", "data_visualization"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 25.0
            },
            "memo_agent": {
                "domains": ["memory", "storage"],
                "capabilities": ["data_storage", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 8.0
            },
            "analysis_agent": {
                "domains": ["analysis", "research"],
                "capabilities": ["data_analysis", "research"],
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_time": 30.0
            }
        }
        
        # 에이전트 간 의존성 규칙 (기본 템플릿)
        self.dependency_rules = {
            "chart_agent": ["internet_agent", "finance_agent", "calculate_agent"],
            "analysis_agent": ["internet_agent", "finance_agent"],
            "memo_agent": ["internet_agent", "analysis_agent", "chart_agent"],
            "calculate_agent": ["internet_agent", "finance_agent"]
        }
        
        # 설치된 에이전트 정보를 기반으로 능력 캐시 구축
        self._build_capabilities_cache()
        
        logger.info(f"🎯 SmartWorkflowDesigner 초기화 완료 - 설치된 에이전트: {len(self.installed_agents_info)}개")
    
    def _build_capabilities_cache(self):
        """설치된 에이전트 정보를 기반으로 능력 캐시 구축"""
        try:
            logger.info(f"🔧 에이전트 능력 캐시 구축 시작 - 총 {len(self.installed_agents_info)}개 에이전트")
            
            for i, agent_info in enumerate(self.installed_agents_info):
                agent_id = agent_info.get('agent_id', '')
                if not agent_id:
                    logger.warning(f"에이전트 {i+1}: agent_id가 없습니다.")
                    continue
                
                # 에이전트 데이터에서 능력 정보 추출
                agent_data = agent_info.get('agent_data', {})
                capabilities_info = self._extract_capabilities_from_agent_data(agent_id, agent_data)
                
                self.agents_capabilities_cache[agent_id] = capabilities_info
                
                # 상세 로깅
                logger.info(f"  🤖 에이전트 {i+1}: {agent_id}")
                logger.info(f"    - 이름: {capabilities_info.get('name', agent_id)}")
                logger.info(f"    - 도메인: {capabilities_info.get('domains', [])}")
                logger.info(f"    - 능력: {len(capabilities_info.get('capabilities', []))}개")
                logger.info(f"    - 복잡도: {capabilities_info.get('complexity', 'UNKNOWN')}")
                logger.info(f"    - 예상 시간: {capabilities_info.get('estimated_time', 0)}초")
                
                logger.debug(f"에이전트 {agent_id} 능력 정보 캐시됨: {capabilities_info}")
            
            logger.info(f"✅ 에이전트 능력 캐시 구축 완료 - {len(self.agents_capabilities_cache)}개 에이전트")
                
        except Exception as e:
            logger.error(f"능력 캐시 구축 실패: {e}")
            logger.error(f"설치된 에이전트 정보: {self.installed_agents_info}")
    
    def _extract_capabilities_from_agent_data(self, agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 데이터에서 능력 정보 추출"""
        try:
            logger.debug(f"🔍 에이전트 {agent_id} 능력 정보 추출 시작")
            
            # 기본 정보
            capabilities_info = {
                "domains": [],
                "capabilities": [],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0,
                "description": agent_data.get('description', ''),
                "name": agent_data.get('name', agent_id)
            }
            
            # capabilities에서 정보 추출
            if 'capabilities' in agent_data:
                logger.debug(f"  능력 정보 추출: {len(agent_data['capabilities'])}개")
                for capability in agent_data['capabilities']:
                    cap_id = capability.get('id', '')
                    cap_name = capability.get('name', '')
                    cap_desc = capability.get('description', '')
                    
                    capabilities_info["capabilities"].append(cap_id)
                    
                    # 도메인 추론
                    domain = self._infer_domain_from_capability(cap_id, cap_name, cap_desc)
                    if domain and domain not in capabilities_info["domains"]:
                        capabilities_info["domains"].append(domain)
                        logger.debug(f"    도메인 추가: {domain} (from {cap_id})")
            
            # 태그에서 도메인 정보 추출
            if 'tags' in agent_data:
                logger.debug(f"  태그 정보 추출: {len(agent_data['tags'])}개")
                for tag in agent_data['tags']:
                    domain = self._infer_domain_from_tag(tag)
                    if domain and domain not in capabilities_info["domains"]:
                        capabilities_info["domains"].append(domain)
                        logger.debug(f"    도메인 추가: {domain} (from tag: {tag})")
            
            # 메타데이터에서 에이전트 타입 추출
            metadata = agent_data.get('metadata', {})
            agent_type = metadata.get('agent_type', '')
            if agent_type:
                type_info = self._get_type_based_info(agent_type)
                capabilities_info.update(type_info)
                logger.debug(f"  타입 기반 정보 적용: {agent_type}")
            
            # 복잡도 및 시간 추정
            complexity, estimated_time = self._estimate_complexity_and_time(agent_id, agent_data)
            capabilities_info["complexity"] = complexity
            capabilities_info["estimated_time"] = estimated_time
            
            # 도메인이 비어있으면 에이전트 ID로부터 추론
            if not capabilities_info["domains"]:
                capabilities_info["domains"] = self._infer_domains_from_agent_id(agent_id)
                logger.debug(f"  ID 기반 도메인 추론: {capabilities_info['domains']}")
            
            # 능력이 비어있으면 기본값 설정
            if not capabilities_info["capabilities"]:
                capabilities_info["capabilities"] = ["general_processing"]
                logger.debug(f"  기본 능력 설정: general_processing")
            
            logger.debug(f"✅ 에이전트 {agent_id} 능력 정보 추출 완료")
            return capabilities_info
            
        except Exception as e:
            logger.error(f"에이전트 {agent_id} 능력 정보 추출 실패: {e}")
            return self._get_fallback_capabilities(agent_id)
    
    def _infer_domain_from_capability(self, cap_id: str, cap_name: str, cap_desc: str) -> str:
        """능력 정보로부터 도메인 추론"""
        text = f"{cap_id} {cap_name} {cap_desc}".lower()
        
        if any(keyword in text for keyword in ["search", "internet", "web", "scraping"]):
            return "web"
        elif any(keyword in text for keyword in ["memo", "note", "storage", "save"]):
            return "memory"
        elif any(keyword in text for keyword in ["schedule", "calendar", "time", "event"]):
            return "scheduling"
        elif any(keyword in text for keyword in ["restaurant", "food", "place", "location"]):
            return "location"
        elif any(keyword in text for keyword in ["calculate", "math", "arithmetic", "compute"]):
            return "math"
        elif any(keyword in text for keyword in ["currency", "exchange", "money", "finance"]):
            return "finance"
        elif any(keyword in text for keyword in ["fortune", "luck", "zodiac", "prediction"]):
            return "entertainment"
        elif any(keyword in text for keyword in ["chart", "graph", "visual", "plot"]):
            return "visualization"
        else:
            return "general"
    
    def _infer_domain_from_tag(self, tag: str) -> str:
        """태그로부터 도메인 추론"""
        tag_lower = tag.lower()
        
        if any(keyword in tag_lower for keyword in ["인터넷", "검색", "웹", "internet", "search", "web"]):
            return "web"
        elif any(keyword in tag_lower for keyword in ["메모", "노트", "저장", "memo", "note", "storage"]):
            return "memory"
        elif any(keyword in tag_lower for keyword in ["일정", "캘린더", "스케줄", "schedule", "calendar"]):
            return "scheduling"
        elif any(keyword in tag_lower for keyword in ["맛집", "음식", "레스토랑", "restaurant", "food"]):
            return "location"
        elif any(keyword in tag_lower for keyword in ["계산", "수학", "math", "calculate"]):
            return "math"
        elif any(keyword in tag_lower for keyword in ["환율", "통화", "금융", "currency", "finance"]):
            return "finance"
        elif any(keyword in tag_lower for keyword in ["운세", "점", "fortune", "luck"]):
            return "entertainment"
        elif any(keyword in tag_lower for keyword in ["차트", "그래프", "시각화", "chart", "graph"]):
            return "visualization"
        else:
            return "general"
    
    def _get_type_based_info(self, agent_type: str) -> Dict[str, Any]:
        """에이전트 타입 기반 정보 반환"""
        type_mapping = {
            "INTERNET_SEARCH": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 15.0
            },
            "MEMO": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 8.0
            },
            "SCHEDULER": {
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 12.0
            },
            "RESTAURANT_FINDER": {
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 18.0
            },
            "CALCULATOR": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 5.0
            },
            "CUSTOM": {
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 15.0
            },
            "FORECASTING": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 10.0
            }
        }
        
        return type_mapping.get(agent_type, {
            "complexity": WorkflowComplexity.MODERATE,
            "estimated_time": 20.0
        })
    
    def _estimate_complexity_and_time(self, agent_id: str, agent_data: Dict[str, Any]) -> Tuple[WorkflowComplexity, float]:
        """복잡도와 예상 시간 추정"""
        try:
            # 메타데이터에서 타입 확인
            metadata = agent_data.get('metadata', {})
            agent_type = metadata.get('agent_type', '')
            
            type_info = self._get_type_based_info(agent_type)
            complexity = type_info.get('complexity', WorkflowComplexity.MODERATE)
            estimated_time = type_info.get('estimated_time', 20.0)
            
            # 능력 개수에 따른 조정
            capabilities_count = len(agent_data.get('capabilities', []))
            if capabilities_count > 5:
                if complexity == WorkflowComplexity.SIMPLE:
                    complexity = WorkflowComplexity.MODERATE
                estimated_time *= 1.2
            elif capabilities_count > 10:
                if complexity == WorkflowComplexity.MODERATE:
                    complexity = WorkflowComplexity.COMPLEX
                estimated_time *= 1.5
            
            return complexity, estimated_time
            
        except Exception as e:
            logger.error(f"복잡도 추정 실패: {e}")
            return WorkflowComplexity.MODERATE, 20.0
    
    def _infer_domains_from_agent_id(self, agent_id: str) -> List[str]:
        """에이전트 ID로부터 도메인 추론"""
        agent_id_lower = agent_id.lower()
        domains = []
        
        if any(keyword in agent_id_lower for keyword in ["internet", "web", "search"]):
            domains.append("web")
        if any(keyword in agent_id_lower for keyword in ["memo", "note", "storage"]):
            domains.append("memory")
        if any(keyword in agent_id_lower for keyword in ["schedule", "calendar"]):
            domains.append("scheduling")
        if any(keyword in agent_id_lower for keyword in ["restaurant", "finder", "location"]):
            domains.append("location")
        if any(keyword in agent_id_lower for keyword in ["calculator", "calc", "math"]):
            domains.append("math")
        if any(keyword in agent_id_lower for keyword in ["currency", "exchange", "finance"]):
            domains.append("finance")
        if any(keyword in agent_id_lower for keyword in ["fortune", "daily"]):
            domains.append("entertainment")
        
        return domains if domains else ["general"]
    
    def _get_fallback_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """폴백 능력 정보 반환"""
        return {
            "domains": self._infer_domains_from_agent_id(agent_id),
            "capabilities": ["general_processing"],
            "complexity": WorkflowComplexity.MODERATE,
            "estimated_time": 20.0,
            "description": f"General agent: {agent_id}",
            "name": agent_id
        }
    
    def _get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """에이전트 능력 정보 조회 (실제 설치된 에이전트 정보 기반)"""
        try:
            # 캐시에서 먼저 확인
            if agent_id in self.agents_capabilities_cache:
                return self.agents_capabilities_cache[agent_id].copy()
            
            # 설치된 에이전트 정보에서 찾기
            for agent_info in self.installed_agents_info:
                if agent_info.get('agent_id') == agent_id:
                    agent_data = agent_info.get('agent_data', {})
                    capabilities = self._extract_capabilities_from_agent_data(agent_id, agent_data)
                    
                    # 캐시에 저장
                    self.agents_capabilities_cache[agent_id] = capabilities
                    logger.info(f"🔍 에이전트 {agent_id} 실제 능력 정보 조회: {capabilities}")
                    return capabilities.copy()
            
            # 템플릿에서 찾기 (폴백)
            if agent_id in self.agent_capability_templates:
                logger.info(f"🔍 에이전트 {agent_id} 템플릿 능력 정보 사용")
                return self.agent_capability_templates[agent_id].copy()
            
            # 최후의 수단: ID 기반 추론
            inferred_capabilities = self._infer_agent_capabilities(agent_id)
            logger.info(f"🔍 에이전트 {agent_id} 능력 추론: {inferred_capabilities}")
            return inferred_capabilities
            
        except Exception as e:
            logger.error(f"에이전트 {agent_id} 능력 정보 조회 실패: {e}")
            return self._get_fallback_capabilities(agent_id)
    
    def _infer_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """에이전트 ID를 기반으로 능력 추론"""
        agent_id_lower = agent_id.lower()
        
        # 키워드 기반 추론
        if any(keyword in agent_id_lower for keyword in ["internet", "web", "search"]):
            return {
                "domains": ["web", "search", "information"],
                "capabilities": ["search", "web_scraping", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 15.0
            }
        elif any(keyword in agent_id_lower for keyword in ["finance", "stock", "money"]):
            return {
                "domains": ["finance", "stock", "currency"],
                "capabilities": ["financial_data", "market_analysis"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0
            }
        elif any(keyword in agent_id_lower for keyword in ["weather", "climate"]):
            return {
                "domains": ["weather", "climate"],
                "capabilities": ["weather_forecast", "climate_data"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 10.0
            }
        elif any(keyword in agent_id_lower for keyword in ["calc", "math", "compute"]):
            return {
                "domains": ["math", "calculation"],
                "capabilities": ["arithmetic", "computation"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 5.0
            }
        elif any(keyword in agent_id_lower for keyword in ["chart", "graph", "visual"]):
            return {
                "domains": ["visualization", "chart"],
                "capabilities": ["chart_creation", "data_visualization"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 25.0
            }
        elif any(keyword in agent_id_lower for keyword in ["memo", "note", "storage"]):
            return {
                "domains": ["memory", "storage"],
                "capabilities": ["data_storage", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 8.0
            }
        elif any(keyword in agent_id_lower for keyword in ["analysis", "analyze", "research"]):
            return {
                "domains": ["analysis", "research"],
                "capabilities": ["data_analysis", "research"],
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_time": 30.0
            }
        else:
            # 기본값
            return {
                "domains": ["general"],
                "capabilities": ["general_processing"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0
            }
    
    async def design_workflow(self, 
                            semantic_query: SemanticQuery,
                            available_agents: List[str]) -> WorkflowPlan:
        """
        워크플로우 설계
        
        Args:
            semantic_query: 의미론적 쿼리
            available_agents: 사용 가능한 에이전트 목록 (사용자별 설치된 에이전트)
        
        Returns:
            설계된 워크플로우 계획
        """
        try:
            logger.info(f"🎯 워크플로우 설계 시작 - 쿼리: {semantic_query.natural_language[:100]}...")
            logger.info(f"🎯 사용 가능한 에이전트: {available_agents}")
            
            if not available_agents:
                logger.warning("사용 가능한 에이전트가 없습니다. 기본 워크플로우를 생성합니다.")
                return self._create_fallback_workflow(semantic_query, [])
            
            # 1. 필요한 에이전트 선택
            required_agents = self._select_required_agents(semantic_query, available_agents)
            logger.info(f"선택된 에이전트: {required_agents}")
            
            if not required_agents:
                logger.warning("선택된 에이전트가 없습니다. 첫 번째 사용 가능한 에이전트를 사용합니다.")
                required_agents = [available_agents[0]]
            
            # 2. 워크플로우 단계 생성
            workflow_steps = self._create_workflow_steps(semantic_query, required_agents)
            
            # 3. 실행 그래프 구축
            execution_graph = self._build_execution_graph(workflow_steps)
            
            # 4. 최적화 전략 결정
            optimization_strategy = self._determine_optimization_strategy(semantic_query, workflow_steps)
            
            # 5. 메트릭 추정
            estimated_quality, estimated_time = self._estimate_workflow_metrics(workflow_steps)
            
            # 6. 추론 체인 생성
            reasoning_chain = self._generate_reasoning_chain(
                semantic_query, workflow_steps, optimization_strategy
            )
            
            # 7. 워크플로우 계획 생성
            workflow_plan = WorkflowPlan.create_simple(
                plan_id=f"workflow_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                semantic_query=semantic_query,
                steps=workflow_steps,
                execution_graph=execution_graph,
                strategy=optimization_strategy,  # strategy 파라미터 사용
                quality=estimated_quality,
                time=estimated_time,
                reasoning=reasoning_chain
            )
            
            logger.info(f"✅ 워크플로우 설계 완료 - {len(workflow_steps)}개 단계, 예상 시간: {estimated_time:.1f}초")
            return workflow_plan
            
        except Exception as e:
            logger.error(f"❌ 워크플로우 설계 실패: {e}")
            return self._create_fallback_workflow(semantic_query, available_agents)
    
    def optimize_workflow(self, workflow_plan: WorkflowPlan) -> WorkflowPlan:
        """워크플로우 최적화"""
        try:
            logger.info("🔧 워크플로우 최적화 시작")
            
            # 1. 병렬 처리 가능성 분석
            parallel_groups = self._identify_parallel_groups(workflow_plan.steps)
            
            # 2. 중복 단계 제거
            optimized_steps = self._remove_duplicate_steps(workflow_plan.steps)
            
            # 3. 의존성 최적화
            optimized_graph = self._optimize_dependencies(workflow_plan.execution_graph)
            
            # 4. 실행 전략 재평가
            optimized_strategy = self._reevaluate_strategy(optimized_steps, parallel_groups)
            
            # 최적화된 플랜 생성
            optimized_plan = WorkflowPlan.create_simple(
                semantic_query=workflow_plan.semantic_query,
                steps=optimized_steps,
                strategy=optimized_strategy,
                quality=workflow_plan.estimated_quality * 1.1,  # 최적화로 품질 향상
                time=workflow_plan.estimated_time * 0.9,  # 최적화로 시간 단축
                reasoning=workflow_plan.reasoning_chain + ["워크플로우 최적화 적용"]
            )
            
            optimized_plan.execution_graph = optimized_graph
            
            logger.info(f"✅ 워크플로우 최적화 완료: {len(optimized_steps)}개 단계")
            return optimized_plan
            
        except Exception as e:
            logger.error(f"❌ 워크플로우 최적화 실패: {e}")
            return workflow_plan
    
    def validate_workflow(self, workflow_plan: WorkflowPlan) -> bool:
        """워크플로우 유효성 검증 (추상 메서드 구현)"""
        try:
            # 1. 기본 구조 검증
            if not workflow_plan or not workflow_plan.steps:
                logger.warning("워크플로우 플랜이 비어있습니다.")
                return False
            
            # 2. 단계별 검증
            for step in workflow_plan.steps:
                if not step.step_id or not step.agent_id:
                    logger.warning(f"단계 {step.step_id}에 필수 정보가 누락되었습니다.")
                    return False
                
                # 에이전트 능력 조회 가능 여부 확인 (동적 조회)
                try:
                    agent_info = self._get_agent_capabilities(step.agent_id)
                    if not agent_info:
                        logger.warning(f"에이전트 {step.agent_id}의 능력 정보를 조회할 수 없습니다.")
                        return False
                except Exception as e:
                    logger.warning(f"에이전트 {step.agent_id} 능력 조회 실패: {e}")
                    return False
            
            # 3. 의존성 검증
            if hasattr(workflow_plan, 'execution_graph') and workflow_plan.execution_graph:
                # 순환 참조 검사
                if not nx.is_directed_acyclic_graph(workflow_plan.execution_graph):
                    logger.warning("워크플로우에 순환 참조가 있습니다.")
                    return False
                
                # 모든 단계가 그래프에 포함되어 있는지 확인
                step_ids = {step.step_id for step in workflow_plan.steps}
                graph_nodes = set(workflow_plan.execution_graph.nodes())
                if step_ids != graph_nodes:
                    logger.warning("실행 그래프와 단계 목록이 일치하지 않습니다.")
                    return False
            
            # 4. 의존성 규칙 검증 (동적으로 확인)
            for step in workflow_plan.steps:
                if step.agent_id in self.dependency_rules:
                    required_deps = self.dependency_rules[step.agent_id]
                    available_agents = [s.agent_id for s in workflow_plan.steps]
                    
                    # 필요한 의존성이 있는지 확인
                    has_dependency = any(dep in available_agents for dep in required_deps)
                    if not has_dependency and len(workflow_plan.steps) > 1:
                        logger.warning(f"에이전트 {step.agent_id}의 의존성이 충족되지 않았습니다.")
                        # 경고만 하고 실패로 처리하지는 않음
            
            # 5. 시간 추정 검증
            if hasattr(workflow_plan, 'estimated_time') and workflow_plan.estimated_time <= 0:
                logger.warning("워크플로우 예상 시간이 유효하지 않습니다.")
                return False
            
            logger.info("✅ 워크플로우 유효성 검증 통과")
            return True
            
        except Exception as e:
            logger.error(f"❌ 워크플로우 유효성 검증 실패: {e}")
            return False
    
    def _select_required_agents(self, semantic_query: SemanticQuery, available_agents: List[str]) -> List[str]:
        """필요한 에이전트 선택 - 단순화된 로직"""
        required_agents = []
        query_text = semantic_query.natural_language.lower()
        
        logger.info(f"🎯 에이전트 선택 시작 - 쿼리: '{semantic_query.natural_language}'")
        logger.info(f"🎯 사용 가능한 에이전트: {available_agents}")
        
        # Enhanced Query Processor 사용
        from ..core.enhanced_query_processor import get_enhanced_query_processor
        enhanced_processor = get_enhanced_query_processor()
        
        # 설치된 에이전트 정보 설정 (이미 설정되어 있어야 하지만 안전하게)
        if self.installed_agents_info:
            enhanced_processor.set_installed_agents_info(self.installed_agents_info)
        
        # Enhanced Query Processor의 간단한 에이전트 선택 로직 사용
        selected_agents = enhanced_processor._select_agents_simple(semantic_query.natural_language, available_agents)
        
        if selected_agents:
            required_agents.extend(selected_agents)
            logger.info(f"Enhanced Processor 선택 결과: {selected_agents}")
        else:
            # 폴백: 기본 선택 로직
            logger.warning("Enhanced Processor 선택 실패, 폴백 로직 사용")
            
            # 구조화된 쿼리에서 명시적 에이전트 확인
            if "required_agents" in semantic_query.structured_query:
                explicit_agents = semantic_query.structured_query["required_agents"]
                required_agents.extend([agent for agent in explicit_agents if agent in available_agents])
                logger.info(f"명시적 에이전트 선택: {required_agents}")
            
            # 기본 키워드 매칭 (매우 간단)
            if not required_agents:
                basic_mappings = {
                    "INTERNET_SEARCH": ["검색", "찾아", "정보"],
                    "MEMO": ["저장", "메모"],
                    "CALCULATOR": ["계산", "수학"]
                }
                
                for agent_id in available_agents:
                    agent_type = self._extract_agent_type_from_id(agent_id)
                    if agent_type in basic_mappings:
                        keywords = basic_mappings[agent_type]
                        if any(kw in query_text for kw in keywords):
                            required_agents.append(agent_id)
                            break
            
            # 최후 폴백: 인터넷 검색 에이전트 선택
            if not required_agents and available_agents:
                internet_agent = None
                for agent_id in available_agents:
                    if "internet" in agent_id.lower() or "search" in agent_id.lower():
                        internet_agent = agent_id
                        break
                
                if internet_agent:
                    required_agents.append(internet_agent)
                    logger.info(f"기본 인터넷 에이전트 선택: {internet_agent}")
                else:
                    required_agents.append(available_agents[0])
                    logger.info(f"기본 에이전트 선택: {available_agents[0]}")
        
        logger.info(f"🎯 최종 선택된 에이전트: {required_agents}")
        return required_agents
    
    def _extract_agent_type_from_id(self, agent_id: str) -> str:
        """에이전트 ID에서 타입 추출"""
        # 설치된 에이전트 정보에서 타입 확인
        for agent_info in self.installed_agents_info:
            if agent_info.get('agent_id') == agent_id:
                agent_data = agent_info.get('agent_data', {})
                agent_type = agent_data.get('metadata', {}).get('agent_type', '')
                if agent_type:
                    return agent_type
        
        # 폴백: ID에서 패턴 매칭으로 추출
        agent_id_lower = agent_id.lower()
        if 'memo' in agent_id_lower:
            return 'MEMO'
        elif 'internet' in agent_id_lower or 'search' in agent_id_lower:
            return 'INTERNET_SEARCH'
        elif 'schedule' in agent_id_lower or 'calendar' in agent_id_lower:
            return 'SCHEDULER'
        elif 'restaurant' in agent_id_lower or 'finder' in agent_id_lower:
            return 'RESTAURANT_FINDER'
        elif 'calculator' in agent_id_lower or 'calc' in agent_id_lower:
            return 'CALCULATOR'
        elif 'fortune' in agent_id_lower or 'daily' in agent_id_lower:
            return 'FORECASTING'
        elif 'currency' in agent_id_lower or 'exchange' in agent_id_lower:
            return 'CURRENCY'
        else:
            return 'CUSTOM'
    
    def _create_workflow_steps(self, semantic_query: SemanticQuery, required_agents: List[str]) -> List[WorkflowStep]:
        """워크플로우 단계 생성"""
        steps = []
        
        for i, agent_id in enumerate(required_agents):
            agent_info = self._get_agent_capabilities(agent_id)
            
            # 단계 목적 생성
            purpose = self._generate_step_purpose(semantic_query, agent_id, i)
            
            # 의존성 결정
            dependencies = self._determine_dependencies(agent_id, required_agents[:i])
            
            # 워크플로우 단계 생성
            step = WorkflowStep.create_simple(
                agent_id=agent_id,
                purpose=purpose,
                concepts=semantic_query.concepts,
                complexity=agent_info.get("complexity", WorkflowComplexity.MODERATE),
                depends_on=dependencies,
                estimated_time=agent_info.get("estimated_time", 30.0)
            )
            
            steps.append(step)
        
        return steps
    
    def _generate_step_purpose(self, semantic_query: SemanticQuery, agent_id: str, step_index: int) -> str:
        """단계 목적 생성"""
        agent_info = self._get_agent_capabilities(agent_id)
        primary_capability = agent_info.get("capabilities", ["processing"])[0]
        
        if step_index == 0:
            return f"Primary {primary_capability} for: {semantic_query.natural_language[:100]}"
        else:
            return f"Secondary {primary_capability} based on previous results"
    
    def _determine_dependencies(self, agent_id: str, previous_agents: List[str]) -> List[str]:
        """의존성 결정"""
        dependencies = []
        
        # 실제 에이전트 ID에서 기본 타입 추출 (예: memo_agent_dcf1704b61d4250e8762ba41f -> memo_agent)
        def extract_base_agent_type(full_agent_id: str) -> str:
            # 설치된 에이전트 정보에서 타입 확인
            for agent_info in self.installed_agents_info:
                if agent_info.get('agent_id') == full_agent_id:
                    agent_data = agent_info.get('agent_data', {})
                    agent_type = agent_data.get('metadata', {}).get('agent_type', '')
                    
                    # 에이전트 타입을 기본 에이전트 이름으로 매핑
                    type_mapping = {
                        'INTERNET_SEARCH': 'internet_agent',
                        'MEMO': 'memo_agent',
                        'SCHEDULER': 'schedule_agent',
                        'RESTAURANT_FINDER': 'restaurant_agent',
                        'CALCULATOR': 'calculate_agent',
                        'CUSTOM': 'custom_agent',
                        'FORECASTING': 'fortune_agent'
                    }
                    
                    if agent_type in type_mapping:
                        return type_mapping[agent_type]
            
            # 폴백: ID에서 패턴 매칭으로 추출
            agent_id_lower = full_agent_id.lower()
            if 'memo' in agent_id_lower:
                return 'memo_agent'
            elif 'internet' in agent_id_lower or 'search' in agent_id_lower:
                return 'internet_agent'
            elif 'schedule' in agent_id_lower or 'calendar' in agent_id_lower:
                return 'schedule_agent'
            elif 'restaurant' in agent_id_lower or 'finder' in agent_id_lower:
                return 'restaurant_agent'
            elif 'calculator' in agent_id_lower or 'calc' in agent_id_lower:
                return 'calculate_agent'
            elif 'fortune' in agent_id_lower or 'daily' in agent_id_lower:
                return 'fortune_agent'
            elif 'currency' in agent_id_lower or 'exchange' in agent_id_lower:
                return 'currency_agent'
            else:
                return 'general_agent'
        
        # 현재 에이전트의 기본 타입 추출
        base_agent_type = extract_base_agent_type(agent_id)
        
        # 규칙 기반 의존성 확인
        if base_agent_type in self.dependency_rules:
            required_deps = self.dependency_rules[base_agent_type]
            
            for dep_type in required_deps:
                # 이전 에이전트들 중에서 해당 타입과 매칭되는 에이전트 찾기
                for prev_agent in previous_agents:
                    prev_base_type = extract_base_agent_type(prev_agent)
                    if prev_base_type == dep_type:
                        # 해당 에이전트의 단계 ID 생성
                        dep_index = previous_agents.index(prev_agent)
                        dep_step_id = f"step_{dep_index:06d}"
                        dependencies.append(dep_step_id)
                        logger.debug(f"의존성 추가: {agent_id} -> {prev_agent} (단계: {dep_step_id})")
                        break
        
        return dependencies
    
    def _build_execution_graph(self, workflow_steps: List[WorkflowStep]) -> nx.DiGraph:
        """실행 그래프 구성"""
        graph = nx.DiGraph()
        
        # 노드 추가
        for step in workflow_steps:
            graph.add_node(step.step_id, step=step)
        
        # 의존성 엣지 추가
        for step in workflow_steps:
            for dependency in step.depends_on:
                if dependency in [s.step_id for s in workflow_steps]:
                    graph.add_edge(dependency, step.step_id)
        
        # 순환 참조 확인 및 제거
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("순환 참조 감지, 제거 중...")
            self._remove_cycles(graph)
        
        return graph
    
    def _remove_cycles(self, graph: nx.DiGraph):
        """순환 참조 제거"""
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    # 마지막 엣지 제거
                    graph.remove_edge(cycle[-1], cycle[0])
                    logger.info(f"순환 제거: {cycle[-1]} -> {cycle[0]}")
        except Exception as e:
            logger.error(f"순환 제거 실패: {e}")
    
    def _determine_optimization_strategy(self, 
                                       semantic_query: SemanticQuery, 
                                       workflow_steps: List[WorkflowStep]) -> OptimizationStrategy:
        """최적화 전략 결정"""
        query_text = semantic_query.natural_language.lower()
        
        # 속도 우선 키워드
        if any(word in query_text for word in ["빠르게", "즉시", "급하게", "신속"]):
            return OptimizationStrategy.SPEED_FIRST
        
        # 품질 우선 키워드
        if any(word in query_text for word in ["정확하게", "자세히", "완벽하게", "정밀"]):
            return OptimizationStrategy.QUALITY_FIRST
        
        # 자원 효율 키워드
        if any(word in query_text for word in ["효율적", "절약", "최소"]):
            return OptimizationStrategy.RESOURCE_EFFICIENT
        
        # 단계 수에 따른 기본 전략
        if len(workflow_steps) <= 2:
            return OptimizationStrategy.SPEED_FIRST
        elif len(workflow_steps) >= 5:
            return OptimizationStrategy.QUALITY_FIRST
        else:
            return OptimizationStrategy.BALANCED
    
    def _estimate_workflow_metrics(self, workflow_steps: List[WorkflowStep]) -> Tuple[float, float]:
        """워크플로우 메트릭스 추정"""
        # 품질 추정 (에이전트 복잡도 기반)
        complexity_scores = {
            WorkflowComplexity.SIMPLE: 0.7,
            WorkflowComplexity.MODERATE: 0.8,
            WorkflowComplexity.COMPLEX: 0.9,
            WorkflowComplexity.SOPHISTICATED: 0.95
        }
        
        quality_scores = [complexity_scores.get(step.estimated_complexity, 0.8) for step in workflow_steps]
        estimated_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.8
        
        # 시간 추정 (병렬 처리 고려)
        total_time = sum(step.estimated_time for step in workflow_steps)
        
        # 병렬 처리 가능성 고려
        independent_steps = [step for step in workflow_steps if not step.depends_on]
        if len(independent_steps) > 1:
            # 병렬 처리로 시간 단축
            estimated_time = total_time * 0.7
        else:
            estimated_time = total_time
        
        return estimated_quality, estimated_time
    
    def _generate_reasoning_chain(self, 
                                semantic_query: SemanticQuery,
                                workflow_steps: List[WorkflowStep],
                                optimization_strategy: OptimizationStrategy) -> List[str]:
        """추론 체인 생성"""
        reasoning = []
        
        # 쿼리 분석
        reasoning.append(f"쿼리 의도 분석: {semantic_query.intent}")
        reasoning.append(f"필요한 개념: {', '.join(semantic_query.concepts[:3])}")
        
        # 에이전트 선택
        agent_names = [step.agent_id for step in workflow_steps]
        reasoning.append(f"선택된 에이전트: {', '.join(agent_names)}")
        
        # 실행 전략
        reasoning.append(f"최적화 전략: {optimization_strategy.value}")
        
        # 의존성 분석
        dependent_steps = [step for step in workflow_steps if step.depends_on]
        if dependent_steps:
            reasoning.append(f"의존성 단계: {len(dependent_steps)}개")
        else:
            reasoning.append("모든 단계가 독립적으로 실행 가능")
        
        return reasoning
    
    def _create_fallback_workflow(self, semantic_query: SemanticQuery, available_agents: List[str]) -> WorkflowPlan:
        """폴백 워크플로우 생성"""
        logger.warning("폴백 워크플로우 생성")
        
        # 기본 에이전트 선택
        fallback_agent = "internet_agent" if "internet_agent" in available_agents else available_agents[0]
        
        # 단순한 단계 생성
        fallback_step = WorkflowStep.create_simple(
            agent_id=fallback_agent,
            purpose=f"Fallback processing for: {semantic_query.natural_language}",
            concepts=["general_processing"],
            complexity=WorkflowComplexity.SIMPLE,
            estimated_time=30.0
        )
        
        return WorkflowPlan.create_simple(
            semantic_query=semantic_query,
            steps=[fallback_step],
            strategy=OptimizationStrategy.BALANCED,
            quality=0.6,
            time=30.0,
            reasoning=["폴백 워크플로우로 생성됨"]
        )
    
    def _identify_parallel_groups(self, workflow_steps: List[WorkflowStep]) -> List[List[WorkflowStep]]:
        """병렬 그룹 식별"""
        parallel_groups = []
        
        # 독립적인 단계들 그룹화
        independent_steps = [step for step in workflow_steps if not step.depends_on]
        if len(independent_steps) > 1:
            parallel_groups.append(independent_steps)
        
        # 같은 의존성을 가진 단계들 그룹화
        dependency_groups = {}
        for step in workflow_steps:
            if step.depends_on:
                dep_key = tuple(sorted(step.depends_on))
                if dep_key not in dependency_groups:
                    dependency_groups[dep_key] = []
                dependency_groups[dep_key].append(step)
        
        for group in dependency_groups.values():
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def _remove_duplicate_steps(self, workflow_steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """중복 단계 제거 - 연속된 동일 에이전트 호출 병합"""
        if not workflow_steps:
            return []
            
        optimized_steps = []
        i = 0
        
        while i < len(workflow_steps):
            current_step = workflow_steps[i]
            
            # 다음 단계들 중에서 연속된 동일 에이전트 찾기
            j = i + 1
            merged_purposes = [current_step.purpose]
            
            while j < len(workflow_steps):
                next_step = workflow_steps[j]
                
                # 같은 에이전트인지 확인
                if next_step.agent_id == current_step.agent_id:
                    # 의존성이 있는 경우, 연속성이 깨지므로 병합하지 않음
                    # (다른 에이전트에 의존하는 경우)
                    other_dependencies = [dep for dep in next_step.depends_on 
                                        if dep != current_step.step_id]
                    if other_dependencies:
                        break
                    
                    # 병합 가능한 단계
                    logger.info(f"🔀 연속된 동일 에이전트 호출 감지: {current_step.agent_id}")
                    logger.info(f"   단계 {i+1}와 단계 {j+1} 병합")
                    merged_purposes.append(next_step.purpose)
                    
                    # 개념 병합
                    for concept in next_step.required_concepts:
                        if concept not in current_step.required_concepts:
                            current_step.required_concepts.append(concept)
                    
                    # 추정 시간은 더 긴 것으로 설정 (병렬 처리 가능)
                    current_step.estimated_time = max(
                        current_step.estimated_time, 
                        next_step.estimated_time
                    )
                    
                    j += 1
                else:
                    # 다른 에이전트면 중단
                    break
            
            # 병합된 목적 업데이트
            if len(merged_purposes) > 1:
                current_step.purpose = f"병합된 작업 ({len(merged_purposes)}개): " + "; ".join(merged_purposes[:2])
                if len(merged_purposes) > 2:
                    current_step.purpose += f" 외 {len(merged_purposes)-2}개"
            
            optimized_steps.append(current_step)
            i = j  # 병합된 단계들을 건너뛰기
        
        logger.info(f"✅ 중복 제거 완료: {len(workflow_steps)}개 → {len(optimized_steps)}개 단계")
        return optimized_steps
    
    def _optimize_dependencies(self, execution_graph: nx.DiGraph) -> nx.DiGraph:
        """의존성 최적화"""
        optimized_graph = execution_graph.copy()
        
        # 불필요한 의존성 제거 (전이적 의존성)
        transitive_edges = []
        for node in optimized_graph.nodes():
            for successor in optimized_graph.successors(node):
                for indirect_successor in optimized_graph.successors(successor):
                    if optimized_graph.has_edge(node, indirect_successor):
                        transitive_edges.append((node, indirect_successor))
        
        for edge in transitive_edges:
            optimized_graph.remove_edge(*edge)
            logger.info(f"전이적 의존성 제거: {edge[0]} -> {edge[1]}")
        
        return optimized_graph
    
    def _reevaluate_strategy(self, 
                           optimized_steps: List[WorkflowStep],
                           parallel_groups: List[List[WorkflowStep]]) -> OptimizationStrategy:
        """실행 전략 재평가"""
        # 병렬 그룹이 많으면 병렬 처리 우선
        if len(parallel_groups) >= 2:
            return OptimizationStrategy.SPEED_FIRST
        
        # 복잡한 단계가 많으면 품질 우선
        complex_steps = [step for step in optimized_steps 
                        if step.estimated_complexity in [WorkflowComplexity.COMPLEX, WorkflowComplexity.SOPHISTICATED]]
        
        if len(complex_steps) >= len(optimized_steps) * 0.5:
            return OptimizationStrategy.QUALITY_FIRST
        
        # 기본적으로 균형 전략
        return OptimizationStrategy.BALANCED 