"""
🧠 Ontology System
통합 온톨로지 시스템

모든 컴포넌트를 통합하는 메인 시스템
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any
import asyncio
from loguru import logger

from ..core.models import SemanticQuery, ExecutionContext, AgentExecutionResult, WorkflowPlan
from ..engines.semantic_query_manager import SemanticQueryManager
from ..engines.execution_engine import AdvancedExecutionEngine, QueryComplexityAnalyzer
from ..engines.workflow_designer import SmartWorkflowDesigner
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine
from ..core.unified_query_processor import get_unified_query_processor
from .result_integration import ResultIntegrator
from .strategy_manager import StrategyManager
from .progress_callback import SimpleProgressCallback
from .knowledge_graph_manager import KnowledgeGraphManager
from .visualization_manager import VisualizationManager
from .result_processor import ResultProcessor
from .reasoning_generator import ReasoningGenerator
from .workflow_utils import WorkflowUtils
from .complexity_analyzer import ComplexityAnalyzer


class OntologySystem:
    """🧠 통합 온톨로지 시스템"""
    
    def __init__(self, 
                 email: str = "system@ontology.ai",
                 session_id: str = None,
                 project_id: str = None):
        
        # 세션 정보
        self.email = email
        self.session_id = session_id or f"session_{int(time.time())}"
        self.project_id = project_id or "default_project"
        
        # 핵심 엔진들 초기화
        self.semantic_query_manager = SemanticQueryManager()
        self.execution_engine = AdvancedExecutionEngine()
        self.knowledge_graph = KnowledgeGraphEngine()
        self.result_integrator = ResultIntegrator()
        
        # 워크플로우 설계자는 나중에 설치된 에이전트 정보와 함께 초기화
        self.workflow_designer = None
        self.installed_agents_info = []  # 설치된 에이전트 정보 저장
        
        # 복잡도 분석기
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # 전략 관리자
        self.strategy_manager = StrategyManager()
        
        # 분할된 관리자들 초기화
        self.knowledge_graph_manager = KnowledgeGraphManager(self.knowledge_graph, self.session_id)
        self.visualization_manager = VisualizationManager(self.knowledge_graph)
        self.result_processor = ResultProcessor()
        self.reasoning_generator = ReasoningGenerator()
        self.workflow_utils = WorkflowUtils()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # 기본 에이전트 목록 (폴백용)
        self.available_agents = [
            "internet_agent", "finance_agent", "weather_agent",
            "calculate_agent", "chart_agent", "memo_agent", "analysis_agent"
        ]
        
        # 시스템 상태
        self.is_initialized = False
        self.execution_history = []
        
        logger.info(f"🧠 온톨로지 시스템 초기화: {self.session_id}")
    
    def _initialize_workflow_designer(self, installed_agents_info: List[Dict[str, Any]] = None):
        """워크플로우 설계자 초기화 (설치된 에이전트 정보와 함께)"""
        try:
            if installed_agents_info:
                self.installed_agents_info = installed_agents_info
                logger.info(f"🎯 설치된 에이전트 정보 업데이트: {len(installed_agents_info)}개")
            
            # SmartWorkflowDesigner를 설치된 에이전트 정보와 함께 초기화
            self.workflow_designer = SmartWorkflowDesigner(self.installed_agents_info)
            logger.info(f"🎯 워크플로우 설계자 초기화 완료 - 에이전트 정보: {len(self.installed_agents_info)}개")
            
        except Exception as e:
            logger.error(f"워크플로우 설계자 초기화 실패: {e}")
            # 폴백: 기본 워크플로우 설계자 사용
            self.workflow_designer = SmartWorkflowDesigner()

    async def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 온톨로지 시스템 초기화 시작")
            
            # 각 컴포넌트 초기화 (필요한 경우)
            # 현재는 생성자에서 이미 초기화됨
            
            self.is_initialized = True
            logger.info("✅ 온톨로지 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 온톨로지 시스템 초기화 실패: {e}")
            raise
    
    async def process_query(self, query_text: str, execution_context: ExecutionContext) -> Dict[str, Any]:
        """쿼리 처리 - 메인 진입점"""
        try:
            logger.info(f"🚀 온톨로지 시스템 쿼리 처리 시작: '{query_text[:50]}...'")
            
            # 설치된 에이전트 정보 추출 및 워크플로우 설계자 초기화
            installed_agents_info = self.workflow_utils.extract_installed_agents_info(execution_context)
            available_agents = self.workflow_utils.extract_available_agents(execution_context)
            
            logger.info(f"🤖 사용 가능한 에이전트: {len(available_agents)}개 - {available_agents}")
            
            # 통합 쿼리 프로세서 사용 (NEW!)
            unified_processor = get_unified_query_processor()
            
            if installed_agents_info:
                unified_processor.set_installed_agents_info(installed_agents_info)
            
            # 통합 처리 실행 - 한 번의 LLM 호출로 모든 분석 완료
            logger.info("🚀 통합 LLM 처리 시작...")
            start_time = datetime.now()
            
            unified_result = await unified_processor.process_unified_query(
                query_text, 
                available_agents
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"⏱️ 통합 LLM 처리 완료: {processing_time:.2f}초")
            
            # 통합 결과를 기존 형식으로 변환
            if installed_agents_info:
                self.workflow_utils.update_installed_agents_info(installed_agents_info)
            workflow_plan = self.workflow_utils.convert_unified_result_to_workflow(unified_result, query_text, execution_context)
            
            # 워크플로우 실행
            logger.info(f"🔧 워크플로우 실행 시작: {len(workflow_plan.steps)}개 단계")
            execution_results = await self.execution_engine.execute_workflow(
                workflow_plan.query, workflow_plan, execution_context
            )
            
            # 결과 통합
            logger.info(f"🔗 실행 결과 통합 중...")
            # SemanticQuery 생성 (결과 통합에 필요)
            semantic_query = workflow_plan.query if workflow_plan and workflow_plan.query else SemanticQuery(
                query_id=f'semantic_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent='information_retrieval',
                entities=[],
                concepts=[],
                relations=[],
                complexity_score=0.7,
                created_at=datetime.now(),
                metadata={}
            )
            
            # 분할된 결과 처리자 사용
            integrated_result = await self.result_processor.integrate_results(
                execution_results, workflow_plan, semantic_query
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # 성능 메트릭 기록 - 안전한 타입 변환
            raw_quality_score = unified_result.get('quality_assessment', {}).get('overall_confidence', 0.8)
            try:
                # quality_score를 안전하게 float로 변환
                if isinstance(raw_quality_score, str):
                    quality_score = float(raw_quality_score)
                elif isinstance(raw_quality_score, (int, float)):
                    quality_score = float(raw_quality_score)
                else:
                    quality_score = 0.8  # 기본값
            except (ValueError, TypeError):
                quality_score = 0.8  # 기본값
            
            performance_metrics = {
                "total_processing_time": total_time,
                "llm_processing_time": processing_time,
                "execution_time": total_time - processing_time,
                "agents_used": len(workflow_plan.steps),
                "execution_strategy": workflow_plan.optimization_strategy.value if hasattr(workflow_plan.optimization_strategy, 'value') else str(workflow_plan.optimization_strategy),
                "unified_processing": True,  # 통합 처리 표시
                "quality_score": quality_score
            }
            
            # 최종 응답 구성
            final_response = {
                "query": query_text,
                "response": integrated_result.get("integrated_content", "응답을 생성할 수 없습니다."),
                "execution_summary": integrated_result.get("execution_summary", {}),
                "performance_metrics": performance_metrics,
                "workflow_visualization": integrated_result.get("workflow_visualization", ""),
                "confidence_score": integrated_result.get("confidence_score", 0.7),
                "sources": integrated_result.get("sources", []),
                "processing_metadata": {
                    "unified_analysis": unified_result.get('query_analysis', {}),
                    "agent_mappings": unified_result.get('agent_mappings', []),
                    "execution_plan": unified_result.get('execution_plan', {}),
                    "fallback_mode": unified_result.get('fallback_mode', False)
                }
            }
            
            logger.info(f"✅ 온톨로지 쿼리 처리 완료 (총 {total_time:.2f}초, 품질: {performance_metrics['quality_score']:.2f})")
            
            return final_response
            
        except Exception as e:
            logger.error(f"온톨로지 쿼리 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 폴백: 기존 방식으로 처리
            return await self._fallback_query_processing(query_text, execution_context)



    async def _fallback_query_processing(self, query_text: str, execution_context: ExecutionContext) -> Dict[str, Any]:
        """폴백 쿼리 처리"""
        logger.warning("🔄 폴백 모드로 쿼리 처리")
        
        try:
            # 기존 방식으로 처리 시도
            workflow_plan = self.workflow_utils.create_minimal_workflow(query_text, execution_context)
            
            execution_results = await self.execution_engine.execute_workflow(
                workflow_plan.query, workflow_plan, execution_context
            )
            
            # SemanticQuery 생성 (폴백용)
            semantic_query = workflow_plan.query if workflow_plan and workflow_plan.query else SemanticQuery(
                query_id=f'fallback_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent='information_retrieval',
                entities=[],
                concepts=[],
                relations=[],
                complexity_score=0.5,
                created_at=datetime.now(),
                metadata={"fallback_mode": True}
            )
            
            integrated_result = await self.result_processor.integrate_results(
                execution_results, workflow_plan, semantic_query
            )
            
            return {
                "query": query_text,
                "response": integrated_result.get("content", "죄송합니다. 처리 중 오류가 발생했습니다."),
                "execution_summary": integrated_result.get("metadata", {}),
                "performance_metrics": {
                    "total_processing_time": 30,
                    "fallback_mode": True
                },
                "confidence_score": 0.5,
                "sources": [],
                "processing_metadata": {"fallback_processing": True}
            }
            
        except Exception as e:
            logger.error(f"폴백 처리도 실패: {e}")
            return {
                "query": query_text,
                "response": "시스템 오류로 인해 요청을 처리할 수 없습니다. 나중에 다시 시도해주세요.",
                "execution_summary": {},
                "performance_metrics": {"error": True},
                "confidence_score": 0.0,
                "sources": [],
                "processing_metadata": {"system_error": True, "error_message": str(e)}
            }
    
    async def _generate_detailed_reasoning(
        self, 
        execution_results: List[AgentExecutionResult],
        workflow_plan: WorkflowPlan,
        semantic_query: SemanticQuery,
        complexity_analysis: Dict[str, Any],
        integrated_result: Dict[str, Any]
    ) -> str:
        """상세한 reasoning 생성 - 분할된 관리자 사용"""
        return await self.reasoning_generator.generate_detailed_reasoning(
            execution_results, workflow_plan, semantic_query, complexity_analysis, integrated_result
        )
    
    def _generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
        """워크플로우 계획을 Mermaid 다이어그램으로 생성"""
        try:
            mermaid_lines = ["graph TD"]
            
            # 시작 노드
            mermaid_lines.append('    Start(["🚀 시작"]) --> Query["📝 쿼리 분석"]')
            
            # 각 단계를 노드로 추가
            for i, step in enumerate(workflow_plan.steps):
                step_id = f"Step{i+1}"
                agent_name = step.agent_id.replace('_', ' ').title()
                purpose = step.semantic_purpose[:30] + "..." if len(step.semantic_purpose) > 30 else step.semantic_purpose
                
                # 노드 정의
                mermaid_lines.append(f'    {step_id}["🤖 {agent_name}<br/>{purpose}"]')
                
                # 연결 관계
                if i == 0:
                    mermaid_lines.append(f'    Query --> {step_id}')
                else:
                    prev_step_id = f"Step{i}"
                    mermaid_lines.append(f'    {prev_step_id} --> {step_id}')
                
                # 의존성이 있는 경우
                if hasattr(step, 'depends_on') and step.depends_on:
                    for dep in step.depends_on:
                        # 의존성 단계 찾기
                        for j, dep_step in enumerate(workflow_plan.steps):
                            if dep_step.step_id == dep:
                                dep_step_id = f"Step{j+1}"
                                mermaid_lines.append(f'    {dep_step_id} -.-> {step_id}')
                                break
            
            # 마지막 단계에서 결과로 연결
            if workflow_plan.steps:
                last_step_id = f"Step{len(workflow_plan.steps)}"
                mermaid_lines.append(f'    {last_step_id} --> Result[["✅ 결과 통합"]]')
                mermaid_lines.append('    Result --> End(["🎉 완료"])')
            else:
                mermaid_lines.append('    Query --> End(["🎉 완료"])')
            
            # 스타일 추가
            mermaid_lines.extend([
                "",
                "    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
                "    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px", 
                "    classDef agent fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
                "    classDef result fill:#fff3e0,stroke:#e65100,stroke-width:2px",
                "",
                "    class Start,End startEnd",
                "    class Query,Result result"
            ])
            
            # 에이전트 노드들에 스타일 적용
            for i in range(len(workflow_plan.steps)):
                mermaid_lines.append(f"    class Step{i+1} agent")
            
            return "\n".join(mermaid_lines)
            
        except Exception as e:
            logger.error(f"Mermaid 다이어그램 생성 실패: {e}")
            return f'graph TD\n    Start(["시작"]) --> Error["다이어그램 생성 실패: {str(e)}"]\n    Error --> End(["종료"])'
    
    def _safe_analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """안전한 복잡도 분석 - 분할된 관리자 사용"""
        return self.complexity_analyzer.safe_analyze_complexity(semantic_query)
    
    async def _integrate_results(self, 
                               execution_results: List[AgentExecutionResult],
                               workflow_plan: WorkflowPlan,
                               semantic_query: SemanticQuery) -> Dict[str, Any]:
        """결과 통합"""
        try:
            logger.info(f"🔄 결과 통합 시작 - 총 {len(execution_results)}개 결과")
            
            # 모든 결과 상세 로깅
            for i, result in enumerate(execution_results):
                logger.info(f"  결과 {i+1}: {result.agent_id}")
                logger.info(f"    성공: {result.success}")
                logger.info(f"    실행시간: {result.execution_time:.2f}초")
                logger.info(f"    신뢰도: {result.confidence}")
                
                if result.data:
                    logger.info(f"    데이터 타입: {type(result.data)}")
                    if isinstance(result.data, dict):
                        logger.info(f"    데이터 키: {list(result.data.keys())}")
                        # 실제 데이터 내용 일부 로깅
                        for key, value in result.data.items():
                            if isinstance(value, str) and len(value) > 100:
                                logger.info(f"      {key}: {value[:100]}...")
                            else:
                                logger.info(f"      {key}: {value}")
                    else:
                        logger.info(f"    데이터 내용: {str(result.data)[:200]}...")
                
                if result.error_message:
                    logger.warning(f"    오류: {result.error_message}")
            
            # 성공한 결과들만 추출
            successful_results = [r for r in execution_results if r.success]
            logger.info(f"📊 성공한 결과: {len(successful_results)}개")
            
            if not successful_results:
                logger.warning("⚠️ 모든 에이전트 실행이 실패했습니다.")
                return {
                    "status": "failed",
                    "message": "모든 에이전트 실행이 실패했습니다.",
                    "error_details": [r.error_message for r in execution_results if r.error_message]
                }
            
            # 결과 데이터 수집
            result_data = []
            for result in successful_results:
                if result.data:
                    processed_data = {
                        "agent_id": result.agent_id,
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time
                    }
                    result_data.append(processed_data)
                    logger.info(f"✅ 결과 데이터 추가: {result.agent_id}")
            
            logger.info(f"📋 처리할 결과 데이터: {len(result_data)}개")
            
            # 의도별 결과 통합
            intent = getattr(semantic_query, 'intent', 'general')
            logger.info(f"🎯 의도별 통합 시작: {intent}")
            
            if intent == "information_retrieval":
                integrated_content = self._integrate_information_results(result_data)
            elif intent == "analysis":
                integrated_content = self._integrate_analysis_results(result_data)
            elif intent == "comparison":
                integrated_content = self._integrate_comparison_results(result_data)
            else:
                integrated_content = self._integrate_general_results(result_data)
            
            logger.info(f"✅ 통합 완료 - 내용 길이: {len(integrated_content)}자")
            
            return {
                "status": "success",
                "content": integrated_content,
                "metadata": {
                    "total_agents": len(execution_results),
                    "successful_agents": len(successful_results),
                    "total_execution_time": sum(r.execution_time for r in execution_results),
                    "average_confidence": sum(r.confidence for r in successful_results) / len(successful_results) if successful_results else 0,
                    "strategy_used": getattr(workflow_plan, 'optimization_strategy', 'AUTO')
                }
            }
            
        except Exception as e:
            logger.error(f"결과 통합 실패: {e}")
            return {
                "status": "error",
                "message": f"결과 통합 중 오류 발생: {str(e)}",
                "partial_results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in execution_results]
            }
    
    def _integrate_information_results(self, result_data: List[Dict[str, Any]]) -> str:
        """정보 검색 결과 통합"""
        logger.info(f"🔍 정보 검색 결과 통합 시작 - {len(result_data)}개 결과")
        
        contents = []
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            
            logger.info(f"  처리 중: {agent_id}")
            logger.info(f"  데이터 타입: {type(data)}")
            
            content = None
            
            if isinstance(data, dict):
                logger.info(f"  데이터 키들: {list(data.keys())}")
                
                # answer 키가 있으면 우선적으로 사용
                if "answer" in data:
                    content = data["answer"]
                    logger.info(f"  📝 answer 키에서 내용 추출: {len(str(content))}자")
                # 다양한 형태의 검색 결과 처리
                elif "search_results" in data:
                    search_results = data["search_results"]
                    if isinstance(search_results, list) and search_results:
                        content = f"검색 결과 {len(search_results)}개 항목:\n"
                        for i, item in enumerate(search_results[:5], 1):  # 상위 5개만
                            title = item.get("title", "제목 없음")
                            snippet = item.get("snippet", item.get("description", ""))
                            content += f"{i}. {title}\n   {snippet}\n"
                    else:
                        content = "검색 결과가 없습니다."
                elif "items" in data:
                    items = data["items"]
                    if isinstance(items, list) and items:
                        content = f"검색 항목 {len(items)}개:\n"
                        for i, item in enumerate(items[:5], 1):
                            title = item.get("title", item.get("name", "제목 없음"))
                            content += f"{i}. {title}\n"
                    else:
                        content = "검색 항목이 없습니다."
                elif "content" in data:
                    content = data["content"]
                elif "text" in data:
                    content = data["text"]
                elif "result" in data:
                    # result가 딕셔너리이고 answer가 있는지 확인
                    result_value = data["result"]
                    if isinstance(result_value, dict) and "answer" in result_value:
                        content = result_value["answer"]
                        logger.info(f"  📝 result.answer에서 내용 추출: {len(str(content))}자")
                    else:
                        content = str(result_value)
                else:
                    # 다른 키가 없으면 전체 딕셔너리를 문자열로 변환하지 말고 주요 내용만 추출
                    logger.warning(f"  ⚠️ 알려진 키가 없음, 전체 구조: {data}")
                    # 딕셔너리에서 가장 중요해 보이는 텍스트 내용을 찾아서 추출
                    for key in ['response', 'message', 'output', 'data']:
                        if key in data:
                            potential_content = data[key]
                            if isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                                content = potential_content
                                logger.info(f"  📝 {key} 키에서 내용 추출")
                                break
                    
                    # 여전히 content가 없으면 전체를 문자열로 변환
                    if content is None:
                        content = str(data)
                        logger.warning(f"  ⚠️ 폴백: 전체 딕셔너리를 문자열로 변환")
            else:
                content = str(data)
                logger.info(f"  📝 문자열/기타 타입에서 직접 변환")
            
            if content and str(content).strip():
                contents.append(content)  # agent_id 제거
                logger.info(f"  ✅ 내용 추가: {len(str(content))}자 (agent_id 제외)")
            else:
                logger.warning(f"  ⚠️ 빈 내용: {agent_id}")
        
        final_content = "\n\n".join(contents) if contents else "검색 결과를 찾을 수 없습니다."
        logger.info(f"🔍 정보 검색 결과 통합 완료 - 최종 길이: {len(final_content)}자")
        
        return final_content
    
    def _integrate_analysis_results(self, result_data: List[Dict[str, Any]]) -> str:
        """분석 결과 통합"""
        analysis_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            confidence = result["confidence"]
            
            if isinstance(data, dict):
                insights = data.get("insights", [])
                if insights:
                    analysis_parts.append(f"**분석 결과 (신뢰도: {confidence:.2f})**")  # agent_id 제거
                    for insight in insights:
                        analysis_parts.append(f"- {insight}")
                else:
                    analysis_parts.append(str(data))  # agent_id 제거
            else:
                analysis_parts.append(str(data))  # agent_id 제거
        
        return "\n".join(analysis_parts)
    
    def _integrate_comparison_results(self, result_data: List[Dict[str, Any]]) -> str:
        """비교 결과 통합"""
        comparison_parts = ["## 비교 분석 결과"]
        
        for i, result in enumerate(result_data, 1):
            agent_id = result["agent_id"]
            data = result["data"]
            
            comparison_parts.append(f"\n### {i}. 결과")  # agent_id 제거
            comparison_parts.append(str(data))
        
        return "\n".join(comparison_parts)
    
    def _integrate_general_results(self, result_data: List[Dict[str, Any]]) -> str:
        """일반 결과 통합"""
        general_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            
            general_parts.append(str(data))  # agent_id 제거
        
        return "\n".join(general_parts)
    
    async def _update_knowledge_graph(self, 
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
            
            # 5. 도메인 및 개념 노드들 생성
            await self._create_domain_and_concept_nodes(semantic_query, execution_results, workflow_id)
            
            # 6. 태스크 및 능력 노드들 생성
            await self._create_task_and_capability_nodes(semantic_query, execution_results, workflow_id)
            
            # 7. 성능 및 품질 메트릭 노드들 생성
            await self._create_performance_metric_nodes(execution_results, workflow_id)
            
            # 8. 에이전트 간 협업 관계 생성
            await self._create_agent_collaboration_network(execution_results, workflow_id)
            
            # 9. 시간적 순서 관계 생성
            await self._create_temporal_sequence_relations(execution_results, workflow_id)
            
            # 10. 지식 패턴 및 학습 노드 생성
            await self._create_knowledge_pattern_nodes(semantic_query, execution_results, integrated_result, workflow_id)
            
            # 11. 컨텍스트 및 환경 노드 생성
            await self._create_context_environment_nodes(semantic_query, workflow_id)
            
            logger.info(f"✅ 풍부한 온톨로지 지식 그래프 업데이트 완료 - 워크플로우: {workflow_id}")
            
        except Exception as e:
            logger.error(f"지식 그래프 업데이트 실패: {e}")
    
    def _infer_agent_capabilities(self, agent_id: str) -> List[str]:
        """에이전트 ID로부터 능력 추론 - 분할된 관리자 사용"""
        return self.complexity_analyzer.infer_agent_capabilities(agent_id)
    
    def _classify_result_type(self, result_data: Any) -> str:
        """결과 데이터 타입 분류 - 분할된 관리자 사용"""
        return self.complexity_analyzer.classify_result_type(result_data)
    
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
                "user_email": self.email,
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
    
    async def _add_domain_concept_relations(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult]):
        """도메인별 개념 관계 추가"""
        try:
            # 쿼리 의도에 따른 도메인 분류
            intent = semantic_query.intent.lower()
            domain = "general"
            
            if any(keyword in intent for keyword in ["weather", "날씨", "기상"]):
                domain = "weather"
            elif any(keyword in intent for keyword in ["finance", "금융", "환율", "주식"]):
                domain = "finance"
            elif any(keyword in intent for keyword in ["calculation", "계산", "수학"]):
                domain = "calculation"
            elif any(keyword in intent for keyword in ["search", "검색", "정보"]):
                domain = "information"
            
            # 도메인 노드 추가
            domain_id = f"domain_{domain}"
            await self.knowledge_graph.add_concept(domain_id, "domain", {
                "domain_name": domain,
                "query_count": 1,
                "last_accessed": datetime.now().isoformat()
            })
            
            # 쿼리-도메인 관계
            query_id = f"query_{semantic_query.query_id}"
            await self.knowledge_graph.add_relation(
                query_id, "belongs_to_domain", domain_id,
                {"confidence": 0.8}
            )
            
            # 에이전트-도메인 관계
            for result in execution_results:
                if result.is_successful():
                    agent_id = f"agent_{result.agent_id}"
                    await self.knowledge_graph.add_relation(
                        agent_id, "specializes_in_domain", domain_id,
                        {
                            "performance_score": result.confidence,
                            "execution_time": result.execution_time
                        }
                    )
            
        except Exception as e:
            logger.warning(f"도메인 개념 관계 추가 실패: {e}")
    
    async def _add_performance_based_relations(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """성능 기반 관계 추가"""
        try:
            # 고성능 에이전트들 식별 (신뢰도 0.8 이상)
            high_performance_agents = [
                r for r in execution_results 
                if r.is_successful() and r.confidence >= 0.8
            ]
            
            # 고성능 에이전트들 간의 "high_performance_collaboration" 관계
            for i, agent1 in enumerate(high_performance_agents):
                for agent2 in high_performance_agents[i+1:]:
                    await self.knowledge_graph.add_relation(
                        f"agent_{agent1.agent_id}",
                        "high_performance_collaboration",
                        f"agent_{agent2.agent_id}",
                        {
                            "workflow_id": workflow_id,
                            "avg_confidence": (agent1.confidence + agent2.confidence) / 2,
                            "performance_tier": "high"
                        }
                    )
            
            # 빠른 실행 에이전트들 식별 (5초 이하)
            fast_agents = [
                r for r in execution_results 
                if r.is_successful() and r.execution_time <= 5.0
            ]
            
            for agent in fast_agents:
                await self.knowledge_graph.add_concept(
                    f"performance_fast_{agent.agent_id}",
                    "performance_metric",
                    {
                        "agent_id": agent.agent_id,
                        "metric_type": "fast_execution",
                        "execution_time": agent.execution_time,
                        "workflow_id": workflow_id
                    }
                )
            
        except Exception as e:
            logger.warning(f"성능 기반 관계 추가 실패: {e}")
    
    async def _ensure_basic_ontology_concepts(self):
        """기본 온톨로지 개념들 추가 (그래프가 비어있을 때)"""
        try:
            # 그래프가 비어있거나 노드가 적으면 기본 개념들 추가
            if self.knowledge_graph.graph.number_of_nodes() < 5:
                logger.info("🏗️ 기본 온톨로지 개념들 추가 중...")
                
                # 기본 에이전트 타입들
                basic_agents = [
                    ("internet_agent", "인터넷 검색"),
                    ("calculator_agent", "계산 처리"),
                    ("weather_agent", "날씨 정보"),
                    ("memo_agent", "메모 관리"),
                    ("calendar_agent", "일정 관리")
                ]
                
                for agent_id, description in basic_agents:
                    await self.knowledge_graph.add_concept(f"agent_{agent_id}", "agent", {
                        "agent_id": agent_id,
                        "description": description,
                        "type": "basic_agent"
                    })
                
                # 기본 도메인들
                basic_domains = [
                    ("information", "정보 검색"),
                    ("calculation", "계산 처리"),
                    ("weather", "날씨 정보"),
                    ("productivity", "생산성 도구"),
                    ("general", "일반 처리")
                ]
                
                for domain_id, description in basic_domains:
                    await self.knowledge_graph.add_concept(f"domain_{domain_id}", "domain", {
                        "domain_name": domain_id,
                        "description": description,
                        "type": "basic_domain"
                    })
                
                # 기본 능력들
                basic_capabilities = [
                    ("search", "검색 능력"),
                    ("calculate", "계산 능력"),
                    ("analyze", "분석 능력"),
                    ("generate", "생성 능력"),
                    ("process", "처리 능력")
                ]
                
                for capability_id, description in basic_capabilities:
                    await self.knowledge_graph.add_concept(f"capability_{capability_id}", "capability", {
                        "capability_name": capability_id,
                        "description": description,
                        "type": "basic_capability"
                    })
                
                # 기본 관계들 추가
                await self._add_basic_ontology_relations()
                
                logger.info("✅ 기본 온톨로지 개념들 추가 완료")
                
        except Exception as e:
            logger.warning(f"기본 온톨로지 개념 추가 실패: {e}")
    
    async def _add_basic_ontology_relations(self):
        """기본 온톨로지 관계들 추가"""
        try:
            # 에이전트-능력 관계
            agent_capability_map = {
                "internet_agent": ["search", "analyze"],
                "calculator_agent": ["calculate", "process"],
                "weather_agent": ["search", "analyze"],
                "memo_agent": ["generate", "process"],
                "calendar_agent": ["process", "analyze"]
            }
            
            for agent_id, capabilities in agent_capability_map.items():
                for capability in capabilities:
                    await self.knowledge_graph.add_relation(
                        f"agent_{agent_id}",
                        "hasCapability",
                        f"capability_{capability}",
                        {"type": "basic_relation", "confidence": 0.9}
                    )
            
            # 에이전트-도메인 관계
            agent_domain_map = {
                "internet_agent": "information",
                "calculator_agent": "calculation",
                "weather_agent": "weather",
                "memo_agent": "productivity",
                "calendar_agent": "productivity"
            }
            
            for agent_id, domain in agent_domain_map.items():
                await self.knowledge_graph.add_relation(
                    f"agent_{agent_id}",
                    "specializes_in_domain",
                    f"domain_{domain}",
                    {"type": "basic_relation", "confidence": 0.8}
                )
            
            # 도메인-능력 관계
            domain_capability_map = {
                "information": ["search", "analyze"],
                "calculation": ["calculate", "process"],
                "weather": ["search", "analyze"],
                "productivity": ["generate", "process", "analyze"]
            }
            
            for domain, capabilities in domain_capability_map.items():
                for capability in capabilities:
                    await self.knowledge_graph.add_relation(
                        f"domain_{domain}",
                        "requires",
                        f"capability_{capability}",
                        {"type": "basic_relation", "confidence": 0.7}
                    )
            
        except Exception as e:
            logger.warning(f"기본 온톨로지 관계 추가 실패: {e}")
    
    async def _add_domain_specific_relations(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult]):
        """도메인별 특화 관계 추가"""
        try:
            # 쿼리에서 도메인 추출
            query_text = semantic_query.natural_language.lower()
            
            # 도메인별 키워드 매핑
            domain_keywords = {
                "weather": ["날씨", "기상", "온도", "습도", "비", "눈", "바람"],
                "finance": ["환율", "주식", "투자", "금융", "돈", "가격", "시세"],
                "technology": ["기술", "컴퓨터", "소프트웨어", "프로그래밍", "AI", "인공지능"],
                "health": ["건강", "의료", "병원", "약", "치료", "진료"],
                "education": ["교육", "학습", "공부", "학교", "대학", "강의"],
                "entertainment": ["게임", "영화", "음악", "스포츠", "오락"]
            }
            
            # 도메인 감지
            detected_domains = []
            for domain, keywords in domain_keywords.items():
                if any(keyword in query_text for keyword in keywords):
                    detected_domains.append(domain)
            
            # 도메인별 관계 생성
            for domain in detected_domains:
                domain_id = f"domain_{domain}"
                await self.knowledge_graph.add_concept(domain_id, "domain", {
                    "name": domain,
                    "keywords": domain_keywords[domain],
                    "detected_in_query": True
                })
                
                # 쿼리와 도메인 관계
                query_id = f"query_{semantic_query.query_id}"
                await self.knowledge_graph.add_relation(query_id, "belongs_to_domain", domain_id, {
                    "confidence": 0.8,
                    "detection_method": "keyword_matching"
                })
                
                # 성공한 에이전트들과 도메인 관계
                for result in execution_results:
                    if result.success:
                        agent_id = f"agent_{result.agent_id}"
                        await self.knowledge_graph.add_relation(agent_id, "specializes_in_domain", domain_id, {
                            "confidence": result.confidence,
                            "execution_time": result.execution_time
                        })
            
            logger.info(f"🏷️ 도메인별 관계 추가 완료: {detected_domains}")
            
        except Exception as e:
            logger.warning(f"도메인별 관계 추가 실패: {e}")
    
    async def _add_temporal_relations(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """시간 기반 관계 추가"""
        try:
            # 실행 시간 순서대로 정렬
            sorted_results = sorted(execution_results, key=lambda x: x.created_at)
            
            # 시간 기반 순서 관계 추가
            for i in range(len(sorted_results) - 1):
                current_result = sorted_results[i]
                next_result = sorted_results[i + 1]
                
                current_agent = f"agent_{current_result.agent_id}"
                next_agent = f"agent_{next_result.agent_id}"
                
                # 시간적 선행 관계
                await self.knowledge_graph.add_relation(current_agent, "precedes_in_time", next_agent, {
                    "time_gap": (next_result.created_at - current_result.created_at).total_seconds(),
                    "workflow_id": workflow_id,
                    "sequence_order": i + 1
                })
            
            # 성능 기반 시간 관계
            fast_agents = [r for r in execution_results if r.execution_time < 5.0]
            slow_agents = [r for r in execution_results if r.execution_time > 10.0]
            
            # 빠른 에이전트들 간의 관계
            for agent in fast_agents:
                agent_id = f"agent_{agent.agent_id}"
                performance_id = f"performance_fast"
                await self.knowledge_graph.add_concept(performance_id, "performance_metric", {
                    "category": "fast_execution",
                    "threshold": 5.0
                })
                await self.knowledge_graph.add_relation(agent_id, "has_performance", performance_id, {
                    "execution_time": agent.execution_time,
                    "category": "fast"
                })
            
            logger.info(f"⏰ 시간 기반 관계 추가 완료: {len(execution_results)}개 에이전트")
            
        except Exception as e:
            logger.warning(f"시간 기반 관계 추가 실패: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭스 조회"""
        try:
            return {
                "session_info": {
                    "session_id": self.session_id,
                    "email": self.email,
                    "project_id": self.project_id,
                    "is_initialized": self.is_initialized
                },
                "execution_history": {
                    "total_executions": len(self.execution_history),
                    "recent_executions": self.execution_history[-5:] if self.execution_history else []
                },
                "semantic_query_manager": self.semantic_query_manager.get_metrics(),
                "execution_engine": self.execution_engine.get_metrics(),
                "knowledge_graph": {
                    "metadata": self.knowledge_graph.metadata,
                    "available_agents": self.available_agents
                }
            }
        except Exception as e:
            logger.error(f"시스템 메트릭스 조회 실패: {e}")
            return {"error": str(e)}
    
    def get_knowledge_graph_visualization(self, max_nodes: int = 50) -> Dict[str, Any]:
        """지식 그래프 시각화 데이터 조회 (첫 번째 이미지 스타일)"""
        try:
            logger.info(f"🎨 온톨로지 지식 그래프 시각화 요청 - 최대 노드: {max_nodes}")
            
            # 그래프가 비어있으면 기본 온톨로지 데이터 생성
            current_nodes = self.knowledge_graph.graph.number_of_nodes()
            if current_nodes == 0:
                logger.info("📦 빈 그래프 감지, 기본 온톨로지 데이터 생성 중...")
                asyncio.create_task(self._create_default_ontology_data())
                # 기본 데이터 생성 후 다시 확인
                current_nodes = self.knowledge_graph.graph.number_of_nodes()
            
            logger.info(f"📊 현재 그래프 노드 수: {current_nodes}")
            
            # 새로운 지식 그래프 엔진에서 직접 풍부한 시각화 데이터 생성
            knowledge_graph_visualization = self.knowledge_graph.generate_visualization(max_nodes=max_nodes)
            
            # 그래프가 여전히 비어있으면 하드코딩된 시각화 데이터 반환
            if not knowledge_graph_visualization.get("nodes") and not knowledge_graph_visualization.get("edges"):
                logger.warning("⚠️ 시각화 데이터가 비어있음, 하드코딩된 데이터 반환")
                return self._create_hardcoded_visualization_data()
            
            logger.info(f"✅ 풍부한 온톨로지 그래프 시각화 완료")
            return knowledge_graph_visualization
            
        except Exception as e:
            logger.error(f"지식 그래프 시각화 생성 실패: {e}")
            return self._create_fallback_visualization(str(e))
    
    def _create_rich_ontology_visualization(self, base_data: Dict[str, Any], max_nodes: int) -> Dict[str, Any]:
        """첫 번째 이미지 스타일의 풍부한 온톨로지 시각화 생성 (동적 데이터 기반)"""
        try:
            # 기본 데이터에서 실제 노드와 엣지 추출
            base_nodes = base_data.get("nodes", [])
            base_edges = base_data.get("edges", [])
            
            # 실제 워크플로우 ID 및 에이전트 정보 추출
            workflow_nodes = [node for node in base_nodes if node.get("type") == "workflow"]
            agent_nodes = [node for node in base_nodes if node.get("type") == "agent"]
            task_nodes = [node for node in base_nodes if node.get("type") == "task"]
            
            # 중앙 워크플로우 전략 ID 생성 (실제 데이터 기반)
            if workflow_nodes:
                main_workflow_id = workflow_nodes[0]["id"]
                workflow_strategy = workflow_nodes[0].get("attributes", {}).get("optimization_strategy", "resource_efficiency")
            else:
                main_workflow_id = "workflow_llm_owp_727c54cc5c4243988a66c0b15fd4ccdf"
                workflow_strategy = "resource_efficiency"
            
            # Edges 배열 생성 (실제 데이터 기반)
            edges = []
            edge_id = 0
            
            # 실제 에이전트들과 워크플로우 연결
            for i, agent_node in enumerate(agent_nodes[:9]):  # 최대 9개 에이전트
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": main_workflow_id,
                    "target": agent_node["id"],
                    "color": "#5fd2c9" if i % 2 == 0 else "#fd79a8",
                    "label": f"에이전트 연결",
                    "type": "Task",
                    "size": 12,
                    "weight": 1,
                    "properties": {
                        "domain": agent_node.get("attributes", {}).get("domain", "general"),
                        "complexity": "medium",
                        "query_related": True,
                        "selected_agent": False,
                        "shape": "circle",
                        "size": 12,
                        "special_type": "standard",
                        "type": "Task",
                        "x": 50 + (i % 5) * 100,
                        "y": 50 + (i // 5) * 100,
                        "agent_type": agent_node.get("attributes", {}).get("agent_id", "unknown"),
                        "execution_time": agent_node.get("attributes", {}).get("execution_time", 0),
                        "confidence": agent_node.get("attributes", {}).get("confidence", 0.8)
                    },
                    "relevance_score": agent_node.get("confidence", 0.8),
                    "confidence": agent_node.get("confidence", 0.8)
                }
                edges.append(edge)
                edge_id += 1
            
            # 실제 태스크들과 워크플로우 연결
            for i, task_node in enumerate(task_nodes[:9]):  # 최대 9개 태스크
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": main_workflow_id,
                    "target": task_node["id"],
                    "color": "#74b9ff",
                    "label": f"태스크 연결",
                    "type": "TaskExecution",
                    "size": 10,
                    "weight": 0.9,
                    "properties": {
                        "task_description": task_node.get("attributes", {}).get("task_description", ""),
                        "complexity": task_node.get("attributes", {}).get("complexity_level", "medium"),
                        "query_related": True,
                        "task_type": "workflow_task"
                    },
                    "relevance_score": 0.9
                }
                edges.append(edge)
                edge_id += 1
            
            # 실제 기본 엣지들을 변환하여 추가
            for base_edge in base_edges[:10]:  # 최대 10개 추가 엣지
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": base_edge["source"],
                    "target": base_edge["target"],
                    "color": base_edge.get("color", "#96CEB4"),
                    "label": base_edge.get("label", "관계"),
                    "type": base_edge.get("metadata", {}).get("relationship_type", "relation"),
                    "size": int(base_edge.get("weight", 1) * 8),
                    "weight": base_edge.get("weight", 1),
                    "properties": {
                        "relationship_type": base_edge.get("metadata", {}).get("relationship_type", "general"),
                        "strength": base_edge.get("weight", 1),
                        "attributes": base_edge.get("attributes", {})
                    },
                    "relevance_score": base_edge.get("weight", 0.7)
                }
                edges.append(edge)
                edge_id += 1
            
            # 에이전트 간 협력 관계 추가 (실제 데이터 기반)
            collaboration_pairs = []
            for i in range(min(len(agent_nodes), 8)):
                for j in range(i+1, min(len(agent_nodes), i+3)):  # 각 에이전트가 최대 2개와 협력
                    if j < len(agent_nodes):
                        collaboration_pairs.append((agent_nodes[i]["id"], agent_nodes[j]["id"]))
            
            collaboration_types = ["collaboration", "data_flow", "parallel", "aggregation", "finalization"]
            for i, (source, target) in enumerate(collaboration_pairs[:5]):
                collab_type = collaboration_types[i % len(collaboration_types)]
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": source,
                    "target": target,
                    "color": "#96CEB4",
                    "label": f"{collab_type} 관계",
                    "type": collab_type,
                    "size": 8,
                    "weight": 0.7,
                    "properties": {
                        "collaboration_type": collab_type,
                        "strength": "medium",
                        "bidirectional": True
                    },
                    "relevance_score": 0.7
                }
                edges.append(edge)
                edge_id += 1
            
            # 실제 데이터를 기반으로 한 통계 계산
            actual_edge_types = {}
            for edge in edges:
                edge_type = edge.get("type", "unknown")
                actual_edge_types[edge_type] = actual_edge_types.get(edge_type, 0) + 1
            
            actual_node_types = {}
            for node in base_nodes:
                node_type = node.get("type", "unknown")
                actual_node_types[node_type] = actual_node_types.get(node_type, 0) + 1
            
            # Metadata 생성 (실제 데이터 반영)
            metadata = {
                "description": f"{workflow_strategy} workflow 온톨로지 지식 그래프",
                "edge_types": actual_edge_types,
                "graph_type": "query_focused_workflow",
                "layer_suggestions": {
                    "layer_0": [main_workflow_id, "workflow_strategy"],
                    "layer_1": [node["id"] for node in agent_nodes[:3]],
                    "layer_2": [node["id"] for node in agent_nodes[3:6]], 
                    "layer_3": [node["id"] for node in agent_nodes[6:9]]
                },
                "node_types": actual_node_types,
                "parallelWith": len([e for e in edges if e.get("type") == "parallel"]),
                "partOf": len([e for e in edges if "part" in e.get("type", "").lower()]),
                "requires": len([e for e in edges if "require" in e.get("type", "").lower()]),
                "relevance_states": {
                    "avg_node_relevance": sum(e.get("relevance_score", 0.5) for e in edges) / len(edges) if edges else 0.5,
                    "task_nodes": len(task_nodes),
                    "total_edges": len(edges),
                    "total_nodes": len(base_nodes),
                    "workflow_coverage": (len(workflow_nodes) / len(base_nodes) * 100) if base_nodes else 0,
                    "workflow_nodes": len(workflow_nodes)
                },
                "selected_agents": [node["id"] for node in agent_nodes if node.get("attributes", {}).get("success", True)],
                "styling": {
                    "node_colors": {
                        "agent": "#fd79a8",
                        "workflow": "#fdcb6e", 
                        "task": "#74b9ff",
                        "capability": "#00cec9",
                        "domain": "#a29bfe",
                        "query": "#e17055",
                        "result": "#00b894"
                    },
                    "edge_colors": {
                        "Task": "#5fd2c9",
                        "TaskExecution": "#74b9ff",
                        "collaboration": "#96CEB4",
                        "parallelWith": "#fd79a8",
                        "data_flow": "#00cec9",
                        "aggregation": "#a29bfe"
                    },
                    "node_size_range": {"min": 5, "max": 20},
                    "edge_size_range": {"min": 1, "max": 15}
                },
                "generated_at": datetime.now().isoformat(),
                "graph_type": "ontology_hierarchical", 
                "layout": "force",
                "rendering_hints": {
                    "focus_nodes": [main_workflow_id],
                    "highlight_paths": True,
                    "interactive_zoom": True,
                    "node_labels": True
                },
                "real_data_integration": {
                    "agents_integrated": len(agent_nodes),
                    "tasks_integrated": len(task_nodes),
                    "workflows_integrated": len(workflow_nodes),
                    "total_base_nodes": len(base_nodes),
                    "total_base_edges": len(base_edges)
                }
            }
            
            # Workflow Stats 생성 (실제 데이터 기반)
            workflow_stats = {
                "avg_workflow_relevance": metadata["relevance_states"]["avg_node_relevance"],
                "task_nodes": len(task_nodes),
                "total_workflow_related": len(workflow_nodes) + len(task_nodes),
                "workflow_coverage": metadata["relevance_states"]["workflow_coverage"],
                "workflow_nodes": len(workflow_nodes),
                "agent_performance": {
                    "total_agents": len(agent_nodes),
                    "successful_agents": len([n for n in agent_nodes if n.get("attributes", {}).get("success", True)]),
                    "avg_confidence": sum(n.get("confidence", 0.5) for n in agent_nodes) / len(agent_nodes) if agent_nodes else 0.5,
                    "avg_execution_time": sum(n.get("attributes", {}).get("execution_time", 0) for n in agent_nodes) / len(agent_nodes) if agent_nodes else 0
                }
            }
            
            # 최종 구조 (첫 번째 이미지와 동일하되 실제 데이터 반영)
            knowledge_graph_visualization = {
                "edges": edges,
                "metadata": metadata,
                "workflow_stats": workflow_stats,
                "average_degree": (len(edges) * 2) / len(base_nodes) if base_nodes else 0,
                "is_connected": len(base_nodes) > 0 and len(edges) >= len(base_nodes) - 1,
                "node_types": actual_node_types,
                "total_concepts": len(base_nodes),
                "total_relations": len(edges)
            }
            
            logger.info(f"🎨 동적 온톨로지 시각화 생성: {len(edges)}개 엣지, {len(base_nodes)}개 노드 기반")
            logger.info(f"📊 에이전트: {len(agent_nodes)}개, 태스크: {len(task_nodes)}개, 워크플로우: {len(workflow_nodes)}개")
            
            return knowledge_graph_visualization
            
        except Exception as e:
            logger.error(f"풍부한 온톨로지 시각화 생성 실패: {e}")
            return self._create_fallback_visualization(str(e))
    
    async def _create_default_ontology_data(self):
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

    def _create_hardcoded_visualization_data(self) -> Dict[str, Any]:
        """하드코딩된 시각화 데이터 생성 (폴백용)"""
        logger.info("🎨 하드코딩된 시각화 데이터 생성")
        
        # 샘플 노드들
        nodes = [
            {
                "id": "workflow_main",
                "label": "🔄 메인 워크플로우",
                "type": "workflow",
                "size": 25,
                "color": "#fdcb6e",
                "special_type": "workflow",
                "properties": {"workflow_type": "main", "optimization_strategy": "balanced"},
                "confidence": 0.9,
                "relevance_score": 1.0,
                "attributes": {"domain": "system", "complexity": "medium"}
            },
            {
                "id": "agent_internet",
                "label": "🌐 인터넷 에이전트",
                "type": "agent",
                "size": 20,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "internet_agent", "capabilities": ["search", "web"]},
                "confidence": 0.85,
                "relevance_score": 0.9,
                "attributes": {"domain": "information", "execution_time": 2.5}
            },
            {
                "id": "agent_calculator",
                "label": "🔢 계산 에이전트",
                "type": "agent",
                "size": 18,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "calculator_agent", "capabilities": ["math", "calculation"]},
                "confidence": 0.88,
                "relevance_score": 0.8,
                "attributes": {"domain": "calculation", "execution_time": 1.2}
            },
            {
                "id": "agent_analysis",
                "label": "📊 분석 에이전트",
                "type": "agent",
                "size": 19,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "analysis_agent", "capabilities": ["analysis", "insights"]},
                "confidence": 0.82,
                "relevance_score": 0.85,
                "attributes": {"domain": "analysis", "execution_time": 3.1}
            },
            {
                "id": "domain_information",
                "label": "📚 정보 도메인",
                "type": "domain",
                "size": 16,
                "color": "#a29bfe",
                "special_type": "domain",
                "properties": {"domain_name": "information"},
                "confidence": 0.9,
                "relevance_score": 0.7,
                "attributes": {"domain_type": "knowledge"}
            },
            {
                "id": "capability_search",
                "label": "🔍 검색 능력",
                "type": "capability",
                "size": 14,
                "color": "#00cec9",
                "special_type": "capability",
                "properties": {"capability_name": "search"},
                "confidence": 0.85,
                "relevance_score": 0.75,
                "attributes": {"capability_type": "core"}
            },
            {
                "id": "task_data_collection",
                "label": "📋 데이터 수집",
                "type": "task",
                "size": 17,
                "color": "#74b9ff",
                "special_type": "task",
                "properties": {"task_type": "data_collection"},
                "confidence": 0.87,
                "relevance_score": 0.8,
                "attributes": {"complexity": "medium"}
            },
            {
                "id": "result_analysis",
                "label": "📈 분석 결과",
                "type": "result",
                "size": 15,
                "color": "#00b894",
                "special_type": "result",
                "properties": {"result_type": "analysis"},
                "confidence": 0.83,
                "relevance_score": 0.78,
                "attributes": {"quality": "high"}
            }
        ]
        
        # 샘플 엣지들
        edges = [
            {
                "id": "edge_1",
                "source": "workflow_main",
                "target": "agent_internet",
                "label": "실행",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#5fd2c9",
                "size": 3,
                "weight": 1.0,
                "confidence": 0.9,
                "properties": {"execution_order": 1},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_2",
                "source": "workflow_main",
                "target": "agent_calculator",
                "label": "실행",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#74b9ff",
                "size": 3,
                "weight": 0.9,
                "confidence": 0.85,
                "properties": {"execution_order": 2},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_3",
                "source": "workflow_main",
                "target": "agent_analysis",
                "label": "실행",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#fd79a8",
                "size": 3,
                "weight": 0.85,
                "confidence": 0.87,
                "properties": {"execution_order": 3},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_4",
                "source": "agent_internet",
                "target": "domain_information",
                "label": "전문화",
                "type": "specialization",
                "relationship_type": "specializes_in",
                "color": "#96CEB4",
                "size": 2,
                "weight": 0.8,
                "confidence": 0.9,
                "properties": {"specialization_level": "high"},
                "attributes": {"bidirectional": False, "strength": "medium"}
            },
            {
                "id": "edge_5",
                "source": "agent_internet",
                "target": "capability_search",
                "label": "능력",
                "type": "capability",
                "relationship_type": "has_capability",
                "color": "#00cec9",
                "size": 2,
                "weight": 0.9,
                "confidence": 0.88,
                "properties": {"proficiency": "expert"},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_6",
                "source": "agent_internet",
                "target": "task_data_collection",
                "label": "생성",
                "type": "production",
                "relationship_type": "produces",
                "color": "#e17055",
                "size": 2,
                "weight": 0.75,
                "confidence": 0.82,
                "properties": {"output_quality": "high"},
                "attributes": {"bidirectional": False, "strength": "medium"}
            },
            {
                "id": "edge_7",
                "source": "agent_analysis",
                "target": "result_analysis",
                "label": "생성",
                "type": "production",
                "relationship_type": "produces",
                "color": "#00b894",
                "size": 2,
                "weight": 0.88,
                "confidence": 0.85,
                "properties": {"output_quality": "high"},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_8",
                "source": "agent_internet",
                "target": "agent_analysis",
                "label": "협력",
                "type": "collaboration",
                "relationship_type": "collaborated_with",
                "color": "#a29bfe",
                "size": 2,
                "weight": 0.7,
                "confidence": 0.8,
                "properties": {"collaboration_type": "sequential"},
                "attributes": {"bidirectional": True, "strength": "medium"}
            }
        ]
        
        # 메타데이터
        metadata = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {
                "workflow": 1,
                "agent": 3,
                "domain": 1,
                "capability": 1,
                "task": 1,
                "result": 1
            },
            "edge_types": {
                "execution": 3,
                "specialization": 1,
                "capability": 1,
                "production": 2,
                "collaboration": 1
            },
            "graph_metrics": {
                "density": 0.25,
                "average_degree": 2.0,
                "clustering_coefficient": 0.15,
                "connected_components": 1
            },
            "semantic_layers": {
                "workflow_layer": ["workflow_main"],
                "agent_layer": ["agent_internet", "agent_calculator", "agent_analysis"],
                "domain_layer": ["domain_information"],
                "capability_layer": ["capability_search"],
                "task_layer": ["task_data_collection"]
            },
            "layout_suggestions": {
                "recommended": "hierarchical",
                "type_centers": {
                    "workflow": (0.5, 0.1),
                    "agent": (0.3, 0.5),
                    "domain": (0.7, 0.3),
                    "capability": (0.7, 0.7),
                    "task": (0.3, 0.8)
                },
                "force_settings": {
                    "node_repulsion": 800,
                    "link_strength": 0.6,
                    "charge_strength": -250
                }
            },
            "styling": {
                "node_colors": {
                    "workflow": "#fdcb6e",
                    "agent": "#fd79a8",
                    "domain": "#a29bfe",
                    "capability": "#00cec9",
                    "task": "#74b9ff",
                    "result": "#00b894"
                },
                "edge_colors": {
                    "execution": "#5fd2c9",
                    "specialization": "#96CEB4",
                    "capability": "#00cec9",
                    "production": "#e17055",
                    "collaboration": "#a29bfe"
                },
                "node_size_range": {"min": 10, "max": 30},
                "edge_size_range": {"min": 1, "max": 5}
            },
            "generated_at": datetime.now().isoformat(),
            "version": "2.0",
            "graph_type": "hardcoded_demo",
            "description": "온톨로지 시스템 데모 그래프"
        }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata
        }

    def _create_fallback_visualization(self, error_message: str) -> Dict[str, Any]:
        """폴백 시각화 생성"""
        return {
            "nodes": [
                {
                    "id": "error_node",
                    "label": "⚠️ 오류 발생",
                    "type": "error",
                    "size": 20,
                    "color": "#E74C3C",
                    "properties": {"error_message": error_message}
                }
            ],
            "edges": [],
            "metadata": {
                "description": f"온톨로지 그래프 생성 오류: {error_message}",
                "error": True,
                "generated_at": datetime.now().isoformat(),
                "graph_type": "error_fallback",
                "total_nodes": 1,
                "total_edges": 0
            }
        }
    
    async def close(self):
        """시스템 종료"""
        try:
            logger.info("🔄 온톨로지 시스템 종료 중...")
            
            # 캐시 정리
            await self.semantic_query_manager.invalidate_cache()
            
            # 실행 기록 정리 (필요한 경우)
            self.execution_history.clear()
            
            logger.info("✅ 온톨로지 시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"❌ 시스템 종료 실패: {e}")
            raise
    
    def _extract_available_agents(self, execution_context: ExecutionContext = None) -> List[str]:
        """실행 컨텍스트에서 사용 가능한 에이전트 목록 추출"""
        try:
            if execution_context and hasattr(execution_context, 'custom_config'):
                installed_agents = execution_context.custom_config.get('installed_agents', [])
                
                if installed_agents:
                    # TaskAgent에서 전달된 설치된 에이전트 정보를 agent_id 목록으로 변환
                    agent_ids = []
                    for agent_info in installed_agents:
                        if isinstance(agent_info, dict):
                            agent_id = agent_info.get('agent_id')
                            if agent_id:
                                agent_ids.append(agent_id)
                                logger.debug(f"설치된 에이전트 추가: {agent_id}")
                    
                    if agent_ids:
                        logger.info(f"🎯 사용자별 설치된 에이전트 {len(agent_ids)}개 발견: {agent_ids}")
                        return agent_ids
                    else:
                        logger.warning("설치된 에이전트 정보가 있지만 agent_id를 추출할 수 없습니다.")
                else:
                    logger.warning("execution_context에 installed_agents 정보가 없습니다.")
            else:
                logger.warning("execution_context 또는 custom_config가 없습니다.")
            
            # 폴백: 기본 에이전트 목록 사용
            logger.info("🔄 기본 에이전트 목록으로 폴백")
            return self.available_agents
            
        except Exception as e:
            logger.error(f"설치된 에이전트 추출 실패: {e}")
            # 폴백: 기본 에이전트 목록 사용
            return self.available_agents
    
    def _extract_installed_agents_info(self, execution_context: ExecutionContext = None) -> List[Dict[str, Any]]:
        """실행 컨텍스트에서 설치된 에이전트 정보 추출"""
        try:
            if execution_context and hasattr(execution_context, 'custom_config'):
                installed_agents = execution_context.custom_config.get('installed_agents', [])
                
                if installed_agents:
                    logger.info(f"🎯 설치된 에이전트 정보 추출: {len(installed_agents)}개")
                    
                    # 각 에이전트 정보 상세 로깅
                    for i, agent_info in enumerate(installed_agents):
                        agent_id = agent_info.get('agent_id', 'Unknown')
                        agent_data = agent_info.get('agent_data', {})
                        agent_name = agent_data.get('name', agent_id)
                        agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
                        capabilities_count = len(agent_data.get('capabilities', []))
                        tags = agent_data.get('tags', [])
                        
                        logger.info(f"  📋 에이전트 {i+1}: {agent_id}")
                        logger.info(f"    - 이름: {agent_name}")
                        logger.info(f"    - 타입: {agent_type}")
                        logger.info(f"    - 능력 수: {capabilities_count}개")
                        logger.info(f"    - 태그: {tags[:3]}{'...' if len(tags) > 3 else ''}")  # 처음 3개만 표시
                    
                    return installed_agents
                else:
                    logger.warning("execution_context에 installed_agents 정보가 없습니다.")
            else:
                logger.warning("execution_context 또는 custom_config가 없습니다.")
            
            return []
            
        except Exception as e:
            logger.error(f"설치된 에이전트 정보 추출 실패: {e}")
            return [] 