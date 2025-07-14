"""
🧠 Clean Ontology System
간소화된 온톨로지 시스템

분할된 모듈들을 활용한 깔끔한 통합 시스템
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, ExecutionContext, AgentExecutionResult, 
    WorkflowPlan, get_system_metrics
)
from ..core.interfaces import ProgressCallback
from ..engines.execution_engine import AdvancedExecutionEngine
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine

# 새로운 분할된 모듈들 import
from .query_processing import QueryProcessor
from .result_integration import ResultIntegrator
from .knowledge_management import KnowledgeGraphManager
from .metrics_manager import MetricsManager


class SimpleProgressCallback(ProgressCallback):
    """간단한 진행 상황 콜백"""
    
    def __init__(self):
        self.messages = []
        self.current_progress = 0.0
        self.completed_steps = []
        self.errors = []
    
    async def on_progress(self, message: str, progress: float, metadata: Dict[str, Any] = None):
        """진행 상황 업데이트"""
        self.messages.append({
            "message": message,
            "progress": progress,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        self.current_progress = progress
        logger.info(f"Progress ({progress:.1%}): {message}")
    
    async def on_step_complete(self, step_id: str, result: AgentExecutionResult):
        """단계 완료 알림"""
        self.completed_steps.append({
            "step_id": step_id,
            "agent_id": result.agent_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })
        logger.info(f"Step completed: {step_id} ({result.agent_id}) - {'✅' if result.success else '❌'}")
    
    async def on_error(self, error_message: str, error_details: Dict[str, Any] = None):
        """오류 발생 알림"""
        self.errors.append({
            "message": error_message,
            "details": error_details or {},
            "timestamp": time.time()
        })
        logger.error(f"Error: {error_message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """진행 상황 요약"""
        return {
            "current_progress": self.current_progress,
            "total_messages": len(self.messages),
            "completed_steps": len(self.completed_steps),
            "errors": len(self.errors),
            "success_rate": (
                sum(1 for step in self.completed_steps if step["success"]) / 
                len(self.completed_steps) if self.completed_steps else 0
            )
        }


class CleanOntologySystem:
    """🧠 깔끔한 온톨로지 시스템 - 분할된 모듈 활용"""
    
    def __init__(self, 
                 email: str = "system@ontology.ai",
                 session_id: str = None,
                 project_id: str = None):
        
        # 세션 정보
        self.email = email
        self.session_id = session_id or f"session_{int(time.time())}"
        self.project_id = project_id or "default_project"
        
        # 핵심 엔진들 초기화
        self.execution_engine = AdvancedExecutionEngine()
        self.knowledge_graph = KnowledgeGraphEngine()
        
        # 분할된 모듈들 초기화
        self.query_processor = QueryProcessor()
        self.result_integrator = ResultIntegrator()
        self.knowledge_manager = KnowledgeGraphManager(self.knowledge_graph)
        self.metrics_manager = MetricsManager()
        
        # 시스템 상태
        self.is_initialized = False
        self.execution_history = []
        
        logger.info(f"🧠 깔끔한 온톨로지 시스템 초기화: {self.session_id}")
    
    async def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 깔끔한 온톨로지 시스템 초기화 시작")
            
            # 각 모듈 초기화 (필요한 경우)
            await self.query_processor.initialize()
            
            self.is_initialized = True
            logger.info("✅ 깔끔한 온톨로지 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            raise
    
    async def process_query(self, query_text: str, execution_context: ExecutionContext) -> Dict[str, Any]:
        """쿼리 처리 - 메인 진입점"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 쿼리 처리 시작: '{query_text[:50]}...'")
            
            # 1. 쿼리 처리 및 워크플로우 생성 (QueryProcessor에 위임)
            semantic_query, workflow_plan = await self.query_processor.process_query(
                query_text, execution_context
            )
            
            # 2. 워크플로우 실행
            execution_results = await self._execute_workflow(workflow_plan, execution_context)
            
            # 3. 결과 통합 (ResultIntegrator에 위임)
            integrated_result = await self.result_integrator.integrate_results(
                execution_results, workflow_plan, semantic_query
            )
            
            # 4. 지식 그래프 업데이트 (KnowledgeGraphManager에 위임)
            await self.knowledge_manager.update_knowledge_graph(
                semantic_query, workflow_plan, execution_results, integrated_result
            )
            
            # 5. 메트릭 기록 (MetricsManager에 위임)
            total_execution_time = time.time() - start_time
            self.metrics_manager.record_workflow_execution(
                semantic_query, workflow_plan, execution_results, integrated_result, total_execution_time
            )
            
            # 6. 최종 결과 구성
            final_result = {
                **integrated_result,
                'semantic_query': semantic_query.to_dict(),
                'workflow_plan': self._workflow_plan_to_dict(workflow_plan),
                'execution_results': [r.to_dict() for r in execution_results],
                'total_execution_time': total_execution_time,
                'system_metrics': self.metrics_manager.get_system_status()
            }
            
            # 실행 히스토리에 추가
            self.execution_history.append({
                'query_id': semantic_query.query_id,
                'query_text': query_text,
                'success': integrated_result.get('success', False),
                'execution_time': total_execution_time,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"✅ 쿼리 처리 완료: {total_execution_time:.2f}초")
            return final_result
            
        except Exception as e:
            total_execution_time = time.time() - start_time
            logger.error(f"❌ 쿼리 처리 실패: {e}")
            
            # 오류 메트릭 기록
            get_system_metrics().failed_executions += 1
            
            return {
                'success': False,
                'error': str(e),
                'query_text': query_text,
                'execution_time': total_execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_workflow(self, workflow_plan: WorkflowPlan, execution_context: ExecutionContext) -> List[AgentExecutionResult]:
        """워크플로우 실행"""
        try:
            logger.info(f"⚡ 워크플로우 실행 시작: {len(workflow_plan.steps)}개 단계")
            
            # 진행 상황 콜백 생성
            progress_callback = SimpleProgressCallback()
            
            # 실행 엔진에 워크플로우 실행 위임
            execution_results = await self.execution_engine.execute_workflow(
                workflow_plan=workflow_plan,
                execution_context=execution_context,
                progress_callback=progress_callback
            )
            
            logger.info(f"⚡ 워크플로우 실행 완료: {len(execution_results)}개 결과")
            return execution_results
            
        except Exception as e:
            logger.error(f"워크플로우 실행 실패: {e}")
            # 빈 결과 리스트 반환 (오류 처리를 위해)
            return []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 조회"""
        return self.metrics_manager.get_system_status()
    
    def get_system_performance_report(self) -> str:
        """시스템 성능 보고서 조회"""
        return self.metrics_manager.generate_performance_report()
    
    def get_knowledge_graph_visualization(self, max_nodes: int = 50) -> Dict[str, Any]:
        """지식 그래프 시각화 데이터 조회"""
        try:
            logger.info(f"🎨 지식 그래프 시각화 데이터 생성: 최대 {max_nodes}개 노드")
            
            visualization_data = self.knowledge_graph.generate_visualization(max_nodes)
            
            # 추가 메타데이터
            visualization_data['system_info'] = {
                'session_id': self.session_id,
                'total_queries': len(self.execution_history),
                'generated_at': datetime.now().isoformat()
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"지식 그래프 시각화 실패: {e}")
            return {
                'nodes': [],
                'edges': [],
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """실행 히스토리 조회"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    async def close(self):
        """시스템 종료 및 정리"""
        try:
            logger.info("🔄 온톨로지 시스템 종료 시작")
            
            # 각 모듈 정리 (필요한 경우)
            if hasattr(self.execution_engine, 'close'):
                await self.execution_engine.close()
            
            if hasattr(self.knowledge_graph, 'close'):
                await self.knowledge_graph.close()
            
            # 최종 메트릭 출력
            final_report = self.metrics_manager.generate_performance_report()
            logger.info(f"📊 최종 성능 보고서:\n{final_report}")
            
            logger.info("✅ 온톨로지 시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"시스템 종료 중 오류: {e}")
    
    def _workflow_plan_to_dict(self, workflow_plan: WorkflowPlan) -> Dict[str, Any]:
        """WorkflowPlan을 딕셔너리로 변환"""
        try:
            return {
                'plan_id': workflow_plan.plan_id,
                'estimated_time': workflow_plan.estimated_time,
                'estimated_quality': workflow_plan.estimated_quality,
                'optimization_strategy': getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy)),
                'steps_count': len(workflow_plan.steps),
                'steps': [
                    {
                        'step_id': step.step_id,
                        'agent_id': step.agent_id,
                        'semantic_purpose': step.semantic_purpose,
                        'estimated_complexity': getattr(step.estimated_complexity, 'value', str(step.estimated_complexity)),
                        'estimated_time': step.estimated_time,
                        'depends_on': step.depends_on
                    } for step in workflow_plan.steps
                ],
                'created_at': workflow_plan.created_at.isoformat() if hasattr(workflow_plan.created_at, 'isoformat') else str(workflow_plan.created_at)
            }
        except Exception as e:
            logger.error(f"워크플로우 플랜 변환 실패: {e}")
            return {'error': str(e)}


# 기본 시스템 클래스에 대한 별칭 (하위 호환성)
OntologySystem = CleanOntologySystem 