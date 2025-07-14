"""
🔍 Query Processing Module
쿼리 처리 모듈

쿼리 분석, 복잡도 계산, 워크플로우 생성을 담당
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, ExecutionContext, WorkflowPlan, 
    ExecutionStrategy, ComplexityAnalysis, AgentExecutionResult
)
from ..engines.semantic_query_manager import SemanticQueryManager
from ..engines.workflow_designer import SmartWorkflowDesigner
from ..engines.execution_engine import QueryComplexityAnalyzer


class QueryProcessor:
    """🔍 쿼리 처리기 - 쿼리 분석 및 워크플로우 생성 전담"""
    
    def __init__(self):
        self.semantic_query_manager = SemanticQueryManager()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.workflow_designer = None
        self.installed_agents_info = []
        
        logger.info("🔍 쿼리 처리기 초기화 완료")
    
    def initialize_workflow_designer(self, installed_agents_info: List[Dict[str, Any]] = None):
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
    
    async def process_query_to_workflow(self, 
                                      query_text: str, 
                                      execution_context: ExecutionContext,
                                      available_agents: List[str]) -> tuple[SemanticQuery, Dict[str, Any], WorkflowPlan]:
        """쿼리를 분석해서 워크플로우까지 생성"""
        try:
            logger.info(f"🔍 쿼리 분석 시작: '{query_text[:50]}...'")
            
            # 1. 의미론적 쿼리 생성
            semantic_query = await self.semantic_query_manager.create_semantic_query(
                query_text, execution_context
            )
            logger.info(f"📝 의미론적 쿼리 생성 완료: ID={semantic_query.query_id}")
            
            # 2. 복잡도 분석
            complexity_analysis = self.analyze_complexity(semantic_query)
            logger.info(f"🔍 복잡도 분석 완료: {complexity_analysis.get('recommended_strategy', 'AUTO')} (점수: {complexity_analysis.get('complexity_score', 0):.2f})")
            
            # 3. 워크플로우 설계
            logger.info(f"🤖 사용 가능한 에이전트: {len(available_agents)}개 - {available_agents}")
            
            workflow_plan = await self.workflow_designer.design_workflow(
                semantic_query, available_agents
            )
            
            # 워크플로우 설계 상세 로깅
            self._log_workflow_details(workflow_plan)
            
            return semantic_query, complexity_analysis, workflow_plan
            
        except Exception as e:
            logger.error(f"쿼리 처리 실패: {e}")
            raise
    
    def analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """안전하고 단순한 복잡도 분석"""
        try:
            # SemanticQuery가 딕셔너리인지 객체인지 안전하게 확인
            if isinstance(semantic_query, dict):
                query_text = semantic_query.get('natural_language', '')
                required_agents = semantic_query.get('required_agents', [])
            else:
                query_text = getattr(semantic_query, 'natural_language', '')
                required_agents = getattr(semantic_query, 'required_agents', [])
                if hasattr(semantic_query, 'structured_query'):
                    structured_query = getattr(semantic_query, 'structured_query', {})
                    if isinstance(structured_query, dict) and 'required_agents' in structured_query:
                        required_agents = structured_query['required_agents']
            
            # 기본 복잡도 분석 - 단순화
            analysis = {
                'complexity_score': 0.5,  # 기본값
                'query_type': 'GENERAL',
                'required_agents_count': len(required_agents) if required_agents else 0,
                'estimated_processing_time': 30.0,  # 기본 30초
                'recommended_strategy': ExecutionStrategy.AUTO
            }
            
            # 텍스트 길이 기반 복잡도 추정
            if query_text:
                text_length = len(query_text)
                if text_length > 100:
                    analysis['complexity_score'] = 0.7
                elif text_length > 50:
                    analysis['complexity_score'] = 0.6
                else:
                    analysis['complexity_score'] = 0.4
                
                # 핵심 키워드 기반 복잡도 조정
                complex_keywords = ['분석', '비교', '차트', '그래프', '계산']
                keyword_count = sum(1 for keyword in complex_keywords if keyword in query_text)
                if keyword_count > 0:
                    analysis['complexity_score'] = min(analysis['complexity_score'] + 0.1 * keyword_count, 1.0)
            
            # 에이전트 수에 따른 전략 결정 - 단순화
            agents_count = analysis['required_agents_count']
            if agents_count <= 1:
                analysis['recommended_strategy'] = ExecutionStrategy.SINGLE_AGENT
                analysis['estimated_processing_time'] = 15.0
            elif agents_count == 2:
                analysis['recommended_strategy'] = ExecutionStrategy.PARALLEL
                analysis['estimated_processing_time'] = 20.0
            else:
                analysis['recommended_strategy'] = ExecutionStrategy.HYBRID
                analysis['estimated_processing_time'] = agents_count * 15.0
            
            # 복잡도 점수에 따른 추가 조정
            if analysis['complexity_score'] >= 0.8:
                analysis['recommended_strategy'] = ExecutionStrategy.HYBRID
            elif analysis['complexity_score'] >= 0.6 and agents_count > 1:
                analysis['recommended_strategy'] = ExecutionStrategy.PARALLEL
            
            logger.info(f"🔍 복잡도 분석 완료: 점수={analysis['complexity_score']:.2f}, 전략={analysis['recommended_strategy']}")
            return analysis
            
        except Exception as e:
            logger.error(f"복잡도 분석 실패: {e}")
            # 안전한 폴백 분석
            return {
                'complexity_score': 0.5,
                'query_type': 'GENERAL',
                'required_agents_count': 1,
                'estimated_processing_time': 30.0,
                'recommended_strategy': ExecutionStrategy.SINGLE_AGENT,
                'error': str(e),
                'fallback_mode': True
            }
    
    def _log_workflow_details(self, workflow_plan: WorkflowPlan):
        """워크플로우 상세 정보 로깅"""
        logger.info(f"🔧 워크플로우 설계 완료:")
        logger.info(f"  - 플랜 ID: {workflow_plan.plan_id}")
        logger.info(f"  - 단계 수: {len(workflow_plan.steps)}")
        logger.info(f"  - 최적화 전략: {getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy))}")
        logger.info(f"  - 예상 품질: {workflow_plan.estimated_quality:.2f}")
        logger.info(f"  - 예상 시간: {workflow_plan.estimated_time:.1f}초")
        
        # 각 단계 상세 로깅
        for i, step in enumerate(workflow_plan.steps):
            logger.info(f"    단계 {i+1}: {step.step_id}")
            logger.info(f"      - 에이전트: {step.agent_id}")
            logger.info(f"      - 목적: {step.semantic_purpose}")
            logger.info(f"      - 복잡도: {getattr(step.estimated_complexity, 'value', str(step.estimated_complexity))}")
            logger.info(f"      - 예상 시간: {step.estimated_time:.1f}초")
            if step.depends_on:
                logger.info(f"      - 의존성: {step.depends_on}")
    
    def generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
        """워크플로우 Mermaid 다이어그램 생성"""
        try:
            if not workflow_plan or not workflow_plan.steps:
                return 'graph TD\n    A["빈 워크플로우"] --> B["단계 없음"]'
            
            mermaid_lines = ["graph TD"]
            
            # 시작 노드
            mermaid_lines.append(f'    START["{workflow_plan.query.query_text[:30]}..."]')
            
            # 단계들 정의
            for i, step in enumerate(workflow_plan.steps):
                step_label = f"{step.agent_id}\\n{step.semantic_purpose[:20]}..."
                mermaid_lines.append(f'    {step.step_id}["{step_label}"]')
                
                # 의존성이 없는 첫 단계는 START와 연결
                if not step.depends_on:
                    mermaid_lines.append(f'    START --> {step.step_id}')
            
            # 의존성 관계 추가
            for step in workflow_plan.steps:
                for dep in step.depends_on:
                    mermaid_lines.append(f'    {dep} --> {step.step_id}')
            
            # 마지막 단계들을 END와 연결
            last_steps = []
            for step in workflow_plan.steps:
                # 다른 단계의 의존성에 없는 단계는 마지막 단계
                is_last = True
                for other_step in workflow_plan.steps:
                    if step.step_id in other_step.depends_on:
                        is_last = False
                        break
                if is_last:
                    last_steps.append(step.step_id)
            
            # END 노드 추가
            mermaid_lines.append('    END["완료"]')
            for last_step in last_steps:
                mermaid_lines.append(f'    {last_step} --> END')
            
            return "\n".join(mermaid_lines)
            
        except Exception as e:
            logger.error(f"Mermaid 다이어그램 생성 실패: {e}")
            return f'graph TD\n    A["오류"] --> B["{str(e)[:50]}..."]' 