"""
🔧 Workflow Utils
워크플로우 유틸리티

워크플로우 생성, 변환 및 분석 관련 유틸리티
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple
from loguru import logger

from ..core.models import (
    SemanticQuery, ExecutionContext, WorkflowPlan, WorkflowStep, 
    OptimizationStrategy, WorkflowComplexity, ExecutionStrategy
)
from .strategy_manager import StrategyManager
from ..engines.workflow_designer import SmartWorkflowDesigner


class WorkflowUtils:
    """🔧 워크플로우 유틸리티"""
    
    def __init__(self, installed_agents_info: List[Dict[str, Any]] = None):
        self.strategy_manager = StrategyManager()
        self.installed_agents_info = installed_agents_info or []
        self.workflow_designer = None
        self._initialize_workflow_designer()
    
    def _initialize_workflow_designer(self):
        """워크플로우 설계자 초기화"""
        try:
            self.workflow_designer = SmartWorkflowDesigner(self.installed_agents_info)
            logger.info(f"🎯 워크플로우 설계자 초기화 완료 - 에이전트 정보: {len(self.installed_agents_info)}개")
        except Exception as e:
            logger.error(f"워크플로우 설계자 초기화 실패: {e}")
            self.workflow_designer = SmartWorkflowDesigner()
    
    def convert_unified_result_to_workflow(self, 
                                         unified_result: Dict[str, Any], 
                                         query_text: str, 
                                         execution_context: ExecutionContext) -> WorkflowPlan:
        """통합 결과를 WorkflowPlan으로 변환"""
        try:
            # 통합 결과에서 워크플로우 정보 추출
            workflow_info = unified_result.get('workflow_design', {})
            
            if not workflow_info.get('steps'):
                logger.warning("통합 결과에 워크플로우 정보가 없음. 최소 워크플로우 생성")
                return self.create_minimal_workflow(query_text, execution_context)
            
            # SemanticQuery 생성 (통합 결과에서)
            semantic_info = unified_result.get('semantic_analysis', {})
            semantic_query = SemanticQuery(
                query_id=f'semantic_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent=semantic_info.get('intent', 'information_retrieval'),
                entities=semantic_info.get('entities', []),
                concepts=semantic_info.get('concepts', []),
                relations=semantic_info.get('relations', []),
                complexity_score=semantic_info.get('complexity_score', 0.7),
                created_at=datetime.now(),
                metadata=semantic_info.get('metadata', {})
            )
            
            # 워크플로우 단계들 변환
            steps = []
            for step_info in workflow_info.get('steps', []):
                steps.append({
                    'agent_id': step_info.get('agent_id', 'unknown_agent'),
                    'task': step_info.get('task', 'Execute task'),
                    'expected_output': step_info.get('expected_output', 'Task result'),
                    'dependencies': step_info.get('dependencies', []),
                    'priority': step_info.get('priority', 1),
                    'timeout': step_info.get('timeout', 30),
                    'metadata': step_info.get('metadata', {})
                })
            
            # 실행 전략 결정
            strategy_name = workflow_info.get('execution_strategy', 'parallel')
            try:
                execution_strategy = ExecutionStrategy(strategy_name)
            except ValueError:
                execution_strategy = ExecutionStrategy.PARALLEL
            
            # WorkflowPlan 생성
            workflow_plan = WorkflowPlan(
                workflow_id=f'workflow_{uuid.uuid4().hex[:8]}',
                query=semantic_query,
                steps=steps,
                optimization_strategy=execution_strategy,
                estimated_time=workflow_info.get('estimated_time', 30),
                complexity_score=semantic_info.get('complexity_score', 0.7),
                confidence_score=unified_result.get('quality_assessment', {}).get('overall_confidence', 0.8),
                metadata={
                    'unified_processing': True,
                    'workflow_type': workflow_info.get('workflow_type', 'standard'),
                    'reasoning': unified_result.get('strategic_reasoning', '')
                }
            )
            
            logger.info(f"✅ 통합 결과를 워크플로우로 변환 완료: {len(steps)}개 단계")
            return workflow_plan
            
        except Exception as e:
            logger.error(f"❌ 워크플로우 변환 실패: {e}")
            return self.create_minimal_workflow(query_text, execution_context)

    def create_minimal_workflow(self, query_text: str, execution_context: ExecutionContext) -> WorkflowPlan:
        """최소한의 워크플로우 생성 (폴백)"""
        # 사용 가능한 첫 번째 에이전트 선택
        available_agents = list(execution_context.available_agents.keys()) if execution_context.available_agents else ['memo_agent']
        selected_agent = available_agents[0] if available_agents else 'memo_agent'
        
        step = WorkflowStep(
            step_id="minimal_step",
            agent_id=selected_agent,
            semantic_purpose=query_text,
            required_concepts=[],  # 필수 파라미터
            depends_on=[],
            estimated_time=30.0,
            estimated_complexity=WorkflowComplexity.MODERATE,
            execution_context={"fallback_mode": True, "timeout": 120, "retry_count": 2}
        )
        
        workflow_plan = WorkflowPlan(
            plan_id=f"minimal_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query=SemanticQuery(
                query_id=f'minimal_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent='information_retrieval',
                entities=[],
                concepts=[],
                relations=[],
                complexity_score=0.5,
                created_at=datetime.now(),
                metadata={"fallback_mode": True}
            ),
            steps=[step],
            optimization_strategy=OptimizationStrategy.SPEED_FIRST,  # 단일 에이전트는 속도 우선
            estimated_time=30,
            estimated_quality=0.7,
            dependencies={},
            reasoning_chain=["최소한의 폴백 워크플로우", "단일 에이전트 사용", "신속 처리 우선"]
        )
        
        logger.warning("🔄 최소한의 폴백 워크플로우 생성")
        return workflow_plan

    def safe_analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """안전한 복잡도 분석"""
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
    
    def generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
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
                
                # 노드 정의 - 텍스트를 쌍따옴표로 감싸기
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

    def extract_available_agents(self, execution_context: ExecutionContext = None) -> List[str]:
        """사용 가능한 에이전트 목록 추출"""
        try:
            if not execution_context or not execution_context.available_agents:
                # 기본 에이전트 목록 반환
                return [
                    "internet_agent", "finance_agent", "weather_agent",
                    "calculate_agent", "chart_agent", "memo_agent", "analysis_agent"
                ]
            
            available_agents = []
            for agent_id, agent_info in execution_context.available_agents.items():
                if isinstance(agent_info, dict) and agent_info.get('available', True):
                    available_agents.append(agent_id)
                elif hasattr(agent_info, 'agent_id'):
                    available_agents.append(agent_info.agent_id)
                else:
                    available_agents.append(agent_id)
            
            logger.info(f"🤖 사용 가능한 에이전트 추출: {len(available_agents)}개")
            return available_agents
            
        except Exception as e:
            logger.error(f"에이전트 목록 추출 실패: {e}")
            return ["memo_agent"]  # 안전한 폴백
    
    def extract_installed_agents_info(self, execution_context: ExecutionContext = None) -> List[Dict[str, Any]]:
        """설치된 에이전트 정보 추출"""
        try:
            if not execution_context or not execution_context.available_agents:
                logger.warning("ExecutionContext 또는 available_agents가 없음")
                return []
            
            installed_agents_info = []
            
            for agent_id, agent_info in execution_context.available_agents.items():
                try:
                    if isinstance(agent_info, dict):
                        # 딕셔너리 형태의 에이전트 정보
                        agent_data = {
                            "agent_id": agent_id,
                            "name": agent_info.get('name', agent_id),
                            "description": agent_info.get('description', f'{agent_id} 에이전트'),
                            "capabilities": agent_info.get('capabilities', []),
                            "available": agent_info.get('available', True),
                            "metadata": agent_info.get('metadata', {})
                        }
                    elif hasattr(agent_info, '__dict__'):
                        # 객체 형태의 에이전트 정보
                        agent_data = {
                            "agent_id": getattr(agent_info, 'agent_id', agent_id),
                            "name": getattr(agent_info, 'name', agent_id),
                            "description": getattr(agent_info, 'description', f'{agent_id} 에이전트'),
                            "capabilities": getattr(agent_info, 'capabilities', []),
                            "available": getattr(agent_info, 'available', True),
                            "metadata": getattr(agent_info, 'metadata', {})
                        }
                    else:
                        # 기본 정보만 있는 경우
                        agent_data = {
                            "agent_id": agent_id,
                            "name": agent_id,
                            "description": f'{agent_id} 에이전트',
                            "capabilities": [],
                            "available": True,
                            "metadata": {}
                        }
                    
                    installed_agents_info.append(agent_data)
                    
                except Exception as e:
                    logger.warning(f"에이전트 {agent_id} 정보 처리 실패: {e}")
                    # 최소 정보라도 포함
                    installed_agents_info.append({
                        "agent_id": agent_id,
                        "name": agent_id,
                        "description": f'{agent_id} 에이전트',
                        "capabilities": [],
                        "available": True,
                        "metadata": {}
                    })
            
            logger.info(f"📊 설치된 에이전트 정보 추출 완료: {len(installed_agents_info)}개")
            return installed_agents_info
            
        except Exception as e:
            logger.error(f"설치된 에이전트 정보 추출 실패: {e}")
            return []
    
    def update_installed_agents_info(self, installed_agents_info: List[Dict[str, Any]]):
        """설치된 에이전트 정보 업데이트"""
        self.installed_agents_info = installed_agents_info
        self._initialize_workflow_designer()
        logger.info(f"🔄 에이전트 정보 업데이트: {len(installed_agents_info)}개") 