"""
🧠 Core Data Models
핵심 데이터 모델 정의
중복 호출 방지와 효율적인 데이터 관리를 위한 통일된 데이터 구조

LLM 통합 버전 - 모든 모델이 LLM과 함께 작동
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
import networkx as nx
import asyncio

# LLM 관리자 import (지연 import로 순환 import 방지)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .llm_manager import OntologyLLMManager


class QueryType(Enum):
    """쿼리 유형 분류"""
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"


class ExecutionStrategy(Enum):
    """실행 전략 유형"""
    SINGLE_AGENT = "single_agent"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    AUTO = "auto"


class AgentType(Enum):
    """에이전트 유형"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    GENERAL = "general"


class ExecutionStatus(Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataTransformationType(Enum):
    """데이터 변환 타입"""
    DIRECT_PASS = "direct_pass"
    FORMAT_CONVERSION = "format_conversion"
    DATA_EXTRACTION = "data_extraction"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"


class WorkflowComplexity(Enum):
    """워크플로우 복잡도"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    SOPHISTICATED = "sophisticated"


class OptimizationStrategy(Enum):
    """최적화 전략"""
    SPEED_FIRST = "speed_first"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    RESOURCE_EFFICIENT = "resource_efficient"


class LLMProvider(Enum):
    """LLM 제공업체"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class OntologyLLMType(Enum):
    """온톨로지 LLM 타입 정의"""
    SEMANTIC_ANALYZER = "semantic_analyzer"      # 의미론적 분석 전용
    WORKFLOW_DESIGNER = "workflow_designer"      # 워크플로우 설계 전용
    KNOWLEDGE_REASONER = "knowledge_reasoner"    # 지식 추론 전용
    RESULT_INTEGRATOR = "result_integrator"      # 결과 통합 전용
    QUERY_PROCESSOR = "query_processor"          # 쿼리 처리 전용
    GRAPH_BUILDER = "graph_builder"              # 그래프 구축 전용
    PERFORMANCE_OPTIMIZER = "performance_optimizer"  # 성능 최적화 전용
    CREATIVE_REASONER = "creative_reasoner"      # 창의적 추론 전용


@dataclass
class OntologyLLMConfig:
    """온톨로지 LLM 설정"""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    request_timeout: Optional[float] = None
    max_retries: int = 3
    streaming: bool = False
    
    # 온톨로지 특화 설정
    description: str = ""
    use_case: str = ""
    specialization: str = ""
    reasoning_depth: str = "standard"  # shallow, standard, deep
    creativity_level: str = "balanced"  # conservative, balanced, creative
    precision_level: str = "high"       # low, medium, high
    
    # 캐싱 및 성능 설정
    cache_enabled: bool = True
    cache_ttl_minutes: int = 30
    parallel_safe: bool = True


@dataclass
class SemanticQuery:
    """의미론적 쿼리 모델 - 중복 호출 방지를 위한 고유 식별자 포함"""
    query_text: str
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_type: QueryType = QueryType.SIMPLE
    complexity_score: float = 0.0
    required_agents: List[AgentType] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    # 새로운 속성들 (기존 시스템과의 호환성을 위해)
    intent: str = "information_retrieval"
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    structured_query: Dict[str, Any] = field(default_factory=dict)
    natural_language: str = field(default="")
    
    def __post_init__(self):
        """초기화 후 처리"""
        # natural_language가 비어있으면 query_text로 설정
        if not self.natural_language:
            self.natural_language = self.query_text
    
    def __hash__(self):
        """캐싱을 위한 해시 함수"""
        return hash((self.query_text, tuple(sorted(self.context.items()))))
    
    def get_cache_key(self) -> str:
        """캐시 키 생성"""
        return f"query_{hash(self)}_{getattr(self.query_type, 'value', str(self.query_type))}"
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            'query_text': self.query_text,
            'query_id': self.query_id,
            'query_type': getattr(self.query_type, 'value', str(self.query_type)),
            'complexity_score': self.complexity_score,
            'required_agents': [getattr(agent, "value", str(agent)) for agent in self.required_agents],
            'context': self.context,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'intent': self.intent,
            'entities': self.entities,
            'concepts': self.concepts,
            'relations': self.relations,
            'structured_query': self.structured_query,
            'natural_language': self.natural_language
        }
    
    @classmethod
    def create_from_text(cls, 
                        query_text: str,
                        intent: str = "information_retrieval",
                        entities: List[str] = None,
                        concepts: List[str] = None,
                        relations: List[str] = None,
                        structured_query: Dict[str, Any] = None,
                        **kwargs) -> 'SemanticQuery':
        """텍스트로부터 SemanticQuery 생성"""
        return cls(
            query_text=query_text,
            natural_language=query_text,  # natural_language도 설정
            intent=intent,
            entities=entities or [],
            concepts=concepts or [],
            relations=relations or [],
            structured_query=structured_query or {},
            **kwargs
        )
    
    # LLM 통합 메서드들
    async def analyze_with_llm(self, llm_manager: Optional['OntologyLLMManager'] = None) -> Dict[str, Any]:
        """LLM을 사용한 쿼리 분석"""
        if llm_manager is None:
            from .llm_manager import OntologyLLMManager
            llm_manager = OntologyLLMManager()
        
        # 의미론적 분석 LLM 사용
        analysis_prompt = f"""
        다음 쿼리를 분석해주세요:
        - 쿼리 텍스트: {self.query_text}
        - 의도: {self.intent}
        - 현재 개체들: {self.entities}
        - 현재 개념들: {self.concepts}
        
        분석 결과를 JSON 형식으로 제공해주세요:
        {{
            "enhanced_entities": [...],
            "enhanced_concepts": [...],
            "enhanced_relations": [...],
            "complexity_score": 0.0-1.0,
            "required_agent_types": [...],
            "reasoning": "분석 근거"
        }}
        """
        
        result = await llm_manager.call_llm_async('SEMANTIC_ANALYZER', analysis_prompt)
        
        if result and 'enhanced_entities' in result:
            # 분석 결과로 현재 객체 업데이트
            self.entities.extend([e for e in result['enhanced_entities'] if e not in self.entities])
            self.concepts.extend([c for c in result['enhanced_concepts'] if c not in self.concepts])
            self.relations.extend([r for r in result['enhanced_relations'] if r not in self.relations])
            self.complexity_score = result.get('complexity_score', self.complexity_score)
            
            # 필요한 에이전트 타입들을 AgentType enum으로 변환
            if 'required_agent_types' in result:
                try:
                    self.required_agents = [
                        AgentType(agent_str.lower()) for agent_str in result['required_agent_types']
                        if agent_str.lower() in [e.value for e in AgentType]
                    ]
                except ValueError:
                    pass  # 잘못된 에이전트 타입은 무시
        
        return result or {}
    
    def enhance_with_llm_sync(self, llm_manager: Optional['OntologyLLMManager'] = None) -> 'SemanticQuery':
        """동기식 LLM 개선 (호환성을 위해)"""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프에서는 별도 태스크로 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.analyze_with_llm(llm_manager))
                    future.result(timeout=30)
            else:
                asyncio.run(self.analyze_with_llm(llm_manager))
        except Exception as e:
            # LLM 호출 실패 시에도 기본 객체는 반환
            self.metadata['llm_enhancement_error'] = str(e)
        
        return self
    
    def get_optimized_agent_selection(self, available_agents: List[AgentType] = None) -> List[AgentType]:
        """최적화된 에이전트 선택"""
        if available_agents is None:
            available_agents = list(AgentType)
        
        # 쿼리 복잡도에 따른 에이전트 선택
        if self.complexity_score >= 0.8:
            # 높은 복잡도: 전문 에이전트들 필요
            recommended = [AgentType.ANALYSIS, AgentType.TECHNICAL]
        elif self.complexity_score >= 0.5:
            # 중간 복잡도: 연구 및 분석 에이전트
            recommended = [AgentType.RESEARCH, AgentType.ANALYSIS]
        else:
            # 낮은 복잡도: 일반 에이전트
            recommended = [AgentType.GENERAL]
        
        # 의도에 따른 추가 에이전트
        if self.intent in ['creative_generation', 'brainstorming']:
            recommended.append(AgentType.CREATIVE)
        
        # 사용 가능한 에이전트와 교집합
        return [agent for agent in recommended if agent in available_agents]


@dataclass
class ExecutionContext:
    """실행 컨텍스트 - 실행 환경과 상태 관리"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO
    max_parallel_agents: int = 3
    timeout_seconds: int = 300
    retry_count: int = 3
    cache_enabled: bool = True
    debug_mode: bool = False
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # 호환성을 위한 추가 속성
    user_profile: Dict[str, Any] = field(default_factory=dict)
    available_agents: Dict[str, Any] = field(default_factory=dict)  # 사용 가능한 에이전트들
    
    def __post_init__(self):
        """초기화 후 처리"""
        # user_profile이 비어있고 custom_config에 사용자 정보가 있으면 설정
        if not self.user_profile and self.custom_config:
            user_email = self.custom_config.get('user_email', '')
            if user_email:
                self.user_profile = {
                    'email': user_email,
                    'user_id': self.user_id or user_email.split('@')[0]
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'execution_strategy': getattr(self.execution_strategy, 'value', str(self.execution_strategy)),
            'max_parallel_agents': self.max_parallel_agents,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'cache_enabled': self.cache_enabled,
            'debug_mode': self.debug_mode,
            'custom_config': self.custom_config,
            'user_profile': self.user_profile,
            'available_agents': self.available_agents
        }


@dataclass
class AgentExecutionResult:
    """에이전트 실행 결과 - 통일된 결과 형식"""
    result_data: Any
    execution_time: float
    status: ExecutionStatus
    agent_type: AgentType = AgentType.GENERAL  # 기본값 설정
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # 추가 속성들 (호환성을 위해)
    agent_id: str = ""
    data: Any = None
    confidence: float = 0.8
    success: bool = field(default=False)
    
    def __post_init__(self):
        """초기화 후 처리"""
        # agent_id가 비어있으면 agent_type으로 설정
        if not self.agent_id:
            self.agent_id = getattr(self.agent_type, 'value', str(self.agent_type))
        
        # data가 None이면 result_data로 설정
        if self.data is None:
            self.data = self.result_data
        
        # success 상태 설정
        self.success = self.is_successful()
        
        # created_at이 설정되지 않았으면 timestamp와 동일하게 설정
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = self.timestamp
    
    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> 'AgentExecutionResult':
        """딕셔너리로부터 AgentExecutionResult 생성 (호환성 메서드)"""
        # 필수 필드들 추출
        result_data = data.get('result_data', data.get('data', data.get('result', None)))
        execution_time = data.get('execution_time', 0.0)
        
        # status 처리
        status_str = data.get('status', 'completed' if data.get('success', True) else 'failed')
        if isinstance(status_str, str):
            try:
                status = ExecutionStatus(status_str)
            except ValueError:
                status = ExecutionStatus.COMPLETED if data.get('success', True) else ExecutionStatus.FAILED
        else:
            status = status_str
        
        # agent_type 처리
        agent_type_str = data.get('agent_type', 'general')
        if isinstance(agent_type_str, str):
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                agent_type = AgentType.GENERAL
        else:
            agent_type = agent_type_str
        
        return cls(
            result_data=result_data,
            execution_time=execution_time,
            status=status,
            agent_type=agent_type,
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {}),
            agent_id=data.get('agent_id', ''),
            confidence=data.get('confidence', 0.8)
        )
    
    @classmethod
    def create_simple(cls, agent_id: str, success: bool, data: Any, execution_time: float = 0.0, **kwargs) -> 'AgentExecutionResult':
        """간단한 생성 메서드 (호환성)"""
        return cls(
            result_data=data,
            execution_time=execution_time,
            status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            agent_type=AgentType.GENERAL,
            error_message=kwargs.get('error_message'),
            metadata=kwargs.get('metadata', {}),
            agent_id=agent_id,
            confidence=kwargs.get('confidence', 0.8)
        )
    
    def is_successful(self) -> bool:
        """실행 성공 여부"""
        return self.status == ExecutionStatus.COMPLETED and self.error_message is None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'agent_type': getattr(self.agent_type, 'value', str(self.agent_type)),
            'agent_id': self.agent_id,
            'result_data': self.result_data,
            'data': self.data,
            'execution_time': self.execution_time,
            'status': self.status.value,
            'success': self.success,
            'confidence': self.confidence,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class WorkflowStep:
    """워크플로우 단계"""
    step_id: str
    semantic_purpose: str
    required_concepts: List[str]
    agent_id: str
    estimated_complexity: WorkflowComplexity
    prerequisites: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    confidence: float = 1.0
    execution_context: Dict[str, Any] = field(default_factory=dict)
    alternative_agents: List[str] = field(default_factory=list)
    estimated_time: float = 30.0
    
    @classmethod
    def create_simple(cls, agent_id: str, purpose: str, **kwargs) -> 'WorkflowStep':
        """간단한 워크플로우 단계 생성"""
        return cls(
            step_id=f"step_{uuid.uuid4().hex[:6]}",
            semantic_purpose=purpose,
            required_concepts=kwargs.get('concepts', []),
            agent_id=agent_id,
            estimated_complexity=kwargs.get('complexity', WorkflowComplexity.MODERATE),
            depends_on=kwargs.get('depends_on', []),
            estimated_time=kwargs.get('estimated_time', 30.0)
        )


@dataclass
class WorkflowPlan:
    """워크플로우 실행 계획"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: SemanticQuery = None
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time: float = 0.0
    required_resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    optimization_hints: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    # 추가 속성들 (호환성을 위해)
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    steps: List = field(default_factory=list)  # WorkflowStep 객체들의 리스트
    estimated_quality: float = 0.8
    reasoning_chain: List[str] = field(default_factory=list)
    execution_graph: Any = None  # nx.DiGraph
    
    def add_step(self, agent_type: AgentType, config: Dict[str, Any], dependencies: List[str] = None):
        """실행 단계 추가"""
        step = {
            'step_id': str(uuid.uuid4()),
            'agent_type': agent_type.value,
            'config': config,
            'dependencies': dependencies or []
        }
        self.execution_steps.append(step)
        
        if dependencies:
            self.dependencies[step['step_id']] = dependencies
    
    @classmethod
    def create_simple(cls, 
                     semantic_query: SemanticQuery = None,
                     query: SemanticQuery = None,
                     steps: List = None,
                     strategy: OptimizationStrategy = None,
                     quality: float = 0.8,
                     time: float = 30.0,
                     reasoning: List[str] = None,
                     agent_type: AgentType = AgentType.GENERAL,
                     estimated_time: float = 30.0,
                     **kwargs) -> 'WorkflowPlan':
        """간단한 워크플로우 플랜 생성"""
        # semantic_query와 query 중 하나를 사용
        target_query = semantic_query or query
        if not target_query:
            raise ValueError("semantic_query 또는 query 중 하나는 필수입니다.")
        
        plan = cls(
            query=target_query,
            estimated_time=time or estimated_time,
            optimization_strategy=strategy or kwargs.get('optimization_strategy', OptimizationStrategy.BALANCED),
            steps=steps or [],
            estimated_quality=quality,
            reasoning_chain=reasoning or [],
            **kwargs
        )
        
        # steps가 제공되면 사용, 아니면 기본 단계 생성
        if steps:
            # steps가 WorkflowStep 객체들의 리스트인 경우
            plan.steps = steps
            for step in steps:
                if hasattr(step, 'agent_id'):
                    plan.add_step(
                        agent_type=AgentType.GENERAL,  # 기본값
                        config={
                            'step_id': step.step_id,
                            'agent_id': step.agent_id,
                            'purpose': step.semantic_purpose,
                            'complexity': step.estimated_complexity.value if hasattr(step.estimated_complexity, 'value') else 'moderate'
                        }
                    )
        else:
            # 기본 실행 단계 추가
            plan.add_step(
                agent_type=agent_type,
                config={
                    'query_text': target_query.query_text,
                    'intent': target_query.intent,
                    'complexity': 'simple'
                }
            )
        
        return plan

    # LLM 통합 메서드들
    async def optimize_with_llm(self, llm_manager: Optional['OntologyLLMManager'] = None) -> Dict[str, Any]:
        """LLM을 사용한 워크플로우 최적화"""
        if llm_manager is None:
            from .llm_manager import OntologyLLMManager
            llm_manager = OntologyLLMManager()
        
        # 워크플로우 설계 전문 LLM 사용
        optimization_prompt = f"""
        다음 워크플로우 계획을 최적화해주세요:
        
        쿼리: {self.query.query_text if self.query else "알 수 없음"}
        현재 단계 수: {len(self.execution_steps)}
        예상 시간: {self.estimated_time}초
        최적화 전략: {self.optimization_strategy.value if hasattr(self.optimization_strategy, 'value') else str(self.optimization_strategy)}
        
        현재 단계들:
        {[step.get('config', {}).get('purpose', 'Unknown') for step in self.execution_steps]}
        
        최적화 결과를 JSON 형식으로 제공해주세요:
        {{
            "optimized_steps": [
                {{
                    "agent_type": "agent_type",
                    "purpose": "단계 목적",
                    "estimated_time": 30.0,
                    "dependencies": [],
                    "can_parallel": true/false
                }}
            ],
            "total_estimated_time": 120.0,
            "parallel_groups": [["step1", "step2"], ["step3"]],
            "optimization_reasoning": "최적화 근거",
            "estimated_quality_improvement": 0.0-1.0
        }}
        """
        
        result = await llm_manager.call_llm_async('WORKFLOW_DESIGNER', optimization_prompt)
        
        if result and 'optimized_steps' in result:
            # 기존 단계들을 새로운 최적화된 단계들로 교체
            self.execution_steps.clear()
            self.dependencies.clear()
            
            for step_data in result['optimized_steps']:
                try:
                    agent_type = AgentType(step_data.get('agent_type', 'general'))
                except ValueError:
                    agent_type = AgentType.GENERAL
                
                self.add_step(
                    agent_type=agent_type,
                    config={
                        'purpose': step_data.get('purpose', ''),
                        'estimated_time': step_data.get('estimated_time', 30.0),
                        'can_parallel': step_data.get('can_parallel', False)
                    },
                    dependencies=step_data.get('dependencies', [])
                )
            
            # 전체 예상 시간 업데이트
            if 'total_estimated_time' in result:
                self.estimated_time = result['total_estimated_time']
            
            # 품질 개선 정보 저장
            if 'estimated_quality_improvement' in result:
                self.estimated_quality = min(1.0, self.estimated_quality + result['estimated_quality_improvement'])
            
            # 최적화 힌트 추가
            if 'optimization_reasoning' in result:
                self.optimization_hints.append(result['optimization_reasoning'])
        
        return result or {}
    
    def get_parallel_execution_groups(self) -> List[List[str]]:
        """병렬 실행 가능한 단계 그룹들 반환"""
        if not self.execution_steps:
            return []
        
        groups = []
        remaining_steps = self.execution_steps.copy()
        
        while remaining_steps:
            # 의존성이 없는 단계들을 찾아서 그룹으로 묶기
            parallel_group = []
            completed_steps = set()
            
            for step in remaining_steps[:]:
                dependencies = self.dependencies.get(step['step_id'], [])
                
                # 모든 의존성이 완료되었거나 의존성이 없는 경우
                if not dependencies or all(dep in completed_steps for dep in dependencies):
                    parallel_group.append(step['step_id'])
                    remaining_steps.remove(step)
                    completed_steps.add(step['step_id'])
            
            if parallel_group:
                groups.append(parallel_group)
            else:
                # 더 이상 진행할 수 없는 경우 (순환 의존성 등)
                break
        
        return groups
    
    def estimate_execution_time(self, consider_parallel: bool = True) -> float:
        """실행 시간 추정"""
        if not self.execution_steps:
            return 0.0
        
        if not consider_parallel:
            # 순차 실행 시간
            return sum(
                step.get('config', {}).get('estimated_time', 30.0) 
                for step in self.execution_steps
            )
        
        # 병렬 실행 고려한 시간
        parallel_groups = self.get_parallel_execution_groups()
        total_time = 0.0
        
        for group in parallel_groups:
            # 각 그룹에서 가장 오래 걸리는 단계의 시간
            group_times = []
            for step_id in group:
                step = next((s for s in self.execution_steps if s['step_id'] == step_id), None)
                if step:
                    group_times.append(step.get('config', {}).get('estimated_time', 30.0))
            
            if group_times:
                total_time += max(group_times)
        
        return total_time
    
    def validate_dependencies(self) -> List[str]:
        """의존성 검증 (순환 의존성 등)"""
        errors = []
        
        # 모든 단계 ID 수집
        step_ids = set(step['step_id'] for step in self.execution_steps)
        
        for step_id, deps in self.dependencies.items():
            # 존재하지 않는 의존성 확인
            for dep in deps:
                if dep not in step_ids:
                    errors.append(f"Step {step_id} depends on non-existent step {dep}")
            
            # 순환 의존성 확인 (간단한 검사)
            if step_id in deps:
                errors.append(f"Step {step_id} has circular dependency on itself")
        
        return errors
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """실행 계획 요약"""
        parallel_groups = self.get_parallel_execution_groups()
        validation_errors = self.validate_dependencies()
        
        return {
            'plan_id': self.plan_id,
            'query_text': self.query.query_text if self.query else None,
            'total_steps': len(self.execution_steps),
            'estimated_time_sequential': self.estimate_execution_time(consider_parallel=False),
            'estimated_time_parallel': self.estimate_execution_time(consider_parallel=True),
            'parallel_groups_count': len(parallel_groups),
            'max_parallel_steps': max(len(group) for group in parallel_groups) if parallel_groups else 0,
            'validation_errors': validation_errors,
            'optimization_strategy': getattr(self.optimization_strategy, 'value', str(self.optimization_strategy)),
            'estimated_quality': self.estimated_quality,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ComplexityAnalysis:
    """복잡도 분석 결과"""
    complexity_score: float
    strategy: ExecutionStrategy
    indicators: Dict[str, Any]
    parallel_potential: Dict[str, Any]
    estimated_agents: int
    estimated_time: float
    reasoning: str = ""
    
    def is_simple(self) -> bool:
        """단순한 쿼리인지 확인"""
        return self.strategy == ExecutionStrategy.SINGLE_AGENT
    
    def requires_parallel(self) -> bool:
        """병렬 처리가 필요한지 확인"""
        return self.strategy in [ExecutionStrategy.PARALLEL, ExecutionStrategy.HYBRID]


@dataclass
class CachedSemanticQuery:
    """캐시된 SemanticQuery"""
    query: SemanticQuery
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    query_hash: str = ""
    
    def is_expired(self, ttl_minutes: int = 30) -> bool:
        """캐시 만료 여부 확인"""
        from datetime import timedelta
        return datetime.now() - self.created_at > timedelta(minutes=ttl_minutes)
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: int = 3600  # 1시간 기본 TTL
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class SystemMetrics:
    """시스템 성능 메트릭"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    duplicate_calls_prevented: int = 0
    parallel_executions: int = 0
    failed_executions: int = 0
    analysis_calls: int = 0
    average_execution_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    # LLM 관련 메트릭
    llm_calls_total: int = 0
    llm_calls_successful: int = 0
    llm_calls_failed: int = 0
    average_llm_response_time: float = 0.0
    llm_cache_hits: int = 0
    llm_cache_misses: int = 0
    llm_enhancement_count: int = 0  # LLM으로 개선된 쿼리 수
    llm_optimization_count: int = 0  # LLM으로 최적화된 워크플로우 수
    
    def update_cache(self, hit: bool):
        """캐시 히트/미스 업데이트"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def update_llm_cache(self, hit: bool):
        """LLM 캐시 히트/미스 업데이트"""
        if hit:
            self.llm_cache_hits += 1
        else:
            self.llm_cache_misses += 1
    
    def record_llm_call(self, success: bool, response_time: float = 0.0):
        """LLM 호출 기록"""
        self.llm_calls_total += 1
        if success:
            self.llm_calls_successful += 1
        else:
            self.llm_calls_failed += 1
        
        # 평균 응답 시간 업데이트
        if response_time > 0:
            total_time = self.average_llm_response_time * (self.llm_calls_total - 1) + response_time
            self.average_llm_response_time = total_time / self.llm_calls_total
    
    def record_llm_enhancement(self):
        """LLM 개선 기록"""
        self.llm_enhancement_count += 1
    
    def record_llm_optimization(self):
        """LLM 최적화 기록"""
        self.llm_optimization_count += 1
    
    def get_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def get_llm_cache_hit_rate(self) -> float:
        """LLM 캐시 히트율 계산"""
        total = self.llm_cache_hits + self.llm_cache_misses
        return (self.llm_cache_hits / total * 100) if total > 0 else 0.0
    
    def get_llm_success_rate(self) -> float:
        """LLM 성공률 계산"""
        return (self.llm_calls_successful / self.llm_calls_total * 100) if self.llm_calls_total > 0 else 0.0
    
    def get_success_rate(self) -> float:
        """성공률 계산"""
        total = self.total_queries
        successful = total - self.failed_executions
        return (successful / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            # 기존 메트릭
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'average_response_time': self.average_response_time,
            'duplicate_calls_prevented': self.duplicate_calls_prevented,
            'parallel_executions': self.parallel_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.get_success_rate(),
            'analysis_calls': self.analysis_calls,
            'average_execution_time': self.average_execution_time,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            
            # LLM 메트릭
            'llm_calls_total': self.llm_calls_total,
            'llm_calls_successful': self.llm_calls_successful,
            'llm_calls_failed': self.llm_calls_failed,
            'llm_success_rate': self.get_llm_success_rate(),
            'average_llm_response_time': self.average_llm_response_time,
            'llm_cache_hits': self.llm_cache_hits,
            'llm_cache_misses': self.llm_cache_misses,
            'llm_cache_hit_rate': self.get_llm_cache_hit_rate(),
            'llm_enhancement_count': self.llm_enhancement_count,
            'llm_optimization_count': self.llm_optimization_count
        }


@dataclass
class AgentCapability:
    """에이전트 능력 정의"""
    agent_type: AgentType
    supported_query_types: List[QueryType]
    processing_time_estimate: float  # 초 단위
    resource_requirements: Dict[str, Any]
    confidence_threshold: float = 0.7
    max_concurrent_tasks: int = 1
    
    def can_handle(self, query: SemanticQuery) -> bool:
        """쿼리 처리 가능 여부"""
        return query.query_type in self.supported_query_types


# 전역 상수 및 설정
DEFAULT_CACHE_TTL = 3600  # 1시간
MAX_CACHE_SIZE = 1000
DEFAULT_TIMEOUT = 300  # 5분
MAX_RETRY_COUNT = 3

# 에이전트 기본 능력 정의
DEFAULT_AGENT_CAPABILITIES = {
    AgentType.RESEARCH: AgentCapability(
        agent_type=AgentType.RESEARCH,
        supported_query_types=[QueryType.SIMPLE, QueryType.COMPLEX, QueryType.ANALYTICAL],
        processing_time_estimate=10.0,
        resource_requirements={'memory': 'medium', 'cpu': 'low'},
        max_concurrent_tasks=2
    ),
    AgentType.ANALYSIS: AgentCapability(
        agent_type=AgentType.ANALYSIS,
        supported_query_types=[QueryType.ANALYTICAL, QueryType.COMPLEX],
        processing_time_estimate=15.0,
        resource_requirements={'memory': 'high', 'cpu': 'medium'},
        max_concurrent_tasks=1
    ),
    AgentType.CREATIVE: AgentCapability(
        agent_type=AgentType.CREATIVE,
        supported_query_types=[QueryType.CREATIVE, QueryType.SIMPLE],
        processing_time_estimate=20.0,
        resource_requirements={'memory': 'medium', 'cpu': 'high'},
        max_concurrent_tasks=1
    ),
    AgentType.TECHNICAL: AgentCapability(
        agent_type=AgentType.TECHNICAL,
        supported_query_types=[QueryType.COMPLEX, QueryType.MULTI_STEP],
        processing_time_estimate=12.0,
        resource_requirements={'memory': 'high', 'cpu': 'high'},
        max_concurrent_tasks=2
    ),
    AgentType.GENERAL: AgentCapability(
        agent_type=AgentType.GENERAL,
        supported_query_types=list(QueryType),
        processing_time_estimate=8.0,
        resource_requirements={'memory': 'low', 'cpu': 'low'},
        max_concurrent_tasks=3
    )
} 

# LLM 통합을 위한 유틸리티 함수들
def create_enhanced_semantic_query(query_text: str, **kwargs) -> SemanticQuery:
    """LLM으로 개선된 SemanticQuery 생성"""
    query = SemanticQuery.create_from_text(query_text, **kwargs)
    
    # 동기식으로 LLM 개선 시도
    try:
        enhanced_query = query.enhance_with_llm_sync()
        return enhanced_query
    except Exception:
        # LLM 개선 실패 시 기본 쿼리 반환
        return query

def create_optimized_workflow_plan(semantic_query: SemanticQuery, **kwargs) -> WorkflowPlan:
    """LLM으로 최적화된 WorkflowPlan 생성"""
    plan = WorkflowPlan.create_simple(semantic_query=semantic_query, **kwargs)
    
    # 비동기로 최적화하되, 실패 시 기본 플랜 반환
    try:
        import asyncio
        asyncio.run(plan.optimize_with_llm())
    except Exception:
        pass  # 최적화 실패 시 기본 플랜 사용
    
    return plan

async def batch_enhance_queries(queries: List[SemanticQuery], 
                               llm_manager: Optional['OntologyLLMManager'] = None) -> List[SemanticQuery]:
    """여러 쿼리를 배치로 LLM 개선"""
    if llm_manager is None:
        from .llm_manager import OntologyLLMManager
        llm_manager = OntologyLLMManager()
    
    enhanced_queries = []
    
    # 배치 처리로 효율성 증대
    for query in queries:
        try:
            await query.analyze_with_llm(llm_manager)
            enhanced_queries.append(query)
        except Exception as e:
            # 개별 쿼리 개선 실패 시 원본 유지
            query.metadata['llm_enhancement_error'] = str(e)
            enhanced_queries.append(query)
    
    return enhanced_queries

# 글로벌 시스템 메트릭 인스턴스
_global_metrics = SystemMetrics()

def get_system_metrics() -> SystemMetrics:
    """글로벌 시스템 메트릭 반환"""
    return _global_metrics

def reset_system_metrics():
    """시스템 메트릭 초기화"""
    global _global_metrics
    _global_metrics = SystemMetrics()

# 캐시 및 성능 관련 유틸리티
class QueryCache:
    """쿼리 캐시 관리"""
    
    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl_seconds: int = DEFAULT_CACHE_TTL):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key not in self.cache:
            _global_metrics.update_cache(hit=False)
            return None
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            _global_metrics.update_cache(hit=False)
            return None
        
        entry.update_access()
        _global_metrics.update_cache(hit=True)
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """캐시에 값 저장"""
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # LRU 방식으로 가장 오래된 항목 제거
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].last_accessed)
            del self.cache[oldest_key]
        
        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds or self.ttl_seconds
        )
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': _global_metrics.get_cache_hit_rate(),
            'cache_hits': _global_metrics.cache_hits,
            'cache_misses': _global_metrics.cache_misses
        }

# 글로벌 쿼리 캐시 인스턴스
_global_query_cache = QueryCache()

def get_query_cache() -> QueryCache:
    """글로벌 쿼리 캐시 반환"""
    return _global_query_cache

# 성능 모니터링 데코레이터
def track_execution_time(func):
    """실행 시간 추적 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 메트릭 업데이트
            _global_metrics.analysis_calls += 1
            total_time = _global_metrics.average_execution_time * (_global_metrics.analysis_calls - 1) + execution_time
            _global_metrics.average_execution_time = total_time / _global_metrics.analysis_calls
            
            return result
        except Exception as e:
            _global_metrics.failed_executions += 1
            raise e
    
    return wrapper

# 비동기 버전 데코레이터
def track_execution_time_async(func):
    """비동기 실행 시간 추적 데코레이터"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 메트릭 업데이트
            _global_metrics.analysis_calls += 1
            total_time = _global_metrics.average_execution_time * (_global_metrics.analysis_calls - 1) + execution_time
            _global_metrics.average_execution_time = total_time / _global_metrics.analysis_calls
            
            return result
        except Exception as e:
            _global_metrics.failed_executions += 1
            raise e
    
    return wrapper 