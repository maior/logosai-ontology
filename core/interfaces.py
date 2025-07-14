"""
🧠 Core Interfaces
핵심 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator, AsyncIterator
from .models import (
    SemanticQuery, 
    ExecutionContext, 
    AgentExecutionResult, 
    WorkflowPlan, 
    ComplexityAnalysis,
    DataTransformationType,
    AgentType,
    ExecutionStrategy,
    SystemMetrics
)


class QueryAnalyzer(ABC):
    """쿼리 분석기 인터페이스"""
    
    @abstractmethod
    async def analyze_query(self, query_text: str, context: Dict[str, Any] = None) -> SemanticQuery:
        """쿼리 분석 및 SemanticQuery 객체 생성"""
        pass
    
    @abstractmethod
    def estimate_complexity(self, query: SemanticQuery) -> float:
        """쿼리 복잡도 추정 (0.0 ~ 1.0)"""
        pass
    
    @abstractmethod
    def suggest_agents(self, query: SemanticQuery) -> List[AgentType]:
        """쿼리에 적합한 에이전트 추천"""
        pass


class ExecutionEngine(ABC):
    """실행 엔진 인터페이스"""
    
    @abstractmethod
    async def execute_query(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> List[AgentExecutionResult]:
        """쿼리 실행"""
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[ExecutionStrategy]:
        """지원하는 실행 전략 목록"""
        pass
    
    @abstractmethod
    async def estimate_execution_time(
        self, 
        query: SemanticQuery, 
        strategy: ExecutionStrategy
    ) -> float:
        """실행 시간 추정"""
        pass


class DataTransformer(ABC):
    """데이터 변환기 인터페이스"""
    
    @abstractmethod
    async def transform_input(
        self, 
        data: Any, 
        source_agent: AgentType, 
        target_agent: AgentType
    ) -> Any:
        """입력 데이터 변환"""
        pass
    
    @abstractmethod
    async def transform_output(
        self, 
        result: AgentExecutionResult, 
        target_format: str
    ) -> Any:
        """출력 데이터 변환"""
        pass
    
    @abstractmethod
    def get_supported_transformations(self) -> Dict[str, List[str]]:
        """지원하는 변환 목록"""
        pass


class ResultProcessor(ABC):
    """결과 처리기 인터페이스"""
    
    @abstractmethod
    async def process_results(self, 
                            results: List[AgentExecutionResult],
                            context: ExecutionContext) -> Dict[str, Any]:
        """결과 처리"""
        pass
    
    @abstractmethod
    def validate_result(self, result: AgentExecutionResult) -> bool:
        """결과 유효성 검증"""
        pass


class WorkflowDesigner(ABC):
    """워크플로우 설계자 인터페이스"""
    
    @abstractmethod
    async def design_workflow(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> WorkflowPlan:
        """워크플로우 설계"""
        pass
    
    @abstractmethod
    async def optimize_workflow(self, plan: WorkflowPlan) -> WorkflowPlan:
        """워크플로우 최적화"""
        pass
    
    @abstractmethod
    def validate_workflow(self, plan: WorkflowPlan) -> bool:
        """워크플로우 유효성 검증"""
        pass


class CacheManager(ABC):
    """캐시 관리자 인터페이스"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """캐시에 값 저장"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """캐시 전체 삭제"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        pass


class KnowledgeGraph(ABC):
    """지식 그래프 인터페이스"""
    
    @abstractmethod
    async def add_concept(self, concept: str, properties: Dict[str, Any] = None):
        """개념 추가"""
        pass
    
    @abstractmethod
    async def add_relation(self, source: str, target: str, relation_type: str, properties: Dict[str, Any] = None):
        """관계 추가"""
        pass
    
    @abstractmethod
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """그래프 쿼리"""
        pass
    
    @abstractmethod
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """관련 개념 찾기"""
        pass
    
    @abstractmethod
    def visualize_graph(self, output_path: str = None) -> str:
        """그래프 시각화"""
        pass


class AgentCaller(ABC):
    """에이전트 호출자 인터페이스"""
    
    @abstractmethod
    async def call_agent(
        self, 
        agent_type: AgentType, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """단일 에이전트 호출"""
        pass
    
    @abstractmethod
    async def call_agents_parallel(
        self, 
        agent_calls: List[Dict[str, Any]]
    ) -> List[AgentExecutionResult]:
        """병렬 에이전트 호출"""
        pass
    
    @abstractmethod
    def get_agent_status(self, agent_type: AgentType) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        pass


class ProgressCallback(ABC):
    """진행 상황 콜백 인터페이스"""
    
    @abstractmethod
    async def on_progress(self, message: str, progress: float, metadata: Dict[str, Any] = None):
        """진행 상황 업데이트"""
        pass
    
    @abstractmethod
    async def on_step_complete(self, step_id: str, result: AgentExecutionResult):
        """단계 완료 알림"""
        pass
    
    @abstractmethod
    async def on_error(self, error_message: str, error_details: Dict[str, Any] = None):
        """오류 발생 알림"""
        pass


class SystemMonitor(ABC):
    """시스템 모니터 인터페이스"""
    
    @abstractmethod
    def record_execution(self, 
                        strategy: str, 
                        execution_time: float, 
                        success: bool, 
                        metadata: Dict[str, Any] = None):
        """실행 기록"""
        pass
    
    @abstractmethod
    def record_cache_access(self, hit: bool, key: str = None):
        """캐시 접근 기록"""
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭스 조회"""
        pass


class MetricsCollector(ABC):
    """메트릭 수집기 인터페이스"""
    
    @abstractmethod
    def record_query_execution(
        self, 
        query: SemanticQuery, 
        execution_time: float, 
        success: bool
    ):
        """쿼리 실행 기록"""
        pass
    
    @abstractmethod
    def record_cache_access(self, hit: bool):
        """캐시 접근 기록"""
        pass
    
    @abstractmethod
    def record_duplicate_prevention(self):
        """중복 호출 방지 기록"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> SystemMetrics:
        """메트릭 조회"""
        pass
    
    @abstractmethod
    async def export_metrics(self, format: str = "json") -> str:
        """메트릭 내보내기"""
        pass


class EventListener(ABC):
    """이벤트 리스너 인터페이스"""
    
    @abstractmethod
    async def on_query_start(self, query: SemanticQuery, context: ExecutionContext):
        """쿼리 시작 이벤트"""
        pass
    
    @abstractmethod
    async def on_query_complete(self, query: SemanticQuery, results: List[AgentExecutionResult]):
        """쿼리 완료 이벤트"""
        pass
    
    @abstractmethod
    async def on_agent_call(self, agent_type: AgentType, query: SemanticQuery):
        """에이전트 호출 이벤트"""
        pass
    
    @abstractmethod
    async def on_cache_hit(self, key: str):
        """캐시 히트 이벤트"""
        pass
    
    @abstractmethod
    async def on_error(self, error: Exception, context: Dict[str, Any]):
        """오류 이벤트"""
        pass


class ConfigurationManager(ABC):
    """설정 관리자 인터페이스"""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """설정 값 저장"""
        pass
    
    @abstractmethod
    def load_config_file(self, file_path: str) -> bool:
        """설정 파일 로드"""
        pass
    
    @abstractmethod
    def save_config_file(self, file_path: str) -> bool:
        """설정 파일 저장"""
        pass
    
    @abstractmethod
    def get_all_configs(self) -> Dict[str, Any]:
        """모든 설정 조회"""
        pass


class HealthChecker(ABC):
    """헬스 체커 인터페이스"""
    
    @abstractmethod
    async def check_system_health(self) -> Dict[str, Any]:
        """시스템 전체 헬스 체크"""
        pass
    
    @abstractmethod
    async def check_agent_health(self, agent_type: AgentType) -> Dict[str, Any]:
        """특정 에이전트 헬스 체크"""
        pass
    
    @abstractmethod
    async def check_cache_health(self) -> Dict[str, Any]:
        """캐시 시스템 헬스 체크"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> str:
        """전체 헬스 상태 (healthy/degraded/unhealthy)"""
        pass


class ProgressTracker(ABC):
    """진행 상황 추적기 인터페이스"""
    
    @abstractmethod
    async def start_tracking(self, session_id: str, total_steps: int):
        """추적 시작"""
        pass
    
    @abstractmethod
    async def update_progress(self, session_id: str, completed_steps: int, message: str = None):
        """진행 상황 업데이트"""
        pass
    
    @abstractmethod
    async def complete_tracking(self, session_id: str, success: bool = True):
        """추적 완료"""
        pass
    
    @abstractmethod
    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """진행 상황 조회"""
        pass
    
    @abstractmethod
    async def subscribe_progress(self, session_id: str) -> AsyncIterator[Dict[str, Any]]:
        """진행 상황 구독 (실시간 업데이트)"""
        pass 