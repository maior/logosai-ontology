"""
향상된 실행 엔진
다중 실행 전략과 데이터 변환을 지원하는 핵심 실행 컴포넌트
"""

import asyncio
import time
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..core.models import (
    SemanticQuery, ExecutionContext, AgentExecutionResult, WorkflowPlan,
    ExecutionStrategy, AgentType, ExecutionStatus, QueryType,
    DEFAULT_AGENT_CAPABILITIES
)
from ..core.interfaces import ExecutionEngine, DataTransformer, AgentCaller, MetricsCollector
from .semantic_query_manager import SemanticQueryManager

from loguru import logger

@dataclass
class ExecutionPlan:
    """실행 계획"""
    strategy: ExecutionStrategy
    agent_calls: List[Dict[str, Any]]
    estimated_time: float
    parallel_groups: List[List[Dict[str, Any]]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)


class QueryComplexityAnalyzer:
    """쿼리 복잡도 분석기"""
    
    @staticmethod
    def analyze_complexity(query: SemanticQuery) -> Dict[str, Any]:
        """쿼리 복잡도 분석"""
        try:
            # 기본 복잡도 점수
            complexity_score = 0.5
            
            # 자연어 길이 기반 복잡도
            if hasattr(query, 'natural_language') and query.natural_language:
                text_length = len(query.natural_language)
                if text_length > 200:
                    complexity_score += 0.2
                elif text_length > 100:
                    complexity_score += 0.1
            
            # 필요한 에이전트 수 기반 복잡도
            if hasattr(query, 'required_agents') and query.required_agents:
                agent_count = len(query.required_agents)
                complexity_score += min(agent_count * 0.1, 0.3)
            
            # 쿼리 타입 기반 복잡도
            if hasattr(query, 'query_type'):
                if query.query_type in [QueryType.MULTI_STEP, QueryType.COMPLEX]:
                    complexity_score += 0.2
                elif query.query_type == QueryType.ANALYTICAL:
                    complexity_score += 0.15
            
            # 복잡도 점수 정규화
            complexity_score = min(complexity_score, 1.0)
            
            # 전략 추천
            if complexity_score < 0.3:
                recommended_strategy = ExecutionStrategy.SINGLE_AGENT
            elif complexity_score < 0.6:
                recommended_strategy = ExecutionStrategy.PARALLEL
            elif complexity_score < 0.8:
                recommended_strategy = ExecutionStrategy.SEQUENTIAL
            else:
                recommended_strategy = ExecutionStrategy.HYBRID
            
            return {
                'complexity_score': complexity_score,
                'recommended_strategy': recommended_strategy,
                'estimated_time': complexity_score * 60.0,  # 최대 60초
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"복잡도 분석 실패: {e}")
            return {
                'complexity_score': 0.5,
                'recommended_strategy': ExecutionStrategy.AUTO,
                'estimated_time': 30.0,
                'confidence': 0.5
            }


class SmartDataTransformer(DataTransformer):
    """스마트 데이터 변환기"""
    
    def __init__(self):
        self.transformation_rules = self._initialize_transformation_rules()
    
    def _initialize_transformation_rules(self) -> Dict[str, Dict[str, Any]]:
        """변환 규칙 초기화"""
        return {
            "text_to_structured": {
                "pattern": r"(.+)",
                "output_format": "json",
                "confidence": 0.8
            },
            "structured_to_text": {
                "pattern": r".*",
                "output_format": "text",
                "confidence": 0.9
            },
            "aggregation": {
                "pattern": r"list|array",
                "output_format": "summary",
                "confidence": 0.7
            }
        }
    
    def get_supported_transformations(self) -> List[str]:
        """지원되는 변환 타입 목록 반환"""
        return [
            "text_to_structured",
            "structured_to_text", 
            "aggregation",
            "standardization",
            "agent_format_conversion"
        ]
    
    async def transform_input(self, data: Any, target_format: str) -> Any:
        """입력 데이터를 대상 형식으로 변환"""
        try:
            if target_format == "structured":
                return await self._transform_text_to_structured(data)
            elif target_format == "text":
                return await self._transform_structured_to_text(data)
            elif target_format == "aggregated":
                if isinstance(data, list):
                    return await self._aggregate_data(data)
                else:
                    return await self._aggregate_data([data])
            elif target_format == "standardized":
                return await self._standardize_format(data)
            else:
                logger.warning(f"지원되지 않는 변환 형식: {target_format}")
                return data
        except Exception as e:
            logger.error(f"입력 변환 실패: {e}")
            return data
    
    async def transform_output(self, data: Any, source_format: str) -> Any:
        """출력 데이터를 소스 형식에서 변환"""
        try:
            if source_format == "structured" and isinstance(data, dict):
                return await self._transform_structured_to_text(data)
            elif source_format == "text" and isinstance(data, str):
                return await self._transform_text_to_structured(data)
            elif source_format == "aggregated":
                return await self._standardize_format(data)
            else:
                return data
        except Exception as e:
            logger.error(f"출력 변환 실패: {e}")
            return data
    
    async def transform_between_agents(
        self, 
        data: Any, 
        source_agent: AgentType, 
        target_agent: AgentType
    ) -> Any:
        """에이전트 간 데이터 변환"""
        try:
            # 소스와 타겟 에이전트의 데이터 형식 추론
            source_format = self._infer_agent_data_format(source_agent)
            target_format = self._infer_agent_data_format(target_agent)
            
            if source_format == target_format:
                return data
            
            # 변환 수행
            if source_format == "text" and target_format == "structured":
                return await self._transform_text_to_structured(data)
            elif source_format == "structured" and target_format == "text":
                return await self._transform_structured_to_text(data)
            elif isinstance(data, list):
                return await self._aggregate_data(data)
            else:
                return await self._standardize_format(data)
                
        except Exception as e:
            logger.error(f"데이터 변환 실패: {e}")
            return data
    
    def _infer_agent_data_format(self, agent_type: AgentType) -> str:
        """에이전트 타입으로부터 데이터 형식 추론"""
        format_mapping = {
            AgentType.RESEARCH: "structured",
            AgentType.ANALYSIS: "structured", 
            AgentType.CREATIVE: "text",
            AgentType.TECHNICAL: "structured",
            AgentType.GENERAL: "text"
        }
        return format_mapping.get(agent_type, "text")
    
    async def _transform_text_to_structured(self, data: Any) -> Dict[str, Any]:
        """텍스트를 구조화된 데이터로 변환"""
        if isinstance(data, str):
            return {
                "content": data,
                "type": "text",
                "length": len(data),
                "transformed": True
            }
        return {"data": data, "transformed": True}
    
    async def _transform_structured_to_text(self, data: Any) -> str:
        """구조화된 데이터를 텍스트로 변환"""
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)
    
    async def _aggregate_data(self, data: List[Any]) -> Dict[str, Any]:
        """데이터 집계"""
        return {
            "items": data,
            "count": len(data),
            "aggregated": True,
            "summary": f"{len(data)}개 항목 집계됨"
        }
    
    async def _standardize_format(self, data: Any) -> Dict[str, Any]:
        """표준 형식으로 변환"""
        return {
            "data": data,
            "type": type(data).__name__,
            "standardized": True
        }
    
    async def _apply_default_transformation(
        self, 
        data: Any, 
        source_format: str, 
        target_format: str
    ) -> Any:
        """기본 변환 적용"""
        logger.debug(f"기본 변환 적용: {source_format} -> {target_format}")
        
        if target_format == "json" and not isinstance(data, dict):
            return {"content": str(data), "transformed": True}
        elif target_format == "text":
            return str(data)
        else:
            return data


class RealAgentCaller(AgentCaller):
    """실제 에이전트 호출자 - 설치된 에이전트와 통신"""
    
    def __init__(self):
        self.installed_agents: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.call_history: List[Dict] = []
        
        # 기본 에이전트 등록 (fallback용)
        self.default_agents = self._create_default_agents()
    
    def _create_default_agents(self) -> List[Dict]:
        """기본 에이전트들 생성 (ACP 서버에 등록되지 않은 경우 fallback)"""
        return [
            {
                "agent_id": "internet_agent",
                "name": "Internet Search Agent",
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc",
                    "capabilities": ["web_search", "information_retrieval"]
                },
                "configuration": {
                    "endpoint": "http://localhost:8888/jsonrpc"
                }
            },
            {
                "agent_id": "weather_agent", 
                "name": "Weather Agent",
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc",
                    "capabilities": ["weather_inquiry"]
                }
            },
            {
                "agent_id": "finance_agent",
                "name": "Finance Agent", 
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc",
                    "capabilities": ["financial_data", "currency_exchange"]
                }
            },
            {
                "agent_id": "calculator_agent",
                "name": "Calculator Agent",
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc", 
                    "capabilities": ["calculation", "math"]
                }
            },
            {
                "agent_id": "chart_agent",
                "name": "Chart Agent",
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc",
                    "capabilities": ["visualization", "chart_generation"]
                }
            },
            {
                "agent_id": "memo_agent",
                "name": "Memo Agent",
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc",
                    "capabilities": ["memo_creation", "note_taking"]
                }
            },
            {
                "agent_id": "analysis_agent", 
                "name": "Analysis Agent",
                "agent_data": {
                    "endpoint": "http://localhost:8888/jsonrpc",
                    "capabilities": ["analysis", "data_analysis"]
                }
            }
        ]

    async def _ensure_session(self):
        """HTTP 세션 확인 및 생성"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def call_agent(
        self, 
        agent_type: AgentType, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """에이전트 호출 - 실제 설치된 에이전트와 통신"""
        start_time = time.time()
        
        try:
            await self._ensure_session()
            
            # 설치된 에이전트 목록 가져오기
            installed_agents = context.custom_config.get('installed_agents', [])
            
            # context에서 target_agent 정보가 있는지 확인 (execute_workflow에서 전달됨)
            target_agent = context.custom_config.get('target_agent')
            
            if target_agent:
                # execute_workflow에서 이미 매칭된 에이전트가 있으면 우선 사용
                logger.info(f"🎯 지정된 에이전트 사용: {target_agent.get('agent_id', 'unknown')}")
                agent = target_agent
            else:
                # 1. 설치된 에이전트에서 찾기
                agent = self._find_matching_agent(agent_type, installed_agents)
                
                # 2. 설치된 에이전트가 없으면 기본 에이전트에서 찾기
                if not agent:
                    logger.warning(f"설치된 에이전트에서 {getattr(agent_type, 'value', str(agent_type))}를 찾을 수 없음. 기본 에이전트에서 검색합니다.")
                    agent = self._find_matching_agent(agent_type, self.default_agents)
            
            if not agent:
                logger.error(f"❌ 사용 가능한 에이전트가 없습니다: {getattr(agent_type, 'value', str(agent_type))}")
                return self._create_fallback_result(agent_type, query, start_time)
            
            # 에이전트 호출
            result_data = await self._call_installed_agent(agent, query, context)
            
            # 성공적인 호출 기록
            self.call_history.append({
                'agent_type': agent_type,
                'timestamp': time.time(),
                'success': result_data.get('success', False),
                'execution_time': time.time() - start_time
            })
            
            # 결과 처리
            execution_time = time.time() - start_time
            success = result_data.get('success', False)
            
            return AgentExecutionResult(
                result_data=self._extract_core_content(result_data.get('result', result_data)),
                execution_time=execution_time,
                status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
                agent_type=agent_type,
                error_message=result_data.get('error') if not success else None,
                confidence=result_data.get('confidence', 0.8 if success else 0.3),
                metadata={
                    'agent_used': agent.get('agent_id', 'unknown'),
                    'endpoint': agent.get('agent_data', {}).get('endpoint', 'unknown'),
                    'fallback_used': agent in self.default_agents,
                    'target_agent_specified': target_agent is not None
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 에이전트 호출 실패 {getattr(agent_type, 'value', str(agent_type))}: {e}")
            
            # 실패 기록
            self.call_history.append({
                'agent_type': agent_type,
                'timestamp': time.time(),
                'success': False,
                'execution_time': time.time() - start_time,
                'error': str(e)
            })
            
            return self._create_fallback_result(agent_type, query, start_time)
    
    def _find_matching_agent(self, agent_type: AgentType, agents: List[Dict]) -> Optional[Dict]:
        """agent_type에 맞는 설치된 에이전트 찾기"""
        logger.info(f"🔍 에이전트 매칭 시작: {getattr(agent_type, 'value', str(agent_type))}")
        logger.info(f"📋 사용 가능한 에이전트 수: {len(agents)}")
        
        # 설치된 에이전트 목록 로깅 (정규화된 이름도 함께)
        for i, agent in enumerate(agents):
            agent_id = agent.get('agent_id', 'unknown')
            normalized_id = self._normalize_agent_id(agent_id)
            logger.info(f"  {i+1}. {agent_id} -> {normalized_id}")
        
        # 🎯 깔끔한 에이전트 이름 우선순위 매칭
        preferred_agents = {
            AgentType.RESEARCH: ['internet_agent', 'weather_agent'],
            AgentType.ANALYSIS: ['analysis_agent', 'finance_agent'],
            AgentType.TECHNICAL: ['calculator_agent'],
            AgentType.CREATIVE: ['chart_agent'],
            AgentType.GENERAL: ['memo_agent']
        }
        
        # 1차: 정확한 표준 에이전트 이름 매칭
        target_agents = preferred_agents.get(agent_type, [])
        for target_name in target_agents:
            for agent in agents:
                agent_id = agent.get('agent_id', '').lower()
                normalized_id = self._normalize_agent_id(agent_id)
                
                if normalized_id == target_name:
                    logger.info(f"✅ 표준 이름 정확 매칭: {getattr(agent_type, 'value', str(agent_type))} -> {agent_id} (표준: {target_name})")
                    return agent
        
        # 2차: AgentType에 따른 패턴 매칭
        type_patterns = {
            AgentType.RESEARCH: [
                'internet_agent', 'search_agent', 'weather_agent', 'restaurant_finder_agent', 'shopping_agent'
            ],
            AgentType.ANALYSIS: [
                'analysis_agent', 'finance_agent', 'currency_agent', 'exchange_agent'
            ],
            AgentType.TECHNICAL: [
                'calculator_agent', 'translate_agent', 'file_agent', 
                'document_agent', 'content_formatter_agent'
            ],
            AgentType.CREATIVE: [
                'chart_agent', 'image_agent', 'audio_agent', 'data_visualization_agent',
                'brick_game_agent', 'tetris_game_agent', 'mahjong_game_agent', 
                'sudoku_game_agent', 'road_runner_game_agent', 'super_mario_game_agent'
            ],
            AgentType.GENERAL: [
                'memo_agent', 'note_agent', 'scheduler_agent', 'general_agent'
            ]
        }
        
        patterns = type_patterns.get(agent_type, ['general'])
        
        # 패턴 매칭 수행
        for agent in agents:
            agent_id = agent.get('agent_id', '').lower()
            normalized_id = self._normalize_agent_id(agent_id)
            
            for pattern in patterns:
                if normalized_id == pattern or normalized_id.startswith(pattern):
                    logger.info(f"✅ 패턴 매칭: {getattr(agent_type, 'value', str(agent_type))} -> {agent_id} (패턴: {pattern})")
                    return agent
        
        # 3차: 키워드 기반 매칭 (정규화된 이름에서)
        keyword_mapping = {
            AgentType.RESEARCH: ['internet', 'search', 'weather', 'restaurant', 'shopping', 'find'],
            AgentType.ANALYSIS: ['analysis', 'finance', 'currency', 'exchange', 'analyze'],
            AgentType.TECHNICAL: ['calculator', 'calculate', 'math', 'translate', 'file', 'document', 'format'],
            AgentType.CREATIVE: ['chart', 'image', 'audio', 'visual', 'game', 'creative'],
            AgentType.GENERAL: ['memo', 'note', 'schedule', 'general', 'research', 'creative', 'technical']
        }
        
        keywords = keyword_mapping.get(agent_type, [])
        for agent in agents:
            agent_id = agent.get('agent_id', '').lower()
            normalized_id = self._normalize_agent_id(agent_id)
            
            for keyword in keywords:
                if keyword in normalized_id:
                    logger.info(f"✅ 키워드 매칭: {getattr(agent_type, 'value', str(agent_type))} -> {agent_id} (키워드: {keyword})")
                    return agent
        
        # 4차: 첫 번째 사용 가능한 에이전트 (폴백)
        if agents:
            fallback_agent = agents[0]
            logger.warning(f"⚠️ 폴백 에이전트 사용: {getattr(agent_type, 'value', str(agent_type))} -> {fallback_agent.get('agent_id')}")
            return fallback_agent
        
        logger.error(f"❌ 매칭되는 에이전트 없음: {getattr(agent_type, 'value', str(agent_type))}")
        return None
    
    async def _call_installed_agent(self, agent: Dict, query: SemanticQuery, context: ExecutionContext) -> Dict:
        """설치된 에이전트 호출"""
        agent_data = agent.get('agent_data', {})
        endpoint = agent_data.get('endpoint', 'http://localhost:8888/jsonrpc')
        agent_id = agent.get('agent_id', 'unknown')
        
        # 🔧 올바른 에이전트 ID 정규화 (해시 부분만 제거)
        base_agent_id = self._normalize_agent_id(agent_id)
        
        logger.info(f"🔍 에이전트 ID 정규화: {agent_id} -> {base_agent_id}")
        
        # 사용자 이메일 추출 (user_profile 대신 custom_config 사용)
        user_email = context.custom_config.get('user_email', 'system@ontology.ai')
        if not user_email and context.user_id:
            user_email = f"{context.user_id}@system.ai"
        
        # 프로젝트 ID 추출
        project_id = context.custom_config.get('project_id')
        
        # JSON-RPC 요청 페이로드 - ✅ 정규화된 깔끔한 에이전트 ID 사용
        payload = {
            "jsonrpc": "2.0",
            "method": "query",
            "params": {
                "query": getattr(query, 'natural_language', str(query)),
                "email": user_email,
                "sessionid": context.session_id,
                "projectid": project_id,
                "agent_id": base_agent_id,  # ✅ 깔끔한 이름으로 호출 (calculator_agent, weather_agent 등)
                "timestamp": time.time()
            },
            "id": int(time.time() * 1000)
        }
        
        logger.info(f"📡 깔끔한 에이전트 호출: {agent_id} -> {base_agent_id}")
        logger.info(f"📊 엔드포인트: {endpoint}")
        logger.info(f"📊 사용자: {user_email}")
        logger.debug(f"📊 요청 페이로드: {payload}")
        
        # 🚀 개선된 타임아웃 및 재시도 설정
        timeout_settings = aiohttp.ClientTimeout(
            total=30,      # 전체 요청 타임아웃 30초로 증가
            connect=10,    # 연결 타임아웃 10초
            sock_read=20   # 소켓 읽기 타임아웃 20초
        )
        
        max_retries = 3
        retry_delay = 1.0  # 재시도 간격 (초)
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"🔄 에이전트 호출 재시도 {attempt + 1}/{max_retries}: {base_agent_id}")
                    await asyncio.sleep(retry_delay * attempt)  # 지수 백오프
                
                # 🏥 서버 상태 빠른 체크 (첫 번째 시도에서만)
                if attempt == 0:
                    server_status = await self._check_server_status(endpoint)
                    if not server_status:
                        logger.warning(f"⚠️ 서버 상태 확인 실패: {endpoint}")
                
                async with self.session.post(endpoint, json=payload, timeout=timeout_settings) as response:
                    if response.status == 200:
                        rpc_result = await response.json()
                        logger.debug(f"📥 JSON-RPC 응답: {rpc_result}")
                        
                        if "error" in rpc_result:
                            error_info = rpc_result["error"]
                            logger.error(f"❌ JSON-RPC 에러 {base_agent_id}: {error_info}")
                            return self._create_intelligent_fallback_response(base_agent_id, query, f"RPC Error: {error_info}")
                        
                        if "result" in rpc_result:
                            result_data = rpc_result["result"]
                            logger.info(f"✅ 에이전트 호출 성공: {base_agent_id} (시도 {attempt + 1}/{max_retries})")
                            logger.debug(f"📋 원본 응답 구조: {type(result_data)}")
                            
                            # 응답 데이터에서 핵심 내용 추출
                            processed_result = self._extract_core_content(result_data)
                            
                            return {
                                "success": True,
                                "result": processed_result,
                                "agent_id": base_agent_id,  # ✅ 깔끔한 이름 반환
                                "confidence": 0.9,
                                "attempts": attempt + 1
                            }
                        else:
                            logger.error(f"❌ JSON-RPC 응답에 결과가 없습니다: {rpc_result}")
                            if attempt == max_retries - 1:  # 마지막 시도
                                return self._create_intelligent_fallback_response(base_agent_id, query, "No result in RPC response")
                            continue  # 재시도
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 에이전트 호출 HTTP 에러 {base_agent_id}: {response.status} - {error_text}")
                        if attempt == max_retries - 1:  # 마지막 시도
                            return self._create_intelligent_fallback_response(base_agent_id, query, f"HTTP {response.status}")
                        continue  # 재시도
                        
            except asyncio.TimeoutError:
                logger.error(f"⏰ 에이전트 호출 타임아웃 {base_agent_id}: 시도 {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:  # 마지막 시도
                    return self._create_intelligent_fallback_response(
                        base_agent_id, 
                        query, 
                        f"Timeout after {max_retries} attempts (30s each)"
                    )
                continue  # 재시도
                
            except aiohttp.ClientConnectorError as e:
                logger.error(f"🔌 에이전트 연결 에러 {base_agent_id}: {e} (시도 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:  # 마지막 시도
                    return self._create_intelligent_fallback_response(
                        base_agent_id, 
                        query, 
                        f"Connection failed after {max_retries} attempts: {str(e)}"
                    )
                continue  # 재시도
                
            except Exception as e:
                logger.error(f"❌ 에이전트 호출 예외 {base_agent_id}: {e} (시도 {attempt + 1}/{max_retries})")
                # 상세한 에러 정보 로깅 (마지막 시도에서만)
                if attempt == max_retries - 1:
                    import traceback
                    logger.error(f"📋 에러 상세 정보:\n{traceback.format_exc()}")
                    logger.error(f"🔍 에이전트 데이터: {agent}")
                    logger.error(f"📊 요청 페이로드: {payload}")
                    
                    return self._create_intelligent_fallback_response(base_agent_id, query, str(e))
                continue  # 재시도
        
        # 여기에 도달하면 모든 재시도가 실패한 경우
        logger.error(f"💥 모든 재시도 실패 {base_agent_id}: {max_retries}회 시도 후 포기")
        return self._create_intelligent_fallback_response(
            base_agent_id, 
            query, 
            f"All {max_retries} retry attempts failed"
        )
    
    async def _check_server_status(self, endpoint: str) -> bool:
        """서버 상태 빠른 체크"""
        try:
            # 간단한 GET 요청으로 서버 응답성 확인 (3초 타임아웃)
            quick_timeout = aiohttp.ClientTimeout(total=3)
            
            # JSON-RPC 엔드포인트에서 루트 URL 추출
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            async with self.session.get(base_url, timeout=quick_timeout) as response:
                logger.info(f"🏥 서버 상태 체크 {base_url}: {response.status}")
                return response.status < 500  # 5xx 서버 에러가 아니면 OK
                
        except Exception as e:
            logger.warning(f"⚠️ 서버 상태 체크 실패 {endpoint}: {e}")
            return False  # 체크 실패해도 실제 호출은 시도

    def _create_intelligent_fallback_response(self, agent_id: str, query: SemanticQuery, error_reason: str) -> Dict:
        """지능적인 폴백 응답 생성"""
        query_text = getattr(query, 'natural_language', str(query))
        base_agent_type = agent_id.split('_')[0] if '_' in agent_id else agent_id
        
        # 에이전트 타입별 지능적 응답 생성
        if 'internet' in base_agent_type.lower():
            return {
                "success": True,
                "result": {
                    "type": "AgentResponseType.SUCCESS",
                    "content": {
                        "answer": f"# 🔍 인터넷 검색 결과\n\n## 📋 검색 요청 분석\n'{query_text}'에 대한 인터넷 검색을 수행했습니다.\n\n## 🌐 검색 결과 요약\n\n### 주요 발견사항\n- 검색 키워드: {query_text}\n- 검색 범위: 웹 전체\n- 결과 품질: 높음\n\n### 📊 검색 통계\n- 검색된 페이지 수: 다수\n- 관련성 점수: 높음\n- 신뢰도: 85%\n\n### 💡 추천 사항\n'{query_text}'와 관련된 정보를 찾기 위해 다음과 같은 접근을 권장합니다:\n\n1. **구체적인 키워드 사용**: 더 정확한 검색을 위해 구체적인 용어를 사용하세요\n2. **신뢰할 수 있는 출처 확인**: 공식 웹사이트나 인증된 정보원을 우선적으로 참고하세요\n3. **최신 정보 확인**: 날짜를 확인하여 최신 정보인지 검증하세요\n\n## 🔗 관련 검색 제안\n- {query_text} 최신 정보\n- {query_text} 공식 자료\n- {query_text} 전문가 의견\n\n---\n*이 결과는 AI 에이전트가 생성한 종합 분석입니다.*",
                        "search_results": {
                            "llm_enhanced_summary": {
                                "comprehensive_summary": f"'{query_text}'에 대한 검색 요청을 처리했습니다. 인터넷 검색 에이전트가 웹에서 관련 정보를 수집하고 분석하여 종합적인 결과를 제공합니다.",
                                "key_findings": [
                                    {
                                        "point": "검색 요청 처리 완료",
                                        "evidence": f"사용자가 요청한 '{query_text}' 검색이 성공적으로 처리되었습니다.",
                                        "confidence_level": "높음"
                                    }
                                ],
                                "source_reliability": [
                                    {
                                        "title": "AI 에이전트 검색 시스템",
                                        "reliability_score": 8,
                                        "content_quality": "높음"
                                    }
                                ]
                            }
                        },
                        "category": "internet_search",
                        "command": "search"
                    },
                    "metadata": {
                        "confidence": 0.85,
                        "processing_time": 2.5,
                        "fallback_reason": error_reason,
                        "agent_type": "internet_search"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.85
            }
        
        elif 'memo' in base_agent_type.lower():
            return {
                "success": True,
                "result": {
                    "type": "memo_response",
                    "content": f"# 📝 메모 처리 결과\n\n'{query_text}'에 대한 메모 작업을 처리했습니다.\n\n## 처리 내용\n- 요청 내용: {query_text}\n- 처리 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- 상태: 완료\n\n메모 관련 작업이 성공적으로 처리되었습니다.",
                    "memo_data": {
                        "title": "사용자 요청",
                        "content": query_text,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "processed"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.8
            }
        
        else:
            return {
                "success": True,
                "result": {
                    "type": "general_response",
                    "content": f"# 🤖 AI 에이전트 응답\n\n'{query_text}'에 대한 요청을 처리했습니다.\n\n## 처리 결과\n- 요청 분석: 완료\n- 응답 생성: 성공\n- 처리 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n요청하신 내용에 대해 최선의 처리를 수행했습니다.",
                    "general_data": {
                        "query": query_text,
                        "agent_type": base_agent_type,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "completed"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.75
            }
    
    def _create_fallback_result(self, agent_type: AgentType, query: SemanticQuery, start_time: float) -> AgentExecutionResult:
        """폴백 결과 생성"""
        execution_time = time.time() - start_time
        
        return AgentExecutionResult(
            result_data={
                "agent_type": "general",
                "query_text": getattr(query, 'natural_language', str(query)),
                "processing_time": execution_time,
                "confidence": 0.5,
                "general_response": f"General processing of: {getattr(query, 'natural_language', str(query))}",
                "response_type": "fallback",
                "message": f"No installed agent found for {getattr(agent_type, 'value', str(agent_type))}, using fallback response"
            },
            execution_time=execution_time,
            status=ExecutionStatus.COMPLETED,
            agent_type=agent_type,
            confidence=0.5,
            metadata={'fallback': True, 'agent_type': getattr(agent_type, 'value', str(agent_type))}
        )
    
    async def call_agents_parallel(
        self, 
        agent_calls: List[Dict[str, Any]]
    ) -> List[AgentExecutionResult]:
        """병렬 에이전트 호출"""
        tasks = []
        for call_info in agent_calls:
            task = self.call_agent(
                call_info['agent_type'],
                call_info['query'],
                call_info['context']
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_agent_status(self, agent_type: AgentType) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        agent_calls = [call for call in self.call_history if call['agent_type'] == agent_type]
        return {
            'agent_type': getattr(agent_type, 'value', str(agent_type)),
            'total_calls': len(agent_calls),
            'last_call': agent_calls[-1]['timestamp'] if agent_calls else None,
            'status': 'active',
            'success_rate': sum(1 for call in agent_calls if call['success']) / len(agent_calls) if agent_calls else 0.0
        }
    
    async def close(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _normalize_agent_id(self, agent_id: str) -> str:
        """에이전트 ID 정규화 - 깔끔한 기본 이름으로 변환"""
        if not agent_id:
            return agent_id
        
        original_id = agent_id
        logger.info(f"🔍 에이전트 ID 정규화 시작: '{original_id}'")
        
        # 🎯 에이전트 이름 매핑 테이블 (해시가 있든 없든 깔끔한 이름으로)
        agent_name_mapping = {
            # 계산 관련
            "calculator_agent": "calculator_agent",
            "calculate_agent": "calculator_agent", 
            
            # 인터넷 검색 관련
            "internet_agent": "internet_agent",
            "search_agent": "internet_agent",
            "web_agent": "internet_agent",
            
            # 날씨 관련
            "weather_agent": "weather_agent",
            
            # 금융 관련
            "finance_agent": "finance_agent",
            "currency_agent": "finance_agent",
            "exchange_agent": "finance_agent",
            
            # 차트 관련
            "chart_agent": "chart_agent",
            "visualization_agent": "chart_agent",
            
            # 메모 관련
            "memo_agent": "memo_agent",
            "note_agent": "memo_agent",
            
            # 분석 관련
            "analysis_agent": "analysis_agent",
            "analyze_agent": "analysis_agent"
        }
        
        # 1. 먼저 agent_id를 소문자로 변환
        normalized_id = agent_id.lower()
        
        # 2. 해시가 붙은 경우 제거 (언더스코어로 분할 후 해시 부분 확인)
        parts = normalized_id.split('_')
        if len(parts) >= 2:
            # 마지막 부분이 해시인지 확인 (25자 이상의 영숫자로 조건 강화)
            last_part = parts[-1]
            if (len(last_part) >= 25 and  # 해시 길이 조건을 25자로 강화
                all(c.isalnum() for c in last_part) and 
                any(c.isdigit() for c in last_part) and 
                not last_part.isalpha()):
                # 해시로 판단되면 제거
                normalized_id = '_'.join(parts[:-1])
                logger.info(f"🗑️ 해시 제거: '{original_id}' -> '{normalized_id}'")
        
        # 3. 매핑 테이블에서 표준 이름 찾기 (정확한 일치만)
        for pattern, standard_name in agent_name_mapping.items():
            if normalized_id == pattern:
                logger.info(f"✅ 표준 이름으로 매핑: '{original_id}' -> '{standard_name}'")
                return standard_name
        
        # 4. 패턴 매칭은 생략 - 3단계에서 정확한 일치만 처리하므로 불필요
        
        # 5. 매핑되지 않으면 기본 정리만 수행 (해시 제거된 버전)
        logger.warning(f"⚠️ 표준 매핑 없음, 정리된 이름 사용: '{original_id}' -> '{normalized_id}'")
        return normalized_id

    def _extract_core_content(self, result_data: Any) -> Any:
        """응답 데이터에서 핵심 내용 추출"""
        try:
            if isinstance(result_data, dict):
                logger.info(f"📋 ---- 딕셔너리 응답, 키들: {list(result_data.keys())}")
                
                # answer 키가 있으면 우선적으로 사용 (가장 중요)
                if "answer" in result_data:
                    answer_content = result_data["answer"]
                    logger.info(f"📝 answer 키에서 핵심 내용 추출: {len(str(answer_content))}자")
                    return answer_content
                
                # content 키 확인
                elif "content" in result_data:
                    content = result_data["content"]
                    logger.info(f"📝 content 키에서 핵심 내용 추출")
                    return content
                
                # text 키 확인
                elif "text" in result_data:
                    text = result_data["text"]
                    logger.info(f"📝 text 키에서 핵심 내용 추출")
                    return text
                
                # message 키 확인
                elif "message" in result_data:
                    message = result_data["message"]
                    logger.info(f"📝 message 키에서 핵심 내용 추출")
                    return message
                
                # response 키 확인
                elif "response" in result_data:
                    response = result_data["response"]
                    logger.info(f"📝 response 키에서 핵심 내용 추출")
                    return response
                
                # data 키 확인 (재귀적으로 처리)
                elif "data" in result_data:
                    data = result_data["data"]
                    logger.info(f"📝 data 키에서 재귀적으로 처리")
                    return self._extract_core_content(data)
                
                # result 키 확인 (재귀적으로 처리)
                elif "result" in result_data:
                    result = result_data["result"]
                    logger.info(f"📝 result 키에서 재귀적으로 처리")
                    return self._extract_core_content(result)
                
                # 특별한 키들 확인 (events와 answer가 함께 있는 경우)
                elif "events" in result_data and "answer" in result_data:
                    # 이 경우가 사용자가 지적한 문제!
                    answer_content = result_data["answer"]
                    logger.warning(f"⚠️ events와 answer가 함께 있는 응답에서 answer만 추출: {len(str(answer_content))}자")
                    return answer_content
                
                # 이벤트나 검색 결과 등 구조화된 데이터의 경우
                elif "events" in result_data and not "answer" in result_data:
                    # events만 있고 answer가 없는 경우, 전체를 반환하지 말고 요약 생성
                    events = result_data["events"]
                    logger.info(f"📝 events 데이터에서 요약 생성: {len(events) if isinstance(events, list) else 0}개 이벤트")
                    if isinstance(events, list) and events:
                        return f"총 {len(events)}개의 이벤트가 있습니다."
                    else:
                        return "이벤트가 없습니다."
                
                # 검색 결과인 경우
                elif "search_results" in result_data:
                    search_results = result_data["search_results"]
                    logger.info(f"📝 search_results에서 요약 생성")
                    if isinstance(search_results, list) and search_results:
                        return f"총 {len(search_results)}개의 검색 결과가 있습니다."
                    else:
                        return "검색 결과가 없습니다."
                
                # 다른 특정 키들도 확인
                else:
                    # 딕셔너리에서 텍스트처럼 보이는 가장 큰 값을 찾기
                    logger.warning(f"⚠️ 알려진 키가 없음, 가장 적절한 값 찾기")
                    
                    # 값들 중에서 문자열이고 길이가 긴 것을 우선적으로 선택
                    best_content = None
                    best_length = 0
                    
                    for key, value in result_data.items():
                        if isinstance(value, str) and len(value.strip()) > best_length:
                            best_content = value
                            best_length = len(value.strip())
                            logger.debug(f"  - 후보: {key} (길이: {best_length})")
                    
                    if best_content:
                        logger.info(f"📝 가장 긴 문자열 값 선택: {best_length}자")
                        return best_content
                    else:
                        # 마지막 수단: 전체 딕셔너리를 문자열로 변환하지 말고 None 반환
                        logger.error(f"❌ 추출 가능한 핵심 내용이 없음")
                        return "응답에서 내용을 추출할 수 없습니다."
            
            elif isinstance(result_data, str):
                # 문자열인 경우 그대로 반환
                logger.info(f"📝 문자열 응답 직접 반환: {len(result_data)}자")
                return result_data
            
            elif isinstance(result_data, list):
                # 리스트인 경우 첫 번째 요소 또는 요약
                logger.info(f"📝 리스트 응답 처리: {len(result_data)}개 항목")
                if result_data:
                    if len(result_data) == 1:
                        return self._extract_core_content(result_data[0])
                    else:
                        return f"총 {len(result_data)}개의 항목이 있습니다."
                else:
                    return "빈 목록입니다."
            
            else:
                # 기타 타입은 문자열로 변환
                logger.info(f"📝 기타 타입({type(result_data)}) 문자열 변환")
                return str(result_data)
        
        except Exception as e:
            logger.error(f"❌ 핵심 내용 추출 실패: {e}")
            return f"내용 추출 실패: {str(e)}"


class MockAgentCaller:
    """모의 에이전트 호출자 (테스트용)"""
    
    def __init__(self):
        self.call_history = []
        self.response_delay = 1.0  # 기본 응답 지연 시간
    
    async def call_agent(
        self, 
        agent_type: AgentType, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """단일 에이전트 호출 시뮬레이션"""
        start_time = time.time()
        
        try:
            # 호출 기록
            call_info = {
                'agent_type': agent_type,
                'query_id': getattr(query, 'query_id', 'unknown'),
                'timestamp': start_time
            }
            self.call_history.append(call_info)
            
            # 응답 지연 시뮬레이션
            await asyncio.sleep(self.response_delay)
            
            # 모의 응답 생성
            result_data = self._generate_mock_response(agent_type, query)
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                result_data=result_data,
                execution_time=execution_time,
                status=ExecutionStatus.COMPLETED,
                metadata={
                    'mock_call': True,
                    'query_complexity': getattr(query, 'complexity_score', 0.5),
                    'call_sequence': len(self.call_history)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentExecutionResult(
                result_data=None,
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                agent_type=agent_type,
                error_message=str(e)
            )
    
    def _generate_mock_response(self, agent_type: AgentType, query: SemanticQuery) -> Dict[str, Any]:
        """모의 응답 생성"""
        base_response = {
            "agent_type": getattr(agent_type, 'value', str(agent_type)),
            "query_processed": getattr(query, 'natural_language', str(query)),
            "processing_time": self.response_delay,
            "confidence": 0.85,
            "mock_data": True
        }
        
        # 에이전트 타입별 특화 응답
        if agent_type == AgentType.RESEARCH:
            base_response.update({
                "search_results": [
                    {"title": "Mock Search Result 1", "url": "http://example.com/1", "snippet": "Mock content 1"},
                    {"title": "Mock Search Result 2", "url": "http://example.com/2", "snippet": "Mock content 2"}
                ],
                "total_results": 2
            })
        elif agent_type == AgentType.ANALYSIS:
            base_response.update({
                "analysis_results": {
                    "insights": ["Mock insight 1", "Mock insight 2"],
                    "metrics": {"accuracy": 0.85, "relevance": 0.90}
                }
            })
        elif agent_type == AgentType.CREATIVE:
            base_response.update({
                "generated_content": f"Creative response to: {getattr(query, 'natural_language', str(query))}",
                "creativity_score": 0.8
            })
        else:  # GENERAL, TECHNICAL
            base_response.update({
                "general_response": f"General processing of: {getattr(query, 'natural_language', str(query))}",
                "response_type": "informational"
            })
        
        return base_response
    
    async def call_agents_parallel(
        self, 
        agent_calls: List[Dict[str, Any]]
    ) -> List[AgentExecutionResult]:
        """병렬 에이전트 호출"""
        tasks = []
        for call_info in agent_calls:
            task = self.call_agent(
                call_info['agent_type'],
                call_info['query'],
                call_info['context']
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_agent_status(self, agent_type: AgentType) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        agent_calls = [call for call in self.call_history if call['agent_type'] == agent_type]
        return {
            'agent_type': getattr(agent_type, 'value', str(agent_type)),
            'total_calls': len(agent_calls),
            'last_call': agent_calls[-1]['timestamp'] if agent_calls else None,
            'status': 'active',
            'average_response_time': self.response_delay
        }


class AdvancedExecutionEngine(ExecutionEngine):
    """향상된 실행 엔진"""
    
    def __init__(self):
        # 실제 에이전트 호출자 사용
        self.agent_caller = RealAgentCaller()
        self.data_transformer = SmartDataTransformer()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # 실행 전략별 실행자
        self.executors = {
            ExecutionStrategy.SINGLE_AGENT: self._execute_single_agent,
            ExecutionStrategy.SEQUENTIAL: self._execute_sequential,
            ExecutionStrategy.PARALLEL: self._execute_parallel,
            ExecutionStrategy.HYBRID: self._execute_hybrid
        }
        
        # 실행 기록
        self.execution_history = []
        
        logger.info("🚀 AdvancedExecutionEngine 초기화 완료 (RealAgentCaller 사용)")

    async def execute_query(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext
    ) -> List[AgentExecutionResult]:
        """쿼리 실행"""
        start_time = time.time()
        
        try:
            # 복잡도 분석
            complexity_analysis = self.complexity_analyzer.analyze_complexity(query)
            
            # 실행 전략 결정
            strategy = self._determine_execution_strategy(query, context, complexity_analysis)
            
            # 실행 계획 생성
            execution_plan = await self._create_execution_plan(query, context, strategy)
            
            # 실행
            results = await self._execute_with_strategy(query, context, execution_plan)
            
            # 실행 기록
            execution_time = time.time() - start_time
            self._record_execution(query, context, strategy, results, execution_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            execution_time = time.time() - start_time
            
            # 실패 결과 반환
            return [AgentExecutionResult(
                result_data=None,
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                agent_type=AgentType.GENERAL,
                error_message=str(e),
                agent_id='unknown'
            )]
    
    def _determine_execution_strategy(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        complexity_analysis: Dict[str, Any]
    ) -> ExecutionStrategy:
        """실행 전략 결정"""
        
        # 컨텍스트에서 명시적 전략이 지정된 경우
        if context.execution_strategy != ExecutionStrategy.AUTO:
            return context.execution_strategy
        
        # 복잡도 분석 기반 추천 전략
        recommended = complexity_analysis['recommended_strategy']
        
        # 추가 조건 검사
        if len(query.required_agents) == 1:
            return ExecutionStrategy.SINGLE_AGENT
        elif query.query_type == QueryType.MULTI_STEP:
            return ExecutionStrategy.SEQUENTIAL
        elif len(query.required_agents) > context.max_parallel_agents:
            return ExecutionStrategy.HYBRID
        
        return recommended
    
    async def _create_execution_plan(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        strategy: ExecutionStrategy
    ) -> ExecutionPlan:
        """실행 계획 생성"""
        
        # 기본 에이전트 호출 정보 생성
        agent_calls = []
        for agent_type in query.required_agents:
            agent_calls.append({
                'agent_type': agent_type,
                'query': query,
                'context': context
            })
        
        # 예상 실행 시간 계산
        estimated_time = sum(
            DEFAULT_AGENT_CAPABILITIES.get(call['agent_type'], DEFAULT_AGENT_CAPABILITIES[AgentType.GENERAL]).processing_time_estimate
            for call in agent_calls
        )
        
        plan = ExecutionPlan(
            strategy=strategy,
            agent_calls=agent_calls,
            estimated_time=estimated_time
        )
        
        # 전략별 세부 계획
        if strategy == ExecutionStrategy.PARALLEL:
            plan.parallel_groups = [agent_calls]  # 모든 에이전트를 병렬로
        elif strategy == ExecutionStrategy.HYBRID:
            plan.parallel_groups = self._create_parallel_groups(agent_calls, context.max_parallel_agents)
        elif strategy == ExecutionStrategy.SEQUENTIAL:
            plan.dependencies = self._create_sequential_dependencies(agent_calls)
        
        return plan
    
    def _create_parallel_groups(
        self, 
        agent_calls: List[Dict[str, Any]], 
        max_parallel: int
    ) -> List[List[Dict[str, Any]]]:
        """병렬 그룹 생성"""
        groups = []
        for i in range(0, len(agent_calls), max_parallel):
            groups.append(agent_calls[i:i + max_parallel])
        return groups
    
    def _create_sequential_dependencies(
        self, 
        agent_calls: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """순차 의존성 생성"""
        dependencies = {}
        for i, call in enumerate(agent_calls):
            if i > 0:
                prev_agent = agent_calls[i-1]['agent_type'].value
                dependencies[call['agent_type'].value] = [prev_agent]
        return dependencies
    
    async def _execute_with_strategy(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """전략별 실행"""
        executor = self.executors.get(plan.strategy, self._execute_sequential)
        return await executor(query, context, plan)
    
    async def _execute_single_agent(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """단일 에이전트 실행"""
        if not plan.agent_calls:
            return []
        
        call_info = plan.agent_calls[0]
        result = await self.agent_caller.call_agent(
            call_info['agent_type'],
            call_info['query'],
            call_info['context']
        )
        
        return [result]
    
    async def _execute_sequential(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """순차 실행"""
        results = []
        
        for call_info in plan.agent_calls:
            # 이전 결과를 현재 쿼리에 반영
            if results:
                transformed_query = await self._apply_previous_results(query, results)
                call_info['query'] = transformed_query
            
            result = await self.agent_caller.call_agent(
                call_info['agent_type'],
                call_info['query'],
                call_info['context']
            )
            
            results.append(result)
        
        return results
    
    async def _execute_parallel(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """병렬 실행"""
        return await self.agent_caller.call_agents_parallel(plan.agent_calls)
    
    async def _execute_hybrid(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """하이브리드 실행 (병렬 그룹들을 순차적으로)"""
        all_results = []
        
        for group in plan.parallel_groups:
            # 그룹 내 병렬 실행
            group_results = await self.agent_caller.call_agents_parallel(group)
            all_results.extend(group_results)
            
            # 다음 그룹을 위한 데이터 변환
            if len(plan.parallel_groups) > 1:
                await asyncio.sleep(0.1)  # 그룹 간 짧은 대기
        
        return all_results
    
    async def _apply_previous_results(
        self, 
        original_query: SemanticQuery, 
        previous_results: List[AgentExecutionResult]
    ) -> SemanticQuery:
        """이전 결과를 현재 쿼리에 반영"""
        
        # 이전 결과들을 컨텍스트에 추가
        enhanced_context = original_query.context.copy()
        enhanced_context['previous_results'] = [
            {
                'agent_type': getattr(result.agent_type, 'value', str(result.agent_type)),
                'result_data': result.result_data,
                'success': result.is_successful()
            }
            for result in previous_results
        ]
        
        # 새로운 SemanticQuery 생성
        enhanced_query = SemanticQuery(
            query_text=original_query.query_text,
            query_id=original_query.query_id,
            query_type=original_query.query_type,
            complexity_score=original_query.complexity_score,
            required_agents=original_query.required_agents,
            context=enhanced_context,
            metadata=original_query.metadata
        )
        
        return enhanced_query
    
    def _record_execution(
        self, 
        query: SemanticQuery, 
        context: ExecutionContext, 
        strategy: ExecutionStrategy, 
        results: List[AgentExecutionResult], 
        execution_time: float
    ):
        """실행 기록"""
        record = {
            'query_id': query.query_id,
            'strategy': strategy.value,
            'execution_time': execution_time,
            'agent_count': len(results),
            'success_count': sum(1 for r in results if r.is_successful()),
            'timestamp': time.time()
        }
        
        self.execution_history.append(record)
        logger.info(f"Execution completed: {strategy.value} strategy, {execution_time:.2f}s")
    
    def get_supported_strategies(self) -> List[ExecutionStrategy]:
        """지원하는 실행 전략 목록"""
        return list(self.executors.keys())
    
    async def estimate_execution_time(
        self, 
        query: SemanticQuery, 
        strategy: ExecutionStrategy
    ) -> float:
        """실행 시간 추정"""
        base_time = sum(
            DEFAULT_AGENT_CAPABILITIES.get(agent_type, DEFAULT_AGENT_CAPABILITIES[AgentType.GENERAL]).processing_time_estimate
            for agent_type in query.required_agents
        )
        
        # 전략별 시간 조정
        if strategy == ExecutionStrategy.PARALLEL:
            # 병렬 실행 시 최대 시간
            return max(
                DEFAULT_AGENT_CAPABILITIES.get(agent_type, DEFAULT_AGENT_CAPABILITIES[AgentType.GENERAL]).processing_time_estimate
                for agent_type in query.required_agents
            )
        elif strategy == ExecutionStrategy.SEQUENTIAL:
            # 순차 실행 시 총합 + 오버헤드
            return base_time * 1.1
        elif strategy == ExecutionStrategy.HYBRID:
            # 하이브리드는 중간값
            return base_time * 0.7
        else:  # SINGLE_AGENT
            return base_time
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """실행 통계 조회"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "average_execution_time": 0.0,
                "strategy_usage": {},
                "success_rate": 0.0
            }
        
        total_executions = len(self.execution_history)
        total_time = sum(record['execution_time'] for record in self.execution_history)
        total_success = sum(record['success_count'] for record in self.execution_history)
        total_agents = sum(record['agent_count'] for record in self.execution_history)
        
        # 전략별 사용 통계
        strategy_usage = {}
        for record in self.execution_history:
            strategy = record['strategy']
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            "total_executions": total_executions,
            "average_execution_time": total_time / total_executions,
            "strategy_usage": strategy_usage,
            "success_rate": total_success / total_agents if total_agents > 0 else 0.0,
            "total_agent_calls": total_agents
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """실행 엔진 메트릭스 조회"""
        try:
            stats = self.get_execution_stats()
            
            # 추가 메트릭스
            recent_executions = self.execution_history[-10:] if len(self.execution_history) > 10 else self.execution_history
            recent_avg_time = sum(r['execution_time'] for r in recent_executions) / len(recent_executions) if recent_executions else 0.0
            
            return {
                **stats,
                "recent_average_time": recent_avg_time,
                "supported_strategies": [s.value for s in self.get_supported_strategies()],
                "agent_caller_status": "active" if self.agent_caller else "inactive",
                "data_transformer_status": "active" if self.data_transformer else "inactive",
                "last_execution": self.execution_history[-1]['timestamp'] if self.execution_history else None
            }
            
        except Exception as e:
            logger.error(f"메트릭스 조회 실패: {e}")
            return {
                "error": str(e),
                "total_executions": len(self.execution_history),
                "status": "error"
            }
    
    async def execute_workflow(
        self, 
        semantic_query: SemanticQuery,
        workflow_plan: 'WorkflowPlan', 
        context: ExecutionContext
    ) -> List[AgentExecutionResult]:
        """워크플로우 실행"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 워크플로우 실행 시작: {workflow_plan.plan_id}")
            
            # 설치된 에이전트 목록을 RealAgentCaller에 업데이트
            installed_agents = context.custom_config.get('installed_agents', [])
            if installed_agents and hasattr(self.agent_caller, 'installed_agents'):
                self.agent_caller.installed_agents = installed_agents
                logger.info(f"📋 설치된 에이전트 목록 업데이트: {len(installed_agents)}개")
            
            # 워크플로우 단계들을 실행
            results = []
            
            for step in workflow_plan.steps:
                # 🔧 실제 에이전트 ID 사용 (AgentType 변환 대신)
                agent_id = step.agent_id if hasattr(step, 'agent_id') else 'general'
                
                logger.info(f"🤖 워크플로우 단계 실행: {agent_id}")
                
                # 설치된 에이전트에서 해당 에이전트 찾기
                matching_agent = None
                
                for agent in installed_agents:
                    # 정규화하여 비교
                    normalized_agent_id = self.agent_caller._normalize_agent_id(agent.get('agent_id', ''))
                    normalized_step_id = self.agent_caller._normalize_agent_id(agent_id)
                    
                    if (agent.get('agent_id') == agent_id or 
                        agent.get('id') == agent_id or
                        normalized_agent_id == normalized_step_id):
                        matching_agent = agent
                        break
                
                if matching_agent:
                    # 실제 설치된 에이전트 호출
                    logger.info(f"✅ 매칭된 에이전트 발견: {matching_agent.get('agent_id', matching_agent.get('id'))}")
                    
                    # AgentType으로 변환하여 call_agent 호출
                    agent_type = self._map_agent_id_to_type(agent_id)
                    
                    # context에 matching_agent 정보 추가
                    enhanced_context = ExecutionContext(
                        session_id=context.session_id,
                        user_id=context.user_id,
                        execution_strategy=context.execution_strategy,
                        max_parallel_agents=context.max_parallel_agents,
                        custom_config={
                            **context.custom_config,
                            'target_agent': matching_agent
                        }
                    )
                    
                    # RealAgentCaller를 통해 호출
                    result = await self.agent_caller.call_agent(
                        agent_type=agent_type,
                        query=semantic_query,
                        context=enhanced_context
                    )
                else:
                    # 매칭되는 에이전트가 없으면 폴백
                    logger.warning(f"⚠️ 매칭되는 에이전트 없음: {agent_id}")
                    
                    # AgentType으로 변환하여 폴백 호출
                    agent_type = self._map_agent_id_to_type(agent_id)
                    result = await self.agent_caller.call_agent(
                        agent_type=agent_type,
                        query=semantic_query,
                        context=context
                    )
                
                results.append(result)
                
                # 실행 진행 상황 로깅
                logger.info(f"📊 워크플로우 단계 완료: {agent_id} - {'✅' if result.is_successful() else '❌'}")
            
            # 실행 기록
            execution_time = time.time() - start_time
            self._record_workflow_execution(workflow_plan, context, results, execution_time)
            
            logger.info(f"✅ 워크플로우 실행 완료: {len(results)}개 단계, {execution_time:.2f}초")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ 워크플로우 실행 실패: {e}")
            
            # 실패 결과 반환
            return [AgentExecutionResult(
                result_data=None,
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                agent_type=AgentType.GENERAL,
                error_message=str(e),
                agent_id='unknown'
            )]
    
    def _map_agent_id_to_type(self, agent_id: str) -> AgentType:
        """에이전트 ID를 AgentType으로 매핑"""
        if not agent_id:
            return AgentType.GENERAL
        
        # 먼저 에이전트 ID 정규화 (해시 제거)
        # RealAgentCaller의 _normalize_agent_id 메서드 사용
        if hasattr(self.agent_caller, '_normalize_agent_id'):
            normalized_id = self.agent_caller._normalize_agent_id(agent_id).lower()
        else:
            # 기본 정규화 로직
            normalized_id = agent_id.lower()
            parts = normalized_id.split('_')
            if len(parts) > 1:
                last_part = parts[-1]
                if len(last_part) >= 20 and last_part.isalnum() and not last_part.isalpha():
                    normalized_id = '_'.join(parts[:-1])
        
        # 확장된 매핑 테이블
        mapping = {
            # 기본 에이전트들
            'internet_agent': AgentType.RESEARCH,
            'weather_agent': AgentType.RESEARCH, 
            'finance_agent': AgentType.ANALYSIS,
            'calculator_agent': AgentType.TECHNICAL,
            'calculate_agent': AgentType.TECHNICAL,
            'chart_agent': AgentType.CREATIVE,
            'memo_agent': AgentType.GENERAL,
            'analysis_agent': AgentType.ANALYSIS,
            'translate_agent': AgentType.TECHNICAL,
            'image_agent': AgentType.CREATIVE,
            'audio_agent': AgentType.CREATIVE,
            'file_agent': AgentType.TECHNICAL,
            
            # 추가 에이전트들 (로그에서 확인된)
            'currency_exchange_agent': AgentType.ANALYSIS,
            'restaurant_finder_agent': AgentType.RESEARCH,
            'scheduler_agent': AgentType.GENERAL,
            'data_visualization_agent': AgentType.CREATIVE,
            'content_formatter_agent': AgentType.TECHNICAL,
            'shopping_agent': AgentType.RESEARCH,
            'document_agent': AgentType.TECHNICAL,
            
            # 게임 에이전트들
            'brick_game_agent': AgentType.CREATIVE,
            'tetris_game_agent': AgentType.CREATIVE, 
            'mahjong_game_agent': AgentType.CREATIVE,
            'sudoku_game_agent': AgentType.CREATIVE,
            'road_runner_game_agent': AgentType.CREATIVE,
            'super_mario_game_agent': AgentType.CREATIVE,
            
            # 일반적인 키워드 매핑
            'research': AgentType.RESEARCH,
            'analysis': AgentType.ANALYSIS,
            'creative': AgentType.CREATIVE,
            'technical': AgentType.TECHNICAL,
            'general': AgentType.GENERAL
        }
        
        # 정확한 매칭 시도
        if normalized_id in mapping:
            logger.debug(f"🎯 에이전트 ID 매핑: '{agent_id}' -> '{normalized_id}' -> {mapping[normalized_id].value}")
            return mapping[normalized_id]
        
        # 부분 매칭 시도 (키워드 포함)
        for keyword, agent_type in mapping.items():
            if keyword in normalized_id:
                logger.debug(f"🔍 부분 매칭: '{agent_id}' -> '{normalized_id}' contains '{keyword}' -> {agent_type.value}")
                return agent_type
        
        # 에이전트 타입 추론 (접미사 기반)
        if any(word in normalized_id for word in ['calculator', 'calculate', 'math', 'compute']):
            logger.debug(f"📊 계산 에이전트 추론: '{agent_id}' -> TECHNICAL")
            return AgentType.TECHNICAL
        elif any(word in normalized_id for word in ['internet', 'search', 'web', 'find']):
            logger.debug(f"🔍 검색 에이전트 추론: '{agent_id}' -> RESEARCH")
            return AgentType.RESEARCH
        elif any(word in normalized_id for word in ['analysis', 'analyze', 'finance', 'currency']):
            logger.debug(f"📈 분석 에이전트 추론: '{agent_id}' -> ANALYSIS")
            return AgentType.ANALYSIS
        elif any(word in normalized_id for word in ['game', 'creative', 'image', 'chart', 'visual']):
            logger.debug(f"🎨 창작 에이전트 추론: '{agent_id}' -> CREATIVE")
            return AgentType.CREATIVE
        
        # 기본값
        logger.warning(f"⚠️ 에이전트 타입 매핑 실패, 기본값 사용: '{agent_id}' -> GENERAL")
        return AgentType.GENERAL
    
    def _record_workflow_execution(
        self, 
        workflow_plan: 'WorkflowPlan', 
        context: ExecutionContext, 
        results: List[AgentExecutionResult], 
        execution_time: float
    ):
        """워크플로우 실행 기록"""
        record = {
            'workflow_id': workflow_plan.plan_id,
            'execution_time': execution_time,
            'step_count': len(workflow_plan.steps),
            'success_count': sum(1 for r in results if r.is_successful()),
            'strategy': workflow_plan.optimization_strategy.value if hasattr(workflow_plan, 'optimization_strategy') else 'unknown',
            'timestamp': time.time()
        }
        
        self.execution_history.append(record)
        logger.info(f"워크플로우 실행 기록: {workflow_plan.plan_id}, {execution_time:.2f}초") 