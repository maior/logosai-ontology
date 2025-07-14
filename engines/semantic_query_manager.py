"""
🧠 Semantic Query Manager
의미론적 쿼리 중앙 관리자

중복 호출 문제를 해결하고 효율적인 쿼리 분석을 제공합니다.
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from loguru import logger

from ..core.models import SemanticQuery, CachedSemanticQuery, SystemMetrics, ComplexityAnalysis, ExecutionStrategy, AgentType
from ..core.interfaces import QueryAnalyzer, CacheManager


class InMemoryCacheManager(CacheManager):
    """인메모리 캐시 관리자"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 1800):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.ttls: Dict[str, datetime] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        # TTL 확인
        if key in self.ttls and datetime.now() > self.ttls[key]:
            await self._evict(key)
            self.stats["misses"] += 1
            return None
        
        # 접근 시간 업데이트
        self.access_times[key] = datetime.now()
        self.stats["hits"] += 1
        return self.cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        try:
            # 캐시 크기 확인 및 정리
            if len(self.cache) >= self.max_size:
                await self._cleanup_lru()
            
            self.cache[key] = value
            self.access_times[key] = datetime.now()
            
            # TTL 설정
            if ttl is not None:
                self.ttls[key] = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                self.ttls[key] = datetime.now() + timedelta(seconds=self.default_ttl)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False
    
    async def invalidate(self, pattern: str = None) -> int:
        """캐시 무효화"""
        if pattern is None:
            # 전체 캐시 클리어
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            self.ttls.clear()
            return count
        
        # 패턴 매칭으로 삭제
        keys_to_delete = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_delete:
            await self._evict(key)
        
        return len(keys_to_delete)
    
    async def delete(self, key: str) -> bool:
        """캐시에서 특정 키 삭제 (추상 메서드 구현)"""
        if key in self.cache:
            await self._evict(key)
            return True
        return False
    
    async def clear(self) -> bool:
        """캐시 전체 삭제 (추상 메서드 구현)"""
        self.cache.clear()
        self.access_times.clear()
        self.ttls.clear()
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            **self.stats
        }
    
    async def _evict(self, key: str):
        """키 제거"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.ttls.pop(key, None)
        self.stats["evictions"] += 1
    
    async def _cleanup_lru(self):
        """LRU 기반 정리"""
        if not self.access_times:
            return
        
        # 가장 오래된 항목 찾기
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self._evict(oldest_key)


class LLMSemanticQueryAnalyzer(QueryAnalyzer):
    """LLM 기반 의미론적 쿼리 분석기"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.fallback_patterns = self._initialize_fallback_patterns()
    
    async def analyze_query(self, query_text: str, context: Dict[str, Any] = None) -> SemanticQuery:
        """쿼리 분석 및 SemanticQuery 객체 생성 (추상 메서드 구현)"""
        return await self.analyze_semantic_query(query_text)
    
    def estimate_complexity(self, query: SemanticQuery) -> float:
        """쿼리 복잡도 추정 (추상 메서드 구현)"""
        complexity_analysis = self.analyze_complexity(query)
        return complexity_analysis.complexity_score / 10.0  # 0.0 ~ 1.0 범위로 정규화
    
    def suggest_agents(self, query: SemanticQuery) -> List[AgentType]:
        """쿼리에 적합한 에이전트 추천 (추상 메서드 구현)"""
        from ..core.models import AgentType
        
        # 쿼리 텍스트 기반 에이전트 추천
        text_lower = query.natural_language.lower()
        suggested_agents = []
        
        # 도메인별 에이전트 매핑
        if any(word in text_lower for word in ["분석", "데이터", "통계", "평가"]):
            suggested_agents.append(AgentType.ANALYSIS)
        
        if any(word in text_lower for word in ["연구", "조사", "정보", "검색"]):
            suggested_agents.append(AgentType.RESEARCH)
        
        if any(word in text_lower for word in ["기술", "개발", "프로그래밍", "코딩"]):
            suggested_agents.append(AgentType.TECHNICAL)
        
        if any(word in text_lower for word in ["창작", "디자인", "아이디어", "브레인스토밍"]):
            suggested_agents.append(AgentType.CREATIVE)
        
        # 기본값으로 GENERAL 에이전트 추가
        if not suggested_agents:
            suggested_agents.append(AgentType.GENERAL)
        
        return suggested_agents
    
    async def analyze_semantic_query(self, text: str) -> SemanticQuery:
        """텍스트를 의미론적 쿼리로 분석"""
        try:
            if self.llm_client:
                return await self._analyze_with_llm(text)
            else:
                return self._analyze_with_patterns(text)
                
        except Exception as e:
            logger.error(f"의미론적 쿼리 분석 실패: {e}")
            return self._create_fallback_query(text)
    
    def analyze_complexity(self, semantic_query: SemanticQuery) -> 'ComplexityAnalysis':
        """쿼리 복잡도 분석"""
        from ..core.models import ComplexityAnalysis, ExecutionStrategy
        
        query_text = semantic_query.natural_language.lower()
        
        # 복잡도 지표들
        indicators = {
            "agent_count": len(semantic_query.structured_query.get("required_agents", [])),
            "concept_count": len(semantic_query.concepts),
            "entity_count": len(semantic_query.entities),
            "has_comparison": any(word in query_text for word in ["비교", "차이", "대비", "vs"]),
            "has_analysis": any(word in query_text for word in ["분석", "평가", "검토", "조사"]),
            "has_multiple_tasks": "그리고" in query_text or "또한" in query_text or "," in query_text,
            "has_time_dependency": any(word in query_text for word in ["먼저", "다음", "그 후", "이후"]),
            "requires_data_processing": any(word in query_text for word in ["계산", "차트", "그래프", "표"])
        }
        
        # 복잡도 점수 계산
        complexity_score = 0
        complexity_score += indicators["agent_count"] * 2
        complexity_score += indicators["concept_count"] * 0.5
        complexity_score += indicators["entity_count"] * 0.3
        complexity_score += 3 if indicators["has_comparison"] else 0
        complexity_score += 2 if indicators["has_analysis"] else 0
        complexity_score += 2 if indicators["has_multiple_tasks"] else 0
        complexity_score += 3 if indicators["has_time_dependency"] else 0
        complexity_score += 2 if indicators["requires_data_processing"] else 0
        
        # 실행 전략 결정
        if complexity_score <= 2:
            strategy = ExecutionStrategy.SINGLE_AGENT
        elif complexity_score <= 5:
            strategy = ExecutionStrategy.SEQUENTIAL
        elif indicators["has_time_dependency"]:
            strategy = ExecutionStrategy.SEQUENTIAL
        elif indicators["has_multiple_tasks"] and not indicators["has_time_dependency"]:
            strategy = ExecutionStrategy.PARALLEL
        else:
            strategy = ExecutionStrategy.HYBRID
        
        # 병렬 처리 가능성 분석
        parallel_potential = self._analyze_parallel_potential(semantic_query)
        
        return ComplexityAnalysis(
            complexity_score=complexity_score,
            strategy=strategy,
            indicators=indicators,
            parallel_potential=parallel_potential,
            estimated_agents=indicators["agent_count"],
            estimated_time=complexity_score * 5,
            reasoning=self._generate_reasoning(strategy, indicators)
        )
    
    async def _analyze_with_llm(self, text: str) -> SemanticQuery:
        """LLM을 사용한 분석"""
        # LLM 분석 로직 (실제 구현 시 LLM 클라이언트 사용)
        prompt = f"""
        다음 텍스트를 분석하여 의미론적 쿼리로 변환해주세요:
        "{text}"
        
        다음 형식으로 응답해주세요:
        {{
            "intent": "의도",
            "entities": ["엔티티1", "엔티티2"],
            "concepts": ["개념1", "개념2"],
            "relations": ["관계1", "관계2"],
            "required_agents": ["에이전트1", "에이전트2"]
        }}
        """
        
        # 실제 LLM 호출 (여기서는 패턴 기반 분석으로 대체)
        return self._analyze_with_patterns(text)
    
    def _analyze_with_patterns(self, text: str) -> SemanticQuery:
        """패턴 기반 분석"""
        text_lower = text.lower()
        
        # 의도 분석
        intent = "information_retrieval"
        if any(word in text_lower for word in ["분석", "평가", "검토"]):
            intent = "analysis"
        elif any(word in text_lower for word in ["비교", "차이"]):
            intent = "comparison"
        elif any(word in text_lower for word in ["계산", "산출"]):
            intent = "calculation"
        
        # 엔티티 추출
        entities = []
        for pattern, entity_list in self.fallback_patterns["entities"].items():
            if any(word in text_lower for word in pattern.split("|")):
                entities.extend(entity_list)
        
        # 개념 추출
        concepts = []
        for pattern, concept_list in self.fallback_patterns["concepts"].items():
            if any(word in text_lower for word in pattern.split("|")):
                concepts.extend(concept_list)
        
        # 필요한 에이전트 추정
        required_agents = self._estimate_required_agents(text_lower)
        
        return SemanticQuery.create_from_text(
            text,
            intent=intent,
            entities=list(set(entities)),
            concepts=list(set(concepts)),
            relations=[f"requires_{concept}" for concept in concepts],
            structured_query={
                "required_agents": required_agents,
                "complexity_level": "moderate",
                "primary_domain": self._detect_domain(text_lower)
            }
        )
    
    def _create_fallback_query(self, text: str) -> SemanticQuery:
        """폴백 쿼리 생성"""
        return SemanticQuery.create_from_text(
            text,
            intent="information_retrieval",
            entities=[text[:50]],
            concepts=["general_processing"],
            relations=["requires_general_processing"],
            structured_query={
                "required_agents": ["internet_agent"],
                "complexity_level": "simple",
                "primary_domain": "general",
                "fallback_generated": True
            }
        )
    
    def _initialize_fallback_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """폴백 패턴 초기화"""
        return {
            "entities": {
                "finance|환율|주가|금융": ["currency", "stock", "finance"],
                "weather|날씨|기상": ["weather", "temperature", "forecast"],
                "calculation|계산|수학": ["numbers", "calculation", "math"],
                "chart|그래프|시각화": ["chart", "graph", "visualization"]
            },
            "concepts": {
                "analysis|분석": ["data_analysis", "research"],
                "comparison|비교": ["comparison", "evaluation"],
                "information|정보": ["information_retrieval", "search"],
                "creation|생성|작성": ["content_creation", "generation"]
            }
        }
    
    def _estimate_required_agents(self, text: str) -> List[str]:
        """필요한 에이전트 추정"""
        agents = []
        
        # 도메인별 에이전트 매핑
        if any(word in text for word in ["환율", "주가", "금융", "투자"]):
            agents.append("finance_agent")
        
        if any(word in text for word in ["날씨", "기상", "온도"]):
            agents.append("weather_agent")
        
        if any(word in text for word in ["계산", "수학", "산출"]):
            agents.append("calculate_agent")
        
        if any(word in text for word in ["차트", "그래프", "시각화"]):
            agents.append("chart_agent")
        
        if any(word in text for word in ["메모", "저장", "기록"]):
            agents.append("memo_agent")
        
        if any(word in text for word in ["검색", "찾기", "정보", "최신"]):
            agents.append("internet_agent")
        
        # 기본 에이전트
        if not agents:
            agents.append("internet_agent")
        
        return agents
    
    def _detect_domain(self, text: str) -> str:
        """도메인 감지"""
        if any(word in text for word in ["환율", "주가", "금융"]):
            return "finance"
        elif any(word in text for word in ["날씨", "기상"]):
            return "weather"
        elif any(word in text for word in ["계산", "수학"]):
            return "calculation"
        elif any(word in text for word in ["분석", "연구"]):
            return "analysis"
        else:
            return "general"
    
    def _analyze_parallel_potential(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """병렬 처리 가능성 분석"""
        required_agents = semantic_query.structured_query.get("required_agents", [])
        
        # 에이전트 간 의존성 분석
        agent_dependencies = {
            "internet_agent": [],
            "finance_agent": [],
            "weather_agent": [],
            "calculate_agent": ["finance_agent", "internet_agent"],
            "chart_agent": ["internet_agent", "finance_agent", "calculate_agent"],
            "memo_agent": ["internet_agent", "analysis_agent", "chart_agent"],
            "analysis_agent": ["internet_agent"]
        }
        
        # 독립 에이전트 식별
        independent_agents = [agent for agent in required_agents 
                            if not agent_dependencies.get(agent, [])]
        
        # 의존성 체인 구성
        dependent_chains = []
        for agent in required_agents:
            if agent not in independent_agents:
                dependencies = agent_dependencies.get(agent, [])
                available_deps = [dep for dep in dependencies if dep in required_agents]
                if available_deps:
                    dependent_chains.append({
                        "agent": agent,
                        "dependencies": available_deps
                    })
        
        return {
            "independent_agents": independent_agents,
            "dependent_chains": dependent_chains,
            "parallel_groups": len(independent_agents) > 1,
            "max_parallelism": len(independent_agents),
            "sequential_depth": len(dependent_chains)
        }
    
    def _generate_reasoning(self, strategy: 'ExecutionStrategy', indicators: Dict[str, Any]) -> str:
        """전략 선택 이유 생성"""
        if strategy.value == "single_agent":
            return f"단순한 쿼리로 단일 에이전트로 충분합니다."
        elif strategy.value == "sequential":
            reasons = []
            if indicators["has_time_dependency"]:
                reasons.append("시간적 의존성")
            if indicators["requires_data_processing"]:
                reasons.append("데이터 처리 필요")
            return f"순차 처리 필요: {', '.join(reasons)}"
        elif strategy.value == "parallel":
            return "독립적인 작업들이 있어 병렬 처리 가능"
        else:
            return "복잡한 워크플로우로 하이브리드 전략 필요"


class SemanticQueryManager:
    """🧠 SemanticQuery 중앙 관리자"""
    
    def __init__(self, query_analyzer: QueryAnalyzer = None, cache_manager: CacheManager = None):
        self.query_analyzer = query_analyzer or LLMSemanticQueryAnalyzer()
        self.cache_manager = cache_manager or InMemoryCacheManager()
        
        # 세션별 캐시
        self._session_cache: Dict[str, SemanticQuery] = {}
        
        # 동시성 제어
        self._analysis_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        
        # 메트릭스
        self.metrics = SystemMetrics()
        
        # 설정
        self.cache_ttl_minutes = 30
        self.max_cache_size = 100
        self.cleanup_interval_minutes = 10
        
        logger.info("🧠 SemanticQuery 중앙 관리자 초기화 완료")
    
    async def get_semantic_query(self, 
                               query_text: str, 
                               session_id: str = None,
                               force_refresh: bool = False) -> SemanticQuery:
        """
        의미론적 쿼리 조회 (캐시 우선)
        
        Args:
            query_text: 쿼리 텍스트
            session_id: 세션 ID (선택적)
            force_refresh: 강제 새로고침 여부
        
        Returns:
            SemanticQuery 객체
        """
        start_time = time.time()
        
        try:
            # 강제 새로고침이 아닌 경우 캐시 확인
            if not force_refresh:
                # 1. 세션별 캐시 확인
                if session_id:
                    cached_query = await self._get_session_cached_query(session_id, query_text)
                    if cached_query:
                        logger.debug(f"세션 캐시 히트: {session_id}")
                        self.metrics.cache_hits += 1
                        return cached_query
                
                # 2. 글로벌 캐시 확인
                cached_query = await self._get_cached_query(query_text)
                if cached_query:
                    logger.debug(f"글로벌 캐시 히트: {query_text[:50]}...")
                    self.metrics.cache_hits += 1
                    return cached_query
            
            # 3. 캐시 미스 - 새로운 분석 수행
            logger.debug(f"캐시 미스 - 새로운 분석 수행: {query_text[:50]}...")
            self.metrics.cache_misses += 1
            
            semantic_query = await self._perform_analysis(query_text)
            
            # 4. 캐시에 저장
            await self._store_cached_query(query_text, semantic_query)
            if session_id:
                await self._store_session_cache(session_id, query_text, semantic_query)
            
            # 5. 메트릭스 업데이트
            analysis_time = time.time() - start_time
            self._update_analysis_metrics(analysis_time)
            
            logger.debug(f"의미론적 쿼리 분석 완료: {analysis_time:.3f}초")
            return semantic_query
            
        except Exception as e:
            logger.error(f"의미론적 쿼리 조회 실패: {e}")
            # 폴백 쿼리 반환
            return self._create_fallback_semantic_query(query_text)
    
    async def create_semantic_query(self, query_text: str, execution_context=None) -> SemanticQuery:
        """
        새로운 의미론적 쿼리 생성
        
        Args:
            query_text: 쿼리 텍스트
            execution_context: 실행 컨텍스트 (선택적)
        
        Returns:
            SemanticQuery 객체
        """
        try:
            # 실행 컨텍스트에서 세션 ID 추출
            session_id = None
            if execution_context and hasattr(execution_context, 'session_id'):
                session_id = execution_context.session_id
            
            # 기존 get_semantic_query 메서드 활용
            return await self.get_semantic_query(query_text, session_id)
            
        except Exception as e:
            logger.error(f"의미론적 쿼리 생성 실패: {e}")
            return self._create_fallback_semantic_query(query_text)
    
    async def _get_session_cached_query(self, session_id: str, query_text: str) -> Optional[SemanticQuery]:
        """세션 캐시에서 쿼리 조회"""
        cache_key = f"session_{session_id}_{self._generate_query_hash(query_text)}"
        return self._session_cache.get(cache_key)
    
    async def _get_cached_query(self, query_text: str) -> Optional[SemanticQuery]:
        """글로벌 캐시에서 쿼리 조회"""
        cache_key = f"query_{self._generate_query_hash(query_text)}"
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data and isinstance(cached_data, CachedSemanticQuery):
            if not cached_data.is_expired(self.cache_ttl_minutes):
                cached_data.update_access()
                return cached_data.query
        
        return None
    
    async def _perform_analysis(self, query_text: str) -> SemanticQuery:
        """실제 SemanticQuery 분석 수행"""
        async with self._analysis_lock:
            start_time = time.time()
            self.metrics.analysis_calls += 1
            
            try:
                logger.info(f"🔍 새로운 SemanticQuery 분석 시작: {query_text[:50]}...")
                
                if self.query_analyzer:
                    semantic_query = await self.query_analyzer.analyze_semantic_query(query_text)
                else:
                    semantic_query = self._create_fallback_semantic_query(query_text)
                
                analysis_time = time.time() - start_time
                self._update_analysis_metrics(analysis_time)
                
                logger.info(f"✅ SemanticQuery 분석 완료 ({analysis_time:.3f}초): {semantic_query.intent}")
                return semantic_query
                
            except Exception as e:
                analysis_time = time.time() - start_time
                logger.error(f"❌ SemanticQuery 분석 실패 ({analysis_time:.3f}초): {e}")
                return self._create_fallback_semantic_query(query_text)
    
    async def _store_cached_query(self, query_text: str, semantic_query: SemanticQuery):
        """글로벌 캐시에 쿼리 저장"""
        cache_key = f"query_{self._generate_query_hash(query_text)}"
        cached_query = CachedSemanticQuery(
            query=semantic_query,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            query_hash=self._generate_query_hash(query_text)
        )
        
        await self.cache_manager.set(cache_key, cached_query, ttl=self.cache_ttl_minutes * 60)
    
    async def _store_session_cache(self, session_id: str, query_text: str, semantic_query: SemanticQuery):
        """세션 캐시에 쿼리 저장"""
        cache_key = f"session_{session_id}_{self._generate_query_hash(query_text)}"
        self._session_cache[cache_key] = semantic_query
        
        # 세션 캐시 크기 제한
        if len(self._session_cache) > 50:
            # 가장 오래된 항목 제거
            oldest_key = min(self._session_cache.keys())
            del self._session_cache[oldest_key]
    
    def _generate_query_hash(self, query_text: str) -> str:
        """쿼리 해시 생성"""
        return hashlib.md5(query_text.encode('utf-8')).hexdigest()
    
    def _update_analysis_metrics(self, analysis_time: float):
        """분석 메트릭스 업데이트"""
        if self.metrics.analysis_calls == 1:
            self.metrics.average_execution_time = analysis_time
        else:
            current_avg = self.metrics.average_execution_time
            total_calls = self.metrics.analysis_calls
            self.metrics.average_execution_time = (
                (current_avg * (total_calls - 1) + analysis_time) / total_calls
            )
    
    def _create_fallback_semantic_query(self, query_text: str) -> SemanticQuery:
        """폴백 SemanticQuery 생성"""
        return SemanticQuery.create_from_text(
            query_text,
            intent="information_retrieval",
            entities=[query_text[:50]],
            concepts=["general_processing"],
            relations=["requires_general_processing"],
            structured_query={
                "required_agents": ["internet_agent"],
                "complexity_level": "simple",
                "primary_domain": "general",
                "fallback_generated": True
            }
        )
    
    async def invalidate_cache(self, pattern: str = None) -> int:
        """캐시 무효화"""
        async with self._cache_lock:
            # 글로벌 캐시 무효화
            global_count = await self.cache_manager.invalidate(pattern)
            
            # 세션 캐시 무효화
            if pattern is None:
                session_count = len(self._session_cache)
                self._session_cache.clear()
            else:
                keys_to_delete = [key for key in self._session_cache.keys() if pattern in key]
                for key in keys_to_delete:
                    del self._session_cache[key]
                session_count = len(keys_to_delete)
            
            logger.info(f"캐시 무효화 완료: 글로벌 {global_count}개, 세션 {session_count}개")
            return global_count + session_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            "global_cache": cache_stats,
            "session_cache_size": len(self._session_cache),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_ratio": self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0.0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭스 조회"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            **self.metrics.to_dict(),
            "cache_stats": cache_stats,
            "session_cache_size": len(self._session_cache)
        }
    
    def get_complexity_analysis(self, query_text: str) -> 'ComplexityAnalysis':
        """복잡도 분석 (동기 버전)"""
        # 간단한 SemanticQuery 생성 후 분석
        simple_query = SemanticQuery.create_from_text(query_text)
        return self.query_analyzer.analyze_complexity(simple_query) 