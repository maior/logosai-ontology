"""
🧠 Semantic Query Manager

Solves duplicate call problems and provides efficient query analysis.
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
    """In-memory cache manager"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 1800):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.ttls: Dict[str, datetime] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}
    
    async def get(self, key: str) -> Optional[Any]:
        """Look up a value in the cache"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        # Check TTL
        if key in self.ttls and datetime.now() > self.ttls[key]:
            await self._evict(key)
            self.stats["misses"] += 1
            return None

        # Update access time
        self.access_times[key] = datetime.now()
        self.stats["hits"] += 1
        return self.cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in the cache"""
        try:
            # Check and clean cache size
            if len(self.cache) >= self.max_size:
                await self._cleanup_lru()

            self.cache[key] = value
            self.access_times[key] = datetime.now()

            # Set TTL
            if ttl is not None:
                self.ttls[key] = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                self.ttls[key] = datetime.now() + timedelta(seconds=self.default_ttl)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache store failed: {e}")
            return False
    
    async def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache"""
        if pattern is None:
            # Clear entire cache
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            self.ttls.clear()
            return count
        
        # Delete by pattern matching
        keys_to_delete = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_delete:
            await self._evict(key)
        
        return len(keys_to_delete)
    
    async def delete(self, key: str) -> bool:
        """Delete a specific key from the cache (abstract method implementation)"""
        if key in self.cache:
            await self._evict(key)
            return True
        return False
    
    async def clear(self) -> bool:
        """Delete entire cache (abstract method implementation)"""
        self.cache.clear()
        self.access_times.clear()
        self.ttls.clear()
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
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
        """Evict a key"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.ttls.pop(key, None)
        self.stats["evictions"] += 1
    
    async def _cleanup_lru(self):
        """LRU-based cleanup"""
        if not self.access_times:
            return

        # Find oldest entry
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self._evict(oldest_key)


class LLMSemanticQueryAnalyzer(QueryAnalyzer):
    """LLM-based semantic query analyzer"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.fallback_patterns = self._initialize_fallback_patterns()
    
    async def analyze_query(self, query_text: str, context: Dict[str, Any] = None) -> SemanticQuery:
        """Analyze query and create SemanticQuery object (abstract method implementation)"""
        return await self.analyze_semantic_query(query_text)
    
    def estimate_complexity(self, query: SemanticQuery) -> float:
        """Estimate query complexity (abstract method implementation)"""
        complexity_analysis = self.analyze_complexity(query)
        return complexity_analysis.complexity_score / 10.0  # Normalize to range 0.0 ~ 1.0
    
    def suggest_agents(self, query: SemanticQuery) -> List[AgentType]:
        """Recommend suitable agents for the query (abstract method implementation)"""
        from ..core.models import AgentType

        # Recommend agents based on query text
        text_lower = query.natural_language.lower()
        suggested_agents = []

        # Domain-based agent mapping
        if any(word in text_lower for word in ["분석", "데이터", "통계", "평가"]):
            suggested_agents.append(AgentType.ANALYSIS)

        if any(word in text_lower for word in ["연구", "조사", "정보", "검색"]):
            suggested_agents.append(AgentType.RESEARCH)

        if any(word in text_lower for word in ["기술", "개발", "프로그래밍", "코딩"]):
            suggested_agents.append(AgentType.TECHNICAL)

        if any(word in text_lower for word in ["창작", "디자인", "아이디어", "브레인스토밍"]):
            suggested_agents.append(AgentType.CREATIVE)

        # Add GENERAL agent as default
        if not suggested_agents:
            suggested_agents.append(AgentType.GENERAL)
        
        return suggested_agents
    
    async def analyze_semantic_query(self, text: str) -> SemanticQuery:
        """Analyze text as a semantic query"""
        try:
            if self.llm_client:
                return await self._analyze_with_llm(text)
            else:
                return self._analyze_with_patterns(text)

        except Exception as e:
            logger.error(f"Semantic query analysis failed: {e}")
            return self._create_fallback_query(text)
    
    def analyze_complexity(self, semantic_query: SemanticQuery) -> 'ComplexityAnalysis':
        """Analyze query complexity"""
        from ..core.models import ComplexityAnalysis, ExecutionStrategy

        query_text = semantic_query.natural_language.lower()

        # Complexity indicators
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
        
        # Calculate complexity score
        complexity_score = 0
        complexity_score += indicators["agent_count"] * 2
        complexity_score += indicators["concept_count"] * 0.5
        complexity_score += indicators["entity_count"] * 0.3
        complexity_score += 3 if indicators["has_comparison"] else 0
        complexity_score += 2 if indicators["has_analysis"] else 0
        complexity_score += 2 if indicators["has_multiple_tasks"] else 0
        complexity_score += 3 if indicators["has_time_dependency"] else 0
        complexity_score += 2 if indicators["requires_data_processing"] else 0
        
        # Determine execution strategy
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
        
        # Analyze parallel processing potential
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
        """Analysis using LLM"""
        # LLM analysis logic (use LLM client in actual implementation)
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

        # Actual LLM call (replaced with pattern-based analysis here)
        return self._analyze_with_patterns(text)
    
    def _analyze_with_patterns(self, text: str) -> SemanticQuery:
        """Pattern-based analysis"""
        text_lower = text.lower()

        # Intent analysis
        intent = "information_retrieval"
        if any(word in text_lower for word in ["분석", "평가", "검토"]):
            intent = "analysis"
        elif any(word in text_lower for word in ["비교", "차이"]):
            intent = "comparison"
        elif any(word in text_lower for word in ["계산", "산출"]):
            intent = "calculation"

        # Entity extraction
        entities = []
        for pattern, entity_list in self.fallback_patterns["entities"].items():
            if any(word in text_lower for word in pattern.split("|")):
                entities.extend(entity_list)

        # Concept extraction
        concepts = []
        for pattern, concept_list in self.fallback_patterns["concepts"].items():
            if any(word in text_lower for word in pattern.split("|")):
                concepts.extend(concept_list)

        # Estimate required agents
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
        """Create fallback query"""
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
        """Initialize fallback patterns"""
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
        """Estimate required agents"""
        agents = []

        # Domain-based agent mapping
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

        # Default agent
        if not agents:
            agents.append("internet_agent")
        
        return agents
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain"""
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
        """Analyze parallel processing potential"""
        required_agents = semantic_query.structured_query.get("required_agents", [])

        # Analyze inter-agent dependencies
        agent_dependencies = {
            "internet_agent": [],
            "finance_agent": [],
            "weather_agent": [],
            "calculate_agent": ["finance_agent", "internet_agent"],
            "chart_agent": ["internet_agent", "finance_agent", "calculate_agent"],
            "memo_agent": ["internet_agent", "analysis_agent", "chart_agent"],
            "analysis_agent": ["internet_agent"]
        }
        
        # Identify independent agents
        independent_agents = [agent for agent in required_agents
                            if not agent_dependencies.get(agent, [])]

        # Build dependency chains
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
        """Generate reasoning for strategy selection"""
        if strategy.value == "single_agent":
            return "Simple query - a single agent is sufficient."
        elif strategy.value == "sequential":
            reasons = []
            if indicators["has_time_dependency"]:
                reasons.append("temporal dependency")
            if indicators["requires_data_processing"]:
                reasons.append("data processing required")
            return f"Sequential processing needed: {', '.join(reasons)}"
        elif strategy.value == "parallel":
            return "Independent tasks allow parallel processing"
        else:
            return "Complex workflow requires hybrid strategy"


class SemanticQueryManager:
    """🧠 SemanticQuery Central Manager"""

    def __init__(self, query_analyzer: QueryAnalyzer = None, cache_manager: CacheManager = None):
        self.query_analyzer = query_analyzer or LLMSemanticQueryAnalyzer()
        self.cache_manager = cache_manager or InMemoryCacheManager()

        # Per-session cache
        self._session_cache: Dict[str, SemanticQuery] = {}

        # Concurrency control
        self._analysis_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()

        # Metrics
        self.metrics = SystemMetrics()

        # Settings
        self.cache_ttl_minutes = 30
        self.max_cache_size = 100
        self.cleanup_interval_minutes = 10

        logger.info("🧠 SemanticQuery central manager initialized")
    
    async def get_semantic_query(self,
                               query_text: str,
                               session_id: str = None,
                               force_refresh: bool = False) -> SemanticQuery:
        """
        Look up semantic query (cache-first)

        Args:
            query_text: Query text
            session_id: Session ID (optional)
            force_refresh: Whether to force refresh

        Returns:
            SemanticQuery object
        """
        start_time = time.time()

        try:
            # Check cache if not force refresh
            if not force_refresh:
                # 1. Check per-session cache
                if session_id:
                    cached_query = await self._get_session_cached_query(session_id, query_text)
                    if cached_query:
                        logger.debug(f"Session cache hit: {session_id}")
                        self.metrics.cache_hits += 1
                        return cached_query

                # 2. Check global cache
                cached_query = await self._get_cached_query(query_text)
                if cached_query:
                    logger.debug(f"Global cache hit: {query_text[:50]}...")
                    self.metrics.cache_hits += 1
                    return cached_query

            # 3. Cache miss - perform new analysis
            logger.debug(f"Cache miss - performing new analysis: {query_text[:50]}...")
            self.metrics.cache_misses += 1

            semantic_query = await self._perform_analysis(query_text)

            # 4. Store in cache
            await self._store_cached_query(query_text, semantic_query)
            if session_id:
                await self._store_session_cache(session_id, query_text, semantic_query)

            # 5. Update metrics
            analysis_time = time.time() - start_time
            self._update_analysis_metrics(analysis_time)

            logger.debug(f"Semantic query analysis completed: {analysis_time:.3f}s")
            return semantic_query

        except Exception as e:
            logger.error(f"Semantic query lookup failed: {e}")
            # Return fallback query
            return self._create_fallback_semantic_query(query_text)
    
    async def create_semantic_query(self, query_text: str, execution_context=None) -> SemanticQuery:
        """
        Create a new semantic query

        Args:
            query_text: Query text
            execution_context: Execution context (optional)

        Returns:
            SemanticQuery object
        """
        try:
            # Extract session ID from execution context
            session_id = None
            if execution_context and hasattr(execution_context, 'session_id'):
                session_id = execution_context.session_id

            # Use existing get_semantic_query method
            return await self.get_semantic_query(query_text, session_id)

        except Exception as e:
            logger.error(f"Semantic query creation failed: {e}")
            return self._create_fallback_semantic_query(query_text)
    
    async def _get_session_cached_query(self, session_id: str, query_text: str) -> Optional[SemanticQuery]:
        """Retrieve query from session cache"""
        cache_key = f"session_{session_id}_{self._generate_query_hash(query_text)}"
        return self._session_cache.get(cache_key)

    async def _get_cached_query(self, query_text: str) -> Optional[SemanticQuery]:
        """Retrieve query from global cache"""
        cache_key = f"query_{self._generate_query_hash(query_text)}"
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data and isinstance(cached_data, CachedSemanticQuery):
            if not cached_data.is_expired(self.cache_ttl_minutes):
                cached_data.update_access()
                return cached_data.query
        
        return None
    
    async def _perform_analysis(self, query_text: str) -> SemanticQuery:
        """Perform actual SemanticQuery analysis"""
        async with self._analysis_lock:
            start_time = time.time()
            self.metrics.analysis_calls += 1

            try:
                logger.info(f"🔍 Starting new SemanticQuery analysis: {query_text[:50]}...")

                if self.query_analyzer:
                    semantic_query = await self.query_analyzer.analyze_semantic_query(query_text)
                else:
                    semantic_query = self._create_fallback_semantic_query(query_text)

                analysis_time = time.time() - start_time
                self._update_analysis_metrics(analysis_time)

                logger.info(f"✅ SemanticQuery analysis completed ({analysis_time:.3f}s): {semantic_query.intent}")
                return semantic_query

            except Exception as e:
                analysis_time = time.time() - start_time
                logger.error(f"❌ SemanticQuery analysis failed ({analysis_time:.3f}s): {e}")
                return self._create_fallback_semantic_query(query_text)
    
    async def _store_cached_query(self, query_text: str, semantic_query: SemanticQuery):
        """Store query in global cache"""
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
        """Store query in session cache"""
        cache_key = f"session_{session_id}_{self._generate_query_hash(query_text)}"
        self._session_cache[cache_key] = semantic_query

        # Limit session cache size
        if len(self._session_cache) > 50:
            # Remove the oldest entry
            oldest_key = min(self._session_cache.keys())
            del self._session_cache[oldest_key]
    
    def _generate_query_hash(self, query_text: str) -> str:
        """Generate query hash"""
        return hashlib.md5(query_text.encode('utf-8')).hexdigest()

    def _update_analysis_metrics(self, analysis_time: float):
        """Update analysis metrics"""
        if self.metrics.analysis_calls == 1:
            self.metrics.average_execution_time = analysis_time
        else:
            current_avg = self.metrics.average_execution_time
            total_calls = self.metrics.analysis_calls
            self.metrics.average_execution_time = (
                (current_avg * (total_calls - 1) + analysis_time) / total_calls
            )
    
    def _create_fallback_semantic_query(self, query_text: str) -> SemanticQuery:
        """Create fallback SemanticQuery"""
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
        """Invalidate cache"""
        async with self._cache_lock:
            # Invalidate global cache
            global_count = await self.cache_manager.invalidate(pattern)

            # Invalidate session cache
            if pattern is None:
                session_count = len(self._session_cache)
                self._session_cache.clear()
            else:
                keys_to_delete = [key for key in self._session_cache.keys() if pattern in key]
                for key in keys_to_delete:
                    del self._session_cache[key]
                session_count = len(keys_to_delete)

            logger.info(f"Cache invalidation complete: global {global_count}, session {session_count}")
            return global_count + session_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retrieve cache statistics"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            "global_cache": cache_stats,
            "session_cache_size": len(self._session_cache),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_ratio": self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0.0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve metrics"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            **self.metrics.to_dict(),
            "cache_stats": cache_stats,
            "session_cache_size": len(self._session_cache)
        }
    
    def get_complexity_analysis(self, query_text: str) -> 'ComplexityAnalysis':
        """Complexity analysis (synchronous version)"""
        # Create a simple SemanticQuery and analyze it
        simple_query = SemanticQuery.create_from_text(query_text)
        return self.query_analyzer.analyze_complexity(simple_query)