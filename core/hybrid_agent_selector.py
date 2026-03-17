"""
🧠 Hybrid Agent Selector v3.0
GNN + RL + Knowledge Graph + LLM integrated agent selector

Phase 0: GNN+RL Selection (NEW in v3.0)
  - Learns Knowledge Graph structure with Graph Neural Network
  - Optimizes agent selection policy with Reinforcement Learning (PPO)
  - Directly selects when confidence is high (>0.8)

Phase 1: Knowledge Graph Analysis
  - Entity extraction and related concept exploration
  - Query past success patterns (with time decay)
  - Graph-based candidate recommendation

Phase 2: LLM Final Decision
  - Utilizes graph insights as context
  - Final decision combining semantic analysis + graph evidence

Phase 3: Feedback Loop
  - Stores successful query-agent mappings in the graph
  - Stores to GNN+RL experience buffer (for training)
  - LLM-based semantic query analysis (no hardcoding)
  - Applies Time Decay
  - Pattern generalization learning (category-based)

v3.0 Updates (2026-02-01):
  - GNN+RL integration: IntelligentAgentSelector connected
  - 3-stage selection: GNN+RL → KG → LLM
  - Confidence-based fallback: proceeds to next stage on low confidence

v2.0 Updates (2026-01-31):
  - Time decay: higher weight for recent patterns
  - LLM-based semantic matching: no hardcoded keywords
  - Pattern generalization: query category-based learning
"""

import asyncio
import json
import math
import re
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .llm_manager import get_ontology_llm_manager, OntologyLLMType


# Time decay configuration
TIME_DECAY_CONFIG = {
    "half_life_days": 30,  # Weight decreases to 50% after 30 days
    "min_weight": 0.1,     # Minimum weight (maintains at least 10% no matter how old)
    "max_age_days": 365    # Patterns older than 1 year get minimum weight
}


class HybridAgentSelector:
    """
    🧠 Hybrid Agent Selector v3.0

    Selects the optimal agent by combining GNN+RL + Knowledge Graph + LLM.

    Features:
    - GNN+RL based intelligent agent selection (v3.0)
    - Auto-sync with Agent Marketplace on startup
    - Knowledge graph based pattern learning
    - LLM-based semantic analysis
    """

    # GNN+RL enable flag and confidence threshold
    USE_GNN_RL = True  # GNN+RL enabled
    GNN_RL_CONFIDENCE_THRESHOLD = 0.7  # Use GNN+RL result directly above this threshold

    def __init__(self, knowledge_graph=None, llm_manager=None, auto_sync: bool = True, use_gnn_rl: bool = True):
        """
        Args:
            knowledge_graph: KnowledgeGraphEngine instance (lazy loaded if None)
            llm_manager: LLM manager (uses default manager if None)
            auto_sync: Whether to auto-sync agents on startup
            use_gnn_rl: Whether to use GNN+RL (v3.0)
        """
        self._knowledge_graph = knowledge_graph
        self._llm_manager = llm_manager
        self._intelligent_selector = None  # v3.0: GNN+RL selector
        self._sync_service = None
        self._initialized = False
        self._gnn_rl_enabled = use_gnn_rl and self.USE_GNN_RL

        # Statistics tracking
        self.stats = {
            "total_selections": 0,
            "gnn_rl_selections": 0,     # v3.0: GNN+RL direct selection count
            "gnn_rl_fallback": 0,       # v3.0: GNN+RL fallback due to low confidence
            "graph_assisted": 0,
            "llm_only": 0,
            "feedback_stored": 0,
            "agents_synced": 0,
            "semantic_analysis_count": 0,
            "pattern_generalizations": 0,
            "time_decay_applied": 0
        }

        # v3.0: History buffers for dashboard visualization
        self._selection_history: deque = deque(maxlen=200)
        self._training_history: deque = deque(maxlen=100)

        # v2.0: Query category cache (minimize LLM calls)
        self._category_cache: Dict[str, Dict[str, Any]] = {}

        # Auto sync
        if auto_sync:
            asyncio.create_task(self._initialize_async())

        logger.info(f"Hybrid agent selector v3.0 initialized (GNN+RL: {'enabled' if self._gnn_rl_enabled else 'disabled'})")

    async def _initialize_async(self):
        """Async initialization (agent sync)"""
        if self._initialized:
            return

        try:
            from .agent_sync_service import get_sync_service, initialize_agent_sync

            self._sync_service = get_sync_service()

            # Configure Knowledge Graph
            if self.knowledge_graph:
                self._sync_service._knowledge_graph = self.knowledge_graph

            # Run full sync
            result = await self._sync_service.full_sync()
            self.stats["agents_synced"] = result.get("total_agents", 0)

            self._initialized = True
            logger.info(f"🔄 Agent sync complete: {self.stats['agents_synced']} agents")

        except Exception as e:
            logger.warning(f"⚠️ Agent auto-sync failed (manual sync required): {e}")

    async def ensure_initialized(self):
        """Ensure initialization (wait for sync)"""
        if not self._initialized:
            await self._initialize_async()

    @property
    def knowledge_graph(self):
        """Lazy load knowledge graph — uses shared singleton"""
        if self._knowledge_graph is None:
            try:
                from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
                self._knowledge_graph = get_knowledge_graph_engine()
                logger.info("📊 Knowledge graph engine loaded (shared singleton)")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge graph load failed, using LLM only: {e}")
        return self._knowledge_graph

    @property
    def llm_manager(self):
        """Lazy load LLM manager"""
        if self._llm_manager is None:
            self._llm_manager = get_ontology_llm_manager()
        return self._llm_manager

    @property
    def intelligent_selector(self):
        """v3.0: Lazy load GNN+RL intelligent selector"""
        if self._intelligent_selector is None and self._gnn_rl_enabled:
            try:
                from ..ml.intelligent_selector import IntelligentAgentSelector
                self._intelligent_selector = IntelligentAgentSelector(
                    auto_load=True,  # Auto-load saved models
                    device='cpu'
                )
                logger.info("🤖 GNN+RL IntelligentAgentSelector loaded")
            except Exception as e:
                logger.warning(f"⚠️ GNN+RL selector load failed, using fallback: {e}")
                self._gnn_rl_enabled = False
        return self._intelligent_selector

    async def select_agent(
        self,
        query: str,
        available_agents: List[str],
        agents_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        🧠 Hybrid agent selection (main method) v3.0

        Selection flow:
        1. Phase 0 (v3.0): GNN+RL selection → return directly if confidence is high
        2. Phase 1: Knowledge Graph analysis → collect insights
        3. Phase 2: LLM final decision → utilize graph insights

        Args:
            query: User query
            available_agents: List of available agent IDs
            agents_info: Agent metadata {agent_id: {name, description, capabilities, tags}}
            context: Additional context (optional)

        Returns:
            Tuple[selected_agent_id, selection_metadata]
        """
        self.stats["total_selections"] += 1
        start_time = datetime.now()

        # ========== Phase 0 (v3.0): GNN+RL Selection ==========
        gnn_rl_result = None
        if self._gnn_rl_enabled and self.intelligent_selector:
            try:
                gnn_rl_agent, gnn_rl_meta = await self.intelligent_selector.select_agent(
                    query=query,
                    available_agents=available_agents,
                    deterministic=False  # Allow exploration
                )

                gnn_rl_confidence = gnn_rl_meta.get('confidence', 0.0)
                gnn_rl_result = {
                    "agent": gnn_rl_agent,
                    "confidence": gnn_rl_confidence,
                    "value_estimate": gnn_rl_meta.get('value_estimate', 0.0),
                    "method": "gnn_rl"
                }

                logger.info(
                    f"🤖 GNN+RL selection: {gnn_rl_agent} (confidence: {gnn_rl_confidence:.1%})"
                )

                # Return directly if confidence is sufficiently high
                if gnn_rl_confidence >= self.GNN_RL_CONFIDENCE_THRESHOLD:
                    if gnn_rl_agent in available_agents:
                        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                        self.stats["gnn_rl_selections"] += 1

                        metadata = {
                            "selected_agent": gnn_rl_agent,
                            "reasoning": f"GNN+RL model selected with {gnn_rl_confidence:.1%} confidence",
                            "gnn_rl_result": gnn_rl_result,
                            "selection_method": "gnn_rl",
                            "elapsed_ms": elapsed_ms,
                            "timestamp": datetime.now().isoformat()
                        }

                        self._selection_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "query": query[:80],
                            "selected_agent": gnn_rl_agent,
                            "method": "gnn_rl",
                            "confidence": gnn_rl_confidence,
                            "value_estimate": gnn_rl_result.get("value_estimate", 0.0),
                            "elapsed_ms": round(elapsed_ms, 1),
                            # Enriched metadata for detail view
                            "reasoning": f"GNN+RL model selected with {gnn_rl_confidence:.1%} confidence (direct)",
                            "graph_insights": {
                                "has_insights": False,
                                "entities": [],
                                "related_concepts": [],
                                "past_patterns": [],
                                "recommended_agents": [],
                                "kg_confidence": 0.0,
                            },
                            "gnn_rl": {
                                "raw_confidence": gnn_rl_confidence,
                                "value_estimate": gnn_rl_result.get("value_estimate", 0.0),
                                "suggested_agent": gnn_rl_agent,
                                "used_directly": True,
                            },
                        })

                        logger.info(
                            f"🤖 GNN+RL direct selection complete: {gnn_rl_agent} "
                            f"(confidence: {gnn_rl_confidence:.1%}, {elapsed_ms:.0f}ms)"
                        )
                        return gnn_rl_agent, metadata
                    else:
                        logger.warning(f"⚠️ GNN+RL selection {gnn_rl_agent} not in available_agents, falling back")

                # Fall back if confidence is low
                self.stats["gnn_rl_fallback"] += 1
                logger.info(f"🔄 GNN+RL confidence low ({gnn_rl_confidence:.1%}), falling back to KG+LLM")

            except Exception as e:
                logger.warning(f"GNN+RL selection failed, falling back to KG+LLM: {e}")

        # ========== Phase 1: Knowledge Graph Analysis ==========
        graph_insights = await self._analyze_with_knowledge_graph(query, available_agents)

        # Add GNN+RL result to graph insights (for LLM reference)
        if gnn_rl_result:
            graph_insights["gnn_rl_suggestion"] = gnn_rl_result

        # ========== Phase 2: LLM Final Decision ==========
        selected_agent, reasoning = await self._select_with_llm(
            query, available_agents, agents_info, graph_insights
        )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Build metadata
        selection_method = "hybrid"
        if gnn_rl_result:
            selection_method = "gnn_rl_assisted"
        elif graph_insights.get("has_insights"):
            selection_method = "kg_assisted"
        else:
            selection_method = "llm_only"

        metadata = {
            "selected_agent": selected_agent,
            "reasoning": reasoning,
            "graph_insights": graph_insights,
            "gnn_rl_result": gnn_rl_result,
            "selection_method": selection_method,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat()
        }

        if graph_insights.get("has_insights"):
            self.stats["graph_assisted"] += 1
        else:
            self.stats["llm_only"] += 1

        self._selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query[:80],
            "selected_agent": selected_agent,
            "method": selection_method,
            "confidence": gnn_rl_result["confidence"] if gnn_rl_result else 0.0,
            "value_estimate": gnn_rl_result["value_estimate"] if gnn_rl_result else 0.0,
            "elapsed_ms": round(elapsed_ms, 1),
            # Enriched metadata for detail view
            "reasoning": reasoning,
            "graph_insights": {
                "has_insights": graph_insights.get("has_insights", False),
                "entities": graph_insights.get("entities", [])[:10],
                "related_concepts": graph_insights.get("related_concepts", [])[:10],
                "past_patterns": graph_insights.get("past_patterns", [])[:3],
                "recommended_agents": graph_insights.get("recommended_agents", [])[:3],
                "kg_confidence": graph_insights.get("confidence", 0.0),
            },
            "gnn_rl": {
                "raw_confidence": gnn_rl_result["confidence"],
                "value_estimate": gnn_rl_result["value_estimate"],
                "suggested_agent": gnn_rl_result["agent"],
                "used_directly": False,
            } if gnn_rl_result else None,
        })

        logger.info(
            f"🧠 Hybrid selection complete: {selected_agent} "
            f"(method: {selection_method}, {elapsed_ms:.0f}ms)"
        )

        return selected_agent, metadata

    async def _analyze_with_knowledge_graph(
        self,
        query: str,
        available_agents: List[str]
    ) -> Dict[str, Any]:
        """
        📊 Phase 1: Knowledge Graph analysis

        - Entity extraction
        - Related concept exploration
        - Query past success patterns
        """
        insights = {
            "has_insights": False,
            "entities": [],
            "related_concepts": [],
            "past_patterns": [],
            "recommended_agents": [],
            "confidence": 0.0
        }

        if not self.knowledge_graph:
            return insights

        try:
            # 1. Entity extraction (simple rule-based + graph matching)
            entities = await self._extract_entities(query)
            insights["entities"] = entities

            # 2. Related concept exploration
            related_concepts = []
            for entity in entities[:5]:  # Top 5 only
                concepts = await self.knowledge_graph.find_related_concepts(entity, max_depth=2)
                related_concepts.extend(concepts[:10])
            insights["related_concepts"] = list(set(related_concepts))[:20]

            # 3. Query past success patterns
            past_patterns = await self._find_past_patterns(query, entities)
            insights["past_patterns"] = past_patterns

            # 4. Derive agent recommendations
            recommended = await self._derive_agent_recommendations(
                entities, related_concepts, past_patterns, available_agents
            )
            insights["recommended_agents"] = recommended

            # 5. Calculate confidence
            if recommended or past_patterns:
                insights["has_insights"] = True
                insights["confidence"] = self._calculate_confidence(insights)

                logger.info(
                f"📊 Graph analysis complete: {len(entities)} entities, "
                f"{len(related_concepts)} related concepts, "
                f"{len(past_patterns)} past patterns"
            )

        except Exception as e:
            logger.warning(f"⚠️ Knowledge graph analysis failed: {e}")

        return insights

    async def _extract_entities(self, query: str) -> List[str]:
        """Entity extraction (simple rule-based)"""
        entities = []

        # Korean noun/proper noun patterns
        # Extend later if more sophisticated NER is needed
        patterns = [
            r'삼성전자|삼성|애플|구글|테슬라|마이크로소프트',  # Company names
            r'주가|환율|날씨|뉴스|가격|실적|매출',  # Domain keywords
            r'\d+원|\d+달러|\d+%',  # Numeric patterns
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)

        # Match with nodes in the graph
        if self.knowledge_graph and hasattr(self.knowledge_graph, 'graph_engine'):
            graph = self.knowledge_graph.graph_engine.graph
            words = query.split()
            for word in words:
                if word in graph.nodes:
                    entities.append(word)

        return list(set(entities))

    async def _find_past_patterns(
        self,
        query: str,
        entities: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Query past success patterns (v2.0: time decay + semantic matching)

        v2.0 improvements:
        - LLM-based semantic query analysis
        - Apply time decay weighting
        - Pattern generalization matching (category-based)
        """
        patterns = []

        if not self.knowledge_graph:
            return patterns

        try:
            graph = self.knowledge_graph.graph_engine.graph

            # v2.0: LLM-based semantic analysis (no hardcoding)
            query_semantics = await self._analyze_query_semantics(query)
            query_category = query_semantics.get("category", "general")
            query_pattern = query_semantics.get("generalization_pattern", "general_query")
            query_keywords = query_semantics.get("keywords", [])

            for node_id, attrs in graph.nodes(data=True):
                node_type = attrs.get('type', '')

                # Find past successful query-agent mappings
                if node_type == 'query_agent_mapping':
                    past_agent = attrs.get('selected_agent', '')
                    success_rate = attrs.get('success_rate', 0.0)
                    usage_count = attrs.get('usage_count', 1)
                    last_used = attrs.get('last_used')

                    # v2.0: Apply time decay
                    time_weight = self._calculate_time_decay(last_used)
                    self.stats["time_decay_applied"] += 1

                    # Calculate matching score (considering multiple factors)
                    match_score = 0.0

                    # 1. Category matching (v2.0: generalized pattern matching)
                    past_category = attrs.get('category', '')
                    past_pattern = attrs.get('generalization_pattern', '')

                    if past_pattern == query_pattern:
                        match_score += 1.0  # Exact pattern match
                        self.stats["pattern_generalizations"] += 1
                    elif past_category == query_category:
                        match_score += 0.7  # Category match

                    # 2. Keyword matching (semantic)
                    past_keywords = attrs.get('keywords', [])
                    if past_keywords:
                        keyword_overlap = len(set(past_keywords) & set(query_keywords))
                        match_score += min(keyword_overlap * 0.2, 0.5)

                    # 3. Entity matching
                    past_entities = attrs.get('entities', [])
                    if past_entities and entities:
                        entity_overlap = len(set(past_entities) & set(entities))
                        match_score += min(entity_overlap * 0.1, 0.3)

                    # v2.0: Final score = match_score × success_rate × time_weight
                    if match_score > 0:
                        final_score = match_score * success_rate * time_weight

                        patterns.append({
                            'agent': past_agent,
                            'category': past_category,
                            'generalization_pattern': past_pattern,
                            'success_rate': success_rate,
                            'usage_count': usage_count,
                            'time_weight': round(time_weight, 3),
                            'match_score': round(match_score, 3),
                            'final_score': round(final_score, 3),
                            'last_used': last_used
                        })

                # Find agent-domain mapping
                if node_type == 'agent':
                    agent_id = node_id
                    for neighbor in graph.neighbors(agent_id):
                        neighbor_attrs = graph.nodes.get(neighbor, {})
                        if neighbor_attrs.get('type') == 'domain':
                            domain_name = neighbor_attrs.get('domain_name', '')

                            # v2.0: Category-based domain matching
                            if query_category in domain_name.lower() or any(kw in domain_name.lower() for kw in query_keywords):
                                patterns.append({
                                    'agent': agent_id,
                                    'domain': domain_name,
                                    'category': query_category,
                                    'success_rate': attrs.get('success_rate', 0.5),
                                    'usage_count': attrs.get('usage_count', 1),
                                    'final_score': 0.3  # Lower score for domain match
                                })

            # v2.0: Sort by final score (time decay applied)
            patterns.sort(key=lambda x: x.get('final_score', 0), reverse=True)

            logger.info(
                f"📊 Past patterns found: {len(patterns)} "
                f"(category={query_category}, pattern={query_pattern})"
            )

        except Exception as e:
            logger.warning(f"⚠️ Past pattern query failed: {e}")

        return patterns[:5]  # Top 5 only

    # =========================================================================
    # v2.0 NEW: Time Decay
    # =========================================================================

    def _calculate_time_decay(self, last_used_str: Optional[str]) -> float:
        """
        🕐 v2.0: Calculate time decay weight

        Assigns higher weight to recent patterns, lower weight to older patterns.

        Formula: weight = max(min_weight, 0.5 ^ (days / half_life))

        Returns:
            float: Weight between 0.1 and 1.0
        """
        if not last_used_str:
            return TIME_DECAY_CONFIG["min_weight"]

        try:
            last_used = datetime.fromisoformat(last_used_str.replace('Z', '+00:00'))
            if last_used.tzinfo:
                last_used = last_used.replace(tzinfo=None)

            days_ago = (datetime.now() - last_used).days

            if days_ago <= 0:
                return 1.0  # Pattern used today

            if days_ago >= TIME_DECAY_CONFIG["max_age_days"]:
                return TIME_DECAY_CONFIG["min_weight"]

            # Exponential decay: 0.5 ^ (days / half_life)
            half_life = TIME_DECAY_CONFIG["half_life_days"]
            decay_weight = math.pow(0.5, days_ago / half_life)

            return max(TIME_DECAY_CONFIG["min_weight"], decay_weight)

        except Exception as e:
            logger.warning(f"⚠️ Time decay calculation failed: {e}")
            return 0.5  # Default value

    # =========================================================================
    # v2.0 NEW: LLM-based semantic query analysis (no hardcoding)
    # =========================================================================

    async def _analyze_query_semantics(self, query: str) -> Dict[str, Any]:
        """
        🧠 v2.0: LLM-based query semantic analysis

        Uses LLM instead of hardcoded keyword matching to analyze query semantics.

        Returns:
            {
                "category": "금융|날씨|쇼핑|정보검색|...",
                "intent": "조회|분석|비교|계산|...",
                "entities": ["삼성전자", "주가", ...],
                "keywords": ["검색", "금융", ...],
                "generalization_pattern": "stock_price_query"
            }
        """
        # Check cache (prevent re-analysis of same query)
        cache_key = query[:100]
        if cache_key in self._category_cache:
            return self._category_cache[cache_key]

        self.stats["semantic_analysis_count"] += 1

        prompt = f"""당신은 사용자 쿼리를 분석하는 전문가입니다.

쿼리: "{query}"

다음 정보를 JSON으로 추출하세요:

1. category: 쿼리의 주요 도메인 카테고리
   - 가능한 값: finance(금융), weather(날씨), shopping(쇼핑), coding(코딩),
     research(연구/논문), news(뉴스), travel(여행), calculation(계산),
     scheduling(일정), general(일반정보), other(기타)

2. intent: 사용자의 주요 의도
   - 가능한 값: search(검색), analyze(분석), compare(비교), calculate(계산),
     visualize(시각화), monitor(모니터링), create(생성), other(기타)

3. entities: 쿼리에 언급된 주요 개체들 (회사명, 상품명, 지역명 등)

4. keywords: 쿼리의 핵심 키워드 (동사, 도메인 용어 등)

5. generalization_pattern: 이 쿼리의 일반화된 패턴명
   - 예시: stock_price_query, weather_forecast, product_comparison, etc.
   - 비슷한 유형의 쿼리가 같은 패턴으로 묶일 수 있도록 명명

응답 형식 (JSON만):
{{"category": "...", "intent": "...", "entities": [...], "keywords": [...], "generalization_pattern": "..."}}
"""

        try:
            llm = self.llm_manager.get_llm(OntologyLLMType.QUERY_PROCESSOR)
            response = await llm.ainvoke(prompt)

            response_text = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                # Normalize result
                semantic_result = {
                    "category": result.get("category", "general"),
                    "intent": result.get("intent", "search"),
                    "entities": result.get("entities", []),
                    "keywords": result.get("keywords", []),
                    "generalization_pattern": result.get("generalization_pattern", "general_query")
                }

                # Cache result
                self._category_cache[cache_key] = semantic_result

                logger.info(
                    f"🧠 Query semantic analysis complete: category={semantic_result['category']}, "
                    f"pattern={semantic_result['generalization_pattern']}"
                )

                return semantic_result

        except Exception as e:
            logger.warning(f"⚠️ LLM semantic analysis failed, using fallback: {e}")

        # Fallback: basic analysis
        return self._fallback_semantic_analysis(query)

    def _fallback_semantic_analysis(self, query: str) -> Dict[str, Any]:
        """Basic semantic analysis when LLM fails (minimal rules)"""
        query_lower = query.lower()

        # Infer category (minimal rules)
        category = "general"
        if any(w in query_lower for w in ['주가', '환율', '주식', '펀드', '투자']):
            category = "finance"
        elif any(w in query_lower for w in ['날씨', '기온', '비', '눈']):
            category = "weather"
        elif any(w in query_lower for w in ['가격', '구매', '쇼핑', '상품']):
            category = "shopping"
        elif any(w in query_lower for w in ['코드', '프로그래밍', '개발', '버그']):
            category = "coding"

        return {
            "category": category,
            "intent": "search",
            "entities": [],
            "keywords": query.split()[:5],
            "generalization_pattern": f"{category}_query"
        }

    def _extract_intent_keywords(self, query: str) -> List[str]:
        """
        Extract intent keywords from query (v2.0: for backward compatibility)

        NOTE: This method is maintained for compatibility,
        but new code should use _analyze_query_semantics().
        """
        # Use basic analysis in contexts where async calls are not possible
        fallback = self._fallback_semantic_analysis(query)
        return fallback.get("keywords", [])

    async def _derive_agent_recommendations(
        self,
        entities: List[str],
        related_concepts: List[str],
        past_patterns: List[Dict[str, Any]],
        available_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Derive graph-based agent recommendations (v2.0: with time decay)

        v2.0 improvements:
        - Uses final_score (time decay already applied)
        - Includes generalization pattern information
        """
        recommendations = []
        agent_scores = {}
        agent_details = {}  # Recommendation detail info

        # 1. Score based on past patterns (v2.0: uses final_score with time decay)
        for pattern in past_patterns:
            agent = pattern.get('agent', '')
            if agent in available_agents:
                # v2.0: final_score already has time decay applied
                score = pattern.get('final_score', 0)
                agent_scores[agent] = agent_scores.get(agent, 0) + score

                # Store detail info
                if agent not in agent_details:
                    agent_details[agent] = {
                        'patterns': [],
                        'categories': set()
                    }
                agent_details[agent]['patterns'].append({
                    'pattern': pattern.get('generalization_pattern', ''),
                    'success_rate': pattern.get('success_rate', 0),
                    'time_weight': pattern.get('time_weight', 1.0)
                })
                if pattern.get('category'):
                    agent_details[agent]['categories'].add(pattern.get('category'))

        # 2. Score based on domain relevance (graph traversal)
        if self.knowledge_graph and hasattr(self.knowledge_graph, 'graph_engine'):
            graph = self.knowledge_graph.graph_engine.graph

            for agent_id in available_agents:
                if agent_id in graph.nodes:
                    connected = list(graph.neighbors(agent_id))

                    overlap_entities = len(set(connected) & set(entities))
                    overlap_concepts = len(set(connected) & set(related_concepts))

                    if overlap_entities > 0 or overlap_concepts > 0:
                        score = overlap_entities * 0.3 + overlap_concepts * 0.1
                        agent_scores[agent_id] = agent_scores.get(agent_id, 0) + score

        # Generate recommendations (v2.0: includes detail info)
        for agent_id, score in sorted(agent_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                details = agent_details.get(agent_id, {})
                recommendations.append({
                    'agent_id': agent_id,
                    'graph_score': round(score, 3),
                    'reason': 'graph_based_recommendation',
                    'matched_categories': list(details.get('categories', [])),
                    'pattern_count': len(details.get('patterns', [])),
                    'avg_time_weight': round(
                        sum(p.get('time_weight', 1.0) for p in details.get('patterns', [])) /
                        max(len(details.get('patterns', [])), 1),
                        2
                    ) if details.get('patterns') else 1.0
                })

        return recommendations[:3]  # Top 3

    def _calculate_confidence(self, insights: Dict[str, Any]) -> float:
        """Calculate graph insight confidence"""
        confidence = 0.0

        # +0.3 if past patterns exist
        if insights.get('past_patterns'):
            best_pattern = insights['past_patterns'][0]
            confidence += 0.3 * best_pattern.get('success_rate', 0.5)

        # +0.3 if recommended agents exist
        if insights.get('recommended_agents'):
            confidence += 0.3

        # +0.2 if many related concepts
        if len(insights.get('related_concepts', [])) > 5:
            confidence += 0.2

        # +0.2 if entities were extracted
        if insights.get('entities'):
            confidence += 0.2

        return min(confidence, 1.0)

    async def _select_with_llm(
        self,
        query: str,
        available_agents: List[str],
        agents_info: Dict[str, Any],
        graph_insights: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        🤖 Phase 2: LLM final decision (utilizing graph insights)
        """
        # Build agent list string
        agents_list = []
        for agent_id in available_agents:
            info = agents_info.get(agent_id, {})
            agent_data = info.get('agent_data', info)
            name = agent_data.get('name', agent_id)
            description = agent_data.get('description', '')[:200]
            capabilities = agent_data.get('capabilities', [])
            tags = agent_data.get('tags', [])

            cap_str = ', '.join([
                c.get('name', str(c)) if isinstance(c, dict) else str(c)
                for c in capabilities[:5]
            ])
            tag_str = ', '.join(tags[:5]) if tags else ''

            agents_list.append(
                f"- {agent_id}: {name} | {description} | 능력: {cap_str} | 태그: {tag_str}"
            )

        agents_formatted = '\n'.join(agents_list)

        # Build graph insights context
        graph_context = ""
        if graph_insights.get("has_insights"):
            graph_context = f"""
📊 지식그래프 분석 결과 (참고용):
- 추출된 엔티티: {', '.join(graph_insights.get('entities', [])[:5])}
- 관련 개념: {', '.join(graph_insights.get('related_concepts', [])[:5])}
- 과거 성공 패턴: {json.dumps(graph_insights.get('past_patterns', [])[:3], ensure_ascii=False)}
- 그래프 추천 에이전트: {json.dumps(graph_insights.get('recommended_agents', [])[:3], ensure_ascii=False)}
- 그래프 신뢰도: {graph_insights.get('confidence', 0):.1%}

위 그래프 분석 결과를 참고하되, 최종 판단은 쿼리 의도와 에이전트 능력을 종합적으로 고려하세요.
"""

        # LLM prompt
        prompt = f"""당신은 사용자 쿼리를 분석하여 가장 적합한 에이전트를 선택하는 전문가입니다.

사용자 쿼리: "{query}"
{graph_context}
사용 가능한 에이전트 목록:
{agents_formatted}

선택 규칙:
1. 전문 에이전트 우선: 특정 도메인 전용 에이전트가 있으면 반드시 우선 사용
   - 날씨 쿼리 + weather_agent 존재 → weather_agent 사용
   - 쇼핑 쿼리 + shopping_agent 존재 → shopping_agent 사용
   - 코딩 쿼리 + code_agent 존재 → code_agent 사용
2. 범용 에이전트(internet_agent, llm_search_agent)는 전문 에이전트가 없을 때만 사용
3. 그래프 분석 결과가 있으면 참고하되, 최종 판단은 쿼리 의도 기반
4. 반드시 위 목록에 있는 정확한 agent_id를 반환

응답 형식 (JSON만):
{{"selected_agent": "agent_id_here", "reasoning": "선택 이유 (그래프 근거 포함)"}}
"""

        try:
            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(prompt)

            response_text = response.content if hasattr(response, 'content') else str(response)

            # JSON parsing
            json_match = re.search(
                r'\{[^{}]*"selected_agent"\s*:\s*"([^"]+)"[^{}]*"reasoning"\s*:\s*"([^"]*)"[^{}]*\}',
                response_text, re.DOTALL
            )

            if json_match:
                selected_agent = json_match.group(1)
                reasoning = json_match.group(2)

                # Validate selected agent
                if selected_agent in available_agents:
                    return selected_agent, reasoning

                # Try partial match
                for agent in available_agents:
                    if selected_agent.lower() in agent.lower() or agent.lower() in selected_agent.lower():
                        return agent, reasoning

            # Use graph recommendation on parse failure
            if graph_insights.get("recommended_agents"):
                recommended = graph_insights["recommended_agents"][0]
                return recommended["agent_id"], f"Graph recommendation (LLM parse failed): {recommended.get('reason', '')}"

            logger.warning(f"LLM response parse failed: {response_text[:200]}")

        except Exception as e:
            logger.error(f"LLM selection failed: {e}")

            # Fall back to graph recommendation
            if graph_insights.get("recommended_agents"):
                recommended = graph_insights["recommended_agents"][0]
                return recommended["agent_id"], f"Graph-based fallback (LLM failed): {str(e)[:50]}"

        # Final fallback
        return available_agents[0] if available_agents else 'unknown', "Fallback: first agent"

    async def store_feedback(
        self,
        query: str,
        selected_agent: str,
        success: bool,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        📝 Phase 3: Store feedback (v2.0 Enhanced learning loop)

        v2.0 improvements:
        - Store LLM-based semantic analysis results
        - Store category and generalization pattern
        - Store entities and keywords
        - Record time information accurately
        """
        if not self.knowledge_graph:
            return False

        try:
            # v2.0: LLM-based semantic analysis
            query_semantics = await self._analyze_query_semantics(query)

            # Mapping node ID (v2.0: more general ID based on pattern)
            pattern = query_semantics.get("generalization_pattern", "general_query")
            mapping_id = f"mapping_{selected_agent}_{pattern}_{hash(query) % 1000}"

            # Check existing mapping
            graph = self.knowledge_graph.graph_engine.graph

            if mapping_id in graph.nodes:
                # Update existing mapping
                attrs = graph.nodes[mapping_id]
                usage_count = attrs.get('usage_count', 1) + 1

                # Update success rate (exponential moving average - higher weight for recent results)
                # v2.0: Use EMA (Exponential Moving Average)
                alpha = 0.3  # Weight for recent results
                old_success_rate = attrs.get('success_rate', 0.5)
                new_success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_success_rate

                attrs['usage_count'] = usage_count
                attrs['success_rate'] = new_success_rate
                attrs['last_used'] = datetime.now().isoformat()

                # v2.0: Update sample queries (maintain diversity)
                existing_samples = attrs.get('query_samples', [])
                if len(existing_samples) < 5 and query[:50] not in existing_samples:
                    existing_samples.append(query[:50])
                    attrs['query_samples'] = existing_samples

                    logger.info(
                    f"📝 Feedback updated: {selected_agent} ({pattern}) "
                    f"[success rate: {old_success_rate:.2f}→{new_success_rate:.2f}, usage: {usage_count}]"
                )

            else:
                # v2.0: Create new mapping (including semantic analysis results)
                await self.knowledge_graph.add_concept(
                    mapping_id,
                    "query_agent_mapping",
                    {
                        "query_sample": query[:100],
                        "query_samples": [query[:50]],  # v2.0: Store diverse samples
                        "selected_agent": selected_agent,
                        # v2.0: Semantic analysis results
                        "category": query_semantics.get("category", "general"),
                        "intent": query_semantics.get("intent", "search"),
                        "generalization_pattern": pattern,
                        "entities": query_semantics.get("entities", []),
                        "keywords": query_semantics.get("keywords", []),
                        # Success rate and usage statistics
                        "success_rate": 1.0 if success else 0.0,
                        "usage_count": 1,
                        # Time information
                        "created_at": datetime.now().isoformat(),
                        "last_used": datetime.now().isoformat()
                    }
                )

                # Connect agent with mapping
                await self.knowledge_graph.add_relationship(
                    selected_agent, mapping_id, "has_mapping"
                )

                # v2.0: Also connect with category node (pattern generalization)
                category = query_semantics.get("category", "general")
                category_node_id = f"category_{category}"

                # Create category node if it does not exist
                if category_node_id not in graph.nodes:
                    await self.knowledge_graph.add_concept(
                        category_node_id,
                        "query_category",
                        {"name": category, "created_at": datetime.now().isoformat()}
                    )

                await self.knowledge_graph.add_relationship(
                    mapping_id, category_node_id, "belongs_to_category"
                )

                logger.info(
                    f"📝 New feedback stored: {selected_agent} "
                    f"(category={category}, pattern={pattern})"
                )

            self.stats["feedback_stored"] += 1

            # Enrich matching selection_history entry with feedback data
            query_short = query[:80]
            for entry in reversed(self._selection_history):
                if entry.get("query") == query_short and entry.get("selected_agent") == selected_agent:
                    # Determine EMA success rate from current node attrs
                    ema_rate = None
                    if mapping_id in graph.nodes:
                        ema_rate = graph.nodes[mapping_id].get("success_rate")
                    entry["feedback"] = {
                        "success": success,
                        "query_semantics": query_semantics,
                        "ema_success_rate": ema_rate,
                        "kg_nodes_updated": True,
                    }
                    break

            # Periodic save: checkpoint KG + stats every 50 feedbacks
            if self.stats["feedback_stored"] % 50 == 0:
                try:
                    self.knowledge_graph.save_to_disk()
                    self.save_stats()
                    logger.info(f"💾 Periodic checkpoint at feedback #{self.stats['feedback_stored']}")
                except Exception as save_err:
                    logger.warning(f"⚠️ Periodic save failed: {save_err}")

            # v3.0: Also store feedback in GNN+RL experience buffer
            if self._gnn_rl_enabled and self.intelligent_selector:
                try:
                    await self.intelligent_selector.store_feedback(
                        success=success,
                        execution_result=execution_result
                    )
                    logger.info(f"🤖 GNN+RL feedback stored: {selected_agent} (success={success})")

                    # Record training loss for dashboard
                    last_loss = self.intelligent_selector.stats.get("last_train_loss")
                    if last_loss is not None:
                        self._training_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "total_loss": last_loss,
                            "training_steps": self.intelligent_selector.stats.get("training_steps", 0),
                            "buffer_size": self.intelligent_selector.experience_buffer.size,
                        })
                except Exception as gnn_rl_error:
                    logger.warning(f"⚠️ GNN+RL feedback store failed: {gnn_rl_error}")

            return True

        except Exception as e:
            logger.warning(f"⚠️ Feedback store failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics (v3.0: includes history for dashboard)."""
        return {
            **self.stats,
            "graph_assist_ratio": round(
                self.stats["graph_assisted"] / max(self.stats["total_selections"], 1),
                3
            ),
            "generalization_ratio": round(
                self.stats["pattern_generalizations"] / max(self.stats["time_decay_applied"], 1),
                3
            ) if self.stats["time_decay_applied"] > 0 else 0.0,
            "version": "3.0",
            # v3.0: History buffers for dashboard visualization
            "selection_history": list(self._selection_history),
            "training_history": list(self._training_history),
            "gnn_rl_enabled": self._gnn_rl_enabled,
            "ml_stats": (
                self.intelligent_selector.stats
                if self._gnn_rl_enabled and self.intelligent_selector
                else None
            ),
            "buffer_size": (
                self.intelligent_selector.experience_buffer.size
                if self._gnn_rl_enabled and self.intelligent_selector
                else 0
            ),
        }

    def save_stats(self) -> bool:
        """Save selector stats + category cache to disk."""
        try:
            from pathlib import Path
            import json

            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            save_path = data_dir / "selector_stats.json"

            checkpoint = {
                "stats": self.stats,
                "category_cache": dict(self._category_cache),
                "selection_history": list(self._selection_history),
                "training_history": list(self._training_history),
                "saved_at": datetime.now().isoformat(),
            }
            tmp_path = save_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, default=str)
            import os
            os.replace(str(tmp_path), str(save_path))
            logger.info(f"💾 Selector stats saved: {self.stats['total_selections']} selections, {self.stats['feedback_stored']} feedbacks")
            return True
        except Exception as e:
            logger.error(f"Selector stats save failed: {e}")
            return False

    def load_stats(self) -> bool:
        """Load selector stats + category cache from disk."""
        try:
            from pathlib import Path
            import json

            load_path = Path(__file__).parent.parent / "data" / "selector_stats.json"
            if not load_path.exists():
                logger.info("No selector stats checkpoint found — starting fresh")
                return False

            with open(load_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            saved_stats = checkpoint.get("stats", {})
            if saved_stats:
                self.stats.update(saved_stats)

            saved_cache = checkpoint.get("category_cache", {})
            if saved_cache:
                self._category_cache.update(saved_cache)

            saved_selection_history = checkpoint.get("selection_history", [])
            if saved_selection_history:
                self._selection_history.extend(saved_selection_history)

            saved_training_history = checkpoint.get("training_history", [])
            if saved_training_history:
                self._training_history.extend(saved_training_history)

            logger.info(
                f"📂 Selector stats loaded: {self.stats['total_selections']} selections, "
                f"{self.stats['feedback_stored']} feedbacks, {len(self._category_cache)} cached categories, "
                f"{len(self._selection_history)} selection history, {len(self._training_history)} training history"
            )
            return True
        except Exception as e:
            logger.error(f"Selector stats load failed: {e}")
            return False

    def clear_category_cache(self):
        """v2.0: Clear category cache"""
        self._category_cache.clear()
        logger.info("🧹 Category cache cleared")


# Singleton instance
_hybrid_selector_instance = None


def get_hybrid_selector() -> HybridAgentSelector:
    """Return hybrid selector singleton instance (loads stats from checkpoint)"""
    global _hybrid_selector_instance
    if _hybrid_selector_instance is None:
        _hybrid_selector_instance = HybridAgentSelector()
        _hybrid_selector_instance.load_stats()
    return _hybrid_selector_instance


    logger.info("🧠 Hybrid agent selector v2.0 module loaded (time decay + LLM semantic analysis + pattern generalization)")
