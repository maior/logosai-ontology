"""
🧠 Hybrid Agent Selector v2.0
지식그래프 + LLM 하이브리드 에이전트 선택기

Phase 1: Knowledge Graph Analysis
  - 엔티티 추출 및 관련 개념 탐색
  - 과거 성공 패턴 조회 (시간 감쇠 적용)
  - 그래프 기반 후보 추천

Phase 2: LLM Final Decision
  - 그래프 인사이트를 컨텍스트로 활용
  - 의미론적 분석 + 그래프 근거로 최종 결정

Phase 3: Feedback Loop (v2.0 Enhanced)
  - 성공적인 쿼리-에이전트 매핑을 그래프에 저장
  - LLM 기반 의미론적 쿼리 분석 (하드코딩 제거)
  - 시간 감쇠 (Time Decay) 적용
  - 패턴 일반화 학습 (카테고리 기반)

v2.0 Updates (2026-01-31):
  - 시간 감쇠: 최근 패턴에 높은 가중치
  - LLM 기반 의미론적 매칭: 하드코딩 키워드 제거
  - 패턴 일반화: 쿼리 카테고리 기반 학습
"""

import asyncio
import json
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .llm_manager import get_ontology_llm_manager, OntologyLLMType


# 시간 감쇠 설정
TIME_DECAY_CONFIG = {
    "half_life_days": 30,  # 30일 후 가중치 50%로 감소
    "min_weight": 0.1,     # 최소 가중치 (아무리 오래되어도 10%는 유지)
    "max_age_days": 365    # 1년 이상된 패턴은 최소 가중치 적용
}


class HybridAgentSelector:
    """
    🧠 하이브리드 에이전트 선택기

    Knowledge Graph + LLM을 결합하여 최적의 에이전트를 선택합니다.

    Features:
    - 시작 시 Agent Marketplace와 자동 동기화
    - 지식그래프 기반 패턴 학습
    - LLM 기반 의미론적 분석
    """

    def __init__(self, knowledge_graph=None, llm_manager=None, auto_sync: bool = True):
        """
        Args:
            knowledge_graph: KnowledgeGraphEngine 인스턴스 (없으면 지연 로딩)
            llm_manager: LLM 매니저 (없으면 기본 매니저 사용)
            auto_sync: 시작 시 에이전트 자동 동기화 여부
        """
        self._knowledge_graph = knowledge_graph
        self._llm_manager = llm_manager
        self._sync_service = None
        self._initialized = False

        # 통계 추적
        self.stats = {
            "total_selections": 0,
            "graph_assisted": 0,
            "llm_only": 0,
            "feedback_stored": 0,
            "agents_synced": 0,
            "semantic_analysis_count": 0,  # v2.0: LLM 의미 분석 횟수
            "pattern_generalizations": 0,   # v2.0: 패턴 일반화 횟수
            "time_decay_applied": 0         # v2.0: 시간 감쇠 적용 횟수
        }

        # v2.0: 쿼리 카테고리 캐시 (LLM 호출 최소화)
        self._category_cache: Dict[str, Dict[str, Any]] = {}

        # 자동 동기화
        if auto_sync:
            asyncio.create_task(self._initialize_async())

        logger.info("🧠 하이브리드 에이전트 선택기 초기화 완료")

    async def _initialize_async(self):
        """비동기 초기화 (에이전트 동기화)"""
        if self._initialized:
            return

        try:
            from .agent_sync_service import get_sync_service, initialize_agent_sync

            self._sync_service = get_sync_service()

            # Knowledge Graph 설정
            if self.knowledge_graph:
                self._sync_service._knowledge_graph = self.knowledge_graph

            # 전체 동기화 실행
            result = await self._sync_service.full_sync()
            self.stats["agents_synced"] = result.get("total_agents", 0)

            self._initialized = True
            logger.info(f"🔄 에이전트 동기화 완료: {self.stats['agents_synced']}개")

        except Exception as e:
            logger.warning(f"⚠️ 에이전트 자동 동기화 실패 (수동 동기화 필요): {e}")

    async def ensure_initialized(self):
        """초기화 보장 (동기화 완료 대기)"""
        if not self._initialized:
            await self._initialize_async()

    @property
    def knowledge_graph(self):
        """지식그래프 지연 로딩"""
        if self._knowledge_graph is None:
            try:
                from ..engines.knowledge_graph_clean import KnowledgeGraphEngine
                self._knowledge_graph = KnowledgeGraphEngine(fast_mode=True)
                logger.info("📊 지식그래프 엔진 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ 지식그래프 로드 실패, LLM만 사용: {e}")
        return self._knowledge_graph

    @property
    def llm_manager(self):
        """LLM 매니저 지연 로딩"""
        if self._llm_manager is None:
            self._llm_manager = get_ontology_llm_manager()
        return self._llm_manager

    async def select_agent(
        self,
        query: str,
        available_agents: List[str],
        agents_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        🧠 하이브리드 에이전트 선택 (메인 메서드)

        Args:
            query: 사용자 쿼리
            available_agents: 사용 가능한 에이전트 ID 목록
            agents_info: 에이전트 메타데이터 {agent_id: {name, description, capabilities, tags}}
            context: 추가 컨텍스트 (선택사항)

        Returns:
            Tuple[selected_agent_id, selection_metadata]
        """
        self.stats["total_selections"] += 1
        start_time = datetime.now()

        # Phase 1: Knowledge Graph Analysis
        graph_insights = await self._analyze_with_knowledge_graph(query, available_agents)

        # Phase 2: LLM Final Decision (with graph insights)
        selected_agent, reasoning = await self._select_with_llm(
            query, available_agents, agents_info, graph_insights
        )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        # 메타데이터 구성
        metadata = {
            "selected_agent": selected_agent,
            "reasoning": reasoning,
            "graph_insights": graph_insights,
            "selection_method": "hybrid" if graph_insights.get("has_insights") else "llm_only",
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat()
        }

        if graph_insights.get("has_insights"):
            self.stats["graph_assisted"] += 1
        else:
            self.stats["llm_only"] += 1

        logger.info(
            f"🧠 하이브리드 선택 완료: {selected_agent} "
            f"(방식: {metadata['selection_method']}, {elapsed_ms:.0f}ms)"
        )

        return selected_agent, metadata

    async def _analyze_with_knowledge_graph(
        self,
        query: str,
        available_agents: List[str]
    ) -> Dict[str, Any]:
        """
        📊 Phase 1: 지식그래프 분석

        - 엔티티 추출
        - 관련 개념 탐색
        - 과거 성공 패턴 조회
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
            # 1. 엔티티 추출 (간단한 규칙 기반 + 그래프 매칭)
            entities = await self._extract_entities(query)
            insights["entities"] = entities

            # 2. 관련 개념 탐색
            related_concepts = []
            for entity in entities[:5]:  # 상위 5개만
                concepts = await self.knowledge_graph.find_related_concepts(entity, max_depth=2)
                related_concepts.extend(concepts[:10])
            insights["related_concepts"] = list(set(related_concepts))[:20]

            # 3. 과거 성공 패턴 조회
            past_patterns = await self._find_past_patterns(query, entities)
            insights["past_patterns"] = past_patterns

            # 4. 추천 에이전트 도출
            recommended = await self._derive_agent_recommendations(
                entities, related_concepts, past_patterns, available_agents
            )
            insights["recommended_agents"] = recommended

            # 5. 신뢰도 계산
            if recommended or past_patterns:
                insights["has_insights"] = True
                insights["confidence"] = self._calculate_confidence(insights)

            logger.info(
                f"📊 그래프 분석 완료: {len(entities)} 엔티티, "
                f"{len(related_concepts)} 관련 개념, "
                f"{len(past_patterns)} 과거 패턴"
            )

        except Exception as e:
            logger.warning(f"⚠️ 지식그래프 분석 실패: {e}")

        return insights

    async def _extract_entities(self, query: str) -> List[str]:
        """엔티티 추출 (간단한 규칙 기반)"""
        entities = []

        # 한국어 명사/고유명사 패턴
        # 더 정교한 NER이 필요하면 추후 확장
        patterns = [
            r'삼성전자|삼성|애플|구글|테슬라|마이크로소프트',  # 회사명
            r'주가|환율|날씨|뉴스|가격|실적|매출',  # 도메인 키워드
            r'\d+원|\d+달러|\d+%',  # 숫자 패턴
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)

        # 그래프에 있는 노드와 매칭
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
        과거 성공 패턴 조회 (v2.0: 시간 감쇠 + 의미론적 매칭)

        v2.0 개선사항:
        - LLM 기반 의미론적 쿼리 분석
        - 시간 감쇠 가중치 적용
        - 패턴 일반화 매칭 (카테고리 기반)
        """
        patterns = []

        if not self.knowledge_graph:
            return patterns

        try:
            graph = self.knowledge_graph.graph_engine.graph

            # v2.0: LLM 기반 의미 분석 (하드코딩 제거)
            query_semantics = await self._analyze_query_semantics(query)
            query_category = query_semantics.get("category", "general")
            query_pattern = query_semantics.get("generalization_pattern", "general_query")
            query_keywords = query_semantics.get("keywords", [])

            for node_id, attrs in graph.nodes(data=True):
                node_type = attrs.get('type', '')

                # 과거 성공적인 쿼리-에이전트 매핑 찾기
                if node_type == 'query_agent_mapping':
                    past_agent = attrs.get('selected_agent', '')
                    success_rate = attrs.get('success_rate', 0.0)
                    usage_count = attrs.get('usage_count', 1)
                    last_used = attrs.get('last_used')

                    # v2.0: 시간 감쇠 적용
                    time_weight = self._calculate_time_decay(last_used)
                    self.stats["time_decay_applied"] += 1

                    # 매칭 점수 계산 (여러 요소 고려)
                    match_score = 0.0

                    # 1. 카테고리 매칭 (v2.0: 일반화된 패턴 매칭)
                    past_category = attrs.get('category', '')
                    past_pattern = attrs.get('generalization_pattern', '')

                    if past_pattern == query_pattern:
                        match_score += 1.0  # 정확한 패턴 매칭
                        self.stats["pattern_generalizations"] += 1
                    elif past_category == query_category:
                        match_score += 0.7  # 카테고리 매칭

                    # 2. 키워드 매칭 (의미론적)
                    past_keywords = attrs.get('keywords', [])
                    if past_keywords:
                        keyword_overlap = len(set(past_keywords) & set(query_keywords))
                        match_score += min(keyword_overlap * 0.2, 0.5)

                    # 3. 엔티티 매칭
                    past_entities = attrs.get('entities', [])
                    if past_entities and entities:
                        entity_overlap = len(set(past_entities) & set(entities))
                        match_score += min(entity_overlap * 0.1, 0.3)

                    # v2.0: 최종 점수 = 매칭점수 × 성공률 × 시간가중치
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

                # 에이전트-도메인 매핑 찾기
                if node_type == 'agent':
                    agent_id = node_id
                    for neighbor in graph.neighbors(agent_id):
                        neighbor_attrs = graph.nodes.get(neighbor, {})
                        if neighbor_attrs.get('type') == 'domain':
                            domain_name = neighbor_attrs.get('domain_name', '')

                            # v2.0: 카테고리 기반 도메인 매칭
                            if query_category in domain_name.lower() or any(kw in domain_name.lower() for kw in query_keywords):
                                patterns.append({
                                    'agent': agent_id,
                                    'domain': domain_name,
                                    'category': query_category,
                                    'success_rate': attrs.get('success_rate', 0.5),
                                    'usage_count': attrs.get('usage_count', 1),
                                    'final_score': 0.3  # 도메인 매칭은 낮은 점수
                                })

            # v2.0: 최종 점수 기준 정렬 (시간 감쇠 반영)
            patterns.sort(key=lambda x: x.get('final_score', 0), reverse=True)

            logger.info(
                f"📊 과거 패턴 조회: {len(patterns)}개 발견 "
                f"(category={query_category}, pattern={query_pattern})"
            )

        except Exception as e:
            logger.warning(f"⚠️ 과거 패턴 조회 실패: {e}")

        return patterns[:5]  # 상위 5개만

    # =========================================================================
    # v2.0 NEW: 시간 감쇠 (Time Decay)
    # =========================================================================

    def _calculate_time_decay(self, last_used_str: Optional[str]) -> float:
        """
        🕐 v2.0: 시간 감쇠 가중치 계산

        최근 패턴에 높은 가중치, 오래된 패턴에 낮은 가중치를 부여합니다.

        Formula: weight = max(min_weight, 0.5 ^ (days / half_life))

        Returns:
            float: 0.1 ~ 1.0 사이의 가중치
        """
        if not last_used_str:
            return TIME_DECAY_CONFIG["min_weight"]

        try:
            last_used = datetime.fromisoformat(last_used_str.replace('Z', '+00:00'))
            if last_used.tzinfo:
                last_used = last_used.replace(tzinfo=None)

            days_ago = (datetime.now() - last_used).days

            if days_ago <= 0:
                return 1.0  # 오늘 사용된 패턴

            if days_ago >= TIME_DECAY_CONFIG["max_age_days"]:
                return TIME_DECAY_CONFIG["min_weight"]

            # 지수 감쇠: 0.5 ^ (days / half_life)
            half_life = TIME_DECAY_CONFIG["half_life_days"]
            decay_weight = math.pow(0.5, days_ago / half_life)

            return max(TIME_DECAY_CONFIG["min_weight"], decay_weight)

        except Exception as e:
            logger.warning(f"⚠️ 시간 감쇠 계산 실패: {e}")
            return 0.5  # 기본값

    # =========================================================================
    # v2.0 NEW: LLM 기반 의미론적 쿼리 분석 (하드코딩 제거)
    # =========================================================================

    async def _analyze_query_semantics(self, query: str) -> Dict[str, Any]:
        """
        🧠 v2.0: LLM 기반 쿼리 의미 분석

        하드코딩된 키워드 매칭 대신 LLM을 사용하여 쿼리의 의미를 분석합니다.

        Returns:
            {
                "category": "금융|날씨|쇼핑|정보검색|...",
                "intent": "조회|분석|비교|계산|...",
                "entities": ["삼성전자", "주가", ...],
                "keywords": ["검색", "금융", ...],
                "generalization_pattern": "stock_price_query"
            }
        """
        # 캐시 확인 (동일 쿼리 재분석 방지)
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

            # JSON 추출
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                # 결과 정규화
                semantic_result = {
                    "category": result.get("category", "general"),
                    "intent": result.get("intent", "search"),
                    "entities": result.get("entities", []),
                    "keywords": result.get("keywords", []),
                    "generalization_pattern": result.get("generalization_pattern", "general_query")
                }

                # 캐시 저장
                self._category_cache[cache_key] = semantic_result

                logger.info(
                    f"🧠 쿼리 의미 분석 완료: category={semantic_result['category']}, "
                    f"pattern={semantic_result['generalization_pattern']}"
                )

                return semantic_result

        except Exception as e:
            logger.warning(f"⚠️ LLM 의미 분석 실패, 기본 분석 사용: {e}")

        # 폴백: 기본 분석
        return self._fallback_semantic_analysis(query)

    def _fallback_semantic_analysis(self, query: str) -> Dict[str, Any]:
        """LLM 실패 시 기본 의미 분석 (최소한의 규칙)"""
        query_lower = query.lower()

        # 카테고리 추론 (최소한의 규칙)
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
        쿼리에서 의도 키워드 추출 (v2.0: 호환성 유지용)

        NOTE: 이 메서드는 호환성을 위해 유지되지만,
        새로운 코드에서는 _analyze_query_semantics()를 사용하세요.
        """
        # 비동기 호출이 불가능한 컨텍스트에서는 기본 분석 사용
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
        그래프 기반 에이전트 추천 도출 (v2.0: 시간 감쇠 반영)

        v2.0 개선사항:
        - final_score 사용 (시간 감쇠 이미 적용됨)
        - 일반화 패턴 정보 포함
        """
        recommendations = []
        agent_scores = {}
        agent_details = {}  # 추천 상세 정보

        # 1. 과거 패턴 기반 점수 (v2.0: 시간 감쇠 반영된 final_score 사용)
        for pattern in past_patterns:
            agent = pattern.get('agent', '')
            if agent in available_agents:
                # v2.0: final_score는 이미 시간 감쇠가 적용됨
                score = pattern.get('final_score', 0)
                agent_scores[agent] = agent_scores.get(agent, 0) + score

                # 상세 정보 저장
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

        # 2. 도메인 연관성 기반 점수 (그래프 탐색)
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

        # 추천 목록 생성 (v2.0: 상세 정보 포함)
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

        return recommendations[:3]  # 상위 3개

    def _calculate_confidence(self, insights: Dict[str, Any]) -> float:
        """그래프 인사이트 신뢰도 계산"""
        confidence = 0.0

        # 과거 패턴이 있으면 +0.3
        if insights.get('past_patterns'):
            best_pattern = insights['past_patterns'][0]
            confidence += 0.3 * best_pattern.get('success_rate', 0.5)

        # 추천 에이전트가 있으면 +0.3
        if insights.get('recommended_agents'):
            confidence += 0.3

        # 관련 개념이 많으면 +0.2
        if len(insights.get('related_concepts', [])) > 5:
            confidence += 0.2

        # 엔티티가 추출되었으면 +0.2
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
        🤖 Phase 2: LLM 최종 결정 (그래프 인사이트 활용)
        """
        # 에이전트 목록 문자열 생성
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

        # 그래프 인사이트 컨텍스트 구성
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

        # LLM 프롬프트
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

            # JSON 파싱
            json_match = re.search(
                r'\{[^{}]*"selected_agent"\s*:\s*"([^"]+)"[^{}]*"reasoning"\s*:\s*"([^"]*)"[^{}]*\}',
                response_text, re.DOTALL
            )

            if json_match:
                selected_agent = json_match.group(1)
                reasoning = json_match.group(2)

                # 선택된 에이전트 검증
                if selected_agent in available_agents:
                    return selected_agent, reasoning

                # 부분 매칭 시도
                for agent in available_agents:
                    if selected_agent.lower() in agent.lower() or agent.lower() in selected_agent.lower():
                        return agent, reasoning

            # 파싱 실패 시 그래프 추천 사용
            if graph_insights.get("recommended_agents"):
                recommended = graph_insights["recommended_agents"][0]
                return recommended["agent_id"], f"그래프 추천 (LLM 파싱 실패): {recommended.get('reason', '')}"

            logger.warning(f"⚠️ LLM 응답 파싱 실패: {response_text[:200]}")

        except Exception as e:
            logger.error(f"❌ LLM 선택 실패: {e}")

            # 그래프 추천으로 폴백
            if graph_insights.get("recommended_agents"):
                recommended = graph_insights["recommended_agents"][0]
                return recommended["agent_id"], f"그래프 기반 폴백 (LLM 실패): {str(e)[:50]}"

        # 최종 폴백
        return available_agents[0] if available_agents else 'unknown', "폴백: 첫 번째 에이전트"

    async def store_feedback(
        self,
        query: str,
        selected_agent: str,
        success: bool,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        📝 Phase 3: 피드백 저장 (v2.0 Enhanced 학습 루프)

        v2.0 개선사항:
        - LLM 기반 의미론적 분석 결과 저장
        - 카테고리 및 일반화 패턴 저장
        - 엔티티 및 키워드 저장
        - 시간 정보 정확히 기록
        """
        if not self.knowledge_graph:
            return False

        try:
            # v2.0: LLM 기반 의미 분석
            query_semantics = await self._analyze_query_semantics(query)

            # 매핑 노드 ID (v2.0: 패턴 기반으로 더 일반적인 ID)
            pattern = query_semantics.get("generalization_pattern", "general_query")
            mapping_id = f"mapping_{selected_agent}_{pattern}_{hash(query) % 1000}"

            # 기존 매핑 확인
            graph = self.knowledge_graph.graph_engine.graph

            if mapping_id in graph.nodes:
                # 기존 매핑 업데이트
                attrs = graph.nodes[mapping_id]
                usage_count = attrs.get('usage_count', 1) + 1

                # 성공률 업데이트 (지수 이동 평균 - 최근 결과에 더 높은 가중치)
                # v2.0: EMA (Exponential Moving Average) 사용
                alpha = 0.3  # 최근 결과 가중치
                old_success_rate = attrs.get('success_rate', 0.5)
                new_success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_success_rate

                attrs['usage_count'] = usage_count
                attrs['success_rate'] = new_success_rate
                attrs['last_used'] = datetime.now().isoformat()

                # v2.0: 샘플 쿼리 업데이트 (다양성 유지)
                existing_samples = attrs.get('query_samples', [])
                if len(existing_samples) < 5 and query[:50] not in existing_samples:
                    existing_samples.append(query[:50])
                    attrs['query_samples'] = existing_samples

                logger.info(
                    f"📝 피드백 업데이트: {selected_agent} ({pattern}) "
                    f"[성공률: {old_success_rate:.2f}→{new_success_rate:.2f}, 사용: {usage_count}회]"
                )

            else:
                # v2.0: 새 매핑 생성 (의미론적 분석 결과 포함)
                await self.knowledge_graph.add_concept(
                    mapping_id,
                    "query_agent_mapping",
                    {
                        "query_sample": query[:100],
                        "query_samples": [query[:50]],  # v2.0: 다양한 샘플 저장
                        "selected_agent": selected_agent,
                        # v2.0: 의미론적 분석 결과
                        "category": query_semantics.get("category", "general"),
                        "intent": query_semantics.get("intent", "search"),
                        "generalization_pattern": pattern,
                        "entities": query_semantics.get("entities", []),
                        "keywords": query_semantics.get("keywords", []),
                        # 성공률 및 사용 통계
                        "success_rate": 1.0 if success else 0.0,
                        "usage_count": 1,
                        # 시간 정보
                        "created_at": datetime.now().isoformat(),
                        "last_used": datetime.now().isoformat()
                    }
                )

                # 에이전트와 매핑 연결
                await self.knowledge_graph.add_relationship(
                    selected_agent, mapping_id, "has_mapping"
                )

                # v2.0: 카테고리 노드와도 연결 (패턴 일반화)
                category = query_semantics.get("category", "general")
                category_node_id = f"category_{category}"

                # 카테고리 노드가 없으면 생성
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
                    f"📝 새 피드백 저장: {selected_agent} "
                    f"(category={category}, pattern={pattern})"
                )

            self.stats["feedback_stored"] += 1

            return True

        except Exception as e:
            logger.warning(f"⚠️ 피드백 저장 실패: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        선택기 통계 조회 (v2.0: 확장된 통계)

        Returns:
            Dict with statistics including:
            - total_selections: 총 선택 횟수
            - graph_assisted: 그래프 보조 선택 횟수
            - llm_only: LLM 단독 선택 횟수
            - feedback_stored: 저장된 피드백 수
            - agents_synced: 동기화된 에이전트 수
            - semantic_analysis_count: LLM 의미 분석 횟수 (v2.0)
            - pattern_generalizations: 패턴 일반화 매칭 횟수 (v2.0)
            - time_decay_applied: 시간 감쇠 적용 횟수 (v2.0)
            - graph_assist_ratio: 그래프 보조 비율
        """
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
            "version": "2.0"
        }

    def clear_category_cache(self):
        """v2.0: 카테고리 캐시 초기화"""
        self._category_cache.clear()
        logger.info("🧹 카테고리 캐시 초기화 완료")


# 싱글톤 인스턴스
_hybrid_selector_instance = None


def get_hybrid_selector() -> HybridAgentSelector:
    """하이브리드 선택기 싱글톤 인스턴스 반환"""
    global _hybrid_selector_instance
    if _hybrid_selector_instance is None:
        _hybrid_selector_instance = HybridAgentSelector()
    return _hybrid_selector_instance


logger.info("🧠 하이브리드 에이전트 선택기 v2.0 모듈 로드 완료 (시간감쇠 + LLM 의미분석 + 패턴일반화)")
