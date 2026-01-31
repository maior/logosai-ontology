"""
🧠 Hybrid Agent Selector
지식그래프 + LLM 하이브리드 에이전트 선택기

Phase 1: Knowledge Graph Analysis
  - 엔티티 추출 및 관련 개념 탐색
  - 과거 성공 패턴 조회
  - 그래프 기반 후보 추천

Phase 2: LLM Final Decision
  - 그래프 인사이트를 컨텍스트로 활용
  - 의미론적 분석 + 그래프 근거로 최종 결정

Phase 3: Feedback Loop
  - 성공적인 쿼리-에이전트 매핑을 그래프에 저장
  - 시간이 지날수록 더 정확한 추천
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .llm_manager import get_ontology_llm_manager, OntologyLLMType


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
            "agents_synced": 0
        }

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
        """과거 성공 패턴 조회"""
        patterns = []

        if not self.knowledge_graph:
            return patterns

        try:
            graph = self.knowledge_graph.graph_engine.graph

            # 쿼리 의도 키워드로 과거 패턴 검색
            intent_keywords = self._extract_intent_keywords(query)

            for node_id, attrs in graph.nodes(data=True):
                node_type = attrs.get('type', '')

                # 과거 성공적인 쿼리-에이전트 매핑 찾기
                if node_type == 'query_agent_mapping':
                    past_intent = attrs.get('intent_keywords', [])
                    past_agent = attrs.get('selected_agent', '')
                    success_rate = attrs.get('success_rate', 0.0)

                    # 의도 키워드 매칭
                    if any(kw in past_intent for kw in intent_keywords):
                        patterns.append({
                            'agent': past_agent,
                            'intent_keywords': past_intent,
                            'success_rate': success_rate,
                            'usage_count': attrs.get('usage_count', 1)
                        })

                # 에이전트-도메인 매핑 찾기
                if node_type == 'agent':
                    agent_id = node_id
                    # 이 에이전트와 연결된 도메인/개념 확인
                    for neighbor in graph.neighbors(agent_id):
                        neighbor_attrs = graph.nodes.get(neighbor, {})
                        if neighbor_attrs.get('type') == 'domain':
                            domain_name = neighbor_attrs.get('domain_name', '')
                            # 쿼리와 도메인 매칭
                            if any(kw in domain_name.lower() for kw in intent_keywords):
                                patterns.append({
                                    'agent': agent_id,
                                    'domain': domain_name,
                                    'success_rate': attrs.get('success_rate', 0.5),
                                    'usage_count': attrs.get('usage_count', 1)
                                })

            # 성공률 기준 정렬
            patterns.sort(key=lambda x: (x.get('success_rate', 0), x.get('usage_count', 0)), reverse=True)

        except Exception as e:
            logger.warning(f"⚠️ 과거 패턴 조회 실패: {e}")

        return patterns[:5]  # 상위 5개만

    def _extract_intent_keywords(self, query: str) -> List[str]:
        """쿼리에서 의도 키워드 추출"""
        intent_map = {
            '검색': ['검색', '찾아', '알려', '보여'],
            '분석': ['분석', '비교', '평가', '리뷰'],
            '계산': ['계산', '얼마', '몇', '환산'],
            '시각화': ['그래프', '차트', '시각화', '그려'],
            '정보': ['뉴스', '소식', '정보', '현황'],
            '실시간': ['지금', '현재', '오늘', '실시간'],
            '금융': ['주가', '환율', '가격', '시세', '주식'],
            '날씨': ['날씨', '기온', '비', '눈'],
            '쇼핑': ['구매', '주문', '쇼핑', '상품', '가격비교']
        }

        keywords = []
        query_lower = query.lower()

        for intent, triggers in intent_map.items():
            if any(t in query_lower for t in triggers):
                keywords.append(intent)

        return keywords

    async def _derive_agent_recommendations(
        self,
        entities: List[str],
        related_concepts: List[str],
        past_patterns: List[Dict[str, Any]],
        available_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """그래프 기반 에이전트 추천 도출"""
        recommendations = []
        agent_scores = {}

        # 1. 과거 패턴 기반 점수
        for pattern in past_patterns:
            agent = pattern.get('agent', '')
            if agent in available_agents:
                score = pattern.get('success_rate', 0.5) * pattern.get('usage_count', 1) * 0.1
                agent_scores[agent] = agent_scores.get(agent, 0) + score

        # 2. 도메인 연관성 기반 점수 (그래프 탐색)
        if self.knowledge_graph and hasattr(self.knowledge_graph, 'graph_engine'):
            graph = self.knowledge_graph.graph_engine.graph

            for agent_id in available_agents:
                if agent_id in graph.nodes:
                    # 에이전트와 연결된 개념들
                    connected = list(graph.neighbors(agent_id))

                    # 현재 쿼리의 엔티티/개념과 겹치는 정도
                    overlap_entities = len(set(connected) & set(entities))
                    overlap_concepts = len(set(connected) & set(related_concepts))

                    if overlap_entities > 0 or overlap_concepts > 0:
                        score = overlap_entities * 0.3 + overlap_concepts * 0.1
                        agent_scores[agent_id] = agent_scores.get(agent_id, 0) + score

        # 추천 목록 생성
        for agent_id, score in sorted(agent_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                recommendations.append({
                    'agent_id': agent_id,
                    'graph_score': round(score, 3),
                    'reason': 'graph_based_recommendation'
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
        📝 Phase 3: 피드백 저장 (학습 루프)

        성공적인 쿼리-에이전트 매핑을 그래프에 저장하여
        향후 유사한 쿼리에 대해 더 나은 추천을 제공합니다.
        """
        if not self.knowledge_graph:
            return False

        try:
            # 의도 키워드 추출
            intent_keywords = self._extract_intent_keywords(query)

            # 매핑 노드 ID
            mapping_id = f"mapping_{selected_agent}_{hash(query) % 10000}"

            # 기존 매핑 확인
            graph = self.knowledge_graph.graph_engine.graph

            if mapping_id in graph.nodes:
                # 기존 매핑 업데이트
                attrs = graph.nodes[mapping_id]
                usage_count = attrs.get('usage_count', 1) + 1

                # 성공률 업데이트 (이동 평균)
                old_success_rate = attrs.get('success_rate', 0.5)
                new_success_rate = (old_success_rate * (usage_count - 1) + (1.0 if success else 0.0)) / usage_count

                attrs['usage_count'] = usage_count
                attrs['success_rate'] = new_success_rate
                attrs['last_used'] = datetime.now().isoformat()

            else:
                # 새 매핑 생성
                await self.knowledge_graph.add_concept(
                    mapping_id,
                    "query_agent_mapping",
                    {
                        "query_sample": query[:100],
                        "selected_agent": selected_agent,
                        "intent_keywords": intent_keywords,
                        "success_rate": 1.0 if success else 0.0,
                        "usage_count": 1,
                        "created_at": datetime.now().isoformat(),
                        "last_used": datetime.now().isoformat()
                    }
                )

                # 에이전트와 매핑 연결
                await self.knowledge_graph.add_relationship(
                    selected_agent, mapping_id, "has_mapping"
                )

            self.stats["feedback_stored"] += 1
            logger.info(f"📝 피드백 저장 완료: {selected_agent} ({'성공' if success else '실패'})")

            return True

        except Exception as e:
            logger.warning(f"⚠️ 피드백 저장 실패: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """선택기 통계 조회"""
        return {
            **self.stats,
            "graph_assist_ratio": (
                self.stats["graph_assisted"] / max(self.stats["total_selections"], 1)
            )
        }


# 싱글톤 인스턴스
_hybrid_selector_instance = None


def get_hybrid_selector() -> HybridAgentSelector:
    """하이브리드 선택기 싱글톤 인스턴스 반환"""
    global _hybrid_selector_instance
    if _hybrid_selector_instance is None:
        _hybrid_selector_instance = HybridAgentSelector()
    return _hybrid_selector_instance


logger.info("🧠 하이브리드 에이전트 선택기 모듈 로드 완료")
