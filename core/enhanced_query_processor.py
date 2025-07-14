"""
🧠 고도화된 쿼리 프로세서
Enhanced Query Processor

복잡한 쿼리를 의미 분석하고, 적절한 에이전트를 선택하며, 
각 에이전트에 최적화된 쿼리를 생성하여 전달하는 통합 시스템
"""

import asyncio
import json
import uuid
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from .models import SemanticQuery, ExecutionStrategy, AgentExecutionResult, WorkflowPlan
from .llm_manager import get_ontology_llm_manager, OntologyLLMType


class QueryDecomposition:
    """쿼리 분해 결과"""
    def __init__(self, 
                 query_parts: List[Dict[str, Any]], 
                 execution_strategy: ExecutionStrategy,
                 dependencies: Dict[str, List[str]] = None):
        self.query_parts = query_parts
        self.execution_strategy = execution_strategy
        self.dependencies = dependencies or {}
        self.created_at = datetime.now()


class AgentQueryMapping:
    """에이전트-쿼리 매핑"""
    def __init__(self, 
                 agent_id: str, 
                 optimized_query: str, 
                 context: Dict[str, Any],
                 priority: int = 1,
                 depends_on: List[str] = None):
        self.agent_id = agent_id
        self.optimized_query = optimized_query
        self.context = context
        self.priority = priority
        self.depends_on = depends_on or []
        self.mapping_id = str(uuid.uuid4())


class EnhancedQueryProcessor:
    """🧠 고도화된 쿼리 프로세서 - LLM 분석 + 키워드 매칭 조합"""
    
    def __init__(self):
        self.llm_manager = get_ontology_llm_manager()
        self.installed_agents_info = []
        
        # 핵심 키워드 매핑 - 실제 에이전트 타입 기반으로 단순화
        self.core_keyword_mappings = {
            "WEATHER": ["날씨", "기온", "예보", "weather", "temperature"],
            "CURRENCY": ["환율", "달러", "유로", "엔", "원", "currency", "exchange"],
            "CRAWLER": ["주가", "주식", "삼성전자", "크롤링", "stock", "price"],
            "INTERNET_SEARCH": ["검색", "찾아", "알아봐", "정보", "인터넷", "웹", "조사"],
            "MEMO": ["저장", "기록", "메모", "보관", "기억"],
            "CALCULATOR": ["계산", "더하기", "빼기", "곱하기", "나누기", "수학"],
            "SCHEDULER": ["일정", "스케줄", "약속", "캘린더"],
            "RESTAURANT_FINDER": ["맛집", "음식점", "레스토랑", "식당"],
            "FORECASTING": ["운세", "점", "오늘의", "fortune"],
        }
        
        # 에이전트별 특화된 쿼리 템플릿
        self.agent_query_templates = {
            "weather_agent": "오늘 {location}의 날씨 정보를 자세히 알려주세요. 온도, 습도, 강수확률을 포함해주세요.",
            "currency_exchange_agent": "현재 {currency} 환율 정보를 알려주세요. 매수/매도 가격과 전일 대비 변동률을 포함해주세요.",
            "crawler_agent": "{company} 주가 정보를 알려주세요. 현재가, 변동률, 거래량을 포함해주세요.",
            "internet_agent": "{topic}에 대해 최신 정보를 검색하여 알려주세요.",
            "memo_agent": "다음 내용을 메모로 저장해주세요: {content}",
            "calculator_agent": "{expression}을 계산해주세요.",
            "scheduler_agent": "{schedule} 일정을 관리해주세요.",
            "restaurant_finder_agent": "{location}에서 {cuisine} 맛집을 찾아주세요."
        }

    def set_installed_agents_info(self, installed_agents_info: List[Dict[str, Any]]):
        """설치된 에이전트 정보 설정"""
        self.installed_agents_info = installed_agents_info
        logger.info(f"🎯 설치된 에이전트 정보 업데이트: {len(installed_agents_info)}개")

    async def process_complex_query(self, 
                                  query: str, 
                                  available_agents: List[str]) -> Tuple[QueryDecomposition, List[AgentQueryMapping]]:
        """
        복잡한 쿼리를 처리하여 에이전트별 최적화된 쿼리 생성
        LLM 분석 + 키워드 매칭 조합으로 향상된 정확도
        
        Args:
            query: 원본 사용자 쿼리
            available_agents: 사용 가능한 에이전트 목록
            
        Returns:
            Tuple[QueryDecomposition, List[AgentQueryMapping]]: 쿼리 분해 결과와 에이전트 매핑
        """
        try:
            logger.info(f"🧠 향상된 쿼리 처리 시작: {query}")
            
            # 1. LLM을 활용한 쿼리 분석 및 분해
            llm_analysis = await self._analyze_query_with_llm(query, available_agents)
            
            # 2. 키워드 기반 백업 분석
            keyword_analysis = self._analyze_query_with_keywords(query, available_agents)
            
            # 3. LLM과 키워드 분석 결과 통합
            final_analysis = self._merge_analysis_results(llm_analysis, keyword_analysis, query)
            
            # 4. 분해 및 매핑 생성
            decomposition, mappings = self._create_enhanced_decomposition_and_mappings(
                query, final_analysis, available_agents
            )
            
            logger.info(f"🎯 향상된 처리 완료: {len(mappings)}개 매핑 생성")
            return decomposition, mappings
            
        except Exception as e:
            logger.error(f"향상된 쿼리 처리 실패: {e}")
            # 폴백: 키워드 기반 처리
            return self._create_fallback_result(query, available_agents)

    async def _analyze_query_with_llm(self, query: str, available_agents: List[str]) -> Dict[str, Any]:
        """LLM을 활용한 쿼리 분석"""
        try:
            # 실제 에이전트 정보 구성
            agents_info_str = self._build_agents_summary(available_agents)
            
            prompt = f"""
당신은 사용자 쿼리를 분석하여 적절한 에이전트를 선택하는 전문가입니다.

**사용자 쿼리:** "{query}"

**사용 가능한 에이전트:**
{agents_info_str}

**분석 원칙:**
1. 쿼리를 독립적인 작업들로 분해
2. 각 작업에 가장 적합한 에이전트 선택
3. 에이전트별로 최적화된 메시지 생성

**분석 예시:**
- "오늘 날씨와 환율 알려줘" → 날씨(weather_agent) + 환율(currency_exchange_agent)
- "삼성전자 주가 검색해줘" → 주가 조회(crawler_agent)

**응답 형식 (JSON):**
{{
    "multi_task": true,
    "tasks": [
        {{
            "task_type": "weather_inquiry",
            "keywords": ["날씨", "오늘"],
            "best_agent": "weather_agent",
            "optimized_message": "오늘 서울의 날씨 정보를 알려주세요",
            "confidence": 0.95
        }}
    ],
    "reasoning": "분석 근거"
}}
"""
            
            # LLM 호출
            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(prompt)
            
            # 응답 처리
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # JSON 파싱
            try:
                # JSON 블록 추출
                if '```json' in response_text:
                    start = response_text.find('```json') + 7
                    end = response_text.find('```', start)
                    json_str = response_text[start:end].strip()
                else:
                    json_str = response_text.strip()
                
                result = json.loads(json_str)
                logger.info(f"✅ LLM 분석 성공: {len(result.get('tasks', []))}개 작업 식별")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"LLM 응답 JSON 파싱 실패: {e}")
                return {}
            
        except Exception as e:
            logger.error(f"LLM 쿼리 분석 실패: {e}")
            return {}

    def _analyze_query_with_keywords(self, query: str, available_agents: List[str]) -> Dict[str, Any]:
        """키워드 기반 쿼리 분석 (백업 시스템)"""
        query_lower = query.lower()
        detected_tasks = []
        
        # 연결어로 쿼리 분리
        segments = self._split_query_by_connectors(query)
        
        for i, segment in enumerate(segments):
            segment_lower = segment.lower()
            best_match = None
            best_score = 0
            
            # 각 에이전트별 매칭 점수 계산
            for agent_id in available_agents:
                agent_type = self._get_agent_type_from_installed_info(agent_id)
                score = self._calculate_keyword_score(segment_lower, agent_type, agent_id)
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        "task_type": agent_type.lower() if agent_type else "general",
                        "keywords": self._extract_keywords_from_segment(segment),
                        "best_agent": agent_id,
                        "optimized_message": self._generate_optimized_message(segment, agent_id),
                        "confidence": min(score / 30.0, 0.9)  # 정규화
                    }
            
            if best_match and best_match["confidence"] > 0.3:
                detected_tasks.append(best_match)
        
        return {
            "multi_task": len(detected_tasks) > 1,
            "tasks": detected_tasks,
            "reasoning": "키워드 기반 분석"
        }

    def _split_query_by_connectors(self, query: str) -> List[str]:
        """연결어로 쿼리 분리"""
        # 연결어 패턴
        connectors = ['그리고', '하고', '또', '그런데', '그다음에', '그리고나서', '또한', '과', '와', ',']
        
        segments = [query]
        for connector in connectors:
            new_segments = []
            for segment in segments:
                new_segments.extend([s.strip() for s in segment.split(connector)])
            segments = new_segments
        
        # 빈 문자열 제거
        return [s for s in segments if s.strip()]

    def _calculate_keyword_score(self, segment: str, agent_type: str, agent_id: str) -> int:
        """키워드 매칭 점수 계산"""
        score = 0
        
        # 1. 에이전트 타입별 키워드 매칭
        if agent_type and agent_type in self.core_keyword_mappings:
            keywords = self.core_keyword_mappings[agent_type]
            for keyword in keywords:
                if keyword in segment:
                    score += 15
        
        # 2. 에이전트 ID 직접 매칭
        agent_keywords = {
            'weather_agent': ['날씨', '기온', '온도'],
            'currency_exchange_agent': ['환율', '달러', '원'],
            'crawler_agent': ['주가', '주식', '삼성전자'],
            'internet_agent': ['검색', '정보', '알려줘']
        }
        
        if agent_id in agent_keywords:
            for keyword in agent_keywords[agent_id]:
                if keyword in segment:
                    score += 20
        
        # 3. 특별 패턴 매칭
        if '날씨' in segment and 'weather' in agent_id:
            score += 25
        if '환율' in segment and 'currency' in agent_id:
            score += 25
        if any(word in segment for word in ['주가', '주식', '삼성전자']) and 'crawler' in agent_id:
            score += 25
        
        return score

    def _extract_keywords_from_segment(self, segment: str) -> List[str]:
        """세그먼트에서 키워드 추출"""
        # 간단한 키워드 추출
        keywords = []
        common_words = ['오늘', '정보', '알려줘', '확인', '검색']
        
        for word in common_words:
            if word in segment:
                keywords.append(word)
        
        # 특정 도메인 키워드
        domain_keywords = ['날씨', '환율', '주가', '삼성전자', '달러', '원']
        for word in domain_keywords:
            if word in segment:
                keywords.append(word)
        
        return keywords

    def _generate_optimized_message(self, segment: str, agent_id: str) -> str:
        """에이전트별 최적화된 메시지 생성"""
        if agent_id in self.agent_query_templates:
            template = self.agent_query_templates[agent_id]
            
            # 템플릿별 파라미터 추출 및 적용
            if 'weather_agent' in agent_id:
                return template.format(location="서울")
            elif 'currency_exchange_agent' in agent_id:
                return template.format(currency="원달러")
            elif 'crawler_agent' in agent_id:
                if '삼성전자' in segment:
                    return template.format(company="삼성전자")
                else:
                    return template.format(company="관련 종목")
            elif 'internet_agent' in agent_id:
                # 핵심 키워드 추출
                topic = segment.replace('알려줘', '').replace('검색', '').strip()
                return template.format(topic=topic)
        
        # 기본 메시지
        return f"{segment} - {agent_id}를 통해 처리해주세요"

    def _merge_analysis_results(self, llm_analysis: Dict[str, Any], 
                              keyword_analysis: Dict[str, Any], 
                              query: str) -> Dict[str, Any]:
        """LLM과 키워드 분석 결과 통합"""
        
        # LLM 분석이 성공한 경우 우선 사용
        if llm_analysis and llm_analysis.get('tasks'):
            logger.info("🧠 LLM 분석 결과 사용")
            return llm_analysis
        
        # LLM 분석 실패 시 키워드 분석 사용
        if keyword_analysis and keyword_analysis.get('tasks'):
            logger.info("🔤 키워드 분석 결과 사용")
            return keyword_analysis
        
        # 둘 다 실패한 경우 기본 분석
        logger.warning("⚠️ 기본 분석으로 폴백")
        return {
            "multi_task": False,
            "tasks": [{
                "task_type": "general",
                "keywords": [],
                "best_agent": self._select_default_agent(query, []),
                "optimized_message": f"{query} - 이 요청을 처리해주세요",
                "confidence": 0.5
            }],
            "reasoning": "폴백 분석"
        }

    def _create_enhanced_decomposition_and_mappings(self, 
                                                  query: str,
                                                  analysis: Dict[str, Any],
                                                  available_agents: List[str]) -> Tuple[QueryDecomposition, List[AgentQueryMapping]]:
        """향상된 분해 및 매핑 생성"""
        
        tasks = analysis.get('tasks', [])
        is_multi_task = analysis.get('multi_task', False)
        
        # 실행 전략 결정
        if len(tasks) <= 1:
            strategy = ExecutionStrategy.SINGLE_AGENT
        elif len(tasks) == 2:
            strategy = ExecutionStrategy.PARALLEL
        else:
            strategy = ExecutionStrategy.HYBRID
        
        # 쿼리 부분 생성
        query_parts = []
        for i, task in enumerate(tasks):
            query_parts.append({
                "part_id": f"task_{i+1}",
                "description": task.get('task_type', 'general'),
                "keywords": task.get('keywords', []),
                "required_agents": [task.get('best_agent', 'internet_agent')],
                "priority": 1,
                "estimated_time": 15.0,
                "confidence": task.get('confidence', 0.8)
            })
        
        # 분해 결과 생성
        decomposition = QueryDecomposition(
            query_parts=query_parts,
            execution_strategy=strategy,
            dependencies={}
        )

        # 매핑 생성
        mappings = []
        for i, task in enumerate(tasks):
            agent_id = task.get('best_agent', 'internet_agent')
            optimized_message = task.get('optimized_message', query)
            
            mapping = AgentQueryMapping(
                agent_id=agent_id,
                optimized_query=optimized_message,
                context={
                    "task_type": task.get('task_type', 'general'),
                    "expected_output": "text",
                    "priority": "medium",
                    "original_query": query,
                    "confidence": task.get('confidence', 0.8),
                    "keywords": task.get('keywords', [])
                },
                priority=1,
                depends_on=[]
            )
            mappings.append(mapping)
            
            logger.info(f"🎯 향상된 매핑 생성: {agent_id} -> {optimized_message[:50]}...")
        
        return decomposition, mappings

    def _build_agents_summary(self, available_agents: List[str]) -> str:
        """에이전트 정보 요약 생성 - tags 포함으로 개선"""
        summary_lines = []
        
        for agent_id in available_agents:
            agent_info = self._find_agent_info(agent_id)
            
            if agent_info:
                agent_data = agent_info.get('agent_data', {})
                name = agent_data.get('name', agent_id)
                agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
                description = agent_data.get('description', '')
                tags = agent_data.get('tags', [])
                capabilities = agent_data.get('capabilities', [])
                
                # 주요 능력 3개 추출
                main_capabilities = []
                for cap in capabilities[:3]:
                    if isinstance(cap, dict):
                        cap_name = cap.get('name', '알 수 없음')
                        main_capabilities.append(cap_name)
                    else:
                        main_capabilities.append(str(cap))
                
                # 에이전트 정보 구성 (tags 포함)
                summary_lines.append(f"- {agent_id}: {name} ({agent_type})")
                if description:
                    summary_lines.append(f"  설명: {description[:80]}...")
                if main_capabilities:
                    summary_lines.append(f"  능력: {', '.join(main_capabilities)}")
                if tags:
                    # 태그를 5개까지만 표시
                    display_tags = tags[:5]
                    if len(tags) > 5:
                        display_tags.append(f"외 {len(tags)-5}개")
                    summary_lines.append(f"  🏷️ 태그: {', '.join(display_tags)}")
            else:
                summary_lines.append(f"- {agent_id}: {self._infer_type_from_id(agent_id)} 에이전트")
        
        return "\n".join(summary_lines)

    def _find_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """에이전트 정보 찾기"""
        for agent_info in self.installed_agents_info:
            if agent_info.get('agent_id') == agent_id:
                return agent_info
        return None

    def _get_agent_type_from_installed_info(self, agent_id: str) -> Optional[str]:
        """설치된 에이전트 정보에서 타입 추출"""
        agent_info = self._find_agent_info(agent_id)
        if agent_info:
            agent_data = agent_info.get('agent_data', {})
            return agent_data.get('metadata', {}).get('agent_type', '')
        return self._infer_type_from_id(agent_id)

    def _infer_type_from_id(self, agent_id: str) -> str:
        """에이전트 ID에서 타입 추론"""
        agent_id_lower = agent_id.lower()
        
        if 'weather' in agent_id_lower:
            return 'WEATHER'
        elif 'currency' in agent_id_lower:
            return 'CURRENCY'
        elif 'crawler' in agent_id_lower:
            return 'CRAWLER'
        elif any(keyword in agent_id_lower for keyword in ['internet', 'search', 'web']):
            return 'INTERNET_SEARCH'
        elif any(keyword in agent_id_lower for keyword in ['memo', 'note', 'save']):
            return 'MEMO'
        elif any(keyword in agent_id_lower for keyword in ['calc', 'calculator', 'math']):
            return 'CALCULATOR'
        elif any(keyword in agent_id_lower for keyword in ['schedule', 'calendar']):
            return 'SCHEDULER'
        elif any(keyword in agent_id_lower for keyword in ['restaurant', 'food']):
            return 'RESTAURANT_FINDER'
        elif any(keyword in agent_id_lower for keyword in ['fortune', 'daily']):
            return 'FORECASTING'
        else:
            return 'CUSTOM'

    def _select_default_agent(self, query: str, available_agents: List[str]) -> str:
        """기본 에이전트 선택"""
        if available_agents:
            # 인터넷 검색 에이전트 우선
            for agent_id in available_agents:
                if any(keyword in agent_id.lower() for keyword in ['internet', 'search']):
                    return agent_id
            return available_agents[0]
        return "internet_agent"

    def _create_fallback_result(self, query: str, available_agents: List[str]) -> Tuple[QueryDecomposition, List[AgentQueryMapping]]:
        """폴백 결과 생성"""
        logger.warning("🔄 향상된 프로세서 폴백 모드")
        
        # 기본 에이전트 선택
        selected_agent = self._select_default_agent(query, available_agents)
        
        query_parts = [{
            "part_id": "fallback_task",
            "description": "폴백 작업 처리",
            "keywords": [],
            "required_agents": [selected_agent],
            "priority": 1,
            "estimated_time": 30.0
        }]
        
        decomposition = QueryDecomposition(
            query_parts=query_parts,
            execution_strategy=ExecutionStrategy.SINGLE_AGENT,
            dependencies={}
        )
        
        mappings = [AgentQueryMapping(
            agent_id=selected_agent,
            optimized_query=f"{query} - 이 요청을 처리해주세요",
            context={
                "task_type": "general",
                "expected_output": "text",
                "priority": "medium",
                "original_query": query,
                "fallback_mode": True
            },
            priority=1,
            depends_on=[]
        )]
        
        return decomposition, mappings


# 전역 인스턴스
_query_processor = None

def get_enhanced_query_processor() -> EnhancedQueryProcessor:
    """전역 쿼리 프로세서 인스턴스 반환"""
    global _query_processor
    if _query_processor is None:
        _query_processor = EnhancedQueryProcessor()
    return _query_processor


logger.info("🧠 고도화된 쿼리 프로세서 로드 완료! (LLM + 키워드 조합)") 