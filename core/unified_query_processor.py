"""
🚀 통합 쿼리 프로세서
Unified Query Processor

기존의 여러 번 나뉜 LLM 호출을 하나로 통합하여 
쿼리 분석 → 에이전트 선택 → 워크플로우 설계 → 쿼리 최적화까지 
한 번의 LLM 호출로 처리하는 효율적인 시스템

🔄 v3.1: 다국어 지원 및 하드코딩 제거
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


class LanguageConfig:
    """언어별 설정 클래스"""
    
    def __init__(self):
        self.configs = {
            'ko': {
                'name': '한국어',
                'stopwords': {'은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '하고', '그리고', '또는', '또'},
                'connectors': ['그리고', '하고', '또', '그런데', '그다음에', '그리고나서', '또한', '과', '와', ',', '한 다음', '후에', '다음에', '이후'],
                'sequential_patterns': [
                    r'(.+?)(?:확인|조회|검색|찾기|알아보기).*?(?:하고|한\s*다음|후에|다음에|이후).*?(.+?)(?:계산|분석|처리|변환|생성)',
                    r'(.+?)(?:가져오기|수집|조회).*?(?:하고|한\s*다음|후에|다음에|이후).*?(.+?)(?:분석|처리|계산|변환)',
                    r'(.+?)(?:시세|가격|정보).*?(?:확인|조회).*?(?:하고|한\s*다음|후에|다음에|이후).*?(.+?)(?:계산|변환|분석)',
                    r'(.+?)(?:데이터|정보).*?(?:수집|조회|확인).*?(?:하고|한\s*다음|후에|다음에|이후).*?(.+?)(?:분석|처리|생성)',
                    r'(.+?)(?:주가|환율|시세).*?(?:확인|조회).*?(?:하고|한\s*다음|후에|다음에|이후).*?(.+?)(?:분석|추천|평가)',
                    r'(.+?)(?:조사|수집|검색).*?(?:해서|하고|한\s*다음).*?(.+?)(?:분석|처리).*?(?:하고|한\s*다음).*?(.+?)(?:만들어|생성|작성)',
                    r'(.+?)(?:트렌드|동향|현황).*?(?:조사|분석).*?(?:해서|하고|한\s*다음).*?(.+?)(?:보고서|리포트|문서).*?(?:만들어|생성|작성)',
                    r'(.+?)(?:조사|검색|수집).*?(?:해서|하고).*?(.+?)(?:분석|정리).*?(?:하고|한\s*다음).*?(.+?)(?:시각|그래프|차트)',
                    r'(.+?)(?:정보|데이터).*?(?:조사|수집).*?(?:해서|하고).*?(.+?)(?:분석|처리).*?(?:하고).*?(.+?)(?:보고서|문서|리포트)',
                    r'(.+?)(?:기술|트렌드|동향).*?(?:조사|분석).*?(?:해서|하고).*?(.+?)(?:시각|그래프|차트|보고서)'
                ],
                'intent_keywords': {
                    'search': ['검색', '찾기', '조회', '알아보기', '확인', '알려줘', '알아봐'],
                    'analyze': ['분석', '해석', '평가', '검토'],
                    'calculate': ['계산', '산출', '변환', '환산'],
                    'generate': ['생성', '만들기', '작성', '제작'],
                    'visualize': ['시각화', '차트', '그래프', '도표'],
                    'collect': ['수집', '모으기', '가져오기', '크롤링'],
                    'compare': ['비교', '대조', '견주기'],
                    'summarize': ['요약', '정리', '종합'],
                    # 🚀 여행/추천 의도 추가
                    'recommend': ['추천', '추천해줘', '추천해', '좋을까', '괜찮을까', '어떨까', '어디가', '뭐가'],
                    'plan': ['계획', '일정', '코스', '여정', '플랜']
                },
                'agent_keywords': {
                    'weather': ['날씨', '기상', '기온', '온도', '일기예보'],
                    'currency': ['환율', '달러', '유로', '엔', '원', '통화'],
                    'stock': ['주가', '주식', '증권', '종목', '투자'],
                    'calculation': ['계산', '수학', '산수'],
                    'search': ['검색', '정보', '조사', '알려줘', '알아봐'],
                    'analysis': ['분석', '해석', '평가'],
                    'visualization': ['시각화', '차트', '그래프'],
                    # 🗓️ 일정/캘린더 도메인 추가
                    'schedule': ['일정', '스케줄', '약속', '캘린더', '등록', '추가해줘', '추가해', '알려줘', '목요일', '금요일', '이번주', '다음주'],
                    'calendar': ['일정', '스케줄', '약속', '캘린더', '등록', '추가해줘', '추가해', '알려줘', '이번주', '다음주'],
                    # 🚀 여행/추천 도메인 추가
                    'travel': ['여행', '관광', '여행지', '가볼만한', '가볼곳', '방문', '코스'],
                    'recommendation': ['추천', '추천해줘', '추천해', '어디', '뭐', '뭘', '좋을까', '괜찮을까', '어떨까'],
                    'food': ['맛집', '음식', '식당', '레스토랑', '카페', '먹을곳', '먹거리', '맛있는'],
                    'attraction': ['명소', '관광지', '볼거리', '볼곳', '핫플', '핫플레이스'],
                    'accommodation': ['숙소', '호텔', '펜션', '리조트', '민박', '숙박']
                }
            },
            'en': {
                'name': 'English',
                'stopwords': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'},
                'connectors': ['and', 'then', 'after', 'next', 'following', 'subsequently', 'afterwards', ',', 'and then', 'after that'],
                'sequential_patterns': [
                    r'(.+?)(?:check|verify|search|find|get).*?(?:and|then|after|next).*?(.+?)(?:calculate|analyze|process|convert|generate)',
                    r'(.+?)(?:collect|gather|fetch).*?(?:and|then|after|next).*?(.+?)(?:analyze|process|calculate|transform)',
                    r'(.+?)(?:price|rate|data|info|information).*?(?:check|get).*?(?:and|then|after|next).*?(.+?)(?:calculate|convert|analyze)',
                    r'(.+?)(?:data|information).*?(?:collect|gather|get).*?(?:and|then|after|next).*?(.+?)(?:analyze|process|generate)',
                    r'(.+?)(?:stock|exchange|rate).*?(?:check|get).*?(?:and|then|after|next).*?(.+?)(?:analyze|recommend|evaluate)',
                    r'(.+?)(?:research|collect|search).*?(?:and|then).*?(.+?)(?:analyze|process).*?(?:and|then).*?(.+?)(?:create|generate|make)',
                    r'(.+?)(?:trend|status|current).*?(?:research|analyze).*?(?:and|then).*?(.+?)(?:report|document).*?(?:create|generate|make)',
                    r'(.+?)(?:research|search|collect).*?(?:and|then).*?(.+?)(?:analyze|organize).*?(?:and|then).*?(.+?)(?:visual|chart|graph)',
                    r'(.+?)(?:information|data).*?(?:research|collect).*?(?:and|then).*?(.+?)(?:analyze|process).*?(?:and).*?(.+?)(?:report|document)',
                    r'(.+?)(?:technology|trend).*?(?:research|analyze).*?(?:and|then).*?(.+?)(?:visual|chart|graph|report)'
                ],
                'intent_keywords': {
                    'search': ['search', 'find', 'look', 'check', 'get', 'fetch', 'tell me'],
                    'analyze': ['analyze', 'examine', 'evaluate', 'review'],
                    'calculate': ['calculate', 'compute', 'convert', 'transform'],
                    'generate': ['generate', 'create', 'make', 'produce'],
                    'visualize': ['visualize', 'chart', 'graph', 'plot'],
                    'collect': ['collect', 'gather', 'fetch', 'crawl'],
                    'compare': ['compare', 'contrast', 'versus'],
                    'summarize': ['summarize', 'sum up', 'conclude'],
                    # 🚀 Travel/Recommendation intents added
                    'recommend': ['recommend', 'suggest', 'best', 'top', 'where should', 'what should'],
                    'plan': ['plan', 'itinerary', 'schedule', 'route']
                },
                'agent_keywords': {
                    'weather': ['weather', 'temperature', 'climate', 'forecast'],
                    'currency': ['currency', 'exchange', 'dollar', 'euro', 'yen'],
                    'stock': ['stock', 'share', 'equity', 'market', 'investment'],
                    'calculation': ['calculation', 'math', 'arithmetic'],
                    'search': ['search', 'information', 'research', 'find', 'look'],
                    'analysis': ['analysis', 'analytics', 'evaluation'],
                    'visualization': ['visualization', 'chart', 'graph'],
                    # 🗓️ Schedule/Calendar domains added
                    'schedule': ['schedule', 'appointment', 'calendar', 'meeting', 'event', 'add', 'register', 'show', 'this week', 'next week', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
                    'calendar': ['schedule', 'appointment', 'calendar', 'meeting', 'event', 'add', 'register', 'show', 'this week', 'next week'],
                    # 🚀 Travel/Recommendation domains added
                    'travel': ['travel', 'trip', 'tour', 'visit', 'destination', 'itinerary', 'vacation'],
                    'recommendation': ['recommend', 'suggestion', 'where', 'what', 'which', 'best', 'top'],
                    'food': ['food', 'restaurant', 'cafe', 'dining', 'eat', 'cuisine', 'meal'],
                    'attraction': ['attraction', 'sightseeing', 'landmark', 'tourist', 'place'],
                    'accommodation': ['hotel', 'accommodation', 'stay', 'lodging', 'resort', 'hostel']
                }
            }
        }
    
    def detect_language(self, text: str) -> str:
        """텍스트에서 언어 감지"""
        # 간단한 언어 감지 로직
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if korean_chars > english_chars:
            return 'ko'
        else:
            return 'en'
    
    def get_config(self, language: str) -> Dict[str, Any]:
        """언어별 설정 반환"""
        return self.configs.get(language, self.configs['en'])
    
    def get_stopwords(self, language: str) -> set:
        """언어별 불용어 반환"""
        return self.get_config(language).get('stopwords', set())
    
    def get_connectors(self, language: str) -> List[str]:
        """언어별 연결어 반환"""
        return self.get_config(language).get('connectors', [])
    
    def get_sequential_patterns(self, language: str) -> List[str]:
        """언어별 순차 패턴 반환"""
        return self.get_config(language).get('sequential_patterns', [])
    
    def get_intent_keywords(self, language: str) -> Dict[str, List[str]]:
        """언어별 의도 키워드 반환"""
        return self.get_config(language).get('intent_keywords', {})
    
    def get_agent_keywords(self, language: str) -> Dict[str, List[str]]:
        """언어별 에이전트 키워드 반환"""
        return self.get_config(language).get('agent_keywords', {})


class TaskDependency:
    """작업 의존성 정보"""
    def __init__(self, task_id: str, depends_on: List[str], data_flow: Dict[str, str]):
        self.task_id = task_id
        self.depends_on = depends_on  # 의존하는 작업 ID들
        self.data_flow = data_flow    # 데이터 흐름 정보 {"from_task": "data_field"}


class QueryChain:
    """쿼리 체인 - 의존성 있는 작업들의 순서와 데이터 전달"""
    def __init__(self, chain_id: str, tasks: List[Dict[str, Any]], dependencies: List[TaskDependency]):
        self.chain_id = chain_id
        self.tasks = tasks
        self.dependencies = dependencies
        self.execution_order = self._calculate_execution_order()
    
    def _calculate_execution_order(self) -> List[List[str]]:
        """의존성을 고려한 실행 순서 계산 (위상 정렬)"""
        # 간단한 위상 정렬 구현
        in_degree = {task["task_id"]: 0 for task in self.tasks}
        
        # 각 작업의 in-degree 계산
        for dep in self.dependencies:
            in_degree[dep.task_id] = len(dep.depends_on)
        
        execution_order = []
        remaining_tasks = set(task["task_id"] for task in self.tasks)
        
        while remaining_tasks:
            # in-degree가 0인 작업들 찾기 (동시 실행 가능)
            ready_tasks = [task_id for task_id in remaining_tasks if in_degree[task_id] == 0]
            
            if not ready_tasks:
                # 순환 의존성이 있는 경우 남은 작업들을 순서대로 처리
                ready_tasks = [list(remaining_tasks)[0]]
            
            execution_order.append(ready_tasks)
            
            # 처리된 작업들 제거 및 의존성 업데이트
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                
                # 이 작업에 의존하는 다른 작업들의 in-degree 감소
                for dep in self.dependencies:
                    if task_id in dep.depends_on:
                        in_degree[dep.task_id] -= 1
        
        return execution_order


class AgentResultContext:
    """에이전트 실행 결과 컨텍스트 - 순차 처리를 위한 결과 전달"""
    def __init__(self):
        self.results = {}  # {task_id: result}
        self.execution_history = []  # [(task_id, timestamp, result)]
    
    def add_result(self, task_id: str, result: Any):
        """결과 추가"""
        self.results[task_id] = result
        self.execution_history.append((task_id, datetime.now(), result))
    
    def get_result(self, task_id: str) -> Any:
        """특정 작업 결과 가져오기"""
        return self.results.get(task_id)
    
    def get_previous_results(self, current_task_id: str, dependency_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """현재 작업이 의존하는 이전 결과들 가져오기"""
        dependent_tasks = dependency_map.get(current_task_id, [])
        previous_results = {}
        for dep_task in dependent_tasks:
            if dep_task in self.results:
                previous_results[dep_task] = self.results[dep_task]
        return previous_results
    
    def format_context_for_agent(self, task_id: str, dependency_map: Dict[str, List[str]]) -> str:
        """에이전트에게 전달할 컨텍스트 포맷팅"""
        previous_results = self.get_previous_results(task_id, dependency_map)
        if not previous_results:
            return ""
        
        context_parts = ["이전 작업 결과를 참고하여 처리하세요:"]
        for dep_task, result in previous_results.items():
            # 결과에서 핵심 정보 추출
            result_summary = self._extract_key_info_from_result(result)
            context_parts.append(f"- {dep_task}: {result_summary}")
        
        return "\n".join(context_parts)
    
    def _extract_key_info_from_result(self, result: Any) -> str:
        """결과에서 핵심 정보 추출"""
        if isinstance(result, dict):
            # answer, content, result 등의 키에서 정보 추출
            for key in ['answer', 'content', 'result', 'data', 'output']:
                if key in result:
                    value = result[key]
                    if isinstance(value, str):
                        return value[:200] + "..." if len(value) > 200 else value
                    else:
                        return str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            
            # 전체 결과를 문자열로 변환
            return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
        elif isinstance(result, str):
            return result[:200] + "..." if len(result) > 200 else result
        else:
            return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)


class UnifiedQueryProcessor:
    """🚀 통합 쿼리 프로세서 - 다국어 지원 및 하드코딩 제거"""
    
    def __init__(self):
        self.llm_manager = get_ontology_llm_manager()
        self.installed_agents_info = []
        self.result_context = AgentResultContext()
        self.language_config = LanguageConfig()
        
    def set_installed_agents_info(self, installed_agents_info: List[Dict[str, Any]]):
        """설치된 에이전트 정보 설정"""
        self.installed_agents_info = installed_agents_info
        logger.info(f"🎯 설치된 에이전트 정보 업데이트: {len(installed_agents_info)}개")
        
        # 에이전트 정보 요약 로깅
        for agent in installed_agents_info:
            agent_id = agent.get('agent_id', 'unknown')
            agent_data = agent.get('agent_data', {})
            agent_name = agent_data.get('name', agent_id)
            agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
            capabilities = agent_data.get('capabilities', [])
            
            logger.info(f"  📋 {agent_id}: {agent_name} ({agent_type}) - {len(capabilities)}개 능력")

    async def process_unified_query(self, 
                                  query: str, 
                                  available_agents: List[str]) -> Dict[str, Any]:
        """
        통합 쿼리 처리 - 다국어 지원 LLM 기반 완전 자동화 + 에이전트 간 결과 전달
        
        Args:
            query: 원본 사용자 쿼리
            available_agents: 사용 가능한 에이전트 목록
            
        Returns:
            Dict: 통합 분석 결과 (에이전트 간 결과 전달 포함)
        """
        try:
            # 언어 감지
            detected_language = self.language_config.detect_language(query)
            logger.info(f"🌐 감지된 언어: {self.language_config.get_config(detected_language)['name']}")
            
            logger.info(f"🚀 통합 쿼리 처리 시작 (v3.1 - 다국어 지원): {query}")
            
            # 결과 컨텍스트 초기화
            self.result_context = AgentResultContext()
            
            # 설치된 에이전트 정보 구성
            agents_info = self._build_agents_info_for_llm(available_agents)
            
            # LLM 기반 완전 통합 분석
            unified_prompt = self._create_llm_based_unified_prompt(query, agents_info, detected_language)
            
            logger.info("🧠 LLM 기반 완전 통합 분석 실행...")
            start_time = datetime.now()
            
            # LLM 호출
            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(unified_prompt["query"])
            
            # AIMessage에서 content 추출
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"⏱️ LLM 처리 완료: {processing_time:.2f}초")

            # 🔍 DEBUG: LLM 원본 응답 로깅
            logger.info(f"🔍 [DEBUG] LLM 원본 응답 (처음 800자): {response_text[:800] if len(response_text) > 800 else response_text}")

            # 결과 파싱 및 검증 (에이전트 간 결과 전달 포함)
            result = await self._parse_and_validate_llm_response(response_text, query, available_agents, detected_language)
            
            logger.info(f"✅ 다국어 지원 통합 처리 완료: {len(result.get('agent_mappings', []))}개 매핑, {len(result.get('query_chains', []))}개 체인 생성")
            return result
            
        except Exception as e:
            logger.error(f"통합 쿼리 처리 실패: {e}")
            # 폴백 처리 (다국어 지원)
            return await self._create_llm_based_fallback(query, available_agents, detected_language)

    def _create_llm_based_unified_prompt(self, query: str, agents_info: Dict[str, Any], language: str) -> Dict[str, str]:
        """다국어 지원 완전 LLM 기반 통합 프롬프트 생성 - 🚀 개선된 에이전트 정보 포맷팅"""

        # 🚀 개선: 에이전트 정보를 LLM이 이해하기 쉬운 상세 형태로 정리
        agents_list = []
        for agent_id, info in agents_info.items():
            agent_data = info.get('agent_data', info)  # agent_data가 있으면 사용
            agent_type = agent_data.get('agent_type', info.get('agent_type', 'UNKNOWN'))
            description = agent_data.get('description', info.get('description', '설명 없음'))
            name = agent_data.get('name', info.get('name', agent_id))
            capabilities = agent_data.get('capabilities', info.get('capabilities', []))
            tags = agent_data.get('tags', info.get('tags', []))

            # 🚀 개선: 모든 능력 추출 (최대 10개)
            all_capabilities = []
            for cap in capabilities[:10]:
                if isinstance(cap, dict):
                    cap_name = cap.get('name', cap.get('capability', ''))
                    cap_desc = cap.get('description', '')
                    if cap_name:
                        all_capabilities.append(f"{cap_name}" + (f"({cap_desc[:30]})" if cap_desc else ""))
                elif cap:
                    all_capabilities.append(str(cap))

            # 🚀 개선: 태그를 도메인 힌트로 활용 (최대 15개)
            all_tags = tags[:15] if tags else []

            # 🚀 개선: 에이전트 ID와 이름을 명확히 구분
            agent_entry = f"""
📌 에이전트 ID: {agent_id}
   이름: {name}
   타입: {agent_type}
   설명: {description[:300]}{'...' if len(description) > 300 else ''}
   능력: {', '.join(all_capabilities) if all_capabilities else '일반 처리'}
   도메인 태그: {', '.join(all_tags) if all_tags else '일반'}
   ✅ 이 에이전트를 선택하려면 agent_mappings의 selected_agent에 "{agent_id}"를 입력하세요.
"""
            agents_list.append(agent_entry)
        
        agents_formatted = '\n'.join(agents_list)
        
        # 언어별 예시 및 패턴
        if language == 'ko':
            examples = {
                'single_agent': '"제주도 여행 어디 추천해줘" → 여행/검색 에이전트가 제주도 관광지, 맛집, 명소 정보를 종합 검색',
                'sequential': '"금 시세 확인하고 1온스당 원화 가격 계산" → 1단계: 금시세 조회 → 2단계: 금시세 결과를 받아서 원화 계산',
                'parallel': '"제주도 맛집과 관광지 알려줘" → 맛집 검색 || 관광지 검색 (동시 실행)',
                'hybrid': '"날씨와 환율 확인하고 여행 경비 계산" → 1단계: (날씨 || 환율) → 2단계: 두 결과를 받아서 경비 계산',
                'visualization': '"제주도 여행 5일 일정을 플로우차트로" → 여행 계획 생성 에이전트 + 플로우차트 시각화 에이전트',
                'recommendation': '"서울 근교 당일치기 여행지 추천해줘" → 인터넷 검색 에이전트로 최신 추천 정보 검색'
            }
        else:
            examples = {
                'single_agent': '"Where should I travel in Jeju?" → Travel/search agent comprehensively searches for attractions, restaurants, and landmarks',
                'sequential': '"check gold price and calculate ounce to won" → Step1: get gold price → Step2: calculate won price using gold price result',
                'parallel': '"tell me restaurants and attractions in Seoul" → restaurant search || attraction search (simultaneous execution)',
                'hybrid': '"check weather and exchange rate then calculate travel cost" → Step1: (weather || exchange rate) → Step2: calculate cost using both results',
                'visualization': '"create flowchart for 5-day Jeju trip" → travel planning agent + flowchart visualization agent',
                'recommendation': '"recommend day trip destinations near Seoul" → internet search agent for latest recommendations'
            }
        
        # LLM 기반 쿼리 분석 프롬프트
        unified_prompt = f"""당신은 사용자 쿼리를 분석하여 최적의 에이전트를 선택하는 전문가입니다.

언어: {self.language_config.get_config(language)['name']}
사용자 쿼리: "{query}"

═══════════════════════════════════════════════════════════════════
🚨 **핵심 규칙: 에이전트 선택 시 반드시 아래 목록의 정확한 ID를 사용하세요!**
═══════════════════════════════════════════════════════════════════

사용 가능한 에이전트들:
{agents_formatted}

═══════════════════════════════════════════════════════════════════
⚠️ **에이전트 선택 필수 규칙 (LLM 기반 분석):**
1. agent_mappings의 selected_agent에는 **반드시 위 목록에 있는 정확한 에이전트 ID**를 입력
2. 에이전트 ID는 대소문자를 정확히 일치시켜야 함
3. **쿼리의 의도와 에이전트의 설명(description), 능력(capabilities), 태그(tags)를 의미론적으로 분석**하여 가장 적합한 에이전트 선택
4. **절대 특정 에이전트 이름을 하드코딩하지 마세요** - 오직 에이전트 메타데이터만으로 판단
5. **적합한 에이전트가 없으면**: no_suitable_agent 필드를 true로 설정하고 사용자에게 알림 메시지 생성

🔴 **에이전트 선택 원칙 (LLM 기반 의미론적 매칭):**

**핵심 원칙: 모든 에이전트는 동등하게 평가됩니다. 쿼리 의도와 에이전트 capabilities의 의미론적 매칭으로 선택하세요.**

1. **의미론적 매칭**: 각 에이전트의 description, capabilities, tags를 분석하여 쿼리 의도와 가장 잘 맞는 에이전트 선택
   - 에이전트마다 고유한 전문 영역이 있음
   - 쿼리가 요구하는 기능과 에이전트가 제공하는 기능을 매칭

2. **에이전트별 전문 영역** (참고용 - 실제로는 메타데이터 기반 판단):
   - scheduler: 개인 일정/캘린더 관리 (사용자 시스템 연결됨)
   - shopping: 상품 가격 검색/비교
   - llm_search: 개념 설명/정의 질문
   - internet: 최신 뉴스, 실시간 정보, 웹 검색이 필요한 경우
   - weather: 날씨/기온 정보
   - rag_search: 업로드된 문서에서 검색 (PDF, 보고서 등)
   - samsung_gateway: 삼성/반도체 관련 전문 분석

3. **적합한 에이전트가 없을 때 - 유용한 피드백 제공**:
   **no_suitable_agent = true로 설정하고, 다음 중 하나의 피드백 제공:**

   a) **질문 구체화 유도** (모호한 쿼리):
      - user_message: "어떤 종류의 도움이 필요하신지 더 구체적으로 알려주시면 정확한 도움을 드릴 수 있습니다."
      - clarification_needed: ["어떤 정보가 필요한가요?", "구체적인 요청을 알려주세요"]

   b) **대안 제시** (기능 부재):
      - user_message: "요청하신 기능은 현재 지원되지 않지만, 다음과 같은 도움은 드릴 수 있습니다."
      - suggested_alternatives: ["가능한 대안 에이전트/기능 목록"]

   c) **불가능 이유 설명** (현실적으로 불가능):
      - user_message: "죄송합니다, 이 요청은 [이유]로 처리가 어렵습니다."
      - impossible_reason: "구체적인 불가능 이유"

4. **RAG/문서 검색**: "보고서에서", "PDF에서", "문서에서" 등 **명시적 문서 참조가 있을 때만** rag_search 사용
═══════════════════════════════════════════════════════════════════

🎯 **1단계: 쿼리 타입 분류 (가장 먼저 수행 - 필수)**

쿼리를 분석하여 아래 4가지 타입 중 하나로 정확히 분류하세요:

**1. DIRECT_ANSWER (직접 답변 가능 - 에이전트 불필요)**
   하위 타입:
   - math_expression: 수학 표현식 계산
     예시: "1+1", "2*3+5", "sqrt(16)", "10/2", "2^3"
     특징: 숫자와 연산자(+,-,*,/,^,sqrt,sin,cos 등)만 포함

   - simple_fact: 간단한 사실 질문
     예시: "한국의 수도", "지구의 둘레", "광속은?"
     특징: 일반 상식, 즉시 답변 가능

   - time_query: 시간/날짜 질문
     예시: "지금 몇 시", "오늘 날짜", "현재 시간"
     특징: 시간 관련 실시간 정보

**2. SEARCH_REQUIRED (문서/웹 검색 필요)**
   하위 타입:
   - document_search: PDF/문서에서 정보 검색
     예시: "계약서에서 3조 찾기", "보고서의 결론 부분"
     특징: "PDF", "문서", "파일", "보고서" 키워드 포함

   - web_search: 인터넷 정보 검색
     예시: "2024년 최신 뉴스", "현재 주가", "날씨"
     특징: 최신 정보, 실시간 데이터 필요

   - travel_recommendation: 🚀 여행/관광지 추천
     예시: "제주도 여행 어디 추천해줘", "서울 근교 당일치기 여행지", "부산 가볼만한 곳"
     특징: "여행", "관광", "가볼", "추천", "어디" 키워드 포함

   - food_recommendation: 🍽️ 맛집/음식 추천
     예시: "제주도 맛집 추천", "서울 데이트 코스 맛집", "강남역 맛있는 집"
     특징: "맛집", "음식", "식당", "먹을곳" 키워드 포함

   - place_recommendation: 📍 장소 추천
     예시: "서울 핫플레이스", "데이트 코스 추천", "주말에 어디 가면 좋을까"
     특징: "추천", "어디", "좋을까", "핫플" 키워드 포함

**3. AGENT_TASK (에이전트 작업 필요)**
   하위 타입:
   - data_analysis: 데이터 분석 작업
   - code_generation: 코드 생성
   - visualization: 시각화 생성
   - complex_reasoning: 복잡한 추론/분석

**4. CONVERSATIONAL (대화형)**
   하위 타입:
   - greeting: 인사, 감사
     예시: "안녕", "고마워", "감사합니다"
   - help_request: 도움말 요청
     예시: "사용법 알려줘", "어떻게 써?"

⚠️ **분류 우선순위:**
1. 수학 표현식이면 무조건 DIRECT_ANSWER/math_expression
2. "PDF", "문서", "파일" 키워드가 있으면 SEARCH_REQUIRED/document_search
3. "여행", "추천", "맛집", "관광" 키워드가 있으면 SEARCH_REQUIRED/travel_recommendation 또는 food_recommendation
4. 분석/시각화/코드 생성 요청이면 AGENT_TASK
5. 위 모두 아니면 적절한 타입 선택

🧠 **LLM 기반 에이전트 선택 원칙:**
- 쿼리의 의도와 에이전트의 **description**, **capabilities**, **tags**를 의미론적으로 분석
- 특정 에이전트 이름을 기억하지 말고, **에이전트 메타데이터만으로** 판단
- 복합 쿼리(예: 여행+맛집)는 관련 에이전트 조합으로 병렬(parallel) 전략 사용 가능

🚨 **절대 규칙 - 반드시 준수하세요:**
1. "분석하고 제안/개선/제시" 패턴은 **무조건 1개 작업**으로 처리
2. 작업 분할은 **완전히 다른 도메인**일 때만 허용
3. **의미론적 매칭**: 쿼리 의도와 에이전트 메타데이터(description, capabilities, tags)를 분석하여 가장 적합한 에이전트 선택
4. **모든 에이전트는 동등**: 특정 에이전트를 "폴백"으로 취급하지 않음. 각 에이전트의 전문 영역에 맞게 선택

5. **적합한 에이전트가 없을 때 - 유용한 피드백 제공:**
   - no_suitable_agent: true 설정
   - feedback_type: "clarification_needed" | "alternative_suggested" | "impossible_request"
   - user_message: 사용자에게 전달할 메시지
   - 예시:
     * 모호한 쿼리 → "어떤 종류의 도움이 필요하신지 더 구체적으로 알려주세요"
     * 기능 부재 → "이 기능은 지원되지 않지만, 대신 X 기능은 가능합니다"
     * 불가능한 요청 → "죄송합니다, 이 요청은 [이유]로 처리가 어렵습니다"

✅ **올바른 에이전트 매칭 예시:**
- "이번주 일정 알려줘" → scheduler (개인 캘린더 접근 가능)
- "아이폰 가격 검색해줘" → shopping (상품 가격 전문)
- "양자역학이란 무엇인가요?" → llm_search (개념 설명 전문)
- "최신 AI 뉴스 알려줘" → internet (실시간 웹 검색 전문)
- "제주도 여행지 추천해줘" → internet (웹에서 정보 검색)
- "보고서에서 3분기 실적 찾아줘" → rag_search (문서 검색 전문)
- "삼성전자 반도체 NAND 분석" → samsung_gateway (삼성/반도체 전문)

❌ **적합한 에이전트 없음 처리 예시:**
- "달나라에서 피자 주문해줘" → no_suitable_agent: true
  - feedback_type: "impossible_request"
  - user_message: "죄송합니다, 달나라 배달은 현실적으로 불가능합니다. 대신 서울 지역 피자 배달 정보는 찾아드릴 수 있어요."
- "뭔가 추천해줘" → no_suitable_agent: true
  - feedback_type: "clarification_needed"
  - user_message: "어떤 종류의 추천을 원하시나요? (여행지, 맛집, 상품 등)"

❌ **잘못된 분할 예시 (절대 금지):**
- "수율 분석하고 개선방안 제시" → 2개 작업 ❌
- "현황 파악하고 전략 수립" → 2개 작업 ❌
- "트렌드 분석하고 예측" → 2개 작업 ❌

✅ **올바른 처리 예시:**
쿼리: "삼성반도체 DDR5 Etch 공정의 수율 추이를 분석하고 개선방안을 제시해주세요"
응답:
- task_breakdown: [{{"task_id": "task_1", "individual_query": "{query}"}}]
- strategy: "single_agent"
- agent_mappings: [{{"selected_agent": "samsung_gateway_agent_xxx"}}]

전략별 예시:
- 단일 작업: {examples.get('single_agent', '"분석하고 제안" 같은 통합 요청')}
- 순차 처리: {examples['sequential']}
- 병렬 처리: {examples['parallel']}

아래 JSON 형식으로 정확히 응답하세요:

{{
    "language_detected": "{language}",
    "query_analysis": {{
        "original_query": "{query}",
        "query_type": "DIRECT_ANSWER|SEARCH_REQUIRED|AGENT_TASK|CONVERSATIONAL",
        "query_subtype": "math_expression|simple_fact|time_query|document_search|web_search|travel_recommendation|food_recommendation|place_recommendation|data_analysis|code_generation|visualization|complex_reasoning|greeting|help_request",
        "classification_confidence": 0.0-1.0,
        "classification_reasoning": "쿼리 타입 분류 근거: 왜 이 타입으로 분류했는지 구체적으로 설명 (예: '1+1'은 숫자와 연산자만 포함하므로 DIRECT_ANSWER/math_expression)",
        "complexity": "simple|moderate|complex",
        "multi_task": true|false,
        "task_count": 숫자,
        "primary_intent": "정보검색|계산|분석|시각화|계획수립|여행추천|장소추천|맛집추천|기타",
        "domains": ["도메인1", "도메인2", ...],
        "output_format": "text|table|chart|flowchart|list|json",
        "dependency_detected": true|false,
        "sequential_required": true|false,
        "reasoning": "분석 근거: 왜 이런 복잡도로 평가했는지, 왜 multi_task로 판단했는지 등의 이유를 상세히 설명"
    }},
    "task_breakdown": [
        {{
            "task_id": "고유_작업_ID",
            "task_description": "구체적인 작업 설명",
            "individual_query": "이 작업을 위한 쿼리 (원본과 동일해도 됨)",
            "extracted_keywords": ["키워드1", "키워드2"],
            "domain": "작업_도메인",
            "complexity": "simple|moderate|complex",
            "depends_on": ["의존하는_작업_ID들"],
            "data_requirements": ["필요한_데이터_필드들"],
            "expected_output_type": "text|number|json|url|table|chart|flowchart",
            "output_fields": ["필드1", "필드2"]
        }}
    ],
    "dependency_analysis": {{
        "has_dependencies": true|false,
        "dependency_type": "sequential|parallel|hybrid",
        "dependency_chains": [
            {{
                "chain_id": "체인_ID",
                "tasks_in_order": ["task1", "task2", "task3"],
                "data_flow": [
                    {{
                        "from_task": "task1",
                        "to_task": "task2", 
                        "data_field": "전달할_데이터_필드",
                        "transformation": "데이터_변환_방법"
                    }}
                ]
            }}
        ],
        "parallel_groups": [["동시실행_가능한_작업들"]],
        "execution_order": [["단계1_작업들"], ["단계2_작업들"]],
        "reasoning": "의존성 분석 근거: 왜 sequential/parallel/hybrid로 결정했는지, 어떤 작업들이 왜 서로 의존적인지 구체적으로 설명"
    }},
    "agent_mappings": [
        {{
            "task_id": "작업_ID",
            "selected_agent": "실제_에이전트_ID",
            "agent_type": "에이전트_타입",
            "selection_reasoning": "선택 근거 (태그 매칭 포함)",
            "individual_query": "에이전트에게 전달할 개별 최적화 쿼리",
            "context_integration": "이전 결과 활용 방법",
            "input_dependencies": ["의존하는_작업_ID들"],
            "expected_output": "예상 출력",
            "confidence": 0.0-1.0
        }}
    ],
    "execution_plan": {{
        "strategy": "single_agent|parallel|sequential|hybrid",
        "estimated_time": 예상시간_초,
        "parallel_groups": [["단계별_병렬_그룹들"]],
        "execution_order": [["단계1"], ["단계2"], ["단계3"]],
        "data_passing_required": true|false,
        "result_integration_method": "통합_방법",
        "reasoning": "실행 전략 선택 이유: 왜 이 전략(single_agent/sequential/parallel/hybrid)을 선택했는지, 작업들 간의 관계와 의존성을 고려한 구체적 사유"
    }},
    "result_context": {{
        "requires_context_passing": true|false,
        "context_format": "결과_전달_형식",
        "integration_points": ["통합_지점들"],
        "final_output_format": "최종_출력_형식"
    }},
    "quality_assessment": {{
        "completeness": 0.0-1.0,
        "agent_match_quality": 0.0-1.0,
        "dependency_accuracy": 0.0-1.0,
        "execution_efficiency": 0.0-1.0,
        "context_flow_quality": 0.0-1.0,
        "overall_confidence": 0.0-1.0
    }},
    "agent_availability": {{
        "no_suitable_agent": true|false,
        "feedback_type": "clarification_needed|alternative_suggested|impossible_request|none",
        "availability_reasoning": "적합한 에이전트가 없는 이유 또는 선택된 에이전트가 적합한 이유",
        "user_message": "사용자에게 전달할 건설적인 메시지",
        "clarification_questions": ["구체화를 위한 질문들 (feedback_type이 clarification_needed일 때)"],
        "suggested_alternatives": ["대안 제안 (feedback_type이 alternative_suggested일 때)"],
        "required_capabilities": ["필요하지만 없는 능력들"],
        "impossible_reason": "불가능한 이유 (feedback_type이 impossible_request일 때)"
    }}
}}

‼️ **최종 지침:**

1. **[필수] 가장 먼저 query_type과 query_subtype을 정확히 분류하세요**
   - "1+1" 같은 수학 표현식은 무조건 DIRECT_ANSWER/math_expression
   - PDF/문서 키워드가 있으면 SEARCH_REQUIRED/document_search
   - classification_confidence와 classification_reasoning 필수 포함

2. **[필수] 에이전트 선택은 메타데이터 기반으로만 수행**
   - 에이전트의 description, capabilities, tags를 쿼리 의도와 의미론적으로 비교
   - 특정 에이전트 이름을 하드코딩하지 마세요 (예: "일정" → scheduler 같은 키워드 매칭 금지)
   - 에이전트 목록에서 쿼리 의도에 가장 부합하는 에이전트 선택

3. **[필수] 적합한 에이전트가 없으면 명확히 표시**
   - agent_availability.no_suitable_agent = true 설정
   - user_message에 사용자에게 알릴 메시지 작성
   - required_capabilities에 필요하지만 없는 능력 명시

4. **전략 선택 기준 (strategy)**
   - single_agent: 단일 에이전트로 처리 가능한 경우
   - parallel: 독립적인 여러 작업을 동시에 처리 (예: 맛집 검색 || 관광지 검색)
   - sequential: 이전 결과가 다음 작업에 필요한 경우 (예: 가격 조회 → 계산)
   - hybrid: parallel + sequential 조합

5. 감지된 언어({language})로 응답 생성
6. **작업 수를 최소화하세요** - 하나의 에이전트가 처리 가능하면 하나로 통합
7. reasoning 필드에 "왜 그렇게 결정했는지" 구체적 이유 명시"""

        return {"query": unified_prompt}

    async def _parse_and_validate_llm_response(self, 
                                             response: str, 
                                             original_query: str, 
                                             available_agents: List[str],
                                             language: str) -> Dict[str, Any]:
        """LLM 응답 파싱 및 검증 - 다국어 지원"""
        try:
            logger.info(f"🔍 LLM 응답 파싱 시작... (길이: {len(response)})")
            
            # JSON 추출 및 파싱
            result = self._safe_json_parse_from_response(response)
            
            if not result:
                logger.warning("JSON 파싱 실패, LLM 기반 폴백 분석으로 전환")
                return await self._create_llm_based_fallback(original_query, available_agents, language)
            
            # 결과 검증 및 보완
            result = self._validate_and_enhance_llm_result(result, original_query, available_agents, language)
            
            # 에이전트 간 결과 전달을 위한 쿼리 체인 생성
            result = self._create_context_aware_query_chains(result)
            
            # 실행 계획 검증 및 보완
            result = self._validate_execution_plan(result)
            
            logger.info(f"✅ LLM 응답 검증 완료: {len(result.get('agent_mappings', []))}개 유효한 매핑")
            return result
            
        except Exception as e:
            logger.error(f"LLM 응답 파싱 중 오류: {e}")
            return await self._create_llm_based_fallback(original_query, available_agents, language)

    def _safe_json_parse_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """응답에서 안전한 JSON 파싱"""
        try:
            # JSON 블록 찾기
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif response.strip().startswith('{'):
                json_str = response.strip()
            else:
                # JSON 부분만 추출 시도
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    return None
            
            # JSON 문자열 정리
            cleaned_json = self._clean_json_string(json_str)
            
            # JSON 파싱
            return json.loads(cleaned_json)
            
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            return None

    def _validate_and_enhance_llm_result(self, result: Dict[str, Any], original_query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """LLM 결과 검증 및 향상"""

        # 필수 키 확인
        required_keys = ['query_analysis', 'task_breakdown', 'agent_mappings', 'execution_plan', 'dependency_analysis']
        for key in required_keys:
            if key not in result:
                result[key] = self._create_default_section(key, original_query, available_agents, language)

        # query_analysis 검증 및 향상
        query_analysis = result.get('query_analysis', {})

        # query_type 필드 검증 및 기본값 설정
        if 'query_type' not in query_analysis:
            # 간단한 휴리스틱으로 기본 타입 추정
            query_lower = original_query.lower().strip()
            if any(op in query_lower for op in ['+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos']) and query_lower.replace(' ', '').replace('+', '').replace('-', '').replace('*', '').replace('/', '').replace('.', '').isdigit():
                query_analysis['query_type'] = 'DIRECT_ANSWER'
                query_analysis['query_subtype'] = 'math_expression'
            elif any(keyword in query_lower for keyword in ['pdf', '문서', '파일', '보고서', 'document']):
                query_analysis['query_type'] = 'SEARCH_REQUIRED'
                query_analysis['query_subtype'] = 'document_search'
            elif any(keyword in query_lower for keyword in ['분석', '시각화', '코드', '생성', 'analysis', 'visualization', 'code']):
                query_analysis['query_type'] = 'AGENT_TASK'
                query_analysis['query_subtype'] = 'data_analysis'
            elif any(keyword in query_lower for keyword in ['안녕', '고마워', '감사', 'hello', 'hi', 'thanks']):
                query_analysis['query_type'] = 'CONVERSATIONAL'
                query_analysis['query_subtype'] = 'greeting'
            else:
                query_analysis['query_type'] = 'AGENT_TASK'
                query_analysis['query_subtype'] = 'complex_reasoning'

        # classification_confidence 기본값 설정
        if 'classification_confidence' not in query_analysis:
            query_analysis['classification_confidence'] = 0.7

        # classification_reasoning 기본값 설정
        if 'classification_reasoning' not in query_analysis:
            query_analysis['classification_reasoning'] = f"자동 분류: {query_analysis.get('query_type', 'AGENT_TASK')}/{query_analysis.get('query_subtype', 'unknown')}"

        result['query_analysis'] = query_analysis
        logger.info(f"🎯 쿼리 타입 분류: {query_analysis.get('query_type')}/{query_analysis.get('query_subtype')} (신뢰도: {query_analysis.get('classification_confidence')})")

        # 🚀 개선된 에이전트 매핑 검증 - LLM 의도 최대한 존중
        # 🔍 DEBUG: LLM이 선택한 에이전트 매핑 원본
        raw_mappings = result.get('agent_mappings', [])
        logger.info(f"🔍 [DEBUG] LLM agent_mappings 원본: {raw_mappings}")
        logger.info(f"🔍 [DEBUG] available_agents: {available_agents[:5]}... (총 {len(available_agents)}개)")

        valid_mappings = []
        for mapping in raw_mappings:
            agent_id = mapping.get('selected_agent')
            selection_reasoning = mapping.get('selection_reasoning', '')
            individual_query = mapping.get('individual_query', original_query)

            if agent_id in available_agents:
                # 정확히 매칭됨
                logger.info(f"✅ LLM 에이전트 선택 유효: {agent_id}")
                valid_mappings.append(mapping)
            else:
                # 🚀 개선: LLM의 의도(selection_reasoning)와 원본 쿼리를 함께 전달
                similar_agent = self._find_similar_agent(
                    agent_id,
                    available_agents,
                    selection_reasoning=selection_reasoning,
                    query=individual_query
                )
                if similar_agent:
                    original_agent = agent_id
                    mapping['selected_agent'] = similar_agent
                    mapping['selection_reasoning'] = f"{selection_reasoning} (🔄 LLM 의도 기반 대체: {original_agent} → {similar_agent})"
                    mapping['llm_original_selection'] = original_agent  # 원래 LLM 선택 기록
                    valid_mappings.append(mapping)
                    logger.info(f"🔄 LLM 의도 기반 에이전트 대체: {original_agent} → {similar_agent}")
                else:
                    logger.warning(f"⚠️ 에이전트 매칭 실패, 건너뜀: {agent_id}")

        # 🚀 개선: 유효한 매핑이 없을 때만 폴백 (최후의 수단)
        if not valid_mappings:
            logger.warning(f"⚠️ 모든 에이전트 매핑 실패, 쿼리 기반 폴백 실행")
            # LLM이 선택하려던 첫 번째 에이전트의 의도 활용
            first_mapping = result.get('agent_mappings', [{}])[0] if result.get('agent_mappings') else {}
            first_reasoning = first_mapping.get('selection_reasoning', '')

            # 폴백 에이전트 선택 (쿼리 분석 기반)
            fallback_agent = self._select_agent_by_content(original_query, available_agents, language)
            valid_mappings = [{
                "task_id": "fallback_task",
                "selected_agent": fallback_agent,
                "agent_type": "GENERAL",
                "selection_reasoning": f"폴백 선택: {fallback_agent} (원래 LLM 추론: {first_reasoning[:100]})" if first_reasoning else f"폴백 선택: {fallback_agent}",
                "individual_query": original_query,
                "context_integration": "없음",
                "confidence": 0.6,
                "is_fallback": True
            }]

        result['agent_mappings'] = valid_mappings

        # 🧠 agent_availability 검증 및 처리
        agent_availability = result.get('agent_availability', {})
        no_suitable_agent = agent_availability.get('no_suitable_agent', False)

        if no_suitable_agent:
            # LLM이 적합한 에이전트가 없다고 판단한 경우 - 유용한 피드백 제공
            feedback_type = agent_availability.get('feedback_type', 'none')
            user_message = agent_availability.get('user_message', '요청하신 작업에 적합한 에이전트를 찾지 못했습니다.')
            required_capabilities = agent_availability.get('required_capabilities', [])

            logger.warning(f"⚠️ 적합한 에이전트 없음 (피드백 타입: {feedback_type})")
            logger.warning(f"   메시지: {user_message}")

            # 피드백 타입별 추가 정보
            feedback_details = {}
            if feedback_type == 'clarification_needed':
                feedback_details['clarification_questions'] = agent_availability.get('clarification_questions', [])
                logger.info(f"   구체화 질문: {feedback_details['clarification_questions']}")
            elif feedback_type == 'alternative_suggested':
                feedback_details['suggested_alternatives'] = agent_availability.get('suggested_alternatives', [])
                logger.info(f"   대안 제안: {feedback_details['suggested_alternatives']}")
            elif feedback_type == 'impossible_request':
                feedback_details['impossible_reason'] = agent_availability.get('impossible_reason', '')
                logger.info(f"   불가능 이유: {feedback_details['impossible_reason']}")

            # 결과에 사용자 알림 정보 추가
            result['no_suitable_agent_info'] = {
                'status': True,
                'feedback_type': feedback_type,
                'user_message': user_message,
                'required_capabilities': required_capabilities,
                **feedback_details
            }
        else:
            # 적합한 에이전트가 있는 경우 - agent_availability 기본값 설정
            if 'agent_availability' not in result:
                result['agent_availability'] = {
                    'no_suitable_agent': False,
                    'feedback_type': 'none',
                    'availability_reasoning': '쿼리에 적합한 에이전트를 찾았습니다.',
                    'user_message': '',
                    'suggested_alternatives': [],
                    'required_capabilities': []
                }

        # 의존성 분석 검증
        dependency_analysis = result.get('dependency_analysis', {})
        if not dependency_analysis.get('execution_order'):
            task_ids = [task.get('task_id') for task in result.get('task_breakdown', [])]
            dependency_analysis['execution_order'] = [task_ids] if task_ids else []
        
        if not dependency_analysis.get('parallel_groups'):
            strategy = result.get('execution_plan', {}).get('strategy', 'parallel')
            task_ids = [task.get('task_id') for task in result.get('task_breakdown', [])]
            
            if strategy == 'sequential':
                dependency_analysis['parallel_groups'] = [[task_id] for task_id in task_ids]
            elif strategy == 'parallel':
                dependency_analysis['parallel_groups'] = [task_ids] if task_ids else []
            else:  # hybrid
                dependency_analysis['parallel_groups'] = [task_ids] if task_ids else []
        
        result['dependency_analysis'] = dependency_analysis
        
        return result

    def _create_context_aware_query_chains(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 인식 쿼리 체인 생성 - 에이전트 간 결과 전달 포함"""
        try:
            dependency_analysis = result.get('dependency_analysis', {})
            tasks = result.get('task_breakdown', [])
            agent_mappings = result.get('agent_mappings', [])
            
            query_chains = []
            
            # 의존성 체인이 있는 경우
            if dependency_analysis.get('has_dependencies', False):
                dependency_chains = dependency_analysis.get('dependency_chains', [])
                
                for chain_info in dependency_chains:
                    chain_id = chain_info.get('chain_id', f'chain_{len(query_chains)}')
                    tasks_in_order = chain_info.get('tasks_in_order', [])
                    data_flow = chain_info.get('data_flow', [])
                    
                    # TaskDependency 객체들 생성
                    dependencies = []
                    dependency_map = {}
                    
                    for flow in data_flow:
                        from_task = flow.get('from_task')
                        to_task = flow.get('to_task')
                        
                        if to_task not in dependency_map:
                            dependency_map[to_task] = []
                        dependency_map[to_task].append(from_task)
                    
                    for task_id in tasks_in_order:
                        depends_on = dependency_map.get(task_id, [])
                        if depends_on:
                            data_flow_info = {}
                            for dep in depends_on:
                                data_flow_info[dep] = 'result'  # 기본값
                            
                            dependency = TaskDependency(task_id, depends_on, data_flow_info)
                            dependencies.append(dependency)
                    
                    # 체인에 포함된 작업들 필터링
                    chain_tasks = [task for task in tasks if task.get('task_id') in tasks_in_order]
                    
                    # 에이전트 매핑에 컨텍스트 전달 정보 추가
                    for mapping in agent_mappings:
                        task_id = mapping.get('task_id')
                        if task_id in dependency_map:
                            # 의존성이 있는 작업의 경우 컨텍스트 전달 방법 추가
                            mapping['requires_context'] = True
                            mapping['context_dependencies'] = dependency_map[task_id]
                            mapping['context_integration'] = f"이전 작업 결과를 참고: {', '.join(dependency_map[task_id])}"
                        else:
                            mapping['requires_context'] = False
                    
                    query_chain = QueryChain(chain_id, chain_tasks, dependencies)
                    query_chains.append({
                        'chain_id': query_chain.chain_id,
                        'tasks': query_chain.tasks,
                        'execution_order': query_chain.execution_order,
                        'dependencies': [
                            {
                                'task_id': dep.task_id,
                                'depends_on': dep.depends_on,
                                'data_flow': dep.data_flow
                            } for dep in query_chain.dependencies
                        ],
                        'dependency_map': dependency_map
                    })
            
            result['query_chains'] = query_chains
            result['context_passing_enabled'] = len(query_chains) > 0
            
            logger.info(f"✅ 컨텍스트 인식 쿼리 체인 생성 완료: {len(query_chains)}개")
            return result
            
        except Exception as e:
            logger.error(f"컨텍스트 인식 쿼리 체인 생성 실패: {e}")
            result['query_chains'] = []
            result['context_passing_enabled'] = False
            return result

    def _validate_execution_plan(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """실행 계획 검증 및 보완"""
        try:
            execution_plan = result.get('execution_plan', {})
            dependency_analysis = result.get('dependency_analysis', {})
            
            # 전략 검증
            strategy = execution_plan.get('strategy', 'parallel')
            if strategy not in ['single_agent', 'parallel', 'sequential', 'hybrid']:
                strategy = 'parallel'
            
            # 병렬 그룹 검증
            parallel_groups = dependency_analysis.get('parallel_groups', [])
            if not parallel_groups:
                task_ids = [task.get('task_id') for task in result.get('task_breakdown', [])]
                if strategy == 'sequential':
                    parallel_groups = [[task_id] for task_id in task_ids]
                elif strategy == 'parallel':
                    parallel_groups = [task_ids] if task_ids else []
                else:  # hybrid
                    parallel_groups = [task_ids] if task_ids else []
            
            # 실행 순서 검증
            execution_order = dependency_analysis.get('execution_order', [])
            if not execution_order:
                task_ids = [task.get('task_id') for task in result.get('task_breakdown', [])]
                if strategy == 'sequential':
                    execution_order = task_ids
                else:
                    execution_order = task_ids
            
            # 실행 계획 업데이트
            execution_plan.update({
                'strategy': strategy,
                'parallel_groups': parallel_groups,
                'execution_order': execution_order,
                'estimated_time': execution_plan.get('estimated_time', len(parallel_groups) * 15.0),
                'data_passing_required': dependency_analysis.get('has_dependencies', False)
            })
            
            result['execution_plan'] = execution_plan
            dependency_analysis['parallel_groups'] = parallel_groups
            dependency_analysis['execution_order'] = execution_order
            result['dependency_analysis'] = dependency_analysis
            
            logger.info(f"✅ 실행 계획 검증 완료: {strategy} 전략, {len(parallel_groups)}개 그룹")
            return result
            
        except Exception as e:
            logger.error(f"실행 계획 검증 실패: {e}")
            return result

    async def _create_llm_based_fallback(self, query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """LLM 기반 폴백 결과 생성"""
        logger.warning("🔄 LLM 기반 폴백 모드로 통합 결과 생성")
        
        try:
            # 간단한 LLM 프롬프트로 기본 분석
            simple_prompt = f"""사용자 쿼리를 분석하여 기본적인 작업 분해를 수행하세요.

쿼리: "{query}"
사용 가능한 에이전트: {', '.join(available_agents)}

다음 형식으로 간단히 응답하세요:
{{
    "tasks": [
        {{
            "task_id": "task_1",
            "description": "작업 설명",
            "agent": "선택된_에이전트",
            "query": "에이전트용 쿼리"
        }}
    ],
    "strategy": "parallel|sequential",
    "has_dependencies": true|false
}}"""

            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(simple_prompt)
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # 간단한 파싱
            fallback_result = self._safe_json_parse_from_response(response_text)
            
            if fallback_result:
                return self._convert_fallback_to_full_result(fallback_result, query, available_agents, language)
            
        except Exception as e:
            logger.error(f"LLM 기반 폴백도 실패: {e}")
        
        # 최종 폴백 - 기본 구조
        return self._create_emergency_fallback(query, available_agents, language)

    def _create_emergency_fallback(self, query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """최종 폴백 - 기본 구조 생성"""
        logger.warning("🚨 최종 폴백 모드: 기본 구조로 결과 생성")
        
        # 가장 기본적인 작업 분해
        if not available_agents:
            available_agents = ['default_agent']
        
        # 삼성 쿼리 확인 후 적절한 에이전트 선택
        if self._is_samsung_domain_query(query):
            samsung_agents = [agent for agent in available_agents if "samsung_gateway" in agent.lower()]
            primary_agent = samsung_agents[0] if samsung_agents else available_agents[0]
        else:
            primary_agent = available_agents[0]
        
        # 언어 감지
        if language is None:
            language = self.language_config.detect_language(query)
        
        # 단순 키워드 기반 분석
        keywords = self._extract_simple_keywords(query, language)
        
        # 정규식 기반 순차 처리 패턴 감지
        sequential_detected = self._detect_sequential_pattern_regex(query, language)
        
        if sequential_detected:
            # 순차 처리가 필요한 경우 - 작업을 분해
            task_breakdown, agent_mappings = self._create_sequential_tasks_emergency(query, available_agents, keywords, language)
            
            # 순차 실행 계획
            task_ids = [task["task_id"] for task in task_breakdown]
            parallel_groups = [[task_id] for task_id in task_ids]  # 각 작업을 개별 그룹으로
            execution_order = task_ids
            strategy = "sequential"
        else:
            # 기본 단일 작업
            task_breakdown = [{
                "task_id": "emergency_task_1",
                "task_description": f"'{query}' 처리",
                "individual_query": query,
                "extracted_keywords": keywords,
                "domain": "일반",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            }]
            
            # 기본 에이전트 매핑
            agent_mappings = [{
                "task_id": "emergency_task_1",
                "selected_agent": primary_agent,
                "agent_type": "GENERAL",
                "selection_reasoning": "최종 폴백 - 기본 에이전트 선택",
                "individual_query": query,
                "context_integration": "없음",
                "confidence": 0.5
            }]
            
            # 기본 실행 계획
            parallel_groups = [["emergency_task_1"]]
            execution_order = ["emergency_task_1"]
            strategy = "single_agent"
        
        return {
            "query_analysis": {
                "original_query": query,
                "complexity": "simple",
                "multi_task": False,
                "task_count": 1,
                "primary_intent": "정보검색",
                "domains": ["일반"],
                "dependency_detected": False,
                "reasoning": "최종 폴백 - 기본 분석"
            },
            "task_breakdown": task_breakdown,
            "agent_mappings": agent_mappings,
            "dependency_analysis": {
                "has_dependencies": False,
                "dependency_type": "single_agent",
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "reasoning": "최종 폴백 - 단일 작업"
            },
            "execution_plan": {
                "strategy": strategy,
                "estimated_time": len(task_breakdown) * 15.0,
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "data_passing_required": sequential_detected,
                "reasoning": f"최종 폴백: {strategy} 전략을 선택한 이유는 작업들 간의 의존성 분석 결과 기반"
            },
            "quality_assessment": {
                "completeness": 0.5,
                "agent_match_quality": 0.4,
                "execution_efficiency": 0.6,
                "overall_confidence": 0.5
            },
            "fallback_mode": "emergency"
        }

    def _extract_simple_keywords(self, query: str, language: str = None) -> List[str]:
        """다국어 지원 간단한 키워드 추출"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        # 언어별 불용어 가져오기
        stopwords = self.language_config.get_stopwords(language)
        
        # 간단한 토큰화 (공백 기준)
        tokens = query.split()
        
        # 불용어 제거 및 길이 필터링
        keywords = [token for token in tokens if token not in stopwords and len(token) > 1]
        
        return keywords[:5]  # 최대 5개만

    def _detect_sequential_pattern_regex(self, query: str, language: str = None) -> bool:
        """다국어 지원 정규식 기반 순차 처리 패턴 감지"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        # 언어별 순차 패턴 가져오기
        sequential_patterns = self.language_config.get_sequential_patterns(language)
        
        query_lower = query.lower()
        for pattern in sequential_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def _create_sequential_tasks_emergency(self, query: str, available_agents: List[str], keywords: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """다국어 지원 긴급 폴백용 순차 작업 분해"""
        import re
        
        # 언어별 특별 패턴 처리
        if language == 'ko':
            # 금 시세 + 계산 패턴 특별 처리
            if re.search(r'금.*?시세.*?(?:하고|한\s*다음|후에|다음에|이후).*?(?:계산|변환|원화)', query.lower()):
                return self._create_gold_price_sequential_tasks(query, available_agents, language)
            
            # AI 트렌드 + 분석 + 시각화 패턴 특별 처리
            if re.search(r'(?:AI|인공지능|기술).*?(?:트렌드|동향).*?(?:조사|분석).*?(?:해서|하고).*?(?:시각|보고서|리포트)', query.lower()):
                return self._create_ai_trend_analysis_tasks(query, available_agents, language)
            
            # 일반적인 조사 + 분석 + 생성 패턴
            if re.search(r'(.+?)(?:조사|수집).*?(?:해서|하고).*?(.+?)(?:분석|처리).*?(?:하고).*?(.+?)(?:만들어|생성|작성)', query.lower()):
                return self._create_research_analysis_generation_tasks(query, available_agents, language)
        else:  # English
            # Gold price + calculation pattern
            if re.search(r'gold.*?price.*?(?:and|then).*?(?:calculate|convert|won)', query.lower()):
                return self._create_gold_price_sequential_tasks(query, available_agents, language)
            
            # AI trend + analysis + visualization pattern
            if re.search(r'(?:AI|artificial intelligence|technology).*?(?:trend|trends).*?(?:research|analyze).*?(?:and|then).*?(?:visual|report|chart)', query.lower()):
                return self._create_ai_trend_analysis_tasks(query, available_agents, language)
            
            # General research + analysis + generation pattern
            if re.search(r'(.+?)(?:research|collect).*?(?:and|then).*?(.+?)(?:analyze|process).*?(?:and|then).*?(.+?)(?:create|generate|make)', query.lower()):
                return self._create_research_analysis_generation_tasks(query, available_agents, language)
        
        # 일반적인 순차 패턴 분해
        connectors = self.language_config.get_connectors(language)
        for connector in connectors:
            if connector in query:
                parts = query.split(connector)
                if len(parts) >= 2:
                    return self._create_general_sequential_tasks(parts, available_agents, language)
        
        # 폴백: 기본 단일 작업
        return self._create_default_single_task(query, available_agents, keywords, language)

    def _create_gold_price_sequential_tasks(self, query: str, available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """다국어 지원 금 시세 + 계산 순차 작업 생성"""
        # 1단계: 금 시세 조회
        # 2단계: 원화 계산
        # 3단계: 정리
        
        # 적절한 에이전트 선택
        crawler_agent = self._find_best_agent_for_task(['crawler_agent', 'internet_agent', 'llm_search_agent'], available_agents)
        calculator_agent = self._find_best_agent_for_task(['calculator_agent', 'currency_exchange_agent', 'math_agent'], available_agents)
        analysis_agent = self._find_best_agent_for_task(['analysis_agent', 'content_formatter_agent', 'llm_search_agent'], available_agents)
        
        task_breakdown = [
            {
                "task_id": "gold_price_task",
                "task_description": "오늘의 금 시세를 확인합니다.",
                "individual_query": "현재 금의 시세 정보를 알려주세요. 가격과 변동률을 포함해주세요.",
                "extracted_keywords": ["금", "시세", "가격"],
                "domain": "금융",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            },
            {
                "task_id": "calculation_task",
                "task_description": "확인된 금 시세를 1온스당 원화 가격으로 계산합니다.",
                "individual_query": "1온스당 원화 가격을 정확히 계산해주세요. 계산 과정과 결과를 포함해주세요.",
                "extracted_keywords": ["1온스", "원화", "계산"],
                "domain": "계산",
                "complexity": "moderate",
                "depends_on": ["gold_price_task"],
                "expected_output_type": "number"
            },
            {
                "task_id": "summary_task",
                "task_description": "금 시세와 계산 결과를 정리합니다.",
                "individual_query": "금 시세 정보와 계산 결과를 사용자 친화적으로 정리해주세요.",
                "extracted_keywords": ["정리", "요약"],
                "domain": "분석",
                "complexity": "simple",
                "depends_on": ["calculation_task"],
                "expected_output_type": "text"
            }
        ]
        
        agent_mappings = [
            {
                "task_id": "gold_price_task",
                "selected_agent": crawler_agent,
                "agent_type": "CRAWLER",
                "selection_reasoning": "금 시세 조회를 위한 크롤링 에이전트 선택",
                "individual_query": "현재 금의 시세 정보를 알려주세요. 가격과 변동률을 포함해주세요.",
                "context_integration": "없음",
                "confidence": 0.85
            },
            {
                "task_id": "calculation_task",
                "selected_agent": calculator_agent,
                "agent_type": "CALCULATOR",
                "selection_reasoning": "원화 계산을 위한 계산기 에이전트 선택",
                "individual_query": "1온스당 원화 가격을 정확히 계산해주세요. 계산 과정과 결과를 포함해주세요.",
                "context_integration": "이전 금 시세 결과 활용",
                "confidence": 0.90
            },
            {
                "task_id": "summary_task",
                "selected_agent": analysis_agent,
                "agent_type": "ANALYSIS",
                "selection_reasoning": "결과 정리를 위한 분석 에이전트 선택",
                "individual_query": "금 시세 정보와 계산 결과를 사용자 친화적으로 정리해주세요.",
                "context_integration": "이전 모든 결과 활용",
                "confidence": 0.80
            }
        ]
        
        return task_breakdown, agent_mappings

    def _create_ai_trend_analysis_tasks(self, query: str, available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """다국어 지원 AI 트렌드 분석 + 시각화 순차 작업 생성"""
        # 1단계: AI 기술 트렌드 조사
        # 2단계: 수집된 데이터 분석  
        # 3단계: 시각적 보고서 생성
        
        # 적절한 에이전트 선택
        research_agent = self._find_best_agent_for_task(['internet_agent', 'crawler_agent', 'llm_search_agent'], available_agents)
        analysis_agent = self._find_best_agent_for_task(['analysis_agent', 'task_classifier_agent', 'llm_search_agent'], available_agents)
        visualization_agent = self._find_best_agent_for_task(['data_visualization_agent', 'content_formatter_agent', 'document_agent'], available_agents)
        
        task_breakdown = [
            {
                "task_id": "ai_trend_research_task",
                "task_description": "AI 기술 트렌드를 조사합니다.",
                "individual_query": "최신 AI 기술 트렌드와 동향을 조사해주세요. 주요 기술, 시장 동향, 핵심 플레이어들에 대한 정보를 수집해주세요.",
                "extracted_keywords": ["AI", "기술", "트렌드", "동향", "조사"],
                "domain": "기술조사",
                "complexity": "moderate",
                "depends_on": [],
                "expected_output_type": "text"
            },
            {
                "task_id": "ai_trend_analysis_task", 
                "task_description": "수집된 AI 기술 트렌드 데이터를 분석합니다.",
                "individual_query": "AI 기술 트렌드 데이터를 분석하여 패턴, 성장률, 주요 인사이트를 도출해주세요. 데이터 간의 상관관계와 미래 전망을 포함해주세요.",
                "extracted_keywords": ["분석", "패턴", "인사이트", "전망"],
                "domain": "데이터분석",
                "complexity": "high",
                "depends_on": ["ai_trend_research_task"],
                "expected_output_type": "structured_data"
            },
            {
                "task_id": "visual_report_task",
                "task_description": "분석 결과를 바탕으로 시각적 보고서를 생성합니다.",
                "individual_query": "AI 기술 트렌드 분석 결과를 시각적 보고서로 만들어주세요. 차트, 그래프, 인포그래픽을 포함한 종합적인 보고서를 작성해주세요.",
                "extracted_keywords": ["시각화", "보고서", "차트", "그래프"],
                "domain": "시각화",
                "complexity": "high", 
                "depends_on": ["ai_trend_analysis_task"],
                "expected_output_type": "visual_report"
            }
        ]
        
        agent_mappings = [
            {
                "task_id": "ai_trend_research_task",
                "selected_agent": research_agent,
                "agent_type": "INTERNET_SEARCH",
                "selection_reasoning": "AI 기술 트렌드 조사를 위한 인터넷 검색 에이전트 선택",
                "individual_query": "최신 AI 기술 트렌드와 동향을 조사해주세요. 주요 기술, 시장 동향, 핵심 플레이어들에 대한 정보를 수집해주세요.",
                "context_integration": "없음",
                "confidence": 0.90
            },
            {
                "task_id": "ai_trend_analysis_task",
                "selected_agent": analysis_agent,
                "agent_type": "ANALYSIS",
                "selection_reasoning": "AI 트렌드 데이터 분석을 위한 분석 에이전트 선택",
                "individual_query": "AI 기술 트렌드 데이터를 분석하여 패턴, 성장률, 주요 인사이트를 도출해주세요. 데이터 간의 상관관계와 미래 전망을 포함해주세요.",
                "context_integration": "이전 조사 결과 활용",
                "confidence": 0.85
            },
            {
                "task_id": "visual_report_task",
                "selected_agent": visualization_agent,
                "agent_type": "DATA_VISUALIZATION",
                "selection_reasoning": "시각적 보고서 생성을 위한 데이터 시각화 에이전트 선택",
                "individual_query": "AI 기술 트렌드 분석 결과를 시각적 보고서로 만들어주세요. 차트, 그래프, 인포그래픽을 포함한 종합적인 보고서를 작성해주세요.",
                "context_integration": "이전 모든 결과 활용",
                "confidence": 0.95
            }
        ]
        
        return task_breakdown, agent_mappings

    def _create_research_analysis_generation_tasks(self, query: str, available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """다국어 지원 일반적인 조사 + 분석 + 생성 순차 작업 생성"""
        import re
        
        # 쿼리에서 주요 부분 추출
        match = re.search(r'(.+?)(?:조사|수집).*?(?:해서|하고).*?(.+?)(?:분석|처리).*?(?:하고).*?(.+?)(?:만들어|생성|작성)', query.lower())
        
        if match:
            research_topic = match.group(1).strip()
            analysis_focus = match.group(2).strip() if len(match.groups()) > 1 else "데이터"
            output_type = match.group(3).strip() if len(match.groups()) > 2 else "결과"
        else:
            # 폴백: 간단한 분할
            parts = query.split('하고')
            research_topic = parts[0].strip() if len(parts) > 0 else query
            analysis_focus = parts[1].strip() if len(parts) > 1 else "분석"
            output_type = parts[2].strip() if len(parts) > 2 else "보고서"
        
        # 적절한 에이전트 선택
        research_agent = self._find_best_agent_for_task(['internet_agent', 'crawler_agent', 'llm_search_agent'], available_agents)
        analysis_agent = self._find_best_agent_for_task(['analysis_agent', 'task_classifier_agent', 'math_agent'], available_agents)
        generation_agent = self._find_best_agent_for_task(['content_formatter_agent', 'document_agent', 'data_visualization_agent'], available_agents)
        
        task_breakdown = [
            {
                "task_id": "research_task",
                "task_description": f"{research_topic}에 대해 조사합니다.",
                "individual_query": f"{research_topic}에 대한 최신 정보와 데이터를 조사하여 수집해주세요. 신뢰할 수 있는 출처에서 종합적인 정보를 제공해주세요.",
                "extracted_keywords": research_topic.split()[:3],
                "domain": "조사",
                "complexity": "moderate",
                "depends_on": [],
                "expected_output_type": "text"
            },
            {
                "task_id": "analysis_task",
                "task_description": f"수집된 데이터를 {analysis_focus} 관점에서 분석합니다.",
                "individual_query": f"수집된 정보를 {analysis_focus} 관점에서 분석해주세요. 주요 패턴, 트렌드, 인사이트를 도출하고 의미 있는 결론을 제시해주세요.",
                "extracted_keywords": analysis_focus.split()[:3],
                "domain": "분석",
                "complexity": "high",
                "depends_on": ["research_task"],
                "expected_output_type": "structured_data"
            },
            {
                "task_id": "generation_task",
                "task_description": f"분석 결과를 바탕으로 {output_type}을 생성합니다.",
                "individual_query": f"분석 결과를 바탕으로 {output_type}을 생성해주세요. 사용자가 이해하기 쉽고 실용적인 형태로 정리해주세요.",
                "extracted_keywords": output_type.split()[:3],
                "domain": "생성",
                "complexity": "moderate",
                "depends_on": ["analysis_task"],
                "expected_output_type": "document"
            }
        ]
        
        agent_mappings = [
            {
                "task_id": "research_task",
                "selected_agent": research_agent,
                "agent_type": "INTERNET_SEARCH",
                "selection_reasoning": f"{research_topic} 조사를 위한 검색 에이전트 선택",
                "individual_query": f"{research_topic}에 대한 최신 정보와 데이터를 조사하여 수집해주세요. 신뢰할 수 있는 출처에서 종합적인 정보를 제공해주세요.",
                "context_integration": "없음",
                "confidence": 0.80
            },
            {
                "task_id": "analysis_task",
                "selected_agent": analysis_agent,
                "agent_type": "ANALYSIS",
                "selection_reasoning": f"{analysis_focus} 분석을 위한 분석 에이전트 선택",
                "individual_query": f"수집된 정보를 {analysis_focus} 관점에서 분석해주세요. 주요 패턴, 트렌드, 인사이트를 도출하고 의미 있는 결론을 제시해주세요.",
                "context_integration": "이전 조사 결과 활용",
                "confidence": 0.85
            },
            {
                "task_id": "generation_task",
                "selected_agent": generation_agent,
                "agent_type": "CONTENT_FORMATTING",
                "selection_reasoning": f"{output_type} 생성을 위한 콘텐츠 포맷터 에이전트 선택",
                "individual_query": f"분석 결과를 바탕으로 {output_type}을 생성해주세요. 사용자가 이해하기 쉽고 실용적인 형태로 정리해주세요.",
                "context_integration": "이전 모든 결과 활용",
                "confidence": 0.80
            }
        ]
        
        return task_breakdown, agent_mappings

    def _create_general_sequential_tasks(self, parts: List[str], available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """다국어 지원 일반적인 순차 작업 생성"""
        task_breakdown = []
        agent_mappings = []
        
        for i, part in enumerate(parts[:3]):  # 최대 3개 작업
            part = part.strip()
            if not part:
                continue
                
            task_id = f"sequential_task_{i+1}"
            
            # 작업 내용에 따른 에이전트 선택
            best_agent = self._select_agent_by_content(part, available_agents, language)
            
            task_breakdown.append({
                "task_id": task_id,
                "task_description": f"{part}를 처리합니다.",
                "individual_query": f"{part}에 대해 처리해주세요.",
                "extracted_keywords": part.split()[:3],
                "domain": "일반",
                "complexity": "simple",
                "depends_on": [f"sequential_task_{i}"] if i > 0 else [],
                "expected_output_type": "text"
            })
            
            agent_mappings.append({
                "task_id": task_id,
                "selected_agent": best_agent,
                "agent_type": "GENERAL",
                "selection_reasoning": f"내용 기반 에이전트 선택: {part}",
                "individual_query": f"{part}에 대해 처리해주세요.",
                "context_integration": "이전 결과 활용" if i > 0 else "없음",
                "confidence": 0.7
            })
        
        return task_breakdown, agent_mappings

    def _create_default_single_task(self, query: str, available_agents: List[str], keywords: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """다국어 지원 기본 단일 작업 생성 - 최적화된 메시지 포함"""
        
        # 더 나은 에이전트 선택
        best_agent = self._select_agent_by_content(query, available_agents, language)
        
        # 에이전트별 최적화된 메시지 생성
        optimized_message = self._generate_optimized_message_for_agent(query, best_agent, language)
        
        task_breakdown = [{
            "task_id": "emergency_task_1",
            "task_description": f"'{query}' 처리",
            "individual_query": optimized_message,
            "extracted_keywords": keywords,
            "domain": "일반",
            "complexity": "simple",
            "depends_on": [],
            "expected_output_type": "text"
        }]
        
        agent_mappings = [{
            "task_id": "emergency_task_1",
            "selected_agent": best_agent,
            "agent_type": self._infer_agent_type_from_query(query, language),
            "selection_reasoning": f"쿼리 내용 기반 에이전트 선택: {best_agent}",
            "individual_query": optimized_message,
            "context_integration": "없음",
            "confidence": 0.7
        }]
        
        return task_breakdown, agent_mappings

    def _generate_optimized_message_for_agent(self, query: str, agent_id: str, language: str = None) -> str:
        """다국어 지원 에이전트별 최적화된 메시지 생성"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        query_lower = query.lower()
        intent_keywords = self.language_config.get_intent_keywords(language)
        
        # 언어별 메시지 템플릿
        if language == 'ko':
            templates = {
                'search': f"{query} - 최신 정보를 검색하여 상세히 알려주세요.",
                'analysis': f"{query} - 데이터를 분석하여 인사이트와 패턴을 도출해주세요.",
                'visualization': f"{query} - 시각적 자료와 차트를 포함하여 보고서를 작성해주세요.",
                'content': f"{query} - 사용자 친화적인 형태로 정리하여 제공해주세요.",
                'document': f"{query} - 문서 형태로 종합적인 결과를 작성해주세요.",
                'crawler': f"{query} - 웹에서 관련 정보를 수집하여 정리해주세요.",
                'llm': f"{query} - 전문적인 지식을 바탕으로 상세한 답변을 제공해주세요.",
                'default': f"{query} - 이 요청을 전문적으로 처리해주세요."
            }
        else:  # English
            templates = {
                'search': f"{query} - Please search for the latest information and provide detailed results.",
                'analysis': f"{query} - Please analyze the data and derive insights and patterns.",
                'visualization': f"{query} - Please create a report with visual materials and charts.",
                'content': f"{query} - Please organize and present in a user-friendly format.",
                'document': f"{query} - Please create a comprehensive document.",
                'crawler': f"{query} - Please collect and organize relevant information from the web.",
                'llm': f"{query} - Please provide detailed answers based on professional knowledge.",
                'default': f"{query} - Please handle this request professionally."
            }
        
        # 에이전트 타입별 최적화
        if 'internet' in agent_id.lower() or 'search' in agent_id.lower():
            return templates['search']
        elif 'analysis' in agent_id.lower():
            return templates['analysis']
        elif 'visualization' in agent_id.lower() or 'data_visualization' in agent_id.lower():
            return templates['visualization']
        elif 'content' in agent_id.lower() or 'formatter' in agent_id.lower():
            return templates['content']
        elif 'document' in agent_id.lower():
            return templates['document']
        elif 'crawler' in agent_id.lower():
            return templates['crawler']
        elif 'llm' in agent_id.lower():
            return templates['llm']
        else:
            # 의도 기반 최적화
            if any(keyword in query_lower for keyword in intent_keywords.get('search', [])):
                return templates['search']
            elif any(keyword in query_lower for keyword in intent_keywords.get('analyze', [])):
                return templates['analysis']
            elif any(keyword in query_lower for keyword in intent_keywords.get('generate', [])):
                return templates['content']
            elif any(keyword in query_lower for keyword in intent_keywords.get('visualize', [])):
                return templates['visualization']
            else:
                return templates['default']

    def _infer_agent_type_from_query(self, query: str, language: str = None) -> str:
        """다국어 지원 쿼리 내용에서 에이전트 타입 추론"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        query_lower = query.lower()
        intent_keywords = self.language_config.get_intent_keywords(language)
        
        if any(keyword in query_lower for keyword in intent_keywords.get('search', [])):
            return "INTERNET_SEARCH"
        elif any(keyword in query_lower for keyword in intent_keywords.get('analyze', [])):
            return "ANALYSIS"
        elif any(keyword in query_lower for keyword in intent_keywords.get('visualize', [])):
            return "DATA_VISUALIZATION"
        elif any(keyword in query_lower for keyword in intent_keywords.get('generate', [])):
            return "CONTENT_FORMATTING"
        elif any(keyword in query_lower for keyword in intent_keywords.get('collect', [])):
            return "CRAWLER"
        elif any(keyword in query_lower for keyword in intent_keywords.get('calculate', [])):
            return "CALCULATOR"
        else:
            return "GENERAL"

    def _find_best_agent_for_task(self, preferred_agents: List[str], available_agents: List[str]) -> str:
        """작업에 가장 적합한 에이전트 찾기"""
        for preferred in preferred_agents:
            if preferred in available_agents:
                return preferred
        
        # 폴백: 첫 번째 사용 가능한 에이전트
        return available_agents[0] if available_agents else 'unknown'

    def _is_samsung_domain_query(self, query: str) -> bool:
        """삼성 관련 업무 쿼리 자동 감지 (오타 보정 포함)"""
        
        # 오타 보정
        normalized_query = self._normalize_samsung_typos(query)
        
        # 회사/브랜드 키워드
        company_keywords = [
            "삼성", "samsung", "삼성반도체", "samsung semiconductor",
            "삼성전자", "삼성디스플레이", "삼성SDI"
        ]
        
        # 제품/기술 키워드  
        product_keywords = [
            "ddr4", "ddr5", "gddr6", "lpddr5", "hbm3", 
            "메모리", "memory", "반도체", "semiconductor",
            "nand", "dram", "ssd", "플래시", "flash",
            "particle", "파티클", "yield", "수율", "defect", "불량",
            "fab", "팹", "wafer", "웨이퍼", "foundry", "파운드리",
            "cleanroom", "클린룸", "lithography", "리소그래피",
            "etching", "에칭", "deposition", "증착"
        ]
        
        # 업무 키워드
        business_keywords = [
            "수율", "yield", "불량", "defect", "품질", "quality",
            "공정", "process", "fab", "공급망", "supply chain",
            "시장점유율", "market share", "매출", "revenue",
            "생산", "production", "제조", "manufacturing"
        ]
        
        # 분석 키워드 (업무 깊이 판단)
        analysis_keywords = [
            "분석", "analysis", "추이", "trend", "개선방안", "improvement",
            "최적화", "optimization", "보고서", "report", "예측", "forecast",
            "대시보드", "dashboard", "평가", "assessment"
        ]
        
        # 정규화된 쿼리 사용
        query_lower = normalized_query.lower()
        
        # 삼성 + (제품 OR 업무) 패턴
        has_company = any(keyword in query_lower for keyword in company_keywords)
        has_product = any(keyword in query_lower for keyword in product_keywords)
        has_business = any(keyword in query_lower for keyword in business_keywords)
        has_analysis = any(keyword in query_lower for keyword in analysis_keywords)
        
        # 패턴 매칭 로직
        if has_company and (has_product or has_business):
            logger.info(f"🏢 삼성 도메인 감지됨: 회사 키워드 + 제품/업무")
            return True
        
        # 삼성 없어도 반도체 + 분석이면 삼성으로 간주 (도메인 특화)
        if has_product and has_business and has_analysis:
            logger.info(f"🏢 삼성 도메인 감지됨: 반도체 업무 분석 패턴")
            return True
        
        # 반도체 전문 용어만 있어도 삼성으로 라우팅 (기본 도메인)
        semiconductor_specific = [
            "particle", "파티클", "yield", "수율", "defect", "불량",
            "fab", "팹", "cleanroom", "클린룸"
        ]
        if any(term in query_lower for term in semiconductor_specific):
            logger.info(f"🏢 삼성 도메인 감지됨: 반도체 전문 용어")
            return True
            
        return False
    
    def _normalize_samsung_typos(self, query: str) -> str:
        """삼성 관련 오타 정규화"""
        typo_corrections = {
            '삼상': '삼성',
            '삼숭': '삼성',
            '삼송': '삼성',
            '삼선': '삼성',
            '삼셩': '삼성',
            '반도채': '반도체',
            '번도체': '반도체',
            '반도처': '반도체',
            '수율': '수율',  # 표준화
            '슈율': '수율',
            '수류': '수율',
            'partical': 'particle',
            'particel': 'particle',
            'yeild': 'yield',
            'yiled': 'yield',
            'defact': 'defect',
            'deffect': 'defect'
        }
        
        normalized = query
        for typo, correct in typo_corrections.items():
            normalized = normalized.replace(typo, correct)
        
        return normalized

    async def _select_agent_by_llm(self, content: str, available_agents: List[str], language: str = None) -> str:
        """
        🧠 LLM 기반 에이전트 선택 (하드코딩 제거)

        쿼리 내용과 에이전트 설명을 LLM이 분석하여 최적의 에이전트를 선택합니다.
        """
        if language is None:
            language = self.language_config.detect_language(content)

        # 에이전트 정보 구성
        agents_info = self._build_agents_info_for_llm(available_agents)

        if not agents_info:
            logger.warning("⚠️ 사용 가능한 에이전트 정보 없음, 기본 에이전트 반환")
            return available_agents[0] if available_agents else 'unknown'

        # 에이전트 목록 문자열 생성
        agents_list = []
        for agent_id, info in agents_info.items():
            agent_data = info.get('agent_data', info)
            name = agent_data.get('name', info.get('name', agent_id))
            description = agent_data.get('description', info.get('description', ''))[:200]
            capabilities = agent_data.get('capabilities', info.get('capabilities', []))
            tags = agent_data.get('tags', info.get('tags', []))

            cap_str = ', '.join([c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in capabilities[:5]])
            tag_str = ', '.join(tags[:5]) if tags else ''

            agents_list.append(f"- {agent_id}: {name} | {description} | 능력: {cap_str} | 태그: {tag_str}")

        agents_formatted = '\n'.join(agents_list)

        # LLM 프롬프트
        prompt = f"""당신은 사용자 쿼리를 분석하여 가장 적합한 에이전트를 선택하는 전문가입니다.

사용자 쿼리: "{content}"

사용 가능한 에이전트 목록:
{agents_formatted}

규칙:
1. 쿼리의 의도를 정확히 파악하세요
2. 에이전트의 이름, 설명, 능력, 태그를 분석하세요
3. 가장 적합한 에이전트 ID를 하나만 선택하세요
4. 반드시 위 목록에 있는 정확한 agent_id를 반환하세요

응답 형식 (JSON만):
{{"selected_agent": "agent_id_here", "reasoning": "선택 이유"}}
"""

        try:
            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(prompt)

            response_text = response.content if hasattr(response, 'content') else str(response)

            # JSON 파싱
            import re
            json_match = re.search(r'\{[^{}]*"selected_agent"\s*:\s*"([^"]+)"[^{}]*\}', response_text, re.DOTALL)

            if json_match:
                selected_agent = json_match.group(1)

                # 선택된 에이전트가 사용 가능한 목록에 있는지 확인
                if selected_agent in available_agents:
                    logger.info(f"🧠 LLM 에이전트 선택 완료: {selected_agent} (쿼리: {content[:50]}...)")
                    return selected_agent
                else:
                    # 부분 매칭 시도
                    for agent in available_agents:
                        if selected_agent.lower() in agent.lower() or agent.lower() in selected_agent.lower():
                            logger.info(f"🧠 LLM 에이전트 부분 매칭: {agent} (원본: {selected_agent})")
                            return agent

            logger.warning(f"⚠️ LLM 에이전트 선택 파싱 실패, 응답: {response_text[:200]}")

        except Exception as e:
            logger.error(f"❌ LLM 에이전트 선택 실패: {e}")

        # LLM 실패 시 첫 번째 에이전트 반환
        return available_agents[0] if available_agents else 'unknown'

    def _select_agent_by_content(self, content: str, available_agents: List[str], language: str = None) -> str:
        """
        🔄 동기 래퍼: LLM 기반 에이전트 선택을 동기 컨텍스트에서 호출

        비동기 _select_agent_by_llm()을 동기 방식으로 호출합니다.
        """
        if language is None:
            language = self.language_config.detect_language(content)

        # 오타 정규화 적용
        normalized_content = self._normalize_samsung_typos(content)

        # 🏢 삼성 도메인 우선 체크 (비즈니스 로직 - 삼성 도메인은 특수 처리 필요)
        if self._is_samsung_domain_query(normalized_content):
            samsung_agents = [agent for agent in available_agents if "samsung_gateway" in agent.lower()]
            if samsung_agents:
                logger.info(f"🚀 Samsung Gateway Agent 선택: {samsung_agents[0]}")
                return samsung_agents[0]

            samsung_sub_agents = [
                agent for agent in available_agents
                if any(keyword in agent.lower() for keyword in [
                    "samsung_yield", "samsung_market", "samsung_quality",
                    "samsung_supply", "samsung_business"
                ])
            ]
            if samsung_sub_agents:
                logger.info(f"🚀 Samsung Sub-agent 선택: {samsung_sub_agents[0]}")
                return samsung_sub_agents[0]

        # 🧠 LLM 기반 에이전트 선택 (비동기 호출)
        try:
            import asyncio

            # 이벤트 루프 확인 및 실행
            try:
                loop = asyncio.get_running_loop()
                # 이미 루프가 실행 중이면 새 태스크로 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._select_agent_by_llm(normalized_content, available_agents, language)
                    )
                    selected_agent = future.result(timeout=30)
                    return selected_agent
            except RuntimeError:
                # 루프가 없으면 직접 실행
                selected_agent = asyncio.run(
                    self._select_agent_by_llm(normalized_content, available_agents, language)
                )
                return selected_agent

        except Exception as e:
            logger.error(f"❌ LLM 기반 에이전트 선택 실패: {e}")
            # 폴백: 첫 번째 사용 가능한 에이전트
            if available_agents:
                logger.info(f"🔍 폴백 에이전트 선택: {available_agents[0]}")
                return available_agents[0]
            return 'unknown'

    def _convert_fallback_to_full_result(self, fallback_result: Dict[str, Any], query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """폴백 결과를 전체 구조로 변환"""
        tasks = fallback_result.get('tasks', [])
        strategy = fallback_result.get('strategy', 'parallel')
        has_dependencies = fallback_result.get('has_dependencies', False)
        
        # 작업 분해
        task_breakdown = []
        agent_mappings = []
        
        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i+1}')
            task_breakdown.append({
                "task_id": task_id,
                "task_description": task.get('description', f'작업 {i+1}'),
                "individual_query": task.get('query', query),
                "extracted_keywords": [],
                "domain": "일반",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            })
            
            agent_mappings.append({
                "task_id": task_id,
                "selected_agent": task.get('agent', available_agents[0] if available_agents else 'unknown'),
                "agent_type": "GENERAL",
                "selection_reasoning": "LLM 기반 폴백 선택",
                "individual_query": task.get('query', query),
                "context_integration": "없음",
                "confidence": 0.7
            })
        
        # 실행 계획
        task_ids = [task['task_id'] for task in task_breakdown]
        if strategy == 'sequential':
            parallel_groups = [[task_id] for task_id in task_ids]
            execution_order = task_ids
        else:
            parallel_groups = [task_ids] if task_ids else []
            execution_order = task_ids
        
        return {
            "query_analysis": {
                "original_query": query,
                "complexity": "moderate",
                "multi_task": len(tasks) > 1,
                "task_count": len(tasks),
                "primary_intent": "정보검색",
                "domains": ["일반"],
                "dependency_detected": has_dependencies,
                "reasoning": "LLM 기반 폴백 분석"
            },
            "task_breakdown": task_breakdown,
            "agent_mappings": agent_mappings,
            "dependency_analysis": {
                "has_dependencies": has_dependencies,
                "dependency_type": strategy,
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "reasoning": "LLM 기반 폴백 의존성 분석"
            },
            "execution_plan": {
                "strategy": strategy,
                "estimated_time": len(tasks) * 15.0,
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "data_passing_required": has_dependencies,
                "reasoning": f"LLM 기반 폴백: {strategy} 전략 선택 - 작업 복잡도와 에이전트 능력을 고려한 최적 전략"
            },
            "quality_assessment": {
                "completeness": 0.7,
                "agent_match_quality": 0.6,
                "execution_efficiency": 0.7,
                "overall_confidence": 0.7
            },
            "fallback_mode": "llm_based"
        }

    async def create_context_aware_query(self, task_id: str, original_query: str, 
                                        dependency_map: Dict[str, List[str]]) -> str:
        """컨텍스트 인식 쿼리 생성 - 이전 결과를 활용한 쿼리"""
        
        # 이전 결과 컨텍스트 생성
        context = self.result_context.format_context_for_agent(task_id, dependency_map)
        
        if not context:
            return original_query
        
        # LLM을 사용하여 컨텍스트가 포함된 쿼리 생성
        try:
            context_prompt = f"""이전 작업 결과를 활용하여 현재 작업에 최적화된 쿼리를 생성하세요.

원본 쿼리: "{original_query}"
현재 작업 ID: {task_id}

{context}

이전 결과를 참고하여 현재 작업에 특화된 구체적이고 실행 가능한 쿼리를 생성하세요.
단순히 "이전 결과를 참고하여"라는 문구를 추가하지 말고, 구체적인 지시사항을 포함하세요.

응답 형식: 최적화된 쿼리만 반환"""

            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(context_prompt)
            
            if hasattr(response, 'content'):
                optimized_query = response.content.strip()
                logger.info(f"🎯 컨텍스트 인식 쿼리 생성: {task_id} -> {optimized_query[:50]}...")
                return optimized_query
            
        except Exception as e:
            logger.error(f"컨텍스트 인식 쿼리 생성 실패: {e}")
        
        # 폴백: 컨텍스트를 직접 추가
        return f"{context}\n\n{original_query}"

    def add_agent_result(self, task_id: str, result: Any):
        """에이전트 실행 결과 추가"""
        self.result_context.add_result(task_id, result)
        logger.info(f"📊 에이전트 결과 저장: {task_id}")

    def get_execution_context(self) -> AgentResultContext:
        """실행 컨텍스트 반환"""
        return self.result_context

    def _build_agents_info_for_llm(self, available_agents: List[str]) -> Dict[str, Any]:
        """LLM용 에이전트 정보 구성"""
        agents_info = {}
        
        for agent_id in available_agents:
            agent_info = self._find_agent_info(agent_id)
            if agent_info:
                agents_info[agent_id] = agent_info
            else:
                # 기본 정보 생성
                agents_info[agent_id] = {
                    'agent_type': self._infer_type_from_id(agent_id),
                    'description': f'{agent_id} 에이전트',
                    'capabilities': [],
                    'tags': [agent_id.replace('_agent', '')]
                }
        
        return agents_info

    def _find_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """에이전트 정보 찾기"""
        for agent_info in self.installed_agents_info:
            if agent_info.get('agent_id') == agent_id:
                agent_data = agent_info.get('agent_data', {})
                metadata = agent_data.get('metadata', {})
                
                return {
                    'agent_type': metadata.get('agent_type', 'GENERAL'),
                    'description': agent_data.get('description', f'{agent_id} 에이전트'),
                    'capabilities': agent_data.get('capabilities', []),
                    'tags': metadata.get('tags', [])
                }
        return None

    def _infer_type_from_id(self, agent_id: str) -> str:
        """에이전트 ID에서 타입 추론"""
        type_mapping = {
            'weather': 'WEATHER',
            'currency': 'CURRENCY',
            'calculator': 'CALCULATOR',
            'crawler': 'CRAWLER',
            'internet': 'INTERNET_SEARCH',
            'analysis': 'ANALYSIS',
            'llm': 'LLM_SEARCH',
            'search': 'SEARCH',
            'math': 'MATH',
            'fortune': 'FORECASTING',
            'restaurant': 'RESTAURANT_FINDER',
            'scheduler': 'SCHEDULER',
            'game': 'GAME',
            'shopping': 'SHOPPING',
            'memo': 'MEMO',
            'document': 'DOCUMENT_PROCESSING'
        }
        
        agent_lower = agent_id.lower()
        for keyword, agent_type in type_mapping.items():
            if keyword in agent_lower:
                return agent_type
        
        return 'GENERAL'

    def _clean_json_string(self, json_str: str) -> str:
        """JSON 문자열 정리"""
        # 주석 제거
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # // 주석 제거
            if '//' in line:
                line = line[:line.index('//')]
            cleaned_lines.append(line)
        
        # 다시 합치기
        cleaned_json = '\n'.join(cleaned_lines)
        
        # 불완전한 JSON 수정 시도
        cleaned_json = cleaned_json.strip()
        if cleaned_json.endswith(','):
            cleaned_json = cleaned_json[:-1]
        
        return cleaned_json

    def _find_similar_agent(self, target_agent: str, available_agents: List[str],
                             selection_reasoning: str = None, query: str = None) -> Optional[str]:
        """
        🚀 개선된 유사 에이전트 찾기 - LLM의 의도를 최대한 존중

        Args:
            target_agent: LLM이 선택한 에이전트 ID (존재하지 않을 수 있음)
            available_agents: 실제 사용 가능한 에이전트 목록
            selection_reasoning: LLM의 선택 이유 (의도 파악에 활용)
            query: 원본 쿼리 (도메인 힌트 추출에 활용)

        Returns:
            가장 적합한 에이전트 ID 또는 None
        """
        if not target_agent:
            return None

        target_lower = target_agent.lower().strip()

        # === 1단계: 정확한 매칭 (대소문자 무시) ===
        for agent in available_agents:
            if agent.lower() == target_lower:
                logger.info(f"✅ 정확한 매칭 성공: {target_agent} → {agent}")
                return agent

        # === 2단계: 부분 매칭 (포함 관계) ===
        for agent in available_agents:
            if target_lower in agent.lower() or agent.lower() in target_lower:
                logger.info(f"✅ 부분 매칭 성공: {target_agent} → {agent}")
                return agent

        # === 3단계: 키워드 기반 매칭 (언더스코어 분리) ===
        keywords = [kw for kw in target_lower.replace('_', ' ').replace('-', ' ').split() if len(kw) > 2]
        for agent in available_agents:
            agent_lower = agent.lower()
            # 키워드 중 하나라도 에이전트 ID에 포함되면 매칭
            if any(keyword in agent_lower for keyword in keywords):
                logger.info(f"✅ 키워드 매칭 성공: {target_agent} → {agent} (키워드: {keywords})")
                return agent

        # === 4단계: 도메인 유사성 매칭 (LLM 의도 기반) ===
        # LLM이 선택한 에이전트 이름에서 도메인 힌트 추출
        domain_mappings = {
            # 여행/관광 도메인
            'travel': ['internet_agent', 'llm_search_agent', 'search_agent', 'tour_agent'],
            'tour': ['internet_agent', 'llm_search_agent', 'search_agent', 'travel_agent'],
            'trip': ['internet_agent', 'llm_search_agent', 'search_agent'],
            # 음식/맛집 도메인
            'food': ['restaurant_agent', 'matzip_agent', 'shopping_agent', 'internet_agent', 'llm_search_agent'],
            'restaurant': ['matzip_agent', 'food_agent', 'shopping_agent', 'internet_agent', 'llm_search_agent'],
            'matzip': ['restaurant_agent', 'food_agent', 'shopping_agent', 'internet_agent'],
            # 검색/정보 도메인
            'search': ['internet_agent', 'llm_search_agent', 'search_agent'],
            'info': ['internet_agent', 'llm_search_agent', 'search_agent'],
            'internet': ['llm_search_agent', 'search_agent', 'web_agent'],
            'web': ['internet_agent', 'llm_search_agent', 'search_agent'],
            # 추천 도메인
            'recommend': ['internet_agent', 'llm_search_agent', 'recommendation_agent'],
            'recommendation': ['internet_agent', 'llm_search_agent', 'recommend_agent'],
            # 분석 도메인
            'analysis': ['analysis_agent', 'data_agent', 'analytics_agent'],
            'data': ['analysis_agent', 'data_analysis_agent', 'analytics_agent'],
            # 날씨 도메인
            'weather': ['weather_agent', 'internet_agent', 'llm_search_agent'],
            # 쇼핑 도메인
            'shopping': ['shopping_agent', 'matzip_agent', 'internet_agent'],
            # 주식/금융 도메인
            'stock': ['stock_agent', 'finance_agent', 'internet_agent'],
            'finance': ['stock_agent', 'finance_agent', 'analysis_agent'],
        }

        # LLM이 선택한 에이전트에서 도메인 힌트 추출
        for domain_key, preferred_agents in domain_mappings.items():
            if domain_key in target_lower:
                # 선호 에이전트 중 사용 가능한 것 찾기
                for preferred in preferred_agents:
                    for agent in available_agents:
                        if preferred in agent.lower() or agent.lower() in preferred:
                            logger.info(f"✅ 도메인 유사성 매칭: {target_agent} → {agent} (도메인: {domain_key})")
                            return agent

        # === 5단계: selection_reasoning에서 힌트 추출 (LLM의 의도 분석) ===
        if selection_reasoning:
            reasoning_lower = selection_reasoning.lower()
            # reasoning에서 도메인 키워드 찾기
            reasoning_domains = {
                '여행': ['internet_agent', 'llm_search_agent', 'search_agent'],
                '관광': ['internet_agent', 'llm_search_agent', 'search_agent'],
                '맛집': ['restaurant_agent', 'matzip_agent', 'internet_agent', 'llm_search_agent'],
                '음식': ['restaurant_agent', 'matzip_agent', 'internet_agent', 'llm_search_agent'],
                '추천': ['internet_agent', 'llm_search_agent'],
                '검색': ['internet_agent', 'llm_search_agent', 'search_agent'],
                '정보': ['internet_agent', 'llm_search_agent', 'search_agent'],
                '분석': ['analysis_agent', 'data_agent'],
                '날씨': ['weather_agent', 'internet_agent'],
                '주식': ['stock_agent', 'finance_agent'],
                '쇼핑': ['shopping_agent'],
                'travel': ['internet_agent', 'llm_search_agent'],
                'food': ['restaurant_agent', 'matzip_agent', 'internet_agent', 'llm_search_agent'],
                'search': ['internet_agent', 'llm_search_agent'],
                'recommend': ['internet_agent', 'llm_search_agent'],
            }

            for domain_word, preferred_agents in reasoning_domains.items():
                if domain_word in reasoning_lower:
                    for preferred in preferred_agents:
                        for agent in available_agents:
                            if preferred in agent.lower():
                                logger.info(f"✅ 추론 기반 매칭: {target_agent} → {agent} (추론 키워드: {domain_word})")
                                return agent

        # === 6단계: 원본 쿼리에서 힌트 추출 ===
        if query:
            query_lower = query.lower()
            query_domains = {
                '여행': ['internet_agent', 'llm_search_agent'],
                '제주': ['internet_agent', 'llm_search_agent'],
                '관광': ['internet_agent', 'llm_search_agent'],
                '맛집': ['restaurant_agent', 'matzip_agent', 'internet_agent', 'llm_search_agent'],
                '추천': ['internet_agent', 'llm_search_agent'],
                '어디': ['internet_agent', 'llm_search_agent'],
                '뭐': ['internet_agent', 'llm_search_agent'],
            }

            for domain_word, preferred_agents in query_domains.items():
                if domain_word in query_lower:
                    for preferred in preferred_agents:
                        for agent in available_agents:
                            if preferred in agent.lower():
                                logger.info(f"✅ 쿼리 기반 매칭: {target_agent} → {agent} (쿼리 키워드: {domain_word})")
                                return agent

        # === 7단계: 범용 에이전트로 폴백 (최후의 수단) ===
        # 범용 에이전트 우선순위
        general_fallbacks = ['internet_agent', 'llm_search_agent', 'search_agent', 'general_agent']
        for fallback in general_fallbacks:
            for agent in available_agents:
                if fallback in agent.lower():
                    logger.warning(f"⚠️ 범용 에이전트 폴백: {target_agent} → {agent}")
                    return agent

        logger.warning(f"❌ 유사 에이전트를 찾지 못함: {target_agent}")
        return None

    def _create_default_section(self, section_key: str, query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """기본 섹션 생성"""
        if section_key == 'query_analysis':
            return {
                "original_query": query,
                "complexity": "simple",
                "multi_task": False,
                "task_count": 1,
                "primary_intent": "정보검색",
                "domains": ["일반"],
                "dependency_detected": False,
                "reasoning": "기본 분석"
            }
        elif section_key == 'task_breakdown':
            return [{
                "task_id": "default_task",
                "task_description": f"'{query}' 처리",
                "individual_query": query,
                "extracted_keywords": [],
                "domain": "일반",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            }]
        elif section_key == 'agent_mappings':
            return [{
                "task_id": "default_task",
                "selected_agent": available_agents[0] if available_agents else 'unknown',
                "agent_type": "GENERAL",
                "selection_reasoning": "기본 선택",
                "individual_query": query,
                "context_integration": "없음",
                "confidence": 0.5
            }]
        elif section_key == 'execution_plan':
            return {
                "strategy": "single_agent",
                "estimated_time": 15.0,
                "parallel_groups": [["default_task"]],
                "execution_order": ["default_task"],
                "data_passing_required": False,
                "reasoning": "single_agent 기본 전략: 단일 에이전트로 처리 가능하여 가장 효율적인 방법"
            }
        elif section_key == 'dependency_analysis':
            return {
                "has_dependencies": False,
                "dependency_type": "single_agent",
                "parallel_groups": [["default_task"]],
                "execution_order": ["default_task"],
                "reasoning": "기본 의존성 분석"
            }
        else:
            return {}


# 전역 인스턴스
_unified_processor = None

def get_unified_query_processor() -> UnifiedQueryProcessor:
    """전역 통합 쿼리 프로세서 인스턴스 반환"""
    global _unified_processor
    if _unified_processor is None:
        _unified_processor = UnifiedQueryProcessor()
    return _unified_processor


logger.info("🚀 통합 쿼리 프로세서 로드 완료!") 