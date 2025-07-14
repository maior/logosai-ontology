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
                    'search': ['검색', '찾기', '조회', '알아보기', '확인'],
                    'analyze': ['분석', '해석', '평가', '검토'],
                    'calculate': ['계산', '산출', '변환', '환산'],
                    'generate': ['생성', '만들기', '작성', '제작'],
                    'visualize': ['시각화', '차트', '그래프', '도표'],
                    'collect': ['수집', '모으기', '가져오기', '크롤링'],
                    'compare': ['비교', '대조', '견주기'],
                    'summarize': ['요약', '정리', '종합']
                },
                'agent_keywords': {
                    'weather': ['날씨', '기상', '기온', '온도'],
                    'currency': ['환율', '달러', '유로', '엔', '원', '통화'],
                    'stock': ['주가', '주식', '증권', '종목'],
                    'calculation': ['계산', '수학', '산수'],
                    'search': ['검색', '정보', '조사'],
                    'analysis': ['분석', '해석', '평가'],
                    'visualization': ['시각화', '차트', '그래프']
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
                    'search': ['search', 'find', 'look', 'check', 'get', 'fetch'],
                    'analyze': ['analyze', 'examine', 'evaluate', 'review'],
                    'calculate': ['calculate', 'compute', 'convert', 'transform'],
                    'generate': ['generate', 'create', 'make', 'produce'],
                    'visualize': ['visualize', 'chart', 'graph', 'plot'],
                    'collect': ['collect', 'gather', 'fetch', 'crawl'],
                    'compare': ['compare', 'contrast', 'versus'],
                    'summarize': ['summarize', 'sum up', 'conclude']
                },
                'agent_keywords': {
                    'weather': ['weather', 'temperature', 'climate', 'forecast'],
                    'currency': ['currency', 'exchange', 'dollar', 'euro', 'yen'],
                    'stock': ['stock', 'share', 'equity', 'market'],
                    'calculation': ['calculation', 'math', 'arithmetic'],
                    'search': ['search', 'information', 'research'],
                    'analysis': ['analysis', 'analytics', 'evaluation'],
                    'visualization': ['visualization', 'chart', 'graph']
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
            
            # 결과 파싱 및 검증 (에이전트 간 결과 전달 포함)
            result = await self._parse_and_validate_llm_response(response_text, query, available_agents, detected_language)
            
            logger.info(f"✅ 다국어 지원 통합 처리 완료: {len(result.get('agent_mappings', []))}개 매핑, {len(result.get('query_chains', []))}개 체인 생성")
            return result
            
        except Exception as e:
            logger.error(f"통합 쿼리 처리 실패: {e}")
            # 폴백 처리 (다국어 지원)
            return await self._create_llm_based_fallback(query, available_agents, detected_language)

    def _create_llm_based_unified_prompt(self, query: str, agents_info: Dict[str, Any], language: str) -> Dict[str, str]:
        """다국어 지원 완전 LLM 기반 통합 프롬프트 생성"""
        
        # 실제 에이전트 정보를 구조화된 형태로 정리
        agents_list = []
        for agent_id, info in agents_info.items():
            agent_type = info.get('agent_type', 'UNKNOWN')
            description = info.get('description', '설명 없음')
            capabilities = info.get('capabilities', [])
            tags = info.get('tags', [])

            # 주요 능력 3개만 추출
            main_capabilities = []
            for cap in capabilities[:3]:
                if isinstance(cap, dict):
                    cap_name = cap.get('name', '알 수 없음')
                    main_capabilities.append(cap_name)
                else:
                    main_capabilities.append(str(cap))
            
            agent_entry = f"""
{agent_id}:
  - 타입: {agent_type}
  - 주요 능력: {', '.join(main_capabilities) if main_capabilities else '기본 처리'}
  - 🏷️ 태그: {', '.join(tags[:6]) if tags else '없음'}{'...' if len(tags) > 6 else ''}
  - 설명: {description[:100]}{'...' if len(description) > 100 else ''}
            """
            agents_list.append(agent_entry)
        
        agents_formatted = '\n'.join(agents_list)
        
        # 언어별 예시 및 패턴
        if language == 'ko':
            examples = {
                'sequential': '"금 시세 확인하고 1온스당 원화 가격 계산" → 1단계: 금시세 조회 → 2단계: 금시세 결과를 받아서 원화 계산',
                'parallel': '"날씨와 환율 알려줘" → 날씨 조회 || 환율 조회 (동시 실행)',
                'hybrid': '"날씨와 환율 확인하고 여행 경비 계산" → 1단계: (날씨 || 환율) → 2단계: 두 결과를 받아서 경비 계산'
            }
        else:
            examples = {
                'sequential': '"check gold price and calculate ounce to won" → Step1: get gold price → Step2: calculate won price using gold price result',
                'parallel': '"tell me weather and exchange rate" → weather query || exchange rate query (simultaneous execution)',
                'hybrid': '"check weather and exchange rate then calculate travel cost" → Step1: (weather || exchange rate) → Step2: calculate cost using both results'
            }
        
        # LLM 기반 완전 통합 프롬프트
        unified_prompt = f"""당신은 사용자 쿼리를 분석하여 에이전트 실행 체인을 설계하는 전문가입니다.
특히 에이전트 간 결과 전달과 의존성 분석에 특화되어 있습니다.

🌐 감지된 언어: {self.language_config.get_config(language)['name']}
사용자 쿼리: "{query}"

🤖 사용 가능한 실제 에이전트들:
{agents_formatted}

🔗 **핵심 임무: 에이전트 간 결과 전달 분석**

1. **순차 처리 (Sequential)**: 앞 에이전트의 결과가 다음 에이전트의 입력이 되는 경우
   - 예: {examples['sequential']}
   - 각 단계마다 이전 결과를 활용한 새로운 개별 쿼리 생성

2. **병렬 처리 (Parallel)**: 독립적인 작업들을 동시에 처리
   - 예: {examples['parallel']}

3. **하이브리드 처리 (Hybrid)**: 병렬 수집 후 통합 분석
   - 예: {examples['hybrid']}

📋 **중요 지시사항:**
1. 각 작업에 **특화된 개별 쿼리**를 생성하세요 (원본 쿼리 그대로 사용 금지)
2. **데이터 전달 흐름**을 명확히 정의하세요
3. **실행 순서와 병렬 그룹**을 정확히 구성하세요
4. 순차 처리 시 이전 결과를 활용하는 방법을 명시하세요
5. 반드시 위에 나열된 에이전트만 사용하세요
6. 감지된 언어에 맞는 응답을 생성하세요

🎯 **에이전트 선택 기준:**
- 에이전트의 태그와 능력을 기반으로 최적 매칭
- 작업 도메인과 에이전트 특성의 일치도
- 에이전트 간 협업 가능성 고려

아래 JSON 형식으로 정확히 응답하세요:

{{
    "language_detected": "{language}",
    "query_analysis": {{
        "original_query": "{query}",
        "complexity": "simple|moderate|complex",
        "multi_task": true|false,
        "task_count": 숫자,
        "primary_intent": "정보검색|계산|분석|기타",
        "domains": ["도메인1", "도메인2", ...],
        "dependency_detected": true|false,
        "sequential_required": true|false,
        "reasoning": "분석 근거"
    }},
    "task_breakdown": [
        {{
            "task_id": "고유_작업_ID",
            "task_description": "구체적인 작업 설명",
            "individual_query": "이 작업만을 위한 최적화된 개별 쿼리",
            "extracted_keywords": ["키워드1", "키워드2"],
            "domain": "작업_도메인",
            "complexity": "simple|moderate|complex",
            "depends_on": ["의존하는_작업_ID들"],
            "data_requirements": ["필요한_데이터_필드들"],
            "expected_output_type": "text|number|json|url",
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
        "reasoning": "의존성 분석 근거"
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
        "reasoning": "실행 전략 근거"
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
    }}
}}

🚨 중요: 
1. individual_query는 각 작업에 특화된 개별 쿼리여야 합니다
2. 의존성이 있는 작업은 이전 결과 활용 방법을 명시해야 합니다
3. parallel_groups와 execution_order를 정확히 구성하세요
4. 실제 사용 가능한 에이전트만 사용하고, 태그 정보를 적극 활용하세요
5. 감지된 언어({language})에 맞는 응답을 생성하세요"""

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
        
        # 에이전트 매핑 검증
        valid_mappings = []
        for mapping in result.get('agent_mappings', []):
            agent_id = mapping.get('selected_agent')
            if agent_id in available_agents:
                valid_mappings.append(mapping)
            else:
                # 유사한 에이전트 찾기
                similar_agent = self._find_similar_agent(agent_id, available_agents)
                if similar_agent:
                    mapping['selected_agent'] = similar_agent
                    mapping['selection_reasoning'] += f" (대체: {agent_id} → {similar_agent})"
                    valid_mappings.append(mapping)
        
        result['agent_mappings'] = valid_mappings
        
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
                "reasoning": f"최종 폴백 - {strategy} 처리"
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

    def _select_agent_by_content(self, content: str, available_agents: List[str], language: str = None) -> str:
        """다국어 지원 내용 기반 에이전트 선택"""
        if language is None:
            language = self.language_config.detect_language(content)
        
        content_lower = content.lower()
        agent_keywords = self.language_config.get_agent_keywords(language)
        
        # 언어별 키워드 기반 매칭
        if any(word in content_lower for word in agent_keywords.get('calculation', [])):
            return self._find_best_agent_for_task(['calculator_agent', 'math_agent'], available_agents)
        elif any(word in content_lower for word in agent_keywords.get('weather', [])):
            return self._find_best_agent_for_task(['weather_agent'], available_agents)
        elif any(word in content_lower for word in agent_keywords.get('currency', [])):
            return self._find_best_agent_for_task(['currency_exchange_agent'], available_agents)
        elif any(word in content_lower for word in agent_keywords.get('search', [])):
            return self._find_best_agent_for_task(['internet_agent', 'crawler_agent', 'llm_search_agent'], available_agents)
        elif any(word in content_lower for word in agent_keywords.get('analysis', [])):
            return self._find_best_agent_for_task(['analysis_agent', 'content_formatter_agent'], available_agents)
        elif any(word in content_lower for word in agent_keywords.get('visualization', [])):
            return self._find_best_agent_for_task(['data_visualization_agent', 'content_formatter_agent'], available_agents)
        else:
            return self._find_best_agent_for_task(['llm_search_agent', 'analysis_agent'], available_agents)

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
                "reasoning": f"LLM 기반 폴백 - {strategy} 전략"
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

    def _find_similar_agent(self, target_agent: str, available_agents: List[str]) -> Optional[str]:
        """유사한 에이전트 찾기"""
        target_lower = target_agent.lower()
        
        # 정확한 매칭 시도
        for agent in available_agents:
            if agent.lower() == target_lower:
                return agent
        
        # 부분 매칭 시도
        for agent in available_agents:
            if target_lower in agent.lower() or agent.lower() in target_lower:
                return agent
        
        # 키워드 기반 매칭
        keywords = target_lower.replace('_', ' ').split()
        for agent in available_agents:
            agent_lower = agent.lower()
            if any(keyword in agent_lower for keyword in keywords):
                return agent
        
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
                "reasoning": "기본 실행 계획"
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