"""
🚀 Unified Query Processor

An efficient system that consolidates previously fragmented LLM calls into one,
handling query analysis → agent selection → workflow design → query optimization
in a single LLM call.

🔄 v3.1: Multilingual support and hardcoding removal
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
from .hybrid_agent_selector import get_hybrid_selector, HybridAgentSelector


class LanguageConfig:
    """Language-specific configuration class"""
    
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
                    # 🚀 Travel/recommendation intent added
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
                    # 🗓️ Schedule/calendar domain added
                    'schedule': ['일정', '스케줄', '약속', '캘린더', '등록', '추가해줘', '추가해', '알려줘', '목요일', '금요일', '이번주', '다음주'],
                    'calendar': ['일정', '스케줄', '약속', '캘린더', '등록', '추가해줘', '추가해', '알려줘', '이번주', '다음주'],
                    # 🚀 Travel/recommendation domain added
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
        """Detect language from text"""
        # Simple language detection logic
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if korean_chars > english_chars:
            return 'ko'
        else:
            return 'en'
    
    def get_config(self, language: str) -> Dict[str, Any]:
        """Return language-specific configuration"""
        return self.configs.get(language, self.configs['en'])
    
    def get_stopwords(self, language: str) -> set:
        """Return language-specific stopwords"""
        return self.get_config(language).get('stopwords', set())
    
    def get_connectors(self, language: str) -> List[str]:
        """Return language-specific connectors"""
        return self.get_config(language).get('connectors', [])
    
    def get_sequential_patterns(self, language: str) -> List[str]:
        """Return language-specific sequential patterns"""
        return self.get_config(language).get('sequential_patterns', [])
    
    def get_intent_keywords(self, language: str) -> Dict[str, List[str]]:
        """Return language-specific intent keywords"""
        return self.get_config(language).get('intent_keywords', {})
    
    def get_agent_keywords(self, language: str) -> Dict[str, List[str]]:
        """Return language-specific agent keywords"""
        return self.get_config(language).get('agent_keywords', {})


class TaskDependency:
    """Task dependency information"""
    def __init__(self, task_id: str, depends_on: List[str], data_flow: Dict[str, str]):
        self.task_id = task_id
        self.depends_on = depends_on  # Dependent task IDs
        self.data_flow = data_flow    # Data flow information {"from_task": "data_field"}


class QueryChain:
    """Query chain - execution order and data passing for dependent tasks"""
    def __init__(self, chain_id: str, tasks: List[Dict[str, Any]], dependencies: List[TaskDependency]):
        self.chain_id = chain_id
        self.tasks = tasks
        self.dependencies = dependencies
        self.execution_order = self._calculate_execution_order()
    
    def _calculate_execution_order(self) -> List[List[str]]:
        """Calculate execution order considering dependencies (topological sort)"""
        # Simple topological sort implementation
        in_degree = {task["task_id"]: 0 for task in self.tasks}
        
        # Calculate in-degree for each task
        for dep in self.dependencies:
            in_degree[dep.task_id] = len(dep.depends_on)
        
        execution_order = []
        remaining_tasks = set(task["task_id"] for task in self.tasks)
        
        while remaining_tasks:
            # Find tasks with in-degree 0 (can run concurrently)
            ready_tasks = [task_id for task_id in remaining_tasks if in_degree[task_id] == 0]
            
            if not ready_tasks:
                # Circular dependency detected - process remaining tasks in order
                ready_tasks = [list(remaining_tasks)[0]]
            
            execution_order.append(ready_tasks)
            
            # Remove processed tasks and update dependencies
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                
                # Decrease in-degree for tasks that depend on this one
                for dep in self.dependencies:
                    if task_id in dep.depends_on:
                        in_degree[dep.task_id] -= 1
        
        return execution_order


class AgentResultContext:
    """Agent execution result context - result passing for sequential processing"""
    def __init__(self):
        self.results = {}  # {task_id: result}
        self.execution_history = []  # [(task_id, timestamp, result)]
    
    def add_result(self, task_id: str, result: Any):
        """Add result"""
        self.results[task_id] = result
        self.execution_history.append((task_id, datetime.now(), result))
    
    def get_result(self, task_id: str) -> Any:
        """Get result for a specific task"""
        return self.results.get(task_id)
    
    def get_previous_results(self, current_task_id: str, dependency_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get previous results that the current task depends on"""
        dependent_tasks = dependency_map.get(current_task_id, [])
        previous_results = {}
        for dep_task in dependent_tasks:
            if dep_task in self.results:
                previous_results[dep_task] = self.results[dep_task]
        return previous_results
    
    def format_context_for_agent(self, task_id: str, dependency_map: Dict[str, List[str]]) -> str:
        """Format context to pass to agent"""
        previous_results = self.get_previous_results(task_id, dependency_map)
        if not previous_results:
            return ""
        
        context_parts = ["Refer to the previous task results:"]
        for dep_task, result in previous_results.items():
            # Extract key information from result
            result_summary = self._extract_key_info_from_result(result)
            context_parts.append(f"- {dep_task}: {result_summary}")
        
        return "\n".join(context_parts)
    
    def _extract_key_info_from_result(self, result: Any) -> str:
        """Extract key information from result"""
        if isinstance(result, dict):
            # Extract info from keys like answer, content, result, etc.
            for key in ['answer', 'content', 'result', 'data', 'output']:
                if key in result:
                    value = result[key]
                    if isinstance(value, str):
                        return value[:200] + "..." if len(value) > 200 else value
                    else:
                        return str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            
            # Convert entire result to string
            return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
        elif isinstance(result, str):
            return result[:200] + "..." if len(result) > 200 else result
        else:
            return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)


class UnifiedQueryProcessor:
    """🚀 Unified Query Processor - multilingual support and hardcoding removal"""
    
    def __init__(self):
        self.llm_manager = get_ontology_llm_manager()
        self.installed_agents_info = []
        self.result_context = AgentResultContext()
        self.language_config = LanguageConfig()
        
    def set_installed_agents_info(self, installed_agents_info: List[Dict[str, Any]]):
        """Set installed agent information"""
        self.installed_agents_info = installed_agents_info
        logger.info(f"🎯 Installed agent info updated: {len(installed_agents_info)} agents")
        
        # Log agent info summary
        for agent in installed_agents_info:
            agent_id = agent.get('agent_id', 'unknown')
            agent_data = agent.get('agent_data', {})
            agent_name = agent_data.get('name', agent_id)
            agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
            capabilities = agent_data.get('capabilities', [])
            
            logger.info(f"  📋 {agent_id}: {agent_name} ({agent_type}) - {len(capabilities)} capabilities")

    async def process_unified_query(self, 
                                  query: str, 
                                  available_agents: List[str]) -> Dict[str, Any]:
        """
        Unified query processing - multilingual LLM-based full automation + inter-agent result passing
        
        Args:
            query: Original user query
            available_agents: List of available agents
            
        Returns:
            Dict: Integrated analysis result (including inter-agent result passing)
        """
        try:
            # Detect language
            detected_language = self.language_config.detect_language(query)
            logger.info(f"🌐 Detected language: {self.language_config.get_config(detected_language)['name']}")
            
            logger.info(f"🚀 Unified query processing started (v3.1 - multilingual): {query}")

            # 🏢 Samsung domain priority check (business logic - handled before LLM call)
            # CLAUDE.md guideline: "Samsung domain is special business logic requiring routing to dedicated gateway agent"
            normalized_query = self._normalize_samsung_typos(query)
            if self._is_samsung_domain_query(normalized_query):
                samsung_agents = [agent for agent in available_agents if "samsung_gateway" in agent.lower()]
                if samsung_agents:
                    samsung_agent = samsung_agents[0]
                    logger.info(f"🏢 Samsung domain detected - priority routing: {samsung_agent}")
                    # Return result with direct routing to Samsung agent
                    return {
                        "intent": "samsung_domain_analysis",
                        "complexity": {"level": "moderate", "score": 0.6},
                        "execution_plan": {
                            "strategy": "SINGLE_AGENT",
                            "reasoning": f"Samsung/semiconductor domain specialist agent ({samsung_agent}) is optimal for this query.",
                            "estimated_time": 30
                        },
                        "agent_mappings": [{
                            "task_id": "samsung_task_1",
                            "selected_agent": samsung_agent,
                            "task_type": "samsung_domain_analysis",
                            "individual_query": query,
                            "confidence": 0.95,
                            "reasoning": "Samsung/semiconductor domain query - priority routing to specialist agent"
                        }],
                        "query_chains": [],
                        "reasoning": f"Detected as Samsung/semiconductor-related query, routing directly to specialist agent ({samsung_agent}).",
                        "language": detected_language,
                        "samsung_priority_routing": True
                    }
                else:
                    logger.warning(f"⚠️ Samsung domain detected, but samsung_gateway_agent is not installed. Proceeding with LLM-based selection.")

            # Initialize result context
            self.result_context = AgentResultContext()
            
            # Build installed agent information
            agents_info = self._build_agents_info_for_llm(available_agents)
            
            # LLM-based complete unified analysis
            unified_prompt = self._create_llm_based_unified_prompt(query, agents_info, detected_language)
            
            logger.info("🧠 Running LLM-based complete unified analysis...")
            start_time = datetime.now()
            
            # Call LLM
            llm = self.llm_manager.get_llm(OntologyLLMType.SEMANTIC_ANALYZER)
            response = await llm.ainvoke(unified_prompt["query"])
            
            # Extract content from AIMessage
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"⏱️ LLM processing complete: {processing_time:.2f}s")

            # 🔍 DEBUG: Log raw LLM response
            logger.info(f"🔍 [DEBUG] Raw LLM response (first 800 chars): {response_text[:800] if len(response_text) > 800 else response_text}")

            # Parse and validate result (including inter-agent result passing)
            result = await self._parse_and_validate_llm_response(response_text, query, available_agents, detected_language)
            
            logger.info(f"✅ Multilingual unified processing complete: {len(result.get('agent_mappings', []))} mappings, {len(result.get('query_chains', []))} chains created")
            return result
            
        except Exception as e:
            logger.error(f"Unified query processing failed: {e}")
            # Fallback processing (multilingual)
            return await self._create_llm_based_fallback(query, available_agents, detected_language)

    def _create_llm_based_unified_prompt(self, query: str, agents_info: Dict[str, Any], language: str) -> Dict[str, str]:
        """Generate multilingual LLM-based unified prompt - 🚀 improved agent info formatting"""

        # 🚀 Improvement: Format agent info in detail for LLM to understand easily
        agents_list = []
        for agent_id, info in agents_info.items():
            agent_data = info.get('agent_data', info)  # use agent_data if available
            agent_type = agent_data.get('agent_type', info.get('agent_type', 'UNKNOWN'))
            description = agent_data.get('description', info.get('description', 'No description'))
            name = agent_data.get('name', info.get('name', agent_id))
            capabilities = agent_data.get('capabilities', info.get('capabilities', []))
            tags = agent_data.get('tags', info.get('tags', []))

            # 🚀 Improvement: Extract all capabilities (max 10)
            all_capabilities = []
            for cap in capabilities[:10]:
                if isinstance(cap, dict):
                    cap_name = cap.get('name', cap.get('capability', ''))
                    cap_desc = cap.get('description', '')
                    if cap_name:
                        all_capabilities.append(f"{cap_name}" + (f"({cap_desc[:30]})" if cap_desc else ""))
                elif cap:
                    all_capabilities.append(str(cap))

            # 🚀 Improvement: Use tags as domain hints (max 15)
            all_tags = tags[:15] if tags else []

            # 🚀 Extract data type spec (for intelligent workflow)
            input_spec = info.get('input_spec', {})
            output_spec = info.get('output_spec', {})
            input_type = input_spec.get('type', 'text') if input_spec else 'text'
            output_type = output_spec.get('type', 'text') if output_spec else 'text'

            # 🚀 Improvement: Clearly distinguish agent ID and name + include data type spec
            agent_entry = f"""
📌 Agent ID: {agent_id}
   Name: {name}
   Type: {agent_type}
   Description: {description[:300]}{'...' if len(description) > 300 else ''}
   Capabilities: {', '.join(all_capabilities) if all_capabilities else 'General processing'}
   Domain tags: {', '.join(all_tags) if all_tags else 'General'}
   📥 Input type: {input_type}
   📤 Output type: {output_type}
   ✅ To select this agent, enter "{agent_id}" in selected_agent of agent_mappings.
"""
            agents_list.append(agent_entry)
        
        agents_formatted = '\n'.join(agents_list)
        
        # Language-specific examples and patterns
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
        
        # LLM-based query analysis prompt
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
   - samsung_gateway: 삼성/반도체 **제조 공정** 전문 분석 (Particle, Yield, FAB, Etch, EUV 등)

   ### 🏢 삼성 도메인 구분 규칙 (매우 중요!)
   **samsung_gateway_agent는 삼성/반도체 "제조 공정" 전문 에이전트입니다.**
   **주식, 금융, 투자 관련 쿼리는 절대 처리하지 않습니다.**

   ✅ samsung_gateway 사용: 반도체 제조 관련
      - "삼성전자 반도체 NAND 분석", "삼성 FAB Particle 이슈", "DDR5 수율 개선", "Etch 공정 최적화"

   ❌ samsung_gateway 금지: 주식/금융/투자 관련
      - "삼성전자 주가", "삼성전자 종가/시가", "삼성전자 시가총액", "삼성전자 배당금", "삼성전자 매출/실적"
      - 키워드: 주가, 종가, 시가, 고가, 저가, 시총, 배당, 매수, 매도, 거래, 실적, 매출

   🔑 핵심: "삼성전자"가 포함되어도 금융 키워드가 있으면 → internet_agent 사용!

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

🧠 **지능형 워크플로우 설계 - 데이터 흐름 추론 (Backward Chaining):**

## 📊 에이전트 데이터 타입 규칙:
각 에이전트는 특정 데이터 타입을 입력받고 출력합니다:
- **raw_text** (비정형 텍스트): HTML, 웹 검색 결과, 자연어 등
- **structured_data** (정형 데이터): JSON, 배열, 객체, 테이블 등
- **visual** (시각적 출력): 차트, 그래프, HTML/SVG 등
- **text** (자연어 쿼리): 사용자 입력, 검색 키워드 등
- **numeric** (숫자): 계산 결과, 수치 데이터 등

## 🔄 워크플로우 설계 방법:
### Step 1: 최종 출력 분석
- 쿼리에서 요구하는 최종 출력 형태를 파악 (예: 그래프 → visual)

### Step 2: 역방향 추론 (Backward Chaining)
- 최종 출력을 생성할 수 있는 에이전트를 찾음
- 해당 에이전트의 필요 입력 타입을 확인
- 그 입력을 제공할 수 있는 이전 에이전트를 찾음
- 데이터 소스(검색 등)에 도달할 때까지 반복

### Step 3: 데이터 호환성 검증
- **중요**: 에이전트 간 데이터 타입이 호환되어야 함!
- 비정형(raw_text) → 정형(structured_data) 변환 필요 시 분석 에이전트 삽입
- 예: 웹 검색(raw_text) → 시각화(structured_data 필요) = 중간에 분석 에이전트 필요!

### Step 4: 갭 분석 (Gap Analysis)
- 출력 타입과 다음 에이전트 입력 타입이 맞지 않으면 → 변환 에이전트 추가
- raw_text ≠ structured_data → 분석 에이전트로 변환 필요

## ⚡ 예시:
쿼리: "삼성전자 주가 5일치 그래프로 그려줘"
1. 최종 출력: visual (그래프)
2. 시각화 에이전트 필요 입력: structured_data
3. 데이터 소스: 인터넷 검색 출력: raw_text
4. 갭 발견: raw_text ≠ structured_data → 분석 에이전트 삽입
5. 결과: internet(raw_text) → analysis(structured_data) → visualization(visual) = 3단계

쿼리: "분석하고 개선방안 제시해줘"
1. 같은 분석 에이전트가 분석과 제안 모두 처리 가능
2. 데이터 흐름 변환 불필요
3. 결과: 1개 작업

🚨 **핵심 규칙 (반드시 준수):**
1. **⛔ 데이터 타입 불일치 시 직접 연결 금지**:
   - 📥 입력 타입과 📤 출력 타입이 맞지 않으면 **절대** 직접 연결하지 마세요
   - 예: internet(출력: raw_text) → visualization(입력: structured_data) = ❌ 금지!
   - 반드시 중간에 analysis 에이전트(any→structured_data 변환 가능) 삽입

2. **⚠️ 시각화 에이전트 제한**:
   - 시각화 에이전트는 **structured_data만** 입력으로 받을 수 있음
   - raw_text를 직접 처리할 수 없음
   - 검색 결과(raw_text)를 시각화하려면 **반드시** 분석 에이전트 거쳐야 함

3. **✅ 갭 발견 시 변환 에이전트 삽입 필수**:
   - raw_text → structured_data 변환: analysis_agent 필수
   - 예: "주가 검색해서 그래프로" = 3단계: internet → analysis → visualization

4. **같은 능력이면 통합**: 한 에이전트가 처리 가능하면 분할하지 않음
5. **의미론적 매칭**: 쿼리 의도와 에이전트 메타데이터를 분석하여 가장 적합한 에이전트 선택
6. **모든 에이전트는 동등**: 특정 에이전트를 "폴백"으로 취급하지 않음

7. **적합한 에이전트가 없을 때 - 유용한 피드백 제공:**
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

❌ **같은 능력으로 처리 가능한 경우 - 분할 금지:**
- "수율 분석하고 개선방안 제시" → 같은 분석 에이전트가 둘 다 가능 → 1개 작업 ✅
- "현황 파악하고 전략 수립" → 같은 분석 에이전트가 둘 다 가능 → 1개 작업 ✅
- "트렌드 분석하고 예측" → 같은 분석 에이전트가 둘 다 가능 → 1개 작업 ✅

✅ **다른 능력이 필요한 경우 - 반드시 분할:**
- "삼성전자 주가 검색해서 그래프로 그려줘" → 3개 작업 (sequential):
  - task_1: internet_agent (주가 데이터 검색)
  - task_2: analysis_agent (데이터 정리/구조화) - depends_on: task_1
  - task_3: data_visualization_agent (그래프 시각화) - depends_on: task_2
- "비트코인 시세 확인하고 원화로 환산" → 2개 작업 (sequential):
  - task_1: internet_agent (시세 검색)
  - task_2: calculator_agent (환율 계산) - depends_on: task_1

✅ **같은 능력으로 처리 가능한 예시:**
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
6. **작업 분할 기준** - 같은 에이전트가 처리 가능하면 통합, 다른 능력이 필요하면 반드시 분할 (예: 검색+시각화는 2개 작업)
7. reasoning 필드에 "왜 그렇게 결정했는지" 구체적 이유 명시"""

        return {"query": unified_prompt}

    async def _parse_and_validate_llm_response(self, 
                                             response: str, 
                                             original_query: str, 
                                             available_agents: List[str],
                                             language: str) -> Dict[str, Any]:
        """Parse and validate LLM response - multilingual support"""
        try:
            logger.info(f"🔍 LLM response parsing started... (length: {len(response)})")
            
            # Extract and parse JSON
            result = self._safe_json_parse_from_response(response)
            
            if not result:
                logger.warning("JSON parsing failed, switching to LLM-based fallback analysis")
                return await self._create_llm_based_fallback(original_query, available_agents, language)
            
            # Validate and supplement result
            result = self._validate_and_enhance_llm_result(result, original_query, available_agents, language)
            
            # Create query chains for inter-agent result passing
            result = self._create_context_aware_query_chains(result)
            
            # Validate and supplement execution plan
            result = self._validate_execution_plan(result)
            
            logger.info(f"✅ LLM response validation complete: {len(result.get('agent_mappings', []))} valid mappings")
            return result
            
        except Exception as e:
            logger.error(f"Error during LLM response parsing: {e}")
            return await self._create_llm_based_fallback(original_query, available_agents, language)

    def _safe_json_parse_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Safe JSON parsing from response"""
        try:
            # Find JSON block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif response.strip().startswith('{'):
                json_str = response.strip()
            else:
                # Attempt to extract JSON portion only
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    return None
            
            # Clean JSON string
            cleaned_json = self._clean_json_string(json_str)
            
            # Parse JSON
            return json.loads(cleaned_json)
            
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            return None

    def _validate_and_enhance_llm_result(self, result: Dict[str, Any], original_query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """Validate and enhance LLM result"""

        # Check required keys
        required_keys = ['query_analysis', 'task_breakdown', 'agent_mappings', 'execution_plan', 'dependency_analysis']
        for key in required_keys:
            if key not in result:
                result[key] = self._create_default_section(key, original_query, available_agents, language)

        # Validate and enhance query_analysis
        query_analysis = result.get('query_analysis', {})

        # Validate query_type field and set default
        if 'query_type' not in query_analysis:
            # Infer basic type with simple heuristic
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

        # Set classification_confidence default
        if 'classification_confidence' not in query_analysis:
            query_analysis['classification_confidence'] = 0.7

        # Set classification_reasoning default
        if 'classification_reasoning' not in query_analysis:
            query_analysis['classification_reasoning'] = f"Auto-classified: {query_analysis.get('query_type', 'AGENT_TASK')}/{query_analysis.get('query_subtype', 'unknown')}"

        result['query_analysis'] = query_analysis
        logger.info(f"🎯 Query type classified: {query_analysis.get('query_type')}/{query_analysis.get('query_subtype')} (confidence: {query_analysis.get('classification_confidence')})")

        # 🚀 Improved agent mapping validation - respect LLM intent as much as possible
        # 🔍 DEBUG: Raw agent mapping selected by LLM
        raw_mappings = result.get('agent_mappings', [])
        logger.info(f"🔍 [DEBUG] LLM agent_mappings raw: {raw_mappings}")
        logger.info(f"🔍 [DEBUG] available_agents: {available_agents[:5]}... (total: {len(available_agents)})")

        valid_mappings = []
        for mapping in raw_mappings:
            agent_id = mapping.get('selected_agent')
            selection_reasoning = mapping.get('selection_reasoning', '')
            individual_query = mapping.get('individual_query', original_query)

            if agent_id in available_agents:
                # Exact match found
                logger.info(f"✅ LLM agent selection valid: {agent_id}")
                valid_mappings.append(mapping)
            else:
                # 🚀 Improvement: Pass LLM intent (selection_reasoning) along with original query
                similar_agent = self._find_similar_agent(
                    agent_id,
                    available_agents,
                    selection_reasoning=selection_reasoning,
                    query=individual_query
                )
                if similar_agent:
                    original_agent = agent_id
                    mapping['selected_agent'] = similar_agent
                    mapping['selection_reasoning'] = f"{selection_reasoning} (🔄 LLM intent-based replacement: {original_agent} → {similar_agent})"
                    mapping['llm_original_selection'] = original_agent  # Record original LLM selection
                    valid_mappings.append(mapping)
                    logger.info(f"🔄 LLM intent-based agent replacement: {original_agent} → {similar_agent}")
                else:
                    logger.warning(f"⚠️ Agent matching failed, skipping: {agent_id}")

        # 🚀 Improvement: Fallback only when no valid mappings (last resort)
        if not valid_mappings:
            logger.warning(f"⚠️ All agent mappings failed, running query-based fallback")
            # Use intent from the first agent LLM tried to select
            first_mapping = result.get('agent_mappings', [{}])[0] if result.get('agent_mappings') else {}
            first_reasoning = first_mapping.get('selection_reasoning', '')

            # Select fallback agent (based on query analysis)
            fallback_agent = self._select_agent_by_content(original_query, available_agents, language)
            valid_mappings = [{
                "task_id": "fallback_task",
                "selected_agent": fallback_agent,
                "agent_type": "GENERAL",
                "selection_reasoning": f"Fallback selection: {fallback_agent} (original LLM reasoning: {first_reasoning[:100]})" if first_reasoning else f"Fallback selection: {fallback_agent}",
                "individual_query": original_query,
                "context_integration": "None",
                "confidence": 0.6,
                "is_fallback": True
            }]

        result['agent_mappings'] = valid_mappings

        # 🧠 Validate and process agent_availability
        agent_availability = result.get('agent_availability', {})
        no_suitable_agent = agent_availability.get('no_suitable_agent', False)

        if no_suitable_agent:
            # LLM determined no suitable agent - provide useful feedback
            feedback_type = agent_availability.get('feedback_type', 'none')
            user_message = agent_availability.get('user_message', 'No suitable agent found for the requested task.')
            required_capabilities = agent_availability.get('required_capabilities', [])

            logger.warning(f"⚠️ No suitable agent (feedback type: {feedback_type})")
            logger.warning(f"   Message: {user_message}")

            # Additional info by feedback type
            feedback_details = {}
            if feedback_type == 'clarification_needed':
                feedback_details['clarification_questions'] = agent_availability.get('clarification_questions', [])
                logger.info(f"   Clarification questions: {feedback_details['clarification_questions']}")
            elif feedback_type == 'alternative_suggested':
                feedback_details['suggested_alternatives'] = agent_availability.get('suggested_alternatives', [])
                logger.info(f"   Suggested alternatives: {feedback_details['suggested_alternatives']}")
            elif feedback_type == 'impossible_request':
                feedback_details['impossible_reason'] = agent_availability.get('impossible_reason', '')
                logger.info(f"   Reason impossible: {feedback_details['impossible_reason']}")

            # Add user notification info to result
            result['no_suitable_agent_info'] = {
                'status': True,
                'feedback_type': feedback_type,
                'user_message': user_message,
                'required_capabilities': required_capabilities,
                **feedback_details
            }
        else:
            # Suitable agent found - set agent_availability defaults
            if 'agent_availability' not in result:
                result['agent_availability'] = {
                    'no_suitable_agent': False,
                    'feedback_type': 'none',
                    'availability_reasoning': 'Found a suitable agent for the query.',
                    'user_message': '',
                    'suggested_alternatives': [],
                    'required_capabilities': []
                }

        # Validate dependency analysis
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
        """Create context-aware query chains - including inter-agent result passing"""
        try:
            dependency_analysis = result.get('dependency_analysis', {})
            tasks = result.get('task_breakdown', [])
            agent_mappings = result.get('agent_mappings', [])
            
            query_chains = []
            
            # When dependency chains exist
            if dependency_analysis.get('has_dependencies', False):
                dependency_chains = dependency_analysis.get('dependency_chains', [])
                
                for chain_info in dependency_chains:
                    chain_id = chain_info.get('chain_id', f'chain_{len(query_chains)}')
                    tasks_in_order = chain_info.get('tasks_in_order', [])
                    data_flow = chain_info.get('data_flow', [])
                    
                    # Create TaskDependency objects
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
                                data_flow_info[dep] = 'result'  # default value
                            
                            dependency = TaskDependency(task_id, depends_on, data_flow_info)
                            dependencies.append(dependency)
                    
                    # Filter tasks included in chain
                    chain_tasks = [task for task in tasks if task.get('task_id') in tasks_in_order]
                    
                    # Add context passing info to agent mappings
                    for mapping in agent_mappings:
                        task_id = mapping.get('task_id')
                        if task_id in dependency_map:
                            # Add context passing method for tasks with dependencies
                            mapping['requires_context'] = True
                            mapping['context_dependencies'] = dependency_map[task_id]
                            mapping['context_integration'] = f"Refer to previous task results: {', '.join(dependency_map[task_id])}"
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
            
            logger.info(f"✅ Context-aware query chains created: {len(query_chains)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create context-aware query chains: {e}")
            result['query_chains'] = []
            result['context_passing_enabled'] = False
            return result

    def _validate_execution_plan(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and supplement execution plan"""
        try:
            execution_plan = result.get('execution_plan', {})
            dependency_analysis = result.get('dependency_analysis', {})
            
            # Validate strategy
            strategy = execution_plan.get('strategy', 'parallel')
            if strategy not in ['single_agent', 'parallel', 'sequential', 'hybrid']:
                strategy = 'parallel'
            
            # Validate parallel groups
            parallel_groups = dependency_analysis.get('parallel_groups', [])
            if not parallel_groups:
                task_ids = [task.get('task_id') for task in result.get('task_breakdown', [])]
                if strategy == 'sequential':
                    parallel_groups = [[task_id] for task_id in task_ids]
                elif strategy == 'parallel':
                    parallel_groups = [task_ids] if task_ids else []
                else:  # hybrid
                    parallel_groups = [task_ids] if task_ids else []
            
            # Validate execution order
            execution_order = dependency_analysis.get('execution_order', [])
            if not execution_order:
                task_ids = [task.get('task_id') for task in result.get('task_breakdown', [])]
                if strategy == 'sequential':
                    execution_order = task_ids
                else:
                    execution_order = task_ids
            
            # Update execution plan
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
            
            logger.info(f"✅ Execution plan validation complete: {strategy} strategy, {len(parallel_groups)} groups")
            return result
            
        except Exception as e:
            logger.error(f"Execution plan validation failed: {e}")
            return result

    async def _create_llm_based_fallback(self, query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """Generate LLM-based fallback result"""
        logger.warning("🔄 Generating unified result in LLM-based fallback mode")
        
        try:
            # Basic analysis with simple LLM prompt
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
            
            # Simple parsing
            fallback_result = self._safe_json_parse_from_response(response_text)
            
            if fallback_result:
                return self._convert_fallback_to_full_result(fallback_result, query, available_agents, language)
            
        except Exception as e:
            logger.error(f"LLM-based fallback also failed: {e}")
        
        # Final fallback - basic structure
        return self._create_emergency_fallback(query, available_agents, language)

    def _create_emergency_fallback(self, query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """Final fallback - create basic structure"""
        logger.warning("🚨 Final fallback mode: generating result with basic structure")
        
        # Most basic task decomposition
        if not available_agents:
            available_agents = ['default_agent']
        
        # Check for Samsung query then select appropriate agent
        if self._is_samsung_domain_query(query):
            samsung_agents = [agent for agent in available_agents if "samsung_gateway" in agent.lower()]
            primary_agent = samsung_agents[0] if samsung_agents else available_agents[0]
        else:
            primary_agent = available_agents[0]
        
        # Detect language
        if language is None:
            language = self.language_config.detect_language(query)
        
        # Simple keyword-based analysis
        keywords = self._extract_simple_keywords(query, language)
        
        # Detect sequential processing pattern via regex
        sequential_detected = self._detect_sequential_pattern_regex(query, language)
        
        if sequential_detected:
            # Sequential processing required - decompose tasks
            task_breakdown, agent_mappings = self._create_sequential_tasks_emergency(query, available_agents, keywords, language)
            
            # Sequential execution plan
            task_ids = [task["task_id"] for task in task_breakdown]
            parallel_groups = [[task_id] for task_id in task_ids]  # Each task as its own group
            execution_order = task_ids
            strategy = "sequential"
        else:
            # Default single task
            task_breakdown = [{
                "task_id": "emergency_task_1",
                "task_description": f"'{query}' processing",
                "individual_query": query,
                "extracted_keywords": keywords,
                "domain": "general",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            }]
            
            # Default agent mapping
            agent_mappings = [{
                "task_id": "emergency_task_1",
                "selected_agent": primary_agent,
                "agent_type": "GENERAL",
                "selection_reasoning": "Final fallback - default agent selection",
                "individual_query": query,
                "context_integration": "None",
                "confidence": 0.5
            }]
            
            # Default execution plan
            parallel_groups = [["emergency_task_1"]]
            execution_order = ["emergency_task_1"]
            strategy = "single_agent"
        
        return {
            "query_analysis": {
                "original_query": query,
                "complexity": "simple",
                "multi_task": False,
                "task_count": 1,
                "primary_intent": "information_search",
                "domains": ["general"],
                "dependency_detected": False,
                "reasoning": "Final fallback - basic analysis"
            },
            "task_breakdown": task_breakdown,
            "agent_mappings": agent_mappings,
            "dependency_analysis": {
                "has_dependencies": False,
                "dependency_type": "single_agent",
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "reasoning": "Final fallback - single task"
            },
            "execution_plan": {
                "strategy": strategy,
                "estimated_time": len(task_breakdown) * 15.0,
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "data_passing_required": sequential_detected,
                "reasoning": f"Final fallback: {strategy} strategy selected based on task dependency analysis"
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
        """Multilingual simple keyword extraction"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        # Get language-specific stopwords
        stopwords = self.language_config.get_stopwords(language)
        
        # Simple tokenization (whitespace-based)
        tokens = query.split()
        
        # Remove stopwords and filter by length
        keywords = [token for token in tokens if token not in stopwords and len(token) > 1]
        
        return keywords[:5]  # Max 5

    def _detect_sequential_pattern_regex(self, query: str, language: str = None) -> bool:
        """Multilingual regex-based sequential processing pattern detection"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        # Get language-specific sequential patterns
        sequential_patterns = self.language_config.get_sequential_patterns(language)
        
        query_lower = query.lower()
        for pattern in sequential_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def _create_sequential_tasks_emergency(self, query: str, available_agents: List[str], keywords: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Multilingual emergency fallback sequential task decomposition"""
        import re
        
        # Language-specific special pattern handling
        if language == 'ko':
            # Gold price + calculation pattern special handling
            if re.search(r'금.*?시세.*?(?:하고|한\s*다음|후에|다음에|이후).*?(?:계산|변환|원화)', query.lower()):
                return self._create_gold_price_sequential_tasks(query, available_agents, language)
            
            # AI trend + analysis + visualization pattern special handling
            if re.search(r'(?:AI|인공지능|기술).*?(?:트렌드|동향).*?(?:조사|분석).*?(?:해서|하고).*?(?:시각|보고서|리포트)', query.lower()):
                return self._create_ai_trend_analysis_tasks(query, available_agents, language)
            
            # General research + analysis + generation pattern
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
        
        # General sequential pattern decomposition
        connectors = self.language_config.get_connectors(language)
        for connector in connectors:
            if connector in query:
                parts = query.split(connector)
                if len(parts) >= 2:
                    return self._create_general_sequential_tasks(parts, available_agents, language)
        
        # Fallback: default single task
        return self._create_default_single_task(query, available_agents, keywords, language)

    def _create_gold_price_sequential_tasks(self, query: str, available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Multilingual gold price + calculation sequential task creation"""
        # Step 1: Get gold price
        # Step 2: Calculate KRW amount
        # Step 3: Summarize
        
        # Select appropriate agents
        crawler_agent = self._find_best_agent_for_task(['crawler_agent', 'internet_agent', 'llm_search_agent'], available_agents)
        calculator_agent = self._find_best_agent_for_task(['calculator_agent', 'currency_exchange_agent', 'math_agent'], available_agents)
        analysis_agent = self._find_best_agent_for_task(['analysis_agent', 'content_formatter_agent', 'llm_search_agent'], available_agents)
        
        task_breakdown = [
            {
                "task_id": "gold_price_task",
                "task_description": "Check today's gold price.",
                "individual_query": "Please provide the current gold price information. Include the price and change rate.",
                "extracted_keywords": ["gold", "price", "rate"],
                "domain": "finance",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            },
            {
                "task_id": "calculation_task",
                "task_description": "Calculate the confirmed gold price per ounce in KRW.",
                "individual_query": "Please accurately calculate the KRW price per ounce. Include the calculation process and result.",
                "extracted_keywords": ["1oz", "KRW", "calculate"],
                "domain": "calculation",
                "complexity": "moderate",
                "depends_on": ["gold_price_task"],
                "expected_output_type": "number"
            },
            {
                "task_id": "summary_task",
                "task_description": "Summarize the gold price and calculation results.",
                "individual_query": "Please organize the gold price information and calculation results in a user-friendly format.",
                "extracted_keywords": ["summarize", "organize"],
                "domain": "analysis",
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
                "selection_reasoning": "Crawler agent selected for gold price lookup",
                "individual_query": "Please provide the current gold price information. Include the price and change rate.",
                "context_integration": "None",
                "confidence": 0.85
            },
            {
                "task_id": "calculation_task",
                "selected_agent": calculator_agent,
                "agent_type": "CALCULATOR",
                "selection_reasoning": "Calculator agent selected for KRW calculation",
                "individual_query": "Please accurately calculate the KRW price per ounce. Include the calculation process and result.",
                "context_integration": "Use previous gold price result",
                "confidence": 0.90
            },
            {
                "task_id": "summary_task",
                "selected_agent": analysis_agent,
                "agent_type": "ANALYSIS",
                "selection_reasoning": "Analysis agent selected for result summarization",
                "individual_query": "Please organize the gold price information and calculation results in a user-friendly format.",
                "context_integration": "Use all previous results",
                "confidence": 0.80
            }
        ]
        
        return task_breakdown, agent_mappings

    def _create_ai_trend_analysis_tasks(self, query: str, available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Multilingual AI trend analysis + visualization sequential task creation"""
        # Step 1: Research AI technology trends
        # Step 2: Analyze collected data
        # Step 3: Generate visual report
        
        # Select appropriate agents
        research_agent = self._find_best_agent_for_task(['internet_agent', 'crawler_agent', 'llm_search_agent'], available_agents)
        analysis_agent = self._find_best_agent_for_task(['analysis_agent', 'task_classifier_agent', 'llm_search_agent'], available_agents)
        visualization_agent = self._find_best_agent_for_task(['data_visualization_agent', 'content_formatter_agent', 'document_agent'], available_agents)
        
        task_breakdown = [
            {
                "task_id": "ai_trend_research_task",
                "task_description": "Research AI technology trends.",
                "individual_query": "Please research the latest AI technology trends. Collect information on key technologies, market trends, and major players.",
                "extracted_keywords": ["AI", "technology", "trend", "research"],
                "domain": "technology_research",
                "complexity": "moderate",
                "depends_on": [],
                "expected_output_type": "text"
            },
            {
                "task_id": "ai_trend_analysis_task", 
                "task_description": "Analyze the collected AI technology trend data.",
                "individual_query": "Analyze AI technology trend data to derive patterns, growth rates, and key insights. Include correlations between data and future outlook.",
                "extracted_keywords": ["analysis", "pattern", "insight", "outlook"],
                "domain": "data_analysis",
                "complexity": "high",
                "depends_on": ["ai_trend_research_task"],
                "expected_output_type": "structured_data"
            },
            {
                "task_id": "visual_report_task",
                "task_description": "Generate a visual report based on the analysis results.",
                "individual_query": "Please create a visual report of the AI technology trend analysis. Write a comprehensive report including charts, graphs, and infographics.",
                "extracted_keywords": ["visualization", "report", "chart", "graph"],
                "domain": "visualization",
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
                "selection_reasoning": "Internet search agent selected for AI technology trend research",
                "individual_query": "Please research the latest AI technology trends. Collect information on key technologies, market trends, and major players.",
                "context_integration": "None",
                "confidence": 0.90
            },
            {
                "task_id": "ai_trend_analysis_task",
                "selected_agent": analysis_agent,
                "agent_type": "ANALYSIS",
                "selection_reasoning": "Analysis agent selected for AI trend data analysis",
                "individual_query": "Analyze AI technology trend data to derive patterns, growth rates, and key insights. Include correlations between data and future outlook.",
                "context_integration": "Use previous research result",
                "confidence": 0.85
            },
            {
                "task_id": "visual_report_task",
                "selected_agent": visualization_agent,
                "agent_type": "DATA_VISUALIZATION",
                "selection_reasoning": "Data visualization agent selected for visual report generation",
                "individual_query": "Please create a visual report of the AI technology trend analysis. Write a comprehensive report including charts, graphs, and infographics.",
                "context_integration": "Use all previous results",
                "confidence": 0.95
            }
        ]
        
        return task_breakdown, agent_mappings

    def _create_research_analysis_generation_tasks(self, query: str, available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Multilingual general research + analysis + generation sequential task creation"""
        import re
        
        # Extract main parts from query
        match = re.search(r'(.+?)(?:조사|수집).*?(?:해서|하고).*?(.+?)(?:분석|처리).*?(?:하고).*?(.+?)(?:만들어|생성|작성)', query.lower())
        
        if match:
            research_topic = match.group(1).strip()
            analysis_focus = match.group(2).strip() if len(match.groups()) > 1 else "data"
            output_type = match.group(3).strip() if len(match.groups()) > 2 else "result"
        else:
            # Fallback: simple split
            parts = query.split('하고')
            research_topic = parts[0].strip() if len(parts) > 0 else query
            analysis_focus = parts[1].strip() if len(parts) > 1 else "analysis"
            output_type = parts[2].strip() if len(parts) > 2 else "report"
        
        # Select appropriate agents
        research_agent = self._find_best_agent_for_task(['internet_agent', 'crawler_agent', 'llm_search_agent'], available_agents)
        analysis_agent = self._find_best_agent_for_task(['analysis_agent', 'task_classifier_agent', 'math_agent'], available_agents)
        generation_agent = self._find_best_agent_for_task(['content_formatter_agent', 'document_agent', 'data_visualization_agent'], available_agents)
        
        task_breakdown = [
            {
                "task_id": "research_task",
                "task_description": f"Researching {research_topic}.",
                "individual_query": f"Please research and collect the latest information and data on {research_topic}. Provide comprehensive information from reliable sources.",
                "extracted_keywords": research_topic.split()[:3],
                "domain": "research",
                "complexity": "moderate",
                "depends_on": [],
                "expected_output_type": "text"
            },
            {
                "task_id": "analysis_task",
                "task_description": f"Analyzing collected data from the {analysis_focus} perspective.",
                "individual_query": f"Please analyze the collected information from the {analysis_focus} perspective. Identify key patterns, trends, and insights and provide meaningful conclusions.",
                "extracted_keywords": analysis_focus.split()[:3],
                "domain": "analysis",
                "complexity": "high",
                "depends_on": ["research_task"],
                "expected_output_type": "structured_data"
            },
            {
                "task_id": "generation_task",
                "task_description": f"Generating {output_type} based on analysis results.",
                "individual_query": f"Please generate {output_type} based on the analysis results. Organize it in a user-friendly and practical format.",
                "extracted_keywords": output_type.split()[:3],
                "domain": "generation",
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
                "selection_reasoning": f"Selecting search agent for {research_topic} research",
                "individual_query": f"Please research and collect the latest information and data on {research_topic}. Provide comprehensive information from reliable sources.",
                "context_integration": "None",
                "confidence": 0.80
            },
            {
                "task_id": "analysis_task",
                "selected_agent": analysis_agent,
                "agent_type": "ANALYSIS",
                "selection_reasoning": f"Selecting analysis agent for {analysis_focus} analysis",
                "individual_query": f"Please analyze the collected information from the {analysis_focus} perspective. Identify key patterns, trends, and insights and provide meaningful conclusions.",
                "context_integration": "Use previous research result",
                "confidence": 0.85
            },
            {
                "task_id": "generation_task",
                "selected_agent": generation_agent,
                "agent_type": "CONTENT_FORMATTING",
                "selection_reasoning": f"Selecting content formatter agent for generating {output_type}",
                "individual_query": f"Please generate {output_type} based on the analysis results. Organize it in a user-friendly and practical format.",
                "context_integration": "Use all previous results",
                "confidence": 0.80
            }
        ]
        
        return task_breakdown, agent_mappings

    def _create_general_sequential_tasks(self, parts: List[str], available_agents: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Multilingual general sequential task creation"""
        task_breakdown = []
        agent_mappings = []
        
        for i, part in enumerate(parts[:3]):  # Max 3 tasks
            part = part.strip()
            if not part:
                continue
                
            task_id = f"sequential_task_{i+1}"
            
            # Select agent based on task content
            best_agent = self._select_agent_by_content(part, available_agents, language)
            
            task_breakdown.append({
                "task_id": task_id,
                "task_description": f"Process {part}.",
                "individual_query": f"Please process the following: {part}.",
                "extracted_keywords": part.split()[:3],
                "domain": "general",
                "complexity": "simple",
                "depends_on": [f"sequential_task_{i}"] if i > 0 else [],
                "expected_output_type": "text"
            })
            
            agent_mappings.append({
                "task_id": task_id,
                "selected_agent": best_agent,
                "agent_type": "GENERAL",
                "selection_reasoning": f"Agent selected based on content: {part}",
                "individual_query": f"Please process the following: {part}.",
                "context_integration": "Use previous results" if i > 0 else "None",
                "confidence": 0.7
            })
        
        return task_breakdown, agent_mappings

    def _create_default_single_task(self, query: str, available_agents: List[str], keywords: List[str], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Multilingual default single task creation - includes optimized message"""
        
        # Select a better agent
        best_agent = self._select_agent_by_content(query, available_agents, language)
        
        # Generate agent-specific optimized message
        optimized_message = self._generate_optimized_message_for_agent(query, best_agent, language)
        
        task_breakdown = [{
            "task_id": "emergency_task_1",
            "task_description": f"'{query}' processing",
            "individual_query": optimized_message,
            "extracted_keywords": keywords,
            "domain": "general",
            "complexity": "simple",
            "depends_on": [],
            "expected_output_type": "text"
        }]
        
        agent_mappings = [{
            "task_id": "emergency_task_1",
            "selected_agent": best_agent,
            "agent_type": self._infer_agent_type_from_query(query, language),
            "selection_reasoning": f"Agent selected based on query content: {best_agent}",
            "individual_query": optimized_message,
            "context_integration": "None",
            "confidence": 0.7
        }]
        
        return task_breakdown, agent_mappings

    def _generate_optimized_message_for_agent(self, query: str, agent_id: str, language: str = None) -> str:
        """Multilingual agent-specific optimized message generation"""
        if language is None:
            language = self.language_config.detect_language(query)
        
        query_lower = query.lower()
        intent_keywords = self.language_config.get_intent_keywords(language)
        
        # Language-specific message templates
        if language == 'ko':
            templates = {
                'search': f"{query} - Please search for the latest information and provide detailed findings.",
                'analysis': f"{query} - Please analyze the data and derive insights and patterns.",
                'visualization': f"{query} - Please write a report including visual materials and charts.",
                'content': f"{query} - Please organize and provide in a user-friendly format.",
                'document': f"{query} - Please write a comprehensive result in document format.",
                'crawler': f"{query} - Please collect and organize relevant information from the web.",
                'llm': f"{query} - Please provide a detailed answer based on professional knowledge.",
                'default': f"{query} - Please handle this request professionally."
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
        
        # Optimization by agent type
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
            # Intent-based optimization
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
        """Multilingual inference of agent type from query content"""
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
        """Find the most suitable agent for the task"""
        for preferred in preferred_agents:
            if preferred in available_agents:
                return preferred
        
        # Fallback: first available agent
        return available_agents[0] if available_agents else 'unknown'

    def _is_samsung_domain_query(self, query: str) -> bool:
        """Auto-detect Samsung-related business queries (including typo correction)

        Note: Stock/finance-related queries are excluded from the Samsung domain.
        - "Samsung Electronics stock price" → stock agent
        - "Samsung Electronics Particle issue" → Samsung semiconductor agent
        """

        # Typo correction
        normalized_query = self._normalize_samsung_typos(query)
        query_lower = normalized_query.lower()

        # 🚫 Exclude stock/finance queries (check first)
        stock_finance_keywords = [
            # Stock price
            "종가", "시가", "고가", "저가", "주가", "주식", "stock", "price",
            "closing", "opening", "시세",
            # Trading
            "거래", "trading", "매수", "매도", "buy", "sell", "거래량", "volume",
            # Charts/graphs
            "그래프", "차트", "chart", "graph", "캔들", "candle",
            # Investment
            "투자", "invest", "포트폴리오", "portfolio", "배당", "dividend",
            # Market
            "코스피", "코스닥", "nasdaq", "kospi", "kosdaq",
            # Securities
            "증권", "주권", "상장", "시총", "시가총액"
        ]

        has_stock_finance = any(keyword in query_lower for keyword in stock_finance_keywords)
        if has_stock_finance:
            logger.info(f"📈 Stock/finance query detected - excluded from Samsung domain: {query[:50]}...")
            return False

        # Company/brand keywords
        company_keywords = [
            "삼성", "samsung", "삼성반도체", "samsung semiconductor",
            "삼성전자", "삼성디스플레이", "삼성SDI"
        ]

        # Product/technology keywords
        product_keywords = [
            "ddr4", "ddr5", "gddr6", "lpddr5", "hbm3",
            "메모리", "memory", "반도체", "semiconductor",
            "nand", "dram", "ssd", "플래시", "flash",
            "particle", "파티클", "yield", "수율", "defect", "불량",
            "fab", "팹", "wafer", "웨이퍼", "foundry", "파운드리",
            "cleanroom", "클린룸", "lithography", "리소그래피",
            "etching", "에칭", "deposition", "증착"
        ]

        # Business keywords
        business_keywords = [
            "수율", "yield", "불량", "defect", "품질", "quality",
            "공정", "process", "fab", "공급망", "supply chain",
            "시장점유율", "market share", "매출", "revenue",
            "생산", "production", "제조", "manufacturing"
        ]

        # Analysis keywords (assess business depth)
        analysis_keywords = [
            "분석", "analysis", "추이", "trend", "개선방안", "improvement",
            "최적화", "optimization", "보고서", "report", "예측", "forecast",
            "대시보드", "dashboard", "평가", "assessment"
        ]

        # Samsung + (product OR business) pattern
        has_company = any(keyword in query_lower for keyword in company_keywords)
        has_product = any(keyword in query_lower for keyword in product_keywords)
        has_business = any(keyword in query_lower for keyword in business_keywords)
        has_analysis = any(keyword in query_lower for keyword in analysis_keywords)

        # Pattern matching logic
        if has_company and (has_product or has_business):
            logger.info(f"🏢 Samsung domain detected: company keyword + product/business")
            return True

        # Consider Samsung domain even without Samsung keyword if semiconductor + analysis (domain-specific)
        if has_product and has_business and has_analysis:
            logger.info(f"🏢 Samsung domain detected: semiconductor business analysis pattern")
            return True

        # Route to Samsung even with just semiconductor-specific terms (base domain)
        semiconductor_specific = [
            "particle", "파티클", "yield", "수율", "defect", "불량",
            "fab", "팹", "cleanroom", "클린룸"
        ]
        if any(term in query_lower for term in semiconductor_specific):
            logger.info(f"🏢 Samsung domain detected: semiconductor-specific terminology")
            return True

        return False
    
    def _normalize_samsung_typos(self, query: str) -> str:
        """Normalize Samsung-related typos"""
        typo_corrections = {
            '삼상': '삼성',
            '삼숭': '삼성',
            '삼송': '삼성',
            '삼선': '삼성',
            '삼셩': '삼성',
            '반도채': '반도체',
            '번도체': '반도체',
            '반도처': '반도체',
            '수율': '수율',  # Normalization
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
        🧠 Hybrid agent selection (Knowledge Graph + LLM)

        Phase 1: Knowledge graph analysis (entity extraction, related concepts, past patterns)
        Phase 2: LLM final decision (using graph insights)
        Phase 3: Store feedback (learn successful mappings)
        """
        if language is None:
            language = self.language_config.detect_language(content)

        # Build agent information
        agents_info = self._build_agents_info_for_llm(available_agents)

        if not agents_info:
            logger.warning("⚠️ No available agent info, returning default agent")
            return available_agents[0] if available_agents else 'unknown'

        try:
            # 🧠 Use hybrid selector
            hybrid_selector = get_hybrid_selector()
            selected_agent, metadata = await hybrid_selector.select_agent(
                query=content,
                available_agents=available_agents,
                agents_info=agents_info,
                context={"language": language}
            )

            # Log selection result
            selection_method = metadata.get('selection_method', 'unknown')
            graph_confidence = metadata.get('graph_insights', {}).get('confidence', 0)

            logger.info(
                f"🧠 Hybrid agent selection complete: {selected_agent} "
                f"(method: {selection_method}, graph confidence: {graph_confidence:.1%}, "
                f"query: {content[:50]}...)"
            )

            # Verify the selected agent is valid
            if selected_agent in available_agents:
                return selected_agent
            else:
                # Attempt partial matching
                for agent in available_agents:
                    if selected_agent.lower() in agent.lower() or agent.lower() in selected_agent.lower():
                        logger.info(f"🧠 Agent partial match: {agent} (original: {selected_agent})")
                        return agent

            logger.warning(f"⚠️ Selected agent not in list: {selected_agent}")

        except Exception as e:
            logger.error(f"❌ Hybrid agent selection failed: {e}")

        # Fallback: return first agent
        return available_agents[0] if available_agents else 'unknown'

    async def store_agent_selection_feedback(
        self,
        query: str,
        selected_agent: str,
        success: bool,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        📝 Store agent selection feedback (learning loop)

        Saves successful query-agent mappings to the knowledge graph
        to provide better recommendations for similar future queries.
        """
        try:
            hybrid_selector = get_hybrid_selector()
            return await hybrid_selector.store_feedback(
                query=query,
                selected_agent=selected_agent,
                success=success,
                execution_result=execution_result
            )
        except Exception as e:
            logger.warning(f"⚠️ Feedback storage failed: {e}")
            return False

    def _select_agent_by_content(self, content: str, available_agents: List[str], language: str = None) -> str:
        """
        🔄 Sync wrapper: Call LLM-based agent selection from a synchronous context

        Calls the async _select_agent_by_llm() in a synchronous way.
        """
        if language is None:
            language = self.language_config.detect_language(content)

        # Apply typo normalization
        normalized_content = self._normalize_samsung_typos(content)

        # 🏢 Samsung domain priority check (business logic - Samsung domain requires special handling)
        if self._is_samsung_domain_query(normalized_content):
            samsung_agents = [agent for agent in available_agents if "samsung_gateway" in agent.lower()]
            if samsung_agents:
                logger.info(f"🚀 Samsung Gateway Agent selected: {samsung_agents[0]}")
                return samsung_agents[0]

            samsung_sub_agents = [
                agent for agent in available_agents
                if any(keyword in agent.lower() for keyword in [
                    "samsung_yield", "samsung_market", "samsung_quality",
                    "samsung_supply", "samsung_business"
                ])
            ]
            if samsung_sub_agents:
                logger.info(f"🚀 Samsung Sub-agent selected: {samsung_sub_agents[0]}")
                return samsung_sub_agents[0]

        # 🧠 LLM-based agent selection (async call)
        try:
            import asyncio

            # Check and run event loop
            try:
                loop = asyncio.get_running_loop()
                # If loop already running, execute as new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._select_agent_by_llm(normalized_content, available_agents, language)
                    )
                    selected_agent = future.result(timeout=30)
                    return selected_agent
            except RuntimeError:
                # If no loop, run directly
                selected_agent = asyncio.run(
                    self._select_agent_by_llm(normalized_content, available_agents, language)
                )
                return selected_agent

        except Exception as e:
            logger.error(f"❌ LLM-based agent selection failed: {e}")
            # Fallback: first available agent
            if available_agents:
                logger.info(f"🔍 Fallback agent selected: {available_agents[0]}")
                return available_agents[0]
            return 'unknown'

    def _convert_fallback_to_full_result(self, fallback_result: Dict[str, Any], query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """Convert fallback result to full structure"""
        tasks = fallback_result.get('tasks', [])
        strategy = fallback_result.get('strategy', 'parallel')
        has_dependencies = fallback_result.get('has_dependencies', False)
        
        # Task decomposition
        task_breakdown = []
        agent_mappings = []
        
        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i+1}')
            task_breakdown.append({
                "task_id": task_id,
                "task_description": task.get('description', f'Task {i+1}'),
                "individual_query": task.get('query', query),
                "extracted_keywords": [],
                "domain": "general",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            })
            
            agent_mappings.append({
                "task_id": task_id,
                "selected_agent": task.get('agent', available_agents[0] if available_agents else 'unknown'),
                "agent_type": "GENERAL",
                "selection_reasoning": "LLM-based fallback selection",
                "individual_query": task.get('query', query),
                "context_integration": "None",
                "confidence": 0.7
            })
        
        # Execution plan
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
                "primary_intent": "information_search",
                "domains": ["general"],
                "dependency_detected": has_dependencies,
                "reasoning": "LLM-based fallback analysis"
            },
            "task_breakdown": task_breakdown,
            "agent_mappings": agent_mappings,
            "dependency_analysis": {
                "has_dependencies": has_dependencies,
                "dependency_type": strategy,
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "reasoning": "LLM-based fallback dependency analysis"
            },
            "execution_plan": {
                "strategy": strategy,
                "estimated_time": len(tasks) * 15.0,
                "parallel_groups": parallel_groups,
                "execution_order": execution_order,
                "data_passing_required": has_dependencies,
                "reasoning": f"LLM-based fallback: {strategy} strategy selected - optimal strategy considering task complexity and agent capabilities"
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
        """Generate context-aware query - query using previous results"""
        
        # Generate previous result context
        context = self.result_context.format_context_for_agent(task_id, dependency_map)
        
        if not context:
            return original_query
        
        # Generate context-enriched query using LLM
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
                logger.info(f"🎯 Context-aware query created: {task_id} -> {optimized_query[:50]}...")
                return optimized_query
            
        except Exception as e:
            logger.error(f"Failed to create context-aware query: {e}")
        
        # Fallback: directly prepend context
        return f"{context}\n\n{original_query}"

    def add_agent_result(self, task_id: str, result: Any):
        """Add agent execution result"""
        self.result_context.add_result(task_id, result)
        logger.info(f"📊 Agent result stored: {task_id}")

    def get_execution_context(self) -> AgentResultContext:
        """Return execution context"""
        return self.result_context

    def _build_agents_info_for_llm(self, available_agents: List[str]) -> Dict[str, Any]:
        """Build agent information for LLM (including input_spec/output_spec)"""
        agents_info = {}

        for agent_id in available_agents:
            agent_info = self._find_agent_info(agent_id)
            if agent_info:
                agents_info[agent_id] = agent_info
            else:
                # Generate basic info (including data type spec)
                agent_type = self._infer_type_from_id(agent_id)
                input_spec, output_spec = self._get_default_data_specs(agent_type)
                agents_info[agent_id] = {
                    'agent_type': agent_type,
                    'description': f'{agent_id} agent',
                    'capabilities': [],
                    'tags': [agent_id.replace('_agent', '')],
                    'input_spec': input_spec,
                    'output_spec': output_spec
                }

        return agents_info

    def _get_default_data_specs(self, agent_type: str) -> tuple:
        """Return default data spec based on agent type"""
        # Default input_spec, output_spec by agent type
        specs_mapping = {
            'INTERNET_SEARCH': (
                {'type': 'text', 'format': ['query', 'string'], 'description': 'Search keywords'},
                {'type': 'raw_text', 'format': ['html', 'text'], 'description': 'Web search result (unstructured)'}
            ),
            'CRAWLER': (
                {'type': 'text', 'format': ['query', 'string'], 'description': 'Search keywords'},
                {'type': 'raw_text', 'format': ['html', 'text'], 'description': 'Crawling result (unstructured)'}
            ),
            'ANALYSIS': (
                {'type': 'any', 'format': ['raw_text', 'structured_data'], 'description': 'Data to analyze'},
                {'type': 'structured_data', 'format': ['json', 'array'], 'description': 'Analyzed structured data'}
            ),
            'DATA_VISUALIZATION': (
                {'type': 'structured_data', 'format': ['json', 'array'], 'description': 'Structured data to visualize'},
                {'type': 'visual', 'format': ['html', 'svg', 'chart'], 'description': 'Chart/graph'}
            ),
            'CALCULATOR': (
                {'type': 'numeric', 'format': ['number', 'expression'], 'description': 'Value to calculate'},
                {'type': 'numeric', 'format': ['number'], 'description': 'Calculation result'}
            ),
            'CURRENCY': (
                {'type': 'numeric', 'format': ['number', 'currency'], 'description': 'Amount to exchange'},
                {'type': 'numeric', 'format': ['number'], 'description': 'Exchange result'}
            ),
            'WEATHER': (
                {'type': 'text', 'format': ['location', 'query'], 'description': 'Location information'},
                {'type': 'structured_data', 'format': ['json'], 'description': 'Weather information'}
            ),
            'LLM_SEARCH': (
                {'type': 'text', 'format': ['query', 'question'], 'description': 'Question'},
                {'type': 'text', 'format': ['string', 'explanation'], 'description': 'Answer/explanation'}
            ),
            'RAG_SEARCH': (
                {'type': 'text', 'format': ['query'], 'description': 'Document search query'},
                {'type': 'raw_text', 'format': ['text', 'document'], 'description': 'Retrieved document'}
            ),
            'SCHEDULER': (
                {'type': 'text', 'format': ['command', 'date'], 'description': 'Schedule-related request'},
                {'type': 'structured_data', 'format': ['json', 'calendar'], 'description': 'Schedule information'}
            ),
            'SHOPPING': (
                {'type': 'text', 'format': ['query', 'product'], 'description': 'Product search query'},
                {'type': 'structured_data', 'format': ['json', 'product_list'], 'description': 'Product list'}
            ),
            'WRITER': (
                {'type': 'any', 'format': ['text', 'structured_data'], 'description': 'Content to write'},
                {'type': 'text', 'format': ['document', 'markdown'], 'description': 'Written document'}
            )
        }

        # Default
        default_spec = (
            {'type': 'text', 'format': ['query'], 'description': 'Input'},
            {'type': 'text', 'format': ['string'], 'description': 'Output'}
        )

        return specs_mapping.get(agent_type, default_spec)

    def _find_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Find agent information (including input_spec/output_spec)"""
        for agent_info in self.installed_agents_info:
            if agent_info.get('agent_id') == agent_id:
                agent_data = agent_info.get('agent_data', {})
                metadata = agent_data.get('metadata', {})
                agent_type = metadata.get('agent_type', 'GENERAL')

                # Check input_spec/output_spec in agent data, use default if absent
                input_spec = agent_data.get('input_spec') or metadata.get('input_spec')
                output_spec = agent_data.get('output_spec') or metadata.get('output_spec')

                if not input_spec or not output_spec:
                    input_spec, output_spec = self._get_default_data_specs(agent_type)

                return {
                    'agent_type': agent_type,
                    'description': agent_data.get('description', f'{agent_id} agent'),
                    'capabilities': agent_data.get('capabilities', []),
                    'tags': metadata.get('tags', []),
                    'input_spec': input_spec,
                    'output_spec': output_spec
                }
        return None

    def _infer_type_from_id(self, agent_id: str) -> str:
        """Infer type from agent ID"""
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
        """Clean JSON string"""
        # Remove comments
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove // comments
            if '//' in line:
                line = line[:line.index('//')]
            cleaned_lines.append(line)
        
        # Rejoin
        cleaned_json = '\n'.join(cleaned_lines)
        
        # Attempt to fix incomplete JSON
        cleaned_json = cleaned_json.strip()
        if cleaned_json.endswith(','):
            cleaned_json = cleaned_json[:-1]
        
        return cleaned_json

    def _find_similar_agent(self, target_agent: str, available_agents: List[str],
                             selection_reasoning: str = None, query: str = None) -> Optional[str]:
        """
        🚀 Improved similar agent search - maximize respect for LLM intent

        Args:
            target_agent: Agent ID selected by LLM (may not exist)
            available_agents: Actually available agent list
            selection_reasoning: LLM's selection reason (used for intent analysis)
            query: Original query (used for domain hint extraction)

        Returns:
            Most suitable agent ID or None
        """
        if not target_agent:
            return None

        target_lower = target_agent.lower().strip()

        # === Step 1: Exact match (case-insensitive) ===
        for agent in available_agents:
            if agent.lower() == target_lower:
                logger.info(f"✅ Exact match succeeded: {target_agent} → {agent}")
                return agent

        # === Step 2: Partial match (containment) ===
        for agent in available_agents:
            if target_lower in agent.lower() or agent.lower() in target_lower:
                logger.info(f"✅ Partial match succeeded: {target_agent} → {agent}")
                return agent

        # === Step 3: Keyword-based matching (underscore split) ===
        keywords = [kw for kw in target_lower.replace('_', ' ').replace('-', ' ').split() if len(kw) > 2]
        for agent in available_agents:
            agent_lower = agent.lower()
            # Match if any keyword is contained in agent ID
            if any(keyword in agent_lower for keyword in keywords):
                logger.info(f"✅ Keyword match succeeded: {target_agent} → {agent} (keywords: {keywords})")
                return agent

        # === Step 4: Domain similarity matching (LLM intent-based) ===
        # Extract domain hints from agent name selected by LLM
        domain_mappings = {
            # Travel/tourism domain
            'travel': ['internet_agent', 'llm_search_agent', 'search_agent', 'tour_agent'],
            'tour': ['internet_agent', 'llm_search_agent', 'search_agent', 'travel_agent'],
            'trip': ['internet_agent', 'llm_search_agent', 'search_agent'],
            # Food/restaurant domain
            'food': ['restaurant_agent', 'matzip_agent', 'shopping_agent', 'internet_agent', 'llm_search_agent'],
            'restaurant': ['matzip_agent', 'food_agent', 'shopping_agent', 'internet_agent', 'llm_search_agent'],
            'matzip': ['restaurant_agent', 'food_agent', 'shopping_agent', 'internet_agent'],
            # Search/information domain
            'search': ['internet_agent', 'llm_search_agent', 'search_agent'],
            'info': ['internet_agent', 'llm_search_agent', 'search_agent'],
            'internet': ['llm_search_agent', 'search_agent', 'web_agent'],
            'web': ['internet_agent', 'llm_search_agent', 'search_agent'],
            # Recommendation domain
            'recommend': ['internet_agent', 'llm_search_agent', 'recommendation_agent'],
            'recommendation': ['internet_agent', 'llm_search_agent', 'recommend_agent'],
            # Analysis domain
            'analysis': ['analysis_agent', 'data_agent', 'analytics_agent'],
            'data': ['analysis_agent', 'data_analysis_agent', 'analytics_agent'],
            # Weather domain
            'weather': ['weather_agent', 'internet_agent', 'llm_search_agent'],
            # Shopping domain
            'shopping': ['shopping_agent', 'matzip_agent', 'internet_agent'],
            # Stock/finance domain
            'stock': ['stock_agent', 'finance_agent', 'internet_agent'],
            'finance': ['stock_agent', 'finance_agent', 'analysis_agent'],
        }

        # Extract domain hints from agent selected by LLM
        for domain_key, preferred_agents in domain_mappings.items():
            if domain_key in target_lower:
                # Find available agent among preferred ones
                for preferred in preferred_agents:
                    for agent in available_agents:
                        if preferred in agent.lower() or agent.lower() in preferred:
                            logger.info(f"✅ Domain similarity match: {target_agent} → {agent} (domain: {domain_key})")
                            return agent

        # === Step 5: Extract hints from selection_reasoning (LLM intent analysis) ===
        if selection_reasoning:
            reasoning_lower = selection_reasoning.lower()
            # Find domain keywords in reasoning
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
                                logger.info(f"✅ Reasoning-based match: {target_agent} → {agent} (reasoning keyword: {domain_word})")
                                return agent

        # === Step 6: Extract hints from original query ===
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
                                logger.info(f"✅ Query-based match: {target_agent} → {agent} (query keyword: {domain_word})")
                                return agent

        # === Step 7: Fallback to general agent (last resort) ===
        # General agent priority
        general_fallbacks = ['internet_agent', 'llm_search_agent', 'search_agent', 'general_agent']
        for fallback in general_fallbacks:
            for agent in available_agents:
                if fallback in agent.lower():
                    logger.warning(f"⚠️ General agent fallback: {target_agent} → {agent}")
                    return agent

        logger.warning(f"❌ No similar agent found: {target_agent}")
        return None

    def _create_default_section(self, section_key: str, query: str, available_agents: List[str], language: str) -> Dict[str, Any]:
        """Create a default section"""
        if section_key == 'query_analysis':
            return {
                "original_query": query,
                "complexity": "simple",
                "multi_task": False,
                "task_count": 1,
                "primary_intent": "information_search",
                "domains": ["general"],
                "dependency_detected": False,
                "reasoning": "Default analysis"
            }
        elif section_key == 'task_breakdown':
            return [{
                "task_id": "default_task",
                "task_description": f"'{query}' processing",
                "individual_query": query,
                "extracted_keywords": [],
                "domain": "general",
                "complexity": "simple",
                "depends_on": [],
                "expected_output_type": "text"
            }]
        elif section_key == 'agent_mappings':
            return [{
                "task_id": "default_task",
                "selected_agent": available_agents[0] if available_agents else 'unknown',
                "agent_type": "GENERAL",
                "selection_reasoning": "Default selection",
                "individual_query": query,
                "context_integration": "None",
                "confidence": 0.5
            }]
        elif section_key == 'execution_plan':
            return {
                "strategy": "single_agent",
                "estimated_time": 15.0,
                "parallel_groups": [["default_task"]],
                "execution_order": ["default_task"],
                "data_passing_required": False,
                "reasoning": "single_agent default strategy: most efficient as a single agent can handle this"
            }
        elif section_key == 'dependency_analysis':
            return {
                "has_dependencies": False,
                "dependency_type": "single_agent",
                "parallel_groups": [["default_task"]],
                "execution_order": ["default_task"],
                "reasoning": "Default dependency analysis"
            }
        else:
            return {}


# Global instance
_unified_processor = None

def get_unified_query_processor() -> UnifiedQueryProcessor:
    """Return the global unified query processor instance"""
    global _unified_processor
    if _unified_processor is None:
        _unified_processor = UnifiedQueryProcessor()
    return _unified_processor


logger.info("🚀 Unified query processor loaded successfully!") 