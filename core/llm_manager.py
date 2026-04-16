"""
🧠 Ontology LLM Central Manager

LLM configuration and management optimized for the ontology system
- Dedicated LLM for semantic analysis
- Dedicated LLM for workflow design
- Dedicated LLM for knowledge graph reasoning
- Dedicated LLM for result integration

Supported LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic Claude (Claude-3.5-Sonnet, Claude-3-Haiku)
- Google Gemini (Gemini-1.5-Pro, Gemini-1.5-Flash)
"""

import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from loguru import logger
import asyncio

# LangChain imports for different providers
from langchain_openai import ChatOpenAI

# Optional imports for other providers
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

# Model imports
from .models import LLMProvider, OntologyLLMType, OntologyLLMConfig


class GeminiLLMWrapper:
    """Google Gemini API wrapper class (LangChain compatible)"""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2000, api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=self.api_key)
    
    async def ainvoke(self, messages):
        """Async invocation (LangChain compatible)"""
        try:
            # Message processing
            if isinstance(messages, list):
                # Separate SystemMessage and HumanMessage
                system_instruction = ""
                content = ""
                
                for msg in messages:
                    if hasattr(msg, 'type'):
                        if msg.type == "system":
                            system_instruction = msg.content
                        elif msg.type == "human":
                            content = msg.content
                    else:
                        content += str(msg) + "\n"
            else:
                content = str(messages)
                system_instruction = "You are a helpful AI assistant."
            
            # Gemini API call
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=system_instruction if system_instruction else None
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                config=config,
                contents=content
            )
            
            # Create LangChain compatible response object
            return GeminiResponse(response.text)
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {e}")
    
    def invoke(self, messages):
        """Synchronous invocation (LangChain compatible)"""
        import asyncio
        return asyncio.run(self.ainvoke(messages))


class GeminiResponse:
    """Gemini response wrapper (LangChain compatible)"""
    
    def __init__(self, content: str):
        self.content = content


class OntologyLLMManager:
    """Ontology LLM Central Manager"""
    
    def __init__(self, config_profile: str = None, config_path: str = None):
        """Initialize"""
        self.configs: Dict[OntologyLLMType, OntologyLLMConfig] = {}
        self.instances: Dict[OntologyLLMType, BaseLanguageModel] = {}
        self.prompt_templates: Dict[OntologyLLMType, ChatPromptTemplate] = {}
        self.call_counts: Dict[OntologyLLMType, int] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize config loader
        from .llm_config_loader import LLMConfigLoader
        self.config_loader = LLMConfigLoader(config_path)
        
        # Load configuration
        profile_to_use = config_profile or self.config_loader.get_default_profile()
        self._load_configs_from_file(profile_to_use)
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
        
        logger.info(f"🧠 Ontology LLM manager initialized (profile: {profile_to_use})")
    
    def _load_configs_from_file(self, profile_name: str):
        """Load LLM configuration from config file"""
        try:
            logger.info(f"📁 Loading config profile: {profile_name}")
            
            # Load profile configuration
            configs = self.config_loader.load_profile_configs(profile_name)
            
            if not configs:
                logger.warning(f"Profile '{profile_name}' is empty or not found. Using default configuration.")
                self._load_fallback_configs()
                return
            
            self.configs.update(configs)
            self.current_profile = profile_name
            
            # Summary of loaded configurations
            logger.info(f"✅ {len(configs)} LLM configurations loaded:")
            for llm_type, config in configs.items():
                logger.info(f"   🤖 {llm_type.value}: {config.provider.value}/{config.model}")
                
        except Exception as e:
            logger.error(f"❌ Config file load failed: {e}")
            self._load_fallback_configs()
    
    def _load_fallback_configs(self):
        """Load fallback configuration using the default profile from config file"""
        logger.info("📋 Using fallback configuration - loading from default profile")
        
        try:
            # Load fallback config using the default profile
            fallback_profile = self.config_loader.get_default_profile()
            logger.info(f"📁 Using '{fallback_profile}' profile as fallback")
            
            configs = self.config_loader.load_profile_configs(fallback_profile)
            
            if configs:
                self.configs.update(configs)
                self.current_profile = f"{fallback_profile}_fallback"
                logger.info(f"✅ Fallback configuration loaded: {len(configs)} LLM settings")
            else:
                # If profile load fails, use minimal hardcoded fallback
                self._load_minimal_fallback_configs()
                
        except Exception as e:
            logger.error(f"❌ Fallback configuration load failed: {e}")
            # Final fallback: minimal hardcoded configuration
            self._load_minimal_fallback_configs()

    def _load_minimal_fallback_configs(self):
        """Minimal hardcoded fallback configuration (used when config file also fails)"""
        logger.warning("⚠️ Using minimal hardcoded fallback configuration")
        
        # Default provider priority: Google > OpenAI > Anthropic
        fallback_provider = LLMProvider.GOOGLE
        fallback_model = "gemini-2.5-flash-lite"
        
        # Use OpenAI if Google API key is unavailable
        if not os.getenv("GOOGLE_API_KEY"):
            if os.getenv("OPENAI_API_KEY"):
                fallback_provider = LLMProvider.OPENAI
                fallback_model = "gpt-4.1-mini"
            elif os.getenv("ANTHROPIC_API_KEY"):
                fallback_provider = LLMProvider.ANTHROPIC
                fallback_model = "claude-3.5-sonnet"
        
        logger.info(f"🔧 Final fallback: {fallback_provider.value}/{fallback_model}")
        
        # Apply the same fallback configuration to all LLM types
        fallback_config = OntologyLLMConfig(
            provider=fallback_provider,
            model=fallback_model,
            temperature=0.7,
            description=f"Minimal fallback configuration ({fallback_provider.value})"
        )
        
        fallback_configs = {llm_type: fallback_config for llm_type in OntologyLLMType}
        
        self.configs.update(fallback_configs)
        self.current_profile = "minimal_fallback"
    
    def load_profile(self, profile_name: str):
        """Load a new profile"""
        try:
            logger.info(f"🔄 Switching profile: {profile_name}")
            
            # Clean up existing instances
            self.instances.clear()
            
            # Load new configuration
            self._load_configs_from_file(profile_name)
            
            logger.info(f"✅ Profile '{profile_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Profile load failed: {e}")
            raise e
    
    def get_current_profile(self) -> str:
        """Get current profile name"""
        return getattr(self, 'current_profile', 'unknown')
    
    def get_available_profiles(self) -> List[str]:
        """List of available profiles"""
        return self.config_loader.get_available_profiles()
    
    def get_profile_info(self) -> Dict[str, str]:
        """Information for all profiles"""
        return self.config_loader.list_profiles_info()
    
    def create_custom_profile(self, profile_name: str, description: str, 
                            llm_settings: Dict[str, Dict[str, str]]):
        """Create a custom profile"""
        try:
            self.config_loader.add_custom_profile(profile_name, description, llm_settings)
            logger.info(f"✅ Custom profile created: {profile_name}")
            
        except Exception as e:
            logger.error(f"❌ Profile creation failed: {e}")
            raise e
    
    def _create_llm_instance(self, config: OntologyLLMConfig) -> BaseLanguageModel:
        """Create LLM instance"""
        try:
            if config.provider == LLMProvider.OPENAI:
                return ChatOpenAI(
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    request_timeout=config.request_timeout,
                    max_retries=config.max_retries,
                    streaming=config.streaming,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            
            elif config.provider == LLMProvider.ANTHROPIC:
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("Anthropic provider is not available. Install langchain_anthropic package.")
                return ChatAnthropic(
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    timeout=config.request_timeout,
                    max_retries=config.max_retries,
                    streaming=config.streaming,
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            
            elif config.provider == LLMProvider.GOOGLE:
                if not GOOGLE_AVAILABLE:
                    raise ImportError("Google provider is not available. Install google-genai package.")
                # Google Gemini uses a custom wrapper
                return GeminiLLMWrapper(
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {config.provider}")
                
        except Exception as e:
            logger.error(f"LLM instance creation failed ({config.provider.value}/{config.model}): {e}")
            # Smart fallback: try available providers in order
            return self._create_fallback_llm_instance(config)
    
    def _create_fallback_llm_instance(self, original_config: OntologyLLMConfig) -> BaseLanguageModel:
        """Create smart fallback LLM instance"""
        
        # Fallback priority: try providers from default profile first
        try:
            fallback_profile = self.config_loader.get_default_profile()
            profile_configs = self.config_loader.load_profile_configs(fallback_profile)
            
            if profile_configs:
                # Try with the first configuration from the default profile
                first_config = next(iter(profile_configs.values()))
                logger.warning(f"Fallback attempt: {first_config.provider.value}/{first_config.model}")
                
                if first_config.provider == LLMProvider.GOOGLE:
                    return GeminiLLMWrapper(
                        model=first_config.model,
                        temperature=original_config.temperature,
                        max_tokens=original_config.max_tokens or 2000,
                        api_key=os.getenv("GOOGLE_API_KEY")
                    )
                elif first_config.provider == LLMProvider.OPENAI:
                    return ChatOpenAI(
                        model=first_config.model,
                        temperature=original_config.temperature,
                        max_tokens=original_config.max_tokens or 2000,
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                elif first_config.provider == LLMProvider.ANTHROPIC:
                    return ChatAnthropic(
                        model=first_config.model,
                        temperature=original_config.temperature,
                        max_tokens=original_config.max_tokens or 2000,
                        api_key=os.getenv("ANTHROPIC_API_KEY")
                    )
        except Exception as e:
            logger.error(f"Profile-based fallback failed: {e}")
        
        # Hardcoded fallback: try providers with available API keys in order
        fallback_attempts = [
            (LLMProvider.GOOGLE, "gemini-2.5-flash-lite", "GOOGLE_API_KEY"),
            (LLMProvider.OPENAI, "gpt-4.1-mini", "OPENAI_API_KEY"),
            (LLMProvider.ANTHROPIC, "claude-3.5-sonnet", "ANTHROPIC_API_KEY")
        ]
        
        for provider, model, api_key_name in fallback_attempts:
            if os.getenv(api_key_name):
                try:
                    logger.warning(f"Hardcoded fallback attempt: {provider.value}/{model}")
                    
                    if provider == LLMProvider.GOOGLE:
                        return GeminiLLMWrapper(
                            model=model,
                            temperature=original_config.temperature,
                            max_tokens=original_config.max_tokens or 2000,
                            api_key=os.getenv(api_key_name)
                        )
                    elif provider == LLMProvider.OPENAI:
                        return ChatOpenAI(
                            model=model,
                            temperature=original_config.temperature,
                            max_tokens=original_config.max_tokens or 2000,
                            api_key=os.getenv(api_key_name)
                        )
                    elif provider == LLMProvider.ANTHROPIC:
                        return ChatAnthropic(
                            model=model,
                            temperature=original_config.temperature,
                            max_tokens=original_config.max_tokens or 2000,
                            api_key=os.getenv(api_key_name)
                        )
                        
                except Exception as e:
                    logger.error(f"Fallback attempt failed ({provider.value}): {e}")
                    continue
        
        # Raise exception when all fallbacks fail
        raise RuntimeError("All LLM provider fallbacks failed. Please check your API keys.")
    
    def update_llm_config(self, llm_type: OntologyLLMType, 
                         provider: LLMProvider = None,
                         model: str = None,
                         **kwargs):
        """Update LLM configuration"""
        try:
            config = self.configs[llm_type]
            
            if provider:
                config.provider = provider
            if model:
                config.model = model
            
            # Update other settings
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Remove existing instance so it gets recreated
            if llm_type in self.instances:
                del self.instances[llm_type]
            
            logger.info(f"LLM config updated: {llm_type.value} -> {config.provider.value}/{config.model}")
            
        except Exception as e:
            logger.error(f"LLM config update failed: {e}")
    
    def set_all_provider(self, provider: LLMProvider, model_mapping: Dict[str, str] = None):
        """Set all LLMs to a specific provider"""
        try:
            # Default model mapping per provider
            default_models = {
                LLMProvider.OPENAI: {
                    "high_performance": "gpt-4.1",
                    "fast": "gpt-4.1-mini",
                    "creative": "gpt-4.1-mini"
                },
                LLMProvider.ANTHROPIC: {
                    "high_performance": "claude-4-sonnet-20240528",
                    "fast": "claude-4-sonnet-20240528",
                    "creative": "claude-4-sonnet-20240528"
                },
                LLMProvider.GOOGLE: {
                    "high_performance": "gemini-2.5-flash-lite",
                    "fast": "gemini-2.5-flash-lite",
                    "creative": "gemini-2.5-flash-lite"
                }
            }
            
            models = model_mapping or default_models.get(provider, {})
            
            for llm_type in OntologyLLMType:
                config = self.configs[llm_type]
                
                # Select model based on performance requirements
                if llm_type in [OntologyLLMType.QUERY_PROCESSOR, OntologyLLMType.PERFORMANCE_OPTIMIZER]:
                    model = models.get("fast", list(models.values())[0] if models else config.model)
                elif llm_type == OntologyLLMType.CREATIVE_REASONER:
                    model = models.get("creative", list(models.values())[0] if models else config.model)
                else:
                    model = models.get("high_performance", list(models.values())[0] if models else config.model)
                
                self.update_llm_config(llm_type, provider=provider, model=model)
            
            logger.info(f"All LLMs set to {provider.value}")
            
        except Exception as e:
            logger.error(f"Bulk LLM provider configuration failed: {e}")
    
    def get_available_models(self, provider: LLMProvider) -> List[str]:
        """List of available models per provider"""
        models = {
            LLMProvider.OPENAI: [
                "gpt-4.1-mini", "gpt-4.1-mini", "gpt-4.1-mini", "gpt-4.1-mini", "gpt-4.1-mini"
            ],
            LLMProvider.ANTHROPIC: [
                "claude-3.5-sonnet", "claude-3.5-sonnet", 
                "claude-3.5-sonnet", "claude-3.5-sonnet", "claude-3.5-sonnet"
            ],
            LLMProvider.GOOGLE: [
                "gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash-lite"
            ]
        }
        
        return models.get(provider, [])
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Check status per provider"""
        status = {}
        
        for provider in LLMProvider:
            try:
                # Check API key
                if provider == LLMProvider.OPENAI:
                    api_key = os.getenv("OPENAI_API_KEY")
                    package_available = True
                elif provider == LLMProvider.ANTHROPIC:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    package_available = ANTHROPIC_AVAILABLE
                elif provider == LLMProvider.GOOGLE:
                    api_key = os.getenv("GOOGLE_API_KEY")
                    package_available = GOOGLE_AVAILABLE
                else:
                    api_key = None
                    package_available = False
                
                status[provider.value] = {
                    "available": bool(api_key) and package_available,
                    "api_key_configured": bool(api_key),
                    "package_available": package_available,
                    "models": self.get_available_models(provider) if package_available else []
                }
                
            except Exception as e:
                status[provider.value] = {
                    "available": False,
                    "error": str(e),
                    "models": []
                }
        
        return status
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates"""
        
        # Semantic analysis prompt - specialized for complex queries
        self.prompt_templates[OntologyLLMType.SEMANTIC_ANALYZER] = ChatPromptTemplate.from_messages([
            ("system", """당신은 온톨로지 시스템의 복합 쿼리 분석 전문가입니다.
사용자의 쿼리를 분석하여 개별 작업들을 정확히 식별하고 분리해주세요.

🔍 **복합 쿼리 분석 핵심 원칙**:
1. **작업 단위 분리**: 연결어("그리고", "도", "또한", ",")로 연결된 각 부분을 독립적 작업으로 인식
2. **개별 작업 식별**: 각 작업이 서로 다른 도메인/에이전트를 필요로 하는지 확인
3. **병렬 처리 우선**: 서로 독립적인 작업들은 병렬 실행 가능으로 분류
4. **에이전트 매칭**: 각 작업에 가장 적합한 전문 에이전트 식별

📊 **작업 유형별 분류**:
- **날씨 작업**: "날씨", "기온", "예보" → weather_agent
- **환율/금융 작업**: "환율", "달러", "주가" → currency_agent 또는 internet_agent
- **계산 작업**: "계산", "수학", "더하기" → calculator_agent
- **검색 작업**: "알려줘", "정보", "찾아줘" → internet_agent
- **분석 작업**: "분석", "비교", "평가" → analysis_agent

⚡ **복합 쿼리 예시 분석**:
"오늘 날씨를 확인하고, 환율 정보도 알려주고, 간단한 계산도 해줘"
→ 작업 1: "오늘 날씨를 확인" (weather_agent)
→ 작업 2: "환율 정보도 알려주고" (currency_agent)  
→ 작업 3: "간단한 계산도 해줘" (calculator_agent)
→ 실행 전략: parallel (3개 독립 작업)

🎯 **중요한 분석 규칙**:
- 각 연결어 앞뒤로 별개의 작업이 있는지 확인
- 작업별로 필요한 전문 지식이 다른지 판단
- 작업 간 데이터 의존성이 있는지 확인 (없으면 병렬 처리)
- 단순한 나열이 아닌 실제 독립적인 작업인지 검증

정확한 작업 분리와 에이전트 매칭을 위해 structured_query에 상세 정보를 포함해주세요."""),
            ("human", """다음 쿼리를 분석해주세요:

쿼리: "{query}"

다음 JSON 형식으로 응답해주세요:
{{
    "multi_task": true/false,
    "task_count": 개별_작업_수,
    "complexity": "simple/moderate/complex",
    "individual_tasks": [
        {{
            "task_id": "task_1",
            "task_description": "구체적인 작업 설명",
            "domain": "weather/finance/calculation/search/analysis",
            "required_agent": "적합한_에이전트명",
            "keywords": ["관련_키워드들"],
            "independent": true/false
        }}
    ],
    "execution_strategy": "single/parallel/sequential/hybrid",
    "dependencies": {{"task_2": ["task_1"]}},
    "reasoning": "작업 분리와 분석 근거",
    "intent": "정보_검색_또는_계산_또는_분석",
    "entities": ["추출된_엔티티들"],
    "concepts": ["관련_개념들"],
    "relations": ["개념_간_관계들"],
    "constraints": {{
        "time_sensitive": true/false,
        "accuracy_required": "high/medium/low",
        "real_time_needed": true/false
    }},
    "structured_query": {{
        "primary_domain": "주요_도메인",
        "required_agents": ["필요한_에이전트_목록"],
        "query_parts": ["분리된_쿼리_부분들"],
        "task_types": ["작업_유형들"]
    }}
}}""")
        ])
        
        # Workflow design prompt
        self.prompt_templates[OntologyLLMType.WORKFLOW_DESIGNER] = ChatPromptTemplate.from_messages([
            ("system", """당신은 복합 쿼리 전용 워크플로우 설계 전문가입니다.
의미론적 분석 결과를 바탕으로 각 개별 작업에 최적의 에이전트를 매칭하고 효율적인 실행 워크플로우를 설계해주세요.

🎯 **워크플로우 설계 핵심 목표**:
1. **개별 작업 워크플로우**: 복합 쿼리의 각 부분을 독립적 실행 단계로 설계
2. **전문 에이전트 할당**: 각 작업 도메인에 가장 적합한 에이전트 매칭
3. **병렬 처리 우선**: 독립적 작업들은 동시 실행으로 성능 최적화
4. **사용자 경험**: 개별 결과들을 통합하여 완성도 높은 응답 제공

⚙️ **정확한 에이전트 매칭 규칙**:
- **날씨/기상**: "날씨", "기온", "예보", "미세먼지" → internet_agent
- **환율/금융**: "환율", "달러", "원", "주가", "금시세" → internet_agent
- **계산/수학**: "계산", "수학", "더하기", "곱셈", "나누기" → calculator_agent
- **일반 검색**: "정보", "알려줘", "찾아줘", "검색" → internet_agent
- **데이터 분석**: "분석", "비교", "평가", "추이" → analysis_agent
- **차트/시각화**: "차트", "그래프", "시각화" → chart_agent

🔄 **실행 전략 우선순위**:
1. **parallel**: 각 작업이 완전 독립적 (⭐ 최우선 선택)
2. **sequential**: 한 작업 결과가 다른 작업 입력으로 필요
3. **hybrid**: 일부 병렬 + 일부 순차
4. **single_agent**: 모든 작업을 하나의 에이전트가 처리 가능

💡 **복합 쿼리 처리 예시**:
"오늘 날씨를 확인하고, 환율 정보도 알려주고, 간단한 계산도 해줘"
→ 워크플로우 설계:
  Step 1: 날씨 확인 (internet_agent, "오늘 날씨 정보를 알려주세요")
  Step 2: 환율 정보 (internet_agent, "현재 달러 환율을 알려주세요")  
  Step 3: 계산 수행 (calculator_agent, "간단한 계산을 해주세요")
→ 실행 전략: parallel (3개 독립 작업 동시 실행)

각 단계별로 구체적인 에이전트 할당과 최적화된 메시지를 제공해주세요."""),
            ("human", """다음 의미론적 분석 결과를 바탕으로 워크플로우를 JSON 형식으로 설계해주세요:

{query_info}

사용 가능한 에이전트: {available_agents}

⚠️ **중요**: 반드시 다음 JSON 형식으로만 응답해주세요:

{{
    "workflow_id": "unique_id",
    "execution_strategy": "parallel/sequential/hybrid/single_agent",
    "total_steps": 총_단계_수,
    "estimated_time": 예상_실행_시간_초,
    "steps": [
        {{
            "step_id": "step_1",
            "task_description": "구체적인 작업 설명",
            "assigned_agent": "담당_에이전트명",
            "agent_message": "해당 에이전트에게 전달할 최적화된 메시지",
            "expected_output": "예상 출력 형태",
            "dependencies": [],
            "parallel_group": "group_1",
            "priority": "high/medium/low",
            "timeout": 제한시간_초
        }}
    ],
    "parallel_groups": {{
        "group_1": ["step_1", "step_2", "step_3"]
    }},
    "integration_plan": {{
        "method": "결과_통합_방법",
        "final_format": "최종_응답_형태"
    }},
    "reasoning": "워크플로우 설계 근거"
}}

JSON 형식 외에는 다른 텍스트를 포함하지 마세요.""")
        ])
        
        # Knowledge reasoning prompt
        self.prompt_templates[OntologyLLMType.KNOWLEDGE_REASONER] = ChatPromptTemplate.from_messages([
            ("system", """당신은 온톨로지 시스템의 지식 추론 전문가입니다.
주어진 정보를 바탕으로 깊이 있는 추론을 수행해주세요.

추론 영역:
1. 개념 간의 숨겨진 관계 발견
2. 지식 패턴 인식
3. 온톨로지 확장 가능성 제시
4. 논리적 일관성 검증

논리적이고 근거 있는 추론을 JSON 형태로 제공해주세요."""),
            ("human", "{reasoning_context}")
        ])
        
        # Result integration prompt
        self.prompt_templates[OntologyLLMType.RESULT_INTEGRATOR] = ChatPromptTemplate.from_messages([
            ("system", """당신은 온톨로지 시스템의 결과 통합 전문가입니다.
여러 에이전트의 실행 결과를 일관성 있고 사용자 친화적으로 통합해주세요.

통합 원칙:
1. 정보의 일관성 보장
2. 중복 정보 제거
3. 사용자 친화적 표현
4. 핵심 인사이트 강조
5. 시각적 요소 활용 제안

통합되고 완성도 높은 결과를 자연스러운 한국어로 제공해주세요."""),
            ("human", "원본 쿼리: {original_query}\n\n에이전트 실행 결과들:\n{execution_results}")
        ])
        
        logger.info("📝 Ontology prompt templates initialized")
    
    def get_llm(self, llm_type: OntologyLLMType, force_new: bool = False) -> BaseLanguageModel:
        """Get LLM instance (singleton pattern)"""
        try:
            if force_new or llm_type not in self.instances:
                config = self.configs[llm_type]
                
                # Create LLM instance
                llm_instance = self._create_llm_instance(config)
                
                self.instances[llm_type] = llm_instance
                self.call_counts[llm_type] = 0
                logger.debug(f"🔨 {llm_type.value} LLM instance created: {config.provider.value}/{config.model}")
            
            return self.instances[llm_type]
                 
        except Exception as e:
            logger.error(f"LLM instance lookup failed ({llm_type.value}): {str(e)}")
            # Smart fallback: use default configuration
            fallback_config = OntologyLLMConfig(
                provider=LLMProvider.GOOGLE,  # Default value
                model="gemini-2.5-flash-lite",
                temperature=0.7
            )
            return self._create_fallback_llm_instance(fallback_config)
    
    async def invoke_llm(self, llm_type: OntologyLLMType, 
                        messages: Union[str, List, Dict], 
                        **kwargs) -> str:
        """Invoke LLM with performance tracking"""
        start_time = datetime.now()
        
        try:
            llm = self.get_llm(llm_type)
            
            # Normalize message format
            if isinstance(messages, str):
                # Use prompt template
                if llm_type in self.prompt_templates:
                    prompt = self.prompt_templates[llm_type]
                    formatted_messages = prompt.format_messages(query=messages, **kwargs)
                else:
                    formatted_messages = [HumanMessage(content=messages)]
            elif isinstance(messages, dict):
                # Use prompt template with parameters
                if llm_type in self.prompt_templates:
                    prompt = self.prompt_templates[llm_type]
                    formatted_messages = prompt.format_messages(**messages)
                else:
                    formatted_messages = [HumanMessage(content=str(messages))]
            else:
                formatted_messages = messages
            
            # Invoke LLM
            response = await llm.ainvoke(formatted_messages)
            
            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(llm_type, execution_time, True)
            
            return response.content
             
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(llm_type, execution_time, False)
            logger.error(f"LLM invocation failed ({llm_type.value}): {str(e)}")
            raise e
    
    def _update_performance_metrics(self, llm_type: OntologyLLMType, 
                                   execution_time: float, success: bool):
        """Update performance metrics"""
        type_name = llm_type.value
        
        if type_name not in self.performance_metrics:
            self.performance_metrics[type_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "last_call": None
            }
        
        metrics = self.performance_metrics[type_name]
        metrics["total_calls"] += 1
        metrics["total_time"] += execution_time
        metrics["average_time"] = metrics["total_time"] / metrics["total_calls"]
        metrics["last_call"] = datetime.now().isoformat()
        
        if success:
            metrics["successful_calls"] += 1
        else:
            metrics["failed_calls"] += 1
        
        # Update call count
        self.call_counts[llm_type] = self.call_counts.get(llm_type, 0) + 1
    
    def get_config(self, llm_type: OntologyLLMType) -> OntologyLLMConfig:
        """Get LLM configuration"""
        return self.configs.get(llm_type, self.configs[OntologyLLMType.SEMANTIC_ANALYZER])
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return {
            "performance_metrics": self.performance_metrics,
            "call_counts": {k.value: v for k, v in self.call_counts.items()},
            "active_instances": len(self.instances),
            "total_llm_types": len(self.configs),
            "provider_status": self.get_provider_status(),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_current_configuration_summary(self) -> Dict[str, Any]:
        """Current configuration summary"""
        summary = {}
        for llm_type, config in self.configs.items():
            summary[llm_type.value] = {
                "provider": config.provider.value,
                "model": config.model,
                "temperature": config.temperature,
                "description": config.description,
                "specialization": config.specialization
            }
        return summary


# Global ontology LLM manager instance
_global_ontology_llm_manager: Optional[OntologyLLMManager] = None


def get_ontology_llm_manager() -> OntologyLLMManager:
    """Get global ontology LLM manager instance (singleton)"""
    global _global_ontology_llm_manager
    if _global_ontology_llm_manager is None:
        _global_ontology_llm_manager = OntologyLLMManager()
    return _global_ontology_llm_manager


# Convenience functions - individual LLM calls

def get_semantic_analyzer() -> BaseLanguageModel:
    """Get dedicated semantic analysis LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.SEMANTIC_ANALYZER)

def get_workflow_designer() -> BaseLanguageModel:
    """Get dedicated workflow design LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.WORKFLOW_DESIGNER)

def get_knowledge_reasoner() -> BaseLanguageModel:
    """Get dedicated knowledge reasoning LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.KNOWLEDGE_REASONER)

def get_result_integrator() -> BaseLanguageModel:
    """Get dedicated result integration LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.RESULT_INTEGRATOR)

def get_query_processor() -> BaseLanguageModel:
    """Get dedicated query processing LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.QUERY_PROCESSOR)

def get_graph_builder() -> BaseLanguageModel:
    """Get dedicated graph building LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.GRAPH_BUILDER)

def get_performance_optimizer() -> BaseLanguageModel:
    """Get dedicated performance optimization LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.PERFORMANCE_OPTIMIZER)

def get_creative_reasoner() -> BaseLanguageModel:
    """Get dedicated creative reasoning LLM"""
    return get_ontology_llm_manager().get_llm(OntologyLLMType.CREATIVE_REASONER)


# Context-specific LLM selection functions

async def analyze_semantic_query(query: str, **kwargs) -> str:
    """Semantic query analysis"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(OntologyLLMType.SEMANTIC_ANALYZER, query, **kwargs)

async def design_workflow(query_info: str, available_agents: List[str], **kwargs) -> str:
    """Workflow design"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(
        OntologyLLMType.WORKFLOW_DESIGNER, 
        {"query_info": query_info, "available_agents": available_agents, **kwargs}
    )

async def reason_knowledge(reasoning_context: str, **kwargs) -> str:
    """Knowledge reasoning"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(
        OntologyLLMType.KNOWLEDGE_REASONER, 
        {"reasoning_context": reasoning_context, **kwargs}
    )

async def integrate_results(original_query: str, execution_results: List[Dict], **kwargs) -> str:
    """Result integration"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(
        OntologyLLMType.RESULT_INTEGRATOR, 
        {"original_query": original_query, "execution_results": execution_results, **kwargs}
    )

async def process_query_fast(query: str, **kwargs) -> str:
    """Fast query processing"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(OntologyLLMType.QUERY_PROCESSOR, query, **kwargs)

async def build_graph_structure(graph_context: str, **kwargs) -> str:
    """Build graph structure"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(OntologyLLMType.GRAPH_BUILDER, graph_context, **kwargs)

async def optimize_performance(performance_context: str, **kwargs) -> str:
    """Performance optimization"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(OntologyLLMType.PERFORMANCE_OPTIMIZER, performance_context, **kwargs)

async def creative_reasoning(creative_context: str, **kwargs) -> str:
    """Creative reasoning"""
    manager = get_ontology_llm_manager()
    return await manager.invoke_llm(OntologyLLMType.CREATIVE_REASONER, creative_context, **kwargs)


# Configuration management convenience functions

def configure_all_openai():
    """Set all LLMs to OpenAI"""
    manager = get_ontology_llm_manager()
    try:
        manager.load_profile("all_openai")
        logger.info("✅ All LLMs have been set to OpenAI")
    except Exception as e:
        logger.error(f"❌ OpenAI configuration failed: {e}")
        raise

def configure_all_claude():
    """Set all LLMs to Claude"""
    manager = get_ontology_llm_manager()
    try:
        manager.load_profile("all_claude")
        logger.info("✅ All LLMs have been set to Claude")
    except Exception as e:
        logger.error(f"❌ Claude configuration failed: {e}")
        raise

def configure_all_gemini():
    """Set all LLMs to Gemini"""
    manager = get_ontology_llm_manager()
    try:
        manager.load_profile("all_gemini")
        logger.info("✅ All LLMs have been set to Gemini")
    except Exception as e:
        logger.error(f"❌ Gemini configuration failed: {e}")
        raise

def configure_mixed_optimal():
    """Mixed configuration for optimal performance"""
    manager = get_ontology_llm_manager() 
    try:
        manager.load_profile("mixed_optimal")
        logger.info("✅ Mixed optimization configuration applied")
    except Exception as e:
        logger.error(f"❌ Mixed optimization configuration failed: {e}")
        raise

def configure_budget_friendly():
    """Cost-efficient configuration"""
    manager = get_ontology_llm_manager()
    try:
        manager.load_profile("budget_friendly")
        logger.info("✅ Cost-efficient configuration applied")
    except Exception as e:
        logger.error(f"❌ Cost-efficient configuration failed: {e}")
        raise

def quick_configure(provider: str):
    """Quick configuration change (provider: 'openai', 'claude', 'gemini', 'mixed', 'budget')"""
    provider_map = {
        'openai': configure_all_openai,
        'claude': configure_all_claude,
        'gemini': configure_all_gemini,
        'mixed': configure_mixed_optimal,
        'budget': configure_budget_friendly
    }
    
    if provider.lower() in provider_map:
        provider_map[provider.lower()]()
        logger.info(f"🚀 Quickly switched to '{provider}' configuration!")
    else:
        available = list(provider_map.keys())
        raise ValueError(f"Unsupported provider: {provider}. Available: {available}")

def auto_configure():
    """Auto-configure based on available API keys"""
    manager = get_ontology_llm_manager()
    
    # Check API key status
    api_keys = {
        'google': bool(os.getenv("GOOGLE_API_KEY")),
        'openai': bool(os.getenv("OPENAI_API_KEY")),
        'anthropic': bool(os.getenv("ANTHROPIC_API_KEY"))
    }
    
    available_providers = [k for k, v in api_keys.items() if v]
    logger.info(f"🔍 Available API keys: {available_providers}")
    
    if not available_providers:
        raise RuntimeError("No available API keys. Please set the environment variables.")
    
    # Priority: Google > OpenAI > Anthropic
    if 'google' in available_providers:
        configure_all_gemini()
        logger.info("🎯 Google API key found - auto-configured to Gemini")
    elif 'openai' in available_providers:
        configure_all_openai()
        logger.info("🎯 OpenAI API key found - auto-configured to OpenAI")
    elif 'anthropic' in available_providers:
        configure_all_claude()
        logger.info("🎯 Anthropic API key found - auto-configured to Claude")


# Utility functions

def get_llm_performance_report() -> Dict[str, Any]:
    """Get LLM performance report"""
    return get_ontology_llm_manager().get_performance_report()

def get_provider_status() -> Dict[str, Any]:
    """Get provider status"""
    return get_ontology_llm_manager().get_provider_status()

def get_current_configuration() -> Dict[str, Any]:
    """Get current configuration"""
    return get_ontology_llm_manager().get_current_configuration_summary()

def get_available_models(provider: str) -> List[str]:
    """List of available models per provider"""
    try:
        provider_enum = LLMProvider(provider.lower())
        return get_ontology_llm_manager().get_available_models(provider_enum)
    except ValueError:
        return []

def cleanup_llm_instances():
    """Clean up unused LLM instances"""
    manager = get_ontology_llm_manager()
    manager.instances.clear()
    logger.info("🧹 LLM instances cleaned up")

def get_all_llm_configs() -> Dict[str, Dict[str, Any]]:
    """Get all LLM configurations"""
    manager = get_ontology_llm_manager()
    result = {}
    for llm_type, config in manager.configs.items():
        result[llm_type.value] = {
            "provider": config.provider.value,
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "description": config.description,
            "use_case": config.use_case,
            "specialization": config.specialization,
            "reasoning_depth": config.reasoning_depth,
            "creativity_level": config.creativity_level,
            "precision_level": config.precision_level
        }
    return result


# Config file management functions

def get_available_profiles() -> List[str]:
    """List of available profiles"""
    return get_ontology_llm_manager().get_available_profiles()

def get_profile_info() -> Dict[str, str]:
    """Information for all profiles"""
    return get_ontology_llm_manager().get_profile_info()

def get_current_profile() -> str:
    """Get current profile name"""
    return get_ontology_llm_manager().get_current_profile()

def load_profile(profile_name: str):
    """Load profile"""
    get_ontology_llm_manager().load_profile(profile_name)

def create_custom_profile(profile_name: str, description: str, 
                        llm_settings: Dict[str, Dict[str, str]]):
    """Create a custom profile
    
    Args:
        profile_name: Profile name
        description: Profile description
        llm_settings: LLM settings (e.g. {"semantic_analyzer": {"provider": "anthropic", "model_tier": "high_performance"}})
    """
    get_ontology_llm_manager().create_custom_profile(profile_name, description, llm_settings)

def get_config_summary() -> Dict[str, Any]:
    """Configuration file summary info"""
    manager = get_ontology_llm_manager()
    return manager.config_loader.get_config_summary()

def reload_config():
    """Reload configuration file"""
    from .llm_config_loader import reload_llm_config
    reload_llm_config()
    logger.info("🔄 Config file reloaded successfully")


logger.info("🧠 Ontology LLM Central Manager loaded!") 