"""
Agent Registry

Central registry for all available agents with their metadata,
capabilities, and I/O schemas. Used by Query Planner to select
appropriate agents for task execution.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AgentSchema, AgentRegistryEntry
from .exceptions import AgentNotFoundError

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for managing agent metadata.

    Features:
    - Default agent definitions with detailed metadata
    - Dynamic agent registration/deregistration
    - Query by capability, tags, or agent ID
    - Build prompt context for Query Planner

    Example:
        registry = AgentRegistry()
        registry.initialize()

        # Get all agents
        agents = registry.get_all_agents()

        # Get agent by ID
        agent = registry.get_agent("internet_agent")

        # Get agents by capability
        search_agents = registry.get_agents_by_capability("web_search")

        # Build context for LLM prompt
        context = registry.build_prompt_context()
    """

    # Default agent definitions with comprehensive metadata
    DEFAULT_AGENTS = [
        AgentRegistryEntry(
            agent_id="internet_agent",
            name="인터넷 검색 에이전트",
            description=(
                "웹 검색을 수행하여 실시간 정보를 수집합니다. "
                "주가, 환율, 뉴스 등 최신 데이터를 인터넷에서 검색합니다. "
                "날씨 정보는 weather_agent를 사용하세요."
            ),
            capabilities=[
                "web_search",
                "real_time_data",
                "news_search",
                "price_lookup",
            ],
            tags=["검색", "실시간", "인터넷", "데이터수집"],
            schema=AgentSchema(
                input_type="query",
                output_type="text",
            ),
            display_name="Internet Search",
            display_name_ko="인터넷 검색",
            icon="🌐",
            color="#3b82f6",
            priority=5,
        ),
        AgentRegistryEntry(
            agent_id="weather_agent",
            name="날씨 정보 에이전트",
            description=(
                "실시간 날씨 정보를 제공하는 전문 에이전트입니다. "
                "현재 날씨, 기온, 습도, 미세먼지, 주간 예보를 조회합니다. "
                "날씨, 기온, 온도, 비, 눈, 미세먼지 관련 질문에 사용하세요."
            ),
            capabilities=[
                "weather_forecast",
                "current_weather",
                "temperature",
                "humidity",
                "air_quality",
                "weekly_forecast",
            ],
            tags=["날씨", "기온", "온도", "미세먼지", "예보", "weather"],
            schema=AgentSchema(
                input_type="query",
                output_type="json",
            ),
            display_name="Weather Info",
            display_name_ko="날씨 정보",
            icon="🌤️",
            color="#0ea5e9",
            priority=50,  # High priority for weather queries
        ),
        AgentRegistryEntry(
            agent_id="analysis_agent",
            name="데이터 분석 에이전트",
            description=(
                "수집된 데이터를 분석하고 구조화합니다. "
                "숫자 데이터 추출, 트렌드 분석, 통계 계산을 수행합니다. "
                "분석 결과를 JSON 형태로 반환합니다."
            ),
            capabilities=[
                "data_analysis",
                "number_extraction",
                "trend_analysis",
                "statistical_calculation",
                "data_structuring",
            ],
            tags=["분석", "데이터", "통계", "구조화"],
            schema=AgentSchema(
                input_type="structured_data",
                output_type="json",
            ),
            display_name="Data Analysis",
            display_name_ko="데이터 분석",
            icon="📊",
            color="#8b5cf6",
            priority=10,
        ),
        AgentRegistryEntry(
            agent_id="data_visualization_agent",
            name="데이터 시각화 에이전트",
            description=(
                "분석된 데이터를 차트, 그래프, 표 등으로 시각화합니다. "
                "라인 차트, 바 차트, 파이 차트, 테이블 등을 생성합니다. "
                "HTML/SVG 형태의 시각화 결과를 반환합니다."
            ),
            capabilities=[
                "chart_generation",
                "graph_creation",
                "table_creation",
                "svg_generation",
                "data_visualization",
            ],
            tags=["시각화", "차트", "그래프", "SVG"],
            schema=AgentSchema(
                input_type="json",
                output_type="html",
            ),
            display_name="Data Visualization",
            display_name_ko="데이터 시각화",
            icon="📈",
            color="#10b981",
            priority=15,
        ),
        AgentRegistryEntry(
            agent_id="llm_search_agent",
            name="LLM 검색 에이전트",
            description=(
                "대규모 언어 모델을 활용한 지능형 검색 및 답변 생성. "
                "복잡한 질문에 대한 종합적인 답변, 설명, 요약을 제공합니다. "
                "일반 지식, 개념 설명, 비교 분석에 적합합니다."
            ),
            capabilities=[
                "question_answering",
                "explanation",
                "summarization",
                "concept_explanation",
                "comparison_analysis",
            ],
            tags=["LLM", "질문답변", "설명", "요약"],
            schema=AgentSchema(
                input_type="query",
                output_type="text",
            ),
            display_name="LLM Search",
            display_name_ko="LLM 검색",
            icon="🔍",
            color="#f59e0b",
            priority=8,
        ),
        AgentRegistryEntry(
            agent_id="samsung_gateway_agent",
            name="삼성 게이트웨이 에이전트",
            description=(
                "삼성전자 내부 데이터 및 시스템에 접근하는 전문 에이전트. "
                "삼성 반도체, NAND, 반도체 공정, 삼성 내부 데이터 관련 쿼리를 처리합니다. "
                "Samsung, 반도체, FAB, 수율, EUV 관련 질문에 사용됩니다."
            ),
            capabilities=[
                "samsung_data_access",
                "semiconductor_analysis",
                "internal_data_query",
                "fab_data",
                "yield_analysis",
            ],
            tags=["삼성", "반도체", "내부데이터", "Samsung", "NAND"],
            schema=AgentSchema(
                input_type="query",
                output_type="json",
            ),
            display_name="Samsung Gateway",
            display_name_ko="삼성 게이트웨이",
            icon="📱",
            color="#1e40af",
            priority=100,  # High priority for Samsung queries
        ),
        AgentRegistryEntry(
            agent_id="shopping_agent",
            name="쇼핑 검색 에이전트",
            description=(
                "온라인 쇼핑몰에서 상품을 검색하고 가격을 비교합니다. "
                "상품 검색, 가격 비교, 최저가 찾기, 쇼핑 추천에 사용됩니다."
            ),
            capabilities=[
                "product_search",
                "price_comparison",
                "shopping_recommendation",
                "deal_finder",
            ],
            tags=["쇼핑", "가격비교", "상품검색", "최저가"],
            schema=AgentSchema(
                input_type="query",
                output_type="json",
            ),
            display_name="Shopping Search",
            display_name_ko="쇼핑 검색",
            icon="🛒",
            color="#ec4899",
            priority=7,
        ),
        AgentRegistryEntry(
            agent_id="scheduler_agent",
            name="일정 관리 에이전트",
            description=(
                "일정 및 스케줄을 관리하는 전문 에이전트입니다. "
                "일정 조회, 일정 추가, 일정 수정, 캘린더 관리를 수행합니다. "
                "일정, 스케줄, 약속, 캘린더 관련 질문에 사용하세요."
            ),
            capabilities=[
                "schedule_management",
                "calendar_access",
                "event_creation",
                "event_query",
                "reminder_setting",
            ],
            tags=["일정", "스케줄", "캘린더", "약속", "schedule"],
            schema=AgentSchema(
                input_type="query",
                output_type="json",
            ),
            display_name="Scheduler",
            display_name_ko="일정 관리",
            icon="📅",
            color="#f97316",
            priority=50,  # High priority for schedule queries
        ),
        AgentRegistryEntry(
            agent_id="calculator_agent",
            name="계산기 에이전트",
            description=(
                "수학 계산 및 단위 변환을 수행하는 전문 에이전트입니다. "
                "사칙연산, 퍼센트 계산, 단위 변환, 환율 계산을 처리합니다. "
                "계산, 더하기, 빼기, 곱하기, 나누기 관련 질문에 사용하세요."
            ),
            capabilities=[
                "math_calculation",
                "unit_conversion",
                "currency_conversion",
                "percentage_calculation",
            ],
            tags=["계산", "수학", "단위변환", "환율", "calculator"],
            schema=AgentSchema(
                input_type="query",
                output_type="json",
            ),
            display_name="Calculator",
            display_name_ko="계산기",
            icon="🔢",
            color="#84cc16",
            priority=40,
        ),
        AgentRegistryEntry(
            agent_id="code_generation_agent",
            name="코드 생성 에이전트",
            description=(
                "다양한 프로그래밍 언어에서 고품질 코드를 생성하는 전문 에이전트입니다. "
                "Python, JavaScript, Java, C++ 등 다양한 언어로 코드를 작성합니다. "
                "함수, 클래스, 알고리즘 구현, 버그 수정, 코드 최적화를 수행합니다."
            ),
            capabilities=[
                "code_generation",
                "code_analysis",
                "bug_fixing",
                "code_explanation",
                "optimization",
                "algorithm_implementation",
            ],
            tags=["코드", "프로그래밍", "개발", "코딩"],
            schema=AgentSchema(
                input_type="query",
                output_type="text",
            ),
            display_name="Code Generator",
            display_name_ko="코드 생성",
            icon="💻",
            color="#6366f1",
            priority=6,
        ),
        AgentRegistryEntry(
            agent_id="rag_search_agent",
            name="RAG 검색 에이전트",
            description=(
                "벡터 데이터베이스 기반 문서 검색 및 답변 생성. "
                "업로드된 문서에서 관련 정보를 검색하고 답변을 생성합니다."
            ),
            capabilities=[
                "document_search",
                "vector_search",
                "context_retrieval",
                "document_qa",
            ],
            tags=["RAG", "문서검색", "벡터검색", "문서"],
            schema=AgentSchema(
                input_type="query",
                output_type="text",
            ),
            display_name="RAG Search",
            display_name_ko="문서 검색",
            icon="📚",
            color="#14b8a6",
            priority=9,
        ),
        AgentRegistryEntry(
            agent_id="currency_exchange_agent",
            name="환율 변환 에이전트",
            description=(
                "실시간 환율 정보를 제공하고 통화 변환을 수행합니다. "
                "USD, EUR, JPY, CNY, GBP 등 30개 이상의 통화를 지원합니다. "
                "환율 조회, 통화 변환, 환율 추이 분석에 사용하세요. "
                "exchange rate, currency conversion, 환율, 달러, 엔화 관련 질문에 적합합니다."
            ),
            capabilities=[
                "exchange_rate",
                "currency_conversion",
                "real_time_rate",
                "rate_history",
                "multi_currency",
            ],
            tags=["환율", "통화", "달러", "엔화", "유로", "exchange", "currency", "USD", "EUR", "JPY"],
            schema=AgentSchema(
                input_type="query",
                output_type="json",
            ),
            display_name="Currency Exchange",
            display_name_ko="환율 변환",
            icon="💱",
            color="#f59e0b",
            priority=50,  # High priority for currency queries
        ),
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the agent registry.

        Args:
            config_path: Optional path to agent configuration file
        """
        self._agents: Dict[str, AgentRegistryEntry] = {}
        self._config_path = config_path
        self._initialized = False

    def initialize(self) -> None:
        """Initialize registry with default agents and config file if available"""
        if self._initialized:
            return

        # Load default agents
        for agent in self.DEFAULT_AGENTS:
            self._agents[agent.agent_id] = agent

        # Load from config file if provided
        if self._config_path and self._config_path.exists():
            self._load_from_config(self._config_path)

        self._initialized = True
        logger.info(f"Agent registry initialized with {len(self._agents)} agents")

    def _load_from_config(self, config_path: Path) -> None:
        """Load additional agent configurations from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            for agent_id, agent_data in config.items():
                if agent_id not in self._agents:
                    # Create new entry from config
                    entry = self._create_entry_from_config(agent_id, agent_data)
                    if entry:
                        self._agents[agent_id] = entry
                else:
                    # Update existing entry with config data
                    self._update_entry_from_config(agent_id, agent_data)

            logger.info(f"Loaded agent config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load agent config from {config_path}: {e}")

    def _create_entry_from_config(
        self,
        agent_id: str,
        config: Dict[str, Any]
    ) -> Optional[AgentRegistryEntry]:
        """Create AgentRegistryEntry from config dictionary"""
        try:
            schema = AgentSchema(
                input_type=config.get("input_type", "query"),
                output_type=config.get("output_type", "text"),
            )
            return AgentRegistryEntry(
                agent_id=agent_id,
                name=config.get("name", agent_id),
                description=config.get("description", ""),
                capabilities=config.get("capabilities", []),
                tags=config.get("tags", []),
                schema=schema,
                display_name=config.get("display_name"),
                display_name_ko=config.get("display_name_ko"),
                icon=config.get("icon", "🤖"),
                color=config.get("color", "#6366f1"),
                priority=config.get("priority", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to create agent entry for {agent_id}: {e}")
            return None

    def _update_entry_from_config(
        self,
        agent_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Update existing agent entry with config data"""
        entry = self._agents[agent_id]

        # Update fields if present in config
        if "description" in config and config["description"]:
            entry.description = config["description"]
        if "capabilities" in config and config["capabilities"]:
            entry.capabilities = config["capabilities"]
        if "tags" in config and config["tags"]:
            entry.tags = config["tags"]

    def get_all_agents(self) -> List[AgentRegistryEntry]:
        """Get all registered agents"""
        if not self._initialized:
            self.initialize()
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentRegistryEntry:
        """
        Get agent by ID.

        Raises:
            AgentNotFoundError: If agent is not found
        """
        if not self._initialized:
            self.initialize()

        if agent_id not in self._agents:
            raise AgentNotFoundError(
                agent_id=agent_id,
                available_agents=list(self._agents.keys())
            )
        return self._agents[agent_id]

    def get_agent_safe(self, agent_id: str) -> Optional[AgentRegistryEntry]:
        """Get agent by ID, returns None if not found"""
        if not self._initialized:
            self.initialize()
        return self._agents.get(agent_id)

    def has_agent(self, agent_id: str) -> bool:
        """Check if agent exists in registry"""
        if not self._initialized:
            self.initialize()
        return agent_id in self._agents

    def register_agent(self, entry: AgentRegistryEntry) -> None:
        """Register a new agent or update existing"""
        if not self._initialized:
            self.initialize()
        self._agents[entry.agent_id] = entry
        logger.info(f"Registered agent: {entry.agent_id}")

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent. Returns True if agent was found and removed."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def get_agents_by_capability(self, capability: str) -> List[AgentRegistryEntry]:
        """Get all agents with a specific capability"""
        if not self._initialized:
            self.initialize()

        return [
            agent for agent in self._agents.values()
            if capability.lower() in [c.lower() for c in agent.capabilities]
        ]

    def get_agents_by_tag(self, tag: str) -> List[AgentRegistryEntry]:
        """Get all agents with a specific tag"""
        if not self._initialized:
            self.initialize()

        return [
            agent for agent in self._agents.values()
            if tag.lower() in [t.lower() for t in agent.tags]
        ]

    def get_available_agents(self) -> List[AgentRegistryEntry]:
        """Get all currently available agents (is_available=True)"""
        if not self._initialized:
            self.initialize()

        return [
            agent for agent in self._agents.values()
            if agent.is_available
        ]

    def get_agent_ids(self) -> List[str]:
        """Get list of all agent IDs"""
        if not self._initialized:
            self.initialize()
        return list(self._agents.keys())

    def build_prompt_context(self, include_schema: bool = False) -> str:
        """
        Build context string for Query Planner LLM prompt.

        This creates a formatted string describing all available agents
        that can be included in the planning prompt.

        Args:
            include_schema: Whether to include I/O schema details

        Returns:
            Formatted string describing available agents
        """
        if not self._initialized:
            self.initialize()

        agents = self.get_available_agents()
        lines = ["# 사용 가능한 에이전트 목록\n"]

        for agent in sorted(agents, key=lambda a: -a.priority):
            lines.append(f"## {agent.agent_id}")
            lines.append(f"- 이름: {agent.name}")
            lines.append(f"- 설명: {agent.description}")
            # capabilities와 tags를 문자열로 변환 (dict 객체 처리)
            capabilities_str = ', '.join(
                c if isinstance(c, str) else (c.get('name', str(c)) if isinstance(c, dict) else str(c))
                for c in agent.capabilities
            ) if agent.capabilities else ''
            tags_str = ', '.join(
                t if isinstance(t, str) else (t.get('name', str(t)) if isinstance(t, dict) else str(t))
                for t in agent.tags
            ) if agent.tags else ''
            lines.append(f"- 능력: {capabilities_str}")
            lines.append(f"- 태그: {tags_str}")

            if include_schema:
                lines.append(f"- 입력 타입: {agent.schema.input_type}")
                lines.append(f"- 출력 타입: {agent.schema.output_type}")

            lines.append("")

        return "\n".join(lines)

    def build_agents_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Build dictionary of agents for JSON serialization.

        Returns:
            Dictionary with agent_id as key and agent details as value
        """
        if not self._initialized:
            self.initialize()

        return {
            agent.agent_id: agent.to_dict()
            for agent in self.get_available_agents()
        }

    def get_schema_compatibility(
        self,
        source_agent_id: str,
        target_agent_id: str
    ) -> bool:
        """
        Check if source agent's output is compatible with target agent's input.

        Returns:
            True if schemas are compatible (possibly with transformation)
        """
        source = self.get_agent_safe(source_agent_id)
        target = self.get_agent_safe(target_agent_id)

        if not source or not target:
            return False

        return source.schema.is_compatible_with(target.schema)

    def __len__(self) -> int:
        if not self._initialized:
            self.initialize()
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return self.has_agent(agent_id)

    def __iter__(self):
        if not self._initialized:
            self.initialize()
        return iter(self._agents.values())


# Singleton instance for global access
_default_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get the default global agent registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentRegistry()
        _default_registry.initialize()
    return _default_registry
