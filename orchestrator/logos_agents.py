"""Logos 기본 에이전트 — SDK 레지스트리에 넣을 수 있는 Logos 인스턴스 데이터.

logosai.orchestration.AgentRegistry 는 빈 상태로 시작한다 (2026-10-05). 이 목록은
그 전까지 SDK 의 AgentRegistry.DEFAULT_AGENTS 였고, 소스 텍스트를 그대로 옮겼다
(tests/test_logos_agents.py 가 옮기기 전 기록과 필드 단위로 같음을 확인한다).

쓰는 곳: 레지스트리를 DB 로 채우지 못했을 때의 폴백.
    registry = AgentRegistry(defaults=logos_default_agents())

알려진 유령 (KNOWN_GHOSTS): rag_search_agent — ACP 라이브(103개)·agents.json·운영 DB
레지스트리 어디에도 없다 (실제 있는 것은 rag_agent). 그런데 이 목록에 있어 운영 플래너
프롬프트에 매번 후보로 실린다. **일부러 남겼다**: 이 블록(230자) 하나만 빼도 계획 회귀
하네스의 안정 시나리오 30개 중 6개가 바뀌었다(다시 넣으면 0, 2026-10-05 실측). 바뀐
계획 일부는 오히려 나아 보여(A3 시간→날씨 오답 해소) '변화'가 아니라 '정답' 기준
평가가 필요하다 — 이 이전(지식 분리)과 섞지 않고 별도 단계로 뺀다.
"""

from typing import List

from .models import AgentRegistryEntry, AgentSchema

#: 실체 없이 목록에 남아 있는 에이전트 (위 docstring)
KNOWN_GHOSTS = ("rag_search_agent",)


def logos_default_agents() -> List[AgentRegistryEntry]:
    """Logos 기본 에이전트 12개 — 호출할 때마다 새 항목을 만든다 (공유 변경 방지)."""
    return [
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
