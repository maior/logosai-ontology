"""
Query Planner

Uses gemini-2.5-flash-lite (non-thinking) for single-call execution planning.
Based on 4-model comparison test: Flash-Lite achieves 100% accuracy at 3.63s avg.

Key Design Principles:
- Single LLM call for complete planning
- No thinking mode (proven less accurate for this task)
- Deterministic execution after planning
- HybridAgentSelector (GNN+RL) for intelligent agent selection (v3.0)
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .models import (
    AgentTask,
    ExecutionPlan,
    ExecutionStage,
)
from .agent_registry import AgentRegistry, get_registry
from .progress_streamer import ProgressStreamer
from .exceptions import PlanningError, NoSuitableAgentError

logger = logging.getLogger(__name__)

# Import HybridAgentSelector for GNN+RL agent selection
try:
    from ..core.hybrid_agent_selector import get_hybrid_selector, HybridAgentSelector
    HYBRID_SELECTOR_AVAILABLE = True
except ImportError:
    HYBRID_SELECTOR_AVAILABLE = False
    get_hybrid_selector = None
    HybridAgentSelector = None
    logger.warning("[QueryPlanner] HybridAgentSelector not available, using LLM-only selection")


# Import Google Gemini client
try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None


def detect_explicit_capability_gap(query: str) -> Optional[Dict[str, Any]]:
    """Code-level safety net (2026-05-09): query 에 명시적 에이전트 생성 패턴이
    있으면 capability_gap 강제 trigger.

    LLM (flash-lite) 이 prompt 강화에도 internet_agent 같은 generic 으로 fallback
    하는 보수적 분류 깨지지 않을 때의 backstop. 단순 키워드 매칭이지만 의미가
    명백한 의도만 잡으므로 false-positive 위험 낮음.

    Args:
        query: 사용자 쿼리 원문

    Returns:
        capability_gap dict (detected, missing_capabilities, suggested_agent_description, reason)
        패턴 매칭 안 되면 None.
    """
    if not query:
        return None
    explicit_patterns = [
        # 한국어
        "에이전트 만들", "에이전트를 만들", "에이전트 생성", "에이전트를 생성",
        "에이전트 추가", "에이전트를 추가", "에이전트 만들어", "agent 만들",
        # 영어
        "build agent", "create agent", "make agent",
        "build an agent", "create an agent", "make an agent",
        "build me an agent", "create me an agent",
    ]
    q_lower = query.lower()
    if not any(p.lower() in q_lower for p in explicit_patterns):
        return None
    return {
        "detected": True,
        "missing_capabilities": ["explicit_agent_creation_request"],
        "suggested_agent_description": query,
        "reason": "사용자가 명시적으로 새 에이전트 생성을 요청 (code safety net)",
    }


def merge_independent_stages(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LLM 이 만든 stages 에서 데이터 의존성 없는 1-agent stages 를 parallel 로 병합.

    flash-lite 의 흔한 실수 (독립적 multi-domain 쿼리를 1-agent-per-stage 로 분리) 의 backstop.

    Rules:
      - 인접한 stages 들이 모두 1 agent 이고 input_from 이 모두 null 이면 → 같은 parallel stage 로 병합
      - 데이터 의존성 (input_from 에 stage_X 참조) 있으면 그 자리에서 분리 유지
      - 이미 parallel 인 stage 는 건드리지 않음

    Args:
        stages: LLM 이 만든 raw stages list (각 dict: stage_id, execution_type, agents)

    Returns:
        Post-processed stages with stage_ids renumbered.
    """
    if not stages or len(stages) < 2:
        return stages

    def _is_independent_singleton(stage: Dict[str, Any]) -> bool:
        agents = stage.get("agents", [])
        if len(agents) != 1:
            return False
        input_from = agents[0].get("input_from")
        # input_from 이 None / [] / 빈 리스트 면 의존성 없음
        if input_from is None:
            return True
        if isinstance(input_from, list) and len(input_from) == 0:
            return True
        return False

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(stages):
        cur = stages[i]
        if _is_independent_singleton(cur):
            # 인접한 independent singletons 수집
            group_agents = list(cur.get("agents", []))
            j = i + 1
            while j < len(stages) and _is_independent_singleton(stages[j]):
                group_agents.extend(stages[j].get("agents", []))
                j += 1
            if len(group_agents) > 1:
                # 병합
                out.append({
                    "stage_id": len(out) + 1,
                    "execution_type": "parallel",
                    "agents": group_agents,
                })
                i = j
                continue
        # 그대로 추가 + stage_id 재번호
        new_stage = dict(cur)
        new_stage["stage_id"] = len(out) + 1
        out.append(new_stage)
        i += 1

    # 후속 stages 의 input_from 참조도 새 stage_id 로 매핑해야 하지만,
    # 현재 input_from 형식이 "stage_N.agent_id" 인데 stage 번호 변경이 일어남.
    # 안전성: ExecutionEngine 이 stage 번호 의존성보다는 stage 순서로 처리한다고 가정 (검증 필요).
    # 단, 1-agent → parallel 병합은 stage 1 에서 일어나므로 stage 1 이름은 유지됨.
    return out


class QueryPlanner:
    """
    Query analysis and execution planning using gemini-2.5-flash-lite.

    Based on comprehensive 4-model comparison testing:
    - gemini-2.5-flash-lite: 100% accuracy, 3.63s avg (WINNER)
    - gemini-2.0-flash: 100% accuracy, 9.39s avg
    - gemini-2.5-flash-lite+thinking: 81.8% accuracy
    - gemini-2.0-flash+thinking: 54.5% accuracy

    Single LLM call generates complete execution plan including:
    - Workflow strategy (sequential/parallel/hybrid)
    - Stage definitions with execution types
    - Agent assignments with sub-queries
    - Data flow between agents
    - Final aggregation strategy
    """

    # LLM Configuration
    MODEL = "gemini-2.5-flash-lite"
    TEMPERATURE = 0.3
    MAX_TOKENS = 4096

    # GNN+RL Configuration
    USE_HYBRID_SELECTOR = True  # Enable HybridAgentSelector for agent selection
    HYBRID_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to use hybrid selection

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        streamer: Optional[ProgressStreamer] = None,
        api_key: Optional[str] = None,
        hybrid_selector: Optional["HybridAgentSelector"] = None,
    ):
        """
        Initialize Query Planner.

        Args:
            registry: Agent registry (uses default if not provided)
            streamer: Progress streamer for real-time updates
            api_key: Google API key (uses env var if not provided)
            hybrid_selector: HybridAgentSelector for GNN+RL agent selection
        """
        self.registry = registry or get_registry()
        self.streamer = streamer
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        # Initialize HybridAgentSelector
        self._hybrid_selector = hybrid_selector
        self._hybrid_selector_enabled = (
            self.USE_HYBRID_SELECTOR and
            HYBRID_SELECTOR_AVAILABLE and
            hybrid_selector is not None
        )

        if self.USE_HYBRID_SELECTOR and HYBRID_SELECTOR_AVAILABLE and hybrid_selector is None:
            try:
                self._hybrid_selector = get_hybrid_selector()
                self._hybrid_selector_enabled = True
                logger.info("[QueryPlanner] HybridAgentSelector (GNN+RL) enabled")
            except Exception as e:
                logger.warning(f"[QueryPlanner] Failed to initialize HybridAgentSelector: {e}")
                self._hybrid_selector_enabled = False

        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-genai package is required. "
                "Install with: pip install google-genai"
            )

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Set it or provide api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)

    async def create_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        Create execution plan for the given query.

        Uses HybridAgentSelector (GNN+RL) for intelligent agent selection,
        then LLM for workflow design.

        Args:
            query: User query to analyze
            context: Optional additional context

        Returns:
            ExecutionPlan with stages, agents, and aggregation strategy

        Raises:
            PlanningError: If planning fails
            NoSuitableAgentError: If no suitable agents found
        """
        start_time = time.time()
        plan_id = str(uuid.uuid4())[:8]

        # Emit planning start event
        if self.streamer:
            await self.streamer.planning_start(query)

        try:
            # Phase 0: GNN+RL Agent Selection (if enabled)
            recommended_agent = None
            hybrid_metadata = None
            if self._hybrid_selector_enabled and self._hybrid_selector:
                try:
                    recommended_agent, hybrid_metadata = await self._select_agent_via_hybrid(query)
                    if recommended_agent:
                        logger.info(
                            f"[QueryPlanner] GNN+RL recommended: {recommended_agent} "
                            f"(confidence: {hybrid_metadata.get('confidence', 0):.1%})"
                        )
                except Exception as e:
                    logger.warning(f"[QueryPlanner] HybridAgentSelector failed: {e}")

            # Build the prompt (with hybrid recommendation if available)
            prompt = self._build_planning_prompt(query, context, recommended_agent, hybrid_metadata)

            # Call Gemini
            logger.info(f"[QueryPlanner] Calling {self.MODEL} for query: {query[:50]}...")
            response = await self._call_llm(prompt)

            # Parse response
            plan_data = self._parse_llm_response(response)

            # Build ExecutionPlan from parsed data
            plan = self._build_execution_plan(query, plan_data, plan_id)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[QueryPlanner] Plan created in {elapsed_ms:.0f}ms: "
                f"{plan.get_stage_count()} stages, {plan.get_total_agents()} agents"
            )

            # Emit planning complete event
            if self.streamer:
                await self.streamer.planning_complete(plan)

            return plan

        except Exception as e:
            logger.error(f"[QueryPlanner] Planning failed: {e}")
            if self.streamer:
                await self.streamer.planning_error(str(e))
            raise PlanningError(
                message=f"Failed to create execution plan: {e}",
                query=query,
            )

    async def _select_agent_via_hybrid(
        self,
        query: str,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Select agent using HybridAgentSelector (GNN+RL + Knowledge Graph + LLM).

        Returns:
            Tuple of (agent_id, metadata) or (None, None) if selection failed
        """
        if not self._hybrid_selector:
            return None, None

        try:
            # Get available agents from registry
            available_agents = [entry.agent_id for entry in self.registry.get_available_agents()]

            # Build agents_info from registry
            agents_info = {}
            for entry in self.registry.get_available_agents():
                agents_info[entry.agent_id] = {
                    "name": entry.name,
                    "description": entry.description,
                    "capabilities": entry.capabilities,
                    "tags": entry.tags,
                }

            # Call HybridAgentSelector
            agent_id, metadata = await self._hybrid_selector.select_agent(
                query=query,
                available_agents=available_agents,
                agents_info=agents_info,
            )

            # Check confidence threshold
            confidence = metadata.get("confidence", 0) if metadata else 0
            if confidence < self.HYBRID_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"[QueryPlanner] Hybrid confidence {confidence:.1%} below threshold "
                    f"{self.HYBRID_CONFIDENCE_THRESHOLD:.1%}, passing as hint (not mandatory)"
                )
                # 신뢰도 낮아도 힌트로 전달 (LLM이 참고하도록)
                if metadata:
                    metadata["is_hint"] = True
                return agent_id, metadata

            return agent_id, metadata

        except Exception as e:
            logger.warning(f"[QueryPlanner] Hybrid selection failed: {e}")
            return None, None

    async def store_execution_feedback(
        self,
        query: str,
        agent_id: str,
        success: bool,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store execution feedback to HybridAgentSelector for learning.

        This enables the GNN+RL system to learn from execution outcomes.

        Args:
            query: Original user query
            agent_id: Agent that was executed
            success: Whether the execution was successful
            execution_result: Optional execution result for additional context
        """
        if not self._hybrid_selector_enabled or not self._hybrid_selector:
            return

        try:
            await self._hybrid_selector.store_feedback(
                query=query,
                selected_agent=agent_id,
                success=success,
            )
            logger.info(
                f"[QueryPlanner] Stored feedback: {agent_id} "
                f"({'success' if success else 'failure'}) for query: {query[:30]}..."
            )
        except Exception as e:
            logger.warning(f"[QueryPlanner] Failed to store feedback: {e}")

    def _build_planning_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        recommended_agent: Optional[str] = None,
        hybrid_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the complete prompt for the LLM"""

        # Get agent information from registry
        agents_context = self.registry.build_prompt_context(include_schema=True)

        # Build GNN+RL recommendation section
        gnn_rl_section = ""
        if recommended_agent and hybrid_metadata:
            confidence = hybrid_metadata.get("confidence", 0)
            selection_source = hybrid_metadata.get("selection_source", "unknown")
            is_hint = hybrid_metadata.get("is_hint", False)

            if is_hint:
                gnn_rl_section = f"""
## 🔍 GNN+RL 추천 에이전트 (참고)
GNN+RL 지능형 시스템이 아래 에이전트를 추천합니다 (신뢰도 낮음, 참고용):
- **추천 에이전트**: `{recommended_agent}`
- **신뢰도**: {confidence:.1%}
- **선택 근거**: {selection_source}

💡 이 에이전트가 쿼리에 적합한지 description/capabilities를 확인하고, 적합하면 우선 사용하세요.
"""
            else:
                gnn_rl_section = f"""
## 🎯 GNN+RL 추천 에이전트 (IMPORTANT)
GNN+RL 지능형 시스템이 아래 에이전트를 **강력 추천**합니다:
- **추천 에이전트**: `{recommended_agent}`
- **신뢰도**: {confidence:.1%}
- **선택 근거**: {selection_source}

⚠️ **중요**: GNN+RL 신뢰도가 60% 이상이면 이 에이전트를 **반드시 첫 번째 단계**에서 사용하세요.
다른 에이전트를 선택하려면 명확한 이유가 있어야 합니다.
"""

        # Build conversation history section
        conversation_history_section = ""
        if context and context.get("conversation_history"):
            conversation_history_section = f"""# 이전 대화 내용 (최근)
{context["conversation_history"]}

⚠️ 대화 맥락 활용 원칙:
- 사용자가 "그것", "현재가", "이전 결과" 등 이전 대화를 참조하면, 위 대화 내용을 기반으로 쿼리를 해석하세요
- 이전 대화에서 언급된 주제(종목명, 키워드 등)를 현재 쿼리와 연결하세요
- 예: 이전 "삼성전자 주식" + 현재 "현재가는?" → "삼성전자 현재 주가"로 해석

"""

        # Build user memory section
        user_memory_section = ""
        if context and context.get("user_memories"):
            user_memory_section = f"""{context["user_memories"]}

⚠️ 메모리 활용 원칙:
- 사용자의 현재 쿼리 의도가 항상 최우선입니다
- "지시사항"은 항상 따르되, "사용자 정보"는 쿼리와 직접 관련된 경우에만 활용하세요
- 메모리와 현재 쿼리가 충돌하면 현재 쿼리를 따르세요
- 메모리를 근거로 사용자가 명시하지 않은 내용을 추측하지 마세요

"""

        # Build the full prompt
        prompt = f"""# 역할
당신은 사용자 쿼리를 분석하여 최적의 에이전트 워크플로우를 설계하는 전문가입니다.
{gnn_rl_section}
{conversation_history_section}# 사용 가능한 에이전트
{agents_context}

# 핵심 원칙

## 1. 데이터 흐름 원칙
- 실시간 데이터(주가, 환율, 뉴스)가 필요하면 → internet_agent 먼저
- 데이터 가공/분석이 필요하면 → analysis_agent
- 시각화(차트, 그래프)가 필요하면 → data_visualization_agent
- 일반 지식/설명이 필요하면 → llm_search_agent

## 2. 전문 에이전트 우선 원칙 (CRITICAL)
- 특정 도메인 전용 에이전트가 있으면 반드시 해당 에이전트를 우선 사용
- 에이전트 목록의 description/capabilities를 분석하여 가장 적합한 에이전트 선택
- 범용 에이전트(internet_agent, llm_search_agent)는 전문 에이전트가 없을 때만 사용
- 예시:
  - 날씨 쿼리 + weather_agent 존재 → weather_agent 사용 (internet_agent 아님)
  - 쇼핑 쿼리 + shopping_agent 존재 → shopping_agent 사용
  - 코딩 쿼리 + code_generation_agent 존재 → code_generation_agent 사용

## 3. 도메인 구분 원칙
- 삼성반도체 제조공정/FAB/NAND/수율/EUV/Particle 이슈 → samsung_gateway_agent
  (주의: 삼성전자 주가/재무/투자 정보는 samsung_gateway_agent가 아닌 internet_agent 사용!)
- 상품 검색/가격 비교 → shopping_agent
- 코드/프로그래밍 → code_generation_agent
- 문서 검색 → rag_search_agent

## 4. 워크플로우 전략
- sequential: 이전 결과가 다음 작업에 필요할 때 (예: 데이터 수집 → 분석 → 시각화)
- parallel: 독립적인 작업들을 동시 처리할 때 (예: 두 회사 정보 동시 검색)
- hybrid: sequential과 parallel 혼합

## 5. 결과 정리 원칙 (IMPORTANT)
- 모든 쿼리 결과는 사용자에게 깔끔하게 정리되어야 함
- 단순 정보 검색(날씨, 뉴스, 일반 질문)도 마지막에 llm_search_agent로 결과 정리
- 최종 단계에서 사용자 친화적인 형태로 요약 및 포맷팅

### 응답 포맷 가이드라인
최종 정리 에이전트(llm_search_agent 등)의 sub_query에 아래 포맷 지시를 포함하세요:

**검색/리서치 쿼리** (트렌드, 뉴스, 최신 정보 등):
→ "결과를 Markdown 형식으로 정리: 핵심 요약을 먼저, 각 항목은 ## 소제목과 bullet point로 구분, 출처가 있으면 말미에 표기"

**계산/단순 답변 쿼리** (수학, 환율, 날씨 등):
→ "간결하게 핵심 답변을 먼저 제시하고, 필요시 부연 설명 추가"

**비교/분석 쿼리** (제품 비교, 장단점, 분석 등):
→ "Markdown 표(table) 또는 항목별 비교 형식으로 정리, 결론을 마지막에 제시"

**코드/기술 쿼리** (프로그래밍, 기술 설명 등):
→ "코드는 ```언어 코드블록으로 감싸고, 설명은 단계별로 정리"

**일반 원칙**:
- 긴 텍스트 덩어리(wall of text) 금지 — 반드시 구조화된 Markdown 사용
- 핵심 내용을 상단에 배치 (inverted pyramid)
- 항목이 3개 이상이면 bullet point 또는 번호 목록 사용

## 6. 워크플로우 설계 규칙
- 불필요한 에이전트 추가 금지 (필요한 에이전트만 선택)
- 같은 에이전트 중복 호출 금지 (한 스테이지 내에서)
- 데이터 의존성 무시 금지 (실시간 데이터 먼저 수집)

### 단일 에이전트 vs 다단계 판단 기준 (CRITICAL)
- **단일 에이전트 충분**: 정보 검색, Q&A, 번역, 코드 생성, 요약, 일반 대화
  → 전문 에이전트 1개로 바로 결과 반환 (예: "테슬라 주식 어때?" → internet_agent만)
- **2단계 이상 필요**: 데이터 분석 후 시각화, 비교 분석, 복잡한 계산
  → 데이터 수집 → 분석/시각화 (예: "삼성전자 5일 종가 그래프" → internet → visualization)
- **analysis_agent 사용 조건**: 반드시 구체적인 숫자/통계 데이터가 제공될 때만
  → "주식 어때?" 같은 일반 질문에 analysis_agent 사용 금지
- **llm_search_agent 정리 단계**: 최종 정리가 꼭 필요한 복합 쿼리에만 추가
  → 단순 검색/Q&A에는 불필요

### 병렬화 판단 기준 (CRITICAL — 자주 누락됨)
**같은 stage 안에 여러 agent 를 parallel 로 묶어야 하는 경우**:
- 사용자 쿼리에 **서로 독립적인 sub-task 가 N개** 포함 (N≥2)
  → 데이터 의존성 없으면 **모두 같은 stage_id 안에 execution_type="parallel"**
  → 서로 다른 agent (예: 날씨 + 환율 + 검색) 들도 마찬가지 — agent 가 다르다고 stage 분리 금지
- 표지: 쿼리에 "그리고", "와/과", "동시에", "각각", "모두", 콤마로 나열된 N개 항목
- **잘못된 패턴**: agent 마다 stage 분리 → execution_type 이 sequential 인데 의존성 없는 N개 stage 가 줄지어 늘어섬 (불필요한 latency)
- **올바른 패턴**: 의존성 없는 모든 agent 를 **stage 1 에 parallel 로 묶고**, 종합이 필요하면 stage 2 에 sequential 로 종합 agent 1개

**판단 알고리즘**:
1. 쿼리를 sub-task 들로 분해
2. 각 sub-task 간 데이터 의존성 그래프 작성
3. 같은 의존성 레이어 → 같은 stage 안에 parallel
4. 다른 레이어 → 다른 stage 에 sequential

# 예시

## 예시 1: "삼성전자 5일 종가 그래프로 그려줘"
{{
  "workflow_strategy": "sequential",
  "stages": [
    {{
      "stage_id": 1,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "internet_agent",
          "sub_query": "삼성전자 최근 5일간 종가 데이터",
          "input_from": null,
          "output_to": ["stage_2"]
        }}
      ]
    }},
    {{
      "stage_id": 2,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "analysis_agent",
          "sub_query": "주가 데이터 분석 및 구조화",
          "input_from": ["stage_1.internet_agent"],
          "output_to": ["stage_3"]
        }}
      ]
    }},
    {{
      "stage_id": 3,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "data_visualization_agent",
          "sub_query": "종가 추이 라인 차트 생성",
          "input_from": ["stage_2.analysis_agent"],
          "output_to": ["final"]
        }}
      ]
    }}
  ],
  "final_aggregation": {{
    "type": "single",
    "format": "chart"
  }},
  "reasoning": "실시간 주가 데이터 수집 → 데이터 구조화 → 차트 시각화의 순차적 파이프라인"
}}

## 예시 2: "테슬라 주식 어때?" (단일 에이전트 — 실시간 검색만 필요)
{{
  "workflow_strategy": "sequential",
  "stages": [
    {{
      "stage_id": 1,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "internet_agent",
          "sub_query": "테슬라 현재 주가 및 최근 동향",
          "input_from": null,
          "output_to": ["final"]
        }}
      ]
    }}
  ],
  "final_aggregation": {{
    "type": "single",
    "format": "report"
  }},
  "reasoning": "단순 정보 검색 쿼리 — internet_agent 1개 스테이지로 충분. analysis_agent 불필요."
}}

## 예시 3: "양자역학이란 무엇인가?" (일반 지식 검색)
{{
  "workflow_strategy": "sequential",
  "stages": [
    {{
      "stage_id": 1,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "llm_search_agent",
          "sub_query": "양자역학의 기본 개념, 원리, 주요 특징 설명",
          "input_from": null,
          "output_to": ["final"]
        }}
      ]
    }}
  ],
  "final_aggregation": {{
    "type": "single",
    "format": "summary"
  }},
  "reasoning": "일반 지식 질문은 llm_search_agent가 직접 답변 (전문 에이전트 불필요)"
}}

## 예시 3: "삼성전자와 애플 실적 비교"
{{
  "workflow_strategy": "hybrid",
  "stages": [
    {{
      "stage_id": 1,
      "execution_type": "parallel",
      "agents": [
        {{
          "agent_id": "internet_agent",
          "sub_query": "삼성전자 최근 분기 실적",
          "input_from": null,
          "output_to": ["stage_2"]
        }},
        {{
          "agent_id": "internet_agent",
          "sub_query": "애플 최근 분기 실적",
          "input_from": null,
          "output_to": ["stage_2"]
        }}
      ]
    }},
    {{
      "stage_id": 2,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "analysis_agent",
          "sub_query": "삼성전자와 애플 실적 비교 분석. Markdown 표(table)로 주요 지표를 비교하고, 결론을 마지막에 제시",
          "input_from": ["stage_1.internet_agent"],
          "output_to": ["final"]
        }}
      ]
    }}
  ],
  "final_aggregation": {{
    "type": "combine",
    "format": "comparison_report"
  }},
  "reasoning": "두 회사 데이터를 병렬로 수집하고 통합 분석"
}}

## 예시 4: 독립적 다중 정보 요청 — **다른 agent 들도 같은 stage 에 parallel** (CRITICAL)
"여러 종류의 독립적 정보를 한 번에 알려줘" (예: A 정보, B 정보, C 정보 — 데이터 의존성 없음)
{{
  "workflow_strategy": "parallel",
  "stages": [
    {{
      "stage_id": 1,
      "execution_type": "parallel",
      "agents": [
        {{ "agent_id": "<A 도메인 전문 agent>", "sub_query": "A 정보 요청", "input_from": null, "output_to": ["stage_2"] }},
        {{ "agent_id": "<B 도메인 전문 agent>", "sub_query": "B 정보 요청", "input_from": null, "output_to": ["stage_2"] }},
        {{ "agent_id": "<C 도메인 전문 agent>", "sub_query": "C 정보 요청", "input_from": null, "output_to": ["stage_2"] }}
      ]
    }},
    {{
      "stage_id": 2,
      "execution_type": "sequential",
      "agents": [
        {{
          "agent_id": "llm_search_agent",
          "sub_query": "수집된 A/B/C 정보를 사용자 친화적으로 종합 정리",
          "input_from": ["stage_1.<A>", "stage_1.<B>", "stage_1.<C>"],
          "output_to": ["final"]
        }}
      ]
    }}
  ],
  "final_aggregation": {{
    "type": "combine",
    "format": "summary"
  }},
  "reasoning": "독립적인 N개 정보 요청 → 모두 같은 stage 에 parallel 로 묶어 latency 최소화. agent 가 서로 달라도 의존성 없으면 같은 stage."
}}
**핵심 원칙**: agent 가 다르다고 stage 분리 금지. **데이터 의존성만이 stage 분리 기준**.

## 예시 5 (CRITICAL — 자주 실수): 잘못된 분리 vs 올바른 병합
**쿼리**: "오늘 날씨, 환율, 비트코인 가격을 모두 알려줘"
**상황**: 3개 sub-task. 서로 독립 (의존성 없음). 다른 도메인 agent 들 필요.

❌ **잘못된 패턴** (절대 피할 것 — flash-lite 의 흔한 실수):
```
{{ "stages": [
  {{ "stage_id": 1, "execution_type": "sequential", "agents": [{{ weather_agent }}] }},
  {{ "stage_id": 2, "execution_type": "sequential", "agents": [{{ currency_agent }}] }},
  {{ "stage_id": 3, "execution_type": "sequential", "agents": [{{ internet_agent }}] }},
  {{ "stage_id": 4, "execution_type": "sequential", "agents": [{{ llm_search_agent }}] }}
]}}
```
**왜 잘못인가**: 3개 sub-task 가 서로 독립인데 stage 4개로 분리. 결과적으로 3배 latency. agent 가 다른 것을 stage 분리의 이유로 삼지 않는다.

✅ **올바른 패턴**:
```
{{ "stages": [
  {{
    "stage_id": 1,
    "execution_type": "parallel",
    "agents": [
      {{ "agent_id": "weather_agent",  "sub_query": "...", "input_from": null, "output_to": ["stage_2"] }},
      {{ "agent_id": "currency_exchange_agent", "sub_query": "...", "input_from": null, "output_to": ["stage_2"] }},
      {{ "agent_id": "internet_agent", "sub_query": "비트코인 가격 ...", "input_from": null, "output_to": ["stage_2"] }}
    ]
  }},
  {{ "stage_id": 2, "execution_type": "sequential", "agents": [{{ "agent_id": "llm_search_agent", "sub_query": "수집 정보 종합", "input_from": ["stage_1.weather_agent","stage_1.currency_exchange_agent","stage_1.internet_agent"], "output_to": ["final"] }}] }}
]}}
```
**핵심**: 모든 독립 sub-task 를 stage 1 에 parallel. 종합은 stage 2.

## 예시 6: 능력 부재 — capability_gap 선언 (CRITICAL)
**출력 필드**: `capability_gap` (optional). 다음 3 시그널 중 하나라도 해당하면 **detected=true** 로 선언:

### 시그널 A — 명시적 에이전트 생성 요청
사용자가 "에이전트 만들어줘", "에이전트 생성", "build agent", "create agent" 같이
**새 에이전트 자체를 요구**하면 등록된 generic agent 가 우회 처리 가능해도 **무조건 detected=true**.

### 시그널 B — 외부 API 전용성
특정 외부 API (Mastodon, Discord, Slack, Notion, Stripe, GitHub API 등) 호출이 필요한데
그 서비스 전용 에이전트가 등록 목록에 없으면 → **internet_agent 의 일반 검색으로 대체 불가**
→ detected=true. 일반 검색은 API 호출과 본질이 다름 (인증, 페이로드, 권한 미지원).

### 시그널 C — 약한 매칭 금지
internet_agent / analysis_agent / llm_search_agent 같은 범용 에이전트가 있다고 해서
특화 도메인 쿼리를 그쪽으로 fallback 하면 안 됨. 약한 매칭은 detected=true 와 같다.

### 출력 형식
```
{{
  "workflow_strategy": "sequential",
  "stages": [],
  "capability_gap": {{
    "detected": true,
    "missing_capabilities": ["mastodon_api", "sentiment_analysis"],
    "suggested_agent_description": "Mastodon API 로 toot fetch + sentiment 분석",
    "reason": "Mastodon 전용 에이전트 부재, internet_agent 의 일반 검색으로 대체 불가"
  }},
  "reasoning": "...",
  "final_aggregation": {{"type": "single"}}
}}
```
**주의**: capability_gap.detected=true 면 stages 는 빈 list 또는 부분 워크플로우 (전처리만).
실제 FORGE 호출 + 새 에이전트 등록은 logos_api 가 ACP /stream/multi 로 fallback 해서 처리.

### ❌ 잘못된 예 (자주 발생):
query: "Mastodon API 로 toot 5개 가져와서 sentiment 분석하는 에이전트 만들어줘"
→ 잘못된 응답: capability_gap 누락, stages=[internet_agent, analysis_agent]
→ 진짜 문제: (1) 사용자가 명시적 "에이전트 만들어줘" — 시그널 A 위반.
              (2) Mastodon API 는 internet_agent 로 호출 불가 — 시그널 B 위반.

### ✅ 올바른 예:
같은 query → capability_gap.detected=true, missing=["mastodon_api"], stages=[]

{user_memory_section}# 사용자 쿼리
"{query}"

# 지시사항

## 대화 맥락 기반 쿼리 확장 (CRITICAL)
사용자 쿼리가 이전 대화를 참조하는 경우(예: "현재가는?", "그래프로 보여줘", "비교해줘" 등 불완전한 쿼리):
1. **이전 대화 내용**을 참고하여 쿼리의 주제/대상을 파악하세요
2. **sub_query를 반드시 자기완결적(self-contained) 문장으로 작성**하세요
3. 예시:
   - 이전: "삼성전자 주식은 어때?" → 현재: "현재가는?"
   - sub_query: "삼성전자 현재 주가" (O) — "현재가" (X)
   - 이전: "테슬라 분석해줘" → 현재: "일주일 전 대비 변화율은?"
   - sub_query: "테슬라 주가 일주일 전 대비 변화율" (O) — "일주일 전 대비 변화율" (X)
4. sub_query만 보고도 어떤 데이터를 가져와야 하는지 알 수 있어야 합니다
5. **사용자가 명시하지 않은 조건을 sub_query에 추가하지 마세요** (CRITICAL)
   - "파일 찾아줘" → sub_query: "oars 관련 파일 검색" (O) — "데스크탑 폴더에서 oars 파일 검색" (X, 사용자가 데스크탑이라고 안 함)
   - "날씨 알려줘" → sub_query: "서울 날씨" (O, 대화 맥락에서 추론) — "내일 오전 서울 강남구 날씨" (X, 과도한 추가)
   - 에이전트가 알아서 판단할 영역(검색 범위, 정렬 순서 등)을 sub_query에서 제한하지 마세요

위 쿼리에 대한 실행 계획을 JSON 형식으로 작성하세요.
반드시 위 JSON 형식을 정확히 따르세요.
JSON 외에 다른 텍스트는 포함하지 마세요.

```json
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            config = types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                max_output_tokens=self.MAX_TOKENS,
            )

            response = self.client.models.generate_content(
                model=self.MODEL,
                config=config,
                contents=prompt,
            )

            return response.text

        except Exception as e:
            logger.error(f"[QueryPlanner] Gemini API call failed: {e}")
            raise

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from LLM"""
        try:
            # Clean up response - extract JSON
            text = response.strip()

            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            # Parse JSON
            return json.loads(text)

        except json.JSONDecodeError as e:
            logger.error(f"[QueryPlanner] Failed to parse JSON: {e}")
            logger.error(f"[QueryPlanner] Raw response: {response[:500]}...")
            raise PlanningError(
                message=f"Invalid JSON in LLM response: {e}",
                llm_response=response[:500],
            )

    def _build_execution_plan(
        self,
        query: str,
        plan_data: Dict[str, Any],
        plan_id: str,
    ) -> ExecutionPlan:
        """Build ExecutionPlan from parsed LLM response.

        Post-processing safety net (2026-05-09): merge_independent_stages 가
        flash-lite 의 흔한 실수 (의존성 없는 1-agent stages 의 sequential 분리) 를
        자동으로 parallel 로 병합. prompt 가 안 통할 때의 backstop.
        """

        raw_stages = plan_data.get("stages", [])
        merged_stages = merge_independent_stages(raw_stages)

        # workflow_strategy 도 함께 갱신 (병합 결과 반영)
        workflow_strategy = plan_data.get("workflow_strategy", "sequential")
        if len(merged_stages) != len(raw_stages):
            n_parallel = sum(1 for s in merged_stages if s.get("execution_type") == "parallel")
            n_seq = sum(1 for s in merged_stages if s.get("execution_type") != "parallel")
            workflow_strategy = (
                "parallel" if n_parallel > 0 and n_seq == 0
                else "hybrid" if n_parallel > 0 and n_seq > 0
                else "sequential"
            )
            logger.info(
                f"  Stage merger: {len(raw_stages)} → {len(merged_stages)} stages "
                f"(strategy: {plan_data.get('workflow_strategy')} → {workflow_strategy})"
            )
            plan_data["workflow_strategy"] = workflow_strategy

        stages = []
        for stage_data in merged_stages:
            agents = []
            for agent_data in stage_data.get("agents", []):
                agent = AgentTask(
                    agent_id=agent_data.get("agent_id"),
                    sub_query=agent_data.get("sub_query", query),
                    input_from=agent_data.get("input_from"),
                    output_to=agent_data.get("output_to"),
                    expected_output=agent_data.get("expected_output"),
                )
                agents.append(agent)

            stage = ExecutionStage(
                stage_id=stage_data.get("stage_id", len(stages) + 1),
                execution_type=stage_data.get("execution_type", "sequential"),
                agents=agents,
            )
            stages.append(stage)

        # capability_gap 결정 (LLM 응답 우선, safety net 으로 backfill)
        capability_gap = plan_data.get("capability_gap")
        if not (isinstance(capability_gap, dict) and capability_gap.get("detected")):
            # LLM 이 안 잡았으면 safety net (명시적 패턴) 검사
            safety = detect_explicit_capability_gap(query)
            if safety:
                logger.info(
                    f"  Code safety net: 명시적 에이전트 생성 패턴 감지 → "
                    f"capability_gap 강제 trigger (LLM 누락 보강)"
                )
                capability_gap = safety
            else:
                capability_gap = None

        plan = ExecutionPlan(
            query=query,
            workflow_strategy=plan_data.get("workflow_strategy", "sequential"),
            stages=stages,
            final_aggregation=plan_data.get("final_aggregation", {"type": "combine"}),
            reasoning=plan_data.get("reasoning", ""),
            plan_id=plan_id,
            capability_gap=capability_gap,
        )

        return plan

    async def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Quick validation of query before full planning.

        Returns:
            Dict with validation results
        """
        # Check for empty query
        if not query or not query.strip():
            return {
                "valid": False,
                "error": "Empty query",
            }

        # Check for minimum length
        if len(query.strip()) < 2:
            return {
                "valid": False,
                "error": "Query too short",
            }

        # Check for available agents
        agents = self.registry.get_available_agents()
        if not agents:
            return {
                "valid": False,
                "error": "No agents available",
            }

        return {
            "valid": True,
            "available_agents": len(agents),
        }


# Factory function for creating QueryPlanner
def create_query_planner(
    registry: Optional[AgentRegistry] = None,
    streamer: Optional[ProgressStreamer] = None,
) -> QueryPlanner:
    """Create a QueryPlanner instance with default configuration"""
    return QueryPlanner(registry=registry, streamer=streamer)
