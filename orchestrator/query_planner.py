"""
Query Planner

Uses gemini-2.0-flash-lite (non-thinking) for single-call execution planning.
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


class QueryPlanner:
    """
    Query analysis and execution planning using gemini-2.0-flash-lite.

    Based on comprehensive 4-model comparison testing:
    - gemini-2.0-flash-lite: 100% accuracy, 3.63s avg (WINNER)
    - gemini-2.0-flash: 100% accuracy, 9.39s avg
    - gemini-2.0-flash-lite+thinking: 81.8% accuracy
    - gemini-2.0-flash+thinking: 54.5% accuracy

    Single LLM call generates complete execution plan including:
    - Workflow strategy (sequential/parallel/hybrid)
    - Stage definitions with execution types
    - Agent assignments with sub-queries
    - Data flow between agents
    - Final aggregation strategy
    """

    # LLM Configuration
    MODEL = "gemini-2.0-flash-lite"
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
                logger.debug(
                    f"[QueryPlanner] Hybrid confidence {confidence:.1%} below threshold "
                    f"{self.HYBRID_CONFIDENCE_THRESHOLD:.1%}, will let LLM decide"
                )
                return None, None

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
        """Build ExecutionPlan from parsed LLM response"""

        stages = []
        for stage_data in plan_data.get("stages", []):
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

        plan = ExecutionPlan(
            query=query,
            workflow_strategy=plan_data.get("workflow_strategy", "sequential"),
            stages=stages,
            final_aggregation=plan_data.get("final_aggregation", {"type": "combine"}),
            reasoning=plan_data.get("reasoning", ""),
            plan_id=plan_id,
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
