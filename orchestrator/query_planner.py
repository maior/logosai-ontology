"""
Query Planner

Uses gemini-2.5-flash-lite (non-thinking) for single-call execution planning.
Based on 4-model comparison test: Flash-Lite achieves 100% accuracy at 3.63s avg.

Key Design Principles:
- Single LLM call for complete planning
- No thinking mode (proven less accurate for this task)
- Deterministic execution after planning
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
    Query analysis and execution planning using gemini-2.5-flash-lite.

    Based on comprehensive 4-model comparison testing:
    - gemini-2.5-flash-lite: 100% accuracy, 3.63s avg (WINNER)
    - gemini-2.5-flash: 100% accuracy, 9.39s avg
    - gemini-2.5-flash-lite+thinking: 81.8% accuracy
    - gemini-2.5-flash+thinking: 54.5% accuracy

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

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        streamer: Optional[ProgressStreamer] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Query Planner.

        Args:
            registry: Agent registry (uses default if not provided)
            streamer: Progress streamer for real-time updates
            api_key: Google API key (uses env var if not provided)
        """
        self.registry = registry or get_registry()
        self.streamer = streamer
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

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
            # Build the prompt
            prompt = self._build_planning_prompt(query, context)

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

    def _build_planning_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the complete prompt for the LLM"""

        # Get agent information from registry
        agents_context = self.registry.build_prompt_context(include_schema=True)

        # Build the full prompt
        prompt = f"""# 역할
당신은 사용자 쿼리를 분석하여 최적의 에이전트 워크플로우를 설계하는 전문가입니다.

# 사용 가능한 에이전트
{agents_context}

# 핵심 원칙

## 1. 데이터 흐름 원칙
- 실시간 데이터(주가, 환율, 뉴스)가 필요하면 → internet_agent 먼저
- 데이터 가공/분석이 필요하면 → analysis_agent
- 시각화(차트, 그래프)가 필요하면 → data_visualization_agent
- 일반 지식/설명이 필요하면 → llm_search_agent

## 2. 도메인 구분 원칙
- 삼성/반도체/FAB/NAND/수율 관련 → samsung_gateway_agent 우선
- 상품 검색/가격 비교 → shopping_agent
- 코드/프로그래밍 → code_agent
- 문서 검색 → rag_search_agent

## 3. 워크플로우 전략
- sequential: 이전 결과가 다음 작업에 필요할 때 (예: 데이터 수집 → 분석 → 시각화)
- parallel: 독립적인 작업들을 동시 처리할 때 (예: 두 회사 정보 동시 검색)
- hybrid: sequential과 parallel 혼합

## 4. 금지 사항
- 불필요한 에이전트 추가 금지 (필요한 에이전트만 선택)
- 같은 에이전트 중복 호출 금지 (한 스테이지 내에서)
- 데이터 의존성 무시 금지 (실시간 데이터 먼저 수집)

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

## 예시 2: "삼성전자와 애플 실적 비교"
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
          "sub_query": "삼성전자와 애플 실적 비교 분석",
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

# 사용자 쿼리
"{query}"

# 지시사항
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
