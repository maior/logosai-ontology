"""
Query Planner — Logos 운영 플래너.

Uses gemini-2.5-flash-lite (non-thinking) for single-call execution planning.
Based on 4-model comparison test: Flash-Lite achieves 100% accuracy at 3.63s avg.

계획 흐름(메커니즘)은 logosai.orchestration.planner 가 정본이다 (2026-10-04,
orchestrator-unify P2). 이 모듈은 그 기반 클래스를 상속해 Logos 의 지식을 훅으로 넣는다:
  _build_planning_prompt  Logos 라우팅 규칙·예시가 담긴 프롬프트
  _call_llm               Gemini 직접 호출 (모델·온도·재시도)
  _recommend              HybridAgentSelector (GNN+RL) 힌트
  _apply_exclusion_gate   description 의 "대상이 아닙니다" 배제 관문
  _explicit_gap           "에이전트 만들어줘" 류 키워드 안전망
옮기기 전후로 LLM 에 보내는 프롬프트와 최종 계획이 같음을 tests/test_planner_golden.py 가 확인한다.

Key Design Principles:
- Single LLM call for complete planning
- No thinking mode (proven less accurate for this task)
- Deterministic execution after planning
- HybridAgentSelector (GNN+RL) for intelligent agent selection (v3.0)
"""

import asyncio
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

__sdk_base__ = "logosai.orchestration.planner"
try:
    from logosai.orchestration.planner import (  # noqa: F401 — 공개 이름 재노출
        QueryPlanner as _SDKQueryPlanner,
        backfill_gap_for_empty_plan,
        critique_plan,
        merge_independent_stages,
        normalize_capability_gap,
    )
except ModuleNotFoundError as _e:
    if _e.name not in ("logosai", "logosai.orchestration", "logosai.orchestration.planner"):
        raise
    raise ImportError(
        "ontology.orchestrator 의 계획 흐름은 logosai.orchestration.planner 로 옮겨졌다 — "
        "`pip install logosai-ontology[logosai]` 로 logosai 를 설치하라."
    ) from _e


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
        "required_resources": [],  # unknown from an explicit-creation phrase; LLM/FORGE refine later
        "suggested_agent_description": query,
        "reason": "사용자가 명시적으로 새 에이전트 생성을 요청 (code safety net)",
    }


EXCLUSION_MARKER = "대상이 아닙니다"


async def enforce_exclusion_gate(
    query: str,
    plan_data: Dict[str, Any],
    agent_descriptions: Dict[str, str],
    llm_invoke,
) -> tuple:
    """배제 관문 (2026-07-11): 계획에 선택된 에이전트의 description 배제 조건
    ("~는 대상이 아닙니다") 위반을 좁은 단일 LLM 판정으로 집행한다.

    배경: C-3 프롬프트 규칙 + 추천 기각 단서에도 flash-lite 가 temperature 0
    에서 배제를 무시 (점자→모스, 추천 주입 없이도 재현). 계획 전체를 시키면
    무시하지만 좁은 매칭 질문은 안정적 — find_equivalent 관문과 동일 원리.

    Returns:
        (plan_data, violations) — 위반 에이전트는 stages 에서 제거되고,
        남는 stage 가 없으면 capability_gap 을 강제한다 (기존 detected=true
        는 보존). LLM 에러는 fail-open (계획을 막지 않음).
    """
    stages = plan_data.get("stages") or []
    # 배제 마커 보유 에이전트만 검사 (대부분의 쿼리는 LLM 콜 0회)
    candidates: Dict[str, str] = {}
    for st in stages:
        for ag in st.get("agents", []):
            aid = ag.get("agent_id", "")
            desc = agent_descriptions.get(aid, "") or ""
            if EXCLUSION_MARKER in desc:
                candidates[aid] = desc

    violations: List[str] = []
    for aid, desc in candidates.items():
        prompt = (
            "다음 에이전트 설명에는 배제 조건이 명시되어 있습니다.\n"
            f"에이전트 설명: {desc}\n\n"
            f"사용자 요청: {query}\n\n"
            "이 요청이 설명의 배제 조건('~는 대상이 아닙니다'에 해당하는 작업)에 "
            "해당합니까? 반드시 '해당' 또는 '무관' 한 단어로만 답하세요."
        )
        try:
            answer = str(await llm_invoke(prompt)).strip()
        except Exception as e:  # 관문 실패는 계획을 막지 않는다 (fail-open)
            logger.warning(f"[ExclusionGate] 판정 실패 ({aid}): {e}")
            continue
        if answer.startswith("해당"):
            violations.append(aid)
            logger.info(f"[ExclusionGate] 배제 위반 기각: {aid} (query: {query[:40]})")

    if violations:
        new_stages = []
        for st in stages:
            kept = [a for a in st.get("agents", []) if a.get("agent_id") not in violations]
            if kept:
                new_stages.append({**st, "agents": kept})
        plan_data["stages"] = new_stages
        existing = plan_data.get("capability_gap")
        if not new_stages and not (isinstance(existing, dict) and existing.get("detected")):
            plan_data["capability_gap"] = {
                "detected": True,
                "missing_capabilities": ["excluded_capability"],
                "required_resources": [],
                "suggested_agent_description": query,
                "reason": f"선택 후보가 description 배제 조건 위반으로 기각됨: {violations}",
            }
    return plan_data, violations




class QueryPlanner(_SDKQueryPlanner):
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
        super().__init__(registry=registry, streamer=streamer)
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

    async def _recommend(self, query: str):
        """Phase 0: GNN+RL Agent Selection (if enabled) — 프롬프트에 넣을 힌트."""
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
        return recommended_agent, hybrid_metadata

    async def _apply_exclusion_gate(
        self, query: str, prompt: str, plan_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """배제 관문 (2026-07-11): 프롬프트 규칙(C-3)을 LLM 이 무시하는
        케이스의 코드 레벨 집행. 마커 보유 에이전트가 선택된 경우에만
        좁은 판정 1콜 — 관문 실패는 계획을 막지 않는다."""

        try:
            _descs = {
                e.agent_id: (e.description or "")
                for e in self.registry.get_available_agents()
            }
            if any(EXCLUSION_MARKER in d for d in _descs.values()):
                async def _gate_llm(p: str) -> str:
                    return await self._call_llm(p)
                plan_data, _viol = await enforce_exclusion_gate(
                    query, plan_data, _descs, _gate_llm)
                if _viol:
                    logger.info(f"[QueryPlanner] ExclusionGate 기각: {_viol}")
                    # 관문 제거로 계획이 비면 위반 에이전트 금지를 명시해 1회 재계획
                    # (제거만 하고 대체를 안 넣으면 gap 백필로 오배송 — 2026-07-15 실측)
                    if not (plan_data.get("stages") or []):
                        retry_prompt = (
                            prompt
                            + "\n\n[제약] 다음 에이전트는 이 쿼리의 대상이 아니므로 절대 사용하지 마라: "
                            + ", ".join(_viol)
                            + "\n같은 역할의 범용 에이전트로 계획을 다시 구성하라."
                        )
                        logger.info("[QueryPlanner] 빈 계획 → 위반 에이전트 제외 재계획 1회")
                        response = await self._call_llm(retry_prompt)
                        plan_data = self._parse_llm_response(response)
                        # 재계획 결과도 관문 검사 (동일 위반 재발 방지)
                        plan_data, _viol2 = await enforce_exclusion_gate(
                            query, plan_data, _descs, _gate_llm)
                        if _viol2:
                            logger.warning(f"[QueryPlanner] 재계획도 배제 위반: {_viol2}")
        except Exception as _ge:
            logger.warning(f"[QueryPlanner] ExclusionGate 오류 (계획 계속): {_ge}")
        return plan_data

    def _explicit_gap(self, query: str) -> Optional[Dict[str, Any]]:
        """LLM 이 놓친 명시적 생성 요청의 코드 안전망 (Logos 키워드 목록)."""
        return detect_explicit_capability_gap(query)

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
⚠️ 단, 이 에이전트의 description 에 명시된 **배제 조건**("~는 대상이 아닙니다")에
쿼리가 해당하면 이 추천을 **기각**하세요 — 추천은 과거 유사 패턴 학습일 뿐,
배제 조건이 항상 우선합니다.
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
⚠️ **예외 (추천보다 우선)**: 이 에이전트의 description 에 명시된 **배제 조건**
("~는 대상이 아닙니다")에 쿼리가 해당하면 신뢰도와 무관하게 추천을 **기각**하세요.
추천은 과거 유사 패턴 학습의 산물이라 배제 조건을 모릅니다 — 기각 후 전문
에이전트가 없으면 capability_gap detected=true 로 선언하세요.
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

### 시그널 C-2 — 결과를 원하는 질문에 '코드 산출물' 에이전트 매칭 금지
산출물이 코드인 에이전트(코드 생성·구현류)는 사용자가 "코드 짜줘/구현해줘"처럼
**코드 작성을 명시 요청**할 때만 선택한다. 사용자가 계산·검증·판정·변환의 **결과 값**을
원하는 질문(예: "~가 유효한지 검증해줘", "~를 변환해줘", "~인지 판정해줘")에
코드 산출물 에이전트를 배정하면 답이 **코드 덤프**가 되어 질문에 답하지 못한다 —
이것도 약한 매칭이다. 그 기능의 전문 에이전트가 등록돼 있으면 그쪽을 쓰고,
없으면 detected=true 로 선언한다. (사용자가 코드를 작성해 달라고 한 경우는
정상적으로 코드 생성 에이전트를 쓴다.)

### 시그널 C-3 — description 의 배제 조건 존중 (CRITICAL)
에이전트 description 에 "~는 대상이 아닙니다" 같은 **배제 조건**이 명시돼 있으면
그 배제를 반드시 존중한다. 배제된 작업을 그 에이전트에 배정하는 것은
"비슷해 보인다"는 이유의 약한 매칭이다 (예: 점자 변환을 '모스 부호 전용,
점자는 대상 아님' 에이전트에 배정 — 결과는 오답). 배제를 피해 갈 전문
에이전트가 없으면 detected=true 로 선언한다.

### 출력 형식
```
{{
  "workflow_strategy": "sequential",
  "stages": [],
  "capability_gap": {{
    "detected": true,
    "missing_capabilities": ["mastodon_api", "sentiment_analysis"],
    "required_resources": ["api:mastodon"],
    "suggested_agent_description": "Mastodon API 로 toot fetch + sentiment 분석",
    "reason": "Mastodon 전용 에이전트 부재, internet_agent 의 일반 검색으로 대체 불가"
  }},

### required_resources (배치 힌트)
새 에이전트가 **런타임에 의존할 자원**을 태그로 명시한다 (없으면 `[]`). logos_api 가 이 태그로
자원을 갖춘 ACP 노드에 에이전트를 배치한다. 형식:
- `api:<name>` — 외부 API 의존 (예: `api:mastodon`, `api:stripe`)
- `desktop:<app>` — 데스크톱 앱 필요 (예: `desktop:kakaotalk`)
- `region:<code>` — 지역 제약 (예: `region:kr`)
- `db:<name>` — 특정 DB 접근 (예: `db:logosus`)
특정 에이전트 이름을 넣지 말 것 — 자원 태그만.
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
6. **사용자가 명시한 조건(특히 시간 범위)은 sub_query에서 삭제·축소하지 마세요** (CRITICAL)
   - 사용자가 직접 말한 시간 범위(이번주/오늘/내일/이번달/올해/지난주/최근 N일 등)는 그대로 보존
   - "이번주 날씨" → sub_query: "이번주 날씨" (O) — "현재 날씨" (X, 사용자가 말한 '이번주'를 임의로 '현재'로 좁힘)
   - "오늘 환율" → sub_query: "오늘 환율" (O) — "현재 환율" (X)
   - '현재'는 사용자가 시간을 명시하지 않았고 대화 맥락상 최신 값이 필요할 때만 추론 (예: 주식 후속질문 "현재가는?" → "삼성전자 현재 주가")
   - 5번(명시 안 한 조건 추가 금지)과 6번(명시한 조건 삭제 금지)은 하나의 원칙 — 사용자 의도를 그대로 유지하라는 것

위 쿼리에 대한 실행 계획을 JSON 형식으로 작성하세요.
반드시 위 JSON 형식을 정확히 따르세요.
JSON 외에 다른 텍스트는 포함하지 마세요.

```json
"""
        return prompt

    # 일시 오류(수요 스파이크) 백오프 — 503 은 수십 초 지속 실측 (2026-07-14)
    _TRANSIENT_RETRY_DELAYS = (1.5, 3.0, 6.0)
    _TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")

    async def _call_llm(self, prompt: str) -> str:
        """Call Gemini API (일시 오류는 백오프 재시도)"""
        config = types.GenerateContentConfig(
            temperature=self.TEMPERATURE,
            max_output_tokens=self.MAX_TOKENS,
        )

        last_error: Exception = RuntimeError("no attempt")
        for attempt, delay in enumerate((0,) + self._TRANSIENT_RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                # 동기 API 를 스레드로 — 그대로 부르면 호출 내내 이벤트 루프가 멈춘다
                # (실측 0.92s 정지). logos_api 는 루프 하나로 모든 사용자를 처리한다.
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.MODEL,
                    config=config,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                msg = str(e)
                if not any(m in msg for m in self._TRANSIENT_MARKERS):
                    logger.error(f"[QueryPlanner] Gemini API call failed: {e}")
                    raise
                logger.warning(
                    f"[QueryPlanner] 일시 오류 (attempt {attempt + 1}/"
                    f"{len(self._TRANSIENT_RETRY_DELAYS) + 1}): {msg[:120]}"
                )

        logger.error(f"[QueryPlanner] Gemini API call failed after retries: {last_error}")
        raise last_error



# Factory function for creating QueryPlanner
def create_query_planner(
    registry: Optional[AgentRegistry] = None,
    streamer: Optional[ProgressStreamer] = None,
) -> QueryPlanner:
    """Create a QueryPlanner instance with default configuration"""
    return QueryPlanner(registry=registry, streamer=streamer)
