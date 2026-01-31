"""
Progress Streamer for Real-time Frontend Updates

Provides comprehensive streaming of workflow execution progress
via WebSocket, SSE, or callback mechanisms.

Key Features:
- Multiple streaming modes (WebSocket, SSE, Callback)
- Buffered events for reliability
- Automatic reconnection support
- Localized messages (Korean/English)
- Detailed timing information
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from collections import deque

from .models import (
    ProgressEvent,
    ProgressEventType,
    AgentStatus,
    ExecutionPlan,
    AgentTask,
    ExecutionStage,
)
from .exceptions import StreamingError

logger = logging.getLogger(__name__)


class ProgressStreamer:
    """
    Real-time progress streaming for workflow execution.

    Supports multiple streaming modes:
    - Callback: Direct function calls for each event
    - AsyncGenerator: Async iteration over events
    - SSE: Server-Sent Events format
    - WebSocket: For bidirectional communication

    Example usage:
        streamer = ProgressStreamer(workflow_id="abc123")

        # Callback mode
        async def handle_event(event):
            print(f"Event: {event.type} - {event.message}")

        streamer.on_event(handle_event)

        # Generator mode
        async for event in streamer.events():
            process_event(event)
    """

    # Default messages for each event type (English and Korean)
    DEFAULT_MESSAGES = {
        ProgressEventType.WORKFLOW_START: {
            "en": "Starting workflow execution",
            "ko": "워크플로우 실행 시작"
        },
        ProgressEventType.WORKFLOW_COMPLETE: {
            "en": "Workflow completed successfully",
            "ko": "워크플로우 완료"
        },
        ProgressEventType.WORKFLOW_ERROR: {
            "en": "Workflow execution failed",
            "ko": "워크플로우 실행 실패"
        },
        ProgressEventType.PLANNING_START: {
            "en": "Analyzing query and creating execution plan",
            "ko": "쿼리 분석 및 실행 계획 생성 중"
        },
        ProgressEventType.PLANNING_COMPLETE: {
            "en": "Execution plan created",
            "ko": "실행 계획 생성 완료"
        },
        ProgressEventType.VALIDATION_START: {
            "en": "Validating execution plan",
            "ko": "실행 계획 검증 중"
        },
        ProgressEventType.VALIDATION_COMPLETE: {
            "en": "Execution plan validated",
            "ko": "실행 계획 검증 완료"
        },
        ProgressEventType.STAGE_START: {
            "en": "Starting stage {stage_id}",
            "ko": "스테이지 {stage_id} 시작"
        },
        ProgressEventType.STAGE_COMPLETE: {
            "en": "Stage {stage_id} completed",
            "ko": "스테이지 {stage_id} 완료"
        },
        ProgressEventType.AGENT_QUEUED: {
            "en": "Agent '{agent_id}' queued for execution",
            "ko": "에이전트 '{agent_id}' 실행 대기 중"
        },
        ProgressEventType.AGENT_START: {
            "en": "Agent '{agent_id}' started processing",
            "ko": "에이전트 '{agent_id}' 처리 시작"
        },
        ProgressEventType.AGENT_PROGRESS: {
            "en": "Agent '{agent_id}' processing... {progress}%",
            "ko": "에이전트 '{agent_id}' 처리 중... {progress}%"
        },
        ProgressEventType.AGENT_COMPLETE: {
            "en": "Agent '{agent_id}' completed",
            "ko": "에이전트 '{agent_id}' 완료"
        },
        ProgressEventType.AGENT_ERROR: {
            "en": "Agent '{agent_id}' failed: {error}",
            "ko": "에이전트 '{agent_id}' 실패: {error}"
        },
        ProgressEventType.AGENT_RETRY: {
            "en": "Retrying agent '{agent_id}' (attempt {retry})",
            "ko": "에이전트 '{agent_id}' 재시도 중 ({retry}번째 시도)"
        },
        ProgressEventType.TRANSFORM_START: {
            "en": "Transforming data from '{source}' to '{target}'",
            "ko": "'{source}'에서 '{target}'로 데이터 변환 중"
        },
        ProgressEventType.TRANSFORM_COMPLETE: {
            "en": "Data transformation completed",
            "ko": "데이터 변환 완료"
        },
        ProgressEventType.AGGREGATION_START: {
            "en": "Aggregating results",
            "ko": "결과 통합 중"
        },
        ProgressEventType.AGGREGATION_COMPLETE: {
            "en": "Results aggregated",
            "ko": "결과 통합 완료"
        },
    }

    # Icons for different event types and agents
    ICONS = {
        ProgressEventType.WORKFLOW_START: "🚀",
        ProgressEventType.WORKFLOW_COMPLETE: "✅",
        ProgressEventType.WORKFLOW_ERROR: "❌",
        ProgressEventType.PLANNING_START: "🧠",
        ProgressEventType.PLANNING_COMPLETE: "📋",
        ProgressEventType.VALIDATION_START: "🔍",
        ProgressEventType.VALIDATION_COMPLETE: "✓",
        ProgressEventType.STAGE_START: "▶️",
        ProgressEventType.STAGE_COMPLETE: "✅",
        ProgressEventType.STAGE_ERROR: "❌",
        ProgressEventType.AGENT_QUEUED: "⏳",
        ProgressEventType.AGENT_START: "🔄",
        ProgressEventType.AGENT_PROGRESS: "⚡",
        ProgressEventType.AGENT_COMPLETE: "✅",
        ProgressEventType.AGENT_ERROR: "❌",
        ProgressEventType.AGENT_RETRY: "🔁",
        ProgressEventType.TRANSFORM_START: "🔄",
        ProgressEventType.TRANSFORM_COMPLETE: "✅",
        ProgressEventType.AGGREGATION_START: "📊",
        ProgressEventType.AGGREGATION_COMPLETE: "📈",
        ProgressEventType.INFO: "ℹ️",
        ProgressEventType.WARNING: "⚠️",
        ProgressEventType.DEBUG: "🔧",
    }

    # Agent-specific icons
    AGENT_ICONS = {
        "internet_agent": "🌐",
        "analysis_agent": "📊",
        "data_visualization_agent": "📈",
        "visualization_agent": "📈",
        "samsung_gateway_agent": "📱",
        "llm_search_agent": "🔍",
        "shopping_agent": "🛒",
        "code_agent": "💻",
        "weather_agent": "🌤️",
        "scheduler_agent": "📅",
    }

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        buffer_size: int = 100,
        include_debug: bool = False,
    ):
        """
        Initialize progress streamer.

        Args:
            workflow_id: Unique identifier for the workflow
            buffer_size: Maximum number of events to buffer
            include_debug: Whether to emit debug-level events
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.buffer_size = buffer_size
        self.include_debug = include_debug

        # Event storage
        self._event_buffer: deque = deque(maxlen=buffer_size)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._listeners: List[Callable[[ProgressEvent], Any]] = []
        self._async_listeners: List[Callable[[ProgressEvent], Any]] = []

        # Timing
        self._start_time: Optional[float] = None
        self._stage_start_times: Dict[int, float] = {}
        self._agent_start_times: Dict[str, float] = {}

        # State tracking
        self._current_stage: Optional[int] = None
        self._active_agents: Dict[str, AgentStatus] = {}
        self._completed_stages: List[int] = []

        # Plan reference for progress calculation
        self._plan: Optional[ExecutionPlan] = None
        self._total_agents: int = 0
        self._completed_agents: int = 0

        # Streaming control
        self._is_streaming: bool = False
        self._is_closed: bool = False

    def set_plan(self, plan: ExecutionPlan) -> None:
        """Set the execution plan for progress calculation"""
        self._plan = plan
        self._total_agents = plan.get_total_agents()

    def on_event(self, callback: Callable[[ProgressEvent], Any]) -> None:
        """Register a synchronous event callback"""
        self._listeners.append(callback)

    def on_event_async(self, callback: Callable[[ProgressEvent], Any]) -> None:
        """Register an asynchronous event callback"""
        self._async_listeners.append(callback)

    def remove_listener(self, callback: Callable) -> None:
        """Remove an event callback"""
        if callback in self._listeners:
            self._listeners.remove(callback)
        if callback in self._async_listeners:
            self._async_listeners.remove(callback)

    async def emit(self, event: ProgressEvent) -> None:
        """
        Emit a progress event to all listeners and buffers.

        Args:
            event: The progress event to emit
        """
        if self._is_closed:
            return

        # Skip debug events if not enabled
        if event.type == ProgressEventType.DEBUG and not self.include_debug:
            return

        # Ensure workflow_id is set
        event.workflow_id = self.workflow_id

        # Add icon if not set
        if not event.icon:
            if event.agent_id and event.agent_id in self.AGENT_ICONS:
                event.icon = self.AGENT_ICONS[event.agent_id]
            else:
                event.icon = self.ICONS.get(event.type, "📌")

        # Add to buffer
        self._event_buffer.append(event)

        # Add to async queue for generators
        await self._event_queue.put(event)

        # Call synchronous listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(f"Error in sync listener: {e}")

        # Call asynchronous listeners
        for listener in self._async_listeners:
            try:
                await listener(event)
            except Exception as e:
                logger.warning(f"Error in async listener: {e}")

        # Log the event
        self._log_event(event)

    def _log_event(self, event: ProgressEvent) -> None:
        """Log the event for debugging"""
        log_msg = f"[{event.type.value}] {event.icon} {event.message}"
        if event.stage_id is not None:
            log_msg = f"Stage {event.stage_id}: {log_msg}"
        if event.agent_id:
            log_msg = f"[{event.agent_id}] {log_msg}"

        if event.type in [ProgressEventType.WORKFLOW_ERROR, ProgressEventType.AGENT_ERROR]:
            logger.error(log_msg)
        elif event.type == ProgressEventType.WARNING:
            logger.warning(log_msg)
        elif event.type == ProgressEventType.DEBUG:
            logger.debug(log_msg)
        else:
            logger.info(log_msg)

    def _format_message(
        self,
        event_type: ProgressEventType,
        **kwargs
    ) -> tuple[str, str]:
        """Format message with placeholders replaced"""
        messages = self.DEFAULT_MESSAGES.get(event_type, {"en": "", "ko": ""})
        en_msg = messages.get("en", "").format(**kwargs)
        ko_msg = messages.get("ko", "").format(**kwargs)
        return en_msg, ko_msg

    def _calculate_elapsed_time(self) -> float:
        """Calculate elapsed time from workflow start in milliseconds"""
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) * 1000

    def _calculate_overall_progress(self) -> float:
        """Calculate overall workflow progress percentage"""
        if self._total_agents == 0:
            return 0.0
        return (self._completed_agents / self._total_agents) * 100

    # ========== High-level event emission methods ==========

    async def workflow_start(self, query: str) -> None:
        """Emit workflow start event"""
        self._start_time = time.time()
        self._is_streaming = True

        event = ProgressEvent(
            type=ProgressEventType.WORKFLOW_START,
            status=AgentStatus.RUNNING,
            message="Starting workflow execution",
            message_ko="워크플로우 실행 시작",
            data={"query": query},
            progress_percent=0.0,
        )
        await self.emit(event)

    async def workflow_complete(
        self,
        success: bool = True,
        final_output: Any = None,
        error: Optional[str] = None
    ) -> None:
        """Emit workflow completion event"""
        elapsed = self._calculate_elapsed_time()

        event = ProgressEvent(
            type=ProgressEventType.WORKFLOW_COMPLETE if success else ProgressEventType.WORKFLOW_ERROR,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            message="Workflow completed successfully" if success else f"Workflow failed: {error}",
            message_ko="워크플로우 완료" if success else f"워크플로우 실패: {error}",
            data={"final_output": final_output} if final_output else None,
            progress_percent=100.0 if success else self._calculate_overall_progress(),
            elapsed_time_ms=elapsed,
            error=error,
        )
        await self.emit(event)
        self._is_streaming = False

    async def planning_start(self, query: str) -> None:
        """Emit planning phase start"""
        event = ProgressEvent(
            type=ProgressEventType.PLANNING_START,
            status=AgentStatus.RUNNING,
            message="Analyzing query and creating execution plan",
            message_ko="쿼리 분석 및 실행 계획 생성 중",
            data={"query": query[:200]},  # Truncate long queries
            progress_percent=5.0,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def planning_complete(self, plan: ExecutionPlan) -> None:
        """Emit planning phase completion"""
        self.set_plan(plan)

        # Build stage summary for display
        stage_summary = []
        for stage in plan.stages:
            agent_names = [a.agent_id for a in stage.agents]
            stage_summary.append({
                "stage_id": stage.stage_id,
                "execution_type": stage.execution_type,
                "agents": agent_names,
            })

        event = ProgressEvent(
            type=ProgressEventType.PLANNING_COMPLETE,
            status=AgentStatus.COMPLETED,
            message=f"Execution plan created: {plan.get_stage_count()} stages, {plan.get_total_agents()} agents",
            message_ko=f"실행 계획 생성 완료: {plan.get_stage_count()}개 스테이지, {plan.get_total_agents()}개 에이전트",
            data={
                "workflow_strategy": plan.workflow_strategy,
                "stage_count": plan.get_stage_count(),
                "total_agents": plan.get_total_agents(),
                "stages": stage_summary,
                "reasoning": plan.reasoning,
            },
            progress_percent=10.0,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def planning_error(self, error: str) -> None:
        """Emit planning error"""
        event = ProgressEvent(
            type=ProgressEventType.PLANNING_ERROR,
            status=AgentStatus.FAILED,
            message=f"Failed to create execution plan: {error}",
            message_ko=f"실행 계획 생성 실패: {error}",
            error=error,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def validation_start(self) -> None:
        """Emit validation phase start"""
        event = ProgressEvent(
            type=ProgressEventType.VALIDATION_START,
            status=AgentStatus.RUNNING,
            message="Validating execution plan",
            message_ko="실행 계획 검증 중",
            progress_percent=12.0,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def validation_complete(self) -> None:
        """Emit validation phase completion"""
        event = ProgressEvent(
            type=ProgressEventType.VALIDATION_COMPLETE,
            status=AgentStatus.COMPLETED,
            message="Execution plan validated",
            message_ko="실행 계획 검증 완료",
            progress_percent=15.0,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def validation_error(self, errors: List[str]) -> None:
        """Emit validation error"""
        event = ProgressEvent(
            type=ProgressEventType.VALIDATION_ERROR,
            status=AgentStatus.FAILED,
            message=f"Plan validation failed: {len(errors)} errors",
            message_ko=f"실행 계획 검증 실패: {len(errors)}개 오류",
            data={"validation_errors": errors},
            error="; ".join(errors),
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def stage_start(
        self,
        stage_id: int,
        execution_type: str,
        agent_ids: List[str]
    ) -> None:
        """Emit stage start event"""
        self._current_stage = stage_id
        self._stage_start_times[stage_id] = time.time()

        # Calculate progress based on completed stages
        base_progress = 15.0  # Planning + validation
        stage_progress = (len(self._completed_stages) / max(self._plan.get_stage_count(), 1)) * 80 if self._plan else 0
        current_progress = base_progress + stage_progress

        event = ProgressEvent(
            type=ProgressEventType.STAGE_START,
            stage_id=stage_id,
            status=AgentStatus.RUNNING,
            message=f"Starting stage {stage_id} ({execution_type}): {len(agent_ids)} agent(s)",
            message_ko=f"스테이지 {stage_id} 시작 ({execution_type}): {len(agent_ids)}개 에이전트",
            data={
                "execution_type": execution_type,
                "agents": agent_ids,
            },
            progress_percent=current_progress,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def stage_complete(
        self,
        stage_id: int,
        success: bool = True,
        results_count: int = 0
    ) -> None:
        """Emit stage completion event"""
        self._completed_stages.append(stage_id)

        stage_elapsed = 0.0
        if stage_id in self._stage_start_times:
            stage_elapsed = (time.time() - self._stage_start_times[stage_id]) * 1000

        # Calculate updated progress
        base_progress = 15.0
        stage_progress = (len(self._completed_stages) / max(self._plan.get_stage_count(), 1)) * 80 if self._plan else 0
        current_progress = min(base_progress + stage_progress, 95.0)

        event = ProgressEvent(
            type=ProgressEventType.STAGE_COMPLETE if success else ProgressEventType.STAGE_ERROR,
            stage_id=stage_id,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            message=f"Stage {stage_id} completed ({results_count} results)",
            message_ko=f"스테이지 {stage_id} 완료 ({results_count}개 결과)",
            data={"results_count": results_count},
            progress_percent=current_progress,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def agent_queued(
        self,
        agent_id: str,
        stage_id: int,
        sub_query: str,
        display_name: Optional[str] = None
    ) -> None:
        """Emit agent queued event"""
        self._active_agents[agent_id] = AgentStatus.QUEUED

        event = ProgressEvent(
            type=ProgressEventType.AGENT_QUEUED,
            stage_id=stage_id,
            agent_id=agent_id,
            status=AgentStatus.QUEUED,
            message=f"Agent '{agent_id}' queued",
            message_ko=f"에이전트 '{agent_id}' 대기 중",
            data={"sub_query": sub_query[:100]},
            display_name=display_name or agent_id,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def agent_start(
        self,
        agent_id: str,
        stage_id: int,
        sub_query: str,
        display_name: Optional[str] = None
    ) -> None:
        """Emit agent start event"""
        self._active_agents[agent_id] = AgentStatus.RUNNING
        self._agent_start_times[agent_id] = time.time()

        event = ProgressEvent(
            type=ProgressEventType.AGENT_START,
            stage_id=stage_id,
            agent_id=agent_id,
            status=AgentStatus.RUNNING,
            message=f"Agent '{agent_id}' started",
            message_ko=f"에이전트 '{agent_id}' 시작",
            data={"sub_query": sub_query[:100]},
            display_name=display_name or agent_id,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def agent_progress(
        self,
        agent_id: str,
        stage_id: int,
        progress_percent: float,
        message: Optional[str] = None
    ) -> None:
        """Emit agent progress update (for long-running agents)"""
        event = ProgressEvent(
            type=ProgressEventType.AGENT_PROGRESS,
            stage_id=stage_id,
            agent_id=agent_id,
            status=AgentStatus.RUNNING,
            message=message or f"Agent '{agent_id}' processing... {progress_percent:.0f}%",
            message_ko=message or f"에이전트 '{agent_id}' 처리 중... {progress_percent:.0f}%",
            progress_percent=progress_percent,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def agent_complete(
        self,
        agent_id: str,
        stage_id: int,
        success: bool = True,
        result_preview: Optional[str] = None,
        full_result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """Emit agent completion event"""
        self._active_agents[agent_id] = AgentStatus.COMPLETED if success else AgentStatus.FAILED
        self._completed_agents += 1

        agent_elapsed = 0.0
        if agent_id in self._agent_start_times:
            agent_elapsed = (time.time() - self._agent_start_times[agent_id]) * 1000

        # data에 preview와 full_result 모두 포함
        event_data = {
            "result_preview": result_preview[:200] if result_preview else None,
        }
        # full_result가 있으면 포함 (프론트엔드에서 전체 결과 사용 가능)
        if full_result is not None:
            event_data["full_result"] = full_result

        event = ProgressEvent(
            type=ProgressEventType.AGENT_COMPLETE if success else ProgressEventType.AGENT_ERROR,
            stage_id=stage_id,
            agent_id=agent_id,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            message=f"Agent '{agent_id}' completed in {agent_elapsed:.0f}ms" if success else f"Agent '{agent_id}' failed: {error}",
            message_ko=f"에이전트 '{agent_id}' 완료 ({agent_elapsed:.0f}ms)" if success else f"에이전트 '{agent_id}' 실패: {error}",
            data=event_data,
            progress_percent=self._calculate_overall_progress(),
            elapsed_time_ms=self._calculate_elapsed_time(),
            error=error,
        )
        await self.emit(event)

    async def agent_retry(
        self,
        agent_id: str,
        stage_id: int,
        retry_count: int,
        reason: str
    ) -> None:
        """Emit agent retry event"""
        self._active_agents[agent_id] = AgentStatus.RETRYING

        event = ProgressEvent(
            type=ProgressEventType.AGENT_RETRY,
            stage_id=stage_id,
            agent_id=agent_id,
            status=AgentStatus.RETRYING,
            message=f"Retrying agent '{agent_id}' (attempt {retry_count}): {reason}",
            message_ko=f"에이전트 '{agent_id}' 재시도 ({retry_count}번째): {reason}",
            data={"retry_count": retry_count, "reason": reason},
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def transform_start(
        self,
        source_agent: str,
        target_agent: str
    ) -> None:
        """Emit data transformation start"""
        event = ProgressEvent(
            type=ProgressEventType.TRANSFORM_START,
            status=AgentStatus.RUNNING,
            message=f"Transforming data from '{source_agent}' to '{target_agent}'",
            message_ko=f"'{source_agent}'에서 '{target_agent}'로 데이터 변환 중",
            data={"source": source_agent, "target": target_agent},
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def transform_complete(
        self,
        source_agent: str,
        target_agent: str,
        success: bool = True
    ) -> None:
        """Emit data transformation completion"""
        event = ProgressEvent(
            type=ProgressEventType.TRANSFORM_COMPLETE if success else ProgressEventType.TRANSFORM_ERROR,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            message="Data transformation completed" if success else "Data transformation failed",
            message_ko="데이터 변환 완료" if success else "데이터 변환 실패",
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def aggregation_start(self, aggregation_type: str) -> None:
        """Emit result aggregation start"""
        event = ProgressEvent(
            type=ProgressEventType.AGGREGATION_START,
            status=AgentStatus.RUNNING,
            message=f"Aggregating results ({aggregation_type})",
            message_ko=f"결과 통합 중 ({aggregation_type})",
            data={"aggregation_type": aggregation_type},
            progress_percent=95.0,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def aggregation_complete(self, result_type: str) -> None:
        """Emit result aggregation completion"""
        event = ProgressEvent(
            type=ProgressEventType.AGGREGATION_COMPLETE,
            status=AgentStatus.COMPLETED,
            message=f"Results aggregated ({result_type})",
            message_ko=f"결과 통합 완료 ({result_type})",
            progress_percent=98.0,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def info(self, message: str, message_ko: Optional[str] = None, data: Optional[Dict] = None) -> None:
        """Emit informational event"""
        event = ProgressEvent(
            type=ProgressEventType.INFO,
            message=message,
            message_ko=message_ko or message,
            data=data,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def warning(self, message: str, message_ko: Optional[str] = None, data: Optional[Dict] = None) -> None:
        """Emit warning event"""
        event = ProgressEvent(
            type=ProgressEventType.WARNING,
            message=message,
            message_ko=message_ko or message,
            data=data,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    async def debug(self, message: str, data: Optional[Dict] = None) -> None:
        """Emit debug event (only if include_debug is True)"""
        event = ProgressEvent(
            type=ProgressEventType.DEBUG,
            message=message,
            message_ko=message,
            data=data,
            elapsed_time_ms=self._calculate_elapsed_time(),
        )
        await self.emit(event)

    # ========== Streaming generators ==========

    async def events(self) -> AsyncGenerator[ProgressEvent, None]:
        """
        Async generator for streaming events.

        Usage:
            async for event in streamer.events():
                process_event(event)
        """
        while not self._is_closed:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.5
                )
                yield event
            except asyncio.TimeoutError:
                # Check if we should stop
                if not self._is_streaming:
                    break
                continue

    async def sse_stream(self) -> AsyncGenerator[str, None]:
        """
        Generator for Server-Sent Events format.

        Usage with FastAPI:
            @app.get("/stream")
            async def stream():
                return StreamingResponse(
                    streamer.sse_stream(),
                    media_type="text/event-stream"
                )
        """
        async for event in self.events():
            yield event.to_sse()

    async def json_stream(self) -> AsyncGenerator[str, None]:
        """
        Generator for newline-delimited JSON format.
        """
        async for event in self.events():
            yield event.to_json() + "\n"

    # ========== Utility methods ==========

    def get_buffered_events(self) -> List[ProgressEvent]:
        """Get all buffered events (for replay)"""
        return list(self._event_buffer)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status summary"""
        return {
            "workflow_id": self.workflow_id,
            "is_streaming": self._is_streaming,
            "current_stage": self._current_stage,
            "completed_stages": self._completed_stages,
            "active_agents": {k: v.value for k, v in self._active_agents.items()},
            "progress_percent": self._calculate_overall_progress(),
            "elapsed_time_ms": self._calculate_elapsed_time(),
            "events_count": len(self._event_buffer),
        }

    async def close(self) -> None:
        """Close the streamer and cleanup"""
        self._is_closed = True
        self._is_streaming = False
        self._listeners.clear()
        self._async_listeners.clear()

    def __aiter__(self):
        return self.events()
