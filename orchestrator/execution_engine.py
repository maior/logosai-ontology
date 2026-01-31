"""
Execution Engine

Executes the validated plan with support for sequential, parallel,
and hybrid execution strategies.

Features:
- Stage-based execution (sequential within stage, parallel across agents)
- Automatic retry with exponential backoff
- Data transformation between agents
- Real-time progress streaming
- Graceful error handling
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    AgentTask,
    AgentResult,
    AgentStatus,
    ExecutionPlan,
    ExecutionStage,
    StageResult,
    WorkflowResult,
)
from .agent_registry import AgentRegistry, get_registry
from .data_transformer import DataTransformer
from .progress_streamer import ProgressStreamer
from .exceptions import (
    ExecutionError,
    AgentExecutionError,
    AgentTimeoutError,
)

logger = logging.getLogger(__name__)


# Type for agent executor function
AgentExecutor = Callable[[str, str, Optional[Dict[str, Any]]], Any]


class ExecutionEngine:
    """
    Executes validated execution plans with parallel/sequential support.

    Handles:
    - Sequential stage execution (stages run one after another)
    - Parallel/sequential agent execution within stages
    - Data transformation between agents
    - Retry logic for failed agents
    - Progress streaming for frontend visualization

    Example:
        engine = ExecutionEngine(agent_executor=my_executor)
        result = await engine.execute(plan)
    """

    DEFAULT_TIMEOUT_MS = 30000  # 30 seconds
    MAX_RETRIES = 2
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(
        self,
        agent_executor: Optional[AgentExecutor] = None,
        registry: Optional[AgentRegistry] = None,
        transformer: Optional[DataTransformer] = None,
        streamer: Optional[ProgressStreamer] = None,
    ):
        """
        Initialize Execution Engine.

        Args:
            agent_executor: Function to execute agents (agent_id, sub_query, context) -> result
            registry: Agent registry
            transformer: Data transformer for agent I/O
            streamer: Progress streamer for real-time updates
        """
        self.agent_executor = agent_executor or self._default_executor
        self.registry = registry or get_registry()
        self.transformer = transformer or DataTransformer(registry=self.registry)
        self.streamer = streamer

        # Execution state
        self._stage_results: Dict[int, StageResult] = {}
        self._agent_results: Dict[str, AgentResult] = {}

    async def execute(
        self,
        plan: ExecutionPlan,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Execute the given plan.

        Args:
            plan: Validated execution plan
            context: Optional execution context

        Returns:
            WorkflowResult with all stage and agent results
        """
        workflow_id = plan.plan_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(
            f"[ExecutionEngine] Starting workflow {workflow_id}: "
            f"{plan.get_stage_count()} stages, {plan.get_total_agents()} agents"
        )

        # Initialize result tracking
        self._stage_results = {}
        self._agent_results = {}

        # Emit workflow start
        if self.streamer:
            await self.streamer.workflow_start(plan.query)

        try:
            # Execute stages sequentially
            all_stage_results: List[StageResult] = []
            current_output: Any = None

            for stage in plan.stages:
                stage_result = await self._execute_stage(
                    stage=stage,
                    previous_output=current_output,
                    context=context,
                )

                all_stage_results.append(stage_result)
                self._stage_results[stage.stage_id] = stage_result

                # Update current output for next stage
                current_output = stage_result.aggregated_output

                # Stop on stage failure if critical
                if not stage_result.success:
                    logger.warning(
                        f"[ExecutionEngine] Stage {stage.stage_id} failed, "
                        f"continuing with partial results"
                    )

            # Calculate statistics
            total_time_ms = (time.time() - start_time) * 1000
            total_agents = sum(len(sr.results) for sr in all_stage_results)
            successful_agents = sum(
                len(sr.get_successful_results()) for sr in all_stage_results
            )
            failed_agents = total_agents - successful_agents

            # Build final output
            final_output = self._build_final_output(
                all_stage_results, plan.final_aggregation
            )

            # Determine overall success
            success = failed_agents == 0 or (
                successful_agents > 0 and final_output is not None
            )

            result = WorkflowResult(
                success=success,
                workflow_id=workflow_id,
                query=plan.query,
                stages=all_stage_results,
                final_output=final_output,
                start_time=datetime.now(),
                total_time_ms=total_time_ms,
                plan=plan,
                total_agents_executed=total_agents,
                successful_agents=successful_agents,
                failed_agents=failed_agents,
            )

            # Emit workflow complete
            if self.streamer:
                await self.streamer.workflow_complete(
                    success=success,
                    final_output=final_output,
                )

            logger.info(
                f"[ExecutionEngine] Workflow {workflow_id} completed in "
                f"{total_time_ms:.0f}ms: {successful_agents}/{total_agents} agents succeeded"
            )

            return result

        except Exception as e:
            logger.error(f"[ExecutionEngine] Workflow failed: {e}")

            # Emit workflow error
            if self.streamer:
                await self.streamer.workflow_complete(
                    success=False,
                    error=str(e),
                )

            raise ExecutionError(
                message=f"Workflow execution failed: {e}",
            )

    async def _execute_stage(
        self,
        stage: ExecutionStage,
        previous_output: Any,
        context: Optional[Dict[str, Any]],
    ) -> StageResult:
        """Execute a single stage"""
        stage_start = time.time()

        # Emit stage start
        agent_ids = [a.agent_id for a in stage.agents]
        if self.streamer:
            await self.streamer.stage_start(
                stage_id=stage.stage_id,
                execution_type=stage.execution_type,
                agent_ids=agent_ids,
            )

        logger.info(
            f"[ExecutionEngine] Stage {stage.stage_id} ({stage.execution_type}): "
            f"{len(stage.agents)} agents"
        )

        try:
            if stage.execution_type == "parallel":
                results = await self._execute_parallel(
                    stage=stage,
                    previous_output=previous_output,
                    context=context,
                )
            else:
                results = await self._execute_sequential(
                    stage=stage,
                    previous_output=previous_output,
                    context=context,
                )

            # Aggregate stage results
            stage_time = (time.time() - stage_start) * 1000
            success = all(r.success for r in results)

            # Create aggregated output for next stage
            if success:
                if len(results) == 1:
                    aggregated = results[0].data
                else:
                    aggregated = [r.data for r in results]
            else:
                # Include successful results even if some failed
                aggregated = [r.data for r in results if r.success]

            stage_result = StageResult(
                stage_id=stage.stage_id,
                execution_type=stage.execution_type,
                success=success,
                results=results,
                total_time_ms=stage_time,
                aggregated_output=aggregated,
            )

            # Emit stage complete
            if self.streamer:
                await self.streamer.stage_complete(
                    stage_id=stage.stage_id,
                    success=success,
                    results_count=len(results),
                )

            return stage_result

        except Exception as e:
            logger.error(f"[ExecutionEngine] Stage {stage.stage_id} failed: {e}")

            stage_time = (time.time() - stage_start) * 1000
            return StageResult(
                stage_id=stage.stage_id,
                execution_type=stage.execution_type,
                success=False,
                results=[],
                total_time_ms=stage_time,
            )

    async def _execute_parallel(
        self,
        stage: ExecutionStage,
        previous_output: Any,
        context: Optional[Dict[str, Any]],
    ) -> List[AgentResult]:
        """Execute agents in parallel"""
        tasks = []

        for agent_task in stage.agents:
            task = asyncio.create_task(
                self._execute_agent(
                    agent_task=agent_task,
                    stage_id=stage.stage_id,
                    previous_output=previous_output,
                    context=context,
                )
            )
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        agent_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Task raised exception
                agent_task = stage.agents[i]
                agent_results.append(
                    AgentResult(
                        agent_id=agent_task.agent_id,
                        stage_id=stage.stage_id,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                agent_results.append(result)

        return agent_results

    async def _execute_sequential(
        self,
        stage: ExecutionStage,
        previous_output: Any,
        context: Optional[Dict[str, Any]],
    ) -> List[AgentResult]:
        """Execute agents sequentially, passing output to next"""
        results = []
        current_input = previous_output

        for agent_task in stage.agents:
            result = await self._execute_agent(
                agent_task=agent_task,
                stage_id=stage.stage_id,
                previous_output=current_input,
                context=context,
            )
            results.append(result)

            # Pass output to next agent (for sequential chains)
            if result.success:
                current_input = result.data
            else:
                # Continue with None if agent failed
                current_input = None

        return results

    async def _execute_agent(
        self,
        agent_task: AgentTask,
        stage_id: int,
        previous_output: Any,
        context: Optional[Dict[str, Any]],
    ) -> AgentResult:
        """Execute a single agent with retry logic"""
        agent_id = agent_task.agent_id
        sub_query = agent_task.sub_query
        timeout_ms = agent_task.timeout_ms or self.DEFAULT_TIMEOUT_MS
        max_retries = agent_task.max_retries

        # Emit agent start
        agent_entry = self.registry.get_agent_safe(agent_id)
        display_name = agent_entry.display_name if agent_entry else agent_id

        if self.streamer:
            await self.streamer.agent_queued(
                agent_id=agent_id,
                stage_id=stage_id,
                sub_query=sub_query,
                display_name=display_name,
            )

        # Prepare input data
        input_data = previous_output
        if agent_task.input_from and previous_output is not None:
            # Transform data if needed
            for input_ref in agent_task.input_from:
                if "." in input_ref:
                    source_agent = input_ref.split(".")[-1]
                    input_data = await self.transformer.transform(
                        source_agent=source_agent,
                        target_agent=agent_id,
                        data=previous_output,
                    )
                    break

        # Execute with retry
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if self.streamer:
                    await self.streamer.agent_start(
                        agent_id=agent_id,
                        stage_id=stage_id,
                        sub_query=sub_query,
                        display_name=display_name,
                    )

                start_time = time.time()

                # Execute with timeout
                result = await asyncio.wait_for(
                    self._call_agent(agent_id, sub_query, input_data, context),
                    timeout=timeout_ms / 1000,
                )

                execution_time = (time.time() - start_time) * 1000

                # Create success result
                agent_result = AgentResult(
                    agent_id=agent_id,
                    stage_id=stage_id,
                    success=True,
                    data=result,
                    execution_time_ms=execution_time,
                    retry_count=attempt,
                )

                # Store result
                self._agent_results[f"stage_{stage_id}.{agent_id}"] = agent_result

                # Emit agent complete
                if self.streamer:
                    result_preview = str(result)[:100] if result else None
                    await self.streamer.agent_complete(
                        agent_id=agent_id,
                        stage_id=stage_id,
                        success=True,
                        result_preview=result_preview,
                        full_result=result,  # 전체 결과도 전달
                    )

                return agent_result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout_ms}ms"
                logger.warning(
                    f"[ExecutionEngine] Agent {agent_id} timeout (attempt {attempt + 1})"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[ExecutionEngine] Agent {agent_id} failed (attempt {attempt + 1}): {e}"
                )

            # Retry with delay
            if attempt < max_retries:
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)

                if self.streamer:
                    await self.streamer.agent_retry(
                        agent_id=agent_id,
                        stage_id=stage_id,
                        retry_count=attempt + 1,
                        reason=last_error,
                    )

                await asyncio.sleep(delay)

        # All retries failed
        agent_result = AgentResult(
            agent_id=agent_id,
            stage_id=stage_id,
            success=False,
            error=last_error,
            retry_count=max_retries,
        )

        self._agent_results[f"stage_{stage_id}.{agent_id}"] = agent_result

        # Emit agent error
        if self.streamer:
            await self.streamer.agent_complete(
                agent_id=agent_id,
                stage_id=stage_id,
                success=False,
                error=last_error,
            )

        return agent_result

    async def _call_agent(
        self,
        agent_id: str,
        sub_query: str,
        input_data: Any,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Call the actual agent executor"""
        # Build execution context
        exec_context = context.copy() if context else {}
        exec_context["input_data"] = input_data

        # 🔥 이전 단계의 결과를 sub_query에 포함하여 에이전트가 활용할 수 있도록 함
        enriched_query = self._enrich_query_with_input(sub_query, input_data, agent_id)

        # Call executor
        result = await self.agent_executor(agent_id, enriched_query, exec_context)

        return result

    def _extract_core_result(self, data: Any, depth: int = 0) -> str:
        """
        중첩된 데이터에서 핵심 결과(answer, result, content)를 재귀적으로 추출.

        Args:
            data: 추출할 데이터
            depth: 재귀 깊이 (무한 재귀 방지)

        Returns:
            핵심 결과 문자열 또는 빈 문자열
        """
        if depth > 5:  # 무한 재귀 방지
            return ""

        if data is None:
            return ""

        if isinstance(data, str):
            return data

        if isinstance(data, (int, float, bool)):
            return str(data)

        if isinstance(data, dict):
            # 1. 직접 answer 필드 확인
            if "answer" in data:
                answer = data["answer"]
                if isinstance(answer, str):
                    return answer
                return self._extract_core_result(answer, depth + 1)

            # 2. result 필드 확인 (중첩 가능)
            if "result" in data:
                result = data["result"]
                if isinstance(result, str):
                    return result
                extracted = self._extract_core_result(result, depth + 1)
                if extracted:
                    return extracted

            # 3. content 필드 확인
            if "content" in data:
                content = data["content"]
                if isinstance(content, str):
                    return content
                return self._extract_core_result(content, depth + 1)

            # 4. data 필드 확인 (중첩 가능)
            if "data" in data:
                inner_data = data["data"]
                extracted = self._extract_core_result(inner_data, depth + 1)
                if extracted:
                    return extracted

            # 5. text 필드 확인
            if "text" in data:
                text = data["text"]
                if isinstance(text, str):
                    return text

            # 6. 없으면 전체 JSON으로 변환
            try:
                return json.dumps(data, ensure_ascii=False, indent=2)
            except:
                return str(data)

        if isinstance(data, list):
            if len(data) == 1:
                return self._extract_core_result(data[0], depth + 1)
            try:
                return json.dumps(data, ensure_ascii=False, indent=2)
            except:
                return str(data)

        return str(data)

    def _enrich_query_with_input(
        self,
        sub_query: str,
        input_data: Any,
        agent_id: str,
    ) -> str:
        """
        이전 단계 결과를 sub_query에 통합.

        추상적인 sub_query (예: "계산 결과를 전달")를
        실제 데이터가 포함된 구체적인 쿼리로 변환합니다.
        """
        if input_data is None:
            return sub_query

        # input_data에서 핵심 결과 추출
        input_str = self._extract_core_result(input_data)

        if not input_str:
            return sub_query

        # 너무 긴 경우 잘라냄
        max_input_len = 2000
        if len(input_str) > max_input_len:
            input_str = input_str[:max_input_len] + "... (생략)"

        # 쿼리 구성
        enriched_query = f"""[이전 단계 결과]
{input_str}

[요청]
{sub_query}

위의 이전 단계 결과를 활용하여 요청에 응답해주세요."""

        logger.info(
            f"[ExecutionEngine] Enriched query for {agent_id}: "
            f"input_data 길이={len(input_str)}, 원래 쿼리='{sub_query[:50]}...'"
        )

        return enriched_query

    def _build_final_output(
        self,
        stage_results: List[StageResult],
        aggregation: Dict[str, Any],
    ) -> Any:
        """Build final output from all stage results"""
        if not stage_results:
            return None

        # Get last stage's output
        last_stage = stage_results[-1]

        if not last_stage.results:
            return None

        aggregation_type = aggregation.get("type", "combine")

        if aggregation_type == "single":
            # Return single result (usually visualization)
            if last_stage.results:
                return last_stage.results[-1].data
            return None

        elif aggregation_type == "combine":
            # Combine all results
            all_data = []
            for stage in stage_results:
                for result in stage.results:
                    if result.success and result.data:
                        all_data.append({
                            "agent_id": result.agent_id,
                            "stage_id": result.stage_id,
                            "data": result.data,
                        })
            return all_data

        elif aggregation_type == "last":
            # Return only last result
            return last_stage.aggregated_output

        else:
            # Default: return aggregated from last stage
            return last_stage.aggregated_output

    async def _default_executor(
        self,
        agent_id: str,
        sub_query: str,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Default agent executor (for testing)"""
        logger.warning(
            f"[ExecutionEngine] Using default executor for {agent_id}. "
            f"Provide a real executor for production use."
        )

        # Simulate execution
        await asyncio.sleep(0.5)

        return {
            "agent_id": agent_id,
            "query": sub_query,
            "result": f"Mock result from {agent_id}",
            "timestamp": datetime.now().isoformat(),
        }

    def get_agent_result(self, stage_id: int, agent_id: str) -> Optional[AgentResult]:
        """Get result for specific agent"""
        key = f"stage_{stage_id}.{agent_id}"
        return self._agent_results.get(key)

    def get_stage_result(self, stage_id: int) -> Optional[StageResult]:
        """Get result for specific stage"""
        return self._stage_results.get(stage_id)


# Factory function
def create_execution_engine(
    agent_executor: Optional[AgentExecutor] = None,
    registry: Optional[AgentRegistry] = None,
    streamer: Optional[ProgressStreamer] = None,
) -> ExecutionEngine:
    """Create an ExecutionEngine instance with default configuration"""
    return ExecutionEngine(
        agent_executor=agent_executor,
        registry=registry,
        streamer=streamer,
    )
