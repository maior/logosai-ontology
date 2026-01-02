"""
Result Aggregator

Aggregates results from multiple agents into a final output.

Aggregation Strategies:
1. combine: Simple merge of all results (no LLM)
2. summarize: LLM-generated summary of results
3. format: Apply specific formatting to results
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .models import StageResult, WorkflowResult, AgentResult
from .progress_streamer import ProgressStreamer
from .exceptions import AggregationError

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates results from multiple agents.

    Supports multiple aggregation strategies:
    - combine: Merge results (fast, no LLM)
    - summarize: Generate summary with LLM
    - format: Apply specific output format

    Example:
        aggregator = ResultAggregator()
        final = await aggregator.aggregate(
            results=workflow_result.stages,
            strategy={"type": "combine", "format": "json"}
        )
    """

    def __init__(
        self,
        streamer: Optional[ProgressStreamer] = None,
        enable_llm: bool = True,
    ):
        """
        Initialize Result Aggregator.

        Args:
            streamer: Progress streamer for status updates
            enable_llm: Whether to use LLM for summarization
        """
        self.streamer = streamer
        self.enable_llm = enable_llm

    async def aggregate(
        self,
        stages: List[StageResult],
        strategy: Dict[str, Any],
        original_query: Optional[str] = None,
    ) -> Any:
        """
        Aggregate results from all stages.

        Args:
            stages: List of stage results
            strategy: Aggregation strategy configuration
            original_query: Original user query (for context)

        Returns:
            Aggregated result
        """
        aggregation_type = strategy.get("type", "combine")
        output_format = strategy.get("format", "auto")

        # Emit aggregation start
        if self.streamer:
            await self.streamer.aggregation_start(aggregation_type)

        try:
            if aggregation_type == "single":
                result = self._aggregate_single(stages)
            elif aggregation_type == "combine":
                result = self._aggregate_combine(stages, output_format)
            elif aggregation_type == "summarize":
                result = await self._aggregate_summarize(stages, original_query)
            elif aggregation_type == "format":
                result = self._aggregate_format(stages, output_format)
            elif aggregation_type == "comparison_report":
                result = self._aggregate_comparison(stages)
            elif aggregation_type == "chart":
                result = self._aggregate_chart(stages)
            else:
                # Default to combine
                result = self._aggregate_combine(stages, output_format)

            # Emit aggregation complete
            if self.streamer:
                await self.streamer.aggregation_complete(output_format)

            return result

        except Exception as e:
            logger.error(f"[ResultAggregator] Aggregation failed: {e}")
            raise AggregationError(
                message=f"Result aggregation failed: {e}",
                aggregation_type=aggregation_type,
            )

    def _aggregate_single(self, stages: List[StageResult]) -> Any:
        """Return single result from last stage"""
        if not stages:
            return None

        last_stage = stages[-1]
        if not last_stage.results:
            return None

        # Get the last successful result
        for result in reversed(last_stage.results):
            if result.success and result.data:
                return result.data

        return None

    def _aggregate_combine(
        self,
        stages: List[StageResult],
        output_format: str,
    ) -> Any:
        """Combine all results"""
        all_results = []

        for stage in stages:
            for result in stage.results:
                if result.success and result.data:
                    all_results.append({
                        "agent_id": result.agent_id,
                        "stage_id": result.stage_id,
                        "data": result.data,
                        "execution_time_ms": result.execution_time_ms,
                    })

        if not all_results:
            return None

        # Format based on output type
        if output_format == "json":
            return all_results
        elif output_format == "text":
            return self._results_to_text(all_results)
        elif output_format == "markdown":
            return self._results_to_markdown(all_results)
        else:
            # Auto-detect
            if len(all_results) == 1:
                return all_results[0]["data"]
            return all_results

    async def _aggregate_summarize(
        self,
        stages: List[StageResult],
        original_query: Optional[str],
    ) -> str:
        """Generate LLM summary of results"""
        if not self.enable_llm:
            # Fallback to combine
            return self._results_to_text(
                self._collect_all_results(stages)
            )

        # Collect all results
        all_results = self._collect_all_results(stages)

        # Build summary prompt
        results_text = self._results_to_text(all_results)

        # TODO: Implement LLM summarization
        # For now, return formatted text
        summary = f"# 결과 요약\n\n"
        if original_query:
            summary += f"**질문**: {original_query}\n\n"
        summary += f"**처리된 에이전트**: {len(all_results)}개\n\n"
        summary += results_text

        return summary

    def _aggregate_format(
        self,
        stages: List[StageResult],
        output_format: str,
    ) -> Any:
        """Format results in specific format"""
        all_results = self._collect_all_results(stages)

        if output_format == "html":
            return self._results_to_html(all_results)
        elif output_format == "markdown":
            return self._results_to_markdown(all_results)
        elif output_format == "text":
            return self._results_to_text(all_results)
        else:
            return all_results

    def _aggregate_comparison(
        self,
        stages: List[StageResult],
    ) -> Dict[str, Any]:
        """Create comparison report from results"""
        all_results = self._collect_all_results(stages)

        comparison = {
            "type": "comparison_report",
            "items": [],
            "summary": "",
        }

        for result in all_results:
            data = result.get("data", {})
            if isinstance(data, dict):
                comparison["items"].append({
                    "source": result.get("agent_id"),
                    "data": data,
                })
            else:
                comparison["items"].append({
                    "source": result.get("agent_id"),
                    "content": str(data),
                })

        comparison["summary"] = f"{len(comparison['items'])}개 항목 비교"

        return comparison

    def _aggregate_chart(
        self,
        stages: List[StageResult],
    ) -> Any:
        """Extract chart data from results"""
        # Look for visualization result
        for stage in reversed(stages):
            for result in stage.results:
                if result.success and result.data:
                    # Check if it's chart data
                    if isinstance(result.data, dict):
                        if any(k in result.data for k in ["chart", "svg", "html"]):
                            return result.data
                    elif isinstance(result.data, str):
                        if "<svg" in result.data or "<html" in result.data:
                            return result.data

        # Fallback to last result
        return self._aggregate_single(stages)

    def _collect_all_results(
        self,
        stages: List[StageResult],
    ) -> List[Dict[str, Any]]:
        """Collect all successful results"""
        all_results = []

        for stage in stages:
            for result in stage.results:
                if result.success and result.data:
                    all_results.append({
                        "agent_id": result.agent_id,
                        "stage_id": result.stage_id,
                        "data": result.data,
                    })

        return all_results

    def _results_to_text(self, results: List[Dict[str, Any]]) -> str:
        """Convert results to plain text"""
        lines = []

        for i, result in enumerate(results, 1):
            agent_id = result.get("agent_id", "unknown")
            data = result.get("data", "")

            lines.append(f"## {i}. {agent_id}")

            if isinstance(data, dict):
                lines.append(json.dumps(data, ensure_ascii=False, indent=2))
            elif isinstance(data, list):
                lines.append(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                lines.append(str(data))

            lines.append("")

        return "\n".join(lines)

    def _results_to_markdown(self, results: List[Dict[str, Any]]) -> str:
        """Convert results to Markdown format"""
        md_lines = ["# 실행 결과\n"]

        for i, result in enumerate(results, 1):
            agent_id = result.get("agent_id", "unknown")
            stage_id = result.get("stage_id", 0)
            data = result.get("data", "")

            md_lines.append(f"## {i}. {agent_id} (Stage {stage_id})\n")

            if isinstance(data, dict):
                md_lines.append("```json")
                md_lines.append(json.dumps(data, ensure_ascii=False, indent=2))
                md_lines.append("```")
            elif isinstance(data, str):
                if "<svg" in data or "<html" in data:
                    md_lines.append("```html")
                    md_lines.append(data[:500] + "..." if len(data) > 500 else data)
                    md_lines.append("```")
                else:
                    md_lines.append(data)
            else:
                md_lines.append(str(data))

            md_lines.append("")

        return "\n".join(md_lines)

    def _results_to_html(self, results: List[Dict[str, Any]]) -> str:
        """Convert results to HTML format"""
        html_parts = ['<div class="workflow-results">']

        for i, result in enumerate(results, 1):
            agent_id = result.get("agent_id", "unknown")
            data = result.get("data", "")

            html_parts.append(f'<div class="result-item">')
            html_parts.append(f'<h3>{i}. {agent_id}</h3>')

            if isinstance(data, str) and ("<svg" in data or "<html" in data):
                # Embed raw HTML/SVG
                html_parts.append(f'<div class="content">{data}</div>')
            elif isinstance(data, dict) or isinstance(data, list):
                html_parts.append(f'<pre>{json.dumps(data, ensure_ascii=False, indent=2)}</pre>')
            else:
                html_parts.append(f'<p>{data}</p>')

            html_parts.append('</div>')

        html_parts.append('</div>')

        return "\n".join(html_parts)

    def aggregate_sync(
        self,
        stages: List[StageResult],
        strategy: Dict[str, Any],
    ) -> Any:
        """Synchronous aggregation (no LLM)"""
        import asyncio

        # Disable LLM for sync
        original_enable = self.enable_llm
        self.enable_llm = False

        try:
            return asyncio.run(self.aggregate(stages, strategy))
        finally:
            self.enable_llm = original_enable


# Factory function
def create_result_aggregator(
    streamer: Optional[ProgressStreamer] = None,
) -> ResultAggregator:
    """Create a ResultAggregator instance with default configuration"""
    return ResultAggregator(streamer=streamer)
