"""
Data Transformer

Transforms data between agents with different input/output schemas.

Transformation Strategies:
1. Rule-based: Fast, deterministic transformations (preferred)
2. LLM-based: Fallback for complex transformations

Transformation Matrix:
| Source Agent | Target Agent | Transformation |
|--------------|--------------|----------------|
| internet_agent | analysis_agent | HTML/Text → Structured Data |
| analysis_agent | visualization_agent | Dict → Chart Data Format |
| analysis_agent | llm_search_agent | Dict → Context String |
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .models import AgentResult
from .agent_registry import AgentRegistry, get_registry
from .progress_streamer import ProgressStreamer
from .exceptions import TransformationError

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Transforms data between agent outputs and inputs.

    Uses rule-based transformations first (fast, deterministic),
    falls back to LLM-based transformation for complex cases.

    Example:
        transformer = DataTransformer()

        # Transform internet_agent output for analysis_agent
        result = await transformer.transform(
            source_agent="internet_agent",
            target_agent="analysis_agent",
            data=search_results
        )
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        streamer: Optional[ProgressStreamer] = None,
        enable_llm_fallback: bool = True,
    ):
        """
        Initialize Data Transformer.

        Args:
            registry: Agent registry for schema lookup
            streamer: Progress streamer for status updates
            enable_llm_fallback: Whether to use LLM for complex transformations
        """
        self.registry = registry or get_registry()
        self.streamer = streamer
        self.enable_llm_fallback = enable_llm_fallback

        # Rule-based transformation registry
        self._transformers: Dict[Tuple[str, str], Callable] = {}
        self._register_default_transformers()

    def _register_default_transformers(self):
        """Register default transformation rules"""

        # internet_agent → analysis_agent
        self.register_transformer(
            "internet_agent",
            "analysis_agent",
            self._transform_internet_to_analysis
        )

        # analysis_agent → data_visualization_agent
        self.register_transformer(
            "analysis_agent",
            "data_visualization_agent",
            self._transform_analysis_to_visualization
        )

        # analysis_agent → llm_search_agent
        self.register_transformer(
            "analysis_agent",
            "llm_search_agent",
            self._transform_analysis_to_llm_search
        )

        # internet_agent → data_visualization_agent (direct)
        self.register_transformer(
            "internet_agent",
            "data_visualization_agent",
            self._transform_internet_to_visualization
        )

        # any → llm_search_agent (context string)
        self.register_transformer(
            "*",
            "llm_search_agent",
            self._transform_any_to_context_string
        )

        # samsung_gateway_agent → analysis_agent
        self.register_transformer(
            "samsung_gateway_agent",
            "analysis_agent",
            self._transform_json_pass_through
        )

        # samsung_gateway_agent → data_visualization_agent
        self.register_transformer(
            "samsung_gateway_agent",
            "data_visualization_agent",
            self._transform_analysis_to_visualization
        )

    def register_transformer(
        self,
        source_agent: str,
        target_agent: str,
        transformer: Callable[[Any], Any],
    ):
        """Register a transformation function"""
        self._transformers[(source_agent, target_agent)] = transformer
        logger.debug(
            f"[DataTransformer] Registered: {source_agent} → {target_agent}"
        )

    async def transform(
        self,
        source_agent: str,
        target_agent: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Transform data from source agent format to target agent format.

        Args:
            source_agent: ID of source agent
            target_agent: ID of target agent
            data: Data to transform
            context: Optional additional context

        Returns:
            Transformed data suitable for target agent

        Raises:
            TransformationError: If transformation fails
        """
        # Emit transform start
        if self.streamer:
            await self.streamer.transform_start(source_agent, target_agent)

        try:
            # Try specific transformer first
            key = (source_agent, target_agent)
            if key in self._transformers:
                result = self._transformers[key](data)
                if self.streamer:
                    await self.streamer.transform_complete(source_agent, target_agent)
                return result

            # Try wildcard source transformer
            wildcard_key = ("*", target_agent)
            if wildcard_key in self._transformers:
                result = self._transformers[wildcard_key](data)
                if self.streamer:
                    await self.streamer.transform_complete(source_agent, target_agent)
                return result

            # Try type-based transformation
            result = self._transform_by_type(source_agent, target_agent, data)
            if result is not None:
                if self.streamer:
                    await self.streamer.transform_complete(source_agent, target_agent)
                return result

            # LLM fallback
            if self.enable_llm_fallback:
                result = await self._transform_with_llm(
                    source_agent, target_agent, data, context
                )
                if self.streamer:
                    await self.streamer.transform_complete(source_agent, target_agent)
                return result

            # No transformation available, pass through
            logger.warning(
                f"[DataTransformer] No transformer for "
                f"{source_agent} → {target_agent}, passing through"
            )
            if self.streamer:
                await self.streamer.transform_complete(source_agent, target_agent)
            return data

        except Exception as e:
            logger.error(
                f"[DataTransformer] Transform failed: "
                f"{source_agent} → {target_agent}: {e}"
            )
            if self.streamer:
                await self.streamer.transform_complete(
                    source_agent, target_agent, success=False
                )
            raise TransformationError(
                source_agent=source_agent,
                target_agent=target_agent,
                reason=str(e),
                original_error=e,
            )

    def transform_sync(
        self,
        source_agent: str,
        target_agent: str,
        data: Any,
    ) -> Any:
        """Synchronous transformation (uses rule-based only)"""
        # Try specific transformer
        key = (source_agent, target_agent)
        if key in self._transformers:
            return self._transformers[key](data)

        # Try wildcard
        wildcard_key = ("*", target_agent)
        if wildcard_key in self._transformers:
            return self._transformers[wildcard_key](data)

        # Type-based
        result = self._transform_by_type(source_agent, target_agent, data)
        if result is not None:
            return result

        # Pass through
        return data

    def _transform_by_type(
        self,
        source_agent: str,
        target_agent: str,
        data: Any,
    ) -> Optional[Any]:
        """Transform based on data type and schema"""

        source_entry = self.registry.get_agent_safe(source_agent)
        target_entry = self.registry.get_agent_safe(target_agent)

        if not source_entry or not target_entry:
            return None

        source_type = source_entry.schema.output_type
        target_type = target_entry.schema.input_type

        # Same type - no transformation needed
        if source_type == target_type:
            return data

        # Any type - pass through
        if source_type == "any" or target_type == "any":
            return data

        # text → structured_data
        if source_type == "text" and target_type == "structured_data":
            return self._text_to_structured(data)

        # json → text
        if source_type == "json" and target_type == "query":
            return self._json_to_query_string(data)

        # html → text
        if source_type == "html" and target_type == "text":
            return self._html_to_text(data)

        return None

    # ========== Rule-based transformation functions ==========

    def _transform_internet_to_analysis(self, data: Any) -> Dict[str, Any]:
        """Transform internet search results to analysis-ready format"""
        if isinstance(data, dict):
            return {
                "raw_data": data,
                "data_type": "internet_search_result",
                "content": data.get("content") or data.get("text") or str(data),
            }
        elif isinstance(data, str):
            # Try to extract structured data from text
            return {
                "raw_data": data,
                "data_type": "text",
                "content": data,
                "extracted_numbers": self._extract_numbers(data),
                "extracted_dates": self._extract_dates(data),
            }
        elif isinstance(data, list):
            return {
                "raw_data": data,
                "data_type": "list",
                "items": data,
                "count": len(data),
            }
        return {"raw_data": data, "data_type": "unknown"}

    def _transform_analysis_to_visualization(self, data: Any) -> Dict[str, Any]:
        """Transform analysis results to visualization-ready format"""
        if isinstance(data, dict):
            # Check if already in chart format
            if "chart_data" in data or "chart_type" in data:
                return data

            # Try to create chart data from analysis results
            chart_data = {
                "chart_type": "auto",  # Let visualization agent decide
                "title": data.get("title", ""),
                "data_points": [],
            }

            # Extract data points if available
            if "data" in data and isinstance(data["data"], list):
                chart_data["data_points"] = data["data"]
            elif "items" in data and isinstance(data["items"], list):
                chart_data["data_points"] = data["items"]
            elif "values" in data:
                chart_data["data_points"] = data["values"]
            else:
                # Use the entire dict as context
                chart_data["context"] = data

            return chart_data

        elif isinstance(data, list):
            return {
                "chart_type": "auto",
                "data_points": data,
            }

        return {"chart_type": "auto", "raw_data": data}

    def _transform_analysis_to_llm_search(self, data: Any) -> str:
        """Transform analysis results to LLM context string"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Create context string from dict
            parts = []
            if "summary" in data:
                parts.append(f"Summary: {data['summary']}")
            if "findings" in data:
                parts.append(f"Findings: {data['findings']}")
            if "conclusion" in data:
                parts.append(f"Conclusion: {data['conclusion']}")
            if not parts:
                parts.append(json.dumps(data, ensure_ascii=False, indent=2))
            return "\n".join(parts)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        return str(data)

    def _transform_internet_to_visualization(self, data: Any) -> Dict[str, Any]:
        """Transform internet data directly to visualization format"""
        # First convert to analysis format
        analysis_data = self._transform_internet_to_analysis(data)
        # Then to visualization format
        return self._transform_analysis_to_visualization(analysis_data)

    def _transform_any_to_context_string(self, data: Any) -> str:
        """Transform any data to context string for LLM"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, list):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    def _transform_json_pass_through(self, data: Any) -> Dict[str, Any]:
        """Pass through JSON data with minimal transformation"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {"content": data}
        return {"data": data}

    # ========== Helper functions ==========

    def _text_to_structured(self, data: Any) -> Dict[str, Any]:
        """Convert text to structured data"""
        if isinstance(data, str):
            return {
                "text": data,
                "length": len(data),
                "lines": data.split("\n"),
                "extracted_numbers": self._extract_numbers(data),
            }
        return {"data": data}

    def _json_to_query_string(self, data: Any) -> str:
        """Convert JSON/dict to query string"""
        if isinstance(data, dict):
            if "query" in data:
                return data["query"]
            if "question" in data:
                return data["question"]
            return json.dumps(data, ensure_ascii=False)
        return str(data)

    def _html_to_text(self, data: Any) -> str:
        """Strip HTML tags from text"""
        if isinstance(data, str):
            # Simple HTML tag removal
            clean = re.sub(r'<[^>]+>', '', data)
            # Clean up whitespace
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean
        return str(data)

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        if not isinstance(text, str):
            return []
        # Match numbers including decimals and commas
        pattern = r'[\d,]+\.?\d*'
        matches = re.findall(pattern, text)
        numbers = []
        for match in matches:
            try:
                # Remove commas and convert
                num = float(match.replace(",", ""))
                numbers.append(num)
            except ValueError:
                pass
        return numbers

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date-like patterns from text"""
        if not isinstance(text, str):
            return []
        # Common date patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{4}/\d{2}/\d{2}',  # 2024/01/15
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # 2024년 1월 15일
            r'\d{1,2}월\s*\d{1,2}일',  # 1월 15일
        ]
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates

    async def _transform_with_llm(
        self,
        source_agent: str,
        target_agent: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Use LLM for complex transformation (fallback)"""
        logger.info(
            f"[DataTransformer] Using LLM fallback for "
            f"{source_agent} → {target_agent}"
        )

        # For now, just do basic type conversion
        # TODO: Implement actual LLM-based transformation
        target_entry = self.registry.get_agent_safe(target_agent)
        if target_entry:
            if target_entry.schema.input_type == "query":
                return self._transform_any_to_context_string(data)
            elif target_entry.schema.input_type == "structured_data":
                return self._text_to_structured(data)
            elif target_entry.schema.input_type == "json":
                return self._transform_json_pass_through(data)

        return data

    async def transform_batch(
        self,
        transformations: List[Dict[str, Any]],
    ) -> List[Any]:
        """
        Transform multiple data items.

        Args:
            transformations: List of dicts with source_agent, target_agent, data

        Returns:
            List of transformed data
        """
        results = []
        for t in transformations:
            result = await self.transform(
                source_agent=t["source_agent"],
                target_agent=t["target_agent"],
                data=t["data"],
                context=t.get("context"),
            )
            results.append(result)
        return results

    def get_available_transformers(self) -> List[Tuple[str, str]]:
        """Get list of registered transformation pairs"""
        return list(self._transformers.keys())


# Factory function
def create_data_transformer(
    registry: Optional[AgentRegistry] = None,
    streamer: Optional[ProgressStreamer] = None,
) -> DataTransformer:
    """Create a DataTransformer instance with default configuration"""
    return DataTransformer(registry=registry, streamer=streamer)
