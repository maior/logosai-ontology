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
        """Transform analysis results to visualization-ready format

        Extracts data_values, data_points, etc. from analysis_agent output
        and converts them to a format usable by data_visualization_agent.
        """
        if isinstance(data, dict):
            # Check if already in chart format
            if "chart_data" in data or ("chart_type" in data and "data_points" in data):
                return data

            # Try to create chart data from analysis results
            chart_data = {
                "chart_type": "auto",  # Let visualization agent decide
                "title": data.get("title", ""),
                "data_points": [],
            }

            # 🔥 0. Process analysis_agent results.date_price_pairs (highest priority - date-price pairs)
            results_data = data.get("results", {})
            if isinstance(results_data, dict):
                date_price_pairs = results_data.get("date_price_pairs", [])
                if date_price_pairs and isinstance(date_price_pairs, list) and len(date_price_pairs) >= 2:
                    chart_data["data_points"] = [
                        {
                            "label": f"{pair.get('date', '').split('-')[1]}/{pair.get('date', '').split('-')[2]}" if '-' in pair.get('date', '') else pair.get('date', f'item{i+1}'),
                            "value": float(pair.get("price", 0)),
                            "category": "stock",
                            "type": "stock_price"
                        }
                        for i, pair in enumerate(date_price_pairs) if pair.get("price")
                    ]
                    if chart_data["data_points"]:
                        chart_data["chart_type"] = "line"
                        chart_data["data_type"] = "stock_price"
                        chart_data["source"] = "analysis_agent.results.date_price_pairs"
                        logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data['data_points'])} data points from results.date_price_pairs")
                        return chart_data

                # 🔥 Process results.data_values (no date information)
                result_data_values = results_data.get("data_values", [])
                if result_data_values and isinstance(result_data_values, list) and len(result_data_values) >= 2:
                    # Filter only values within stock price range (50000~200000 KRW)
                    stock_prices = [v for v in result_data_values if isinstance(v, (int, float)) and 50000 <= v <= 200000]
                    if len(stock_prices) >= 2:
                        chart_data["data_points"] = [
                            {"label": f"data{i+1}", "value": float(v), "category": "stock", "type": "stock_price"}
                            for i, v in enumerate(stock_prices)
                        ]
                        chart_data["chart_type"] = "line"
                        chart_data["data_type"] = "stock_price"
                        chart_data["source"] = "analysis_agent.results.data_values"
                        logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data['data_points'])} stock price data points from results.data_values")
                        return chart_data

            # 🔥 1. Process analysis_agent data_values field (numeric array)
            data_values = data.get("data_values", [])
            if data_values and isinstance(data_values, list):
                # Use label information if available
                labels = data.get("labels", data.get("data_labels", []))
                if labels and len(labels) >= len(data_values):
                    chart_data["data_points"] = [
                        {"label": str(labels[i]), "value": float(v), "category": "analysis"}
                        for i, v in enumerate(data_values)
                    ]
                else:
                    # Generate index-based labels if no labels available
                    chart_data["data_points"] = [
                        {"label": f"item{i+1}", "value": float(v), "category": "analysis"}
                        for i, v in enumerate(data_values)
                    ]
                if chart_data["data_points"]:
                    logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data['data_points'])} data points from data_values")
                    return chart_data

            # 🔥 2. Process structured_data if present (internet_agent pass-through)
            structured_data = data.get("structured_data", {})
            if structured_data:
                date_price_pairs = structured_data.get("date_price_pairs", [])
                if date_price_pairs:
                    chart_data["data_points"] = [
                        {
                            "label": f"{pair.get('date', '').split('-')[1]}/{pair.get('date', '').split('-')[2]}" if '-' in pair.get('date', '') else pair.get('date', f'item{i+1}'),
                            "value": float(pair.get("price", 0)),
                            "category": "stock",
                            "type": "stock_price"
                        }
                        for i, pair in enumerate(date_price_pairs) if pair.get("price")
                    ]
                    if chart_data["data_points"]:
                        chart_data["chart_type"] = "line"
                        chart_data["data_type"] = "stock_price"
                        logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data['data_points'])} data points from structured_data")
                        return chart_data

            # 3. Extract data points if available (legacy logic)
            if "data_points" in data and isinstance(data["data_points"], list):
                chart_data["data_points"] = data["data_points"]
            elif "data" in data and isinstance(data["data"], list):
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
        """Transform internet data directly to visualization format

        Specifically converts internet_agent's structured_data (stock prices, dates, etc.)
        into a format immediately usable by data_visualization_agent.
        """
        if isinstance(data, dict):
            # 🔥 Directly extract stock price data from structured_data
            structured_data = data.get("structured_data", {})

            # 1. Use date_price_pairs first if available (most accurate data)
            date_price_pairs = structured_data.get("date_price_pairs", [])
            if date_price_pairs:
                chart_data = []
                for pair in date_price_pairs:
                    date_str = pair.get("date", "")
                    price = pair.get("price", 0)
                    if date_str and price:
                        # Convert date format to MM/DD
                        try:
                            parts = date_str.split("-")
                            if len(parts) >= 3:
                                label = f"{parts[1]}/{parts[2]}"
                            else:
                                label = date_str
                        except:
                            label = date_str
                        chart_data.append({
                            "label": label,
                            "value": float(price),
                            "category": "stock",
                            "type": "stock_price",
                            "date": date_str
                        })
                if chart_data:
                    logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data)} data points from date_price_pairs")
                    return {
                        "chart_type": "line",
                        "title": data.get("query", "Stock Price Trend"),
                        "data_points": chart_data,
                        "data_type": "stock_price",
                        "source": "internet_agent.structured_data"
                    }

            # 2. Use prices array if available
            prices = structured_data.get("prices", [])
            dates = structured_data.get("dates", [])
            if prices:
                chart_data = []
                for i, price_info in enumerate(prices):
                    if isinstance(price_info, dict):
                        value = price_info.get("value", 0)
                        context = price_info.get("context", f"price{i+1}")
                    else:
                        value = price_info
                        context = f"price{i+1}"

                    # Use corresponding date if available
                    if i < len(dates):
                        date_info = dates[i]
                        if isinstance(date_info, dict):
                            label = date_info.get("formatted", date_info.get("original", f"item{i+1}"))
                        else:
                            label = str(date_info)
                    else:
                        label = context

                    chart_data.append({
                        "label": label,
                        "value": float(value),
                        "category": "stock",
                        "type": "stock_price"
                    })
                if chart_data:
                    logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data)} data points from prices array")
                    return {
                        "chart_type": "line" if len(chart_data) > 1 else "bar",
                        "title": data.get("query", "Price Data"),
                        "data_points": chart_data,
                        "data_type": structured_data.get("data_type", "general"),
                        "source": "internet_agent.structured_data"
                    }

            # 3. Attempt to extract stock price data from text
            content = data.get("content") or data.get("answer") or data.get("text") or ""
            if content and isinstance(content, str):
                # Extract stock price data from markdown table
                import re
                markdown_pattern = r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([0-9,]+)\s*\|'
                matches = re.findall(markdown_pattern, content)
                if matches:
                    chart_data = []
                    for date_str, value_str in matches:
                        try:
                            parts = date_str.split("-")
                            label = f"{parts[1]}/{parts[2]}"
                            value = float(value_str.replace(",", ""))
                            if 10000 <= value <= 1000000:  # Validate stock price range
                                chart_data.append({
                                    "label": label,
                                    "value": value,
                                    "category": "stock",
                                    "type": "stock_price",
                                    "date": date_str
                                })
                        except:
                            continue
                    if chart_data:
                        logger.info(f"[DataTransformer] ✅ Extracted {len(chart_data)} data points from markdown table")
                        return {
                            "chart_type": "line",
                            "title": data.get("query", "Stock Price Trend"),
                            "data_points": chart_data,
                            "data_type": "stock_price",
                            "source": "internet_agent.content"
                        }

        # 4. Fall back to legacy logic
        analysis_data = self._transform_internet_to_analysis(data)
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
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # 2024년 1월 15일 (Korean date format)
            r'\d{1,2}월\s*\d{1,2}일',  # 1월 15일 (Korean date format)
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
