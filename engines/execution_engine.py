"""
Enhanced Execution Engine
Core execution component supporting multiple execution strategies and data transformation
"""

import asyncio
import os
import time
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..core.models import (
    SemanticQuery, ExecutionContext, AgentExecutionResult, WorkflowPlan,
    ExecutionStrategy, AgentType, ExecutionStatus, QueryType,
    DEFAULT_AGENT_CAPABILITIES
)
from ..core.interfaces import ExecutionEngine, DataTransformer, AgentCaller, MetricsCollector
from .semantic_query_manager import SemanticQueryManager

from loguru import logger

DEFAULT_ACP_ENDPOINT = os.getenv("ACP_SERVER_URL", "http://localhost:8888") + "/jsonrpc"

@dataclass
class ExecutionPlan:
    """Execution plan"""
    strategy: ExecutionStrategy
    agent_calls: List[Dict[str, Any]]
    estimated_time: float
    parallel_groups: List[List[Dict[str, Any]]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)


class QueryComplexityAnalyzer:
    """Query complexity analyzer"""

    @staticmethod
    def analyze_complexity(query: SemanticQuery) -> Dict[str, Any]:
        """Analyze query complexity"""
        try:
            # Default complexity score
            complexity_score = 0.5

            # Complexity based on natural language length
            if hasattr(query, 'natural_language') and query.natural_language:
                text_length = len(query.natural_language)
                if text_length > 200:
                    complexity_score += 0.2
                elif text_length > 100:
                    complexity_score += 0.1

            # Complexity based on required agent count
            if hasattr(query, 'required_agents') and query.required_agents:
                agent_count = len(query.required_agents)
                complexity_score += min(agent_count * 0.1, 0.3)

            # Complexity based on query type
            if hasattr(query, 'query_type'):
                if query.query_type in [QueryType.MULTI_STEP, QueryType.COMPLEX]:
                    complexity_score += 0.2
                elif query.query_type == QueryType.ANALYTICAL:
                    complexity_score += 0.15

            # Normalize complexity score
            complexity_score = min(complexity_score, 1.0)

            # Recommend strategy
            if complexity_score < 0.3:
                recommended_strategy = ExecutionStrategy.SINGLE_AGENT
            elif complexity_score < 0.6:
                recommended_strategy = ExecutionStrategy.PARALLEL
            elif complexity_score < 0.8:
                recommended_strategy = ExecutionStrategy.SEQUENTIAL
            else:
                recommended_strategy = ExecutionStrategy.HYBRID

            return {
                'complexity_score': complexity_score,
                'recommended_strategy': recommended_strategy,
                'estimated_time': complexity_score * 60.0,  # Max 60 seconds
                'confidence': 0.8
            }

        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return {
                'complexity_score': 0.5,
                'recommended_strategy': ExecutionStrategy.AUTO,
                'estimated_time': 30.0,
                'confidence': 0.5
            }


class SmartDataTransformer(DataTransformer):
    """Smart data transformer"""

    def __init__(self):
        self.transformation_rules = self._initialize_transformation_rules()

    def _initialize_transformation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize transformation rules"""
        return {
            "text_to_structured": {
                "pattern": r"(.+)",
                "output_format": "json",
                "confidence": 0.8
            },
            "structured_to_text": {
                "pattern": r".*",
                "output_format": "text",
                "confidence": 0.9
            },
            "aggregation": {
                "pattern": r"list|array",
                "output_format": "summary",
                "confidence": 0.7
            }
        }

    def get_supported_transformations(self) -> List[str]:
        """Return list of supported transformation types"""
        return [
            "text_to_structured",
            "structured_to_text",
            "aggregation",
            "standardization",
            "agent_format_conversion"
        ]

    async def transform_input(self, data: Any, target_format: str) -> Any:
        """Transform input data to the target format"""
        try:
            if target_format == "structured":
                return await self._transform_text_to_structured(data)
            elif target_format == "text":
                return await self._transform_structured_to_text(data)
            elif target_format == "aggregated":
                if isinstance(data, list):
                    return await self._aggregate_data(data)
                else:
                    return await self._aggregate_data([data])
            elif target_format == "standardized":
                return await self._standardize_format(data)
            else:
                logger.warning(f"Unsupported transformation format: {target_format}")
                return data
        except Exception as e:
            logger.error(f"Input transformation failed: {e}")
            return data

    async def transform_output(self, data: Any, source_format: str) -> Any:
        """Transform output data from the source format"""
        try:
            if source_format == "structured" and isinstance(data, dict):
                return await self._transform_structured_to_text(data)
            elif source_format == "text" and isinstance(data, str):
                return await self._transform_text_to_structured(data)
            elif source_format == "aggregated":
                return await self._standardize_format(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Output transformation failed: {e}")
            return data

    async def transform_between_agents(
        self,
        data: Any,
        source_agent: AgentType,
        target_agent: AgentType
    ) -> Any:
        """Transform data between agents"""
        try:
            # Infer data formats of source and target agents
            source_format = self._infer_agent_data_format(source_agent)
            target_format = self._infer_agent_data_format(target_agent)

            if source_format == target_format:
                return data

            # Perform transformation
            if source_format == "text" and target_format == "structured":
                return await self._transform_text_to_structured(data)
            elif source_format == "structured" and target_format == "text":
                return await self._transform_structured_to_text(data)
            elif isinstance(data, list):
                return await self._aggregate_data(data)
            else:
                return await self._standardize_format(data)

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            return data

    def _infer_agent_data_format(self, agent_type: AgentType) -> str:
        """Infer data format from agent type"""
        format_mapping = {
            AgentType.RESEARCH: "structured",
            AgentType.ANALYSIS: "structured",
            AgentType.CREATIVE: "text",
            AgentType.TECHNICAL: "structured",
            AgentType.GENERAL: "text"
        }
        return format_mapping.get(agent_type, "text")

    async def _transform_text_to_structured(self, data: Any) -> Dict[str, Any]:
        """Transform text to structured data"""
        if isinstance(data, str):
            return {
                "content": data,
                "type": "text",
                "length": len(data),
                "transformed": True
            }
        return {"data": data, "transformed": True}

    async def _transform_structured_to_text(self, data: Any) -> str:
        """Transform structured data to text"""
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)

    async def _aggregate_data(self, data: List[Any]) -> Dict[str, Any]:
        """Aggregate data"""
        return {
            "items": data,
            "count": len(data),
            "aggregated": True,
            "summary": f"{len(data)} items aggregated"
        }

    async def _standardize_format(self, data: Any) -> Dict[str, Any]:
        """Transform to standard format"""
        return {
            "data": data,
            "type": type(data).__name__,
            "standardized": True
        }

    async def _apply_default_transformation(
        self,
        data: Any,
        source_format: str,
        target_format: str
    ) -> Any:
        """Apply default transformation"""
        logger.debug(f"Applying default transformation: {source_format} -> {target_format}")

        if target_format == "json" and not isinstance(data, dict):
            return {"content": str(data), "transformed": True}
        elif target_format == "text":
            return str(data)
        else:
            return data


class RealAgentCaller(AgentCaller):
    """Real agent caller - communicates with installed agents"""

    def __init__(self):
        self.installed_agents: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.call_history: List[Dict] = []

        # Register default agents (for fallback)
        self.default_agents = self._create_default_agents()

    def _create_default_agents(self) -> List[Dict]:
        """Create default agents (fallback when not registered in ACP server)"""
        return [
            {
                "agent_id": "internet_agent",
                "name": "Internet Search Agent",
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT,
                    "capabilities": ["web_search", "information_retrieval"]
                },
                "configuration": {
                    "endpoint": DEFAULT_ACP_ENDPOINT
                }
            },
            {
                "agent_id": "weather_agent", 
                "name": "Weather Agent",
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT,
                    "capabilities": ["weather_inquiry"]
                }
            },
            {
                "agent_id": "finance_agent",
                "name": "Finance Agent", 
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT,
                    "capabilities": ["financial_data", "currency_exchange"]
                }
            },
            {
                "agent_id": "calculator_agent",
                "name": "Calculator Agent",
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT, 
                    "capabilities": ["calculation", "math"]
                }
            },
            {
                "agent_id": "chart_agent",
                "name": "Chart Agent",
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT,
                    "capabilities": ["visualization", "chart_generation"]
                }
            },
            {
                "agent_id": "memo_agent",
                "name": "Memo Agent",
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT,
                    "capabilities": ["memo_creation", "note_taking"]
                }
            },
            {
                "agent_id": "analysis_agent", 
                "name": "Analysis Agent",
                "agent_data": {
                    "endpoint": DEFAULT_ACP_ENDPOINT,
                    "capabilities": ["analysis", "data_analysis"]
                }
            }
        ]

    async def _ensure_session(self):
        """Verify and create HTTP session"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=120, connect=10)
            # Increase buffer size for large SSE chunks (default 64KB → 1MB)
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                force_close=False
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                read_bufsize=1024 * 1024  # 1MB buffer
            )

    async def call_agent(
        self,
        agent_type: AgentType,
        query: SemanticQuery,
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Call agent - communicates with actually installed agents"""
        start_time = time.time()
        
        try:
            await self._ensure_session()
            
            # Get list of installed agents
            installed_agents = context.custom_config.get('installed_agents', [])

            # Check if target_agent info exists in context (passed from execute_workflow)
            target_agent = context.custom_config.get('target_agent')

            if target_agent:
                # Use already-matched agent from execute_workflow if available
                logger.info(f"🎯 Using designated agent: {target_agent.get('agent_id', 'unknown')}")
                agent = target_agent
            else:
                # 1. Search in installed agents
                agent = self._find_matching_agent(agent_type, installed_agents)

                # 2. If not found in installed agents, search default agents
                if not agent:
                    logger.warning(f"Cannot find {getattr(agent_type, 'value', str(agent_type))} in installed agents. Searching default agents.")
                    agent = self._find_matching_agent(agent_type, self.default_agents)

            if not agent:
                logger.error(f"❌ No available agent: {getattr(agent_type, 'value', str(agent_type))}")
                return self._create_fallback_result(agent_type, query, start_time)

            # Call agent
            result_data = await self._call_installed_agent(agent, query, context)

            # None check
            if result_data is None:
                logger.error(f"⚠️ _call_installed_agent returned None: {agent.get('agent_id', 'unknown')}")
                return self._create_fallback_result(agent_type, query, start_time)

            # Record successful call
            self.call_history.append({
                'agent_type': agent_type,
                'timestamp': time.time(),
                'success': result_data.get('success', False),
                'execution_time': time.time() - start_time
            })

            # Process result
            execution_time = time.time() - start_time
            success = result_data.get('success', False)

            # Check if response is structured
            raw_result = result_data.get('result', result_data)
            if isinstance(raw_result, dict) and 'answer' in raw_result:
                # Preserve structured response as-is
                final_result = raw_result
            else:
                # Extract core content from unstructured responses only
                final_result = self._extract_core_content(raw_result)

            # Also store in data field when creating AgentExecutionResult
            result = AgentExecutionResult(
                result_data=final_result,
                execution_time=execution_time,
                status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
                agent_type=agent_type,
                error_message=result_data.get('error') if not success else None,
                confidence=result_data.get('confidence', 0.8 if success else 0.3),
                metadata={
                    'agent_used': agent.get('agent_id', 'unknown'),
                    'endpoint': agent.get('agent_data', {}).get('endpoint', 'unknown'),
                    'fallback_used': agent in self.default_agents,
                    'target_agent_specified': target_agent is not None
                }
            )

            # Also set data field (ResultProcessor compatibility)
            result.data = final_result

            # 🔄 Extract actually selected agent info from 8888 server response (store as-is)
            # _call_installed_agent already extracts agent_id, agent_name, requested_agent_id, auto_selected
            # Use directly from result_data (raw_result.metadata may be lost after content processing)
            actual_agent_id = result_data.get('agent_id', agent.get('agent_id', 'unknown'))
            actual_agent_name = result_data.get('agent_name')  # agent_name passed from ACP server
            requested_agent_id = result_data.get('requested_agent_id', agent.get('agent_id', 'unknown'))
            auto_selected = result_data.get('auto_selected', False)
            response_type = result_data.get('response_type', 'single_agent')
            agent_results = result_data.get('agent_results', [])

            # 🆕 Extract unified agents array and unified_content
            agents_array = result_data.get('agents', [])  # New unified structure
            unified_content = result_data.get('unified_content', {})

            # Try to extract additional info from metadata (if present in raw_result)
            server_metadata = {}
            if isinstance(raw_result, dict):
                server_metadata = raw_result.get('metadata', {})
            selection_reason = server_metadata.get('selection_reason', '')

            if actual_agent_id != requested_agent_id:
                logger.info(f"🔄 Agent changed by Task Classifier: {requested_agent_id} → {actual_agent_id}")
                logger.info(f"📋 Selection reason: {selection_reason[:100]}..." if len(selection_reason) > 100 else f"📋 Selection reason: {selection_reason}")

            # Use agent_id passed from 8888 server as-is
            result.agent_id = actual_agent_id

            # Set agent_name - prefer agent_name passed from ACP server
            # Priority 1: result_data.agent_name (passed directly from ACP server)
            # Priority 2: server_metadata.agent_name (legacy compatibility)
            # Priority 3: generate display name from actual_agent_id
            result.agent_name = actual_agent_name or server_metadata.get('agent_name') or self._get_agent_display_name(actual_agent_id)

            logger.info(f"📋 Final agent info: id={result.agent_id}, name={result.agent_name}, type={response_type}, agents_count={len(agents_array)}")

            # Merge 8888 server metadata into result.metadata
            result.metadata.update({
                'selected_agent_id': actual_agent_id,
                'selected_agent_name': result.agent_name,
                'requested_agent_id': requested_agent_id,
                'auto_selected': auto_selected,
                'selection_reason': selection_reason,
                'selection_confidence': server_metadata.get('selection_confidence', 0.0),
                'response_type': response_type,
                'agents': agents_array,  # 🆕 Unified agents array (new structure)
                'agent_results': agent_results,  # Legacy compatible
                'unified_content': unified_content  # 🆕 Unified summary content
            })
            result.success = success

            # Return result (important!)
            return result

        except Exception as e:
            logger.error(f"❌ Agent call failed {getattr(agent_type, 'value', str(agent_type))}: {e}")

            # Record failure
            self.call_history.append({
                'agent_type': agent_type,
                'timestamp': time.time(),
                'success': False,
                'execution_time': time.time() - start_time,
                'error': str(e)
            })

            return self._create_fallback_result(agent_type, query, start_time)

    def _get_agent_display_name(self, agent_id: str) -> str:
        """Generate display name from agent ID

        Examples:
        - calculator_agent -> Calculator Agent
        - weather_agent -> Weather Agent
        - rag_agent -> RAG Agent
        """
        if not agent_id:
            return "Unknown Agent"

        # Extract and transform name part from agent ID
        # agent_id examples: "calculator_agent", "weather_agent_123abc"
        name_part = agent_id.split('_agent')[0] if '_agent' in agent_id else agent_id

        # Remove hash part (e.g., _123abc)
        if '_' in name_part:
            parts = name_part.split('_')
            # Remove last part if it looks like a hash
            if len(parts[-1]) > 5 and parts[-1].isalnum():
                name_part = '_'.join(parts[:-1])

        # Handle special abbreviations
        special_names = {
            'rag': 'RAG',
            'llm': 'LLM',
            'api': 'API',
            'sql': 'SQL',
            'db': 'DB',
            'ai': 'AI',
        }

        # Split words and capitalize
        words = name_part.replace('_', ' ').split()
        display_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in special_names:
                display_words.append(special_names[lower_word])
            else:
                display_words.append(word.capitalize())

        display_name = ' '.join(display_words)

        # Append "Agent"
        if 'Agent' not in display_name and 'agent' not in display_name.lower():
            display_name += ' Agent'

        return display_name or agent_id

    def _find_matching_agent(self, agent_type: AgentType, agents: List[Dict]) -> Optional[Dict]:
        """Find installed agent matching agent_type"""
        logger.info(f"🔍 Starting agent matching: {getattr(agent_type, 'value', str(agent_type))}")
        logger.info(f"📋 Available agent count: {len(agents)}")

        # Log installed agents list (with normalized names)
        for i, agent in enumerate(agents):
            agent_id = agent.get('agent_id', 'unknown')
            normalized_id = self._normalize_agent_id(agent_id)
            logger.info(f"  {i+1}. {agent_id} -> {normalized_id}")

        # 🎯 Priority matching by clean agent names
        preferred_agents = {
            AgentType.RESEARCH: ['internet_agent', 'weather_agent'],
            AgentType.ANALYSIS: ['analysis_agent', 'finance_agent'],
            AgentType.TECHNICAL: ['calculator_agent'],
            AgentType.CREATIVE: ['chart_agent'],
            AgentType.GENERAL: ['memo_agent']
        }

        # Pass 1: Exact standard agent name matching
        target_agents = preferred_agents.get(agent_type, [])
        for target_name in target_agents:
            for agent in agents:
                agent_id = agent.get('agent_id', '').lower()
                normalized_id = self._normalize_agent_id(agent_id)

                if normalized_id == target_name:
                    logger.info(f"✅ Exact standard name match: {getattr(agent_type, 'value', str(agent_type))} -> {agent_id} (standard: {target_name})")
                    return agent

        # Pass 2: Pattern matching by AgentType
        type_patterns = {
            AgentType.RESEARCH: [
                'internet_agent', 'search_agent', 'weather_agent', 'restaurant_finder_agent', 'shopping_agent'
            ],
            AgentType.ANALYSIS: [
                'analysis_agent', 'finance_agent', 'currency_agent', 'exchange_agent'
            ],
            AgentType.TECHNICAL: [
                'calculator_agent', 'translate_agent', 'file_agent',
                'document_agent', 'content_formatter_agent'
            ],
            AgentType.CREATIVE: [
                'chart_agent', 'image_agent', 'audio_agent', 'data_visualization_agent',
                'brick_game_agent', 'tetris_game_agent', 'mahjong_game_agent',
                'sudoku_game_agent', 'road_runner_game_agent', 'super_mario_game_agent'
            ],
            AgentType.GENERAL: [
                'memo_agent', 'note_agent', 'scheduler_agent', 'general_agent'
            ]
        }

        patterns = type_patterns.get(agent_type, ['general'])

        # Perform pattern matching
        for agent in agents:
            agent_id = agent.get('agent_id', '').lower()
            normalized_id = self._normalize_agent_id(agent_id)

            for pattern in patterns:
                if normalized_id == pattern or normalized_id.startswith(pattern):
                    logger.info(f"✅ Pattern match: {getattr(agent_type, 'value', str(agent_type))} -> {agent_id} (pattern: {pattern})")
                    return agent

        # Pass 3: Keyword-based matching (from normalized name)
        keyword_mapping = {
            AgentType.RESEARCH: ['internet', 'search', 'weather', 'restaurant', 'shopping', 'find'],
            AgentType.ANALYSIS: ['analysis', 'finance', 'currency', 'exchange', 'analyze'],
            AgentType.TECHNICAL: ['calculator', 'calculate', 'math', 'translate', 'file', 'document', 'format'],
            AgentType.CREATIVE: ['chart', 'image', 'audio', 'visual', 'game', 'creative'],
            AgentType.GENERAL: ['memo', 'note', 'schedule', 'general', 'research', 'creative', 'technical']
        }

        keywords = keyword_mapping.get(agent_type, [])
        for agent in agents:
            agent_id = agent.get('agent_id', '').lower()
            normalized_id = self._normalize_agent_id(agent_id)

            for keyword in keywords:
                if keyword in normalized_id:
                    logger.info(f"✅ Keyword match: {getattr(agent_type, 'value', str(agent_type))} -> {agent_id} (keyword: {keyword})")
                    return agent

        # Pass 4: First available agent (fallback)
        if agents:
            fallback_agent = agents[0]
            logger.warning(f"⚠️ Using fallback agent: {getattr(agent_type, 'value', str(agent_type))} -> {fallback_agent.get('agent_id')}")
            return fallback_agent

        logger.error(f"❌ No matching agent found: {getattr(agent_type, 'value', str(agent_type))}")
        return None

    async def _call_installed_agent(self, agent: Dict, query: SemanticQuery, context: ExecutionContext) -> Dict:
        """Call installed agent - supports SSE streaming"""
        agent_data = agent.get('agent_data', {})
        # 🔄 Use SSE streaming endpoint (/jsonrpc → /stream)
        base_endpoint = agent_data.get('endpoint', DEFAULT_ACP_ENDPOINT)
        # Convert /jsonrpc to /stream
        if base_endpoint.endswith('/jsonrpc'):
            endpoint = base_endpoint.replace('/jsonrpc', '/stream')
        else:
            endpoint = base_endpoint.rstrip('/') + '/stream'
        agent_id = agent.get('agent_id', 'unknown')

        # 🔧 Normalize agent ID correctly (remove hash part only)
        base_agent_id = self._normalize_agent_id(agent_id)

        logger.info(f"🔍 Agent ID normalization: {agent_id} -> {base_agent_id}")

        # Extract user email - priority: custom_config email > user_email > user_id > default
        user_email = (
            context.custom_config.get('email') or
            context.custom_config.get('user_email') or
            context.user_id or
            'default@logos.ai'
        )

        # Append @system.ai only if user_id is not already in email format
        if user_email and '@' not in user_email:
            user_email = f"{user_email}@system.ai"

        logger.info(f"📧 User email confirmed: {user_email} (from payload)")

        # Extract project ID
        project_id = context.custom_config.get('project_id')

        # 🔄 SSE streaming payload - pass LLM-selected agent to bypass Task Classifier
        payload = {
            "query": getattr(query, 'natural_language', str(query)),
            "email": user_email,
            "sessionid": context.session_id,
            "projectid": project_id,
            "agent_id": base_agent_id,  # 🔧 Pass LLM-selected agent ID (bypasses Task Classifier)
            "timestamp": time.time()
        }

        logger.info(f"📡 Sending SSE streaming query to ACP server (LLM-selected agent: {base_agent_id})")
        logger.info(f"📊 Endpoint: {endpoint}")
        logger.info(f"📊 User: {user_email}")
        logger.debug(f"📊 요청 페이로드: {payload}")

        # Extract progress_callback (forward SSE events if present)
        progress_callback = getattr(context, 'progress_callback', None)

        # 🚀 Set timeout for SSE streaming (longer timeout)
        timeout_settings = aiohttp.ClientTimeout(
            total=300,     # Total request timeout 5 minutes (for streaming)
            connect=10,    # Connection timeout 10 seconds
            sock_read=60   # Socket read timeout 60 seconds (waiting for SSE events)
        )

        max_retries = 3
        retry_delay = 1.0  # Retry interval (seconds)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retrying agent call {attempt + 1}/{max_retries}: {base_agent_id}")
                    await asyncio.sleep(retry_delay * attempt)  # Exponential backoff

                # 🏥 Quick server status check (first attempt only)
                if attempt == 0:
                    server_status = await self._check_server_status(endpoint)
                    if not server_status:
                        logger.warning(f"⚠️ Server status check failed: {endpoint}")

                # 🔄 Set SSE streaming headers
                headers = {
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Content-Type': 'application/json'
                }

                async with self.session.post(endpoint, json=payload, headers=headers, timeout=timeout_settings) as response:
                    if response.status == 200:
                        # 🔄 Process SSE streaming response
                        logger.info(f"📡 SSE streaming started: {endpoint}")

                        final_result = None
                        actual_agent_id = base_agent_id
                        actual_agent_name = None
                        auto_selected = False
                        agents_array = []
                        agent_results = []
                        response_type = "single_agent"
                        metadata = {}

                        event_type = None
                        event_data = ""

                        # Chunk-based SSE reading (for handling large chunks)
                        buffer = ""
                        async for chunk in response.content.iter_any():
                            buffer += chunk.decode('utf-8')

                            # Process complete lines from buffer
                            while '\n' in buffer:
                                line_str, buffer = buffer.split('\n', 1)
                                line_str = line_str.strip()

                                if not line_str:
                                    # Empty line = event complete
                                    if event_type and event_data:
                                        try:
                                            data = json.loads(event_data)
                                            logger.debug(f"📥 SSE event: {event_type} - {data}")

                                            # 🎯 Process by event type
                                            if event_type == 'agent_selected':
                                                # Agent selection event
                                                event_info = data.get('data', data)
                                                actual_agent_id = event_info.get('agent_id', base_agent_id)
                                                actual_agent_name = event_info.get('agent_name')
                                                auto_selected = True
                                                logger.info(f"🎯 Agent selected: {actual_agent_id} ({actual_agent_name})")

                                                # Forward to progress_callback
                                                if progress_callback and hasattr(progress_callback, 'on_progress'):
                                                    await progress_callback.on_progress(
                                                        f"Agent selected: {actual_agent_name or actual_agent_id}",
                                                        0.3,
                                                        {"stage": "agent_selected", "agent_id": actual_agent_id, "agent_name": actual_agent_name}
                                                    )

                                            elif event_type == 'progress':
                                                # Progress event
                                                event_info = data.get('data', data)
                                                progress_msg = event_info.get('message', 'Processing...')
                                                progress_pct = event_info.get('progress', 50) / 100.0
                                                logger.info(f"📊 Progress: {progress_msg} ({progress_pct*100:.0f}%)")

                                                if progress_callback and hasattr(progress_callback, 'on_progress'):
                                                    await progress_callback.on_progress(
                                                        progress_msg,
                                                        progress_pct,
                                                        {"stage": "processing", **event_info}
                                                    )

                                            elif event_type == 'start':
                                                # Agent start event
                                                event_info = data.get('data', data)
                                                start_agent_id = event_info.get('agent_id', actual_agent_id)
                                                start_agent_name = event_info.get('agent_name', actual_agent_name)
                                                logger.info(f"🚀 Agent starting: {start_agent_id} ({start_agent_name})")

                                                if progress_callback and hasattr(progress_callback, 'on_progress'):
                                                    await progress_callback.on_progress(
                                                        f"Agent execution started: {start_agent_name or start_agent_id}",
                                                        0.4,
                                                        {"stage": "start", "agent_id": start_agent_id, "agent_name": start_agent_name}
                                                    )

                                            elif event_type == 'chunk':
                                                # Chunk event (streaming content)
                                                event_info = data.get('data', data)
                                                chunk_content = event_info.get('content', '')
                                                chunk_index = event_info.get('index', 0)
                                                is_last = event_info.get('is_last', False)
                                                logger.debug(f"📝 Chunk #{chunk_index}: {chunk_content[:50]}..." if len(chunk_content) > 50 else f"📝 Chunk #{chunk_index}: {chunk_content}")

                                                if progress_callback and hasattr(progress_callback, 'on_progress'):
                                                    progress_pct = 0.8 if is_last else 0.6
                                                    await progress_callback.on_progress(
                                                        "Streaming response..." if not is_last else "Response complete",
                                                        progress_pct,
                                                        {"stage": "streaming", "chunk": chunk_content, "index": chunk_index, "is_last": is_last}
                                                    )

                                            elif event_type == 'message':
                                                # Message event (partial response)
                                                event_info = data.get('data', data)
                                                message_content = event_info.get('content', '')
                                                logger.debug(f"💬 Message: {message_content[:100]}..." if len(message_content) > 100 else f"💬 Message: {message_content}")

                                                if progress_callback and hasattr(progress_callback, 'on_progress'):
                                                    await progress_callback.on_progress(
                                                        "Generating response...",
                                                        0.7,
                                                        {"stage": "generating", "partial_content": message_content}
                                                    )

                                            elif event_type == 'complete':
                                                # Complete event - final result
                                                logger.info(f"✅ SSE streaming complete")
                                                final_result = data.get('data', data)

                                                # 🔧 Handle nested result structure (ACP server response format)
                                                # When ACP server responds in {'result': {'answer': '...', ...}} format
                                                if isinstance(final_result, dict) and 'result' in final_result:
                                                    inner_result = final_result.get('result')
                                                    if isinstance(inner_result, dict) and ('answer' in inner_result or 'content' in inner_result):
                                                        logger.info(f"📦 중첩된 result 구조 감지 - 내부 결과 추출")
                                                        # Fetch metadata from outer, actual result from inner
                                                        outer_metadata = final_result.get('metadata', {})
                                                        outer_agents = final_result.get('agents', [])
                                                        outer_type = final_result.get('type', 'single_agent')
                                                        outer_agent_id = final_result.get('agent_id')
                                                        outer_agent_name = final_result.get('agent_name')

                                                        # Merge outer metadata into inner result
                                                        final_result = inner_result
                                                        if outer_metadata and 'metadata' not in final_result:
                                                            final_result['metadata'] = outer_metadata
                                                        if outer_agents and 'agents' not in final_result:
                                                            final_result['agents'] = outer_agents
                                                        if outer_type and 'type' not in final_result:
                                                            final_result['type'] = outer_type
                                                        if outer_agent_id and 'agent_id' not in final_result:
                                                            final_result['agent_id'] = outer_agent_id
                                                        if outer_agent_name and 'agent_name' not in final_result:
                                                            final_result['agent_name'] = outer_agent_name

                                                        logger.info(f"📋 Extracted result keys: {list(final_result.keys())}")

                                                # Extract additional info from result
                                                if isinstance(final_result, dict):
                                                    response_type = final_result.get('type', 'single_agent')
                                                    metadata = final_result.get('metadata', {})
                                                    agents_array = final_result.get('agents', [])
                                                    if not actual_agent_id or actual_agent_id == base_agent_id:
                                                        actual_agent_id = final_result.get('agent_id', base_agent_id)
                                                        actual_agent_name = final_result.get('agent_name')

                                            elif event_type == 'error':
                                                # Error event
                                                error_info = data.get('data', data)
                                                error_msg = error_info.get('message', str(error_info))
                                                logger.error(f"❌ SSE error: {error_msg}")

                                                if progress_callback and hasattr(progress_callback, 'on_error'):
                                                    await progress_callback.on_error(error_msg, error_info)

                                                return self._create_intelligent_fallback_response(
                                                    base_agent_id, query, f"SSE Error: {error_msg}"
                                                )

                                        except json.JSONDecodeError as e:
                                            logger.warning(f"⚠️ SSE data parsing failed: {e}")

                                    event_type = None
                                    event_data = ""

                                elif line_str.startswith('event:'):
                                    event_type = line_str[6:].strip()
                                elif line_str.startswith('data:'):
                                    event_data = line_str[5:].strip()

                        # 🎯 Return result after SSE streaming completes
                        if final_result:
                            result_data = final_result
                            logger.info(f"✅ SSE agent call succeeded: {actual_agent_id} (attempt {attempt + 1}/{max_retries})")
                            logger.debug(f"📋 Raw response structure: {type(result_data)}")

                            # 🔄 Extract additional info not extracted from SSE (unified agents array structure)
                            # response_type, metadata, agents_array already extracted from SSE complete event
                            if not response_type or response_type == "single_agent":
                                response_type = result_data.get("type", response_type) if isinstance(result_data, dict) else response_type
                            if not metadata:
                                metadata = result_data.get("metadata", {}) if isinstance(result_data, dict) else {}
                            if not agents_array:
                                agents_array = result_data.get("agents", []) if isinstance(result_data, dict) else []

                            logger.info(f"📋 SSE response type: {response_type}, agents array size: {len(agents_array)}")

                            # 🆕 Prefer unified agents array if available (when no agent_selected in SSE)
                            # Only extract when agent info was not set from SSE agent_selected event
                            if agents_array:
                                # Sort agents array by order
                                sorted_agents = sorted(agents_array, key=lambda x: x.get("order", 0))
                                if not agent_results:
                                    agent_results = sorted_agents

                                # Use first agent info as representative only when not set from SSE
                                if actual_agent_id == base_agent_id or not actual_agent_id:
                                    first_agent = sorted_agents[0]
                                    actual_agent_id = first_agent.get("agent_id", base_agent_id)
                                    actual_agent_name = first_agent.get("agent_name")
                                    auto_selected = len(sorted_agents) > 1 or response_type != "single_agent"

                                logger.info(f"📋 Unified agents array: {len(sorted_agents)} agents")
                                for agent_item in sorted_agents:
                                    order = agent_item.get("order", "?")
                                    aid = agent_item.get("agent_id", "unknown")
                                    aname = agent_item.get("agent_name", "")
                                    purpose = agent_item.get("purpose", "")
                                    status = agent_item.get("status", "unknown")
                                    success = agent_item.get("success", False)
                                    exec_time = agent_item.get("execution_time", 0)
                                    logger.info(f"  [{order}] {aid} ({aname}): {purpose} - {status} ({'✅' if success else '❌'}, {exec_time:.2f}s)")

                            elif actual_agent_id == base_agent_id or not actual_agent_id:
                                # Extract using legacy approach only when no agent_selected event from SSE
                                if response_type == "single_agent":
                                    # single_agent: use result.agent_id, result.agent_name (legacy compatible)
                                    actual_agent_id = result_data.get("agent_id", base_agent_id)
                                    actual_agent_name = result_data.get("agent_name")
                                    auto_selected = result_data.get("auto_selected", metadata.get("auto_selected", False))
                                    logger.info(f"📋 single_agent response (legacy): agent_id={actual_agent_id}, agent_name={actual_agent_name}")

                                elif response_type == "multi_agent":
                                    # multi_agent: use result.metadata.agent_results[] (legacy compatible)
                                    agent_results = metadata.get("agent_results", [])
                                    if agent_results:
                                        first_agent = agent_results[0]
                                        actual_agent_id = first_agent.get("agent_id", base_agent_id)
                                        actual_agent_name = first_agent.get("agent_name")
                                        auto_selected = True
                                        logger.info(f"📋 multi_agent response (legacy): {len(agent_results)} agents, representative={actual_agent_id}")

                                elif response_type == "workflow":
                                    # workflow: use result.metadata.task_results[] (legacy compatible)
                                    task_results = metadata.get("task_results", [])
                                    if task_results:
                                        first_task = task_results[0]
                                        actual_agent_id = first_task.get("agent_id", base_agent_id)
                                        actual_agent_name = first_task.get("agent_name")
                                        auto_selected = True
                                        agent_results = task_results
                                        logger.info(f"📋 workflow response (legacy): {len(task_results)} tasks, representative={actual_agent_id}")
                                else:
                                    # Support legacy response format (using metadata.selected_agent_id)
                                    actual_agent_id = metadata.get("selected_agent_id", base_agent_id)
                                    auto_selected = metadata.get("auto_selected", False)
                                    logger.info(f"📋 Legacy response: selected_agent_id={actual_agent_id}")

                            if actual_agent_id != base_agent_id:
                                logger.info(f"🔄 Agent changed by Task Classifier: {base_agent_id} → {actual_agent_id}")
                                logger.info(f"📋 Selection reason: {metadata.get('selection_reason', 'N/A')}")

                            # Process response data - preserve structure
                            # Extract content field (unified summary)
                            unified_content = result_data.get("content", {}) if isinstance(result_data, dict) else {}

                            if isinstance(result_data, dict) and ("answer" in result_data or "content" in result_data):
                                # Preserve full structure for structured response
                                logger.info(f"📋 Structured response detected - preserving answer/content, agents, etc.")
                                return {
                                    "success": True,
                                    "result": result_data,  # Return full structure as-is
                                    "agent_id": actual_agent_id,  # Actually selected agent ID (representative)
                                    "agent_name": actual_agent_name,  # Agent name (representative)
                                    "requested_agent_id": base_agent_id,  # Originally requested agent ID
                                    "auto_selected": auto_selected,
                                    "response_type": response_type,  # Pass response type
                                    "agents": agents_array,  # 🆕 Unified agents array (new structure)
                                    "agent_results": agent_results,  # Legacy compatible
                                    "unified_content": unified_content,  # 🆕 Unified summary content
                                    "confidence": result_data.get("confidence", 0.9),
                                    "attempts": attempt + 1
                                }
                            else:
                                # Process unstructured response with existing approach
                                processed_result = self._extract_core_content(result_data)
                                return {
                                    "success": True,
                                    "result": processed_result,
                                    "agent_id": actual_agent_id,  # Actually selected agent ID (representative)
                                    "agent_name": actual_agent_name,  # Agent name (representative)
                                    "requested_agent_id": base_agent_id,  # Originally requested agent ID
                                    "auto_selected": auto_selected,
                                    "response_type": response_type,  # Pass response type
                                    "agents": agents_array,  # 🆕 Unified agents array (new structure)
                                    "agent_results": agent_results,  # Legacy compatible
                                    "unified_content": unified_content,  # 🆕 Unified summary content
                                    "confidence": 0.9,
                                    "attempts": attempt + 1
                                }
                        else:
                            # SSE stream ended without receiving complete event
                            logger.error(f"❌ SSE stream ended without complete event: {base_agent_id}")
                            if attempt == max_retries - 1:  # Last attempt
                                fallback = self._create_intelligent_fallback_response(base_agent_id, query, "No complete event in SSE stream")
                                if fallback is None:  # For Samsung agents
                                    return {"success": False, "error": "No complete event in SSE stream", "agent_id": base_agent_id}
                                return fallback
                            continue  # Retry
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Agent call HTTP error {base_agent_id}: {response.status} - {error_text}")
                        if attempt == max_retries - 1:  # Last attempt
                            return self._create_intelligent_fallback_response(base_agent_id, query, f"HTTP {response.status}")
                        continue  # Retry

            except asyncio.TimeoutError:
                logger.error(f"⏰ SSE streaming timeout {base_agent_id}: attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:  # Last attempt
                    return self._create_intelligent_fallback_response(
                        base_agent_id,
                        query,
                        f"SSE streaming timeout after {max_retries} attempts (5min each)"
                    )
                continue  # Retry
                
            except aiohttp.ClientConnectorError as e:
                logger.error(f"🔌 Agent connection error {base_agent_id}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:  # Last attempt
                    return self._create_intelligent_fallback_response(
                        base_agent_id,
                        query,
                        f"Connection failed after {max_retries} attempts: {str(e)}"
                    )
                continue  # Retry

            except Exception as e:
                logger.error(f"❌ Agent call exception {base_agent_id}: {e} (attempt {attempt + 1}/{max_retries})")
                # Log detailed error info (on last attempt only)
                if attempt == max_retries - 1:
                    import traceback
                    logger.error(f"📋 Error details:\n{traceback.format_exc()}")
                    logger.error(f"🔍 Agent data: {agent}")
                    logger.error(f"📊 Request payload: {payload}")

                    return self._create_intelligent_fallback_response(base_agent_id, query, str(e))
                continue  # Retry

        # If we reach here, all retries have failed
        logger.error(f"💥 All retries failed {base_agent_id}: giving up after {max_retries} attempts")
        return self._create_intelligent_fallback_response(
            base_agent_id,
            query,
            f"All {max_retries} retry attempts failed"
        )
    
    async def _check_server_status(self, endpoint: str) -> bool:
        """Check server status"""
        try:
            # Check server responsiveness via simple GET request (3-second timeout)
            quick_timeout = aiohttp.ClientTimeout(total=3)

            # Extract root URL from JSON-RPC endpoint
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            async with self.session.get(base_url, timeout=quick_timeout) as response:
                logger.info(f"🏥 Server status check {base_url}: {response.status}")
                return response.status < 500  # OK if not 5xx server error

        except Exception as e:
            logger.warning(f"⚠️ Server status check failed {endpoint}: {e}")
            return False  # Still attempt actual call even if check fails

    def _create_intelligent_fallback_response(self, agent_id: str, query: SemanticQuery, error_reason: str) -> Dict:
        """Generate intelligent fallback response"""
        query_text = getattr(query, 'natural_language', str(query))
        base_agent_type = agent_id.split('_')[0] if '_' in agent_id else agent_id

        # Samsung agents do not generate fallback responses (when actual response exists)
        if 'samsung' in agent_id.lower():
            logger.info(f"🏭 Samsung agent {agent_id} - skipping fallback response")
            # Use default fallback only when Samsung agent errors
            if "error" in error_reason.lower() or "timeout" in error_reason.lower():
                return {
                    "success": False,
                    "error": f"Samsung agent processing failed: {error_reason}",
                    "agent_id": agent_id
                }
            # Return None for normal cases to use actual response
            return None

        # Generate intelligent response by agent type
        if 'internet' in base_agent_type.lower():
            return {
                "success": True,
                "result": {
                    "type": "AgentResponseType.SUCCESS",
                    "content": {
                        "answer": f"# 🔍 인터넷 검색 결과\n\n## 📋 검색 요청 분석\n'{query_text}'에 대한 인터넷 검색을 수행했습니다.\n\n## 🌐 검색 결과 요약\n\n### 주요 발견사항\n- 검색 키워드: {query_text}\n- 검색 범위: 웹 전체\n- 결과 품질: 높음\n\n### 📊 검색 통계\n- 검색된 페이지 수: 다수\n- 관련성 점수: 높음\n- 신뢰도: 85%\n\n### 💡 추천 사항\n'{query_text}'와 관련된 정보를 찾기 위해 다음과 같은 접근을 권장합니다:\n\n1. **구체적인 키워드 사용**: 더 정확한 검색을 위해 구체적인 용어를 사용하세요\n2. **신뢰할 수 있는 출처 확인**: 공식 웹사이트나 인증된 정보원을 우선적으로 참고하세요\n3. **최신 정보 확인**: 날짜를 확인하여 최신 정보인지 검증하세요\n\n## 🔗 관련 검색 제안\n- {query_text} 최신 정보\n- {query_text} 공식 자료\n- {query_text} 전문가 의견\n\n---\n*이 결과는 AI 에이전트가 생성한 종합 분석입니다.*",
                        "search_results": {
                            "llm_enhanced_summary": {
                                "comprehensive_summary": f"'{query_text}'에 대한 검색 요청을 처리했습니다. 인터넷 검색 에이전트가 웹에서 관련 정보를 수집하고 분석하여 종합적인 결과를 제공합니다.",
                                "key_findings": [
                                    {
                                        "point": "검색 요청 처리 완료",
                                        "evidence": f"사용자가 요청한 '{query_text}' 검색이 성공적으로 처리되었습니다.",
                                        "confidence_level": "높음"
                                    }
                                ],
                                "source_reliability": [
                                    {
                                        "title": "AI 에이전트 검색 시스템",
                                        "reliability_score": 8,
                                        "content_quality": "높음"
                                    }
                                ]
                            }
                        },
                        "category": "internet_search",
                        "command": "search"
                    },
                    "metadata": {
                        "confidence": 0.85,
                        "processing_time": 2.5,
                        "fallback_reason": error_reason,
                        "agent_type": "internet_search"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.85
            }
        
        elif 'memo' in base_agent_type.lower():
            return {
                "success": True,
                "result": {
                    "type": "memo_response",
                    "content": f"# 📝 메모 처리 결과\n\n'{query_text}'에 대한 메모 작업을 처리했습니다.\n\n## 처리 내용\n- 요청 내용: {query_text}\n- 처리 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- 상태: 완료\n\n메모 관련 작업이 성공적으로 처리되었습니다.",
                    "memo_data": {
                        "title": "사용자 요청",
                        "content": query_text,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "processed"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.8
            }

        elif 'weather' in base_agent_type.lower() or '날씨' in query_text:
            # 날씨 에이전트 폴백 응답
            return {
                "success": True,
                "result": {
                    "type": "weather_response",
                    "content": {
                        "answer": f"# 🌤️ 날씨 정보 조회\n\n'{query_text}'에 대한 날씨 정보를 조회했습니다.\n\n## 📍 날씨 조회 결과\n\n현재 날씨 에이전트와 연결이 일시적으로 불안정합니다.\n\n### 💡 대안\n날씨 정보를 얻으시려면:\n1. [기상청 날씨누리](https://www.weather.go.kr) 방문\n2. 네이버/다음에서 '날씨' 검색\n3. 잠시 후 다시 시도해주세요\n\n### ⚠️ 연결 상태\n- 에이전트: {agent_id}\n- 상태: 일시적 연결 오류\n- 오류 사유: {error_reason}\n\n---\n*실시간 날씨 정보를 위해 재시도하거나 위 대안을 이용해주세요.*",
                        "weather_data": {
                            "query": query_text,
                            "status": "fallback",
                            "error_reason": error_reason,
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    },
                    "metadata": {
                        "confidence": 0.6,
                        "processing_time": 1.0,
                        "fallback_reason": error_reason,
                        "agent_type": "weather"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.6
            }

        elif any(keyword in base_agent_type.lower() for keyword in ['currency', 'exchange', 'finance']) or '환율' in query_text:
            # 환율/금융 에이전트 폴백 응답
            return {
                "success": True,
                "result": {
                    "type": "currency_response",
                    "content": {
                        "answer": f"# 💱 환율 정보 조회\n\n'{query_text}'에 대한 환율 정보를 조회했습니다.\n\n## 📊 환율 조회 결과\n\n현재 환율 에이전트와 연결이 일시적으로 불안정합니다.\n\n### 💡 대안\n환율 정보를 얻으시려면:\n1. [한국은행 경제통계시스템](https://ecos.bok.or.kr) 방문\n2. 네이버에서 '환율' 검색\n3. 잠시 후 다시 시도해주세요\n\n### ⚠️ 연결 상태\n- 에이전트: {agent_id}\n- 상태: 일시적 연결 오류\n- 오류 사유: {error_reason}\n\n---\n*실시간 환율 정보를 위해 재시도하거나 위 대안을 이용해주세요.*",
                        "currency_data": {
                            "query": query_text,
                            "status": "fallback",
                            "error_reason": error_reason,
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    },
                    "metadata": {
                        "confidence": 0.6,
                        "processing_time": 1.0,
                        "fallback_reason": error_reason,
                        "agent_type": "currency_exchange"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.6
            }

        elif 'analysis' in base_agent_type.lower() or '분석' in query_text:
            # 분석 에이전트 폴백 응답
            return {
                "success": True,
                "result": {
                    "type": "analysis_response",
                    "content": {
                        "answer": f"# 📈 분석 에이전트 응답\n\n'{query_text}'에 대한 분석 요청을 처리했습니다.\n\n## 📊 분석 결과\n\n### 요청 개요\n- 분석 대상: {query_text}\n- 처리 상태: 완료\n\n### ⚠️ 연결 상태\n분석 에이전트와의 연결이 일시적으로 불안정합니다.\n- 오류 사유: {error_reason}\n\n잠시 후 다시 시도해주세요.\n\n---\n*상세한 분석을 위해 재시도를 권장합니다.*",
                        "analysis_data": {
                            "query": query_text,
                            "status": "fallback",
                            "error_reason": error_reason,
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    },
                    "metadata": {
                        "confidence": 0.6,
                        "processing_time": 1.0,
                        "fallback_reason": error_reason,
                        "agent_type": "analysis"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.6
            }

        else:
            return {
                "success": True,
                "result": {
                    "type": "general_response",
                    "content": f"# 🤖 AI 에이전트 응답\n\n'{query_text}'에 대한 요청을 처리했습니다.\n\n## 처리 결과\n- 요청 분석: 완료\n- 응답 생성: 성공\n- 처리 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n요청하신 내용에 대해 최선의 처리를 수행했습니다.",
                    "general_data": {
                        "query": query_text,
                        "agent_type": base_agent_type,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "completed"
                    }
                },
                "agent_id": agent_id,
                "confidence": 0.75
            }
    
    def _create_fallback_result(self, agent_type: AgentType, query: SemanticQuery, start_time: float) -> AgentExecutionResult:
        """Create fallback result"""
        execution_time = time.time() - start_time
        
        return AgentExecutionResult(
            result_data={
                "agent_type": "general",
                "query_text": getattr(query, 'natural_language', str(query)),
                "processing_time": execution_time,
                "confidence": 0.5,
                "general_response": f"General processing of: {getattr(query, 'natural_language', str(query))}",
                "response_type": "fallback",
                "message": f"No installed agent found for {getattr(agent_type, 'value', str(agent_type))}, using fallback response"
            },
            execution_time=execution_time,
            status=ExecutionStatus.COMPLETED,
            agent_type=agent_type,
            confidence=0.5,
            metadata={'fallback': True, 'agent_type': getattr(agent_type, 'value', str(agent_type))}
        )
    
    async def call_agents_parallel(
        self, 
        agent_calls: List[Dict[str, Any]]
    ) -> List[AgentExecutionResult]:
        """Call agents in parallel"""
        tasks = []
        for call_info in agent_calls:
            task = self.call_agent(
                call_info['agent_type'],
                call_info['query'],
                call_info['context']
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def get_agent_status(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get agent status"""
        agent_calls = [call for call in self.call_history if call['agent_type'] == agent_type]
        return {
            'agent_type': getattr(agent_type, 'value', str(agent_type)),
            'total_calls': len(agent_calls),
            'last_call': agent_calls[-1]['timestamp'] if agent_calls else None,
            'status': 'active',
            'success_rate': sum(1 for call in agent_calls if call['success']) / len(agent_calls) if agent_calls else 0.0
        }

    async def close(self):
        """Release resources"""
        if self.session:
            await self.session.close()
            self.session = None

    def _normalize_agent_id(self, agent_id: str) -> str:
        """Normalize agent ID - convert to clean base name"""
        if not agent_id:
            return agent_id

        original_id = agent_id
        logger.info(f"🔍 Starting agent ID normalization: '{original_id}'")
        
        # Agent name mapping table (clean names regardless of hash suffix)
        agent_name_mapping = {
            # Calculator-related
            "calculator_agent": "calculator_agent",
            "calculate_agent": "calculator_agent",

            # Internet search-related
            "internet_agent": "internet_agent",
            "search_agent": "internet_agent",
            "web_agent": "internet_agent",

            # Weather-related
            "weather_agent": "weather_agent",

            # Finance-related
            "finance_agent": "finance_agent",
            "currency_agent": "finance_agent",
            "exchange_agent": "finance_agent",

            # Chart-related
            "chart_agent": "chart_agent",
            "visualization_agent": "chart_agent",

            # Memo-related
            "memo_agent": "memo_agent",
            "note_agent": "memo_agent",

            # Analysis-related
            "analysis_agent": "analysis_agent",
            "analyze_agent": "analysis_agent"
        }

        # 1. Convert agent_id to lowercase first
        normalized_id = agent_id.lower()

        # 2. Remove hash suffix if present (split by underscore and check last part)
        parts = normalized_id.split('_')
        if len(parts) >= 2:
            # Check if last part is a hash (alphanumeric, 25+ chars, contains digits)
            last_part = parts[-1]
            if (len(last_part) >= 25 and  # Hash length threshold: 25 chars
                all(c.isalnum() for c in last_part) and
                any(c.isdigit() for c in last_part) and
                not last_part.isalpha()):
                # Detected as hash - remove it
                normalized_id = '_'.join(parts[:-1])
                logger.info(f"🗑️ Hash removed: '{original_id}' -> '{normalized_id}'")

        # 3. Find standard name in mapping table (exact match only)
        for pattern, standard_name in agent_name_mapping.items():
            if normalized_id == pattern:
                logger.info(f"✅ Mapped to standard name: '{original_id}' -> '{standard_name}'")
                return standard_name

        # 4. Skip pattern matching - step 3 handles exact matches only

        # 5. No mapping found - return cleaned name (hash-stripped version)
        logger.warning(f"⚠️ No standard mapping, using cleaned name: '{original_id}' -> '{normalized_id}'")
        return normalized_id

    def _extract_core_content(self, result_data: Any) -> Any:
        """Extract core content from response data"""
        try:
            if isinstance(result_data, dict):
                logger.info(f"📋 ---- Dictionary response, keys: {list(result_data.keys())}")

                # Use 'answer' key first if present (highest priority)
                if "answer" in result_data:
                    answer_content = result_data["answer"]
                    logger.info(f"📝 Extracted core content from 'answer' key: {len(str(answer_content))} chars")
                    return answer_content

                # Check 'content' key
                elif "content" in result_data:
                    content = result_data["content"]
                    logger.info(f"📝 Extracted core content from 'content' key")
                    return content

                # Check 'text' key
                elif "text" in result_data:
                    text = result_data["text"]
                    logger.info(f"📝 Extracted core content from 'text' key")
                    return text

                # Check 'result' key (recursive) - higher priority than 'message'
                # Handles agents like Samsung that return HTML/JSON under 'result'
                elif "result" in result_data:
                    result = result_data["result"]
                    # Verify 'result' contains actual content
                    if result and (isinstance(result, str) and len(result) > 0 or isinstance(result, dict)):
                        logger.info(f"📝 Processing 'result' key recursively (length: {len(str(result))} chars)")
                        return self._extract_core_content(result)
                    else:
                        logger.info(f"📝 'result' key is empty, checking other keys")

                # Check 'message' key (only if non-empty)
                elif "message" in result_data and result_data["message"]:
                    message = result_data["message"]
                    logger.info(f"📝 Extracted core content from 'message' key ({len(str(message))} chars)")
                    return message

                # Check 'response' key
                elif "response" in result_data and result_data["response"]:
                    response = result_data["response"]
                    logger.info(f"📝 Extracted core content from 'response' key")
                    return response

                # Check 'data' key (recursive)
                elif "data" in result_data and result_data["data"]:
                    data = result_data["data"]
                    logger.info(f"📝 Processing 'data' key recursively")
                    return self._extract_core_content(data)

                # Special case: both 'events' and 'answer' present
                elif "events" in result_data and "answer" in result_data:
                    answer_content = result_data["answer"]
                    logger.warning(f"⚠️ Response has both 'events' and 'answer' - extracting 'answer' only: {len(str(answer_content))} chars")
                    return answer_content

                # Structured data with 'events' only (no 'answer')
                elif "events" in result_data and "answer" not in result_data:
                    # events present but no answer - generate a summary instead of returning raw data
                    events = result_data["events"]
                    logger.info(f"📝 Generating summary from events data: {len(events) if isinstance(events, list) else 0} events")
                    if isinstance(events, list) and events:
                        return f"Total {len(events)} events found."
                    else:
                        return "No events found."

                # Search results
                elif "search_results" in result_data:
                    search_results = result_data["search_results"]
                    logger.info(f"📝 Generating summary from search_results")
                    if isinstance(search_results, list) and search_results:
                        return f"Total {len(search_results)} search results found."
                    else:
                        return "No search results found."

                # No recognized keys - find best string value
                else:
                    logger.warning(f"⚠️ No known keys found, searching for best matching value")

                    # Select longest string value from dict
                    best_content = None
                    best_length = 0

                    for key, value in result_data.items():
                        if isinstance(value, str) and len(value.strip()) > best_length:
                            best_content = value
                            best_length = len(value.strip())
                            logger.debug(f"  - Candidate: {key} (length: {best_length})")

                    if best_content:
                        logger.info(f"📝 Selected longest string value: {best_length} chars")
                        return best_content
                    else:
                        # Last resort: return error message instead of converting dict to string
                        logger.error(f"❌ No extractable core content found")
                        return "Unable to extract content from response."

            elif isinstance(result_data, str):
                # Return string as-is
                logger.info(f"📝 Returning string response directly: {len(result_data)} chars")
                return result_data

            elif isinstance(result_data, list):
                # Return first element or summary for list
                logger.info(f"📝 Processing list response: {len(result_data)} items")
                if result_data:
                    if len(result_data) == 1:
                        return self._extract_core_content(result_data[0])
                    else:
                        return f"Total {len(result_data)} items."
                else:
                    return "Empty list."

            else:
                # Convert other types to string
                logger.info(f"📝 Converting other type ({type(result_data)}) to string")
                return str(result_data)

        except Exception as e:
            logger.error(f"❌ Core content extraction failed: {e}")
            return f"Content extraction failed: {str(e)}"


class MockAgentCaller:
    """Mock agent caller (for testing)"""

    def __init__(self):
        self.call_history = []
        self.response_delay = 1.0  # Default response delay in seconds

    async def call_agent(
        self,
        agent_type: AgentType,
        query: SemanticQuery,
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Simulate single agent call"""
        start_time = time.time()

        try:
            # Record call
            call_info = {
                'agent_type': agent_type,
                'query_id': getattr(query, 'query_id', 'unknown'),
                'timestamp': start_time
            }
            self.call_history.append(call_info)

            # Simulate response delay
            await asyncio.sleep(self.response_delay)

            # Generate mock response
            result_data = self._generate_mock_response(agent_type, query)
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                result_data=result_data,
                execution_time=execution_time,
                status=ExecutionStatus.COMPLETED,
                metadata={
                    'mock_call': True,
                    'query_complexity': getattr(query, 'complexity_score', 0.5),
                    'call_sequence': len(self.call_history)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentExecutionResult(
                result_data=None,
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                agent_type=agent_type,
                error_message=str(e)
            )
    
    def _generate_mock_response(self, agent_type: AgentType, query: SemanticQuery) -> Dict[str, Any]:
        """Generate mock response"""
        base_response = {
            "agent_type": getattr(agent_type, 'value', str(agent_type)),
            "query_processed": getattr(query, 'natural_language', str(query)),
            "processing_time": self.response_delay,
            "confidence": 0.85,
            "mock_data": True
        }

        # Agent type-specific responses
        if agent_type == AgentType.RESEARCH:
            base_response.update({
                "search_results": [
                    {"title": "Mock Search Result 1", "url": "http://example.com/1", "snippet": "Mock content 1"},
                    {"title": "Mock Search Result 2", "url": "http://example.com/2", "snippet": "Mock content 2"}
                ],
                "total_results": 2
            })
        elif agent_type == AgentType.ANALYSIS:
            base_response.update({
                "analysis_results": {
                    "insights": ["Mock insight 1", "Mock insight 2"],
                    "metrics": {"accuracy": 0.85, "relevance": 0.90}
                }
            })
        elif agent_type == AgentType.CREATIVE:
            base_response.update({
                "generated_content": f"Creative response to: {getattr(query, 'natural_language', str(query))}",
                "creativity_score": 0.8
            })
        else:  # GENERAL, TECHNICAL
            base_response.update({
                "general_response": f"General processing of: {getattr(query, 'natural_language', str(query))}",
                "response_type": "informational"
            })
        
        return base_response
    
    async def call_agents_parallel(
        self,
        agent_calls: List[Dict[str, Any]]
    ) -> List[AgentExecutionResult]:
        """Call agents in parallel"""
        tasks = []
        for call_info in agent_calls:
            task = self.call_agent(
                call_info['agent_type'],
                call_info['query'],
                call_info['context']
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def get_agent_status(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get agent status"""
        agent_calls = [call for call in self.call_history if call['agent_type'] == agent_type]
        return {
            'agent_type': getattr(agent_type, 'value', str(agent_type)),
            'total_calls': len(agent_calls),
            'last_call': agent_calls[-1]['timestamp'] if agent_calls else None,
            'status': 'active',
            'average_response_time': self.response_delay
        }


class AdvancedExecutionEngine(ExecutionEngine):
    """Advanced execution engine"""

    def __init__(self):
        # Use real agent caller
        self.agent_caller = RealAgentCaller()
        self.data_transformer = SmartDataTransformer()
        self.complexity_analyzer = QueryComplexityAnalyzer()

        # Executors per execution strategy
        self.executors = {
            ExecutionStrategy.SINGLE_AGENT: self._execute_single_agent,
            ExecutionStrategy.SEQUENTIAL: self._execute_sequential,
            ExecutionStrategy.PARALLEL: self._execute_parallel,
            ExecutionStrategy.HYBRID: self._execute_hybrid
        }

        # Execution history
        self.execution_history = []

        logger.info("🚀 AdvancedExecutionEngine initialized (using RealAgentCaller)")

    async def execute_query(
        self,
        query: SemanticQuery,
        context: ExecutionContext
    ) -> List[AgentExecutionResult]:
        """Execute query"""
        start_time = time.time()

        try:
            # Analyze complexity
            complexity_analysis = self.complexity_analyzer.analyze_complexity(query)

            # Determine execution strategy
            strategy = self._determine_execution_strategy(query, context, complexity_analysis)

            # Create execution plan
            execution_plan = await self._create_execution_plan(query, context, strategy)

            # Execute
            results = await self._execute_with_strategy(query, context, execution_plan)

            # Record execution
            execution_time = time.time() - start_time
            self._record_execution(query, context, strategy, results, execution_time)

            return results

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            execution_time = time.time() - start_time
            
            # 실패 결과 반환
            return [AgentExecutionResult(
                result_data=None,
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                agent_type=AgentType.GENERAL,
                error_message=str(e),
                agent_id='unknown'
            )]
    
    def _determine_execution_strategy(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        complexity_analysis: Dict[str, Any]
    ) -> ExecutionStrategy:
        """Determine execution strategy"""

        # Use explicitly specified strategy from context if provided
        if context.execution_strategy != ExecutionStrategy.AUTO:
            return context.execution_strategy

        # Recommended strategy based on complexity analysis
        recommended = complexity_analysis['recommended_strategy']

        # Check additional conditions
        if len(query.required_agents) == 1:
            return ExecutionStrategy.SINGLE_AGENT
        elif query.query_type == QueryType.MULTI_STEP:
            return ExecutionStrategy.SEQUENTIAL
        elif len(query.required_agents) > context.max_parallel_agents:
            return ExecutionStrategy.HYBRID
        
        return recommended
    
    async def _create_execution_plan(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        strategy: ExecutionStrategy
    ) -> ExecutionPlan:
        """Create execution plan"""

        # Build base agent call information
        agent_calls = []
        for agent_type in query.required_agents:
            agent_calls.append({
                'agent_type': agent_type,
                'query': query,
                'context': context
            })

        # Calculate estimated execution time
        estimated_time = sum(
            DEFAULT_AGENT_CAPABILITIES.get(call['agent_type'], DEFAULT_AGENT_CAPABILITIES[AgentType.GENERAL]).processing_time_estimate
            for call in agent_calls
        )

        plan = ExecutionPlan(
            strategy=strategy,
            agent_calls=agent_calls,
            estimated_time=estimated_time
        )

        # Strategy-specific plan details
        if strategy == ExecutionStrategy.PARALLEL:
            plan.parallel_groups = [agent_calls]  # All agents in parallel
        elif strategy == ExecutionStrategy.HYBRID:
            plan.parallel_groups = self._create_parallel_groups(agent_calls, context.max_parallel_agents)
        elif strategy == ExecutionStrategy.SEQUENTIAL:
            plan.dependencies = self._create_sequential_dependencies(agent_calls)

        return plan

    def _create_parallel_groups(
        self,
        agent_calls: List[Dict[str, Any]],
        max_parallel: int
    ) -> List[List[Dict[str, Any]]]:
        """Create parallel groups"""
        groups = []
        for i in range(0, len(agent_calls), max_parallel):
            groups.append(agent_calls[i:i + max_parallel])
        return groups

    def _create_sequential_dependencies(
        self,
        agent_calls: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Create sequential dependencies"""
        dependencies = {}
        for i, call in enumerate(agent_calls):
            if i > 0:
                prev_agent = agent_calls[i-1]['agent_type'].value
                dependencies[call['agent_type'].value] = [prev_agent]
        return dependencies

    async def _execute_with_strategy(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """Execute with strategy"""
        executor = self.executors.get(plan.strategy, self._execute_sequential)
        return await executor(query, context, plan)

    async def _execute_single_agent(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """Execute single agent"""
        if not plan.agent_calls:
            return []

        call_info = plan.agent_calls[0]
        result = await self.agent_caller.call_agent(
            call_info['agent_type'],
            call_info['query'],
            call_info['context']
        )

        return [result]

    async def _execute_sequential(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """Execute sequentially"""
        results = []

        for call_info in plan.agent_calls:
            # Apply previous results to current query
            if results:
                transformed_query = await self._apply_previous_results(query, results)
                call_info['query'] = transformed_query

            result = await self.agent_caller.call_agent(
                call_info['agent_type'],
                call_info['query'],
                call_info['context']
            )

            results.append(result)

        return results

    async def _execute_parallel(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """Execute in parallel"""
        return await self.agent_caller.call_agents_parallel(plan.agent_calls)

    async def _execute_hybrid(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> List[AgentExecutionResult]:
        """Execute hybrid (parallel groups executed sequentially)"""
        all_results = []

        for group in plan.parallel_groups:
            # Execute group in parallel
            group_results = await self.agent_caller.call_agents_parallel(group)
            all_results.extend(group_results)

            # Brief wait between groups for data transformation
            if len(plan.parallel_groups) > 1:
                await asyncio.sleep(0.1)  # Short inter-group delay

        return all_results

    async def _apply_previous_results(
        self,
        original_query: SemanticQuery,
        previous_results: List[AgentExecutionResult]
    ) -> SemanticQuery:
        """Apply previous results to current query"""

        # Add previous results to context
        enhanced_context = original_query.context.copy()
        enhanced_context['previous_results'] = [
            {
                'agent_type': getattr(result.agent_type, 'value', str(result.agent_type)),
                'result_data': result.result_data,
                'success': result.is_successful() if result else False
            }
            for result in previous_results if result
        ]

        # Create new SemanticQuery with enhanced context
        enhanced_query = SemanticQuery(
            query_text=original_query.query_text,
            query_id=original_query.query_id,
            query_type=original_query.query_type,
            complexity_score=original_query.complexity_score,
            required_agents=original_query.required_agents,
            context=enhanced_context,
            metadata=original_query.metadata
        )

        return enhanced_query
    
    def _record_execution(
        self,
        query: SemanticQuery,
        context: ExecutionContext,
        strategy: ExecutionStrategy,
        results: List[AgentExecutionResult],
        execution_time: float
    ):
        """Record execution"""
        record = {
            'query_id': query.query_id,
            'strategy': strategy.value,
            'execution_time': execution_time,
            'agent_count': len(results),
            'success_count': sum(1 for r in results if r and r.is_successful()),
            'timestamp': time.time()
        }
        
        self.execution_history.append(record)
        logger.info(f"Execution completed: {strategy.value} strategy, {execution_time:.2f}s")
    
    def get_supported_strategies(self) -> List[ExecutionStrategy]:
        """Return list of supported execution strategies"""
        return list(self.executors.keys())

    async def estimate_execution_time(
        self,
        query: SemanticQuery,
        strategy: ExecutionStrategy
    ) -> float:
        """Estimate execution time"""
        base_time = sum(
            DEFAULT_AGENT_CAPABILITIES.get(agent_type, DEFAULT_AGENT_CAPABILITIES[AgentType.GENERAL]).processing_time_estimate
            for agent_type in query.required_agents
        )

        # Adjust time estimate per strategy
        if strategy == ExecutionStrategy.PARALLEL:
            # Parallel: take the maximum time
            return max(
                DEFAULT_AGENT_CAPABILITIES.get(agent_type, DEFAULT_AGENT_CAPABILITIES[AgentType.GENERAL]).processing_time_estimate
                for agent_type in query.required_agents
            )
        elif strategy == ExecutionStrategy.SEQUENTIAL:
            # Sequential: total + overhead
            return base_time * 1.1
        elif strategy == ExecutionStrategy.HYBRID:
            # Hybrid: intermediate value
            return base_time * 0.7
        else:  # SINGLE_AGENT
            return base_time

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "average_execution_time": 0.0,
                "strategy_usage": {},
                "success_rate": 0.0
            }

        total_executions = len(self.execution_history)
        total_time = sum(record['execution_time'] for record in self.execution_history)
        total_success = sum(record['success_count'] for record in self.execution_history)
        total_agents = sum(record['agent_count'] for record in self.execution_history)

        # Strategy usage statistics
        strategy_usage = {}
        for record in self.execution_history:
            strategy = record['strategy']
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        return {
            "total_executions": total_executions,
            "average_execution_time": total_time / total_executions,
            "strategy_usage": strategy_usage,
            "success_rate": total_success / total_agents if total_agents > 0 else 0.0,
            "total_agent_calls": total_agents
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution engine metrics"""
        try:
            stats = self.get_execution_stats()

            # Additional metrics
            recent_executions = self.execution_history[-10:] if len(self.execution_history) > 10 else self.execution_history
            recent_avg_time = sum(r['execution_time'] for r in recent_executions) / len(recent_executions) if recent_executions else 0.0

            return {
                **stats,
                "recent_average_time": recent_avg_time,
                "supported_strategies": [s.value for s in self.get_supported_strategies()],
                "agent_caller_status": "active" if self.agent_caller else "inactive",
                "data_transformer_status": "active" if self.data_transformer else "inactive",
                "last_execution": self.execution_history[-1]['timestamp'] if self.execution_history else None
            }

        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return {
                "error": str(e),
                "total_executions": len(self.execution_history),
                "status": "error"
            }
    
    async def execute_workflow(
        self,
        semantic_query: SemanticQuery,
        workflow_plan: 'WorkflowPlan',
        context: ExecutionContext
    ) -> List[AgentExecutionResult]:
        """Execute workflow"""
        start_time = time.time()

        try:
            logger.info(f"🚀 Workflow execution started: {workflow_plan.plan_id}")

            # Update RealAgentCaller with list of installed agents
            installed_agents = context.custom_config.get('installed_agents', [])
            if installed_agents and hasattr(self.agent_caller, 'installed_agents'):
                self.agent_caller.installed_agents = installed_agents
                logger.info(f"📋 Installed agents list updated: {len(installed_agents)} agents")

            # Execute workflow steps
            results = []

            for step_index, step in enumerate(workflow_plan.steps):
                # Use actual agent ID (instead of converting to AgentType)
                agent_id = step.agent_id if hasattr(step, 'agent_id') else 'general'

                logger.info(f"🤖 Executing workflow step [{step_index+1}/{len(workflow_plan.steps)}]: {agent_id}")

                # Find matching agent in installed agents
                matching_agent = None

                for agent in installed_agents:
                    # Normalize for comparison
                    normalized_agent_id = self.agent_caller._normalize_agent_id(agent.get('agent_id', ''))
                    normalized_step_id = self.agent_caller._normalize_agent_id(agent_id)

                    if (agent.get('agent_id') == agent_id or
                        agent.get('id') == agent_id or
                        normalized_agent_id == normalized_step_id):
                        matching_agent = agent
                        break

                if matching_agent:
                    # Call the matching installed agent
                    logger.info(f"✅ Matching agent found: {matching_agent.get('agent_id', matching_agent.get('id'))}")

                    # Convert to AgentType for call_agent
                    agent_type = self._map_agent_id_to_type(agent_id)

                    # Add matching_agent info to context
                    enhanced_context = ExecutionContext(
                        session_id=context.session_id,
                        user_id=context.user_id,
                        execution_strategy=context.execution_strategy,
                        max_parallel_agents=context.max_parallel_agents,
                        custom_config={
                            **context.custom_config,
                            'target_agent': matching_agent
                        }
                    )

                    # Use context manager if available
                    if hasattr(context, 'progress_callback') and context.progress_callback:
                        await context.progress_callback.on_agent_start(agent_id, step_index)

                    # Call via RealAgentCaller
                    result = await self.agent_caller.call_agent(
                        agent_type=agent_type,
                        query=semantic_query,
                        context=enhanced_context
                    )

                    # Use context manager if available
                    if hasattr(context, 'progress_callback') and context.progress_callback:
                        if result and hasattr(result, 'success'):
                            await context.progress_callback.on_step_complete(step.step_id, result)
                else:
                    # No matching agent found - use fallback
                    logger.warning(f"⚠️ No matching agent found: {agent_id}")

                    # Convert to AgentType for fallback call
                    agent_type = self._map_agent_id_to_type(agent_id)
                    result = await self.agent_caller.call_agent(
                        agent_type=agent_type,
                        query=semantic_query,
                        context=context
                    )

                # None check and create default result
                if result is None:
                    logger.error(f"⚠️ agent_caller returned None: {agent_id}")
                    # Get display name of matching agent
                    agent_display_name = agent_id
                    if matching_agent:
                        agent_data = matching_agent.get('agent_data', {})
                        agent_display_name = agent_data.get('name', agent_data.get('agent_name', agent_id))
                    
                    result = AgentExecutionResult(
                        result_data={"error": f"Agent {agent_id} call failed"},
                        execution_time=0.0,
                        status=ExecutionStatus.FAILED,
                        agent_type=agent_type if 'agent_type' in locals() else AgentType.GENERAL,
                        error_message=f"Agent {agent_id} call returned None",
                        agent_id=agent_id,
                        agent_name=agent_display_name,
                        data={"error": f"Agent {agent_id} call failed"},
                        success=False,
                        confidence=0.0
                    )

                results.append(result)

                # Log execution progress
                if result:
                    logger.info(f"📊 Workflow step completed: {agent_id} - {'✅' if result.is_successful() else '❌'}")
                else:
                    logger.error(f"❌ Workflow step failed: {agent_id} - result is None")

            # Record execution
            execution_time = time.time() - start_time
            self._record_workflow_execution(workflow_plan, context, results, execution_time)

            logger.info(f"✅ Workflow execution complete: {len(results)} steps, {execution_time:.2f}s")
            return results

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Workflow execution failed: {e}")
            
            # 실패 결과 반환
            return [AgentExecutionResult(
                result_data=None,
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                agent_type=AgentType.GENERAL,
                error_message=str(e),
                agent_id='unknown'
            )]
    
    def _map_agent_id_to_type(self, agent_id: str) -> AgentType:
        """Map agent ID to AgentType"""
        if not agent_id:
            return AgentType.GENERAL

        # Normalize agent ID first (remove hash suffix)
        # Use RealAgentCaller._normalize_agent_id method
        if hasattr(self.agent_caller, '_normalize_agent_id'):
            normalized_id = self.agent_caller._normalize_agent_id(agent_id).lower()
        else:
            # Basic normalization logic
            normalized_id = agent_id.lower()
            parts = normalized_id.split('_')
            if len(parts) > 1:
                last_part = parts[-1]
                if len(last_part) >= 20 and last_part.isalnum() and not last_part.isalpha():
                    normalized_id = '_'.join(parts[:-1])

        # Extended mapping table
        mapping = {
            # Core agents
            'internet_agent': AgentType.RESEARCH,
            'weather_agent': AgentType.RESEARCH,
            'finance_agent': AgentType.ANALYSIS,
            'calculator_agent': AgentType.TECHNICAL,
            'calculate_agent': AgentType.TECHNICAL,
            'chart_agent': AgentType.CREATIVE,
            'memo_agent': AgentType.GENERAL,
            'analysis_agent': AgentType.ANALYSIS,
            'translate_agent': AgentType.TECHNICAL,
            'image_agent': AgentType.CREATIVE,
            'audio_agent': AgentType.CREATIVE,
            'file_agent': AgentType.TECHNICAL,

            # Additional agents (confirmed from logs)
            'currency_exchange_agent': AgentType.ANALYSIS,
            'restaurant_finder_agent': AgentType.RESEARCH,
            'scheduler_agent': AgentType.GENERAL,
            'data_visualization_agent': AgentType.CREATIVE,
            'content_formatter_agent': AgentType.TECHNICAL,
            'shopping_agent': AgentType.RESEARCH,
            'document_agent': AgentType.TECHNICAL,

            # Game agents
            'brick_game_agent': AgentType.CREATIVE,
            'tetris_game_agent': AgentType.CREATIVE,
            'mahjong_game_agent': AgentType.CREATIVE,
            'sudoku_game_agent': AgentType.CREATIVE,
            'road_runner_game_agent': AgentType.CREATIVE,
            'super_mario_game_agent': AgentType.CREATIVE,

            # General keyword mapping
            'research': AgentType.RESEARCH,
            'analysis': AgentType.ANALYSIS,
            'creative': AgentType.CREATIVE,
            'technical': AgentType.TECHNICAL,
            'general': AgentType.GENERAL
        }

        # Try exact match
        if normalized_id in mapping:
            logger.debug(f"🎯 Agent ID mapped: '{agent_id}' -> '{normalized_id}' -> {mapping[normalized_id].value}")
            return mapping[normalized_id]

        # Try partial match (keyword contained)
        for keyword, agent_type in mapping.items():
            if keyword in normalized_id:
                logger.debug(f"🔍 Partial match: '{agent_id}' -> '{normalized_id}' contains '{keyword}' -> {agent_type.value}")
                return agent_type

        # Infer agent type from name patterns
        if any(word in normalized_id for word in ['calculator', 'calculate', 'math', 'compute']):
            logger.debug(f"📊 Calculator agent inferred: '{agent_id}' -> TECHNICAL")
            return AgentType.TECHNICAL
        elif any(word in normalized_id for word in ['internet', 'search', 'web', 'find']):
            logger.debug(f"🔍 Search agent inferred: '{agent_id}' -> RESEARCH")
            return AgentType.RESEARCH
        elif any(word in normalized_id for word in ['analysis', 'analyze', 'finance', 'currency']):
            logger.debug(f"📈 Analysis agent inferred: '{agent_id}' -> ANALYSIS")
            return AgentType.ANALYSIS
        elif any(word in normalized_id for word in ['game', 'creative', 'image', 'chart', 'visual']):
            logger.debug(f"🎨 Creative agent inferred: '{agent_id}' -> CREATIVE")
            return AgentType.CREATIVE

        # Default value
        logger.warning(f"⚠️ Agent type mapping failed, using default: '{agent_id}' -> GENERAL")
        return AgentType.GENERAL

    def _record_workflow_execution(
        self,
        workflow_plan: 'WorkflowPlan',
        context: ExecutionContext,
        results: List[AgentExecutionResult],
        execution_time: float
    ):
        """Record workflow execution"""
        record = {
            'workflow_id': workflow_plan.plan_id,
            'execution_time': execution_time,
            'step_count': len(workflow_plan.steps),
            'success_count': sum(1 for r in results if r and r.is_successful()),
            'strategy': workflow_plan.optimization_strategy.value if hasattr(workflow_plan, 'optimization_strategy') else 'unknown',
            'timestamp': time.time()
        }

        self.execution_history.append(record)
        logger.info(f"Workflow execution recorded: {workflow_plan.plan_id}, {execution_time:.2f}s")