"""
Enhanced Ontology Query Processor
Integrates with the enhanced workflow processor for better agent selection and query analysis
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class QueryCategory(str, Enum):
    INFORMATION_RETRIEVAL = "information_retrieval"
    DATA_ANALYSIS = "data_analysis" 
    CONTENT_GENERATION = "content_generation"
    TASK_AUTOMATION = "task_automation"
    CONVERSATION = "conversation"
    COMPOUND_TASK = "compound_task"

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class QueryAnalysisResult:
    """Enhanced query analysis result with detailed insights"""
    query: str
    category: QueryCategory
    complexity: ComplexityLevel
    intent: str
    entities: List[str]
    required_capabilities: List[str]
    subtasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    parallel_groups: List[List[str]]
    suggested_agents: List[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class AgentCapability:
    """Enhanced agent capability representation"""
    agent_id: str
    agent_type: str
    name: str
    domains: List[str]
    capabilities: List[str]
    complexity_range: Tuple[str, str]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class EnhancedOntologyQueryProcessor:
    """
    Enhanced query processor that integrates with the workflow processor
    for better agent selection and query understanding
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=2000
        )
        self.agent_capabilities_cache: Dict[str, AgentCapability] = {}
        
    async def analyze_query(self, query: str, available_agents: List[Dict[str, Any]]) -> QueryAnalysisResult:
        """
        Perform comprehensive query analysis with enhanced understanding
        """
        # Update agent capabilities cache
        self._update_agent_capabilities(available_agents)
        
        # Parallel analysis tasks
        analysis_tasks = [
            self._analyze_category_and_complexity(query),
            self._extract_entities_and_intent(query),
            self._decompose_into_subtasks(query),
            self._analyze_dependencies(query)
        ]
        
        results = await asyncio.gather(*analysis_tasks)
        category_result, entity_result, subtask_result, dependency_result = results
        
        # Determine required capabilities based on analysis
        required_capabilities = self._determine_required_capabilities(
            category_result, entity_result, subtask_result
        )
        
        # Select optimal agents
        suggested_agents = await self._select_optimal_agents(
            required_capabilities, subtask_result, available_agents
        )
        
        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(subtask_result, dependency_result)
        
        return QueryAnalysisResult(
            query=query,
            category=category_result['category'],
            complexity=category_result['complexity'],
            intent=entity_result['intent'],
            entities=entity_result['entities'],
            required_capabilities=required_capabilities,
            subtasks=subtask_result,
            dependencies=dependency_result,
            parallel_groups=parallel_groups,
            suggested_agents=suggested_agents,
            confidence=category_result.get('confidence', 0.8),
            metadata={
                'analysis_version': '2.0',
                'processor': 'enhanced_ontology'
            }
        )
    
    async def _analyze_category_and_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query category and complexity using LLM"""
        prompt = f"""Analyze this query and determine its category and complexity.

Query: {query}

Categories:
- information_retrieval: Searching or fetching information
- data_analysis: Analyzing, calculating, or processing data
- content_generation: Creating new content or documents
- task_automation: Automating tasks or workflows
- conversation: General conversation or Q&A
- compound_task: Multiple different tasks combined

Complexity levels:
- simple: Single straightforward task
- moderate: 2-3 related tasks or some processing required
- complex: Multiple tasks with dependencies or extensive processing

Return JSON format:
{{
    "category": "category_name",
    "complexity": "complexity_level",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a query analysis expert."),
            HumanMessage(content=prompt)
        ])
        
        try:
            result = json.loads(response.content)
            return {
                'category': QueryCategory(result['category']),
                'complexity': ComplexityLevel(result['complexity']),
                'confidence': result.get('confidence', 0.8),
                'reasoning': result.get('reasoning', '')
            }
        except:
            return {
                'category': QueryCategory.INFORMATION_RETRIEVAL,
                'complexity': ComplexityLevel.SIMPLE,
                'confidence': 0.5
            }
    
    async def _extract_entities_and_intent(self, query: str) -> Dict[str, Any]:
        """Extract entities and primary intent from query"""
        prompt = f"""Extract key information from this query.

Query: {query}

Return JSON format:
{{
    "intent": "primary user intent",
    "entities": ["entity1", "entity2", ...],
    "action_verbs": ["verb1", "verb2", ...],
    "constraints": ["constraint1", "constraint2", ...]
}}"""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an NLP expert in entity and intent extraction."),
            HumanMessage(content=prompt)
        ])
        
        try:
            result = json.loads(response.content)
            return result
        except:
            return {
                'intent': 'information_retrieval',
                'entities': [],
                'action_verbs': [],
                'constraints': []
            }
    
    async def _decompose_into_subtasks(self, query: str) -> List[Dict[str, Any]]:
        """Decompose query into subtasks with detailed analysis"""
        prompt = f"""Decompose this query into subtasks that can be executed by different agents.

Query: {query}

For each subtask, provide:
- id: unique identifier (e.g., "task_1")
- description: what needs to be done
- type: the type of task (search, analyze, generate, etc.)
- input_required: what input this task needs
- output_expected: what output this task produces
- estimated_duration: rough estimate in seconds

Return JSON format:
{{
    "subtasks": [
        {{
            "id": "task_1",
            "description": "...",
            "type": "...",
            "input_required": ["..."],
            "output_expected": "...",
            "estimated_duration": 10
        }}
    ]
}}"""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a task decomposition expert."),
            HumanMessage(content=prompt)
        ])
        
        try:
            result = json.loads(response.content)
            return result.get('subtasks', [])
        except:
            return [{
                'id': 'task_1',
                'description': query,
                'type': 'general',
                'input_required': [],
                'output_expected': 'response',
                'estimated_duration': 30
            }]
    
    async def _analyze_dependencies(self, query: str) -> Dict[str, List[str]]:
        """Analyze dependencies between subtasks"""
        prompt = f"""Analyze task dependencies for this query.

Query: {query}

Identify which tasks depend on outputs from other tasks.
Return a dictionary where keys are task IDs and values are lists of task IDs they depend on.

Return JSON format:
{{
    "dependencies": {{
        "task_2": ["task_1"],
        "task_3": ["task_1", "task_2"]
    }}
}}"""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a workflow dependency expert."),
            HumanMessage(content=prompt)
        ])
        
        try:
            result = json.loads(response.content)
            return result.get('dependencies', {})
        except:
            return {}
    
    def _determine_required_capabilities(
        self, 
        category_result: Dict[str, Any],
        entity_result: Dict[str, Any],
        subtasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Determine required agent capabilities based on analysis"""
        capabilities = set()
        
        # Category-based capabilities
        category_capabilities = {
            QueryCategory.INFORMATION_RETRIEVAL: ['search', 'retrieve', 'query'],
            QueryCategory.DATA_ANALYSIS: ['analyze', 'calculate', 'process'],
            QueryCategory.CONTENT_GENERATION: ['generate', 'create', 'write'],
            QueryCategory.TASK_AUTOMATION: ['automate', 'execute', 'schedule'],
            QueryCategory.CONVERSATION: ['chat', 'discuss', 'respond'],
            QueryCategory.COMPOUND_TASK: ['coordinate', 'integrate', 'manage']
        }
        
        if category_result['category'] in category_capabilities:
            capabilities.update(category_capabilities[category_result['category']])
        
        # Action verb based capabilities
        for verb in entity_result.get('action_verbs', []):
            capabilities.add(verb.lower())
        
        # Subtask type based capabilities
        for subtask in subtasks:
            task_type = subtask.get('type', '').lower()
            if task_type:
                capabilities.add(task_type)
        
        return list(capabilities)
    
    async def _select_optimal_agents(
        self,
        required_capabilities: List[str],
        subtasks: List[Dict[str, Any]],
        available_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Select optimal agents using enhanced matching logic"""
        selected_agents = []
        
        for subtask in subtasks:
            best_agent = await self._find_best_agent_for_task(
                subtask, required_capabilities, available_agents
            )
            if best_agent and best_agent not in selected_agents:
                selected_agents.append(best_agent)
        
        # Ensure we have at least one agent
        if not selected_agents and available_agents:
            # Fallback to internet search agent
            for agent in available_agents:
                if 'internet' in agent.get('id', '').lower() or \
                   'search' in agent.get('id', '').lower():
                    selected_agents.append(agent['id'])
                    break
            
            # Last resort: first available agent
            if not selected_agents:
                selected_agents.append(available_agents[0]['id'])
        
        return selected_agents
    
    async def _find_best_agent_for_task(
        self,
        subtask: Dict[str, Any],
        required_capabilities: List[str],
        available_agents: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Find the best agent for a specific subtask"""
        task_type = subtask.get('type', '').lower()
        task_description = subtask.get('description', '').lower()
        
        best_agent = None
        best_score = 0
        
        for agent in available_agents:
            score = 0
            agent_id = agent.get('id', '').lower()
            agent_name = agent.get('name', '').lower()
            agent_description = agent.get('description', '').lower()
            
            # Type matching
            if task_type in agent_id or task_type in agent_name:
                score += 3
            
            # Capability matching
            if agent_id in self.agent_capabilities_cache:
                agent_cap = self.agent_capabilities_cache[agent_id]
                for cap in required_capabilities:
                    if cap in agent_cap.capabilities:
                        score += 2
            
            # Description matching
            keywords = task_description.split()
            for keyword in keywords:
                if len(keyword) > 3:  # Skip short words
                    if keyword in agent_id or keyword in agent_name or keyword in agent_description:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_agent = agent['id']
        
        return best_agent
    
    def _identify_parallel_groups(
        self,
        subtasks: List[Dict[str, Any]],
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Identify groups of tasks that can be executed in parallel"""
        if not subtasks:
            return []
        
        # Build task ID list
        task_ids = [task['id'] for task in subtasks]
        
        # Find tasks with no dependencies (can start immediately)
        independent_tasks = []
        for task_id in task_ids:
            if task_id not in dependencies or not dependencies[task_id]:
                independent_tasks.append(task_id)
        
        # Group tasks by dependency level
        groups = []
        processed = set()
        
        # First group: independent tasks
        if independent_tasks:
            groups.append(independent_tasks)
            processed.update(independent_tasks)
        
        # Subsequent groups: tasks whose dependencies are all processed
        while len(processed) < len(task_ids):
            next_group = []
            for task_id in task_ids:
                if task_id not in processed:
                    deps = dependencies.get(task_id, [])
                    if all(dep in processed for dep in deps):
                        next_group.append(task_id)
            
            if next_group:
                groups.append(next_group)
                processed.update(next_group)
            else:
                # Prevent infinite loop - add remaining tasks
                remaining = [t for t in task_ids if t not in processed]
                if remaining:
                    groups.append(remaining)
                break
        
        return groups
    
    def _update_agent_capabilities(self, available_agents: List[Dict[str, Any]]):
        """Update agent capabilities cache"""
        for agent in available_agents:
            agent_id = agent.get('id', '')
            
            # Extract capabilities from agent metadata
            capabilities = []
            agent_type = agent.get('type', '').lower()
            
            # Type-based capability inference
            type_capabilities = {
                'search': ['search', 'query', 'find', 'retrieve'],
                'analysis': ['analyze', 'calculate', 'process', 'evaluate'],
                'generation': ['generate', 'create', 'write', 'produce'],
                'chat': ['chat', 'converse', 'discuss', 'respond'],
                'tool': ['execute', 'run', 'perform', 'automate']
            }
            
            for key, caps in type_capabilities.items():
                if key in agent_type:
                    capabilities.extend(caps)
            
            # Extract from description
            description = agent.get('description', '').lower()
            capability_keywords = [
                'search', 'analyze', 'generate', 'create', 'calculate',
                'process', 'extract', 'summarize', 'translate', 'convert'
            ]
            
            for keyword in capability_keywords:
                if keyword in description:
                    capabilities.append(keyword)
            
            # Create capability object
            self.agent_capabilities_cache[agent_id] = AgentCapability(
                agent_id=agent_id,
                agent_type=agent_type,
                name=agent.get('name', ''),
                domains=agent.get('domains', []),
                capabilities=list(set(capabilities)),
                complexity_range=('simple', 'complex'),
                performance_metrics={},
                metadata=agent
            )
    
    async def select_agents_simple(self, query: str, available_agents: List[Dict[str, Any]]) -> List[str]:
        """
        Simple agent selection method for backward compatibility
        Used by the existing workflow designer
        """
        analysis = await self.analyze_query(query, available_agents)
        return analysis.suggested_agents