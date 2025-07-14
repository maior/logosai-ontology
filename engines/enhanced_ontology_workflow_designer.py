"""
Enhanced Ontology Workflow Designer
Integrates with the enhanced workflow processor for better workflow planning
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import time
import uuid
import asyncio

from ..core.models import (
    SemanticQuery, WorkflowPlan, WorkflowStep, 
    ExecutionStrategy, OptimizationStrategy, WorkflowComplexity
)
from ..core.interfaces import WorkflowDesigner as IWorkflowDesigner
from ..processors.enhanced_ontology_query_processor import (
    EnhancedOntologyQueryProcessor, QueryAnalysisResult
)


class EnhancedOntologyWorkflowDesigner(IWorkflowDesigner):
    """Enhanced workflow designer with improved query analysis and parallel execution support"""
    
    def __init__(self, installed_agents_info: List[Dict[str, Any]] = None):
        # 실제 설치된 에이전트 정보 저장
        self.installed_agents_info = installed_agents_info or []
        self.agents_capabilities_cache = {}
        
        # Enhanced query processor
        self.query_processor = EnhancedOntologyQueryProcessor()
        
        # 에이전트 간 의존성 규칙
        self.dependency_rules = {
            "chart": ["search", "data", "analysis"],
            "analysis": ["search", "data"],
            "memo": ["search", "analysis", "chart"],
            "report": ["search", "analysis", "data"]
        }
        
        logger.info(f"🎯 EnhancedOntologyWorkflowDesigner 초기화 완료 - 설치된 에이전트: {len(self.installed_agents_info)}개")
    
    async def design_workflow(self, 
                            semantic_query: SemanticQuery,
                            available_agents: List[str]) -> WorkflowPlan:
        """
        Enhanced workflow design with comprehensive query analysis
        """
        try:
            logger.info(f"🎯 Enhanced 워크플로우 설계 시작 - 쿼리: {semantic_query.natural_language[:100]}...")
            
            if not available_agents:
                logger.warning("사용 가능한 에이전트가 없습니다.")
                return self._create_fallback_workflow(semantic_query, [])
            
            # Convert agent IDs to agent info format for query processor
            agent_info_list = self._convert_to_agent_info(available_agents)
            
            # 1. Enhanced query analysis
            query_analysis = await self.query_processor.analyze_query(
                semantic_query.natural_language, 
                agent_info_list
            )
            
            logger.info(f"Query Analysis Result:")
            logger.info(f"  - Category: {query_analysis.category}")
            logger.info(f"  - Complexity: {query_analysis.complexity}")
            logger.info(f"  - Intent: {query_analysis.intent}")
            logger.info(f"  - Suggested Agents: {query_analysis.suggested_agents}")
            logger.info(f"  - Parallel Groups: {query_analysis.parallel_groups}")
            
            # 2. Create workflow steps based on analysis
            workflow_steps = await self._create_enhanced_workflow_steps(
                semantic_query, query_analysis, available_agents
            )
            
            # 3. Build execution graph with parallel support
            execution_graph = self._build_enhanced_execution_graph(
                workflow_steps, query_analysis
            )
            
            # 4. Determine execution strategy
            execution_strategy = self._determine_execution_strategy(query_analysis)
            
            # 5. Estimate metrics
            estimated_quality, estimated_time = self._estimate_enhanced_metrics(
                workflow_steps, query_analysis
            )
            
            # 6. Generate reasoning chain
            reasoning_chain = self._generate_enhanced_reasoning_chain(
                query_analysis, workflow_steps, execution_strategy
            )
            
            # 7. Create workflow plan
            workflow_plan = WorkflowPlan.create_simple(
                plan_id=f"workflow_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                semantic_query=semantic_query,
                steps=workflow_steps,
                execution_graph=execution_graph,
                strategy=self._map_to_optimization_strategy(execution_strategy),
                quality=estimated_quality,
                time=estimated_time,
                reasoning=reasoning_chain
            )
            
            # Add metadata for execution engine
            workflow_plan.metadata = {
                'query_analysis': query_analysis,
                'execution_strategy': execution_strategy,
                'parallel_groups': query_analysis.parallel_groups,
                'complexity_level': query_analysis.complexity.value
            }
            
            logger.info(f"✅ Enhanced 워크플로우 설계 완료 - {len(workflow_steps)}개 단계, 예상 시간: {estimated_time:.1f}초")
            return workflow_plan
            
        except Exception as e:
            logger.error(f"❌ Enhanced 워크플로우 설계 실패: {e}")
            return self._create_fallback_workflow(semantic_query, available_agents)
    
    def _convert_to_agent_info(self, agent_ids: List[str]) -> List[Dict[str, Any]]:
        """Convert agent IDs to agent info format"""
        agent_info_list = []
        
        for agent_id in agent_ids:
            # Find in installed agents
            agent_info = None
            for installed in self.installed_agents_info:
                if installed.get('agent_id') == agent_id:
                    agent_data = installed.get('agent_data', {})
                    agent_info = {
                        'id': agent_id,
                        'name': agent_data.get('name', agent_id),
                        'type': agent_data.get('metadata', {}).get('agent_type', 'CUSTOM'),
                        'description': agent_data.get('description', ''),
                        'capabilities': [cap.get('id', '') for cap in agent_data.get('capabilities', [])]
                    }
                    break
            
            # Fallback: create basic info
            if not agent_info:
                agent_info = {
                    'id': agent_id,
                    'name': agent_id,
                    'type': self._infer_type_from_id(agent_id),
                    'description': f'Agent {agent_id}',
                    'capabilities': []
                }
            
            agent_info_list.append(agent_info)
        
        return agent_info_list
    
    def _infer_type_from_id(self, agent_id: str) -> str:
        """Infer agent type from ID"""
        agent_id_lower = agent_id.lower()
        
        if 'internet' in agent_id_lower or 'search' in agent_id_lower:
            return 'search'
        elif 'memo' in agent_id_lower:
            return 'storage'
        elif 'chart' in agent_id_lower or 'visual' in agent_id_lower:
            return 'visualization'
        elif 'analysis' in agent_id_lower or 'analyze' in agent_id_lower:
            return 'analysis'
        elif 'calc' in agent_id_lower or 'math' in agent_id_lower:
            return 'calculation'
        else:
            return 'general'
    
    async def _create_enhanced_workflow_steps(
        self,
        semantic_query: SemanticQuery,
        query_analysis: QueryAnalysisResult,
        available_agents: List[str]
    ) -> List[WorkflowStep]:
        """Create workflow steps with enhanced analysis"""
        steps = []
        step_index = 0
        
        # Map subtasks to agents
        for subtask in query_analysis.subtasks:
            # Find best agent for this subtask
            agent_id = None
            for suggested_agent in query_analysis.suggested_agents:
                if suggested_agent in available_agents:
                    # Check if agent matches subtask type
                    if self._agent_matches_subtask(suggested_agent, subtask):
                        agent_id = suggested_agent
                        break
            
            # Fallback to any available agent
            if not agent_id and query_analysis.suggested_agents:
                agent_id = query_analysis.suggested_agents[0]
            
            if agent_id:
                # Determine dependencies
                dependencies = []
                if subtask['id'] in query_analysis.dependencies:
                    for dep_task_id in query_analysis.dependencies[subtask['id']]:
                        # Find step ID for dependent task
                        for prev_step in steps:
                            if prev_step.metadata.get('subtask_id') == dep_task_id:
                                dependencies.append(prev_step.step_id)
                
                # Create step
                step = WorkflowStep.create_simple(
                    step_id=f"step_{step_index:06d}",
                    agent_id=agent_id,
                    purpose=subtask['description'],
                    concepts=query_analysis.entities,
                    complexity=self._map_complexity(query_analysis.complexity),
                    depends_on=dependencies,
                    estimated_time=subtask.get('estimated_duration', 20.0)
                )
                
                # Add metadata
                step.metadata = {
                    'subtask_id': subtask['id'],
                    'subtask_type': subtask['type'],
                    'input_required': subtask.get('input_required', []),
                    'output_expected': subtask.get('output_expected', '')
                }
                
                steps.append(step)
                step_index += 1
        
        # If no steps created, use suggested agents directly
        if not steps and query_analysis.suggested_agents:
            for agent_id in query_analysis.suggested_agents:
                if agent_id in available_agents:
                    step = WorkflowStep.create_simple(
                        step_id=f"step_{step_index:06d}",
                        agent_id=agent_id,
                        purpose=f"Process query: {semantic_query.natural_language[:100]}",
                        concepts=query_analysis.entities,
                        complexity=self._map_complexity(query_analysis.complexity),
                        depends_on=[],
                        estimated_time=30.0
                    )
                    steps.append(step)
                    step_index += 1
                    break
        
        return steps
    
    def _agent_matches_subtask(self, agent_id: str, subtask: Dict[str, Any]) -> bool:
        """Check if agent matches subtask requirements"""
        agent_type = self._infer_type_from_id(agent_id)
        task_type = subtask.get('type', '').lower()
        
        # Type-based matching
        type_mapping = {
            'search': ['search', 'retrieve', 'find', 'query'],
            'analysis': ['analyze', 'process', 'evaluate', 'research'],
            'visualization': ['chart', 'graph', 'visualize', 'plot'],
            'storage': ['save', 'store', 'memo', 'record'],
            'calculation': ['calculate', 'compute', 'math']
        }
        
        for agent_key, task_types in type_mapping.items():
            if agent_type == agent_key and task_type in task_types:
                return True
        
        # Keyword matching in description
        task_desc = subtask.get('description', '').lower()
        return agent_type in task_desc or task_type in agent_id.lower()
    
    def _build_enhanced_execution_graph(
        self,
        workflow_steps: List[WorkflowStep],
        query_analysis: QueryAnalysisResult
    ) -> nx.DiGraph:
        """Build execution graph with parallel execution support"""
        graph = nx.DiGraph()
        
        # Add nodes
        for step in workflow_steps:
            graph.add_node(step.step_id, step=step)
        
        # Add dependency edges
        for step in workflow_steps:
            for dependency in step.depends_on:
                if dependency in [s.step_id for s in workflow_steps]:
                    graph.add_edge(dependency, step.step_id)
        
        # Add parallel group metadata
        if hasattr(graph, 'graph'):
            graph.graph['parallel_groups'] = query_analysis.parallel_groups
        
        # Verify DAG
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("순환 참조 감지, 제거 중...")
            self._remove_cycles(graph)
        
        return graph
    
    def _determine_execution_strategy(self, query_analysis: QueryAnalysisResult) -> ExecutionStrategy:
        """Determine optimal execution strategy"""
        # Based on complexity and parallel opportunities
        if query_analysis.complexity.value == "simple":
            return ExecutionStrategy.SEQUENTIAL
        
        if len(query_analysis.parallel_groups) > 1:
            # Multiple parallel groups - use parallel or hybrid
            if query_analysis.complexity.value == "complex":
                return ExecutionStrategy.HYBRID
            else:
                return ExecutionStrategy.PARALLEL
        
        # Default based on subtask count
        if len(query_analysis.subtasks) == 1:
            return ExecutionStrategy.SINGLE_AGENT
        elif len(query_analysis.subtasks) <= 3:
            return ExecutionStrategy.SEQUENTIAL
        else:
            return ExecutionStrategy.PARALLEL
    
    def _map_complexity(self, complexity_level) -> WorkflowComplexity:
        """Map query complexity to workflow complexity"""
        mapping = {
            "simple": WorkflowComplexity.SIMPLE,
            "moderate": WorkflowComplexity.MODERATE,
            "complex": WorkflowComplexity.COMPLEX
        }
        return mapping.get(complexity_level.value, WorkflowComplexity.MODERATE)
    
    def _map_to_optimization_strategy(self, execution_strategy: ExecutionStrategy) -> OptimizationStrategy:
        """Map execution strategy to optimization strategy"""
        mapping = {
            ExecutionStrategy.SINGLE_AGENT: OptimizationStrategy.SPEED_FIRST,
            ExecutionStrategy.SEQUENTIAL: OptimizationStrategy.BALANCED,
            ExecutionStrategy.PARALLEL: OptimizationStrategy.SPEED_FIRST,
            ExecutionStrategy.HYBRID: OptimizationStrategy.QUALITY_FIRST,
            ExecutionStrategy.AUTO: OptimizationStrategy.BALANCED
        }
        return mapping.get(execution_strategy, OptimizationStrategy.BALANCED)
    
    def _estimate_enhanced_metrics(
        self,
        workflow_steps: List[WorkflowStep],
        query_analysis: QueryAnalysisResult
    ) -> Tuple[float, float]:
        """Estimate quality and time with parallel execution consideration"""
        # Quality based on agent count and complexity
        base_quality = 0.7
        quality_bonus = len(workflow_steps) * 0.05
        complexity_bonus = {
            "simple": 0.0,
            "moderate": 0.1,
            "complex": 0.2
        }.get(query_analysis.complexity.value, 0.1)
        
        estimated_quality = min(0.95, base_quality + quality_bonus + complexity_bonus)
        
        # Time estimation with parallel execution
        if not workflow_steps:
            return estimated_quality, 30.0
        
        # Calculate critical path time
        if query_analysis.parallel_groups:
            # Time is max of each parallel group
            group_times = []
            for group in query_analysis.parallel_groups:
                group_time = 0
                for task_id in group:
                    # Find corresponding step
                    for step in workflow_steps:
                        if step.metadata.get('subtask_id') == task_id:
                            group_time = max(group_time, step.estimated_time)
                            break
                if group_time > 0:
                    group_times.append(group_time)
            
            estimated_time = sum(group_times) if group_times else sum(s.estimated_time for s in workflow_steps)
        else:
            # Sequential execution
            estimated_time = sum(s.estimated_time for s in workflow_steps)
        
        # Apply optimization factor
        if len(query_analysis.parallel_groups) > 1:
            estimated_time *= 0.7  # 30% improvement from parallelization
        
        return estimated_quality, estimated_time
    
    def _generate_enhanced_reasoning_chain(
        self,
        query_analysis: QueryAnalysisResult,
        workflow_steps: List[WorkflowStep],
        execution_strategy: ExecutionStrategy
    ) -> List[str]:
        """Generate detailed reasoning chain"""
        reasoning = []
        
        # Query understanding
        reasoning.append(f"Query category: {query_analysis.category.value}")
        reasoning.append(f"Complexity level: {query_analysis.complexity.value}")
        reasoning.append(f"User intent: {query_analysis.intent}")
        
        # Entity recognition
        if query_analysis.entities:
            reasoning.append(f"Key entities: {', '.join(query_analysis.entities[:5])}")
        
        # Task decomposition
        reasoning.append(f"Decomposed into {len(query_analysis.subtasks)} subtasks")
        
        # Agent selection
        reasoning.append(f"Selected {len(workflow_steps)} agents for execution")
        
        # Execution strategy
        reasoning.append(f"Execution strategy: {execution_strategy.value}")
        
        # Parallel optimization
        if len(query_analysis.parallel_groups) > 1:
            reasoning.append(f"Identified {len(query_analysis.parallel_groups)} parallel execution groups")
            reasoning.append("Parallel execution will improve performance by ~30%")
        
        # Dependencies
        dep_count = sum(1 for step in workflow_steps if step.depends_on)
        if dep_count > 0:
            reasoning.append(f"{dep_count} steps have dependencies")
        
        return reasoning
    
    def _remove_cycles(self, graph: nx.DiGraph):
        """Remove cycles from graph"""
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    # Remove last edge
                    graph.remove_edge(cycle[-1], cycle[0])
                    logger.info(f"Removed cycle: {cycle[-1]} -> {cycle[0]}")
        except Exception as e:
            logger.error(f"Failed to remove cycles: {e}")
    
    def _create_fallback_workflow(self, semantic_query: SemanticQuery, available_agents: List[str]) -> WorkflowPlan:
        """Create fallback workflow"""
        logger.warning("Creating fallback workflow")
        
        # Use first available agent or default
        fallback_agent = available_agents[0] if available_agents else "default_agent"
        
        fallback_step = WorkflowStep.create_simple(
            agent_id=fallback_agent,
            purpose=f"Process query: {semantic_query.natural_language[:100]}",
            concepts=["general"],
            complexity=WorkflowComplexity.SIMPLE,
            estimated_time=30.0
        )
        
        return WorkflowPlan.create_simple(
            semantic_query=semantic_query,
            steps=[fallback_step],
            strategy=OptimizationStrategy.BALANCED,
            quality=0.6,
            time=30.0,
            reasoning=["Fallback workflow created"]
        )
    
    def optimize_workflow(self, workflow_plan: WorkflowPlan) -> WorkflowPlan:
        """Optimize workflow with enhanced strategies"""
        try:
            logger.info("🔧 Enhanced workflow optimization starting")
            
            # Extract metadata
            query_analysis = workflow_plan.metadata.get('query_analysis')
            if not query_analysis:
                logger.warning("No query analysis metadata, using basic optimization")
                return workflow_plan
            
            # Apply optimizations based on execution strategy
            execution_strategy = workflow_plan.metadata.get('execution_strategy', ExecutionStrategy.SEQUENTIAL)
            
            if execution_strategy in [ExecutionStrategy.PARALLEL, ExecutionStrategy.HYBRID]:
                # Already optimized for parallel execution
                logger.info("Workflow already optimized for parallel execution")
                return workflow_plan
            
            # Check if we can convert sequential to parallel
            if self._can_parallelize(workflow_plan):
                logger.info("Converting sequential workflow to parallel")
                workflow_plan.metadata['execution_strategy'] = ExecutionStrategy.PARALLEL
                workflow_plan.estimated_time *= 0.7
                workflow_plan.reasoning_chain.append("Optimized for parallel execution")
            
            return workflow_plan
            
        except Exception as e:
            logger.error(f"❌ Enhanced optimization failed: {e}")
            return workflow_plan
    
    def _can_parallelize(self, workflow_plan: WorkflowPlan) -> bool:
        """Check if workflow can be parallelized"""
        # Count independent steps
        independent_count = sum(1 for step in workflow_plan.steps if not step.depends_on)
        return independent_count > 1
    
    def validate_workflow(self, workflow_plan: WorkflowPlan) -> bool:
        """Validate enhanced workflow"""
        try:
            # Basic validation
            if not workflow_plan or not workflow_plan.steps:
                logger.warning("Empty workflow plan")
                return False
            
            # Validate each step
            for step in workflow_plan.steps:
                if not step.step_id or not step.agent_id:
                    logger.warning(f"Invalid step: {step}")
                    return False
            
            # Validate graph structure
            if hasattr(workflow_plan, 'execution_graph') and workflow_plan.execution_graph:
                if not nx.is_directed_acyclic_graph(workflow_plan.execution_graph):
                    logger.warning("Workflow contains cycles")
                    return False
            
            # Validate metadata
            if hasattr(workflow_plan, 'metadata'):
                if 'execution_strategy' not in workflow_plan.metadata:
                    logger.warning("Missing execution strategy metadata")
            
            logger.info("✅ Enhanced workflow validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow validation failed: {e}")
            return False