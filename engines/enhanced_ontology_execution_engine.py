"""
Enhanced Ontology Execution Engine
Handles intelligent data passing between agents and parallel/hybrid execution
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from loguru import logger
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import traceback

from ..core.models import (
    WorkflowPlan, WorkflowStep, ExecutionStrategy,
    WorkflowComplexity, ExecutionResult
)
from ..core.interfaces import ExecutionEngine as IExecutionEngine


@dataclass
class StepExecutionResult:
    """Result of a single step execution"""
    step_id: str
    agent_id: str
    status: str  # 'success', 'error', 'skipped'
    result: Any
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedOntologyExecutionEngine(IExecutionEngine):
    """
    Enhanced execution engine with intelligent data passing and parallel execution
    """
    
    def __init__(self, agent_executor=None):
        self.agent_executor = agent_executor
        self.execution_history: Dict[str, List[StepExecutionResult]] = {}
        self.data_transformers = self._initialize_data_transformers()
        
        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        
        logger.info("🚀 Enhanced Ontology Execution Engine initialized")
    
    def _initialize_data_transformers(self) -> Dict[str, Any]:
        """Initialize data transformation rules between agent types"""
        return {
            # Search -> Analysis: Extract relevant data
            ('search', 'analysis'): self._transform_search_to_analysis,
            # Analysis -> Chart: Format for visualization
            ('analysis', 'chart'): self._transform_analysis_to_chart,
            # Any -> Memo: Summarize for storage
            ('*', 'memo'): self._transform_any_to_memo,
            # Data -> Analysis: Structure data
            ('data', 'analysis'): self._transform_data_to_analysis,
            # Search -> Chart: Extract visualizable data
            ('search', 'chart'): self._transform_search_to_chart,
        }
    
    async def execute_workflow(
        self,
        workflow_plan: WorkflowPlan,
        context: Dict[str, Any]
    ) -> AsyncGenerator[ExecutionResult, None]:
        """
        Execute workflow with intelligent data passing and parallel support
        """
        execution_id = workflow_plan.plan_id
        self.execution_history[execution_id] = []
        
        try:
            # Extract execution metadata
            execution_strategy = workflow_plan.metadata.get(
                'execution_strategy', 
                ExecutionStrategy.SEQUENTIAL
            )
            parallel_groups = workflow_plan.metadata.get('parallel_groups', [])
            
            logger.info(f"🚀 Workflow execution started: {execution_id}")
            logger.info(f"  Execution strategy: {execution_strategy}")
            logger.info(f"  Total steps: {len(workflow_plan.steps)}")
            
            # Yield initial status
            yield ExecutionResult(
                step_id="init",
                agent_id="system",
                status="starting",
                result={
                    "message": "Starting workflow execution",
                    "total_steps": len(workflow_plan.steps),
                    "strategy": execution_strategy.value
                },
                timestamp=datetime.now()
            )
            
            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.SINGLE_AGENT:
                async for result in self._execute_single_agent(workflow_plan, context):
                    yield result
                    
            elif execution_strategy == ExecutionStrategy.PARALLEL:
                async for result in self._execute_parallel(workflow_plan, context, parallel_groups):
                    yield result
                    
            elif execution_strategy == ExecutionStrategy.HYBRID:
                async for result in self._execute_hybrid(workflow_plan, context, parallel_groups):
                    yield result
                    
            else:  # SEQUENTIAL or AUTO
                async for result in self._execute_sequential(workflow_plan, context):
                    yield result
            
            # Yield completion status
            yield ExecutionResult(
                step_id="complete",
                agent_id="system",
                status="completed",
                result={
                    "message": "Workflow execution completed",
                    "total_results": len(self.execution_history[execution_id])
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            logger.error(traceback.format_exc())
            
            yield ExecutionResult(
                step_id="error",
                agent_id="system",
                status="error",
                result={"error": str(e)},
                error=str(e),
                timestamp=datetime.now()
            )
    
    async def _execute_single_agent(
        self,
        workflow_plan: WorkflowPlan,
        context: Dict[str, Any]
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Execute single agent workflow"""
        if not workflow_plan.steps:
            return
        
        step = workflow_plan.steps[0]
        async for result in self._execute_step(step, context, {}):
            yield result
    
    async def _execute_sequential(
        self,
        workflow_plan: WorkflowPlan,
        context: Dict[str, Any]
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Execute workflow sequentially with data passing"""
        step_results = {}
        
        for step in workflow_plan.steps:
            # Prepare input data from previous steps
            input_data = self._prepare_step_input(step, step_results)
            
            # Execute step
            async for result in self._execute_step(step, context, input_data):
                yield result
                
                # Store result for data passing
                if result.status == "completed":
                    step_results[step.step_id] = result
    
    async def _execute_parallel(
        self,
        workflow_plan: WorkflowPlan,
        context: Dict[str, Any],
        parallel_groups: List[List[str]]
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Execute workflow with parallel groups"""
        step_results = {}
        steps_by_id = {step.step_id: step for step in workflow_plan.steps}
        
        # If no parallel groups defined, create them based on dependencies
        if not parallel_groups:
            parallel_groups = self._create_parallel_groups(workflow_plan)
        
        # Execute each parallel group
        for group_idx, group in enumerate(parallel_groups):
            logger.info(f"🔄 Parallel group {group_idx + 1} execution started: {group}")
            
            # Collect tasks for parallel execution
            tasks = []
            for task_id in group:
                # Map task_id to step
                step = None
                if task_id.startswith('step_'):
                    # Direct step ID
                    step = steps_by_id.get(task_id)
                else:
                    # Subtask ID - find corresponding step
                    for s in workflow_plan.steps:
                        if s.metadata.get('subtask_id') == task_id:
                            step = s
                            break
                
                if step:
                    input_data = self._prepare_step_input(step, step_results)
                    tasks.append(self._execute_step_async(step, context, input_data))
            
            # Execute tasks in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for idx, result_list in enumerate(results):
                    if isinstance(result_list, Exception):
                        logger.error(f"Parallel execution error: {result_list}")
                        continue
                    
                    # Yield all results from the step
                    for result in result_list:
                        yield result
                        
                        # Store successful results
                        if result.status == "completed":
                            step_results[result.step_id] = result
    
    async def _execute_hybrid(
        self,
        workflow_plan: WorkflowPlan,
        context: Dict[str, Any],
        parallel_groups: List[List[str]]
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Execute hybrid workflow (mix of sequential and parallel)"""
        # Hybrid is essentially parallel with dependency ordering
        async for result in self._execute_parallel(workflow_plan, context, parallel_groups):
            yield result
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Execute a single workflow step"""
        start_time = datetime.now()
        
        try:
            logger.info(f"📍 Step execution: {step.step_id} - Agent: {step.agent_id}")
            
            # Yield starting status
            yield ExecutionResult(
                step_id=step.step_id,
                agent_id=step.agent_id,
                status="running",
                result={"message": f"Running {step.agent_id}..."},
                timestamp=start_time
            )
            
            # Execute agent
            if self.agent_executor:
                # Prepare agent input
                agent_input = self._prepare_agent_input(step, context, input_data)
                
                # Execute agent
                result = await self.agent_executor.execute(
                    agent_id=step.agent_id,
                    input_data=agent_input,
                    context=context
                )
                
                # Process result
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Create execution result
                step_result = StepExecutionResult(
                    step_id=step.step_id,
                    agent_id=step.agent_id,
                    status="success",
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        'execution_time': execution_time,
                        'input_data': input_data
                    }
                )
                
                self.execution_history[context.get('execution_id', 'default')].append(step_result)
                
                # Yield completed status
                yield ExecutionResult(
                    step_id=step.step_id,
                    agent_id=step.agent_id,
                    status="completed",
                    result=result,
                    timestamp=end_time,
                    metadata={
                        'execution_time': execution_time,
                        'step_purpose': step.purpose
                    }
                )
            else:
                # Mock execution for testing
                await asyncio.sleep(1)
                
                mock_result = {
                    "agent": step.agent_id,
                    "purpose": step.purpose,
                    "mock_data": f"Mock result for {step.agent_id}",
                    "input_received": input_data
                }
                
                yield ExecutionResult(
                    step_id=step.step_id,
                    agent_id=step.agent_id,
                    status="completed",
                    result=mock_result,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Step execution error {step.step_id}: {str(e)}")
            
            yield ExecutionResult(
                step_id=step.step_id,
                agent_id=step.agent_id,
                status="error",
                result={"error": str(e)},
                error=str(e),
                timestamp=datetime.now()
            )
    
    async def _execute_step_async(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> List[ExecutionResult]:
        """Execute step and return all results as a list (for parallel execution)"""
        results = []
        async for result in self._execute_step(step, context, input_data):
            results.append(result)
        return results
    
    def _prepare_step_input(
        self,
        step: WorkflowStep,
        previous_results: Dict[str, StepExecutionResult]
    ) -> Dict[str, Any]:
        """Prepare input data for a step based on dependencies"""
        input_data = {}
        
        # Collect data from dependencies
        for dep_id in step.depends_on:
            if dep_id in previous_results:
                dep_result = previous_results[dep_id]
                
                # Apply data transformation
                transformed_data = self._transform_data(
                    source_agent=dep_result.agent_id,
                    target_agent=step.agent_id,
                    source_data=dep_result.result
                )
                
                input_data[dep_id] = transformed_data
        
        # Add step metadata
        if step.metadata:
            input_data['step_metadata'] = step.metadata
        
        return input_data
    
    def _prepare_agent_input(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare final input for agent execution"""
        agent_input = {
            'query': context.get('query', ''),
            'purpose': step.purpose,
            'concepts': step.concepts,
            'previous_results': input_data
        }
        
        # Add specific requirements from step metadata
        if step.metadata:
            if 'input_required' in step.metadata:
                agent_input['requirements'] = step.metadata['input_required']
            if 'output_expected' in step.metadata:
                agent_input['expected_output'] = step.metadata['output_expected']
        
        return agent_input
    
    def _transform_data(
        self,
        source_agent: str,
        target_agent: str,
        source_data: Any
    ) -> Any:
        """Transform data between agents based on their types"""
        # Extract agent types
        source_type = self._get_agent_type(source_agent)
        target_type = self._get_agent_type(target_agent)
        
        # Find transformer
        transformer_key = (source_type, target_type)
        transformer = self.data_transformers.get(transformer_key)
        
        # Try wildcard transformer
        if not transformer:
            transformer = self.data_transformers.get(('*', target_type))
        
        # Apply transformation
        if transformer:
            try:
                return transformer(source_data)
            except Exception as e:
                logger.error(f"Data transformation error: {e}")
                return source_data
        
        # Default: pass through
        return source_data
    
    def _get_agent_type(self, agent_id: str) -> str:
        """Extract agent type from agent ID"""
        agent_id_lower = agent_id.lower()
        
        if 'search' in agent_id_lower or 'internet' in agent_id_lower:
            return 'search'
        elif 'analysis' in agent_id_lower or 'analyze' in agent_id_lower:
            return 'analysis'
        elif 'chart' in agent_id_lower or 'visual' in agent_id_lower:
            return 'chart'
        elif 'memo' in agent_id_lower:
            return 'memo'
        elif 'data' in agent_id_lower:
            return 'data'
        else:
            return 'general'
    
    def _create_parallel_groups(self, workflow_plan: WorkflowPlan) -> List[List[str]]:
        """Create parallel groups from workflow steps"""
        # Build dependency graph
        dep_graph = nx.DiGraph()
        for step in workflow_plan.steps:
            dep_graph.add_node(step.step_id)
            for dep in step.depends_on:
                dep_graph.add_edge(dep, step.step_id)
        
        # Topological generations give us parallel groups
        try:
            generations = list(nx.topological_generations(dep_graph))
            return generations
        except nx.NetworkXError:
            # Not a DAG, fall back to sequential
            return [[step.step_id] for step in workflow_plan.steps]
    
    # Data transformation methods
    def _transform_search_to_analysis(self, data: Any) -> Dict[str, Any]:
        """Transform search results for analysis"""
        if isinstance(data, dict):
            return {
                'data_points': data.get('results', []),
                'source': 'search',
                'metadata': data.get('metadata', {}),
                'summary': data.get('summary', '')
            }
        elif isinstance(data, list):
            return {
                'data_points': data,
                'source': 'search'
            }
        else:
            return {'raw_data': data}
    
    def _transform_analysis_to_chart(self, data: Any) -> Dict[str, Any]:
        """Transform analysis results for charting"""
        if isinstance(data, dict):
            return {
                'chart_data': data.get('analyzed_data', data),
                'chart_type': data.get('suggested_chart_type', 'bar'),
                'labels': data.get('labels', []),
                'values': data.get('values', [])
            }
        else:
            return {'chart_data': data}
    
    def _transform_any_to_memo(self, data: Any) -> Dict[str, Any]:
        """Transform any data for memo storage"""
        if isinstance(data, dict):
            return {
                'title': data.get('title', 'Workflow Result'),
                'content': json.dumps(data, ensure_ascii=False, indent=2),
                'timestamp': datetime.now().isoformat(),
                'type': 'workflow_result'
            }
        else:
            return {
                'title': 'Workflow Result',
                'content': str(data),
                'timestamp': datetime.now().isoformat()
            }
    
    def _transform_data_to_analysis(self, data: Any) -> Dict[str, Any]:
        """Transform raw data for analysis"""
        return {
            'dataset': data,
            'analysis_required': True,
            'preprocessing': 'auto'
        }
    
    def _transform_search_to_chart(self, data: Any) -> Dict[str, Any]:
        """Transform search results directly for charting"""
        if isinstance(data, dict) and 'results' in data:
            # Extract numeric data for visualization
            numeric_data = []
            labels = []
            
            for item in data.get('results', []):
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (int, float)):
                            labels.append(key)
                            numeric_data.append(value)
            
            return {
                'chart_type': 'bar',
                'labels': labels,
                'values': numeric_data,
                'title': 'Search Results Visualization'
            }
        
        return {'raw_data': data}
    
    async def validate_execution(self, workflow_plan: WorkflowPlan, context: Dict[str, Any]) -> bool:
        """Validate execution readiness"""
        try:
            # Check workflow plan
            if not workflow_plan or not workflow_plan.steps:
                logger.error("Workflow plan is empty")
                return False

            # Check context
            if not context:
                logger.error("Execution context is missing")
                return False

            # Check agent executor
            if not self.agent_executor:
                logger.warning("Agent executor not configured (mock execution mode)")

            # Validate execution strategy
            if hasattr(workflow_plan, 'metadata'):
                strategy = workflow_plan.metadata.get('execution_strategy')
                if strategy and strategy not in ExecutionStrategy:
                    logger.error(f"Invalid execution strategy: {strategy}")
                    return False

            logger.info("✅ Execution validation passed")
            return True

        except Exception as e:
            logger.error(f"Execution validation failed: {e}")
            return False
    
    def get_execution_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get execution metrics"""
        if execution_id not in self.execution_history:
            return {}
        
        history = self.execution_history[execution_id]
        
        # Calculate metrics
        total_steps = len(history)
        successful_steps = sum(1 for r in history if r.status == 'success')
        failed_steps = sum(1 for r in history if r.status == 'error')
        
        # Calculate execution time
        if history:
            start_time = min(r.start_time for r in history if r.start_time)
            end_time = max(r.end_time for r in history if r.end_time)
            total_time = (end_time - start_time).total_seconds() if start_time and end_time else 0
        else:
            total_time = 0
        
        return {
            'execution_id': execution_id,
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'total_execution_time': total_time,
            'average_step_time': total_time / total_steps if total_steps > 0 else 0
        }