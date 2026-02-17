"""
🔍 Query Processing Module
Query Processing Module

Responsible for query analysis, complexity calculation, and workflow creation
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, ExecutionContext, WorkflowPlan, 
    ExecutionStrategy, ComplexityAnalysis, AgentExecutionResult
)
from ..engines.semantic_query_manager import SemanticQueryManager
from ..engines.workflow_designer import SmartWorkflowDesigner
from ..engines.execution_engine import QueryComplexityAnalyzer


class QueryProcessor:
    """🔍 Query Processor - dedicated to query analysis and workflow creation"""
    
    def __init__(self):
        self.semantic_query_manager = SemanticQueryManager()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.workflow_designer = None
        self.installed_agents_info = []
        
        logger.info("🔍 Query processor initialized")
    
    def initialize_workflow_designer(self, installed_agents_info: List[Dict[str, Any]] = None):
        """Initialize workflow designer with installed agent information"""
        try:
            if installed_agents_info:
                self.installed_agents_info = installed_agents_info
                logger.info(f"🎯 Updated installed agent information: {len(installed_agents_info)} agents")
            
            # Initialize SmartWorkflowDesigner with installed agent information
            self.workflow_designer = SmartWorkflowDesigner(self.installed_agents_info)
            logger.info(f"🎯 Workflow designer initialized - agent count: {len(self.installed_agents_info)}")
            
        except Exception as e:
            logger.error(f"Workflow designer initialization failed: {e}")
            # Fallback: use default workflow designer
            self.workflow_designer = SmartWorkflowDesigner()
    
    async def process_query_to_workflow(self, 
                                      query_text: str, 
                                      execution_context: ExecutionContext,
                                      available_agents: List[str]) -> tuple[SemanticQuery, Dict[str, Any], WorkflowPlan]:
        """Analyze query and create workflow"""
        try:
            logger.info(f"🔍 Starting query analysis: '{query_text[:50]}...'")
            
            # 1. Create semantic query
            semantic_query = await self.semantic_query_manager.create_semantic_query(
                query_text, execution_context
            )
            logger.info(f"📝 Semantic query created: ID={semantic_query.query_id}")
            
            # 2. Complexity analysis
            complexity_analysis = self.analyze_complexity(semantic_query)
            logger.info(f"🔍 Complexity analysis complete: {complexity_analysis.get('recommended_strategy', 'AUTO')} (score: {complexity_analysis.get('complexity_score', 0):.2f})")
            
            # 3. Design workflow
            logger.info(f"🤖 Available agents: {len(available_agents)} - {available_agents}")
            
            workflow_plan = await self.workflow_designer.design_workflow(
                semantic_query, available_agents
            )
            
            # Optimize workflow (including duplicate removal)
            workflow_plan = self.workflow_designer.optimize_workflow(workflow_plan)
            
            # Log workflow design details
            self._log_workflow_details(workflow_plan)
            
            return semantic_query, complexity_analysis, workflow_plan
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    def analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Safe and simple complexity analysis"""
        try:
            # Safely check if SemanticQuery is a dict or object
            if isinstance(semantic_query, dict):
                query_text = semantic_query.get('natural_language', '')
                required_agents = semantic_query.get('required_agents', [])
            else:
                query_text = getattr(semantic_query, 'natural_language', '')
                required_agents = getattr(semantic_query, 'required_agents', [])
                if hasattr(semantic_query, 'structured_query'):
                    structured_query = getattr(semantic_query, 'structured_query', {})
                    if isinstance(structured_query, dict) and 'required_agents' in structured_query:
                        required_agents = structured_query['required_agents']
            
            # Basic complexity analysis - simplified
            analysis = {
                'complexity_score': 0.5,  # default value
                'query_type': 'GENERAL',
                'required_agents_count': len(required_agents) if required_agents else 0,
                'estimated_processing_time': 30.0,  # default 30 seconds
                'recommended_strategy': ExecutionStrategy.AUTO
            }
            
            # Estimate complexity based on text length
            if query_text:
                text_length = len(query_text)
                if text_length > 100:
                    analysis['complexity_score'] = 0.7
                elif text_length > 50:
                    analysis['complexity_score'] = 0.6
                else:
                    analysis['complexity_score'] = 0.4
                
                # Adjust complexity based on key keywords
                complex_keywords = ['분석', '비교', '차트', '그래프', '계산']  # analysis, comparison, chart, graph, calculation
                keyword_count = sum(1 for keyword in complex_keywords if keyword in query_text)
                if keyword_count > 0:
                    analysis['complexity_score'] = min(analysis['complexity_score'] + 0.1 * keyword_count, 1.0)
            
            # Determine strategy based on agent count - simplified
            agents_count = analysis['required_agents_count']
            if agents_count <= 1:
                analysis['recommended_strategy'] = ExecutionStrategy.SINGLE_AGENT
                analysis['estimated_processing_time'] = 15.0
            elif agents_count == 2:
                analysis['recommended_strategy'] = ExecutionStrategy.PARALLEL
                analysis['estimated_processing_time'] = 20.0
            else:
                analysis['recommended_strategy'] = ExecutionStrategy.HYBRID
                analysis['estimated_processing_time'] = agents_count * 15.0
            
            # Additional adjustment based on complexity score
            if analysis['complexity_score'] >= 0.8:
                analysis['recommended_strategy'] = ExecutionStrategy.HYBRID
            elif analysis['complexity_score'] >= 0.6 and agents_count > 1:
                analysis['recommended_strategy'] = ExecutionStrategy.PARALLEL
            
            logger.info(f"🔍 Complexity analysis complete: score={analysis['complexity_score']:.2f}, strategy={analysis['recommended_strategy']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            # Safe fallback analysis
            return {
                'complexity_score': 0.5,
                'query_type': 'GENERAL',
                'required_agents_count': 1,
                'estimated_processing_time': 30.0,
                'recommended_strategy': ExecutionStrategy.SINGLE_AGENT,
                'error': str(e),
                'fallback_mode': True
            }
    
    def _log_workflow_details(self, workflow_plan: WorkflowPlan):
        """Log workflow details"""
        logger.info(f"🔧 Workflow design complete:")
        logger.info(f"  - Plan ID: {workflow_plan.plan_id}")
        logger.info(f"  - Step count: {len(workflow_plan.steps)}")
        logger.info(f"  - Optimization strategy: {getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy))}")
        logger.info(f"  - Estimated quality: {workflow_plan.estimated_quality:.2f}")
        logger.info(f"  - Estimated time: {workflow_plan.estimated_time:.1f}s")
        
        # Log each step in detail
        for i, step in enumerate(workflow_plan.steps):
            logger.info(f"    Step {i+1}: {step.step_id}")
            logger.info(f"      - Agent: {step.agent_id}")
            logger.info(f"      - Purpose: {step.semantic_purpose}")
            logger.info(f"      - Complexity: {getattr(step.estimated_complexity, 'value', str(step.estimated_complexity))}")
            logger.info(f"      - Estimated time: {step.estimated_time:.1f}s")
            if step.depends_on:
                logger.info(f"      - Dependencies: {step.depends_on}")
    
    def generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
        """Generate Mermaid diagram for workflow"""
        try:
            if not workflow_plan or not workflow_plan.steps:
                return 'graph TD\n    A["Empty workflow"] --> B["No steps"]'
            
            mermaid_lines = ["graph TD"]
            
            # Start node
            mermaid_lines.append(f'    START["{workflow_plan.query.query_text[:30]}..."]')
            
            # Define steps
            for i, step in enumerate(workflow_plan.steps):
                step_label = f"{step.agent_id}\\n{step.semantic_purpose[:20]}..."
                mermaid_lines.append(f'    {step.step_id}["{step_label}"]')
                
                # First steps with no dependencies are connected to START
                if not step.depends_on:
                    mermaid_lines.append(f'    START --> {step.step_id}')
            
            # Add dependency relationships
            for step in workflow_plan.steps:
                for dep in step.depends_on:
                    mermaid_lines.append(f'    {dep} --> {step.step_id}')
            
            # Connect last steps to END
            last_steps = []
            for step in workflow_plan.steps:
                # Steps not in any other step's dependencies are the last steps
                is_last = True
                for other_step in workflow_plan.steps:
                    if step.step_id in other_step.depends_on:
                        is_last = False
                        break
                if is_last:
                    last_steps.append(step.step_id)
            
            # Add END node
            mermaid_lines.append('    END["Complete"]')
            for last_step in last_steps:
                mermaid_lines.append(f'    {last_step} --> END')
            
            return "\n".join(mermaid_lines)
            
        except Exception as e:
            logger.error(f"Mermaid diagram generation failed: {e}")
            return f'graph TD\n    A["Error"] --> B["{str(e)[:50]}..."]' 