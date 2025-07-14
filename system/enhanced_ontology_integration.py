"""
Enhanced Ontology System Integration
Integrates all enhanced components for the ontology system
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import json
from loguru import logger

from ..processors.enhanced_ontology_query_processor import EnhancedOntologyQueryProcessor
from ..engines.enhanced_ontology_workflow_designer import EnhancedOntologyWorkflowDesigner
from ..engines.enhanced_ontology_execution_engine import EnhancedOntologyExecutionEngine
from ..core.models import SemanticQuery, WorkflowPlan, ExecutionResult, ExecutionContext
from ..core.interfaces import OntologySystem as IOntologySystem

# Import existing components
from .semantic_query_manager import SemanticQueryManager
from .knowledge_graph_engine import KnowledgeGraphEngine
from .result_processor import OntologyResultProcessor


class EnhancedOntologySystem(IOntologySystem):
    """
    Enhanced Ontology System with improved query processing, workflow design, and execution
    """
    
    def __init__(self):
        # Initialize enhanced components
        self.query_processor = EnhancedOntologyQueryProcessor()
        self.workflow_designer = None  # Will be initialized with agents
        self.execution_engine = None  # Will be initialized with executor
        
        # Initialize existing components
        self.semantic_manager = SemanticQueryManager()
        self.knowledge_graph = KnowledgeGraphEngine()
        self.result_processor = OntologyResultProcessor()
        
        # State management
        self.installed_agents: List[Dict[str, Any]] = []
        self.agent_executor = None
        
        logger.info("🚀 Enhanced Ontology System 초기화 완료")
    
    def initialize(self, installed_agents: List[Dict[str, Any]], agent_executor: Any = None):
        """
        Initialize the system with installed agents and executor
        """
        self.installed_agents = installed_agents
        self.agent_executor = agent_executor
        
        # Initialize workflow designer with agents
        self.workflow_designer = EnhancedOntologyWorkflowDesigner(installed_agents)
        
        # Initialize execution engine with executor
        self.execution_engine = EnhancedOntologyExecutionEngine(agent_executor)
        
        logger.info(f"✅ Ontology System 초기화: {len(installed_agents)}개 에이전트")
    
    async def process_query(
        self,
        query: str,
        context: ExecutionContext,
        streaming: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query through the enhanced ontology system
        """
        try:
            logger.info(f"🎯 쿼리 처리 시작: {query}")
            
            # 1. Create semantic query
            semantic_query = await self.semantic_manager.analyze_query(query)
            logger.info(f"의미 분석 완료: 의도={semantic_query.intent}, 개념={len(semantic_query.concepts)}개")
            
            # 2. Get available agents for user
            available_agents = self._get_user_agents(context)
            if not available_agents:
                yield {
                    "type": "error",
                    "message": "사용 가능한 에이전트가 없습니다"
                }
                return
            
            # 3. Design workflow with enhanced designer
            workflow_plan = await self.workflow_designer.design_workflow(
                semantic_query, available_agents
            )
            
            # Log workflow details
            logger.info(f"워크플로우 설계 완료:")
            logger.info(f"  - 단계: {len(workflow_plan.steps)}개")
            logger.info(f"  - 전략: {workflow_plan.optimization_strategy}")
            if hasattr(workflow_plan, 'metadata'):
                logger.info(f"  - 실행 전략: {workflow_plan.metadata.get('execution_strategy')}")
                logger.info(f"  - 병렬 그룹: {len(workflow_plan.metadata.get('parallel_groups', []))}개")
            
            # 4. Yield workflow plan
            if streaming:
                yield {
                    "type": "workflow_plan",
                    "plan": {
                        "steps": len(workflow_plan.steps),
                        "agents": [step.agent_id for step in workflow_plan.steps],
                        "strategy": str(workflow_plan.optimization_strategy),
                        "estimated_time": workflow_plan.estimated_time,
                        "reasoning": workflow_plan.reasoning_chain
                    }
                }
            
            # 5. Execute workflow with enhanced engine
            execution_context = {
                'query': query,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'execution_id': workflow_plan.plan_id
            }
            
            results = []
            async for execution_result in self.execution_engine.execute_workflow(
                workflow_plan, execution_context
            ):
                # Process and yield results
                if streaming:
                    yield self._format_execution_result(execution_result)
                
                # Collect results
                if execution_result.status == "completed":
                    results.append(execution_result)
            
            # 6. Process final results
            if results:
                final_result = await self.result_processor.process_results(
                    results, semantic_query
                )
                
                # Update knowledge graph
                await self.knowledge_graph.update_from_execution(
                    semantic_query, workflow_plan, results
                )
                
                if streaming:
                    yield {
                        "type": "final_result",
                        "result": final_result
                    }
            
            # 7. Get execution metrics
            metrics = self.execution_engine.get_execution_metrics(workflow_plan.plan_id)
            if streaming:
                yield {
                    "type": "metrics",
                    "metrics": metrics
                }
            
        except Exception as e:
            logger.error(f"쿼리 처리 오류: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "message": "쿼리 처리 중 오류가 발생했습니다"
            }
    
    def _get_user_agents(self, context: ExecutionContext) -> List[str]:
        """Get available agents for the user"""
        # In a real implementation, this would check user permissions
        # For now, return all agent IDs
        return [agent['agent_id'] for agent in self.installed_agents]
    
    def _format_execution_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """Format execution result for streaming"""
        formatted = {
            "type": "execution_update",
            "step_id": result.step_id,
            "agent_id": result.agent_id,
            "status": result.status,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
        
        # Add appropriate content based on status
        if result.status == "running":
            formatted["message"] = result.result.get("message", "실행 중...")
        elif result.status == "completed":
            formatted["result"] = result.result
            if result.metadata:
                formatted["execution_time"] = result.metadata.get("execution_time")
        elif result.status == "error":
            formatted["error"] = result.error
        
        return formatted
    
    async def analyze_query(self, query: str) -> SemanticQuery:
        """Analyze query to extract semantic information"""
        return await self.semantic_manager.analyze_query(query)
    
    async def select_agents(self, semantic_query: SemanticQuery, available_agents: List[str]) -> List[str]:
        """Select optimal agents for the query"""
        # Convert to agent info format
        agent_info_list = self.workflow_designer._convert_to_agent_info(available_agents)
        
        # Use enhanced query processor
        analysis = await self.query_processor.analyze_query(
            semantic_query.natural_language,
            agent_info_list
        )
        
        return analysis.suggested_agents
    
    async def design_workflow(self, semantic_query: SemanticQuery, selected_agents: List[str]) -> WorkflowPlan:
        """Design workflow for selected agents"""
        return await self.workflow_designer.design_workflow(semantic_query, selected_agents)
    
    async def execute_workflow(
        self,
        workflow_plan: WorkflowPlan,
        context: ExecutionContext
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Execute workflow plan"""
        execution_context = {
            'query': workflow_plan.semantic_query.natural_language,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'execution_id': workflow_plan.plan_id
        }
        
        async for result in self.execution_engine.execute_workflow(
            workflow_plan, execution_context
        ):
            yield result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "status": "active",
            "components": {
                "query_processor": "enhanced",
                "workflow_designer": "enhanced",
                "execution_engine": "enhanced"
            },
            "installed_agents": len(self.installed_agents),
            "capabilities": {
                "parallel_execution": True,
                "intelligent_data_passing": True,
                "query_decomposition": True,
                "dependency_analysis": True
            }
        }


# Factory function
def create_enhanced_ontology_system(
    installed_agents: List[Dict[str, Any]],
    agent_executor: Any = None
) -> EnhancedOntologySystem:
    """
    Create and initialize an enhanced ontology system
    """
    system = EnhancedOntologySystem()
    system.initialize(installed_agents, agent_executor)
    return system