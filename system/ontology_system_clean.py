"""
🧠 Clean Ontology System
Simplified Ontology System

Clean integrated system utilizing split modules
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, ExecutionContext, AgentExecutionResult, 
    WorkflowPlan, get_system_metrics
)
from ..core.interfaces import ProgressCallback
from ..engines.execution_engine import AdvancedExecutionEngine
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine

# Import new split modules
from .query_processing import QueryProcessor
from .result_integration import ResultIntegrator
from .knowledge_management import KnowledgeGraphManager
from .metrics_manager import MetricsManager


class SimpleProgressCallback(ProgressCallback):
    """Simple progress callback"""
    
    def __init__(self):
        self.messages = []
        self.current_progress = 0.0
        self.completed_steps = []
        self.errors = []
    
    async def on_progress(self, message: str, progress: float, metadata: Dict[str, Any] = None):
        """Update progress"""
        self.messages.append({
            "message": message,
            "progress": progress,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        self.current_progress = progress
        logger.info(f"Progress ({progress:.1%}): {message}")
    
    async def on_step_complete(self, step_id: str, result: AgentExecutionResult):
        """Step completion notification"""
        self.completed_steps.append({
            "step_id": step_id,
            "agent_id": result.agent_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })
        logger.info(f"Step completed: {step_id} ({result.agent_id}) - {'✅' if result.success else '❌'}")
    
    async def on_error(self, error_message: str, error_details: Dict[str, Any] = None):
        """Error notification"""
        self.errors.append({
            "message": error_message,
            "details": error_details or {},
            "timestamp": time.time()
        })
        logger.error(f"Error: {error_message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Progress summary"""
        return {
            "current_progress": self.current_progress,
            "total_messages": len(self.messages),
            "completed_steps": len(self.completed_steps),
            "errors": len(self.errors),
            "success_rate": (
                sum(1 for step in self.completed_steps if step["success"]) / 
                len(self.completed_steps) if self.completed_steps else 0
            )
        }


class CleanOntologySystem:
    """🧠 Clean Ontology System - utilizing split modules"""
    
    def __init__(self, 
                 email: str = "system@ontology.ai",
                 session_id: str = None,
                 project_id: str = None):
        
        # Session information
        self.email = email
        self.session_id = session_id or f"session_{int(time.time())}"
        self.project_id = project_id or "default_project"
        
        # Initialize core engines
        self.execution_engine = AdvancedExecutionEngine()
        self.knowledge_graph = KnowledgeGraphEngine()
        
        # Initialize split modules
        self.query_processor = QueryProcessor()
        self.result_integrator = ResultIntegrator()
        self.knowledge_manager = KnowledgeGraphManager(self.knowledge_graph)
        self.metrics_manager = MetricsManager()
        
        # System state
        self.is_initialized = False
        self.execution_history = []
        
        logger.info(f"🧠 Clean ontology system initialized: {self.session_id}")
    
    async def initialize(self):
        """Initialize the system"""
        try:
            logger.info("🚀 Starting clean ontology system initialization")
            
            # Initialize each module (if needed)
            await self.query_processor.initialize()
            
            self.is_initialized = True
            logger.info("✅ Clean ontology system initialization complete")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def process_query(self, query_text: str, execution_context: ExecutionContext) -> Dict[str, Any]:
        """Query processing - main entry point"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting query processing: '{query_text[:50]}...'")
            
            # 1. Query processing and workflow creation (delegated to QueryProcessor)
            semantic_query, workflow_plan = await self.query_processor.process_query(
                query_text, execution_context
            )
            
            # 2. Execute workflow
            execution_results = await self._execute_workflow(workflow_plan, execution_context)
            
            # 3. Result integration (delegated to ResultIntegrator)
            integrated_result = await self.result_integrator.integrate_results(
                execution_results, workflow_plan, semantic_query
            )
            
            # 4. Knowledge graph update (delegated to KnowledgeGraphManager)
            await self.knowledge_manager.update_knowledge_graph(
                semantic_query, workflow_plan, execution_results, integrated_result
            )
            
            # 5. Metrics recording (delegated to MetricsManager)
            total_execution_time = time.time() - start_time
            self.metrics_manager.record_workflow_execution(
                semantic_query, workflow_plan, execution_results, integrated_result, total_execution_time
            )
            
            # 6. Build final result
            final_result = {
                **integrated_result,
                'semantic_query': semantic_query.to_dict(),
                'workflow_plan': self._workflow_plan_to_dict(workflow_plan),
                'execution_results': [r.to_dict() for r in execution_results],
                'total_execution_time': total_execution_time,
                'system_metrics': self.metrics_manager.get_system_status()
            }
            
            # Add to execution history
            self.execution_history.append({
                'query_id': semantic_query.query_id,
                'query_text': query_text,
                'success': integrated_result.get('success', False),
                'execution_time': total_execution_time,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"✅ Query processing complete: {total_execution_time:.2f}s")
            return final_result
            
        except Exception as e:
            total_execution_time = time.time() - start_time
            logger.error(f"❌ Query processing failed: {e}")
            
            # Record error metrics
            get_system_metrics().failed_executions += 1
            
            return {
                'success': False,
                'error': str(e),
                'query_text': query_text,
                'execution_time': total_execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_workflow(self, workflow_plan: WorkflowPlan, execution_context: ExecutionContext) -> List[AgentExecutionResult]:
        """Execute workflow"""
        try:
            logger.info(f"⚡ Starting workflow execution: {len(workflow_plan.steps)} steps")
            
            # Create progress callback
            progress_callback = SimpleProgressCallback()
            
            # Delegate workflow execution to execution engine
            execution_results = await self.execution_engine.execute_workflow(
                workflow_plan=workflow_plan,
                execution_context=execution_context,
                progress_callback=progress_callback
            )
            
            logger.info(f"⚡ Workflow execution complete: {len(execution_results)} results")
            return execution_results
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Return empty result list (for error handling)
            return []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Retrieve system metrics"""
        return self.metrics_manager.get_system_status()
    
    def get_system_performance_report(self) -> str:
        """Retrieve system performance report"""
        return self.metrics_manager.generate_performance_report()
    
    def get_knowledge_graph_visualization(self, max_nodes: int = 50) -> Dict[str, Any]:
        """Retrieve knowledge graph visualization data"""
        try:
            logger.info(f"🎨 Generating knowledge graph visualization data: max {max_nodes} nodes")
            
            visualization_data = self.knowledge_graph.generate_visualization(max_nodes)
            
            # Additional metadata
            visualization_data['system_info'] = {
                'session_id': self.session_id,
                'total_queries': len(self.execution_history),
                'generated_at': datetime.now().isoformat()
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Knowledge graph visualization failed: {e}")
            return {
                'nodes': [],
                'edges': [],
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve execution history"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    async def close(self):
        """System shutdown and cleanup"""
        try:
            logger.info("🔄 Starting ontology system shutdown")
            
            # Clean up each module (if needed)
            if hasattr(self.execution_engine, 'close'):
                await self.execution_engine.close()
            
            if hasattr(self.knowledge_graph, 'close'):
                await self.knowledge_graph.close()
            
            # Print final metrics
            final_report = self.metrics_manager.generate_performance_report()
            logger.info(f"📊 Final performance report:\n{final_report}")
            
            logger.info("✅ Ontology system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
    
    def _workflow_plan_to_dict(self, workflow_plan: WorkflowPlan) -> Dict[str, Any]:
        """Convert WorkflowPlan to dictionary"""
        try:
            return {
                'plan_id': workflow_plan.plan_id,
                'estimated_time': workflow_plan.estimated_time,
                'estimated_quality': workflow_plan.estimated_quality,
                'optimization_strategy': getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy)),
                'steps_count': len(workflow_plan.steps),
                'steps': [
                    {
                        'step_id': step.step_id,
                        'agent_id': step.agent_id,
                        'semantic_purpose': step.semantic_purpose,
                        'estimated_complexity': getattr(step.estimated_complexity, 'value', str(step.estimated_complexity)),
                        'estimated_time': step.estimated_time,
                        'depends_on': step.depends_on
                    } for step in workflow_plan.steps
                ],
                'created_at': workflow_plan.created_at.isoformat() if hasattr(workflow_plan.created_at, 'isoformat') else str(workflow_plan.created_at)
            }
        except Exception as e:
            logger.error(f"Workflow plan conversion failed: {e}")
            return {'error': str(e)}


# Alias for base system class (backward compatibility)
OntologySystem = CleanOntologySystem 