"""
🧠 Ontology System
Integrated Ontology System

Main system that integrates all components
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any
import asyncio
from loguru import logger

from ..core.models import SemanticQuery, ExecutionContext, AgentExecutionResult, WorkflowPlan
from ..engines.semantic_query_manager import SemanticQueryManager
from ..engines.execution_engine import AdvancedExecutionEngine, QueryComplexityAnalyzer
from ..engines.workflow_designer import SmartWorkflowDesigner
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine
from ..core.unified_query_processor import get_unified_query_processor
from .result_integration import ResultIntegrator
from .strategy_manager import StrategyManager
from .progress_callback import SimpleProgressCallback
from .knowledge_graph_manager import KnowledgeGraphManager
from .visualization_manager import VisualizationManager
from .result_processor import ResultProcessor
from .reasoning_generator import ReasoningGenerator
from .workflow_utils import WorkflowUtils
from .complexity_analyzer import ComplexityAnalyzer


class OntologySystem:
    """🧠 Integrated Ontology System"""
    
    def __init__(self, 
                 email: str = "system@ontology.ai",
                 session_id: str = None,
                 project_id: str = None):
        
        # Session information
        self.email = email
        self.session_id = session_id or f"session_{int(time.time())}"
        self.project_id = project_id or "default_project"
        
        # Initialize core engines
        self.semantic_query_manager = SemanticQueryManager()
        self.execution_engine = AdvancedExecutionEngine()
        self.knowledge_graph = KnowledgeGraphEngine()
        self.result_integrator = ResultIntegrator()
        
        # Workflow designer is initialized later with installed agent information
        self.workflow_designer = None
        self.installed_agents_info = []  # Store installed agent information
        
        # Complexity analyzer
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Strategy manager
        self.strategy_manager = StrategyManager()
        
        # Initialize split managers
        self.knowledge_graph_manager = KnowledgeGraphManager(self.knowledge_graph, self.session_id)
        self.visualization_manager = VisualizationManager(self.knowledge_graph)
        self.result_processor = ResultProcessor()
        self.reasoning_generator = ReasoningGenerator()
        self.workflow_utils = WorkflowUtils()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Default agent list (for fallback)
        self.available_agents = [
            "internet_agent", "finance_agent", "weather_agent",
            "calculate_agent", "chart_agent", "memo_agent", "analysis_agent"
        ]
        
        # System state
        self.is_initialized = False
        self.execution_history = []
        
        logger.info(f"🧠 Ontology system initialized: {self.session_id}")
    
    def _initialize_workflow_designer(self, installed_agents_info: List[Dict[str, Any]] = None):
        """Initialize workflow designer with installed agent information"""
        try:
            if installed_agents_info:
                self.installed_agents_info = installed_agents_info
                logger.info(f"🎯 Updated installed agent information: {len(installed_agents_info)} agents")
            
            # Initialize SmartWorkflowDesigner with installed agent information
            self.workflow_designer = SmartWorkflowDesigner(self.installed_agents_info)
            logger.info(f"🎯 Workflow designer initialized - agent count: {len(self.installed_agents_info)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow designer: {e}")
            # Fallback: use default workflow designer
            self.workflow_designer = SmartWorkflowDesigner()

    async def initialize(self):
        """Initialize the system"""
        try:
            logger.info("🚀 Starting ontology system initialization")
            
            # Initialize each component (if needed)
            # Currently already initialized in constructor
            
            self.is_initialized = True
            logger.info("✅ Ontology system initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Ontology system initialization failed: {e}")
            raise
    
    async def process_query(self, query_text: str, execution_context: ExecutionContext) -> Dict[str, Any]:
        """Query processing - main entry point"""
        try:
            logger.info(f"🚀 Ontology system query processing started: '{query_text[:50]}...'")
            
            # Extract installed agent information and initialize workflow designer
            installed_agents_info = self.workflow_utils.extract_installed_agents_info(execution_context)
            available_agents = self.workflow_utils.extract_available_agents(execution_context)
            
            logger.info(f"🤖 Available agents: {len(available_agents)} - {available_agents}")
            
            # Use unified query processor (NEW!)
            unified_processor = get_unified_query_processor()
            
            if installed_agents_info:
                unified_processor.set_installed_agents_info(installed_agents_info)
            
            # Execute unified processing - complete all analysis in one LLM call
            logger.info("🚀 Starting unified LLM processing...")
            start_time = datetime.now()
            
            unified_result = await unified_processor.process_unified_query(
                query_text, 
                available_agents
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"⏱️ Unified LLM processing complete: {processing_time:.2f}s")
            
            # Convert unified result to existing format
            if installed_agents_info:
                self.workflow_utils.update_installed_agents_info(installed_agents_info)
            workflow_plan = self.workflow_utils.convert_unified_result_to_workflow(unified_result, query_text, execution_context)
            
            # 🔍 Log query analysis results
            logger.warning("="*80)
            logger.warning("🔍 Query Analysis Results")
            logger.warning(f"📝 Original query: {query_text}")
            logger.warning(f"🎯 Query intent: {unified_result.get('intent', 'unknown')}")
            logger.warning(f"📊 Complexity: {unified_result.get('complexity', {}).get('level', 'unknown')} (score: {unified_result.get('complexity', {}).get('score', 0)})")
            
            # Execution strategy information
            execution_plan = unified_result.get('execution_plan', {})
            strategy = execution_plan.get('strategy', 'unknown')
            strategy_reasoning = execution_plan.get('reasoning', 'N/A')
            
            # Strategy name mapping
            strategy_kr = {
                'SINGLE_AGENT': 'Single Agent',
                'SEQUENTIAL': 'Sequential', 
                'PARALLEL': 'Parallel',
                'HYBRID': 'Hybrid'
            }.get(strategy, strategy)
            
            logger.warning(f"⚡ Execution strategy: {strategy_kr} ({strategy})")
            logger.warning(f"🤔 Strategy reasoning: {strategy_reasoning}")
            logger.warning(f"⏱️  Estimated time: {execution_plan.get('estimated_time', 0)}s")
            logger.warning(f"🔄 Workflow type: {workflow_plan.optimization_strategy if workflow_plan else 'unknown'}")
            
            logger.warning(f"\n🤖 Selected agents and individual queries:")
            agent_mappings = unified_result.get('agent_mappings', [])
            logger.warning(f"📊 Total {len(agent_mappings)} tasks")
            
            for i, mapping in enumerate(agent_mappings):
                agent_id = mapping.get('selected_agent', 'unknown')
                task_type = mapping.get('task_type', 'unknown')
                individual_query = mapping.get('individual_query', 'N/A')
                confidence = mapping.get('confidence', 0)
                
                logger.warning(f"\n   Task {i+1}: {agent_id}")
                logger.warning(f"      📋 Task type: {task_type}")
                logger.warning(f"      🔍 Individual query: '{individual_query}'")
                logger.warning(f"      📈 Confidence: {confidence}")
                
                # Task decomposition reason (if present)
                task_reasoning = mapping.get('reasoning', '')
                if task_reasoning:
                    logger.warning(f"      💭 Task decomposition reason: {task_reasoning}")
            
            # Overall reasoning and analysis
            overall_reasoning = unified_result.get('reasoning', 'N/A')
            if overall_reasoning and overall_reasoning != 'N/A':
                logger.warning(f"\n📝 Overall analysis reasoning: {overall_reasoning}")
            
            # Decomposition or integration decision reason
            query_analysis = unified_result.get('query_analysis', {})
            if query_analysis:
                decomposition = query_analysis.get('task_decomposition', {})
                if decomposition:
                    logger.warning(f"\n🔬 Task decomposition analysis:")
                    logger.warning(f"   - Needs decomposition: {decomposition.get('needs_decomposition', 'unknown')}")
                    logger.warning(f"   - Decomposition reason: {decomposition.get('reasoning', 'N/A')}")
                    
            logger.warning("="*80)
            
            # Execute workflow
            logger.info(f"🔧 Starting workflow execution: {len(workflow_plan.steps)} steps")
            execution_results = await self.execution_engine.execute_workflow(
                workflow_plan.query, workflow_plan, execution_context
            )
            
            # 🔍 Log execution results
            logger.warning("="*80)
            logger.warning("📊 Agent Execution Results")
            logger.warning(f"📝 Original query: {query_text}")
            logger.warning(f"⚡ Execution strategy: {strategy_kr} ({strategy})")
            logger.warning(f"✅ Successful agents: {len([r for r in execution_results if r.success])}")
            logger.warning(f"❌ Failed agents: {len([r for r in execution_results if not r.success])}")
            
            # More detailed execution results per agent
            for i, result in enumerate(execution_results):
                # Find the original query for this agent
                original_query = "N/A"
                if i < len(agent_mappings):
                    original_query = agent_mappings[i].get('individual_query', 'N/A')
                elif len(agent_mappings) == 1:
                    # Use original query for single agent case
                    original_query = query_text
                
                logger.warning(f"\n🤖 [{i+1}] {result.agent_id}")
                logger.warning(f"   🔍 Individual query: '{original_query}'")
                logger.warning(f"   📊 Status: {'✅ Success' if result.success else '❌ Failed'}")
                logger.warning(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
                logger.warning(f"   📈 Confidence: {result.confidence}")
                
                # Check metadata from result data
                if isinstance(result.result_data, dict):
                    metadata = result.result_data.get('metadata', {})
                    if metadata:
                        # Extract category - special handling for Samsung agent
                        category = metadata.get('category')
                        if not category:
                            agent_type_check = metadata.get('agent', '')
                            if 'samsung' in agent_type_check.lower():
                                category = 'samsung'
                            else:
                                category = 'general'
                        
                        agent_type = metadata.get('agent', 'unknown')
                        query_type = metadata.get('query_type', 'unknown')
                        logger.warning(f"   📊 Category: {category}")
                        logger.warning(f"   🤖 Agent Type: {agent_type}")
                        logger.warning(f"   📋 Query Type: {query_type}")
                
                # Result preview
                if result.result_data:
                    if isinstance(result.result_data, dict) and 'content' in result.result_data:
                        content = result.result_data['content']
                        if isinstance(content, dict) and 'answer' in content:
                            preview = str(content['answer'])[:100]
                        else:
                            preview = str(content)[:100]
                    else:
                        preview = str(result.result_data)[:100]
                    logger.warning(f"   - Result preview: {preview}...")
            
            logger.warning("="*80)
            
            # Integrate results
            logger.info(f"🔗 Integrating execution results...")
            # Create SemanticQuery (needed for result integration)
            semantic_query = workflow_plan.query if workflow_plan and workflow_plan.query else SemanticQuery(
                query_id=f'semantic_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent='information_retrieval',
                entities=[],
                concepts=[],
                relations=[],
                complexity_score=0.7,
                created_at=datetime.now(),
                metadata={}
            )
            
            # Use split result processor
            integrated_result = await self.result_processor.integrate_results(
                execution_results, workflow_plan, semantic_query
            )
            
            # 🔍 Log integrated result
            logger.warning("="*80)
            logger.warning("🎯 Final Integrated Result")
            
            if isinstance(integrated_result, dict):
                # Check content
                content = integrated_result.get('content', '')
                if content:
                    logger.warning(f"📝 Content type: {type(content).__name__}")
                    if isinstance(content, str):
                        logger.warning(f"📝 Content length: {len(content)} chars")
                        logger.warning(f"📝 Content preview: {content[:200]}...")
                
                # Check metadata
                metadata = integrated_result.get('metadata', {})
                if metadata:
                    logger.warning(f"📊 Metadata:")
                    logger.warning(f"   - Agents used: {metadata.get('agent_count', 0)}")
                    logger.warning(f"   - Integration strategy: {metadata.get('integration_strategy', 'unknown')}")
                    logger.warning(f"   - Category: {metadata.get('category', 'NOT_SET')}")
                
                # Special check for category
                if 'category' in integrated_result:
                    logger.warning(f"🏷️  Final Category: {integrated_result.get('category', 'NOT_SET')}")
                
                # Check reasoning
                if 'reasoning' in integrated_result:
                    reasoning = integrated_result.get('reasoning', '')
                    logger.warning(f"🧠 Reasoning: {reasoning[:200]}...")
            
            logger.warning("="*80)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Record performance metrics - safe type conversion
            raw_quality_score = unified_result.get('quality_assessment', {}).get('overall_confidence', 0.8)
            try:
                # Safely convert quality_score to float
                if isinstance(raw_quality_score, str):
                    quality_score = float(raw_quality_score)
                elif isinstance(raw_quality_score, (int, float)):
                    quality_score = float(raw_quality_score)
                else:
                    quality_score = 0.8  # default value
            except (ValueError, TypeError):
                quality_score = 0.8  # default value
            
            performance_metrics = {
                "total_processing_time": total_time,
                "llm_processing_time": processing_time,
                "execution_time": total_time - processing_time,
                "agents_used": len(workflow_plan.steps),
                "execution_strategy": workflow_plan.optimization_strategy.value if hasattr(workflow_plan.optimization_strategy, 'value') else str(workflow_plan.optimization_strategy),
                "unified_processing": True,  # indicates unified processing
                "quality_score": quality_score
            }
            
            # Generate knowledge graph visualization data
            knowledge_graph_data = {}
            try:
                # Update knowledge graph
                await self._update_knowledge_graph(semantic_query, workflow_plan, execution_results, integrated_result)
                
                # Retrieve visualization data
                knowledge_graph_data = self.get_knowledge_graph_visualization(max_nodes=100)
                logger.info(f"📊 Knowledge graph generated: {len(knowledge_graph_data.get('nodes', []))} nodes, {len(knowledge_graph_data.get('edges', []))} edges")
            except Exception as kg_error:
                logger.error(f"Knowledge graph generation failed: {kg_error}")
                # Fallback to empty structure on failure
                knowledge_graph_data = {
                    "nodes": [],
                    "edges": [],
                    "metadata": {"error": str(kg_error)}
                }
            
            # Build final response - structured response processing
            logger.info("📦 [Structured Response] Building final response")
            logger.info(f"  - integrated_result type: {type(integrated_result)}")
            if isinstance(integrated_result, dict):
                logger.info(f"  - integrated_result keys: {list(integrated_result.keys())}")
            
            # Check if integrated_result is already a structured response
            # Note: uses 'result' key when coming from ontology_enhanced_multi_agent_system
            # uses 'content' key when coming from result_processor
            if isinstance(integrated_result, dict) and (
                ('result' in integrated_result and 'reasoning' in integrated_result) or
                ('content' in integrated_result and 'reasoning' in integrated_result and 'agent_results' in integrated_result)
            ):
                logger.info("✅ [Structured Response] Structured response format detected")
                
                # Convert content to result when coming from result_processor
                if 'content' in integrated_result and 'result' not in integrated_result:
                    logger.info(f"  - Converting content to result")
                    
                    # Category extraction - Special handling for Samsung agent
                    category = "general"
                    agent_results = integrated_result.get("agent_results", [])
                    logger.warning(f"🔍 [Category Fix] agent_results count: {len(agent_results)}")
                    for ar in agent_results:
                        agent_id = ar.get("agent_id", "")
                        logger.warning(f"🔍 [Category Fix] Checking agent_id: '{agent_id}'")
                        if "samsung" in agent_id.lower():
                            category = "samsung"
                            logger.warning(f"✅ [Category Fix] Setting category to 'samsung' based on agent_id!")
                            break
                        # Also check in metadata
                        metadata = ar.get("metadata", {})
                        if metadata.get("category") == "samsung":
                            category = "samsung"
                            logger.warning(f"✅ [Category Fix] Setting category to 'samsung' based on metadata!")
                            break
                    
                    final_response = {
                        "result": integrated_result.get("content", ""),
                        "reasoning": integrated_result.get("reasoning", ""),
                        "success": integrated_result.get("status") == "success",
                        "category": category,
                        "source_agent": integrated_result.get("agent_results", [{}])[0].get("agent_id") if integrated_result.get("agent_results") else None,
                        "execution_time": integrated_result.get("metadata", {}).get("total_execution_time", 0),
                        "single_purpose": len(integrated_result.get("agent_results", [])) == 1,
                        "knowledge_graph_visualization": knowledge_graph_data,
                        "agent_results": integrated_result.get("agent_results", []),
                        "query": query_text,
                        "execution_summary": integrated_result.get("metadata", {}),
                        "performance_metrics": performance_metrics,
                        "workflow_visualization": integrated_result.get("workflow_visualization", ""),
                        "confidence_score": integrated_result.get("metadata", {}).get("average_confidence", 0.7),
                        "sources": integrated_result.get("sources", []),
                        "processing_metadata": {
                            "unified_analysis": unified_result.get('query_analysis', {}),
                            "agent_mappings": unified_result.get('agent_mappings', []),
                            "execution_plan": unified_result.get('execution_plan', {}),
                            "fallback_mode": unified_result.get('fallback_mode', False)
                        }
                    }
                else:
                    # Use existing format as-is
                    logger.info(f"  - result: {integrated_result.get('result', '')[:100]}...")
                    logger.info(f"  - reasoning: {integrated_result.get('reasoning', '')}")
                    logger.info(f"  - agent_results count: {len(integrated_result.get('agent_results', []))}")
                    logger.info(f"  - single_purpose: {integrated_result.get('single_purpose', False)}")
                    
                    final_response = integrated_result.copy()
                    # Merge additional fields
                    final_response.update({
                        "query": query_text,
                        "knowledge_graph_visualization": knowledge_graph_data,
                        "performance_metrics": performance_metrics,
                        "processing_metadata": {
                            "unified_analysis": unified_result.get('query_analysis', {}),
                            "agent_mappings": unified_result.get('agent_mappings', []),
                            "execution_plan": unified_result.get('execution_plan', {}),
                            "fallback_mode": unified_result.get('fallback_mode', False)
                        }
                    })
            else:
                logger.info("⚠️ [Structured Response] Using basic response format")
                logger.info(f"  - content: {integrated_result.get('content', '')[:100]}...")
                
                # Process structured result returned by ResultProcessor
                if isinstance(integrated_result, dict) and 'reasoning' in integrated_result and 'agent_results' in integrated_result:
                    logger.info("✅ [Structured Response] ResultProcessor structured response detected")
                    logger.info(f"  - reasoning: {integrated_result.get('reasoning', '')[:100]}...")
                    logger.info(f"  - agent_results count: {len(integrated_result.get('agent_results', []))}")
                    
                    # Convert to structured response
                    
                    # Category extraction - Special handling for Samsung agent
                    category = "general"
                    agent_results = integrated_result.get("agent_results", [])
                    logger.warning(f"🔍 [Category Fix] agent_results count: {len(agent_results)}")
                    for ar in agent_results:
                        agent_id = ar.get("agent_id", "")
                        logger.warning(f"🔍 [Category Fix] Checking agent_id: '{agent_id}'")
                        if "samsung" in agent_id.lower():
                            category = "samsung"
                            logger.warning(f"✅ [Category Fix] Setting category to 'samsung' based on agent_id!")
                            break
                        # Also check in metadata
                        metadata = ar.get("metadata", {})
                        if metadata.get("category") == "samsung":
                            category = "samsung"
                            logger.warning(f"✅ [Category Fix] Set category to 'samsung' based on metadata!")
                            break
                    
                    final_response = {
                        "result": integrated_result.get("content", "Unable to generate a response."),
                        "reasoning": integrated_result.get("reasoning", ""),
                        "success": True,
                        "category": category,
                        "source_agent": integrated_result.get("agent_results", [{}])[0].get("agent_id") if integrated_result.get("agent_results") else None,
                        "execution_time": integrated_result.get("metadata", {}).get("total_execution_time", 0),
                        "single_purpose": len(integrated_result.get("agent_results", [])) == 1,
                        "knowledge_graph_visualization": knowledge_graph_data,
                        "agent_results": integrated_result.get("agent_results", []),
                        "query": query_text,
                        "execution_summary": integrated_result.get("metadata", {}),
                        "performance_metrics": performance_metrics,
                        "workflow_visualization": integrated_result.get("workflow_visualization", ""),
                        "confidence_score": integrated_result.get("metadata", {}).get("average_confidence", 0.7),
                        "sources": integrated_result.get("sources", []),
                        "processing_metadata": {
                            "unified_analysis": unified_result.get('query_analysis', {}),
                            "agent_mappings": unified_result.get('agent_mappings', []),
                            "execution_plan": unified_result.get('execution_plan', {}),
                            "fallback_mode": unified_result.get('fallback_mode', False)
                        }
                    }
                else:
                    # Default format
                    
                    # Category extraction - Special handling for Samsung agent
                    category = "general"
                    agent_results = integrated_result.get("agent_results", [])
                    logger.warning(f"🔍 [Category Fix] Number of agent_results: {len(agent_results)}")
                    for ar in agent_results:
                        agent_id = ar.get("agent_id", "")
                        logger.warning(f"🔍 [Category Fix] Checking agent_id: '{agent_id}'")
                        if "samsung" in agent_id.lower():
                            category = "samsung"
                            logger.warning(f"✅ [Category Fix] Set category to 'samsung' based on agent_id!")
                            break
                        # Also check in metadata
                        metadata = ar.get("metadata", {})
                        if metadata.get("category") == "samsung":
                            category = "samsung"
                            logger.warning(f"✅ [Category Fix] Set category to 'samsung' based on metadata!")
                            break
                    
                    final_response = {
                        "result": integrated_result.get("content", "Unable to generate a response."),
                        "reasoning": integrated_result.get("reasoning", ""),  # also check reasoning
                        "success": integrated_result.get("status") == "success",
                        "category": category,
                        "source_agent": None,
                        "execution_time": integrated_result.get("metadata", {}).get("total_execution_time", 0),
                        "single_purpose": False,
                        "knowledge_graph_visualization": knowledge_graph_data,
                        "agent_results": [],
                        "query": query_text,
                        "execution_summary": integrated_result.get("metadata", {}),
                        "performance_metrics": performance_metrics,
                        "workflow_visualization": integrated_result.get("workflow_visualization", ""),
                        "confidence_score": integrated_result.get("metadata", {}).get("average_confidence", 0.7),
                        "sources": integrated_result.get("sources", []),
                        "processing_metadata": {
                            "unified_analysis": unified_result.get('query_analysis', {}),
                            "agent_mappings": unified_result.get('agent_mappings', []),
                            "execution_plan": unified_result.get('execution_plan', {}),
                            "fallback_mode": unified_result.get('fallback_mode', False)
                        }
                    }
            
            logger.info(f"✅ Ontology query processing complete (total {total_time:.2f}s, quality: {performance_metrics['quality_score']:.2f})")
            
            # 🔍 Log final return value
            logger.warning("="*80)
            logger.warning("📦 Final return value")
            
            if isinstance(final_response, dict):
                # Basic info
                logger.warning(f"🎯 Success: {'✅' if final_response.get('success', False) else '❌'}")
                logger.warning(f"🏷️  Category: {final_response.get('category', 'NOT_SET')}")
                logger.warning(f"🤖 Source Agent: {final_response.get('source_agent', 'None')}")
                logger.warning(f"⏱️  Execution time: {final_response.get('execution_time', 0):.2f}s")
                logger.warning(f"💯 Confidence: {final_response.get('confidence_score', 0):.2f}")
                
                # result content
                result = final_response.get('result', '')
                if result:
                    logger.warning(f"📝 Result type: {type(result).__name__}")
                    if isinstance(result, str):
                        logger.warning(f"📝 Result length: {len(result)} chars")
                        if result.startswith('<!DOCTYPE') or result.startswith('<html'):
                            logger.warning(f"📝 Result: HTML content")
                        else:
                            logger.warning(f"📝 Result preview: {result[:200]}...")
                
                # agent_results summary
                agent_results = final_response.get('agent_results', [])
                if agent_results:
                    logger.warning(f"🤖 Number of agent results: {len(agent_results)}")
                    for i, ar in enumerate(agent_results):
                        agent_id = ar.get('agent_id', 'unknown')
                        execution_time = ar.get('execution_time', 0)
                        
                        # Category extraction - Special handling for Samsung agent
                        category = ar.get('category')
                        logger.warning(f"🔍 [Category Fix] agent_id '{agent_id}' direct category: '{category}'")
                        if not category:
                            # Find category in result.metadata
                            result_data = ar.get('result', {})
                            if isinstance(result_data, dict):
                                metadata = result_data.get('metadata', {})
                                category = metadata.get('category')
                                logger.warning(f"🔍 [Category Fix] category from result.metadata: '{category}'")
                            
                            # If still not found, infer from agent_id
                            if not category:
                                if 'samsung' in agent_id.lower():
                                    category = 'samsung'
                                    logger.warning(f"✅ [Category Fix] Set 'samsung' via agent_id inference!")
                                else:
                                    category = 'general'
                        else:
                            logger.warning(f"✅ [Category Fix] Keeping existing category '{category}'")
                        
                        agent_type = ar.get('agentType', 'unknown')
                        confidence = ar.get('confidence', 0)
                        
                        logger.warning(f"   [{i+1}] {agent_id}")
                        logger.warning(f"       ⏱️  Execution time: {execution_time:.2f}s")
                        logger.warning(f"       📊 Category: {category}")
                        logger.warning(f"       🤖 Type: {agent_type}")
                        logger.warning(f"       📈 Confidence: {confidence}")
                        
                        # Individual query info (if present)
                        if i < len(agent_mappings):
                            individual_query = agent_mappings[i].get('individual_query', 'N/A')
                            logger.warning(f"       🔍 Individual query: '{individual_query}'")
                
                # Add processing_metadata info
                processing_metadata = final_response.get('processing_metadata', {})
                if processing_metadata:
                    logger.warning(f"\n🔍 Processing metadata:")
                    
                    # execution_plan info
                    execution_plan = processing_metadata.get('execution_plan', {})
                    if execution_plan:
                        strategy = execution_plan.get('strategy', 'unknown')
                        reasoning = execution_plan.get('reasoning', 'N/A')
                        logger.warning(f"   ⚡ Final execution strategy: {strategy}")
                        logger.warning(f"   🤔 Strategy selection reason: {reasoning}")
                    
                    # agent_mappings info
                    agent_mappings_meta = processing_metadata.get('agent_mappings', [])
                    if agent_mappings_meta:
                        logger.warning(f"   📊 Total {len(agent_mappings_meta)} task mappings")
                
                # metadata summary
                exec_summary = final_response.get('execution_summary', {})
                if exec_summary:
                    logger.warning(f"\n📊 Execution summary:")
                    logger.warning(f"   - Total agents: {exec_summary.get('total_agents', 0)}")
                    logger.warning(f"   - Successful agents: {exec_summary.get('successful_agents', 0)}")
                    logger.warning(f"   - Average confidence: {exec_summary.get('average_confidence', 0):.2f}")
                    
                # Final category check - Special handling for Samsung agent
                final_category = final_response.get('category')
                logger.warning(f"🔍 [Final Category] Base category from final_response: '{final_category}'")
                if not final_category:
                    # Check if Samsung agent exists in agent_results
                    agent_results = final_response.get('agent_results', [])
                    logger.warning(f"🔍 [Final Category] Number of agent_results: {len(agent_results)}")
                    for ar in agent_results:
                        agent_id = ar.get('agent_id', '')
                        logger.warning(f"🔍 [Final Category] Checking agent_id: '{agent_id}'")
                        if 'samsung' in agent_id.lower():
                            final_category = 'samsung'
                            logger.warning(f"✅ [Final Category] Samsung agent found! Set category to 'samsung'!")
                            break
                        # Also check in agent result metadata
                        result_data = ar.get('result', {})
                        if isinstance(result_data, dict):
                            metadata = result_data.get('metadata', {})
                            if metadata.get('category') == 'samsung':
                                final_category = 'samsung'
                                logger.warning(f"✅ [Final Category] Samsung category found in metadata! Set to 'samsung'!")
                                break
                    
                    # If still not found, use default value
                    if not final_category:
                        final_category = 'general'
                        logger.warning(f"⚠️ [Final Category] Samsung agent not found, set to 'general'")
                else:
                    logger.warning(f"✅ [Final Category] Keeping existing category '{final_category}'")
                
                logger.warning(f"\n📊 Final Category: {final_category}")
            
            # Overall processing summary
            total_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"\n⏱️  Total processing time: {total_time:.2f}s")
            logger.warning(f"🎯 Query processing strategy: {strategy_kr} ({strategy})")
            logger.warning(f"📊 Agent count: {len(agent_mappings)}")
            logger.warning(f"📈 Overall success rate: {len([r for r in execution_results if r.success]) / len(execution_results) * 100:.1f}%")
            
            logger.warning("="*80)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Ontology query processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: process using legacy method
            return await self._fallback_query_processing(query_text, execution_context)



    async def _fallback_query_processing(self, query_text: str, execution_context: ExecutionContext) -> Dict[str, Any]:
        """Fallback query processing"""
        logger.warning("🔄 Processing query in fallback mode")
        
        try:
            # Try processing using legacy method
            workflow_plan = self.workflow_utils.create_minimal_workflow(query_text, execution_context)
            
            execution_results = await self.execution_engine.execute_workflow(
                workflow_plan.query, workflow_plan, execution_context
            )
            
            # Create SemanticQuery (for fallback)
            semantic_query = workflow_plan.query if workflow_plan and workflow_plan.query else SemanticQuery(
                query_id=f'fallback_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent='information_retrieval',
                entities=[],
                concepts=[],
                relations=[],
                complexity_score=0.5,
                created_at=datetime.now(),
                metadata={"fallback_mode": True}
            )
            
            integrated_result = await self.result_processor.integrate_results(
                execution_results, workflow_plan, semantic_query
            )
            
            # Also generate knowledge graph in fallback mode
            knowledge_graph_data = self.get_knowledge_graph_visualization(max_nodes=50)
            
            return {
                "query": query_text,
                "response": integrated_result.get("content", "Sorry, an error occurred during processing."),
                "execution_summary": integrated_result.get("metadata", {}),
                "performance_metrics": {
                    "total_processing_time": 30,
                    "fallback_mode": True
                },
                "confidence_score": 0.5,
                "sources": [],
                "knowledge_graph": knowledge_graph_data,  # Add knowledge graph
                "processing_metadata": {"fallback_processing": True}
            }
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            
            # Minimal knowledge graph structure
            minimal_knowledge_graph = {
                "nodes": [
                    {
                        "id": f"error_{uuid.uuid4().hex[:8]}",
                        "label": "Processing failed",
                        "type": "error",
                        "properties": {"error": str(e)[:100]}
                    }
                ],
                "edges": [],
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "error": True
                }
            }
            
            return {
                "query": query_text,
                "response": "Unable to process the request due to a system error. Please try again later.",
                "execution_summary": {},
                "performance_metrics": {"error": True},
                "confidence_score": 0.0,
                "sources": [],
                "knowledge_graph": minimal_knowledge_graph,  # Include knowledge graph even on error
                "processing_metadata": {"system_error": True, "error_message": str(e)}
            }
    
    async def _generate_detailed_reasoning(
        self, 
        execution_results: List[AgentExecutionResult],
        workflow_plan: WorkflowPlan,
        semantic_query: SemanticQuery,
        complexity_analysis: Dict[str, Any],
        integrated_result: Dict[str, Any]
    ) -> str:
        """Generate detailed reasoning - using split managers"""
        return await self.reasoning_generator.generate_detailed_reasoning(
            execution_results, workflow_plan, semantic_query, complexity_analysis, integrated_result
        )
    
    def _generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
        """Generate Mermaid diagram from workflow plan"""
        try:
            mermaid_lines = ["graph TD"]
            
            # Start node
            mermaid_lines.append('    Start(["🚀 Start"]) --> Query["📝 Query Analysis"]')
            
            # Add each step as a node
            for i, step in enumerate(workflow_plan.steps):
                step_id = f"Step{i+1}"
                agent_name = step.agent_id.replace('_', ' ').title()
                purpose = step.semantic_purpose[:30] + "..." if len(step.semantic_purpose) > 30 else step.semantic_purpose
                
                # Node definition
                mermaid_lines.append(f'    {step_id}["🤖 {agent_name}<br/>{purpose}"]')
                
                # Connection relationships
                if i == 0:
                    mermaid_lines.append(f'    Query --> {step_id}')
                else:
                    prev_step_id = f"Step{i}"
                    mermaid_lines.append(f'    {prev_step_id} --> {step_id}')
                
                # If there are dependencies
                if hasattr(step, 'depends_on') and step.depends_on:
                    for dep in step.depends_on:
                        # Find dependency steps
                        for j, dep_step in enumerate(workflow_plan.steps):
                            if dep_step.step_id == dep:
                                dep_step_id = f"Step{j+1}"
                                mermaid_lines.append(f'    {dep_step_id} -.-> {step_id}')
                                break
            
            # Connect from last step to result
            if workflow_plan.steps:
                last_step_id = f"Step{len(workflow_plan.steps)}"
                mermaid_lines.append(f'    {last_step_id} --> Result[["✅ Result Integration"]]')
                mermaid_lines.append('    Result --> End(["🎉 Complete"])')
            else:
                mermaid_lines.append('    Query --> End(["🎉 Complete"])')
            
            # Add styles
            mermaid_lines.extend([
                "",
                "    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
                "    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px", 
                "    classDef agent fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
                "    classDef result fill:#fff3e0,stroke:#e65100,stroke-width:2px",
                "",
                "    class Start,End startEnd",
                "    class Query,Result result"
            ])
            
            # Apply style to agent nodes
            for i in range(len(workflow_plan.steps)):
                mermaid_lines.append(f"    class Step{i+1} agent")
            
            return "\n".join(mermaid_lines)
            
        except Exception as e:
            logger.error(f"Mermaid diagram generation failed: {e}")
            return f'graph TD\n    Start(["Start"]) --> Error["Diagram generation failed: {str(e)}"]\n    Error --> End(["End"])'
    
    def _safe_analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Safe complexity analysis - using split managers"""
        return self.complexity_analyzer.safe_analyze_complexity(semantic_query)
    
    async def _integrate_results(self, 
                               execution_results: List[AgentExecutionResult],
                               workflow_plan: WorkflowPlan,
                               semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Result integration"""
        try:
            logger.info(f"🔄 Starting result integration - total {len(execution_results)} results")
            
            # Detailed logging of all results
            for i, result in enumerate(execution_results):
                logger.info(f"  Result {i+1}: {result.agent_id}")
                logger.info(f"    Success: {result.success}")
                logger.info(f"    Execution time: {result.execution_time:.2f}s")
                logger.info(f"    Confidence: {result.confidence}")
                
                if result.data:
                    logger.info(f"    Data type: {type(result.data)}")
                    if isinstance(result.data, dict):
                        logger.info(f"    Data keys: {list(result.data.keys())}")
                        # Log a portion of the actual data content
                        for key, value in result.data.items():
                            if isinstance(value, str) and len(value) > 100:
                                logger.info(f"      {key}: {value[:100]}...")
                            else:
                                logger.info(f"      {key}: {value}")
                    else:
                        logger.info(f"    Data content: {str(result.data)[:200]}...")
                
                if result.error_message:
                    logger.warning(f"    Error: {result.error_message}")
            
            # Extract only successful results
            successful_results = [r for r in execution_results if r.success]
            logger.info(f"📊 Successful results: {len(successful_results)}")
            
            if not successful_results:
                logger.warning("⚠️ All agent executions failed.")
                return {
                    "status": "failed",
                    "message": "All agent executions failed.",
                    "error_details": [r.error_message for r in execution_results if r.error_message]
                }
            
            # Collect result data
            result_data = []
            for result in successful_results:
                if result.data:
                    processed_data = {
                        "agent_id": result.agent_id,
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time
                    }
                    result_data.append(processed_data)
                    logger.info(f"✅ Result data added: {result.agent_id}")
            
            logger.info(f"📋 Result data to process: {len(result_data)}")
            
            # Integrate results by intent
            intent = getattr(semantic_query, 'intent', 'general')
            logger.info(f"🎯 Starting intent-based integration: {intent}")
            
            if intent == "information_retrieval":
                integrated_content = self._integrate_information_results(result_data)
            elif intent == "analysis":
                integrated_content = self._integrate_analysis_results(result_data)
            elif intent == "comparison":
                integrated_content = self._integrate_comparison_results(result_data)
            else:
                integrated_content = self._integrate_general_results(result_data)
            
            logger.info(f"✅ Integration complete - content length: {len(integrated_content)} chars")
            
            return {
                "status": "success",
                "content": integrated_content,
                "metadata": {
                    "total_agents": len(execution_results),
                    "successful_agents": len(successful_results),
                    "total_execution_time": sum(r.execution_time for r in execution_results),
                    "average_confidence": sum(r.confidence for r in successful_results) / len(successful_results) if successful_results else 0,
                    "strategy_used": getattr(workflow_plan, 'optimization_strategy', 'AUTO')
                }
            }
            
        except Exception as e:
            logger.error(f"Result integration failed: {e}")
            return {
                "status": "error",
                "message": f"Error during result integration: {str(e)}",
                "partial_results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in execution_results]
            }
    
    def _integrate_information_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate information search results"""
        logger.info(f"🔍 Starting information search result integration - {len(result_data)} results")
        
        contents = []
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            
            logger.info(f"  Processing: {agent_id}")
            logger.info(f"  Data type: {type(data)}")
            
            content = None
            
            if isinstance(data, dict):
                logger.info(f"  Data keys: {list(data.keys())}")
                
                # Use answer key preferentially if present
                if "answer" in data:
                    content = data["answer"]
                    logger.info(f"  📝 Extracting content from answer key: {len(str(content))} chars")
                # Handle various forms of search results
                elif "search_results" in data:
                    search_results = data["search_results"]
                    if isinstance(search_results, list) and search_results:
                        content = f"Search results {len(search_results)} items:\n"
                        for i, item in enumerate(search_results[:5], 1):  # top 5 only
                            title = item.get("title", "No title")
                            snippet = item.get("snippet", item.get("description", ""))
                            content += f"{i}. {title}\n   {snippet}\n"
                    else:
                        content = "No search results found."
                elif "items" in data:
                    items = data["items"]
                    if isinstance(items, list) and items:
                        content = f"Search items {len(items)}:\n"
                        for i, item in enumerate(items[:5], 1):
                            title = item.get("title", item.get("name", "No title"))
                            content += f"{i}. {title}\n"
                    else:
                        content = "No search items found."
                elif "content" in data:
                    content = data["content"]
                elif "text" in data:
                    content = data["text"]
                elif "result" in data:
                    # Check if result is a dict and has answer key
                    result_value = data["result"]
                    if isinstance(result_value, dict) and "answer" in result_value:
                        content = result_value["answer"]
                        logger.info(f"  📝 Extracting content from result.answer: {len(str(content))} chars")
                    else:
                        content = str(result_value)
                else:
                    # If no known key, extract main content without converting entire dict to string
                    logger.warning(f"  ⚠️ No known key found, full structure: {data}")
                    # Find and extract the most important text content from the dictionary
                    for key in ['response', 'message', 'output', 'data']:
                        if key in data:
                            potential_content = data[key]
                            if isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                                content = potential_content
                                logger.info(f"  📝 Extracting content from {key} key")
                                break
                    
                    # If still no content, convert entire dict to string
                    if content is None:
                        content = str(data)
                        logger.warning(f"  ⚠️ Fallback: converting entire dictionary to string")
            else:
                content = str(data)
                logger.info(f"  📝 Direct conversion from string/other type")
            
            if content and str(content).strip():
                contents.append(content)  # exclude agent_id
                logger.info(f"  ✅ Content added: {len(str(content))} chars (agent_id excluded)")
            else:
                logger.warning(f"  ⚠️ Empty content: {agent_id}")
        
        final_content = "\n\n".join(contents) if contents else "No search results found."
        logger.info(f"🔍 Information search result integration complete - final length: {len(final_content)} chars")
        
        return final_content
    
    def _integrate_analysis_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate analysis results"""
        analysis_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            confidence = result["confidence"]
            
            if isinstance(data, dict):
                insights = data.get("insights", [])
                if insights:
                    analysis_parts.append(f"**Analysis Result (confidence: {confidence:.2f})**")  # exclude agent_id
                    for insight in insights:
                        analysis_parts.append(f"- {insight}")
                else:
                    analysis_parts.append(str(data))  # exclude agent_id
            else:
                analysis_parts.append(str(data))  # exclude agent_id
        
        return "\n".join(analysis_parts)
    
    def _integrate_comparison_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate comparison results"""
        comparison_parts = ["## Comparison Analysis Results"]
        
        for i, result in enumerate(result_data, 1):
            agent_id = result["agent_id"]
            data = result["data"]
            
            comparison_parts.append(f"\n### {i}. Result")  # exclude agent_id
            comparison_parts.append(str(data))
        
        return "\n".join(comparison_parts)
    
    def _integrate_general_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate general results"""
        general_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            
            general_parts.append(str(data))  # exclude agent_id
        
        return "\n".join(general_parts)
    
    async def _update_knowledge_graph(self, 
                                    semantic_query: SemanticQuery,
                                    workflow_plan: WorkflowPlan,
                                    execution_results: List[AgentExecutionResult],
                                    integrated_result: Dict[str, Any]):
        """Update ontology knowledge graph - generate rich graph based on image reference"""
        try:
            logger.info("🔗 Starting rich ontology knowledge graph update")
            
            workflow_id = workflow_plan.plan_id
            
            # 1. Create core query node (central hub)
            query_id = f"query_{semantic_query.query_id}"
            await self.knowledge_graph.add_concept(query_id, "query", {
                "natural_language": semantic_query.natural_language,
                "intent": semantic_query.intent,
                "complexity": getattr(semantic_query, 'complexity_score', 0.5),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            })
            
            # 2. Create workflow plan node
            workflow_node = f"workflow_{workflow_id}"
            await self.knowledge_graph.add_concept(workflow_node, "workflow", {
                "plan_id": workflow_id,
                "optimization_strategy": str(workflow_plan.optimization_strategy),
                "estimated_quality": workflow_plan.estimated_quality,
                "estimated_time": workflow_plan.estimated_time,
                "total_steps": len(workflow_plan.steps),
                "session_id": self.session_id,
                "reasoning_chain": workflow_plan.reasoning_chain[:3] if workflow_plan.reasoning_chain else []
            })
            
            # 3. Query → workflow relationship
            await self.knowledge_graph.add_relation(query_id, "triggers", workflow_node, {
                "trigger_type": "user_request",
                "confidence": 0.9
            })
            
            # 4. Create agent nodes and execution results
            for i, result in enumerate(execution_results):
                agent_node = f"agent_{result.agent_id}"
                result_node = f"result_{workflow_id}_{i}"
                
                # Agent node
                await self.knowledge_graph.add_concept(agent_node, "agent", {
                    "agent_id": result.agent_id,
                    "agent_type": str(result.agent_type) if hasattr(result, 'agent_type') else "unknown",
                    "capabilities": self._infer_agent_capabilities(result.agent_id),
                    "performance_history": {
                        "last_execution_time": result.execution_time,
                        "last_confidence": result.confidence,
                        "last_success": result.is_successful()
                    }
                })
                
                # Execution result node
                await self.knowledge_graph.add_concept(result_node, "execution_result", {
                    "agent_id": result.agent_id,
                    "execution_time": result.execution_time,
                    "confidence": result.confidence,
                    "success": result.is_successful(),
                    "workflow_step": i + 1,
                    "result_type": self._classify_result_type(result.result_data)
                })
                
                # Create relationships
                await self.knowledge_graph.add_relation(workflow_node, "executes_with", agent_node, {
                    "execution_order": i + 1,
                    "step_purpose": f"Step {i+1} execution"
                })
                
                await self.knowledge_graph.add_relation(agent_node, "produces", result_node, {
                    "production_time": result.execution_time,
                    "quality_score": result.confidence
                })
                
                await self.knowledge_graph.add_relation(result_node, "contributes_to", workflow_node, {
                    "contribution_weight": result.confidence,
                    "step_number": i + 1
                })
            
            # 5. Create domain and concept nodes
            await self._create_domain_and_concept_nodes(semantic_query, execution_results, workflow_id)
            
            # 6. Create task and capability nodes
            await self._create_task_and_capability_nodes(semantic_query, execution_results, workflow_id)
            
            # 7. Create performance and quality metric nodes
            await self._create_performance_metric_nodes(execution_results, workflow_id)
            
            # 8. Create inter-agent collaboration relationships
            await self._create_agent_collaboration_network(execution_results, workflow_id)
            
            # 9. Create temporal ordering relationships
            await self._create_temporal_sequence_relations(execution_results, workflow_id)
            
            # 10. Create knowledge pattern and learning nodes
            await self._create_knowledge_pattern_nodes(semantic_query, execution_results, integrated_result, workflow_id)
            
            # 11. Create context and environment nodes
            await self._create_context_environment_nodes(semantic_query, workflow_id)
            
            logger.info(f"✅ Rich ontology knowledge graph update complete - workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Knowledge graph update failed: {e}")
    
    def _infer_agent_capabilities(self, agent_id: str) -> List[str]:
        """Infer capabilities from agent ID - using split managers"""
        return self.complexity_analyzer.infer_agent_capabilities(agent_id)
    
    def _classify_result_type(self, result_data: Any) -> str:
        """Classify result data type - using split managers"""
        return self.complexity_analyzer.classify_result_type(result_data)
    
    async def _create_domain_and_concept_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create domain and concept nodes"""
        try:
            # Extract domain from query
            query_text = semantic_query.natural_language.lower()
            domains = []
            
            if any(word in query_text for word in ['날씨', 'weather', '기상', '온도']):
                domains.append('weather')
            if any(word in query_text for word in ['환율', 'exchange', '달러', '유로']):
                domains.append('finance')
            if any(word in query_text for word in ['계산', 'calculate', '수학', 'math']):
                domains.append('calculation')
            if any(word in query_text for word in ['검색', 'search', '찾아', 'find']):
                domains.append('information')
            if any(word in query_text for word in ['메모', 'memo', '기록', 'note']):
                domains.append('productivity')
            
            if not domains:
                domains = ['general']
            
            # Create domain nodes
            for domain in domains:
                domain_node = f"domain_{domain}"
                await self.knowledge_graph.add_concept(domain_node, "domain", {
                    "domain_name": domain,
                    "query_relevance": 0.8,
                    "last_accessed": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                })
                
                # Query-domain relationship
                query_id = f"query_{semantic_query.query_id}"
                await self.knowledge_graph.add_relation(query_id, "belongs_to_domain", domain_node, {
                    "relevance_score": 0.8
                })
                
                # Agent-domain relationship
                for result in execution_results:
                    if result.is_successful():
                        agent_node = f"agent_{result.agent_id}"
                        await self.knowledge_graph.add_relation(agent_node, "operates_in_domain", domain_node, {
                            "performance_score": result.confidence,
                            "execution_time": result.execution_time
                        })
            
            # Create concept entities
            entities = getattr(semantic_query, 'entities', [])
            for entity in entities[:5]:  # max 5
                entity_node = f"entity_{entity}"
                await self.knowledge_graph.add_concept(entity_node, "entity", {
                    "entity_name": entity,
                    "source_query": semantic_query.query_id,
                    "extraction_confidence": 0.7
                })
                
                # Query-entity relationship
                await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "contains_entity", entity_node, {
                    "entity_importance": 0.7
                })
                
        except Exception as e:
            logger.warning(f"Domain and concept node creation failed: {e}")
    
    async def _create_task_and_capability_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create task and capability nodes"""
        try:
            # Create task node
            task_node = f"task_{workflow_id}"
            await self.knowledge_graph.add_concept(task_node, "task", {
                "task_description": semantic_query.natural_language,
                "task_intent": semantic_query.intent,
                "complexity_level": getattr(semantic_query, 'complexity_score', 0.5),
                "workflow_id": workflow_id
            })
            
            # Query-task relationship
            await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "defines_task", task_node, {
                "task_clarity": 0.8
            })
            
            # Create capability nodes for each agent
            for result in execution_results:
                capabilities = self._infer_agent_capabilities(result.agent_id)
                
                for capability in capabilities:
                    capability_node = f"capability_{capability}"
                    await self.knowledge_graph.add_concept(capability_node, "capability", {
                        "capability_name": capability,
                        "capability_type": "agent_skill",
                        "last_used": datetime.now().isoformat()
                    })
                    
                    # Agent-capability relationship
                    agent_node = f"agent_{result.agent_id}"
                    await self.knowledge_graph.add_relation(agent_node, "has_capability", capability_node, {
                        "proficiency_level": result.confidence,
                        "usage_frequency": 1
                    })
                    
                    # Task-capability relationship
                    await self.knowledge_graph.add_relation(task_node, "requires_capability", capability_node, {
                        "requirement_strength": 0.6
                    })
                    
        except Exception as e:
            logger.warning(f"Task and capability node creation failed: {e}")
    
    async def _create_performance_metric_nodes(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create performance and quality metric nodes"""
        try:
            # Overall workflow performance metrics
            total_time = sum(r.execution_time for r in execution_results)
            avg_confidence = sum(r.confidence for r in execution_results) / len(execution_results) if execution_results else 0
            success_rate = sum(1 for r in execution_results if r.is_successful()) / len(execution_results) if execution_results else 0
            
            workflow_performance_node = f"performance_workflow_{workflow_id}"
            await self.knowledge_graph.add_concept(workflow_performance_node, "performance_metric", {
                "metric_type": "workflow_performance",
                "total_execution_time": total_time,
                "average_confidence": avg_confidence,
                "success_rate": success_rate,
                "agent_count": len(execution_results),
                "workflow_id": workflow_id
            })
            
            # Individual agent performance metrics
            for i, result in enumerate(execution_results):
                agent_performance_node = f"performance_{result.agent_id}_{workflow_id}"
                await self.knowledge_graph.add_concept(agent_performance_node, "performance_metric", {
                    "metric_type": "agent_performance",
                    "agent_id": result.agent_id,
                    "execution_time": result.execution_time,
                    "confidence": result.confidence,
                    "success": result.is_successful(),
                    "performance_tier": "high" if result.confidence > 0.8 else "medium" if result.confidence > 0.5 else "low"
                })
                
                # Agent-performance relationship
                agent_node = f"agent_{result.agent_id}"
                await self.knowledge_graph.add_relation(agent_node, "has_performance", agent_performance_node, {
                    "measurement_timestamp": datetime.now().isoformat()
                })
                
                # Workflow-performance relationship
                workflow_node = f"workflow_{workflow_id}"
                await self.knowledge_graph.add_relation(workflow_node, "measured_by", agent_performance_node, {
                    "step_number": i + 1
                })
                
        except Exception as e:
            logger.warning(f"Performance metric node creation failed: {e}")
    
    async def _create_agent_collaboration_network(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create inter-agent collaboration relationships"""
        try:
            successful_results = [r for r in execution_results if r.is_successful()]
            
            # Collaboration relationships between successful agents
            for i, result1 in enumerate(successful_results):
                for result2 in successful_results[i+1:]:
                    agent1_node = f"agent_{result1.agent_id}"
                    agent2_node = f"agent_{result2.agent_id}"
                    
                    # Collaboration relationship
                    await self.knowledge_graph.add_relation(agent1_node, "collaborated_with", agent2_node, {
                        "workflow_id": workflow_id,
                        "collaboration_success": True,
                        "combined_confidence": (result1.confidence + result2.confidence) / 2,
                        "collaboration_type": "sequential" if abs(i - successful_results.index(result2)) == 1 else "parallel"
                    })
                    
                    # Complementary relationship (when capabilities differ)
                    cap1 = self._infer_agent_capabilities(result1.agent_id)
                    cap2 = self._infer_agent_capabilities(result2.agent_id)
                    
                    if set(cap1) != set(cap2):  # When capabilities differ
                        await self.knowledge_graph.add_relation(agent1_node, "complements", agent2_node, {
                            "complementarity_score": 0.8,
                            "capability_overlap": len(set(cap1) & set(cap2)) / max(len(cap1), len(cap2))
                        })
                        
        except Exception as e:
            logger.warning(f"Agent collaboration network creation failed: {e}")
    
    async def _create_temporal_sequence_relations(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create temporal ordering relationships"""
        try:
            # Predecessor/successor relationships by execution order
            for i in range(len(execution_results) - 1):
                current_result = execution_results[i]
                next_result = execution_results[i + 1]
                
                current_agent = f"agent_{current_result.agent_id}"
                next_agent = f"agent_{next_result.agent_id}"
                
                # Predecessor relationship
                await self.knowledge_graph.add_relation(current_agent, "precedes", next_agent, {
                    "sequence_order": i + 1,
                    "time_gap": 0.1,  # assumed time gap
                    "workflow_id": workflow_id
                })
                
                # Dependencies between results
                current_result_node = f"result_{workflow_id}_{i}"
                next_result_node = f"result_{workflow_id}_{i+1}"
                
                await self.knowledge_graph.add_relation(current_result_node, "influences", next_result_node, {
                    "influence_strength": 0.6,
                    "dependency_type": "sequential"
                })
                
        except Exception as e:
            logger.warning(f"Temporal ordering relationship creation failed: {e}")
    
    async def _create_knowledge_pattern_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], integrated_result: Dict[str, Any], workflow_id: str):
        """Create knowledge pattern and learning nodes"""
        try:
            # Learning pattern node
            pattern_node = f"pattern_{workflow_id}"
            await self.knowledge_graph.add_concept(pattern_node, "knowledge_pattern", {
                "pattern_type": "workflow_execution",
                "query_type": semantic_query.intent,
                "agent_combination": [r.agent_id for r in execution_results],
                "success_pattern": [r.is_successful() for r in execution_results],
                "performance_pattern": [r.confidence for r in execution_results],
                "learned_at": datetime.now().isoformat()
            })
            
            # Query-pattern relationship
            await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "generates_pattern", pattern_node, {
                "pattern_strength": integrated_result.get('confidence', 0.7)
            })
            
            # Insight node (from successful executions)
            successful_results = [r for r in execution_results if r.is_successful()]
            if successful_results:
                insight_node = f"insight_{workflow_id}"
                await self.knowledge_graph.add_concept(insight_node, "insight", {
                    "insight_type": "execution_success",
                    "key_factors": [r.agent_id for r in successful_results],
                    "success_rate": len(successful_results) / len(execution_results),
                    "confidence_level": sum(r.confidence for r in successful_results) / len(successful_results),
                    "discovered_at": datetime.now().isoformat()
                })
                
                # Pattern-insight relationship
                await self.knowledge_graph.add_relation(pattern_node, "reveals", insight_node, {
                    "revelation_confidence": 0.8
                })
                
        except Exception as e:
            logger.warning(f"Knowledge pattern node creation failed: {e}")
    
    async def _create_context_environment_nodes(self, semantic_query: SemanticQuery, workflow_id: str):
        """Create context and environment nodes"""
        try:
            # Session context node
            session_node = f"session_{self.session_id}"
            await self.knowledge_graph.add_concept(session_node, "session_context", {
                "session_id": self.session_id,
                "user_email": self.email,
                "session_start": datetime.now().isoformat(),
                "query_count": 1  # based on current query
            })
            
            # Query-session relationship
            await self.knowledge_graph.add_relation(f"query_{semantic_query.query_id}", "occurs_in_session", session_node, {
                "query_sequence": 1
            })
            
            # Environment node (system environment)
            environment_node = f"environment_{workflow_id}"
            await self.knowledge_graph.add_concept(environment_node, "execution_environment", {
                "system_type": "ontology_multi_agent",
                "execution_mode": "production",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id
            })
            
            # Workflow-environment relationship
            await self.knowledge_graph.add_relation(f"workflow_{workflow_id}", "executes_in", environment_node, {
                "environment_suitability": 0.9
            })
            
        except Exception as e:
            logger.warning(f"Context and environment node creation failed: {e}")
    
    async def _add_domain_concept_relations(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult]):
        """Add domain-specific concept relationships"""
        try:
            # Domain classification by query intent
            intent = semantic_query.intent.lower()
            domain = "general"
            
            if any(keyword in intent for keyword in ["weather", "날씨", "기상"]):
                domain = "weather"
            elif any(keyword in intent for keyword in ["finance", "금융", "환율", "주식"]):
                domain = "finance"
            elif any(keyword in intent for keyword in ["calculation", "계산", "수학"]):
                domain = "calculation"
            elif any(keyword in intent for keyword in ["search", "검색", "정보"]):
                domain = "information"
            
            # Add domain node
            domain_id = f"domain_{domain}"
            await self.knowledge_graph.add_concept(domain_id, "domain", {
                "domain_name": domain,
                "query_count": 1,
                "last_accessed": datetime.now().isoformat()
            })
            
            # Query-domain relationship
            query_id = f"query_{semantic_query.query_id}"
            await self.knowledge_graph.add_relation(
                query_id, "belongs_to_domain", domain_id,
                {"confidence": 0.8}
            )
            
            # Agent-domain relationship
            for result in execution_results:
                if result.is_successful():
                    agent_id = f"agent_{result.agent_id}"
                    await self.knowledge_graph.add_relation(
                        agent_id, "specializes_in_domain", domain_id,
                        {
                            "performance_score": result.confidence,
                            "execution_time": result.execution_time
                        }
                    )
            
        except Exception as e:
            logger.warning(f"Domain concept relationship addition failed: {e}")
    
    async def _add_performance_based_relations(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Add performance-based relationships"""
        try:
            # Identify high-performance agents (confidence >= 0.8)
            high_performance_agents = [
                r for r in execution_results 
                if r.is_successful() and r.confidence >= 0.8
            ]
            
            # "high_performance_collaboration" relationship between high-performance agents
            for i, agent1 in enumerate(high_performance_agents):
                for agent2 in high_performance_agents[i+1:]:
                    await self.knowledge_graph.add_relation(
                        f"agent_{agent1.agent_id}",
                        "high_performance_collaboration",
                        f"agent_{agent2.agent_id}",
                        {
                            "workflow_id": workflow_id,
                            "avg_confidence": (agent1.confidence + agent2.confidence) / 2,
                            "performance_tier": "high"
                        }
                    )
            
            # Identify fast execution agents (under 5s)
            fast_agents = [
                r for r in execution_results 
                if r.is_successful() and r.execution_time <= 5.0
            ]
            
            for agent in fast_agents:
                await self.knowledge_graph.add_concept(
                    f"performance_fast_{agent.agent_id}",
                    "performance_metric",
                    {
                        "agent_id": agent.agent_id,
                        "metric_type": "fast_execution",
                        "execution_time": agent.execution_time,
                        "workflow_id": workflow_id
                    }
                )
            
        except Exception as e:
            logger.warning(f"Performance-based relationship addition failed: {e}")
    
    async def _ensure_basic_ontology_concepts(self):
        """Add basic ontology concepts (when graph is empty)"""
        try:
            # Add basic concepts when graph is empty or has few nodes
            if self.knowledge_graph.graph.number_of_nodes() < 5:
                logger.info("🏗️ Adding basic ontology concepts...")
                
                # Basic agent types
                basic_agents = [
                    ("internet_agent", "Internet Search"),
                    ("calculator_agent", "Calculation Processing"),
                    ("weather_agent", "Weather Information"),
                    ("memo_agent", "Memo Management"),
                    ("calendar_agent", "Calendar Management")
                ]
                
                for agent_id, description in basic_agents:
                    await self.knowledge_graph.add_concept(f"agent_{agent_id}", "agent", {
                        "agent_id": agent_id,
                        "description": description,
                        "type": "basic_agent"
                    })
                
                # Basic domains
                basic_domains = [
                    ("information", "Information Search"),
                    ("calculation", "Calculation Processing"),
                    ("weather", "Weather Information"),
                    ("productivity", "Productivity Tools"),
                    ("general", "General Processing")
                ]
                
                for domain_id, description in basic_domains:
                    await self.knowledge_graph.add_concept(f"domain_{domain_id}", "domain", {
                        "domain_name": domain_id,
                        "description": description,
                        "type": "basic_domain"
                    })
                
                # Basic capabilities
                basic_capabilities = [
                    ("search", "Search Capability"),
                    ("calculate", "Calculation Capability"),
                    ("analyze", "Analysis Capability"),
                    ("generate", "Generation Capability"),
                    ("process", "Processing Capability")
                ]
                
                for capability_id, description in basic_capabilities:
                    await self.knowledge_graph.add_concept(f"capability_{capability_id}", "capability", {
                        "capability_name": capability_id,
                        "description": description,
                        "type": "basic_capability"
                    })
                
                # Add basic relationships
                await self._add_basic_ontology_relations()
                
                logger.info("✅ Basic ontology concepts added")
                
        except Exception as e:
            logger.warning(f"Basic ontology concept addition failed: {e}")
    
    async def _add_basic_ontology_relations(self):
        """Add basic ontology relationships"""
        try:
            # Agent-capability relationship
            agent_capability_map = {
                "internet_agent": ["search", "analyze"],
                "calculator_agent": ["calculate", "process"],
                "weather_agent": ["search", "analyze"],
                "memo_agent": ["generate", "process"],
                "calendar_agent": ["process", "analyze"]
            }
            
            for agent_id, capabilities in agent_capability_map.items():
                for capability in capabilities:
                    await self.knowledge_graph.add_relation(
                        f"agent_{agent_id}",
                        "hasCapability",
                        f"capability_{capability}",
                        {"type": "basic_relation", "confidence": 0.9}
                    )
            
            # Agent-domain relationship
            agent_domain_map = {
                "internet_agent": "information",
                "calculator_agent": "calculation",
                "weather_agent": "weather",
                "memo_agent": "productivity",
                "calendar_agent": "productivity"
            }
            
            for agent_id, domain in agent_domain_map.items():
                await self.knowledge_graph.add_relation(
                    f"agent_{agent_id}",
                    "specializes_in_domain",
                    f"domain_{domain}",
                    {"type": "basic_relation", "confidence": 0.8}
                )
            
            # Domain-capability relationship
            domain_capability_map = {
                "information": ["search", "analyze"],
                "calculation": ["calculate", "process"],
                "weather": ["search", "analyze"],
                "productivity": ["generate", "process", "analyze"]
            }
            
            for domain, capabilities in domain_capability_map.items():
                for capability in capabilities:
                    await self.knowledge_graph.add_relation(
                        f"domain_{domain}",
                        "requires",
                        f"capability_{capability}",
                        {"type": "basic_relation", "confidence": 0.7}
                    )
            
        except Exception as e:
            logger.warning(f"Basic ontology relationship addition failed: {e}")
    
    async def _add_domain_specific_relations(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult]):
        """Add domain-specific specialized relationships"""
        try:
            # Extract domain from query
            query_text = semantic_query.natural_language.lower()
            
            # Domain keyword mapping
            domain_keywords = {
                "weather": ["날씨", "기상", "온도", "습도", "비", "눈", "바람"],
                "finance": ["환율", "주식", "투자", "금융", "돈", "가격", "시세"],
                "technology": ["기술", "컴퓨터", "소프트웨어", "프로그래밍", "AI", "인공지능"],
                "health": ["건강", "의료", "병원", "약", "치료", "진료"],
                "education": ["교육", "학습", "공부", "학교", "대학", "강의"],
                "entertainment": ["게임", "영화", "음악", "스포츠", "오락"]
            }
            
            # Domain detection
            detected_domains = []
            for domain, keywords in domain_keywords.items():
                if any(keyword in query_text for keyword in keywords):
                    detected_domains.append(domain)
            
            # Create domain-specific relationships
            for domain in detected_domains:
                domain_id = f"domain_{domain}"
                await self.knowledge_graph.add_concept(domain_id, "domain", {
                    "name": domain,
                    "keywords": domain_keywords[domain],
                    "detected_in_query": True
                })
                
                # Query-domain relationship
                query_id = f"query_{semantic_query.query_id}"
                await self.knowledge_graph.add_relation(query_id, "belongs_to_domain", domain_id, {
                    "confidence": 0.8,
                    "detection_method": "keyword_matching"
                })
                
                # Relationship between successful agents and domain
                for result in execution_results:
                    if result.success:
                        agent_id = f"agent_{result.agent_id}"
                        await self.knowledge_graph.add_relation(agent_id, "specializes_in_domain", domain_id, {
                            "confidence": result.confidence,
                            "execution_time": result.execution_time
                        })
            
            logger.info(f"🏷️ Domain-specific relationship addition complete: {detected_domains}")
            
        except Exception as e:
            logger.warning(f"Domain-specific relationship addition failed: {e}")
    
    async def _add_temporal_relations(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Add time-based relationships"""
        try:
            # Sort by execution time order
            sorted_results = sorted(execution_results, key=lambda x: x.created_at)
            
            # Add time-based ordering relationships
            for i in range(len(sorted_results) - 1):
                current_result = sorted_results[i]
                next_result = sorted_results[i + 1]
                
                current_agent = f"agent_{current_result.agent_id}"
                next_agent = f"agent_{next_result.agent_id}"
                
                # Temporal predecessor relationship
                await self.knowledge_graph.add_relation(current_agent, "precedes_in_time", next_agent, {
                    "time_gap": (next_result.created_at - current_result.created_at).total_seconds(),
                    "workflow_id": workflow_id,
                    "sequence_order": i + 1
                })
            
            # Performance-based time relationship
            fast_agents = [r for r in execution_results if r.execution_time < 5.0]
            slow_agents = [r for r in execution_results if r.execution_time > 10.0]
            
            # Relationship between fast agents
            for agent in fast_agents:
                agent_id = f"agent_{agent.agent_id}"
                performance_id = f"performance_fast"
                await self.knowledge_graph.add_concept(performance_id, "performance_metric", {
                    "category": "fast_execution",
                    "threshold": 5.0
                })
                await self.knowledge_graph.add_relation(agent_id, "has_performance", performance_id, {
                    "execution_time": agent.execution_time,
                    "category": "fast"
                })
            
            logger.info(f"⏰ Time-based relationship addition complete: {len(execution_results)} agents")
            
        except Exception as e:
            logger.warning(f"Time-based relationship addition failed: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Retrieve system metrics"""
        try:
            return {
                "session_info": {
                    "session_id": self.session_id,
                    "email": self.email,
                    "project_id": self.project_id,
                    "is_initialized": self.is_initialized
                },
                "execution_history": {
                    "total_executions": len(self.execution_history),
                    "recent_executions": self.execution_history[-5:] if self.execution_history else []
                },
                "semantic_query_manager": self.semantic_query_manager.get_metrics(),
                "execution_engine": self.execution_engine.get_metrics(),
                "knowledge_graph": {
                    "metadata": self.knowledge_graph.metadata,
                    "available_agents": self.available_agents
                }
            }
        except Exception as e:
            logger.error(f"System metrics retrieval failed: {e}")
            return {"error": str(e)}
    
    def get_knowledge_graph_visualization(self, max_nodes: int = 50) -> Dict[str, Any]:
        """Retrieve knowledge graph visualization data (first image style)"""
        try:
            logger.info(f"🎨 Ontology knowledge graph visualization requested - max nodes: {max_nodes}")
            
            # Generate default ontology data if graph is empty
            current_nodes = self.knowledge_graph.graph.number_of_nodes()
            if current_nodes == 0:
                logger.info("📦 Empty graph detected, generating basic ontology data...")
                asyncio.create_task(self._create_default_ontology_data())
                # Verify again after generating basic data
                current_nodes = self.knowledge_graph.graph.number_of_nodes()
            
            logger.info(f"📊 Current graph node count: {current_nodes}")
            
            # Generate rich visualization data directly from new knowledge graph engine
            knowledge_graph_visualization = self.knowledge_graph.generate_visualization(max_nodes=max_nodes)
            
            # Return hardcoded visualization data if graph is still empty
            if not knowledge_graph_visualization.get("nodes") and not knowledge_graph_visualization.get("edges"):
                logger.warning("⚠️ Visualization data is empty, returning hardcoded data")
                return self._create_hardcoded_visualization_data()
            
            logger.info(f"✅ Rich ontology graph visualization complete")
            return knowledge_graph_visualization
            
        except Exception as e:
            logger.error(f"Knowledge graph visualization generation failed: {e}")
            return self._create_fallback_visualization(str(e))
    
    def _create_rich_ontology_visualization(self, base_data: Dict[str, Any], max_nodes: int) -> Dict[str, Any]:
        """Generate rich ontology visualization in first image style (based on dynamic data)"""
        try:
            # Extract actual nodes and edges from base data
            base_nodes = base_data.get("nodes", [])
            base_edges = base_data.get("edges", [])
            
            # Extract actual workflow ID and agent info
            workflow_nodes = [node for node in base_nodes if node.get("type") == "workflow"]
            agent_nodes = [node for node in base_nodes if node.get("type") == "agent"]
            task_nodes = [node for node in base_nodes if node.get("type") == "task"]
            
            # Generate central workflow strategy ID (based on actual data)
            if workflow_nodes:
                main_workflow_id = workflow_nodes[0]["id"]
                workflow_strategy = workflow_nodes[0].get("attributes", {}).get("optimization_strategy", "resource_efficiency")
            else:
                main_workflow_id = "workflow_llm_owp_727c54cc5c4243988a66c0b15fd4ccdf"
                workflow_strategy = "resource_efficiency"
            
            # Build Edges array (based on actual data)
            edges = []
            edge_id = 0
            
            # Connect actual agents with workflow
            for i, agent_node in enumerate(agent_nodes[:9]):  # max 9 agents
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": main_workflow_id,
                    "target": agent_node["id"],
                    "color": "#5fd2c9" if i % 2 == 0 else "#fd79a8",
                    "label": f"Agent connection",
                    "type": "Task",
                    "size": 12,
                    "weight": 1,
                    "properties": {
                        "domain": agent_node.get("attributes", {}).get("domain", "general"),
                        "complexity": "medium",
                        "query_related": True,
                        "selected_agent": False,
                        "shape": "circle",
                        "size": 12,
                        "special_type": "standard",
                        "type": "Task",
                        "x": 50 + (i % 5) * 100,
                        "y": 50 + (i // 5) * 100,
                        "agent_type": agent_node.get("attributes", {}).get("agent_id", "unknown"),
                        "execution_time": agent_node.get("attributes", {}).get("execution_time", 0),
                        "confidence": agent_node.get("attributes", {}).get("confidence", 0.8)
                    },
                    "relevance_score": agent_node.get("confidence", 0.8),
                    "confidence": agent_node.get("confidence", 0.8)
                }
                edges.append(edge)
                edge_id += 1
            
            # Connect actual tasks with workflow
            for i, task_node in enumerate(task_nodes[:9]):  # max 9 tasks
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": main_workflow_id,
                    "target": task_node["id"],
                    "color": "#74b9ff",
                    "label": f"Task connection",
                    "type": "TaskExecution",
                    "size": 10,
                    "weight": 0.9,
                    "properties": {
                        "task_description": task_node.get("attributes", {}).get("task_description", ""),
                        "complexity": task_node.get("attributes", {}).get("complexity_level", "medium"),
                        "query_related": True,
                        "task_type": "workflow_task"
                    },
                    "relevance_score": 0.9
                }
                edges.append(edge)
                edge_id += 1
            
            # Convert and add actual base edges
            for base_edge in base_edges[:10]:  # max 10 additional edges
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": base_edge["source"],
                    "target": base_edge["target"],
                    "color": base_edge.get("color", "#96CEB4"),
                    "label": base_edge.get("label", "relationship"),
                    "type": base_edge.get("metadata", {}).get("relationship_type", "relation"),
                    "size": int(base_edge.get("weight", 1) * 8),
                    "weight": base_edge.get("weight", 1),
                    "properties": {
                        "relationship_type": base_edge.get("metadata", {}).get("relationship_type", "general"),
                        "strength": base_edge.get("weight", 1),
                        "attributes": base_edge.get("attributes", {})
                    },
                    "relevance_score": base_edge.get("weight", 0.7)
                }
                edges.append(edge)
                edge_id += 1
            
            # Add inter-agent collaboration relationships (based on actual data)
            collaboration_pairs = []
            for i in range(min(len(agent_nodes), 8)):
                for j in range(i+1, min(len(agent_nodes), i+3)):  # each agent collaborates with up to 2 others
                    if j < len(agent_nodes):
                        collaboration_pairs.append((agent_nodes[i]["id"], agent_nodes[j]["id"]))
            
            collaboration_types = ["collaboration", "data_flow", "parallel", "aggregation", "finalization"]
            for i, (source, target) in enumerate(collaboration_pairs[:5]):
                collab_type = collaboration_types[i % len(collaboration_types)]
                edge = {
                    "id": f"edge_{edge_id}",
                    "source": source,
                    "target": target,
                    "color": "#96CEB4",
                    "label": f"{collab_type} relationship",
                    "type": collab_type,
                    "size": 8,
                    "weight": 0.7,
                    "properties": {
                        "collaboration_type": collab_type,
                        "strength": "medium",
                        "bidirectional": True
                    },
                    "relevance_score": 0.7
                }
                edges.append(edge)
                edge_id += 1
            
            # Calculate statistics based on actual data
            actual_edge_types = {}
            for edge in edges:
                edge_type = edge.get("type", "unknown")
                actual_edge_types[edge_type] = actual_edge_types.get(edge_type, 0) + 1
            
            actual_node_types = {}
            for node in base_nodes:
                node_type = node.get("type", "unknown")
                actual_node_types[node_type] = actual_node_types.get(node_type, 0) + 1
            
            # Generate Metadata (reflecting actual data)
            metadata = {
                "description": f"{workflow_strategy} workflow ontology knowledge graph",
                "edge_types": actual_edge_types,
                "graph_type": "query_focused_workflow",
                "layer_suggestions": {
                    "layer_0": [main_workflow_id, "workflow_strategy"],
                    "layer_1": [node["id"] for node in agent_nodes[:3]],
                    "layer_2": [node["id"] for node in agent_nodes[3:6]], 
                    "layer_3": [node["id"] for node in agent_nodes[6:9]]
                },
                "node_types": actual_node_types,
                "parallelWith": len([e for e in edges if e.get("type") == "parallel"]),
                "partOf": len([e for e in edges if "part" in e.get("type", "").lower()]),
                "requires": len([e for e in edges if "require" in e.get("type", "").lower()]),
                "relevance_states": {
                    "avg_node_relevance": sum(e.get("relevance_score", 0.5) for e in edges) / len(edges) if edges else 0.5,
                    "task_nodes": len(task_nodes),
                    "total_edges": len(edges),
                    "total_nodes": len(base_nodes),
                    "workflow_coverage": (len(workflow_nodes) / len(base_nodes) * 100) if base_nodes else 0,
                    "workflow_nodes": len(workflow_nodes)
                },
                "selected_agents": [node["id"] for node in agent_nodes if node.get("attributes", {}).get("success", True)],
                "styling": {
                    "node_colors": {
                        "agent": "#fd79a8",
                        "workflow": "#fdcb6e", 
                        "task": "#74b9ff",
                        "capability": "#00cec9",
                        "domain": "#a29bfe",
                        "query": "#e17055",
                        "result": "#00b894"
                    },
                    "edge_colors": {
                        "Task": "#5fd2c9",
                        "TaskExecution": "#74b9ff",
                        "collaboration": "#96CEB4",
                        "parallelWith": "#fd79a8",
                        "data_flow": "#00cec9",
                        "aggregation": "#a29bfe"
                    },
                    "node_size_range": {"min": 5, "max": 20},
                    "edge_size_range": {"min": 1, "max": 15}
                },
                "generated_at": datetime.now().isoformat(),
                "graph_type": "ontology_hierarchical", 
                "layout": "force",
                "rendering_hints": {
                    "focus_nodes": [main_workflow_id],
                    "highlight_paths": True,
                    "interactive_zoom": True,
                    "node_labels": True
                },
                "real_data_integration": {
                    "agents_integrated": len(agent_nodes),
                    "tasks_integrated": len(task_nodes),
                    "workflows_integrated": len(workflow_nodes),
                    "total_base_nodes": len(base_nodes),
                    "total_base_edges": len(base_edges)
                }
            }
            
            # Generate Workflow Stats (based on actual data)
            workflow_stats = {
                "avg_workflow_relevance": metadata["relevance_states"]["avg_node_relevance"],
                "task_nodes": len(task_nodes),
                "total_workflow_related": len(workflow_nodes) + len(task_nodes),
                "workflow_coverage": metadata["relevance_states"]["workflow_coverage"],
                "workflow_nodes": len(workflow_nodes),
                "agent_performance": {
                    "total_agents": len(agent_nodes),
                    "successful_agents": len([n for n in agent_nodes if n.get("attributes", {}).get("success", True)]),
                    "avg_confidence": sum(n.get("confidence", 0.5) for n in agent_nodes) / len(agent_nodes) if agent_nodes else 0.5,
                    "avg_execution_time": sum(n.get("attributes", {}).get("execution_time", 0) for n in agent_nodes) / len(agent_nodes) if agent_nodes else 0
                }
            }
            
            # Final structure (same as first image but reflecting actual data)
            knowledge_graph_visualization = {
                "edges": edges,
                "metadata": metadata,
                "workflow_stats": workflow_stats,
                "average_degree": (len(edges) * 2) / len(base_nodes) if base_nodes else 0,
                "is_connected": len(base_nodes) > 0 and len(edges) >= len(base_nodes) - 1,
                "node_types": actual_node_types,
                "total_concepts": len(base_nodes),
                "total_relations": len(edges)
            }
            
            logger.info(f"🎨 Dynamic ontology visualization generated: {len(edges)} edges, {len(base_nodes)} base nodes")
            logger.info(f"📊 Agents: {len(agent_nodes)}, Tasks: {len(task_nodes)}, Workflows: {len(workflow_nodes)}")
            
            return knowledge_graph_visualization
            
        except Exception as e:
            logger.error(f"Rich ontology visualization generation failed: {e}")
            return self._create_fallback_visualization(str(e))
    
    async def _create_default_ontology_data(self):
        """Generate basic ontology data"""
        try:
            logger.info("🏗️ Starting basic ontology data generation")
            
            # 1. Add basic agents
            agents = [
                ("internet_agent", "Internet Search Agent", ["search", "information_retrieval"]),
                ("calculator_agent", "Calculator Agent", ["calculation", "mathematical_operations"]),
                ("weather_agent", "Weather Agent", ["weather_data", "location_services"]),
                ("memo_agent", "Memo Agent", ["text_storage", "note_management"]),
                ("analysis_agent", "Analysis Agent", ["data_analysis", "pattern_recognition"]),
                ("chart_agent", "Chart Agent", ["data_visualization", "chart_generation"])
            ]
            
            for agent_id, description, capabilities in agents:
                await self.knowledge_graph.add_concept(f"agent_{agent_id}", "agent", {
                    "agent_id": agent_id,
                    "description": description,
                    "capabilities": capabilities,
                    "agent_type": "system_agent",
                    "confidence": 0.9,
                    "success": True
                })
            
            # 2. Add basic domains
            domains = [
                ("information", "Information Search Domain"),
                ("calculation", "Calculation Processing Domain"),
                ("weather", "Weather Information Domain"),
                ("productivity", "Productivity Tools Domain"),
                ("analysis", "Data Analysis Domain"),
                ("visualization", "Visualization Domain")
            ]
            
            for domain_id, description in domains:
                await self.knowledge_graph.add_concept(f"domain_{domain_id}", "domain", {
                    "domain_name": domain_id,
                    "description": description,
                    "domain_type": "system_domain"
                })
            
            # 3. Add basic capabilities
            capabilities = [
                ("search", "Search Capability"),
                ("calculate", "Calculation Capability"),
                ("analyze", "Analysis Capability"),
                ("generate", "Generation Capability"),
                ("visualize", "Visualization Capability"),
                ("store", "Storage Capability")
            ]
            
            for capability_id, description in capabilities:
                await self.knowledge_graph.add_concept(f"capability_{capability_id}", "capability", {
                    "capability_name": capability_id,
                    "description": description,
                    "capability_type": "system_capability"
                })
            
            # 4. Add sample workflow
            await self.knowledge_graph.add_concept("workflow_sample", "workflow", {
                "workflow_id": "sample_workflow",
                "description": "Sample workflow",
                "workflow_type": "demo",
                "optimization_strategy": "balanced"
            })
            
            # 5. Add relationships
            relationships = [
                ("agent_internet_agent", "specializes_in_domain", "domain_information"),
                ("agent_calculator_agent", "specializes_in_domain", "domain_calculation"),
                ("agent_weather_agent", "specializes_in_domain", "domain_weather"),
                ("agent_memo_agent", "specializes_in_domain", "domain_productivity"),
                ("agent_analysis_agent", "specializes_in_domain", "domain_analysis"),
                ("agent_chart_agent", "specializes_in_domain", "domain_visualization"),
                
                ("agent_internet_agent", "has_capability", "capability_search"),
                ("agent_calculator_agent", "has_capability", "capability_calculate"),
                ("agent_analysis_agent", "has_capability", "capability_analyze"),
                ("agent_chart_agent", "has_capability", "capability_visualize"),
                ("agent_memo_agent", "has_capability", "capability_store"),
                
                ("workflow_sample", "executes_with", "agent_internet_agent"),
                ("workflow_sample", "executes_with", "agent_analysis_agent"),
                ("workflow_sample", "executes_with", "agent_chart_agent")
            ]
            
            for source, relationship, target in relationships:
                await self.knowledge_graph.add_relationship(source, target, relationship, {
                    "relationship_type": relationship,
                    "confidence": 0.8,
                    "context": "default_ontology"
                })
            
            logger.info(f"✅ Basic ontology data generation complete - nodes: {self.knowledge_graph.graph.number_of_nodes()}")
            
        except Exception as e:
            logger.error(f"Basic ontology data generation failed: {e}")

    def _create_hardcoded_visualization_data(self) -> Dict[str, Any]:
        """Generate hardcoded visualization data (for fallback)"""
        logger.info("🎨 Generating hardcoded visualization data")
        
        # Sample nodes
        nodes = [
            {
                "id": "workflow_main",
                "label": "🔄 Main Workflow",
                "type": "workflow",
                "size": 25,
                "color": "#fdcb6e",
                "special_type": "workflow",
                "properties": {"workflow_type": "main", "optimization_strategy": "balanced"},
                "confidence": 0.9,
                "relevance_score": 1.0,
                "attributes": {"domain": "system", "complexity": "medium"}
            },
            {
                "id": "agent_internet",
                "label": "🌐 Internet Agent",
                "type": "agent",
                "size": 20,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "internet_agent", "capabilities": ["search", "web"]},
                "confidence": 0.85,
                "relevance_score": 0.9,
                "attributes": {"domain": "information", "execution_time": 2.5}
            },
            {
                "id": "agent_calculator",
                "label": "🔢 Calculator Agent",
                "type": "agent",
                "size": 18,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "calculator_agent", "capabilities": ["math", "calculation"]},
                "confidence": 0.88,
                "relevance_score": 0.8,
                "attributes": {"domain": "calculation", "execution_time": 1.2}
            },
            {
                "id": "agent_analysis",
                "label": "📊 Analysis Agent",
                "type": "agent",
                "size": 19,
                "color": "#fd79a8",
                "special_type": "agent",
                "properties": {"agent_id": "analysis_agent", "capabilities": ["analysis", "insights"]},
                "confidence": 0.82,
                "relevance_score": 0.85,
                "attributes": {"domain": "analysis", "execution_time": 3.1}
            },
            {
                "id": "domain_information",
                "label": "📚 Information Domain",
                "type": "domain",
                "size": 16,
                "color": "#a29bfe",
                "special_type": "domain",
                "properties": {"domain_name": "information"},
                "confidence": 0.9,
                "relevance_score": 0.7,
                "attributes": {"domain_type": "knowledge"}
            },
            {
                "id": "capability_search",
                "label": "🔍 Search Capability",
                "type": "capability",
                "size": 14,
                "color": "#00cec9",
                "special_type": "capability",
                "properties": {"capability_name": "search"},
                "confidence": 0.85,
                "relevance_score": 0.75,
                "attributes": {"capability_type": "core"}
            },
            {
                "id": "task_data_collection",
                "label": "📋 Data Collection",
                "type": "task",
                "size": 17,
                "color": "#74b9ff",
                "special_type": "task",
                "properties": {"task_type": "data_collection"},
                "confidence": 0.87,
                "relevance_score": 0.8,
                "attributes": {"complexity": "medium"}
            },
            {
                "id": "result_analysis",
                "label": "📈 Analysis Result",
                "type": "result",
                "size": 15,
                "color": "#00b894",
                "special_type": "result",
                "properties": {"result_type": "analysis"},
                "confidence": 0.83,
                "relevance_score": 0.78,
                "attributes": {"quality": "high"}
            }
        ]
        
        # Sample edges
        edges = [
            {
                "id": "edge_1",
                "source": "workflow_main",
                "target": "agent_internet",
                "label": "Execute",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#5fd2c9",
                "size": 3,
                "weight": 1.0,
                "confidence": 0.9,
                "properties": {"execution_order": 1},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_2",
                "source": "workflow_main",
                "target": "agent_calculator",
                "label": "Execute",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#74b9ff",
                "size": 3,
                "weight": 0.9,
                "confidence": 0.85,
                "properties": {"execution_order": 2},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_3",
                "source": "workflow_main",
                "target": "agent_analysis",
                "label": "Execute",
                "type": "execution",
                "relationship_type": "executes_with",
                "color": "#fd79a8",
                "size": 3,
                "weight": 0.85,
                "confidence": 0.87,
                "properties": {"execution_order": 3},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_4",
                "source": "agent_internet",
                "target": "domain_information",
                "label": "Specialization",
                "type": "specialization",
                "relationship_type": "specializes_in",
                "color": "#96CEB4",
                "size": 2,
                "weight": 0.8,
                "confidence": 0.9,
                "properties": {"specialization_level": "high"},
                "attributes": {"bidirectional": False, "strength": "medium"}
            },
            {
                "id": "edge_5",
                "source": "agent_internet",
                "target": "capability_search",
                "label": "Capability",
                "type": "capability",
                "relationship_type": "has_capability",
                "color": "#00cec9",
                "size": 2,
                "weight": 0.9,
                "confidence": 0.88,
                "properties": {"proficiency": "expert"},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_6",
                "source": "agent_internet",
                "target": "task_data_collection",
                "label": "Produce",
                "type": "production",
                "relationship_type": "produces",
                "color": "#e17055",
                "size": 2,
                "weight": 0.75,
                "confidence": 0.82,
                "properties": {"output_quality": "high"},
                "attributes": {"bidirectional": False, "strength": "medium"}
            },
            {
                "id": "edge_7",
                "source": "agent_analysis",
                "target": "result_analysis",
                "label": "Produce",
                "type": "production",
                "relationship_type": "produces",
                "color": "#00b894",
                "size": 2,
                "weight": 0.88,
                "confidence": 0.85,
                "properties": {"output_quality": "high"},
                "attributes": {"bidirectional": False, "strength": "strong"}
            },
            {
                "id": "edge_8",
                "source": "agent_internet",
                "target": "agent_analysis",
                "label": "Collaborate",
                "type": "collaboration",
                "relationship_type": "collaborated_with",
                "color": "#a29bfe",
                "size": 2,
                "weight": 0.7,
                "confidence": 0.8,
                "properties": {"collaboration_type": "sequential"},
                "attributes": {"bidirectional": True, "strength": "medium"}
            }
        ]
        
        # Metadata
        metadata = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {
                "workflow": 1,
                "agent": 3,
                "domain": 1,
                "capability": 1,
                "task": 1,
                "result": 1
            },
            "edge_types": {
                "execution": 3,
                "specialization": 1,
                "capability": 1,
                "production": 2,
                "collaboration": 1
            },
            "graph_metrics": {
                "density": 0.25,
                "average_degree": 2.0,
                "clustering_coefficient": 0.15,
                "connected_components": 1
            },
            "semantic_layers": {
                "workflow_layer": ["workflow_main"],
                "agent_layer": ["agent_internet", "agent_calculator", "agent_analysis"],
                "domain_layer": ["domain_information"],
                "capability_layer": ["capability_search"],
                "task_layer": ["task_data_collection"]
            },
            "layout_suggestions": {
                "recommended": "hierarchical",
                "type_centers": {
                    "workflow": (0.5, 0.1),
                    "agent": (0.3, 0.5),
                    "domain": (0.7, 0.3),
                    "capability": (0.7, 0.7),
                    "task": (0.3, 0.8)
                },
                "force_settings": {
                    "node_repulsion": 800,
                    "link_strength": 0.6,
                    "charge_strength": -250
                }
            },
            "styling": {
                "node_colors": {
                    "workflow": "#fdcb6e",
                    "agent": "#fd79a8",
                    "domain": "#a29bfe",
                    "capability": "#00cec9",
                    "task": "#74b9ff",
                    "result": "#00b894"
                },
                "edge_colors": {
                    "execution": "#5fd2c9",
                    "specialization": "#96CEB4",
                    "capability": "#00cec9",
                    "production": "#e17055",
                    "collaboration": "#a29bfe"
                },
                "node_size_range": {"min": 10, "max": 30},
                "edge_size_range": {"min": 1, "max": 5}
            },
            "generated_at": datetime.now().isoformat(),
            "version": "2.0",
            "graph_type": "hardcoded_demo",
            "description": "Ontology system demo graph"
        }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata
        }

    def _create_fallback_visualization(self, error_message: str) -> Dict[str, Any]:
        """Generate fallback visualization"""
        return {
            "nodes": [
                {
                    "id": "error_node",
                    "label": "⚠️ Error Occurred",
                    "type": "error",
                    "size": 20,
                    "color": "#E74C3C",
                    "properties": {"error_message": error_message}
                }
            ],
            "edges": [],
            "metadata": {
                "description": f"Ontology graph generation error: {error_message}",
                "error": True,
                "generated_at": datetime.now().isoformat(),
                "graph_type": "error_fallback",
                "total_nodes": 1,
                "total_edges": 0
            }
        }
    
    async def close(self):
        """System shutdown"""
        try:
            logger.info("🔄 Shutting down ontology system...")
            
            # Clear cache
            await self.semantic_query_manager.invalidate_cache()
            
            # Clear execution history (if needed)
            self.execution_history.clear()
            
            logger.info("✅ Ontology system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ System shutdown failed: {e}")
            raise
    
    def _extract_available_agents(self, execution_context: ExecutionContext = None) -> List[str]:
        """Extract available agent list from execution context"""
        try:
            if execution_context and hasattr(execution_context, 'custom_config'):
                installed_agents = execution_context.custom_config.get('installed_agents', [])
                
                if installed_agents:
                    # Convert installed agent info passed from TaskAgent to agent_id list
                    agent_ids = []
                    for agent_info in installed_agents:
                        if isinstance(agent_info, dict):
                            agent_id = agent_info.get('agent_id')
                            if agent_id:
                                agent_ids.append(agent_id)
                                logger.debug(f"Installed agent added: {agent_id}")
                    
                    if agent_ids:
                        logger.info(f"🎯 Found {len(agent_ids)} user-installed agents: {agent_ids}")
                        return agent_ids
                    else:
                        logger.warning("Installed agent info exists but agent_id could not be extracted.")
                else:
                    logger.warning("No installed_agents info in execution_context.")
            else:
                logger.warning("execution_context or custom_config is missing.")
            
            # Fallback: use default agent list
            logger.info("🔄 Falling back to default agent list")
            return self.available_agents
            
        except Exception as e:
            logger.error(f"Installed agent extraction failed: {e}")
            # Fallback: use default agent list
            return self.available_agents
    
    def _extract_installed_agents_info(self, execution_context: ExecutionContext = None) -> List[Dict[str, Any]]:
        """Extract installed agent info from execution context"""
        try:
            if execution_context and hasattr(execution_context, 'custom_config'):
                installed_agents = execution_context.custom_config.get('installed_agents', [])
                
                if installed_agents:
                    logger.info(f"🎯 Extracting installed agent info: {len(installed_agents)}")
                    
                    # Detailed logging of each agent info
                    for i, agent_info in enumerate(installed_agents):
                        agent_id = agent_info.get('agent_id', 'Unknown')
                        agent_data = agent_info.get('agent_data', {})
                        agent_name = agent_data.get('name', agent_id)
                        agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
                        capabilities_count = len(agent_data.get('capabilities', []))
                        tags = agent_data.get('tags', [])
                        
                        logger.info(f"  📋 Agent {i+1}: {agent_id}")
                        logger.info(f"    - Name: {agent_name}")
                        logger.info(f"    - Type: {agent_type}")
                        logger.info(f"    - Capability count: {capabilities_count}")
                        logger.info(f"    - Tags: {tags[:3]}{'...' if len(tags) > 3 else ''}")  # show only first 3
                    
                    return installed_agents
                else:
                    logger.warning("No installed_agents info in execution_context.")
            else:
                logger.warning("execution_context or custom_config is missing.")
            
            return []
            
        except Exception as e:
            logger.error(f"Installed agent info extraction failed: {e}")
            return [] 