"""
🔧 Workflow Utils
Workflow Utilities

Utilities for workflow creation, conversion and analysis
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple
from loguru import logger

from ..core.models import (
    SemanticQuery, ExecutionContext, WorkflowPlan, WorkflowStep, 
    OptimizationStrategy, WorkflowComplexity, ExecutionStrategy
)
from .strategy_manager import StrategyManager
from ..engines.workflow_designer import SmartWorkflowDesigner


class WorkflowUtils:
    """🔧 Workflow Utilities"""
    
    def __init__(self, installed_agents_info: List[Dict[str, Any]] = None):
        self.strategy_manager = StrategyManager()
        self.installed_agents_info = installed_agents_info or []
        self.workflow_designer = None
        self._initialize_workflow_designer()
    
    def _initialize_workflow_designer(self):
        """Initialize workflow designer"""
        try:
            self.workflow_designer = SmartWorkflowDesigner(self.installed_agents_info)
            logger.info(f"🎯 Workflow designer initialized - agent count: {len(self.installed_agents_info)}")
        except Exception as e:
            logger.error(f"Workflow designer initialization failed: {e}")
            self.workflow_designer = SmartWorkflowDesigner()
    
    def convert_unified_result_to_workflow(self, 
                                         unified_result: Dict[str, Any], 
                                         query_text: str, 
                                         execution_context: ExecutionContext) -> WorkflowPlan:
        """Convert unified result to WorkflowPlan"""
        try:
            # Extract workflow information from unified result - new structure
            agent_mappings = unified_result.get('agent_mappings', [])
            task_breakdown = unified_result.get('task_breakdown', [])
            execution_plan = unified_result.get('execution_plan', {})
            
            if not agent_mappings:
                logger.warning("No agent mappings in unified result. Creating minimal workflow")
                return self.create_minimal_workflow(query_text, execution_context)
            
            # Create SemanticQuery (from unified result)
            semantic_info = unified_result.get('semantic_analysis', {})
            semantic_query = SemanticQuery(
                query_id=f'semantic_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent=semantic_info.get('intent', 'information_retrieval'),
                entities=semantic_info.get('entities', []),
                concepts=semantic_info.get('concepts', []),
                relations=semantic_info.get('relations', []),
                complexity_score=semantic_info.get('complexity_score', 0.7),
                created_at=datetime.now(),
                metadata=semantic_info.get('metadata', {})
            )
            
            # Convert workflow steps - from new structure
            steps = []
            for mapping in agent_mappings:
                # Find the relevant task information
                task_info = next((t for t in task_breakdown if t.get('task_id') == mapping.get('task_id')), {})
                
                step = WorkflowStep(
                    step_id=mapping.get('task_id', f'step_{len(steps)+1}'),
                    agent_id=mapping.get('selected_agent', 'unknown_agent'),
                    semantic_purpose=task_info.get('task_description', mapping.get('individual_query', 'Execute task')),
                    required_concepts=task_info.get('extracted_keywords', []),
                    depends_on=task_info.get('depends_on', []),
                    estimated_time=30.0,
                    estimated_complexity=WorkflowComplexity.MODERATE,
                    execution_context={
                        "query": mapping.get('individual_query', query_text),
                        "expected_output": mapping.get('expected_output', ''),
                        "context_integration": mapping.get('context_integration', ''),
                        "confidence": mapping.get('confidence', 0.8)
                    }
                )
                steps.append(step)
            
            # Determine execution strategy - from new structure
            strategy_map = {
                'single_agent': OptimizationStrategy.SPEED_FIRST,  # single agent prioritizes speed
                'parallel': OptimizationStrategy.BALANCED,
                'sequential': OptimizationStrategy.QUALITY_FIRST,
                'hybrid': OptimizationStrategy.BALANCED
            }
            
            strategy_name = execution_plan.get('strategy', 'parallel')
            optimization_strategy = strategy_map.get(strategy_name, OptimizationStrategy.BALANCED)
            
            # Handle execution order and dependencies
            execution_order = execution_plan.get('execution_order', [])
            if not execution_order and steps:
                # Generate default execution order
                execution_order = [[step.step_id] for step in steps]
            
            # Create WorkflowPlan
            workflow_plan = WorkflowPlan(
                plan_id=f'workflow_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                query=semantic_query,
                steps=steps,
                optimization_strategy=optimization_strategy,
                estimated_time=execution_plan.get('estimated_time', 30),
                estimated_quality=0.8,
                dependencies={},
                reasoning_chain=[
                    unified_result.get('query_analysis', {}).get('reasoning', ''),
                    execution_plan.get('reasoning', '')
                ]
            )
            
            logger.info(f"✅ Unified result converted to workflow: {len(steps)} steps")
            return workflow_plan
            
        except Exception as e:
            logger.error(f"❌ Workflow conversion failed: {e}")
            return self.create_minimal_workflow(query_text, execution_context)

    def create_minimal_workflow(self, query_text: str, execution_context: ExecutionContext) -> WorkflowPlan:
        """Create minimal workflow (fallback) - select suitable agent for query"""
        # List of available agents
        available_agents = list(execution_context.available_agents.keys()) if execution_context.available_agents else ['memo_agent']
        
        # Select agent suitable for the query
        selected_agent = self._select_best_fallback_agent(query_text, available_agents, execution_context)
        if not selected_agent:
            selected_agent = available_agents[0] if available_agents else 'memo_agent'
        
        step = WorkflowStep(
            step_id="minimal_step",
            agent_id=selected_agent,
            semantic_purpose=query_text,
            required_concepts=[],  # required parameter
            depends_on=[],
            estimated_time=30.0,
            estimated_complexity=WorkflowComplexity.MODERATE,
            execution_context={"fallback_mode": True, "timeout": 120, "retry_count": 2}
        )
        
        workflow_plan = WorkflowPlan(
            plan_id=f"minimal_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query=SemanticQuery(
                query_id=f'minimal_{uuid.uuid4().hex[:8]}',
                query_text=query_text,
                natural_language=query_text,
                intent='information_retrieval',
                entities=[],
                concepts=[],
                relations=[],
                complexity_score=0.5,
                created_at=datetime.now(),
                metadata={"fallback_mode": True}
            ),
            steps=[step],
            optimization_strategy=OptimizationStrategy.SPEED_FIRST,  # single agent prioritizes speed
            estimated_time=30,
            estimated_quality=0.7,
            dependencies={},
            reasoning_chain=["Minimal fallback workflow", "Using single agent", "Speed-first processing"]
        )
        
        logger.warning("🔄 Creating minimal fallback workflow")
        return workflow_plan

    def safe_analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Safe complexity analysis"""
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
    
    def generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
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
                
                # Node definition - wrap text in double quotes
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

    def extract_available_agents(self, execution_context: ExecutionContext = None) -> List[str]:
        """Extract list of available agents"""
        try:
            if not execution_context or not execution_context.available_agents:
                # ⚠️ No agents in ExecutionContext - using fallback
                logger.warning("⚠️ extract_available_agents: ExecutionContext.available_agents is empty!")
                logger.warning("⚠️ Falling back to hardcoded default agents")
                return [
                    "internet_agent", "finance_agent", "weather_agent",
                    "calculate_agent", "chart_agent", "memo_agent", "analysis_agent"
                ]

            logger.info(f"✅ extract_available_agents: Found {len(execution_context.available_agents)} agents from ExecutionContext")
            
            available_agents = []
            for agent_id, agent_info in execution_context.available_agents.items():
                if isinstance(agent_info, dict) and agent_info.get('available', True):
                    available_agents.append(agent_id)
                elif hasattr(agent_info, 'agent_id'):
                    available_agents.append(agent_info.agent_id)
                else:
                    available_agents.append(agent_id)
            
            logger.info(f"🤖 Available agents extracted: {len(available_agents)}")
            return available_agents
            
        except Exception as e:
            logger.error(f"Agent list extraction failed: {e}")
            return ["memo_agent"]  # safe fallback
    
    def extract_installed_agents_info(self, execution_context: ExecutionContext = None) -> List[Dict[str, Any]]:
        """Extract installed agent info - with enhanced metadata"""
        try:
            if not execution_context or not execution_context.available_agents:
                logger.warning("ExecutionContext or available_agents is missing")
                return []
            
            # Use agent metadata extractor
            try:
                from ..core.agent_metadata_extractor import get_agent_metadata_extractor
                extractor = get_agent_metadata_extractor()
                use_enhanced_extraction = True
            except ImportError:
                logger.warning("AgentMetadataExtractor unavailable, using basic extraction")
                use_enhanced_extraction = False
            
            installed_agents_info = []
            
            for agent_id, agent_info in execution_context.available_agents.items():
                try:
                    if isinstance(agent_info, dict):
                        if use_enhanced_extraction:
                            # Enhanced metadata extraction
                            agent_data = extractor.extract_agent_metadata({
                                'agent_id': agent_id,
                                'agent_data': agent_info
                            })
                        else:
                            # Basic extraction
                            agent_data = {
                                "agent_id": agent_id,
                                "name": agent_info.get('name', agent_id),
                                "description": agent_info.get('description', f'{agent_id} agent'),
                                "capabilities": agent_info.get('capabilities', []),
                                "available": agent_info.get('available', True),
                                "metadata": agent_info.get('metadata', {}),
                                "tags": agent_info.get('tags', [])
                            }
                    elif hasattr(agent_info, '__dict__'):
                        # Agent info in object form
                        agent_data = {
                            "agent_id": getattr(agent_info, 'agent_id', agent_id),
                            "name": getattr(agent_info, 'name', agent_id),
                            "description": getattr(agent_info, 'description', f'{agent_id} agent'),
                            "capabilities": getattr(agent_info, 'capabilities', []),
                            "available": getattr(agent_info, 'available', True),
                            "metadata": getattr(agent_info, 'metadata', {})
                        }
                    else:
                        # Only basic info available
                        agent_data = {
                            "agent_id": agent_id,
                            "name": agent_id,
                            "description": f'{agent_id} agent',
                            "capabilities": [],
                            "available": True,
                            "metadata": {}
                        }
                    
                    installed_agents_info.append(agent_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process agent {agent_id} info: {e}")
                    # Include at least minimal info
                    installed_agents_info.append({
                        "agent_id": agent_id,
                        "name": agent_id,
                        "description": f'{agent_id} agent',
                        "capabilities": [],
                        "available": True,
                        "metadata": {}
                    })
            
            logger.info(f"📊 Installed agent info extraction complete: {len(installed_agents_info)}")
            return installed_agents_info
            
        except Exception as e:
            logger.error(f"Installed agent info extraction failed: {e}")
            return []
    
    def _select_best_fallback_agent(self, query_text: str, available_agents: List[str], 
                                   execution_context: ExecutionContext) -> str:
        """Select best fallback agent for the query"""
        try:
            query_lower = query_text.lower()
            
            # Query keyword-based agent mapping
            agent_patterns = {
                # Visualization-related
                'chart_agent': ['플로우차트', '차트', '그래프', '표', '다이어그램', 'flowchart', 'chart', 'graph', 'diagram'],
                # Travel-related  
                'travel_agent': ['여행', '일정', '계획', 'travel', 'trip', 'schedule', 'plan'],
                # Weather-related
                'weather_agent': ['날씨', '기상', '온도', 'weather', 'temperature'],
                # Finance-related
                'finance_agent': ['주식', '환율', '금융', 'stock', 'finance', 'currency'],
                # Calculation-related (lowest priority)
                'calculator_agent': ['계산', '수식', '덧셈', '뺄셈', 'calculate', 'math', 'compute'],
                # Search-related
                'internet_agent': ['검색', '조회', '찾아', 'search', 'find', 'lookup'],
                # Analysis-related
                'analysis_agent': ['분석', '비교', '평가', 'analysis', 'analyze', 'compare']
            }
            
            # Calculate score by priority
            agent_scores = {}
            
            for agent_id in available_agents:
                score = 0
                agent_base = agent_id.replace('_agent', '')
                
                # Check exact agent name match
                if agent_id in agent_patterns:
                    patterns = agent_patterns[agent_id]
                    for pattern in patterns:
                        if pattern in query_lower:
                            score += 10  # High score for exact pattern match
                
                # Also check by agent base name
                elif agent_base in query_lower:
                    score += 5
                
                # Special handling by domain
                if 'chart' in agent_id or 'visual' in agent_id:
                    if any(keyword in query_lower for keyword in ['플로우차트', '차트', '시각화', 'flowchart', 'visual']):
                        score += 15  # Prioritize for visualization requests
                
                if 'travel' in agent_id or 'plan' in agent_id:
                    if any(keyword in query_lower for keyword in ['여행', '일정', '계획', 'travel', 'plan']):
                        score += 12
                
                # Deduct score for calculator agent if not clearly a calculation request
                if 'calculator' in agent_id or 'calc' in agent_id:
                    if not any(keyword in query_lower for keyword in ['계산', '수식', 'calculate', 'math']):
                        if any(keyword in query_lower for keyword in ['플로우차트', '여행', '일정', 'flowchart', 'travel']):
                            score -= 10  # Deduct for clearly non-calculation tasks
                
                agent_scores[agent_id] = score
            
            # Select agent with highest score
            if agent_scores:
                best_agent = max(agent_scores.items(), key=lambda x: x[1])
                if best_agent[1] > 0:  # Only if positive score
                    logger.info(f"🎯 Fallback agent selected: {best_agent[0]} (score: {best_agent[1]})")
                    return best_agent[0]
            
            # If all scores are 0 or below, select first non-calculator agent
            non_calc_agents = [agent for agent in available_agents 
                             if 'calculator' not in agent and 'calc' not in agent]
            if non_calc_agents:
                selected = non_calc_agents[0]
                logger.info(f"🔄 Default fallback agent selected: {selected}")
                return selected
            
            # If still none, select first agent
            return available_agents[0] if available_agents else 'memo_agent'
            
        except Exception as e:
            logger.error(f"Fallback agent selection failed: {e}")
            # Return a safe non-calculator agent on error
            safe_agents = [agent for agent in available_agents 
                          if 'calculator' not in agent and 'calc' not in agent]
            return safe_agents[0] if safe_agents else (available_agents[0] if available_agents else 'memo_agent')

    def update_installed_agents_info(self, installed_agents_info: List[Dict[str, Any]]):
        """Update installed agent information"""
        self.installed_agents_info = installed_agents_info
        self._initialize_workflow_designer()
        logger.info(f"🔄 Agent info update: {len(installed_agents_info)}") 