"""
🎯 Workflow Designer

Designs optimal workflows based on SemanticQuery.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import time
import uuid

from ..core.models import (
    SemanticQuery, WorkflowPlan, WorkflowStep, 
    ExecutionStrategy, OptimizationStrategy, WorkflowComplexity
)
from ..core.interfaces import WorkflowDesigner as IWorkflowDesigner


class SmartWorkflowDesigner(IWorkflowDesigner):
    """🎯 Smart Workflow Designer"""

    def __init__(self, installed_agents_info: List[Dict[str, Any]] = None):
        # Store information about actually installed agents
        self.installed_agents_info = installed_agents_info or []
        self.agents_capabilities_cache = {}  # Agent capability cache

        # Default agent capability templates (fallback)
        self.agent_capability_templates = {
            "internet_agent": {
                "domains": ["web", "search", "information"],
                "capabilities": ["search", "web_scraping", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 15.0
            },
            "finance_agent": {
                "domains": ["finance", "stock", "currency"],
                "capabilities": ["financial_data", "market_analysis"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0
            },
            "weather_agent": {
                "domains": ["weather", "climate"],
                "capabilities": ["weather_forecast", "climate_data"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 10.0
            },
            "calculate_agent": {
                "domains": ["math", "calculation"],
                "capabilities": ["arithmetic", "computation"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 5.0
            },
            "chart_agent": {
                "domains": ["visualization", "chart"],
                "capabilities": ["chart_creation", "data_visualization"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 25.0
            },
            "memo_agent": {
                "domains": ["memory", "storage"],
                "capabilities": ["data_storage", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 8.0
            },
            "analysis_agent": {
                "domains": ["analysis", "research"],
                "capabilities": ["data_analysis", "research"],
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_time": 30.0
            }
        }
        
        # Inter-agent dependency rules (default templates)
        self.dependency_rules = {
            "chart_agent": ["internet_agent", "finance_agent", "calculate_agent"],
            "analysis_agent": ["internet_agent", "finance_agent"],
            "memo_agent": ["internet_agent", "analysis_agent", "chart_agent"],
            "calculate_agent": ["internet_agent", "finance_agent"]
        }
        
        # Build capability cache based on installed agent information
        self._build_capabilities_cache()

        logger.info(f"🎯 SmartWorkflowDesigner initialized - installed agents: {len(self.installed_agents_info)}")
    
    def _build_capabilities_cache(self):
        """Build capability cache based on installed agent information"""
        try:
            logger.info(f"🔧 Agent capability cache build started - total {len(self.installed_agents_info)} agents")

            for i, agent_info in enumerate(self.installed_agents_info):
                agent_id = agent_info.get('agent_id', '')
                if not agent_id:
                    logger.warning(f"Agent {i+1}: agent_id is missing.")
                    continue

                # Extract capability information from agent data
                agent_data = agent_info.get('agent_data', {})
                capabilities_info = self._extract_capabilities_from_agent_data(agent_id, agent_data)

                self.agents_capabilities_cache[agent_id] = capabilities_info

                # Detailed logging
                logger.info(f"  🤖 Agent {i+1}: {agent_id}")
                logger.info(f"    - Name: {capabilities_info.get('name', agent_id)}")
                logger.info(f"    - Domains: {capabilities_info.get('domains', [])}")
                logger.info(f"    - Capabilities: {len(capabilities_info.get('capabilities', []))}")
                logger.info(f"    - Complexity: {capabilities_info.get('complexity', 'UNKNOWN')}")
                logger.info(f"    - Estimated time: {capabilities_info.get('estimated_time', 0)}s")

                logger.debug(f"Agent {agent_id} capability info cached: {capabilities_info}")

            logger.info(f"✅ Agent capability cache built - {len(self.agents_capabilities_cache)} agents")

        except Exception as e:
            logger.error(f"Capability cache build failed: {e}")
            logger.error(f"Installed agent information: {self.installed_agents_info}")
    
    def _extract_capabilities_from_agent_data(self, agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract capability information from agent data"""
        try:
            logger.debug(f"🔍 Agent {agent_id} capability extraction started")

            # Base information
            capabilities_info = {
                "domains": [],
                "capabilities": [],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0,
                "description": agent_data.get('description', ''),
                "name": agent_data.get('name', agent_id)
            }
            
            # Extract information from capabilities
            if 'capabilities' in agent_data:
                logger.debug(f"  Capability extraction: {len(agent_data['capabilities'])} entries")
                for capability in agent_data['capabilities']:
                    cap_id = capability.get('id', '')
                    cap_name = capability.get('name', '')
                    cap_desc = capability.get('description', '')

                    capabilities_info["capabilities"].append(cap_id)

                    # Infer domain
                    domain = self._infer_domain_from_capability(cap_id, cap_name, cap_desc)
                    if domain and domain not in capabilities_info["domains"]:
                        capabilities_info["domains"].append(domain)
                        logger.debug(f"    Domain added: {domain} (from {cap_id})")

            # Extract domain information from tags
            if 'tags' in agent_data:
                logger.debug(f"  Tag extraction: {len(agent_data['tags'])} entries")
                for tag in agent_data['tags']:
                    domain = self._infer_domain_from_tag(tag)
                    if domain and domain not in capabilities_info["domains"]:
                        capabilities_info["domains"].append(domain)
                        logger.debug(f"    Domain added: {domain} (from tag: {tag})")

            # Extract agent type from metadata
            metadata = agent_data.get('metadata', {})
            agent_type = metadata.get('agent_type', '')
            if agent_type:
                type_info = self._get_type_based_info(agent_type)
                capabilities_info.update(type_info)
                logger.debug(f"  Type-based info applied: {agent_type}")

            # Estimate complexity and time
            complexity, estimated_time = self._estimate_complexity_and_time(agent_id, agent_data)
            capabilities_info["complexity"] = complexity
            capabilities_info["estimated_time"] = estimated_time

            # Infer domains from agent ID if domains are empty
            if not capabilities_info["domains"]:
                capabilities_info["domains"] = self._infer_domains_from_agent_id(agent_id)
                logger.debug(f"  ID-based domain inference: {capabilities_info['domains']}")

            # Set default capability if capabilities are empty
            if not capabilities_info["capabilities"]:
                capabilities_info["capabilities"] = ["general_processing"]
                logger.debug(f"  Default capability set: general_processing")

            logger.debug(f"✅ Agent {agent_id} capability extraction completed")
            return capabilities_info

        except Exception as e:
            logger.error(f"Agent {agent_id} capability extraction failed: {e}")
            return self._get_fallback_capabilities(agent_id)
    
    def _infer_domain_from_capability(self, cap_id: str, cap_name: str, cap_desc: str) -> str:
        """Infer domain from capability information"""
        text = f"{cap_id} {cap_name} {cap_desc}".lower()
        
        if any(keyword in text for keyword in ["search", "internet", "web", "scraping"]):
            return "web"
        elif any(keyword in text for keyword in ["memo", "note", "storage", "save"]):
            return "memory"
        elif any(keyword in text for keyword in ["schedule", "calendar", "time", "event"]):
            return "scheduling"
        elif any(keyword in text for keyword in ["restaurant", "food", "place", "location"]):
            return "location"
        elif any(keyword in text for keyword in ["calculate", "math", "arithmetic", "compute"]):
            return "math"
        elif any(keyword in text for keyword in ["currency", "exchange", "money", "finance"]):
            return "finance"
        elif any(keyword in text for keyword in ["fortune", "luck", "zodiac", "prediction"]):
            return "entertainment"
        elif any(keyword in text for keyword in ["chart", "graph", "visual", "plot"]):
            return "visualization"
        else:
            return "general"
    
    def _infer_domain_from_tag(self, tag: str) -> str:
        """Infer domain from tag"""
        tag_lower = tag.lower()

        if any(keyword in tag_lower for keyword in ["인터넷", "검색", "웹", "internet", "search", "web"]):
            return "web"
        elif any(keyword in tag_lower for keyword in ["메모", "노트", "저장", "memo", "note", "storage"]):
            return "memory"
        elif any(keyword in tag_lower for keyword in ["일정", "캘린더", "스케줄", "schedule", "calendar"]):
            return "scheduling"
        elif any(keyword in tag_lower for keyword in ["맛집", "음식", "레스토랑", "restaurant", "food"]):
            return "location"
        elif any(keyword in tag_lower for keyword in ["계산", "수학", "math", "calculate"]):
            return "math"
        elif any(keyword in tag_lower for keyword in ["환율", "통화", "금융", "currency", "finance"]):
            return "finance"
        elif any(keyword in tag_lower for keyword in ["운세", "점", "fortune", "luck"]):
            return "entertainment"
        elif any(keyword in tag_lower for keyword in ["차트", "그래프", "시각화", "chart", "graph"]):
            return "visualization"
        else:
            return "general"
    
    def _get_type_based_info(self, agent_type: str) -> Dict[str, Any]:
        """Return information based on agent type"""
        type_mapping = {
            "INTERNET_SEARCH": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 15.0
            },
            "MEMO": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 8.0
            },
            "SCHEDULER": {
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 12.0
            },
            "RESTAURANT_FINDER": {
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 18.0
            },
            "CALCULATOR": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 5.0
            },
            "CUSTOM": {
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 15.0
            },
            "FORECASTING": {
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 10.0
            }
        }
        
        return type_mapping.get(agent_type, {
            "complexity": WorkflowComplexity.MODERATE,
            "estimated_time": 20.0
        })
    
    def _estimate_complexity_and_time(self, agent_id: str, agent_data: Dict[str, Any]) -> Tuple[WorkflowComplexity, float]:
        """Estimate complexity and expected time"""
        try:
            # Check type from metadata
            metadata = agent_data.get('metadata', {})
            agent_type = metadata.get('agent_type', '')

            type_info = self._get_type_based_info(agent_type)
            complexity = type_info.get('complexity', WorkflowComplexity.MODERATE)
            estimated_time = type_info.get('estimated_time', 20.0)

            # Adjust based on number of capabilities
            capabilities_count = len(agent_data.get('capabilities', []))
            if capabilities_count > 5:
                if complexity == WorkflowComplexity.SIMPLE:
                    complexity = WorkflowComplexity.MODERATE
                estimated_time *= 1.2
            elif capabilities_count > 10:
                if complexity == WorkflowComplexity.MODERATE:
                    complexity = WorkflowComplexity.COMPLEX
                estimated_time *= 1.5
            
            return complexity, estimated_time
            
        except Exception as e:
            logger.error(f"Complexity estimation failed: {e}")
            return WorkflowComplexity.MODERATE, 20.0
    
    def _infer_domains_from_agent_id(self, agent_id: str) -> List[str]:
        """Infer domains from agent ID"""
        agent_id_lower = agent_id.lower()
        domains = []
        
        if any(keyword in agent_id_lower for keyword in ["internet", "web", "search"]):
            domains.append("web")
        if any(keyword in agent_id_lower for keyword in ["memo", "note", "storage"]):
            domains.append("memory")
        if any(keyword in agent_id_lower for keyword in ["schedule", "calendar"]):
            domains.append("scheduling")
        if any(keyword in agent_id_lower for keyword in ["restaurant", "finder", "location"]):
            domains.append("location")
        if any(keyword in agent_id_lower for keyword in ["calculator", "calc", "math"]):
            domains.append("math")
        if any(keyword in agent_id_lower for keyword in ["currency", "exchange", "finance"]):
            domains.append("finance")
        if any(keyword in agent_id_lower for keyword in ["fortune", "daily"]):
            domains.append("entertainment")
        
        return domains if domains else ["general"]
    
    def _get_fallback_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Return fallback capability information"""
        return {
            "domains": self._infer_domains_from_agent_id(agent_id),
            "capabilities": ["general_processing"],
            "complexity": WorkflowComplexity.MODERATE,
            "estimated_time": 20.0,
            "description": f"General agent: {agent_id}",
            "name": agent_id
        }
    
    def _get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Look up agent capability information (based on actually installed agent data)"""
        try:
            # Check cache first
            if agent_id in self.agents_capabilities_cache:
                return self.agents_capabilities_cache[agent_id].copy()

            # Look up in installed agent information
            for agent_info in self.installed_agents_info:
                if agent_info.get('agent_id') == agent_id:
                    agent_data = agent_info.get('agent_data', {})
                    capabilities = self._extract_capabilities_from_agent_data(agent_id, agent_data)

                    # Store in cache
                    self.agents_capabilities_cache[agent_id] = capabilities
                    logger.info(f"🔍 Agent {agent_id} actual capability lookup: {capabilities}")
                    return capabilities.copy()

            # Look up in templates (fallback)
            if agent_id in self.agent_capability_templates:
                logger.info(f"🔍 Agent {agent_id} using template capability info")
                return self.agent_capability_templates[agent_id].copy()

            # Last resort: ID-based inference
            inferred_capabilities = self._infer_agent_capabilities(agent_id)
            logger.info(f"🔍 Agent {agent_id} capability inferred: {inferred_capabilities}")
            return inferred_capabilities

        except Exception as e:
            logger.error(f"Agent {agent_id} capability lookup failed: {e}")
            return self._get_fallback_capabilities(agent_id)
    
    def _infer_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Infer capabilities based on agent ID"""
        agent_id_lower = agent_id.lower()

        # Keyword-based inference
        if any(keyword in agent_id_lower for keyword in ["internet", "web", "search"]):
            return {
                "domains": ["web", "search", "information"],
                "capabilities": ["search", "web_scraping", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 15.0
            }
        elif any(keyword in agent_id_lower for keyword in ["finance", "stock", "money"]):
            return {
                "domains": ["finance", "stock", "currency"],
                "capabilities": ["financial_data", "market_analysis"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0
            }
        elif any(keyword in agent_id_lower for keyword in ["weather", "climate"]):
            return {
                "domains": ["weather", "climate"],
                "capabilities": ["weather_forecast", "climate_data"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 10.0
            }
        elif any(keyword in agent_id_lower for keyword in ["calc", "math", "compute"]):
            return {
                "domains": ["math", "calculation"],
                "capabilities": ["arithmetic", "computation"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 5.0
            }
        elif any(keyword in agent_id_lower for keyword in ["chart", "graph", "visual"]):
            return {
                "domains": ["visualization", "chart"],
                "capabilities": ["chart_creation", "data_visualization"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 25.0
            }
        elif any(keyword in agent_id_lower for keyword in ["memo", "note", "storage"]):
            return {
                "domains": ["memory", "storage"],
                "capabilities": ["data_storage", "information_retrieval"],
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_time": 8.0
            }
        elif any(keyword in agent_id_lower for keyword in ["analysis", "analyze", "research"]):
            return {
                "domains": ["analysis", "research"],
                "capabilities": ["data_analysis", "research"],
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_time": 30.0
            }
        else:
            # Default value
            return {
                "domains": ["general"],
                "capabilities": ["general_processing"],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 20.0
            }
    
    async def design_workflow(self,
                            semantic_query: SemanticQuery,
                            available_agents: List[str]) -> WorkflowPlan:
        """
        Design a workflow

        Args:
            semantic_query: Semantic query
            available_agents: List of available agents (user-installed agents)

        Returns:
            Designed workflow plan
        """
        try:
            logger.info(f"🎯 Workflow design started - query: {semantic_query.natural_language[:100]}...")
            logger.info(f"🎯 Available agents: {available_agents}")

            if not available_agents:
                logger.warning("No available agents. Creating default workflow.")
                return self._create_fallback_workflow(semantic_query, [])

            # 1. Select required agents
            required_agents = self._select_required_agents(semantic_query, available_agents)
            logger.info(f"Selected agents: {required_agents}")

            if not required_agents:
                logger.warning("No agents selected. Using first available agent.")
                required_agents = [available_agents[0]]

            # 2. Create workflow steps
            workflow_steps = self._create_workflow_steps(semantic_query, required_agents)

            # 3. Build execution graph
            execution_graph = self._build_execution_graph(workflow_steps)

            # 4. Determine optimization strategy
            optimization_strategy = self._determine_optimization_strategy(semantic_query, workflow_steps)

            # 5. Estimate metrics
            estimated_quality, estimated_time = self._estimate_workflow_metrics(workflow_steps)

            # 6. Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(
                semantic_query, workflow_steps, optimization_strategy
            )

            # 7. Create workflow plan
            workflow_plan = WorkflowPlan.create_simple(
                plan_id=f"workflow_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                semantic_query=semantic_query,
                steps=workflow_steps,
                execution_graph=execution_graph,
                strategy=optimization_strategy,  # strategy parameter
                quality=estimated_quality,
                time=estimated_time,
                reasoning=reasoning_chain
            )

            logger.info(f"✅ Workflow design completed - {len(workflow_steps)} steps, estimated time: {estimated_time:.1f}s")
            return workflow_plan

        except Exception as e:
            logger.error(f"❌ Workflow design failed: {e}")
            return self._create_fallback_workflow(semantic_query, available_agents)
    
    def optimize_workflow(self, workflow_plan: WorkflowPlan) -> WorkflowPlan:
        """Optimize workflow"""
        try:
            logger.info("🔧 Workflow optimization started")

            # 1. Analyze parallel processing opportunities
            parallel_groups = self._identify_parallel_groups(workflow_plan.steps)

            # 2. Remove duplicate steps
            optimized_steps = self._remove_duplicate_steps(workflow_plan.steps)

            # 3. Optimize dependencies
            optimized_graph = self._optimize_dependencies(workflow_plan.execution_graph)

            # 4. Re-evaluate execution strategy
            optimized_strategy = self._reevaluate_strategy(optimized_steps, parallel_groups)

            # Create optimized plan
            optimized_plan = WorkflowPlan.create_simple(
                semantic_query=workflow_plan.semantic_query,
                steps=optimized_steps,
                strategy=optimized_strategy,
                quality=workflow_plan.estimated_quality * 1.1,  # quality improved by optimization
                time=workflow_plan.estimated_time * 0.9,  # time reduced by optimization
                reasoning=workflow_plan.reasoning_chain + ["Workflow optimization applied"]
            )

            optimized_plan.execution_graph = optimized_graph

            logger.info(f"✅ Workflow optimization completed: {len(optimized_steps)} steps")
            return optimized_plan

        except Exception as e:
            logger.error(f"❌ Workflow optimization failed: {e}")
            return workflow_plan
    
    def validate_workflow(self, workflow_plan: WorkflowPlan) -> bool:
        """Validate workflow (abstract method implementation)"""
        try:
            # 1. Validate basic structure
            if not workflow_plan or not workflow_plan.steps:
                logger.warning("Workflow plan is empty.")
                return False

            # 2. Validate each step
            for step in workflow_plan.steps:
                if not step.step_id or not step.agent_id:
                    logger.warning(f"Step {step.step_id} is missing required information.")
                    return False

                # Check whether agent capability lookup is possible (dynamic lookup)
                try:
                    agent_info = self._get_agent_capabilities(step.agent_id)
                    if not agent_info:
                        logger.warning(f"Cannot look up capability information for agent {step.agent_id}.")
                        return False
                except Exception as e:
                    logger.warning(f"Agent {step.agent_id} capability lookup failed: {e}")
                    return False

            # 3. Validate dependencies
            if hasattr(workflow_plan, 'execution_graph') and workflow_plan.execution_graph:
                # Check for cyclic references
                if not nx.is_directed_acyclic_graph(workflow_plan.execution_graph):
                    logger.warning("Workflow has cyclic references.")
                    return False

                # Verify all steps are included in the graph
                step_ids = {step.step_id for step in workflow_plan.steps}
                graph_nodes = set(workflow_plan.execution_graph.nodes())
                if step_ids != graph_nodes:
                    logger.warning("Execution graph and step list do not match.")
                    return False
            
            # 4. Validate dependency rules (dynamically)
            for step in workflow_plan.steps:
                if step.agent_id in self.dependency_rules:
                    required_deps = self.dependency_rules[step.agent_id]
                    available_agents = [s.agent_id for s in workflow_plan.steps]

                    # Check if required dependencies are present
                    has_dependency = any(dep in available_agents for dep in required_deps)
                    if not has_dependency and len(workflow_plan.steps) > 1:
                        logger.warning(f"Dependencies for agent {step.agent_id} are not satisfied.")
                        # Only warn, do not treat as failure

            # 5. Validate time estimate
            if hasattr(workflow_plan, 'estimated_time') and workflow_plan.estimated_time <= 0:
                logger.warning("Workflow estimated time is invalid.")
                return False

            logger.info("✅ Workflow validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Workflow validation failed: {e}")
            return False
    
    def _select_required_agents(self, semantic_query: SemanticQuery, available_agents: List[str]) -> List[str]:
        """Select required agents - multi-agent support"""
        required_agents = []
        query_text = semantic_query.natural_language.lower()

        logger.info(f"🎯 Agent selection started - query: '{semantic_query.natural_language}'")
        logger.info(f"🎯 Available agents: {available_agents}")

        # Use Enhanced Query Processor
        from ..core.enhanced_query_processor import get_enhanced_query_processor
        enhanced_processor = get_enhanced_query_processor()

        # Set installed agent information (should already be set, but set safely)
        if self.installed_agents_info:
            enhanced_processor.set_installed_agents_info(self.installed_agents_info)

        # Use Enhanced Query Processor's multi-agent selection logic
        selected_agents = enhanced_processor._select_agents_simple(semantic_query.natural_language, available_agents)

        if selected_agents:
            required_agents.extend(selected_agents)
            logger.info(f"✅ Enhanced Processor selection result: {len(selected_agents)} agents")
            for agent in selected_agents:
                logger.info(f"   📍 {agent}")
        else:
            # Fallback: basic multi-agent selection logic
            logger.warning("Enhanced Processor selection failed, using fallback logic")

            # Check for explicit agents in structured query
            if "required_agents" in semantic_query.structured_query:
                explicit_agents = semantic_query.structured_query["required_agents"]
                required_agents.extend([agent for agent in explicit_agents if agent in available_agents])
                logger.info(f"Explicit agent selection: {required_agents}")

            # Multi-agent keyword matching - compound query support
            if not required_agents:
                # Select agents for each domain
                domain_mappings = {
                    "INTERNET_SEARCH": ["검색", "찾아", "정보", "알아봐", "인터넷"],
                    "MEMO": ["저장", "메모", "기록", "기억"],
                    "CALCULATOR": ["계산", "수학", "더하기", "빼기"],
                    "WEATHER": ["날씨", "기온", "예보"],
                    "CURRENCY": ["환율", "달러", "원"],
                    "CRAWLER": ["주가", "주식", "삼성전자"],
                    "SCHEDULER": ["일정", "스케줄", "약속", "캘린더"],
                    "RESTAURANT_FINDER": ["맛집", "음식점", "레스토랑"],
                }

                # Detect compound query: select all agents matching each domain
                for agent_id in available_agents:
                    agent_type = self._extract_agent_type_from_id(agent_id)
                    if agent_type in domain_mappings:
                        keywords = domain_mappings[agent_type]
                        if any(kw in query_text for kw in keywords):
                            if agent_id not in required_agents:
                                required_agents.append(agent_id)
                                logger.info(f"📍 Domain match: {agent_type} → {agent_id}")

            # Last resort fallback: select internet search agent
            if not required_agents and available_agents:
                internet_agent = None
                for agent_id in available_agents:
                    if "internet" in agent_id.lower() or "search" in agent_id.lower():
                        internet_agent = agent_id
                        break

                if internet_agent:
                    required_agents.append(internet_agent)
                    logger.info(f"📍 Fallback internet agent: {internet_agent}")
                else:
                    required_agents.append(available_agents[0])
                    logger.info(f"📍 Fallback default agent: {available_agents[0]}")

        logger.info(f"🎯 Final selected agents: {len(required_agents)}")
        for i, agent in enumerate(required_agents):
            logger.info(f"   [{i+1}] {agent}")
        return required_agents
    
    def _extract_agent_type_from_id(self, agent_id: str) -> str:
        """Extract type from agent ID"""
        # Check type in installed agent information
        for agent_info in self.installed_agents_info:
            if agent_info.get('agent_id') == agent_id:
                agent_data = agent_info.get('agent_data', {})
                agent_type = agent_data.get('metadata', {}).get('agent_type', '')
                if agent_type:
                    return agent_type
        
        # Fallback: extract via pattern matching from ID
        agent_id_lower = agent_id.lower()
        if 'memo' in agent_id_lower:
            return 'MEMO'
        elif 'internet' in agent_id_lower or 'search' in agent_id_lower:
            return 'INTERNET_SEARCH'
        elif 'schedule' in agent_id_lower or 'calendar' in agent_id_lower:
            return 'SCHEDULER'
        elif 'restaurant' in agent_id_lower or 'finder' in agent_id_lower:
            return 'RESTAURANT_FINDER'
        elif 'calculator' in agent_id_lower or 'calc' in agent_id_lower:
            return 'CALCULATOR'
        elif 'fortune' in agent_id_lower or 'daily' in agent_id_lower:
            return 'FORECASTING'
        elif 'currency' in agent_id_lower or 'exchange' in agent_id_lower:
            return 'CURRENCY'
        else:
            return 'CUSTOM'
    
    def _create_workflow_steps(self, semantic_query: SemanticQuery, required_agents: List[str]) -> List[WorkflowStep]:
        """Create workflow steps"""
        steps = []

        for i, agent_id in enumerate(required_agents):
            agent_info = self._get_agent_capabilities(agent_id)

            # Generate step purpose
            purpose = self._generate_step_purpose(semantic_query, agent_id, i)

            # Determine dependencies
            dependencies = self._determine_dependencies(agent_id, required_agents[:i])

            # Create workflow step
            step = WorkflowStep.create_simple(
                agent_id=agent_id,
                purpose=purpose,
                concepts=semantic_query.concepts,
                complexity=agent_info.get("complexity", WorkflowComplexity.MODERATE),
                depends_on=dependencies,
                estimated_time=agent_info.get("estimated_time", 30.0)
            )
            
            steps.append(step)
        
        return steps
    
    def _generate_step_purpose(self, semantic_query: SemanticQuery, agent_id: str, step_index: int) -> str:
        """Generate step purpose"""
        agent_info = self._get_agent_capabilities(agent_id)
        primary_capability = agent_info.get("capabilities", ["processing"])[0]
        
        if step_index == 0:
            return f"Primary {primary_capability} for: {semantic_query.natural_language[:100]}"
        else:
            return f"Secondary {primary_capability} based on previous results"
    
    def _determine_dependencies(self, agent_id: str, previous_agents: List[str]) -> List[str]:
        """Determine dependencies"""
        dependencies = []

        # Extract base type from actual agent ID (e.g., memo_agent_dcf1704b61d4250e8762ba41f -> memo_agent)
        def extract_base_agent_type(full_agent_id: str) -> str:
            # Check type in installed agent information
            for agent_info in self.installed_agents_info:
                if agent_info.get('agent_id') == full_agent_id:
                    agent_data = agent_info.get('agent_data', {})
                    agent_type = agent_data.get('metadata', {}).get('agent_type', '')
                    
                    # Map agent type to base agent name
                    type_mapping = {
                        'INTERNET_SEARCH': 'internet_agent',
                        'MEMO': 'memo_agent',
                        'SCHEDULER': 'schedule_agent',
                        'RESTAURANT_FINDER': 'restaurant_agent',
                        'CALCULATOR': 'calculate_agent',
                        'CUSTOM': 'custom_agent',
                        'FORECASTING': 'fortune_agent'
                    }
                    
                    if agent_type in type_mapping:
                        return type_mapping[agent_type]
            
            # Fallback: extract via pattern matching from ID
            agent_id_lower = full_agent_id.lower()
            if 'memo' in agent_id_lower:
                return 'memo_agent'
            elif 'internet' in agent_id_lower or 'search' in agent_id_lower:
                return 'internet_agent'
            elif 'schedule' in agent_id_lower or 'calendar' in agent_id_lower:
                return 'schedule_agent'
            elif 'restaurant' in agent_id_lower or 'finder' in agent_id_lower:
                return 'restaurant_agent'
            elif 'calculator' in agent_id_lower or 'calc' in agent_id_lower:
                return 'calculate_agent'
            elif 'fortune' in agent_id_lower or 'daily' in agent_id_lower:
                return 'fortune_agent'
            elif 'currency' in agent_id_lower or 'exchange' in agent_id_lower:
                return 'currency_agent'
            else:
                return 'general_agent'
        
        # Extract base type for current agent
        base_agent_type = extract_base_agent_type(agent_id)

        # Check rule-based dependencies
        if base_agent_type in self.dependency_rules:
            required_deps = self.dependency_rules[base_agent_type]

            for dep_type in required_deps:
                # Find matching agent among previous agents
                for prev_agent in previous_agents:
                    prev_base_type = extract_base_agent_type(prev_agent)
                    if prev_base_type == dep_type:
                        # Generate step ID for this agent
                        dep_index = previous_agents.index(prev_agent)
                        dep_step_id = f"step_{dep_index:06d}"
                        dependencies.append(dep_step_id)
                        logger.debug(f"Dependency added: {agent_id} -> {prev_agent} (step: {dep_step_id})")
                        break
        
        return dependencies
    
    def _build_execution_graph(self, workflow_steps: List[WorkflowStep]) -> nx.DiGraph:
        """Build execution graph"""
        graph = nx.DiGraph()

        # Add nodes
        for step in workflow_steps:
            graph.add_node(step.step_id, step=step)

        # Add dependency edges
        for step in workflow_steps:
            for dependency in step.depends_on:
                if dependency in [s.step_id for s in workflow_steps]:
                    graph.add_edge(dependency, step.step_id)

        # Check and remove cyclic references
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("Cyclic reference detected, removing...")
            self._remove_cycles(graph)
        
        return graph
    
    def _remove_cycles(self, graph: nx.DiGraph):
        """Remove cyclic references"""
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    # Remove last edge
                    graph.remove_edge(cycle[-1], cycle[0])
                    logger.info(f"Cycle removed: {cycle[-1]} -> {cycle[0]}")
        except Exception as e:
            logger.error(f"Cycle removal failed: {e}")
    
    def _determine_optimization_strategy(self,
                                       semantic_query: SemanticQuery,
                                       workflow_steps: List[WorkflowStep]) -> OptimizationStrategy:
        """Determine optimization strategy"""
        query_text = semantic_query.natural_language.lower()

        # Speed-first keywords
        if any(word in query_text for word in ["빠르게", "즉시", "급하게", "신속"]):
            return OptimizationStrategy.SPEED_FIRST

        # Quality-first keywords
        if any(word in query_text for word in ["정확하게", "자세히", "완벽하게", "정밀"]):
            return OptimizationStrategy.QUALITY_FIRST

        # Resource-efficient keywords
        if any(word in query_text for word in ["효율적", "절약", "최소"]):
            return OptimizationStrategy.RESOURCE_EFFICIENT

        # Default strategy based on step count
        if len(workflow_steps) <= 2:
            return OptimizationStrategy.SPEED_FIRST
        elif len(workflow_steps) >= 5:
            return OptimizationStrategy.QUALITY_FIRST
        else:
            return OptimizationStrategy.BALANCED
    
    def _estimate_workflow_metrics(self, workflow_steps: List[WorkflowStep]) -> Tuple[float, float]:
        """Estimate workflow metrics"""
        # Quality estimation (based on agent complexity)
        complexity_scores = {
            WorkflowComplexity.SIMPLE: 0.7,
            WorkflowComplexity.MODERATE: 0.8,
            WorkflowComplexity.COMPLEX: 0.9,
            WorkflowComplexity.SOPHISTICATED: 0.95
        }
        
        quality_scores = [complexity_scores.get(step.estimated_complexity, 0.8) for step in workflow_steps]
        estimated_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.8
        
        # Time estimation (considering parallel processing)
        total_time = sum(step.estimated_time for step in workflow_steps)

        # Consider parallel processing opportunities
        independent_steps = [step for step in workflow_steps if not step.depends_on]
        if len(independent_steps) > 1:
            # Time reduced by parallel processing
            estimated_time = total_time * 0.7
        else:
            estimated_time = total_time
        
        return estimated_quality, estimated_time
    
    def _generate_reasoning_chain(self,
                                semantic_query: SemanticQuery,
                                workflow_steps: List[WorkflowStep],
                                optimization_strategy: OptimizationStrategy) -> List[str]:
        """Generate reasoning chain"""
        reasoning = []

        # Query analysis
        reasoning.append(f"Query intent analysis: {semantic_query.intent}")
        reasoning.append(f"Required concepts: {', '.join(semantic_query.concepts[:3])}")

        # Agent selection
        agent_names = [step.agent_id for step in workflow_steps]
        reasoning.append(f"Selected agents: {', '.join(agent_names)}")

        # Execution strategy
        reasoning.append(f"Optimization strategy: {optimization_strategy.value}")

        # Dependency analysis
        dependent_steps = [step for step in workflow_steps if step.depends_on]
        if dependent_steps:
            reasoning.append(f"Dependent steps: {len(dependent_steps)}")
        else:
            reasoning.append("All steps can execute independently")
        
        return reasoning
    
    def _create_fallback_workflow(self, semantic_query: SemanticQuery, available_agents: List[str]) -> WorkflowPlan:
        """Create fallback workflow"""
        logger.warning("Creating fallback workflow")

        # Select default agent
        fallback_agent = "internet_agent" if "internet_agent" in available_agents else available_agents[0]
        
        # Create simple step
        fallback_step = WorkflowStep.create_simple(
            agent_id=fallback_agent,
            purpose=f"Fallback processing for: {semantic_query.natural_language}",
            concepts=["general_processing"],
            complexity=WorkflowComplexity.SIMPLE,
            estimated_time=30.0
        )
        
        return WorkflowPlan.create_simple(
            semantic_query=semantic_query,
            steps=[fallback_step],
            strategy=OptimizationStrategy.BALANCED,
            quality=0.6,
            time=30.0,
            reasoning=["Created as fallback workflow"]
        )
    
    def _identify_parallel_groups(self, workflow_steps: List[WorkflowStep]) -> List[List[WorkflowStep]]:
        """Identify parallel groups"""
        parallel_groups = []

        # Group independent steps
        independent_steps = [step for step in workflow_steps if not step.depends_on]
        if len(independent_steps) > 1:
            parallel_groups.append(independent_steps)

        # Group steps with the same dependencies
        dependency_groups = {}
        for step in workflow_steps:
            if step.depends_on:
                dep_key = tuple(sorted(step.depends_on))
                if dep_key not in dependency_groups:
                    dependency_groups[dep_key] = []
                dependency_groups[dep_key].append(step)
        
        for group in dependency_groups.values():
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def _remove_duplicate_steps(self, workflow_steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Remove duplicate steps - merge consecutive calls to the same agent"""
        if not workflow_steps:
            return []

        optimized_steps = []
        i = 0

        while i < len(workflow_steps):
            current_step = workflow_steps[i]

            # Look for consecutive same-agent steps
            j = i + 1
            merged_purposes = [current_step.purpose]

            while j < len(workflow_steps):
                next_step = workflow_steps[j]

                # Check if same agent
                if next_step.agent_id == current_step.agent_id:
                    # If there are dependencies, continuity is broken, so don't merge
                    # (when dependent on a different agent)
                    other_dependencies = [dep for dep in next_step.depends_on
                                        if dep != current_step.step_id]
                    if other_dependencies:
                        break

                    # Mergeable step
                    logger.info(f"🔀 Consecutive same-agent call detected: {current_step.agent_id}")
                    logger.info(f"   Merging step {i+1} and step {j+1}")
                    merged_purposes.append(next_step.purpose)

                    # Merge concepts
                    for concept in next_step.required_concepts:
                        if concept not in current_step.required_concepts:
                            current_step.required_concepts.append(concept)

                    # Use longer estimated time (can run in parallel)
                    current_step.estimated_time = max(
                        current_step.estimated_time,
                        next_step.estimated_time
                    )

                    j += 1
                else:
                    # Different agent, stop
                    break

            # Update merged purpose
            if len(merged_purposes) > 1:
                current_step.purpose = f"Merged task ({len(merged_purposes)}): " + "; ".join(merged_purposes[:2])
                if len(merged_purposes) > 2:
                    current_step.purpose += f" and {len(merged_purposes)-2} more"

            optimized_steps.append(current_step)
            i = j  # Skip merged steps

        logger.info(f"✅ Deduplication completed: {len(workflow_steps)} → {len(optimized_steps)} steps")
        return optimized_steps
    
    def _optimize_dependencies(self, execution_graph: nx.DiGraph) -> nx.DiGraph:
        """Optimize dependencies"""
        optimized_graph = execution_graph.copy()

        # Remove unnecessary dependencies (transitive dependencies)
        transitive_edges = []
        for node in optimized_graph.nodes():
            for successor in optimized_graph.successors(node):
                for indirect_successor in optimized_graph.successors(successor):
                    if optimized_graph.has_edge(node, indirect_successor):
                        transitive_edges.append((node, indirect_successor))
        
        for edge in transitive_edges:
            optimized_graph.remove_edge(*edge)
            logger.info(f"Transitive dependency removed: {edge[0]} -> {edge[1]}")
        
        return optimized_graph
    
    def _reevaluate_strategy(self,
                           optimized_steps: List[WorkflowStep],
                           parallel_groups: List[List[WorkflowStep]]) -> OptimizationStrategy:
        """Re-evaluate execution strategy"""
        # Prioritize parallel processing when many parallel groups exist
        if len(parallel_groups) >= 2:
            return OptimizationStrategy.SPEED_FIRST

        # Prioritize quality when many complex steps exist
        complex_steps = [step for step in optimized_steps
                        if step.estimated_complexity in [WorkflowComplexity.COMPLEX, WorkflowComplexity.SOPHISTICATED]]

        if len(complex_steps) >= len(optimized_steps) * 0.5:
            return OptimizationStrategy.QUALITY_FIRST

        # Default: balanced strategy
        return OptimizationStrategy.BALANCED