"""
🧠 Knowledge Graph Management Module
Knowledge Graph Management Module

Responsible for ontology knowledge graph updates, concept addition, and relationship creation
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import SemanticQuery, WorkflowPlan, AgentExecutionResult
from ..engines.knowledge_graph_clean import KnowledgeGraphEngine


class KnowledgeGraphManager:
    """🧠 Knowledge Graph Manager - ontology updates and concept management"""
    
    def __init__(self, knowledge_graph: KnowledgeGraphEngine):
        self.knowledge_graph = knowledge_graph
        
        logger.info("🧠 Knowledge graph manager initialized")
    
    async def update_knowledge_graph(self, 
                                   semantic_query: SemanticQuery,
                                   workflow_plan: WorkflowPlan,
                                   execution_results: List[AgentExecutionResult],
                                   integrated_result: Dict[str, Any]):
        """Main knowledge graph update function"""
        try:
            logger.info("🧠 Starting knowledge graph update")
            
            # Generate workflow ID
            workflow_id = f"workflow_{workflow_plan.plan_id}"
            
            # 1. Ensure basic ontology concepts
            await self._ensure_basic_ontology_concepts()
            
            # 2. Create workflow concept
            await self._create_workflow_concept(workflow_plan, workflow_id)
            
            # 3. Create domain and concept nodes
            await self._create_domain_and_concept_nodes(semantic_query, execution_results, workflow_id)
            
            # 4. Create agent collaboration network
            await self._create_agent_collaboration_network(execution_results, workflow_id)
            
            # 5. Create knowledge pattern nodes
            await self._create_knowledge_pattern_nodes(semantic_query, execution_results, integrated_result, workflow_id)
            
            # 6. Add relationships
            await self._add_relationships(semantic_query, execution_results, workflow_id)
            
            logger.info("✅ Knowledge graph update complete")
            
        except Exception as e:
            logger.error(f"Knowledge graph update failed: {e}")
    
    async def _ensure_basic_ontology_concepts(self):
        """Ensure basic ontology concepts"""
        try:
            basic_concepts = [
                ("concept", "concept", {"name": "concept", "description": "Abstract ideas or thoughts"}),
                ("entity", "concept", {"name": "entity", "description": "Concrete objects or entities"}),
                ("agent", "concept", {"name": "agent", "description": "Entity that performs tasks"}),
                ("workflow", "concept", {"name": "workflow", "description": "Task flow and procedures"}),
                ("domain", "concept", {"name": "domain", "description": "Specific knowledge domain"}),
                ("knowledge", "concept", {"name": "knowledge", "description": "Accumulated information and experience"})
            ]
            
            for concept_id, concept_type, attributes in basic_concepts:
                await self.knowledge_graph.add_concept(concept_id, concept_type, attributes)
            
            logger.info("✅ Basic ontology concepts ensured")
            
        except Exception as e:
            logger.error(f"Failed to ensure basic ontology concepts: {e}")
    
    async def _create_workflow_concept(self, workflow_plan: WorkflowPlan, workflow_id: str):
        """Create workflow concept"""
        try:
            workflow_attributes = {
                "optimization_strategy": getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy)),
                "estimated_time": workflow_plan.estimated_time,
                "estimated_quality": workflow_plan.estimated_quality,
                "step_count": len(workflow_plan.steps),
                "created_at": workflow_plan.created_at.isoformat() if hasattr(workflow_plan.created_at, 'isoformat') else str(workflow_plan.created_at),
                "query_intent": workflow_plan.query.intent if workflow_plan.query else "unknown"
            }
            
            await self.knowledge_graph.add_concept(workflow_id, "workflow", workflow_attributes)
            
            # Link to basic workflow concept
            await self.knowledge_graph.add_relationship(workflow_id, "workflow", "is_instance_of")
            
            logger.info(f"✅ Workflow concept created: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Workflow concept creation failed: {e}")
    
    async def _create_domain_and_concept_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create domain and concept nodes"""
        try:
            # Infer domain based on query intent
            domain_mapping = {
                "information_retrieval": "information_domain",
                "data_analysis": "analysis_domain", 
                "calculation": "computation_domain",
                "creative_generation": "creative_domain",
                "task_automation": "automation_domain"
            }
            
            primary_domain = domain_mapping.get(semantic_query.intent, "general_domain")
            
            # Create domain node
            domain_attributes = {
                "domain_name": primary_domain.replace("_", " ").title(),
                "query_intent": semantic_query.intent,
                "complexity_level": "high" if semantic_query.complexity_score > 0.7 else "medium" if semantic_query.complexity_score > 0.4 else "low"
            }
            
            await self.knowledge_graph.add_concept(primary_domain, "domain", domain_attributes)
            
            # Link workflow and domain
            await self.knowledge_graph.add_relationship(workflow_id, primary_domain, "operates_in_domain")
            
            # Add entities as concepts
            for entity in semantic_query.entities:
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                entity_attributes = {
                    "entity_name": entity,
                    "source_query": semantic_query.query_id,
                    "domain": primary_domain
                }
                
                await self.knowledge_graph.add_concept(entity_id, "entity", entity_attributes)
                await self.knowledge_graph.add_relationship(entity_id, primary_domain, "belongs_to_domain")
                await self.knowledge_graph.add_relationship(workflow_id, entity_id, "processes_entity")
            
            # Add concepts
            for concept in semantic_query.concepts:
                concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                concept_attributes = {
                    "concept_name": concept,
                    "source_query": semantic_query.query_id,
                    "domain": primary_domain
                }
                
                await self.knowledge_graph.add_concept(concept_id, "concept", concept_attributes)
                await self.knowledge_graph.add_relationship(concept_id, primary_domain, "belongs_to_domain")
                await self.knowledge_graph.add_relationship(workflow_id, concept_id, "involves_concept")
            
            logger.info(f"✅ Domain and concept nodes created: {primary_domain}")
            
        except Exception as e:
            logger.error(f"Domain and concept node creation failed: {e}")
    
    async def _create_agent_collaboration_network(self, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create agent collaboration network"""
        try:
            # Create agent nodes
            agent_ids = []
            for result in execution_results:
                agent_attributes = {
                    "agent_type": getattr(result.agent_type, 'value', str(result.agent_type)),
                    "last_execution_time": result.execution_time,
                    "last_success": result.success,
                    "last_confidence": getattr(result, 'confidence', 0.8)
                }
                
                await self.knowledge_graph.add_concept(result.agent_id, "agent", agent_attributes)
                await self.knowledge_graph.add_relationship(workflow_id, result.agent_id, "utilizes_agent")
                agent_ids.append(result.agent_id)
            
            # Create collaboration relationships between agents
            for i in range(len(agent_ids) - 1):
                current_agent = agent_ids[i]
                next_agent = agent_ids[i + 1]
                
                await self.knowledge_graph.add_relationship(
                    current_agent, next_agent, "collaborates_with",
                    {"collaboration_type": "sequential", "workflow_id": workflow_id}
                )
            
            logger.info(f"✅ Agent collaboration network creation complete")
            
        except Exception as e:
            logger.error(f"Agent collaboration network creation failed: {e}")
    
    async def _create_knowledge_pattern_nodes(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], integrated_result: Dict[str, Any], workflow_id: str):
        """Create knowledge pattern nodes"""
        try:
            # Analyze execution patterns
            agent_types = [getattr(r.agent_type, 'value', str(r.agent_type)) for r in execution_results]
            unique_agent_types = list(set(agent_types))
            
            pattern_id = f"pattern_{workflow_id}"
            pattern_attributes = {
                "pattern_name": f"{semantic_query.intent}_pattern",
                "agent_types_used": unique_agent_types,
                "total_agents": len(execution_results),
                "unique_agent_types": len(unique_agent_types),
                "complexity_level": semantic_query.complexity_score,
                "success_pattern": integrated_result.get('success', False),
                "execution_strategy": "multi_agent" if len(execution_results) > 1 else "single_agent"
            }
            
            await self.knowledge_graph.add_concept(pattern_id, "knowledge", pattern_attributes)
            await self.knowledge_graph.add_relationship(workflow_id, pattern_id, "generates_knowledge")
            
            # Knowledge patterns by result type
            if integrated_result.get('components'):
                for component_type, component_data in integrated_result['components'].items():
                    knowledge_id = f"knowledge_{workflow_id}_{component_type}"
                    knowledge_attributes = {
                        "knowledge_type": component_type,
                        "data_size": len(str(component_data)) if component_data else 0,
                        "generated_by": workflow_id
                    }
                    
                    await self.knowledge_graph.add_concept(knowledge_id, "knowledge", knowledge_attributes)
                    await self.knowledge_graph.add_relationship(pattern_id, knowledge_id, "contains_knowledge")
            
            logger.info(f"✅ Knowledge pattern nodes creation complete")
            
        except Exception as e:
            logger.error(f"Knowledge pattern node creation failed: {e}")
    
    async def _add_relationships(self, semantic_query: SemanticQuery, execution_results: List[AgentExecutionResult], workflow_id: str):
        """Create additional relationships"""
        try:
            # Basic relationships
            basic_relations = [
                ("agent", "capability", "has"),
                ("workflow", "task", "contains"),
                ("concept", "domain", "belongs_to"),
                ("entity", "concept", "is_instance_of")
            ]
            
            for subject, object_concept, predicate in basic_relations:
                await self.knowledge_graph.add_relationship(subject, object_concept, predicate)
            
            logger.info(f"✅ Additional relationship creation complete")
            
        except Exception as e:
            logger.error(f"Additional relationship creation failed: {e}") 