"""
🔗 Result Processor
Result Processor

Integrates and processes workflow execution results
"""

from typing import Dict, List, Any
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult, WorkflowPlan


class ResultProcessor:
    """🔗 Result Processor"""
    
    async def integrate_results(self, 
                               execution_results: List[AgentExecutionResult],
                               workflow_plan: WorkflowPlan,
                               semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Integrate results"""
        try:
            logger.info(f"🔄 Starting result integration - total {len(execution_results)} results")
            
            # Log all results in detail
            for i, result in enumerate(execution_results):
                logger.info(f"  Result {i+1}: {result.agent_id}")
                logger.info(f"    Success: {result.success}")
                logger.info(f"    Execution time: {result.execution_time:.2f}s")
                logger.info(f"    Confidence: {result.confidence}")
                
                if result.data:
                    logger.info(f"    Data type: {type(result.data)}")
                    if isinstance(result.data, dict):
                        logger.info(f"    Data keys: {list(result.data.keys())}")
                        # Log partial actual data content
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
                logger.warning("⚠️ All agent executions have failed.")
                return {
                    "status": "failed",
                    "message": "All agent executions have failed.",
                    "error_details": [r.error_message for r in execution_results if r.error_message]
                }
            
            # Collect result data
            result_data = []
            for result in successful_results:
                if result.data:
                    processed_data = {
                        "agent_id": result.agent_id,
                        "agent_name": getattr(result, 'agent_name', result.agent_id),  # add agent_name
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time
                    }
                    result_data.append(processed_data)
                    logger.info(f"✅ Result data added: {result.agent_id} (name: {processed_data['agent_name']})")
            
            logger.info(f"📋 Result data to process: {len(result_data)}")
            
            # Integrate results by intent
            intent = getattr(semantic_query, 'intent', 'general')
            logger.info(f"🎯 Starting intent-based integration: {intent}")
            
            # Return integrated result together with reasoning information
            if intent == "information_retrieval":
                integrated_result = self._integrate_information_results_structured(result_data)
            elif intent == "analysis":
                integrated_result = self._integrate_analysis_results_structured(result_data)
            elif intent == "comparison":
                integrated_result = self._integrate_comparison_results_structured(result_data)
            else:
                integrated_result = self._integrate_general_results_structured(result_data)
            
            # Distinguish between dict and string integrated result
            if isinstance(integrated_result, dict):
                logger.info(f"✅ Structured integration complete - content length: {len(integrated_result.get('content', ''))} chars")
                
                # Generate dynamic reasoning
                dynamic_reasoning = self._generate_dynamic_reasoning(
                    semantic_query=semantic_query,
                    workflow_plan=workflow_plan,
                    execution_results=execution_results,
                    integrated_result=integrated_result
                )
                
                # Merge existing reasoning with dynamic reasoning
                final_reasoning = integrated_result.get('reasoning', '')
                if dynamic_reasoning:
                    final_reasoning = f"{dynamic_reasoning}\n\n{final_reasoning}" if final_reasoning else dynamic_reasoning
                
                return {
                    "status": "success",
                    "content": integrated_result.get('content', ''),
                    "reasoning": final_reasoning,
                    "agent_results": integrated_result.get('agent_results', []),
                    "metadata": {
                        "total_agents": len(execution_results),
                        "successful_agents": len(successful_results),
                        "total_execution_time": sum(r.execution_time for r in execution_results),
                        "average_confidence": sum(r.confidence for r in successful_results) / len(successful_results) if successful_results else 0,
                        "strategy_used": getattr(workflow_plan, 'optimization_strategy', 'AUTO')
                    }
                }
            else:
                # Backward compatibility for string return
                logger.info(f"✅ Integration complete - content length: {len(integrated_result)} chars")
                
                # Generate dynamic reasoning for string result
                temp_integrated_result = {
                    "content": integrated_result,
                    "agent_results": [
                        {
                            "agent_id": r.agent_id,
                            "agent_name": getattr(r, 'agent_name', r.agent_id),  # use agent_name if available, else agent_id
                            "result": r.data if r.data else "Processing complete",
                            "execution_time": r.execution_time,
                            "confidence": r.confidence
                        } for r in successful_results
                    ]
                }
                
                dynamic_reasoning = self._generate_dynamic_reasoning(
                    semantic_query=semantic_query,
                    workflow_plan=workflow_plan,
                    execution_results=execution_results,
                    integrated_result=temp_integrated_result
                )
                
                return {
                    "status": "success",
                    "content": integrated_result,
                    "reasoning": dynamic_reasoning,
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
    
    def _integrate_information_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structured integration of information retrieval results"""
        logger.info(f"🔍 Starting structured information result integration - {len(result_data)} results")
        
        contents = []
        reasonings = []
        agent_results = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)  # use agent_name passed from ExecutionEngine
            data = result["data"]
            
            logger.info(f"  Processing: {agent_id} (name: {agent_name})")
            logger.info(f"  Data type: {type(data)}")
            
            content = None
            reasoning = None
            
            if isinstance(data, dict):
                logger.info(f"  Data keys: {list(data.keys())}")
                
                # Prefer answer and reasoning keys if available
                if "answer" in data:
                    content = data["answer"]
                    reasoning = data.get("reasoning", "")
                    logger.info(f"  📝 Content extracted from answer key: {len(str(content))} chars")
                    if reasoning:
                        logger.info(f"  📝 Reasoning extracted from reasoning key: {len(str(reasoning))} chars")
                # Check if result is a dict and has answer
                elif "result" in data:
                    result_value = data["result"]
                    if isinstance(result_value, dict) and "answer" in result_value:
                        content = result_value["answer"]
                        reasoning = result_value.get("reasoning", "")
                        logger.info(f"  📝 Content extracted from result.answer: {len(str(content))} chars")
                    else:
                        content = str(result_value)
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
                elif "content" in data:
                    content = data["content"]
                    reasoning = data.get("reasoning", "")
                elif "text" in data:
                    content = data["text"]
                    reasoning = data.get("reasoning", "")
                else:
                    # Fallback
                    content = str(data)
                    logger.warning(f"  ⚠️ Fallback: converting entire data to string")
            else:
                content = str(data)
                logger.info(f"  📝 Direct conversion from string/other type")
            
            if content and str(content).strip():
                contents.append(content)
                if reasoning:
                    reasonings.append(reasoning)
                
                # Build agent_results
                agent_results.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "result": {"answer": content, "reasoning": reasoning} if reasoning else content,
                    "execution_time": result.get("execution_time", 0),
                    "confidence": result.get("confidence", 0.8),
                    "hasArtifacts": False,
                    "agentType": self._detect_agent_type(agent_id),
                    "category": self._extract_category(data, agent_id)
                })
                
                logger.info(f"  ✅ Content added: {len(str(content))} chars")
            else:
                logger.warning(f"  ⚠️ Empty content: {agent_id}")
        
        final_content = "\n\n".join(contents) if contents else "No search results found."
        final_reasoning = "\n\n".join(reasonings) if reasonings else ""
        
        logger.info(f"🔍 Information result integration complete - final length: {len(final_content)} chars")
        
        return {
            "content": final_content,
            "reasoning": final_reasoning,
            "agent_results": agent_results
        }
    
    def _integrate_information_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate information search results"""
        logger.info(f"🔍 Starting information search result integration - {len(result_data)} results")
        
        contents = []
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)  # use agent_name passed from ExecutionEngine
            data = result["data"]
            
            logger.info(f"  Processing: {agent_id} (name: {agent_name})")
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
    
    def _integrate_analysis_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structured integration of analysis results"""
        analysis_parts = []
        reasonings = []
        agent_results = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)
            data = result["data"]
            confidence = result["confidence"]
            
            content = None
            reasoning = None
            
            if isinstance(data, dict):
                if "answer" in data:
                    content = data["answer"]
                    reasoning = data.get("reasoning", "")
                elif "insights" in data:
                    insights = data.get("insights", [])
                    if insights:
                        content = f"**Analysis Result (confidence: {confidence:.2f})**\n"
                        for insight in insights:
                            content += f"- {insight}\n"
                    else:
                        content = str(data)
                    reasoning = data.get("reasoning", "")
                else:
                    content = str(data)
                    reasoning = data.get("reasoning", "")
            else:
                content = str(data)
            
            if content:
                analysis_parts.append(content)
                if reasoning:
                    reasonings.append(reasoning)
                
                agent_results.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "result": {"answer": content, "reasoning": reasoning} if reasoning else content,
                    "execution_time": result.get("execution_time", 0),
                    "confidence": confidence,
                    "hasArtifacts": False,
                    "agentType": self._detect_agent_type(agent_id),
                    "category": self._extract_category(data, agent_id)
                })
        
        return {
            "content": "\n".join(analysis_parts),
            "reasoning": "\n\n".join(reasonings) if reasonings else "",
            "agent_results": agent_results
        }
    
    def _integrate_analysis_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate analysis results"""
        analysis_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)
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
    
    def _integrate_comparison_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structured integration of comparison results"""
        comparison_parts = ["## Comparison Analysis Results"]
        reasonings = []
        agent_results = []
        
        for i, result in enumerate(result_data, 1):
            agent_id = result["agent_id"]
            data = result["data"]
            
            content = None
            reasoning = None
            
            if isinstance(data, dict):
                if "answer" in data:
                    content = data["answer"]
                    reasoning = data.get("reasoning", "")
                else:
                    content = str(data)
                    reasoning = data.get("reasoning", "")
            else:
                content = str(data)
            
            if content:
                comparison_parts.append(f"\n### {i}. Result")
                comparison_parts.append(content)
                
                if reasoning:
                    reasonings.append(reasoning)
                
                agent_results.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "result": {"answer": content, "reasoning": reasoning} if reasoning else content,
                    "execution_time": result.get("execution_time", 0),
                    "confidence": result.get("confidence", 0.8),
                    "hasArtifacts": False,
                    "agentType": self._detect_agent_type(agent_id),
                    "category": self._extract_category(data, agent_id)
                })
        
        return {
            "content": "\n".join(comparison_parts),
            "reasoning": "\n\n".join(reasonings) if reasonings else "",
            "agent_results": agent_results
        }
    
    def _integrate_comparison_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate comparison results"""
        comparison_parts = ["## Comparison Analysis Results"]
        
        for i, result in enumerate(result_data, 1):
            agent_id = result["agent_id"]
            data = result["data"]
            
            comparison_parts.append(f"\n### {i}. Result")  # exclude agent_id
            comparison_parts.append(str(data))
        
        return "\n".join(comparison_parts)
    
    def _integrate_general_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structured integration of general results"""
        general_parts = []
        reasonings = []
        agent_results = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)
            data = result["data"]
            
            content = None
            reasoning = None
            
            if isinstance(data, dict):
                if "answer" in data:
                    content = data["answer"]
                    reasoning = data.get("reasoning", "")
                else:
                    content = str(data)
                    reasoning = data.get("reasoning", "")
            else:
                content = str(data)
            
            if content:
                general_parts.append(content)
                
                if reasoning:
                    reasonings.append(reasoning)
                
                agent_results.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "result": {"answer": content, "reasoning": reasoning} if reasoning else content,
                    "execution_time": result.get("execution_time", 0),
                    "confidence": result.get("confidence", 0.8),
                    "hasArtifacts": False,
                    "agentType": self._detect_agent_type(agent_id),
                    "category": self._extract_category(data, agent_id)
                })
        
        return {
            "content": "\n".join(general_parts),
            "reasoning": "\n\n".join(reasonings) if reasonings else "",
            "agent_results": agent_results
        }
    
    def _integrate_general_results(self, result_data: List[Dict[str, Any]]) -> str:
        """Integrate general results"""
        general_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)
            data = result["data"]
            
            general_parts.append(str(data))  # exclude agent_id
        
        return "\n".join(general_parts)
    
    def _generate_dynamic_reasoning(self, 
                                   semantic_query: SemanticQuery,
                                   workflow_plan: WorkflowPlan,
                                   execution_results: List[AgentExecutionResult],
                                   integrated_result: Dict[str, Any]) -> str:
        """Generate dynamic reasoning by combining query analysis and execution results"""
        try:
            logger.info("🧠 Starting dynamic reasoning generation")
            
            # 1. Extract query context
            query_context = {
                "original_query": getattr(semantic_query, 'original_query', ''),
                "intent": getattr(semantic_query, 'intent', 'general'),
                "key_entities": getattr(semantic_query, 'key_entities', []),
                "expected_output": getattr(semantic_query, 'expected_output_type', 'text')
            }
            logger.info(f"Query context: intent={query_context['intent']}, key entities={len(query_context['key_entities'])}")
            
            # 2. Extract workflow info
            workflow_info = {
                "strategy": getattr(workflow_plan, 'optimization_strategy', 'AUTO'),
                "planned_agents": [step.agent_id for step in getattr(workflow_plan, 'steps', [])],
                "successful_agents": [getattr(r, 'agent_name', r.agent_id) for r in execution_results if r.success],
                "failed_agents": [getattr(r, 'agent_name', r.agent_id) for r in execution_results if not r.success]
            }
            logger.info(f"Workflow: strategy={workflow_info['strategy']}, successful={len(workflow_info['successful_agents'])}")
            
            # 3. Compose reasoning by intent
            reasoning_parts = []
            
            # Intent description
            intent_map = {
                "information_retrieval": "Information Search",
                "analysis": "Analysis",
                "comparison": "Comparison",
                "general": "General Query"
            }
            intent_desc = intent_map.get(query_context['intent'], query_context['intent'])
            reasoning_parts.append(f"Your request '{query_context['original_query']}' has been classified as a {intent_desc} task.")
            
            # Approach description
            strategy_map = {
                "PARALLEL": "Parallel Processing",
                "SEQUENTIAL": "Sequential Processing",
                "HYBRID": "Hybrid",
                "AUTO": "Auto Optimization"
            }
            strategy_desc = strategy_map.get(workflow_info['strategy'], workflow_info['strategy'])
            
            if workflow_info['successful_agents']:
                reasoning_parts.append(
                    f"{len(workflow_info['successful_agents'])} specialized agents processed using {strategy_desc}: "
                    f"{', '.join(workflow_info['successful_agents'])}"
                )
            
            # If there are failed agents
            if workflow_info['failed_agents']:
                reasoning_parts.append(
                    f"Some agents ({', '.join(workflow_info['failed_agents'])}) failed, "
                    f"but the answer was constructed from results of other agents."
                )
            
            # Mention key entities (if present)
            if query_context['key_entities']:
                key_entities_str = ', '.join(query_context['key_entities'][:5])  # max 5 only
                reasoning_parts.append(f"Key keywords: {key_entities_str}")
            
            # Summary of key contributions per agent
            agent_contributions = []
            for agent_result in integrated_result.get('agent_results', []):
                # Display agent_id cleanly when agent_name is missing or same
                agent_name = agent_result.get('agent_name', '')
                if not agent_name or agent_name == agent_result.get('agent_id', ''):
                    agent_name = agent_result.get('agent_id', 'Unknown')
                    # Remove _agent suffix for cleaner display
                    if agent_name.endswith('_agent'):
                        agent_name = agent_name[:-6].replace('_', ' ').title() + ' Agent'
                
                if isinstance(agent_result.get('result'), dict):
                    agent_reasoning = agent_result['result'].get('reasoning', '')
                    if agent_reasoning:
                        # Summarize if reasoning is too long
                        if len(agent_reasoning) > 150:
                            agent_reasoning = agent_reasoning[:150] + "..."
                        agent_contributions.append(f"• {agent_name}: {agent_reasoning}")
            
            if agent_contributions:
                reasoning_parts.append("\nPer-agent analysis:\n" + "\n".join(agent_contributions[:5]))  # max 5
            
            # Execution time and confidence info (if metadata available)
            metadata = integrated_result.get('metadata', {})
            if metadata:
                exec_time = metadata.get('total_execution_time', 0)
                avg_confidence = metadata.get('average_confidence', 0)
                if exec_time > 0:
                    reasoning_parts.append(f"\nProcessing time: {exec_time:.2f}s, Average confidence: {avg_confidence:.2%}")
            
            # Combine final reasoning
            final_reasoning = "\n\n".join(reasoning_parts)
            logger.info(f"✅ Dynamic reasoning generation complete - length: {len(final_reasoning)} chars")
            
            return final_reasoning
            
        except Exception as e:
            logger.error(f"❌ Error during dynamic reasoning generation: {str(e)}")
            # Return default reasoning on error
            return f"Processing of '{getattr(semantic_query, 'original_query', 'query')}' is complete."
    
    def _detect_agent_type(self, agent_id: str) -> str:
        """Extract type from agent ID"""
        agent_id_lower = agent_id.lower()
        
        if "calculator" in agent_id_lower or "calc" in agent_id_lower:
            return "calculator"
        elif "search" in agent_id_lower:
            return "search"
        elif "weather" in agent_id_lower:
            return "weather"
        elif "stock" in agent_id_lower:
            return "stock"
        elif "currency" in agent_id_lower or "exchange" in agent_id_lower:
            return "currency"
        elif "table" in agent_id_lower or "chart" in agent_id_lower or "visual" in agent_id_lower:
            return "visualization"
        elif "memo" in agent_id_lower:
            return "memo"
        elif "rag" in agent_id_lower:
            return "rag"
        elif "samsung" in agent_id_lower:
            return "samsung"
        else:
            return "general"
    
    def _extract_category(self, data: Any, agent_id: str) -> str:
        """Extract category from agent result"""
        # 1. Extract from metadata if data is dict
        if isinstance(data, dict):
            # Find category in metadata
            metadata = data.get("metadata", {})
            if metadata.get("category"):
                logger.info(f"✅ Category found in metadata: {metadata['category']}")
                return metadata["category"]
            
            # Find category at top level
            if data.get("category"):
                logger.info(f"✅ Category found at top level: {data['category']}")
                return data["category"]
        
        # 2. Infer from agent_id
        inferred_category = self._infer_category_from_agent_id(agent_id)
        logger.info(f"📊 Category inferred from agent_id '{agent_id}': {inferred_category}")
        return inferred_category
    
    def _infer_category_from_agent_id(self, agent_id: str) -> str:
        """Infer category from agent ID"""
        agent_id_lower = agent_id.lower()
        
        if "samsung" in agent_id_lower:
            return "samsung"
        elif "rag" in agent_id_lower:
            return "rag"
        elif "calculator" in agent_id_lower or "calc" in agent_id_lower:
            return "calculation"
        elif "search" in agent_id_lower:
            return "search"
        elif "weather" in agent_id_lower:
            return "weather"
        elif "stock" in agent_id_lower or "finance" in agent_id_lower:
            return "finance"
        elif "currency" in agent_id_lower or "exchange" in agent_id_lower:
            return "currency"
        elif "visual" in agent_id_lower or "chart" in agent_id_lower or "graph" in agent_id_lower:
            return "visualization"
        elif "table" in agent_id_lower:
            return "table"
        elif "memo" in agent_id_lower:
            return "memo"
        else:
            return "general" 