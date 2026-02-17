"""
🔗 Result Integration Module
Result Integration Module

Semanticaly integrates and analyzes multi-agent execution results
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, WorkflowPlan, AgentExecutionResult,
    ExecutionStatus, get_system_metrics, ExecutionContext
)
from ..core.llm_manager import get_ontology_llm_manager, OntologyLLMType
from ..services.agent_detector import get_agent_detector
from ..services.mermaid_fallback_service import get_mermaid_fallback_service
from ..services.visualization_response_formatter import get_visualization_response_formatter


class ResultIntegrator:
    """🔗 Result Integrator - intelligent integration of multi-agent results"""
    
    def __init__(self):
        self.llm_manager = get_ontology_llm_manager()
        self.metrics = get_system_metrics()
        
        logger.info("🔗 Result integrator initialized")
    
    async def integrate_results(self, 
                              execution_results: List[AgentExecutionResult],
                              workflow_plan: WorkflowPlan,
                              semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Main result integration function"""
        try:
            logger.info(f"🔄 Starting result integration - {len(execution_results)} results")
            
            # 1. Classify and pre-process results
            classified_results = self._classify_results(execution_results)
            
            # 2. Integrate results by type
            integration_result = {}
            
            if classified_results.get('information'):
                integration_result['information'] = await self._integrate_information_results(
                    classified_results['information']
                )
            
            if classified_results.get('analysis'):
                integration_result['analysis'] = await self._integrate_analysis_results(
                    classified_results['analysis']
                )
            
            if classified_results.get('calculation'):
                integration_result['calculation'] = await self._integrate_calculation_results(
                    classified_results['calculation']
                )
            
            if classified_results.get('visualization'):
                integration_result['visualization'] = await self._integrate_visualization_results(
                    classified_results['visualization']
                )
            
            # 3. Generate final integrated result
            final_result = await self._create_final_integration(
                integration_result, semantic_query, workflow_plan
            )
            
            logger.info(f"✅ Result integration complete: {final_result.get('success', False)}")
            return final_result
            
        except Exception as e:
            logger.error(f"Result integration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_results': execution_results
            }
    
    async def integrate_agent_results_with_visualization(self, 
                                                       original_query: str,
                                                       agent_results: List[Dict[str, Any]],
                                                       execution_context: ExecutionContext) -> Dict[str, Any]:
        """Integrate agent results - including visualization processing"""
        try:
            logger.info(f"🔄 Starting agent result integration (with visualization) - {len(agent_results)} results")
            
            # 1. Check if visualization is needed
            needs_visualization = self._needs_visualization(original_query, agent_results)
            
            if needs_visualization:
                logger.info("📊 Visualization need detected - checking visualization agents")
                
                # 2. Check for agents with visualization capability
                detector = get_agent_detector(execution_context)
                
                if detector.has_visualization_capability():
                    # 3-A. Call agent with visualization capability
                    best_agent = detector.get_best_visualization_agent()
                    logger.info(f"✅ Using visualization-capable agent: {best_agent}")
                    
                    # Check if visualization agent result already exists
                    viz_result = next((r for r in agent_results if r.get('agent_id') == best_agent), None)
                    if not viz_result:
                        logger.warning(f"⚠️ No result from {best_agent} - using fallback system")
                        viz_result = await self._create_visualization_fallback(original_query, agent_results)
                        agent_results.append(viz_result)
                else:
                    # 3-B. Use Mermaid fallback system
                    logger.info("🔄 No visualization-capable agent - using Mermaid fallback")
                    viz_result = await self._create_visualization_fallback(original_query, agent_results)
                    agent_results.append(viz_result)
            
            # 4. Execute existing integration logic
            return await self.integrate_agent_results(original_query, agent_results)
            
        except Exception as e:
            logger.error(f"Result integration with visualization failed: {e}")
            # Fallback: run existing integration logic only
            return await self.integrate_agent_results(original_query, agent_results)

    async def integrate_agent_results(self, 
                                    original_query: str,
                                    agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate agent results - interface for testing"""
        try:
            logger.info(f"🔄 Starting agent result integration - {len(agent_results)} results")
            
            # Convert agent results to AgentExecutionResult format
            execution_results = []
            for result in agent_results:
                # Create object with required attributes
                class AgentResult:
                    def __init__(self, agent_id, data, success, execution_time, confidence, metadata=None):
                        self.agent_id = agent_id
                        self.data = data
                        self.success = success
                        self.execution_time = execution_time
                        self.confidence = confidence
                        self.metadata = metadata or {}
                        self.error_message = None if success else "Execution failed"
                
                exec_result = AgentResult(
                    agent_id=result.get('agent_id', 'unknown'),
                    data=result.get('data', {}),
                    success=result.get('success', False),
                    execution_time=result.get('execution_time', 0),
                    confidence=result.get('confidence', 0.8),
                    metadata=result.get('metadata', {})
                )
                execution_results.append(exec_result)
            
            # Classify results
            classified_results = self._classify_results(execution_results)
            logger.info(f"📊 Classified results: {list(classified_results.keys())}")
            
            # Integrate information retrieval results (process all results as information)
            all_contents = []
            for category, results in classified_results.items():
                if results:  # only if not an empty list
                    for result in results:
                        content = self._extract_content_from_data(result['data'], result['agent_id'])
                        if content and content.strip():
                            all_contents.append(content)
                            logger.info(f"✅ Extracted content from {result['agent_id']}: {len(content)} chars")
            
            if not all_contents:
                logger.warning("⚠️ No content extracted.")
                return {
                    "success": False,
                    "message": "No valid content found in agent results.",
                    "agent_results_count": len(agent_results)
                }
            
            # Integrate using LLM
            logger.info(f"🧠 Starting LLM integration - {len(all_contents)} contents")
            integrated_content = await self._llm_integrate_information(all_contents)
            
            # Build final result
            final_result = {
                "success": True,
                "integrated_content": integrated_content,
                "original_query": original_query,
                "agent_results_count": len(agent_results),
                "successful_extractions": len(all_contents),
                "processing_summary": {
                    "total_agents": len(agent_results),
                    "successful_agents": len([r for r in agent_results if r.get('success', False)]),
                    "content_extracted": len(all_contents),
                    "integration_method": "LLM" if len(all_contents) > 1 else "Direct"
                }
            }
            
            logger.info(f"✅ Agent result integration complete")
            return final_result
            
        except Exception as e:
            logger.error(f"Agent result integration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_query": original_query,
                "agent_results_count": len(agent_results)
            }
    
    def _classify_results(self, execution_results: List[AgentExecutionResult]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify results by type"""
        classified = {
            'information': [],
            'analysis': [],
            'calculation': [],
            'visualization': [],
            'general': []
        }
        
        for result in execution_results:
            # Normalize result data format
            result_data = {
                "agent_id": result.agent_id,
                "data": result.data if hasattr(result, 'data') else result.result_data,
                "success": result.success,
                "execution_time": result.execution_time,
                "confidence": getattr(result, 'confidence', 0.8),
                "metadata": result.metadata
            }
            
            # Classify based on agent ID
            agent_id = result.agent_id.lower()
            
            if any(keyword in agent_id for keyword in ['internet', 'search', 'web', 'news']):
                classified['information'].append(result_data)
            elif any(keyword in agent_id for keyword in ['analysis', 'analyze', 'research']):
                classified['analysis'].append(result_data)
            elif any(keyword in agent_id for keyword in ['calculate', 'math', 'finance', 'stock']):
                classified['calculation'].append(result_data)
            elif any(keyword in agent_id for keyword in ['chart', 'graph', 'visual', 'plot']):
                classified['visualization'].append(result_data)
            else:
                classified['general'].append(result_data)
        
        return classified
    
    async def _integrate_information_results(self, results: List[Dict[str, Any]]) -> str:
        """Integrate information retrieval results"""
        logger.info(f"🔍 Starting information result integration - {len(results)} results")
        
        contents = []
        for result in results:
            agent_id = result["agent_id"]
            data = result["data"]
            
            content = self._extract_content_from_data(data, agent_id)
            if content and str(content).strip():
                contents.append(content)
                logger.info(f"  ✅ Content added: {len(str(content))} chars")
        
        # Advanced integration using LLM
        if len(contents) > 1:
            integrated_content = await self._llm_integrate_information(contents)
            if integrated_content:
                return integrated_content
        
        # Fallback: simple concatenation
        final_content = "\n\n".join(contents) if contents else "No search results found."
        logger.info(f"🔍 Information result integration complete - final length: {len(final_content)} chars")
        
        return final_content
    
    async def _integrate_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """Integrate analysis results"""
        logger.info(f"📊 Starting analysis result integration - {len(results)} results")
        
        analysis_contents = []
        
        for result in results:
            agent_id = result["agent_id"]
            data = result["data"]
            
            analysis_content = self._extract_content_from_data(data, agent_id)
            if analysis_content:
                analysis_contents.append({
                    'agent': agent_id,
                    'content': analysis_content,
                    'confidence': result.get('confidence', 0.8)
                })
        
        if not analysis_contents:
            return "Unable to generate analysis results."
        
        # Integrate analysis using LLM
        integrated_analysis = await self._llm_integrate_analysis(analysis_contents)
        
        logger.info(f"📊 Analysis result integration complete")
        return integrated_analysis
    
    async def _integrate_calculation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate calculation results"""
        logger.info(f"🔢 Starting calculation result integration - {len(results)} results")
        
        calculations = []
        
        for result in results:
            agent_id = result["agent_id"]
            data = result["data"]
            
            calc_result = self._extract_calculation_result(data, agent_id)
            if calc_result:
                calculations.append({
                    'agent': agent_id,
                    'result': calc_result,
                    'confidence': result.get('confidence', 0.8)
                })
        
        if not calculations:
            return {"error": "No calculation results found."}
        
        integrated_calc = {
            'calculations': calculations,
            'summary': self._create_calculation_summary(calculations),
            'total_results': len(calculations)
        }
        
        logger.info(f"🔢 Calculation result integration complete")
        return integrated_calc
    
    async def _integrate_visualization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate visualization results - enhanced version"""
        logger.info(f"📈 Starting visualization result integration - {len(results)} results")
        
        visualizations = []
        formatter = get_visualization_response_formatter()
        
        for result in results:
            agent_id = result["agent_id"]
            data = result["data"]
            
            # Process visualization-capable agent results
            if self._is_visualization_agent_result(agent_id, data):
                viz_result = formatter.format_data_visualization_agent_result(
                    result, result.get('original_query', '')
                )
                visualizations.append(viz_result)
            else:
                # Process existing visualization data
                viz_data = self._extract_visualization_data(data, agent_id)
                if viz_data:
                    visualizations.append({
                        'agent': agent_id,
                        'data': viz_data,
                        'type': self._detect_visualization_type(viz_data),
                        'confidence': result.get('confidence', 0.8)
                    })
        
        if not visualizations:
            return {"error": "No visualization results found."}
        
        integrated_viz = {
            'visualizations': visualizations,
            'summary': f"{len(visualizations)} visualizations generated",
            'types': [viz.get('type', 'unknown') for viz in visualizations],
            'formatted': True  # using new formatting system
        }
        
        logger.info(f"📈 Visualization result integration complete")
        return integrated_viz
    
    def _needs_visualization(self, query: str, agent_results: List[Dict[str, Any]]) -> bool:
        """Determine if visualization is needed"""
        query_lower = query.lower()
        
        # 1. Check if query contains visualization keywords
        viz_keywords = [
            '시각화', '차트', '그래프', '플로우차트', '순서도', '다이어그램',
            'visualization', 'chart', 'graph', 'flowchart', 'diagram', 'mermaid'
        ]
        
        if any(keyword in query_lower for keyword in viz_keywords):
            logger.info(f"📊 Visualization keyword found in query: {query}")
            return True
        
        # 2. Check if agent results contain visualizable structured data
        for result in agent_results:
            data = result.get('data', {})
            if self._has_visualizable_data(data):
                logger.info(f"📊 Visualizable data found in {result.get('agent_id')} result")
                return True
        
        # 3. Consider process visualization for multi-agent execution
        if len(agent_results) > 2:
            logger.info(f"📊 Multi-agent execution detected ({len(agent_results)}) - considering process visualization")
            return True
        
        return False
    
    def _has_visualizable_data(self, data: Any) -> bool:
        """Check if data is visualizable"""
        if isinstance(data, dict):
            # Check numeric data
            numeric_keys = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    numeric_keys.append(key)
            
            if len(numeric_keys) >= 2:
                return True
            
            # Check list data
            list_keys = []
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 1:
                    list_keys.append(key)
            
            if len(list_keys) >= 1:
                return True
        
        return False
    
    async def _create_visualization_fallback(self, 
                                           original_query: str,
                                           agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate visualization fallback"""
        try:
            logger.info("🔄 Starting visualization fallback generation")
            
            # Use Mermaid fallback service
            fallback_service = get_mermaid_fallback_service()
            mermaid_result = await fallback_service.generate_mermaid_from_agent_results(
                original_query, agent_results
            )
            
            # Convert to agent result format
            return {
                'agent_id': 'mermaid_fallback_service',
                'data': mermaid_result,
                'success': True,
                'execution_time': 0.5,
                'confidence': 0.7,
                'metadata': {
                    'fallback': True,
                    'visualization_type': mermaid_result.get('format', 'mermaid')
                }
            }
            
        except Exception as e:
            logger.error(f"Visualization fallback generation failed: {e}")
            return {
                'agent_id': 'visualization_error',
                'data': {
                    'type': 'error',
                    'content': f'Visualization generation failed: {str(e)}',
                    'title': 'Visualization Error',
                    'fallback': True
                },
                'success': False,
                'execution_time': 0.1,
                'confidence': 0.1,
                'metadata': {'error': str(e)}
            }
    
    def _is_visualization_agent_result(self, agent_id: str, data: Any) -> bool:
        """Check if agent result is from a visualization agent"""
        try:
            # 1. Check visualization capability via agent ID
            if any(keyword in agent_id.lower() for keyword in [
                'visual', 'chart', 'graph', 'plot', 'diagram', 'mermaid'
            ]):
                return True
            
            # 2. Check via result data structure
            if isinstance(data, dict):
                # Check HTML content
                content = data.get('content', '')
                if isinstance(content, str):
                    if any(tag in content for tag in ['<svg', '<mermaid', 'mermaid', 'graph TD', 'graph LR']):
                        return True
                
                # Check metadata
                metadata = data.get('metadata', {})
                if isinstance(metadata, dict):
                    viz_type = metadata.get('visualization_type', '')
                    if viz_type or metadata.get('contains_mermaid', False) or metadata.get('contains_svg', False):
                        return True
                
                # Check visualization-related keys
                viz_keys = ['chart_data', 'mermaid_code', 'svg_content', 'visualization_type']
                if any(key in data for key in viz_keys):
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Visualization agent result check failed {agent_id}: {e}")
            return False
    
    def _extract_content_from_data(self, data: Any, agent_id: str) -> Optional[str]:
        """Extract text content from data - improved version"""
        if isinstance(data, dict):
            logger.debug(f"📊 {agent_id} data structure: {list(data.keys())}")
            
            # 1st pass: Check direct answer keys
            for key in ['answer', 'content', 'text', 'result', 'response']:
                if key in data:
                    potential_content = data[key]
                    
                    # Handle nested dictionaries recursively
                    if isinstance(potential_content, dict):
                        # result is a dict with answer or content inside
                        if 'answer' in potential_content:
                            return str(potential_content['answer'])
                        elif 'content' in potential_content:
                            return str(potential_content['content'])
                        elif 'result' in potential_content:
                            return str(potential_content['result'])
                        else:
                            # Convert entire dictionary to string cleanly
                            return self._dict_to_readable_text(potential_content)
                    elif isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                        return potential_content.strip()
            
            # 2nd pass: Check agent-specific keys
            if 'weather' in agent_id.lower():
                # Weather agent specific handling
                weather_info = self._extract_weather_info(data)
                if weather_info:
                    return weather_info
            elif 'currency' in agent_id.lower() or 'exchange' in agent_id.lower():
                # Currency exchange agent specific handling
                currency_info = self._extract_currency_info(data)
                if currency_info:
                    return currency_info
            elif 'crawler' in agent_id.lower() or 'stock' in agent_id.lower():
                # Stock/crawling agent specific handling
                stock_info = self._extract_stock_info(data)
                if stock_info:
                    return stock_info
            
            # 3rd pass: Check other keys
            for key in ['message', 'output', 'data', 'value', 'info']:
                if key in data:
                    potential_content = data[key]
                    if isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                        return potential_content.strip()
                    elif isinstance(potential_content, dict):
                        return self._dict_to_readable_text(potential_content)
            
            # 4th pass: Convert entire dictionary to readable text
            return self._dict_to_readable_text(data)
        else:
            return str(data).strip() if str(data).strip() else None

    def _extract_weather_info(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract weather information"""
        try:
            info_parts = []
            
            # Basic weather info
            if 'temperature' in data:
                info_parts.append(f"Temperature: {data['temperature']}")
            if 'humidity' in data:
                info_parts.append(f"Humidity: {data['humidity']}")
            if 'condition' in data:
                info_parts.append(f"Weather: {data['condition']}")
            if 'wind' in data:
                info_parts.append(f"Wind: {data['wind']}")
            
            return ", ".join(info_parts) if info_parts else None
        except:
            return None

    def _extract_currency_info(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract currency exchange information"""
        try:
            info_parts = []
            
            if 'rate' in data:
                info_parts.append(f"Exchange rate: {data['rate']}")
            if 'change' in data:
                info_parts.append(f"Change: {data['change']}")
            if 'change_percent' in data:
                info_parts.append(f"Change rate: {data['change_percent']}")
            
            return ", ".join(info_parts) if info_parts else None
        except:
            return None

    def _extract_stock_info(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract stock price information"""
        try:
            # Handle error case
            if 'error' in data:
                error_msg = data.get('error', 'Unknown error')
                return f"Stock price lookup failed: {error_msg}"
            
            info_parts = []
            
            if 'price' in data:
                info_parts.append(f"Current price: {data['price']}")
            if 'change' in data:
                info_parts.append(f"Day change: {data['change']}")
            if 'change_percent' in data:
                info_parts.append(f"Change rate: {data['change_percent']}")
            if 'volume' in data:
                info_parts.append(f"Volume: {data['volume']}")
            
            return ", ".join(info_parts) if info_parts else None
        except:
            return None

    def _dict_to_readable_text(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to readable text"""
        try:
            readable_parts = []
            
            for key, value in data.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    readable_parts.append(f"{key}: {value.strip()}")
                elif isinstance(value, (int, float)):
                    readable_parts.append(f"{key}: {value}")
                elif isinstance(value, dict) and value:
                    # Represent nested dict briefly
                    readable_parts.append(f"{key}: [complex data]")
                elif isinstance(value, list) and value:
                    readable_parts.append(f"{key}: [list of {len(value)} items]")
            
            return "; ".join(readable_parts) if readable_parts else str(data)
        except:
            return str(data)
    
    def _extract_calculation_result(self, data: Any, agent_id: str) -> Optional[Any]:
        """Extract calculation result"""
        if isinstance(data, dict):
            for key in ['result', 'calculation', 'value', 'answer', 'output']:
                if key in data:
                    return data[key]
        
        # Check if numeric
        try:
            if isinstance(data, (int, float)):
                return data
            elif isinstance(data, str):
                import re
                numbers = re.findall(r'-?\d+\.?\d*', data)
                if numbers:
                    return float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
        except:
            pass
        
        return data
    
    def _extract_visualization_data(self, data: Any, agent_id: str) -> Optional[Any]:
        """Extract visualization data"""
        if isinstance(data, dict):
            for key in ['chart', 'graph', 'plot', 'visualization', 'image', 'figure']:
                if key in data:
                    return data[key]
        
        return data
    
    def _detect_visualization_type(self, viz_data: Any) -> str:
        """Detect visualization type"""
        if isinstance(viz_data, dict):
            if 'type' in viz_data:
                return viz_data['type']
            elif 'chart_type' in viz_data:
                return viz_data['chart_type']
        
        return 'unknown'
    
    async def _generate_user_focused_response(self, 
                                            user_query: str, 
                                            integration_result: Dict[str, Any],
                                            semantic_query: SemanticQuery,
                                            workflow_plan: WorkflowPlan) -> Optional[str]:
        """Generate final response tailored to user query"""
        try:
            logger.info(f"🤖 Starting custom response generation - query: {user_query[:50]}...")
            
            # Collect agent results
            agent_results = []
            
            # information results
            if integration_result.get('information'):
                agent_results.append({
                    "type": "information",
                    "content": integration_result['information']
                })
            
            # analysis results  
            if integration_result.get('analysis'):
                agent_results.append({
                    "type": "analysis",
                    "content": integration_result['analysis']
                })
                
            # calculation results
            if integration_result.get('calculation'):
                calc_result = integration_result['calculation']
                if isinstance(calc_result, dict) and 'summary' in calc_result:
                    content = calc_result['summary']
                else:
                    content = str(calc_result)
                agent_results.append({
                    "type": "calculation",
                    "content": content
                })
            
            # visualization results
            if integration_result.get('visualization'):
                viz_result = integration_result['visualization']
                agent_results.append({
                    "type": "visualization",
                    "content": str(viz_result)
                })
            
            # Return None if no agent results
            if not agent_results:
                logger.warning("No agent results available, skipping custom response generation")
                return None
            
            # Generate prompt
            agent_results_text = "\n\n".join([
                f"[{r['type'].upper()}]\n{r['content']}" for r in agent_results
            ])
            
            prompt = f"""사용자가 다음과 같은 요청을 했습니다:
"{user_query}"

다음은 여러 에이전트가 처리한 결과입니다:
{agent_results_text}

위 에이전트 결과들을 종합하여, 사용자의 원래 요청에 정확히 부합하는 응답을 작성해주세요.

중요 지침:
1. 사용자가 요청한 형식(플로우차트, 표, 리스트 등)이 있다면 반드시 그 형식으로 제공하세요
2. "제주도 여행 5일 일정을 플로우차트로"와 같은 요청이면, 마크다운 플로우차트 형식으로 일정을 구성하세요
3. 에이전트 결과를 단순 나열하지 말고, 사용자 요청에 맞게 재구성하세요
4. 불필요한 설명은 제외하고 핵심 내용만 포함하세요
5. 사용자 언어(한국어/영어)와 동일한 언어로 응답하세요

응답:"""

            # Call LLM
            llm = self.llm_manager.get_llm(OntologyLLMType.RESULT_INTEGRATOR)
            response = await llm.ainvoke(prompt)
            
            # Extract response
            if hasattr(response, 'content'):
                result = response.content.strip()
            else:
                result = str(response).strip()
            
            logger.info(f"✅ Custom response generation complete - length: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Custom response generation failed: {e}")
            return None
    
    def _create_calculation_summary(self, calculations: List[Dict[str, Any]]) -> str:
        """Generate calculation result summary"""
        if not calculations:
            return "No calculation result"
        
        summary_parts = []
        for calc in calculations:
            agent = calc['agent']
            result = calc['result']
            summary_parts.append(f"{agent}: {result}")
        
        return "; ".join(summary_parts)
    
    async def _llm_integrate_information(self, contents: List[str]) -> str:
        """Integrate information using LLM"""
        try:
            # Build content by source
            sources_text = ""
            for i, content in enumerate(contents):
                sources_text += f"\n{'='*50}\n소스 {i+1}:\n{content}\n"
            
            integration_prompt = f"""
다음 여러 에이전트로부터 받은 정보를 통합하여 사용자에게 제공할 일관성 있고 포괄적인 답변을 만들어주세요.

**통합 원칙:**
1. 중복되는 정보는 하나로 합치기
2. 상충하는 정보가 있으면 신뢰도가 높은 것 우선
3. 각 정보의 핵심 내용을 놓치지 않기
4. 사용자가 이해하기 쉬운 구조로 정리
5. 마크다운이나 HTML 태그는 제거하고 깔끔한 텍스트로 변환

**정보 소스들:**{sources_text}

**요구사항:**
- 위 정보들을 종합하여 사용자 친화적인 통합 답변을 생성해주세요
- 각 정보의 핵심 내용을 모두 포함하되, 중복은 제거해주세요
- 정보가 부족하거나 오류가 있는 부분은 명확히 표시해주세요
- 최종 답변은 한국어로 작성해주세요
"""
            
            llm = self.llm_manager.get_llm(OntologyLLMType.RESULT_INTEGRATOR)
            if llm:
                response = await llm.ainvoke(integration_prompt)
                
                if hasattr(response, 'content'):
                    integrated = response.content
                else:
                    integrated = str(response)
                
                if integrated and isinstance(integrated, str) and len(integrated.strip()) > 0:
                    logger.info(f"✅ LLM information integration successful: {len(integrated)} chars")
                    return integrated.strip()
            
        except Exception as e:
            logger.warning(f"LLM information integration failed: {e}")
        
        # Fallback: simple concatenation
        logger.info("🔄 Integrating information in fallback mode")
        return "\n\n".join(contents)
    
    async def _llm_integrate_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """Integrate analysis using LLM"""
        try:
            analysis_texts = []
            for analysis in analyses:
                analysis_texts.append(f"{analysis['agent']} (confidence: {analysis['confidence']:.2f}):\n{analysis['content']}")
            
            integration_prompt = f"""
            다음 여러 분석 결과를 종합하여 통합된 분석 결과를 제공해주세요:
            
            {"=" * 50}
            """.join(analysis_texts)
            
            integrated = await self.llm_manager.call_llm_async(
                OntologyLLMType.RESULT_INTEGRATOR, 
                integration_prompt
            )
            
            if integrated and isinstance(integrated, str):
                return integrated
            
        except Exception as e:
            logger.warning(f"LLM analysis integration failed: {e}")
        
        return "\n\n".join([a['content'] for a in analyses])
    
    async def _create_final_integration(self, 
                                      integration_result: Dict[str, Any],
                                      semantic_query: SemanticQuery,
                                      workflow_plan: WorkflowPlan) -> Dict[str, Any]:
        """Generate final integrated result"""
        
        # Extract user query
        user_query = getattr(semantic_query, 'query_text', '') or getattr(semantic_query, 'natural_language', '')
        
        # Generate final answer tailored to user query using LLM
        main_content = await self._generate_user_focused_response(
            user_query, integration_result, semantic_query, workflow_plan
        )
        
        # Default handling when main answer is not generated
        if not main_content:
            if integration_result.get('information'):
                main_content = integration_result['information']
            elif integration_result.get('analysis'):
                main_content = integration_result['analysis']
            elif integration_result.get('calculation'):
                calc_result = integration_result['calculation']
                if isinstance(calc_result, dict) and 'summary' in calc_result:
                    main_content = calc_result['summary']
                else:
                    main_content = str(calc_result)
            else:
                main_content = "The requested task has been completed."
        
        # Build result in format expected by ontology_system.py
        final_result = {
            'success': True,
            'integrated_content': main_content,  # Main response content
            'execution_summary': {
                'query_id': semantic_query.query_id,
                'workflow_id': workflow_plan.plan_id,
                'integration_timestamp': datetime.now().isoformat(),
                'components_processed': list(integration_result.keys()),
                'total_components': len(integration_result)
            },
            'workflow_visualization': "",  # Initialize to empty string (add if needed)
            'confidence_score': 0.8,  # Default confidence score
            'sources': [],  # Initialize to empty array (add if needed)
            'components': integration_result,  # Preserve original integration result
            'metadata': {
                'query_id': semantic_query.query_id,
                'workflow_id': workflow_plan.plan_id,
                'integration_timestamp': datetime.now().isoformat()
            }
        }
        
        # Include visualization data if available
        if integration_result.get('visualization'):
            final_result['has_visualization'] = True
            final_result['visualization_data'] = integration_result['visualization']
        
        # Include calculation data if available
        if integration_result.get('calculation'):
            final_result['has_calculation'] = True
            final_result['calculation_data'] = integration_result['calculation']
        
        return final_result 