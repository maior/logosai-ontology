"""
🔗 Result Processor
결과 처리자

워크플로우 실행 결과 통합 및 처리
"""

from typing import Dict, List, Any
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult, WorkflowPlan


class ResultProcessor:
    """🔗 결과 처리자"""
    
    async def integrate_results(self, 
                               execution_results: List[AgentExecutionResult],
                               workflow_plan: WorkflowPlan,
                               semantic_query: SemanticQuery) -> Dict[str, Any]:
        """결과 통합"""
        try:
            logger.info(f"🔄 결과 통합 시작 - 총 {len(execution_results)}개 결과")
            
            # 모든 결과 상세 로깅
            for i, result in enumerate(execution_results):
                logger.info(f"  결과 {i+1}: {result.agent_id}")
                logger.info(f"    성공: {result.success}")
                logger.info(f"    실행시간: {result.execution_time:.2f}초")
                logger.info(f"    신뢰도: {result.confidence}")
                
                if result.data:
                    logger.info(f"    데이터 타입: {type(result.data)}")
                    if isinstance(result.data, dict):
                        logger.info(f"    데이터 키: {list(result.data.keys())}")
                        # 실제 데이터 내용 일부 로깅
                        for key, value in result.data.items():
                            if isinstance(value, str) and len(value) > 100:
                                logger.info(f"      {key}: {value[:100]}...")
                            else:
                                logger.info(f"      {key}: {value}")
                    else:
                        logger.info(f"    데이터 내용: {str(result.data)[:200]}...")
                
                if result.error_message:
                    logger.warning(f"    오류: {result.error_message}")
            
            # 성공한 결과들만 추출
            successful_results = [r for r in execution_results if r.success]
            logger.info(f"📊 성공한 결과: {len(successful_results)}개")
            
            if not successful_results:
                logger.warning("⚠️ 모든 에이전트 실행이 실패했습니다.")
                return {
                    "status": "failed",
                    "message": "모든 에이전트 실행이 실패했습니다.",
                    "error_details": [r.error_message for r in execution_results if r.error_message]
                }
            
            # 결과 데이터 수집
            result_data = []
            for result in successful_results:
                if result.data:
                    processed_data = {
                        "agent_id": result.agent_id,
                        "agent_name": getattr(result, 'agent_name', result.agent_id),  # agent_name 추가
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time
                    }
                    result_data.append(processed_data)
                    logger.info(f"✅ 결과 데이터 추가: {result.agent_id} (이름: {processed_data['agent_name']})")
            
            logger.info(f"📋 처리할 결과 데이터: {len(result_data)}개")
            
            # 의도별 결과 통합
            intent = getattr(semantic_query, 'intent', 'general')
            logger.info(f"🎯 의도별 통합 시작: {intent}")
            
            # 통합 결과와 reasoning 정보를 함께 반환
            if intent == "information_retrieval":
                integrated_result = self._integrate_information_results_structured(result_data)
            elif intent == "analysis":
                integrated_result = self._integrate_analysis_results_structured(result_data)
            elif intent == "comparison":
                integrated_result = self._integrate_comparison_results_structured(result_data)
            else:
                integrated_result = self._integrate_general_results_structured(result_data)
            
            # 통합 결과가 딕셔너리인 경우와 문자열인 경우를 구분
            if isinstance(integrated_result, dict):
                logger.info(f"✅ 구조화된 통합 완료 - 내용 길이: {len(integrated_result.get('content', ''))}자")
                
                # 동적 reasoning 생성
                dynamic_reasoning = self._generate_dynamic_reasoning(
                    semantic_query=semantic_query,
                    workflow_plan=workflow_plan,
                    execution_results=execution_results,
                    integrated_result=integrated_result
                )
                
                # 기존 reasoning과 동적 reasoning 통합
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
                # 기존 문자열 반환을 위한 하위 호환성
                logger.info(f"✅ 통합 완료 - 내용 길이: {len(integrated_result)}자")
                
                # 문자열 결과를 위한 동적 reasoning 생성
                temp_integrated_result = {
                    "content": integrated_result,
                    "agent_results": [
                        {
                            "agent_id": r.agent_id,
                            "agent_name": getattr(r, 'agent_name', r.agent_id),  # agent_name이 있으면 사용, 없으면 agent_id
                            "result": r.data if r.data else "처리 완료",
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
            logger.error(f"결과 통합 실패: {e}")
            return {
                "status": "error",
                "message": f"결과 통합 중 오류 발생: {str(e)}",
                "partial_results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in execution_results]
            }
    
    def _integrate_information_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """정보 검색 결과 구조화된 통합"""
        logger.info(f"🔍 정보 검색 결과 구조화 통합 시작 - {len(result_data)}개 결과")
        
        contents = []
        reasonings = []
        agent_results = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)  # ExecutionEngine에서 전달된 agent_name 사용
            data = result["data"]
            
            logger.info(f"  처리 중: {agent_id} (이름: {agent_name})")
            logger.info(f"  데이터 타입: {type(data)}")
            
            content = None
            reasoning = None
            
            if isinstance(data, dict):
                logger.info(f"  데이터 키들: {list(data.keys())}")
                
                # answer와 reasoning 키가 있으면 우선적으로 사용
                if "answer" in data:
                    content = data["answer"]
                    reasoning = data.get("reasoning", "")
                    logger.info(f"  📝 answer 키에서 내용 추출: {len(str(content))}자")
                    if reasoning:
                        logger.info(f"  📝 reasoning 키에서 추론 추출: {len(str(reasoning))}자")
                # result가 딕셔너리이고 answer가 있는지 확인
                elif "result" in data:
                    result_value = data["result"]
                    if isinstance(result_value, dict) and "answer" in result_value:
                        content = result_value["answer"]
                        reasoning = result_value.get("reasoning", "")
                        logger.info(f"  📝 result.answer에서 내용 추출: {len(str(content))}자")
                    else:
                        content = str(result_value)
                # 다양한 형태의 검색 결과 처리
                elif "search_results" in data:
                    search_results = data["search_results"]
                    if isinstance(search_results, list) and search_results:
                        content = f"검색 결과 {len(search_results)}개 항목:\n"
                        for i, item in enumerate(search_results[:5], 1):  # 상위 5개만
                            title = item.get("title", "제목 없음")
                            snippet = item.get("snippet", item.get("description", ""))
                            content += f"{i}. {title}\n   {snippet}\n"
                    else:
                        content = "검색 결과가 없습니다."
                elif "content" in data:
                    content = data["content"]
                    reasoning = data.get("reasoning", "")
                elif "text" in data:
                    content = data["text"]
                    reasoning = data.get("reasoning", "")
                else:
                    # 폴백
                    content = str(data)
                    logger.warning(f"  ⚠️ 폴백: 전체 데이터를 문자열로 변환")
            else:
                content = str(data)
                logger.info(f"  📝 문자열/기타 타입에서 직접 변환")
            
            if content and str(content).strip():
                contents.append(content)
                if reasoning:
                    reasonings.append(reasoning)
                
                # agent_results 구성
                agent_results.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "result": {"answer": content, "reasoning": reasoning} if reasoning else content,
                    "execution_time": result.get("execution_time", 0),
                    "confidence": result.get("confidence", 0.8),
                    "hasArtifacts": False,
                    "agentType": "information_retrieval"
                })
                
                logger.info(f"  ✅ 내용 추가: {len(str(content))}자")
            else:
                logger.warning(f"  ⚠️ 빈 내용: {agent_id}")
        
        final_content = "\n\n".join(contents) if contents else "검색 결과를 찾을 수 없습니다."
        final_reasoning = "\n\n".join(reasonings) if reasonings else ""
        
        logger.info(f"🔍 정보 검색 결과 통합 완료 - 최종 길이: {len(final_content)}자")
        
        return {
            "content": final_content,
            "reasoning": final_reasoning,
            "agent_results": agent_results
        }
    
    def _integrate_information_results(self, result_data: List[Dict[str, Any]]) -> str:
        """정보 검색 결과 통합"""
        logger.info(f"🔍 정보 검색 결과 통합 시작 - {len(result_data)}개 결과")
        
        contents = []
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)  # ExecutionEngine에서 전달된 agent_name 사용
            data = result["data"]
            
            logger.info(f"  처리 중: {agent_id} (이름: {agent_name})")
            logger.info(f"  데이터 타입: {type(data)}")
            
            content = None
            
            if isinstance(data, dict):
                logger.info(f"  데이터 키들: {list(data.keys())}")
                
                # answer 키가 있으면 우선적으로 사용
                if "answer" in data:
                    content = data["answer"]
                    logger.info(f"  📝 answer 키에서 내용 추출: {len(str(content))}자")
                # 다양한 형태의 검색 결과 처리
                elif "search_results" in data:
                    search_results = data["search_results"]
                    if isinstance(search_results, list) and search_results:
                        content = f"검색 결과 {len(search_results)}개 항목:\n"
                        for i, item in enumerate(search_results[:5], 1):  # 상위 5개만
                            title = item.get("title", "제목 없음")
                            snippet = item.get("snippet", item.get("description", ""))
                            content += f"{i}. {title}\n   {snippet}\n"
                    else:
                        content = "검색 결과가 없습니다."
                elif "items" in data:
                    items = data["items"]
                    if isinstance(items, list) and items:
                        content = f"검색 항목 {len(items)}개:\n"
                        for i, item in enumerate(items[:5], 1):
                            title = item.get("title", item.get("name", "제목 없음"))
                            content += f"{i}. {title}\n"
                    else:
                        content = "검색 항목이 없습니다."
                elif "content" in data:
                    content = data["content"]
                elif "text" in data:
                    content = data["text"]
                elif "result" in data:
                    # result가 딕셔너리이고 answer가 있는지 확인
                    result_value = data["result"]
                    if isinstance(result_value, dict) and "answer" in result_value:
                        content = result_value["answer"]
                        logger.info(f"  📝 result.answer에서 내용 추출: {len(str(content))}자")
                    else:
                        content = str(result_value)
                else:
                    # 다른 키가 없으면 전체 딕셔너리를 문자열로 변환하지 말고 주요 내용만 추출
                    logger.warning(f"  ⚠️ 알려진 키가 없음, 전체 구조: {data}")
                    # 딕셔너리에서 가장 중요해 보이는 텍스트 내용을 찾아서 추출
                    for key in ['response', 'message', 'output', 'data']:
                        if key in data:
                            potential_content = data[key]
                            if isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                                content = potential_content
                                logger.info(f"  📝 {key} 키에서 내용 추출")
                                break
                    
                    # 여전히 content가 없으면 전체를 문자열로 변환
                    if content is None:
                        content = str(data)
                        logger.warning(f"  ⚠️ 폴백: 전체 딕셔너리를 문자열로 변환")
            else:
                content = str(data)
                logger.info(f"  📝 문자열/기타 타입에서 직접 변환")
            
            if content and str(content).strip():
                contents.append(content)  # agent_id 제거
                logger.info(f"  ✅ 내용 추가: {len(str(content))}자 (agent_id 제외)")
            else:
                logger.warning(f"  ⚠️ 빈 내용: {agent_id}")
        
        final_content = "\n\n".join(contents) if contents else "검색 결과를 찾을 수 없습니다."
        logger.info(f"🔍 정보 검색 결과 통합 완료 - 최종 길이: {len(final_content)}자")
        
        return final_content
    
    def _integrate_analysis_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분석 결과 구조화된 통합"""
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
                        content = f"**분석 결과 (신뢰도: {confidence:.2f})**\n"
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
                    "agentType": "analysis"
                })
        
        return {
            "content": "\n".join(analysis_parts),
            "reasoning": "\n\n".join(reasonings) if reasonings else "",
            "agent_results": agent_results
        }
    
    def _integrate_analysis_results(self, result_data: List[Dict[str, Any]]) -> str:
        """분석 결과 통합"""
        analysis_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)
            data = result["data"]
            confidence = result["confidence"]
            
            if isinstance(data, dict):
                insights = data.get("insights", [])
                if insights:
                    analysis_parts.append(f"**분석 결과 (신뢰도: {confidence:.2f})**")  # agent_id 제거
                    for insight in insights:
                        analysis_parts.append(f"- {insight}")
                else:
                    analysis_parts.append(str(data))  # agent_id 제거
            else:
                analysis_parts.append(str(data))  # agent_id 제거
        
        return "\n".join(analysis_parts)
    
    def _integrate_comparison_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """비교 결과 구조화된 통합"""
        comparison_parts = ["## 비교 분석 결과"]
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
                comparison_parts.append(f"\n### {i}. 결과")
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
                    "agentType": "comparison"
                })
        
        return {
            "content": "\n".join(comparison_parts),
            "reasoning": "\n\n".join(reasonings) if reasonings else "",
            "agent_results": agent_results
        }
    
    def _integrate_comparison_results(self, result_data: List[Dict[str, Any]]) -> str:
        """비교 결과 통합"""
        comparison_parts = ["## 비교 분석 결과"]
        
        for i, result in enumerate(result_data, 1):
            agent_id = result["agent_id"]
            data = result["data"]
            
            comparison_parts.append(f"\n### {i}. 결과")  # agent_id 제거
            comparison_parts.append(str(data))
        
        return "\n".join(comparison_parts)
    
    def _integrate_general_results_structured(self, result_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """일반 결과 구조화된 통합"""
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
                    "agentType": "general"
                })
        
        return {
            "content": "\n".join(general_parts),
            "reasoning": "\n\n".join(reasonings) if reasonings else "",
            "agent_results": agent_results
        }
    
    def _integrate_general_results(self, result_data: List[Dict[str, Any]]) -> str:
        """일반 결과 통합"""
        general_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            agent_name = result.get("agent_name", agent_id)
            data = result["data"]
            
            general_parts.append(str(data))  # agent_id 제거
        
        return "\n".join(general_parts)
    
    def _generate_dynamic_reasoning(self, 
                                   semantic_query: SemanticQuery,
                                   workflow_plan: WorkflowPlan,
                                   execution_results: List[AgentExecutionResult],
                                   integrated_result: Dict[str, Any]) -> str:
        """쿼리 분석과 실행 결과를 결합한 동적 reasoning 생성"""
        try:
            logger.info("🧠 동적 reasoning 생성 시작")
            
            # 1. 쿼리 맥락 추출
            query_context = {
                "original_query": getattr(semantic_query, 'original_query', ''),
                "intent": getattr(semantic_query, 'intent', 'general'),
                "key_entities": getattr(semantic_query, 'key_entities', []),
                "expected_output": getattr(semantic_query, 'expected_output_type', 'text')
            }
            logger.info(f"쿼리 맥락: 의도={query_context['intent']}, 핵심개체={len(query_context['key_entities'])}개")
            
            # 2. 워크플로우 정보 추출
            workflow_info = {
                "strategy": getattr(workflow_plan, 'optimization_strategy', 'AUTO'),
                "planned_agents": [step.agent_id for step in getattr(workflow_plan, 'steps', [])],
                "successful_agents": [getattr(r, 'agent_name', r.agent_id) for r in execution_results if r.success],
                "failed_agents": [getattr(r, 'agent_name', r.agent_id) for r in execution_results if not r.success]
            }
            logger.info(f"워크플로우: 전략={workflow_info['strategy']}, 성공={len(workflow_info['successful_agents'])}개")
            
            # 3. 의도별 reasoning 구성
            reasoning_parts = []
            
            # 의도 설명
            intent_map = {
                "information_retrieval": "정보 검색",
                "analysis": "분석",
                "comparison": "비교",
                "general": "일반 질의"
            }
            intent_desc = intent_map.get(query_context['intent'], query_context['intent'])
            reasoning_parts.append(f"사용자님의 '{query_context['original_query']}' 요청은 {intent_desc} 작업으로 분류되었습니다.")
            
            # 접근 방법 설명
            strategy_map = {
                "PARALLEL": "병렬 처리",
                "SEQUENTIAL": "순차 처리",
                "HYBRID": "하이브리드",
                "AUTO": "자동 최적화"
            }
            strategy_desc = strategy_map.get(workflow_info['strategy'], workflow_info['strategy'])
            
            if workflow_info['successful_agents']:
                reasoning_parts.append(
                    f"{strategy_desc} 방식으로 {len(workflow_info['successful_agents'])}개의 전문 에이전트가 "
                    f"작업을 수행했습니다: {', '.join(workflow_info['successful_agents'])}"
                )
            
            # 실패한 에이전트가 있는 경우
            if workflow_info['failed_agents']:
                reasoning_parts.append(
                    f"일부 에이전트({', '.join(workflow_info['failed_agents'])})는 처리에 실패했지만, "
                    f"다른 에이전트들의 결과로 답변을 구성했습니다."
                )
            
            # 핵심 개체 언급 (있는 경우)
            if query_context['key_entities']:
                key_entities_str = ', '.join(query_context['key_entities'][:5])  # 최대 5개만
                reasoning_parts.append(f"주요 키워드: {key_entities_str}")
            
            # 에이전트별 핵심 기여 요약
            agent_contributions = []
            for agent_result in integrated_result.get('agent_results', []):
                # agent_name이 없거나 agent_id와 같으면 agent_id를 보기 좋게 표시
                agent_name = agent_result.get('agent_name', '')
                if not agent_name or agent_name == agent_result.get('agent_id', ''):
                    agent_name = agent_result.get('agent_id', 'Unknown')
                    # _agent 접미사 제거하여 더 깔끔하게
                    if agent_name.endswith('_agent'):
                        agent_name = agent_name[:-6].replace('_', ' ').title() + ' Agent'
                
                if isinstance(agent_result.get('result'), dict):
                    agent_reasoning = agent_result['result'].get('reasoning', '')
                    if agent_reasoning:
                        # reasoning이 너무 길면 요약
                        if len(agent_reasoning) > 150:
                            agent_reasoning = agent_reasoning[:150] + "..."
                        agent_contributions.append(f"• {agent_name}: {agent_reasoning}")
            
            if agent_contributions:
                reasoning_parts.append("\n에이전트별 분석:\n" + "\n".join(agent_contributions[:5]))  # 최대 5개
            
            # 실행 시간과 신뢰도 정보 (metadata가 있는 경우)
            metadata = integrated_result.get('metadata', {})
            if metadata:
                exec_time = metadata.get('total_execution_time', 0)
                avg_confidence = metadata.get('average_confidence', 0)
                if exec_time > 0:
                    reasoning_parts.append(f"\n처리 시간: {exec_time:.2f}초, 평균 신뢰도: {avg_confidence:.2%}")
            
            # 최종 reasoning 조합
            final_reasoning = "\n\n".join(reasoning_parts)
            logger.info(f"✅ 동적 reasoning 생성 완료 - 길이: {len(final_reasoning)}자")
            
            return final_reasoning
            
        except Exception as e:
            logger.error(f"❌ 동적 reasoning 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 reasoning 반환
            return f"'{getattr(semantic_query, 'original_query', '질의')}'에 대한 처리를 완료했습니다." 