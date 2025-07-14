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
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time
                    }
                    result_data.append(processed_data)
                    logger.info(f"✅ 결과 데이터 추가: {result.agent_id}")
            
            logger.info(f"📋 처리할 결과 데이터: {len(result_data)}개")
            
            # 의도별 결과 통합
            intent = getattr(semantic_query, 'intent', 'general')
            logger.info(f"🎯 의도별 통합 시작: {intent}")
            
            if intent == "information_retrieval":
                integrated_content = self._integrate_information_results(result_data)
            elif intent == "analysis":
                integrated_content = self._integrate_analysis_results(result_data)
            elif intent == "comparison":
                integrated_content = self._integrate_comparison_results(result_data)
            else:
                integrated_content = self._integrate_general_results(result_data)
            
            logger.info(f"✅ 통합 완료 - 내용 길이: {len(integrated_content)}자")
            
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
            logger.error(f"결과 통합 실패: {e}")
            return {
                "status": "error",
                "message": f"결과 통합 중 오류 발생: {str(e)}",
                "partial_results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in execution_results]
            }
    
    def _integrate_information_results(self, result_data: List[Dict[str, Any]]) -> str:
        """정보 검색 결과 통합"""
        logger.info(f"🔍 정보 검색 결과 통합 시작 - {len(result_data)}개 결과")
        
        contents = []
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            
            logger.info(f"  처리 중: {agent_id}")
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
    
    def _integrate_analysis_results(self, result_data: List[Dict[str, Any]]) -> str:
        """분석 결과 통합"""
        analysis_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
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
    
    def _integrate_comparison_results(self, result_data: List[Dict[str, Any]]) -> str:
        """비교 결과 통합"""
        comparison_parts = ["## 비교 분석 결과"]
        
        for i, result in enumerate(result_data, 1):
            agent_id = result["agent_id"]
            data = result["data"]
            
            comparison_parts.append(f"\n### {i}. 결과")  # agent_id 제거
            comparison_parts.append(str(data))
        
        return "\n".join(comparison_parts)
    
    def _integrate_general_results(self, result_data: List[Dict[str, Any]]) -> str:
        """일반 결과 통합"""
        general_parts = []
        
        for result in result_data:
            agent_id = result["agent_id"]
            data = result["data"]
            
            general_parts.append(str(data))  # agent_id 제거
        
        return "\n".join(general_parts) 