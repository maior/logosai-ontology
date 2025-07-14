"""
🔗 Result Integration Module
결과 통합 모듈

다중 에이전트 실행 결과를 의미론적으로 통합하고 분석
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, WorkflowPlan, AgentExecutionResult,
    ExecutionStatus, get_system_metrics
)
from ..core.llm_manager import get_ontology_llm_manager, OntologyLLMType


class ResultIntegrator:
    """🔗 결과 통합기 - 다중 에이전트 결과의 지능적 통합"""
    
    def __init__(self):
        self.llm_manager = get_ontology_llm_manager()
        self.metrics = get_system_metrics()
        
        logger.info("🔗 결과 통합기 초기화 완료")
    
    async def integrate_results(self, 
                              execution_results: List[AgentExecutionResult],
                              workflow_plan: WorkflowPlan,
                              semantic_query: SemanticQuery) -> Dict[str, Any]:
        """메인 결과 통합 함수"""
        try:
            logger.info(f"🔄 결과 통합 시작 - {len(execution_results)}개 결과")
            
            # 1. 결과 분류 및 전처리
            classified_results = self._classify_results(execution_results)
            
            # 2. 결과 유형별 통합
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
            
            # 3. 최종 통합 결과 생성
            final_result = await self._create_final_integration(
                integration_result, semantic_query, workflow_plan
            )
            
            logger.info(f"✅ 결과 통합 완료: {final_result.get('success', False)}")
            return final_result
            
        except Exception as e:
            logger.error(f"결과 통합 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_results': execution_results
            }
    
    async def integrate_agent_results(self, 
                                    original_query: str,
                                    agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """에이전트 결과 통합 - 테스트용 인터페이스"""
        try:
            logger.info(f"🔄 에이전트 결과 통합 시작 - {len(agent_results)}개 결과")
            
            # 에이전트 결과를 AgentExecutionResult 형식으로 변환
            execution_results = []
            for result in agent_results:
                # 필요한 속성들을 가진 객체 생성
                class AgentResult:
                    def __init__(self, agent_id, data, success, execution_time, confidence, metadata=None):
                        self.agent_id = agent_id
                        self.data = data
                        self.success = success
                        self.execution_time = execution_time
                        self.confidence = confidence
                        self.metadata = metadata or {}
                        self.error_message = None if success else "실행 실패"
                
                exec_result = AgentResult(
                    agent_id=result.get('agent_id', 'unknown'),
                    data=result.get('data', {}),
                    success=result.get('success', False),
                    execution_time=result.get('execution_time', 0),
                    confidence=result.get('confidence', 0.8),
                    metadata=result.get('metadata', {})
                )
                execution_results.append(exec_result)
            
            # 결과 분류
            classified_results = self._classify_results(execution_results)
            logger.info(f"📊 분류된 결과: {list(classified_results.keys())}")
            
            # 정보 검색 결과 통합 (모든 결과를 정보로 처리)
            all_contents = []
            for category, results in classified_results.items():
                if results:  # 빈 리스트가 아닌 경우만
                    for result in results:
                        content = self._extract_content_from_data(result['data'], result['agent_id'])
                        if content and content.strip():
                            all_contents.append(content)
                            logger.info(f"✅ {result['agent_id']}에서 내용 추출: {len(content)}자")
            
            if not all_contents:
                logger.warning("⚠️ 추출된 내용이 없습니다.")
                return {
                    "success": False,
                    "message": "에이전트 결과에서 유효한 내용을 찾을 수 없습니다.",
                    "agent_results_count": len(agent_results)
                }
            
            # LLM을 사용한 통합
            logger.info(f"🧠 LLM 통합 시작 - {len(all_contents)}개 내용")
            integrated_content = await self._llm_integrate_information(all_contents)
            
            # 최종 결과 구성
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
            
            logger.info(f"✅ 에이전트 결과 통합 완료")
            return final_result
            
        except Exception as e:
            logger.error(f"에이전트 결과 통합 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_query": original_query,
                "agent_results_count": len(agent_results)
            }
    
    def _classify_results(self, execution_results: List[AgentExecutionResult]) -> Dict[str, List[Dict[str, Any]]]:
        """결과를 유형별로 분류"""
        classified = {
            'information': [],
            'analysis': [],
            'calculation': [],
            'visualization': [],
            'general': []
        }
        
        for result in execution_results:
            # 결과 데이터 형식 통일
            result_data = {
                "agent_id": result.agent_id,
                "data": result.data if hasattr(result, 'data') else result.result_data,
                "success": result.success,
                "execution_time": result.execution_time,
                "confidence": getattr(result, 'confidence', 0.8),
                "metadata": result.metadata
            }
            
            # 에이전트 ID 기반 분류
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
        """정보 검색 결과 통합"""
        logger.info(f"🔍 정보 검색 결과 통합 시작 - {len(results)}개 결과")
        
        contents = []
        for result in results:
            agent_id = result["agent_id"]
            data = result["data"]
            
            content = self._extract_content_from_data(data, agent_id)
            if content and str(content).strip():
                contents.append(content)
                logger.info(f"  ✅ 내용 추가: {len(str(content))}자")
        
        # LLM을 사용한 고급 통합
        if len(contents) > 1:
            integrated_content = await self._llm_integrate_information(contents)
            if integrated_content:
                return integrated_content
        
        # 폴백: 단순 결합
        final_content = "\n\n".join(contents) if contents else "검색 결과를 찾을 수 없습니다."
        logger.info(f"🔍 정보 검색 결과 통합 완료 - 최종 길이: {len(final_content)}자")
        
        return final_content
    
    async def _integrate_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """분석 결과 통합"""
        logger.info(f"📊 분석 결과 통합 시작 - {len(results)}개 결과")
        
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
            return "분석 결과를 생성할 수 없습니다."
        
        # LLM을 사용한 분석 통합
        integrated_analysis = await self._llm_integrate_analysis(analysis_contents)
        
        logger.info(f"📊 분석 결과 통합 완료")
        return integrated_analysis
    
    async def _integrate_calculation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """계산 결과 통합"""
        logger.info(f"🔢 계산 결과 통합 시작 - {len(results)}개 결과")
        
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
            return {"error": "계산 결과를 찾을 수 없습니다."}
        
        integrated_calc = {
            'calculations': calculations,
            'summary': self._create_calculation_summary(calculations),
            'total_results': len(calculations)
        }
        
        logger.info(f"🔢 계산 결과 통합 완료")
        return integrated_calc
    
    async def _integrate_visualization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시각화 결과 통합"""
        logger.info(f"📈 시각화 결과 통합 시작 - {len(results)}개 결과")
        
        visualizations = []
        
        for result in results:
            agent_id = result["agent_id"]
            data = result["data"]
            
            viz_data = self._extract_visualization_data(data, agent_id)
            if viz_data:
                visualizations.append({
                    'agent': agent_id,
                    'data': viz_data,
                    'type': self._detect_visualization_type(viz_data),
                    'confidence': result.get('confidence', 0.8)
                })
        
        if not visualizations:
            return {"error": "시각화 결과를 찾을 수 없습니다."}
        
        integrated_viz = {
            'visualizations': visualizations,
            'summary': f"{len(visualizations)}개의 시각화 생성됨",
            'types': [viz['type'] for viz in visualizations]
        }
        
        logger.info(f"📈 시각화 결과 통합 완료")
        return integrated_viz
    
    def _extract_content_from_data(self, data: Any, agent_id: str) -> Optional[str]:
        """데이터에서 텍스트 콘텐츠 추출 - 개선된 버전"""
        if isinstance(data, dict):
            logger.debug(f"📊 {agent_id} 데이터 구조: {list(data.keys())}")
            
            # 1차: 직접적인 답변 키들 확인
            for key in ['answer', 'content', 'text', 'result', 'response']:
                if key in data:
                    potential_content = data[key]
                    
                    # 중첩된 딕셔너리인 경우 재귀적으로 처리
                    if isinstance(potential_content, dict):
                        # result가 딕셔너리이고 그 안에 answer나 content가 있는 경우
                        if 'answer' in potential_content:
                            return str(potential_content['answer'])
                        elif 'content' in potential_content:
                            return str(potential_content['content'])
                        elif 'result' in potential_content:
                            return str(potential_content['result'])
                        else:
                            # 딕셔너리 전체를 문자열로 변환하되 깔끔하게
                            return self._dict_to_readable_text(potential_content)
                    elif isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                        return potential_content.strip()
            
            # 2차: 에이전트별 특화된 키 확인
            if 'weather' in agent_id.lower():
                # 날씨 에이전트 특화 처리
                weather_info = self._extract_weather_info(data)
                if weather_info:
                    return weather_info
            elif 'currency' in agent_id.lower() or 'exchange' in agent_id.lower():
                # 환율 에이전트 특화 처리
                currency_info = self._extract_currency_info(data)
                if currency_info:
                    return currency_info
            elif 'crawler' in agent_id.lower() or 'stock' in agent_id.lower():
                # 주가/크롤링 에이전트 특화 처리
                stock_info = self._extract_stock_info(data)
                if stock_info:
                    return stock_info
            
            # 3차: 기타 키들 확인
            for key in ['message', 'output', 'data', 'value', 'info']:
                if key in data:
                    potential_content = data[key]
                    if isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                        return potential_content.strip()
                    elif isinstance(potential_content, dict):
                        return self._dict_to_readable_text(potential_content)
            
            # 4차: 전체 딕셔너리를 읽기 쉬운 텍스트로 변환
            return self._dict_to_readable_text(data)
        else:
            return str(data).strip() if str(data).strip() else None

    def _extract_weather_info(self, data: Dict[str, Any]) -> Optional[str]:
        """날씨 정보 추출"""
        try:
            info_parts = []
            
            # 기본 날씨 정보
            if 'temperature' in data:
                info_parts.append(f"기온: {data['temperature']}")
            if 'humidity' in data:
                info_parts.append(f"습도: {data['humidity']}")
            if 'condition' in data:
                info_parts.append(f"날씨: {data['condition']}")
            if 'wind' in data:
                info_parts.append(f"바람: {data['wind']}")
            
            return ", ".join(info_parts) if info_parts else None
        except:
            return None

    def _extract_currency_info(self, data: Dict[str, Any]) -> Optional[str]:
        """환율 정보 추출"""
        try:
            info_parts = []
            
            if 'rate' in data:
                info_parts.append(f"환율: {data['rate']}")
            if 'change' in data:
                info_parts.append(f"변동: {data['change']}")
            if 'change_percent' in data:
                info_parts.append(f"변동률: {data['change_percent']}")
            
            return ", ".join(info_parts) if info_parts else None
        except:
            return None

    def _extract_stock_info(self, data: Dict[str, Any]) -> Optional[str]:
        """주가 정보 추출"""
        try:
            # 에러가 있는 경우 처리
            if 'error' in data:
                error_msg = data.get('error', '알 수 없는 오류')
                return f"주가 정보 조회 실패: {error_msg}"
            
            info_parts = []
            
            if 'price' in data:
                info_parts.append(f"현재가: {data['price']}")
            if 'change' in data:
                info_parts.append(f"전일대비: {data['change']}")
            if 'change_percent' in data:
                info_parts.append(f"등락률: {data['change_percent']}")
            if 'volume' in data:
                info_parts.append(f"거래량: {data['volume']}")
            
            return ", ".join(info_parts) if info_parts else None
        except:
            return None

    def _dict_to_readable_text(self, data: Dict[str, Any]) -> str:
        """딕셔너리를 읽기 쉬운 텍스트로 변환"""
        try:
            readable_parts = []
            
            for key, value in data.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    readable_parts.append(f"{key}: {value.strip()}")
                elif isinstance(value, (int, float)):
                    readable_parts.append(f"{key}: {value}")
                elif isinstance(value, dict) and value:
                    # 중첩 딕셔너리는 간단히 표현
                    readable_parts.append(f"{key}: [복합 데이터]")
                elif isinstance(value, list) and value:
                    readable_parts.append(f"{key}: [목록 {len(value)}개 항목]")
            
            return "; ".join(readable_parts) if readable_parts else str(data)
        except:
            return str(data)
    
    def _extract_calculation_result(self, data: Any, agent_id: str) -> Optional[Any]:
        """계산 결과 추출"""
        if isinstance(data, dict):
            for key in ['result', 'calculation', 'value', 'answer', 'output']:
                if key in data:
                    return data[key]
        
        # 숫자인지 확인
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
        """시각화 데이터 추출"""
        if isinstance(data, dict):
            for key in ['chart', 'graph', 'plot', 'visualization', 'image', 'figure']:
                if key in data:
                    return data[key]
        
        return data
    
    def _detect_visualization_type(self, viz_data: Any) -> str:
        """시각화 타입 감지"""
        if isinstance(viz_data, dict):
            if 'type' in viz_data:
                return viz_data['type']
            elif 'chart_type' in viz_data:
                return viz_data['chart_type']
        
        return 'unknown'
    
    def _create_calculation_summary(self, calculations: List[Dict[str, Any]]) -> str:
        """계산 결과 요약 생성"""
        if not calculations:
            return "계산 결과 없음"
        
        summary_parts = []
        for calc in calculations:
            agent = calc['agent']
            result = calc['result']
            summary_parts.append(f"{agent}: {result}")
        
        return "; ".join(summary_parts)
    
    async def _llm_integrate_information(self, contents: List[str]) -> str:
        """LLM을 사용한 정보 통합"""
        try:
            # 소스별 내용 구성
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
                    logger.info(f"✅ LLM 정보 통합 성공: {len(integrated)}자")
                    return integrated.strip()
            
        except Exception as e:
            logger.warning(f"LLM 정보 통합 실패: {e}")
        
        # 폴백: 단순 결합
        logger.info("🔄 폴백 모드로 정보 통합")
        return "\n\n".join(contents)
    
    async def _llm_integrate_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """LLM을 사용한 분석 통합"""
        try:
            analysis_texts = []
            for analysis in analyses:
                analysis_texts.append(f"{analysis['agent']} (신뢰도: {analysis['confidence']:.2f}):\n{analysis['content']}")
            
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
            logger.warning(f"LLM 분석 통합 실패: {e}")
        
        return "\n\n".join([a['content'] for a in analyses])
    
    async def _create_final_integration(self, 
                                      integration_result: Dict[str, Any],
                                      semantic_query: SemanticQuery,
                                      workflow_plan: WorkflowPlan) -> Dict[str, Any]:
        """최종 통합 결과 생성"""
        
        # 메인 답변 생성
        main_content = ""
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
            main_content = "요청하신 작업이 완료되었습니다."
        
        # ontology_system.py에서 기대하는 형식으로 결과 구성
        final_result = {
            'success': True,
            'integrated_content': main_content,  # 메인 응답 콘텐츠
            'execution_summary': {
                'query_id': semantic_query.query_id,
                'workflow_id': workflow_plan.plan_id,
                'integration_timestamp': datetime.now().isoformat(),
                'components_processed': list(integration_result.keys()),
                'total_components': len(integration_result)
            },
            'workflow_visualization': "",  # 빈 문자열로 초기화 (필요시 추가)
            'confidence_score': 0.8,  # 기본 신뢰도 점수
            'sources': [],  # 빈 배열로 초기화 (필요시 추가)
            'components': integration_result,  # 원본 통합 결과 보존
            'metadata': {
                'query_id': semantic_query.query_id,
                'workflow_id': workflow_plan.plan_id,
                'integration_timestamp': datetime.now().isoformat()
            }
        }
        
        # 시각화 데이터가 있으면 포함
        if integration_result.get('visualization'):
            final_result['has_visualization'] = True
            final_result['visualization_data'] = integration_result['visualization']
        
        # 계산 데이터가 있으면 포함
        if integration_result.get('calculation'):
            final_result['has_calculation'] = True
            final_result['calculation_data'] = integration_result['calculation']
        
        return final_result 