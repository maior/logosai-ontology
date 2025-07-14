#!/usr/bin/env python3
"""
🚀 완전한 쿼리 처리 시스템 테스트
Complete Query Processing System Test

쿼리 분석 → 에이전트 선택 → 에이전트 실행 → 결과 통합 → 최종 응답 생성
전체 파이프라인을 테스트합니다.
"""

import asyncio
import json
import sys
import time
import uuid
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

# 필요한 모듈 import
try:
    from core.unified_query_processor import get_unified_query_processor
    UNIFIED_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.info(f"⚠️ 통합 프로세서 import 실패: {e}")
    UNIFIED_PROCESSOR_AVAILABLE = False

# LLM 직접 활용을 위한 import
try:
    from core.llm_manager import get_ontology_llm_manager, OntologyLLMType
    LLM_MANAGER_AVAILABLE = True
    logger.info("✅ LLM 관리자 import 성공")
except ImportError as e:
    logger.info(f"⚠️ LLM 관리자 import 실패: {e}")
    LLM_MANAGER_AVAILABLE = False

# ontology result integrator import 추가
try:
    # 프로젝트 루트에서 system 모듈을 import
    sys.path.insert(0, str(Path(__file__).parent))
    from system.result_integration import ResultIntegrator
    RESULT_INTEGRATOR_AVAILABLE = True
    logger.info("✅ ontology ResultIntegrator import 성공")
except ImportError as e:
    logger.info(f"⚠️ 결과 통합기 import 실패: {e}")
    RESULT_INTEGRATOR_AVAILABLE = False


class RealAgent:
    """실제 에이전트 클래스 - 서버 호출"""
    
    def __init__(self, agent_id: str, agent_data: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_data = agent_data
        self.agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
        self.name = agent_data.get('name', agent_id)
        self.endpoint = agent_data.get('endpoint', 'http://localhost:8888/jsonrpc')
        self.api_key = agent_data.get('api_key', '')
        
    async def execute(self, query: str, context: Dict[str, Any] = None, 
                     email: str = "maiordba@gmail.com", 
                     session_id: str = None) -> Dict[str, Any]:
        """실제 에이전트 서버 호출"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        # JSON-RPC 형식으로 페이로드 구성
        payload = {
            'jsonrpc': '2.0',
            'method': 'query',
            'params': {
                'query': query,
                'email': email,
                'sessionid': session_id,
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'context': context or {}
            },
            'id': str(uuid.uuid4())
        }
        
        logger.info(f"   📤 서버 호출: {self.endpoint}")
        logger.info(f"   📝 페이로드: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    execution_time = time.time() - start_time
                    
                    if response.status == 200:
                        result_data = await response.json()
                        logger.info(f"   ✅ 서버 응답 성공 ({execution_time:.2f}초)")
                        logger.info(f"   📥 응답 데이터: {json.dumps(result_data, ensure_ascii=False, indent=2)[:500]}...")
                        
                        # JSON-RPC 오류 체크
                        if result_data.get('error'):
                            error_msg = result_data['error'].get('message', 'Unknown error')
                            logger.info(f"   ⚠️ 서버 에러 응답: {error_msg}")
                            # 서버 오류 시 모의 응답으로 폴백
                            logger.info(f"   🔄 모의 응답으로 폴백")
                            return await self._create_mock_response(query, execution_time)
                        
                        # JSON-RPC 성공 응답 처리
                        if result_data.get('result'):
                            agent_response = result_data['result']
                            return {
                                "success": True,
                                "agent_id": self.agent_id,
                                "result_type": self._determine_result_type(agent_response),
                                "data": agent_response,
                                "execution_time": execution_time,
                                "confidence": self._extract_confidence(agent_response),
                                "server_response": result_data
                            }
                        else:
                            logger.info(f"   ⚠️ 결과 없음, 모의 응답으로 폴백")
                            return await self._create_mock_response(query, execution_time)
                    else:
                        error_text = await response.text()
                        logger.info(f"   ❌ 서버 오류 ({response.status}): {error_text}")
                        
                        return {
                            "success": False,
                            "agent_id": self.agent_id,
                            "error": f"HTTP {response.status}: {error_text}",
                            "execution_time": execution_time
                        }
                        
        except aiohttp.ClientError as e:
            execution_time = time.time() - start_time
            logger.info(f"   ❌ 연결 실패: {str(e)}")
            
            # 연결 실패 시 모의 응답 생성 (폴백)
            logger.info(f"   🔄 모의 응답으로 폴백")
            return await self._create_mock_response(query, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.info(f"   ❌ 예상치 못한 오류: {str(e)}")
            
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _determine_result_type(self, data: Dict[str, Any]) -> str:
        """응답 데이터에서 결과 타입 추론"""
        # 응답 내용 기반 분석
        content = data.get('content', '')
        if isinstance(content, dict):
            content = str(content)
        content_lower = content.lower()
        
        # 에이전트 ID 기반 분석
        if 'weather' in self.agent_id.lower() or self.agent_type == 'WEATHER':
            return "weather_info"
        elif 'currency' in self.agent_id.lower() or self.agent_type == 'CURRENCY':
            return "currency_info"
        elif 'stock' in self.agent_id.lower() or 'crawler' in self.agent_id.lower():
            return "stock_info"
        elif 'search' in self.agent_id.lower() or self.agent_type in ['SEARCH', 'INTERNET_SEARCH']:
            return "search_results"
        # 내용 기반 추론
        elif any(keyword in content_lower for keyword in ['날씨', '기온', '온도', 'weather']):
            return "weather_info"
        elif any(keyword in content_lower for keyword in ['환율', '달러', '원', 'currency']):
            return "currency_info"
        elif any(keyword in content_lower for keyword in ['주가', '주식', 'stock']):
            return "stock_info"
        else:
            return "general_response"
    
    def _extract_confidence(self, data: Dict[str, Any]) -> float:
        """응답 데이터에서 신뢰도 추출"""
        if isinstance(data, dict):
            # metadata에서 신뢰도 찾기
            metadata = data.get('metadata', {})
            if isinstance(metadata, dict):
                for key in ['confidence', 'score', 'reliability', 'accuracy']:
                    if key in metadata:
                        try:
                            return float(metadata[key])
                        except (ValueError, TypeError):
                            pass
            
            # 직접 신뢰도 찾기
            for key in ['confidence', 'score', 'reliability', 'accuracy']:
                if key in data:
                    try:
                        return float(data[key])
                    except (ValueError, TypeError):
                        pass
        
        # 기본 신뢰도
        return 0.85
    
    async def _create_mock_response(self, query: str, execution_time: float) -> Dict[str, Any]:
        """연결 실패 시 모의 응답 생성"""
        
        # 에이전트 타입별 다른 응답 생성
        if 'weather' in self.agent_id.lower() or self.agent_type == 'WEATHER':
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result_type": "weather_info",
                "data": {
                    "type": "text",
                    "content": {
                        "location": "서울",
                        "temperature": "22°C",
                        "humidity": "65%",
                        "condition": "맑음",
                        "forecast": "오늘은 맑은 날씨가 예상됩니다.",
                        "note": "모의 응답 (서버 연결 실패)"
                    },
                    "metadata": {
                        "confidence": 0.95,
                        "source": "mock_server"
                    }
                },
                "execution_time": execution_time,
                "confidence": 0.95,
                "is_mock": True
            }
            
        elif 'currency' in self.agent_id.lower() or self.agent_type == 'CURRENCY':
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result_type": "currency_info",
                "data": {
                    "type": "text",
                    "content": {
                        "pair": "USD/KRW",
                        "rate": "1,320.50",
                        "change": "+5.20",
                        "change_percent": "+0.39%",
                        "last_updated": "2025-06-20 21:00:00",
                        "note": "모의 응답 (서버 연결 실패)"
                    },
                    "metadata": {
                        "confidence": 0.92,
                        "source": "mock_server"
                    }
                },
                "execution_time": execution_time,
                "confidence": 0.92,
                "is_mock": True
            }
            
        elif 'stock' in self.agent_id.lower() or 'crawler' in self.agent_id.lower():
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result_type": "stock_info",
                "data": {
                    "type": "text",
                    "content": {
                        "company": "삼성전자",
                        "symbol": "005930",
                        "price": "71,000",
                        "change": "-500",
                        "change_percent": "-0.70%",
                        "volume": "12,345,678",
                        "market_cap": "424조원",
                        "note": "모의 응답 (서버 연결 실패)"
                    },
                    "metadata": {
                        "confidence": 0.88,
                        "source": "mock_server"
                    }
                },
                "execution_time": execution_time,
                "confidence": 0.88,
                "is_mock": True
            }
            
        else:
            # 일반 에이전트
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result_type": "general_response",
                "data": {
                    "type": "text",
                    "content": {
                        "response": f"{self.name}에서 처리한 결과입니다: {query}",
                        "processing_info": f"쿼리를 {self.agent_type} 에이전트로 처리했습니다.",
                        "note": "모의 응답 (서버 연결 실패)"
                    },
                    "metadata": {
                        "confidence": 0.80,
                        "source": "mock_server"
                    }
                },
                "execution_time": execution_time,
                "confidence": 0.80,
                "is_mock": True
            }


class DirectLLMResultIntegrator:
    """직접 LLM을 활용한 결과 통합기"""
    
    def __init__(self):
        if LLM_MANAGER_AVAILABLE:
            self.llm_manager = get_ontology_llm_manager()
        else:
            self.llm_manager = None
    
    async def integrate_agent_results(self, original_query: str, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """LLM을 직접 활용한 에이전트 결과 통합"""
        try:
            logger.info(f"   🧠 직접 LLM 기반 결과 통합 시작...")
            logger.info(f"   📊 처리할 에이전트 결과: {len(agent_results)}개")
            
            # 성공한 결과만 추출
            successful_results = [r for r in agent_results if r.get('success', False)]
            if not successful_results:
                return {
                    "success": False,
                    "message": "성공한 에이전트 결과가 없습니다.",
                    "agent_results_count": len(agent_results)
                }
            
            # 각 에이전트 결과에서 내용 추출
            extracted_contents = []
            for result in successful_results:
                agent_id = result.get('agent_id', 'unknown')
                data = result.get('data', {})
                
                # 다양한 형태의 데이터에서 텍스트 추출
                content = self._extract_content_from_data(data, agent_id)
                if content and content.strip():
                    extracted_contents.append({
                        'agent_id': agent_id,
                        'content': content,
                        'confidence': result.get('confidence', 0.8)
                    })
                    logger.info(f"   ✅ {agent_id}: {len(content)}자 추출")
            
            if not extracted_contents:
                return {
                    "success": False,
                    "message": "추출된 내용이 없습니다.",
                    "agent_results_count": len(agent_results)
                }
            
            # LLM을 사용한 통합
            if self.llm_manager and len(extracted_contents) > 1:
                logger.info(f"   🤖 LLM 통합 시도...")
                integrated_content = await self._llm_integrate_contents(original_query, extracted_contents)
                integration_method = "LLM"
            else:
                logger.info(f"   📝 단순 결합 방식...")
                integrated_content = self._simple_combine_contents(extracted_contents)
                integration_method = "Simple"
            
            return {
                "success": True,
                "integrated_content": integrated_content,
                "original_query": original_query,
                "agent_results_count": len(agent_results),
                "successful_extractions": len(extracted_contents),
                "processing_summary": {
                    "total_agents": len(agent_results),
                    "successful_agents": len(successful_results),
                    "content_extracted": len(extracted_contents),
                    "integration_method": integration_method
                }
            }
            
        except Exception as e:
            logger.info(f"   ❌ 직접 LLM 통합 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_query": original_query,
                "agent_results_count": len(agent_results)
            }
    
    def _extract_content_from_data(self, data: Any, agent_id: str) -> str:
        """데이터에서 텍스트 콘텐츠 추출"""
        if isinstance(data, dict):
            # 1차: 직접적인 답변 키들 확인
            for key in ['answer', 'content', 'text', 'result', 'response']:
                if key in data:
                    potential_content = data[key]
                    
                    if isinstance(potential_content, dict):
                        # 중첩된 딕셔너리에서 재귀적으로 찾기
                        for inner_key in ['answer', 'content', 'result', 'text']:
                            if inner_key in potential_content:
                                return str(potential_content[inner_key]).strip()
                        
                        # 에이전트별 특화 처리
                        if 'weather' in agent_id.lower():
                            return self._extract_weather_summary(potential_content)
                        elif 'currency' in agent_id.lower():
                            return self._extract_currency_summary(potential_content)
                        elif 'crawler' in agent_id.lower() or 'stock' in agent_id.lower():
                            return self._extract_stock_summary(potential_content)
                        else:
                            return str(potential_content)
                    
                    elif isinstance(potential_content, str) and len(potential_content.strip()) > 0:
                        return potential_content.strip()
            
            # 2차: 전체 딕셔너리를 문자열로 변환
            return str(data)
        else:
            return str(data).strip() if str(data).strip() else ""
    
    def _extract_weather_summary(self, data: Dict) -> str:
        """날씨 정보 요약 추출"""
        try:
            parts = []
            if 'temperature' in data:
                parts.append(f"기온 {data['temperature']}")
            if 'humidity' in data:
                parts.append(f"습도 {data['humidity']}")
            if 'condition' in data:
                parts.append(f"날씨 {data['condition']}")
            
            if parts:
                return f"서울 날씨: {', '.join(parts)}"
            else:
                return "날씨 정보를 확인했습니다."
        except:
            return str(data)
    
    def _extract_currency_summary(self, data: Dict) -> str:
        """환율 정보 요약 추출"""
        try:
            if 'rate' in data:
                rate = data['rate']
                change = data.get('change', '')
                return f"USD/KRW 환율: {rate}원 {change}"
            else:
                return "환율 정보를 확인했습니다."
        except:
            return str(data)
    
    def _extract_stock_summary(self, data: Dict) -> str:
        """주가 정보 요약 추출"""
        try:
            if 'error' in data:
                return f"주가 정보 조회 실패: {data['error']}"
            elif 'price' in data:
                price = data['price']
                change = data.get('change', '')
                return f"삼성전자 주가: {price}원 {change}"
            else:
                return "주가 정보를 확인했습니다."
        except:
            return str(data)
    
    async def _llm_integrate_contents(self, original_query: str, contents: List[Dict]) -> str:
        """LLM을 사용한 내용 통합"""
        try:
            # 소스별 내용 구성
            sources_text = ""
            for i, item in enumerate(contents):
                agent_id = item['agent_id']
                content = item['content']
                confidence = item['confidence']
                sources_text += f"\n{'='*30}\n에이전트 {i+1} ({agent_id}, 신뢰도: {confidence:.2f}):\n{content}\n"
            
            prompt = f"""
사용자가 "{original_query}"라고 요청했습니다.

다음은 여러 에이전트들이 처리한 결과입니다:{sources_text}

**요구사항:**
1. 위 정보들을 종합하여 사용자 친화적인 통합 답변을 생성해주세요
2. 각 정보의 핵심 내용을 모두 포함하되, 중복은 제거해주세요
3. 마크다운이나 HTML 태그는 제거하고 깔끔한 텍스트로 변환해주세요
4. 정보가 부족하거나 오류가 있는 부분은 명확히 표시해주세요
5. 최종 답변은 한국어로 작성해주세요

**답변 형식:**
- 간결하고 명확하게 작성
- 각 정보를 구분해서 제시
- 사용자가 이해하기 쉽게 구성
"""
            
            # LLM 호출
            llm = self.llm_manager.get_llm(OntologyLLMType.RESULT_INTEGRATOR)
            if llm:
                response = await llm.ainvoke(prompt)
                
                if hasattr(response, 'content'):
                    integrated = response.content
                else:
                    integrated = str(response)
                
                if integrated and isinstance(integrated, str) and len(integrated.strip()) > 0:
                    logger.info(f"   ✅ LLM 통합 성공: {len(integrated)}자")
                    return integrated.strip()
            
            logger.info(f"   ⚠️ LLM 통합 실패, 단순 결합으로 폴백")
            return self._simple_combine_contents(contents)
            
        except Exception as e:
            logger.info(f"   ❌ LLM 통합 오류: {e}")
            return self._simple_combine_contents(contents)
    
    def _simple_combine_contents(self, contents: List[Dict]) -> str:
        """단순 내용 결합"""
        parts = ["요청하신 정보를 확인했습니다.\n"]
        
        for i, item in enumerate(contents, 1):
            agent_id = item['agent_id']
            content = item['content']
            
            # 에이전트 이름을 사용자 친화적으로 변환
            if 'weather' in agent_id:
                agent_name = "날씨"
            elif 'currency' in agent_id:
                agent_name = "환율"
            elif 'crawler' in agent_id or 'stock' in agent_id:
                agent_name = "주가"
            else:
                agent_name = agent_id
            
            parts.append(f"{i}. {agent_name} 정보:")
            parts.append(f"   {content}\n")
        
        return "\n".join(parts)


class CompleteQueryProcessor:
    """완전한 쿼리 처리 시스템"""
    
    def __init__(self):
        self.agents_json_path = "/Users/maior/Development/skku/Logos/logosai/logosai/examples/configs/agents.json"
        self.test_agents_info = []
        self.available_agents = []
        self.real_agents = {}  # 실제 에이전트들
        self.session_id = str(uuid.uuid4())  # 세션 ID 생성
        
        # ontology의 ResultIntegrator 사용 시도
        if RESULT_INTEGRATOR_AVAILABLE:
            try:
                self.result_integrator = ResultIntegrator()
                logger.info("✅ ontology ResultIntegrator 로드 완료")
            except Exception as e:
                logger.info(f"⚠️ ontology ResultIntegrator 초기화 실패: {e}")
                self.result_integrator = DirectLLMResultIntegrator()
                logger.info("✅ 직접 LLM ResultIntegrator 사용")
        else:
            # 직접 LLM 결과 통합기 사용
            self.result_integrator = DirectLLMResultIntegrator()
            logger.info("✅ 직접 LLM ResultIntegrator 로드 완료")
        
        if UNIFIED_PROCESSOR_AVAILABLE:
            self.unified_processor = get_unified_query_processor()
        else:
            self.unified_processor = None
    
    def load_agents_info(self):
        """에이전트 정보 로드 및 실제 에이전트 생성"""
        try:
            if Path(self.agents_json_path).exists():
                with open(self.agents_json_path, 'r', encoding='utf-8') as f:
                    agents_data = json.load(f)
                
                for agent in agents_data.get('agents', []):
                    agent_id = agent.get('agent_id')
                    if agent_id:
                        self.test_agents_info.append({
                            'agent_id': agent_id,
                            'agent_data': agent
                        })
                        self.available_agents.append(agent_id)
                        
                        # 실제 에이전트 생성
                        self.real_agents[agent_id] = RealAgent(agent_id, agent)
                
                logger.info(f"✅ {len(self.test_agents_info)}개 에이전트 로드 및 실제 에이전트 생성 완료")
                logger.info(f"📋 로드된 에이전트:")
                for i, info in enumerate(self.test_agents_info[:5], 1):  # 처음 5개만 표시
                    agent_data = info['agent_data']
                    logger.info(f"   {i}. {info['agent_id']}: {agent_data.get('name', 'Unknown')}")
                    logger.info(f"      엔드포인트: {agent_data.get('endpoint', 'N/A')}")
                
                if len(self.test_agents_info) > 5:
                    logger.info(f"   ... 외 {len(self.test_agents_info) - 5}개")
                
            else:
                logger.info(f"❌ agents.json 파일을 찾을 수 없음: {self.agents_json_path}")
                self._create_fallback_agents()
                
        except Exception as e:
            logger.info(f"❌ agents.json 로드 실패: {e}")
            self._create_fallback_agents()
        
        # 통합 프로세서에 에이전트 정보 설정
        if self.unified_processor:
            self.unified_processor.set_installed_agents_info(self.test_agents_info)
    
    def _create_fallback_agents(self):
        """폴백 에이전트 생성"""
        fallback_agents = [
            {
                'agent_id': 'llm_search_agent',
                'agent_data': {
                    'name': 'LLM 검색 에이전트',
                    'description': 'LLM을 활용한 검색 에이전트',
                    'endpoint': 'http://localhost:8888/jsonrpc',
                    'metadata': {'agent_type': 'LLM_SEARCH'},
                    'capabilities': [{'name': '지식 검색', 'description': 'LLM 지식 기반 검색'}]
                }
            },
            {
                'agent_id': 'analysis_agent',
                'agent_data': {
                    'name': '데이터 분석 에이전트',
                    'description': '데이터 분석 전문 에이전트',
                    'endpoint': 'http://localhost:8888/jsonrpc',
                    'metadata': {'agent_type': 'ANALYSIS'},
                    'capabilities': [{'name': '데이터 분석', 'description': '통계 및 텍스트 분석'}]
                }
            }
        ]
        
        for agent_info in fallback_agents:
            agent_id = agent_info['agent_id']
            self.test_agents_info.append(agent_info)
            self.available_agents.append(agent_id)
            self.real_agents[agent_id] = RealAgent(agent_id, agent_info['agent_data'])
        
        logger.info(f"🔄 폴백 에이전트 {len(fallback_agents)}개 생성 완료")
    
    async def process_complete_query(self, query: str) -> Dict[str, Any]:
        """완전한 쿼리 처리 파이프라인"""
        
        logger.info(f"\n🚀 완전한 쿼리 처리 시작")
        logger.info(f"📝 쿼리: {query}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1단계: 쿼리 분석 및 에이전트 선택
            logger.info(f"\n📊 1단계: 쿼리 분석 및 에이전트 선택")
            
            if not self.unified_processor:
                raise Exception("통합 프로세서를 사용할 수 없습니다.")
            
            analysis_result = await self.unified_processor.process_unified_query(
                query, self.available_agents
            )
            
            query_analysis = analysis_result.get('query_analysis', {})
            agent_mappings = analysis_result.get('agent_mappings', [])
            execution_plan = analysis_result.get('execution_plan', {})
            
            logger.info(f"   ✅ 쿼리 분석 완료")
            logger.info(f"   📊 복잡도: {query_analysis.get('complexity', 'unknown')}")
            logger.info(f"   🎯 선택된 에이전트: {len(agent_mappings)}개")
            
            # 실행 전략 표시
            strategy = execution_plan.get('strategy', 'unknown')
            estimated_time = execution_plan.get('estimated_time', 0)
            logger.info(f"   ⚡ 실행 전략: {strategy}")
            logger.info(f"   ⏱️ 예상 처리 시간: {estimated_time:.1f}초")
            
            if strategy == 'parallel':
                logger.info(f"   🔀 병렬 처리: {len(agent_mappings)}개 에이전트 동시 실행")
            elif strategy == 'sequential':
                logger.info(f"   🔗 순차 처리: {len(agent_mappings)}개 에이전트 순서대로 실행")
            elif strategy == 'hybrid':
                logger.info(f"   🎭 하이브리드 처리: 의존성에 따른 혼합 실행")
            else:
                logger.info(f"   🎯 단일 처리: 1개 에이전트 실행")
            
            # 에이전트별 쿼리 생성 로그 출력
            logger.info(f"\n🔍 에이전트별 최적화된 쿼리 생성:")
            for i, mapping in enumerate(agent_mappings, 1):
                agent_id = mapping.get('selected_agent', 'unknown')
                optimized_message = mapping.get('optimized_message', query)
                confidence = mapping.get('confidence', 0.0)
                
                try:
                    confidence_float = float(confidence)
                except (ValueError, TypeError):
                    confidence_float = 0.0
                
                logger.info(f"   {i}. 🤖 에이전트: {agent_id}")
                logger.info(f"      📝 최적화된 쿼리: {optimized_message}")
                logger.info(f"      📊 신뢰도: {confidence_float:.2f}")
                logger.info(f"      🎯 컨텍스트: {mapping.get('context', {})}")
                logger.info(f"      ─" * 50)
            
            # 2단계: 에이전트 실행
            strategy = execution_plan.get('strategy', 'parallel')
            logger.info(f"\n⚡ 2단계: 에이전트 실행 ({strategy})")
            
            agent_results = []
            execution_tasks = []
            
            # 실행 전략에 따른 처리
            if strategy == 'sequential':
                logger.info(f"   🔗 순차 실행 모드 (이전 결과를 다음 에이전트에 전달)")
                # 순차 실행 - 이전 결과를 다음 에이전트에 전달
                agent_results = await self._execute_agents_sequentially_with_context(agent_mappings, query, analysis_result)
            elif strategy == 'hybrid':
                logger.info(f"   🎭 하이브리드 실행 모드 (의존성 고려한 혼합)")
                agent_results = await self._execute_agents_hybrid_mode(agent_mappings, analysis_result, query)
            else:
                # 병렬 실행 (parallel, single_agent)
                if strategy == 'parallel':
                    logger.info(f"   🔀 병렬 실행 모드")
                else:
                    logger.info(f"   🎯 단일 실행 모드")
                
                for mapping in agent_mappings:
                    agent_id = mapping.get('selected_agent')
                    optimized_message = mapping.get('optimized_message', query)
                    
                    if agent_id in self.real_agents:
                        logger.info(f"   🤖 {agent_id} 실행 중...")
                        task = self._execute_agent_async(agent_id, optimized_message, mapping)
                        execution_tasks.append(task)
                    else:
                        logger.info(f"   ⚠️ {agent_id} 에이전트를 찾을 수 없습니다.")
                
                # 병렬 실행
                if execution_tasks:
                    agent_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # 예외 처리
                valid_results = []
                for result in agent_results:
                    if isinstance(result, Exception):
                        logger.info(f"   ❌ 에이전트 실행 오류: {result}")
                    else:
                        valid_results.append(result)
                        logger.info(f"   ✅ {result.get('agent_id')} 실행 완료")
                
                agent_results = valid_results
            
            # 3단계: 결과 통합
            logger.info(f"\n🔄 3단계: 결과 통합 및 최종 응답 생성")
            
            if self.result_integrator:
                # ontology의 ResultIntegrator 사용
                logger.info(f"   🧠 LLM 기반 결과 통합 시작...")
                integration_result = await self.result_integrator.integrate_agent_results(
                    query, agent_results
                )
                
                logger.info(f"   ✅ 통합 완료")
                logger.info(f"   📊 통합 결과:")
                logger.info(f"      - 성공 여부: {integration_result.get('success', False)}")
                logger.info(f"      - 처리된 에이전트: {integration_result.get('agent_results_count', 0)}개")
                logger.info(f"      - 추출된 내용: {integration_result.get('successful_extractions', 0)}개")
                logger.info(f"      - 통합 방법: {integration_result.get('processing_summary', {}).get('integration_method', 'Unknown')}")
                
                # 통합된 응답 추출
                final_response = integration_result.get('integrated_content', '결과를 생성할 수 없습니다.')
                
            else:
                # 폴백: 간단한 결과 통합
                logger.info(f"   ⚠️ ontology ResultIntegrator 사용 불가, 폴백 모드")
                integration_result = await self._fallback_integrate_results(query, agent_results, query_analysis)
                final_response = integration_result.get("final_response", "결과를 생성할 수 없습니다.")
            
            total_time = time.time() - start_time
            
            logger.info(f"   ⏱️ 총 처리 시간: {total_time:.2f}초")
            
            # 최종 결과 구성
            final_result = {
                "success": True,
                "query": query,
                "total_processing_time": total_time,
                "analysis_result": analysis_result,
                "agent_results": agent_results,
                "integration_result": integration_result,
                "final_response": final_response,
                "metadata": {
                    "agents_executed": len(agent_results),
                    "integration_method": "LLM" if self.result_integrator else "Fallback"
                }
            }
            
            return final_result
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.info(f"❌ 처리 실패: {e}")
            import traceback
            traceback.logger.info_exc()
            
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "processing_time": error_time
            }
    
    async def _execute_agents_sequentially_with_context(self, agent_mappings: List[Dict], original_query: str, analysis_result: Dict) -> List[Dict[str, Any]]:
        """순차 실행 - 이전 결과를 다음 에이전트에 전달"""
        agent_results = []
        context_data = {}  # 이전 결과들을 저장
        
        # 의존성 분석 결과 가져오기
        dependency_analysis = analysis_result.get('dependency_analysis', {})
        execution_order = dependency_analysis.get('execution_order', [])
        
        # 실행 순서가 있으면 그 순서대로, 없으면 매핑 순서대로
        if execution_order and isinstance(execution_order[0], list):
            # 실행 순서가 2차원 배열인 경우 (병렬 그룹들)
            ordered_mappings = []
            for group in execution_order:
                for task_id in group:
                    for mapping in agent_mappings:
                        if mapping.get('task_id') == task_id:
                            ordered_mappings.append(mapping)
                            break
        else:
            # 실행 순서가 1차원 배열이거나 없는 경우
            ordered_mappings = agent_mappings
        
        for i, mapping in enumerate(ordered_mappings, 1):
            agent_id = mapping.get('selected_agent')
            task_id = mapping.get('task_id', f'task_{i}')
            optimized_message = mapping.get('optimized_message', original_query)
            
            # 이전 결과를 활용한 컨텍스트 생성
            if i > 1 and context_data:
                enhanced_context = self._create_enhanced_context(
                    optimized_message, context_data, task_id, mapping
                )
                logger.debug('--------------------------------')
                logger.info(f"   🔗 [{i}/{len(ordered_mappings)}] {agent_id}: 이전 결과 활용")
                logger.info(f"      📝 컨텍스트: {enhanced_context[:100]}...")
            else:
                logger.debug('--------------------------------')
                enhanced_context = optimized_message
                logger.info(f"   🎯 [{i}/{len(ordered_mappings)}] {agent_id}: 첫 번째 작업")
            
            if agent_id in self.real_agents:
                logger.info(f"   🤖 {agent_id} 실행 중...")
                
                # 컨텍스트 정보 포함하여 실행
                enhanced_mapping = mapping.copy()
                enhanced_mapping['context_data'] = context_data
                enhanced_mapping['enhanced_query'] = enhanced_context
                
                result = await self._execute_agent_async(agent_id, enhanced_context, enhanced_mapping)
                
                # 결과 저장
                agent_results.append(result)
                
                # 다음 에이전트를 위해 결과 저장
                if result.get('success', False):
                    context_data[task_id] = {
                        'agent_id': agent_id,
                        'result': result,
                        'summary': self._extract_result_summary(result)
                    }
                    logger.info(f"   ✅ {agent_id} 완료 - 결과를 다음 에이전트에 전달")
                else:
                    logger.info(f"   ❌ {agent_id} 실패 - 결과 전달 불가")
            else:
                logger.info(f"   ⚠️ {agent_id} 에이전트를 찾을 수 없습니다.")
        
        return agent_results

    async def _execute_agents_hybrid_mode(self, agent_mappings: List[Dict], analysis_result: Dict, original_query: str) -> List[Dict[str, Any]]:
        """하이브리드 실행 - 의존성에 따라 병렬과 순차 혼합"""
        dependency_analysis = analysis_result.get('dependency_analysis', {})
        parallel_groups = dependency_analysis.get('parallel_groups', [])
        
        if not parallel_groups:
            # 병렬 그룹이 없으면 순차 실행으로 폴백
            return await self._execute_agents_sequentially_with_context(agent_mappings, original_query, analysis_result)
        
        agent_results = []
        context_data = {}
        
        logger.info(f"   🎭 하이브리드 실행: {len(parallel_groups)}개 그룹")
        
        for group_idx, group_task_ids in enumerate(parallel_groups, 1):
            logger.info(f"   📦 그룹 {group_idx}: {len(group_task_ids)}개 작업")
            
            # 현재 그룹의 매핑들 찾기
            group_mappings = []
            for task_id in group_task_ids:
                for mapping in agent_mappings:
                    if mapping.get('task_id') == task_id:
                        group_mappings.append(mapping)
                        break
            
            if len(group_mappings) == 1:
                # 단일 작업 - 순차 실행
                mapping = group_mappings[0]
                agent_id = mapping.get('selected_agent')
                task_id = mapping.get('task_id')
                optimized_message = mapping.get('optimized_message', original_query)
                
                # 이전 결과 활용
                if context_data:
                    enhanced_context = self._create_enhanced_context(
                        optimized_message, context_data, task_id, mapping
                    )
                    logger.info(f"      🔗 {agent_id}: 이전 결과 활용")
                else:
                    enhanced_context = optimized_message
                    logger.info(f"      🎯 {agent_id}: 독립 실행")
                
                if agent_id in self.real_agents:
                    enhanced_mapping = mapping.copy()
                    enhanced_mapping['context_data'] = context_data
                    enhanced_mapping['enhanced_query'] = enhanced_context
                    
                    result = await self._execute_agent_async(agent_id, enhanced_context, enhanced_mapping)
                    agent_results.append(result)
                    
                    if result.get('success', False):
                        context_data[task_id] = {
                            'agent_id': agent_id,
                            'result': result,
                            'summary': self._extract_result_summary(result)
                        }
                        logger.info(f"      ✅ {agent_id} 완료")
                    else:
                        logger.info(f"      ❌ {agent_id} 실패")
            else:
                # 다중 작업 - 병렬 실행
                logger.info(f"      🔀 {len(group_mappings)}개 작업 병렬 실행")
                tasks = []
                for mapping in group_mappings:
                    agent_id = mapping.get('selected_agent')
                    optimized_message = mapping.get('optimized_message', original_query)
                    
                    if agent_id in self.real_agents:
                        enhanced_mapping = mapping.copy()
                        enhanced_mapping['context_data'] = context_data
                        task = self._execute_agent_async(agent_id, optimized_message, enhanced_mapping)
                        tasks.append(task)
                
                if tasks:
                    group_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in group_results:
                        if isinstance(result, Exception):
                            logger.info(f"      ❌ 병렬 실행 오류: {result}")
                        else:
                            agent_results.append(result)
                            
                            # 컨텍스트 데이터 업데이트
                            if result.get('success', False):
                                task_id = result.get('task_id', f'task_{len(context_data)}')
                                context_data[task_id] = {
                                    'agent_id': result.get('agent_id'),
                                    'result': result,
                                    'summary': self._extract_result_summary(result)
                                }
                                logger.info(f"      ✅ {result.get('agent_id')} 완료")
        
        return agent_results

    def _create_enhanced_context(self, original_message: str, context_data: Dict, current_task_id: str, mapping: Dict) -> str:
        """이전 결과를 활용한 향상된 컨텍스트 생성"""
        if not context_data:
            return original_message
        
        # 이전 결과들 요약
        context_summary = []
        for task_id, data in context_data.items():
            agent_id = data.get('agent_id', 'unknown')
            summary = data.get('summary', '결과 없음')
            context_summary.append(f"- {agent_id}: {summary}")
        
        # 향상된 쿼리 생성
        enhanced_query = f"""이전 작업 결과를 참고하여 다음 작업을 수행하세요:

이전 결과:
{chr(10).join(context_summary)}

현재 작업: {original_message}

위의 이전 결과를 활용하여 더 정확하고 구체적인 답변을 제공해주세요."""

        return enhanced_query

    def _extract_result_summary(self, result: Dict[str, Any]) -> str:
        """결과에서 요약 정보 추출"""
        if not result.get('success', False):
            return "처리 실패"
        
        data = result.get('data', {})
        
        # 다양한 형태의 데이터에서 요약 추출
        if isinstance(data, dict):
            # 직접적인 답변 키들 확인
            for key in ['answer', 'content', 'result', 'response']:
                if key in data:
                    content = data[key]
                    if isinstance(content, dict):
                        # 중첩된 딕셔너리에서 텍스트 추출
                        for inner_key in ['response', 'content', 'result']:
                            if inner_key in content:
                                text = str(content[inner_key])
                                return text[:100] + "..." if len(text) > 100 else text
                    elif isinstance(content, str):
                        return content[:100] + "..." if len(content) > 100 else content
            
            # 전체 데이터를 문자열로 변환
            data_str = str(data)
            return data_str[:100] + "..." if len(data_str) > 100 else data_str
        else:
            data_str = str(data)
            return data_str[:100] + "..." if len(data_str) > 100 else data_str

    async def _execute_agent_async(self, agent_id: str, message: str, context: Dict) -> Dict[str, Any]:
        """비동기 에이전트 실행 - 실제 서버 호출"""
        if agent_id in self.real_agents:
            return await self.real_agents[agent_id].execute(
                message, 
                context, 
                email="maiordba@gmail.com",
                session_id=self.session_id
            )
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "error": "Agent not found"
            }
    
    def display_final_result(self, result: Dict[str, Any]):
        """최종 결과 표시"""
        logger.info(f"\n" + "=" * 60)
        logger.info(f"🎯 최종 처리 결과")
        logger.info(f"=" * 60)
        
        if result.get("success"):
            logger.info(f"✅ 처리 성공")
            logger.info(f"📝 원본 쿼리: {result.get('query')}")
            logger.info(f"⏱️ 총 처리 시간: {result.get('total_processing_time', 0):.2f}초")
            logger.info(f"🤖 실행된 에이전트: {result.get('metadata', {}).get('agents_executed', 0)}개")
            logger.info(f"📈 전체 신뢰도: {result.get('metadata', {}).get('overall_confidence', 0):.2f}")
            
            logger.info(f"\n📋 최종 사용자 응답:")
            logger.info(f"-" * 40)
            logger.info(result.get("final_response", "응답을 생성할 수 없습니다."))
            logger.info(f"-" * 40)
            
        else:
            logger.info(f"❌ 처리 실패")
            logger.info(f"📝 쿼리: {result.get('query')}")
            logger.info(f"❌ 오류: {result.get('error')}")
            logger.info(f"⏱️ 처리 시간: {result.get('processing_time', 0):.2f}초")

    async def _fallback_integrate_results(self, query: str, agent_results: List[Dict], query_analysis: Dict) -> Dict[str, Any]:
        """폴백 결과 통합 (ontology ResultIntegrator 사용 불가 시)"""
        logger.info(f"   🔄 폴백 결과 통합 시작...")
        
        successful_results = [r for r in agent_results if r.get('success', False)]
        if not successful_results:
            return {
                "success": False,
                "final_response": "모든 에이전트 실행이 실패했습니다.",
                "agent_results_count": len(agent_results)
            }
        
        # 간단한 결과 결합
        response_parts = ["요청하신 정보를 확인했습니다.\n\n"]
        
        for i, result in enumerate(successful_results, 1):
            agent_id = result.get('agent_id', 'Unknown')
            data = result.get('data', {})
            
            response_parts.append(f"{i}. {agent_id} 결과:\n")
            
            # 간단한 데이터 추출
            if isinstance(data, dict):
                content = data.get('content', data.get('result', data.get('answer', str(data))))
                if isinstance(content, dict):
                    content = str(content)
                response_parts.append(f"{content}\n\n")
            else:
                response_parts.append(f"{str(data)}\n\n")
        
        final_response = "".join(response_parts)
        
        return {
            "success": True,
            "final_response": final_response,
            "agent_results_count": len(agent_results),
            "successful_results": len(successful_results)
        }


async def run_complete_test():
    """완전한 테스트 실행"""
    logger.info("🚀 완전한 쿼리 처리 시스템 테스트")
    logger.info("=" * 60)
    
    processor = CompleteQueryProcessor()
    processor.load_agents_info()
    
    # 테스트 쿼리 (1개만)
    #test_query = "오늘 날씨를 확인하고, 환율 정보도 알려주고, 삼성전자 주가 정보도 알려줘."
    #test_query = "오늘 금 시세 확인하고 1온스당 원화 가격 계산해서 정리해줘."
    test_query = "AI 기술 트렌드를 조사해서 분석하고 시각적 보고서로 만들어줘"
    
    logger.info(f"\n🧪 단일 테스트 실행")
    logger.info("=" * 60)
    
    result = await processor.process_complete_query(test_query)
    processor.display_final_result(result)
    
    logger.info(f"\n✅ 모든 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(run_complete_test()) 