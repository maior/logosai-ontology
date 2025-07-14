#!/usr/bin/env python3
"""
🧠 LLM 기반 결과 통합 테스트
Test LLM-based Result Integration

ontology의 ResultIntegrator를 직접 테스트합니다.
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from system.result_integration import ResultIntegrator


async def test_llm_result_integration():
    """LLM 기반 결과 통합 테스트"""
    
    print("🧠 LLM 기반 결과 통합 테스트 시작")
    print("=" * 60)
    
    # ResultIntegrator 인스턴스 생성
    integrator = ResultIntegrator()
    
    # 테스트 에이전트 결과
    test_agent_results = [
        {
            "agent_id": "weather_agent",
            "success": True,
            "data": {
                "content": {
                    "answer": "서울 현재 날씨는 맑음이며, 기온은 22.5°C, 습도는 65%입니다. 오늘은 야외 활동하기 좋은 날씨입니다."
                }
            },
            "execution_time": 1.2,
            "confidence": 0.95
        },
        {
            "agent_id": "currency_exchange_agent", 
            "success": True,
            "data": {
                "content": {
                    "result": "현재 USD/KRW 환율은 1,320원입니다. 전일 대비 +5원(+0.38%) 상승했습니다."
                }
            },
            "execution_time": 0.8,
            "confidence": 0.92
        },
        {
            "agent_id": "crawler_agent",
            "success": True,
            "data": {
                "content": {
                    "error": "주가 정보 조회 중 오류가 발생했습니다."
                }
            },
            "execution_time": 2.1,
            "confidence": 0.75
        }
    ]
    
    # 원본 쿼리
    original_query = "오늘 날씨를 확인하고, 환율 정보도 알려주고, 삼성전자 주가 정보도 알려줘."
    
    print(f"📝 원본 쿼리: {original_query}")
    print(f"🤖 테스트 에이전트 결과: {len(test_agent_results)}개")
    print()
    
    # 결과 통합 실행
    try:
        result = await integrator.integrate_agent_results(
            original_query, test_agent_results
        )
        
        print("✅ 결과 통합 성공!")
        print(f"📊 통합 결과:")
        print(f"   - 성공 여부: {result.get('success', False)}")
        print(f"   - 처리된 에이전트: {result.get('agent_results_count', 0)}개")
        print(f"   - 추출된 내용: {result.get('successful_extractions', 0)}개")
        print(f"   - 통합 방법: {result.get('processing_summary', {}).get('integration_method', 'Unknown')}")
        print()
        
        # 통합된 최종 응답 출력
        integrated_content = result.get('integrated_content', '결과를 생성할 수 없습니다.')
        print("📋 LLM 통합 최종 응답:")
        print("-" * 40)
        print(integrated_content)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ 결과 통합 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_llm_result_integration()) 