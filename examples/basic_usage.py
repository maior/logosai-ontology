"""
🚀 Basic Usage Examples
기본 사용 예제

새로운 온톨로지 시스템의 기본 사용법을 보여줍니다.
"""

import asyncio
from ontology import OntologySystem, SimpleProgressCallback


async def basic_example():
    """기본 사용 예제"""
    print("🧠 온톨로지 시스템 기본 예제")
    
    # 시스템 초기화
    system = OntologySystem(
        email="user@example.com",
        session_id="demo_session",
        project_id="demo_project"
    )
    
    try:
        await system.initialize()
        print("✅ 시스템 초기화 완료\n")
        
        # 간단한 쿼리 처리
        query = "오늘 날씨 알려줘"
        print(f"📝 쿼리: {query}")
        
        async for result in system.process_query(query):
            if result["type"] == "semantic_analysis":
                print(f"🔍 의미 분석: {result['data']['intent']}")
            elif result["type"] == "complexity_analysis":
                print(f"📊 복잡도: {result['data']['strategy']} 전략")
            elif result["type"] == "workflow_design":
                steps = result['data']['steps']
                print(f"🎯 워크플로우: {len(steps)}개 단계")
            elif result["type"] == "final_result":
                success = result['data']['success']
                print(f"✅ 최종 결과: {'성공' if success else '실패'}")
        
    finally:
        await system.close()


async def progress_callback_example():
    """진행 상황 콜백 예제"""
    print("\n🔄 진행 상황 콜백 예제")
    
    system = OntologySystem(session_id="progress_demo")
    callback = SimpleProgressCallback()
    
    try:
        await system.initialize()
        
        query = "USD/KRW 환율 조회해서 차트로 만들어줘"
        print(f"📝 복잡한 쿼리: {query}")
        
        async for result in system.process_query(query, callback):
            if result["type"] == "final_result":
                print(f"✅ 처리 완료!")
                break
        
        # 진행 상황 요약
        summary = callback.get_summary()
        print(f"📊 진행 요약:")
        print(f"   - 진행률: {summary['current_progress']:.1%}")
        print(f"   - 메시지 수: {summary['total_messages']}")
        print(f"   - 완료 단계: {summary['completed_steps']}")
        print(f"   - 성공률: {summary['success_rate']:.1%}")
        
    finally:
        await system.close()


async def metrics_example():
    """메트릭스 조회 예제"""
    print("\n📊 시스템 메트릭스 예제")
    
    system = OntologySystem(session_id="metrics_demo")
    
    try:
        await system.initialize()
        
        # 여러 쿼리 실행
        queries = [
            "날씨 정보",
            "환율 조회",
            "계산 요청",
            "뉴스 검색"
        ]
        
        for query in queries:
            print(f"처리 중: {query}")
            async for result in system.process_query(query):
                if result["type"] == "final_result":
                    break
        
        # 메트릭스 조회
        metrics = system.get_system_metrics()
        
        print(f"\n📈 시스템 메트릭스:")
        print(f"   세션 ID: {metrics['session_info']['session_id']}")
        print(f"   총 실행 횟수: {metrics['execution_history']['total_executions']}")
        
        # 캐시 통계
        cache_stats = metrics['semantic_query_manager']['cache_stats']
        print(f"   캐시 히트율: {cache_stats['hit_rate']:.2%}")
        print(f"   캐시 크기: {cache_stats['cache_size']}")
        
        # 실행 엔진 통계
        engine_stats = metrics['execution_engine']
        print(f"   평균 실행 시간: {engine_stats['average_execution_time']:.2f}초")
        print(f"   성공률: {engine_stats['success_rate']:.2%}")
        
    finally:
        await system.close()


async def knowledge_graph_example():
    """지식 그래프 예제"""
    print("\n🧠 지식 그래프 예제")
    
    system = OntologySystem(session_id="kg_demo")
    
    try:
        await system.initialize()
        
        # 몇 개의 쿼리로 지식 그래프 구축
        queries = [
            "환율 정보 조회",
            "날씨 데이터 분석",
            "주식 차트 생성"
        ]
        
        for query in queries:
            async for result in system.process_query(query):
                if result["type"] == "final_result":
                    break
        
        # 지식 그래프 시각화 데이터
        viz_data = system.get_knowledge_graph_visualization(max_nodes=20)
        
        print(f"🕸️ 지식 그래프 정보:")
        print(f"   노드 수: {len(viz_data['nodes'])}")
        print(f"   엣지 수: {len(viz_data['edges'])}")
        
        # 노드 타입별 통계
        stats = viz_data['stats']
        print(f"   노드 타입별 분포:")
        for node_type, count in stats['node_types'].items():
            print(f"     - {node_type}: {count}개")
        
        # 엣지 타입별 통계
        print(f"   관계 타입별 분포:")
        for edge_type, count in stats['edge_types'].items():
            print(f"     - {edge_type}: {count}개")
        
    finally:
        await system.close()


async def error_handling_example():
    """오류 처리 예제"""
    print("\n⚠️ 오류 처리 예제")
    
    system = OntologySystem(session_id="error_demo")
    
    try:
        await system.initialize()
        
        # 문제가 있을 수 있는 쿼리들
        problematic_queries = [
            "",  # 빈 쿼리
            "매우 " * 1000 + "복잡한 쿼리",  # 너무 긴 쿼리
            "알 수 없는 도메인의 매우 특수한 요청"  # 처리하기 어려운 쿼리
        ]
        
        for i, query in enumerate(problematic_queries, 1):
            print(f"\n테스트 {i}: {'빈 쿼리' if not query else query[:50]}...")
            
            try:
                async for result in system.process_query(query):
                    if result["type"] == "error":
                        print(f"❌ 오류 발생: {result['data']['error_message']}")
                        break
                    elif result["type"] == "final_result":
                        success = result['data']['success']
                        print(f"{'✅' if success else '⚠️'} 결과: {'성공' if success else '부분 성공'}")
                        break
                        
            except Exception as e:
                print(f"❌ 예외 발생: {e}")
        
    finally:
        await system.close()


async def advanced_usage_example():
    """고급 사용 예제"""
    print("\n🚀 고급 사용 예제")
    
    system = OntologySystem(
        email="advanced@example.com",
        session_id="advanced_demo",
        project_id="advanced_project"
    )
    
    try:
        await system.initialize()
        
        # 복잡한 멀티 스텝 쿼리
        complex_query = """
        다음 작업을 순서대로 수행해줘:
        1. USD/KRW 환율을 조회하고
        2. 최근 1주일 데이터로 차트를 만들고
        3. 분석 결과를 요약해서
        4. 메모에 저장해줘
        """
        
        print(f"📝 복잡한 쿼리 처리:")
        print(complex_query.strip())
        
        callback = SimpleProgressCallback()
        
        async for result in system.process_query(complex_query, callback):
            if result["type"] == "workflow_design":
                steps = result['data']['steps']
                print(f"\n🎯 설계된 워크플로우 ({len(steps)}단계):")
                for i, step in enumerate(steps, 1):
                    print(f"   {i}. {step['agent_id']}: {step['purpose'][:60]}...")
                    
            elif result["type"] == "execution_results":
                results = result['data']['results']
                print(f"\n⚡ 실행 결과:")
                for res in results:
                    status = "✅" if res['success'] else "❌"
                    print(f"   {status} {res['agent_id']}: {res['execution_time']:.2f}초")
                    
            elif result["type"] == "final_result":
                summary = result['data']['execution_summary']
                print(f"\n📊 최종 요약:")
                print(f"   총 실행 시간: {summary['total_time']:.2f}초")
                print(f"   실행 단계: {summary['steps_executed']}개")
                print(f"   성공률: {summary['success_rate']:.1%}")
                print(f"   사용 전략: {summary['strategy_used']}")
                break
        
        # 최종 시스템 상태
        metrics = system.get_system_metrics()
        print(f"\n🔍 시스템 상태:")
        print(f"   캐시 효율: {metrics['semantic_query_manager']['cache_hit_rate']:.1%}")
        print(f"   지식 그래프: {metrics['knowledge_graph']['metadata']['total_concepts']}개 개념")
        
    finally:
        await system.close()


async def main():
    """모든 예제 실행"""
    print("🌟 온톨로지 시스템 사용 예제 모음\n")
    
    await basic_example()
    await progress_callback_example()
    await metrics_example()
    await knowledge_graph_example()
    await error_handling_example()
    await advanced_usage_example()
    
    print("\n🎉 모든 예제 완료!")


if __name__ == "__main__":
    asyncio.run(main()) 