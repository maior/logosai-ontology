"""
고급 사용 예제
온톨로지 시스템의 고급 기능들을 보여주는 예제
"""

import asyncio
import time
from typing import Dict, List, Any

from ontology.system.ontology_system import OntologySystem
from ontology.core.models import (
    ExecutionStrategy, QueryType, AgentType, ExecutionContext
)
from ontology.engines.semantic_query_manager import semantic_query_manager


async def example_parallel_processing():
    """병렬 처리 예제"""
    print("=== 병렬 처리 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 복잡한 분석 쿼리
        query = """
        Analyze the current state of artificial intelligence in healthcare, 
        research the latest developments in medical AI, 
        and provide recommendations for implementation in hospitals.
        """
        
        print("병렬 실행 시작...")
        start_time = time.time()
        
        result = await system.process_query(
            query,
            execution_strategy=ExecutionStrategy.PARALLEL
        )
        
        execution_time = time.time() - start_time
        
        print(f"실행 완료: {execution_time:.2f}초")
        print(f"성공 여부: {result['success']}")
        print(f"사용된 에이전트 수: {len(result['execution_results'])}")
        print(f"병렬 실행 수: {result['metrics'].get('parallel_executions', 0)}")
        
        # 결과 요약 출력
        if result['success']:
            print("\n=== 실행 결과 요약 ===")
            for i, exec_result in enumerate(result['execution_results']):
                print(f"에이전트 {i+1} ({exec_result.agent_type.value}): "
                      f"{exec_result.execution_time:.2f}초, "
                      f"상태: {exec_result.status.value}")
    
    finally:
        await system.shutdown()


async def example_sequential_workflow():
    """순차 워크플로우 예제"""
    print("\n=== 순차 워크플로우 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 단계별 처리가 필요한 쿼리
        query = """
        First, research the fundamentals of quantum computing.
        Then, analyze its potential applications in cryptography.
        Finally, create a roadmap for quantum computing adoption in enterprise.
        """
        
        print("순차 실행 시작...")
        start_time = time.time()
        
        result = await system.process_query(
            query,
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        execution_time = time.time() - start_time
        
        print(f"실행 완료: {execution_time:.2f}초")
        print(f"워크플로우 단계 수: {len(result.get('workflow_plan', {}).get('execution_steps', []))}")
        
        # 단계별 결과 출력
        if result['success'] and 'execution_results' in result:
            print("\n=== 단계별 실행 결과 ===")
            for i, exec_result in enumerate(result['execution_results']):
                print(f"단계 {i+1}: {exec_result.agent_type.value}")
                print(f"  실행 시간: {exec_result.execution_time:.2f}초")
                print(f"  상태: {exec_result.status.value}")
                if hasattr(exec_result.result_data, 'get'):
                    summary = str(exec_result.result_data)[:100] + "..."
                    print(f"  결과 요약: {summary}")
    
    finally:
        await system.shutdown()


async def example_caching_and_optimization():
    """캐싱 및 최적화 예제"""
    print("\n=== 캐싱 및 최적화 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        queries = [
            "What is machine learning?",
            "Explain deep learning concepts",
            "What is machine learning?",  # 중복 쿼리 (캐시 테스트)
            "How does neural network training work?",
            "What is machine learning?",  # 또 다른 중복 쿼리
        ]
        
        print("캐싱 효과 테스트 시작...")
        
        execution_times = []
        for i, query in enumerate(queries):
            start_time = time.time()
            result = await system.process_query(query)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            print(f"쿼리 {i+1}: {execution_time:.2f}초 "
                  f"({'캐시 히트 가능' if query in queries[:i] else '새 쿼리'})")
        
        # 캐시 통계 출력
        cache_stats = semantic_query_manager.get_cache_stats()
        print(f"\n=== 캐시 통계 ===")
        print(f"캐시 히트율: {cache_stats['cache_hit_rate']:.1f}%")
        print(f"중복 호출 방지: {cache_stats['duplicate_calls_prevented']}회")
        print(f"전체 쿼리 수: {cache_stats['total_queries']}")
    
    finally:
        await system.shutdown()


async def example_custom_execution_context():
    """커스텀 실행 컨텍스트 예제"""
    print("\n=== 커스텀 실행 컨텍스트 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 커스텀 실행 컨텍스트 생성
        custom_context = ExecutionContext(
            session_id="advanced_example_session",
            user_id="advanced_user",
            execution_strategy=ExecutionStrategy.HYBRID,
            max_parallel_agents=2,
            timeout_seconds=120,
            cache_enabled=True,
            debug_mode=True,
            custom_config={
                "priority": "high",
                "detailed_logging": True,
                "experimental_features": True
            }
        )
        
        query = "Analyze the impact of AI on job markets and suggest adaptation strategies"
        
        print("커스텀 컨텍스트로 실행 시작...")
        result = await system.process_query(
            query,
            execution_context=custom_context
        )
        
        print(f"실행 전략: {custom_context.execution_strategy.value}")
        print(f"최대 병렬 에이전트: {custom_context.max_parallel_agents}")
        print(f"디버그 모드: {custom_context.debug_mode}")
        print(f"실행 결과: {result['success']}")
        
        # 세션별 캐시 확인
        cache_stats = semantic_query_manager.get_cache_stats()
        print(f"세션 캐시 수: {cache_stats['session_cache_count']}")
    
    finally:
        await system.shutdown()


async def example_knowledge_graph_building():
    """지식 그래프 구축 예제"""
    print("\n=== 지식 그래프 구축 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 관련된 여러 쿼리로 지식 그래프 구축
        knowledge_queries = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "How do neural networks work?",
            "What are the applications of deep learning?",
            "Describe natural language processing",
            "What is computer vision?",
            "How does reinforcement learning work?"
        ]
        
        print("지식 그래프 구축을 위한 쿼리 실행...")
        
        for i, query in enumerate(knowledge_queries):
            print(f"처리 중: {i+1}/{len(knowledge_queries)} - {query[:50]}...")
            await system.process_query(query)
        
        # 지식 그래프 통계
        kg_stats = system.knowledge_graph.get_graph_stats()
        print(f"\n=== 지식 그래프 통계 ===")
        print(f"노드 수: {kg_stats['node_count']}")
        print(f"엣지 수: {kg_stats['edge_count']}")
        print(f"연결 컴포넌트 수: {kg_stats['connected_components']}")
        
        # 관련 개념 찾기 예제
        if kg_stats['node_count'] > 0:
            related_concepts = await system.knowledge_graph.find_related_concepts(
                "machine learning", max_depth=2
            )
            print(f"'machine learning'과 관련된 개념들: {related_concepts[:5]}")
    
    finally:
        await system.shutdown()


async def example_error_handling_and_recovery():
    """오류 처리 및 복구 예제"""
    print("\n=== 오류 처리 및 복구 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 다양한 오류 상황 테스트
        test_cases = [
            ("", "빈 쿼리"),
            ("a" * 10000, "매우 긴 쿼리"),
            ("!@#$%^&*()", "특수 문자만 포함된 쿼리"),
            ("Valid query about AI", "정상 쿼리")
        ]
        
        print("다양한 오류 상황 테스트...")
        
        for query, description in test_cases:
            try:
                print(f"\n테스트: {description}")
                result = await system.process_query(query)
                
                if result['success']:
                    print("✅ 성공적으로 처리됨")
                else:
                    print(f"❌ 처리 실패: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"🔥 예외 발생: {str(e)}")
        
        # 시스템 메트릭 확인
        metrics = system.get_system_metrics()
        print(f"\n=== 시스템 메트릭 ===")
        print(f"총 쿼리 수: {metrics.total_queries}")
        print(f"실패한 실행 수: {metrics.failed_executions}")
        print(f"성공률: {metrics.get_success_rate():.1f}%")
    
    finally:
        await system.shutdown()


async def example_concurrent_processing():
    """동시 처리 예제"""
    print("\n=== 동시 처리 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 동시에 처리할 쿼리들
        concurrent_queries = [
            "Research renewable energy technologies",
            "Analyze climate change impacts",
            "Study sustainable development goals",
            "Investigate green technology trends",
            "Examine environmental policy frameworks"
        ]
        
        print(f"{len(concurrent_queries)}개 쿼리 동시 처리 시작...")
        start_time = time.time()
        
        # 모든 쿼리를 동시에 실행
        tasks = [
            system.process_query(query) 
            for query in concurrent_queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        print(f"동시 처리 완료: {total_time:.2f}초")
        
        # 결과 분석
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful
        
        print(f"성공: {successful}개, 실패: {failed}개")
        
        # 중복 호출 방지 효과 확인
        metrics = system.get_system_metrics()
        print(f"중복 호출 방지: {metrics.duplicate_calls_prevented}회")
    
    finally:
        await system.shutdown()


async def example_performance_monitoring():
    """성능 모니터링 예제"""
    print("\n=== 성능 모니터링 예제 ===")
    
    system = OntologySystem()
    await system.initialize()
    
    try:
        # 성능 측정을 위한 다양한 쿼리
        performance_queries = [
            ("Simple query", QueryType.SIMPLE),
            ("Complex analysis of market trends", QueryType.COMPLEX),
            ("Step by step guide creation", QueryType.MULTI_STEP),
            ("Creative writing task", QueryType.CREATIVE),
            ("Data analysis and insights", QueryType.ANALYTICAL)
        ]
        
        print("다양한 쿼리 타입별 성능 측정...")
        
        performance_data = []
        
        for query_text, expected_type in performance_queries:
            start_time = time.time()
            result = await system.process_query(query_text)
            execution_time = time.time() - start_time
            
            performance_data.append({
                'query_type': expected_type.value,
                'execution_time': execution_time,
                'success': result['success'],
                'agent_count': len(result.get('execution_results', []))
            })
            
            print(f"{expected_type.value}: {execution_time:.2f}초")
        
        # 성능 요약
        print(f"\n=== 성능 요약 ===")
        avg_time = sum(p['execution_time'] for p in performance_data) / len(performance_data)
        print(f"평균 실행 시간: {avg_time:.2f}초")
        
        # 시스템 전체 메트릭
        system_metrics = system.get_system_metrics()
        print(f"시스템 평균 응답 시간: {system_metrics.average_response_time:.2f}초")
        print(f"캐시 히트율: {system_metrics.get_cache_hit_rate():.1f}%")
    
    finally:
        await system.shutdown()


async def main():
    """모든 고급 예제 실행"""
    print("🚀 온톨로지 시스템 고급 사용 예제 시작\n")
    
    examples = [
        example_parallel_processing,
        example_sequential_workflow,
        example_caching_and_optimization,
        example_custom_execution_context,
        example_knowledge_graph_building,
        example_error_handling_and_recovery,
        example_concurrent_processing,
        example_performance_monitoring
    ]
    
    for example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(1)  # 예제 간 간격
        except Exception as e:
            print(f"❌ 예제 실행 중 오류: {e}")
    
    print("\n✅ 모든 고급 예제 완료!")


if __name__ == "__main__":
    asyncio.run(main()) 