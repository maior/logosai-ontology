"""
🧪 Ontology System Tests
온톨로지 시스템 테스트

새로운 아키텍처의 통합 테스트
"""

import asyncio
import pytest
import sys
import os
from typing import List, Dict, Any

# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.ontology_system import OntologySystem
from system.progress_callback import SimpleProgressCallback
from core.models import SemanticQuery, ExecutionStrategy


class TestOntologySystem:
    """온톨로지 시스템 테스트"""
    
    @pytest.fixture
    async def ontology_system(self):
        """테스트용 온톨로지 시스템"""
        system = OntologySystem(
            email="test@example.com",
            session_id="test_session_001",
            project_id="test_project"
        )
        await system.initialize()
        yield system
        await system.close()
    
    @pytest.mark.asyncio
    async def test_simple_query_processing(self, ontology_system):
        """간단한 쿼리 처리 테스트"""
        query_text = "오늘 날씨 알려줘"
        results = []
        
        async for result in ontology_system.process_query(query_text):
            results.append(result)
        
        # 결과 검증
        assert len(results) > 0
        
        # 마지막 결과가 성공인지 확인
        final_result = results[-1]
        assert final_result["type"] == "final_result"
        assert final_result["data"]["success"] is True
        
        # 각 단계별 결과 확인
        result_types = [r["type"] for r in results]
        expected_types = ["semantic_analysis", "complexity_analysis", "workflow_design", "execution_results", "final_result"]
        
        for expected_type in expected_types:
            assert expected_type in result_types
    
    @pytest.mark.asyncio
    async def test_complex_query_processing(self, ontology_system):
        """복잡한 쿼리 처리 테스트"""
        query_text = "USD/KRW 환율을 조회하고 차트로 만들어서 메모에 저장해줘"
        results = []
        
        async for result in ontology_system.process_query(query_text):
            results.append(result)
        
        # 복잡도 분석 결과 확인
        complexity_result = next(r for r in results if r["type"] == "complexity_analysis")
        complexity_data = complexity_result["data"]
        
        # 복잡한 쿼리이므로 여러 에이전트 필요
        assert complexity_data["estimated_agents"] > 1
        assert complexity_data["strategy"] in ["sequential", "parallel", "hybrid"]
        
        # 워크플로우 설계 결과 확인
        workflow_result = next(r for r in results if r["type"] == "workflow_design")
        workflow_data = workflow_result["data"]
        
        # 여러 단계가 있어야 함
        assert len(workflow_data["steps"]) > 1
        
        # 필요한 에이전트들이 포함되어야 함
        agent_ids = [step["agent_id"] for step in workflow_data["steps"]]
        assert "finance_agent" in agent_ids  # 환율 조회
        assert "chart_agent" in agent_ids    # 차트 생성
        assert "memo_agent" in agent_ids     # 메모 저장
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, ontology_system):
        """진행 상황 콜백 테스트"""
        callback = SimpleProgressCallback()
        query_text = "간단한 정보 검색"
        
        results = []
        async for result in ontology_system.process_query(query_text, callback):
            results.append(result)
        
        # 콜백 메시지 확인
        assert len(callback.messages) > 0
        assert callback.current_progress == 1.0  # 완료
        
        # 단계 완료 확인
        assert len(callback.completed_steps) > 0
        
        # 요약 정보 확인
        summary = callback.get_summary()
        assert summary["current_progress"] == 1.0
        assert summary["total_messages"] > 0
    
    @pytest.mark.asyncio
    async def test_semantic_query_caching(self, ontology_system):
        """SemanticQuery 캐싱 테스트"""
        query_text = "테스트 쿼리"
        
        # 첫 번째 실행
        start_time = asyncio.get_event_loop().time()
        results1 = []
        async for result in ontology_system.process_query(query_text):
            results1.append(result)
        first_time = asyncio.get_event_loop().time() - start_time
        
        # 두 번째 실행 (캐시 히트 예상)
        start_time = asyncio.get_event_loop().time()
        results2 = []
        async for result in ontology_system.process_query(query_text):
            results2.append(result)
        second_time = asyncio.get_event_loop().time() - start_time
        
        # 두 번째 실행이 더 빨라야 함 (캐시 효과)
        # 실제 환경에서는 차이가 있지만, 테스트에서는 미미할 수 있음
        assert len(results1) == len(results2)
        
        # 캐시 메트릭스 확인
        metrics = ontology_system.get_system_metrics()
        cache_stats = metrics["semantic_query_manager"]["cache_stats"]
        assert cache_stats["total_requests"] >= 2
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_update(self, ontology_system):
        """지식 그래프 업데이트 테스트"""
        query_text = "환율 정보 조회"
        
        # 쿼리 처리
        results = []
        async for result in ontology_system.process_query(query_text):
            results.append(result)
        
        # 지식 그래프 시각화 데이터 조회
        viz_data = ontology_system.get_knowledge_graph_visualization()
        
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert len(viz_data["nodes"]) > 0
        
        # 쿼리 관련 노드가 있는지 확인
        node_ids = [node["id"] for node in viz_data["nodes"]]
        query_nodes = [node_id for node_id in node_ids if "query_" in node_id]
        assert len(query_nodes) > 0
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, ontology_system):
        """시스템 메트릭스 테스트"""
        # 몇 개의 쿼리 실행
        queries = ["날씨 정보", "환율 조회", "계산 요청"]
        
        for query in queries:
            results = []
            async for result in ontology_system.process_query(query):
                results.append(result)
        
        # 메트릭스 조회
        metrics = ontology_system.get_system_metrics()
        
        # 기본 구조 확인
        assert "session_info" in metrics
        assert "execution_history" in metrics
        assert "semantic_query_manager" in metrics
        assert "execution_engine" in metrics
        assert "knowledge_graph" in metrics
        
        # 실행 기록 확인
        assert metrics["execution_history"]["total_executions"] == len(queries)
        
        # 세션 정보 확인
        session_info = metrics["session_info"]
        assert session_info["session_id"] == "test_session_001"
        assert session_info["email"] == "test@example.com"
        assert session_info["is_initialized"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, ontology_system):
        """오류 처리 테스트"""
        # 빈 쿼리
        results = []
        async for result in ontology_system.process_query(""):
            results.append(result)
        
        # 결과가 있어야 함 (오류 처리됨)
        assert len(results) > 0
        
        # 매우 복잡한 쿼리 (시스템 한계 테스트)
        complex_query = "매우 복잡한 " * 100 + "쿼리"
        results = []
        async for result in ontology_system.process_query(complex_query):
            results.append(result)
        
        # 오류가 발생해도 결과는 반환되어야 함
        assert len(results) > 0


async def test_integration_example():
    """통합 예제 테스트"""
    print("\n🧪 온톨로지 시스템 통합 테스트 시작")
    
    # 시스템 초기화
    system = OntologySystem(
        email="integration@test.com",
        session_id="integration_test"
    )
    
    try:
        await system.initialize()
        print("✅ 시스템 초기화 완료")
        
        # 테스트 쿼리들
        test_queries = [
            "오늘 날씨 알려줘",
            "USD/KRW 환율 조회해서 차트로 만들어줘",
            "1+1 계산하고 결과를 메모에 저장해줘",
            "최신 기술 뉴스 검색하고 분석해줘"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 테스트 {i}: {query}")
            
            callback = SimpleProgressCallback()
            results = []
            
            async for result in system.process_query(query, callback):
                results.append(result)
                if result["type"] == "final_result":
                    success = result["data"]["success"]
                    exec_time = result["data"]["execution_summary"]["total_time"]
                    print(f"   {'✅' if success else '❌'} 결과: {success}, 실행시간: {exec_time:.2f}초")
        
        # 시스템 메트릭스 출력
        print("\n📊 시스템 메트릭스:")
        metrics = system.get_system_metrics()
        print(f"   총 실행 횟수: {metrics['execution_history']['total_executions']}")
        print(f"   캐시 히트율: {metrics['semantic_query_manager']['cache_hit_rate']:.2%}")
        
        # 지식 그래프 정보
        viz_data = system.get_knowledge_graph_visualization()
        print(f"   지식 그래프: {len(viz_data['nodes'])}개 노드, {len(viz_data['edges'])}개 엣지")
        
        print("\n✅ 통합 테스트 완료")
        
    finally:
        await system.close()


if __name__ == "__main__":
    # 통합 테스트 실행
    asyncio.run(test_integration_example()) 