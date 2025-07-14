"""
통합 테스트
전체 온톨로지 시스템의 통합 동작을 검증
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any

from ontology.core.models import (
    SemanticQuery, ExecutionContext, AgentType, QueryType, 
    ExecutionStrategy, ExecutionStatus
)
from ontology.system.ontology_system import OntologySystem
from ontology.engines.semantic_query_manager import semantic_query_manager
from ontology.engines.execution_engine import AdvancedExecutionEngine
from ontology.engines.workflow_designer import SmartWorkflowDesigner
from ontology.engines.knowledge_graph import NetworkXKnowledgeGraph


class TestSystemIntegration:
    """시스템 통합 테스트"""
    
    @pytest.fixture
    async def ontology_system(self):
        """온톨로지 시스템 픽스처"""
        system = OntologySystem()
        await system.initialize()
        yield system
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_simple_query_processing(self, ontology_system):
        """간단한 쿼리 처리 테스트"""
        query_text = "What is artificial intelligence?"
        
        result = await ontology_system.process_query(query_text)
        
        assert result is not None
        assert 'final_result' in result
        assert 'execution_results' in result
        assert 'metrics' in result
        assert result['success'] is True
    
    @pytest.mark.asyncio
    async def test_complex_query_processing(self, ontology_system):
        """복잡한 쿼리 처리 테스트"""
        query_text = "Analyze the impact of machine learning on healthcare and provide recommendations for implementation"
        
        result = await ontology_system.process_query(
            query_text,
            execution_strategy=ExecutionStrategy.HYBRID
        )
        
        assert result is not None
        assert result['success'] is True
        assert len(result['execution_results']) > 1  # 여러 에이전트 사용
        assert result['metrics']['execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, ontology_system):
        """병렬 실행 테스트"""
        query_text = "Research current trends in AI and analyze their market impact"
        
        start_time = time.time()
        result = await ontology_system.process_query(
            query_text,
            execution_strategy=ExecutionStrategy.PARALLEL
        )
        execution_time = time.time() - start_time
        
        assert result['success'] is True
        assert execution_time < 10  # 병렬 실행으로 시간 단축
        assert result['metrics']['parallel_executions'] > 0
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, ontology_system):
        """캐싱 기능 테스트"""
        query_text = "What are the benefits of renewable energy?"
        
        # 첫 번째 실행
        result1 = await ontology_system.process_query(query_text)
        first_time = result1['metrics']['execution_time']
        
        # 두 번째 실행 (캐시 히트 예상)
        result2 = await ontology_system.process_query(query_text)
        second_time = result2['metrics']['execution_time']
        
        assert result1['success'] is True
        assert result2['success'] is True
        # 두 번째 실행이 더 빨라야 함 (캐시 효과)
        assert second_time <= first_time
    
    @pytest.mark.asyncio
    async def test_error_handling(self, ontology_system):
        """오류 처리 테스트"""
        # 빈 쿼리
        result = await ontology_system.process_query("")
        assert result['success'] is False
        assert 'error' in result
        
        # 매우 긴 쿼리
        long_query = "test " * 1000
        result = await ontology_system.process_query(long_query)
        # 시스템이 처리하거나 적절히 오류 처리해야 함
        assert 'final_result' in result or 'error' in result
    
    @pytest.mark.asyncio
    async def test_workflow_design_integration(self, ontology_system):
        """워크플로우 설계 통합 테스트"""
        query_text = "Create a step-by-step plan for implementing a machine learning project"
        
        result = await ontology_system.process_query(
            query_text,
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        assert result['success'] is True
        assert 'workflow_plan' in result
        workflow_plan = result['workflow_plan']
        assert len(workflow_plan.execution_steps) > 1
        assert workflow_plan.estimated_time > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_integration(self, ontology_system):
        """지식 그래프 통합 테스트"""
        # 여러 관련 쿼리 실행하여 지식 그래프 구축
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?",
            "Explain artificial intelligence applications"
        ]
        
        for query in queries:
            await ontology_system.process_query(query)
        
        # 지식 그래프 상태 확인
        kg_stats = ontology_system.knowledge_graph.get_graph_stats()
        assert kg_stats['node_count'] > 0
        assert kg_stats['edge_count'] >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, ontology_system):
        """메트릭 수집 테스트"""
        queries = [
            "Simple query test",
            "Another test query for metrics",
            "Third query to build metrics"
        ]
        
        for query in queries:
            await ontology_system.process_query(query)
        
        metrics = ontology_system.get_system_metrics()
        assert metrics.total_queries >= len(queries)
        assert metrics.get_cache_hit_rate() >= 0
        assert metrics.get_success_rate() > 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, ontology_system):
        """세션 관리 테스트"""
        session_id = "test_session_123"
        query_text = "Test query for session management"
        
        # 세션별 쿼리 실행
        result1 = await ontology_system.process_query(
            query_text, 
            session_id=session_id
        )
        result2 = await ontology_system.process_query(
            query_text, 
            session_id=session_id
        )
        
        assert result1['success'] is True
        assert result2['success'] is True
        
        # 세션 캐시 확인
        cache_stats = semantic_query_manager.get_cache_stats()
        assert cache_stats['session_cache_count'] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, ontology_system):
        """동시 쿼리 처리 테스트"""
        queries = [
            "What is blockchain technology?",
            "Explain quantum computing",
            "How does cloud computing work?",
            "What are the benefits of edge computing?"
        ]
        
        # 동시 실행
        tasks = [
            ontology_system.process_query(query) 
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        
        # 모든 쿼리가 성공적으로 처리되어야 함
        for result in results:
            assert result['success'] is True
        
        # 중복 호출 방지 확인
        metrics = ontology_system.get_system_metrics()
        assert metrics.duplicate_calls_prevented >= 0


class TestComponentIntegration:
    """컴포넌트 간 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_semantic_query_manager_integration(self):
        """의미론적 쿼리 관리자 통합 테스트"""
        query_text = "Test semantic query manager integration"
        
        # 첫 번째 호출
        query1 = await semantic_query_manager.get_or_create_semantic_query(query_text)
        
        # 두 번째 호출 (캐시에서 가져와야 함)
        query2 = await semantic_query_manager.get_or_create_semantic_query(query_text)
        
        assert query1.query_id == query2.query_id
        assert query1.query_text == query2.query_text
        
        # 캐시 통계 확인
        stats = semantic_query_manager.get_cache_stats()
        assert stats['cache_hit_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_execution_engine_integration(self):
        """실행 엔진 통합 테스트"""
        engine = AdvancedExecutionEngine()
        
        query = SemanticQuery(
            query_text="Test execution engine integration",
            query_type=QueryType.SIMPLE,
            required_agents=[AgentType.GENERAL]
        )
        
        context = ExecutionContext()
        
        results = await engine.execute_query(query, context)
        
        assert len(results) > 0
        assert results[0].status == ExecutionStatus.COMPLETED
        assert results[0].execution_time > 0
    
    @pytest.mark.asyncio
    async def test_workflow_designer_integration(self):
        """워크플로우 설계자 통합 테스트"""
        designer = SmartWorkflowDesigner()
        
        query = SemanticQuery(
            query_text="Design a workflow for data analysis project",
            query_type=QueryType.MULTI_STEP,
            required_agents=[AgentType.RESEARCH, AgentType.ANALYSIS]
        )
        
        context = ExecutionContext()
        
        workflow_plan = await designer.design_workflow(query, context)
        
        assert workflow_plan is not None
        assert len(workflow_plan.execution_steps) > 0
        assert workflow_plan.estimated_time > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_integration(self):
        """지식 그래프 통합 테스트"""
        kg = NetworkXKnowledgeGraph()
        
        # 개념과 관계 추가
        await kg.add_concept("Machine Learning", {"type": "technology"})
        await kg.add_concept("Artificial Intelligence", {"type": "field"})
        await kg.add_relation("Machine Learning", "Artificial Intelligence", "part_of")
        
        # 쿼리 테스트
        results = await kg.query_graph("Machine Learning")
        assert len(results) > 0
        
        # 관련 개념 찾기
        related = await kg.find_related_concepts("Machine Learning")
        assert "Artificial Intelligence" in related


class TestPerformanceIntegration:
    """성능 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(self):
        """부하 상황에서의 시스템 성능 테스트"""
        system = OntologySystem()
        await system.initialize()
        
        try:
            # 다양한 쿼리 생성
            queries = [
                f"Test query number {i} for performance testing"
                for i in range(20)
            ]
            
            start_time = time.time()
            
            # 배치 처리
            batch_size = 5
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                tasks = [system.process_query(query) for query in batch]
                results = await asyncio.gather(*tasks)
                
                # 모든 결과가 성공적이어야 함
                for result in results:
                    assert result['success'] is True
            
            total_time = time.time() - start_time
            
            # 성능 메트릭 확인
            metrics = system.get_system_metrics()
            assert metrics.total_queries >= len(queries)
            assert total_time < 60  # 1분 내 완료
            
        finally:
            await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """메모리 효율성 테스트"""
        system = OntologySystem()
        await system.initialize()
        
        try:
            # 많은 쿼리 실행
            for i in range(50):
                query = f"Memory test query {i}"
                result = await system.process_query(query)
                assert result['success'] is True
            
            # 캐시 정리 테스트
            await semantic_query_manager.cleanup_expired_entries()
            
            # 시스템이 여전히 정상 동작해야 함
            final_result = await system.process_query("Final test query")
            assert final_result['success'] is True
            
        finally:
            await system.shutdown()


if __name__ == "__main__":
    # 간단한 테스트 실행
    async def run_basic_test():
        system = OntologySystem()
        await system.initialize()
        
        try:
            result = await system.process_query("Test the integrated ontology system")
            print(f"Test result: {result['success']}")
            print(f"Execution time: {result['metrics']['execution_time']:.2f}s")
            
            metrics = system.get_system_metrics()
            print(f"System metrics: {metrics.to_dict()}")
            
        finally:
            await system.shutdown()
    
    asyncio.run(run_basic_test()) 