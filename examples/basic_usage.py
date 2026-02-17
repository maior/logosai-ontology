"""
🚀 Basic Usage Examples

Demonstrates the basic usage of the ontology system.
"""

import asyncio
from ontology import OntologySystem, SimpleProgressCallback


async def basic_example():
    """Basic usage example."""
    print("🧠 Ontology system basic example")

    # Initialize the system
    system = OntologySystem(
        email="user@example.com",
        session_id="demo_session",
        project_id="demo_project"
    )
    
    try:
        await system.initialize()
        print("✅ System initialization complete\n")

        # Process a simple query (Korean query kept as-is: demonstrates Korean language support)
        query = "오늘 날씨 알려줘"
        print(f"📝 Query: {query}")

        async for result in system.process_query(query):
            if result["type"] == "semantic_analysis":
                print(f"🔍 Semantic analysis: {result['data']['intent']}")
            elif result["type"] == "complexity_analysis":
                print(f"📊 Complexity: {result['data']['strategy']} strategy")
            elif result["type"] == "workflow_design":
                steps = result['data']['steps']
                print(f"🎯 Workflow: {len(steps)} steps")
            elif result["type"] == "final_result":
                success = result['data']['success']
                print(f"✅ Final result: {'success' if success else 'failure'}")

    finally:
        await system.close()


async def progress_callback_example():
    """Progress callback example."""
    print("\n🔄 Progress Callback Example")

    system = OntologySystem(session_id="progress_demo")
    callback = SimpleProgressCallback()

    try:
        await system.initialize()

        # Korean query kept as-is: demonstrates Korean language support
        query = "USD/KRW 환율 조회해서 차트로 만들어줘"
        print(f"📝 Complex query: {query}")

        async for result in system.process_query(query, callback):
            if result["type"] == "final_result":
                print(f"✅ Processing complete!")
                break

        # Progress summary
        summary = callback.get_summary()
        print(f"📊 Progress summary:")
        print(f"   - Progress: {summary['current_progress']:.1%}")
        print(f"   - Messages: {summary['total_messages']}")
        print(f"   - Completed steps: {summary['completed_steps']}")
        print(f"   - Success rate: {summary['success_rate']:.1%}")

    finally:
        await system.close()


async def metrics_example():
    """System metrics example."""
    print("\n📊 System Metrics Example")

    system = OntologySystem(session_id="metrics_demo")

    try:
        await system.initialize()

        # Run several queries (Korean queries kept as-is: demonstrate Korean language support)
        queries = [
            "날씨 정보",
            "환율 조회",
            "계산 요청",
            "뉴스 검색"
        ]

        for query in queries:
            print(f"Processing: {query}")
            async for result in system.process_query(query):
                if result["type"] == "final_result":
                    break

        # Retrieve metrics
        metrics = system.get_system_metrics()

        print(f"\n📈 System metrics:")
        print(f"   Session ID: {metrics['session_info']['session_id']}")
        print(f"   Total executions: {metrics['execution_history']['total_executions']}")

        # Cache statistics
        cache_stats = metrics['semantic_query_manager']['cache_stats']
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"   Cache size: {cache_stats['cache_size']}")

        # Execution engine statistics
        engine_stats = metrics['execution_engine']
        print(f"   Average execution time: {engine_stats['average_execution_time']:.2f}s")
        print(f"   Success rate: {engine_stats['success_rate']:.2%}")

    finally:
        await system.close()


async def knowledge_graph_example():
    """Knowledge graph example."""
    print("\n🧠 Knowledge Graph Example")

    system = OntologySystem(session_id="kg_demo")

    try:
        await system.initialize()

        # Build the knowledge graph with a few queries (Korean queries: demonstrate Korean support)
        queries = [
            "환율 정보 조회",
            "날씨 데이터 분석",
            "주식 차트 생성"
        ]

        for query in queries:
            async for result in system.process_query(query):
                if result["type"] == "final_result":
                    break

        # Knowledge graph visualization data
        viz_data = system.get_knowledge_graph_visualization(max_nodes=20)

        print(f"🕸️ Knowledge graph info:")
        print(f"   Nodes: {len(viz_data['nodes'])}")
        print(f"   Edges: {len(viz_data['edges'])}")

        # Node type distribution
        stats = viz_data['stats']
        print(f"   Node type distribution:")
        for node_type, count in stats['node_types'].items():
            print(f"     - {node_type}: {count}")

        # Edge type distribution
        print(f"   Relationship type distribution:")
        for edge_type, count in stats['edge_types'].items():
            print(f"     - {edge_type}: {count}")

    finally:
        await system.close()


async def error_handling_example():
    """Error handling example."""
    print("\n⚠️ Error Handling Example")

    system = OntologySystem(session_id="error_demo")

    try:
        await system.initialize()

        # Queries that may cause issues
        problematic_queries = [
            "",  # empty query
            "매우 " * 1000 + "복잡한 쿼리",  # excessively long query
            "알 수 없는 도메인의 매우 특수한 요청"  # query that is hard to process
        ]

        for i, query in enumerate(problematic_queries, 1):
            print(f"\nTest {i}: {'empty query' if not query else query[:50]}...")

            try:
                async for result in system.process_query(query):
                    if result["type"] == "error":
                        print(f"❌ Error: {result['data']['error_message']}")
                        break
                    elif result["type"] == "final_result":
                        success = result['data']['success']
                        print(f"{'✅' if success else '⚠️'} Result: {'success' if success else 'partial success'}")
                        break

            except Exception as e:
                print(f"❌ Exception: {e}")

    finally:
        await system.close()


async def advanced_usage_example():
    """Advanced usage example."""
    print("\n🚀 Advanced Usage Example")

    system = OntologySystem(
        email="advanced@example.com",
        session_id="advanced_demo",
        project_id="advanced_project"
    )
    
    try:
        await system.initialize()
        
        # Complex multi-step query (Korean kept as-is: demonstrates Korean language support)
        complex_query = """
        다음 작업을 순서대로 수행해줘:
        1. USD/KRW 환율을 조회하고
        2. 최근 1주일 데이터로 차트를 만들고
        3. 분석 결과를 요약해서
        4. 메모에 저장해줘
        """

        print(f"📝 Processing complex query:")
        print(complex_query.strip())

        callback = SimpleProgressCallback()

        async for result in system.process_query(complex_query, callback):
            if result["type"] == "workflow_design":
                steps = result['data']['steps']
                print(f"\n🎯 Designed workflow ({len(steps)} steps):")
                for i, step in enumerate(steps, 1):
                    print(f"   {i}. {step['agent_id']}: {step['purpose'][:60]}...")

            elif result["type"] == "execution_results":
                results = result['data']['results']
                print(f"\n⚡ Execution results:")
                for res in results:
                    status = "✅" if res['success'] else "❌"
                    print(f"   {status} {res['agent_id']}: {res['execution_time']:.2f}s")

            elif result["type"] == "final_result":
                summary = result['data']['execution_summary']
                print(f"\n📊 Final summary:")
                print(f"   Total execution time: {summary['total_time']:.2f}s")
                print(f"   Steps executed: {summary['steps_executed']}")
                print(f"   Success rate: {summary['success_rate']:.1%}")
                print(f"   Strategy used: {summary['strategy_used']}")
                break

        # Final system state
        metrics = system.get_system_metrics()
        print(f"\n🔍 System state:")
        print(f"   Cache efficiency: {metrics['semantic_query_manager']['cache_hit_rate']:.1%}")
        print(f"   Knowledge graph: {metrics['knowledge_graph']['metadata']['total_concepts']} concepts")

    finally:
        await system.close()


async def main():
    """Run all examples."""
    print("🌟 Ontology system usage examples\n")

    await basic_example()
    await progress_callback_example()
    await metrics_example()
    await knowledge_graph_example()
    await error_handling_example()
    await advanced_usage_example()

    print("\n🎉 All examples complete!")


if __name__ == "__main__":
    asyncio.run(main()) 