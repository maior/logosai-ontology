"""
Advanced Usage Examples

Demonstrates advanced features of the ontology system.
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
    """Parallel processing example."""
    print("=== Parallel Processing Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Complex analysis query
        query = """
        Analyze the current state of artificial intelligence in healthcare,
        research the latest developments in medical AI,
        and provide recommendations for implementation in hospitals.
        """

        print("Starting parallel execution...")
        start_time = time.time()

        result = await system.process_query(
            query,
            execution_strategy=ExecutionStrategy.PARALLEL
        )

        execution_time = time.time() - start_time

        print(f"Execution complete: {execution_time:.2f}s")
        print(f"Success: {result['success']}")
        print(f"Agents used: {len(result['execution_results'])}")
        print(f"Parallel executions: {result['metrics'].get('parallel_executions', 0)}")

        # Print result summary
        if result['success']:
            print("\n=== Execution Result Summary ===")
            for i, exec_result in enumerate(result['execution_results']):
                print(f"Agent {i+1} ({exec_result.agent_type.value}): "
                      f"{exec_result.execution_time:.2f}s, "
                      f"status: {exec_result.status.value}")

    finally:
        await system.shutdown()


async def example_sequential_workflow():
    """Sequential workflow example."""
    print("\n=== Sequential Workflow Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Query requiring step-by-step processing
        query = """
        First, research the fundamentals of quantum computing.
        Then, analyze its potential applications in cryptography.
        Finally, create a roadmap for quantum computing adoption in enterprise.
        """

        print("Starting sequential execution...")
        start_time = time.time()

        result = await system.process_query(
            query,
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )

        execution_time = time.time() - start_time

        print(f"Execution complete: {execution_time:.2f}s")
        print(f"Workflow steps: {len(result.get('workflow_plan', {}).get('execution_steps', []))}")

        # Print per-step results
        if result['success'] and 'execution_results' in result:
            print("\n=== Per-Step Execution Results ===")
            for i, exec_result in enumerate(result['execution_results']):
                print(f"Step {i+1}: {exec_result.agent_type.value}")
                print(f"  Execution time: {exec_result.execution_time:.2f}s")
                print(f"  Status: {exec_result.status.value}")
                if hasattr(exec_result.result_data, 'get'):
                    summary = str(exec_result.result_data)[:100] + "..."
                    print(f"  Result summary: {summary}")

    finally:
        await system.shutdown()


async def example_caching_and_optimization():
    """Caching and optimization example."""
    print("\n=== Caching and Optimization Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        queries = [
            "What is machine learning?",
            "Explain deep learning concepts",
            "What is machine learning?",  # duplicate query (cache test)
            "How does neural network training work?",
            "What is machine learning?",  # another duplicate query
        ]

        print("Starting cache effectiveness test...")

        execution_times = []
        for i, query in enumerate(queries):
            start_time = time.time()
            result = await system.process_query(query)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            print(f"Query {i+1}: {execution_time:.2f}s "
                  f"({'possible cache hit' if query in queries[:i] else 'new query'})")

        # Print cache statistics
        cache_stats = semantic_query_manager.get_cache_stats()
        print(f"\n=== Cache Statistics ===")
        print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1f}%")
        print(f"Duplicate calls prevented: {cache_stats['duplicate_calls_prevented']}")
        print(f"Total queries: {cache_stats['total_queries']}")

    finally:
        await system.shutdown()


async def example_custom_execution_context():
    """Custom execution context example."""
    print("\n=== Custom Execution Context Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Create a custom execution context
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

        print("Starting execution with custom context...")
        result = await system.process_query(
            query,
            execution_context=custom_context
        )

        print(f"Execution strategy: {custom_context.execution_strategy.value}")
        print(f"Max parallel agents: {custom_context.max_parallel_agents}")
        print(f"Debug mode: {custom_context.debug_mode}")
        print(f"Result: {result['success']}")

        # Check per-session cache
        cache_stats = semantic_query_manager.get_cache_stats()
        print(f"Session cache count: {cache_stats['session_cache_count']}")

    finally:
        await system.shutdown()


async def example_knowledge_graph_building():
    """Knowledge graph building example."""
    print("\n=== Knowledge Graph Building Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Build the knowledge graph with multiple related queries
        knowledge_queries = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "How do neural networks work?",
            "What are the applications of deep learning?",
            "Describe natural language processing",
            "What is computer vision?",
            "How does reinforcement learning work?"
        ]
        
        print("Running queries to build the knowledge graph...")

        for i, query in enumerate(knowledge_queries):
            print(f"Processing: {i+1}/{len(knowledge_queries)} - {query[:50]}...")
            await system.process_query(query)

        # Knowledge graph statistics
        kg_stats = system.knowledge_graph.get_graph_stats()
        print(f"\n=== Knowledge Graph Statistics ===")
        print(f"Node count: {kg_stats['node_count']}")
        print(f"Edge count: {kg_stats['edge_count']}")
        print(f"Connected components: {kg_stats['connected_components']}")

        # Example: finding related concepts
        if kg_stats['node_count'] > 0:
            related_concepts = await system.knowledge_graph.find_related_concepts(
                "machine learning", max_depth=2
            )
            print(f"Concepts related to 'machine learning': {related_concepts[:5]}")

    finally:
        await system.shutdown()


async def example_error_handling_and_recovery():
    """Error handling and recovery example."""
    print("\n=== Error Handling and Recovery Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Test various error conditions
        test_cases = [
            ("", "empty query"),
            ("a" * 10000, "very long query"),
            ("!@#$%^&*()", "special characters only"),
            ("Valid query about AI", "normal query")
        ]

        print("Testing various error conditions...")

        for query, description in test_cases:
            try:
                print(f"\nTest: {description}")
                result = await system.process_query(query)

                if result['success']:
                    print("✅ Processed successfully")
                else:
                    print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"🔥 Exception raised: {str(e)}")

        # Check system metrics
        metrics = system.get_system_metrics()
        print(f"\n=== System Metrics ===")
        print(f"Total queries: {metrics.total_queries}")
        print(f"Failed executions: {metrics.failed_executions}")
        print(f"Success rate: {metrics.get_success_rate():.1f}%")

    finally:
        await system.shutdown()


async def example_concurrent_processing():
    """Concurrent processing example."""
    print("\n=== Concurrent Processing Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Queries to process concurrently
        concurrent_queries = [
            "Research renewable energy technologies",
            "Analyze climate change impacts",
            "Study sustainable development goals",
            "Investigate green technology trends",
            "Examine environmental policy frameworks"
        ]
        
        print(f"Starting concurrent processing of {len(concurrent_queries)} queries...")
        start_time = time.time()

        # Execute all queries concurrently
        tasks = [
            system.process_query(query)
            for query in concurrent_queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        print(f"Concurrent processing complete: {total_time:.2f}s")

        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful

        print(f"Succeeded: {successful}, Failed: {failed}")

        # Check duplicate call prevention effect
        metrics = system.get_system_metrics()
        print(f"Duplicate calls prevented: {metrics.duplicate_calls_prevented}")

    finally:
        await system.shutdown()


async def example_performance_monitoring():
    """Performance monitoring example."""
    print("\n=== Performance Monitoring Example ===")

    system = OntologySystem()
    await system.initialize()

    try:
        # Various queries for performance measurement
        performance_queries = [
            ("Simple query", QueryType.SIMPLE),
            ("Complex analysis of market trends", QueryType.COMPLEX),
            ("Step by step guide creation", QueryType.MULTI_STEP),
            ("Creative writing task", QueryType.CREATIVE),
            ("Data analysis and insights", QueryType.ANALYTICAL)
        ]
        
        print("Measuring performance across various query types...")

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

            print(f"{expected_type.value}: {execution_time:.2f}s")

        # Performance summary
        print(f"\n=== Performance Summary ===")
        avg_time = sum(p['execution_time'] for p in performance_data) / len(performance_data)
        print(f"Average execution time: {avg_time:.2f}s")

        # Overall system metrics
        system_metrics = system.get_system_metrics()
        print(f"System average response time: {system_metrics.average_response_time:.2f}s")
        print(f"Cache hit rate: {system_metrics.get_cache_hit_rate():.1f}%")

    finally:
        await system.shutdown()


async def main():
    """Run all advanced examples."""
    print("🚀 Starting ontology system advanced usage examples\n")

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
            await asyncio.sleep(1)  # pause between examples
        except Exception as e:
            print(f"❌ Error during example execution: {e}")

    print("\n✅ All advanced examples complete!")


if __name__ == "__main__":
    asyncio.run(main()) 