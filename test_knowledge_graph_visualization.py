#!/usr/bin/env python3
"""
🎨 지식 그래프 시각화 테스트
Knowledge Graph Visualization Test

온톨로지 시스템의 지식 그래프 시각화 기능을 테스트하고 
실제 비주얼 출력을 확인합니다.
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 기존 온톨로지 시스템 사용 (임포트 문제 회피)
try:
    from system.ontology_system_clean import CleanOntologySystem as OntologySystem
    from system.ontology_system_clean import SimpleProgressCallback
    print("✅ CleanOntologySystem 사용")
except ImportError:
    try:
        from system.ontology_system import OntologySystem
        from system.progress_callback import SimpleProgressCallback
        print("✅ 기존 OntologySystem 사용")
    except ImportError:
        print("❌ 온톨로지 시스템을 불러올 수 없습니다.")
        print("   직접 구현된 시스템으로 대체합니다.")
        
        # 간단한 대체 시스템
        class MockOntologySystem:
            def __init__(self, **kwargs):
                self.session_id = kwargs.get('session_id', 'mock_session')
                
            async def initialize(self):
                return True
                
            async def process_query(self, query, callback=None):
                # 간단한 모의 결과 생성
                yield {"type": "semantic_analysis", "data": {"query": query}}
                yield {"type": "final_result", "data": {"success": True}}
                
            def get_knowledge_graph_visualization(self):
                return {
                    "nodes": [
                        {"id": "query_1", "type": "query", "label": "사용자 쿼리"},
                        {"id": "agent_1", "type": "agent", "label": "처리 에이전트"},
                        {"id": "result_1", "type": "result", "label": "실행 결과"}
                    ],
                    "edges": [
                        {"source": "query_1", "target": "agent_1", "type": "executes"},
                        {"source": "agent_1", "target": "result_1", "type": "produces"}
                    ]
                }
                
            def get_system_metrics(self):
                return {
                    "execution_history": {"total_executions": 1},
                    "semantic_query_manager": {"cache_hit_rate": 0.0}
                }
                
            async def close(self):
                pass
        
        class MockProgressCallback:
            def __init__(self):
                self.messages = []
                
        OntologySystem = MockOntologySystem
        SimpleProgressCallback = MockProgressCallback


class VisualizationTester:
    """지식 그래프 시각화 테스터"""
    
    def __init__(self):
        self.system = None
        self.test_results = []
    
    async def initialize(self):
        """시스템 초기화"""
        print("🚀 온톨로지 시스템 초기화 중...")
        self.system = OntologySystem(
            email="maiordba@gmail.com",
            session_id=f"viz_test_{int(time.time())}",
            project_id="knowledge_graph_visualization"
        )
        
        try:
            await self.system.initialize()
            print("✅ 시스템 초기화 완료")
            return True
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    async def run_test_query(self, query: str, description: str) -> Dict[str, Any]:
        """테스트 쿼리 실행 및 결과 분석"""
        print(f"\n📝 테스트: {description}")
        print(f"   쿼리: {query}")
        
        start_time = time.time()
        callback = SimpleProgressCallback()
        results = []
        
        try:
            async for result in self.system.process_query(query, callback):
                results.append(result)
            
            execution_time = time.time() - start_time
            
            # 최종 결과 확인
            final_result = next((r for r in results if r["type"] == "final_result"), None)
            success = final_result["data"]["success"] if final_result else False
            
            print(f"   {'✅' if success else '❌'} 실행 완료 ({execution_time:.2f}초)")
            
            return {
                "query": query,
                "description": description,
                "success": success,
                "execution_time": execution_time,
                "results": results,
                "callback_messages": callback.messages
            }
            
        except Exception as e:
            print(f"   ❌ 실행 실패: {e}")
            return {
                "query": query,
                "description": description,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def analyze_knowledge_graph(self) -> Dict[str, Any]:
        """지식 그래프 분석"""
        print("\n🔍 지식 그래프 분석 중...")
        
        try:
            viz_data = self.system.get_knowledge_graph_visualization()
            
            # 기본 구조 분석
            nodes = viz_data.get("nodes", [])
            edges = viz_data.get("edges", [])
            
            print(f"   📊 노드 수: {len(nodes)}")
            print(f"   🔗 엣지 수: {len(edges)}")
            
            # 노드 타입별 분석
            node_types = {}
            for node in nodes:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"   🏷️ 노드 타입별 분포:")
            for node_type, count in sorted(node_types.items()):
                print(f"      - {node_type}: {count}개")
            
            # 엣지 타입별 분석
            edge_types = {}
            for edge in edges:
                edge_type = edge.get("type", "unknown")
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            print(f"   🔗 엣지 타입별 분포:")
            for edge_type, count in sorted(edge_types.items()):
                print(f"      - {edge_type}: {count}개")
            
            return {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": node_types,
                "edge_types": edge_types,
                "visualization_data": viz_data
            }
            
        except Exception as e:
            print(f"   ❌ 지식 그래프 분석 실패: {e}")
            return {"error": str(e)}
    
    def generate_mermaid_diagram(self, viz_data: Dict[str, Any]) -> str:
        """Mermaid 다이어그램 생성"""
        print("\n🎨 Mermaid 다이어그램 생성 중...")
        
        try:
            nodes = viz_data.get("nodes", [])
            edges = viz_data.get("edges", [])
            
            # Mermaid 다이어그램 시작
            mermaid_lines = ["graph TD"]
            
            # 노드 정의 (처음 20개만)
            for i, node in enumerate(nodes[:20]):
                node_id = node.get("id", f"node_{i}")
                node_label = node.get("label", node_id)[:20]  # 라벨 길이 제한
                node_type = node.get("type", "default")
                
                # 노드 타입에 따른 스타일링
                if node_type == "query":
                    style = f'["{node_label}"]'
                elif node_type == "agent":
                    style = f'("{node_label}")'
                elif node_type == "concept":
                    style = f'{{{node_label}}}'
                else:
                    style = f'["{node_label}"]'
                
                mermaid_lines.append(f'    {node_id}{style}')
            
            # 엣지 정의 (처음 30개만)
            for edge in edges[:30]:
                source = edge.get("source", "")
                target = edge.get("target", "")
                edge_type = edge.get("type", "")
                
                if source and target:
                    if edge_type == "executes":
                        arrow = "-->"
                    elif edge_type == "depends_on":
                        arrow = "-..->"
                    elif edge_type == "related_to":
                        arrow = "---"
                    else:
                        arrow = "-->"
                    
                    mermaid_lines.append(f'    {source} {arrow} {target}')
            
            mermaid_diagram = "\n".join(mermaid_lines)
            
            print(f"   ✅ Mermaid 다이어그램 생성 완료 ({len(mermaid_lines)}줄)")
            return mermaid_diagram
            
        except Exception as e:
            print(f"   ❌ Mermaid 다이어그램 생성 실패: {e}")
            return f'graph TD\n    Error["다이어그램 생성 실패: {str(e)}"]'
    
    def save_visualization_data(self, data: Dict[str, Any], filename: str):
        """시각화 데이터 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"   💾 데이터 저장 완료: {filename}")
        except Exception as e:
            print(f"   ❌ 데이터 저장 실패: {e}")
    
    def display_sample_nodes(self, viz_data: Dict[str, Any], limit: int = 10):
        """샘플 노드 표시"""
        print(f"\n📋 샘플 노드 ({limit}개):")
        
        nodes = viz_data.get("nodes", [])
        for i, node in enumerate(nodes[:limit]):
            node_id = node.get("id", "unknown")
            node_type = node.get("type", "unknown")
            node_label = node.get("label", node_id)
            
            print(f"   {i+1:2d}. [{node_type}] {node_id}")
            print(f"       라벨: {node_label}")
            
            # 추가 속성 표시
            for key, value in node.items():
                if key not in ["id", "type", "label"] and len(str(value)) < 50:
                    print(f"       {key}: {value}")
            print()
    
    def display_sample_edges(self, viz_data: Dict[str, Any], limit: int = 10):
        """샘플 엣지 표시"""
        print(f"\n🔗 샘플 엣지 ({limit}개):")
        
        edges = viz_data.get("edges", [])
        for i, edge in enumerate(edges[:limit]):
            source = edge.get("source", "unknown")
            target = edge.get("target", "unknown")
            edge_type = edge.get("type", "unknown")
            
            print(f"   {i+1:2d}. {source} --[{edge_type}]--> {target}")
            
            # 추가 속성 표시
            for key, value in edge.items():
                if key not in ["source", "target", "type"] and len(str(value)) < 50:
                    print(f"       {key}: {value}")
    
    async def cleanup(self):
        """정리 작업"""
        if self.system:
            try:
                await self.system.close()
                print("🧹 시스템 정리 완료")
            except Exception as e:
                print(f"⚠️ 시스템 정리 중 오류: {e}")


async def run_comprehensive_visualization_test():
    """종합 시각화 테스트 실행"""
    print("🎨 지식 그래프 시각화 종합 테스트 시작")
    print("=" * 60)
    
    tester = VisualizationTester()
    
    try:
        # 1. 시스템 초기화
        if not await tester.initialize():
            return
        
        # 2. 다양한 테스트 쿼리 실행
        test_queries = [
            ("오늘 날씨 알려줘", "간단한 정보 조회"),
            ("USD/KRW 환율을 조회하고 차트로 만들어줘", "복합 작업 요청"),
            ("1+1을 계산하고 결과를 메모에 저장해줘", "계산 및 저장"),
            ("최신 AI 기술 동향을 검색하고 분석해줘", "검색 및 분석"),
            ("Python으로 간단한 웹 크롤러 만드는 방법 알려줘", "기술 질문")
        ]
        
        print(f"\n📝 {len(test_queries)}개 테스트 쿼리 실행 중...")
        
        for query, description in test_queries:
            result = await tester.run_test_query(query, description)
            tester.test_results.append(result)
            
            # 잠시 대기 (시스템 안정화)
            await asyncio.sleep(1)
        
        # 3. 지식 그래프 분석
        kg_analysis = tester.analyze_knowledge_graph()
        
        if "error" not in kg_analysis:
            viz_data = kg_analysis["visualization_data"]
            
            # 4. 샘플 데이터 표시
            tester.display_sample_nodes(viz_data, 10)
            tester.display_sample_edges(viz_data, 10)
            
            # 5. Mermaid 다이어그램 생성
            mermaid_diagram = tester.generate_mermaid_diagram(viz_data)
            
            # 6. 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 시각화 데이터 저장
            viz_filename = f"knowledge_graph_viz_{timestamp}.json"
            tester.save_visualization_data(viz_data, viz_filename)
            
            # Mermaid 다이어그램 저장
            mermaid_filename = f"knowledge_graph_mermaid_{timestamp}.md"
            try:
                with open(mermaid_filename, 'w', encoding='utf-8') as f:
                    f.write("# 지식 그래프 Mermaid 다이어그램\n\n")
                    f.write("```mermaid\n")
                    f.write(mermaid_diagram)
                    f.write("\n```\n")
                print(f"   💾 Mermaid 다이어그램 저장: {mermaid_filename}")
            except Exception as e:
                print(f"   ❌ Mermaid 저장 실패: {e}")
            
            # 7. 테스트 요약
            print("\n📊 테스트 요약:")
            print("=" * 40)
            
            successful_tests = sum(1 for r in tester.test_results if r.get("success", False))
            total_tests = len(tester.test_results)
            
            print(f"   성공한 테스트: {successful_tests}/{total_tests}")
            print(f"   총 실행 시간: {sum(r.get('execution_time', 0) for r in tester.test_results):.2f}초")
            print(f"   지식 그래프 노드: {kg_analysis['total_nodes']}개")
            print(f"   지식 그래프 엣지: {kg_analysis['total_edges']}개")
            
            # 시스템 메트릭스
            try:
                metrics = tester.system.get_system_metrics()
                print(f"   총 쿼리 실행: {metrics['execution_history']['total_executions']}")
                print(f"   캐시 히트율: {metrics['semantic_query_manager']['cache_hit_rate']:.1%}")
            except Exception as e:
                print(f"   메트릭스 조회 실패: {e}")
            
            print(f"\n💾 생성된 파일:")
            print(f"   - {viz_filename} (JSON 시각화 데이터)")
            print(f"   - {mermaid_filename} (Mermaid 다이어그램)")
            
            # 8. 비주얼 확인 안내
            print(f"\n🎨 시각화 확인 방법:")
            print(f"   1. JSON 데이터: {viz_filename} 파일을 JSON 뷰어로 확인")
            print(f"   2. Mermaid 다이어그램: {mermaid_filename}를 Mermaid 지원 에디터에서 확인")
            print(f"      - VS Code + Mermaid Preview 확장")
            print(f"      - GitHub/GitLab (자동 렌더링)")
            print(f"      - mermaid.live 온라인 에디터")
            
        else:
            print(f"❌ 지식 그래프 분석 실패: {kg_analysis['error']}")
        
        print("\n✅ 지식 그래프 시각화 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # 비동기 테스트 실행
    asyncio.run(run_comprehensive_visualization_test()) 