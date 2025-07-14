#!/usr/bin/env python3
"""
🧪 간단한 온톨로지 시스템 테스트
Simple Ontology System Test

기본적인 기능 확인 및 지식 그래프 시각화 테스트
"""

import asyncio
import json
from datetime import datetime

# 간단한 온톨로지 시스템 구현
class SimpleOntologySystem:
    """간단한 온톨로지 시스템"""
    
    def __init__(self, session_id="simple_test"):
        self.session_id = session_id
        self.knowledge_graph = {
            "nodes": [],
            "edges": []
        }
        self.query_count = 0
        
    async def initialize(self):
        """시스템 초기화"""
        print("🚀 간단한 온톨로지 시스템 초기화")
        return True
    
    async def process_query(self, query_text: str):
        """쿼리 처리"""
        self.query_count += 1
        query_id = f"query_{self.query_count}"
        
        print(f"📝 쿼리 처리: {query_text}")
        
        # 쿼리 노드 추가
        query_node = {
            "id": query_id,
            "type": "query",
            "label": query_text[:30] + "..." if len(query_text) > 30 else query_text,
            "full_text": query_text,
            "timestamp": datetime.now().isoformat()
        }
        self.knowledge_graph["nodes"].append(query_node)
        
        # 쿼리 분석 결과 노드
        analysis_id = f"analysis_{self.query_count}"
        analysis_node = {
            "id": analysis_id,
            "type": "analysis",
            "label": "쿼리 분석",
            "complexity": self._analyze_complexity(query_text),
            "keywords": self._extract_keywords(query_text)
        }
        self.knowledge_graph["nodes"].append(analysis_node)
        
        # 쿼리 → 분석 엣지
        self.knowledge_graph["edges"].append({
            "source": query_id,
            "target": analysis_id,
            "type": "analyzed_by",
            "timestamp": datetime.now().isoformat()
        })
        
        # 에이전트 선택 및 실행
        selected_agents = self._select_agents(query_text)
        
        for i, agent_type in enumerate(selected_agents):
            agent_id = f"agent_{agent_type}_{self.query_count}_{i}"
            
            # 에이전트 노드
            agent_node = {
                "id": agent_id,
                "type": "agent",
                "label": f"{agent_type} 에이전트",
                "agent_type": agent_type,
                "status": "executed"
            }
            self.knowledge_graph["nodes"].append(agent_node)
            
            # 분석 → 에이전트 엣지
            self.knowledge_graph["edges"].append({
                "source": analysis_id,
                "target": agent_id,
                "type": "executes",
                "timestamp": datetime.now().isoformat()
            })
            
            # 결과 노드
            result_id = f"result_{agent_type}_{self.query_count}_{i}"
            result_node = {
                "id": result_id,
                "type": "result",
                "label": f"{agent_type} 결과",
                "agent_type": agent_type,
                "success": True,
                "data": f"Mock result from {agent_type}"
            }
            self.knowledge_graph["nodes"].append(result_node)
            
            # 에이전트 → 결과 엣지
            self.knowledge_graph["edges"].append({
                "source": agent_id,
                "target": result_id,
                "type": "produces",
                "timestamp": datetime.now().isoformat()
            })
        
        # 개념 노드 추가 (키워드 기반)
        for keyword in analysis_node["keywords"]:
            concept_id = f"concept_{keyword.lower()}"
            
            # 이미 존재하는 개념인지 확인
            existing_concept = next(
                (node for node in self.knowledge_graph["nodes"] 
                 if node["id"] == concept_id), None
            )
            
            if not existing_concept:
                concept_node = {
                    "id": concept_id,
                    "type": "concept",
                    "label": keyword,
                    "frequency": 1
                }
                self.knowledge_graph["nodes"].append(concept_node)
            else:
                existing_concept["frequency"] = existing_concept.get("frequency", 0) + 1
            
            # 쿼리 → 개념 엣지
            self.knowledge_graph["edges"].append({
                "source": query_id,
                "target": concept_id,
                "type": "relates_to",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "query_id": query_id,
            "agents_used": selected_agents,
            "results_count": len(selected_agents)
        }
    
    def _analyze_complexity(self, query_text: str) -> str:
        """쿼리 복잡도 분석"""
        if len(query_text) > 100:
            return "high"
        elif len(query_text) > 50:
            return "medium"
        else:
            return "low"
    
    def _extract_keywords(self, query_text: str) -> list:
        """키워드 추출"""
        # 간단한 키워드 추출
        common_words = {"을", "를", "이", "가", "은", "는", "에", "의", "로", "와", "과", "해줘", "알려줘"}
        words = query_text.split()
        keywords = [word for word in words if len(word) > 1 and word not in common_words]
        return keywords[:5]  # 최대 5개
    
    def _select_agents(self, query_text: str) -> list:
        """에이전트 선택"""
        agents = []
        query_lower = query_text.lower()
        
        if any(keyword in query_lower for keyword in ["날씨", "기온", "예보"]):
            agents.append("weather")
        
        if any(keyword in query_lower for keyword in ["환율", "달러", "원", "유로"]):
            agents.append("currency")
        
        if any(keyword in query_lower for keyword in ["계산", "더하기", "빼기", "곱하기"]):
            agents.append("calculator")
        
        if any(keyword in query_lower for keyword in ["검색", "찾아", "정보"]):
            agents.append("search")
        
        if any(keyword in query_lower for keyword in ["차트", "그래프", "시각화"]):
            agents.append("visualization")
        
        if any(keyword in query_lower for keyword in ["메모", "저장", "기록"]):
            agents.append("memo")
        
        # 기본 에이전트
        if not agents:
            agents.append("general")
        
        return agents
    
    def get_knowledge_graph_visualization(self):
        """지식 그래프 시각화 데이터 반환"""
        return self.knowledge_graph
    
    def get_system_metrics(self):
        """시스템 메트릭"""
        return {
            "total_queries": self.query_count,
            "total_nodes": len(self.knowledge_graph["nodes"]),
            "total_edges": len(self.knowledge_graph["edges"]),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types()
        }
    
    def _count_node_types(self):
        """노드 타입별 개수"""
        counts = {}
        for node in self.knowledge_graph["nodes"]:
            node_type = node["type"]
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _count_edge_types(self):
        """엣지 타입별 개수"""
        counts = {}
        for edge in self.knowledge_graph["edges"]:
            edge_type = edge["type"]
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    async def close(self):
        """시스템 종료"""
        print("🧹 시스템 종료")


def generate_mermaid_diagram(viz_data):
    """Mermaid 다이어그램 생성"""
    nodes = viz_data.get("nodes", [])
    edges = viz_data.get("edges", [])
    
    lines = ["graph TD"]
    
    # 노드 정의
    for node in nodes:
        node_id = node["id"]
        label = node["label"][:20]  # 라벨 길이 제한
        node_type = node["type"]
        
        if node_type == "query":
            style = f'["{label}"]'
        elif node_type == "agent":
            style = f'("{label}")'
        elif node_type == "concept":
            style = f'{{{label}}}'
        elif node_type == "analysis":
            style = f'<"{label}">'
        else:
            style = f'["{label}"]'
        
        lines.append(f"    {node_id}{style}")
    
    # 엣지 정의
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        edge_type = edge["type"]
        
        if edge_type == "executes":
            arrow = "-->"
        elif edge_type == "analyzed_by":
            arrow = "-..->"
        elif edge_type == "relates_to":
            arrow = "---"
        else:
            arrow = "-->"
        
        lines.append(f"    {source} {arrow} {target}")
    
    return "\n".join(lines)


async def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🧪 간단한 온톨로지 시스템 종합 테스트")
    print("=" * 50)
    
    system = SimpleOntologySystem("comprehensive_test")
    
    try:
        await system.initialize()
        
        # 테스트 쿼리들
        test_queries = [
            "오늘 서울 날씨 알려줘",
            "USD/KRW 환율을 조회하고 차트로 만들어줘",
            "1+1을 계산하고 결과를 메모에 저장해줘",
            "최신 AI 기술 동향을 검색해줘",
            "Python 웹 크롤링 방법 알려줘"
        ]
        
        print(f"\n📝 {len(test_queries)}개 쿼리 처리 중...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. {query}")
            result = await system.process_query(query)
            print(f"   ✅ 성공: {result['agents_used']} 에이전트 사용")
        
        # 지식 그래프 분석
        print("\n🔍 지식 그래프 분석:")
        viz_data = system.get_knowledge_graph_visualization()
        metrics = system.get_system_metrics()
        
        print(f"   📊 총 노드: {metrics['total_nodes']}개")
        print(f"   🔗 총 엣지: {metrics['total_edges']}개")
        print(f"   🏷️ 노드 타입: {metrics['node_types']}")
        print(f"   🔗 엣지 타입: {metrics['edge_types']}")
        
        # Mermaid 다이어그램 생성
        print("\n🎨 Mermaid 다이어그램 생성:")
        mermaid_diagram = generate_mermaid_diagram(viz_data)
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 데이터 저장
        json_filename = f"simple_kg_viz_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, ensure_ascii=False, indent=2)
        print(f"   💾 JSON 데이터 저장: {json_filename}")
        
        # Mermaid 다이어그램 저장
        md_filename = f"simple_kg_mermaid_{timestamp}.md"
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write("# 간단한 온톨로지 시스템 지식 그래프\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_diagram)
            f.write("\n```\n")
        print(f"   💾 Mermaid 다이어그램 저장: {md_filename}")
        
        # 샘플 노드 출력
        print(f"\n📋 샘플 노드 (처음 10개):")
        for i, node in enumerate(viz_data["nodes"][:10]):
            print(f"   {i+1:2d}. [{node['type']}] {node['id']}: {node['label']}")
        
        # 샘플 엣지 출력
        print(f"\n🔗 샘플 엣지 (처음 10개):")
        for i, edge in enumerate(viz_data["edges"][:10]):
            print(f"   {i+1:2d}. {edge['source']} --[{edge['type']}]--> {edge['target']}")
        
        print(f"\n🎨 생성된 시각화 파일:")
        print(f"   - {json_filename} (JSON 데이터)")
        print(f"   - {md_filename} (Mermaid 다이어그램)")
        
        print(f"\n✅ 테스트 완료!")
        
    finally:
        await system.close()


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test()) 