"""
📊 Visualization Response Formatter
시각화 응답 포맷터

D3.js 친화적인 JSON 응답 형식으로 변환
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class VisualizationResponseFormatter:
    """시각화 응답 포맷터 - D3.js 친화적 JSON 변환"""
    
    def format_visualization_response(
        self,
        visualization_data: Dict[str, Any],
        original_query: str,
        agent_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """시각화 데이터를 표준화된 JSON 형식으로 변환"""
        try:
            logger.info(f"📊 시각화 응답 포맷 변환 시작: {visualization_data.get('type', 'unknown')}")
            
            # 1. 기본 메타데이터 구성
            response = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "query": original_query,
                "visualization": self._format_visualization_data(visualization_data),
                "metadata": self._generate_metadata(visualization_data, agent_results)
            }
            
            # 2. 시각화 타입별 데이터 구조 최적화
            viz_type = visualization_data.get('type', 'unknown')
            
            if viz_type == 'mermaid':
                response["visualization"] = self._format_mermaid_data(visualization_data)
            elif viz_type == 'svg':
                response["visualization"] = self._format_svg_data(visualization_data)
            elif viz_type == 'd3':
                response["visualization"] = self._format_d3_data(visualization_data)
            else:
                response["visualization"] = self._format_generic_data(visualization_data)
            
            logger.info(f"✅ 시각화 응답 포맷 변환 완료: {viz_type}")
            return response
            
        except Exception as e:
            logger.error(f"시각화 응답 포맷 변환 실패: {e}")
            return self._create_error_response(original_query, str(e))
    
    def _format_visualization_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 데이터 기본 포맷"""
        return {
            "type": data.get('type', 'unknown'),
            "format": data.get('format', 'mermaid'),
            "title": data.get('title', '시각화'),
            "description": data.get('description', ''),
            "interactive": data.get('interactive', True),
            "fallback": data.get('fallback', False),
            "content": data.get('content', ''),
            "data": data.get('data', {}),
            "options": data.get('options', {})
        }
    
    def _format_mermaid_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mermaid 데이터 포맷"""
        return {
            "type": "mermaid",
            "renderer": "mermaid",
            "code": data.get('content', ''),
            "title": data.get('title', 'Mermaid 다이어그램'),
            "description": data.get('description', ''),
            "config": {
                "theme": "default",
                "themeVariables": {
                    "primaryColor": "#667eea",
                    "primaryTextColor": "#fff",
                    "primaryBorderColor": "#764ba2",
                    "lineColor": "#764ba2"
                }
            },
            "interactive": True,
            "fallback": data.get('fallback', False),
            "raw_data": data.get('data', {}),
            "format": data.get('format', 'flowchart')
        }
    
    def _format_svg_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """SVG 데이터 포맷"""
        return {
            "type": "svg",
            "renderer": "svg",
            "svg_content": data.get('content', ''),
            "title": data.get('title', 'SVG 차트'),
            "description": data.get('description', ''),
            "width": data.get('options', {}).get('width', 600),
            "height": data.get('options', {}).get('height', 400),
            "interactive": data.get('interactive', False),
            "fallback": data.get('fallback', False),
            "raw_data": data.get('data', {}),
            "chart_type": data.get('format', 'bar')
        }
    
    def _format_d3_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """D3.js 데이터 포맷"""
        raw_data = data.get('data', {})
        
        # D3.js 친화적 노드-링크 구조
        d3_data = {
            "type": "d3",
            "renderer": "d3",
            "title": data.get('title', 'D3.js 시각화'),
            "description": data.get('description', ''),
            "chart_type": data.get('format', 'network'),
            "data": self._convert_to_d3_format(raw_data, data.get('format', 'network')),
            "config": {
                "width": data.get('options', {}).get('width', 800),
                "height": data.get('options', {}).get('height', 600),
                "margin": {"top": 20, "right": 20, "bottom": 20, "left": 20}
            },
            "interactive": True,
            "fallback": data.get('fallback', False),
            "raw_data": raw_data
        }
        
        return d3_data
    
    def _format_generic_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """범용 데이터 포맷"""
        return {
            "type": "generic",
            "renderer": "auto",
            "title": data.get('title', '시각화'),
            "description": data.get('description', ''),
            "content": data.get('content', ''),
            "data": data.get('data', {}),
            "options": data.get('options', {}),
            "interactive": data.get('interactive', True),
            "fallback": data.get('fallback', False),
            "format": data.get('format', 'text')
        }
    
    def _convert_to_d3_format(self, raw_data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """원시 데이터를 D3.js 형식으로 변환"""
        
        if chart_type == 'network':
            return self._convert_to_network_format(raw_data)
        elif chart_type == 'tree':
            return self._convert_to_tree_format(raw_data)
        elif chart_type == 'timeline':
            return self._convert_to_timeline_format(raw_data)
        elif chart_type == 'chart':
            return self._convert_to_chart_format(raw_data)
        else:
            return self._convert_to_generic_format(raw_data)
    
    def _convert_to_network_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """네트워크 그래프 형식으로 변환"""
        nodes = []
        links = []
        
        # 에이전트들을 노드로 변환
        agents = raw_data.get('agents', [])
        for i, agent in enumerate(agents):
            nodes.append({
                "id": agent.get('id', f'node_{i}'),
                "name": agent.get('name', f'Node {i+1}'),
                "group": 1,
                "success": agent.get('success', True),
                "size": 10,
                "color": "#4ecdc4" if agent.get('success', True) else "#ff6b6b"
            })
        
        # 관계들을 링크로 변환
        relationships = raw_data.get('relationships', [])
        for rel in relationships:
            links.append({
                "source": rel.get('from', ''),
                "target": rel.get('to', ''),
                "type": rel.get('type', 'default'),
                "strength": 1
            })
        
        # 관계가 없으면 순차 연결 생성
        if not links and len(nodes) > 1:
            for i in range(len(nodes) - 1):
                links.append({
                    "source": nodes[i]["id"],
                    "target": nodes[i + 1]["id"],
                    "type": "sequence",
                    "strength": 1
                })
        
        return {
            "nodes": nodes,
            "links": links
        }
    
    def _convert_to_tree_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """트리 구조 형식으로 변환"""
        agents = raw_data.get('agents', [])
        
        if not agents:
            return {"name": "Root", "children": []}
        
        # 첫 번째 에이전트를 루트로 설정
        root = {
            "name": agents[0].get('name', 'Root'),
            "children": []
        }
        
        # 나머지 에이전트들을 자식으로 추가
        for agent in agents[1:]:
            root["children"].append({
                "name": agent.get('name', 'Node'),
                "success": agent.get('success', True),
                "size": 100
            })
        
        return root
    
    def _convert_to_timeline_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """타임라인 형식으로 변환"""
        agents = raw_data.get('agents', [])
        
        timeline_data = []
        for i, agent in enumerate(agents):
            timeline_data.append({
                "id": agent.get('id', f'event_{i}'),
                "title": agent.get('name', f'Event {i+1}'),
                "start": i * 1000,  # 가상 시간
                "end": (i + 1) * 1000,
                "success": agent.get('success', True),
                "description": f"에이전트 {agent.get('name', '')} 실행"
            })
        
        return {
            "timeline": timeline_data
        }
    
    def _convert_to_chart_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """차트 형식으로 변환"""
        data_points = raw_data.get('data_points', [])
        
        if not data_points:
            # 에이전트 성공/실패 통계로 대체
            agents = raw_data.get('agents', [])
            success_count = sum(1 for agent in agents if agent.get('success', True))
            failure_count = len(agents) - success_count
            
            data_points = [
                {"label": "성공", "value": success_count},
                {"label": "실패", "value": failure_count}
            ]
        
        return {
            "data": data_points,
            "xAxis": "label",
            "yAxis": "value"
        }
    
    def _convert_to_generic_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """범용 형식으로 변환"""
        return {
            "items": raw_data.get('agents', []),
            "metadata": {
                "total_items": len(raw_data.get('agents', [])),
                "data_points": len(raw_data.get('data_points', [])),
                "processes": len(raw_data.get('processes', []))
            }
        }
    
    def _generate_metadata(
        self, 
        visualization_data: Dict[str, Any],
        agent_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """메타데이터 생성"""
        return {
            "source": "LogosAI Visualization System",
            "version": "1.0.0",
            "processing_time": datetime.now().isoformat(),
            "agents_used": len(agent_results or []),
            "visualization_type": visualization_data.get('type', 'unknown'),
            "format": visualization_data.get('format', 'unknown'),
            "fallback_used": visualization_data.get('fallback', False),
            "interactive": visualization_data.get('interactive', True),
            "client_libs": {
                "mermaid": "10.6.1",
                "d3": "7.8.5",
                "recommended_renderer": self._get_recommended_renderer(visualization_data)
            }
        }
    
    def _get_recommended_renderer(self, data: Dict[str, Any]) -> str:
        """추천 렌더러 반환"""
        viz_type = data.get('type', 'unknown')
        
        if viz_type == 'mermaid':
            return 'mermaid'
        elif viz_type == 'svg':
            return 'svg'
        elif viz_type == 'd3':
            return 'd3'
        else:
            return 'auto'
    
    def _create_error_response(self, query: str, error_msg: str) -> Dict[str, Any]:
        """오류 응답 생성"""
        return {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": error_msg,
            "visualization": {
                "type": "error",
                "renderer": "text",
                "content": f"시각화 생성 중 오류가 발생했습니다: {error_msg}",
                "title": "오류",
                "description": "시각화 처리 실패",
                "interactive": False,
                "fallback": True
            },
            "metadata": {
                "source": "LogosAI Visualization System",
                "version": "1.0.0",
                "processing_time": datetime.now().isoformat(),
                "error": True,
                "error_message": error_msg
            }
        }
    
    def format_data_visualization_agent_result(
        self, 
        agent_result: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """data_visualization_agent 결과 포맷팅"""
        try:
            content = agent_result.get('content', '')
            metadata = agent_result.get('metadata', {})
            
            # HTML 컨텐츠에서 시각화 데이터 추출
            if 'mermaid' in content.lower():
                # Mermaid 코드 추출
                import re
                mermaid_match = re.search(r'<div class="mermaid">(.*?)</div>', content, re.DOTALL)
                if mermaid_match:
                    mermaid_code = mermaid_match.group(1).strip()
                    visualization_data = {
                        'type': 'mermaid',
                        'content': mermaid_code,
                        'title': metadata.get('visualization_type', 'Mermaid 다이어그램'),
                        'description': f"Data Visualization Agent 생성 결과",
                        'interactive': True,
                        'fallback': False
                    }
                else:
                    visualization_data = {
                        'type': 'html',
                        'content': content,
                        'title': 'HTML 시각화',
                        'description': 'Data Visualization Agent HTML 결과',
                        'interactive': True,
                        'fallback': False
                    }
            elif '<svg' in content:
                # SVG 컨텐츠
                visualization_data = {
                    'type': 'svg',
                    'content': content,
                    'title': metadata.get('visualization_type', 'SVG 차트'),
                    'description': f"Data Visualization Agent SVG 결과",
                    'interactive': metadata.get('contains_svg', True),
                    'fallback': False
                }
            else:
                # 기본 HTML
                visualization_data = {
                    'type': 'html',
                    'content': content,
                    'title': 'HTML 시각화',
                    'description': 'Data Visualization Agent 결과',
                    'interactive': True,
                    'fallback': False
                }
            
            return self.format_visualization_response(
                visualization_data, original_query
            )
            
        except Exception as e:
            logger.error(f"Data Visualization Agent 결과 포맷팅 실패: {e}")
            return self._create_error_response(original_query, str(e))


def get_visualization_response_formatter() -> VisualizationResponseFormatter:
    """시각화 응답 포맷터 인스턴스 반환"""
    return VisualizationResponseFormatter()