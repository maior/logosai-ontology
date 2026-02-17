"""
📊 Visualization Response Formatter

Converts visualization data to D3.js-friendly JSON response format.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class VisualizationResponseFormatter:
    """Visualization response formatter - converts data to D3.js-friendly JSON."""

    def format_visualization_response(
        self,
        visualization_data: Dict[str, Any],
        original_query: str,
        agent_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convert visualization data to a standardized JSON format."""
        try:
            logger.info(f"📊 Starting visualization response format conversion: {visualization_data.get('type', 'unknown')}")

            # 1. Build base metadata
            response = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "query": original_query,
                "visualization": self._format_visualization_data(visualization_data),
                "metadata": self._generate_metadata(visualization_data, agent_results)
            }
            
            # 2. Optimize data structure based on visualization type
            viz_type = visualization_data.get('type', 'unknown')

            if viz_type == 'mermaid':
                response["visualization"] = self._format_mermaid_data(visualization_data)
            elif viz_type == 'svg':
                response["visualization"] = self._format_svg_data(visualization_data)
            elif viz_type == 'd3':
                response["visualization"] = self._format_d3_data(visualization_data)
            else:
                response["visualization"] = self._format_generic_data(visualization_data)

            logger.info(f"✅ Visualization response format conversion complete: {viz_type}")
            return response

        except Exception as e:
            logger.error(f"Visualization response format conversion failed: {e}")
            return self._create_error_response(original_query, str(e))
    
    def _format_visualization_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic visualization data format."""
        return {
            "type": data.get('type', 'unknown'),
            "format": data.get('format', 'mermaid'),
            "title": data.get('title', 'Visualization'),
            "description": data.get('description', ''),
            "interactive": data.get('interactive', True),
            "fallback": data.get('fallback', False),
            "content": data.get('content', ''),
            "data": data.get('data', {}),
            "options": data.get('options', {})
        }
    
    def _format_mermaid_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mermaid data format."""
        return {
            "type": "mermaid",
            "renderer": "mermaid",
            "code": data.get('content', ''),
            "title": data.get('title', 'Mermaid Diagram'),
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
        """SVG data format."""
        return {
            "type": "svg",
            "renderer": "svg",
            "svg_content": data.get('content', ''),
            "title": data.get('title', 'SVG Chart'),
            "description": data.get('description', ''),
            "width": data.get('options', {}).get('width', 600),
            "height": data.get('options', {}).get('height', 400),
            "interactive": data.get('interactive', False),
            "fallback": data.get('fallback', False),
            "raw_data": data.get('data', {}),
            "chart_type": data.get('format', 'bar')
        }
    
    def _format_d3_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """D3.js data format."""
        raw_data = data.get('data', {})

        # D3.js-friendly node-link structure
        d3_data = {
            "type": "d3",
            "renderer": "d3",
            "title": data.get('title', 'D3.js Visualization'),
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
        """Generic data format."""
        return {
            "type": "generic",
            "renderer": "auto",
            "title": data.get('title', 'Visualization'),
            "description": data.get('description', ''),
            "content": data.get('content', ''),
            "data": data.get('data', {}),
            "options": data.get('options', {}),
            "interactive": data.get('interactive', True),
            "fallback": data.get('fallback', False),
            "format": data.get('format', 'text')
        }
    
    def _convert_to_d3_format(self, raw_data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Convert raw data to D3.js format."""
        
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
        """Convert to network graph format."""
        nodes = []
        links = []

        # Convert agents to nodes
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

        # Convert relationships to links
        relationships = raw_data.get('relationships', [])
        for rel in relationships:
            links.append({
                "source": rel.get('from', ''),
                "target": rel.get('to', ''),
                "type": rel.get('type', 'default'),
                "strength": 1
            })

        # If no relationships exist, create sequential connections
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
        """Convert to tree structure format."""
        agents = raw_data.get('agents', [])

        if not agents:
            return {"name": "Root", "children": []}

        # Set the first agent as the root
        root = {
            "name": agents[0].get('name', 'Root'),
            "children": []
        }

        # Add remaining agents as children
        for agent in agents[1:]:
            root["children"].append({
                "name": agent.get('name', 'Node'),
                "success": agent.get('success', True),
                "size": 100
            })
        
        return root
    
    def _convert_to_timeline_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to timeline format."""
        agents = raw_data.get('agents', [])

        timeline_data = []
        for i, agent in enumerate(agents):
            timeline_data.append({
                "id": agent.get('id', f'event_{i}'),
                "title": agent.get('name', f'Event {i+1}'),
                "start": i * 1000,  # simulated time
                "end": (i + 1) * 1000,
                "success": agent.get('success', True),
                "description": f"Agent {agent.get('name', '')} execution"
            })
        
        return {
            "timeline": timeline_data
        }
    
    def _convert_to_chart_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to chart format."""
        data_points = raw_data.get('data_points', [])

        if not data_points:
            # Fall back to agent success/failure statistics
            agents = raw_data.get('agents', [])
            success_count = sum(1 for agent in agents if agent.get('success', True))
            failure_count = len(agents) - success_count

            data_points = [
                {"label": "Success", "value": success_count},
                {"label": "Failure", "value": failure_count}
            ]
        
        return {
            "data": data_points,
            "xAxis": "label",
            "yAxis": "value"
        }
    
    def _convert_to_generic_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to generic format."""
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
        """Generate metadata."""
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
        """Return the recommended renderer."""
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
        """Create an error response."""
        return {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": error_msg,
            "visualization": {
                "type": "error",
                "renderer": "text",
                "content": f"An error occurred while generating the visualization: {error_msg}",
                "title": "Error",
                "description": "Visualization processing failed",
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
        """Format the result from data_visualization_agent."""
        try:
            content = agent_result.get('content', '')
            metadata = agent_result.get('metadata', {})

            # Extract visualization data from HTML content
            if 'mermaid' in content.lower():
                # Extract Mermaid code
                import re
                mermaid_match = re.search(r'<div class="mermaid">(.*?)</div>', content, re.DOTALL)
                if mermaid_match:
                    mermaid_code = mermaid_match.group(1).strip()
                    visualization_data = {
                        'type': 'mermaid',
                        'content': mermaid_code,
                        'title': metadata.get('visualization_type', 'Mermaid Diagram'),
                        'description': "Data Visualization Agent output",
                        'interactive': True,
                        'fallback': False
                    }
                else:
                    visualization_data = {
                        'type': 'html',
                        'content': content,
                        'title': 'HTML Visualization',
                        'description': 'Data Visualization Agent HTML output',
                        'interactive': True,
                        'fallback': False
                    }
            elif '<svg' in content:
                # SVG content
                visualization_data = {
                    'type': 'svg',
                    'content': content,
                    'title': metadata.get('visualization_type', 'SVG Chart'),
                    'description': "Data Visualization Agent SVG output",
                    'interactive': metadata.get('contains_svg', True),
                    'fallback': False
                }
            else:
                # Default HTML
                visualization_data = {
                    'type': 'html',
                    'content': content,
                    'title': 'HTML Visualization',
                    'description': 'Data Visualization Agent output',
                    'interactive': True,
                    'fallback': False
                }

            return self.format_visualization_response(
                visualization_data, original_query
            )

        except Exception as e:
            logger.error(f"Data Visualization Agent result formatting failed: {e}")
            return self._create_error_response(original_query, str(e))


def get_visualization_response_formatter() -> VisualizationResponseFormatter:
    """Return a VisualizationResponseFormatter instance."""
    return VisualizationResponseFormatter()