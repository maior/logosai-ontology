"""
🔍 Agent Detector

Dynamically detects and manages installed agents.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import ExecutionContext


class AgentDetector:
    """Dynamically detects the installation status of agents."""

    def __init__(self, execution_context: ExecutionContext):
        self.execution_context = execution_context
        self.available_agents = execution_context.available_agents or {}

    def is_agent_installed(self, agent_id: str) -> bool:
        """Check whether a specific agent is installed."""
        try:
            return agent_id in self.available_agents
        except Exception as e:
            logger.error(f"Agent installation check failed for {agent_id}: {e}")
            return False

    def has_visualization_capability(self) -> bool:
        """Check whether any agent with visualization capability exists."""
        viz_agents = self.find_agents_with_capability('visualization')

        if viz_agents:
            logger.info(f"✅ Agents with visualization capability found: {viz_agents}")
            return True

        logger.warning("⚠️ No agents with visualization capability found")
        return False

    def get_best_visualization_agent(self) -> Optional[str]:
        """Return the best-suited visualization agent."""
        viz_agents = self.find_agents_with_capability('visualization')

        if not viz_agents:
            return None

        # Calculate agent scores
        scored_agents = []
        for agent_id in viz_agents:
            score = self._calculate_visualization_score(agent_id)
            scored_agents.append((agent_id, score))

        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        best_agent = scored_agents[0][0]
        logger.info(f"🎯 Selected visualization agent: {best_agent} (score: {scored_agents[0][1]})")
        return best_agent

    def find_agents_with_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        matching_agents = []

        for agent_id, agent_info in self.available_agents.items():
            if self._has_capability(agent_id, agent_info, capability):
                matching_agents.append(agent_id)

        logger.info(f"📊 Agents with '{capability}' capability: {matching_agents}")
        return matching_agents

    def _has_capability(self, agent_id: str, agent_info: Any, capability: str) -> bool:
        """Check whether an agent has a specific capability."""
        try:
            # 1. Check capabilities in agent metadata
            if isinstance(agent_info, dict):
                capabilities = agent_info.get('capabilities', [])
                if self._check_capability_in_list(capabilities, capability):
                    return True

                # Also check tags
                tags = agent_info.get('tags', [])
                if self._check_capability_in_list(tags, capability):
                    return True

                # Check description
                description = agent_info.get('description', '')
                if self._check_capability_in_text(description, capability):
                    return True

            # 2. Infer from agent ID
            if self._check_capability_in_text(agent_id, capability):
                return True

            # 3. Infer from agent name
            if hasattr(agent_info, 'name'):
                if self._check_capability_in_text(agent_info.name, capability):
                    return True
            elif isinstance(agent_info, dict) and 'name' in agent_info:
                if self._check_capability_in_text(agent_info['name'], capability):
                    return True

            return False

        except Exception as e:
            logger.warning(f"Capability check failed for agent {agent_id}: {e}")
            return False

    def _check_capability_in_list(self, items: List, capability: str) -> bool:
        """Check for capability in a list."""
        if not items:
            return False
        
        capability_keywords = self._get_capability_keywords(capability)
        
        for item in items:
            item_text = str(item).lower()
            for keyword in capability_keywords:
                if keyword in item_text:
                    return True
        
        return False
    
    def _check_capability_in_text(self, text: str, capability: str) -> bool:
        """Check for capability in text."""
        if not text:
            return False
        
        text_lower = text.lower()
        capability_keywords = self._get_capability_keywords(capability)
        
        return any(keyword in text_lower for keyword in capability_keywords)
    
    def _get_capability_keywords(self, capability: str) -> List[str]:
        """Keyword mapping per capability."""
        keyword_mapping = {
            'visualization': [
                'visual', 'chart', 'graph', 'plot', 'diagram', 'mermaid',
                'svg', 'd3', 'flowchart', 'timeline', 'mindmap',
                '시각화', '차트', '그래프', '플로우차트', '다이어그램'
            ],
            'analysis': [
                'analy', 'analyze', 'report', 'insight', 'statistic',
                '분석', '통계', '리포트', '인사이트'
            ],
            'calculation': [
                'calc', 'compute', 'math', 'formula', 'calculate',
                '계산', '수식', '수학', '연산'
            ],
            'search': [
                'search', 'find', 'lookup', 'query', 'retrieve',
                '검색', '조회', '찾기', '탐색'
            ],
            'generation': [
                'generate', 'create', 'build', 'make', 'produce',
                '생성', '제작', '만들기', '구축'
            ]
        }
        
        return keyword_mapping.get(capability, [capability])
    
    def _calculate_visualization_score(self, agent_id: str) -> float:
        """Calculate the score for a visualization agent."""
        try:
            agent_info = self.available_agents.get(agent_id, {})
            score = 0.0

            # 1. Score based on agent name
            name_keywords = {
                'data_visualization': 10.0,
                'chart': 8.0,
                'graph': 8.0,
                'visual': 7.0,
                'plot': 6.0,
                'diagram': 6.0,
                'mermaid': 5.0
            }
            
            agent_name = agent_id.lower()
            for keyword, points in name_keywords.items():
                if keyword in agent_name:
                    score += points
                    break
            
            # 2. Score based on capabilities
            if isinstance(agent_info, dict):
                capabilities = agent_info.get('capabilities', [])
                for cap in capabilities:
                    cap_text = str(cap).lower()
                    if 'visual' in cap_text or 'chart' in cap_text:
                        score += 5.0
                    elif 'graph' in cap_text or 'plot' in cap_text:
                        score += 4.0
                    elif 'diagram' in cap_text or 'mermaid' in cap_text:
                        score += 3.0

                # Score based on tags
                tags = agent_info.get('tags', [])
                for tag in tags:
                    tag_text = str(tag).lower()
                    if any(keyword in tag_text for keyword in ['visual', 'chart', 'graph']):
                        score += 2.0

            # 3. Baseline score (all agents)
            if score == 0:
                score = 1.0

            return score

        except Exception as e:
            logger.warning(f"Score calculation failed for agent {agent_id}: {e}")
            return 1.0

    def get_agent_metadata(self, agent_id: str) -> Dict[str, Any]:
        """Return agent metadata."""
        try:
            agent_info = self.available_agents.get(agent_id, {})
            if isinstance(agent_info, dict):
                return agent_info
            elif hasattr(agent_info, '__dict__'):
                return agent_info.__dict__
            else:
                return {
                    'agent_id': agent_id,
                    'name': agent_id,
                    'available': True
                }
        except Exception as e:
            logger.error(f"Agent metadata retrieval failed for {agent_id}: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        total_agents = len(self.available_agents)
        viz_agents = self.find_agents_with_capability('visualization')
        
        return {
            'total_agents': total_agents,
            'visualization_capable_agents': len(viz_agents),
            'visualization_agents_list': viz_agents,
            'has_visualization_capability': len(viz_agents) > 0,
            'best_visualization_agent': self.get_best_visualization_agent(),
            'capability_analysis': {
                'visualization': len(self.find_agents_with_capability('visualization')),
                'analysis': len(self.find_agents_with_capability('analysis')),
                'calculation': len(self.find_agents_with_capability('calculation')),
                'search': len(self.find_agents_with_capability('search')),
                'generation': len(self.find_agents_with_capability('generation'))
            }
        }


def get_agent_detector(execution_context: ExecutionContext) -> AgentDetector:
    """Return an AgentDetector instance."""
    return AgentDetector(execution_context)