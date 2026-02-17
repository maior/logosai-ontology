"""
🔍 Agent Metadata Extractor
Agent Metadata Extractor

Extracts and structures detailed agent information
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class AgentMetadataExtractor:
    """Extract and structure agent metadata"""
    
    def extract_agent_metadata(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract rich metadata from agent information"""
        
        agent_id = agent_info.get('agent_id', '')
        agent_data = agent_info.get('agent_data', {})
        
        # Basic information
        metadata = {
            'agent_id': agent_id,
            'name': agent_data.get('name', agent_id),
            'description': agent_data.get('description', ''),
            'agent_type': self._extract_agent_type(agent_data),
            'endpoint': agent_data.get('endpoint', ''),
            'api_key': agent_data.get('api_key', '')
        }
        
        # Extract capabilities
        capabilities = agent_data.get('capabilities', [])
        metadata['capabilities'] = self._extract_capabilities(capabilities)
        metadata['capability_names'] = [cap.get('name', '') for cap in capabilities if isinstance(cap, dict)]
        
        # Extract and normalize tags
        tags = agent_data.get('tags', [])
        metadata['tags'] = self._normalize_tags(tags)
        metadata['tag_categories'] = self._categorize_tags(metadata['tags'])
        
        # Extract examples
        examples = agent_data.get('examples', [])
        metadata['examples'] = examples
        metadata['example_patterns'] = self._extract_patterns_from_examples(examples)
        
        # Extract domains
        metadata['domains'] = self._extract_domains(agent_data)
        
        # Detect special abilities
        metadata['special_abilities'] = self._detect_special_abilities(metadata)
        
        return metadata
    
    def _extract_agent_type(self, agent_data: Dict[str, Any]) -> str:
        """Extract agent type"""
        # Check metadata first
        if 'metadata' in agent_data and 'agent_type' in agent_data['metadata']:
            return agent_data['metadata']['agent_type']
        
        # Check agent_type field
        if 'agent_type' in agent_data:
            return agent_data['agent_type']
        
        # Infer from name or description
        name = agent_data.get('name', '').lower()
        desc = agent_data.get('description', '').lower()
        
        type_keywords = {
            'CRAWLER': ['crawler', '크롤러', '크롤링'],
            'SEARCH': ['search', '검색', '조회'],
            'ANALYSIS': ['analysis', '분석', 'analyze'],
            'CALCULATION': ['calculator', '계산', 'calc'],
            'VISUALIZATION': ['visual', '시각화', 'chart', '차트', 'graph'],
            'PLANNER': ['planner', '계획', 'schedule', '일정'],
            'WEATHER': ['weather', '날씨'],
            'STOCK': ['stock', '주식', '주가'],
            'CURRENCY': ['currency', '환율', 'exchange']
        }
        
        for agent_type, keywords in type_keywords.items():
            if any(keyword in name or keyword in desc for keyword in keywords):
                return agent_type
        
        return 'GENERAL'
    
    def _extract_capabilities(self, capabilities: List[Any]) -> List[Dict[str, Any]]:
        """Extract and normalize capability information"""
        normalized = []
        
        for cap in capabilities:
            if isinstance(cap, dict):
                normalized.append({
                    'name': cap.get('name', 'Unknown'),
                    'description': cap.get('description', ''),
                    'category': self._categorize_capability(cap.get('name', ''))
                })
            elif isinstance(cap, str):
                normalized.append({
                    'name': cap,
                    'description': cap,
                    'category': self._categorize_capability(cap)
                })
        
        return normalized
    
    def _categorize_capability(self, capability_name: str) -> str:
        """Categorize capability"""
        cap_lower = capability_name.lower()
        
        if any(word in cap_lower for word in ['검색', 'search', '조회', 'find']):
            return 'search'
        elif any(word in cap_lower for word in ['분석', 'analysis', 'analyze']):
            return 'analysis'
        elif any(word in cap_lower for word in ['계산', 'calculate', 'compute']):
            return 'calculation'
        elif any(word in cap_lower for word in ['시각화', 'visual', 'chart', 'graph']):
            return 'visualization'
        elif any(word in cap_lower for word in ['생성', 'create', 'generate']):
            return 'generation'
        else:
            return 'general'
    
    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tags"""
        normalized = []
        
        for tag in tags:
            if isinstance(tag, str):
                # Convert to lowercase and strip whitespace
                normalized_tag = tag.lower().strip()
                if normalized_tag:
                    normalized.append(normalized_tag)
        
        return list(set(normalized))  # Remove duplicates
    
    def _categorize_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """Categorize tags by category"""
        categories = {
            'domain': [],
            'function': [],
            'data_type': [],
            'output_format': []
        }
        
        for tag in tags:
            if tag in ['여행', '금융', '날씨', 'travel', 'finance', 'weather']:
                categories['domain'].append(tag)
            elif tag in ['검색', '분석', '계산', 'search', 'analysis', 'calculation']:
                categories['function'].append(tag)
            elif tag in ['텍스트', '숫자', '이미지', 'text', 'number', 'image']:
                categories['data_type'].append(tag)
            elif tag in ['표', '차트', '플로우차트', 'table', 'chart', 'flowchart']:
                categories['output_format'].append(tag)
        
        return categories
    
    def _extract_patterns_from_examples(self, examples: List[str]) -> List[str]:
        """Extract patterns from examples"""
        patterns = []
        
        for example in examples:
            if isinstance(example, str):
                # Simple pattern extraction (more sophisticated logic needed in production)
                if '계산' in example or 'calculate' in example:
                    patterns.append('calculation_task')
                elif '조회' in example or '확인' in example or 'check' in example:
                    patterns.append('information_retrieval')
                elif '분석' in example or 'analyze' in example:
                    patterns.append('analysis_task')
                elif '생성' in example or 'create' in example:
                    patterns.append('generation_task')
        
        return list(set(patterns))
    
    def _extract_domains(self, agent_data: Dict[str, Any]) -> List[str]:
        """Extract domains handled by the agent"""
        domains = []
        
        # Extract domain from name, description, and tags
        text_to_analyze = ' '.join([
            agent_data.get('name', ''),
            agent_data.get('description', ''),
            ' '.join(agent_data.get('tags', []))
        ]).lower()
        
        domain_keywords = {
            'travel': ['여행', '관광', 'travel', 'tour'],
            'finance': ['금융', '주식', '환율', 'finance', 'stock', 'currency'],
            'weather': ['날씨', '기상', 'weather', 'climate'],
            'technology': ['기술', 'IT', 'tech', 'AI'],
            'data': ['데이터', '정보', 'data', 'information']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ['general']
    
    def _detect_special_abilities(self, metadata: Dict[str, Any]) -> List[str]:
        """Detect special abilities"""
        special_abilities = []
        
        # Flowchart generation ability
        if any('flowchart' in tag or '플로우차트' in tag for tag in metadata['tags']):
            special_abilities.append('flowchart_generation')
        
        # Real-time data processing ability
        if any('실시간' in tag or 'realtime' in tag for tag in metadata['tags']):
            special_abilities.append('realtime_data')
        
        # Multimodal ability
        if any('이미지' in cap['name'] or 'image' in cap['name'] 
               for cap in metadata['capabilities']):
            special_abilities.append('multimodal')
        
        return special_abilities


def get_agent_metadata_extractor() -> AgentMetadataExtractor:
    """Return singleton instance"""
    global _extractor_instance
    if '_extractor_instance' not in globals():
        _extractor_instance = AgentMetadataExtractor()
    return _extractor_instance