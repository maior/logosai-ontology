"""
🔍 Agent Metadata Extractor
에이전트 메타데이터 추출기

에이전트의 상세 정보를 추출하고 구조화
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class AgentMetadataExtractor:
    """에이전트 메타데이터 추출 및 구조화"""
    
    def extract_agent_metadata(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 정보에서 풍부한 메타데이터 추출"""
        
        agent_id = agent_info.get('agent_id', '')
        agent_data = agent_info.get('agent_data', {})
        
        # 기본 정보
        metadata = {
            'agent_id': agent_id,
            'name': agent_data.get('name', agent_id),
            'description': agent_data.get('description', ''),
            'agent_type': self._extract_agent_type(agent_data),
            'endpoint': agent_data.get('endpoint', ''),
            'api_key': agent_data.get('api_key', '')
        }
        
        # capabilities 추출
        capabilities = agent_data.get('capabilities', [])
        metadata['capabilities'] = self._extract_capabilities(capabilities)
        metadata['capability_names'] = [cap.get('name', '') for cap in capabilities if isinstance(cap, dict)]
        
        # tags 추출 및 정규화
        tags = agent_data.get('tags', [])
        metadata['tags'] = self._normalize_tags(tags)
        metadata['tag_categories'] = self._categorize_tags(metadata['tags'])
        
        # examples 추출
        examples = agent_data.get('examples', [])
        metadata['examples'] = examples
        metadata['example_patterns'] = self._extract_patterns_from_examples(examples)
        
        # 도메인 추출
        metadata['domains'] = self._extract_domains(agent_data)
        
        # 특수 능력 감지
        metadata['special_abilities'] = self._detect_special_abilities(metadata)
        
        return metadata
    
    def _extract_agent_type(self, agent_data: Dict[str, Any]) -> str:
        """에이전트 타입 추출"""
        # metadata에서 먼저 확인
        if 'metadata' in agent_data and 'agent_type' in agent_data['metadata']:
            return agent_data['metadata']['agent_type']
        
        # agent_type 필드 확인
        if 'agent_type' in agent_data:
            return agent_data['agent_type']
        
        # 이름이나 설명에서 추론
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
        """능력 정보 추출 및 정규화"""
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
        """능력을 카테고리로 분류"""
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
        """태그 정규화"""
        normalized = []
        
        for tag in tags:
            if isinstance(tag, str):
                # 소문자로 변환하고 공백 제거
                normalized_tag = tag.lower().strip()
                if normalized_tag:
                    normalized.append(normalized_tag)
        
        return list(set(normalized))  # 중복 제거
    
    def _categorize_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """태그를 카테고리별로 분류"""
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
        """예제에서 패턴 추출"""
        patterns = []
        
        for example in examples:
            if isinstance(example, str):
                # 간단한 패턴 추출 (실제로는 더 정교한 로직 필요)
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
        """에이전트가 다루는 도메인 추출"""
        domains = []
        
        # 이름, 설명, 태그에서 도메인 추출
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
        """특수 능력 감지"""
        special_abilities = []
        
        # 플로우차트 생성 능력
        if any('flowchart' in tag or '플로우차트' in tag for tag in metadata['tags']):
            special_abilities.append('flowchart_generation')
        
        # 실시간 데이터 처리 능력
        if any('실시간' in tag or 'realtime' in tag for tag in metadata['tags']):
            special_abilities.append('realtime_data')
        
        # 멀티모달 능력
        if any('이미지' in cap['name'] or 'image' in cap['name'] 
               for cap in metadata['capabilities']):
            special_abilities.append('multimodal')
        
        return special_abilities


def get_agent_metadata_extractor() -> AgentMetadataExtractor:
    """싱글톤 인스턴스 반환"""
    global _extractor_instance
    if '_extractor_instance' not in globals():
        _extractor_instance = AgentMetadataExtractor()
    return _extractor_instance