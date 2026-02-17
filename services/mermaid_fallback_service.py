"""
🔄 Mermaid Fallback Service

Generates Mermaid diagrams using an LLM when data_visualization_agent is not installed.
"""

import json
import re
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.llm_manager import get_ontology_llm_manager, OntologyLLMType


class MermaidFallbackService:
    """Mermaid fallback service."""

    def __init__(self):
        self.llm_manager = get_ontology_llm_manager()

    async def generate_mermaid_from_agent_results(
        self,
        original_query: str,
        agent_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a Mermaid diagram based on agent results."""
        try:
            logger.info(f"🔄 Starting Mermaid fallback generation - {len(agent_results)} results")

            # 1. Extract data from agent results
            extracted_data = self._extract_data_from_results(agent_results)

            # 2. Determine visualization type
            viz_type = self._determine_visualization_type(original_query)

            # 3. Generate Mermaid by type
            if viz_type == "flowchart":
                mermaid_result = await self._generate_flowchart_mermaid(
                    original_query, extracted_data
                )
            elif viz_type == "timeline":
                mermaid_result = await self._generate_timeline_mermaid(
                    original_query, extracted_data
                )
            elif viz_type == "mindmap":
                mermaid_result = await self._generate_mindmap_mermaid(
                    original_query, extracted_data
                )
            elif viz_type == "chart":
                mermaid_result = await self._generate_chart_mermaid(
                    original_query, extracted_data
                )
            else:
                mermaid_result = await self._generate_generic_mermaid(
                    original_query, extracted_data
                )
            
            logger.info(f"✅ Mermaid fallback generation complete: {viz_type}")
            return mermaid_result

        except Exception as e:
            logger.error(f"Mermaid fallback generation failed: {e}")
            return self._create_error_fallback(original_query, str(e))
    
    def _extract_data_from_results(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract visualizable data from agent results."""
        extracted_data = {
            'agents': [],
            'data_points': [],
            'processes': [],
            'relationships': [],
            'timeline_events': []
        }
        
        for result in agent_results:
            try:
                agent_id = result.get('agent_id', 'unknown')
                agent_data = result.get('data', {})

                # Add agent information
                extracted_data['agents'].append({
                    'id': agent_id,
                    'name': agent_id.replace('_', ' ').title(),
                    'success': result.get('success', False),
                    'data': agent_data
                })
                
                # Extract data points
                if isinstance(agent_data, dict):
                    for key, value in agent_data.items():
                        if isinstance(value, (int, float)):
                            extracted_data['data_points'].append({
                                'label': key,
                                'value': value,
                                'source': agent_id
                            })
                        elif isinstance(value, str) and len(value) < 100:
                            extracted_data['processes'].append({
                                'name': key,
                                'description': value,
                                'source': agent_id
                            })

                # Extract relationships (between agents)
                if len(extracted_data['agents']) > 1:
                    prev_agent = extracted_data['agents'][-2]['id']
                    curr_agent = agent_id
                    extracted_data['relationships'].append({
                        'from': prev_agent,
                        'to': curr_agent,
                        'type': 'sequence'
                    })
                        
            except Exception as e:
                logger.warning(f"Data extraction failed for {result.get('agent_id', 'unknown')}: {e}")
                continue
        
        return extracted_data
    
    def _determine_visualization_type(self, query: str) -> str:
        """Determine visualization type by analyzing the query."""
        query_lower = query.lower()

        # Flowchart keywords
        if any(keyword in query_lower for keyword in [
            '플로우차트', '순서도', '과정', '단계', '절차', 'flowchart', 'process', 'workflow'
        ]):
            return "flowchart"

        # Timeline keywords
        if any(keyword in query_lower for keyword in [
            '타임라인', '일정', '스케줄', '시간순', 'timeline', 'schedule'
        ]):
            return "timeline"

        # Mindmap keywords
        if any(keyword in query_lower for keyword in [
            '마인드맵', '아이디어', '구조', 'mindmap', 'mind map', 'idea'
        ]):
            return "mindmap"

        # Chart keywords
        if any(keyword in query_lower for keyword in [
            '차트', '그래프', '통계', '데이터', 'chart', 'graph', 'data'
        ]):
            return "chart"

        # Default
        return "generic"
    
    async def _generate_flowchart_mermaid(
        self,
        query: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a flowchart Mermaid diagram."""
        
        prompt = f"""
사용자 쿼리: {query}

다음 에이전트 실행 결과를 바탕으로 플로우차트 Mermaid 다이어그램을 생성하세요:

에이전트 정보:
{json.dumps(data['agents'], ensure_ascii=False, indent=2)}

프로세스 정보:
{json.dumps(data['processes'], ensure_ascii=False, indent=2)}

요구사항:
1. 에이전트 실행 순서를 플로우차트로 표현
2. 각 단계의 성공/실패 상태 표시
3. 명확한 시작과 종료 노드 포함
4. 한국어 라벨 사용

응답 형식:
```mermaid
graph TD
    A[시작] --> B[단계1]
    B --> C[단계2]
    C --> D[완료]
```

Mermaid 코드만 응답하세요.
"""
        
        try:
            response = await self.llm_manager.call_llm_async(
                OntologyLLMType.WORKFLOW_DESIGNER, prompt
            )
            
            mermaid_code = self._extract_mermaid_code(response)
            
            return {
                'type': 'mermaid',
                'format': 'flowchart',
                'content': mermaid_code,
                'title': 'Processing Flowchart',
                'description': f'{len(data["agents"])} agent execution process',
                'data': data,
                'fallback': True
            }

        except Exception as e:
            logger.error(f"Flowchart generation failed: {e}")
            return self._create_basic_flowchart(data)
    
    async def _generate_timeline_mermaid(
        self,
        query: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a timeline Mermaid diagram."""
        
        prompt = f"""
사용자 쿼리: {query}

다음 에이전트 실행 결과를 바탕으로 타임라인 Mermaid 다이어그램을 생성하세요:

에이전트 정보:
{json.dumps(data['agents'], ensure_ascii=False, indent=2)}

요구사항:
1. 에이전트 실행 순서를 시간순으로 표현
2. 각 단계의 처리 시간 표시 (가능한 경우)
3. 한국어 라벨 사용

응답 형식:
```mermaid
timeline
    title 처리 과정 타임라인
    
    1단계 : 에이전트1 실행
           : 결과 처리
    2단계 : 에이전트2 실행
           : 결과 처리
```

Mermaid 코드만 응답하세요.
"""
        
        try:
            response = await self.llm_manager.call_llm_async(
                OntologyLLMType.WORKFLOW_DESIGNER, prompt
            )
            
            mermaid_code = self._extract_mermaid_code(response)
            
            return {
                'type': 'mermaid',
                'format': 'timeline',
                'content': mermaid_code,
                'title': 'Processing Timeline',
                'description': f'{len(data["agents"])} agent execution timeline',
                'data': data,
                'fallback': True
            }

        except Exception as e:
            logger.error(f"Timeline generation failed: {e}")
            return self._create_basic_timeline(data)
    
    async def _generate_mindmap_mermaid(
        self,
        query: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a mindmap Mermaid diagram."""
        
        prompt = f"""
사용자 쿼리: {query}

다음 에이전트 실행 결과를 바탕으로 마인드맵 Mermaid 다이어그램을 생성하세요:

에이전트 정보:
{json.dumps(data['agents'], ensure_ascii=False, indent=2)}

프로세스 정보:
{json.dumps(data['processes'], ensure_ascii=False, indent=2)}

요구사항:
1. 중심 주제는 사용자 쿼리 기반
2. 각 에이전트를 주요 브랜치로 표현
3. 하위 항목으로 처리 결과 표시
4. 한국어 라벨 사용

응답 형식:
```mermaid
mindmap
  root((중심 주제))
    에이전트1
      결과1
      결과2
    에이전트2
      결과3
      결과4
```

Mermaid 코드만 응답하세요.
"""
        
        try:
            response = await self.llm_manager.call_llm_async(
                OntologyLLMType.WORKFLOW_DESIGNER, prompt
            )
            
            mermaid_code = self._extract_mermaid_code(response)
            
            return {
                'type': 'mermaid',
                'format': 'mindmap',
                'content': mermaid_code,
                'title': 'Processing Result Mindmap',
                'description': f'{len(data["agents"])} agent results structured',
                'data': data,
                'fallback': True
            }

        except Exception as e:
            logger.error(f"Mindmap generation failed: {e}")
            return self._create_basic_mindmap(data)
    
    async def _generate_chart_mermaid(
        self,
        query: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a chart Mermaid diagram."""

        # Build chart if data points are available
        if data['data_points']:
            chart_data = []
            for point in data['data_points']:
                chart_data.append(f"{point['label']}: {point['value']}")

            mermaid_code = f"""graph LR
    subgraph "Data Chart"
        {chr(10).join([f'    {chr(65+i)}["{item}"]' for i, item in enumerate(chart_data)])}
    end"""
        else:
            mermaid_code = self._create_basic_chart_fallback(data)

        return {
            'type': 'mermaid',
            'format': 'chart',
            'content': mermaid_code,
            'title': 'Data Chart',
            'description': f'{len(data["data_points"])} data points',
            'data': data,
            'fallback': True
        }
    
    async def _generate_generic_mermaid(
        self,
        query: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a generic Mermaid diagram."""

        # Represent the agent execution process as a simple flowchart
        return await self._generate_flowchart_mermaid(query, data)
    
    def _extract_mermaid_code(self, response: str) -> str:
        """Extract Mermaid code from an LLM response."""
        try:
            # Find ```mermaid block
            mermaid_match = re.search(r'```mermaid\s*(.*?)\s*```', response, re.DOTALL)
            if mermaid_match:
                return mermaid_match.group(1).strip()

            # Find ``` block
            code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            # If no block found, use the full response
            return response.strip()

        except Exception as e:
            logger.error(f"Mermaid code extraction failed: {e}")
            return "graph TD\n    A[Error] --> B[Code extraction failed]"
    
    def _create_basic_flowchart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic flowchart."""
        mermaid_lines = ["graph TD"]
        mermaid_lines.append('    A[Start] --> B[Processing Begin]')
        
        for i, agent in enumerate(data['agents']):
            node_id = chr(67 + i)  # C, D, E, ...
            agent_name = agent['name']
            status = "✅" if agent['success'] else "❌"
            mermaid_lines.append(f'    {chr(66 + i)} --> {node_id}["{status} {agent_name}"]')
        
        mermaid_lines.append(f'    {chr(66 + len(data["agents"]))} --> Z[Complete]')

        return {
            'type': 'mermaid',
            'format': 'flowchart',
            'content': '\n'.join(mermaid_lines),
            'title': 'Basic Processing Flow',
            'description': f'{len(data["agents"])} agent processing flow',
            'data': data,
            'fallback': True
        }
    
    def _create_basic_timeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic timeline."""
        mermaid_lines = ["timeline"]
        mermaid_lines.append('    title Processing Timeline')
        mermaid_lines.append('')

        for i, agent in enumerate(data['agents']):
            step_num = i + 1
            agent_name = agent['name']
            status = "Complete" if agent['success'] else "Failed"
            mermaid_lines.append(f'    Step {step_num} : {agent_name} execution')
            mermaid_lines.append(f'            : {status}')

        return {
            'type': 'mermaid',
            'format': 'timeline',
            'content': '\n'.join(mermaid_lines),
            'title': 'Basic Timeline',
            'description': f'{len(data["agents"])} agent execution timeline',
            'data': data,
            'fallback': True
        }
    
    def _create_basic_mindmap(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic mindmap."""
        mermaid_lines = ["mindmap"]
        mermaid_lines.append('  root((Processing Result))')

        for agent in data['agents']:
            agent_name = agent['name']
            status = "Success" if agent['success'] else "Failed"
            mermaid_lines.append(f'    {agent_name}')
            mermaid_lines.append(f'      {status}')

        return {
            'type': 'mermaid',
            'format': 'mindmap',
            'content': '\n'.join(mermaid_lines),
            'title': 'Basic Mindmap',
            'description': f'{len(data["agents"])} agent results structured',
            'data': data,
            'fallback': True
        }
    
    def _create_basic_chart_fallback(self, data: Dict[str, Any]) -> str:
        """Basic chart fallback."""
        mermaid_lines = ["graph LR"]
        mermaid_lines.append('    subgraph "Agent Execution Results"')

        for i, agent in enumerate(data['agents']):
            node_id = chr(65 + i)
            agent_name = agent['name']
            status = "Success" if agent['success'] else "Failed"
            mermaid_lines.append(f'        {node_id}["{agent_name}<br/>{status}"]')

        mermaid_lines.append('    end')
        return '\n'.join(mermaid_lines)
    
    def _create_error_fallback(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Create an error fallback diagram."""
        mermaid_code = f"""graph TD
    A[Query: {query[:30]}...] --> B[Error during processing]
    B --> C[Error: {error_msg[:50]}...]
    C --> D[Generating default response]"""

        return {
            'type': 'mermaid',
            'format': 'error',
            'content': mermaid_code,
            'title': 'Processing Error',
            'description': f'Default response due to error: {error_msg}',
            'error': error_msg,
            'fallback': True
        }


def get_mermaid_fallback_service() -> MermaidFallbackService:
    """Return a MermaidFallbackService instance."""
    return MermaidFallbackService()