"""
🔗 Context Manager
컨텍스트 관리자

에이전트 간 결과 전달 및 컨텍스트 관리
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class ExecutionContextManager:
    """실행 컨텍스트 관리자"""
    
    def __init__(self):
        self.execution_contexts = {}  # session_id -> context
        
    def create_session_context(self, session_id: str, original_query: str) -> Dict[str, Any]:
        """새 세션 컨텍스트 생성"""
        context = {
            'session_id': session_id,
            'original_query': original_query,
            'created_at': datetime.now().isoformat(),
            'agent_results': {},  # agent_id -> result
            'task_results': {},   # task_id -> result
            'execution_flow': [], # 실행 순서
            'shared_data': {}     # 공유 데이터
        }
        
        self.execution_contexts[session_id] = context
        return context
    
    def add_agent_result(self, session_id: str, agent_id: str, task_id: str, 
                        result: Dict[str, Any]) -> None:
        """에이전트 실행 결과 추가"""
        if session_id not in self.execution_contexts:
            logger.warning(f"세션 {session_id}를 찾을 수 없음")
            return
        
        context = self.execution_contexts[session_id]
        
        # 에이전트 결과 저장
        context['agent_results'][agent_id] = {
            'task_id': task_id,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'summary': self._extract_result_summary(result)
        }
        
        # 태스크 결과 저장
        context['task_results'][task_id] = {
            'agent_id': agent_id,
            'result': result,
            'summary': self._extract_result_summary(result)
        }
        
        # 실행 순서 기록
        context['execution_flow'].append({
            'agent_id': agent_id,
            'task_id': task_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"✅ 컨텍스트 업데이트: {agent_id} → {task_id}")
    
    def get_context_for_agent(self, session_id: str, agent_id: str, 
                             task_id: str, dependencies: List[str]) -> Dict[str, Any]:
        """특정 에이전트를 위한 컨텍스트 생성"""
        if session_id not in self.execution_contexts:
            return {}
        
        context = self.execution_contexts[session_id]
        
        # 의존성 결과 수집
        dependency_results = {}
        for dep_task_id in dependencies:
            if dep_task_id in context['task_results']:
                dep_result = context['task_results'][dep_task_id]
                dependency_results[dep_task_id] = {
                    'agent_id': dep_result['agent_id'],
                    'summary': dep_result['summary'],
                    'key_data': self._extract_key_data(dep_result['result'])
                }
        
        # 에이전트용 컨텍스트 구성
        agent_context = {
            'original_query': context['original_query'],
            'current_task_id': task_id,
            'dependencies': dependency_results,
            'previous_results': self._get_previous_results(context, agent_id),
            'shared_data': context['shared_data']
        }
        
        return agent_context
    
    def create_enhanced_query(self, original_query: str, agent_context: Dict[str, Any], 
                            individual_query: str) -> str:
        """컨텍스트를 포함한 향상된 쿼리 생성"""
        if not agent_context.get('dependencies'):
            return individual_query
        
        # 의존성 결과를 쿼리에 통합
        context_parts = []
        
        for dep_task_id, dep_data in agent_context['dependencies'].items():
            agent_name = dep_data['agent_id']
            summary = dep_data['summary']
            context_parts.append(f"{agent_name}의 결과: {summary}")
        
        if context_parts:
            context_info = "\n".join(context_parts)
            enhanced_query = f"""이전 작업 결과:
{context_info}

위 정보를 참고하여 다음 작업을 수행하세요:
{individual_query}"""
        else:
            enhanced_query = individual_query
        
        return enhanced_query
    
    def _extract_result_summary(self, result: Dict[str, Any]) -> str:
        """결과에서 요약 추출"""
        if isinstance(result, dict):
            # 다양한 형태의 결과에서 요약 추출
            if 'summary' in result:
                return str(result['summary'])
            elif 'data' in result and isinstance(result['data'], dict):
                data = result['data']
                if 'answer' in data:
                    return str(data['answer'])[:200]
                elif 'content' in data:
                    return str(data['content'])[:200]
                elif 'result' in data:
                    return str(data['result'])[:200]
        
        return str(result)[:200] if result else "결과 없음"
    
    def _extract_key_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """결과에서 핵심 데이터 추출"""
        key_data = {}
        
        if isinstance(result, dict):
            # 숫자 데이터 추출
            for key in ['price', 'value', 'amount', 'rate', 'temperature']:
                if key in result:
                    key_data[key] = result[key]
            
            # 데이터 타입별 추출
            if 'data' in result and isinstance(result['data'], dict):
                data = result['data']
                # 금융 데이터
                if any(key in data for key in ['price', 'rate', 'exchange_rate']):
                    key_data['financial_data'] = {
                        k: v for k, v in data.items() 
                        if k in ['price', 'rate', 'exchange_rate', 'change', 'volume']
                    }
                # 날씨 데이터
                if any(key in data for key in ['temperature', 'humidity', 'condition']):
                    key_data['weather_data'] = {
                        k: v for k, v in data.items()
                        if k in ['temperature', 'humidity', 'condition', 'wind_speed']
                    }
        
        return key_data
    
    def _get_previous_results(self, context: Dict[str, Any], 
                            current_agent_id: str) -> List[Dict[str, Any]]:
        """이전 실행 결과들 반환"""
        previous_results = []
        
        for flow_item in context['execution_flow']:
            if flow_item['agent_id'] != current_agent_id:
                agent_id = flow_item['agent_id']
                if agent_id in context['agent_results']:
                    previous_results.append({
                        'agent_id': agent_id,
                        'summary': context['agent_results'][agent_id]['summary'],
                        'timestamp': flow_item['timestamp']
                    })
        
        return previous_results[-3:]  # 최근 3개만 반환
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """세션 실행 요약"""
        if session_id not in self.execution_contexts:
            return {}
        
        context = self.execution_contexts[session_id]
        
        return {
            'session_id': session_id,
            'original_query': context['original_query'],
            'total_agents': len(context['agent_results']),
            'execution_flow': context['execution_flow'],
            'created_at': context['created_at'],
            'agent_summaries': {
                agent_id: result['summary']
                for agent_id, result in context['agent_results'].items()
            }
        }


def get_execution_context_manager() -> ExecutionContextManager:
    """싱글톤 인스턴스 반환"""
    global _context_manager_instance
    if '_context_manager_instance' not in globals():
        _context_manager_instance = ExecutionContextManager()
    return _context_manager_instance