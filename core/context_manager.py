"""
🔗 Context Manager
Context Manager

Manages result passing and context between agents
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class ExecutionContextManager:
    """Execution context manager"""
    
    def __init__(self):
        self.execution_contexts = {}  # session_id -> context
        
    def create_session_context(self, session_id: str, original_query: str) -> Dict[str, Any]:
        """Create a new session context"""
        context = {
            'session_id': session_id,
            'original_query': original_query,
            'created_at': datetime.now().isoformat(),
            'agent_results': {},  # agent_id -> result
            'task_results': {},   # task_id -> result
            'execution_flow': [], # Execution order
            'shared_data': {}     # Shared data
        }
        
        self.execution_contexts[session_id] = context
        return context
    
    def add_agent_result(self, session_id: str, agent_id: str, task_id: str, 
                        result: Dict[str, Any]) -> None:
        """Add agent execution result"""
        if session_id not in self.execution_contexts:
            logger.warning(f"Session {session_id} not found")
            return
        
        context = self.execution_contexts[session_id]
        
        # Store agent result
        context['agent_results'][agent_id] = {
            'task_id': task_id,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'summary': self._extract_result_summary(result)
        }
        
        # Store task result
        context['task_results'][task_id] = {
            'agent_id': agent_id,
            'result': result,
            'summary': self._extract_result_summary(result)
        }
        
        # Record execution order
        context['execution_flow'].append({
            'agent_id': agent_id,
            'task_id': task_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"✅ Context updated: {agent_id} → {task_id}")
    
    def get_context_for_agent(self, session_id: str, agent_id: str, 
                             task_id: str, dependencies: List[str]) -> Dict[str, Any]:
        """Create context for a specific agent"""
        if session_id not in self.execution_contexts:
            return {}
        
        context = self.execution_contexts[session_id]
        
        # Collect dependency results
        dependency_results = {}
        for dep_task_id in dependencies:
            if dep_task_id in context['task_results']:
                dep_result = context['task_results'][dep_task_id]
                dependency_results[dep_task_id] = {
                    'agent_id': dep_result['agent_id'],
                    'summary': dep_result['summary'],
                    'key_data': self._extract_key_data(dep_result['result'])
                }
        
        # Build context for agent
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
        """Create an enhanced query including context"""
        if not agent_context.get('dependencies'):
            return individual_query
        
        # Integrate dependency results into query
        context_parts = []
        
        for dep_task_id, dep_data in agent_context['dependencies'].items():
            agent_name = dep_data['agent_id']
            summary = dep_data['summary']
            context_parts.append(f"{agent_name} result: {summary}")
        
        if context_parts:
            context_info = "\n".join(context_parts)
            enhanced_query = f"""Previous task results:
{context_info}

Using the information above, perform the following task:
{individual_query}"""
        else:
            enhanced_query = individual_query
        
        return enhanced_query
    
    def _extract_result_summary(self, result: Dict[str, Any]) -> str:
        """Extract summary from result"""
        if isinstance(result, dict):
            # Extract summary from various result formats
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
        
        return str(result)[:200] if result else "No result"
    
    def _extract_key_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key data from result"""
        key_data = {}
        
        if isinstance(result, dict):
            # Extract numeric data
            for key in ['price', 'value', 'amount', 'rate', 'temperature']:
                if key in result:
                    key_data[key] = result[key]
            
            # Extract by data type
            if 'data' in result and isinstance(result['data'], dict):
                data = result['data']
                # Financial data
                if any(key in data for key in ['price', 'rate', 'exchange_rate']):
                    key_data['financial_data'] = {
                        k: v for k, v in data.items() 
                        if k in ['price', 'rate', 'exchange_rate', 'change', 'volume']
                    }
                # Weather data
                if any(key in data for key in ['temperature', 'humidity', 'condition']):
                    key_data['weather_data'] = {
                        k: v for k, v in data.items()
                        if k in ['temperature', 'humidity', 'condition', 'wind_speed']
                    }
        
        return key_data
    
    def _get_previous_results(self, context: Dict[str, Any], 
                            current_agent_id: str) -> List[Dict[str, Any]]:
        """Return previous execution results"""
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
        
        return previous_results[-3:]  # Return only last 3
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Session execution summary"""
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
    """Return singleton instance"""
    global _context_manager_instance
    if '_context_manager_instance' not in globals():
        _context_manager_instance = ExecutionContextManager()
    return _context_manager_instance