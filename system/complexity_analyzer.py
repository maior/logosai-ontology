"""
Complexity Analysis Utilities
"""

from typing import Dict, Any, List
from loguru import logger

from ..core.models import SemanticQuery
from ..engines.execution_engine import QueryComplexityAnalyzer


class ComplexityAnalyzer:
    """Complexity analysis utility class"""
    
    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()
    
    def safe_analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Safe complexity analysis (with error handling)"""
        try:
            # Attempt basic complexity analysis
            complexity_result = self.complexity_analyzer.analyze_query_complexity(semantic_query)
            
            # Validate results and set default values
            if not isinstance(complexity_result, dict):
                logger.warning("Complexity analysis result is not a dict. Using default values")
                return self._get_default_complexity_analysis(semantic_query)
            
            # Check and supplement required fields
            required_fields = {
                'overall_complexity': 0.7,
                'cognitive_load': 0.6,
                'execution_complexity': 0.8,
                'data_complexity': 0.5,
                'reasoning_depth': 0.6,
                'interdependency_level': 0.4,
                'resource_requirements': 'moderate',
                'estimated_steps': 3,
                'confidence_score': 0.8
            }
            
            for field, default_value in required_fields.items():
                if field not in complexity_result:
                    complexity_result[field] = default_value
            
            # Validate score range (between 0 and 1)
            score_fields = ['overall_complexity', 'cognitive_load', 'execution_complexity', 
                          'data_complexity', 'reasoning_depth', 'interdependency_level', 'confidence_score']
            
            for field in score_fields:
                try:
                    score = float(complexity_result[field])
                    complexity_result[field] = max(0.0, min(1.0, score))  # restrict to 0-1 range
                except (ValueError, TypeError):
                    complexity_result[field] = required_fields[field]
            
            # Validate step count
            try:
                steps = int(complexity_result['estimated_steps'])
                complexity_result['estimated_steps'] = max(1, min(10, steps))  # restrict to 1-10 range
            except (ValueError, TypeError):
                complexity_result['estimated_steps'] = 3
            
            # Additional metadata
            complexity_result.update({
                'analysis_timestamp': semantic_query.created_at.isoformat() if semantic_query.created_at else None,
                'query_length': len(semantic_query.query_text),
                'entities_count': len(semantic_query.entities) if semantic_query.entities else 0,
                'concepts_count': len(semantic_query.concepts) if semantic_query.concepts else 0,
                'relations_count': len(semantic_query.relations) if semantic_query.relations else 0,
                'analysis_method': 'advanced_analyzer'
            })
            
            logger.info(f"✅ Complexity analysis complete: {complexity_result['overall_complexity']:.2f}")
            return complexity_result
            
        except Exception as e:
            logger.error(f"❌ Complexity analysis failed: {e}")
            return self._get_default_complexity_analysis(semantic_query)
    
    def _get_default_complexity_analysis(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """Return default complexity analysis result"""
        # Simple complexity estimation based on query text
        query_text = semantic_query.query_text.lower()
        
        # Keyword-based complexity estimation
        complexity_indicators = {
            'high': ['분석', 'compare', '비교', 'complex', '복잡', 'detailed', '상세', 'comprehensive'],
            'medium': ['explain', '설명', 'describe', '묘사', 'summarize', '요약', 'calculate'],
            'low': ['what', '무엇', 'who', '누구', 'when', '언제', 'where', '어디']
        }
        
        complexity_score = 0.5  # default value
        
        for level, keywords in complexity_indicators.items():
            if any(keyword in query_text for keyword in keywords):
                if level == 'high':
                    complexity_score = 0.8
                elif level == 'medium':
                    complexity_score = 0.6
                else:
                    complexity_score = 0.4
                break
        
        # Adjust based on query length
        if len(query_text) > 100:
            complexity_score += 0.1
        elif len(query_text) < 20:
            complexity_score -= 0.1
        
        complexity_score = max(0.1, min(0.9, complexity_score))
        
        return {
            'overall_complexity': complexity_score,
            'cognitive_load': complexity_score * 0.9,
            'execution_complexity': complexity_score * 1.1,
            'data_complexity': complexity_score * 0.8,
            'reasoning_depth': complexity_score,
            'interdependency_level': complexity_score * 0.7,
            'resource_requirements': 'moderate' if complexity_score > 0.6 else 'low',
            'estimated_steps': max(1, int(complexity_score * 5)),
            'confidence_score': 0.6,  # low confidence since this is a fallback analysis
            'analysis_timestamp': semantic_query.created_at.isoformat() if semantic_query.created_at else None,
            'query_length': len(semantic_query.query_text),
            'entities_count': len(semantic_query.entities) if semantic_query.entities else 0,
            'concepts_count': len(semantic_query.concepts) if semantic_query.concepts else 0,
            'relations_count': len(semantic_query.relations) if semantic_query.relations else 0,
            'analysis_method': 'fallback_analyzer',
            'fallback_reason': 'complexity_analyzer_unavailable'
        }
    
    def classify_result_type(self, result_data: Any) -> str:
        """Classify the type of result data"""
        try:
            if isinstance(result_data, dict):
                # Classify based on dictionary content
                if any(key in result_data for key in ['chart', 'graph', 'plot', 'visualization']):
                    return 'visualization'
                elif any(key in result_data for key in ['calculation', 'math', 'number', 'result']):
                    return 'calculation'
                elif any(key in result_data for key in ['analysis', 'insights', 'conclusion']):
                    return 'analysis'
                elif any(key in result_data for key in ['comparison', 'vs', 'versus', 'diff']):
                    return 'comparison'
                elif any(key in result_data for key in ['information', 'data', 'facts']):
                    return 'information'
                else:
                    return 'general'
            
            elif isinstance(result_data, str):
                # Classify based on string content
                content = result_data.lower()
                if any(word in content for word in ['chart', 'graph', 'plot', 'diagram']):
                    return 'visualization'
                elif any(word in content for word in ['calculation', 'math', 'number', '계산']):
                    return 'calculation'
                elif any(word in content for word in ['analysis', 'insight', '분석', '결론']):
                    return 'analysis'
                elif any(word in content for word in ['comparison', 'compare', '비교']):
                    return 'comparison'
                elif any(word in content for word in ['information', 'data', '정보', '데이터']):
                    return 'information'
                else:
                    return 'general'
            
            elif isinstance(result_data, (list, tuple)):
                # Determine from first element for list/tuple
                if len(result_data) > 0:
                    return self.classify_result_type(result_data[0])
                else:
                    return 'general'
            
            else:
                return 'general'
                
        except Exception as e:
            logger.warning(f"Result type classification failed: {e}")
            return 'general'
    
    def infer_agent_capabilities(self, agent_id: str) -> List[str]:
        """Infer capabilities based on agent ID"""
        capability_map = {
            'internet_agent': ['web_search', 'information_retrieval', 'real_time_data'],
            'finance_agent': ['financial_data', 'market_analysis', 'economic_indicators'],
            'weather_agent': ['weather_data', 'climate_information', 'forecasting'],
            'calculate_agent': ['mathematical_operations', 'calculations', 'data_processing'],
            'chart_agent': ['data_visualization', 'charts', 'graphs', 'plotting'],
            'memo_agent': ['note_taking', 'information_storage', 'text_processing'],
            'analysis_agent': ['data_analysis', 'pattern_recognition', 'insights_generation']
        }
        
        return capability_map.get(agent_id, ['general_purpose', 'task_execution']) 