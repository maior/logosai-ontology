"""
복잡도 분석 유틸리티
"""

from typing import Dict, Any, List
from loguru import logger

from ..core.models import SemanticQuery
from ..engines.execution_engine import QueryComplexityAnalyzer


class ComplexityAnalyzer:
    """복잡도 분석 유틸리티 클래스"""
    
    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()
    
    def safe_analyze_complexity(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """안전한 복잡도 분석 (오류 처리 포함)"""
        try:
            # 기본 복잡도 분석 시도
            complexity_result = self.complexity_analyzer.analyze_query_complexity(semantic_query)
            
            # 결과 검증 및 기본값 설정
            if not isinstance(complexity_result, dict):
                logger.warning("복잡도 분석 결과가 딕셔너리가 아님. 기본값 사용")
                return self._get_default_complexity_analysis(semantic_query)
            
            # 필수 필드 확인 및 보완
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
            
            # 점수 범위 검증 (0-1 사이)
            score_fields = ['overall_complexity', 'cognitive_load', 'execution_complexity', 
                          'data_complexity', 'reasoning_depth', 'interdependency_level', 'confidence_score']
            
            for field in score_fields:
                try:
                    score = float(complexity_result[field])
                    complexity_result[field] = max(0.0, min(1.0, score))  # 0-1 범위로 제한
                except (ValueError, TypeError):
                    complexity_result[field] = required_fields[field]
            
            # 단계 수 검증
            try:
                steps = int(complexity_result['estimated_steps'])
                complexity_result['estimated_steps'] = max(1, min(10, steps))  # 1-10 범위로 제한
            except (ValueError, TypeError):
                complexity_result['estimated_steps'] = 3
            
            # 추가 메타데이터
            complexity_result.update({
                'analysis_timestamp': semantic_query.created_at.isoformat() if semantic_query.created_at else None,
                'query_length': len(semantic_query.query_text),
                'entities_count': len(semantic_query.entities) if semantic_query.entities else 0,
                'concepts_count': len(semantic_query.concepts) if semantic_query.concepts else 0,
                'relations_count': len(semantic_query.relations) if semantic_query.relations else 0,
                'analysis_method': 'advanced_analyzer'
            })
            
            logger.info(f"✅ 복잡도 분석 완료: {complexity_result['overall_complexity']:.2f}")
            return complexity_result
            
        except Exception as e:
            logger.error(f"❌ 복잡도 분석 실패: {e}")
            return self._get_default_complexity_analysis(semantic_query)
    
    def _get_default_complexity_analysis(self, semantic_query: SemanticQuery) -> Dict[str, Any]:
        """기본 복잡도 분석 결과 반환"""
        # 쿼리 텍스트 기반 간단한 복잡도 추정
        query_text = semantic_query.query_text.lower()
        
        # 키워드 기반 복잡도 추정
        complexity_indicators = {
            'high': ['분석', 'compare', '비교', 'complex', '복잡', 'detailed', '상세', 'comprehensive'],
            'medium': ['explain', '설명', 'describe', '묘사', 'summarize', '요약', 'calculate'],
            'low': ['what', '무엇', 'who', '누구', 'when', '언제', 'where', '어디']
        }
        
        complexity_score = 0.5  # 기본값
        
        for level, keywords in complexity_indicators.items():
            if any(keyword in query_text for keyword in keywords):
                if level == 'high':
                    complexity_score = 0.8
                elif level == 'medium':
                    complexity_score = 0.6
                else:
                    complexity_score = 0.4
                break
        
        # 쿼리 길이 기반 조정
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
            'confidence_score': 0.6,  # 기본 분석이므로 낮은 신뢰도
            'analysis_timestamp': semantic_query.created_at.isoformat() if semantic_query.created_at else None,
            'query_length': len(semantic_query.query_text),
            'entities_count': len(semantic_query.entities) if semantic_query.entities else 0,
            'concepts_count': len(semantic_query.concepts) if semantic_query.concepts else 0,
            'relations_count': len(semantic_query.relations) if semantic_query.relations else 0,
            'analysis_method': 'fallback_analyzer',
            'fallback_reason': 'complexity_analyzer_unavailable'
        }
    
    def classify_result_type(self, result_data: Any) -> str:
        """결과 데이터의 타입을 분류"""
        try:
            if isinstance(result_data, dict):
                # 딕셔너리 내용 기반 분류
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
                # 문자열 내용 기반 분류
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
                # 리스트/튜플인 경우 첫 번째 요소로 판단
                if len(result_data) > 0:
                    return self.classify_result_type(result_data[0])
                else:
                    return 'general'
            
            else:
                return 'general'
                
        except Exception as e:
            logger.warning(f"결과 타입 분류 실패: {e}")
            return 'general'
    
    def infer_agent_capabilities(self, agent_id: str) -> List[str]:
        """에이전트 ID를 기반으로 능력 추론"""
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