"""
📊 System Metrics Manager Module
시스템 메트릭 관리 모듈

성능 추적, 상태 모니터링, 메트릭 분석을 담당
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ..core.models import (
    SemanticQuery, WorkflowPlan, AgentExecutionResult,
    get_system_metrics, SystemMetrics
)


class MetricsManager:
    """📊 메트릭 관리자 - 시스템 성능 추적 및 분석"""
    
    def __init__(self):
        self.metrics = get_system_metrics()
        self.start_time = time.time()
        
        logger.info("📊 메트릭 관리자 초기화 완료")
    
    def record_workflow_execution(self, 
                                 semantic_query: SemanticQuery,
                                 workflow_plan: WorkflowPlan,
                                 execution_results: List[AgentExecutionResult],
                                 integrated_result: Dict[str, Any],
                                 total_execution_time: float):
        """워크플로우 실행 기록"""
        try:
            # 기본 메트릭 업데이트
            self.metrics.total_queries += 1
            
            # 실행 시간 업데이트
            total_time = self.metrics.average_execution_time * (self.metrics.analysis_calls) + total_execution_time
            self.metrics.analysis_calls += 1
            self.metrics.average_execution_time = total_time / self.metrics.analysis_calls
            
            # 성공/실패 기록
            if not integrated_result.get('success', False):
                self.metrics.failed_executions += 1
            
            # 병렬 실행 기록
            if len(execution_results) > 1:
                self.metrics.parallel_executions += 1
            
            # LLM 관련 메트릭 업데이트
            if hasattr(integrated_result, 'llm_enhanced') and integrated_result.get('llm_enhanced'):
                self.metrics.llm_enhancement_count += 1
            
            if hasattr(workflow_plan, 'llm_optimized'):
                self.metrics.llm_optimization_count += 1
            
            logger.info(f"📊 워크플로우 실행 기록 완료: {semantic_query.query_id}")
            
        except Exception as e:
            logger.error(f"워크플로우 실행 기록 실패: {e}")
    
    def record_agent_performance(self, execution_results: List[AgentExecutionResult]):
        """에이전트 성능 기록"""
        try:
            for result in execution_results:
                # 개별 에이전트 성능 추적
                agent_id = result.agent_id
                
                # 에이전트별 성능 히스토리 (간단한 형태)
                if not hasattr(self, 'agent_history'):
                    self.agent_history = {}
                
                if agent_id not in self.agent_history:
                    self.agent_history[agent_id] = {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'total_time': 0.0,
                        'average_confidence': 0.0
                    }
                
                history = self.agent_history[agent_id]
                history['total_executions'] += 1
                
                if result.success:
                    history['successful_executions'] += 1
                
                history['total_time'] += result.execution_time
                
                # 평균 신뢰도 업데이트  
                current_confidence = getattr(result, 'confidence', 0.8)
                total_confidence = history['average_confidence'] * (history['total_executions'] - 1) + current_confidence
                history['average_confidence'] = total_confidence / history['total_executions']
            
            logger.info(f"📊 에이전트 성능 기록 완료: {len(execution_results)}개 에이전트")
            
        except Exception as e:
            logger.error(f"에이전트 성능 기록 실패: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            status = {
                'system_status': 'healthy' if self.metrics.get_success_rate() > 80 else 'warning' if self.metrics.get_success_rate() > 50 else 'critical',
                'uptime_seconds': uptime,
                'uptime_readable': self._format_uptime(uptime),
                'total_queries': self.metrics.total_queries,
                'success_rate': self.metrics.get_success_rate(),
                'cache_hit_rate': self.metrics.get_cache_hit_rate(),
                'average_response_time': self.metrics.average_response_time,
                'average_execution_time': self.metrics.average_execution_time,
                'parallel_executions': self.metrics.parallel_executions,
                'failed_executions': self.metrics.failed_executions,
                'duplicate_calls_prevented': self.metrics.duplicate_calls_prevented,
                
                # LLM 관련 메트릭
                'llm_metrics': {
                    'total_calls': self.metrics.llm_calls_total,
                    'success_rate': self.metrics.get_llm_success_rate(),
                    'cache_hit_rate': self.metrics.get_llm_cache_hit_rate(),
                    'average_response_time': self.metrics.average_llm_response_time,
                    'enhancement_count': self.metrics.llm_enhancement_count,
                    'optimization_count': self.metrics.llm_optimization_count
                },
                
                # 에이전트 성능 (상위 5개)
                'top_agents': self._get_top_performing_agents(),
                
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """상세 메트릭 조회"""
        try:
            detailed_metrics = self.metrics.to_dict()
            
            # 추가 계산된 메트릭
            detailed_metrics['calculated_metrics'] = {
                'queries_per_minute': self._calculate_queries_per_minute(),
                'average_agents_per_workflow': self._calculate_average_agents_per_workflow(),
                'efficiency_score': self._calculate_efficiency_score(),
                'llm_utilization_rate': self._calculate_llm_utilization_rate()
            }
            
            # 에이전트별 상세 통계
            detailed_metrics['agent_statistics'] = self._get_agent_statistics()
            
            # 시간대별 성능 (간단한 형태)
            detailed_metrics['performance_trends'] = self._get_performance_trends()
            
            return detailed_metrics
            
        except Exception as e:
            logger.error(f"상세 메트릭 조회 실패: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self) -> str:
        """성능 보고서 생성"""
        try:
            status = self.get_system_status()
            
            report_lines = []
            report_lines.append("=" * 50)
            report_lines.append("🔍 ONTOLOGY SYSTEM PERFORMANCE REPORT")
            report_lines.append("=" * 50)
            
            # 시스템 상태
            status_emoji = "🟢" if status['system_status'] == 'healthy' else "🟡" if status['system_status'] == 'warning' else "🔴"
            report_lines.append(f"{status_emoji} System Status: {status['system_status'].upper()}")
            report_lines.append(f"⏱️  Uptime: {status['uptime_readable']}")
            report_lines.append("")
            
            # 기본 메트릭
            report_lines.append("📊 BASIC METRICS:")
            report_lines.append(f"  • Total Queries: {status['total_queries']}")
            report_lines.append(f"  • Success Rate: {status['success_rate']:.1f}%")
            report_lines.append(f"  • Cache Hit Rate: {status['cache_hit_rate']:.1f}%")
            report_lines.append(f"  • Avg Response Time: {status['average_response_time']:.2f}s")
            report_lines.append(f"  • Avg Execution Time: {status['average_execution_time']:.2f}s")
            report_lines.append("")
            
            # 효율성 메트릭
            report_lines.append("⚡ EFFICIENCY METRICS:")
            report_lines.append(f"  • Parallel Executions: {status['parallel_executions']}")
            report_lines.append(f"  • Duplicate Calls Prevented: {status['duplicate_calls_prevented']}")
            report_lines.append(f"  • Failed Executions: {status['failed_executions']}")
            report_lines.append("")
            
            # LLM 메트릭
            llm_metrics = status['llm_metrics']
            report_lines.append("🧠 LLM METRICS:")
            report_lines.append(f"  • Total LLM Calls: {llm_metrics['total_calls']}")
            report_lines.append(f"  • LLM Success Rate: {llm_metrics['success_rate']:.1f}%")
            report_lines.append(f"  • LLM Cache Hit Rate: {llm_metrics['cache_hit_rate']:.1f}%")
            report_lines.append(f"  • LLM Avg Response Time: {llm_metrics['average_response_time']:.2f}s")
            report_lines.append(f"  • Query Enhancements: {llm_metrics['enhancement_count']}")
            report_lines.append(f"  • Workflow Optimizations: {llm_metrics['optimization_count']}")
            report_lines.append("")
            
            # 상위 성능 에이전트
            top_agents = status['top_agents']
            if top_agents:
                report_lines.append("🏆 TOP PERFORMING AGENTS:")
                for i, agent in enumerate(top_agents[:5], 1):
                    report_lines.append(f"  {i}. {agent['agent_id']}: {agent['success_rate']:.1f}% success, {agent['avg_time']:.2f}s avg")
                report_lines.append("")
            
            # 권장사항
            recommendations = self._generate_recommendations(status)
            if recommendations:
                report_lines.append("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    report_lines.append(f"  • {rec}")
                report_lines.append("")
            
            report_lines.append("=" * 50)
            report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("=" * 50)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"성능 보고서 생성 실패: {e}")
            return f"Error generating performance report: {str(e)}"
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """업타임을 읽기 쉬운 형식으로 변환"""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _get_top_performing_agents(self) -> List[Dict[str, Any]]:
        """상위 성능 에이전트 조회"""
        try:
            if not hasattr(self, 'agent_history'):
                return []
            
            agent_stats = []
            
            for agent_id, history in self.agent_history.items():
                if history['total_executions'] > 0:
                    success_rate = (history['successful_executions'] / history['total_executions']) * 100
                    avg_time = history['total_time'] / history['total_executions']
                    
                    agent_stats.append({
                        'agent_id': agent_id,
                        'success_rate': success_rate,
                        'avg_time': avg_time,
                        'total_executions': history['total_executions'],
                        'avg_confidence': history['average_confidence']
                    })
            
            # 성공률 기준으로 정렬
            agent_stats.sort(key=lambda x: (x['success_rate'], -x['avg_time']), reverse=True)
            
            return agent_stats[:10]  # 상위 10개
            
        except Exception as e:
            logger.error(f"상위 성능 에이전트 조회 실패: {e}")
            return []
    
    def _calculate_queries_per_minute(self) -> float:
        """분당 쿼리 수 계산"""
        try:
            current_time = time.time()
            uptime_minutes = (current_time - self.start_time) / 60.0
            
            if uptime_minutes > 0:
                return self.metrics.total_queries / uptime_minutes
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_average_agents_per_workflow(self) -> float:
        """워크플로우당 평균 에이전트 수 계산"""
        try:
            if self.metrics.total_queries > 0:
                # 근사치: 병렬 실행 기록을 바탕으로 추정
                total_agent_calls = self.metrics.total_queries + self.metrics.parallel_executions
                return total_agent_calls / self.metrics.total_queries
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_efficiency_score(self) -> float:
        """효율성 점수 계산 (0-100)"""
        try:
            # 여러 지표를 종합한 효율성 점수
            success_weight = 0.4
            cache_weight = 0.2
            speed_weight = 0.2
            parallel_weight = 0.2
            
            success_score = self.metrics.get_success_rate()
            cache_score = self.metrics.get_cache_hit_rate()
            
            # 속도 점수 (평균 실행 시간이 낮을수록 높은 점수)
            speed_score = max(0, 100 - (self.metrics.average_execution_time * 2))
            
            # 병렬성 점수
            parallel_ratio = self.metrics.parallel_executions / max(1, self.metrics.total_queries)
            parallel_score = min(100, parallel_ratio * 200)  # 50% 병렬 실행 시 100점
            
            efficiency = (
                success_score * success_weight +
                cache_score * cache_weight +
                speed_score * speed_weight +
                parallel_score * parallel_weight
            )
            
            return round(efficiency, 1)
            
        except Exception:
            return 0.0
    
    def _calculate_llm_utilization_rate(self) -> float:
        """LLM 활용률 계산"""
        try:
            if self.metrics.total_queries > 0:
                llm_usage = (
                    self.metrics.llm_enhancement_count + 
                    self.metrics.llm_optimization_count
                ) / self.metrics.total_queries
                
                return round(llm_usage * 100, 1)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_agent_statistics(self) -> Dict[str, Any]:
        """에이전트 통계 조회"""
        try:
            if not hasattr(self, 'agent_history'):
                return {}
            
            stats = {
                'total_unique_agents': len(self.agent_history),
                'agents_detail': {}
            }
            
            for agent_id, history in self.agent_history.items():
                stats['agents_detail'][agent_id] = {
                    'total_executions': history['total_executions'],
                    'successful_executions': history['successful_executions'],
                    'success_rate': (history['successful_executions'] / history['total_executions'] * 100) if history['total_executions'] > 0 else 0,
                    'average_execution_time': history['total_time'] / history['total_executions'] if history['total_executions'] > 0 else 0,
                    'average_confidence': history['average_confidence']
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"에이전트 통계 조회 실패: {e}")
            return {}
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 조회 (간단한 형태)"""
        try:
            # 간단한 트렌드 정보 (실제로는 시계열 데이터가 필요)
            current_success_rate = self.metrics.get_success_rate()
            current_response_time = self.metrics.average_execution_time
            
            return {
                'current_success_rate': current_success_rate,
                'current_response_time': current_response_time,
                'trend_note': "시계열 데이터 수집을 위해서는 별도 저장소가 필요합니다."
            }
            
        except Exception:
            return {}
    
    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """성능 권장사항 생성"""
        recommendations = []
        
        try:
            # 성공률 기반 권장사항
            if status['success_rate'] < 80:
                recommendations.append("성공률이 낮습니다. 에이전트 설정 또는 워크플로우 설계를 검토해보세요.")
            
            # 캐시 히트율 기반 권장사항
            if status['cache_hit_rate'] < 30:
                recommendations.append("캐시 히트율이 낮습니다. 캐시 전략을 개선하여 성능을 향상시킬 수 있습니다.")
            
            # 응답 시간 기반 권장사항
            if status['average_execution_time'] > 60:
                recommendations.append("평균 실행 시간이 길습니다. 병렬 처리나 에이전트 최적화를 고려해보세요.")
            
            # LLM 활용률 기반 권장사항
            llm_utilization = self._calculate_llm_utilization_rate()
            if llm_utilization < 20:
                recommendations.append("LLM 활용률이 낮습니다. 쿼리 개선 및 워크플로우 최적화 기능을 더 활용해보세요.")
            
            # 병렬 실행 기반 권장사항
            if status['parallel_executions'] == 0 and status['total_queries'] > 10:
                recommendations.append("병렬 실행이 활용되지 않고 있습니다. 복잡한 작업에서 다중 에이전트 활용을 고려해보세요.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"권장사항 생성 실패: {e}")
            return ["권장사항 생성 중 오류가 발생했습니다."] 