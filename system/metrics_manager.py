"""
📊 System Metrics Manager Module
System Metrics Management Module

Responsible for performance tracking, status monitoring, and metrics analysis
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
    """📊 Metrics Manager - system performance tracking and analysis"""
    
    def __init__(self):
        self.metrics = get_system_metrics()
        self.start_time = time.time()
        
        logger.info("📊 Metrics manager initialized")
    
    def record_workflow_execution(self, 
                                 semantic_query: SemanticQuery,
                                 workflow_plan: WorkflowPlan,
                                 execution_results: List[AgentExecutionResult],
                                 integrated_result: Dict[str, Any],
                                 total_execution_time: float):
        """Record workflow execution"""
        try:
            # Update basic metrics
            self.metrics.total_queries += 1
            
            # Update execution time
            total_time = self.metrics.average_execution_time * (self.metrics.analysis_calls) + total_execution_time
            self.metrics.analysis_calls += 1
            self.metrics.average_execution_time = total_time / self.metrics.analysis_calls
            
            # Record success/failure
            if not integrated_result.get('success', False):
                self.metrics.failed_executions += 1
            
            # Record parallel executions
            if len(execution_results) > 1:
                self.metrics.parallel_executions += 1
            
            # Update LLM-related metrics
            if hasattr(integrated_result, 'llm_enhanced') and integrated_result.get('llm_enhanced'):
                self.metrics.llm_enhancement_count += 1
            
            if hasattr(workflow_plan, 'llm_optimized'):
                self.metrics.llm_optimization_count += 1
            
            logger.info(f"📊 Workflow execution recorded: {semantic_query.query_id}")
            
        except Exception as e:
            logger.error(f"Workflow execution recording failed: {e}")
    
    def record_agent_performance(self, execution_results: List[AgentExecutionResult]):
        """Record agent performance"""
        try:
            for result in execution_results:
                # Track individual agent performance
                agent_id = result.agent_id
                
                # Agent performance history (simple form)
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
                
                # Update average confidence  
                current_confidence = getattr(result, 'confidence', 0.8)
                total_confidence = history['average_confidence'] * (history['total_executions'] - 1) + current_confidence
                history['average_confidence'] = total_confidence / history['total_executions']
            
            logger.info(f"📊 Agent performance recorded: {len(execution_results)} agents")
            
        except Exception as e:
            logger.error(f"Agent performance recording failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
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
                
                # LLM-related metrics
                'llm_metrics': {
                    'total_calls': self.metrics.llm_calls_total,
                    'success_rate': self.metrics.get_llm_success_rate(),
                    'cache_hit_rate': self.metrics.get_llm_cache_hit_rate(),
                    'average_response_time': self.metrics.average_llm_response_time,
                    'enhancement_count': self.metrics.llm_enhancement_count,
                    'optimization_count': self.metrics.llm_optimization_count
                },
                
                # Agent performance (top 5)
                'top_agents': self._get_top_performing_agents(),
                
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"System status query failed: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        try:
            detailed_metrics = self.metrics.to_dict()
            
            # Additional calculated metrics
            detailed_metrics['calculated_metrics'] = {
                'queries_per_minute': self._calculate_queries_per_minute(),
                'average_agents_per_workflow': self._calculate_average_agents_per_workflow(),
                'efficiency_score': self._calculate_efficiency_score(),
                'llm_utilization_rate': self._calculate_llm_utilization_rate()
            }
            
            # Detailed statistics per agent
            detailed_metrics['agent_statistics'] = self._get_agent_statistics()
            
            # Performance by time period (simple form)
            detailed_metrics['performance_trends'] = self._get_performance_trends()
            
            return detailed_metrics
            
        except Exception as e:
            logger.error(f"Detailed metrics query failed: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self) -> str:
        """Generate performance report"""
        try:
            status = self.get_system_status()
            
            report_lines = []
            report_lines.append("=" * 50)
            report_lines.append("🔍 ONTOLOGY SYSTEM PERFORMANCE REPORT")
            report_lines.append("=" * 50)
            
            # System status
            status_emoji = "🟢" if status['system_status'] == 'healthy' else "🟡" if status['system_status'] == 'warning' else "🔴"
            report_lines.append(f"{status_emoji} System Status: {status['system_status'].upper()}")
            report_lines.append(f"⏱️  Uptime: {status['uptime_readable']}")
            report_lines.append("")
            
            # Basic metrics
            report_lines.append("📊 BASIC METRICS:")
            report_lines.append(f"  • Total Queries: {status['total_queries']}")
            report_lines.append(f"  • Success Rate: {status['success_rate']:.1f}%")
            report_lines.append(f"  • Cache Hit Rate: {status['cache_hit_rate']:.1f}%")
            report_lines.append(f"  • Avg Response Time: {status['average_response_time']:.2f}s")
            report_lines.append(f"  • Avg Execution Time: {status['average_execution_time']:.2f}s")
            report_lines.append("")
            
            # Efficiency metrics
            report_lines.append("⚡ EFFICIENCY METRICS:")
            report_lines.append(f"  • Parallel Executions: {status['parallel_executions']}")
            report_lines.append(f"  • Duplicate Calls Prevented: {status['duplicate_calls_prevented']}")
            report_lines.append(f"  • Failed Executions: {status['failed_executions']}")
            report_lines.append("")
            
            # LLM metrics
            llm_metrics = status['llm_metrics']
            report_lines.append("🧠 LLM METRICS:")
            report_lines.append(f"  • Total LLM Calls: {llm_metrics['total_calls']}")
            report_lines.append(f"  • LLM Success Rate: {llm_metrics['success_rate']:.1f}%")
            report_lines.append(f"  • LLM Cache Hit Rate: {llm_metrics['cache_hit_rate']:.1f}%")
            report_lines.append(f"  • LLM Avg Response Time: {llm_metrics['average_response_time']:.2f}s")
            report_lines.append(f"  • Query Enhancements: {llm_metrics['enhancement_count']}")
            report_lines.append(f"  • Workflow Optimizations: {llm_metrics['optimization_count']}")
            report_lines.append("")
            
            # Top performing agents
            top_agents = status['top_agents']
            if top_agents:
                report_lines.append("🏆 TOP PERFORMING AGENTS:")
                for i, agent in enumerate(top_agents[:5], 1):
                    report_lines.append(f"  {i}. {agent['agent_id']}: {agent['success_rate']:.1f}% success, {agent['avg_time']:.2f}s avg")
                report_lines.append("")
            
            # Recommendations
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
            logger.error(f"Performance report generation failed: {e}")
            return f"Error generating performance report: {str(e)}"
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Convert uptime to human-readable format"""
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
        """Get top performing agents"""
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
            
            # Sort by success rate
            agent_stats.sort(key=lambda x: (x['success_rate'], -x['avg_time']), reverse=True)
            
            return agent_stats[:10]  # top 10
            
        except Exception as e:
            logger.error(f"Top performing agents query failed: {e}")
            return []
    
    def _calculate_queries_per_minute(self) -> float:
        """Calculate queries per minute"""
        try:
            current_time = time.time()
            uptime_minutes = (current_time - self.start_time) / 60.0
            
            if uptime_minutes > 0:
                return self.metrics.total_queries / uptime_minutes
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_average_agents_per_workflow(self) -> float:
        """Calculate average agents per workflow"""
        try:
            if self.metrics.total_queries > 0:
                # Approximate: estimate based on parallel execution records
                total_agent_calls = self.metrics.total_queries + self.metrics.parallel_executions
                return total_agent_calls / self.metrics.total_queries
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score (0-100)"""
        try:
            # Efficiency score combining multiple indicators
            success_weight = 0.4
            cache_weight = 0.2
            speed_weight = 0.2
            parallel_weight = 0.2
            
            success_score = self.metrics.get_success_rate()
            cache_score = self.metrics.get_cache_hit_rate()
            
            # Speed score (lower average execution time = higher score)
            speed_score = max(0, 100 - (self.metrics.average_execution_time * 2))
            
            # Parallelism score
            parallel_ratio = self.metrics.parallel_executions / max(1, self.metrics.total_queries)
            parallel_score = min(100, parallel_ratio * 200)  # 100 points at 50% parallel execution
            
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
        """Calculate LLM utilization rate"""
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
        """Get agent statistics"""
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
            logger.error(f"Agent statistics query failed: {e}")
            return {}
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends (simple form)"""
        try:
            # Simple trend information (real implementation requires time-series data)
            current_success_rate = self.metrics.get_success_rate()
            current_response_time = self.metrics.average_execution_time
            
            return {
                'current_success_rate': current_success_rate,
                'current_response_time': current_response_time,
                'trend_note': "A separate data store is required for time-series data collection."
            }
            
        except Exception:
            return {}
    
    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            # Recommendations based on success rate
            if status['success_rate'] < 80:
                recommendations.append("Success rate is low. Please review agent configuration or workflow design.")
            
            # Recommendations based on cache hit rate
            if status['cache_hit_rate'] < 30:
                recommendations.append("Cache hit rate is low. Improving the caching strategy can enhance performance.")
            
            # Recommendations based on response time
            if status['average_execution_time'] > 60:
                recommendations.append("Average execution time is long. Consider parallel processing or agent optimization.")
            
            # Recommendations based on LLM utilization
            llm_utilization = self._calculate_llm_utilization_rate()
            if llm_utilization < 20:
                recommendations.append("LLM utilization is low. Consider leveraging query enhancement and workflow optimization features more.")
            
            # Recommendations based on parallel execution
            if status['parallel_executions'] == 0 and status['total_queries'] > 10:
                recommendations.append("Parallel execution is not being used. Consider leveraging multiple agents for complex tasks.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["An error occurred while generating recommendations."] 