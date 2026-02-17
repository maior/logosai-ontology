"""
🚀 Ontology System Strategy Manager
System that integrally manages execution and optimization strategies
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# Import from the correct models module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.models import ExecutionStrategy, OptimizationStrategy, WorkflowComplexity, QueryType, AgentType


@dataclass
class StrategyAnalysisResult:
    """Strategy analysis result"""
    execution_strategy: ExecutionStrategy
    optimization_strategy: OptimizationStrategy
    complexity_level: WorkflowComplexity
    reasoning: str
    confidence: float
    estimated_time: float
    estimated_quality: float
    parallel_potential: bool
    agent_count: int


class StrategyManager:
    """Strategy Manager - determines execution and optimization strategies"""
    
    def __init__(self):
        self.strategy_history = []
        self.performance_metrics = {
            'single_agent': {'success_rate': 0.9, 'avg_time': 15.0, 'quality': 0.8},
            'sequential': {'success_rate': 0.85, 'avg_time': 45.0, 'quality': 0.9},
            'parallel': {'success_rate': 0.75, 'avg_time': 20.0, 'quality': 0.85},
            'hybrid': {'success_rate': 0.8, 'avg_time': 35.0, 'quality': 0.92}
        }
    
    def get_strategy_for_execution_plan(self, execution_plan: Dict[str, Any]) -> Tuple[ExecutionStrategy, OptimizationStrategy]:
        """Determine strategy based on execution plan"""
        strategy_str = execution_plan.get('strategy', 'sequential')
        
        # ExecutionStrategy mapping
        execution_strategy_mapping = {
            'single_agent': ExecutionStrategy.SINGLE_AGENT,
            'parallel': ExecutionStrategy.PARALLEL,
            'sequential': ExecutionStrategy.SEQUENTIAL,
            'hybrid': ExecutionStrategy.HYBRID
        }
        
        execution_strategy = execution_strategy_mapping.get(strategy_str, ExecutionStrategy.SEQUENTIAL)
        
        # OptimizationStrategy mapping
        optimization_strategy_mapping = {
            ExecutionStrategy.SINGLE_AGENT: OptimizationStrategy.SPEED_FIRST,
            ExecutionStrategy.PARALLEL: OptimizationStrategy.BALANCED,
            ExecutionStrategy.SEQUENTIAL: OptimizationStrategy.QUALITY_FIRST,
            ExecutionStrategy.HYBRID: OptimizationStrategy.BALANCED
        }
        
        optimization_strategy = optimization_strategy_mapping.get(execution_strategy, OptimizationStrategy.BALANCED)
        
        logger.info(f"🎯 Strategy mapping complete: {strategy_str} -> {execution_strategy.value} / {optimization_strategy.value}")
        
        return execution_strategy, optimization_strategy
    
    def analyze_query_complexity(self, query_text: str, agent_count: int = 1) -> Dict[str, Any]:
        """Analyze query complexity"""
        query_lower = query_text.lower()
        
        # Calculate complexity indicators
        indicators = {
            'has_multiple_tasks': any(word in query_lower for word in ['그리고', '또한', ',', '다음']),  # and, also, next
            'has_comparison': any(word in query_lower for word in ['비교', '차이', '대비', 'vs']),  # compare, difference, contrast
            'has_analysis': any(word in query_lower for word in ['분석', '평가', '검토', '조사']),  # analysis, evaluation, review, investigation
            'has_calculation': any(word in query_lower for word in ['계산', '산출', '구하']),  # calculate, compute, find
            'has_visualization': any(word in query_lower for word in ['차트', '그래프', '표', '그림']),  # chart, graph, table, figure
            'has_time_dependency': any(word in query_lower for word in ['먼저', '다음', '그 후', '이후']),  # first, next, after, then
            'word_count': len(query_text.split())
        }
        
        # Calculate complexity score
        complexity_score = 0.0
        complexity_score += 0.2 if indicators['has_multiple_tasks'] else 0.0
        complexity_score += 0.3 if indicators['has_comparison'] else 0.0
        complexity_score += 0.2 if indicators['has_analysis'] else 0.0
        complexity_score += 0.1 if indicators['has_calculation'] else 0.0
        complexity_score += 0.1 if indicators['has_visualization'] else 0.0
        complexity_score += 0.2 if indicators['has_time_dependency'] else 0.0
        complexity_score += min(indicators['word_count'] / 50.0, 0.3)
        
        # Determine recommended strategy
        if complexity_score <= 0.3 or agent_count == 1:
            recommended_strategy = 'single_agent'
        elif indicators['has_time_dependency']:
            recommended_strategy = 'sequential'
        elif indicators['has_multiple_tasks'] and not indicators['has_time_dependency']:
            recommended_strategy = 'parallel'
        elif complexity_score >= 0.7:
            recommended_strategy = 'hybrid'
        else:
            recommended_strategy = 'sequential'
        
        return {
            'complexity_score': min(complexity_score, 1.0),
            'indicators': indicators,
            'recommended_strategy': recommended_strategy,
            'estimated_time': self._estimate_time_for_strategy(recommended_strategy, complexity_score),
            'agent_count': agent_count
        }
    
    def _estimate_time_for_strategy(self, strategy: str, complexity_score: float) -> float:
        """Calculate estimated time per strategy"""
        base_times = {
            'single_agent': 20.0,
            'sequential': 45.0,
            'parallel': 25.0,
            'hybrid': 35.0
        }
        
        base_time = base_times.get(strategy, 30.0)
        complexity_multiplier = 1.0 + complexity_score
        
        return base_time * complexity_multiplier
    
    def get_strategy_recommendations(self, query_text: str, agent_count: int = 1) -> Dict[str, Any]:
        """Recommend strategy based on query text"""
        analysis = self.analyze_query_complexity(query_text, agent_count)
        
        strategy_str = analysis['recommended_strategy']
        execution_strategy, optimization_strategy = self.get_strategy_for_execution_plan({
            'strategy': strategy_str
        })
        
        return {
            'execution_strategy': execution_strategy.value,
            'optimization_strategy': optimization_strategy.value,
            'complexity_score': analysis['complexity_score'],
            'reasoning': self._generate_reasoning(strategy_str, analysis['indicators']),
            'estimated_time': analysis['estimated_time'],
            'confidence': 0.8
        }
    
    def _generate_reasoning(self, strategy: str, indicators: Dict[str, Any]) -> str:
        """Generate strategy selection reasoning"""
        reasons = []
        
        if strategy == 'single_agent':
            reasons.append("Simple query - single agent processing")
        elif strategy == 'parallel':
            reasons.append("Independent multi-tasks - parallel processing")
        elif strategy == 'sequential':
            reasons.append("Sequential processing due to dependencies or complexity")
        elif strategy == 'hybrid':
            reasons.append("High complexity - hybrid strategy")
        
        if indicators['has_comparison']:
            reasons.append("Comparative analysis required")
        if indicators['has_time_dependency']:
            reasons.append("Temporal dependency exists")
        if indicators['has_analysis']:
            reasons.append("Deep analysis required")
        
        return " | ".join(reasons)
    
    def update_performance_metrics(self, 
                                 strategy: str, 
                                 execution_time: float, 
                                 success: bool, 
                                 quality_score: float = None):
        """Update performance metrics"""
        if strategy in self.performance_metrics:
            metrics = self.performance_metrics[strategy]
            
            # Update success rate (moving average)
            current_success_rate = metrics['success_rate']
            metrics['success_rate'] = (current_success_rate * 0.9 + (1.0 if success else 0.0) * 0.1)
            
            # Update average time
            current_avg_time = metrics['avg_time']
            metrics['avg_time'] = (current_avg_time * 0.9 + execution_time * 0.1)
            
            # Update quality score
            if quality_score is not None:
                current_quality = metrics['quality']
                metrics['quality'] = (current_quality * 0.9 + quality_score * 0.1)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Performance summary information"""
        return {
            'metrics': self.performance_metrics,
            'history_count': len(self.strategy_history),
            'recent_strategies': self.strategy_history[-5:] if self.strategy_history else []
        } 