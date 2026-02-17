"""
Performance Analyzer for Ontology System

Utility for performance analysis and metrics collection.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str = "ms"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    Ontology system performance analyzer.

    Features:
    - Execution time measurement
    - Performance metrics collection
    - Statistical analysis
    """

    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self._enabled = True

    def start_timer(self, operation: str) -> None:
        """Start a timer for an operation."""
        if self._enabled:
            self.start_times[operation] = time.time()

    def stop_timer(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Stop a timer for an operation and record the metric."""
        if not self._enabled or operation not in self.start_times:
            return 0.0

        elapsed = (time.time() - self.start_times[operation]) * 1000  # convert to ms
        del self.start_times[operation]

        metric = PerformanceMetric(
            name=operation,
            value=elapsed,
            unit="ms",
            metadata=metadata or {}
        )
        self.metrics[operation].append(metric)

        return elapsed

    def record_metric(self, name: str, value: float, unit: str = "ms",
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a custom metric."""
        if not self._enabled:
            return

        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        self.metrics[name].append(metric)

    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Return statistics for a specific operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "total": 0}

        values = [m.value for m in self.metrics[operation]]
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "total": sum(values)
        }

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Return statistics for all operations."""
        return {op: self.get_statistics(op) for op in self.metrics}

    def get_recent_metrics(self, operation: str, count: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent metrics for an operation."""
        if operation not in self.metrics:
            return []

        recent = self.metrics[operation][-count:]
        return [
            {
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "timestamp": m.timestamp,
                "metadata": m.metadata
            }
            for m in recent
        ]

    def reset(self, operation: Optional[str] = None) -> None:
        """Reset metrics."""
        if operation:
            self.metrics[operation] = []
            if operation in self.start_times:
                del self.start_times[operation]
        else:
            self.metrics.clear()
            self.start_times.clear()

    def enable(self) -> None:
        """Enable performance analysis."""
        self._enabled = True

    def disable(self) -> None:
        """Disable performance analysis."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check whether performance analysis is enabled."""
        return self._enabled

    def get_summary(self) -> Dict[str, Any]:
        """Return an overall performance summary."""
        all_stats = self.get_all_statistics()
        total_operations = sum(s["count"] for s in all_stats.values())
        total_time = sum(s["total"] for s in all_stats.values())

        return {
            "enabled": self._enabled,
            "total_operations": total_operations,
            "total_time_ms": total_time,
            "operations": all_stats
        }


# Singleton instance
performance_analyzer = PerformanceAnalyzer()
