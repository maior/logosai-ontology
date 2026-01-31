"""
Performance Analyzer for Ontology System
성능 분석 및 메트릭 수집 유틸리티
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """개별 성능 메트릭"""
    name: str
    value: float
    unit: str = "ms"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    온톨로지 시스템 성능 분석기

    기능:
    - 실행 시간 측정
    - 성능 메트릭 수집
    - 통계 분석
    """

    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self._enabled = True

    def start_timer(self, operation: str) -> None:
        """작업 타이머 시작"""
        if self._enabled:
            self.start_times[operation] = time.time()

    def stop_timer(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """작업 타이머 종료 및 메트릭 기록"""
        if not self._enabled or operation not in self.start_times:
            return 0.0

        elapsed = (time.time() - self.start_times[operation]) * 1000  # ms로 변환
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
        """커스텀 메트릭 기록"""
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
        """특정 작업의 통계 반환"""
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
        """모든 작업의 통계 반환"""
        return {op: self.get_statistics(op) for op in self.metrics}

    def get_recent_metrics(self, operation: str, count: int = 10) -> List[Dict[str, Any]]:
        """최근 메트릭 반환"""
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
        """메트릭 초기화"""
        if operation:
            self.metrics[operation] = []
            if operation in self.start_times:
                del self.start_times[operation]
        else:
            self.metrics.clear()
            self.start_times.clear()

    def enable(self) -> None:
        """성능 분석 활성화"""
        self._enabled = True

    def disable(self) -> None:
        """성능 분석 비활성화"""
        self._enabled = False

    def is_enabled(self) -> bool:
        """활성화 상태 확인"""
        return self._enabled

    def get_summary(self) -> Dict[str, Any]:
        """전체 성능 요약"""
        all_stats = self.get_all_statistics()
        total_operations = sum(s["count"] for s in all_stats.values())
        total_time = sum(s["total"] for s in all_stats.values())

        return {
            "enabled": self._enabled,
            "total_operations": total_operations,
            "total_time_ms": total_time,
            "operations": all_stats
        }


# 싱글톤 인스턴스
performance_analyzer = PerformanceAnalyzer()
