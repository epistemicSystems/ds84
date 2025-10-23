"""Performance monitoring and profiling service"""
import time
import asyncio
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class EndpointMetrics:
    """Metrics for a specific API endpoint"""
    endpoint: str
    method: str
    request_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    durations: deque = field(default_factory=lambda: deque(maxlen=1000))
    status_codes: Dict[int, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def avg_duration_ms(self) -> float:
        """Average request duration"""
        return self.total_duration_ms / self.request_count if self.request_count > 0 else 0.0

    @property
    def p50_duration_ms(self) -> float:
        """50th percentile (median) duration"""
        if not self.durations:
            return 0.0
        return statistics.median(self.durations)

    @property
    def p95_duration_ms(self) -> float:
        """95th percentile duration"""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[idx] if idx < len(sorted_durations) else sorted_durations[-1]

    @property
    def p99_duration_ms(self) -> float:
        """99th percentile duration"""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[idx] if idx < len(sorted_durations) else sorted_durations[-1]

    @property
    def error_rate(self) -> float:
        """Error rate (0.0 to 1.0)"""
        return self.error_count / self.request_count if self.request_count > 0 else 0.0


class PerformanceMonitor:
    """Production performance monitoring service"""

    def __init__(self):
        """Initialize performance monitor"""
        self.endpoint_metrics: Dict[str, EndpointMetrics] = {}
        self.system_metrics: List[PerformanceMetric] = deque(maxlen=1000)
        self.start_time = datetime.now()
        self.lock = threading.RLock()

        # Start background monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()

    def record_request(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int
    ):
        """Record API request metrics

        Args:
            endpoint: API endpoint path
            method: HTTP method
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
        """
        with self.lock:
            key = f"{method}:{endpoint}"

            if key not in self.endpoint_metrics:
                self.endpoint_metrics[key] = EndpointMetrics(
                    endpoint=endpoint,
                    method=method
                )

            metrics = self.endpoint_metrics[key]
            metrics.request_count += 1
            metrics.total_duration_ms += duration_ms
            metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
            metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
            metrics.durations.append(duration_ms)
            metrics.last_updated = datetime.now()

            # Track status codes
            metrics.status_codes[status_code] = metrics.status_codes.get(status_code, 0) + 1

            # Track errors (4xx and 5xx)
            if status_code >= 400:
                metrics.error_count += 1

    def get_endpoint_metrics(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for specific endpoint or all endpoints

        Args:
            endpoint: Optional endpoint filter

        Returns:
            Dictionary of endpoint metrics
        """
        with self.lock:
            if endpoint:
                key = endpoint
                if key in self.endpoint_metrics:
                    return self._format_endpoint_metrics(self.endpoint_metrics[key])
                return {}

            # Return all endpoints
            return {
                key: self._format_endpoint_metrics(metrics)
                for key, metrics in self.endpoint_metrics.items()
            }

    def _format_endpoint_metrics(self, metrics: EndpointMetrics) -> Dict[str, Any]:
        """Format endpoint metrics for output

        Args:
            metrics: Endpoint metrics object

        Returns:
            Formatted metrics dictionary
        """
        return {
            "endpoint": metrics.endpoint,
            "method": metrics.method,
            "request_count": metrics.request_count,
            "error_count": metrics.error_count,
            "error_rate": round(metrics.error_rate, 4),
            "duration_ms": {
                "min": round(metrics.min_duration_ms, 2),
                "max": round(metrics.max_duration_ms, 2),
                "avg": round(metrics.avg_duration_ms, 2),
                "p50": round(metrics.p50_duration_ms, 2),
                "p95": round(metrics.p95_duration_ms, 2),
                "p99": round(metrics.p99_duration_ms, 2)
            },
            "status_codes": metrics.status_codes,
            "last_updated": metrics.last_updated.isoformat()
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics

        Returns:
            Dictionary of system metrics
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "cpu": {
                "percent": round(cpu_percent, 2),
                "count": psutil.cpu_count()
            },
            "memory": {
                "total_mb": round(memory.total / (1024 * 1024), 2),
                "available_mb": round(memory.available / (1024 * 1024), 2),
                "used_mb": round(memory.used / (1024 * 1024), 2),
                "percent": round(memory.percent, 2)
            },
            "disk": {
                "total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                "used_gb": round(disk.used / (1024 * 1024 * 1024), 2),
                "free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                "percent": round(disk.percent, 2)
            },
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }

    def get_top_endpoints(self, limit: int = 10, sort_by: str = "request_count") -> List[Dict[str, Any]]:
        """Get top endpoints by metric

        Args:
            limit: Number of endpoints to return
            sort_by: Metric to sort by (request_count, error_rate, avg_duration_ms)

        Returns:
            List of top endpoint metrics
        """
        with self.lock:
            endpoints = list(self.endpoint_metrics.values())

            # Sort by requested metric
            if sort_by == "request_count":
                endpoints.sort(key=lambda x: x.request_count, reverse=True)
            elif sort_by == "error_rate":
                endpoints.sort(key=lambda x: x.error_rate, reverse=True)
            elif sort_by == "avg_duration_ms":
                endpoints.sort(key=lambda x: x.avg_duration_ms, reverse=True)

            return [
                self._format_endpoint_metrics(endpoint)
                for endpoint in endpoints[:limit]
            ]

    def get_slow_endpoints(self, threshold_ms: float = 1000.0, limit: int = 10) -> List[Dict[str, Any]]:
        """Get endpoints with slow response times

        Args:
            threshold_ms: Duration threshold in milliseconds
            limit: Maximum number of endpoints to return

        Returns:
            List of slow endpoints
        """
        with self.lock:
            slow_endpoints = [
                metrics for metrics in self.endpoint_metrics.values()
                if metrics.p95_duration_ms > threshold_ms
            ]

            slow_endpoints.sort(key=lambda x: x.p95_duration_ms, reverse=True)

            return [
                self._format_endpoint_metrics(endpoint)
                for endpoint in slow_endpoints[:limit]
            ]

    def get_error_endpoints(self, min_error_rate: float = 0.05, limit: int = 10) -> List[Dict[str, Any]]:
        """Get endpoints with high error rates

        Args:
            min_error_rate: Minimum error rate threshold (0.0 to 1.0)
            limit: Maximum number of endpoints to return

        Returns:
            List of endpoints with high error rates
        """
        with self.lock:
            error_endpoints = [
                metrics for metrics in self.endpoint_metrics.values()
                if metrics.error_rate >= min_error_rate and metrics.request_count >= 10
            ]

            error_endpoints.sort(key=lambda x: x.error_rate, reverse=True)

            return [
                self._format_endpoint_metrics(endpoint)
                for endpoint in error_endpoints[:limit]
            ]

    def get_summary(self) -> Dict[str, Any]:
        """Get overall system summary

        Returns:
            Summary of all metrics
        """
        with self.lock:
            total_requests = sum(m.request_count for m in self.endpoint_metrics.values())
            total_errors = sum(m.error_count for m in self.endpoint_metrics.values())

            if total_requests > 0:
                overall_error_rate = total_errors / total_requests
                all_durations = []
                for metrics in self.endpoint_metrics.values():
                    all_durations.extend(metrics.durations)

                if all_durations:
                    avg_duration = statistics.mean(all_durations)
                    sorted_durations = sorted(all_durations)
                    p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
                    p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]
                else:
                    avg_duration = 0.0
                    p95_duration = 0.0
                    p99_duration = 0.0
            else:
                overall_error_rate = 0.0
                avg_duration = 0.0
                p95_duration = 0.0
                p99_duration = 0.0

            system_metrics = self.get_system_metrics()

            return {
                "uptime_seconds": system_metrics["uptime_seconds"],
                "requests": {
                    "total": total_requests,
                    "errors": total_errors,
                    "error_rate": round(overall_error_rate, 4)
                },
                "performance": {
                    "avg_duration_ms": round(avg_duration, 2),
                    "p95_duration_ms": round(p95_duration, 2),
                    "p99_duration_ms": round(p99_duration, 2)
                },
                "endpoints": {
                    "total": len(self.endpoint_metrics),
                    "with_errors": len([m for m in self.endpoint_metrics.values() if m.error_count > 0])
                },
                "system": system_metrics
            }

    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.endpoint_metrics.clear()
            self.system_metrics.clear()

    def _monitor_system(self):
        """Background thread to collect system metrics"""
        while self._monitoring:
            try:
                # Collect system metrics every 60 seconds
                time.sleep(60)

                metrics = self.get_system_metrics()

                # Store as performance metrics
                with self.lock:
                    self.system_metrics.append(PerformanceMetric(
                        name="cpu_percent",
                        value=metrics["cpu"]["percent"],
                        timestamp=datetime.now(),
                        unit="percent"
                    ))
                    self.system_metrics.append(PerformanceMetric(
                        name="memory_percent",
                        value=metrics["memory"]["percent"],
                        timestamp=datetime.now(),
                        unit="percent"
                    ))
                    self.system_metrics.append(PerformanceMetric(
                        name="disk_percent",
                        value=metrics["disk"]["percent"],
                        timestamp=datetime.now(),
                        unit="percent"
                    ))

            except Exception:
                # Ignore errors in background monitoring
                pass

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Middleware decorator for automatic request tracking
def track_performance(func: Callable):
    """Decorator to track function performance

    Args:
        func: Function to track

    Returns:
        Wrapped function
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        error_occurred = False

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error_occurred = True
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Try to extract endpoint info from request
            # This is a simplified example
            endpoint = func.__name__
            method = "ASYNC"
            status_code = 500 if error_occurred else 200

            performance_monitor.record_request(
                endpoint=endpoint,
                method=method,
                duration_ms=duration_ms,
                status_code=status_code
            )

    return wrapper
