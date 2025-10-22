"""Performance analysis engine for detecting bottlenecks and optimization opportunities"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass

from app.services.metrics_service import metrics_service


@dataclass
class PerformanceBottleneck:
    """Represents a detected performance bottleneck"""
    state_id: str
    workflow_id: str
    metric_type: str  # duration, cost, tokens
    severity: str  # critical, high, medium, low
    average_value: float
    threshold: float
    impact_score: float
    recommendation: str
    supporting_data: Dict[str, Any]


@dataclass
class OptimizationOpportunity:
    """Represents an optimization opportunity"""
    opportunity_type: str  # prompt, model, routing, caching
    target: str  # state_id or workflow_id
    potential_improvement: float  # percentage
    estimated_savings: Dict[str, float]  # cost, time
    priority: str  # critical, high, medium, low
    action_items: List[str]
    confidence: float


class PerformanceAnalyzer:
    """Analyzes workflow execution metrics to detect bottlenecks and opportunities"""

    def __init__(self, metrics_dir: str = "logs/metrics"):
        """Initialize performance analyzer

        Args:
            metrics_dir: Directory containing metrics files
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_service = metrics_service

        # Performance thresholds
        self.thresholds = {
            "duration_seconds": {
                "critical": 10.0,
                "high": 5.0,
                "medium": 2.0,
                "low": 1.0
            },
            "cost": {
                "critical": 0.10,
                "high": 0.05,
                "medium": 0.02,
                "low": 0.01
            },
            "token_count": {
                "critical": 5000,
                "high": 3000,
                "medium": 1500,
                "low": 800
            }
        }

    def analyze_workflow_performance(
        self,
        workflow_id: str,
        time_window_hours: int = 24,
        min_executions: int = 5
    ) -> Dict[str, Any]:
        """Analyze performance for a specific workflow

        Args:
            workflow_id: Workflow identifier
            time_window_hours: Time window for analysis
            min_executions: Minimum executions needed for analysis

        Returns:
            Analysis results with bottlenecks and opportunities
        """
        # Load execution data
        executions = self._load_execution_data(workflow_id, time_window_hours)

        if len(executions) < min_executions:
            return {
                "workflow_id": workflow_id,
                "status": "insufficient_data",
                "message": f"Need at least {min_executions} executions, found {len(executions)}",
                "executions_analyzed": len(executions)
            }

        # Calculate aggregate statistics
        stats = self._calculate_statistics(executions)

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(workflow_id, executions, stats)

        # Identify optimization opportunities
        opportunities = self._identify_opportunities(workflow_id, executions, stats)

        # Calculate overall health score
        health_score = self._calculate_health_score(stats, bottlenecks)

        return {
            "workflow_id": workflow_id,
            "status": "analyzed",
            "time_window_hours": time_window_hours,
            "executions_analyzed": len(executions),
            "statistics": stats,
            "health_score": health_score,
            "bottlenecks": [self._bottleneck_to_dict(b) for b in bottlenecks],
            "opportunities": [self._opportunity_to_dict(o) for o in opportunities],
            "recommendations": self._generate_recommendations(bottlenecks, opportunities),
            "analyzed_at": datetime.utcnow().isoformat()
        }

    def _load_execution_data(
        self,
        workflow_id: str,
        time_window_hours: int
    ) -> List[Dict[str, Any]]:
        """Load execution data from metrics files

        Args:
            workflow_id: Workflow identifier
            time_window_hours: Time window in hours

        Returns:
            List of execution metrics
        """
        executions = []
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        if not self.metrics_dir.exists():
            return executions

        # Load from metrics files
        for metrics_file in self.metrics_dir.glob("*_metrics.json"):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)

                # Check if it's for the target workflow and within time window
                if data.get("workflow_id") == workflow_id:
                    timestamp = datetime.fromisoformat(data.get("timestamp", ""))
                    if timestamp >= cutoff_time:
                        executions.append(data)

            except Exception:
                continue

        return executions

    def _calculate_statistics(
        self,
        executions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate statistics from executions

        Args:
            executions: List of execution data

        Returns:
            Statistics dictionary
        """
        # Extract metrics
        durations = []
        costs = []
        tokens = []
        state_durations = defaultdict(list)
        state_costs = defaultdict(list)
        state_tokens = defaultdict(list)

        for exec_data in executions:
            metrics = exec_data.get("metrics", {})

            # Overall metrics
            if "total_duration_seconds" in metrics:
                durations.append(metrics["total_duration_seconds"])
            if "total_cost" in metrics:
                costs.append(metrics["total_cost"])
            if "total_tokens" in metrics:
                tokens.append(metrics["total_tokens"])

            # Per-state metrics
            state_metrics = metrics.get("state_metrics", {})
            for state_id, state_metric in state_metrics.items():
                if "duration_seconds" in state_metric:
                    state_durations[state_id].append(state_metric["duration_seconds"])
                if "cost" in state_metric:
                    state_costs[state_id].append(state_metric["cost"])
                if "token_count" in state_metric:
                    state_tokens[state_id].append(state_metric["token_count"])

        # Calculate statistics
        stats = {
            "overall": {
                "duration_seconds": self._calc_stats(durations),
                "cost": self._calc_stats(costs),
                "token_count": self._calc_stats(tokens),
            },
            "per_state": {}
        }

        # Per-state statistics
        all_states = set(state_durations.keys()) | set(state_costs.keys()) | set(state_tokens.keys())
        for state_id in all_states:
            stats["per_state"][state_id] = {
                "duration_seconds": self._calc_stats(state_durations.get(state_id, [])),
                "cost": self._calc_stats(state_costs.get(state_id, [])),
                "token_count": self._calc_stats(state_tokens.get(state_id, []))
            }

        return stats

    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values

        Args:
            values: List of numeric values

        Returns:
            Statistics dictionary
        """
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std_dev": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if count > 1 else 0.0,
            "p95": sorted_values[int(count * 0.95)] if count > 0 else 0.0,
            "p99": sorted_values[int(count * 0.99)] if count > 0 else 0.0
        }

    def _detect_bottlenecks(
        self,
        workflow_id: str,
        executions: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> List[PerformanceBottleneck]:
        """Detect performance bottlenecks

        Args:
            workflow_id: Workflow identifier
            executions: Execution data
            stats: Calculated statistics

        Returns:
            List of detected bottlenecks
        """
        bottlenecks = []

        # Check per-state metrics
        for state_id, state_stats in stats.get("per_state", {}).items():
            # Check duration
            duration_mean = state_stats["duration_seconds"]["mean"]
            severity, threshold = self._assess_severity("duration_seconds", duration_mean)

            if severity:
                impact = self._calculate_impact(
                    state_stats["duration_seconds"],
                    stats["overall"]["duration_seconds"]
                )

                bottlenecks.append(PerformanceBottleneck(
                    state_id=state_id,
                    workflow_id=workflow_id,
                    metric_type="duration",
                    severity=severity,
                    average_value=duration_mean,
                    threshold=threshold,
                    impact_score=impact,
                    recommendation=self._generate_bottleneck_recommendation(
                        state_id, "duration", duration_mean, threshold
                    ),
                    supporting_data=state_stats["duration_seconds"]
                ))

            # Check cost
            cost_mean = state_stats["cost"]["mean"]
            severity, threshold = self._assess_severity("cost", cost_mean)

            if severity:
                impact = self._calculate_impact(
                    state_stats["cost"],
                    stats["overall"]["cost"]
                )

                bottlenecks.append(PerformanceBottleneck(
                    state_id=state_id,
                    workflow_id=workflow_id,
                    metric_type="cost",
                    severity=severity,
                    average_value=cost_mean,
                    threshold=threshold,
                    impact_score=impact,
                    recommendation=self._generate_bottleneck_recommendation(
                        state_id, "cost", cost_mean, threshold
                    ),
                    supporting_data=state_stats["cost"]
                ))

            # Check tokens
            tokens_mean = state_stats["token_count"]["mean"]
            severity, threshold = self._assess_severity("token_count", tokens_mean)

            if severity:
                impact = self._calculate_impact(
                    state_stats["token_count"],
                    stats["overall"]["token_count"]
                )

                bottlenecks.append(PerformanceBottleneck(
                    state_id=state_id,
                    workflow_id=workflow_id,
                    metric_type="tokens",
                    severity=severity,
                    average_value=tokens_mean,
                    threshold=threshold,
                    impact_score=impact,
                    recommendation=self._generate_bottleneck_recommendation(
                        state_id, "tokens", tokens_mean, threshold
                    ),
                    supporting_data=state_stats["token_count"]
                ))

        # Sort by impact score (descending)
        bottlenecks.sort(key=lambda b: b.impact_score, reverse=True)

        return bottlenecks

    def _assess_severity(
        self,
        metric_type: str,
        value: float
    ) -> Tuple[Optional[str], float]:
        """Assess severity level for a metric value

        Args:
            metric_type: Type of metric
            value: Metric value

        Returns:
            Tuple of (severity level, threshold) or (None, 0.0)
        """
        thresholds = self.thresholds.get(metric_type, {})

        if value >= thresholds.get("critical", float('inf')):
            return ("critical", thresholds["critical"])
        elif value >= thresholds.get("high", float('inf')):
            return ("high", thresholds["high"])
        elif value >= thresholds.get("medium", float('inf')):
            return ("medium", thresholds["medium"])
        elif value >= thresholds.get("low", float('inf')):
            return ("low", thresholds["low"])

        return (None, 0.0)

    def _calculate_impact(
        self,
        state_metric: Dict[str, float],
        overall_metric: Dict[str, float]
    ) -> float:
        """Calculate impact score (0.0 to 1.0)

        Args:
            state_metric: State-level metric stats
            overall_metric: Overall metric stats

        Returns:
            Impact score
        """
        if overall_metric["mean"] == 0:
            return 0.0

        # Percentage of total
        contribution = state_metric["mean"] / overall_metric["mean"]

        # Variability (higher std_dev = more unpredictable)
        variability = state_metric["std_dev"] / state_metric["mean"] if state_metric["mean"] > 0 else 0.0

        # Combined impact
        impact = min(contribution * (1 + variability * 0.5), 1.0)

        return impact

    def _generate_bottleneck_recommendation(
        self,
        state_id: str,
        metric_type: str,
        value: float,
        threshold: float
    ) -> str:
        """Generate recommendation for bottleneck

        Args:
            state_id: State identifier
            metric_type: Metric type
            value: Current value
            threshold: Threshold value

        Returns:
            Recommendation string
        """
        excess = ((value - threshold) / threshold) * 100

        if metric_type == "duration":
            return f"State '{state_id}' execution time is {excess:.0f}% above threshold. Consider: (1) Optimizing prompt length, (2) Using faster model, (3) Reducing max_tokens."
        elif metric_type == "cost":
            return f"State '{state_id}' cost is {excess:.0f}% above threshold. Consider: (1) Using cheaper model (gpt-3.5 vs gpt-4), (2) Reducing prompt size, (3) Caching frequent queries."
        elif metric_type == "tokens":
            return f"State '{state_id}' token usage is {excess:.0f}% above threshold. Consider: (1) Shortening prompt, (2) Reducing context window, (3) Using more concise instructions."

        return f"State '{state_id}' {metric_type} exceeds threshold by {excess:.0f}%"

    def _identify_opportunities(
        self,
        workflow_id: str,
        executions: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> List[OptimizationOpportunity]:
        """Identify optimization opportunities

        Args:
            workflow_id: Workflow identifier
            executions: Execution data
            stats: Statistics

        Returns:
            List of optimization opportunities
        """
        opportunities = []

        # Opportunity 1: Model downgrade (GPT-4 -> GPT-3.5 for simple tasks)
        for state_id, state_stats in stats.get("per_state", {}).items():
            cost_mean = state_stats["cost"]["mean"]

            if cost_mean > 0.01:  # Only consider states with meaningful cost
                # Estimate savings from model downgrade (GPT-4 to 3.5 is ~10x cheaper)
                estimated_savings = cost_mean * 0.9  # 90% savings

                opportunities.append(OptimizationOpportunity(
                    opportunity_type="model_downgrade",
                    target=state_id,
                    potential_improvement=90.0,
                    estimated_savings={"cost": estimated_savings, "time": 0.0},
                    priority="high" if cost_mean > 0.05 else "medium",
                    action_items=[
                        f"Test {state_id} with gpt-3.5-turbo instead of gpt-4",
                        "Compare output quality between models",
                        "If quality is acceptable, update workflow definition"
                    ],
                    confidence=0.7
                ))

        # Opportunity 2: Prompt optimization
        for state_id, state_stats in stats.get("per_state", {}).items():
            tokens_mean = state_stats["token_count"]["mean"]

            if tokens_mean > 1500:  # High token usage
                # Estimate savings from prompt optimization (30% reduction)
                token_reduction = tokens_mean * 0.3
                cost_savings = state_stats["cost"]["mean"] * 0.3

                opportunities.append(OptimizationOpportunity(
                    opportunity_type="prompt_optimization",
                    target=state_id,
                    potential_improvement=30.0,
                    estimated_savings={"cost": cost_savings, "time": 0.2},
                    priority="medium",
                    action_items=[
                        f"Review prompt template for {state_id}",
                        "Remove redundant instructions",
                        "Use more concise language",
                        "Test optimized version"
                    ],
                    confidence=0.8
                ))

        # Opportunity 3: Caching
        # (Detect repeated queries by analyzing execution patterns)

        # Opportunity 4: Parallel execution
        # (Identify independent states that could run in parallel)

        return opportunities

    def _calculate_health_score(
        self,
        stats: Dict[str, Any],
        bottlenecks: List[PerformanceBottleneck]
    ) -> float:
        """Calculate overall health score (0.0 to 1.0)

        Args:
            stats: Statistics
            bottlenecks: Detected bottlenecks

        Returns:
            Health score
        """
        # Start at perfect score
        score = 1.0

        # Deduct for bottlenecks
        severity_weights = {
            "critical": 0.25,
            "high": 0.15,
            "medium": 0.10,
            "low": 0.05
        }

        for bottleneck in bottlenecks:
            deduction = severity_weights.get(bottleneck.severity, 0.0)
            deduction *= bottleneck.impact_score  # Weight by impact
            score -= deduction

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        bottlenecks: List[PerformanceBottleneck],
        opportunities: List[OptimizationOpportunity]
    ) -> List[str]:
        """Generate prioritized recommendations

        Args:
            bottlenecks: Detected bottlenecks
            opportunities: Optimization opportunities

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Top bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b.severity == "critical"]
        if critical_bottlenecks:
            recommendations.append(
                f"CRITICAL: Address {len(critical_bottlenecks)} critical bottlenecks immediately"
            )

        # High-priority opportunities
        high_priority_opps = [o for o in opportunities if o.priority == "high"]
        if high_priority_opps:
            total_savings = sum(o.estimated_savings.get("cost", 0.0) for o in high_priority_opps)
            recommendations.append(
                f"HIGH PRIORITY: {len(high_priority_opps)} optimization opportunities could save ${total_savings:.4f} per execution"
            )

        # Specific recommendations from bottlenecks
        for bottleneck in bottlenecks[:3]:  # Top 3
            recommendations.append(bottleneck.recommendation)

        return recommendations

    def _bottleneck_to_dict(self, bottleneck: PerformanceBottleneck) -> Dict[str, Any]:
        """Convert bottleneck to dictionary"""
        return {
            "state_id": bottleneck.state_id,
            "workflow_id": bottleneck.workflow_id,
            "metric_type": bottleneck.metric_type,
            "severity": bottleneck.severity,
            "average_value": bottleneck.average_value,
            "threshold": bottleneck.threshold,
            "impact_score": bottleneck.impact_score,
            "recommendation": bottleneck.recommendation,
            "supporting_data": bottleneck.supporting_data
        }

    def _opportunity_to_dict(self, opportunity: OptimizationOpportunity) -> Dict[str, Any]:
        """Convert opportunity to dictionary"""
        return {
            "opportunity_type": opportunity.opportunity_type,
            "target": opportunity.target,
            "potential_improvement": opportunity.potential_improvement,
            "estimated_savings": opportunity.estimated_savings,
            "priority": opportunity.priority,
            "action_items": opportunity.action_items,
            "confidence": opportunity.confidence
        }


# Global instance
performance_analyzer = PerformanceAnalyzer()
