"""Adaptive workflow routing based on performance and context"""
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from app.services.performance_analyzer import performance_analyzer


class RoutingStrategy(str, Enum):
    """Routing strategy types"""
    PERFORMANCE = "performance"  # Optimize for speed
    COST = "cost"  # Optimize for cost
    QUALITY = "quality"  # Optimize for output quality
    BALANCED = "balanced"  # Balance all factors


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    selected_state: str
    selected_model: str
    selected_temperature: float
    strategy_used: RoutingStrategy
    confidence: float
    reasoning: str
    alternatives_considered: List[Dict[str, Any]]
    estimated_metrics: Dict[str, float]


@dataclass
class StateVariant:
    """Represents a variant of a state with different parameters"""
    variant_id: str
    model: str
    temperature: float
    max_tokens: int
    estimated_duration: float
    estimated_cost: float
    estimated_quality: float
    success_rate: float


class AdaptiveRouter:
    """Routes workflow execution adaptively based on performance data and objectives"""

    def __init__(self):
        """Initialize adaptive router"""
        self.performance_analyzer = performance_analyzer

        # Model performance characteristics (based on general knowledge)
        self.model_profiles = {
            "gpt-4": {
                "quality_score": 1.0,
                "cost_multiplier": 1.0,
                "speed_multiplier": 1.0,
                "use_cases": ["complex_reasoning", "high_stakes", "creative"]
            },
            "gpt-4-turbo": {
                "quality_score": 0.95,
                "cost_multiplier": 0.5,
                "speed_multiplier": 1.5,
                "use_cases": ["complex_reasoning", "fast_response"]
            },
            "gpt-3.5-turbo": {
                "quality_score": 0.75,
                "cost_multiplier": 0.1,
                "speed_multiplier": 2.0,
                "use_cases": ["simple_tasks", "high_volume", "cost_sensitive"]
            },
            "gpt-3.5-turbo-16k": {
                "quality_score": 0.75,
                "cost_multiplier": 0.15,
                "speed_multiplier": 1.8,
                "use_cases": ["long_context", "cost_sensitive"]
            }
        }

        # Routing policies
        self.routing_policies = {
            RoutingStrategy.PERFORMANCE: self._route_for_performance,
            RoutingStrategy.COST: self._route_for_cost,
            RoutingStrategy.QUALITY: self._route_for_quality,
            RoutingStrategy.BALANCED: self._route_balanced
        }

    def route_execution(
        self,
        workflow_id: str,
        current_state_id: str,
        context: Dict[str, Any],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED
    ) -> RoutingDecision:
        """Make adaptive routing decision for workflow execution

        Args:
            workflow_id: Workflow identifier
            current_state_id: Current state ID
            context: Execution context
            strategy: Routing strategy

        Returns:
            Routing decision
        """
        # Get state variants (different model/parameter combinations)
        variants = self._generate_state_variants(current_state_id, context)

        # Get historical performance data
        perf_data = self._get_performance_data(workflow_id, current_state_id)

        # Apply routing strategy
        routing_func = self.routing_policies.get(strategy, self._route_balanced)
        decision = routing_func(variants, perf_data, context)

        return decision

    def _generate_state_variants(
        self,
        state_id: str,
        context: Dict[str, Any]
    ) -> List[StateVariant]:
        """Generate variants of a state with different parameters

        Args:
            state_id: State identifier
            context: Execution context

        Returns:
            List of state variants
        """
        variants = []

        # Get base state configuration (would come from workflow definition)
        base_config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Generate variants for different models
        for model_name, model_profile in self.model_profiles.items():
            # Estimate metrics based on model profile
            base_duration = 2.0  # Base duration in seconds
            base_cost = 0.02  # Base cost

            variant = StateVariant(
                variant_id=f"{state_id}_{model_name}",
                model=model_name,
                temperature=base_config["temperature"],
                max_tokens=base_config["max_tokens"],
                estimated_duration=base_duration / model_profile["speed_multiplier"],
                estimated_cost=base_cost * model_profile["cost_multiplier"],
                estimated_quality=model_profile["quality_score"],
                success_rate=0.95  # Would be calculated from historical data
            )

            variants.append(variant)

        # Generate variants for different temperatures (for same model)
        for temp in [0.1, 0.3, 0.7, 1.0]:
            variant = StateVariant(
                variant_id=f"{state_id}_gpt4_temp{temp}",
                model="gpt-4",
                temperature=temp,
                max_tokens=base_config["max_tokens"],
                estimated_duration=2.0,
                estimated_cost=0.02,
                estimated_quality=1.0 if temp < 0.5 else 0.95,  # Lower temp = more deterministic
                success_rate=0.95
            )
            variants.append(variant)

        return variants

    def _get_performance_data(
        self,
        workflow_id: str,
        state_id: str
    ) -> Dict[str, Any]:
        """Get historical performance data for a state

        Args:
            workflow_id: Workflow identifier
            state_id: State identifier

        Returns:
            Performance data
        """
        # Would query actual performance data from metrics service
        # For now, return empty dict
        return {
            "average_duration": 2.0,
            "average_cost": 0.02,
            "success_rate": 0.95,
            "sample_size": 0
        }

    def _route_for_performance(
        self,
        variants: List[StateVariant],
        perf_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """Route for optimal performance (speed)

        Args:
            variants: State variants
            perf_data: Performance data
            context: Execution context

        Returns:
            Routing decision
        """
        # Sort by estimated duration (fastest first)
        sorted_variants = sorted(variants, key=lambda v: v.estimated_duration)

        best = sorted_variants[0]

        return RoutingDecision(
            selected_state=best.variant_id,
            selected_model=best.model,
            selected_temperature=best.temperature,
            strategy_used=RoutingStrategy.PERFORMANCE,
            confidence=0.85,
            reasoning=f"Selected {best.model} for fastest execution (~{best.estimated_duration:.2f}s)",
            alternatives_considered=[
                {
                    "variant": v.variant_id,
                    "duration": v.estimated_duration,
                    "cost": v.estimated_cost
                }
                for v in sorted_variants[1:4]
            ],
            estimated_metrics={
                "duration": best.estimated_duration,
                "cost": best.estimated_cost,
                "quality": best.estimated_quality
            }
        )

    def _route_for_cost(
        self,
        variants: List[StateVariant],
        perf_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """Route for minimal cost

        Args:
            variants: State variants
            perf_data: Performance data
            context: Execution context

        Returns:
            Routing decision
        """
        # Sort by estimated cost (cheapest first)
        sorted_variants = sorted(variants, key=lambda v: v.estimated_cost)

        best = sorted_variants[0]

        # Ensure quality is acceptable (>0.6)
        if best.estimated_quality < 0.6:
            # Find cheapest variant with acceptable quality
            acceptable = [v for v in sorted_variants if v.estimated_quality >= 0.6]
            if acceptable:
                best = acceptable[0]

        return RoutingDecision(
            selected_state=best.variant_id,
            selected_model=best.model,
            selected_temperature=best.temperature,
            strategy_used=RoutingStrategy.COST,
            confidence=0.9,
            reasoning=f"Selected {best.model} for lowest cost (${best.estimated_cost:.4f})",
            alternatives_considered=[
                {
                    "variant": v.variant_id,
                    "cost": v.estimated_cost,
                    "quality": v.estimated_quality
                }
                for v in sorted_variants[1:4]
            ],
            estimated_metrics={
                "duration": best.estimated_duration,
                "cost": best.estimated_cost,
                "quality": best.estimated_quality
            }
        )

    def _route_for_quality(
        self,
        variants: List[StateVariant],
        perf_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """Route for highest quality output

        Args:
            variants: State variants
            perf_data: Performance data
            context: Execution context

        Returns:
            Routing decision
        """
        # Sort by estimated quality (highest first)
        sorted_variants = sorted(variants, key=lambda v: v.estimated_quality, reverse=True)

        best = sorted_variants[0]

        return RoutingDecision(
            selected_state=best.variant_id,
            selected_model=best.model,
            selected_temperature=best.temperature,
            strategy_used=RoutingStrategy.QUALITY,
            confidence=0.95,
            reasoning=f"Selected {best.model} (temp={best.temperature}) for highest quality output",
            alternatives_considered=[
                {
                    "variant": v.variant_id,
                    "quality": v.estimated_quality,
                    "cost": v.estimated_cost
                }
                for v in sorted_variants[1:4]
            ],
            estimated_metrics={
                "duration": best.estimated_duration,
                "cost": best.estimated_cost,
                "quality": best.estimated_quality
            }
        )

    def _route_balanced(
        self,
        variants: List[StateVariant],
        perf_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """Route for balanced performance/cost/quality

        Args:
            variants: State variants
            perf_data: Performance data
            context: Execution context

        Returns:
            Routing decision
        """
        # Calculate composite score for each variant
        scores = []

        for variant in variants:
            # Normalize metrics (0 to 1, higher is better)
            # For duration and cost, lower is better, so invert

            max_duration = max(v.estimated_duration for v in variants)
            max_cost = max(v.estimated_cost for v in variants)

            duration_score = 1.0 - (variant.estimated_duration / max_duration) if max_duration > 0 else 1.0
            cost_score = 1.0 - (variant.estimated_cost / max_cost) if max_cost > 0 else 1.0
            quality_score = variant.estimated_quality

            # Weighted composite score
            weights = {
                "duration": 0.3,
                "cost": 0.3,
                "quality": 0.4
            }

            composite = (
                duration_score * weights["duration"] +
                cost_score * weights["cost"] +
                quality_score * weights["quality"]
            )

            scores.append((composite, variant))

        # Sort by composite score
        scores.sort(key=lambda x: x[0], reverse=True)

        best_score, best = scores[0]

        return RoutingDecision(
            selected_state=best.variant_id,
            selected_model=best.model,
            selected_temperature=best.temperature,
            strategy_used=RoutingStrategy.BALANCED,
            confidence=0.8,
            reasoning=f"Selected {best.model} for optimal balance (score: {best_score:.2f})",
            alternatives_considered=[
                {
                    "variant": v.variant_id,
                    "score": s,
                    "duration": v.estimated_duration,
                    "cost": v.estimated_cost,
                    "quality": v.estimated_quality
                }
                for s, v in scores[1:4]
            ],
            estimated_metrics={
                "duration": best.estimated_duration,
                "cost": best.estimated_cost,
                "quality": best.estimated_quality,
                "composite_score": best_score
            }
        )

    def recommend_workflow_optimization(
        self,
        workflow_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Recommend workflow-level optimizations based on routing analysis

        Args:
            workflow_id: Workflow identifier
            time_window_hours: Time window for analysis

        Returns:
            Optimization recommendations
        """
        # Analyze performance
        perf_analysis = self.performance_analyzer.analyze_workflow_performance(
            workflow_id=workflow_id,
            time_window_hours=time_window_hours
        )

        if perf_analysis.get("status") != "analyzed":
            return perf_analysis

        recommendations = []

        # For each state, recommend optimal routing strategy
        for state_id, state_stats in perf_analysis["statistics"]["per_state"].items():
            # If cost is high, recommend cost-optimized routing
            if state_stats["cost"]["mean"] > 0.05:
                recommendations.append({
                    "state_id": state_id,
                    "current_avg_cost": state_stats["cost"]["mean"],
                    "recommended_strategy": "COST",
                    "estimated_savings": state_stats["cost"]["mean"] * 0.6,  # 60% savings
                    "action": f"Switch {state_id} to cost-optimized routing (use gpt-3.5-turbo)"
                })

            # If duration is high, recommend performance-optimized routing
            if state_stats["duration_seconds"]["mean"] > 3.0:
                recommendations.append({
                    "state_id": state_id,
                    "current_avg_duration": state_stats["duration_seconds"]["mean"],
                    "recommended_strategy": "PERFORMANCE",
                    "estimated_improvement": "40% faster",
                    "action": f"Switch {state_id} to performance-optimized routing"
                })

        return {
            "workflow_id": workflow_id,
            "analysis": perf_analysis,
            "routing_recommendations": recommendations,
            "overall_recommendation": self._generate_overall_recommendation(recommendations),
            "analyzed_at": datetime.utcnow().isoformat()
        }

    def _generate_overall_recommendation(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> str:
        """Generate overall recommendation summary

        Args:
            recommendations: List of recommendations

        Returns:
            Summary string
        """
        if not recommendations:
            return "No routing optimizations needed. Workflow is performing well."

        cost_recs = [r for r in recommendations if r.get("recommended_strategy") == "COST"]
        perf_recs = [r for r in recommendations if r.get("recommended_strategy") == "PERFORMANCE"]

        parts = []

        if cost_recs:
            total_savings = sum(r.get("estimated_savings", 0) for r in cost_recs)
            parts.append(f"Optimize {len(cost_recs)} states for cost (save ${total_savings:.4f}/execution)")

        if perf_recs:
            parts.append(f"Optimize {len(perf_recs)} states for performance")

        return " | ".join(parts) if parts else "Apply recommended routing strategies"


# Global instance
adaptive_router = AdaptiveRouter()
