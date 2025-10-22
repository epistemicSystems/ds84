"""Cost/Quality optimization balancer for workflow execution"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OptimizationObjective(str, Enum):
    """Optimization objective types"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCED = "balanced"
    COST_CONSTRAINED = "cost_constrained"  # Max quality within cost budget
    QUALITY_CONSTRAINED = "quality_constrained"  # Min cost meeting quality threshold


@dataclass
class CostQualityTradeoff:
    """Represents a cost/quality tradeoff point"""
    configuration: Dict[str, Any]
    estimated_cost: float
    estimated_quality: float
    estimated_duration: float
    efficiency_score: float  # Quality per dollar
    recommendation_reason: str


class CostQualityOptimizer:
    """Optimizes the trade off between cost and quality"""

    def __init__(self):
        """Initialize cost/quality optimizer"""
        # Model cost and quality profiles
        self.model_profiles = {
            "gpt-4": {
                "cost_per_1k_tokens": 0.03,
                "quality_score": 1.0,
                "speed_score": 0.7,
                "best_for": "complex reasoning, high-stakes decisions"
            },
            "gpt-4-turbo": {
                "cost_per_1k_tokens": 0.01,
                "quality_score": 0.95,
                "speed_score": 1.0,
                "best_for": "fast complex tasks"
            },
            "gpt-3.5-turbo": {
                "cost_per_1k_tokens": 0.001,
                "quality_score": 0.75,
                "speed_score": 1.2,
                "best_for": "simple tasks, high volume"
            },
            "gpt-3.5-turbo-16k": {
                "cost_per_1k_tokens": 0.003,
                "quality_score": 0.75,
                "speed_score": 1.1,
                "best_for": "long context, cost-sensitive"
            }
        }

    def optimize(
        self,
        objective: OptimizationObjective,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, float]] = None
    ) -> CostQualityTradeoff:
        """Find optimal cost/quality tradeoff

        Args:
            objective: Optimization objective
            context: Execution context (task complexity, importance, etc.)
            constraints: Optional constraints (max_cost, min_quality)

        Returns:
            Optimal tradeoff configuration
        """
        constraints = constraints or {}

        # Generate candidate configurations
        candidates = self._generate_candidates(context)

        # Filter by constraints
        filtered = self._apply_constraints(candidates, constraints)

        if not filtered:
            # No candidates meet constraints, relax constraints
            filtered = candidates

        # Select best based on objective
        if objective == OptimizationObjective.MINIMIZE_COST:
            return self._select_min_cost(filtered)
        elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
            return self._select_max_quality(filtered)
        elif objective == OptimizationObjective.BALANCED:
            return self._select_balanced(filtered)
        elif objective == OptimizationObjective.COST_CONSTRAINED:
            return self._select_cost_constrained(filtered, constraints)
        elif objective == OptimizationObjective.QUALITY_CONSTRAINED:
            return self._select_quality_constrained(filtered, constraints)

        return self._select_balanced(filtered)

    def _generate_candidates(
        self,
        context: Dict[str, Any]
    ) -> List[CostQualityTradeoff]:
        """Generate candidate configurations

        Args:
            context: Execution context

        Returns:
            List of candidate tradeoffs
        """
        candidates = []

        # Estimate token count from context
        estimated_tokens = context.get("estimated_tokens", 1000)
        task_complexity = context.get("task_complexity", "medium")  # low, medium, high

        for model_name, profile in self.model_profiles.items():
            # Calculate costs
            cost_per_execution = (estimated_tokens / 1000) * profile["cost_per_1k_tokens"]

            # Adjust quality based on task complexity
            quality_score = profile["quality_score"]
            if task_complexity == "low" and model_name == "gpt-3.5-turbo":
                quality_score = 0.9  # 3.5-turbo is good enough for simple tasks
            elif task_complexity == "high" and model_name.startswith("gpt-3.5"):
                quality_score *= 0.8  # Penalize 3.5 for complex tasks

            # Calculate efficiency (quality per dollar)
            efficiency = quality_score / cost_per_execution if cost_per_execution > 0 else 0

            # Estimate duration
            base_duration = 2.0  # seconds
            duration = base_duration / profile["speed_score"]

            candidates.append(CostQualityTradeoff(
                configuration={
                    "model": model_name,
                    "temperature": 0.7,
                    "max_tokens": min(estimated_tokens, 2000)
                },
                estimated_cost=cost_per_execution,
                estimated_quality=quality_score,
                estimated_duration=duration,
                efficiency_score=efficiency,
                recommendation_reason=f"{model_name} - {profile['best_for']}"
            ))

        return candidates

    def _apply_constraints(
        self,
        candidates: List[CostQualityTradeoff],
        constraints: Dict[str, float]
    ) -> List[CostQualityTradeoff]:
        """Filter candidates by constraints

        Args:
            candidates: Candidate configurations
            constraints: Constraints to apply

        Returns:
            Filtered candidates
        """
        filtered = []

        max_cost = constraints.get("max_cost")
        min_quality = constraints.get("min_quality")
        max_duration = constraints.get("max_duration")

        for candidate in candidates:
            # Check constraints
            if max_cost and candidate.estimated_cost > max_cost:
                continue
            if min_quality and candidate.estimated_quality < min_quality:
                continue
            if max_duration and candidate.estimated_duration > max_duration:
                continue

            filtered.append(candidate)

        return filtered

    def _select_min_cost(
        self,
        candidates: List[CostQualityTradeoff]
    ) -> CostQualityTradeoff:
        """Select minimum cost option

        Args:
            candidates: Candidate configurations

        Returns:
            Best candidate
        """
        return min(candidates, key=lambda c: c.estimated_cost)

    def _select_max_quality(
        self,
        candidates: List[CostQualityTradeoff]
    ) -> CostQualityTradeoff:
        """Select maximum quality option

        Args:
            candidates: Candidate configurations

        Returns:
            Best candidate
        """
        return max(candidates, key=lambda c: c.estimated_quality)

    def _select_balanced(
        self,
        candidates: List[CostQualityTradeoff]
    ) -> CostQualityTradeoff:
        """Select balanced option

        Args:
            candidates: Candidate configurations

        Returns:
            Best candidate
        """
        # Select highest efficiency (quality per dollar)
        return max(candidates, key=lambda c: c.efficiency_score)

    def _select_cost_constrained(
        self,
        candidates: List[CostQualityTradeoff],
        constraints: Dict[str, float]
    ) -> CostQualityTradeoff:
        """Select max quality within cost budget

        Args:
            candidates: Candidate configurations
            constraints: Constraints

        Returns:
            Best candidate
        """
        max_cost = constraints.get("max_cost", float('inf'))

        # Filter by cost constraint
        within_budget = [c for c in candidates if c.estimated_cost <= max_cost]

        if not within_budget:
            # Return cheapest option
            return min(candidates, key=lambda c: c.estimated_cost)

        # Return highest quality within budget
        return max(within_budget, key=lambda c: c.estimated_quality)

    def _select_quality_constrained(
        self,
        candidates: List[CostQualityTradeoff],
        constraints: Dict[str, float]
    ) -> CostQualityTradeoff:
        """Select min cost meeting quality threshold

        Args:
            candidates: Candidate configurations
            constraints: Constraints

        Returns:
            Best candidate
        """
        min_quality = constraints.get("min_quality", 0.0)

        # Filter by quality constraint
        meets_quality = [c for c in candidates if c.estimated_quality >= min_quality]

        if not meets_quality:
            # Return highest quality option
            return max(candidates, key=lambda c: c.estimated_quality)

        # Return cheapest option that meets quality
        return min(meets_quality, key=lambda c: c.estimated_cost)

    def analyze_tradeoff_curve(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the full cost/quality tradeoff curve

        Args:
            context: Execution context

        Returns:
            Tradeoff analysis
        """
        # Generate all candidates
        candidates = self._generate_candidates(context)

        # Sort by cost
        candidates.sort(key=lambda c: c.estimated_cost)

        # Identify pareto frontier (configurations where you can't improve
        # quality without increasing cost)
        pareto_frontier = []
        max_quality_seen = 0.0

        for candidate in candidates:
            if candidate.estimated_quality > max_quality_seen:
                pareto_frontier.append(candidate)
                max_quality_seen = candidate.estimated_quality

        return {
            "total_configurations": len(candidates),
            "pareto_optimal_configurations": len(pareto_frontier),
            "cost_range": {
                "min": min(c.estimated_cost for c in candidates),
                "max": max(c.estimated_cost for c in candidates)
            },
            "quality_range": {
                "min": min(c.estimated_quality for c in candidates),
                "max": max(c.estimated_quality for c in candidates)
            },
            "pareto_frontier": [
                {
                    "model": c.configuration["model"],
                    "cost": c.estimated_cost,
                    "quality": c.estimated_quality,
                    "efficiency": c.efficiency_score,
                    "reason": c.recommendation_reason
                }
                for c in pareto_frontier
            ],
            "recommendations": {
                "minimum_cost": self._select_min_cost(candidates).configuration,
                "maximum_quality": self._select_max_quality(candidates).configuration,
                "best_efficiency": self._select_balanced(candidates).configuration
            }
        }

    def recommend_for_use_case(
        self,
        use_case: str,
        context: Dict[str, Any] = None
    ) -> CostQualityTradeoff:
        """Recommend configuration for specific use case

        Args:
            use_case: Use case description
            context: Optional context

        Returns:
            Recommended configuration
        """
        context = context or {}

        # Map use cases to objectives and constraints
        use_case_lower = use_case.lower()

        if "production" in use_case_lower or "high stakes" in use_case_lower:
            # Production: prioritize quality
            return self.optimize(
                objective=OptimizationObjective.MAXIMIZE_QUALITY,
                context={"task_complexity": "high", **context},
                constraints={"min_quality": 0.9}
            )

        elif "testing" in use_case_lower or "development" in use_case_lower:
            # Testing: balance cost and quality
            return self.optimize(
                objective=OptimizationObjective.BALANCED,
                context={"task_complexity": "medium", **context},
                constraints={}
            )

        elif "bulk" in use_case_lower or "high volume" in use_case_lower:
            # Bulk processing: minimize cost
            return self.optimize(
                objective=OptimizationObjective.MINIMIZE_COST,
                context={"task_complexity": "low", **context},
                constraints={"min_quality": 0.6}
            )

        elif "research" in use_case_lower or "analysis" in use_case_lower:
            # Research: maximize quality
            return self.optimize(
                objective=OptimizationObjective.MAXIMIZE_QUALITY,
                context={"task_complexity": "high", **context},
                constraints={}
            )

        else:
            # Default: balanced
            return self.optimize(
                objective=OptimizationObjective.BALANCED,
                context={"task_complexity": "medium", **context},
                constraints={}
            )


# Global instance
cost_quality_optimizer = CostQualityOptimizer()
