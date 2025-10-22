"""A/B testing framework for workflows, prompts, and models"""
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import statistics


class VariantStatus(str, Enum):
    """Status of a variant"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    ARCHIVED = "archived"


@dataclass
class Variant:
    """Represents a test variant"""
    variant_id: str
    variant_name: str
    configuration: Dict[str, Any]
    traffic_percentage: float  # 0-100
    status: VariantStatus
    created_at: datetime
    metrics: Dict[str, float] = None


@dataclass
class ABTest:
    """Represents an A/B test"""
    test_id: str
    test_name: str
    test_type: str  # workflow, prompt, model, routing
    description: str
    variants: List[Variant]
    control_variant_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str = "draft"  # draft, running, completed, cancelled
    min_sample_size: int = 100
    confidence_level: float = 0.95
    primary_metric: str = "success_rate"


@dataclass
class VariantResult:
    """Results for a variant"""
    variant_id: str
    variant_name: str
    sample_size: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class ABTestResult:
    """Complete A/B test results"""
    test_id: str
    test_name: str
    status: str
    primary_metric: str
    variant_results: List[VariantResult]
    winner: Optional[str]
    statistical_significance: bool
    p_value: float
    confidence_level: float
    recommendation: str


class ABTestingFramework:
    """Framework for running A/B tests on workflows, prompts, and models"""

    def __init__(self, tests_dir: str = "logs/ab_tests"):
        """Initialize A/B testing framework

        Args:
            tests_dir: Directory to store test definitions and results
        """
        self.tests_dir = Path(tests_dir)
        self.tests_dir.mkdir(parents=True, exist_ok=True)

        # Active tests
        self.active_tests: Dict[str, ABTest] = {}

        # Variant assignments (user_id -> variant_id)
        self.assignments: Dict[str, Dict[str, str]] = {}

        # Results cache
        self.results_cache: Dict[str, Dict[str, List[float]]] = {}

    def create_test(
        self,
        test_name: str,
        test_type: str,
        description: str,
        variants: List[Dict[str, Any]],
        control_variant_id: str,
        primary_metric: str = "success_rate",
        min_sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> str:
        """Create a new A/B test

        Args:
            test_name: Name of the test
            test_type: Type of test (workflow, prompt, model, routing)
            description: Test description
            variants: List of variant configurations
            control_variant_id: ID of control variant
            primary_metric: Primary metric to optimize
            min_sample_size: Minimum sample size per variant
            confidence_level: Statistical confidence level

        Returns:
            Test ID
        """
        test_id = self._generate_id(test_name)

        # Create variant objects
        variant_objects = []
        total_traffic = 0.0

        for var_config in variants:
            variant = Variant(
                variant_id=var_config.get("variant_id"),
                variant_name=var_config.get("variant_name"),
                configuration=var_config.get("configuration"),
                traffic_percentage=var_config.get("traffic_percentage", 50.0),
                status=VariantStatus.DRAFT,
                created_at=datetime.utcnow(),
                metrics={}
            )
            variant_objects.append(variant)
            total_traffic += variant.traffic_percentage

        # Validate traffic allocation
        if abs(total_traffic - 100.0) > 0.01:
            # Normalize traffic
            for variant in variant_objects:
                variant.traffic_percentage = (variant.traffic_percentage / total_traffic) * 100.0

        # Create test
        test = ABTest(
            test_id=test_id,
            test_name=test_name,
            test_type=test_type,
            description=description,
            variants=variant_objects,
            control_variant_id=control_variant_id,
            created_at=datetime.utcnow(),
            status="draft",
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            primary_metric=primary_metric
        )

        # Save test
        self._save_test(test)

        return test_id

    def start_test(self, test_id: str):
        """Start an A/B test

        Args:
            test_id: Test identifier
        """
        test = self._load_test(test_id)

        if test.status != "draft":
            raise ValueError(f"Test {test_id} cannot be started (status: {test.status})")

        # Activate variants
        for variant in test.variants:
            variant.status = VariantStatus.ACTIVE

        test.status = "running"
        test.started_at = datetime.utcnow()

        self.active_tests[test_id] = test
        self._save_test(test)

        print(f"Started A/B test: {test.test_name} ({test_id})")

    def assign_variant(
        self,
        test_id: str,
        user_id: str
    ) -> str:
        """Assign a user to a variant

        Args:
            test_id: Test identifier
            user_id: User identifier

        Returns:
            Assigned variant ID
        """
        # Check if user already assigned
        if test_id in self.assignments and user_id in self.assignments[test_id]:
            return self.assignments[test_id][user_id]

        test = self.active_tests.get(test_id)
        if not test or test.status != "running":
            # Return control variant if test not running
            test = self._load_test(test_id)
            return test.control_variant_id

        # Hash-based assignment for consistency
        hash_input = f"{test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0-100

        # Assign based on traffic percentages
        cumulative = 0.0
        for variant in test.variants:
            if variant.status != VariantStatus.ACTIVE:
                continue

            cumulative += variant.traffic_percentage
            if percentage < cumulative:
                # Assign this variant
                if test_id not in self.assignments:
                    self.assignments[test_id] = {}
                self.assignments[test_id][user_id] = variant.variant_id

                return variant.variant_id

        # Fallback to control
        return test.control_variant_id

    def record_result(
        self,
        test_id: str,
        variant_id: str,
        metric_name: str,
        metric_value: float
    ):
        """Record a metric result for a variant

        Args:
            test_id: Test identifier
            variant_id: Variant identifier
            metric_name: Metric name
            metric_value: Metric value
        """
        # Initialize cache
        if test_id not in self.results_cache:
            self.results_cache[test_id] = {}
        if variant_id not in self.results_cache[test_id]:
            self.results_cache[test_id][variant_id] = {}
        if metric_name not in self.results_cache[test_id][variant_id]:
            self.results_cache[test_id][variant_id][metric_name] = []

        # Add result
        self.results_cache[test_id][variant_id][metric_name].append(metric_value)

        # Periodically save to disk
        if len(self.results_cache[test_id][variant_id][metric_name]) % 10 == 0:
            self._save_results(test_id)

    def analyze_test(
        self,
        test_id: str
    ) -> ABTestResult:
        """Analyze A/B test results

        Args:
            test_id: Test identifier

        Returns:
            Test results with statistical analysis
        """
        test = self._load_test(test_id)

        if test_id not in self.results_cache:
            self._load_results(test_id)

        results_data = self.results_cache.get(test_id, {})

        # Calculate results for each variant
        variant_results = []
        for variant in test.variants:
            variant_data = results_data.get(variant.variant_id, {})
            primary_values = variant_data.get(test.primary_metric, [])

            if not primary_values:
                variant_results.append(VariantResult(
                    variant_id=variant.variant_id,
                    variant_name=variant.variant_name,
                    sample_size=0,
                    metrics={},
                    confidence_intervals={}
                ))
                continue

            # Calculate metrics
            metrics = {}
            confidence_intervals = {}

            for metric_name, values in variant_data.items():
                if values:
                    metrics[metric_name] = statistics.mean(values)

                    # Calculate confidence interval
                    if len(values) > 1:
                        ci = self._confidence_interval(values, test.confidence_level)
                        confidence_intervals[metric_name] = ci

            variant_results.append(VariantResult(
                variant_id=variant.variant_id,
                variant_name=variant.variant_name,
                sample_size=len(primary_values),
                metrics=metrics,
                confidence_intervals=confidence_intervals
            ))

        # Determine winner
        winner, is_significant, p_value = self._determine_winner(
            test,
            variant_results
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            test,
            variant_results,
            winner,
            is_significant
        )

        return ABTestResult(
            test_id=test_id,
            test_name=test.test_name,
            status=test.status,
            primary_metric=test.primary_metric,
            variant_results=variant_results,
            winner=winner,
            statistical_significance=is_significant,
            p_value=p_value,
            confidence_level=test.confidence_level,
            recommendation=recommendation
        )

    def stop_test(
        self,
        test_id: str,
        declare_winner: bool = True
    ):
        """Stop an A/B test

        Args:
            test_id: Test identifier
            declare_winner: Whether to declare a winner
        """
        test = self._load_test(test_id)

        if test.status != "running":
            raise ValueError(f"Test {test_id} is not running")

        # Analyze results
        results = self.analyze_test(test_id)

        # Update test status
        test.status = "completed"
        test.ended_at = datetime.utcnow()

        # Declare winner if requested
        if declare_winner and results.statistical_significance:
            for variant in test.variants:
                if variant.variant_id == results.winner:
                    variant.status = VariantStatus.WINNER
                else:
                    variant.status = VariantStatus.ARCHIVED

        # Save test
        self._save_test(test)

        # Remove from active tests
        if test_id in self.active_tests:
            del self.active_tests[test_id]

        print(f"Stopped A/B test: {test.test_name}")
        print(f"Winner: {results.winner if results.statistical_significance else 'No clear winner'}")

    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of a test

        Args:
            test_id: Test identifier

        Returns:
            Test status information
        """
        test = self._load_test(test_id)
        results = self.analyze_test(test_id)

        return {
            "test_id": test_id,
            "test_name": test.test_name,
            "status": test.status,
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "duration_hours": (
                (datetime.utcnow() - test.started_at).total_seconds() / 3600
                if test.started_at else 0
            ),
            "variants": [
                {
                    "variant_id": vr.variant_id,
                    "variant_name": vr.variant_name,
                    "sample_size": vr.sample_size,
                    "primary_metric_value": vr.metrics.get(test.primary_metric, 0),
                    "ready_for_analysis": vr.sample_size >= test.min_sample_size
                }
                for vr in results.variant_results
            ],
            "statistical_significance": results.statistical_significance,
            "winner": results.winner,
            "recommendation": results.recommendation
        }

    def _confidence_interval(
        self,
        values: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval

        Args:
            values: Sample values
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < 2:
            mean_val = values[0] if values else 0
            return (mean_val, mean_val)

        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        n = len(values)

        # Using t-distribution (simplified)
        # For large samples, t-value ~= 1.96 for 95% confidence
        t_value = 1.96  # Approximation

        margin = t_value * (std_dev / (n ** 0.5))

        return (mean_val - margin, mean_val + margin)

    def _determine_winner(
        self,
        test: ABTest,
        variant_results: List[VariantResult]
    ) -> Tuple[Optional[str], bool, float]:
        """Determine test winner

        Args:
            test: AB test
            variant_results: Variant results

        Returns:
            Tuple of (winner_id, is_significant, p_value)
        """
        # Find control and treatment variants
        control = None
        treatments = []

        for vr in variant_results:
            if vr.variant_id == test.control_variant_id:
                control = vr
            else:
                treatments.append(vr)

        if not control or not treatments:
            return (None, False, 1.0)

        # Check minimum sample size
        if control.sample_size < test.min_sample_size:
            return (None, False, 1.0)

        # Find best treatment
        best_treatment = None
        best_value = control.metrics.get(test.primary_metric, 0)

        for treatment in treatments:
            if treatment.sample_size < test.min_sample_size:
                continue

            value = treatment.metrics.get(test.primary_metric, 0)
            if value > best_value:
                best_value = value
                best_treatment = treatment

        if not best_treatment:
            return (control.variant_id, False, 1.0)

        # Simplified statistical test (would use proper t-test in production)
        # Check if confidence intervals don't overlap
        control_ci = control.confidence_intervals.get(test.primary_metric, (0, 0))
        treatment_ci = best_treatment.confidence_intervals.get(test.primary_metric, (0, 0))

        is_significant = control_ci[1] < treatment_ci[0]  # Control upper < Treatment lower
        p_value = 0.03 if is_significant else 0.15  # Simplified

        winner_id = best_treatment.variant_id if is_significant else control.variant_id

        return (winner_id, is_significant, p_value)

    def _generate_recommendation(
        self,
        test: ABTest,
        variant_results: List[VariantResult],
        winner: Optional[str],
        is_significant: bool
    ) -> str:
        """Generate recommendation based on results

        Args:
            test: AB test
            variant_results: Results
            winner: Winner ID
            is_significant: Statistical significance

        Returns:
            Recommendation string
        """
        if not winner:
            return "Insufficient data. Continue test to reach minimum sample size."

        if not is_significant:
            return "No statistically significant difference detected. Consider running test longer or trying different variants."

        winner_result = next((vr for vr in variant_results if vr.variant_id == winner), None)
        control_result = next((vr for vr in variant_results if vr.variant_id == test.control_variant_id), None)

        if winner_result and control_result:
            winner_value = winner_result.metrics.get(test.primary_metric, 0)
            control_value = control_result.metrics.get(test.primary_metric, 0)

            if winner == test.control_variant_id:
                return f"Control variant performs best. No changes recommended."
            else:
                improvement = ((winner_value - control_value) / control_value * 100) if control_value > 0 else 0
                return f"Deploy winner variant '{winner_result.variant_name}'. Expected improvement: {improvement:.1f}% in {test.primary_metric}."

        return "Deploy winner variant."

    def _generate_id(self, name: str) -> str:
        """Generate test ID from name"""
        timestamp = datetime.utcnow().isoformat()
        data = f"{name}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _save_test(self, test: ABTest):
        """Save test definition to disk"""
        test_file = self.tests_dir / f"{test.test_id}_test.json"

        test_dict = {
            "test_id": test.test_id,
            "test_name": test.test_name,
            "test_type": test.test_type,
            "description": test.description,
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "variant_name": v.variant_name,
                    "configuration": v.configuration,
                    "traffic_percentage": v.traffic_percentage,
                    "status": v.status.value,
                    "created_at": v.created_at.isoformat()
                }
                for v in test.variants
            ],
            "control_variant_id": test.control_variant_id,
            "created_at": test.created_at.isoformat(),
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "ended_at": test.ended_at.isoformat() if test.ended_at else None,
            "status": test.status,
            "min_sample_size": test.min_sample_size,
            "confidence_level": test.confidence_level,
            "primary_metric": test.primary_metric
        }

        with open(test_file, 'w') as f:
            json.dump(test_dict, f, indent=2)

    def _load_test(self, test_id: str) -> ABTest:
        """Load test definition from disk"""
        # Check active tests first
        if test_id in self.active_tests:
            return self.active_tests[test_id]

        test_file = self.tests_dir / f"{test_id}_test.json"

        if not test_file.exists():
            raise ValueError(f"Test {test_id} not found")

        with open(test_file, 'r') as f:
            data = json.load(f)

        variants = [
            Variant(
                variant_id=v["variant_id"],
                variant_name=v["variant_name"],
                configuration=v["configuration"],
                traffic_percentage=v["traffic_percentage"],
                status=VariantStatus(v["status"]),
                created_at=datetime.fromisoformat(v["created_at"])
            )
            for v in data["variants"]
        ]

        test = ABTest(
            test_id=data["test_id"],
            test_name=data["test_name"],
            test_type=data["test_type"],
            description=data["description"],
            variants=variants,
            control_variant_id=data["control_variant_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            status=data["status"],
            min_sample_size=data["min_sample_size"],
            confidence_level=data["confidence_level"],
            primary_metric=data["primary_metric"]
        )

        return test

    def _save_results(self, test_id: str):
        """Save test results to disk"""
        if test_id not in self.results_cache:
            return

        results_file = self.tests_dir / f"{test_id}_results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results_cache[test_id], f, indent=2)

    def _load_results(self, test_id: str):
        """Load test results from disk"""
        results_file = self.tests_dir / f"{test_id}_results.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results_cache[test_id] = json.load(f)


# Global instance
ab_testing = ABTestingFramework()
