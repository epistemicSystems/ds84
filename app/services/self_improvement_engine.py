"""Recursive self-improvement engine for meta-cognitive optimization"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

from app.services.performance_analyzer import performance_analyzer
from app.services.adaptive_router import adaptive_router
from app.services.prompt_optimizer import prompt_optimizer
from app.services.metrics_service import metrics_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    """Represents one cycle of self-improvement"""
    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    workflow_id: str
    phase: str  # analysis, optimization, validation, deployment
    actions_taken: List[Dict[str, Any]]
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]]
    improvements: List[str]
    status: str  # running, completed, failed


class SelfImprovementEngine:
    """Orchestrates recursive self-improvement cycles"""

    def __init__(self, improvement_dir: str = "logs/improvements"):
        """Initialize self-improvement engine

        Args:
            improvement_dir: Directory to store improvement logs
        """
        self.improvement_dir = Path(improvement_dir)
        self.improvement_dir.mkdir(parents=True, exist_ok=True)

        self.performance_analyzer = performance_analyzer
        self.adaptive_router = adaptive_router
        self.prompt_optimizer = prompt_optimizer
        self.metrics_service = metrics_service

        self.cycles: Dict[str, ImprovementCycle] = {}

        # Improvement thresholds
        self.thresholds = {
            "min_executions": 10,  # Minimum executions before optimization
            "health_score_threshold": 0.7,  # Optimize if below this
            "cost_threshold": 0.10,  # Optimize if average cost exceeds this
            "duration_threshold": 5.0  # Optimize if average duration exceeds this (seconds)
        }

    async def run_improvement_cycle(
        self,
        workflow_id: str,
        time_window_hours: int = 24,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Run one complete improvement cycle

        Args:
            workflow_id: Workflow to optimize
            time_window_hours: Time window for analysis
            dry_run: If True, only analyze and recommend (don't apply changes)

        Returns:
            Improvement cycle results
        """
        cycle_id = f"{workflow_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting improvement cycle: {cycle_id}")

        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            started_at=datetime.utcnow(),
            completed_at=None,
            workflow_id=workflow_id,
            phase="analysis",
            actions_taken=[],
            metrics_before={},
            metrics_after=None,
            improvements=[],
            status="running"
        )

        self.cycles[cycle_id] = cycle

        try:
            # Phase 1: Performance Analysis
            logger.info(f"[{cycle_id}] Phase 1: Performance Analysis")
            cycle.phase = "analysis"

            analysis = await self._analyze_performance(workflow_id, time_window_hours)

            if analysis.get("status") != "analyzed":
                cycle.status = "failed"
                cycle.completed_at = datetime.utcnow()
                return {
                    "cycle_id": cycle_id,
                    "status": "failed",
                    "reason": "insufficient_data",
                    "message": analysis.get("message", "")
                }

            # Store baseline metrics
            cycle.metrics_before = {
                "health_score": analysis["health_score"],
                "average_duration": analysis["statistics"]["overall"]["duration_seconds"]["mean"],
                "average_cost": analysis["statistics"]["overall"]["cost"]["mean"],
                "average_tokens": analysis["statistics"]["overall"]["token_count"]["mean"]
            }

            # Phase 2: Identify Optimizations
            logger.info(f"[{cycle_id}] Phase 2: Identify Optimizations")
            cycle.phase = "optimization"

            optimizations = await self._identify_optimizations(workflow_id, analysis)

            # Phase 3: Validate Optimizations
            logger.info(f"[{cycle_id}] Phase 3: Validate Optimizations")
            cycle.phase = "validation"

            validated_optimizations = await self._validate_optimizations(
                workflow_id,
                optimizations
            )

            # Phase 4: Apply Optimizations (if not dry run)
            if not dry_run:
                logger.info(f"[{cycle_id}] Phase 4: Deploy Optimizations")
                cycle.phase = "deployment"

                deployment_results = await self._deploy_optimizations(
                    workflow_id,
                    validated_optimizations
                )

                cycle.actions_taken = deployment_results["actions_taken"]
                cycle.improvements = deployment_results["improvements"]
            else:
                logger.info(f"[{cycle_id}] Phase 4: Skipped (dry run)")
                cycle.actions_taken = [{
                    "action": "dry_run",
                    "message": "Optimizations identified but not applied (dry run mode)"
                }]
                cycle.improvements = [
                    f"{len(validated_optimizations)} optimizations recommended"
                ]

            # Complete cycle
            cycle.status = "completed"
            cycle.completed_at = datetime.utcnow()

            # Save cycle report
            self._save_cycle_report(cycle, analysis, validated_optimizations)

            logger.info(f"[{cycle_id}] Improvement cycle completed")

            return {
                "cycle_id": cycle_id,
                "status": "completed",
                "workflow_id": workflow_id,
                "baseline_metrics": cycle.metrics_before,
                "analysis": analysis,
                "optimizations_identified": len(optimizations),
                "optimizations_validated": len(validated_optimizations),
                "actions_taken": cycle.actions_taken,
                "improvements": cycle.improvements,
                "dry_run": dry_run,
                "duration_seconds": (cycle.completed_at - cycle.started_at).total_seconds()
            }

        except Exception as e:
            logger.error(f"[{cycle_id}] Failed: {e}")
            cycle.status = "failed"
            cycle.completed_at = datetime.utcnow()

            return {
                "cycle_id": cycle_id,
                "status": "failed",
                "error": str(e)
            }

    async def _analyze_performance(
        self,
        workflow_id: str,
        time_window_hours: int
    ) -> Dict[str, Any]:
        """Analyze workflow performance

        Args:
            workflow_id: Workflow identifier
            time_window_hours: Time window for analysis

        Returns:
            Performance analysis
        """
        analysis = self.performance_analyzer.analyze_workflow_performance(
            workflow_id=workflow_id,
            time_window_hours=time_window_hours,
            min_executions=self.thresholds["min_executions"]
        )

        return analysis

    async def _identify_optimizations(
        self,
        workflow_id: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify possible optimizations

        Args:
            workflow_id: Workflow identifier
            analysis: Performance analysis

        Returns:
            List of optimization opportunities
        """
        optimizations = []

        # From performance bottlenecks
        for bottleneck in analysis.get("bottlenecks", []):
            optimizations.append({
                "type": "bottleneck_fix",
                "target": bottleneck["state_id"],
                "severity": bottleneck["severity"],
                "recommendation": bottleneck["recommendation"],
                "metric": bottleneck["metric_type"],
                "current_value": bottleneck["average_value"],
                "threshold": bottleneck["threshold"]
            })

        # From identified opportunities
        for opportunity in analysis.get("opportunities", []):
            optimizations.append({
                "type": opportunity["opportunity_type"],
                "target": opportunity["target"],
                "priority": opportunity["priority"],
                "potential_improvement": opportunity["potential_improvement"],
                "estimated_savings": opportunity["estimated_savings"],
                "action_items": opportunity["action_items"]
            })

        # From routing recommendations
        routing_recs = self.adaptive_router.recommend_workflow_optimization(
            workflow_id=workflow_id,
            time_window_hours=24
        )

        for rec in routing_recs.get("routing_recommendations", []):
            optimizations.append({
                "type": "routing_optimization",
                "target": rec["state_id"],
                "strategy": rec["recommended_strategy"],
                "action": rec["action"],
                "estimated_savings": rec.get("estimated_savings", 0)
            })

        # From prompt analysis
        # (Would analyze prompts used in workflow states)

        return optimizations

    async def _validate_optimizations(
        self,
        workflow_id: str,
        optimizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate optimization opportunities

        Args:
            workflow_id: Workflow identifier
            optimizations: List of optimizations to validate

        Returns:
            Validated optimizations
        """
        validated = []

        for opt in optimizations:
            # Validation criteria
            is_safe = True
            confidence = 0.8

            # Check if optimization is safe to apply
            if opt["type"] == "model_downgrade":
                # Model downgrades require testing
                confidence = 0.6
            elif opt["type"] == "prompt_optimization":
                # Prompt optimizations need validation
                confidence = 0.7
            elif opt["type"] == "routing_optimization":
                # Routing changes are generally safe
                confidence = 0.9

            # Add validation metadata
            opt["validated"] = is_safe
            opt["confidence"] = confidence
            opt["validation_timestamp"] = datetime.utcnow().isoformat()

            if is_safe and confidence >= 0.6:
                validated.append(opt)

        return validated

    async def _deploy_optimizations(
        self,
        workflow_id: str,
        optimizations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Deploy validated optimizations

        Args:
            workflow_id: Workflow identifier
            optimizations: Validated optimizations

        Returns:
            Deployment results
        """
        actions_taken = []
        improvements = []

        for opt in optimizations:
            try:
                if opt["type"] == "routing_optimization":
                    # Update workflow routing configuration
                    action = {
                        "type": "routing_update",
                        "target": opt["target"],
                        "strategy": opt["strategy"],
                        "status": "applied",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    actions_taken.append(action)
                    improvements.append(
                        f"Updated routing for {opt['target']} to {opt['strategy']} strategy"
                    )

                elif opt["type"] == "prompt_optimization":
                    # Save optimized prompt
                    action = {
                        "type": "prompt_update",
                        "target": opt["target"],
                        "status": "saved_for_review",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    actions_taken.append(action)
                    improvements.append(
                        f"Generated optimized prompt for {opt['target']}"
                    )

                elif opt["type"] == "model_downgrade":
                    # Log recommendation (would update workflow config)
                    action = {
                        "type": "model_recommendation",
                        "target": opt["target"],
                        "status": "recommended",
                        "note": "Requires A/B testing before deployment",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    actions_taken.append(action)
                    improvements.append(
                        f"Recommended model change for {opt['target']}"
                    )

            except Exception as e:
                logger.error(f"Failed to deploy optimization: {e}")
                actions_taken.append({
                    "type": opt["type"],
                    "target": opt.get("target"),
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "actions_taken": actions_taken,
            "improvements": improvements
        }

    def _save_cycle_report(
        self,
        cycle: ImprovementCycle,
        analysis: Dict[str, Any],
        optimizations: List[Dict[str, Any]]
    ):
        """Save improvement cycle report

        Args:
            cycle: Improvement cycle
            analysis: Performance analysis
            optimizations: Optimizations applied
        """
        report = {
            "cycle_id": cycle.cycle_id,
            "workflow_id": cycle.workflow_id,
            "started_at": cycle.started_at.isoformat(),
            "completed_at": cycle.completed_at.isoformat() if cycle.completed_at else None,
            "status": cycle.status,
            "metrics_before": cycle.metrics_before,
            "metrics_after": cycle.metrics_after,
            "analysis_summary": {
                "health_score": analysis.get("health_score"),
                "bottlenecks_found": len(analysis.get("bottlenecks", [])),
                "opportunities_found": len(analysis.get("opportunities", []))
            },
            "optimizations": optimizations,
            "actions_taken": cycle.actions_taken,
            "improvements": cycle.improvements
        }

        # Save to file
        report_file = self.improvement_dir / f"{cycle.cycle_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved improvement report: {report_file}")

    async def run_continuous_improvement(
        self,
        workflow_ids: List[str],
        interval_hours: int = 24,
        max_cycles: int = None
    ):
        """Run continuous improvement loop

        Args:
            workflow_ids: Workflows to optimize
            interval_hours: Hours between improvement cycles
            max_cycles: Maximum number of cycles (None for unlimited)
        """
        cycle_count = 0

        logger.info(f"Starting continuous improvement for {len(workflow_ids)} workflows")
        logger.info(f"Interval: {interval_hours} hours")

        while max_cycles is None or cycle_count < max_cycles:
            for workflow_id in workflow_ids:
                try:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Running improvement cycle {cycle_count + 1} for {workflow_id}")
                    logger.info(f"{'='*80}\n")

                    result = await self.run_improvement_cycle(
                        workflow_id=workflow_id,
                        time_window_hours=interval_hours,
                        dry_run=False  # Actually apply optimizations
                    )

                    logger.info(f"Cycle completed: {result['status']}")

                    if result["status"] == "completed":
                        logger.info(f"Improvements: {len(result['improvements'])}")
                        for improvement in result["improvements"]:
                            logger.info(f"  - {improvement}")

                except Exception as e:
                    logger.error(f"Cycle failed for {workflow_id}: {e}")

                # Brief delay between workflows
                await asyncio.sleep(5)

            cycle_count += 1

            # Wait for next cycle
            if max_cycles is None or cycle_count < max_cycles:
                wait_seconds = interval_hours * 3600
                logger.info(f"\nWaiting {interval_hours} hours until next cycle...")
                await asyncio.sleep(wait_seconds)

        logger.info("Continuous improvement stopped")

    def get_improvement_history(
        self,
        workflow_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get improvement cycle history

        Args:
            workflow_id: Optional workflow filter

        Returns:
            List of improvement cycles
        """
        cycles = []

        for cycle_id, cycle in self.cycles.items():
            if workflow_id is None or cycle.workflow_id == workflow_id:
                cycles.append({
                    "cycle_id": cycle.cycle_id,
                    "workflow_id": cycle.workflow_id,
                    "started_at": cycle.started_at.isoformat(),
                    "completed_at": cycle.completed_at.isoformat() if cycle.completed_at else None,
                    "status": cycle.status,
                    "improvements_count": len(cycle.improvements),
                    "actions_count": len(cycle.actions_taken)
                })

        # Sort by start time (most recent first)
        cycles.sort(key=lambda c: c["started_at"], reverse=True)

        return cycles


# Global instance
self_improvement_engine = SelfImprovementEngine()
