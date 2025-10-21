"""Metrics collection and tracking service for workflow executions"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from app.models.workflow_models import WorkflowExecutionState, StateExecutionMetrics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsService:
    """Service for collecting, tracking, and analyzing workflow metrics"""

    def __init__(self, metrics_dir: str = "logs/metrics"):
        """Initialize metrics service

        Args:
            metrics_dir: Directory to store metrics files
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of recent executions
        self.execution_cache: Dict[str, WorkflowExecutionState] = {}
        self.max_cache_size = 1000

        logger.info(f"MetricsService initialized with directory: {metrics_dir}")

    def log_workflow_start(
        self,
        execution_id: str,
        workflow_id: str,
        input_data: Dict[str, Any]
    ):
        """Log workflow execution start

        Args:
            execution_id: Unique execution identifier
            workflow_id: Workflow identifier
            input_data: Input data for workflow
        """
        logger.info(
            f"Workflow execution started: "
            f"execution_id={execution_id}, "
            f"workflow_id={workflow_id}"
        )

        # Log to file
        self._write_event_log({
            "event": "workflow_start",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "input_size": len(str(input_data))
        })

    def log_workflow_complete(
        self,
        execution_id: str,
        workflow_id: str,
        status: str,
        metrics: Dict[str, Any]
    ):
        """Log workflow execution completion

        Args:
            execution_id: Unique execution identifier
            workflow_id: Workflow identifier
            status: Execution status (completed, failed, timeout)
            metrics: Execution metrics
        """
        logger.info(
            f"Workflow execution {status}: "
            f"execution_id={execution_id}, "
            f"duration={metrics.get('total_duration_seconds', 0):.2f}s, "
            f"cost=${metrics.get('total_cost', 0):.4f}"
        )

        # Log to file
        self._write_event_log({
            "event": "workflow_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": status,
            "metrics": metrics
        })

        # Write detailed metrics file
        self._write_metrics_file(execution_id, workflow_id, metrics)

    def log_state_execution(
        self,
        execution_id: str,
        state_id: str,
        metrics: StateExecutionMetrics
    ):
        """Log individual state execution

        Args:
            execution_id: Workflow execution identifier
            state_id: State identifier
            metrics: State execution metrics
        """
        logger.debug(
            f"State executed: "
            f"execution_id={execution_id}, "
            f"state={state_id}, "
            f"duration={metrics.duration_seconds:.2f}s, "
            f"tokens={metrics.token_count}"
        )

        # Log to file
        self._write_event_log({
            "event": "state_execution",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": execution_id,
            "state_id": state_id,
            "duration_seconds": metrics.duration_seconds,
            "token_count": metrics.token_count,
            "cost": metrics.cost,
            "success": metrics.success,
            "error": metrics.error_message
        })

    def log_state_transition(
        self,
        execution_id: str,
        from_state: str,
        to_state: Optional[str],
        condition_met: Optional[bool] = None
    ):
        """Log state transition

        Args:
            execution_id: Workflow execution identifier
            from_state: Source state ID
            to_state: Target state ID (None for terminal)
            condition_met: Whether transition condition was met
        """
        logger.debug(
            f"State transition: "
            f"execution_id={execution_id}, "
            f"{from_state} -> {to_state or 'END'}"
        )

        self._write_event_log({
            "event": "state_transition",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": execution_id,
            "from_state": from_state,
            "to_state": to_state,
            "condition_met": condition_met
        })

    def log_error(
        self,
        execution_id: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None
    ):
        """Log workflow error

        Args:
            execution_id: Workflow execution identifier
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        logger.error(
            f"Workflow error: "
            f"execution_id={execution_id}, "
            f"type={error_type}, "
            f"message={error_message}"
        )

        self._write_event_log({
            "event": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": execution_id,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        })

    def get_execution_metrics(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metrics for a specific execution

        Args:
            execution_id: Execution identifier

        Returns:
            Metrics dictionary or None if not found
        """
        # Check cache
        if execution_id in self.execution_cache:
            execution = self.execution_cache[execution_id]
            return execution.metrics.dict() if execution.metrics else None

        # Check file system
        metrics_file = self.metrics_dir / f"{execution_id}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)

        return None

    def get_workflow_statistics(self, workflow_id: str) -> Dict[str, Any]:
        """Get aggregate statistics for a workflow

        Args:
            workflow_id: Workflow identifier

        Returns:
            Statistics dictionary
        """
        # In production, this would query a metrics database
        # For now, return placeholder
        return {
            "workflow_id": workflow_id,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration_seconds": 0.0,
            "average_cost": 0.0,
            "average_tokens": 0,
            "note": "Aggregate statistics not yet implemented"
        }

    def _write_event_log(self, event: Dict[str, Any]):
        """Write event to daily log file

        Args:
            event: Event dictionary
        """
        try:
            # Daily log file
            today = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = self.metrics_dir / f"events_{today}.jsonl"

            # Append event as JSON line
            with open(log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')

        except Exception as e:
            logger.error(f"Failed to write event log: {e}")

    def _write_metrics_file(
        self,
        execution_id: str,
        workflow_id: str,
        metrics: Dict[str, Any]
    ):
        """Write detailed metrics file for execution

        Args:
            execution_id: Execution identifier
            workflow_id: Workflow identifier
            metrics: Metrics dictionary
        """
        try:
            metrics_file = self.metrics_dir / f"{execution_id}_metrics.json"

            metrics_data = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics
            }

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to write metrics file: {e}")

    def analyze_performance_trends(
        self,
        workflow_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze performance trends for a workflow

        Args:
            workflow_id: Workflow identifier
            time_window_hours: Time window for analysis

        Returns:
            Analysis results
        """
        # In production, this would analyze historical data
        return {
            "workflow_id": workflow_id,
            "time_window_hours": time_window_hours,
            "trends": {
                "duration": "stable",
                "cost": "stable",
                "success_rate": "stable"
            },
            "note": "Trend analysis not yet implemented"
        }

    def identify_bottlenecks(
        self,
        workflow_id: str,
        min_executions: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in workflow

        Args:
            workflow_id: Workflow identifier
            min_executions: Minimum executions needed for analysis

        Returns:
            List of bottleneck states
        """
        # In production, this would analyze state execution times
        return []

    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up old metrics files

        Args:
            days_to_keep: Number of days to keep metrics
        """
        try:
            cutoff_date = datetime.utcnow().timestamp() - (days_to_keep * 86400)

            cleaned_count = 0
            for metrics_file in self.metrics_dir.glob("*_metrics.json"):
                if metrics_file.stat().st_mtime < cutoff_date:
                    metrics_file.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old metrics files")

        except Exception as e:
            logger.error(f"Failed to cleanup metrics: {e}")


# Global metrics service instance
metrics_service = MetricsService()
