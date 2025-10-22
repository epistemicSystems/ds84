"""Cognitive workflow engine for executing formalized workflows"""
import yaml
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from app.models.workflow_models import (
    WorkflowDefinition,
    CognitiveState,
    StateTransition,
    WorkflowExecutionState,
    WorkflowExecutionMetrics,
    StateExecutionMetrics,
    WorkflowStatus,
    CognitiveStateResult,
    ReasoningTrace,
    SchemaDefinition,
    ContextRequirement
)
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service
from app.services.metrics_service import metrics_service

# Configure logging
logger = logging.getLogger(__name__)


class CognitiveWorkflowEngine:
    """Engine for executing cognitive workflows with formalized state transitions"""

    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecutionState] = {}
        self._load_workflows()

    def _load_workflows(self):
        """Load all workflow definitions from YAML files"""
        if not self.workflows_dir.exists():
            return

        for workflow_file in self.workflows_dir.glob("*.yaml"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)

                # Parse workflow definition
                workflow = self._parse_workflow_definition(workflow_data)
                self.workflows[workflow.id] = workflow

                print(f"Loaded workflow: {workflow.id}")
            except Exception as e:
                print(f"Error loading workflow {workflow_file}: {e}")

    def _parse_workflow_definition(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow definition from YAML data"""
        # Parse states
        states = {}
        for state_id, state_data in data.get('states', {}).items():
            # Parse input/output schemas
            input_schema = SchemaDefinition(**state_data.get('input_schema', {}))
            output_schema = SchemaDefinition(**state_data.get('output_schema', {}))

            # Parse context requirements
            context_reqs = [
                ContextRequirement(**req)
                for req in state_data.get('context_requirements', [])
            ]

            states[state_id] = CognitiveState(
                id=state_data['id'],
                name=state_data['name'],
                state_type=state_data['state_type'],
                agent_type=state_data['agent_type'],
                prompt_template=state_data['prompt_template'],
                input_schema=input_schema,
                output_schema=output_schema,
                context_requirements=context_reqs,
                model=state_data.get('model', 'gpt-4'),
                temperature=state_data.get('temperature', 0.7),
                max_tokens=state_data.get('max_tokens', 1000),
                timeout=state_data.get('timeout', 30),
                retry_count=state_data.get('retry_count', 2),
                description=state_data.get('description')
            )

        # Parse transitions
        transitions = [
            StateTransition(**trans)
            for trans in data.get('transitions', [])
        ]

        return WorkflowDefinition(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            version=data.get('version', '1.0.0'),
            entry_point=data['entry_point'],
            states=states,
            transitions=transitions,
            metadata=data.get('metadata', {})
        )

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a complete cognitive workflow

        Args:
            workflow_id: Workflow identifier
            input_data: Input data for the workflow
            context_data: Optional context data

        Returns:
            Dictionary with workflow results and metrics
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())

        # Initialize execution state
        execution = WorkflowExecutionState(
            workflow_id=workflow_id,
            execution_id=execution_id,
            current_state_id=workflow.entry_point,
            status=WorkflowStatus.IN_PROGRESS,
            data=input_data.copy(),
            context=context_data or {},
            metrics=WorkflowExecutionMetrics(
                workflow_id=workflow_id,
                execution_id=execution_id,
                start_time=datetime.now(),
                status=WorkflowStatus.IN_PROGRESS
            )
        )

        self.executions[execution_id] = execution

        # Log workflow start
        metrics_service.log_workflow_start(
            execution_id=execution_id,
            workflow_id=workflow_id,
            input_data=input_data
        )

        try:
            # Execute workflow states
            while execution.current_state_id:
                current_state = workflow.states[execution.current_state_id]

                # Execute state
                state_result = await self._execute_state(
                    workflow=workflow,
                    state=current_state,
                    execution=execution
                )

                # Update execution state
                execution.data.update(state_result.output_data)
                execution.metrics.states_executed.append(current_state.id)
                execution.metrics.state_metrics[current_state.id] = state_result.metrics

                # Update totals
                execution.metrics.total_tokens += state_result.metrics.token_count or 0
                execution.metrics.total_cost += state_result.metrics.cost or 0.0

                # Log state execution
                metrics_service.log_state_execution(
                    execution_id=execution_id,
                    state_id=current_state.id,
                    metrics=state_result.metrics
                )

                if not state_result.success:
                    execution.status = WorkflowStatus.FAILED
                    execution.metrics.status = WorkflowStatus.FAILED
                    execution.metrics.error = state_result.error

                    # Log error
                    metrics_service.log_error(
                        execution_id=execution_id,
                        error_type="state_execution_failed",
                        error_message=state_result.error or "Unknown error",
                        context={"state_id": current_state.id}
                    )
                    break

                # Find next transition
                next_state_id = self._find_next_state(
                    workflow=workflow,
                    current_state_id=execution.current_state_id,
                    execution_data=execution.data
                )

                # Log state transition
                metrics_service.log_state_transition(
                    execution_id=execution_id,
                    from_state=execution.current_state_id,
                    to_state=next_state_id
                )

                execution.current_state_id = next_state_id

            # Mark as completed if not failed
            if execution.status != WorkflowStatus.FAILED:
                execution.status = WorkflowStatus.COMPLETED
                execution.metrics.status = WorkflowStatus.COMPLETED

            # Update final metrics
            execution.metrics.end_time = datetime.now()
            duration = execution.metrics.end_time - execution.metrics.start_time
            execution.metrics.total_duration_ms = duration.total_seconds() * 1000

            # Log workflow completion
            metrics_service.log_workflow_complete(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=execution.status.value,
                metrics=execution.metrics.dict()
            )

            return {
                "execution_id": execution_id,
                "status": execution.status.value,
                "final_output": execution.data,
                "metrics": execution.metrics.dict()
            }

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.metrics.status = WorkflowStatus.FAILED
            execution.metrics.error = str(e)
            execution.metrics.end_time = datetime.now()

            # Log error
            metrics_service.log_error(
                execution_id=execution_id,
                error_type="workflow_exception",
                error_message=str(e),
                context={"workflow_id": workflow_id}
            )

            # Log workflow completion (with failure)
            duration = execution.metrics.end_time - execution.metrics.start_time
            execution.metrics.total_duration_ms = duration.total_seconds() * 1000
            metrics_service.log_workflow_complete(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status="failed",
                metrics=execution.metrics.dict()
            )

            return {
                "execution_id": execution_id,
                "status": "failed",
                "error": str(e),
                "metrics": execution.metrics.dict()
            }

    async def _execute_state(
        self,
        workflow: WorkflowDefinition,
        state: CognitiveState,
        execution: WorkflowExecutionState
    ) -> CognitiveStateResult:
        """Execute a single cognitive state

        Args:
            workflow: Workflow definition
            state: State to execute
            execution: Current execution state

        Returns:
            State execution result
        """
        start_time = datetime.now()
        metrics = StateExecutionMetrics(
            state_id=state.id,
            start_time=start_time
        )

        try:
            # Validate input data against schema
            # (In production, use jsonschema library for validation)

            # Assemble context for this state
            state_context = self._assemble_context(state, execution)

            # Get prompt template
            prompt = prompt_service.get_prompt(
                state.prompt_template,
                **execution.data,
                **state_context
            )

            # Execute LLM call
            response = await llm_service.complete(
                prompt=prompt,
                model=state.model,
                temperature=state.temperature,
                max_tokens=state.max_tokens
            )

            # Parse output
            output_data = self._parse_state_output(response, state)

            # Update metrics
            metrics.end_time = datetime.now()
            duration = metrics.end_time - metrics.start_time
            metrics.duration_ms = duration.total_seconds() * 1000
            metrics.success = True

            # Estimate token count and cost (rough estimate)
            metrics.token_count = len(prompt.split()) + len(response.split())
            metrics.cost = self._estimate_cost(state.model, metrics.token_count)

            return CognitiveStateResult(
                state_id=state.id,
                success=True,
                output_data=output_data,
                metrics=metrics
            )

        except Exception as e:
            metrics.end_time = datetime.now()
            duration = metrics.end_time - metrics.start_time
            metrics.duration_ms = duration.total_seconds() * 1000
            metrics.success = False
            metrics.error = str(e)

            return CognitiveStateResult(
                state_id=state.id,
                success=False,
                output_data={},
                metrics=metrics,
                error=str(e)
            )

    def _assemble_context(
        self,
        state: CognitiveState,
        execution: WorkflowExecutionState
    ) -> Dict[str, Any]:
        """Assemble context data for state execution

        Args:
            state: State definition
            execution: Current execution state

        Returns:
            Context dictionary
        """
        context = {}

        for req in state.context_requirements:
            if req.type in execution.context:
                context[req.type] = execution.context[req.type]
            elif req.required:
                # In production, raise error if required context missing
                context[req.type] = None

        return context

    def _parse_state_output(
        self,
        response: str,
        state: CognitiveState
    ) -> Dict[str, Any]:
        """Parse LLM response into structured output

        Args:
            response: LLM response text
            state: State definition

        Returns:
            Parsed output data
        """
        # Try to extract JSON from response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # No JSON found, return as text
                return {"response": response}
        except json.JSONDecodeError:
            return {"response": response}

    def _find_next_state(
        self,
        workflow: WorkflowDefinition,
        current_state_id: str,
        execution_data: Dict[str, Any]
    ) -> Optional[str]:
        """Find next state based on transitions

        Args:
            workflow: Workflow definition
            current_state_id: Current state ID
            execution_data: Current execution data

        Returns:
            Next state ID or None if terminal
        """
        # Find transitions from current state
        transitions = [
            t for t in workflow.transitions
            if t.from_state == current_state_id
        ]

        if not transitions:
            return None

        # Evaluate conditions
        for transition in transitions:
            if transition.condition is None:
                # Unconditional transition
                return transition.to_state

            # Evaluate condition
            if self._evaluate_condition(transition.condition, execution_data):
                return transition.to_state

        # No matching transition found
        return transitions[0].to_state if transitions else None

    def _evaluate_condition(
        self,
        condition: Any,
        data: Dict[str, Any]
    ) -> bool:
        """Evaluate transition condition

        Args:
            condition: Condition object
            data: Execution data

        Returns:
            True if condition is met
        """
        if not hasattr(condition, 'field'):
            return True

        field = condition.field
        operator = condition.operator
        value = condition.value

        if field not in data:
            return operator == "exists" and not value

        field_value = data[field]

        if operator == "eq":
            return field_value == value
        elif operator == "ne":
            return field_value != value
        elif operator == "gt":
            return field_value > value
        elif operator == "lt":
            return field_value < value
        elif operator == "contains":
            return value in field_value
        elif operator == "exists":
            return (field_value is not None) == value

        return False

    def _estimate_cost(self, model: str, token_count: int) -> float:
        """Estimate API call cost

        Args:
            model: Model name
            token_count: Estimated token count

        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates (as of 2024)
        cost_per_1k = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "text-embedding-ada-002": 0.0001
        }

        base_cost = cost_per_1k.get(model, 0.01)
        return (token_count / 1000) * base_cost

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID"""
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[str]:
        """List available workflow IDs"""
        return list(self.workflows.keys())

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Get execution state by ID"""
        return self.executions.get(execution_id)


# Global workflow engine instance
workflow_engine = CognitiveWorkflowEngine()
