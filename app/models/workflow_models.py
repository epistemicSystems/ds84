"""Pydantic models for cognitive workflow system"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateType(str, Enum):
    """Cognitive state types"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    GENERATION = "generation"
    EVALUATION = "evaluation"


class SchemaDefinition(BaseModel):
    """Schema definition for state inputs/outputs"""
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class ContextRequirement(BaseModel):
    """Context requirement for a cognitive state"""
    type: str
    scope: Literal["session", "user", "global", "long_term"]
    required: bool = True
    description: Optional[str] = None


class CognitiveState(BaseModel):
    """Definition of a cognitive state in the workflow"""
    id: str
    name: str
    state_type: StateType
    agent_type: str
    prompt_template: str
    input_schema: SchemaDefinition
    output_schema: SchemaDefinition
    context_requirements: List[ContextRequirement] = Field(default_factory=list)
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    retry_count: int = 2
    description: Optional[str] = None


class TransitionCondition(BaseModel):
    """Condition for state transition"""
    field: str
    operator: Literal["eq", "ne", "gt", "lt", "contains", "exists"]
    value: Any
    description: Optional[str] = None


class DataTransformation(BaseModel):
    """Data transformation between states"""
    type: Literal["map", "filter", "extract", "merge"]
    config: Dict[str, Any]
    description: Optional[str] = None


class StateTransition(BaseModel):
    """Transition between cognitive states"""
    from_state: str
    to_state: Optional[str] = None  # None for terminal state
    condition: Optional[TransitionCondition] = None
    transformation: Optional[DataTransformation] = None
    description: Optional[str] = None


class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    entry_point: str
    states: Dict[str, CognitiveState]
    transitions: List[StateTransition]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StateExecutionMetrics(BaseModel):
    """Metrics for a single state execution"""
    state_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    token_count: Optional[int] = None
    cost: Optional[float] = None
    confidence: Optional[float] = None


class WorkflowExecutionMetrics(BaseModel):
    """Metrics for complete workflow execution"""
    workflow_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    status: WorkflowStatus
    states_executed: List[str] = Field(default_factory=list)
    state_metrics: Dict[str, StateExecutionMetrics] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0
    error: Optional[str] = None


class WorkflowExecutionState(BaseModel):
    """Current state of workflow execution"""
    workflow_id: str
    execution_id: str
    current_state_id: Optional[str] = None
    status: WorkflowStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metrics: WorkflowExecutionMetrics
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ReasoningTrace(BaseModel):
    """Trace of reasoning steps in a cognitive process"""
    step_number: int
    operation: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    alternatives_considered: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class CognitiveStateResult(BaseModel):
    """Result of cognitive state execution"""
    state_id: str
    success: bool
    output_data: Dict[str, Any]
    reasoning_trace: List[ReasoningTrace] = Field(default_factory=list)
    confidence: Optional[float] = None
    metrics: StateExecutionMetrics
    error: Optional[str] = None
