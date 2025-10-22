"""FastAPI application entry point for Realtor AI Copilot"""
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from app.services.workflow_service import workflow_service
from app.services.agent_analysis_workflow import agent_workflow
from app.services.cognitive_workflow_engine import workflow_engine
from app.services.context_manager import context_manager
from app.services.performance_analyzer import performance_analyzer
from app.services.adaptive_router import adaptive_router, RoutingStrategy
from app.services.prompt_optimizer import prompt_optimizer
from app.services.self_improvement_engine import self_improvement_engine
from app.services.cost_quality_optimizer import cost_quality_optimizer, OptimizationObjective
from app.services.interaction_logger import interaction_logger, FeedbackType
from app.services.feedback_analyzer import feedback_analyzer
from app.services.preference_learner import preference_learner
from app.services.ab_testing_framework import ab_testing_framework
from app.services.cache_service import cache_service
from app.models.interaction_models import FeedbackRequest, FeedbackResponse
from app.middleware.rate_limiter import rate_limit_middleware
from app.middleware.auth import auth_middleware
from app.middleware.input_validation import input_validation_middleware
from app.config import settings
import uuid
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Realtor AI Copilot API",
    description="Multi-modal AI assistant for real estate agent augmentation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add input validation middleware (first for security)
app.middleware("http")(input_validation_middleware)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Add authentication middleware
app.middleware("http")(auth_middleware)


# Request/Response Models
class PropertySearchRequest(BaseModel):
    """Request model for property search"""
    query: str
    user_id: Optional[str] = None
    search_mode: Literal["vector", "hybrid", "simulated"] = Field(
        default="vector",
        description="Search mode: 'vector' for semantic search, 'hybrid' for vector + filters, 'simulated' for LLM-generated results"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of properties to return"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I need a modern 3-bedroom home with a view of the water, ideally with an open floor plan and within walking distance to restaurants. My budget is around $750,000.",
                "user_id": "user123",
                "search_mode": "vector",
                "limit": 10
            }
        }


class PropertySearchResponse(BaseModel):
    """Response model for property search"""
    query: str
    intent: Optional[Dict[str, Any]] = None
    properties: Optional[List[Dict[str, Any]]] = None
    response: Optional[str] = None
    search_mode: Optional[str] = None
    status: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    environment: str


class AgentAnalysisRequest(BaseModel):
    """Request model for agent analysis"""
    agent_id: str
    area_codes: Optional[List[str]] = None
    comparison_agent_ids: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "A12345",
                "area_codes": ["95113", "95125"],
                "comparison_agent_ids": ["B67890", "C54321"]
            }
        }


class AgentAnalysisResponse(BaseModel):
    """Response model for agent analysis"""
    agent_id: str
    agent_data: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    strategic_insights: Optional[str] = None
    status: str
    error: Optional[str] = None


class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution"""
    workflow_id: str = Field(
        description="ID of the workflow to execute",
        example="property_query_processing"
    )
    input_data: Dict[str, Any] = Field(
        description="Input data for the workflow",
        example={"query": "Find me a 3-bedroom house with a pool"}
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for context and personalization"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "property_query_processing",
                "input_data": {"query": "Modern 3-bedroom with pool in Silicon Valley"},
                "user_id": "user123"
            }
        }


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution"""
    execution_id: str
    workflow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WorkflowListResponse(BaseModel):
    """Response model for workflow listing"""
    workflows: List[Dict[str, Any]]


class WorkflowMetricsResponse(BaseModel):
    """Response model for workflow metrics"""
    execution_id: str
    metrics: Dict[str, Any]


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Realtor AI Copilot API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint for load balancers and container orchestrators"""
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.environment
    }


@app.get("/health/readiness", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies all dependencies are available

    This endpoint checks:
    - Application is running
    - Workflow engine is initialized
    - Cache service is operational

    Returns 200 if ready, 503 if not ready
    """
    try:
        checks = {
            "app": "ok",
            "workflows": "checking",
            "cache": "checking"
        }

        # Check workflow engine
        try:
            workflow_ids = workflow_engine.list_workflows()
            checks["workflows"] = "ok" if len(workflow_ids) > 0 else "no_workflows"
        except Exception as e:
            checks["workflows"] = f"error: {str(e)}"

        # Check cache service
        try:
            cache_stats = cache_service.get_stats()
            checks["cache"] = "ok"
        except Exception as e:
            checks["cache"] = f"error: {str(e)}"

        # Determine overall status
        all_ok = all(status == "ok" or status == "no_workflows" for status in checks.values())

        return {
            "status": "ready" if all_ok else "not_ready",
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.get("/health/liveness", tags=["Health"])
async def liveness_check():
    """Liveness check - verifies application is alive

    Simple check that returns 200 if the application is running.
    Used by container orchestrators to detect if the app needs to be restarted.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with system metrics

    Provides comprehensive system health information including:
    - Service status
    - Cache statistics
    - Workflow information
    - System metrics

    Note: This endpoint may be slower than /health
    """
    try:
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "environment": settings.environment,
            "uptime_seconds": 0,  # Would need tracking
            "services": {},
            "metrics": {}
        }

        # Workflow engine status
        try:
            workflow_ids = workflow_engine.list_workflows()
            health_info["services"]["workflow_engine"] = {
                "status": "ok",
                "workflows_loaded": len(workflow_ids),
                "workflow_ids": workflow_ids
            }
        except Exception as e:
            health_info["services"]["workflow_engine"] = {
                "status": "error",
                "error": str(e)
            }
            health_info["status"] = "degraded"

        # Cache service status
        try:
            cache_stats = cache_service.get_stats()
            health_info["services"]["cache"] = {
                "status": "ok",
                "stats": cache_stats
            }
        except Exception as e:
            health_info["services"]["cache"] = {
                "status": "error",
                "error": str(e)
            }
            health_info["status"] = "degraded"

        # A/B testing framework status
        try:
            tests = ab_testing_framework.list_tests()
            health_info["services"]["ab_testing"] = {
                "status": "ok",
                "active_tests": len([t for t in tests if t.status == "running"])
            }
        except Exception as e:
            health_info["services"]["ab_testing"] = {
                "status": "error",
                "error": str(e)
            }

        # Context manager status
        try:
            # Get count of active sessions (simplified)
            session_count = len(context_manager.sessions)
            health_info["services"]["context_manager"] = {
                "status": "ok",
                "active_sessions": session_count
            }
        except Exception as e:
            health_info["services"]["context_manager"] = {
                "status": "error",
                "error": str(e)
            }

        return health_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/api/property-search", response_model=PropertySearchResponse, tags=["Property Search"])
async def property_search(request: PropertySearchRequest):
    """Execute property search workflow from natural language query

    This endpoint processes natural language property queries through a multi-stage
    cognitive workflow:
    1. Intent Analysis - Parse and structure the query
    2. Property Search - Find matching properties using selected mode:
       - **vector**: Semantic similarity search using embeddings
       - **hybrid**: Combined vector search with structured filters
       - **simulated**: LLM-generated results (for testing without database)
    3. Response Generation - Create natural language response

    Args:
        request: PropertySearchRequest with query, search_mode, and limit

    Returns:
        PropertySearchResponse with intent, properties, and response

    Example:
        ```json
        {
          "query": "Find me a spacious 4-bedroom house with a pool",
          "search_mode": "vector",
          "limit": 10
        }
        ```
    """
    try:
        result = await workflow_service.execute(
            query=request.query,
            user_id=request.user_id,
            search_mode=request.search_mode,
            limit=request.limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent-analysis", response_model=AgentAnalysisResponse, tags=["Agent Analysis"])
async def analyze_agent(request: AgentAnalysisRequest):
    """Execute agent analysis workflow

    This endpoint analyzes real estate agent performance through a multi-stage workflow:
    1. Data Collection - Extract agent and market data
    2. Performance Analysis - Calculate and benchmark metrics
    3. Insight Generation - Generate strategic insights

    Args:
        request: AgentAnalysisRequest with agent_id and optional parameters

    Returns:
        AgentAnalysisResponse with analysis and insights
    """
    try:
        result = await agent_workflow.execute(
            request.agent_id,
            request.area_codes,
            request.comparison_agent_ids
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows", response_model=WorkflowListResponse, tags=["Cognitive Workflows"])
async def list_workflows():
    """List all available cognitive workflows

    Returns a list of workflow definitions with their metadata.

    Returns:
        WorkflowListResponse with list of available workflows
    """
    try:
        workflow_ids = workflow_engine.list_workflows()
        workflows = []

        for workflow_id in workflow_ids:
            workflow = workflow_engine.get_workflow(workflow_id)
            if workflow:
                workflows.append({
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "version": workflow.version,
                    "entry_point": workflow.entry_point,
                    "states_count": len(workflow.states),
                    "transitions_count": len(workflow.transitions),
                    "metadata": workflow.metadata
                })

        return {"workflows": workflows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/execute", response_model=WorkflowExecutionResponse, tags=["Cognitive Workflows"])
async def execute_workflow(
    request: WorkflowExecutionRequest,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID")
):
    """Execute a cognitive workflow with full context management

    This endpoint executes a declarative YAML-based cognitive workflow with:
    - Multi-stage state transitions with conditions
    - Session and user context management
    - Conversation history tracking
    - Comprehensive metrics and reasoning traces
    - Input/output schema validation

    Args:
        request: WorkflowExecutionRequest with workflow_id and input_data
        x_session_id: Optional session ID header for context continuity

    Returns:
        WorkflowExecutionResponse with execution results and metrics

    Example:
        ```json
        {
          "workflow_id": "property_query_processing",
          "input_data": {"query": "Modern 3-bedroom with pool"},
          "user_id": "user123"
        }
        ```
    """
    try:
        # Generate or use provided session ID
        session_id = x_session_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"

        # Get full context for the user/session
        context_data = context_manager.get_full_context(
            session_id=session_id,
            user_id=user_id
        )

        # Add current query to conversation history
        if "query" in request.input_data:
            context_manager.add_message(
                session_id=session_id,
                role="user",
                content=request.input_data["query"],
                metadata={"workflow_id": request.workflow_id}
            )

        # Execute workflow
        result = await workflow_engine.execute_workflow(
            workflow_id=request.workflow_id,
            input_data=request.input_data,
            context_data=context_data
        )

        # Add response to conversation history
        if result.get("status") == "completed" and "final_output" in result:
            response_content = result["final_output"].get("response", "")
            if response_content:
                context_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=response_content,
                    metadata={
                        "workflow_id": request.workflow_id,
                        "execution_id": result.get("execution_id")
                    }
                )

        # Infer preferences from interaction
        if "query" in request.input_data and result.get("status") == "completed":
            # Extract intent from result for preference inference
            intent = result.get("final_output", {}).get("intent", {})
            if intent:
                context_manager.infer_preferences_from_interaction(
                    user_id=user_id,
                    interaction_data={"query": request.input_data["query"], "intent": intent}
                )

        return {
            "execution_id": result.get("execution_id"),
            "workflow_id": request.workflow_id,
            "status": result.get("status"),
            "result": result.get("final_output"),
            "metrics": result.get("metrics")
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/executions/{execution_id}/metrics",
         response_model=WorkflowMetricsResponse,
         tags=["Cognitive Workflows"])
async def get_execution_metrics(execution_id: str):
    """Get detailed metrics for a workflow execution

    Retrieves comprehensive metrics including:
    - Total execution time and cost
    - Per-state execution metrics
    - Token usage statistics
    - State transition history

    Args:
        execution_id: Unique workflow execution identifier

    Returns:
        WorkflowMetricsResponse with detailed metrics
    """
    try:
        # In a production system, this would query a metrics database
        # For now, return a placeholder response
        return {
            "execution_id": execution_id,
            "metrics": {
                "message": "Metrics retrieval not yet implemented",
                "note": "Metrics are returned in the workflow execution response"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/context/sessions/{session_id}", tags=["Context Management"])
async def get_session_context(session_id: str):
    """Get conversation history and context for a session

    Args:
        session_id: Session identifier

    Returns:
        Session context including conversation history
    """
    try:
        context_window = context_manager.get_context_window(
            session_id=session_id,
            max_tokens=4000
        )

        session = context_manager.sessions.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.messages),
            "context_window": context_window
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/context/users/{user_id}/preferences", tags=["Context Management"])
async def get_user_preferences(user_id: str):
    """Get all preferences for a user

    Args:
        user_id: User identifier

    Returns:
        User preferences (explicit and inferred)
    """
    try:
        preferences = context_manager.get_all_preferences(user_id)

        if not preferences:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "user_id": user_id,
            "preferences": preferences
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metacognitive/performance/{workflow_id}", tags=["Meta-cognitive Optimization"])
async def analyze_workflow_performance(
    workflow_id: str,
    time_window_hours: int = 24,
    min_executions: int = 5
):
    """Analyze workflow performance and detect bottlenecks

    Performs comprehensive performance analysis including:
    - Bottleneck detection
    - Optimization opportunities
    - Health score calculation
    - Actionable recommendations

    Args:
        workflow_id: Workflow to analyze
        time_window_hours: Analysis time window
        min_executions: Minimum executions required

    Returns:
        Performance analysis with bottlenecks and recommendations
    """
    try:
        analysis = performance_analyzer.analyze_workflow_performance(
            workflow_id=workflow_id,
            time_window_hours=time_window_hours,
            min_executions=min_executions
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metacognitive/route", tags=["Meta-cognitive Optimization"])
async def adaptive_route(request: Dict[str, Any]):
    """Get adaptive routing recommendation

    Uses performance data to recommend optimal execution path.

    Request body:
    {
      "workflow_id": "property_query_processing",
      "state_id": "intent_analysis",
      "context": {"task_complexity": "medium"},
      "strategy": "balanced"
    }

    Strategies: performance, cost, quality, balanced

    Returns:
        Routing decision with selected model and reasoning
    """
    try:
        workflow_id = request.get("workflow_id")
        state_id = request.get("state_id")
        context = request.get("context", {})
        strategy = request.get("strategy", "balanced")

        decision = adaptive_router.route_execution(
            workflow_id=workflow_id,
            current_state_id=state_id,
            context=context,
            strategy=RoutingStrategy(strategy)
        )

        return {
            "selected_model": decision.selected_model,
            "selected_temperature": decision.selected_temperature,
            "strategy_used": decision.strategy_used.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "estimated_metrics": decision.estimated_metrics,
            "alternatives": decision.alternatives_considered
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metacognitive/optimize-prompt", tags=["Meta-cognitive Optimization"])
async def optimize_prompt_endpoint(request: Dict[str, Any]):
    """Optimize a prompt using meta-cognitive analysis

    Request body:
    {
      "prompt_key": "property_search.intent_analysis",
      "optimization_type": "comprehensive",
      "variables": {}
    }

    Optimization types: conciseness, clarity, structure, comprehensive

    Returns:
        Optimized prompt with improvements and token reduction
    """
    try:
        prompt_key = request.get("prompt_key")
        optimization_type = request.get("optimization_type", "comprehensive")
        variables = request.get("variables")

        optimized = await prompt_optimizer.optimize_prompt(
            prompt_key=prompt_key,
            optimization_type=optimization_type,
            variables=variables
        )

        return {
            "original_key": optimized.original_key,
            "optimized_content": optimized.optimized_content,
            "optimization_type": optimized.optimization_type,
            "token_reduction": optimized.token_reduction,
            "expected_improvements": optimized.expected_improvements,
            "confidence": optimized.confidence,
            "reasoning": optimized.reasoning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metacognitive/self-improve", tags=["Meta-cognitive Optimization"])
async def run_self_improvement(request: Dict[str, Any]):
    """Run self-improvement cycle for a workflow

    Executes complete improvement cycle:
    1. Performance analysis
    2. Identify optimizations
    3. Validate changes
    4. Apply optimizations (if not dry run)

    Request body:
    {
      "workflow_id": "property_query_processing",
      "time_window_hours": 24,
      "dry_run": true
    }

    Returns:
        Improvement cycle results with actions taken
    """
    try:
        workflow_id = request.get("workflow_id")
        time_window_hours = request.get("time_window_hours", 24)
        dry_run = request.get("dry_run", True)

        result = await self_improvement_engine.run_improvement_cycle(
            workflow_id=workflow_id,
            time_window_hours=time_window_hours,
            dry_run=dry_run
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metacognitive/improvement-history", tags=["Meta-cognitive Optimization"])
async def get_improvement_history(workflow_id: Optional[str] = None):
    """Get self-improvement cycle history

    Args:
        workflow_id: Optional filter by workflow

    Returns:
        List of improvement cycles
    """
    try:
        history = self_improvement_engine.get_improvement_history(workflow_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metacognitive/cost-quality", tags=["Meta-cognitive Optimization"])
async def optimize_cost_quality(request: Dict[str, Any]):
    """Optimize cost/quality tradeoff

    Request body:
    {
      "objective": "balanced",
      "context": {
        "estimated_tokens": 1000,
        "task_complexity": "medium"
      },
      "constraints": {
        "max_cost": 0.05,
        "min_quality": 0.8
      }
    }

    Objectives: minimize_cost, maximize_quality, balanced,
                cost_constrained, quality_constrained

    Returns:
        Optimal configuration with cost/quality estimates
    """
    try:
        objective = request.get("objective", "balanced")
        context = request.get("context", {})
        constraints = request.get("constraints")

        tradeoff = cost_quality_optimizer.optimize(
            objective=OptimizationObjective(objective),
            context=context,
            constraints=constraints
        )

        return {
            "configuration": tradeoff.configuration,
            "estimated_cost": tradeoff.estimated_cost,
            "estimated_quality": tradeoff.estimated_quality,
            "estimated_duration": tradeoff.estimated_duration,
            "efficiency_score": tradeoff.efficiency_score,
            "recommendation_reason": tradeoff.recommendation_reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metacognitive/cost-quality/analyze", tags=["Meta-cognitive Optimization"])
async def analyze_cost_quality_tradeoff(
    estimated_tokens: int = 1000,
    task_complexity: str = "medium"
):
    """Analyze full cost/quality tradeoff curve

    Shows pareto frontier of optimal configurations.

    Args:
        estimated_tokens: Estimated token count
        task_complexity: Task complexity (low, medium, high)

    Returns:
        Tradeoff analysis with pareto frontier
    """
    try:
        context = {
            "estimated_tokens": estimated_tokens,
            "task_complexity": task_complexity
        }

        analysis = cost_quality_optimizer.analyze_tradeoff_curve(context)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback Learning"])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for learning and personalization

    Request body:
    {
      "session_id": "session_123",
      "user_id": "user_456",
      "feedback_type": "thumbs_up",
      "related_query_id": "query_789",
      "rating": 5.0
    }

    Returns:
        Feedback confirmation
    """
    try:
        feedback_id = interaction_logger.log_feedback(
            session_id=feedback.session_id,
            user_id=feedback.user_id or "anonymous",
            feedback_type=FeedbackType(feedback.feedback_type),
            related_query_id=feedback.related_query_id,
            related_item_id=feedback.related_item_id,
            rating=feedback.rating,
            comment=feedback.comment
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="success",
            message="Feedback recorded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/analysis/{user_id}", tags=["Feedback Learning"])
async def get_feedback_analysis(user_id: str, days: int = 30):
    """Get feedback analysis for a user

    Analyzes interaction patterns to identify preferences and behaviors.

    Args:
        user_id: User identifier
        days: Days to analyze

    Returns:
        Feedback analysis results
    """
    try:
        analysis = feedback_analyzer.analyze_user_patterns(user_id, days)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preferences/{user_id}", tags=["Feedback Learning"])
async def get_user_preferences_learned(user_id: str):
    """Get learned preferences for a user

    Returns preferences learned from user interactions.

    Args:
        user_id: User identifier

    Returns:
        Learned preferences with confidence scores
    """
    try:
        preferences = preference_learner.learn_preferences(user_id)
        return {
            "user_id": user_id,
            "preferences": preferences,
            "learned_at": preferences.get("learned_at", "").isoformat() if preferences.get("learned_at") else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preferences/{user_id}/explain", tags=["Feedback Learning"])
async def explain_user_preferences(user_id: str):
    """Get human-readable explanation of learned preferences

    Args:
        user_id: User identifier

    Returns:
        Preference explanation
    """
    try:
        explanation = preference_learner.explain_preferences(user_id)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preferences/{user_id}/apply", tags=["Feedback Learning"])
async def apply_preferences_to_search(request: Dict[str, Any]):
    """Apply learned preferences to enhance search query

    Request body:
    {
      "user_id": "user_123",
      "query": "Find me a house",
      "intent": {...}
    }

    Returns:
        Enhanced intent with preferences applied
    """
    try:
        user_id = request.get("user_id")
        query = request.get("query", "")
        intent = request.get("intent", {})

        enhanced_intent = preference_learner.apply_preferences_to_query(
            user_id=user_id,
            query=query,
            intent=intent
        )

        return {
            "original_intent": intent,
            "enhanced_intent": enhanced_intent,
            "preferences_applied": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-testing/tests", tags=["A/B Testing"])
async def create_ab_test(request: Dict[str, Any]):
    """Create a new A/B test

    Request body:
    {
      "test_name": "Property Search Workflow Optimization",
      "test_type": "workflow",
      "variants": [
        {
          "variant_id": "control",
          "variant_name": "Current Workflow",
          "configuration": {...},
          "traffic_percentage": 50.0
        },
        {
          "variant_id": "optimized",
          "variant_name": "Optimized Workflow",
          "configuration": {...},
          "traffic_percentage": 50.0
        }
      ],
      "control_variant_id": "control",
      "primary_metric": "success_rate",
      "min_sample_size": 100
    }

    Returns:
        Test ID and confirmation
    """
    try:
        test_name = request.get("test_name")
        test_type = request.get("test_type")
        variants = request.get("variants", [])
        control_variant_id = request.get("control_variant_id")
        primary_metric = request.get("primary_metric", "success_rate")
        min_sample_size = request.get("min_sample_size", 100)

        test_id = ab_testing_framework.create_test(
            test_name=test_name,
            test_type=test_type,
            variants=variants,
            control_variant_id=control_variant_id,
            primary_metric=primary_metric,
            min_sample_size=min_sample_size
        )

        return {
            "test_id": test_id,
            "status": "created",
            "message": "A/B test created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ab-testing/tests", tags=["A/B Testing"])
async def list_ab_tests(
    status: Optional[str] = None,
    test_type: Optional[str] = None
):
    """List all A/B tests with optional filters

    Args:
        status: Filter by status (running, completed, paused)
        test_type: Filter by type (workflow, prompt, model, routing)

    Returns:
        List of A/B tests
    """
    try:
        tests = ab_testing_framework.list_tests(
            status=status,
            test_type=test_type
        )

        return {
            "tests": [
                {
                    "test_id": test.test_id,
                    "test_name": test.test_name,
                    "test_type": test.test_type,
                    "status": test.status,
                    "created_at": test.created_at.isoformat(),
                    "variants_count": len(test.variants),
                    "primary_metric": test.primary_metric
                }
                for test in tests
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ab-testing/tests/{test_id}", tags=["A/B Testing"])
async def get_ab_test(test_id: str):
    """Get detailed information about an A/B test

    Args:
        test_id: Test identifier

    Returns:
        Test details with variants and results
    """
    try:
        test = ab_testing_framework.get_test(test_id)

        if not test:
            raise HTTPException(status_code=404, detail="Test not found")

        return {
            "test_id": test.test_id,
            "test_name": test.test_name,
            "test_type": test.test_type,
            "status": test.status,
            "created_at": test.created_at.isoformat(),
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "ended_at": test.ended_at.isoformat() if test.ended_at else None,
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "variant_name": v.variant_name,
                    "traffic_percentage": v.traffic_percentage,
                    "samples_collected": v.samples_collected,
                    "metric_values": v.metric_values
                }
                for v in test.variants
            ],
            "control_variant_id": test.control_variant_id,
            "min_sample_size": test.min_sample_size,
            "primary_metric": test.primary_metric
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-testing/tests/{test_id}/assign", tags=["A/B Testing"])
async def assign_variant(test_id: str, request: Dict[str, Any]):
    """Assign a user to a test variant

    Request body:
    {
      "user_id": "user_123"
    }

    Returns:
        Assigned variant ID
    """
    try:
        user_id = request.get("user_id")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        variant_id = ab_testing_framework.assign_variant(test_id, user_id)

        return {
            "test_id": test_id,
            "user_id": user_id,
            "variant_id": variant_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-testing/tests/{test_id}/record", tags=["A/B Testing"])
async def record_test_result(test_id: str, request: Dict[str, Any]):
    """Record a result for an A/B test

    Request body:
    {
      "user_id": "user_123",
      "variant_id": "optimized",
      "metrics": {
        "success_rate": 0.85,
        "latency": 1.2,
        "cost": 0.003
      }
    }

    Returns:
        Confirmation
    """
    try:
        user_id = request.get("user_id")
        variant_id = request.get("variant_id")
        metrics = request.get("metrics", {})

        ab_testing_framework.record_result(
            test_id=test_id,
            user_id=user_id,
            variant_id=variant_id,
            metrics=metrics
        )

        return {
            "status": "recorded",
            "message": "Result recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ab-testing/tests/{test_id}/analyze", tags=["A/B Testing"])
async def analyze_ab_test(test_id: str):
    """Analyze A/B test results with statistical significance

    Args:
        test_id: Test identifier

    Returns:
        Statistical analysis with winner determination
    """
    try:
        result = ab_testing_framework.analyze_test(test_id)

        return {
            "test_id": result.test_id,
            "test_name": result.test_name,
            "analysis_timestamp": result.analysis_timestamp.isoformat(),
            "has_winner": result.has_winner,
            "winner_variant_id": result.winner_variant_id,
            "confidence_level": result.confidence_level,
            "p_value": result.p_value,
            "variant_results": [
                {
                    "variant_id": vr.variant_id,
                    "variant_name": vr.variant_name,
                    "sample_size": vr.sample_size,
                    "mean_value": vr.mean_value,
                    "confidence_interval_lower": vr.confidence_interval_lower,
                    "confidence_interval_upper": vr.confidence_interval_upper,
                    "is_winner": vr.is_winner
                }
                for vr in result.variant_results
            ],
            "recommendation": result.recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-testing/tests/{test_id}/start", tags=["A/B Testing"])
async def start_ab_test(test_id: str):
    """Start an A/B test

    Args:
        test_id: Test identifier

    Returns:
        Confirmation
    """
    try:
        ab_testing_framework.start_test(test_id)

        return {
            "test_id": test_id,
            "status": "running",
            "message": "Test started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-testing/tests/{test_id}/stop", tags=["A/B Testing"])
async def stop_ab_test(test_id: str):
    """Stop an A/B test

    Args:
        test_id: Test identifier

    Returns:
        Confirmation with final results
    """
    try:
        ab_testing_framework.stop_test(test_id)

        return {
            "test_id": test_id,
            "status": "completed",
            "message": "Test stopped successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print(f"Starting {settings.app_name}...")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")

    # Load cognitive workflows
    try:
        workflow_ids = workflow_engine.list_workflows()
        print(f"Loaded {len(workflow_ids)} cognitive workflows:")
        for workflow_id in workflow_ids:
            workflow = workflow_engine.get_workflow(workflow_id)
            if workflow:
                print(f"  - {workflow.name} (v{workflow.version})")
    except Exception as e:
        print(f"Warning: Failed to load workflows: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print(f"Shutting down {settings.app_name}...")

    # Cleanup context manager sessions
    try:
        context_manager.cleanup_expired_sessions()
        print("Context manager cleanup completed")
    except Exception as e:
        print(f"Warning: Context cleanup failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
