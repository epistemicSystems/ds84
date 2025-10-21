"""FastAPI application entry point for Realtor AI Copilot"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from app.services.workflow_service import workflow_service
from app.services.agent_analysis_workflow import agent_workflow
from app.config import settings

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
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.environment
    }


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


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print(f"Starting {settings.app_name}...")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print(f"Shutting down {settings.app_name}...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
