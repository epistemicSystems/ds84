# REALTOR AI COPILOT

A multi-modal AI assistant specifically tailored for real estate agent augmentation, with particular focus on providing competitive intelligence while maintaining ethical and legal compliance.

## Overview

The REALTOR AI COPILOT is an advanced cognitive system that combines natural language processing, vector search, and multi-modal perception to dramatically enhance real estate agent productivity. Built for David Shapiro (Realtor), the system provides:

- **Natural Language Property Search**: Semantic search that transcends rigid MLS parameters
- **Agent Performance Analytics**: Competitive intelligence dashboard with key metrics
- **Personalized AI Assistant**: Learns and adapts to agent-specific preferences
- **Multi-Modal Perception**: Combines visual, textual, and structured data analysis

## Architecture

The system employs a **layered intelligence model** with progressive cognitive enrichment:

```
┌───────────────────────────────────────────────────────┐
│              COGNITIVE META-SYSTEM                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Perception  │  │  Reasoning  │  │  Generation  │  │
│  │ Subsystem   │  │  Subsystem  │  │  Subsystem   │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
└───────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: FastAPI (Python 3.10+)
- **Database**: PostgreSQL 14 with pgvector extension
- **LLM APIs**: OpenAI GPT-4, GPT-3.5, Ada-002 embeddings
- **Vector Search**: pgvector with HNSW indexing
- **Container Runtime**: Docker & Docker Compose

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker Desktop
- OpenAI API key
- 8GB+ RAM, 10GB+ disk space

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/epistemicSystems/ds84.git
cd ds84
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

5. **Start infrastructure**:
```bash
docker-compose up -d
```

6. **Initialize database**:
```bash
python scripts/init_database.py
```

7. **Populate with sample data** (optional but recommended):
```bash
python scripts/populate_database.py --count 50 --verify
```

8. **Run the API server**:
```bash
uvicorn app.main:app --reload
```

9. **Visit API documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

10. **Test the API** (optional):
```bash
python scripts/test_api.py
```

## API Endpoints

### Cognitive Workflows ✨ NEW in Week 3

#### List Available Workflows

**GET** `/api/workflows`

List all available cognitive workflows with their metadata.

```bash
curl http://localhost:8000/api/workflows
```

**Response:**
```json
{
  "workflows": [
    {
      "id": "property_query_processing",
      "name": "Property Query Processing Workflow",
      "description": "Multi-stage property search with intent analysis...",
      "version": "1.0.0",
      "states_count": 4,
      "transitions_count": 5
    }
  ]
}
```

#### Execute Workflow

**POST** `/api/workflows/execute`

Execute a declarative YAML-based cognitive workflow with full context management.

```bash
curl -X POST http://localhost:8000/api/workflows/execute \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session-123" \
  -d '{
    "workflow_id": "property_query_processing",
    "input_data": {
      "query": "Modern 3-bedroom with pool in Silicon Valley"
    },
    "user_id": "user123"
  }'
```

**Features:**
- Multi-stage state transitions with conditions
- Session and user context management
- Conversation history tracking
- Comprehensive metrics and reasoning traces
- Input/output schema validation

**Response:**
```json
{
  "execution_id": "exec_abc123",
  "workflow_id": "property_query_processing",
  "status": "completed",
  "result": {
    "intent": {...},
    "properties": [...],
    "response": "..."
  },
  "metrics": {
    "total_duration_seconds": 2.34,
    "total_tokens": 1523,
    "total_cost": 0.0234,
    "states_executed": ["intent_analysis", "property_search", ...]
  }
}
```

#### Get Execution Metrics

**GET** `/api/workflows/executions/{execution_id}/metrics`

Retrieve detailed metrics for a workflow execution.

### Meta-cognitive Optimization ✨ NEW in Week 4

#### Performance Analysis

**GET** `/api/metacognitive/performance/{workflow_id}`

Analyze workflow performance and detect bottlenecks.

```bash
curl "http://localhost:8000/api/metacognitive/performance/property_query_processing?time_window_hours=24&min_executions=5"
```

**Returns:**
- Health score (0.0-1.0)
- Bottlenecks by severity (critical, high, medium, low)
- Optimization opportunities
- Actionable recommendations
- Per-state performance metrics

#### Adaptive Routing

**POST** `/api/metacognitive/route`

Get intelligent routing recommendation based on strategy.

```bash
curl -X POST http://localhost:8000/api/metacognitive/route \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "property_query_processing",
    "state_id": "intent_analysis",
    "context": {"task_complexity": "medium"},
    "strategy": "balanced"
  }'
```

**Strategies:** `performance`, `cost`, `quality`, `balanced`

#### Prompt Optimization

**POST** `/api/metacognitive/optimize-prompt`

Automatically optimize prompts using meta-cognitive analysis.

```bash
curl -X POST http://localhost:8000/api/metacognitive/optimize-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_key": "property_search.intent_analysis",
    "optimization_type": "comprehensive"
  }'
```

**Optimization types:** `conciseness`, `clarity`, `structure`, `comprehensive`

#### Self-Improvement Cycle

**POST** `/api/metacognitive/self-improve`

Run complete self-improvement cycle for a workflow.

```bash
curl -X POST http://localhost:8000/api/metacognitive/self-improve \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "property_query_processing",
    "time_window_hours": 24,
    "dry_run": true
  }'
```

**Phases:**
1. Performance analysis
2. Identify optimizations
3. Validate changes
4. Deploy optimizations (if not dry run)

#### Cost/Quality Optimization

**POST** `/api/metacognitive/cost-quality`

Optimize cost/quality tradeoff for workflow execution.

```bash
curl -X POST http://localhost:8000/api/metacognitive/cost-quality \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "balanced",
    "context": {"estimated_tokens": 1000, "task_complexity": "medium"},
    "constraints": {"max_cost": 0.05, "min_quality": 0.8}
  }'
```

**GET** `/api/metacognitive/cost-quality/analyze`

Analyze full cost/quality tradeoff curve with Pareto frontier.

### Context Management ✨ NEW in Week 3

#### Get Session Context

**GET** `/api/context/sessions/{session_id}`

Retrieve conversation history and context for a session.

```bash
curl http://localhost:8000/api/context/sessions/my-session-123
```

**Response:**
```json
{
  "session_id": "my-session-123",
  "user_id": "user123",
  "message_count": 5,
  "created_at": "2024-01-15T10:30:00",
  "last_activity": "2024-01-15T10:35:00",
  "context_window": "User: ...\nAssistant: ..."
}
```

#### Get User Preferences

**GET** `/api/context/users/{user_id}/preferences`

Get all preferences for a user (explicit and inferred).

```bash
curl http://localhost:8000/api/context/users/user123/preferences
```

**Response:**
```json
{
  "user_id": "user123",
  "preferences": {
    "preferred_property_types": ["house", "townhouse"],
    "price_range": {"min": 500000, "max": 1000000},
    "must_have_features": ["pool", "garage"]
  }
}
```

### Property Search

**POST** `/api/property-search`

Search for properties using natural language queries with multiple search modes.

#### Search Modes

- **`vector`** (default): Semantic similarity search using embeddings - best for natural language queries
- **`hybrid`**: Combined vector search with structured filters - balances semantic and exact matching
- **`simulated`**: LLM-generated results - useful for testing without a populated database

#### Example: Vector Search

```bash
curl -X POST http://localhost:8000/api/property-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need a modern 3-bedroom home with a view of the water, ideally with an open floor plan and within walking distance to restaurants. My budget is around $750,000.",
    "search_mode": "vector",
    "limit": 10
  }'
```

#### Example: Simulated Search (No Database Required)

```bash
curl -X POST http://localhost:8000/api/property-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find me a family-friendly 4-bedroom house with good schools",
    "search_mode": "simulated",
    "limit": 5
  }'
```

**Response:**
```json
{
  "query": "...",
  "intent": {
    "property_types": ["house", "condo"],
    "bedrooms": {"min": 3, "preferred": 3},
    ...
  },
  "properties": [...],
  "response": "Based on your search for a modern 3-bedroom home...",
  "status": "success"
}
```

### Agent Analysis

**POST** `/api/agent-analysis`

Analyze real estate agent performance metrics.

```bash
curl -X POST http://localhost:8000/api/agent-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "A12345",
    "area_codes": ["95113", "95125"],
    "comparison_agent_ids": ["B67890"]
  }'
```

**Response:**
```json
{
  "agent_id": "A12345",
  "agent_data": {...},
  "performance_analysis": {...},
  "strategic_insights": "# STRATEGIC INSIGHTS...",
  "status": "success"
}
```

## Project Structure

```
realtor-ai/
├── app/
│   ├── models/                         # Pydantic models
│   │   └── workflow_models.py          # ✨ Workflow definitions, state models
│   ├── services/                       # Business logic
│   │   ├── llm_service.py              # LLM interactions
│   │   ├── prompt_service.py           # Prompt management
│   │   ├── workflow_service.py         # Property search workflow
│   │   ├── embedding_service.py        # Embedding generation
│   │   ├── mls_extraction_service.py   # MLS data extraction
│   │   ├── agent_analysis_workflow.py  # Agent analytics
│   │   ├── cognitive_workflow_engine.py # ✨ Workflow execution engine
│   │   ├── context_manager.py          # ✨ Session/user context
│   │   └── metrics_service.py          # ✨ Metrics tracking & logging
│   ├── repositories/                   # Data access
│   │   └── vector_repository.py        # Vector database operations
│   ├── config.py                       # Configuration
│   └── main.py                         # FastAPI app with workflow endpoints
├── workflows/                          # ✨ Declarative workflow definitions
│   └── property_search.yaml            # ✨ Property query workflow
├── prompts/                            # Prompt templates
│   ├── property_search/
│   │   ├── intent_analysis.txt         # Original intent analysis prompt
│   │   ├── intent_analysis_v2.txt      # ✨ Metacognitive version
│   │   └── property_response.txt
│   └── agent_analysis/
│       ├── performance_analysis.txt
│       └── insight_generation.txt
├── scripts/                            # Utility scripts
│   ├── init_database.py                # Database initialization
│   ├── populate_database.py            # Data population
│   ├── test_api.py                     # API testing
│   ├── test_workflow_api.py            # ✨ Cognitive workflow testing
│   ├── visualize_workflow.py           # ✨ Workflow visualization
│   └── evaluate_prompts.py             # ✨ Prompt evaluation framework
├── logs/                               # ✨ Application logs
│   └── metrics/                        # ✨ Workflow execution metrics
├── data/                               # Data files
├── tests/                              # Test suite
├── .claude/                            # Claude Code config
├── docker-compose.yml                  # Infrastructure
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## Cognitive Workflows

The system implements specialized cognitive workflows using a sophisticated state machine architecture.

### Week 3 Enhancement: Declarative Workflows ✨

Workflows are now defined declaratively in YAML files with formalized state transitions:

```yaml
# workflows/property_search.yaml
id: "property_query_processing"
name: "Property Query Processing Workflow"
entry_point: "intent_analysis"

states:
  intent_analysis:
    state_type: "perception"
    agent_type: "semantic_parser"
    model: "gpt-4"
    temperature: 0.3
    prompt_template: "property_search.intent_analysis"
    input_schema:
      type: "object"
      properties:
        query: {type: "string"}
      required: ["query"]
    context_requirements:
      - type: "user_preferences"
        scope: "long_term"

transitions:
  - from_state: "intent_analysis"
    to_state: "property_search"
  - from_state: "property_search"
    to_state: "result_ranking"
    condition:
      field: "properties"
      operator: "exists"
```

**Key Features:**
- **State Types**: PERCEPTION, REASONING, GENERATION, EVALUATION
- **Context Management**: Session, user, and long-term context scopes
- **Conditional Transitions**: Data-driven state routing
- **Schema Validation**: Input/output validation for each state
- **Metrics Tracking**: Comprehensive performance monitoring

### 1. Property Query Processing Workflow

```
Intent Analysis (PERCEPTION)
    ↓
Property Search (REASONING)
    ↓
Result Ranking (EVALUATION)
    ↓
Response Generation (GENERATION)
```

**States:**
1. **Intent Analysis** - Parse natural language query into structured search criteria
2. **Property Search** - Execute vector/hybrid search against database
3. **Result Ranking** - Re-rank results based on relevance
4. **Response Generation** - Create natural language response

### 2. Agent Performance Analysis

```
Agent ID → Data Collection → Metric Calculation → Comparative Analysis → Insight Generation
```

### Workflow Visualization & Debugging

```bash
# List all workflows
python scripts/visualize_workflow.py list

# Show workflow details
python scripts/visualize_workflow.py show --workflow property_query_processing

# Visualize as graph
python scripts/visualize_workflow.py graph --workflow property_query_processing

# Validate workflow definition
python scripts/visualize_workflow.py validate --workflow property_query_processing
```

## Development

### Testing Cognitive Workflows ✨ NEW in Week 3

```bash
# Test cognitive workflow API endpoints
python scripts/test_workflow_api.py

# This comprehensive test suite covers:
# - Listing available workflows
# - Executing workflows with context
# - Retrieving execution metrics
# - Session context management
# - User preference tracking
# - Multi-turn conversations
```

### Prompt Evaluation ✨ NEW in Week 3

```bash
# Evaluate intent analysis prompt
python scripts/evaluate_prompts.py evaluate --prompt intent_analysis

# Compare two prompt versions
python scripts/evaluate_prompts.py compare

# Features:
# - Test prompts against multiple test cases
# - LLM-based evaluation against criteria
# - Automated scoring and reporting
# - A/B testing between prompt versions
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_workflow.py -v
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint
flake8 app/ tests/

# Type check
mypy app/
```

### Metrics & Logging ✨ NEW in Week 3

Comprehensive metrics tracking for all workflow executions:

```bash
# View execution logs
tail -f logs/metrics/events_$(date +%Y-%m-%d).jsonl

# Metrics are automatically collected for:
# - Workflow start/completion
# - State execution times
# - Token usage per state
# - Cost per execution
# - State transitions
# - Errors and failures
```

**Log Files:**
- `logs/metrics/events_YYYY-MM-DD.jsonl` - Daily event logs (JSON Lines format)
- `logs/metrics/{execution_id}_metrics.json` - Detailed execution metrics

**Metrics Include:**
- Total execution duration
- Token count per state
- Cost per state and total
- State execution success/failure
- Reasoning traces
- Transition paths

### Database Management

```bash
# Connect to PostgreSQL
docker exec -it realtor-ai-db psql -U postgres -d realtor_ai

# Connect to vector database
docker exec -it realtor-ai-vector psql -U postgres -d vector_store

# View logs
docker-compose logs -f

# Reset databases (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

### Data Population Scripts

#### Initialize Database Schema

```bash
python scripts/init_database.py
```

Creates the properties table with pgvector extension and necessary indexes.

#### Populate Database

```bash
# Basic usage (generates 50 properties per area for default areas)
python scripts/populate_database.py

# Custom areas and count
python scripts/populate_database.py --areas 95113 94301 --count 100

# With verification
python scripts/populate_database.py --count 50 --verify

# Quiet mode
python scripts/populate_database.py --quiet
```

Generates realistic property data using LLM, creates embeddings, and stores in the vector database.

**Options:**
- `--areas`: ZIP codes to generate properties for (default: 95113, 95125, 94301, 94041, 94103)
- `--count`: Number of properties per area (default: 50)
- `--verify`: Run verification test after population
- `--quiet`: Suppress progress output

#### Test API

```bash
python scripts/test_api.py
```

Runs a comprehensive test suite against the running API:
1. Health check
2. Property search (simulated mode)
3. Property search (vector mode - if database populated)
4. Agent analysis

The script is interactive and will ask before running vector search tests.

## Claude Code Integration

This project includes custom slash commands for rapid development:

- `/setup` - Environment setup guide
- `/implement-workflow` - Create new cognitive workflows
- `/create-prompt` - Generate optimized prompts
- `/test-workflow` - Comprehensive testing
- `/optimize-prompt` - Prompt optimization
- `/review-architecture` - Architecture overview
- `/deploy` - Deployment preparation

See `.claude/README.md` for details.

## Implementation Roadmap

The project follows a progressive implementation strategy across 4 levels:

- **Level 1**: Prompt chaining foundation ✅ **COMPLETE**
  - Basic workflows implemented
  - LLM service layer
  - Prompt management system
- **Level 2**: Enhanced workflows ✅ **COMPLETE** (Week 2)
  - Real vector search integration
  - MLS data extraction service
  - Embedding generation in workflow
  - Bulk data import tools
  - Search mode selection (vector/hybrid/simulated)
- **Level 3**: Formalized cognitive transitions ✅ **COMPLETE** (Week 3)
  - Declarative YAML workflow definitions
  - Cognitive workflow engine with state machines
  - Context management (session/user/long-term)
  - Metacognitive prompt instrumentation
  - Comprehensive metrics tracking and logging
  - Prompt evaluation framework
- **Level 4**: Meta-cognitive optimization ✅ **COMPLETE** (Week 4)
  - Performance analysis and bottleneck detection
  - Adaptive workflow routing (performance/cost/quality strategies)
  - Automated prompt optimization
  - Recursive self-improvement loops
  - Cost/quality optimization balancer
  - Pareto frontier analysis

See `IMPLEMENTATION_ROADMAP.md` for the complete 8-week plan.

## Documentation

- **[PROJECT_SETUP.md](PROJECT_SETUP.md)** - Quick start guide
- **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** - 8-week implementation plan
- **[realtor-ai-prd.md](realtor-ai-prd.md)** - Product requirements
- **[realtor-ai-technical.md](realtor-ai-technical.md)** - Technical architecture
- **[realtor-ai-workflow.md](realtor-ai-workflow.md)** - Cognitive workflows
- **[progressive-implementation.md](progressive-implementation.md)** - Progressive strategy

## Key Features

### Level 4: Meta-cognitive Optimization ✨ NEW in Week 4

#### Performance Analysis Engine
- **Bottleneck detection** - Automatically identify performance bottlenecks by severity
- **Health score calculation** - Overall workflow health metric (0.0-1.0)
- **Per-state metrics** - Detailed analysis of each cognitive state
- **Impact assessment** - Quantify impact of bottlenecks on overall performance
- **Actionable recommendations** - Specific improvement suggestions with priorities

#### Adaptive Routing System
- **Strategy-based routing** - Choose optimal execution path (performance/cost/quality/balanced)
- **Model selection** - Automatically select best LLM based on task and constraints
- **Performance-driven** - Learn from historical execution data
- **Configuration variants** - Generate and evaluate multiple execution strategies
- **Confidence scoring** - Assess routing decision quality

#### Automated Prompt Optimizer
- **Meta-cognitive analysis** - AI analyzes and improves its own prompts
- **Token reduction** - Optimize for cost without sacrificing quality
- **Clarity improvements** - Enhance prompt structure and specificity
- **A/B testing** - Compare prompt versions quantitatively
- **Batch optimization** - Optimize multiple prompts efficiently

#### Recursive Self-Improvement Engine
- **Automated optimization cycles** - Continuous system improvement
- **Four-phase process** - Analysis → Optimization → Validation → Deployment
- **Dry-run mode** - Test optimizations before applying
- **Historical tracking** - Complete audit trail of improvements
- **Intelligent validation** - Safety checks before deploying changes

#### Cost/Quality Optimizer
- **Pareto frontier** - Identify optimal cost/quality tradeoffs
- **Multiple objectives** - Minimize cost, maximize quality, or balance both
- **Constraint satisfaction** - Meet cost budgets or quality thresholds
- **Efficiency scoring** - Quality per dollar metrics
- **Use-case recommendations** - Optimal configurations for different scenarios

### Level 3: Formalized Cognitive Transitions ✨ Week 3

#### Declarative Workflow Engine
- **YAML-based workflow definitions** - Define complex multi-stage workflows without code
- **State machine execution** - Formalized state transitions with conditions
- **Schema validation** - Input/output validation for each cognitive state
- **Type safety** - Full Pydantic model validation throughout
- **Workflow visualization** - Debug and visualize workflows as graphs

#### Context Management System
- **Session context** - Track conversation history per session
- **User context** - Persistent user preferences across sessions
- **Long-term memory** - Extended context for personalization
- **Automatic cleanup** - Session expiration and memory management
- **Token-aware windowing** - Intelligently manage context within token limits
- **Preference inference** - Learn from user interactions

#### Metacognitive Prompts
- **Reasoning instrumentation** - Prompts with self-awareness of reasoning process
- **Confidence assessment** - Explicit confidence levels for each extracted attribute
- **Alternative interpretations** - Consider multiple interpretations of ambiguous inputs
- **Ambiguity detection** - Identify unclear requirements
- **Clarification recommendations** - Suggest questions to resolve ambiguities
- **Reasoning traces** - Transparent decision-making process

#### Comprehensive Metrics & Logging
- **Workflow-level metrics** - Duration, cost, token usage per execution
- **State-level metrics** - Performance tracking for each cognitive state
- **Event logging** - JSON Lines format for easy analysis
- **Error tracking** - Detailed error context and recovery paths
- **Performance trends** - Track improvements over time
- **Cost optimization** - Identify expensive states for optimization

#### Prompt Evaluation Framework
- **Automated testing** - Test prompts against multiple scenarios
- **LLM-based evaluation** - Intelligent assessment against criteria
- **A/B testing** - Compare prompt versions quantitatively
- **Regression detection** - Ensure prompt changes don't degrade performance
- **Scoring system** - Objective quality metrics

### Natural Language Understanding
- Semantic query parsing
- Intent classification
- Implicit preference inference
- Context-aware search
- Metacognitive reasoning traces ✨

### Vector Search ✅ (Implemented in Week 2)
- **Property embedding generation** - Convert properties to semantic vectors
- **Semantic similarity search** - Find properties by meaning, not just keywords
- **Hybrid search** - Combine vector search with structured filters
- **HNSW indexing** - Fast approximate nearest neighbor search
- **Multiple search modes** - Vector, hybrid, or simulated
- **Batch processing** - Efficient embedding generation for large datasets

### Agent Analytics
- Performance metric calculation
- Market benchmarking
- Competitive analysis
- Strategic insight generation

### Multi-Modal Perception (Planned)
- Visual property analysis
- Image feature extraction
- Multi-modal embeddings
- Data fusion pipeline

## Performance

- API response time: <2s (p95)
- Vector search: <100ms for 10k properties
- Cost per query: <$0.10 (optimized)
- Concurrent requests: 100+

## Security

- API key management via environment variables
- Input validation on all endpoints
- Rate limiting (production)
- CORS configuration
- No sensitive data in logs

## Deployment

See `/deploy` slash command or `IMPLEMENTATION_ROADMAP.md` for deployment guidance.

### Production Checklist
- [ ] Configure production environment variables
- [ ] Set up monitoring and logging
- [ ] Configure rate limiting
- [ ] Set appropriate CORS origins
- [ ] Database backups configured
- [ ] SSL/TLS certificates
- [ ] Health check endpoints tested

## Contributing

This is a private project for David Shapiro's real estate business. External contributions are not accepted.

## License

Proprietary - All rights reserved.

## Contact

For questions or support, contact the development team.

---

**Built with Claude Code** | [Documentation](PROJECT_SETUP.md) | [Roadmap](IMPLEMENTATION_ROADMAP.md)
