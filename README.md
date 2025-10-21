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

6. **Run the API server**:
```bash
uvicorn app.main:app --reload
```

7. **Visit API documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Property Search

**POST** `/api/property-search`

Search for properties using natural language queries.

```bash
curl -X POST http://localhost:8000/api/property-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need a modern 3-bedroom home with a view of the water, ideally with an open floor plan and within walking distance to restaurants. My budget is around $750,000.",
    "user_id": "user123"
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
│   ├── api/                   # API endpoints
│   ├── models/                # Pydantic models
│   ├── services/              # Business logic
│   │   ├── llm_service.py     # LLM interactions
│   │   ├── prompt_service.py  # Prompt management
│   │   ├── workflow_service.py
│   │   ├── embedding_service.py
│   │   └── agent_analysis_workflow.py
│   ├── repositories/          # Data access
│   │   └── vector_repository.py
│   ├── config.py              # Configuration
│   └── main.py                # FastAPI app
├── prompts/                   # Prompt templates
│   ├── property_search/
│   │   ├── intent_analysis.txt
│   │   └── property_response.txt
│   └── agent_analysis/
│       ├── performance_analysis.txt
│       └── insight_generation.txt
├── data/                      # Data files
├── tests/                     # Test suite
├── .claude/                   # Claude Code config
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Cognitive Workflows

The system implements specialized cognitive workflows:

### 1. Property Query Processing
```
User Query → Intent Analysis → Property Search → Result Ranking → Response Generation
```

### 2. Agent Performance Analysis
```
Agent ID → Data Collection → Metric Calculation → Comparative Analysis → Insight Generation
```

## Development

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

### Database Management

```bash
# Connect to PostgreSQL
docker exec -it realtor-ai-db psql -U postgres -d realtor_ai

# View logs
docker-compose logs -f

# Reset databases (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

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

- **Level 1**: Prompt chaining foundation ✅ **(Current)**
- **Level 2**: Decomposed cognitive functions
- **Level 3**: Formalized cognitive transitions
- **Level 4**: Meta-cognitive optimization

See `IMPLEMENTATION_ROADMAP.md` for the complete 8-week plan.

## Documentation

- **[PROJECT_SETUP.md](PROJECT_SETUP.md)** - Quick start guide
- **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** - 8-week implementation plan
- **[realtor-ai-prd.md](realtor-ai-prd.md)** - Product requirements
- **[realtor-ai-technical.md](realtor-ai-technical.md)** - Technical architecture
- **[realtor-ai-workflow.md](realtor-ai-workflow.md)** - Cognitive workflows
- **[progressive-implementation.md](progressive-implementation.md)** - Progressive strategy

## Key Features

### Natural Language Understanding
- Semantic query parsing
- Intent classification
- Implicit preference inference
- Context-aware search

### Vector Search
- Property embedding generation
- Semantic similarity search
- Hybrid search (vector + filters)
- HNSW indexing for performance

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
