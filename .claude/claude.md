# REALTOR AI COPILOT Project Context

## Project Overview

This is a **multi-modal AI assistant specifically tailored for real estate agent augmentation**, with particular focus on providing competitive intelligence while avoiding the litigation pitfalls that affected similar platforms. The system is designed for David Shapiro (Realtor) to dramatically enhance his productivity and competitive advantage.

## System Architecture

### Meta-Architecture Philosophy

The system employs a **layered intelligence model** with three core subsystems:

```
┌───────────────────────────────────────────────────────────────┐
│                    COGNITIVE META-SYSTEM                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐  │
│  │ Perception  │   │  Reasoning  │   │ Generation/Response │  │
│  │ Subsystem   │◄─►│  Subsystem  │◄─►│     Subsystem       │  │
│  └─────────────┘   └─────────────┘   └─────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
         ▲                   ▲                     ▲
┌────────┼───────────────────┼─────────────────────┼──────────────┐
│        │                   │                     │              │
│  ┌─────▼──────┐  ┌────────▼─────┐  ┌──────▼─────────────────┐ │
│  │ Embedding  │  │  Knowledge   │  │    Prompt Layer        │ │
│  │   Layer    │  │     Base     │  │                        │ │
│  └────────────┘  └──────────────┘  └────────────────────────┘ │
│                     FOUNDATIONAL LAYER                         │
└────────────────────────────────────────────────────────────────┘
```

### Core Capabilities

1. **Natural Language Property Search**: Semantic search that transcends rigid MLS parameter-based queries
2. **Agent Performance Analytics**: Competitive intelligence dashboard tracking key agent metrics
3. **Personalized AI Assistant**: System becomes increasingly attuned to David's specific preferences and expertise patterns
4. **Multi-Modal Perception**: Combines visual, textual, and structured data fusion

## Technology Stack

- **Backend**: FastAPI (Python) with Uvicorn ASGI server
- **Database**: PostgreSQL with pgvector extension for vector search
- **Embeddings**: OpenAI Ada-002 (text), CLIP (images)
- **LLM**: GPT-4 (reasoning), GPT-3.5 (high-throughput), Claude (alternative)
- **Frontend**: Streamlit (initial prototype), React (production)
- **Infrastructure**: Docker containers, AWS Fargate (production)
- **Orchestration**: Custom cognitive workflow engine

## Project Structure

```
realtor-ai/
├── app/                           # Main application code
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry
│   ├── config.py                  # Configuration management
│   ├── models/                    # Pydantic models
│   ├── api/                       # API endpoints
│   ├── services/                  # Business logic
│   │   ├── llm_service.py         # LLM interaction
│   │   ├── embedding_service.py   # Embedding generation
│   │   ├── workflow_service.py    # Cognitive workflows
│   │   ├── prompt_service.py      # Prompt template management
│   │   └── visual_perception_service.py
│   └── repositories/              # Data access layer
│       └── vector_repository.py   # Vector store operations
├── prompts/                       # Prompt templates
│   ├── property_search/
│   ├── agent_analysis/
│   └── content_generation/
├── data/                          # Sample and seed data
├── tests/                         # Test suite
└── docker-compose.yml             # Local development environment
```

## Key Domain Models

### Property Entity
```typescript
interface Property {
  id: string;
  mlsId?: string;
  embeddings: {
    description: Vector;
    visualFeatures: Vector[];
    amenities: Vector;
  };
  structured: PropertyAttributes;
  unstructured: { description, remarks };
  media: MediaAsset[];
  agentId: string;
  history: TransactionEvent[];
}
```

### Agent Entity
```typescript
interface Agent {
  id: string;
  name: string;
  metrics: {
    closingRate: number;
    avgDaysOnMarket: number;
    priceToListRatio: number;
    geographicSpecialization: GeoDistribution;
  };
  embeddings: { marketingStyle, propertyPreferences };
  historicalTransactions: Transaction[];
}
```

## Cognitive Workflow Architecture

The system uses a **declarative workflow specification** approach with formal state transitions:

### Core Workflows
1. **Property Query Processing**: Natural language → Intent analysis → Search execution → Result ranking → Response generation
2. **Agent Performance Analysis**: Data collection → Metric calculation → Comparative analysis → Insight generation
3. **Content Generation**: Context gathering → Style analysis → Draft generation → Refinement
4. **Feedback Learning**: Interaction logging → Pattern detection → Prompt optimization

### Progressive Implementation Levels

The architecture supports **4 levels of progressive enrichment**:

1. **Level 1**: Prompt chaining foundation (basic sequential LLM calls)
2. **Level 2**: Decomposed cognitive functions (specialized prompt templates)
3. **Level 3**: Formalized cognitive transitions (schema-validated state management)
4. **Level 4**: Meta-cognitive optimization (self-improving prompt structures)

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
- MLS data ingestion and extraction
- Basic natural language query capability
- Initial agent database construction
- Core vector search implementation

### Phase 2: Intelligence (Weeks 3-4)
- Agent performance metric calculation
- Competitive analysis dashboard
- Enhanced property matching algorithms
- Multi-modal perception system

### Phase 3: Personalization (Weeks 5-6)
- Training on David's historical transactions
- Client preference learning mechanisms
- Feedback loop implementation
- Meta-cognitive optimization

## Development Guidelines

### Prompt Engineering Principles
- Use structured prompts with clear cognitive function definitions
- Include explicit reasoning instructions
- Validate outputs with schema enforcement
- Implement confidence assessment mechanisms

### Cognitive Workflow Design
- Decompose tasks by cognitive function (perception, reasoning, generation)
- Design prompt cascades with semantic dependencies
- Reify implicit cognitive states as structured intermediate forms
- Manage multi-scale context propagation

### Testing Strategy
- Develop test cases for each cognitive function
- Implement automated prompt evaluation harness
- Create regression test suite for workflows
- Enable A/B testing of prompt variations

## Key Considerations

### Legal & Ethical
- **MLS Compliance**: Hybrid approach to data access respecting terms of service
- **Private Deployment**: System is for David's private use to avoid litigation risks
- **Competitive Intelligence**: Ethical use of public agent performance data
- **Privacy**: Compliance with real estate regulations

### Performance
- **Cost Optimization**: Tiered model approach (GPT-4 for reasoning, GPT-3.5 for generation)
- **Caching Strategy**: Common queries cached to reduce API costs
- **Token Tracking**: Monitor and optimize token usage across workflows
- **Rate Limiting**: Respect API limits and implement exponential backoff

### Scalability
- **Vector Indexing**: HNSW index for efficient similarity search
- **Batch Processing**: Async operations for data ingestion
- **Worker Separation**: Background workers for long-running tasks
- **Database Optimization**: Proper indexing and query optimization

## Success Metrics

- **40%+ reduction** in research time
- **25%+ increase** in relevant property matches
- **Measurable competitive advantage** in specific neighborhoods
- **Client satisfaction** with property recommendations

## Current Status

**Architecture Phase**: All specifications complete, ready for implementation
**Git Branch**: `claude/real-estate-ai-architecture-011CULqGzNrDqSSM1CKGXM2s`
**Documentation**: 233 KB of comprehensive technical specifications across 11 markdown files

## Quick Reference

### Starting Development
```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d db pgvector

# Run API server
uvicorn app.main:app --reload
```

### Key API Endpoints
- `POST /api/property-search` - Natural language property search
- `POST /api/agent-analysis` - Agent performance analysis
- `GET /health` - Health check endpoint

### Environment Variables
- `LLM_OPENAI_API_KEY` - OpenAI API key (required)
- `LLM_ANTHROPIC_API_KEY` - Anthropic API key (optional)
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_DATABASE` - PostgreSQL config
- `VECTORDB_*` - Vector database configuration

## Related Documentation

- `realtor-ai-prd.md` - Product Requirements Document
- `realtor-ai-technical.md` - Technical Architecture Overview
- `realtor-ai-detailed-technical.md` - Advanced Technical Specification
- `realtor-ai-workflow.md` - Cognitive Workflow Architecture
- `tactical-implementation-plan.md` - Day-by-day Implementation Guide
- `progressive-implementation.md` - Progressive Enhancement Strategy
- `metacognitive-implementation.md` - Meta-cognitive Feedback Loop Spec
- `multi-modal-perception.md` - Multi-modal Perception System
- `cognitive-workflow-prompts.md` - Specialized Prompt Templates

## Development Philosophy

This project embodies the principle of **"conserving conceptual momentum"** - balancing architectural sophistication (potential energy) with implementation velocity (kinetic energy). The cognitive workflow paradigm enables **gradient-based implementation**, allowing the team to ascend the architectural complexity gradient adaptively rather than requiring full realization from inception.

Start simple, iterate rapidly, and progressively enhance cognitive sophistication.
