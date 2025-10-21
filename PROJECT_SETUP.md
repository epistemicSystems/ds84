# REALTOR AI COPILOT: Quick Start Guide

## Overview

This guide will help you set up the REALTOR AI COPILOT development environment and get your first workflow running in under 30 minutes.

## Prerequisites

Before starting, ensure you have:

- **Python 3.10 or higher** installed
- **Docker Desktop** installed and running
- **Git** installed
- An **OpenAI API key** (required)
- An **Anthropic API key** (optional, for fallback)
- At least **8GB of free RAM** for Docker containers
- **10GB of free disk space**

## Quick Start (5 minutes to first API call)

### 1. Clone Repository and Create Environment

```bash
# Navigate to project directory
cd realtor-ai

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify Python version (should be 3.10+)
python --version
```

### 2. Create Requirements File

Create `requirements.txt` with the following content:

```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0

# Data Models
pydantic==2.5.0
pydantic-settings==2.1.0

# HTTP Client
httpx==0.25.2

# Template Engine
jinja2==3.1.2

# Database
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Vector Search
pgvector==0.2.4

# LLM APIs
openai==1.3.8
anthropic==0.7.8

# Image Processing
Pillow==10.1.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Development
black==23.11.0
flake8==6.1.0
mypy==1.7.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Create Environment Configuration

Create `.env` file in project root:

```bash
# LLM API Configuration
LLM_OPENAI_API_KEY=sk-your-openai-key-here
LLM_ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
LLM_DEFAULT_MODEL=gpt-4
LLM_TIMEOUT=30

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_DATABASE=realtor_ai

# Vector Database Configuration
VECTORDB_HOST=localhost
VECTORDB_PORT=5433
VECTORDB_USER=postgres
VECTORDB_PASSWORD=postgres
VECTORDB_DATABASE=vector_store

# Application Configuration
APP_NAME=Realtor AI Copilot
DEBUG=true
ENVIRONMENT=development
```

**Important**: Replace `sk-your-openai-key-here` with your actual OpenAI API key.

### 4. Start Infrastructure with Docker

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  db:
    image: postgres:14
    container_name: realtor-ai-db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: realtor_ai
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  pgvector:
    image: ankane/pgvector:latest
    container_name: realtor-ai-vector
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vector_store
    ports:
      - "5433:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  pgvector_data:
```

Start the databases:

```bash
docker-compose up -d

# Verify containers are running
docker-compose ps

# Check logs if needed
docker-compose logs -f
```

### 5. Create Project Structure

```bash
# Create directory structure
mkdir -p app/{api,models,services,repositories}
mkdir -p prompts/{property_search,agent_analysis,content_generation}
mkdir -p data
mkdir -p tests

# Create __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/repositories/__init__.py
touch tests/__init__.py
```

### 6. Create Basic Application Files

**Create `app/config.py`:**

```python
import os
from pydantic_settings import BaseSettings
from typing import Optional

class LLMConfig(BaseSettings):
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    timeout: int = 30

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"

class DatabaseConfig(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "realtor_ai"

    class Config:
        env_file = ".env"
        env_prefix = "DB_"

class VectorDBConfig(BaseSettings):
    host: str = "localhost"
    port: int = 5433
    user: str = "postgres"
    password: str = "postgres"
    database: str = "vector_store"

    class Config:
        env_file = ".env"
        env_prefix = "VECTORDB_"

class Settings(BaseSettings):
    app_name: str = "Realtor AI Copilot"
    debug: bool = True
    environment: str = "development"
    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    vector_db: VectorDBConfig = VectorDBConfig()

    class Config:
        env_file = ".env"

settings = Settings()
```

**Create `app/main.py`:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Realtor AI Copilot API",
    description="Multi-modal AI assistant for real estate agent augmentation",
    version="0.1.0"
)

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "0.1.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Realtor AI Copilot API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 7. Test Your Setup

Start the API server:

```bash
uvicorn app.main:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using watchgod
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Test the API:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","version":"0.1.0"}

# Or visit in browser:
# http://localhost:8000/docs (Swagger UI)
```

**🎉 Congratulations! Your basic setup is complete!**

---

## Next Steps: Implementing Your First Workflow

Now that your environment is set up, follow these steps to implement the property search workflow:

### Step 1: Create LLM Service

Create `app/services/llm_service.py`:

```python
import httpx
from typing import List
from app.config import settings

class LLMService:
    def __init__(self):
        self.openai_api_key = settings.llm.openai_api_key
        self.timeout = settings.llm.timeout

    async def complete(self,
                      prompt: str,
                      model: str = "gpt-4",
                      temperature: float = 0.7,
                      max_tokens: int = 1000) -> str:
        """Complete a prompt using OpenAI API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"]

llm_service = LLMService()
```

### Step 2: Create First Prompt Template

Create `prompts/property_search/intent_analysis.txt`:

```
You are performing semantic parsing of natural language property queries.
Your role is to transform ambiguous human requests into structured search criteria.

User query: "{{ query }}"

Extract the following information:
1. Property types (house, condo, apartment, etc.)
2. Bedrooms and bathrooms (min, max, preferred)
3. Location preferences (neighborhoods, proximity features)
4. Price range (min, max)
5. Must-have features
6. Nice-to-have features
7. Style preferences
8. Implied preferences from descriptive language

Return your analysis as a JSON object with this structure:
{
  "property_types": ["type1", "type2"],
  "bedrooms": {"min": X, "max": Y, "preferred": Z},
  "bathrooms": {"min": X, "preferred": Y},
  "location": {
    "neighborhoods": ["area1", "area2"],
    "proximity_features": ["feature1", "feature2"]
  },
  "price_range": {"min": X, "max": Y},
  "must_have_features": ["feature1", "feature2"],
  "nice_to_have_features": ["feature1", "feature2"],
  "style_preferences": ["style1", "style2"]
}
```

### Step 3: Create Prompt Service

Create `app/services/prompt_service.py`:

```python
from pathlib import Path
from jinja2 import Template
from typing import Dict

class PromptService:
    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = Path(templates_dir)
        self.templates = {}

    def get_prompt(self, template_id: str, **kwargs) -> str:
        """Get a formatted prompt template"""
        # Load template if not cached
        if template_id not in self.templates:
            template_path = self.templates_dir / f"{template_id.replace('.', '/')}.txt"
            with open(template_path, "r") as f:
                self.templates[template_id] = Template(f.read())

        # Render template with kwargs
        return self.templates[template_id].render(**kwargs)

prompt_service = PromptService()
```

### Step 4: Test Your First LLM Call

Create `test_llm.py` in project root:

```python
import asyncio
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service

async def test_property_search():
    # Create prompt from template
    query = "I need a modern 3-bedroom home with a view of the water"
    prompt = prompt_service.get_prompt("property_search.intent_analysis", query=query)

    # Call LLM
    print("Calling LLM...")
    response = await llm_service.complete(prompt, temperature=0.3)

    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    asyncio.run(test_property_search())
```

Run the test:

```bash
python test_llm.py
```

You should see the LLM's structured analysis of the query!

---

## Development Workflow

### Running the API Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_workflow.py -v
```

### Code Quality

```bash
# Format code with Black
black app/ tests/

# Lint with flake8
flake8 app/ tests/

# Type check with mypy
mypy app/
```

### Database Management

```bash
# Connect to PostgreSQL
docker exec -it realtor-ai-db psql -U postgres -d realtor_ai

# Connect to pgvector
docker exec -it realtor-ai-vector psql -U postgres -d vector_store

# Stop databases
docker-compose stop

# Start databases
docker-compose start

# View logs
docker-compose logs -f db
docker-compose logs -f pgvector

# Reset databases (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

---

## Troubleshooting

### Issue: "Cannot connect to database"

**Solution**:
```bash
# Check if Docker containers are running
docker-compose ps

# Restart containers
docker-compose restart

# Check logs for errors
docker-compose logs
```

### Issue: "OpenAI API error: 401"

**Solution**: Verify your API key in `.env` file is correct and has sufficient credits.

### Issue: "Module not found" errors

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Port already in use

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
# or
netstat -ano | findstr :8000  # Windows

# Kill the process or use a different port
uvicorn app.main:app --reload --port 8001
```

---

## Project Configuration Files

### .gitignore

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Database
*.db
*.sqlite3

# Docker
*.log

# OS
.DS_Store
Thumbs.db
```

---

## Useful Claude Code Commands

Once your project is set up, you can use these Claude Code slash commands:

- `/setup` - Guide through environment setup
- `/implement-workflow` - Implement a new cognitive workflow
- `/create-prompt` - Create a new prompt template
- `/test-workflow` - Test a workflow implementation
- `/optimize-prompt` - Optimize a prompt for better performance
- `/review-architecture` - Review the system architecture
- `/deploy` - Prepare for deployment

---

## Next Implementation Steps

Follow the **IMPLEMENTATION_ROADMAP.md** for detailed day-by-day implementation guidance:

1. **Day 1**: Complete infrastructure setup ✓ (You just did this!)
2. **Day 2**: Implement core services layer
3. **Day 3**: Build property search workflow
4. **Day 4-5**: Implement vector storage and search
5. **Continue following the roadmap...**

---

## Resources

- **Project Documentation**: See markdown files in project root
- **API Documentation**: http://localhost:8000/docs (when server is running)
- **OpenAI API Docs**: https://platform.openai.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **pgvector Docs**: https://github.com/pgvector/pgvector

---

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the detailed technical specs in the markdown files
3. Use Claude Code's `/setup` command for guided assistance
4. Check Docker container logs: `docker-compose logs`
5. Verify environment variables in `.env` file

---

## Success Checklist

- [ ] Python 3.10+ installed and verified
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] .env file created with API keys
- [ ] Docker containers running (db and pgvector)
- [ ] FastAPI server starts successfully
- [ ] Health endpoint returns 200 OK
- [ ] First LLM call works in test script
- [ ] API documentation accessible at /docs

**Once all items are checked, you're ready to start implementing workflows!** 🚀
