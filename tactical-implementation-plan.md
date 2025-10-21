# REALTOR AI COPILOT: Tactical Implementation Plan

## DAY 1: FOUNDATION INFRASTRUCTURE

### Core Environment Setup

```bash
# Project initialization
mkdir -p realtor-ai/{app,prompts,data,tests}
cd realtor-ai

# Environment setup
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pydantic python-dotenv httpx pgvector psycopg2-binary openai anthropic pytest

# Docker infrastructure
cat > docker-compose.yml << EOF
version: '3.8'
services:
  db:
    image: postgres:14
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: realtor_ai
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  pgvector:
    image: ankane/pgvector
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vector_store
    ports:
      - "5433:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  pgvector_data:
EOF

# Start infrastructure
docker-compose up -d
```

### Project Structure

```
realtor-ai/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry
│   ├── config.py              # Configuration management
│   ├── models/                # Pydantic models
│   ├── api/                   # API endpoints
│   ├── services/              # Business logic
│   │   ├── llm_service.py     # LLM interaction
│   │   ├── embedding_service.py
│   │   └── workflow_service.py
│   └── repositories/          # Data access
├── prompts/                   # Prompt templates
│   ├── property_search/
│   ├── agent_analysis/
│   └── content_generation/
├── data/                      # Sample and seed data
└── tests/                     # Test suite
```

### Configuration Management

```python
# app/config.py
import os
from pydantic import BaseSettings
from typing import Dict, List, Optional

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

### LLM Service Implementation

```python
# app/services/llm_service.py
import httpx
import json
import asyncio
from typing import Dict, Any, Optional, List
from app.config import settings

class LLMService:
    def __init__(self):
        self.openai_api_key = settings.llm.openai_api_key
        self.anthropic_api_key = settings.llm.anthropic_api_key
        self.timeout = settings.llm.timeout
        
    async def complete(self, 
                      prompt: str, 
                      model: str = "gpt-4",
                      temperature: float = 0.7,
                      max_tokens: int = 1000,
                      provider: str = "openai") -> str:
        """Complete a prompt using specified LLM provider"""
        if provider == "openai":
            return await self._complete_openai(prompt, model, temperature, max_tokens)
        elif provider == "anthropic":
            return await self._complete_anthropic(prompt, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _complete_openai(self, 
                             prompt: str,
                             model: str,
                             temperature: float,
                             max_tokens: int) -> str:
        """Complete using OpenAI API"""
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
    
    async def _complete_anthropic(self,
                                prompt: str,
                                model: str,
                                temperature: float,
                                max_tokens: int) -> str:
        """Complete using Anthropic API"""
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
            
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.text}")
                
            result = response.json()
            return result["content"][0]["text"]
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """Generate embeddings using OpenAI"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": model
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")
                
            result = response.json()
            return result["data"][0]["embedding"]

llm_service = LLMService()
```

## DAY 2: PROPERTY SEARCH WORKFLOW - CORE COMPONENTS

### Prompt Templates

```python
# app/services/prompt_service.py
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from jinja2 import Template

class PromptService:
    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = Path(templates_dir)
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load all prompt templates from directory structure"""
        templates = {}
        
        for template_file in self.templates_dir.glob("**/*.txt"):
            relative_path = template_file.relative_to(self.templates_dir)
            template_id = str(relative_path).replace("/", ".").replace(".txt", "")
            
            with open(template_file, "r") as f:
                template_content = f.read()
                
            templates[template_id] = template_content
            
        return templates
    
    def get_prompt(self, template_id: str, **kwargs) -> str:
        """Get a formatted prompt template"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
            
        template = Template(self.templates[template_id])
        return template.render(**kwargs)
    
    def add_template(self, template_id: str, template_content: str) -> None:
        """Add a new template programmatically"""
        self.templates[template_id] = template_content
        
        # Save to file system
        template_path = self.templates_dir / f"{template_id.replace('.', '/')}.txt"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_path, "w") as f:
            f.write(template_content)

prompt_service = PromptService()

# Initialize with core templates
os.makedirs("prompts/property_search", exist_ok=True)

with open("prompts/property_search/intent_analysis.txt", "w") as f:
    f.write("""
<cognitive_function>
You are performing semantic parsing of natural language property queries.
Your role is to transform ambiguous human requests into structured search criteria.
</cognitive_function>

<input>
User query: "{{ query }}"
{% if context %}
Context: {{ context }}
{% endif %}
</input>

<reasoning>
First, identify explicit property attributes mentioned in the query:
- Property types (house, condo, apartment, etc.)
- Numeric specifications (bedrooms, bathrooms, square footage)
- Location preferences (neighborhoods, proximity features)
- Price range indicators (explicit or implied)
- Condition descriptors (renovated, new construction, etc.)
- Amenity requirements (pool, garage, view, etc.)
- Style preferences (modern, traditional, craftsman, etc.)

Then, infer implicit preferences from descriptive language:
- Emotional terms suggesting lifestyle preferences
- Adjectives indicating quality expectations
- Temporal constraints (urgency, timeline)
- Trade-off priorities between competing factors
</reasoning>

<output_format>
Provide a JSON object with the following structure:
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
  "style_preferences": ["style1", "style2"],
  "implied_preferences": ["pref1", "pref2"]
}
</output_format>
""")

with open("prompts/property_search/property_response.txt", "w") as f:
    f.write("""
<cognitive_function>
You are a real estate assistant specializing in property descriptions and recommendations.
Your role is to present search results in a compelling, informative way that highlights relevant features.
</cognitive_function>

<input>
Original query: "{{ query }}"
{% if search_intent %}
Parsed intent: {{ search_intent }}
{% endif %}
Properties found: {{ properties | tojson }}
</input>

<reasoning>
For each property:
1. Identify how well it matches the user's explicit criteria
2. Highlight special features that align with implied preferences
3. Note any potential compromises or trade-offs
4. Consider the property's unique selling points

Organize properties by relevance to the user's needs, not just by price or size.
</reasoning>

<output_format>
Provide a conversational response that:
1. Acknowledges the user's search criteria
2. Presents 3-5 top matching properties with relevant details
3. Highlights special features aligned with preferences
4. Offers follow-up questions to refine the search if needed

Use natural, engaging language as a real estate professional would.
</output_format>
""")
```

### Basic Workflow Implementation

```python
# app/services/workflow_service.py
import json
from typing import Dict, List, Any, Optional
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service

class PropertySearchWorkflow:
    def __init__(self):
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        
    async def execute(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Execute the property search workflow"""
        try:
            # Step 1: Analyze user intent
            intent = await self._analyze_intent(query)
            
            # Step 2: Search for properties
            # In prototype, we'll use LLM to simulate property search
            properties = await self._simulate_property_search(intent)
            
            # Step 3: Generate response
            response = await self._generate_response(query, intent, properties)
            
            return {
                "query": query,
                "intent": intent,
                "properties": properties,
                "response": response
            }
        except Exception as e:
            print(f"Workflow error: {e}")
            return {
                "query": query,
                "error": str(e)
            }
    
    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user intent from natural language query"""
        intent_prompt = self.prompt_service.get_prompt(
            "property_search.intent_analysis",
            query=query
        )
        
        intent_response = await self.llm_service.complete(
            prompt=intent_prompt,
            temperature=0.3  # Lower temperature for more consistent parsing
        )
        
        # Extract JSON from response
        try:
            # Find JSON block in response
            json_start = intent_response.find('{')
            json_end = intent_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = intent_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in intent analysis response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse intent analysis response as JSON")
    
    async def _simulate_property_search(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate property search using LLM"""
        search_prompt = f"""
        You are a real estate database. Based on the following search criteria:
        {json.dumps(intent, indent=2)}
        
        Generate 5 realistic property listings that would match these criteria.
        
        Each property should include:
        - Address
        - Price
        - Bedrooms and bathrooms
        - Square footage
        - Year built
        - Property type
        - Key features and amenities
        - A brief description
        
        Return the results as a JSON array of property objects.
        """
        
        search_response = await self.llm_service.complete(
            prompt=search_prompt,
            max_tokens=2000
        )
        
        # Extract JSON from response
        try:
            # Find JSON array in response
            json_start = search_response.find('[')
            json_end = search_response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = search_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON array found in search response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse search response as JSON")
    
    async def _generate_response(self, 
                                query: str, 
                                intent: Dict[str, Any], 
                                properties: List[Dict[str, Any]]) -> str:
        """Generate natural language response for search results"""
        response_prompt = self.prompt_service.get_prompt(
            "property_search.property_response",
            query=query,
            search_intent=json.dumps(intent),
            properties=properties
        )
        
        response = await self.llm_service.complete(
            prompt=response_prompt,
            temperature=0.7,  # Higher temperature for more creative response
            max_tokens=1500
        )
        
        return response

workflow_service = PropertySearchWorkflow()
```

### API Implementation

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from app.services.workflow_service import workflow_service

app = FastAPI(title="Realtor AI Copilot API")

class PropertySearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class PropertySearchResponse(BaseModel):
    query: str
    intent: Optional[Dict[str, Any]] = None
    properties: Optional[List[Dict[str, Any]]] = None
    response: Optional[str] = None
    error: Optional[str] = None

@app.post("/api/property-search", response_model=PropertySearchResponse)
async def property_search(request: PropertySearchRequest):
    """Execute property search workflow from natural language query"""
    result = await workflow_service.execute(request.query, request.user_id)
    return result

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## DAY 3: DATA EXTRACTION AND VECTOR STORAGE

### MLS Data Extraction Service

```python
# app/services/mls_extraction_service.py
import httpx
import asyncio
import re
import json
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from app.config import settings
from app.services.llm_service import llm_service

class MLSExtractionService:
    def __init__(self):
        self.base_url = "https://example-mls-api.com"  # Replace with actual MLS API
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
    async def extract_listings(self, 
                              area_code: str, 
                              limit: int = 100,
                              days_on_market: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract property listings from MLS API or website"""
        # In prototype, we'll use sample data
        # In production, implement actual MLS API integration
        return await self._generate_sample_listings(area_code, limit)
    
    async def _generate_sample_listings(self, 
                                      area_code: str, 
                                      limit: int) -> List[Dict[str, Any]]:
        """Generate sample listings using LLM for prototype"""
        prompt = f"""
        Generate {limit} realistic MLS property listings for area code {area_code}.
        
        Each listing should include:
        - MLS ID (format: AA12345678)
        - Address (street, city, state, zip)
        - List price
        - Bedrooms and bathrooms
        - Square footage
        - Lot size
        - Year built
        - Property type
        - Description (2-3 sentences)
        - Features list (5-10 items)
        - Days on market
        - Listing agent information
        
        Return the data as a JSON array of property objects.
        """
        
        response = await llm_service.complete(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.7
        )
        
        # Extract JSON from response
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON array found in generated listings")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse generated listings as JSON")
    
    async def extract_property_details(self, mls_id: str) -> Dict[str, Any]:
        """Extract detailed property information for a specific MLS ID"""
        # In prototype, generate a detailed listing
        prompt = f"""
        Generate detailed property information for MLS ID {mls_id}.
        
        Include:
        - All basic listing information
        - Detailed room dimensions
        - Complete features list (15-20 items)
        - Tax history (last 3 years)
        - Sales history (up to 3 previous sales)
        - School information
        - Walkability scores
        - Neighborhood data
        
        Return as a single JSON object with nested structure.
        """
        
        response = await llm_service.complete(
            prompt=prompt,
            max_tokens=3000,
            temperature=0.5
        )
        
        # Extract JSON from response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in property details")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse property details as JSON")

mls_service = MLSExtractionService()
```

### Vector Storage Implementation

```python
# app/repositories/vector_repository.py
import asyncio
import asyncpg
import json
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings

class VectorRepository:
    def __init__(self):
        self.config = settings.vector_db
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection pool and schema"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database
            )
            
            # Ensure pgvector extension is installed
            async with self.pool.acquire() as conn:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
                
                # Create properties table if not exists
                await conn.execute('''
                CREATE TABLE IF NOT EXISTS properties (
                    id SERIAL PRIMARY KEY,
                    mls_id TEXT UNIQUE NOT NULL,
                    data JSONB NOT NULL,
                    embedding VECTOR(1536),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
                ''')
                
                # Create index for vector similarity search
                await conn.execute('''
                CREATE INDEX IF NOT EXISTS properties_embedding_idx 
                ON properties USING ivfflat (embedding vector_cosine_ops)
                ''')
    
    async def store_property(self, 
                           mls_id: str, 
                           data: Dict[str, Any], 
                           embedding: List[float] = None) -> int:
        """Store property data and embedding"""
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            if embedding:
                result = await conn.fetchval('''
                INSERT INTO properties (mls_id, data, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (mls_id) 
                DO UPDATE SET data = $2, embedding = $3, updated_at = NOW()
                RETURNING id
                ''', mls_id, json.dumps(data), embedding)
            else:
                result = await conn.fetchval('''
                INSERT INTO properties (mls_id, data)
                VALUES ($1, $2)
                ON CONFLICT (mls_id) 
                DO UPDATE SET data = $2, updated_at = NOW()
                RETURNING id
                ''', mls_id, json.dumps(data))
                
            return result
    
    async def search_by_vector(self, 
                             query_vector: List[float], 
                             limit: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Search properties by vector similarity"""
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch('''
            SELECT data, 1 - (embedding <=> $1) AS similarity
            FROM properties
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            ''', query_vector, limit)
            
            return [(json.loads(r['data']), r['similarity']) for r in results]
    
    async def search_by_filters(self,
                              filters: Dict[str, Any],
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Search properties by JSONB filters"""
        await self.initialize()
        
        # Build dynamic query conditions
        conditions = []
        params = []
        param_idx = 1
        
        for key, value in filters.items():
            path = key.replace('.', '->')
            conditions.append(f"data->>{path} = ${param_idx}")
            params.append(str(value))
            param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        async with self.pool.acquire() as conn:
            query = f'''
            SELECT data
            FROM properties
            WHERE {where_clause}
            LIMIT ${param_idx}
            '''
            
            results = await conn.fetch(query, *params, limit)
            return [json.loads(r['data']) for r in results]
    
    async def hybrid_search(self,
                          query_vector: List[float],
                          filters: Dict[str, Any] = None,
                          limit: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Combined vector and filter search"""
        await self.initialize()
        
        # Build dynamic query conditions for filters
        conditions = []
        params = [query_vector, limit]
        param_idx = 3
        
        if filters:
            for key, value in filters.items():
                path = key.replace('.', '->')
                conditions.append(f"data->>{path} = ${param_idx}")
                params.append(str(value))
                param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        async with self.pool.acquire() as conn:
            query = f'''
            SELECT data, 1 - (embedding <=> $1) AS similarity
            FROM properties
            WHERE embedding IS NOT NULL AND {where_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
            '''
            
            results = await conn.fetch(query, *params)
            return [(json.loads(r['data']), r['similarity']) for r in results]

vector_repository = VectorRepository()
```

## DAY 4-5: AGENT ANALYSIS WORKFLOW

### Agent Data Extraction

```python
# app/services/agent_extraction_service.py
import httpx
import asyncio
import re
import json
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from app.config import settings
from app.services.llm_service import llm_service

class AgentExtractionService:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    async def extract_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """Extract agent data from MLS or public sources"""
        # In prototype, generate sample agent data
        return await self._generate_agent_data(agent_id)
    
    async def _generate_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """Generate sample agent data for prototype"""
        prompt = f"""
        Generate realistic real estate agent profile information for agent ID {agent_id}.
        
        Include:
        - Name, brokerage, and contact information
        - Years of experience
        - Areas of expertise/specialization
        - Transaction history for past 24 months
          - Number of listings
          - Listing prices
          - Days on market
          - List price to sale price ratio
          - Transaction types (buyer/seller representation)
        - Geographic focus areas
        - Client testimonials (3-5)
        - Professional certifications
        
        Return data as a single JSON object with nested structure.
        """
        
        response = await llm_service.complete(
            prompt=prompt,
            max_tokens=2500,
            temperature=0.6
        )
        
        # Extract JSON from response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in agent data response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse agent data as JSON")
    
    async def extract_market_context(self, area_codes: List[str]) -> Dict[str, Any]:
        """Extract market context data for benchmark comparison"""
        # In prototype, generate sample market data
        prompt = f"""
        Generate realistic real estate market statistics for areas {', '.join(area_codes)}.
        
        Include:
        - Average days on market
        - Average list price to sale price ratio
        - Median property prices
        - Transaction volumes
        - Market trends over past 12 months
        - Agent performance benchmarks:
          - Top quartile performance metrics
          - Median performance metrics
          - Bottom quartile performance metrics
        
        Return as a JSON object with nested structure by area code.
        """
        
        response = await llm_service.complete(
            prompt=prompt,
            max_tokens=3000,
            temperature=0.5
        )
        
        # Extract JSON from response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in market context response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse market context as JSON")

agent_service = AgentExtractionService()
```

### Agent Analysis Workflow

```python
# app/services/agent_analysis_workflow.py
import json
from typing import Dict, List, Any, Optional
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service
from app.services.agent_extraction_service import agent_service

class AgentAnalysisWorkflow:
    def __init__(self):
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.agent_service = agent_service
        
        # Initialize prompt templates
        self._init_prompts()
        
    def _init_prompts(self):
        """Initialize prompt templates for agent analysis"""
        os.makedirs("prompts/agent_analysis", exist_ok=True)
        
        with open("prompts/agent_analysis/performance_analysis.txt", "w") as f:
            f.write("""
<cognitive_function>
You are performing quantitative analysis of real estate agent performance metrics.
Your role is to identify strengths, weaknesses, and competitive differentiators.
</cognitive_function>

<input>
Agent data: {{ agent_data | tojson }}
Market context: {{ market_context | tojson }}
{% if comparison_agents %}
Comparison agents: {{ comparison_agents | tojson }}
{% endif %}
</input>

<reasoning>
Analyze the agent's performance across key metrics:
1. Transaction volume relative to market
2. Days on market compared to area averages
3. List-to-sale price ratio performance
4. Geographic concentration and specialization
5. Price tier focus and performance
6. Listing vs. buyer representation balance
7. Seasonal performance patterns
8. Year-over-year growth trends

Compare metrics to:
- Market averages for the same period
- Benchmark data for different performance tiers
- Historical performance of the same agent
</reasoning>

<output_format>
Provide a JSON object with the following structure:
{
  "performance_summary": {
    "overall_assessment": "text description",
    "relative_market_position": "percentile or ranking",
    "key_strengths": ["strength1", "strength2"],
    "improvement_areas": ["area1", "area2"]
  },
  "detailed_metrics": {
    "transaction_volume": {
      "value": X,
      "market_percentile": Y,
      "year_over_year_change": Z
    },
    "days_on_market": {
      "value": X,
      "versus_market_average": Y,
      "trend": "text description"
    },
    // Additional metrics...
  },
  "competitive_differentiators": ["differentiator1", "differentiator2"],
  "recommendations": ["recommendation1", "recommendation2"]
}
</output_format>
""")
        
        with open("prompts/agent_analysis/insight_generation.txt", "w") as f:
            f.write("""
<cognitive_function>
You are generating strategic insights from real estate agent performance analysis.
Your role is to identify actionable opportunities and competitive advantages.
</cognitive_function>

<input>
Performance analysis: {{ performance_analysis | tojson }}
Agent data: {{ agent_data | tojson }}
Market context: {{ market_context | tojson }}
</input>

<reasoning>
Based on the performance analysis:
1. Identify market segments where the agent has comparative advantage
2. Detect underserved niches that align with agent strengths
3. Recognize patterns in successful transactions
4. Find correlation between property features and performance metrics
5. Analyze competitive positioning against other agents
6. Evaluate pricing strategy effectiveness by segment
7. Identify potential operational inefficiencies
</reasoning>

<output_format>
Provide a natural language response with clear sections:

# STRATEGIC INSIGHTS

## Market Positioning
[Insights about optimal positioning based on performance data]

## Competitive Advantages
[2-3 paragraphs on demonstrable competitive advantages]

## Growth Opportunities
[Specific market segments or strategies to pursue]

## Operational Improvements
[Suggestions for improving key metrics]

## Client Value Proposition
[How to articulate unique value to potential clients]
</output_format>
""")
    
    async def execute(self, 
                    agent_id: str,
                    area_codes: List[str] = None,
                    comparison_agent_ids: List[str] = None) -> Dict[str, Any]:
        """Execute the agent analysis workflow"""
        try:
            # Step 1: Extract agent data
            agent_data = await self.agent_service.extract_agent_data(agent_id)
            
            # Step 2: Extract market context
            if not area_codes and 'primary_areas' in agent_data:
                area_codes = agent_data['primary_areas']
            elif not area_codes:
                area_codes = ['95113']  # Default area code
                
            market_context = await self.agent_service.extract_market_context(area_codes)
            
            # Step 3: Extract comparison agent data (if requested)
            comparison_agents = None
            if comparison_agent_ids:
                comparison_agents = []
                for comp_id in comparison_agent_ids:
                    comp_data = await self.agent_service.extract_agent_data(comp_id)
                    comparison_agents.append(comp_data)
            
            # Step 4: Perform performance analysis
            performance_analysis = await self._analyze_performance(
                agent_data, market_context, comparison_agents
            )
            
            # Step 5: Generate strategic insights
            insights = await self._generate_insights(
                performance_analysis, agent_data, market_context
            )
            
            return {
                "agent_id": agent_id,
                "agent_data": agent_data,
                "market_context": market_context,
                "performance_analysis": performance_analysis,
                "strategic_insights": insights
            }
        except Exception as e:
            print(f"Agent analysis workflow error: {e}")
            return {
                "agent_id": agent_id,
                "error": str(e)
            }
    
    async def _analyze_performance(self,
                                 agent_data: Dict[str, Any],
                                 market_context: Dict[str, Any],
                                 comparison_agents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze agent performance metrics"""
        analysis_prompt = self.prompt_service.get_prompt(
            "agent_analysis.performance_analysis",
            agent_data=agent_data,
            market_context=market_context,
            comparison_agents=comparison_agents
        )
        
        analysis_response = await self.llm_service.complete(
            prompt=analysis_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Extract JSON from response
        try:
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in performance analysis response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse performance analysis as JSON")
    
    async def _generate_insights(self,
                              performance_analysis: Dict[str, Any],
                              agent_data: Dict[str, Any],
                              market_context: Dict[str, Any]) -> str:
        """Generate strategic insights from performance analysis"""
        insight_prompt = self.prompt_service.get_prompt(
            "agent_analysis.insight_generation",
            performance_analysis=performance_analysis,
            agent_data=agent_data,
            market_context=market_context
        )
        
        insights = await self.llm_service.complete(
            prompt=insight_prompt,
            temperature=0.7,
            max_tokens=2500
        )
        
        return insights

agent_workflow = AgentAnalysisWorkflow()
```

### API Endpoints for Agent Analysis

```python
# Add to app/main.py
from app.services.agent_analysis_workflow import agent_workflow
from typing import List

class AgentAnalysisRequest(BaseModel):
    agent_id: str
    area_codes: Optional[List[str]] = None
    comparison_agent_ids: Optional[List[str]] = None

class AgentAnalysisResponse(BaseModel):
    agent_id: str
    agent_data: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    strategic_insights: Optional[str] = None
    error: Optional[str] = None

@app.post("/api/agent-analysis", response_model=AgentAnalysisResponse)
async def analyze_agent(request: AgentAnalysisRequest):
    """Execute agent analysis workflow"""
    result = await agent_workflow.execute(
        request.agent_id,
        request.area_codes,
        request.comparison_agent_ids
    )
    return result
```

## STARTUP SCRIPT

```bash
# bootstrap.sh
#!/bin/bash
echo "Starting Realtor AI Copilot setup..."

# Check for Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Create project structure
mkdir -p realtor-ai/{app,prompts,data,tests}
mkdir -p realtor-ai/app/{api,models,services,repositories}
mkdir -p realtor-ai/prompts/{property_search,agent_analysis,content_generation}

# Clone repo if using version control
# git clone https://github.com/yourusername/realtor-ai.git

# Setup virtual environment
cd realtor-ai
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic python-dotenv httpx jinja2 asyncpg pgvector openai pytest

# Create minimal .env file
cat > .env << EOF
# API Keys
LLM_OPENAI_API_KEY=your_openai_key_here
LLM_ANTHROPIC_API_KEY=optional_anthropic_key_here

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
EOF

# Create Docker Compose file
cat > docker-compose.yml << EOF
version: '3.8'
services:
  db:
    image: postgres:14
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: realtor_ai
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  pgvector:
    image: ankane/pgvector
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vector_store
    ports:
      - "5433:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data

  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app
    depends_on:
      - db
      - pgvector
    environment:
      - DB_HOST=db
      - VECTORDB_HOST=pgvector

volumes:
  postgres_data:
  pgvector_data:
EOF

# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create requirements.txt
cat > requirements.txt << EOF
fastapi
uvicorn
pydantic
python-dotenv
httpx
jinja2
asyncpg
pgvector
openai
pytest
EOF

echo "Setup complete. Edit .env file with your API keys, then:"
echo "1. Start infrastructure with: docker-compose up -d db pgvector"
echo "2. Run API with: uvicorn app.main:app --reload"
```

## USAGE EXAMPLES

### Property Search Example

```bash
curl -X POST http://localhost:8000/api/property-search \
     -H "Content-Type: application/json" \
     -d '{"query": "I need a modern 3-bedroom home with a view of the water, ideally with an open floor plan and within walking distance to restaurants. My budget is around $750,000.", "user_id": "user123"}'
```

### Agent Analysis Example

```bash
curl -X POST http://localhost:8000/api/agent-analysis \
     -H "Content-Type: application/json" \
     -d '{"agent_id": "A12345", "area_codes": ["95113", "95125"], "comparison_agent_ids": ["B67890", "C54321"]}'
```
