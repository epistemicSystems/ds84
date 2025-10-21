# Phase II: Multi-Modal Perception System

The initial prototype established basic data extraction capabilities but lacked sophisticated multi-modal perception—the ability to derive deep semantic understanding from heterogeneous data sources including property images, agent content, and market signals. Phase II introduces a comprehensive perception subsystem that transcends simple data extraction to achieve cognitive synthesis across modalities.

## Theoretical Framework

Multi-modal perception operates on the principle that semantic understanding emerges from the fusion of differential information streams. In real estate, this manifests as the integration of:

1. Visual-spatial understanding (property images)
2. Temporal-sequential analysis (market trends)
3. Textual-descriptive processing (property descriptions)
4. Structured attribute comprehension (property metadata)

## Architectural Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                     MULTI-MODAL FUSION ENGINE                       │
│                                                                     │
├─────────────┬─────────────────────────────┬─────────────────────────┤
│             │                             │                         │
│   Visual    │        Textual              │      Structured         │
│  Encoder    │        Encoder              │      Encoder            │
│             │                             │                         │
└─────────────┴─────────────────────────────┴─────────────────────────┘
       │                   │                           │
       ▼                   ▼                           ▼
┌─────────────┐    ┌─────────────────┐     ┌─────────────────────────┐
│  Image      │    │  Description    │     │  Metadata                │
│  Pipeline   │    │  Pipeline       │     │  Pipeline                │
└─────────────┘    └─────────────────┘     └─────────────────────────┘
```

## Implementation Strategy

### Visual Perception Pipeline

```python
# app/services/visual_perception_service.py
import asyncio
import base64
import httpx
import json
from io import BytesIO
from typing import List, Dict, Any, Optional
from PIL import Image
from app.services.llm_service import llm_service

class VisualPerceptionService:
    def __init__(self, llm_service):
        self.llm_service = llm_service
        
    async def analyze_property_images(self, 
                                    image_urls: List[str]) -> Dict[str, Any]:
        """Analyze property images to extract features and attributes"""
        # Fetch and process images
        images = await self._fetch_images(image_urls)
        
        if not images:
            return {
                "error": "No valid images to analyze",
                "features": []
            }
        
        # Extract features using multimodal LLM
        features = await self._extract_image_features(images)
        
        # Categorize room types
        room_types = await self._identify_room_types(images)
        
        # Detect aesthetic qualities
        aesthetics = await self._analyze_aesthetics(images)
        
        # Detect potential issues or red flags
        issues = await self._detect_issues(images)
        
        return {
            "features": features,
            "room_types": room_types,
            "aesthetics": aesthetics,
            "issues": issues,
            "image_count": len(images)
        }
    
    async def _fetch_images(self, image_urls: List[str]) -> List[bytes]:
        """Fetch images from URLs"""
        images = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in image_urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    images.append(response.content)
                except Exception as e:
                    print(f"Error fetching image {url}: {e}")
        
        return images
    
    async def _extract_image_features(self, images: List[bytes]) -> List[Dict[str, Any]]:
        """Extract property features from images using GPT-4 Vision"""
        # Use only first 5 images to stay within context limits
        images_to_analyze = images[:5]  
        
        # Convert images to base64
        base64_images = []
        for img_data in images_to_analyze:
            try:
                img = Image.open(BytesIO(img_data))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_images.append(img_str)
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Prepare the prompt for GPT-4 Vision
        content = [{
            "type": "text",
            "text": """Analyze these property images and extract the following features:
1. Property type and architectural style
2. Notable exterior features
3. Interior features and finishes
4. Room types visible
5. Condition assessment
6. Unique or luxury features
7. Potential renovation opportunities

Format your response as a JSON object with these categories as keys and detailed observations as values."""
        }]
        
        # Add images to content
        for img_str in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                }
            })
        
        # Call GPT-4 Vision
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_service.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "max_tokens": 2000
                }
            )
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    features = json.loads(json_str)
                    return features
                else:
                    # Try to parse the whole response as JSON
                    return json.loads(response_text)
            except json.JSONDecodeError:
                # Return as text if JSON parsing fails
                return {"raw_analysis": response_text}
    
    async def _identify_room_types(self, images: List[bytes]) -> Dict[str, int]:
        """Identify and count room types from property images"""
        # Similar implementation as _extract_image_features but focused on room classification
        # Returns dict like {"kitchen": 2, "bathroom": 3, "bedroom": 4, "living_room": 1}
        pass
    
    async def _analyze_aesthetics(self, images: List[bytes]) -> Dict[str, Any]:
        """Analyze aesthetic qualities of the property"""
        # Implementation to analyze design style, colors, natural light, etc.
        pass
    
    async def _detect_issues(self, images: List[bytes]) -> List[Dict[str, Any]]:
        """Detect potential issues or red flags in property images"""
        # Implementation to identify maintenance issues, outdated features, etc.
        pass

    async def generate_property_embedding(self, 
                                       property_data: Dict[str, Any], 
                                       images: List[bytes] = None) -> Dict[str, List[float]]:
        """Generate multi-modal embedding for a property"""
        embeddings = {}
        
        # Generate text embedding for property description
        if "description" in property_data:
            text_embedding = await self.llm_service.generate_embedding(
                property_data["description"]
            )
            embeddings["description"] = text_embedding
        
        # Generate feature embedding
        features_text = self._serialize_property_features(property_data)
        if features_text:
            features_embedding = await self.llm_service.generate_embedding(
                features_text
            )
            embeddings["features"] = features_embedding
        
        # Generate image embeddings if available
        if images and len(images) > 0:
            # Implementation for image embeddings using CLIP or similar
            # This would require additional API integration
            pass
        
        # Create a composite embedding (simple average for now)
        if embeddings:
            composite = []
            for embed_list in embeddings.values():
                if not composite:
                    composite = embed_list.copy()
                else:
                    for i in range(len(composite)):
                        composite[i] += embed_list[i]
            
            # Normalize
            if composite:
                magnitude = sum(x**2 for x in composite) ** 0.5
                if magnitude > 0:
                    composite = [x/magnitude for x in composite]
                embeddings["composite"] = composite
        
        return embeddings
    
    def _serialize_property_features(self, property_data: Dict[str, Any]) -> str:
        """Convert property features to a text representation for embedding"""
        features = []
        
        # Add property type
        if "property_type" in property_data:
            features.append(f"Property type: {property_data['property_type']}")
        
        # Add bedrooms and bathrooms
        if "bedrooms" in property_data:
            features.append(f"Bedrooms: {property_data['bedrooms']}")
        if "bathrooms" in property_data:
            features.append(f"Bathrooms: {property_data['bathrooms']}")
        
        # Add square footage
        if "square_feet" in property_data:
            features.append(f"Square feet: {property_data['square_feet']}")
        
        # Add features list
        if "features" in property_data and isinstance(property_data["features"], list):
            features.append("Features: " + ", ".join(property_data["features"]))
        
        # Add location information
        if "location" in property_data:
            loc = property_data["location"]
            if isinstance(loc, dict):
                if "neighborhood" in loc:
                    features.append(f"Neighborhood: {loc['neighborhood']}")
                if "city" in loc:
                    features.append(f"City: {loc['city']}")
            elif isinstance(loc, str):
                features.append(f"Location: {loc}")
        
        return " ".join(features)
```

### Enhanced Textual Perception

```python
# app/services/textual_perception_service.py
import asyncio
import json
from typing import Dict, Any, List, Optional
from app.services.llm_service import llm_service

class TextualPerceptionService:
    def __init__(self, llm_service):
        self.llm_service = llm_service
        
    async def analyze_property_description(self, 
                                         description: str) -> Dict[str, Any]:
        """Extract structured information from property description text"""
        analysis_prompt = f"""
        Extract structured information from the following property description.
        Parse the text to identify:
        
        1. Property type and key characteristics
        2. Location details and neighborhood information
        3. Interior features and layout
        4. Exterior features and lot details
        5. Renovations or updates mentioned
        6. Amenities and special features
        7. Agent remarks about property condition
        8. Marketing language and emotional appeal
        9. Investment potential or rental information
        
        Property Description:
        "{description}"
        
        Return the information as a JSON object with these categories as keys.
        Include only information explicitly stated or clearly implied in the text.
        """
        
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
                raise ValueError("No JSON found in description analysis response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse description analysis as JSON")
    
    async def detect_language_patterns(self, 
                                     text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns and sentiment in text"""
        language_prompt = f"""
        Perform a detailed linguistic analysis of the following real estate text:
        
        "{text}"
        
        Analyze the following aspects:
        1. Overall sentiment (positive/negative/neutral)
        2. Emotional tone (e.g., enthusiastic, cautious, urgent)
        3. Target audience signals
        4. Persuasive techniques used
        5. Linguistic patterns characteristic of the author
        6. Unique vocabulary or industry jargon
        7. Formality level
        
        Return the analysis as a JSON object with these categories as keys.
        """
        
        language_response = await self.llm_service.complete(
            prompt=language_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Extract JSON from response
        try:
            json_start = language_response.find('{')
            json_end = language_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = language_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in language analysis response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse language analysis as JSON")
    
    async def extract_comparative_language(self,
                                        agent_content: str) -> Dict[str, Any]:
        """Extract comparative claims and positioning from agent content"""
        comparative_prompt = f"""
        Analyze the following real estate agent content to identify:
        
        "{agent_content}"
        
        Extract:
        1. Competitive differentiation claims
        2. Comparative statements about market position
        3. Unique selling propositions
        4. Experience or success metrics highlighted
        5. Specific promises or guarantees made
        6. Client testimonials or social proof elements
        7. Areas of specialization claimed
        
        Return the analysis as a JSON object with these categories as keys,
        including direct quotes where relevant.
        """
        
        comparative_response = await self.llm_service.complete(
            prompt=comparative_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Extract JSON from response
        try:
            json_start = comparative_response.find('{')
            json_end = comparative_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = comparative_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in comparative analysis response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse comparative analysis as JSON")
```

### Structured Data Perception

```python
# app/services/structured_perception_service.py
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class StructuredPerceptionService:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        
    async def analyze_price_trends(self,
                                 area_code: str,
                                 property_type: str = None,
                                 time_window: int = 180) -> Dict[str, Any]:
        """Analyze price trends for a specific area and property type"""
        start_date = datetime.now() - timedelta(days=time_window)
        
        # Build query conditions
        conditions = ["area_code = $1", "list_date >= $2"]
        params = [area_code, start_date]
        param_idx = 3
        
        if property_type:
            conditions.append(f"property_type = ${param_idx}")
            params.append(property_type)
            param_idx += 1
        
        where_clause = " AND ".join(conditions)
        
        # Execute query
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch(f"""
            SELECT 
                date_trunc('week', list_date) as week,
                COUNT(*) as listings,
                AVG(list_price) as avg_price,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY list_price) as median_price,
                AVG(square_feet) as avg_square_feet,
                AVG(list_price / NULLIF(square_feet, 0)) as avg_price_per_sqft
            FROM properties
            WHERE {where_clause}
            GROUP BY week
            ORDER BY week
            """, *params)
            
            # Calculate trends
            weeks = []
            listings = []
            avg_prices = []
            median_prices = []
            price_per_sqft = []
            
            for row in results:
                weeks.append(row['week'].isoformat())
                listings.append(row['listings'])
                avg_prices.append(float(row['avg_price']))
                median_prices.append(float(row['median_price']))
                price_per_sqft.append(float(row['avg_price_per_sqft']))
            
            # Calculate trend percentages
            trends = {}
            if len(avg_prices) >= 2:
                first, last = avg_prices[0], avg_prices[-1]
                trends["avg_price_change_pct"] = ((last - first) / first) * 100
                
                first, last = median_prices[0], median_prices[-1]
                trends["median_price_change_pct"] = ((last - first) / first) * 100
                
                first, last = price_per_sqft[0], price_per_sqft[-1]
                trends["price_per_sqft_change_pct"] = ((last - first) / first) * 100
            
            return {
                "area_code": area_code,
                "property_type": property_type,
                "time_window_days": time_window,
                "weeks": weeks,
                "listings_count": listings,
                "avg_prices": avg_prices,
                "median_prices": median_prices,
                "price_per_sqft": price_per_sqft,
                "trends": trends
            }
    
    async def analyze_days_on_market(self,
                                   area_code: str,
                                   property_type: str = None,
                                   time_window: int = 180) -> Dict[str, Any]:
        """Analyze days on market trends for a specific area and property type"""
        # Similar implementation to analyze_price_trends but focused on DOM metrics
        pass
    
    async def detect_market_anomalies(self,
                                    area_code: str,
                                    property_type: str = None) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in market data"""
        # Implementation to identify unusual patterns in pricing, DOM, etc.
        pass
    
    async def generate_comparative_market_analysis(self,
                                                property_data: Dict[str, Any],
                                                radius_miles: float = 1.0,
                                                max_comps: int = 5) -> Dict[str, Any]:
        """Generate CMA for a specific property"""
        # Implementation for comparative market analysis
        # Finds similar properties, calculates price per square foot, etc.
        pass
```

### Multi-Modal Fusion Service

The fusion service represents the cognitive synthesis layer, integrating insights from visual, textual, and structured perception into a unified semantic understanding.

```python
# app/services/multimodal_fusion_service.py
import asyncio
import json
from typing import Dict, Any, List, Optional
from app.services.visual_perception_service import VisualPerceptionService
from app.services.textual_perception_service import TextualPerceptionService
from app.services.structured_perception_service import StructuredPerceptionService
from app.services.llm_service import llm_service

class MultiModalFusionService:
    def __init__(self,
               visual_perception: VisualPerceptionService,
               textual_perception: TextualPerceptionService,
               structured_perception: StructuredPerceptionService,
               llm_service):
        self.visual_perception = visual_perception
        self.textual_perception = textual_perception
        self.structured_perception = structured_perception
        self.llm_service = llm_service
        
    async def analyze_property(self,
                             property_data: Dict[str, Any],
                             image_urls: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive multi-modal analysis of a property"""
        analysis_tasks = []
        
        # Add visual analysis if images available
        if image_urls:
            analysis_tasks.append(self.visual_perception.analyze_property_images(image_urls))
        
        # Add textual analysis if description available
        if "description" in property_data:
            analysis_tasks.append(
                self.textual_perception.analyze_property_description(property_data["description"])
            )
        
        # Add market context analysis if location available
        if "area_code" in property_data:
            analysis_tasks.append(
                self.structured_perception.analyze_price_trends(
                    area_code=property_data["area_code"],
                    property_type=property_data.get("property_type")
                )
            )
        
        # Execute all analysis tasks concurrently
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analysis_results = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Analysis error: {result}")
                continue
                
            # Add result to appropriate category
            if i == 0 and image_urls:
                analysis_results["visual_analysis"] = result
            elif i == (1 if image_urls else 0) and "description" in property_data:
                analysis_results["textual_analysis"] = result
            elif "area_code" in property_data:
                analysis_results["market_analysis"] = result
        
        # Generate semantic fusion
        fusion = await self._generate_semantic_fusion(property_data, analysis_results)
        analysis_results["semantic_fusion"] = fusion
        
        # Generate property embeddings
        embeddings = await self.visual_perception.generate_property_embedding(
            property_data=property_data
        )
        analysis_results["embeddings"] = embeddings
        
        return analysis_results
    
    async def analyze_agent(self,
                          agent_data: Dict[str, Any],
                          content_samples: List[str] = None) -> Dict[str, Any]:
        """Perform multi-modal analysis of an agent's online presence"""
        analysis_tasks = []
        
        # Add linguistic analysis of agent content
        if content_samples:
            for i, sample in enumerate(content_samples[:3]):  # Limit to 3 samples
                analysis_tasks.append(
                    self.textual_perception.detect_language_patterns(sample)
                )
                
                # Also add comparative analysis
                analysis_tasks.append(
                    self.textual_perception.extract_comparative_language(sample)
                )
        
        # Execute all analysis tasks concurrently
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analysis_results = {
            "language_patterns": [],
            "comparative_analysis": []
        }
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Analysis error: {result}")
                continue
                
            # Alternate between language patterns and comparative analysis
            if i % 2 == 0:
                analysis_results["language_patterns"].append(result)
            else:
                analysis_results["comparative_analysis"].append(result)
        
        # Generate semantic fusion
        fusion = await self._generate_agent_semantic_fusion(agent_data, analysis_results)
        analysis_results["semantic_fusion"] = fusion
        
        return analysis_results
    
    async def _generate_semantic_fusion(self,
                                      property_data: Dict[str, Any],
                                      analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate semantic fusion of multi-modal property analysis"""
        fusion_prompt = f"""
        Synthesize the following multi-modal property analysis into a coherent semantic understanding.
        
        Property Data:
        {json.dumps(property_data, indent=2)}
        
        Analysis Results:
        {json.dumps(analysis_results, indent=2)}
        
        Generate a comprehensive semantic fusion that includes:
        1. Key property attributes validated across modalities
        2. Discrepancies or inconsistencies between different analyses
        3. Hidden or implicit value factors not explicitly stated
        4. Market positioning insights
        5. Likely buyer persona and appeal factors
        6. Investment considerations
        7. Competitive differentiation factors
        
        Return the semantic fusion as a JSON object with these categories as keys.
        """
        
        fusion_response = await self.llm_service.complete(
            prompt=fusion_prompt,
            temperature=0.4,
            max_tokens=2500
        )
        
        # Extract JSON from response
        try:
            json_start = fusion_response.find('{')
            json_end = fusion_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = fusion_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in semantic fusion response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse semantic fusion as JSON")
    
    async def _generate_agent_semantic_fusion(self,
                                           agent_data: Dict[str, Any],
                                           analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate semantic fusion of agent analysis"""
        # Similar implementation to _generate_semantic_fusion but focused on agent insights
        pass
```

## Integration with Vector Search

To leverage these multi-modal perception capabilities for enhanced search, we need to extend our vector search implementation:

```python
# Enhanced search method in VectorRepository
async def multi_modal_search(self,
                           query_text: str,
                           visual_concepts: List[str] = None,
                           filters: Dict[str, Any] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
    """Execute a multi-modal search combining text and visual concepts"""
    await self.initialize()
    
    # Generate text embedding for query
    text_embedding = await self.llm_service.generate_embedding(query_text)
    
    # Generate embeddings for visual concepts if provided
    visual_embedding = None
    if visual_concepts and len(visual_concepts) > 0:
        visual_text = "Property with " + ", ".join(visual_concepts)
        visual_embedding = await self.llm_service.generate_embedding(visual_text)
    
    # Build dynamic query conditions for filters
    conditions = []
    params = [limit]
    param_idx = 2
    
    if filters:
        for key, value in filters.items():
            path = key.replace('.', '->')
            conditions.append(f"data->>{path} = ${param_idx}")
            params.append(str(value))
            param_idx += 1
    
    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    
    # Execute multi-modal search
    async with self.pool.acquire() as conn:
        if visual_embedding:
            # Combine text and visual embeddings with equal weights
            combined_results = await conn.fetch(f"""
            WITH text_results AS (
                SELECT id, data, 1 - (embedding <=> $1) AS text_similarity
                FROM properties
                WHERE embedding IS NOT NULL
            ),
            visual_results AS (
                SELECT id, data, 1 - (embedding <=> $2) AS visual_similarity
                FROM properties
                WHERE embedding IS NOT NULL
            )
            SELECT 
                tr.id,
                tr.data,
                (tr.text_similarity * 0.7 + vr.visual_similarity * 0.3) AS combined_similarity
            FROM text_results tr
            JOIN visual_results vr ON tr.id = vr.id
            WHERE {where_clause}
            ORDER BY combined_similarity DESC
            LIMIT $3
            """, text_embedding, visual_embedding, *params)
            
            return [
                {**json.loads(r['data']), "similarity": float(r['combined_similarity'])} 
                for r in combined_results
            ]
        else:
            # Text-only search
            results = await conn.fetch(f"""
            SELECT id, data, 1 - (embedding <=> $1) AS similarity
            FROM properties
            WHERE embedding IS NOT NULL AND {where_clause}
            ORDER BY similarity DESC
            LIMIT $2
            """, text_embedding, *params)
            
            return [
                {**json.loads(r['data']), "similarity": float(r['similarity'])} 
                for r in results
            ]
```

## Implementation Timeline

**Week 1: Visual Perception Pipeline**
- Day 1-2: Implement image analysis with GPT-4 Vision
- Day 3-4: Build room classification and aesthetics detection
- Day 5: Develop issue detection and visual embedding generation

**Week 2: Textual & Structured Perception**
- Day 1-2: Implement description analysis and language pattern detection
- Day 3-4: Build market data analysis capabilities
- Day 5: Develop comparative market analysis generation

**Week 3: Multi-Modal Fusion & Search**
- Day 1-2: Implement semantic fusion engine
- Day 3-4: Extend vector search for multi-modal queries
- Day 5: Integrate with cognitive workflow engine
