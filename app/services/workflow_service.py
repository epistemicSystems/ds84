"""Cognitive workflow service for property search"""
import json
from typing import Dict, List, Any, Optional, Literal
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service
from app.services.embedding_service import embedding_service
from app.repositories.vector_repository import vector_repository


SearchMode = Literal["vector", "hybrid", "simulated"]


class PropertySearchWorkflow:
    """Workflow for processing natural language property queries"""

    def __init__(self):
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository

    async def execute(
        self,
        query: str,
        user_id: str = None,
        search_mode: SearchMode = "vector",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Execute the property search workflow

        Args:
            query: Natural language property query
            user_id: Optional user identifier for context
            search_mode: Search mode - "vector", "hybrid", or "simulated"
            limit: Maximum number of properties to return

        Returns:
            Dictionary containing query, intent, properties, and response
        """
        try:
            # Step 1: Analyze user intent
            intent = await self._analyze_intent(query)

            # Step 2: Search for properties
            if search_mode == "vector":
                properties = await self._vector_search(query, intent, limit)
            elif search_mode == "hybrid":
                properties = await self._hybrid_search(query, intent, limit)
            else:  # simulated
                properties = await self._simulate_property_search(intent)

            # Step 3: Generate response
            response = await self._generate_response(query, intent, properties)

            return {
                "query": query,
                "intent": intent,
                "properties": properties,
                "response": response,
                "search_mode": search_mode,
                "status": "success"
            }
        except Exception as e:
            print(f"Workflow error: {e}")
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }

    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user intent from natural language query"""
        intent_prompt = self.prompt_service.get_prompt(
            "property_search.intent_analysis",
            query=query
        )

        intent_response = await self.llm_service.complete(
            prompt=intent_prompt,
            temperature=0.3,  # Lower temperature for more consistent parsing
            max_tokens=800
        )

        # Extract JSON from response
        return self._parse_json_response(intent_response)

    async def _vector_search(
        self,
        query: str,
        intent: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search properties using vector similarity

        Args:
            query: Original user query
            intent: Parsed intent
            limit: Maximum number of results

        Returns:
            List of matching properties with similarity scores
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_query_embedding(query)

        # Search by vector
        results = await self.vector_repository.search_by_vector(
            query_vector=query_embedding,
            limit=limit
        )

        # Extract properties and add similarity scores
        properties = []
        for property_data, similarity in results:
            property_with_score = property_data.copy()
            property_with_score['similarity_score'] = round(similarity, 4)
            properties.append(property_with_score)

        return properties

    async def _hybrid_search(
        self,
        query: str,
        intent: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search properties using hybrid approach (vector + filters)

        Args:
            query: Original user query
            intent: Parsed intent
            limit: Maximum number of results

        Returns:
            List of matching properties
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_query_embedding(query)

        # Build filters from intent
        filters = self._intent_to_filters(intent)

        # Hybrid search
        results = await self.vector_repository.hybrid_search(
            query_vector=query_embedding,
            filters=filters,
            limit=limit
        )

        # Extract properties and add similarity scores
        properties = []
        for property_data, similarity in results:
            property_with_score = property_data.copy()
            property_with_score['similarity_score'] = round(similarity, 4)
            properties.append(property_with_score)

        return properties

    def _intent_to_filters(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed intent to database filters

        Args:
            intent: Parsed search intent

        Returns:
            Dictionary of filters for database query
        """
        filters = {}

        # Property type filter
        if 'property_types' in intent and intent['property_types']:
            # For now, just use the first property type
            # In production, support multiple types
            filters['property_type'] = intent['property_types'][0]

        # Bedroom filter (use minimum if specified)
        if 'bedrooms' in intent and isinstance(intent['bedrooms'], dict):
            if 'min' in intent['bedrooms']:
                filters['bedrooms'] = intent['bedrooms']['min']

        # Add more filters as needed
        # Note: Complex filters (ranges, etc.) need custom query logic

        return filters

    async def _simulate_property_search(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate property search using LLM (for prototype phase)

        In production, this would query a real property database with vector search.
        """
        search_prompt = f"""
You are a real estate database. Based on the following search criteria:
{json.dumps(intent, indent=2)}

Generate 5 realistic property listings that would match these criteria.

Each property should include:
- address (street, city, state, zip)
- price (as integer)
- bedrooms (integer)
- bathrooms (float, e.g., 2.5)
- square_footage (integer)
- lot_size (string, e.g., "0.25 acres")
- year_built (integer)
- property_type (string)
- key_features (array of strings, 5-8 items)
- description (string, 2-3 sentences)

Return ONLY a JSON array of property objects, no additional text.
"""

        search_response = await self.llm_service.complete(
            prompt=search_prompt,
            max_tokens=2000,
            temperature=0.7
        )

        # Extract JSON from response
        return self._parse_json_response(search_response, expect_array=True)

    async def _generate_response(
        self,
        query: str,
        intent: Dict[str, Any],
        properties: List[Dict[str, Any]]
    ) -> str:
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

    def _parse_json_response(
        self,
        response: str,
        expect_array: bool = False
    ) -> Any:
        """Parse JSON from LLM response

        Args:
            response: LLM response text
            expect_array: Whether to expect a JSON array vs object

        Returns:
            Parsed JSON data
        """
        try:
            # Find JSON in response
            if expect_array:
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
            else:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\n\nResponse: {response}")


# Global workflow service instance
workflow_service = PropertySearchWorkflow()
