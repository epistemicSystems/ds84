"""Cognitive workflow service for property search"""
import json
from typing import Dict, List, Any, Optional
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service


class PropertySearchWorkflow:
    """Workflow for processing natural language property queries"""

    def __init__(self):
        self.llm_service = llm_service
        self.prompt_service = prompt_service

    async def execute(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Execute the property search workflow

        Args:
            query: Natural language property query
            user_id: Optional user identifier for context

        Returns:
            Dictionary containing query, intent, properties, and response
        """
        try:
            # Step 1: Analyze user intent
            intent = await self._analyze_intent(query)

            # Step 2: Search for properties (simulated for prototype)
            properties = await self._simulate_property_search(intent)

            # Step 3: Generate response
            response = await self._generate_response(query, intent, properties)

            return {
                "query": query,
                "intent": intent,
                "properties": properties,
                "response": response,
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
