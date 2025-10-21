"""MLS data extraction service for property data ingestion"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from app.services.llm_service import llm_service


class MLSExtractionService:
    """Service for extracting property data from MLS sources

    In prototype phase, generates realistic sample data using LLM.
    In production, this would integrate with actual MLS APIs.
    """

    def __init__(self):
        self.llm_service = llm_service

    async def extract_listings(
        self,
        area_code: str,
        limit: int = 100,
        property_type: Optional[str] = None,
        price_range: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """Extract property listings from MLS

        Args:
            area_code: ZIP code or area code for properties
            limit: Maximum number of listings to extract
            property_type: Optional property type filter
            price_range: Optional price range filter {"min": X, "max": Y}

        Returns:
            List of property data dictionaries
        """
        # In prototype, generate sample data using LLM
        # In production, replace with actual MLS API calls
        return await self._generate_sample_listings(
            area_code, limit, property_type, price_range
        )

    async def _generate_sample_listings(
        self,
        area_code: str,
        limit: int,
        property_type: Optional[str] = None,
        price_range: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """Generate realistic sample property listings using LLM

        Args:
            area_code: ZIP code for properties
            limit: Number of listings to generate
            property_type: Optional property type
            price_range: Optional price range

        Returns:
            List of property dictionaries
        """
        # Build filter description
        filters = []
        if property_type:
            filters.append(f"property type: {property_type}")
        if price_range:
            price_str = f"${price_range.get('min', 0):,} - ${price_range.get('max', 0):,}"
            filters.append(f"price range: {price_str}")

        filter_description = ", ".join(filters) if filters else "various types and prices"

        prompt = f"""
Generate {limit} realistic MLS property listings for ZIP code {area_code} ({filter_description}).

Each listing must include:
- mls_id: Format "MLS" + 8 digits (e.g., "MLS12345678")
- address: Object with street, city, state, zip, neighborhood
- price: Integer (market-appropriate for area)
- bedrooms: Integer (1-6)
- bathrooms: Float (1.0-5.5, use .5 for half bath)
- square_footage: Integer (600-6000)
- lot_size: String (e.g., "0.25 acres", "5000 sqft")
- year_built: Integer (1950-2024)
- property_type: "single_family", "condo", "townhouse", or "multi_family"
- status: "active" or "pending"
- days_on_market: Integer (0-180)
- key_features: Array of 5-10 feature strings
- description: String, 2-4 sentences, compelling and descriptive
- agent_id: Format "AGT" + 5 digits
- listing_date: ISO date string (recent)
- images_count: Integer (5-30)

Create diverse, realistic listings with varied features and prices.
Use real neighborhood names for {area_code} area.

Return ONLY a JSON array of property objects, no additional text.
"""

        response = await self.llm_service.complete(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.8  # Higher temp for diversity
        )

        # Extract JSON from response
        return self._parse_json_response(response, expect_array=True)

    async def extract_property_details(self, mls_id: str) -> Dict[str, Any]:
        """Extract detailed property information for a specific MLS ID

        Args:
            mls_id: MLS identifier

        Returns:
            Detailed property data dictionary
        """
        prompt = f"""
Generate detailed property information for MLS ID {mls_id}.

Include all standard listing fields plus:
- detailed_description: 5-7 sentences with rich detail
- room_details: Array of objects with {{room_type, dimensions, features}}
- amenities: Array of 15-20 specific amenities
- appliances: Array of appliances included
- heating_cooling: Object with heating and cooling types
- parking: Object with {{type, spaces, description}}
- hoa_info: Object with {{fee_monthly, includes, restrictions}} or null
- tax_info: Object with {{annual_amount, year, assessment}}
- utilities: Array of utility information
- school_info: Array of {{name, level, rating, distance}}
- neighborhood_info: Object with {{walkability_score, transit_score, description}}
- recent_updates: Array of recent renovations/updates with years
- comparable_sales: Array of 3 recent comparable properties

Return ONLY a JSON object, no additional text.
"""

        response = await self.llm_service.complete(
            prompt=prompt,
            max_tokens=3000,
            temperature=0.6
        )

        return self._parse_json_response(response)

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

        Raises:
            ValueError: If JSON cannot be parsed
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
            raise ValueError(f"Failed to parse JSON: {e}\n\nResponse: {response[:500]}...")


# Global MLS extraction service instance
mls_service = MLSExtractionService()
