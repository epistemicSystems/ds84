"""Embedding service for generating property embeddings"""
from typing import List, Dict, Any
from app.services.llm_service import llm_service


class EmbeddingService:
    """Service for generating embeddings from property data"""

    def __init__(self):
        self.llm_service = llm_service

    async def generate_property_embedding(
        self,
        property_data: Dict[str, Any]
    ) -> List[float]:
        """Generate embedding for a property

        Args:
            property_data: Property data dictionary

        Returns:
            Embedding vector
        """
        # Combine relevant fields into a text representation
        text = self._property_to_text(property_data)

        # Generate embedding
        embedding = await self.llm_service.generate_embedding(text)

        return embedding

    async def generate_property_embeddings_batch(
        self,
        properties: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Generate embeddings for multiple properties

        Args:
            properties: List of property data dictionaries

        Returns:
            List of embedding vectors
        """
        # Convert properties to text representations
        texts = [self._property_to_text(prop) for prop in properties]

        # Generate embeddings in batch
        embeddings = await self.llm_service.generate_embeddings_batch(texts)

        return embeddings

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query

        Args:
            query: Natural language query

        Returns:
            Embedding vector
        """
        return await self.llm_service.generate_embedding(query)

    def _property_to_text(self, property_data: Dict[str, Any]) -> str:
        """Convert property data to text representation for embedding

        Args:
            property_data: Property data dictionary

        Returns:
            Text representation
        """
        parts = []

        # Add property type
        if 'property_type' in property_data:
            parts.append(f"{property_data['property_type']}")

        # Add bedrooms and bathrooms
        if 'bedrooms' in property_data:
            parts.append(f"{property_data['bedrooms']} bedrooms")
        if 'bathrooms' in property_data:
            parts.append(f"{property_data['bathrooms']} bathrooms")

        # Add square footage
        if 'square_footage' in property_data:
            parts.append(f"{property_data['square_footage']} square feet")

        # Add location
        if 'address' in property_data:
            addr = property_data['address']
            if isinstance(addr, dict):
                location_parts = []
                if 'city' in addr:
                    location_parts.append(addr['city'])
                if 'neighborhood' in addr:
                    location_parts.append(addr['neighborhood'])
                if location_parts:
                    parts.append(f"in {', '.join(location_parts)}")
            else:
                parts.append(f"at {addr}")

        # Add key features
        if 'key_features' in property_data:
            features = property_data['key_features']
            if isinstance(features, list):
                parts.append(f"Features: {', '.join(features)}")

        # Add description
        if 'description' in property_data:
            parts.append(property_data['description'])

        # Add price (for context)
        if 'price' in property_data:
            parts.append(f"Priced at ${property_data['price']:,}")

        return ". ".join(parts)


# Global embedding service instance
embedding_service = EmbeddingService()
