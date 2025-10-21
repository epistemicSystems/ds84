"""Vector repository for property embeddings and similarity search"""
import asyncio
import asyncpg
import json
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings


class VectorRepository:
    """Repository for vector storage and similarity search using pgvector"""

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
                database=self.config.database,
                min_size=2,
                max_size=10
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
                WITH (lists = 100)
                ''')

                # Create index for MLS ID lookups
                await conn.execute('''
                CREATE INDEX IF NOT EXISTS properties_mls_id_idx
                ON properties(mls_id)
                ''')

    async def store_property(
        self,
        mls_id: str,
        data: Dict[str, Any],
        embedding: List[float] = None
    ) -> int:
        """Store property data and embedding

        Args:
            mls_id: MLS identifier
            data: Property data as dictionary
            embedding: Optional embedding vector

        Returns:
            Property ID
        """
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

    async def get_property_by_mls_id(self, mls_id: str) -> Optional[Dict[str, Any]]:
        """Get property by MLS ID

        Args:
            mls_id: MLS identifier

        Returns:
            Property data or None if not found
        """
        await self.initialize()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow('''
            SELECT data FROM properties WHERE mls_id = $1
            ''', mls_id)

            if result:
                return json.loads(result['data'])
            return None

    async def search_by_vector(
        self,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search properties by vector similarity

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of tuples (property_data, similarity_score)
        """
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

    async def search_by_filters(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search properties by JSONB filters

        Args:
            filters: Dictionary of filter conditions
            limit: Maximum number of results

        Returns:
            List of property data dictionaries
        """
        await self.initialize()

        # Build dynamic query conditions
        conditions = []
        params = []
        param_idx = 1

        for key, value in filters.items():
            # Handle nested JSON paths
            if '.' in key:
                path_parts = key.split('.')
                json_path = '->' + '->'.join(f"'{p}'" for p in path_parts[:-1])
                json_path += f"->>'{path_parts[-1]}'"
                conditions.append(f"(data{json_path})::text = ${param_idx}")
            else:
                conditions.append(f"data->>'{key}' = ${param_idx}")

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

    async def hybrid_search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Combined vector and filter search

        Args:
            query_vector: Query embedding vector
            filters: Optional filter conditions
            limit: Maximum number of results

        Returns:
            List of tuples (property_data, similarity_score)
        """
        await self.initialize()

        # Build dynamic query conditions for filters
        conditions = []
        params = [query_vector, limit]
        param_idx = 3

        if filters:
            for key, value in filters.items():
                if '.' in key:
                    path_parts = key.split('.')
                    json_path = '->' + '->'.join(f"'{p}'" for p in path_parts[:-1])
                    json_path += f"->>'{path_parts[-1]}'"
                    conditions.append(f"(data{json_path})::text = ${param_idx}")
                else:
                    conditions.append(f"data->>'{key}' = ${param_idx}")

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

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None


# Global vector repository instance
vector_repository = VectorRepository()
