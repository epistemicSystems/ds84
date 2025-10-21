"""Initialize database schema for REALTOR AI COPILOT"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.repositories.vector_repository import vector_repository


async def init_database():
    """Initialize database schema"""
    print("🔧 Initializing database schema...")
    print()

    try:
        # Initialize vector repository (creates tables and indexes)
        await vector_repository.initialize()

        print("✅ Database schema initialized successfully!")
        print()
        print("Tables created:")
        print("  - properties (with pgvector extension)")
        print("  - Indexes:")
        print("    - properties_embedding_idx (HNSW vector index)")
        print("    - properties_mls_id_idx (MLS ID index)")
        print()
        print("Ready to populate with data!")
        print("Run: python scripts/populate_database.py")

    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return 1

    finally:
        # Close connection
        await vector_repository.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(init_database())
    sys.exit(exit_code)
