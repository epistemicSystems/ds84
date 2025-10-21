"""Script to populate database with sample property data"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.mls_extraction_service import mls_service
from app.services.embedding_service import embedding_service
from app.repositories.vector_repository import vector_repository


async def populate_database(
    area_codes: list[str] = None,
    properties_per_area: int = 50,
    verbose: bool = True
):
    """Populate database with sample properties

    Args:
        area_codes: List of ZIP codes to generate properties for
        properties_per_area: Number of properties to generate per area
        verbose: Whether to print progress
    """
    if area_codes is None:
        # Default to some California ZIP codes
        area_codes = ["95113", "95125", "94301", "94041", "94103"]

    if verbose:
        print(f"🚀 Starting database population...")
        print(f"📍 Area codes: {', '.join(area_codes)}")
        print(f"🏠 Properties per area: {properties_per_area}")
        print()

    total_properties = 0
    failed_properties = 0

    for area_code in area_codes:
        if verbose:
            print(f"📥 Extracting properties for {area_code}...")

        try:
            # Extract properties from MLS (simulated)
            properties = await mls_service.extract_listings(
                area_code=area_code,
                limit=properties_per_area
            )

            if verbose:
                print(f"   ✓ Extracted {len(properties)} properties")
                print(f"   🔢 Generating embeddings...")

            # Generate embeddings for all properties
            embeddings = await embedding_service.generate_property_embeddings_batch(
                properties
            )

            if verbose:
                print(f"   ✓ Generated {len(embeddings)} embeddings")
                print(f"   💾 Storing in database...")

            # Store properties with embeddings
            for property_data, embedding in zip(properties, embeddings):
                try:
                    await vector_repository.store_property(
                        mls_id=property_data['mls_id'],
                        data=property_data,
                        embedding=embedding
                    )
                    total_properties += 1
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Failed to store {property_data.get('mls_id')}: {e}")
                    failed_properties += 1

            if verbose:
                print(f"   ✅ Stored {len(properties)} properties for {area_code}")
                print()

        except Exception as e:
            if verbose:
                print(f"   ❌ Error processing {area_code}: {e}")
                print()
            failed_properties += properties_per_area

    # Print summary
    if verbose:
        print("=" * 60)
        print(f"📊 SUMMARY")
        print("=" * 60)
        print(f"Total properties stored: {total_properties}")
        print(f"Failed properties: {failed_properties}")
        print(f"Success rate: {(total_properties / (total_properties + failed_properties) * 100):.1f}%")
        print("=" * 60)

    return total_properties, failed_properties


async def verify_database(verbose: bool = True):
    """Verify database contents

    Args:
        verbose: Whether to print results
    """
    if verbose:
        print("\n🔍 Verifying database...")

    try:
        # Test vector search with a sample query
        test_query = "modern 3-bedroom house with pool"
        test_embedding = await embedding_service.generate_query_embedding(test_query)

        results = await vector_repository.search_by_vector(
            query_vector=test_embedding,
            limit=5
        )

        if verbose:
            print(f"\n✓ Database verification successful!")
            print(f"  Test query: '{test_query}'")
            print(f"  Found {len(results)} matching properties")

            if results:
                print(f"\n  Top result:")
                top_property, similarity = results[0]
                print(f"    MLS ID: {top_property.get('mls_id')}")
                print(f"    Address: {top_property.get('address', {}).get('street')}")
                print(f"    Price: ${top_property.get('price', 0):,}")
                print(f"    Similarity: {similarity:.3f}")

        return True

    except Exception as e:
        if verbose:
            print(f"\n❌ Database verification failed: {e}")
        return False


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Populate database with sample property data')
    parser.add_argument(
        '--areas',
        nargs='+',
        default=["95113", "95125", "94301", "94041", "94103"],
        help='ZIP codes to generate properties for'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help='Number of properties per area'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify database after population'
    )

    args = parser.parse_args()

    # Populate database
    total, failed = await populate_database(
        area_codes=args.areas,
        properties_per_area=args.count,
        verbose=not args.quiet
    )

    # Verify if requested
    if args.verify:
        await verify_database(verbose=not args.quiet)

    # Close database connection
    await vector_repository.close()

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
