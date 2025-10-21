"""Test script for REALTOR AI COPILOT API"""
import asyncio
import httpx
import json
import sys


API_BASE_URL = "http://localhost:8000"


async def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("1. Testing Health Endpoint")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/health")

        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()

        return response.status_code == 200


async def test_property_search_simulated():
    """Test property search with simulated mode"""
    print("=" * 60)
    print("2. Testing Property Search (Simulated Mode)")
    print("=" * 60)

    query = "I need a modern 3-bedroom home with a pool and a view, budget around $800k"

    print(f"Query: {query}")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/property-search",
            json={
                "query": query,
                "search_mode": "simulated",
                "limit": 5
            }
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Search Mode: {result.get('search_mode')}")
            print(f"Intent: {json.dumps(result.get('intent', {}), indent=2)}")
            print(f"Properties Found: {len(result.get('properties', []))}")
            print()
            print("Response:")
            print(result.get('response', 'No response'))
        else:
            print(f"Error: {response.text}")

        print()
        return response.status_code == 200


async def test_property_search_vector():
    """Test property search with vector mode"""
    print("=" * 60)
    print("3. Testing Property Search (Vector Mode)")
    print("=" * 60)

    query = "Find me a spacious house with good schools nearby and a large backyard"

    print(f"Query: {query}")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/property-search",
            json={
                "query": query,
                "search_mode": "vector",
                "limit": 5
            }
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Search Mode: {result.get('search_mode')}")
            print(f"Properties Found: {len(result.get('properties', []))}")

            # Show top 3 properties with similarity scores
            properties = result.get('properties', [])
            if properties:
                print()
                print("Top Properties:")
                for i, prop in enumerate(properties[:3], 1):
                    print(f"{i}. {prop.get('address', {}).get('street', 'N/A')}")
                    print(f"   Price: ${prop.get('price', 0):,}")
                    print(f"   Beds/Baths: {prop.get('bedrooms')}/{prop.get('bathrooms')}")
                    print(f"   Similarity: {prop.get('similarity_score', 'N/A')}")
                    print()
        else:
            print(f"Error: {response.text}")

        print()
        return response.status_code == 200


async def test_agent_analysis():
    """Test agent analysis endpoint"""
    print("=" * 60)
    print("4. Testing Agent Analysis")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/agent-analysis",
            json={
                "agent_id": "AGT12345",
                "area_codes": ["95113", "95125"]
            }
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Agent ID: {result.get('agent_id')}")

            if 'performance_analysis' in result:
                perf = result['performance_analysis']
                summary = perf.get('performance_summary', {})
                print()
                print("Performance Summary:")
                print(f"  Assessment: {summary.get('overall_assessment', 'N/A')[:100]}...")
                print(f"  Key Strengths: {', '.join(summary.get('key_strengths', [])[:2])}")

            print()
            print("Strategic Insights:")
            insights = result.get('strategic_insights', '')
            print(insights[:300] + "..." if len(insights) > 300 else insights)
        else:
            print(f"Error: {response.text}")

        print()
        return response.status_code == 200


async def main():
    """Run all tests"""
    print()
    print("🧪 REALTOR AI COPILOT - API Test Suite")
    print()

    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            await client.get(API_BASE_URL, timeout=2.0)
    except Exception:
        print("❌ ERROR: API server is not running!")
        print()
        print("Please start the server first:")
        print("  uvicorn app.main:app --reload")
        print()
        return 1

    # Run tests
    results = []

    results.append(await test_health())
    results.append(await test_property_search_simulated())

    # Ask user if they want to test vector search (requires database)
    print("=" * 60)
    print("Vector Search Test")
    print("=" * 60)
    print("Vector search requires a populated database.")
    print("Have you run 'python scripts/populate_database.py'? (y/n): ", end="")

    user_input = input().strip().lower()
    if user_input == 'y':
        results.append(await test_property_search_vector())
    else:
        print("Skipping vector search test.")
        print()

    results.append(await test_agent_analysis())

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total * 100):.0f}%")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
