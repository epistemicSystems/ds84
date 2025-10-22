"""Test script for cognitive workflow API endpoints"""
import asyncio
import httpx
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


async def test_list_workflows():
    """Test listing available workflows"""
    print("\n" + "=" * 80)
    print("TEST: List Available Workflows")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/workflows")

        print(f"Status: {response.status_code}")
        print(f"\nResponse:")
        result = response.json()
        print(json.dumps(result, indent=2))

        if response.status_code == 200:
            workflows = result.get("workflows", [])
            print(f"\n✅ Found {len(workflows)} workflows")
            for wf in workflows:
                print(f"  - {wf['name']} ({wf['id']}) - {wf['states_count']} states")
        else:
            print(f"\n❌ Failed to list workflows")

        return response.status_code == 200


async def test_execute_workflow():
    """Test executing a workflow"""
    print("\n" + "=" * 80)
    print("TEST: Execute Property Query Processing Workflow")
    print("=" * 80)

    # Test data
    payload = {
        "workflow_id": "property_query_processing",
        "input_data": {
            "query": "I'm looking for a modern 3-bedroom house with a pool in Silicon Valley, preferably with mountain views. Budget around $2 million."
        },
        "user_id": "test_user_123"
    }

    headers = {
        "X-Session-ID": "test_session_abc"
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/workflows/execute",
            json=payload,
            headers=headers
        )

        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Workflow executed successfully")
            print(f"\nExecution ID: {result.get('execution_id')}")
            print(f"Status: {result.get('status')}")

            # Print metrics
            if "metrics" in result and result["metrics"]:
                metrics = result["metrics"]
                print(f"\n📊 Metrics:")
                print(f"  Total Duration: {metrics.get('total_duration_seconds', 0):.2f}s")
                print(f"  Total Tokens: {metrics.get('total_tokens', 0)}")
                print(f"  Total Cost: ${metrics.get('total_cost', 0):.4f}")
                print(f"  States Executed: {len(metrics.get('state_executions', []))}")

            # Print result
            if "result" in result and result["result"]:
                print(f"\n📝 Result:")
                output = result["result"]

                # Print intent analysis
                if "intent" in output:
                    intent = output["intent"]
                    print(f"\n  Intent Analysis:")
                    print(f"    Property Types: {intent.get('property_types', [])}")
                    if "bedrooms" in intent:
                        print(f"    Bedrooms: {intent['bedrooms']}")
                    if "location" in intent:
                        print(f"    Location: {intent['location']}")
                    if "price_range" in intent:
                        print(f"    Price Range: ${intent['price_range'].get('min', 0):,} - ${intent['price_range'].get('max', 0):,}")

                # Print properties
                if "properties" in output:
                    properties = output["properties"]
                    print(f"\n  Found {len(properties)} properties:")
                    for i, prop in enumerate(properties[:3], 1):  # Show first 3
                        print(f"\n    Property {i}:")
                        print(f"      Address: {prop.get('address', {}).get('street', 'N/A')}")
                        print(f"      Price: ${prop.get('price', 0):,}")
                        print(f"      Beds/Baths: {prop.get('bedrooms', 'N/A')}/{prop.get('bathrooms', 'N/A')}")
                        if "similarity_score" in prop:
                            print(f"      Similarity: {prop['similarity_score']:.3f}")

                # Print response
                if "response" in output:
                    response_text = output["response"]
                    print(f"\n  Response Preview:")
                    print(f"    {response_text[:300]}...")

            return result.get("execution_id")
        else:
            print(f"\n❌ Workflow execution failed")
            print(f"Error: {response.text}")
            return None


async def test_get_execution_metrics(execution_id: str):
    """Test getting execution metrics"""
    print("\n" + "=" * 80)
    print("TEST: Get Execution Metrics")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/workflows/executions/{execution_id}/metrics"
        )

        print(f"Status: {response.status_code}")
        print(f"\nResponse:")
        result = response.json()
        print(json.dumps(result, indent=2))

        if response.status_code == 200:
            print(f"\n✅ Retrieved metrics successfully")
        else:
            print(f"\n❌ Failed to retrieve metrics")

        return response.status_code == 200


async def test_get_session_context(session_id: str):
    """Test getting session context"""
    print("\n" + "=" * 80)
    print("TEST: Get Session Context")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/context/sessions/{session_id}"
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Retrieved session context")
            print(f"\nSession ID: {result.get('session_id')}")
            print(f"User ID: {result.get('user_id')}")
            print(f"Message Count: {result.get('message_count')}")
            print(f"Created: {result.get('created_at')}")
            print(f"Last Activity: {result.get('last_activity')}")

            context_window = result.get('context_window', '')
            print(f"\nContext Window Preview:")
            print(f"{context_window[:500]}...")

            return True
        else:
            print(f"\n❌ Failed to retrieve session context")
            print(f"Error: {response.text}")
            return False


async def test_get_user_preferences(user_id: str):
    """Test getting user preferences"""
    print("\n" + "=" * 80)
    print("TEST: Get User Preferences")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/context/users/{user_id}/preferences"
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Retrieved user preferences")
            print(f"\nUser ID: {result.get('user_id')}")
            print(f"\nPreferences:")
            print(json.dumps(result.get('preferences', {}), indent=2))
            return True
        else:
            print(f"\n❌ Failed to retrieve user preferences")
            print(f"Error: {response.text}")
            return False


async def test_multi_turn_conversation():
    """Test multi-turn conversation with context"""
    print("\n" + "=" * 80)
    print("TEST: Multi-Turn Conversation with Context")
    print("=" * 80)

    session_id = "test_session_multi_turn"
    user_id = "test_user_456"

    queries = [
        "Show me 4-bedroom houses in Palo Alto",
        "I prefer something with a large backyard",
        "What about houses near good schools?"
    ]

    headers = {"X-Session-ID": session_id}

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n--- Turn {i} ---")
            print(f"Query: {query}")

            payload = {
                "workflow_id": "property_query_processing",
                "input_data": {"query": query},
                "user_id": user_id
            }

            response = await client.post(
                f"{BASE_URL}/api/workflows/execute",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Turn {i} completed")
                print(f"Execution ID: {result.get('execution_id')}")

                # Brief delay between turns
                await asyncio.sleep(1)
            else:
                print(f"❌ Turn {i} failed: {response.text}")
                return False

        # Check session context after conversation
        print(f"\n--- Checking Final Session Context ---")
        context_response = await client.get(
            f"{BASE_URL}/api/context/sessions/{session_id}"
        )

        if context_response.status_code == 200:
            context = context_response.json()
            print(f"\n✅ Session has {context.get('message_count')} messages")
            return True
        else:
            print(f"\n❌ Failed to retrieve session context")
            return False


async def run_all_tests():
    """Run all workflow API tests"""
    print("\n" + "=" * 80)
    print("COGNITIVE WORKFLOW API TEST SUITE")
    print("=" * 80)
    print(f"\nBase URL: {BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  python app/main.py")
    print("\n" + "=" * 80)

    # Wait for user confirmation
    input("\nPress Enter to start tests...")

    results = {}

    # Test 1: List workflows
    results["list_workflows"] = await test_list_workflows()

    # Test 2: Execute workflow
    execution_id = await test_execute_workflow()
    results["execute_workflow"] = execution_id is not None

    # Test 3: Get execution metrics (if execution succeeded)
    if execution_id:
        results["get_metrics"] = await test_get_execution_metrics(execution_id)

    # Test 4: Get session context
    results["get_session"] = await test_get_session_context("test_session_abc")

    # Test 5: Get user preferences
    results["get_preferences"] = await test_get_user_preferences("test_user_123")

    # Test 6: Multi-turn conversation
    results["multi_turn"] = await test_multi_turn_conversation()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\nPassed: {passed}/{total}")
    print("=" * 80)

    return passed == total


async def main():
    """Main entry point"""
    try:
        success = await run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
