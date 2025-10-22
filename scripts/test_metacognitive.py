"""Test script for Level 4 meta-cognitive optimization features"""
import asyncio
import httpx
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


async def test_performance_analysis():
    """Test performance analysis endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Performance Analysis")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            f"{BASE_URL}/api/metacognitive/performance/property_query_processing",
            params={
                "time_window_hours": 24,
                "min_executions": 1
            }
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Performance Analysis Complete")
            print(f"\nWorkflow: {result.get('workflow_id')}")
            print(f"Status: {result.get('status')}")

            if result.get("status") == "analyzed":
                print(f"Health Score: {result.get('health_score', 0):.2f}")
                print(f"Executions Analyzed: {result.get('executions_analyzed')}")

                bottlenecks = result.get("bottlenecks", [])
                print(f"\n📊 Bottlenecks Found: {len(bottlenecks)}")
                for i, b in enumerate(bottlenecks[:3], 1):
                    print(f"\n  {i}. State: {b['state_id']}")
                    print(f"     Metric: {b['metric_type']}")
                    print(f"     Severity: {b['severity']}")
                    print(f"     Impact: {b['impact_score']:.2f}")

                opportunities = result.get("opportunities", [])
                print(f"\n💡 Optimization Opportunities: {len(opportunities)}")
                for i, o in enumerate(opportunities[:3], 1):
                    print(f"\n  {i}. Type: {o['opportunity_type']}")
                    print(f"     Target: {o['target']}")
                    print(f"     Priority: {o['priority']}")

                recs = result.get("recommendations", [])
                if recs:
                    print(f"\n📝 Top Recommendations:")
                    for rec in recs[:3]:
                        print(f"  - {rec}")

            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False


async def test_adaptive_routing():
    """Test adaptive routing endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Adaptive Routing")
    print("=" * 80)

    strategies = ["performance", "cost", "quality", "balanced"]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for strategy in strategies:
            print(f"\n--- Testing Strategy: {strategy} ---")

            payload = {
                "workflow_id": "property_query_processing",
                "state_id": "intent_analysis",
                "context": {
                    "task_complexity": "medium",
                    "estimated_tokens": 1000
                },
                "strategy": strategy
            }

            response = await client.post(
                f"{BASE_URL}/api/metacognitive/route",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model: {result['selected_model']}")
                print(f"   Temperature: {result['selected_temperature']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Reasoning: {result['reasoning']}")

                est = result.get("estimated_metrics", {})
                if est:
                    print(f"   Est. Cost: ${est.get('cost', 0):.4f}")
                    print(f"   Est. Duration: {est.get('duration', 0):.2f}s")
                    print(f"   Est. Quality: {est.get('quality', 0):.2f}")
            else:
                print(f"❌ Failed: {response.text}")
                return False

        return True


async def test_prompt_optimization():
    """Test prompt optimization endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Prompt Optimization")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "prompt_key": "property_search.intent_analysis",
            "optimization_type": "comprehensive",
            "variables": {
                "query": "Find me a 3-bedroom house"
            }
        }

        print(f"\nOptimizing prompt: {payload['prompt_key']}")
        print(f"Optimization type: {payload['optimization_type']}")

        response = await client.post(
            f"{BASE_URL}/api/metacognitive/optimize-prompt",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prompt Optimized")
            print(f"\nToken Reduction: {result['token_reduction']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"\nExpected Improvements:")
            for imp in result.get("expected_improvements", [])[:5]:
                print(f"  - {imp}")

            print(f"\nReasoning: {result.get('reasoning', 'N/A')[:200]}...")

            # Show a preview of optimized content
            opt_content = result.get("optimized_content", "")
            print(f"\nOptimized Content Preview:")
            print(f"{opt_content[:300]}...")

            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False


async def test_self_improvement():
    """Test self-improvement cycle endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Self-Improvement Cycle")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "workflow_id": "property_query_processing",
            "time_window_hours": 24,
            "dry_run": True  # Don't actually apply changes
        }

        print(f"\nRunning improvement cycle (dry run)")
        print(f"Workflow: {payload['workflow_id']}")

        response = await client.post(
            f"{BASE_URL}/api/metacognitive/self-improve",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Improvement Cycle Complete")
            print(f"\nCycle ID: {result.get('cycle_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Duration: {result.get('duration_seconds', 0):.2f}s")

            baseline = result.get("baseline_metrics", {})
            if baseline:
                print(f"\n📊 Baseline Metrics:")
                print(f"  Health Score: {baseline.get('health_score', 0):.2f}")
                print(f"  Avg Duration: {baseline.get('average_duration', 0):.2f}s")
                print(f"  Avg Cost: ${baseline.get('average_cost', 0):.4f}")

            print(f"\nOptimizations Identified: {result.get('optimizations_identified', 0)}")
            print(f"Optimizations Validated: {result.get('optimizations_validated', 0)}")

            improvements = result.get("improvements", [])
            if improvements:
                print(f"\n💡 Improvements:")
                for imp in improvements:
                    print(f"  - {imp}")

            actions = result.get("actions_taken", [])
            if actions:
                print(f"\n⚡ Actions (would be taken if not dry run):")
                for action in actions[:5]:
                    print(f"  - {action.get('action', str(action))}")

            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False


async def test_cost_quality_optimization():
    """Test cost/quality optimization endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Cost/Quality Optimization")
    print("=" * 80)

    objectives = [
        ("minimize_cost", {}),
        ("maximize_quality", {}),
        ("balanced", {}),
        ("cost_constrained", {"max_cost": 0.02, "min_quality": 0.7}),
        ("quality_constrained", {"min_quality": 0.9})
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for objective, constraints in objectives:
            print(f"\n--- Objective: {objective} ---")

            payload = {
                "objective": objective,
                "context": {
                    "estimated_tokens": 1500,
                    "task_complexity": "medium"
                },
                "constraints": constraints
            }

            response = await client.post(
                f"{BASE_URL}/api/metacognitive/cost-quality",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                config = result.get("configuration", {})
                print(f"✅ Model: {config.get('model')}")
                print(f"   Cost: ${result.get('estimated_cost', 0):.4f}")
                print(f"   Quality: {result.get('estimated_quality', 0):.2f}")
                print(f"   Duration: {result.get('estimated_duration', 0):.2f}s")
                print(f"   Efficiency: {result.get('efficiency_score', 0):.2f}")
                print(f"   Reason: {result.get('recommendation_reason', 'N/A')}")
            else:
                print(f"❌ Failed: {response.text}")
                return False

        return True


async def test_cost_quality_analysis():
    """Test cost/quality tradeoff analysis"""
    print("\n" + "=" * 80)
    print("TEST: Cost/Quality Tradeoff Analysis")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            f"{BASE_URL}/api/metacognitive/cost-quality/analyze",
            params={
                "estimated_tokens": 1000,
                "task_complexity": "medium"
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Tradeoff Analysis Complete")

            print(f"\nTotal Configurations: {result.get('total_configurations')}")
            print(f"Pareto Optimal: {result.get('pareto_optimal_configurations')}")

            cost_range = result.get("cost_range", {})
            print(f"\nCost Range: ${cost_range.get('min', 0):.4f} - ${cost_range.get('max', 0):.4f}")

            quality_range = result.get("quality_range", {})
            print(f"Quality Range: {quality_range.get('min', 0):.2f} - {quality_range.get('max', 0):.2f}")

            pareto = result.get("pareto_frontier", [])
            print(f"\n🎯 Pareto Frontier ({len(pareto)} configurations):")
            for config in pareto:
                print(f"\n  Model: {config['model']}")
                print(f"    Cost: ${config['cost']:.4f}")
                print(f"    Quality: {config['quality']:.2f}")
                print(f"    Efficiency: {config['efficiency']:.2f}")
                print(f"    Best For: {config['reason']}")

            recs = result.get("recommendations", {})
            if recs:
                print(f"\n📝 Recommendations:")
                print(f"  Minimum Cost: {recs.get('minimum_cost', {}).get('model')}")
                print(f"  Maximum Quality: {recs.get('maximum_quality', {}).get('model')}")
                print(f"  Best Efficiency: {recs.get('best_efficiency', {}).get('model')}")

            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False


async def test_improvement_history():
    """Test improvement history endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Improvement History")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            f"{BASE_URL}/api/metacognitive/improvement-history"
        )

        if response.status_code == 200:
            result = response.json()
            history = result.get("history", [])

            print(f"\n✅ Retrieved Improvement History")
            print(f"Total Cycles: {len(history)}")

            if history:
                print(f"\nRecent Cycles:")
                for cycle in history[:5]:
                    print(f"\n  Cycle: {cycle.get('cycle_id')}")
                    print(f"    Workflow: {cycle.get('workflow_id')}")
                    print(f"    Status: {cycle.get('status')}")
                    print(f"    Started: {cycle.get('started_at')}")
                    print(f"    Improvements: {cycle.get('improvements_count')}")
                    print(f"    Actions: {cycle.get('actions_count')}")
            else:
                print("\n  No improvement cycles yet")

            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False


async def run_all_tests():
    """Run all meta-cognitive tests"""
    print("\n" + "=" * 80)
    print("META-COGNITIVE OPTIMIZATION TEST SUITE (Level 4)")
    print("=" * 80)
    print(f"\nBase URL: {BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  python app/main.py")
    print("\n" + "=" * 80)

    input("\nPress Enter to start tests...")

    results = {}

    # Test 1: Performance Analysis
    results["performance_analysis"] = await test_performance_analysis()

    # Test 2: Adaptive Routing
    results["adaptive_routing"] = await test_adaptive_routing()

    # Test 3: Prompt Optimization
    results["prompt_optimization"] = await test_prompt_optimization()

    # Test 4: Self-Improvement Cycle
    results["self_improvement"] = await test_self_improvement()

    # Test 5: Cost/Quality Optimization
    results["cost_quality_optimization"] = await test_cost_quality_optimization()

    # Test 6: Cost/Quality Analysis
    results["cost_quality_analysis"] = await test_cost_quality_analysis()

    # Test 7: Improvement History
    results["improvement_history"] = await test_improvement_history()

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
