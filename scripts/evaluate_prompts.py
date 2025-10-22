"""Prompt evaluation framework for cognitive workflows"""
import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service


class PromptEvaluator:
    """Evaluates prompt quality and effectiveness"""

    def __init__(self):
        self.llm_service = llm_service
        self.prompt_service = prompt_service

    async def evaluate_prompt(
        self,
        prompt_key: str,
        test_cases: List[Dict[str, Any]],
        evaluation_criteria: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a prompt against test cases

        Args:
            prompt_key: Prompt key (e.g., "property_search.intent_analysis")
            test_cases: List of test inputs and expected behaviors
            evaluation_criteria: List of criteria to evaluate

        Returns:
            Evaluation results
        """
        print(f"\n{'=' * 80}")
        print(f"EVALUATING PROMPT: {prompt_key}")
        print(f"{'=' * 80}")

        results = {
            "prompt_key": prompt_key,
            "test_cases": [],
            "summary": {
                "total_cases": len(test_cases),
                "passed": 0,
                "failed": 0,
                "average_score": 0.0
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        total_score = 0.0

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}/{len(test_cases)} ---")
            print(f"Input: {test_case.get('description', 'No description')}")

            # Generate prompt
            prompt = self.prompt_service.get_prompt(
                prompt_key,
                **test_case.get('variables', {})
            )

            # Execute prompt
            response = await self.llm_service.complete(
                prompt=prompt,
                temperature=test_case.get('temperature', 0.3),
                max_tokens=test_case.get('max_tokens', 1000)
            )

            # Evaluate response
            evaluation = await self._evaluate_response(
                response=response,
                test_case=test_case,
                criteria=evaluation_criteria
            )

            test_result = {
                "case_number": i,
                "input": test_case.get('variables', {}),
                "response_length": len(response),
                "evaluation": evaluation,
                "passed": evaluation['overall_score'] >= 0.7
            }

            results["test_cases"].append(test_result)

            if test_result["passed"]:
                results["summary"]["passed"] += 1
                print(f"✅ PASSED (Score: {evaluation['overall_score']:.2f})")
            else:
                results["summary"]["failed"] += 1
                print(f"❌ FAILED (Score: {evaluation['overall_score']:.2f})")

            total_score += evaluation['overall_score']

        # Calculate average
        if len(test_cases) > 0:
            results["summary"]["average_score"] = total_score / len(test_cases)

        print(f"\n{'=' * 80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total Cases: {results['summary']['total_cases']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Average Score: {results['summary']['average_score']:.2f}")
        print(f"{'=' * 80}\n")

        return results

    async def _evaluate_response(
        self,
        response: str,
        test_case: Dict[str, Any],
        criteria: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a response against criteria

        Args:
            response: LLM response
            test_case: Test case with expected behaviors
            criteria: Evaluation criteria

        Returns:
            Evaluation scores
        """
        # Build evaluation prompt
        eval_prompt = f"""
You are an expert evaluator of AI system outputs. Evaluate the following response against the specified criteria.

Test Case Description: {test_case.get('description', 'N/A')}

Expected Behaviors:
{json.dumps(test_case.get('expected_behaviors', []), indent=2)}

Actual Response:
{response}

Evaluation Criteria:
{chr(10).join(f"{i+1}. {criterion}" for i, criterion in enumerate(criteria))}

For each criterion, provide:
1. Score (0.0 to 1.0)
2. Reasoning (brief explanation)
3. Suggestions for improvement (if score < 1.0)

Return your evaluation as a JSON object:
{{
  "criteria_scores": {{
    "criterion_name": {{
      "score": 0.0-1.0,
      "reasoning": "explanation",
      "suggestions": ["suggestion1", "suggestion2"]
    }}
  }},
  "overall_score": 0.0-1.0,
  "summary": "brief summary of evaluation"
}}

Return ONLY the JSON object, no additional text.
"""

        # Get evaluation from LLM
        eval_response = await self.llm_service.complete(
            prompt=eval_prompt,
            temperature=0.2,  # Low temperature for consistent evaluation
            max_tokens=1500
        )

        # Parse JSON
        try:
            json_start = eval_response.find('{')
            json_end = eval_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = eval_response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    "overall_score": 0.0,
                    "error": "Failed to parse evaluation response"
                }
        except json.JSONDecodeError as e:
            return {
                "overall_score": 0.0,
                "error": f"JSON parse error: {e}"
            }

    async def compare_prompt_versions(
        self,
        prompt_key_v1: str,
        prompt_key_v2: str,
        test_cases: List[Dict[str, Any]],
        evaluation_criteria: List[str]
    ) -> Dict[str, Any]:
        """Compare two versions of a prompt

        Args:
            prompt_key_v1: First prompt version
            prompt_key_v2: Second prompt version
            test_cases: Test cases
            evaluation_criteria: Evaluation criteria

        Returns:
            Comparison results
        """
        print(f"\n{'=' * 80}")
        print(f"COMPARING PROMPT VERSIONS")
        print(f"{'=' * 80}")
        print(f"Version 1: {prompt_key_v1}")
        print(f"Version 2: {prompt_key_v2}")
        print(f"{'=' * 80}")

        # Evaluate both versions
        results_v1 = await self.evaluate_prompt(
            prompt_key_v1,
            test_cases,
            evaluation_criteria
        )

        results_v2 = await self.evaluate_prompt(
            prompt_key_v2,
            test_cases,
            evaluation_criteria
        )

        # Compare results
        comparison = {
            "version_1": {
                "key": prompt_key_v1,
                "average_score": results_v1["summary"]["average_score"],
                "passed": results_v1["summary"]["passed"]
            },
            "version_2": {
                "key": prompt_key_v2,
                "average_score": results_v2["summary"]["average_score"],
                "passed": results_v2["summary"]["passed"]
            },
            "improvement": {
                "score_delta": results_v2["summary"]["average_score"] - results_v1["summary"]["average_score"],
                "passed_delta": results_v2["summary"]["passed"] - results_v1["summary"]["passed"]
            }
        }

        # Print comparison
        print(f"\n{'=' * 80}")
        print(f"COMPARISON RESULTS")
        print(f"{'=' * 80}")
        print(f"\nVersion 1 ({prompt_key_v1}):")
        print(f"  Average Score: {comparison['version_1']['average_score']:.2f}")
        print(f"  Cases Passed: {comparison['version_1']['passed']}")

        print(f"\nVersion 2 ({prompt_key_v2}):")
        print(f"  Average Score: {comparison['version_2']['average_score']:.2f}")
        print(f"  Cases Passed: {comparison['version_2']['passed']}")

        print(f"\nImprovement:")
        delta = comparison['improvement']['score_delta']
        if delta > 0:
            print(f"  Score: +{delta:.2f} ✅ (Version 2 better)")
        elif delta < 0:
            print(f"  Score: {delta:.2f} ❌ (Version 1 better)")
        else:
            print(f"  Score: {delta:.2f} (No change)")

        print(f"  Cases: {comparison['improvement']['passed_delta']:+d}")
        print(f"{'=' * 80}\n")

        return comparison

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_file: str
    ):
        """Save evaluation results to file

        Args:
            results: Evaluation results
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Results saved to: {output_file}")


async def evaluate_intent_analysis_prompt():
    """Evaluate the intent analysis prompt"""
    evaluator = PromptEvaluator()

    # Define test cases
    test_cases = [
        {
            "description": "Simple property query with explicit requirements",
            "variables": {
                "query": "I need a 3-bedroom house with a pool in San Jose"
            },
            "expected_behaviors": [
                "Extract bedrooms: 3",
                "Extract property_type: house",
                "Extract must_have_features: pool",
                "Extract location: San Jose",
                "High confidence on explicit attributes"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        },
        {
            "description": "Complex query with implicit preferences",
            "variables": {
                "query": "Looking for a cozy family home near good schools, walkable neighborhood, budget around $1.5M"
            },
            "expected_behaviors": [
                "Infer property suitable for families",
                "Extract proximity_features: schools",
                "Extract implied_preferences: walkable, family-friendly",
                "Extract price_range: around 1.5M",
                "Medium confidence on inferred attributes"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        },
        {
            "description": "Ambiguous query requiring clarification",
            "variables": {
                "query": "I want something modern"
            },
            "expected_behaviors": [
                "Identify ambiguity: 'modern' can mean style or age",
                "Provide recommendations_for_clarification",
                "Lower confidence scores",
                "Consider multiple interpretations"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        },
        {
            "description": "Multi-constraint query with trade-offs",
            "variables": {
                "query": "4+ bedrooms, under $800k, close to downtown but quiet neighborhood"
            },
            "expected_behaviors": [
                "Extract bedrooms min: 4",
                "Extract price_range max: 800k",
                "Identify potential trade-off: close to downtown vs quiet",
                "Note competing requirements in reasoning_trace"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        },
        {
            "description": "Lifestyle-focused query",
            "variables": {
                "query": "I work from home and love to entertain, need outdoor space and home office"
            },
            "expected_behaviors": [
                "Extract must_have_features: outdoor space, home office",
                "Infer implied_preferences: suitable for entertaining",
                "Assess confidence appropriately",
                "Consider space requirements"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
    ]

    # Define evaluation criteria
    criteria = [
        "Completeness: All explicit attributes extracted",
        "Accuracy: Attributes match user intent",
        "Inference Quality: Reasonable implicit preferences identified",
        "Confidence Assessment: Appropriate confidence levels",
        "Metacognition: Reasoning trace provided and meaningful",
        "Ambiguity Handling: Ambiguities identified and clarifications suggested",
        "JSON Compliance: Valid JSON structure returned"
    ]

    # Evaluate prompt
    results = await evaluator.evaluate_prompt(
        prompt_key="property_search.intent_analysis",
        test_cases=test_cases,
        evaluation_criteria=criteria
    )

    # Save results
    evaluator.save_evaluation_results(
        results,
        "evaluation_results/intent_analysis_evaluation.json"
    )

    return results


async def compare_intent_analysis_versions():
    """Compare v1 and v2 of intent analysis prompt"""
    evaluator = PromptEvaluator()

    # Test cases
    test_cases = [
        {
            "description": "Standard property query",
            "variables": {
                "query": "3-bedroom house with pool in Palo Alto, under $2M"
            },
            "expected_behaviors": [
                "Extract all attributes",
                "Appropriate confidence levels"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        },
        {
            "description": "Ambiguous query",
            "variables": {
                "query": "Modern home near tech companies"
            },
            "expected_behaviors": [
                "Identify ambiguities",
                "Provide clarifications",
                "Consider alternatives"
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
    ]

    criteria = [
        "Completeness",
        "Accuracy",
        "Metacognitive Awareness",
        "JSON Compliance"
    ]

    # Compare versions
    comparison = await evaluator.compare_prompt_versions(
        prompt_key_v1="property_search.intent_analysis",  # v1
        prompt_key_v2="property_search.intent_analysis",  # v2 (same for now)
        test_cases=test_cases,
        evaluation_criteria=criteria
    )

    return comparison


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate prompts for cognitive workflows')
    parser.add_argument(
        'command',
        choices=['evaluate', 'compare'],
        help='Command to execute'
    )
    parser.add_argument(
        '--prompt',
        help='Prompt key to evaluate'
    )

    args = parser.parse_args()

    if args.command == 'evaluate':
        if args.prompt == 'intent_analysis':
            await evaluate_intent_analysis_prompt()
        else:
            print(f"No evaluation defined for prompt: {args.prompt}")
            print("Available: intent_analysis")
            return 1
    elif args.command == 'compare':
        await compare_intent_analysis_versions()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
