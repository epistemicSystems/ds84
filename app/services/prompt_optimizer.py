"""Automated prompt optimization using meta-cognitive analysis"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service


@dataclass
class PromptAnalysis:
    """Analysis of a prompt's characteristics"""
    prompt_key: str
    token_count: int
    clarity_score: float
    specificity_score: float
    efficiency_score: float
    issues: List[str]
    strengths: List[str]
    optimization_opportunities: List[str]


@dataclass
class OptimizedPrompt:
    """Represents an optimized version of a prompt"""
    original_key: str
    optimized_content: str
    optimization_type: str  # "conciseness", "clarity", "structure", "comprehensive"
    token_reduction: int
    expected_improvements: List[str]
    confidence: float
    reasoning: str


class PromptOptimizer:
    """Automatically optimizes prompts using meta-cognitive analysis"""

    def __init__(self):
        """Initialize prompt optimizer"""
        self.llm_service = llm_service
        self.prompt_service = prompt_service

    async def analyze_prompt(
        self,
        prompt_key: str,
        variables: Dict[str, Any] = None
    ) -> PromptAnalysis:
        """Analyze a prompt to identify optimization opportunities

        Args:
            prompt_key: Prompt identifier
            variables: Template variables

        Returns:
            Prompt analysis
        """
        # Get prompt content
        prompt_content = self.prompt_service.get_prompt(
            prompt_key,
            **(variables or {})
        )

        # Estimate token count
        token_count = len(prompt_content.split()) * 1.3  # Rough estimate

        # Use LLM to analyze prompt
        analysis_prompt = f"""
You are an expert prompt engineer. Analyze the following prompt and provide a detailed assessment.

PROMPT TO ANALYZE:
```
{prompt_content}
```

Provide analysis in the following JSON format:
{{
  "clarity_score": 0.0-1.0,  // How clear and unambiguous the instructions are
  "specificity_score": 0.0-1.0,  // How specific the requirements are
  "efficiency_score": 0.0-1.0,  // How efficiently the prompt achieves its goal
  "issues": ["issue1", "issue2", ...],  // Problems or weaknesses
  "strengths": ["strength1", "strength2", ...],  // What works well
  "optimization_opportunities": ["opportunity1", "opportunity2", ...]  // How to improve
}}

Return ONLY the JSON, no additional text.
"""

        response = await self.llm_service.complete(
            prompt=analysis_prompt,
            temperature=0.3,
            max_tokens=1000
        )

        # Parse response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis_data = json.loads(json_str)

                return PromptAnalysis(
                    prompt_key=prompt_key,
                    token_count=int(token_count),
                    clarity_score=analysis_data.get("clarity_score", 0.0),
                    specificity_score=analysis_data.get("specificity_score", 0.0),
                    efficiency_score=analysis_data.get("efficiency_score", 0.0),
                    issues=analysis_data.get("issues", []),
                    strengths=analysis_data.get("strengths", []),
                    optimization_opportunities=analysis_data.get("optimization_opportunities", [])
                )
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            # Return default analysis
            return PromptAnalysis(
                prompt_key=prompt_key,
                token_count=int(token_count),
                clarity_score=0.5,
                specificity_score=0.5,
                efficiency_score=0.5,
                issues=[f"Analysis failed: {str(e)}"],
                strengths=[],
                optimization_opportunities=[]
            )

    async def optimize_prompt(
        self,
        prompt_key: str,
        optimization_type: str = "comprehensive",
        variables: Dict[str, Any] = None
    ) -> OptimizedPrompt:
        """Generate an optimized version of a prompt

        Args:
            prompt_key: Prompt identifier
            optimization_type: Type of optimization (conciseness, clarity, structure, comprehensive)
            variables: Template variables

        Returns:
            Optimized prompt
        """
        # Get original prompt
        original_prompt = self.prompt_service.get_prompt(
            prompt_key,
            **(variables or {})
        )

        original_tokens = len(original_prompt.split()) * 1.3

        # Analyze first
        analysis = await self.analyze_prompt(prompt_key, variables)

        # Generate optimization prompt based on type
        optimization_instructions = {
            "conciseness": "Reduce token count by at least 30% while preserving all critical instructions. Remove redundancy, use concise language, eliminate filler words.",
            "clarity": "Improve clarity and reduce ambiguity. Make instructions more explicit and unambiguous. Add structure if needed.",
            "structure": "Improve organization and structure. Use clear sections, bullet points, and formatting to enhance readability.",
            "comprehensive": "Perform comprehensive optimization: improve clarity, reduce token count, enhance structure, and fix any identified issues."
        }

        optimization_instruction = optimization_instructions.get(
            optimization_type,
            optimization_instructions["comprehensive"]
        )

        optimization_prompt = f"""
You are an expert prompt engineer. Optimize the following prompt according to these instructions:

OPTIMIZATION GOAL:
{optimization_instruction}

CURRENT ANALYSIS:
- Clarity Score: {analysis.clarity_score:.2f}
- Efficiency Score: {analysis.efficiency_score:.2f}
- Issues: {', '.join(analysis.issues)}
- Opportunities: {', '.join(analysis.optimization_opportunities)}

ORIGINAL PROMPT:
```
{original_prompt}
```

Provide your response in the following JSON format:
{{
  "optimized_prompt": "the optimized version of the prompt",
  "key_changes": ["change1", "change2", ...],
  "expected_improvements": ["improvement1", "improvement2", ...],
  "confidence": 0.0-1.0,
  "reasoning": "explanation of changes made"
}}

IMPORTANT:
1. Preserve all critical functionality and instructions
2. Maintain the same output format requirements
3. Keep all necessary context and examples
4. Only remove or modify content that is truly redundant or inefficient

Return ONLY the JSON, no additional text.
"""

        response = await self.llm_service.complete(
            prompt=optimization_prompt,
            temperature=0.3,
            max_tokens=2000,
            model="gpt-4"  # Use GPT-4 for high-quality optimization
        )

        # Parse response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                opt_data = json.loads(json_str)

                optimized_content = opt_data.get("optimized_prompt", original_prompt)
                optimized_tokens = len(optimized_content.split()) * 1.3
                token_reduction = int(original_tokens - optimized_tokens)

                return OptimizedPrompt(
                    original_key=prompt_key,
                    optimized_content=optimized_content,
                    optimization_type=optimization_type,
                    token_reduction=token_reduction,
                    expected_improvements=opt_data.get("expected_improvements", []),
                    confidence=opt_data.get("confidence", 0.7),
                    reasoning=opt_data.get("reasoning", "")
                )
            else:
                raise ValueError("No JSON in response")

        except Exception as e:
            # Return original as "optimized"
            return OptimizedPrompt(
                original_key=prompt_key,
                optimized_content=original_prompt,
                optimization_type=optimization_type,
                token_reduction=0,
                expected_improvements=[],
                confidence=0.0,
                reasoning=f"Optimization failed: {str(e)}"
            )

    async def batch_optimize_prompts(
        self,
        prompt_keys: List[str],
        optimization_type: str = "comprehensive"
    ) -> Dict[str, OptimizedPrompt]:
        """Optimize multiple prompts in batch

        Args:
            prompt_keys: List of prompt identifiers
            optimization_type: Type of optimization

        Returns:
            Dictionary mapping prompt keys to optimized versions
        """
        results = {}

        for prompt_key in prompt_keys:
            try:
                optimized = await self.optimize_prompt(
                    prompt_key=prompt_key,
                    optimization_type=optimization_type
                )
                results[prompt_key] = optimized

                # Brief delay to avoid rate limits
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Failed to optimize {prompt_key}: {e}")
                continue

        return results

    async def test_optimized_prompt(
        self,
        original_key: str,
        optimized_content: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test an optimized prompt against test cases

        Args:
            original_key: Original prompt key
            optimized_content: Optimized prompt content
            test_cases: Test cases to evaluate

        Returns:
            Test results
        """
        results = {
            "original_key": original_key,
            "test_cases_passed": 0,
            "test_cases_failed": 0,
            "test_results": []
        }

        for i, test_case in enumerate(test_cases):
            # Execute with optimized prompt
            response = await self.llm_service.complete(
                prompt=optimized_content.format(**test_case.get("variables", {})),
                temperature=test_case.get("temperature", 0.3),
                max_tokens=test_case.get("max_tokens", 1000)
            )

            # Evaluate response
            expected_behaviors = test_case.get("expected_behaviors", [])
            behaviors_met = sum(
                1 for behavior in expected_behaviors
                if behavior.lower() in response.lower()
            )

            passed = behaviors_met >= len(expected_behaviors) * 0.7  # 70% threshold

            results["test_results"].append({
                "test_case": i + 1,
                "passed": passed,
                "behaviors_met": behaviors_met,
                "behaviors_total": len(expected_behaviors),
                "response_length": len(response)
            })

            if passed:
                results["test_cases_passed"] += 1
            else:
                results["test_cases_failed"] += 1

        results["pass_rate"] = (
            results["test_cases_passed"] / len(test_cases)
            if test_cases else 0.0
        )

        return results

    def save_optimized_prompt(
        self,
        optimized: OptimizedPrompt,
        output_dir: str = "prompts_optimized"
    ) -> str:
        """Save optimized prompt to file

        Args:
            optimized: Optimized prompt
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{optimized.original_key.replace('.', '_')}_{timestamp}.txt"
        file_path = output_path / filename

        # Create header with metadata
        header = f"""# Optimized Prompt
# Original: {optimized.original_key}
# Optimization Type: {optimized.optimization_type}
# Token Reduction: {optimized.token_reduction}
# Confidence: {optimized.confidence:.2f}
# Generated: {datetime.utcnow().isoformat()}

# Expected Improvements:
{chr(10).join(f'# - {imp}' for imp in optimized.expected_improvements)}

# Reasoning:
# {optimized.reasoning}

# ============================================================================
# OPTIMIZED PROMPT CONTENT
# ============================================================================

"""

        content = header + optimized.optimized_content

        # Write to file
        with open(file_path, 'w') as f:
            f.write(content)

        return str(file_path)

    async def generate_optimization_report(
        self,
        prompt_keys: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report for multiple prompts

        Args:
            prompt_keys: List of prompt keys to analyze

        Returns:
            Optimization report
        """
        report = {
            "prompts_analyzed": len(prompt_keys),
            "analyses": {},
            "optimizations": {},
            "total_token_reduction": 0,
            "recommendations": [],
            "generated_at": datetime.utcnow().isoformat()
        }

        for prompt_key in prompt_keys:
            # Analyze
            analysis = await self.analyze_prompt(prompt_key)
            report["analyses"][prompt_key] = {
                "clarity_score": analysis.clarity_score,
                "specificity_score": analysis.specificity_score,
                "efficiency_score": analysis.efficiency_score,
                "token_count": analysis.token_count,
                "issues_count": len(analysis.issues),
                "issues": analysis.issues,
                "opportunities": analysis.optimization_opportunities
            }

            # Determine if optimization is needed
            needs_optimization = (
                analysis.efficiency_score < 0.7 or
                len(analysis.issues) > 0 or
                analysis.token_count > 1500
            )

            if needs_optimization:
                # Generate optimization
                optimized = await self.optimize_prompt(
                    prompt_key=prompt_key,
                    optimization_type="comprehensive"
                )

                report["optimizations"][prompt_key] = {
                    "token_reduction": optimized.token_reduction,
                    "confidence": optimized.confidence,
                    "expected_improvements": optimized.expected_improvements
                }

                report["total_token_reduction"] += optimized.token_reduction

                # Add recommendation
                if optimized.token_reduction > 100:
                    report["recommendations"].append({
                        "prompt": prompt_key,
                        "action": "Apply optimization",
                        "benefit": f"Reduce tokens by {optimized.token_reduction}",
                        "priority": "high" if optimized.token_reduction > 300 else "medium"
                    })

            # Brief delay
            await asyncio.sleep(0.5)

        # Generate summary
        report["summary"] = self._generate_summary(report)

        return report

    def _generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate summary of optimization report

        Args:
            report: Optimization report

        Returns:
            Summary string
        """
        total = report["prompts_analyzed"]
        optimizations = len(report["optimizations"])
        token_reduction = report["total_token_reduction"]
        high_priority = len([r for r in report["recommendations"] if r.get("priority") == "high"])

        parts = [
            f"Analyzed {total} prompts",
            f"{optimizations} prompts can be optimized",
            f"Total token reduction: {token_reduction}",
        ]

        if high_priority > 0:
            parts.append(f"{high_priority} high-priority optimizations")

        return " | ".join(parts)


# Global instance
prompt_optimizer = PromptOptimizer()
