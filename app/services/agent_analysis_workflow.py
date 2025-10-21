"""Agent analysis workflow for competitive intelligence"""
import json
from typing import Dict, List, Any, Optional
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service


class AgentAnalysisWorkflow:
    """Workflow for analyzing real estate agent performance"""

    def __init__(self):
        self.llm_service = llm_service
        self.prompt_service = prompt_service

    async def execute(
        self,
        agent_id: str,
        area_codes: List[str] = None,
        comparison_agent_ids: List[str] = None
    ) -> Dict[str, Any]:
        """Execute the agent analysis workflow

        Args:
            agent_id: Agent identifier
            area_codes: List of area codes for market context
            comparison_agent_ids: Optional list of agent IDs for comparison

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Step 1: Extract agent data (simulated for prototype)
            agent_data = await self._generate_agent_data(agent_id)

            # Step 2: Extract market context
            if not area_codes and 'primary_areas' in agent_data:
                area_codes = agent_data['primary_areas']
            elif not area_codes:
                area_codes = ['95113']  # Default area code

            market_context = await self._generate_market_context(area_codes)

            # Step 3: Extract comparison agent data (if requested)
            comparison_agents = None
            if comparison_agent_ids:
                comparison_agents = []
                for comp_id in comparison_agent_ids:
                    comp_data = await self._generate_agent_data(comp_id)
                    comparison_agents.append(comp_data)

            # Step 4: Perform performance analysis
            performance_analysis = await self._analyze_performance(
                agent_data, market_context, comparison_agents
            )

            # Step 5: Generate strategic insights
            insights = await self._generate_insights(
                performance_analysis, agent_data, market_context
            )

            return {
                "agent_id": agent_id,
                "agent_data": agent_data,
                "market_context": market_context,
                "performance_analysis": performance_analysis,
                "strategic_insights": insights,
                "status": "success"
            }
        except Exception as e:
            print(f"Agent analysis workflow error: {e}")
            return {
                "agent_id": agent_id,
                "error": str(e),
                "status": "error"
            }

    async def _generate_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """Generate sample agent data for prototype

        In production, this would fetch real agent data from MLS or database.
        """
        prompt = f"""
Generate realistic real estate agent profile information for agent ID {agent_id}.

Include:
- name, brokerage, and contact information
- years_of_experience (integer)
- license_number
- specializations (array of strings)
- transaction_history for past 24 months:
  - total_listings (integer)
  - total_sales (integer)
  - average_list_price (integer)
  - average_sale_price (integer)
  - average_days_on_market (integer)
  - list_to_sale_ratio (float, e.g., 0.98)
- geographic_focus (array of area codes)
- price_tier_focus (string: "luxury", "mid-range", "starter", "mixed")
- client_testimonial_count (integer)
- professional_certifications (array of strings)

Return ONLY a JSON object, no additional text.
"""

        response = await self.llm_service.complete(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.6
        )

        return self._parse_json_response(response)

    async def _generate_market_context(self, area_codes: List[str]) -> Dict[str, Any]:
        """Generate market context data for benchmark comparison

        In production, this would fetch real market data.
        """
        prompt = f"""
Generate realistic real estate market statistics for areas {', '.join(area_codes)}.

Include:
- average_days_on_market (integer)
- average_list_to_sale_ratio (float)
- median_property_price (integer)
- transaction_volume_last_12_months (integer)
- market_trend (string: "hot", "balanced", "cooling")
- agent_performance_benchmarks:
  - top_quartile_days_on_market (integer)
  - median_days_on_market (integer)
  - top_quartile_list_to_sale_ratio (float)
  - median_list_to_sale_ratio (float)

Return ONLY a JSON object with data grouped by area code.
"""

        response = await self.llm_service.complete(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.5
        )

        return self._parse_json_response(response)

    async def _analyze_performance(
        self,
        agent_data: Dict[str, Any],
        market_context: Dict[str, Any],
        comparison_agents: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze agent performance metrics"""
        analysis_prompt = self.prompt_service.get_prompt(
            "agent_analysis.performance_analysis",
            agent_data=agent_data,
            market_context=market_context,
            comparison_agents=comparison_agents
        )

        analysis_response = await self.llm_service.complete(
            prompt=analysis_prompt,
            temperature=0.3,
            max_tokens=2000
        )

        return self._parse_json_response(analysis_response)

    async def _generate_insights(
        self,
        performance_analysis: Dict[str, Any],
        agent_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate strategic insights from performance analysis"""
        insight_prompt = self.prompt_service.get_prompt(
            "agent_analysis.insight_generation",
            performance_analysis=performance_analysis,
            agent_data=agent_data,
            market_context=market_context
        )

        insights = await self.llm_service.complete(
            prompt=insight_prompt,
            temperature=0.7,
            max_tokens=2500
        )

        return insights

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response"""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\n\nResponse: {response}")


# Global agent analysis workflow instance
agent_workflow = AgentAnalysisWorkflow()
