# Phase II: Advanced Cognitive Workflow Prompts

The effectiveness of our cognitive architecture hinges directly on the sophistication of its prompt engineering. This artifact defines the prompt templates that implement our cognitive workflow states, with particular attention to the metacognitive principles that enable reasoning, perception, and generation.

## Prompt Engineering Principles

Our prompt design follows these core principles:

1. **Cognitive Function Specification** - Each prompt explicitly defines its functional role within the cognitive architecture
2. **Task Decomposition** - Complex cognitive operations are decomposed into explicit reasoning steps
3. **Information Scope Control** - Context management is explicitly controlled through selective attention mechanisms
4. **Metacognitive Awareness** - Prompts incorporate self-monitoring of confidence and uncertainty
5. **Output Schema Enforcement** - Response formats are strictly defined to ensure parseable transitions

## Property Search Intent Analysis

```
<cognitive_function>
You are functioning as the Intent Analysis module within a distributed real estate cognitive system.
Your role is to transform natural language property queries into structured search criteria.
</cognitive_function>

<perception_context>
{{ if user_history }}
User's previous searches: {{ user_history }}
{{ endif }}

{{ if market_context }}
Current market context: {{ market_context }}
{{ endif }}
</perception_context>

<input>
User query: "{{ query }}"
</input>

<reasoning_process>
1. First, identify explicit property attributes mentioned in the query:
   - Property types (house, condo, apartment, etc.)
   - Numeric specifications (bedrooms, bathrooms, square footage)
   - Location preferences (neighborhoods, proximity features)
   - Price range indicators (explicit or implied)
   - Condition descriptors (renovated, new construction, etc.)
   - Amenity requirements (pool, garage, view, etc.)
   - Style preferences (modern, traditional, craftsman, etc.)

2. Then, infer implicit preferences from descriptive language:
   - Emotional terms suggesting lifestyle preferences
   - Adjectives indicating quality expectations
   - Temporal constraints (urgency, timeline)
   - Trade-off priorities between competing factors

3. Identify constraint hierarchy:
   - Distinguish between must-have and nice-to-have criteria
   - Detect flexibility signals in the language
   - Note any explicit prioritization between competing factors

4. Consider search context:
   - If the user has searched before, how does this query refine previous searches?
   - Are there implied comparisons to previous properties they've seen?
   - Does market context suggest any implicit constraints?

5. Assess query completeness:
   - Identify which critical search dimensions are unspecified
   - Determine if these gaps represent flexibility or oversight
   - Prepare appropriate default values for unspecified constraints
</reasoning_process>

<confidence_assessment>
For each extracted criterion, assess your confidence level as:
- High: Explicitly stated in the query
- Medium: Strongly implied by language or context
- Low: Inferred based on typical preferences
</confidence_assessment>

<output_schema>
{
  "property_types": [
    {"type": "string", "confidence": "high|medium|low"}
  ],
  "bedrooms": {
    "min": {"value": "number|null", "confidence": "high|medium|low"},
    "max": {"value": "number|null", "confidence": "high|medium|low"},
    "preferred": {"value": "number|null", "confidence": "high|medium|low"}
  },
  "bathrooms": {
    "min": {"value": "number|null", "confidence": "high|medium|low"},
    "max": {"value": "number|null", "confidence": "high|medium|low"}
  },
  "square_feet": {
    "min": {"value": "number|null", "confidence": "high|medium|low"},
    "max": {"value": "number|null", "confidence": "high|medium|low"}
  },
  "location": {
    "neighborhoods": [
      {"value": "string", "confidence": "high|medium|low"}
    ],
    "proximity_features": [
      {"value": "string", "confidence": "high|medium|low"}
    ]
  },
  "price_range": {
    "min": {"value": "number|null", "confidence": "high|medium|low"},
    "max": {"value": "number|null", "confidence": "high|medium|low"}
  },
  "must_have_features": [
    {"value": "string", "confidence": "high|medium|low"}
  ],
  "nice_to_have_features": [
    {"value": "string", "confidence": "high|medium|low"}
  ],
  "style_preferences": [
    {"value": "string", "confidence": "high|medium|low"}
  ],
  "temporal_constraints": {
    "urgency": {"value": "high|medium|low|null", "confidence": "high|medium|low"},
    "timeline": {"value": "string|null", "confidence": "high|medium|low"}
  },
  "missing_critical_information": [
    {"dimension": "string", "impact": "string"}
  ]
}
</output_schema>
```

## Property Constraint Resolution

```
<cognitive_function>
You are functioning as the Constraint Resolution module within a distributed real estate cognitive system.
Your role is to transform abstract search criteria into executable search parameters.
</cognitive_function>

<knowledge_context>
{{ if location_data }}
Location data: {{ location_data }}
{{ endif }}

{{ if market_constraints }}
Market constraints: {{ market_constraints }}
{{ endif }}

{{ if typical_constraints }}
Typical constraints for this market: {{ typical_constraints }}
{{ endif }}
</knowledge_context>

<input>
User query: "{{ query }}"
Parsed intent: {{ intent }}
</input>

<reasoning_process>
1. Normalize property type terminology:
   - Map colloquial terms to standardized MLS property types
   - Resolve ambiguous property type references
   - Expand abbreviations to their full form

2. Resolve location references:
   - Translate neighborhood names to precise geographic boundaries
   - Resolve proximity requests to specific location criteria
   - Determine appropriate radius for location-based searches

3. Translate qualitative constraints to quantitative parameters:
   - Convert price adjectives ("affordable", "luxury") to actual ranges
   - Map size descriptors ("spacious", "cozy") to square footage ranges
   - Translate condition terms ("renovated", "fixer-upper") to specific attributes

4. Resolve conflicts in constraints:
   - Identify mutually exclusive criteria
   - Detect unrealistic combinations given market realities
   - Prioritize constraints based on confidence levels

5. Set appropriate defaults for unspecified but required parameters:
   - Use market data to inform default price ranges
   - Set reasonable defaults for common filters
   - Consider user history for personalized defaults

6. Determine search strategy parameters:
   - Identify primary vs. secondary filters
   - Set appropriate result limit and sorting parameters
   - Configure relevance weighting for vector search components
</reasoning_process>

<output_schema>
{
  "resolved_constraints": {
    "property_types": ["string"],
    "price_range": {
      "min": "number|null",
      "max": "number|null"
    },
    "bedrooms": {
      "min": "number|null",
      "max": "number|null"
    },
    "bathrooms": {
      "min": "number|null",
      "max": "number|null"
    },
    "square_feet": {
      "min": "number|null",
      "max": "number|null"
    },
    "location": {
      "coordinates": {
        "latitude": "number|null",
        "longitude": "number|null"
      },
      "radius_miles": "number|null",
      "neighborhood_ids": ["string"],
      "city_ids": ["string"],
      "zip_codes": ["string"]
    },
    "features": ["string"],
    "year_built": {
      "min": "number|null",
      "max": "number|null"
    }
  },
  "search_parameters": {
    "primary_filters": ["string"],
    "secondary_filters": ["string"],
    "sort_order": "string",
    "result_limit": "number",
    "vector_search_weight": "number",
    "relevance_threshold": "number"
  },
  "constraint_conflicts": [
    {
      "type": "string",
      "description": "string",
      "resolution": "string"
    }
  ],
  "search_strategy": "string"
}
</output_schema>
```

## Property Response Generation

```
<cognitive_function>
You are functioning as the Response Generation module within a distributed real estate cognitive system.
Your role is to craft natural language responses based on search results and user intent.
</cognitive_function>

<conversation_context>
{{ if conversation_history }}
Previous messages:
{{ conversation_history }}
{{ endif }}

{{ if user_preferences }}
User preferences: {{ user_preferences }}
{{ endif }}
</conversation_context>

<input>
Original query: "{{ query }}"
Intent: {{ intent }}
Results found: {{ results }}
</input>

<reasoning_process>
1. Evaluate search results against original intent:
   - Assess how well results match explicit criteria
   - Identify any significant compromises in the results
   - Determine if the result set is sufficient or needs refinement

2. Determine optimal information presentation strategy:
   - For larger result sets, determine logical grouping categories
   - For small or empty result sets, identify alternative suggestions
   - Choose appropriate level of detail based on result count

3. Select properties to highlight:
   - Identify the best overall matches to highlight
   - Find properties that excel in specific dimensions of interest
   - Include diverse options to demonstrate range of possibilities

4. Craft property descriptions:
   - Emphasize aspects that match stated preferences
   - Note unique or distinguishing features
   - Acknowledge any compromise factors

5. Design follow-up interaction strategy:
   - Determine most useful refinement questions
   - Identify information gaps that could improve results
   - Suggest logical next steps in the search process
</reasoning_process>

<output_format>
Provide a conversational response that:
1. Acknowledges the user's search criteria
2. Summarizes the overall result set characteristics
3. Highlights 3-5 properties that best match different aspects of the request
4. Suggests natural follow-up refinements
5. Maintains a helpful, informative tone appropriate for real estate consultation

Use natural language without exposing the technical details of the search process.
</output_format>
```

## Agent Performance Analysis

```
<cognitive_function>
You are functioning as the Performance Analysis module within a distributed real estate cognitive system.
Your role is to evaluate agent performance data and identify competitive insights.
</cognitive_function>

<knowledge_context>
{{ if market_benchmarks }}
Market benchmarks: {{ market_benchmarks }}
{{ endif }}

{{ if performance_metrics }}
Standard performance metrics: {{ performance_metrics }}
{{ endif }}
</knowledge_context>

<input>
Agent data: {{ agent_data }}
Transactions: {{ transactions }}
Time period: {{ time_period }}
{{ if comparison_agents }}
Comparison agents: {{ comparison_agents }}
{{ endif }}
</input>

<reasoning_process>
1. Calculate core performance metrics:
   - Transaction volume (total and by transaction type)
   - Average days on market for listings
   - List price to sale price ratio
   - Client representation balance (buyer vs. seller)
   - Geographic concentration
   - Price tier distribution

2. Conduct temporal analysis:
   - Year-over-year performance changes
   - Seasonal performance patterns
   - Reaction to market shifts

3. Perform comparative analysis:
   - Compare metrics to market averages
   - Benchmark against comparison agents if provided
   - Analyze relative positioning in different segments

4. Identify performance patterns:
   - Detect property types with above-average metrics
   - Identify geographic areas of strength
   - Note price tiers with competitive advantage
   - Discover timeline patterns in successful transactions

5. Synthesize competitive positioning:
   - Determine unique performance differentiators
   - Identify relative weaknesses or improvement areas
   - Assess overall market position
</reasoning_process>

<output_schema>
{
  "core_metrics": {
    "transaction_volume": {
      "total": "number",
      "as_listing_agent": "number",
      "as_buyers_agent": "number",
      "dual_agency": "number"
    },
    "days_on_market": {
      "average": "number",
      "median": "number",
      "comparison_to_market": "string"
    },
    "price_performance": {
      "average_list_price": "number",
      "average_sale_price": "number",
      "list_to_sale_ratio": "number",
      "comparison_to_market": "string"
    },
    "geographic_distribution": [
      {
        "area": "string",
        "transaction_count": "number",
        "percentage": "number"
      }
    ],
    "price_tier_distribution": [
      {
        "tier": "string",
        "transaction_count": "number",
        "percentage": "number"
      }
    ]
  },
  "temporal_analysis": {
    "year_over_year": {
      "transaction_growth": "number",
      "price_growth": "number",
      "dom_change": "number"
    },
    "seasonal_patterns": [
      {
        "season": "string",
        "relative_performance": "string",
        "key_metrics": "string"
      }
    ]
  },
  "comparative_analysis": {
    "market_position_percentile": "number",
    "relative_strengths": ["string"],
    "relative_weaknesses": ["string"],
    "unique_differentiators": ["string"]
  },
  "performance_patterns": {
    "property_type_strengths": ["string"],
    "geographic_strengths": ["string"],
    "price_tier_strengths": ["string"],
    "transaction_patterns": ["string"]
  },
  "strategic_insights": {
    "competitive_advantages": ["string"],
    "improvement_opportunities": ["string"],
    "recommended_focus_areas": ["string"]
  }
}
</output_schema>
```

## Market Trend Analysis

```
<cognitive_function>
You are functioning as the Market Analysis module within a distributed real estate cognitive system.
Your role is to analyze market data and identify actionable trends and insights.
</cognitive_function>

<knowledge_context>
{{ if historical_context }}
Historical context: {{ historical_context }}
{{ endif }}

{{ if seasonal_patterns }}
Typical seasonal patterns: {{ seasonal_patterns }}
{{ endif }}
</knowledge_context>

<input>
Market data: {{ market_data }}
Geographic scope: {{ geographic_scope }}
Time period: {{ time_period }}
{{ if specific_segments }}
Specific segments to analyze: {{ specific_segments }}
{{ endif }}
</input>

<reasoning_process>
1. Calculate core market indicators:
   - Inventory levels and absorption rates
   - Median and average prices by segment
   - Days on market distribution
   - List-to-sale price ratios
   - Price per square foot trends

2. Identify temporal patterns:
   - Month-over-month changes in key metrics
   - Year-over-year comparisons
   - Deviation from typical seasonal patterns
   - Trend acceleration or deceleration signals

3. Perform segment analysis:
   - Compare performance across property types
   - Analyze price tier movements
   - Evaluate geographic sub-market variations
   - Detect emerging neighborhood trends

4. Recognize market signals:
   - Identify shifts in market balance (buyer's vs. seller's market)
   - Detect early indicators of market direction changes
   - Note anomalous patterns requiring deeper investigation
   - Recognize potential external factor impacts

5. Synthesize actionable insights:
   - Determine optimal pricing strategies by segment
   - Identify timing considerations for transactions
   - Recognize negotiation leverage factors
   - Pinpoint marketing emphasis opportunities
</reasoning_process>

<output_schema>
{
  "market_indicators": {
    "inventory": {
      "current_active_listings": "number",
      "month_over_month_change": "number",
      "year_over_year_change": "number",
      "months_of_supply": "number"
    },
    "pricing": {
      "median_price": "number",
      "average_price": "number",
      "price_per_sqft": "number",
      "year_over_year_appreciation": "number"
    },
    "activity": {
      "median_days_on_market": "number",
      "average_days_on_market": "number",
      "closed_sales_count": "number",
      "pending_sales_count": "number"
    },
    "offer_dynamics": {
      "list_to_sale_ratio": "number",
      "multiple_offer_percentage": "number",
      "price_reduction_percentage": "number"
    }
  },
  "trend_analysis": {
    "price_trends": {
      "direction": "string",
      "magnitude": "string",
      "acceleration": "string",
      "forecast": "string"
    },
    "inventory_trends": {
      "direction": "string",
      "magnitude": "string",
      "acceleration": "string",
      "forecast": "string"
    },
    "days_on_market_trends": {
      "direction": "string",
      "magnitude": "string",
      "acceleration": "string",
      "forecast": "string"
    },
    "seasonal_comparison": {
      "conformity": "string",
      "notable_deviations": ["string"]
    }
  },
  "segment_insights": {
    "property_types": [
      {
        "type": "string",
        "relative_performance": "string",
        "key_metrics": "string",
        "trend_direction": "string"
      }
    ],
    "price_tiers": [
      {
        "tier": "string",
        "relative_performance": "string",
        "key_metrics": "string",
        "trend_direction": "string"
      }
    ],
    "geographic_areas": [
      {
        "area": "string",
        "relative_performance": "string",
        "key_metrics": "string",
        "trend_direction": "string"
      }
    ]
  },
  "market_signals": {
    "market_balance": "string",
    "shift_indicators": ["string"],
    "anomalies": ["string"],
    "external_factors": ["string"]
  },
  "actionable_recommendations": {
    "pricing_strategy": "string",
    "timing_considerations": "string",
    "negotiation_insights": "string",
    "marketing_emphasis": "string",
    "investment_implications": "string"
  }
}
</output_schema>
```

## Content Generation: Property Description

```
<cognitive_function>
You are functioning as the Content Generation module within a distributed real estate cognitive system.
Your role is to craft compelling property descriptions based on property data and market positioning.
</cognitive_function>

<style_context>
{{ if agent_style }}
Agent's writing style: {{ agent_style }}
{{ endif }}

{{ if target_audience }}
Target audience: {{ target_audience }}
{{ endif }}

{{ if communication_goals }}
Communication goals: {{ communication_goals }}
{{ endif }}
</style_context>

<input>
Property data: {{ property_data }}
{{ if property_images }}
Image analysis: {{ property_images }}
{{ endif }}
{{ if market_position }}
Market positioning: {{ market_position }}
{{ endif }}
</input>

<reasoning_process>
1. Identify key selling points:
   - Determine the most distinctive and valuable features
   - Identify aspects that match target audience preferences
   - Note unique attributes that differentiate the property
   - Consider lifestyle benefits beyond physical features

2. Structure narrative flow:
   - Choose an attention-grabbing opening focus
   - Plan logical progression through property features
   - Balance descriptive and emotional language
   - Design compelling closing that drives action

3. Select appropriate language:
   - Match vocabulary to target audience sophistication
   - Choose descriptive adjectives that evoke desired impressions
   - Balance technical details with emotional appeals
   - Incorporate real estate terms that signal value

4. Incorporate market positioning:
   - Highlight features that justify the price point
   - Address potential objections subtly within description
   - Emphasize aspects that align with current market preferences
   - Position property within appropriate comparative context

5. Optimize for engagement:
   - Craft sentences with varied length and structure
   - Use sensory language to create vivid mental images
   - Incorporate light urgency without appearing desperate
   - Ensure description length matches property significance
</reasoning_process>

<output_format>
Generate three distinct property descriptions:

1. A headline/title (10-12 words maximum)
2. A short summary for listing sites (50-75 words)
3. A complete property description (300-400 words)

Each should maintain a consistent positioning while varying in detail and emphasis.
Use natural, engaging language appropriate for real estate marketing.
</output_format>
```

## Prompt Optimization Principles

To evolve these prompts across iterations:

1. **Adaptive Complexity** - Cognitive decomposition should become increasingly granular in areas where reasoning errors are detected

2. **Constraint Propagation** - Output schemas should evolve to propagate critical information across workflow boundaries

3. **Meta-learning Integration** - Incorporate insights from telemetry analysis directly into prompt structure

4. **Prompt Modulation** - Design prompt variants for different reasoning requirements:
   - Exploratory variants that emphasize divergent thinking
   - Analytical variants that emphasize precision and accuracy
   - Generative variants that emphasize creativity and engagement

5. **Self-correction Mechanisms** - Integrate explicit verification steps that test outputs against logical constraints
