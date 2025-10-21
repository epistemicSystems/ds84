# Phase II: Meta-Cognitive Learning System

The initial prototype established foundational cognitive workflows, but lacks the recursive self-improvement mechanisms that characterize truly adaptive intelligence. Phase II introduces a meta-cognitive feedback loop that enables progressive optimization through empirical observation of system performance.

## Architectural Evolution

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│               META-COGNITIVE SUPERVISOR                      │
│                                                             │
├─────────────┬─────────────────────────────┬─────────────────┤
│             │                             │                 │
│  Interaction│     Prompt Performance      │    Workflow     │
│    Logger   │         Analyzer            │   Optimizer     │
│             │                             │                 │
└─────────────┴─────────────────────────────┴─────────────────┘
        │                   │                       │
        ▼                   ▼                       ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Telemetry  │    │  Experimental   │    │  Configuration  │
│   Database  │    │  Prompt Store   │    │    Manager      │
└─────────────┘    └─────────────────┘    └─────────────────┘
```

## Implementation Strategy

### Interaction Telemetry System

```python
# app/services/telemetry_service.py
import asyncpg
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

class TelemetryService:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        
    async def initialize(self):
        """Initialize telemetry database schema"""
        async with self.db_pool.acquire() as conn:
            # Create interactions table
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id UUID PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                workflow_id TEXT,
                state_id TEXT,
                input JSON,
                output JSON,
                latency FLOAT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
            ''')
            
            # Create performance_metrics table
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id UUID PRIMARY KEY,
                interaction_id UUID REFERENCES interactions(id),
                metric_type TEXT,
                metric_value FLOAT,
                metadata JSON,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
            ''')
    
    async def log_interaction(self, 
                            user_id: str,
                            session_id: str,
                            workflow_id: str,
                            state_id: str,
                            input_data: Dict[str, Any],
                            output_data: Dict[str, Any],
                            latency: float) -> str:
        """Log a single workflow state interaction"""
        interaction_id = str(uuid.uuid4())
        
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
            INSERT INTO interactions 
            (id, user_id, session_id, workflow_id, state_id, input, output, latency)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
            interaction_id, user_id, session_id, workflow_id, state_id, 
            json.dumps(input_data), json.dumps(output_data), latency)
            
        return interaction_id
    
    async def log_metric(self,
                       interaction_id: str,
                       metric_type: str,
                       metric_value: float,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a performance metric for an interaction"""
        metric_id = str(uuid.uuid4())
        
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
            INSERT INTO performance_metrics
            (id, interaction_id, metric_type, metric_value, metadata)
            VALUES ($1, $2, $3, $4, $5)
            ''',
            metric_id, interaction_id, metric_type, metric_value, 
            json.dumps(metadata) if metadata else json.dumps({}))
            
        return metric_id
    
    async def get_workflow_performance(self,
                                    workflow_id: str,
                                    time_window: Optional[Dict[str, datetime]] = None,
                                    limit: int = 1000) -> List[Dict[str, Any]]:
        """Get performance data for a specific workflow"""
        where_clause = "workflow_id = $1"
        params = [workflow_id]
        
        if time_window:
            where_clause += " AND timestamp >= $2 AND timestamp <= $3"
            params.extend([time_window.get('start'), time_window.get('end')])
        
        async with self.db_pool.acquire() as conn:
            interactions = await conn.fetch(f'''
            SELECT * FROM interactions
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT $4
            ''', *params, limit)
            
            result = []
            for interaction in interactions:
                metrics = await conn.fetch('''
                SELECT * FROM performance_metrics
                WHERE interaction_id = $1
                ''', interaction['id'])
                
                result.append({
                    **dict(interaction),
                    'metrics': [dict(m) for m in metrics]
                })
                
            return result
```

### Prompt Experimentation System

```python
# app/services/prompt_experiment_service.py
import asyncpg
import json
import uuid
import random
from typing import Dict, Any, List, Optional, Tuple

class PromptExperimentService:
    def __init__(self, db_pool, llm_service):
        self.db_pool = db_pool
        self.llm_service = llm_service
        
    async def initialize(self):
        """Initialize experimentation database schema"""
        async with self.db_pool.acquire() as conn:
            # Create prompt_templates table
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id UUID PRIMARY KEY,
                template_id TEXT UNIQUE,
                template_content TEXT,
                version INT,
                metadata JSON,
                is_active BOOLEAN,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            ''')
            
            # Create prompt_experiments table
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS prompt_experiments (
                id UUID PRIMARY KEY,
                experiment_name TEXT,
                template_ids JSON, -- Array of template IDs to compare
                selection_strategy TEXT,
                eval_metric TEXT,
                status TEXT,
                results JSON,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
            ''')
    
    async def store_template(self, 
                           template_id: str,
                           template_content: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           is_active: bool = True) -> str:
        """Store a new prompt template version"""
        # Get current version if template exists
        async with self.db_pool.acquire() as conn:
            current_version = await conn.fetchval('''
            SELECT MAX(version) FROM prompt_templates
            WHERE template_id = $1
            ''', template_id)
            
            version = (current_version or 0) + 1
            
            # Insert new template version
            id = str(uuid.uuid4())
            await conn.execute('''
            INSERT INTO prompt_templates
            (id, template_id, template_content, version, metadata, is_active)
            VALUES ($1, $2, $3, $4, $5, $6)
            ''',
            id, template_id, template_content, version, 
            json.dumps(metadata) if metadata else json.dumps({}), is_active)
            
            # Deactivate previous version if new one is active
            if is_active:
                await conn.execute('''
                UPDATE prompt_templates
                SET is_active = FALSE
                WHERE template_id = $1 AND version != $2
                ''', template_id, version)
            
            return id
    
    async def get_active_template(self, template_id: str) -> Tuple[str, str]:
        """Get the active template content for a template ID"""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow('''
            SELECT id, template_content
            FROM prompt_templates
            WHERE template_id = $1 AND is_active = TRUE
            ''', template_id)
            
            if not result:
                raise ValueError(f"No active template found for {template_id}")
                
            return result['id'], result['template_content']
    
    async def create_experiment(self,
                             experiment_name: str,
                             template_id: str,
                             num_variants: int = 3,
                             eval_metric: str = "accuracy") -> str:
        """Create prompt variants and set up an experiment"""
        # Get current active template
        _, base_template = await self.get_active_template(template_id)
        
        # Generate variants using LLM
        variant_prompt = f"""
        You are an expert prompt engineer tasked with creating {num_variants} variations of the following prompt template.
        Each variant should maintain the same functional purpose but explore different phrasing, structure, or examples
        to potentially improve performance on {eval_metric}.
        
        Original template:
        ```
        {base_template}
        ```
        
        Generate {num_variants} distinct variations, each maintaining the core purpose but approaching the task differently.
        Return the variants in JSON format with each variant as a string in an array.
        """
        
        variant_response = await self.llm_service.complete(
            prompt=variant_prompt,
            temperature=0.8,
            max_tokens=4000
        )
        
        # Extract JSON from response
        try:
            # Find JSON array in response
            json_start = variant_response.find('[')
            json_end = variant_response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = variant_response[json_start:json_end]
                variants = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in variant response")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse variant response as JSON")
        
        # Store variants as new templates
        template_ids = []
        for i, variant in enumerate(variants):
            variant_id = await self.store_template(
                template_id=f"{template_id}_variant_{i+1}",
                template_content=variant,
                metadata={
                    "parent_template": template_id,
                    "experiment_generated": True,
                    "variant_index": i+1
                },
                is_active=False  # Variants start as inactive
            )
            template_ids.append(variant_id)
        
        # Create experiment
        experiment_id = str(uuid.uuid4())
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
            INSERT INTO prompt_experiments
            (id, experiment_name, template_ids, selection_strategy, eval_metric, status)
            VALUES ($1, $2, $3, $4, $5, $6)
            ''',
            experiment_id, experiment_name, json.dumps(template_ids),
            "random_assignment", eval_metric, "active")
            
        return experiment_id
    
    async def select_template_for_interaction(self, 
                                           template_id: str,
                                           context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Select template for an interaction based on active experiments"""
        # Check if template is in an active experiment
        async with self.db_pool.acquire() as conn:
            # First try to find active experiments for this template
            active_experiment = await conn.fetchrow('''
            SELECT id, template_ids, selection_strategy
            FROM prompt_experiments
            WHERE status = 'active' AND experiment_name LIKE $1
            ''', f"%{template_id}%")
            
            if active_experiment:
                # Use experiment selection strategy
                template_ids = json.loads(active_experiment['template_ids'])
                
                if active_experiment['selection_strategy'] == 'random_assignment':
                    # Randomly select a variant
                    selected_id = random.choice(template_ids)
                    
                    # Get the template content
                    template_content = await conn.fetchval('''
                    SELECT template_content FROM prompt_templates
                    WHERE id = $1
                    ''', selected_id)
                    
                    return selected_id, template_content
            
            # If no experiment or strategy failed, return active template
            return await self.get_active_template(template_id)
    
    async def log_template_performance(self,
                                    template_id: str,
                                    metric_value: float,
                                    interaction_id: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log performance for a template usage"""
        perf_id = str(uuid.uuid4())
        
        async with self.db_pool.acquire() as conn:
            # Get template details
            template = await conn.fetchrow('''
            SELECT id, template_id FROM prompt_templates
            WHERE id = $1
            ''', template_id)
            
            if not template:
                raise ValueError(f"Template not found with ID {template_id}")
            
            # Log to performance_metrics
            await conn.execute('''
            INSERT INTO performance_metrics
            (id, interaction_id, metric_type, metric_value, metadata)
            VALUES ($1, $2, $3, $4, $5)
            ''',
            perf_id, interaction_id, f"prompt_performance_{template['template_id']}", 
            metric_value, json.dumps(metadata) if metadata else json.dumps({}))
    
    async def analyze_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze results of an experiment and determine the best variant"""
        async with self.db_pool.acquire() as conn:
            # Get experiment details
            experiment = await conn.fetchrow('''
            SELECT * FROM prompt_experiments
            WHERE id = $1
            ''', experiment_id)
            
            if not experiment:
                raise ValueError(f"Experiment not found with ID {experiment_id}")
                
            template_ids = json.loads(experiment['template_ids'])
            
            # Collect performance metrics for each template
            results = {}
            for template_id in template_ids:
                template = await conn.fetchrow('''
                SELECT id, template_id FROM prompt_templates
                WHERE id = $1
                ''', template_id)
                
                metrics = await conn.fetch('''
                SELECT metric_value FROM performance_metrics
                WHERE metric_type = $1
                ''', f"prompt_performance_{template['template_id']}")
                
                values = [m['metric_value'] for m in metrics]
                
                if values:
                    results[template_id] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
                else:
                    results[template_id] = {
                        "count": 0,
                        "mean": None,
                        "min": None,
                        "max": None
                    }
            
            # Find best performing template
            best_template_id = None
            best_performance = -float('inf')
            
            for template_id, stats in results.items():
                if stats["count"] > 0 and stats["mean"] > best_performance:
                    best_template_id = template_id
                    best_performance = stats["mean"]
            
            # Update experiment with results
            await conn.execute('''
            UPDATE prompt_experiments
            SET results = $1, status = $2, completed_at = NOW()
            WHERE id = $3
            ''',
            json.dumps({
                "template_results": results,
                "best_template_id": best_template_id,
                "best_performance": best_performance
            }),
            "completed", experiment_id)
            
            # Activate best template if clear winner
            if best_template_id:
                template = await conn.fetchrow('''
                SELECT template_id FROM prompt_templates
                WHERE id = $1
                ''', best_template_id)
                
                original_template_id = template['template_id'].split('_variant_')[0]
                
                # Get best template content
                best_template_content = await conn.fetchval('''
                SELECT template_content FROM prompt_templates
                WHERE id = $1
                ''', best_template_id)
                
                # Store as new active version of original template
                await self.store_template(
                    template_id=original_template_id,
                    template_content=best_template_content,
                    metadata={
                        "source_experiment": experiment_id,
                        "performance": best_performance
                    },
                    is_active=True
                )
            
            return {
                "experiment_id": experiment_id,
                "results": results,
                "best_template_id": best_template_id,
                "best_performance": best_performance
            }
```

### Workflow Optimization Service

The Workflow Optimizer represents the apex of the meta-cognitive architecture—a system capable of dynamically reconfiguring cognitive workflows based on empirical performance data.

```python
# app/services/workflow_optimizer_service.py
import json
import uuid
from typing import Dict, Any, List, Optional
from app.services.prompt_experiment_service import PromptExperimentService
from app.services.telemetry_service import TelemetryService

class WorkflowOptimizerService:
    def __init__(self, 
                prompt_experiment_service: PromptExperimentService,
                telemetry_service: TelemetryService,
                llm_service):
        self.prompt_experiment_service = prompt_experiment_service
        self.telemetry_service = telemetry_service
        self.llm_service = llm_service
        
    async def optimize_workflow(self, 
                              workflow_id: str,
                              optimization_target: str = "accuracy",
                              min_data_points: int = 50) -> Dict[str, Any]:
        """Analyze workflow performance and identify optimization opportunities"""
        # Get workflow performance data
        performance_data = await self.telemetry_service.get_workflow_performance(
            workflow_id=workflow_id,
            limit=1000  # Get sufficient data for analysis
        )
        
        if len(performance_data) < min_data_points:
            return {
                "workflow_id": workflow_id,
                "status": "insufficient_data",
                "message": f"Need at least {min_data_points} data points, but only have {len(performance_data)}"
            }
        
        # Group data by state_id to analyze state-specific performance
        states_data = {}
        for interaction in performance_data:
            state_id = interaction['state_id']
            if state_id not in states_data:
                states_data[state_id] = []
            states_data[state_id].append(interaction)
        
        # Analyze each state for performance bottlenecks
        state_analysis = {}
        for state_id, interactions in states_data.items():
            # Calculate key metrics
            latencies = [float(i['latency']) for i in interactions]
            avg_latency = sum(latencies) / len(latencies)
            
            # Extract success/failure patterns
            error_count = sum(1 for i in interactions if 'error' in i['output'])
            error_rate = error_count / len(interactions)
            
            state_analysis[state_id] = {
                "interactions_count": len(interactions),
                "avg_latency": avg_latency,
                "error_rate": error_rate,
                "metrics": self._extract_state_metrics(interactions)
            }
        
        # Identify bottleneck states
        bottlenecks = []
        for state_id, analysis in state_analysis.items():
            is_bottleneck = False
            bottleneck_reasons = []
            
            # Check for high latency
            if analysis["avg_latency"] > 2.0:  # Threshold for concern
                is_bottleneck = True
                bottleneck_reasons.append("high_latency")
                
            # Check for high error rate
            if analysis["error_rate"] > 0.05:  # >5% error rate
                is_bottleneck = True
                bottleneck_reasons.append("high_error_rate")
                
            # Check for poor metric performance
            for metric_name, metric_value in analysis["metrics"].items():
                if metric_name.startswith("accuracy") and metric_value < 0.8:
                    is_bottleneck = True
                    bottleneck_reasons.append(f"low_{metric_name}")
            
            if is_bottleneck:
                bottlenecks.append({
                    "state_id": state_id,
                    "reasons": bottleneck_reasons,
                    "metrics": analysis
                })
        
        # Generate optimization recommendations using LLM
        recommendations = await self._generate_optimization_recommendations(
            workflow_id, bottlenecks, state_analysis, optimization_target
        )
        
        # Trigger experiments for bottleneck states if appropriate
        experiments = []
        for bottleneck in bottlenecks:
            state_id = bottleneck["state_id"]
            
            # Extract prompt template ID from state_id
            # Assuming format like "property_search.intent_analysis"
            if "." in state_id:
                template_id = state_id
            else:
                continue  # Skip if can't determine template
                
            # Create experiment for this template
            try:
                experiment_id = await self.prompt_experiment_service.create_experiment(
                    experiment_name=f"optimize_{workflow_id}_{state_id}",
                    template_id=template_id,
                    num_variants=3,
                    eval_metric=optimization_target
                )
                
                experiments.append({
                    "state_id": state_id,
                    "experiment_id": experiment_id
                })
            except Exception as e:
                print(f"Error creating experiment for {state_id}: {e}")
        
        return {
            "workflow_id": workflow_id,
            "analysis": state_analysis,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "experiments": experiments
        }
    
    def _extract_state_metrics(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract performance metrics for a state from interactions"""
        metrics = {}
        metric_values = {}
        
        # Collect all metrics across interactions
        for interaction in interactions:
            if 'metrics' in interaction:
                for metric in interaction['metrics']:
                    metric_type = metric['metric_type']
                    metric_value = float(metric['metric_value'])
                    
                    if metric_type not in metric_values:
                        metric_values[metric_type] = []
                    
                    metric_values[metric_type].append(metric_value)
        
        # Calculate averages
        for metric_type, values in metric_values.items():
            if values:
                metrics[metric_type] = sum(values) / len(values)
        
        return metrics
    
    async def _generate_optimization_recommendations(self,
                                                  workflow_id: str,
                                                  bottlenecks: List[Dict[str, Any]],
                                                  state_analysis: Dict[str, Any],
                                                  optimization_target: str) -> List[Dict[str, Any]]:
        """Generate optimization recommendations using LLM"""
        if not bottlenecks:
            return []
            
        # Create detailed analysis for LLM
        analysis_prompt = f"""
        You are an AI workflow optimization expert. Analyze the following workflow performance data
        and provide specific optimization recommendations for bottleneck states.
        
        Workflow ID: {workflow_id}
        Optimization Target: {optimization_target}
        
        Performance Analysis:
        {json.dumps(state_analysis, indent=2)}
        
        Identified Bottlenecks:
        {json.dumps(bottlenecks, indent=2)}
        
        For each bottleneck, provide:
        1. Specific prompt engineering improvements
        2. Workflow structure optimizations
        3. Model selection recommendations
        4. Any other architectural changes that could improve performance
        
        Return recommendations as a JSON array where each item has:
        - state_id: The state this recommendation applies to
        - recommendation_type: One of [prompt_engineering, workflow_structure, model_selection, architecture]
        - description: Detailed description of the recommendation
        - expected_impact: Expected impact on performance (high, medium, low)
        """
        
        recommendation_response = await self.llm_service.complete(
            prompt=analysis_prompt,
            temperature=0.4,
            max_tokens=3000
        )
        
        # Extract JSON from response
        try:
            json_start = recommendation_response.find('[')
            json_end = recommendation_response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = recommendation_response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON array found in recommendations")
        except json.JSONDecodeError:
            print(f"Failed to parse recommendations: {recommendation_response}")
            return []
```

## Integration with Cognitive Workflow Engine

To integrate these meta-cognitive capabilities into our workflow engine, we need to enhance the state transition capabilities with telemetry and experiment selection:

```python
# Enhanced state execution in CognitiveWorkflowEngine
async def _execute_state(self,
                       workflow_id: str,
                       state: CognitiveState,
                       state_data: dict,
                       context_data: dict = None) -> dict:
    """Execute a cognitive workflow state with meta-cognitive capabilities"""
    
    start_time = time.time()
    
    try:
        # Select appropriate template with experimentation
        template_id, template_content = await self.prompt_experiment_service.select_template_for_interaction(
            template_id=f"{workflow_id}.{state.id}",
            context=context_data
        )
        
        # Format prompt with state data
        formatted_prompt = self._format_prompt(
            template_content=template_content,
            data={**state_data, **(context_data or {})}
        )
        
        # Select appropriate model based on state requirements
        model_config = self.model_router.select_model(
            state.cognitive_function,
            state.requirements
        )
        
        # Execute LLM call
        response = await self.llm_router.complete(
            prompt=formatted_prompt,
            model=model_config.model,
            provider=model_config.provider,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )
        
        # Parse response based on state output schema
        parsed_output = self._parse_llm_response(
            response=response,
            output_schema=state.output_schema
        )
        
        # Calculate execution metrics
        latency = time.time() - start_time
        success = True
        
        # Log telemetry
        interaction_id = await self.telemetry_service.log_interaction(
            user_id=context_data.get('user_id') if context_data else None,
            session_id=context_data.get('session_id') if context_data else None,
            workflow_id=workflow_id,
            state_id=state.id,
            input_data=state_data,
            output_data=parsed_output,
            latency=latency
        )
        
        # Log prompt performance
        if 'accuracy' in parsed_output:
            await self.prompt_experiment_service.log_template_performance(
                template_id=template_id,
                metric_value=parsed_output['accuracy'],
                interaction_id=interaction_id,
                metadata={
                    'workflow_id': workflow_id,
                    'state_id': state.id
                }
            )
        
        return parsed_output
        
    except Exception as e:
        # Log error telemetry
        latency = time.time() - start_time
        error_data = {'error': str(e)}
        
        await self.telemetry_service.log_interaction(
            user_id=context_data.get('user_id') if context_data else None,
            session_id=context_data.get('session_id') if context_data else None,
            workflow_id=workflow_id,
            state_id=state.id,
            input_data=state_data,
            output_data=error_data,
            latency=latency
        )
        
        raise
```

## Implementation Timeline

**Week 1: Telemetry Foundation**
- Day 1-2: Implement interaction logging infrastructure
- Day 3-4: Build performance metrics collection 
- Day 5: Add visualization dashboard for telemetry data

**Week 2: Prompt Experimentation System**
- Day 1-2: Implement prompt template versioning
- Day 3-4: Build experiment creation and execution
- Day 5: Develop experiment analysis capabilities

**Week 3: Workflow Optimization**
- Day 1-2: Implement workflow analysis algorithms
- Day 3-4: Build recommendation generation system
- Day 5: Integrate automated optimization into workflow engine
