"""Workflow visualization and debugging tool"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.cognitive_workflow_engine import workflow_engine
from app.models.workflow_models import WorkflowDefinition, CognitiveState


def visualize_workflow(workflow_id: str):
    """Visualize workflow structure

    Args:
        workflow_id: Workflow identifier
    """
    workflow = workflow_engine.get_workflow(workflow_id)
    if not workflow:
        print(f"❌ Workflow not found: {workflow_id}")
        return

    print("=" * 80)
    print(f"WORKFLOW: {workflow.name}")
    print("=" * 80)
    print(f"ID: {workflow.id}")
    print(f"Version: {workflow.version}")
    print(f"Description: {workflow.description}")
    print(f"Entry Point: {workflow.entry_point}")
    print()

    # Print states
    print("STATES:")
    print("-" * 80)
    for state_id, state in workflow.states.items():
        print(f"\n[{state_id}]")
        print(f"  Name: {state.name}")
        print(f"  Type: {state.state_type}")
        print(f"  Agent: {state.agent_type}")
        print(f"  Model: {state.model} (temp={state.temperature})")
        print(f"  Prompt: {state.prompt_template}")

        if state.description:
            print(f"  Description: {state.description}")

        # Input schema
        if state.input_schema.required:
            print(f"  Required Input: {', '.join(state.input_schema.required)}")

        # Context requirements
        if state.context_requirements:
            contexts = [f"{req.type} ({req.scope})" for req in state.context_requirements]
            print(f"  Context: {', '.join(contexts)}")

    # Print transitions
    print()
    print("TRANSITIONS:")
    print("-" * 80)
    for i, transition in enumerate(workflow.transitions, 1):
        from_state = transition.from_state
        to_state = transition.to_state or "END"

        condition_str = ""
        if transition.condition:
            cond = transition.condition
            condition_str = f" (if {cond.field} {cond.operator} {cond.value})"

        print(f"{i}. {from_state} → {to_state}{condition_str}")
        if transition.description:
            print(f"   {transition.description}")

    # Print metadata
    if workflow.metadata:
        print()
        print("METADATA:")
        print("-" * 80)
        for key, value in workflow.metadata.items():
            print(f"  {key}: {value}")

    print()
    print("=" * 80)


def visualize_workflow_graph(workflow_id: str, output_format: str = "ascii"):
    """Visualize workflow as a graph

    Args:
        workflow_id: Workflow identifier
        output_format: "ascii" for text graph
    """
    workflow = workflow_engine.get_workflow(workflow_id)
    if not workflow:
        print(f"❌ Workflow not found: {workflow_id}")
        return

    print()
    print(f"WORKFLOW GRAPH: {workflow.name}")
    print("=" * 80)
    print()

    # Build graph representation
    current_state = workflow.entry_point
    visited = set()
    level = 0

    def print_state_tree(state_id: str, indent: int = 0):
        """Recursively print state tree"""
        if state_id in visited or state_id is None:
            return

        visited.add(state_id)

        # Print current state
        state = workflow.states.get(state_id)
        if state:
            prefix = "  " * indent
            print(f"{prefix}┌─ [{state.state_type}] {state.name}")
            print(f"{prefix}│  ID: {state_id}")
            print(f"{prefix}│  Model: {state.model}")
            print(f"{prefix}└─ Prompt: {state.prompt_template}")
            print(f"{prefix}   │")

        # Find next states
        next_transitions = [
            t for t in workflow.transitions
            if t.from_state == state_id
        ]

        for i, trans in enumerate(next_transitions):
            is_last = (i == len(next_transitions) - 1)
            connector = "└─" if is_last else "├─"

            if trans.to_state:
                print(f"{prefix}   {connector}▶")
                print_state_tree(trans.to_state, indent + 1)
            else:
                print(f"{prefix}   {connector}▶ [END]")
                print()

    print_state_tree(workflow.entry_point)
    print()
    print("=" * 80)


def list_all_workflows():
    """List all available workflows"""
    workflows = workflow_engine.list_workflows()

    if not workflows:
        print("No workflows found.")
        return

    print("=" * 80)
    print("AVAILABLE WORKFLOWS")
    print("=" * 80)
    print()

    for workflow_id in workflows:
        workflow = workflow_engine.get_workflow(workflow_id)
        print(f"• {workflow.name}")
        print(f"  ID: {workflow.id}")
        print(f"  Version: {workflow.version}")
        print(f"  States: {len(workflow.states)}")
        print(f"  Transitions: {len(workflow.transitions)}")
        print()

    print("=" * 80)


def validate_workflow(workflow_id: str):
    """Validate workflow definition

    Args:
        workflow_id: Workflow identifier
    """
    workflow = workflow_engine.get_workflow(workflow_id)
    if not workflow:
        print(f"❌ Workflow not found: {workflow_id}")
        return

    print(f"🔍 Validating workflow: {workflow.name}")
    print()

    errors = []
    warnings = []

    # Check entry point exists
    if workflow.entry_point not in workflow.states:
        errors.append(f"Entry point '{workflow.entry_point}' not defined in states")

    # Check all states referenced in transitions exist
    for trans in workflow.transitions:
        if trans.from_state not in workflow.states:
            errors.append(f"Transition references undefined state: {trans.from_state}")
        if trans.to_state and trans.to_state not in workflow.states:
            errors.append(f"Transition references undefined state: {trans.to_state}")

    # Check for unreachable states
    reachable = set([workflow.entry_point])
    changed = True
    while changed:
        changed = False
        for trans in workflow.transitions:
            if trans.from_state in reachable and trans.to_state:
                if trans.to_state not in reachable:
                    reachable.add(trans.to_state)
                    changed = True

    unreachable = set(workflow.states.keys()) - reachable
    if unreachable:
        warnings.append(f"Unreachable states: {', '.join(unreachable)}")

    # Check for states with no outgoing transitions
    states_with_transitions = set([t.from_state for t in workflow.transitions])
    terminal_states = set(workflow.states.keys()) - states_with_transitions

    if len(terminal_states) > 1:
        warnings.append(f"Multiple terminal states: {', '.join(terminal_states)}")

    # Print results
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  • {error}")
        print()

    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
        print()

    if not errors and not warnings:
        print("✅ Workflow is valid!")
        print()

    print(f"States: {len(workflow.states)}")
    print(f"Transitions: {len(workflow.transitions)}")
    print(f"Reachable states: {len(reachable)}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize and debug cognitive workflows')
    parser.add_argument(
        'command',
        choices=['list', 'show', 'graph', 'validate'],
        help='Command to execute'
    )
    parser.add_argument(
        '--workflow',
        '-w',
        help='Workflow ID'
    )

    args = parser.parse_args()

    if args.command == 'list':
        list_all_workflows()
    elif args.command == 'show':
        if not args.workflow:
            print("Error: --workflow required for 'show' command")
            return 1
        visualize_workflow(args.workflow)
    elif args.command == 'graph':
        if not args.workflow:
            print("Error: --workflow required for 'graph' command")
            return 1
        visualize_workflow_graph(args.workflow)
    elif args.command == 'validate':
        if not args.workflow:
            print("Error: --workflow required for 'validate' command")
            return 1
        validate_workflow(args.workflow)

    return 0


if __name__ == "__main__":
    sys.exit(main())
