"""
Example 04: Agent-based Route Analysis

Demonstrates WalkingAgent with different personalities analyzing the same route.
Shows how personality configurations affect decision-making and explanations.

This example:
1. Lists available personality presets
2. Creates agents with different personalities
3. Analyzes the same route with each agent
4. Compares their decisions and explanations
5. Shows semantic vs framework-specific configuration modes
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import WalkingAgent, list_presets
from src.utils.logging import get_logger

logger = get_logger(__name__)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print a formatted section header."""
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def print_agent_info(agent: WalkingAgent):
    """Print agent configuration details."""
    print(f"Agent ID: {agent.metadata.agent_id}")
    print(f"Name: {agent.metadata.name}")
    print(f"Description: {agent.metadata.description}")
    print(f"Explanation Style: {agent.personality.explanation_style}")
    print()
    print("Dimension Weights:")
    for dim, weight in agent.personality.dimension_weights.items():
        print(f"  {dim:15s}: {weight:.2f}")
    print()
    print("Decision Thresholds:")
    for key, value in agent.personality.decision_thresholds.items():
        print(f"  {key:25s}: {value}")
    print()


def print_decision(agent_name: str, result: Dict[str, Any]):
    """Print agent decision details."""
    decision = result.get("decision", {})
    action = result.get("result", {})

    print(f"🤖 {agent_name}")
    print(f"   Recommendation: {decision.get('recommendation', 'N/A').upper()}")
    print(f"   Confidence: {decision.get('confidence', 0):.2%}")
    print(f"   Weighted Score: {decision.get('weighted_score', 0):.1f}/10")
    print()
    print(f"   {action.get('message', 'No message')}")
    print()

    if decision.get("highlights"):
        print("   ✨ Highlights:")
        for highlight in decision["highlights"][:3]:
            print(f"      • {highlight}")
        print()

    if decision.get("concerns"):
        print("   ⚠️  Concerns:")
        for concern in decision["concerns"][:3]:
            print(f"      • {concern}")
        print()


def compare_agents(results: List[Dict[str, Any]]):
    """Print comparison table of agent decisions."""
    print_section_header("AGENT COMPARISON")

    print(f"{'Agent':<20} {'Decision':<12} {'Confidence':<12} {'Score':<8} {'Concerns':<10}")
    print("-" * 72)

    for result in results:
        agent_name = result["metadata"]["name"]
        decision = result["decision"]
        recommendation = decision.get("recommendation", "unknown").upper()
        confidence = decision.get("confidence", 0)
        score = decision.get("weighted_score", 0)
        num_concerns = len(decision.get("concerns", []))

        print(f"{agent_name:<20} {recommendation:<12} {confidence:>10.1%}  {score:>6.1f}  {num_concerns:>8}")

    print()


def demo_personality_listing():
    """Demonstrate listing available personalities."""
    print_section_header("STEP 1: Available Personalities")

    presets = list_presets()
    print(f"Found {len(presets)} personality presets:")
    for i, preset in enumerate(presets, 1):
        print(f"  {i}. {preset}")

    print()


def demo_create_agents(framework_id: str = "sagai_2025"):
    """Demonstrate creating agents with different personalities."""
    print_section_header("STEP 2: Creating Agents")

    personalities = ["safety", "scenic", "balanced"]
    agents = {}

    for personality in personalities:
        print(f"Creating {personality} agent...")
        agent = WalkingAgent.from_preset(
            preset_name=personality,
            framework_id=framework_id,
            use_semantic=True  # Use semantic mapping for flexibility
        )
        agents[personality] = agent
        print(f"  ✓ {agent.metadata.name} created")

    print()
    return agents


def demo_show_agent_configs(agents: Dict[str, WalkingAgent]):
    """Show detailed agent configurations."""
    print_section_header("STEP 3: Agent Configurations")

    for personality, agent in agents.items():
        print(f"--- {personality.upper()} AGENT ---")
        print()
        print_agent_info(agent)
        print()


def demo_analyze_route(agents: Dict[str, WalkingAgent], start, end):
    """Analyze a route with multiple agents and compare results."""
    print_section_header("STEP 4: Route Analysis")

    print(f"Analyzing route:")
    print(f"  Start: {start}")
    print(f"  End: {end}")
    print()

    results = []

    for personality, agent in agents.items():
        print(f"Running {personality} agent...")
        try:
            # Note: This requires actual route data
            # For demo purposes, we're showing the structure
            # In production, you'd call agent.run(start, end)
            print(f"  {agent.metadata.name} would analyze this route")
            print(f"  (Skipping actual route generation for demo)")

            # Mock result structure
            mock_result = {
                "metadata": {
                    "name": agent.metadata.name,
                    "agent_id": agent.metadata.agent_id,
                },
                "decision": {
                    "recommendation": "accept" if personality != "safety" else "reject",
                    "confidence": 0.85 if personality == "balanced" else 0.75,
                    "weighted_score": 7.5 if personality == "scenic" else 6.8,
                    "highlights": [
                        f"Excellent {list(agent.personality.dimension_weights.keys())[0]}"
                    ] if personality == "scenic" else [],
                    "concerns": [
                        "High volatility" if personality == "safety" else ""
                    ],
                },
                "result": {
                    "message": f"{agent.metadata.name}: Route analysis complete"
                }
            }

            results.append(mock_result)

        except Exception as e:
            logger.error(f"Error with {personality} agent", error=str(e))
            print(f"  ✗ Error: {e}")

        print()

    return results


def demo_comparison_modes(framework_id: str = "sagai_2025"):
    """Demonstrate semantic vs framework-specific modes."""
    print_section_header("STEP 5: Semantic vs Framework-Specific Modes")

    print("Creating safety agent with SEMANTIC mapping:")
    semantic_agent = WalkingAgent.from_preset(
        "safety",
        framework_id,
        agent_id="safety_semantic",
        use_semantic=True
    )
    print(f"  ✓ {semantic_agent.metadata.name}")
    print(f"  Weights: {semantic_agent.personality.dimension_weights}")
    print()

    print("Creating safety agent with FRAMEWORK-SPECIFIC config:")
    try:
        specific_agent = WalkingAgent.from_preset(
            "safety",
            framework_id,
            agent_id="safety_specific",
            use_semantic=False
        )
        print(f"  ✓ {specific_agent.metadata.name}")
        print(f"  Weights: {specific_agent.personality.dimension_weights}")
    except ValueError as e:
        print(f"  ⚠️  Framework-specific config not available: {e}")
        print(f"  (This is expected if no explicit config exists for {framework_id})")

    print()


def main():
    """Main demonstration workflow."""
    print()
    print_separator("=")
    print("  CityWalkAgent - Agent-Based Route Analysis Demo")
    print_separator("=")
    print()

    # Configuration
    framework_id = "sagai_2025"  # Or "streetagent_5d" depending on your setup
    test_start = (40.7589, -73.9851)  # Times Square
    test_end = (40.7614, -73.9776)    # Near Central Park

    try:
        # Step 1: List available personalities
        demo_personality_listing()

        # Step 2: Create agents
        agents = demo_create_agents(framework_id)

        # Step 3: Show configurations
        demo_show_agent_configs(agents)

        # Step 4: Analyze route (mocked for demo)
        results = demo_analyze_route(agents, test_start, test_end)

        # Step 5: Compare results
        if results:
            for result in results:
                print_decision(
                    result["metadata"]["name"],
                    result
                )

            compare_agents(results)

        # Step 6: Show different configuration modes
        demo_comparison_modes(framework_id)

        # Summary
        print_section_header("SUMMARY")
        print("✅ Demonstrated:")
        print("  • Listing available personality presets")
        print("  • Creating agents with different personalities")
        print("  • Viewing agent configurations (weights & thresholds)")
        print("  • Route analysis workflow (structure)")
        print("  • Comparing agent decisions")
        print("  • Semantic vs framework-specific modes")
        print()
        print("📝 Note: This demo uses mock data for route analysis.")
        print("   For actual route evaluation, ensure you have:")
        print("   - Valid API keys configured")
        print("   - Route data or ability to generate routes")
        print("   - Street view images available")
        print()

    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
