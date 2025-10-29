"""
Example 04: Agent-based Route Analysis with Real Data

Demonstrates WalkingAgent with different personalities analyzing real routes.
Shows the complete cognitive pipeline: Observe → Analyze → Think → Act → Remember

This example:
1. Lists available personality presets
2. Creates agents with different personalities
3. Analyzes a real route with each agent (full cognitive pipeline)
4. Compares their decisions and explanations
5. Shows memory and state management
6. Demonstrates cache functionality

Prerequisites:
- VLM API key (Claude recommended)
- Google Maps API key (for route generation)
- Or existing route data with images
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import WalkingAgent, list_presets
from src.utils.logging import get_logger
from src.config import settings

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


def print_cognitive_pipeline(result: Dict[str, Any]):
    """Print details of the cognitive pipeline execution."""
    print("🧠 Cognitive Pipeline Execution:")
    print()

    # 1. Observation Phase
    perception = result.get("perception", {})
    route_info = perception.get("route_info", {})
    print(f"  1️⃣  OBSERVE")
    print(f"      Route ID: {route_info.get('route_id', 'N/A')}")
    print(f"      Images evaluated: {route_info.get('num_images', 0)}")
    print(f"      Framework: {perception.get('framework_id', 'N/A')}")

    dimension_stats = perception.get("dimension_stats", {})
    if dimension_stats:
        print(f"      Dimension scores:")
        for dim_id, stats in list(dimension_stats.items())[:3]:  # Show first 3
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            print(f"        • {dim_id}: {mean:.1f} ± {std:.1f}")
    print()

    # 2. Analysis Phase
    decision = result.get("decision", {})
    seq_analysis = decision.get("sequential_analysis", {})
    print(f"  2️⃣  ANALYZE")
    print(f"      Pattern type: {seq_analysis.get('pattern_type', 'N/A')}")
    print(f"      Volatility: {seq_analysis.get('volatility', 0):.2f}")
    print(f"      Barriers detected: {len(seq_analysis.get('barriers', []))}")
    print(f"      Sequential score: {seq_analysis.get('sequential_score', 0):.1f}/10")
    print()

    # 3. Thinking Phase
    print(f"  3️⃣  THINK")
    print(f"      Recommendation: {decision.get('recommendation', 'N/A').upper()}")
    print(f"      Confidence: {decision.get('confidence', 0):.2%}")
    print(f"      Weighted score: {decision.get('weighted_score', 0):.1f}/10")
    print()

    # 4. Action Phase
    action = result.get("result", {})
    print(f"  4️⃣  ACT")
    print(f"      Action: {action.get('action_type', 'N/A')}")
    print(f"      {action.get('message', 'No message')}")
    print()

    # 5. Memory Phase
    state = result.get("state", {})
    print(f"  5️⃣  REMEMBER")
    print(f"      Memory count: {state.get('memory_count', 0)}")
    print(f"      Routes evaluated: {len(state.get('evaluated_routes', []))}")
    print(f"      Waypoint progress: {state.get('waypoint_index', 0)}/{state.get('total_waypoints', 0)}")
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


def demo_analyze_route_real(agents: Dict[str, WalkingAgent], start, end, route_id: str = None):
    """Analyze a REAL route with multiple agents and compare results.

    This demonstrates the full cognitive pipeline with real data.
    """
    print_section_header("STEP 4: Real Route Analysis (Full Cognitive Pipeline)")

    print(f"Analyzing route:")
    print(f"  Start: {start}")
    print(f"  End: {end}")
    if route_id:
        print(f"  Route ID: {route_id}")
    print()

    results = []

    for i, (personality, agent) in enumerate(agents.items(), 1):
        print(f"[{i}/{len(agents)}] Running {agent.metadata.name}...")
        print()

        try:
            # REAL AGENT RUN - Full cognitive pipeline
            result = agent.run(
                start=start,
                end=end,
                route_id=route_id,  # Use same route_id for cache
                interval=50  # Waypoint interval in meters
            )

            # Check if data came from cache
            if route_id:
                print(f"  📦 Cache utilized for consistent comparison")

            # Show the full cognitive pipeline
            print_cognitive_pipeline(result)

            results.append(result)

            print(f"  ✓ {agent.metadata.name} completed analysis")
            print()

        except Exception as e:
            logger.error(f"Error with {personality} agent", error=str(e))
            print(f"  ✗ Error: {e}")
            print()
            # Re-raise if it's a critical error (like missing API keys)
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise

        print("-" * 80)
        print()

    return results


def demo_cache_functionality(agent: WalkingAgent, start, end):
    """Demonstrate cache functionality with the same agent."""
    print_section_header("STEP 5: Cache Functionality")

    print("Running agent twice on the same route to demonstrate caching...")
    print()

    # First run
    print("First run (cache miss - will generate route and evaluate):")
    result1 = agent.run(start=start, end=end, interval=50)
    print(f"  ✓ Completed")
    print(f"  Memory count: {result1['state']['memory_count']}")
    print()

    # Second run with same route
    print("Second run (cache hit - will use cached data):")
    result2 = agent.run(start=start, end=end, interval=50)
    print(f"  ✓ Completed")
    print(f"  Memory count: {result2['state']['memory_count']}")
    print()

    # Compare
    same_route = result1["route_id"] == result2["route_id"]
    print(f"Same route ID: {same_route}")
    print(f"Route ID: {result1['route_id']}")
    print()


def check_prerequisites():
    """Check if required API keys are configured."""
    print_section_header("Checking Prerequisites")

    missing = []

    # Check VLM API key
    if not settings.qwen_vlm_api_key and not hasattr(settings, 'openai_api_key'):
        missing.append("VLM API key (QWEN_VLM_API_KEY or OPENAI_API_KEY)")
    else:
        print("✓ VLM API key configured")

    # Check Google Maps API key (optional but recommended)
    if not settings.google_maps_api_key or settings.google_maps_api_key == "test_key":
        print("⚠️  Google Maps API key not configured (will use simple routes)")
    else:
        print("✓ Google Maps API key configured")

    print()

    if missing:
        print("❌ Missing prerequisites:")
        for item in missing:
            print(f"   • {item}")
        print()
        print("Please configure in .env file or environment variables.")
        print("See README.md for setup instructions.")
        return False

    return True


def main():
    """Main demonstration workflow."""
    print()
    print_separator("=")
    print("  CityWalkAgent - Real Agent-Based Route Analysis")
    print_separator("=")
    print()

    # Check prerequisites
    if not check_prerequisites():
        print()
        print("⚠️  Running in demo mode with limited functionality")
        print()

    # Configuration
    framework_id = "sagai_2025"

    # Real coordinates - Manhattan example
    # Times Square to near Central Park
    test_start = (40.7589, -73.9851)
    test_end = (40.7614, -73.9776)

    # For cache testing - use a deterministic route_id
    # This ensures all agents analyze the same route
    route_id = f"demo_route_{test_start[0]:.6f}_{test_start[1]:.6f}_{test_end[0]:.6f}_{test_end[1]:.6f}_50"

    try:
        # Step 1: List available personalities
        demo_personality_listing()

        # Step 2: Create agents
        agents = demo_create_agents(framework_id)

        # Step 3: Show configurations
        demo_show_agent_configs(agents)

        # Step 4: REAL route analysis with full cognitive pipeline
        print("⚠️  Note: This will make real API calls for VLM evaluation")
        print("         Ensure you have API keys configured and credits available")
        print()

        results = demo_analyze_route_real(
            agents,
            test_start,
            test_end,
            route_id=route_id  # Same route_id ensures cache works
        )

        # Step 5: Compare results
        if results:
            print_section_header("DETAILED DECISIONS")
            for result in results:
                print_decision(
                    result["metadata"]["name"],
                    result
                )
                print()

            compare_agents(results)

        # Step 6: Demonstrate cache
        if results:
            print("💡 Tip: The second and third agents likely used cached data")
            print("   from the first agent's route generation and evaluation.")
            print()

        # Summary
        print_section_header("SUMMARY")
        print("✅ Demonstrated:")
        print("  • Listing available personality presets")
        print("  • Creating agents with different personalities")
        print("  • REAL route analysis with full cognitive pipeline:")
        print("    - Observe: VLM evaluation of street view images")
        print("    - Analyze: Sequential pattern analysis")
        print("    - Think: Personality-driven decision making")
        print("    - Act: Formatted recommendation output")
        print("    - Remember: Experience storage in memory")
        print("  • Comparing agent decisions and reasoning")
        print("  • Cache functionality for efficiency")
        print()
        print("📊 Results:")
        if results:
            print(f"  • {len(results)} agents successfully analyzed the route")
            print(f"  • Route ID: {results[0]['route_id']}")
            print(f"  • Total waypoints: {results[0]['state']['total_waypoints']}")
            print(f"  • Memory entries: {results[0]['state']['memory_count']}")
        print()
        print("💾 Data stored in:")
        print("  • Agent memory: data/agent_memory/")
        print("  • Pipeline results: data/results/pipeline_runs/")
        print()

    except KeyError as e:
        logger.error("Configuration error", error=str(e))
        print(f"❌ Configuration Error: {e}")
        print("   Please check your API keys and configuration.")
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
