"""Demo: Dual-system evaluation with progressive narrative."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent.walking_agent import WalkingAgent
from src.utils.visualization import plot_dual_system_analysis
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run dual-system analysis demo."""
    # Configuration
    start = (40.7589, -73.9851)  # Times Square
    end = (40.7614, -73.9776)  # Bryant Park
    interval = 50  # meters
    output_dir = Path("outputs/dual_system_demo")

    logger.info("Starting dual-system demo")
    logger.info(f"Route: {start} → {end}")
    logger.info(f"Interval: {interval}m")

    # Initialize agent
    agent = WalkingAgent.from_preset("balanced", "sagai_2025")

    # Run analysis with memory
    result = agent.run_with_memory(
        start=start,
        end=end,
        interval=interval,
        output_dir=output_dir,
    )

    # Print statistics
    stats = result["statistics"]
    dual_stats = stats.get("dual_system", {})

    print("\n" + "=" * 60)
    print("DUAL-SYSTEM ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nRoute: {result['route_data']['route_id']}")
    print(f"Distance: {stats.get('route_length_km', 0.0):.2f} km")
    print(f"Waypoints: {stats.get('total_waypoints', 0)}")

    print(
        f"\nSystem 2 Triggers: {dual_stats.get('system2_triggered', 0)} "
        f"({dual_stats.get('system2_trigger_rate', 0.0):.1%})"
    )
    print(f"Avg Score Adjustment: {dual_stats.get('avg_score_adjustment', 0.0):+.2f}")
    print(f"  Positive adjustments: {dual_stats.get('positive_adjustments', 0)}")
    print(f"  Negative adjustments: {dual_stats.get('negative_adjustments', 0)}")
    print(f"  No change: {dual_stats.get('no_change', 0)}")

    print(f"\nNarrative Chapters: {len(result.get('narrative_chapters', []))}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_dir = output_dir / "visualizations"
    viz_paths = plot_dual_system_analysis(
        waypoint_results=result["analysis_results"],
        narrative_chapters=result.get("narrative_chapters"),
        output_dir=viz_dir,
    )

    print("\nVisualizations saved:")
    for name, path in viz_paths.items():
        print(f"  - {name}: {path}")

    print(f"\nComplete narrative: {output_dir / 'complete_narrative.md'}")

    print("\n" + "=" * 60)
    print("✅ Dual-system demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
