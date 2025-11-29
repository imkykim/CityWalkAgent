"""Demo: Dual-system evaluation with progressive narrative."""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent.walking_agent import WalkingAgent
from src.utils.visualization import plot_dual_system_analysis
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_existing_results(output_dir: Path) -> dict:
    """Load prior run outputs to regenerate visualizations without rerunning the agent."""
    analysis_path = output_dir / "analysis_results.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"analysis_results.json not found in {output_dir}. "
            "Run the demo once without --visualize-only to generate it."
        )

    analysis_results = json.loads(analysis_path.read_text())

    # Get waypoint IDs from this run's analysis
    current_waypoint_ids = {r["waypoint_id"] for r in analysis_results}

    # Try multiple sources for narrative chapters (in order of preference)
    narrative_chapters = None

    # 1. Check for JSON in output dir
    narrative_json = output_dir / "narrative_chapters.json"
    if narrative_json.exists():
        narrative_chapters = json.loads(narrative_json.read_text())
        # Filter to current run's waypoints
        narrative_chapters = [
            ch
            for ch in narrative_chapters
            if ch.get("waypoint_id") in current_waypoint_ids
        ]
    else:
        # 2. Check for JSONL in agent memory (common location)
        from src.config import settings

        memory_dir = settings.data_dir / "agent_memory"
        # Try to find the narrative JSONL file
        narrative_jsonl_files = list(memory_dir.glob("*_narrative.jsonl"))
        if narrative_jsonl_files:
            # Use the most recent one
            narrative_jsonl = max(
                narrative_jsonl_files, key=lambda p: p.stat().st_mtime
            )
            all_chapters = []
            with open(narrative_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_chapters.append(json.loads(line))

            # Filter to only chapters matching this run's waypoint IDs
            # and take the most recent chapter for each waypoint
            waypoint_to_chapter = {}
            for chapter in all_chapters:
                wp_id = chapter.get("waypoint_id")
                if wp_id in current_waypoint_ids:
                    # Keep the most recent chapter for this waypoint
                    if wp_id not in waypoint_to_chapter or chapter.get(
                        "timestamp", ""
                    ) > waypoint_to_chapter[wp_id].get("timestamp", ""):
                        waypoint_to_chapter[wp_id] = chapter

            # Sort by waypoint_id to maintain order
            narrative_chapters = (
                sorted(
                    waypoint_to_chapter.values(),
                    key=lambda ch: ch.get("waypoint_id", 0),
                )
                if waypoint_to_chapter
                else None
            )

    return {
        "analysis_results": analysis_results,
        "narrative_chapters": narrative_chapters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-system demo / visualization helper"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dual_system_demo"),
        help="Directory containing or writing outputs.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip running the agent and only render visualizations from existing outputs.",
    )
    parser.add_argument(
        "--framework-id",
        default="streetagent_5d",
        help="Framework id to use for agent (when running) and visualization labels/colors.",
    )
    parser.add_argument(
        "--route-folder",
        type=Path,
        help="Path to existing route images folder (e.g., data/images/singapore/). "
        "If provided, the agent will analyze these images instead of generating new ones.",
    )
    args = parser.parse_args()

    # Configuration
    start = (22.3298, 114.1630)  # Times Square
    end = (22.3250, 114.1550)  # Bryant Park
    interval = 12  # meters
    output_dir = args.output_dir
    framework_id = args.framework_id
    route_folder = args.route_folder

    if args.visualize_only:
        logger.info(
            "Visualize-only mode: loading existing outputs",
            extra={"output_dir": str(output_dir)},
        )
        result = _load_existing_results(output_dir)
    else:
        # Initialize agent
        agent = WalkingAgent.from_preset(
            preset_name="safety",  # personality
            framework_id=framework_id,
        )
        agent.cognitive.phash_threshold = 30
        agent.continuous_analyzer.multi_image_threshold = 30
        agent.continuous_analyzer.enable_multi_image = True

        if route_folder:
            # Run with existing route images
            logger.info("Starting dual-system demo with existing route images")
            logger.info(f"Route folder: {route_folder}")

            result = agent.run_with_memory_from_folder(
                route_folder=route_folder,
                output_dir=output_dir,
            )
        else:
            # Generate new route
            logger.info("Starting dual-system demo")
            logger.info(f"Route: {start} → {end}")
            logger.info(f"Interval: {interval}m")

            # Run analysis with memory
            result = agent.run_with_memory(
                start=start,
                end=end,
                interval=interval,
                output_dir=output_dir,
            )

        # Print statistics
        stats = result.get("statistics", {})
        dual_stats = stats.get("dual_system", {})

        print("\n" + "=" * 60)
        print("DUAL-SYSTEM ANALYSIS RESULTS")
        print("=" * 60)

        route_data = result.get("route_data", {})
        if route_data:
            print(f"\nRoute: {route_data.get('route_id')}")
        print(f"Distance: {stats.get('route_length_km', 0.0):.2f} km")
        print(f"Waypoints: {stats.get('total_waypoints', 0)}")

        print(
            f"\nSystem 2 Triggers: {dual_stats.get('system2_triggered', 0)} "
            f"({dual_stats.get('system2_trigger_rate', 0.0):.1%})"
        )
        print(
            f"Avg Score Adjustment: {dual_stats.get('avg_score_adjustment', 0.0):+.2f}"
        )
        print(f"  Positive adjustments: {dual_stats.get('positive_adjustments', 0)}")
        print(f"  Negative adjustments: {dual_stats.get('negative_adjustments', 0)}")
        print(f"  No change: {dual_stats.get('no_change', 0)}")

        print(f"\nNarrative Chapters: {len(result.get('narrative_chapters', []))}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    narrative_chapters = result.get("narrative_chapters")
    if narrative_chapters:
        print(f"Loaded {len(narrative_chapters)} narrative chapters")
    else:
        print("No narrative chapters available")

    viz_dir = output_dir / "visualizations"
    viz_paths = plot_dual_system_analysis(
        waypoint_results=result["analysis_results"],
        narrative_chapters=narrative_chapters,
        output_dir=viz_dir,
        framework_id=framework_id,
        generate_radar_sets=True,
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
