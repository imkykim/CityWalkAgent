"""Demo: Dual-system evaluation with enhanced personality system.

This demo showcases the dual-system cognitive architecture with enhanced
personality-driven score transformations.

Examples:
    # Run with parent personality (extreme safety focus)
    python examples/dual_system_demo.py --personality parent_with_kids

    # Run photographer personality (aesthetics focus)
    python examples/dual_system_demo.py --personality photographer

    # Compare safety vs scenic
    python examples/dual_system_demo.py --personality safety
    python examples/dual_system_demo.py --personality scenic --output-dir outputs/scenic_demo

    # Use existing route images
    python examples/dual_system_demo.py --route-folder data/images/singapore/ --personality runner

    # Custom coordinates
    python examples/dual_system_demo.py --start "22.3298,114.1630" --end "22.3250,114.1550" --personality elderly_walker
"""

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
        description="Dual-system demo with enhanced personality system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Personalities Available:
  Basic presets (automatically use enhanced versions):
    - safety (→ parent_with_kids): Extreme safety focus, stroller accessibility
    - scenic (→ photographer): Visual interest, aesthetics prioritized
    - balanced (→ homebuyer): Practical livability, family-friendly
    - comfort (→ elderly_walker): Accessibility, rest areas, smooth surfaces
    - explorer (→ photographer): Discovery and visual complexity
    - technical: Neutral analysis (no enhancement)

  Direct enhanced personalities:
    - homebuyer: Family residence evaluation
    - runner: Running route suitability
    - parent_with_kids: Child safety paramount
    - photographer: Photogenic scenes
    - elderly_walker: Mobility considerations
        """
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
        "--personality",
        default="safety",
        choices=["safety", "scenic", "balanced", "comfort", "explorer", "technical",
                 "homebuyer", "runner", "parent_with_kids", "photographer", "elderly_walker"],
        help="Personality preset or enhanced personality to use (default: safety).",
    )
    parser.add_argument(
        "--enable-system2",
        action="store_true",
        default=True,
        help="Enable System 2 deep thinking (enabled by default).",
    )
    parser.add_argument(
        "--disable-system2",
        action="store_true",
        help="Disable System 2 and use only System 1 (fast perception).",
    )
    parser.add_argument(
        "--route-folder",
        type=Path,
        help="Path to existing route images folder (e.g., data/images/singapore/). "
        "If provided, the agent will analyze these images instead of generating new ones.",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start coordinates as 'lat,lng' (default: 22.3298,114.1630).",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End coordinates as 'lat,lng' (default: 22.3250,114.1550).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=12,
        help="Waypoint interval in meters (default: 12).",
    )
    args = parser.parse_args()

    # Configuration
    start = tuple(map(float, args.start.split(','))) if args.start else (22.3298, 114.1630)
    end = tuple(map(float, args.end.split(','))) if args.end else (22.3250, 114.1550)
    interval = args.interval
    output_dir = args.output_dir
    framework_id = args.framework_id
    route_folder = args.route_folder
    personality = args.personality
    enable_system2 = args.enable_system2 and not args.disable_system2

    if args.visualize_only:
        logger.info(
            "Visualize-only mode: loading existing outputs",
            extra={"output_dir": str(output_dir)},
        )
        result = _load_existing_results(output_dir)
    else:
        # Show personality info
        from src.agent.config import (
            get_enhanced_personality,
            list_enhanced_personalities,
            PERSONALITY_ENHANCEMENT_MAP
        )

        print("\n" + "=" * 60)
        print("ENHANCED PERSONALITY SYSTEM")
        print("=" * 60)
        print(f"\nSelected personality: {personality}")

        # Check if this maps to an enhanced personality
        enhanced_id = None
        if personality in PERSONALITY_ENHANCEMENT_MAP:
            enhanced_id = PERSONALITY_ENHANCEMENT_MAP[personality]
            if enhanced_id:
                print(f"  → Maps to enhanced: {enhanced_id}")
        elif personality in list_enhanced_personalities():
            enhanced_id = personality
            print(f"  → Using enhanced personality directly")

        if enhanced_id:
            try:
                enhanced_config = get_enhanced_personality(enhanced_id)
                print(f"\nEnhanced Personality: {enhanced_config.name}")
                print(f"Description: {enhanced_config.description}")
                print(f"Feature modifiers: {len(enhanced_config.scoring_rules.feature_modifiers)}")
                print(f"Sensitivity multipliers: {len(enhanced_config.scoring_rules.sensitivity_multipliers)}")
                print(f"Concern keywords: {len(enhanced_config.scoring_rules.concern_keywords)}")
                print(f"Boost keywords: {len(enhanced_config.scoring_rules.boost_keywords)}")
            except ValueError:
                print("  (No enhanced config available)")
        else:
            print("  (Using basic personality - no enhancement)")

        print(f"\nSystem 2 enabled: {enable_system2}")
        print("=" * 60 + "\n")

        # Initialize agent
        agent = WalkingAgent.from_preset(
            preset_name=personality,
            framework_id=framework_id,
        )

        # Configure System 2 thresholds
        if enable_system2:
            agent.cognitive.phash_threshold = 30
            agent.continuous_analyzer.multi_image_threshold = 30
            agent.continuous_analyzer.enable_multi_image = True
            agent.continuous_analyzer.adaptive = False
        else:
            # Effectively disable System 2 by setting very high thresholds
            agent.thinking_module.enable_score_revision = False
            logger.info("System 2 disabled - using only System 1 perception")

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
        print(f"Personality: {personality}")

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

        # Show enhanced personality impact
        analysis_results = result.get("analysis_results", [])
        system2_results = [r for r in analysis_results if r.get("system2_result")]

        if system2_results:
            print(f"\n--- Enhanced Personality Impact ---")

            # Count transformation applications
            transformations_applied = sum(
                1 for r in system2_results
                if r.get("system2_result", {}).get("memory_influence", {}).get("transformation_applied", False)
            )
            print(f"Transformations applied: {transformations_applied}/{len(system2_results)}")

            # Collect all detected features
            all_features = []
            for r in system2_results:
                memory_inf = r.get("system2_result", {}).get("memory_influence", {})
                features = memory_inf.get("detected_features", [])
                all_features.extend(features)

            if all_features:
                from collections import Counter
                feature_counts = Counter(all_features)
                print(f"\nTop detected features:")
                for feature, count in feature_counts.most_common(5):
                    print(f"  - {feature}: {count}x")

            # Show personality impact levels
            personality_impacts = [
                r.get("system2_result", {}).get("personality_factor", "unknown")
                for r in system2_results
            ]
            from collections import Counter
            impact_counts = Counter(personality_impacts)
            print(f"\nPersonality impact distribution:")
            for impact, count in sorted(impact_counts.items(), reverse=True):
                print(f"  - {impact}: {count}")

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
