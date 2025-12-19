"""Demo: Dual VLM Evaluation - Neutral vs Persona-Aware Scoring.

This demo showcases the dual VLM evaluation system that compares:
- Neutral evaluation (no persona bias)
- Persona-aware evaluation (with personality-specific hints)

The system makes TWO VLM calls per waypoint to reveal how persona hints
influence perception and scoring of walking environments.

Examples:
    # Run with parent personality to see safety bias
    python examples/dual_system_demo.py --personality parent_with_kids

    # Run photographer personality to see aesthetics bias
    python examples/dual_system_demo.py --personality photographer

    # Compare runner vs elderly walker
    python examples/dual_system_demo.py --personality runner
    python examples/dual_system_demo.py --personality elderly_walker --output-dir outputs/elderly_demo

    # Use existing route images
    python examples/dual_system_demo.py --route-folder data/images/singapore/ --personality homebuyer

    # Visualize only (skip re-running VLM)
    python examples/dual_system_demo.py --visualize-only --output-dir outputs/dual_system_demo
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent.walking_agent import WalkingAgent
from src.utils.visualization import RouteVisualizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_existing_results(output_dir: Path) -> dict:
    """Load prior run outputs to regenerate visualizations without rerunning VLM."""
    analysis_path = output_dir / "analysis_results.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"analysis_results.json not found in {output_dir}. "
            "Run the demo once without --visualize-only to generate it."
        )

    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_results = json.load(f)

    return {"analysis_results": analysis_results}


def generate_persona_visualizations(
    analysis_results: list,
    output_dir: Path,
    framework_id: str,
    personality_name: str,
) -> dict:
    """Generate visualizations comparing neutral vs persona-aware evaluations.

    Args:
        analysis_results: List of waypoint result dicts with neutral_scores and persona scores
        output_dir: Directory to save visualizations
        framework_id: Framework ID for dimension labels
        personality_name: Name of personality for titles

    Returns:
        Dict mapping visualization names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz = RouteVisualizer(framework_id=framework_id)

    print("\n" + "=" * 60)
    print("GENERATING PERSONA IMPACT VISUALIZATIONS")
    print("=" * 60)

    viz_paths = {}

    # 1. Persona comparison (line plots with arrows)
    print("\n1. Creating neutral vs persona comparison plots...")
    comparison_path = output_dir / "persona_comparison.png"
    viz.plot_persona_comparison(
        waypoint_results=analysis_results,
        title=f"Neutral vs Persona-Aware Evaluation ({personality_name})",
        save_path=comparison_path,
    )
    viz_paths["persona_comparison"] = comparison_path

    # 2. Summary radar chart
    print("2. Creating persona impact radar chart...")
    radar_path = output_dir / "persona_summary_radar.png"
    viz.plot_persona_summary_radar(
        waypoint_results=analysis_results,
        save_path=radar_path,
    )
    viz_paths["persona_radar"] = radar_path

    # 3. Delta distribution (histograms + box plot)
    print("3. Creating persona adjustment distribution...")
    delta_path = output_dir / "persona_delta_distribution.png"
    viz.plot_persona_delta_distribution(
        waypoint_results=analysis_results,
        save_path=delta_path,
    )
    viz_paths["persona_delta"] = delta_path

    # 4. Individual score plots (neutral and persona separately)
    print("4. Creating individual score timeline plots...")

    # Extract neutral and persona scores
    waypoint_ids = [str(r["waypoint_id"]) for r in analysis_results]
    system2_triggers = [str(r["waypoint_id"]) for r in analysis_results if r.get("system2_triggered")]

    # Get dimensions from first result
    first_result = analysis_results[0]
    neutral_scores_dict = first_result.get("neutral_scores", first_result.get("system1_scores", {}))
    dimensions = list(neutral_scores_dict.keys())

    # Build neutral scores dict
    neutral_scores = {dim: [] for dim in dimensions}
    for r in analysis_results:
        n_scores = r.get("neutral_scores", r.get("system1_scores", {}))
        for dim in dimensions:
            neutral_scores[dim].append(n_scores.get(dim, 0))

    # Build persona scores dict
    persona_scores = {dim: [] for dim in dimensions}
    for r in analysis_results:
        p_scores = r.get("scores", {})
        for dim in dimensions:
            persona_scores[dim].append(p_scores.get(dim, 0))

    # Plot neutral scores
    neutral_path = output_dir / "scores_neutral.png"
    viz.plot_scores_with_trends(
        scores=neutral_scores,
        waypoint_ids=waypoint_ids,
        title="Neutral Evaluation (No Persona Bias)",
        save_path=neutral_path,
        system2_triggered_waypoints=system2_triggers,
    )
    viz_paths["neutral_scores"] = neutral_path

    # Plot persona-aware scores
    persona_path = output_dir / "scores_persona_aware.png"
    viz.plot_scores_with_trends(
        scores=persona_scores,
        waypoint_ids=waypoint_ids,
        title=f"Persona-Aware Evaluation ({personality_name})",
        save_path=persona_path,
        system2_triggered_waypoints=system2_triggers,
    )
    viz_paths["persona_scores"] = persona_path

    print("\n✅ All visualizations generated!")
    return viz_paths


def compute_persona_statistics(analysis_results: list) -> dict:
    """Compute statistics about persona impact on scoring.

    Args:
        analysis_results: List of waypoint result dicts

    Returns:
        Dict with persona impact statistics
    """
    import numpy as np

    # Collect adjustments
    all_adjustments = []
    dimension_adjustments = {}

    for result in analysis_results:
        adjustments = result.get("persona_adjustments", {})
        if adjustments:
            for dim, adj in adjustments.items():
                all_adjustments.append(adj)
                dimension_adjustments.setdefault(dim, []).append(adj)

    if not all_adjustments:
        return {
            "total_waypoints": len(analysis_results),
            "persona_applied": False,
            "mean_adjustment": 0.0,
            "median_adjustment": 0.0,
            "max_positive": 0.0,
            "max_negative": 0.0,
        }

    # Overall statistics
    all_adjustments = np.array(all_adjustments)
    positive_adjustments = all_adjustments[all_adjustments > 0]
    negative_adjustments = all_adjustments[all_adjustments < 0]

    # Per-dimension statistics
    dim_stats = {}
    for dim, adjs in dimension_adjustments.items():
        adjs_arr = np.array(adjs)
        dim_stats[dim] = {
            "mean": float(np.mean(adjs_arr)),
            "median": float(np.median(adjs_arr)),
            "std": float(np.std(adjs_arr)),
            "min": float(np.min(adjs_arr)),
            "max": float(np.max(adjs_arr)),
        }

    return {
        "total_waypoints": len(analysis_results),
        "persona_applied": True,
        "mean_adjustment": float(np.mean(all_adjustments)),
        "median_adjustment": float(np.median(all_adjustments)),
        "std_adjustment": float(np.std(all_adjustments)),
        "max_positive": float(np.max(positive_adjustments)) if len(positive_adjustments) > 0 else 0.0,
        "max_negative": float(np.min(negative_adjustments)) if len(negative_adjustments) > 0 else 0.0,
        "positive_count": int(np.sum(all_adjustments > 0.1)),
        "negative_count": int(np.sum(all_adjustments < -0.1)),
        "neutral_count": int(np.sum(np.abs(all_adjustments) <= 0.1)),
        "dimension_stats": dim_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual VLM evaluation demo: Neutral vs Persona-Aware scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Personalities Available:
  - homebuyer: Family residence evaluation (focus: safety, convenience, amenities)
  - runner: Running route suitability (focus: smooth surfaces, width, air quality)
  - parent_with_kids: Child safety paramount (focus: traffic separation, stroller access)
  - photographer: Photogenic scenes (focus: visual interest, lighting, composition)
  - elderly_walker: Mobility considerations (focus: smooth surfaces, rest areas, gentle slopes)

What This Demo Shows:
  The system makes TWO VLM calls per waypoint:
  1. Neutral evaluation (no persona bias)
  2. Persona-aware evaluation (with personality-specific hints)

  Visualizations reveal how personas influence perception and scoring.
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dual_vlm_demo"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip VLM calls and only render visualizations from existing outputs.",
    )
    parser.add_argument(
        "--framework-id",
        default="streetagent_5d",
        help="Evaluation framework to use.",
    )
    parser.add_argument(
        "--personality",
        default="homebuyer",
        choices=["homebuyer", "runner", "parent_with_kids", "photographer", "elderly_walker"],
        help="Enhanced personality to use (default: homebuyer).",
    )
    parser.add_argument(
        "--route-folder",
        type=Path,
        help="Path to existing route images folder. "
        "If provided, analyze these images instead of generating new ones.",
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
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=30,
        help="pHash threshold for System 2 trigger and multi-image evaluation (default: 30). "
        "Lower values = more sensitive to visual changes.",
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
    phash_threshold = args.phash_threshold

    if args.visualize_only:
        logger.info(
            "Visualize-only mode: loading existing outputs",
            extra={"output_dir": str(output_dir)},
        )
        result = _load_existing_results(output_dir)
        analysis_results = result["analysis_results"]
    else:
        # Show personality info
        from src.agent.config import get_enhanced_personality

        print("\n" + "=" * 70)
        print("DUAL VLM EVALUATION: NEUTRAL vs PERSONA-AWARE")
        print("=" * 70)
        print(f"\nSelected personality: {personality}")

        try:
            enhanced_config = get_enhanced_personality(personality)
            print(f"\nPersonality: {enhanced_config.name}")
            print(f"Description: {enhanced_config.description}")

            # Show persona hint
            if hasattr(enhanced_config, 'system1_persona_hint') and enhanced_config.system1_persona_hint:
                print(f"\nPersona Hint Preview:")
                hint_preview = enhanced_config.system1_persona_hint.strip().split('\n')[0]
                print(f"  {hint_preview}...")

            print(f"\nScoring Configuration:")
            print(f"  Concern keywords: {len(enhanced_config.scoring_rules.concern_keywords)}")
            print(f"  Boost keywords: {len(enhanced_config.scoring_rules.boost_keywords)}")
        except ValueError as e:
            print(f"Error loading personality: {e}")
            return

        print("\n" + "=" * 70)
        print("VLM EVALUATION STRATEGY:")
        print("  For each waypoint, the system makes TWO VLM calls:")
        print("  1. 🔷 Neutral: Unbiased evaluation (no persona hint)")
        print("  2. 🔶 Persona: Evaluation with personality-specific hint")
        print("  ")
        print("  This reveals how personas shift perception and scoring.")
        print("=" * 70 + "\n")

        # Initialize agent
        agent = WalkingAgent.from_preset(
            preset_name=personality,
            framework_id=framework_id,
        )

        # Configure thresholds BEFORE accessing analyzers (they are lazy-loaded)
        agent.set_thresholds(
            phash_threshold=phash_threshold,
        )
        logger.info(
            f"Threshold configured: phash={phash_threshold}"
        )

        # NOTE: The dual VLM calls are now automatic in ContinuousAnalyzer!
        # No additional configuration needed - it will make both calls by default.

        if route_folder:
            logger.info("Starting dual VLM evaluation with existing route images")
            logger.info(f"Route folder: {route_folder}")

            result = agent.run_with_memory_from_folder(
                route_folder=route_folder,
                output_dir=output_dir,
            )
        else:
            logger.info("Starting dual VLM evaluation with new route")
            logger.info(f"Route: {start} → {end}")
            logger.info(f"Interval: {interval}m")

            result = agent.run_with_memory(
                start=start,
                end=end,
                interval=interval,
                output_dir=output_dir,
            )

        analysis_results = result.get("analysis_results", [])

        # Print persona impact statistics
        stats = compute_persona_statistics(analysis_results)

        print("\n" + "=" * 70)
        print("PERSONA IMPACT ANALYSIS")
        print("=" * 70)

        route_data = result.get("route_data", {})
        if route_data:
            print(f"\nRoute: {route_data.get('route_id')}")

        route_length = result.get("statistics", {}).get("route_length_km", 0.0)
        print(f"Distance: {route_length:.2f} km")
        print(f"Waypoints: {stats['total_waypoints']}")
        print(f"Personality: {personality}")

        if stats["persona_applied"]:
            print(f"\n📊 Persona Impact Summary:")
            print(f"  Mean adjustment: {stats['mean_adjustment']:+.2f}")
            print(f"  Median adjustment: {stats['median_adjustment']:+.2f}")
            print(f"  Std deviation: {stats['std_adjustment']:.2f}")
            print(f"  Max positive boost: +{stats['max_positive']:.2f}")
            print(f"  Max negative penalty: {stats['max_negative']:.2f}")

            print(f"\n📈 Adjustment Distribution:")
            print(f"  Positive adjustments (>0.1): {stats['positive_count']}")
            print(f"  Negative adjustments (<-0.1): {stats['negative_count']}")
            print(f"  Neutral (≈0): {stats['neutral_count']}")

            # Show per-dimension stats
            dim_stats = stats.get("dimension_stats", {})
            if dim_stats:
                print(f"\n🎯 Per-Dimension Impact:")
                for dim, ds in sorted(dim_stats.items(), key=lambda x: abs(x[1]["mean"]), reverse=True):
                    print(f"  {dim}:")
                    print(f"    Mean: {ds['mean']:+.2f}, Range: [{ds['min']:.2f}, {ds['max']:.2f}]")
        else:
            print("\n⚠️  No persona adjustments detected (persona hint may not be configured)")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    viz_dir = output_dir / "visualizations"
    viz_paths = {}

    try:
        # 1. Generate persona comparison visualizations (neutral vs persona-aware)
        print("\n=== Persona Impact Visualizations ===")
        persona_viz_paths = generate_persona_visualizations(
            analysis_results=analysis_results,
            output_dir=viz_dir,
            framework_id=framework_id,
            personality_name=personality,
        )
        viz_paths.update(persona_viz_paths)

        # 2. Generate traditional dual-system visualizations (System 1 vs System 2)
        print("\n=== Memory/Thinking System Visualizations ===")
        from src.utils.visualization import plot_dual_system_analysis

        # Load narrative chapters if available
        narrative_chapters = None
        narrative_json = output_dir / "narrative_chapters.json"
        if narrative_json.exists():
            import json
            with open(narrative_json, "r", encoding="utf-8") as f:
                narrative_chapters = json.load(f)
            print(f"Loaded {len(narrative_chapters)} narrative chapters")

        system_viz_paths = plot_dual_system_analysis(
            waypoint_results=analysis_results,
            narrative_chapters=narrative_chapters,
            output_dir=viz_dir,
            framework_id=framework_id,
            generate_radar_sets=False,  # Skip radar sets for speed
        )
        viz_paths.update(system_viz_paths)

        print("\n📁 All visualizations saved:")
        print("\nPersona Impact (Neutral vs Persona-Aware):")
        for name in ["persona_comparison", "persona_radar", "persona_delta", "neutral_scores", "persona_scores"]:
            if name in viz_paths:
                print(f"  ✓ {viz_paths[name].name}")

        print("\nMemory/Thinking System (System 1 vs System 2):")
        for name in ["comparison", "heatmap", "system1", "final_scores", "narrative"]:
            if name in viz_paths:
                print(f"  ✓ {viz_paths[name].name}")

        print(f"\n📂 All outputs in: {output_dir}")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        print(f"\n❌ Visualization error: {e}")

    print("\n" + "=" * 70)
    print("✅ Dual VLM demo complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. View visualizations in: {viz_dir}")
    print(f"  2. Compare with other personalities:")
    print(f"     python examples/dual_system_demo.py --personality photographer")
    print(f"  3. Re-visualize without re-running VLM:")
    print(f"     python examples/dual_system_demo.py --visualize-only")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
