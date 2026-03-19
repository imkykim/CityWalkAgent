#!/usr/bin/env python3
"""
CityWalkAgent — main entry point.

Subcommands:
  run      Analyze a walking route (collect images + evaluate + visualize)
  collect  Collect Street View images along a route

Usage:
  python main.py run     --route-folder data/images/my_route/ --personality homebuyer
  python main.py run     --start 37.5665,126.9780 --end 37.5512,126.9882
  python main.py collect --start 37.5665,126.9780 --end 37.5512,126.9882 --output data/images/my_route
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.core import DEFAULT_FRAMEWORK_ID, settings
from src.agent.orchestrator import CityWalkAgent
from src.utils.visualization import RouteVisualizer
from src.utils.logging import get_logger, set_global_log_level

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_coord(value: str) -> Tuple[float, float]:
    try:
        lat, lon = value.split(",")
        return float(lat.strip()), float(lon.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("Format: 'lat,lon'  e.g. 37.5665,126.9780")


def generate_persona_visualizations(
    analysis_results: list,
    output_dir: Path,
    framework_id: str,
    personality_name: str,
) -> dict:
    """Generate visualizations comparing objective vs persona-aware evaluations.

    Args:
        analysis_results: List of waypoint result dicts with objective_scores and persona_scores
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
    has_objective_scores = any(
        isinstance(r.get("objective_scores"), dict) and bool(r.get("objective_scores"))
        for r in analysis_results
    )

    if has_objective_scores:
        # 1. Persona comparison (line plots with arrows)
        print("\n1. Creating objective vs persona comparison plots...")
        comparison_path = output_dir / "persona_comparison.png"
        viz.plot_comparison(
            waypoint_results=analysis_results,
            title=f"Objective vs Persona-Aware Evaluation ({personality_name})",
            save_path=comparison_path,
            mode="persona",
        )
        viz_paths["persona_comparison"] = comparison_path

        # 2. Summary radar chart
        print("2. Creating persona impact radar chart...")
        radar_path = output_dir / "persona_summary_radar.png"
        viz.plot_summary_radar(
            waypoint_results=analysis_results,
            save_path=radar_path,
            mode="persona",
        )
        viz_paths["persona_radar"] = radar_path

        # 3. Persona delta overview (average trends + delta heatmap)
        print("3. Creating persona adjustment overview...")
        delta_path = output_dir / "persona_delta_distribution.png"
        viz.plot_overview(
            waypoint_results=analysis_results,
            save_path=delta_path,
            mode="persona",
        )
        viz_paths["persona_delta"] = delta_path
    else:
        print("\n1-3. Skipping comparison/radar/overview (objective_scores not found).")

    # 4. Individual score plots (objective and persona separately)
    print("4. Creating individual score timeline plots...")

    # Extract objective and persona scores
    waypoint_ids = [str(r["waypoint_id"]) for r in analysis_results]
    system2_triggers = [
        str(r["waypoint_id"]) for r in analysis_results if r.get("system2_triggered")
    ]

    # Build persona dimensions first (always used for final score plot).
    persona_dimensions = []
    seen_persona_dims = set()
    for r in analysis_results:
        for key in ("persona_scores", "scores"):
            score_map = r.get(key, {})
            if not isinstance(score_map, dict):
                continue
            for dim in score_map.keys():
                if dim not in seen_persona_dims:
                    seen_persona_dims.add(dim)
                    persona_dimensions.append(dim)

    if not persona_dimensions:
        print("⚠️  No persona score dimensions found; skipping individual score timeline plots.")
        print("\n✅ All visualizations generated!")
        return viz_paths

    # Plot objective scores only when objective_scores exist.
    if has_objective_scores:
        objective_scores = {dim: [] for dim in persona_dimensions}
        for r in analysis_results:
            obj_scores = r.get(
                "objective_scores", r.get("neutral_scores", r.get("system1_scores", {}))
            )
            for dim in persona_dimensions:
                objective_scores[dim].append(obj_scores.get(dim, 0))

        objective_path = output_dir / "scores_objective.png"
        viz.plot_scores(
            scores=objective_scores,
            waypoint_ids=waypoint_ids,
            title="Objective Evaluation (Research/Framework-Only)",
            save_path=objective_path,
            dimensions=persona_dimensions,
            markers=system2_triggers,
        )
        viz_paths["objective_scores"] = objective_path
    else:
        print("   - Skipping objective timeline (objective_scores not found).")

    # Build persona scores dict
    persona_scores = {dim: [] for dim in persona_dimensions}
    for r in analysis_results:
        per_scores = r.get("persona_scores", r.get("scores", {}))
        for dim in persona_dimensions:
            persona_scores[dim].append(per_scores.get(dim, 0))

    # Plot persona-aware scores
    persona_path = output_dir / "scores_persona_aware.png"
    viz.plot_scores(
        scores=persona_scores,
        waypoint_ids=waypoint_ids,
        title=f"Persona-Aware Evaluation ({personality_name}) - Final Scores",
        save_path=persona_path,
        dimensions=persona_dimensions,
        markers=system2_triggers,
    )
    viz_paths["persona_scores"] = persona_path

    print("\n✅ All visualizations generated!")
    return viz_paths


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Analyze a walking route end-to-end."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    framework_id = args.framework_id
    personality  = args.personality

    # ── Visualize-only mode ──────────────────────────────────────────────
    if args.visualize_only:
        analysis_path = output_dir / "analysis_results.json"
        if not analysis_path.exists():
            print(f"❌  analysis_results.json not found in {output_dir}")
            sys.exit(1)
        with open(analysis_path, encoding="utf-8") as f:
            analysis_results = json.load(f)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        generate_persona_visualizations(analysis_results, viz_dir, framework_id, personality)
        print(f"✅  Visualizations saved to {viz_dir}")
        return

    # ── Normal analysis mode ─────────────────────────────────────────────
    print("=" * 70)
    print(f"  Persona : {personality}")
    print(f"  Framework: {framework_id}")
    print("=" * 70)

    agent = CityWalkAgent.from_preset(preset_name=personality, framework_id=framework_id)
    agent.set_thresholds(phash_threshold=args.phash_threshold)

    if args.route_folder:
        result = agent.run_with_memory_from_folder(
            route_folder=args.route_folder,
            output_dir=output_dir,
            skip_reasoning=args.system1_only,
        )
    elif args.start and args.end:
        result = agent.run_with_memory(
            start=args.start,
            end=args.end,
            interval=args.interval,
            output_dir=output_dir,
            skip_reasoning=args.system1_only,
        )
    else:
        print("❌  Provide --route-folder OR both --start and --end.")
        sys.exit(1)

    # ── Visualizations ───────────────────────────────────────────────────
    analysis_results = result.get("analysis_results", [])
    if not analysis_results:
        print("⚠️  No analysis results — skipping visualizations.")
        return

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    generate_persona_visualizations(analysis_results, viz_dir, framework_id, personality)

    print(f"\n✅  Done. Outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# `collect` subcommand
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> None:
    """Collect Street View images along a route."""
    from src.data_collection.route_generator import RouteGenerator
    from src.data_collection.image_collector import ImageCollector

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    route_name = args.route_name or f"route_{args.start[0]:.4f}_{args.start[1]:.4f}"

    print(f"Generating route: {args.start} → {args.end}")
    route_gen = RouteGenerator(api_key=settings.google_maps_api_key)
    route = route_gen.create_google_maps_route(
        origin=args.start,
        destination=args.end,
        interval_meters=args.interval,
        route_name=route_name,
    )
    print(f"  ✓ {len(route.waypoints)} waypoints")

    print(f"Collecting Street View images → {output}")
    collector = ImageCollector(api_key=settings.google_maps_api_key)
    results = collector.collect_google_street_view_images_static(
        route,
        output_dir=str(output),
        use_route_direction=True,
        all_around=False,
        fov=90,
        pitch=-5,
        lookahead_distance=2,
        detect_corners=True,
        corner_threshold=30.0,
    )
    success = sum(1 for r in results if r.get("download_success"))
    print(f"  ✓ {success}/{len(results)} images collected")
    print(f"\nRun analysis:")
    print(f"  python main.py run --route-folder {output} --personality balanced")


# ---------------------------------------------------------------------------
# `walk` subcommand
# ---------------------------------------------------------------------------

def cmd_walk(args: argparse.Namespace) -> None:
    """Autonomous walk from start coords to destination coords."""
    import asyncio

    from src.agent.config.personalities import get_preset

    set_global_log_level(args.log_level)

    personality = get_preset(args.personality, "place_pulse_2.0")
    agent = CityWalkAgent(
        agent_id=f"walk_{args.personality}",
        personality=personality,
        framework_id="place_pulse_2.0",
    )

    result = asyncio.run(agent.autonomous_walk(
        start_lat=args.start_lat,
        start_lng=args.start_lng,
        dest_lat=args.dest_lat,
        dest_lng=args.dest_lng,
        max_steps=args.max_steps,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        save_images=args.save_images,
    ))

    print(f"\n{'✅ ARRIVED' if result['arrived'] else '⏹ MAX STEPS REACHED'}")
    print(f"Steps taken : {result['steps']}")
    print(f"Final dist  : {result['final_distance_m']}m")
    print(f"Persona     : {result['persona']}")
    if args.output_dir:
        print(f"Log saved   : {args.output_dir}/walk_log.json")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CityWalkAgent — walkability analysis pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Analyze a walking route")
    src_group = p_run.add_mutually_exclusive_group()
    src_group.add_argument("--route-folder", type=Path, help="Path to existing image folder")
    src_group.add_argument("--start", type=parse_coord, metavar="LAT,LON")
    p_run.add_argument("--end", type=parse_coord, metavar="LAT,LON")
    p_run.add_argument("--personality", default="balanced",
                       help="Persona preset (homebuyer, runner, photographer, …)")
    p_run.add_argument("--framework-id", default=DEFAULT_FRAMEWORK_ID)
    p_run.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging verbosity",
    )
    p_run.add_argument("--output-dir", default="outputs/run", type=str)
    p_run.add_argument("--interval", type=int, default=20, help="Waypoint interval in metres")
    p_run.add_argument("--phash-threshold", type=int, default=30)
    p_run.add_argument("--system1-only", action="store_true",
                       help="Skip ThinkingModule (faster)")
    p_run.add_argument("--visualize-only", action="store_true",
                       help="Regenerate visualizations from existing results")

    # ── walk ─────────────────────────────────────────────────────────────
    p_walk = sub.add_parser("walk", help="Autonomous walk from start to destination")
    p_walk.add_argument("--start-lat",   type=float, required=True)
    p_walk.add_argument("--start-lng",   type=float, required=True)
    p_walk.add_argument("--dest-lat",    type=float, required=True)
    p_walk.add_argument("--dest-lng",    type=float, required=True)
    p_walk.add_argument("--personality", default="homebuyer",
                        help="Persona preset (homebuyer, runner, parent_with_kids, …)")
    p_walk.add_argument("--max-steps",   type=int, default=60)
    p_walk.add_argument("--output-dir",  default="outputs/walk")
    p_walk.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Save Street View images locally to output_dir/images/",
    )
    p_walk.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
    )

    # ── collect ──────────────────────────────────────────────────────────
    p_col = sub.add_parser("collect", help="Collect Street View images along a route")
    p_col.add_argument("--start",  type=parse_coord, required=True, metavar="LAT,LON")
    p_col.add_argument("--end",    type=parse_coord, required=True, metavar="LAT,LON")
    p_col.add_argument("--output", required=True, help="Output directory for images")
    p_col.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging verbosity",
    )
    p_col.add_argument("--interval", type=int, default=20, help="Waypoint interval in metres")
    p_col.add_argument("--route-name", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_global_log_level(args.log_level)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "walk":
        cmd_walk(args)


if __name__ == "__main__":
    main()
