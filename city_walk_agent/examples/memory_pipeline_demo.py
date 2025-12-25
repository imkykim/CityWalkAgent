#!/usr/bin/env python3
"""
Memory Pipeline Demo: Full System Integration

Demonstrates the complete memory system pipeline:
1. ContinuousAnalyzer → per-waypoint VLM analysis with pHash detection
2. ShortTermMemory → sliding window context tracking
3. ThinkingModule → triggered LLM-based reasoning
4. LongTermMemory → key moment curation + pattern extraction

This pipeline showcases how the agent builds understanding through:
- Real-time waypoint analysis
- Temporal context management
- Selective deep reasoning
- Long-term memory formation

Usage:
    python examples/memory_pipeline_demo.py \\
        --route-name "downtown_walk" \\
        --images-dir data/routes/downtown_walk/images \\
        --output-dir outputs/memory_demo \\
        --framework sagai_2025

Prerequisites:
- Waypoint images in images-dir
- Optional: metadata.json with waypoint details
- VLM API configured (for ContinuousAnalyzer)
- LLM API configured (for ThinkingModule, optional)
"""

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import ContinuousAnalyzer, WaypointAnalysis
from src.agent.capabilities import (
    ShortTermMemory,
    ThinkingModule,
    TriggerReason,
    LongTermMemory,
)
from src.config import DEFAULT_FRAMEWORK_ID
from src.utils.exporters import export_evaluations_csv
from src.utils.visualization import plot_analysis_results
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print a formatted section header."""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def calculate_distance(gps1: Tuple[float, float], gps2: Tuple[float, float]) -> float:
    """Calculate distance between two GPS coordinates using Haversine formula.

    Args:
        gps1: (latitude, longitude) tuple for first point
        gps2: (latitude, longitude) tuple for second point

    Returns:
        Distance in meters
    """
    lat1, lon1 = gps1
    lat2, lon2 = gps2

    # Radius of Earth in meters
    R = 6371000

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def load_waypoint_metadata(images_dir: Path) -> List[Dict[str, Any]]:
    """Load waypoint metadata from JSON file or generate dummy data.

    Expected metadata.json format:
    [
        {
            "filename": "waypoint_0.jpg",
            "lat": 37.7749,
            "lon": -122.4194,
            "heading": 90.0,
            "timestamp": "2024-01-01T12:00:00"
        },
        ...
    ]

    Args:
        images_dir: Directory containing waypoint images

    Returns:
        List of metadata dictionaries
    """
    metadata_file = images_dir / "metadata.json"

    if metadata_file.exists():
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        logger.warning(f"No metadata.json found in {images_dir}, generating dummy data")

        # Get image files
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(
            images_dir.glob("*.png")
        )

        # Generate dummy metadata
        metadata = []
        base_lat = 37.7749
        base_lon = -122.4194
        base_time = datetime.now()

        for i, img_file in enumerate(image_files):
            # Simulate a walking route with small GPS changes
            lat = base_lat + (i * 0.0001)  # ~11 meters per waypoint
            lon = base_lon + (i * 0.0001)
            heading = (i * 15) % 360  # Varying heading

            metadata.append(
                {
                    "filename": img_file.name,
                    "lat": lat,
                    "lon": lon,
                    "heading": heading,
                    "timestamp": base_time.isoformat(),
                }
            )

        return metadata


def calculate_route_length(metadata: List[Dict[str, Any]]) -> float:
    """Calculate total route length in kilometers.

    Args:
        metadata: List of waypoint metadata with lat/lon

    Returns:
        Total distance in kilometers
    """
    if len(metadata) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(1, len(metadata)):
        prev_gps = (metadata[i - 1]["lat"], metadata[i - 1]["lon"])
        curr_gps = (metadata[i]["lat"], metadata[i]["lon"])
        total_distance += calculate_distance(prev_gps, curr_gps)

    return total_distance / 1000.0  # Convert to km


# ============================================================================
# Main Pipeline
# ============================================================================


def run_memory_pipeline(
    route_name: str, images_dir: Path, output_dir: Path, framework_id: str
):
    """Run the complete memory pipeline demonstration.

    Args:
        route_name: Name of the route for identification
        images_dir: Directory containing waypoint images
        output_dir: Directory for output files
        framework_id: Evaluation framework to use
    """
    print_section_header("🚶 Memory Pipeline Demo: Full System Integration")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    key_images_dir = output_dir / "key_images"
    key_images_dir.mkdir(exist_ok=True)

    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print_section_header("📁 Step 1: Loading Waypoint Data")

    # Load metadata
    metadata = load_waypoint_metadata(images_dir)
    print(f"✓ Loaded metadata for {len(metadata)} waypoints")

    # Get image paths
    image_paths = [images_dir / m["filename"] for m in metadata]
    existing_images = [p for p in image_paths if p.exists()]

    print(f"✓ Found {len(existing_images)} image files")

    if len(existing_images) == 0:
        print("❌ No images found! Please check the images directory.")
        return

    # Calculate route statistics
    route_length_km = calculate_route_length(metadata)
    print(f"✓ Route length: {route_length_km:.2f} km")
    print(f"✓ Framework: {framework_id}")

    # ========================================================================
    # Step 2: Initialize Components
    # ========================================================================
    print_section_header("⚙️  Step 2: Initializing Pipeline Components")

    try:
        # Initialize ContinuousAnalyzer
        print("Initializing ContinuousAnalyzer...")
        analyzer = ContinuousAnalyzer(
            framework_id=framework_id,
            context_window=3,
            phash_threshold=28.0,
            adaptive_threshold=False,
        )
        print("✓ ContinuousAnalyzer ready")

        # Initialize ShortTermMemory
        print("Initializing ShortTermMemory...")
        stm = ShortTermMemory(window_size=5)
        print("✓ ShortTermMemory ready (window_size=5)")

        # Initialize ThinkingModule (optional, may not have LLM configured)
        print("Initializing ThinkingModule...")
        try:
            thinking = ThinkingModule(
                distance_trigger_meters=600.0,
                score_delta_threshold=1.5,
                enable_vlm_deep_dive=False,
            )
            print("✓ ThinkingModule ready (LLM-based reasoning enabled)")
            use_thinking = True
        except Exception as e:
            logger.warning(f"ThinkingModule initialization failed: {e}")
            print("⚠ ThinkingModule disabled (LLM not configured)")
            use_thinking = False

        # Initialize LongTermMemory
        print("Initializing LongTermMemory...")
        ltm = LongTermMemory(agent_id=f"memory_demo_{route_name}")
        print("✓ LongTermMemory ready")

    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        print(f"❌ Error: {e}")
        return

    # ========================================================================
    # Step 3: Continuous Analysis Phase
    # ========================================================================
    print_section_header("🔍 Step 3: Continuous Analysis (VLM per waypoint)")

    print(
        f"Analyzing {len(existing_images)} waypoints with {framework_id} framework..."
    )
    print("This may take several minutes...\n")

    try:
        analysis_results = analyzer.analyze_route(
            image_paths=existing_images[: len(metadata)], waypoint_metadata=metadata
        )

        # Get statistics
        stats = analyzer.get_statistics()

        print(f"✓ Analysis complete!")
        print(f"  Total waypoints: {stats['total_waypoints']}")
        print(f"  Visual changes detected: {stats['visual_changes_detected']}")
        print(f"  Avg pHash distance: {stats['avg_phash_distance']:.2f}")

        # Save analysis results
        analysis_file = output_dir / "analysis_results.json"
        with open(analysis_file, "w") as f:
            json.dump(
                [
                    {
                        "waypoint_id": a.waypoint_id,
                        "scores": a.scores,
                        "reasoning": a.reasoning,
                        "visual_change": a.visual_change_detected,
                        "phash_distance": a.phash_distance,
                        "gps": a.gps,
                        "timestamp": a.timestamp,
                    }
                    for a in analysis_results
                ],
                f,
                indent=2,
            )

        print(f"✓ Saved analysis results to {analysis_file}")
        analysis_csv = output_dir / "analysis_results.csv"
        export_evaluations_csv(
            (
                {
                    "image_id": analysis.waypoint_id,
                    "dimension_id": dim_id,
                    "dimension_name": dim_id,
                    "score": score,
                    "reasoning": str(analysis.reasoning.get(dim_id, "")),
                }
                for analysis in analysis_results
                for dim_id, score in analysis.scores.items()
            ),
            analysis_csv,
        )
        print(f"✓ Saved analysis CSV to {analysis_csv}")
        try:
            metrics_plot = output_dir / "analysis_metrics.png"
            plot_analysis_results(analysis_results, metrics_plot)
            print(f"✓ Saved analysis metrics plot to {metrics_plot}")
        except Exception as e:
            logger.warning(f"Analysis visualization failed: {e}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis error: {e}")
        return

    # ========================================================================
    # Step 4: Thinking Phase (with STM context)
    # ========================================================================
    print_section_header("🧠 Step 4: Triggered Thinking (STM + LLM reasoning)")

    thinking_results = []
    last_trigger_gps = metadata[0]["lat"], metadata[0]["lon"]
    last_avg_score = 0.0

    for i, analysis in enumerate(analysis_results):
        # Calculate average score
        avg_score = (
            sum(analysis.scores.values()) / len(analysis.scores)
            if analysis.scores
            else 0.0
        )

        # Add to short-term memory
        stm.add(
            waypoint_id=analysis.waypoint_id,
            scores=analysis.scores,
            summary=f"Waypoint with avg score {avg_score:.1f}",
            image_path=analysis.image_path if analysis.visual_change_detected else None,
            gps=analysis.gps,
            timestamp=analysis.timestamp,
        )

        # Check if thinking should trigger
        if use_thinking and i > 0:
            curr_gps = analysis.gps
            distance_from_last = calculate_distance(last_trigger_gps, curr_gps)
            score_delta = abs(avg_score - last_avg_score)

            trigger = thinking.should_trigger(
                waypoint_id=analysis.waypoint_id,
                visual_change=analysis.visual_change_detected,
                score_delta=score_delta,
                distance_from_last=distance_from_last,
                is_exceptional=False,
            )

            if trigger:
                print(
                    f"  Trigger at waypoint {analysis.waypoint_id}: "
                    f"{trigger.value} (score_delta={score_delta:.1f})"
                )

                # Get STM context
                stm_context = stm.get_context()

                # Perform thinking
                try:
                    result = thinking.think_waypoint(
                        waypoint_id=analysis.waypoint_id,
                        trigger_reason=trigger,
                        stm_context=stm_context,
                        current_scores=analysis.scores,
                        current_reasoning=analysis.reasoning,
                        current_image_path=analysis.image_path,
                        route_metadata={
                            "route_id": route_name,
                            "length_km": route_length_km,
                        },
                    )

                    thinking_results.append(result)
                    last_trigger_gps = curr_gps

                    print(
                        f"    → Significance: {result.significance}, "
                        f"Confidence: {result.confidence:.2f}"
                    )

                except Exception as e:
                    logger.warning(f"Thinking failed at waypoint {i}: {e}")

        last_avg_score = avg_score

    if use_thinking:
        print(f"\n✓ Thinking completed: {len(thinking_results)} episodes")

        # Get thinking summary
        thinking_summary = thinking.get_thinking_summary()
        print(f"  Trigger breakdown: {thinking_summary.get('trigger_breakdown', {})}")
        print(f"  Avg confidence: {thinking_summary.get('avg_confidence', 0):.2f}")

        # Save thinking results
        thinking_file = output_dir / "thinking_results.json"
        with open(thinking_file, "w") as f:
            json.dump(
                [
                    {
                        "waypoint_id": t.waypoint_id,
                        "trigger_reason": t.trigger_reason.value,
                        "interpretation": t.interpretation,
                        "significance": t.significance,
                        "pattern_detected": t.pattern_detected,
                        "prediction": t.prediction,
                        "recommendation": t.recommendation,
                        "confidence": t.confidence,
                    }
                    for t in thinking_results
                ],
                f,
                indent=2,
            )

        print(f"✓ Saved thinking results to {thinking_file}")
    else:
        print("⚠ Thinking phase skipped (LLM not configured)")

    # ========================================================================
    # Step 5: Long-Term Memory Formation
    # ========================================================================
    print_section_header("💾 Step 5: Long-Term Memory (curation + patterns)")

    print("Adding candidate moments to long-term memory...")

    # Add moments from thinking results (if available)
    if thinking_results:
        for result in thinking_results:
            # Get corresponding analysis
            analysis = next(
                (a for a in analysis_results if a.waypoint_id == result.waypoint_id),
                None,
            )
            if analysis:
                avg_score = sum(analysis.scores.values()) / len(analysis.scores)

                ltm.add_candidate_moment(
                    waypoint_id=analysis.waypoint_id,
                    image_path=analysis.image_path,
                    scores=analysis.scores,
                    summary=result.interpretation[:200],  # First 200 chars
                    significance=result.significance,
                    gps=analysis.gps,
                    timestamp=analysis.timestamp,
                    thinking_confidence=result.confidence,
                    visual_change_detected=analysis.visual_change_detected,
                    score_delta=None,
                )

        print(f"✓ Added {len(thinking_results)} candidate moments")

    else:
        # Fallback: Add high-scoring waypoints as moments
        print("No thinking results available, using score-based selection...")
        for analysis in analysis_results:
            avg_score = sum(analysis.scores.values()) / len(analysis.scores)

            # Add high-scoring or visually-changed waypoints
            if avg_score >= 7.5 or analysis.visual_change_detected:
                significance = "high" if avg_score >= 8.5 else "medium"

                ltm.add_candidate_moment(
                    waypoint_id=analysis.waypoint_id,
                    image_path=analysis.image_path,
                    scores=analysis.scores,
                    summary=f"High quality waypoint (score: {avg_score:.1f})",
                    significance=significance,
                    gps=analysis.gps,
                    timestamp=analysis.timestamp,
                    thinking_confidence=0.8,
                    visual_change_detected=analysis.visual_change_detected,
                    score_delta=None,
                )

        print(f"✓ Added moments based on scores")

    # Curate moments
    print("\nCurating key moments...")
    ltm.curate_moments(route_length_km=route_length_km)
    print("✓ Moments curated")

    # Extract patterns
    print("Extracting route patterns...")
    analysis_dicts = [
        {"waypoint_id": a.waypoint_id, "scores": a.scores, "summary": ""}
        for a in analysis_results
    ]
    thinking_history = (
        [t.interpretation for t in thinking_results] if thinking_results else []
    )

    ltm.extract_patterns(all_analyses=analysis_dicts, thinking_history=thinking_history)
    print("✓ Patterns extracted")

    # Generate route summary
    print("Generating route summary...")
    summary = ltm.generate_route_summary(
        route_id=route_name,
        total_waypoints=len(analysis_results),
        length_km=route_length_km,
        all_analyses=analysis_dicts,
    )
    print("✓ Route summary generated")

    # Save long-term memory
    ltm_data = {
        "route_summary": {
            "route_id": summary.route_id,
            "total_waypoints": summary.total_waypoints,
            "length_km": summary.length_km,
            "avg_scores": summary.avg_scores,
            "score_volatility": summary.score_volatility,
            "dominant_characteristics": summary.dominant_characteristics,
            "major_barriers": summary.major_barriers,
            "highlights": summary.highlights,
            "patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "start_waypoint": p.start_waypoint,
                    "end_waypoint": p.end_waypoint,
                    "description": p.description,
                    "affected_dimensions": p.affected_dimensions,
                }
                for p in summary.patterns
            ],
            "overall_recommendation": summary.overall_recommendation,
            "recommendation_reason": summary.recommendation_reason,
            "target_users": summary.target_users,
        }
    }

    ltm_file = output_dir / "long_term_memory.json"
    with open(ltm_file, "w") as f:
        json.dump(ltm_data, f, indent=2)

    print(f"✓ Saved long-term memory to {ltm_file}")

    # ========================================================================
    # Step 6: Export Key Images
    # ========================================================================
    print_section_header("📸 Step 6: Exporting Key Images")

    key_image_paths = ltm.get_key_images()
    print(f"Copying {len(key_image_paths)} key moment images...")

    for img_path in key_image_paths:
        if img_path.exists():
            dest = key_images_dir / img_path.name
            shutil.copy2(img_path, dest)

    print(f"✓ Key images saved to {key_images_dir}")

    # ========================================================================
    # Step 7: Generate Narrative
    # ========================================================================
    print_section_header("📝 Step 7: Generating Route Narrative")

    narrative = ltm.get_narrative()
    narrative_file = output_dir / "narrative.md"

    with open(narrative_file, "w") as f:
        f.write(f"# Route Narrative: {route_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Route Length:** {route_length_km:.2f} km\n")
        f.write(f"**Total Waypoints:** {len(analysis_results)}\n")
        f.write(f"**Framework:** {framework_id}\n\n")
        f.write("---\n\n")
        f.write(narrative)
        f.write("\n\n---\n\n")
        f.write("## Route Summary\n\n")
        f.write(f"**Recommendation:** {summary.overall_recommendation.upper()}\n\n")
        f.write(f"**Reason:** {summary.recommendation_reason}\n\n")
        f.write("**Dominant Characteristics:**\n")
        for char in summary.dominant_characteristics:
            f.write(f"- {char}\n")
        f.write("\n**Target Users:**\n")
        for user in summary.target_users:
            f.write(f"- {user}\n")

    print(f"✓ Narrative saved to {narrative_file}")

    # ========================================================================
    # Final Statistics Summary
    # ========================================================================
    print_section_header("📊 Final Statistics Summary")

    print("ANALYSIS PHASE:")
    print(f"  Waypoints analyzed: {len(analysis_results)}")
    print(f"  Visual changes detected: {stats['visual_changes_detected']}")
    print(f"  Avg pHash distance: {stats['avg_phash_distance']:.2f}")
    print()

    if use_thinking and thinking_results:
        print("THINKING PHASE:")
        print(f"  Thinking episodes: {len(thinking_results)}")
        print(f"  Avg confidence: {thinking_summary.get('avg_confidence', 0):.2%}")
        print(
            f"  High significance: {len(thinking_summary.get('high_significance_waypoints', []))}"
        )
        print()

    print("LONG-TERM MEMORY:")
    print(f"  Key moments preserved: {len(key_image_paths)}")
    print(f"  Patterns detected: {len(summary.patterns)}")
    print(f"  Overall recommendation: {summary.overall_recommendation.upper()}")
    print(f"  Route length: {route_length_km:.2f} km")
    print()

    print("OUTPUTS:")
    print(f"  Analysis results: {output_dir / 'analysis_results.json'}")
    print(f"  Analysis CSV: {output_dir / 'analysis_results.csv'}")
    print(f"  Analysis metrics plot: {output_dir / 'analysis_metrics.png'}")
    if use_thinking:
        print(f"  Thinking results: {output_dir / 'thinking_results.json'}")
    print(f"  Long-term memory: {output_dir / 'long_term_memory.json'}")
    print(f"  Narrative: {output_dir / 'narrative.md'}")
    print(f"  Key images: {key_images_dir}")
    print()

    print_separator()
    print("✅ Memory pipeline demo completed successfully!")
    print_separator()


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Main entry point for the memory pipeline demo."""
    parser = argparse.ArgumentParser(
        description="Memory Pipeline Demo: Full System Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python examples/memory_pipeline_demo.py \\
      --route-name downtown_walk \\
      --images-dir data/routes/downtown_walk/images \\
      --output-dir outputs/memory_demo

  # With custom framework
  python examples/memory_pipeline_demo.py \\
      --route-name scenic_route \\
      --images-dir data/routes/scenic/images \\
      --output-dir outputs/scenic_demo \\
      --framework sagai_2025
        """,
    )

    parser.add_argument(
        "--route-name",
        type=str,
        required=True,
        help="Name of the route (for identification)",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing waypoint images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files (will be created if needed)",
    )

    parser.add_argument(
        "--framework",
        type=str,
        default=DEFAULT_FRAMEWORK_ID,
        help=f"Evaluation framework to use (default: {DEFAULT_FRAMEWORK_ID})",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.images_dir.exists():
        print(f"❌ Error: Images directory does not exist: {args.images_dir}")
        sys.exit(1)

    # Run pipeline
    try:
        run_memory_pipeline(
            route_name=args.route_name,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            framework_id=args.framework,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
