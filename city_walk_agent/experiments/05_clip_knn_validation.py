#!/usr/bin/env python3
"""
Experiment 05: CLIP + K-NN Validation

Validates VLM walkability scores against Place Pulse 2.0 human perception data
using CLIP embeddings and K-Nearest Neighbors.

Pipeline:
1. Load VLM evaluation results from previous experiments
2. Load Place Pulse 2.0 anchor set (110K images with crowdsourced scores)
3. Extract CLIP features for both anchor set and our images
4. Use K-NN to predict human perception scores based on visual similarity
5. Compare VLM scores with K-NN predictions using statistical analysis

Usage:
    python experiments/05_clip_knn_validation.py --input results/analysis_results.json
    python experiments/05_clip_knn_validation.py --input results/analysis_results.json --k 10 --use-cache
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root and src to path so both `src.*` and top-level imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Check dependencies
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not available, progress bars disabled")

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("❌ Missing required packages for CLIP")
    print("   Install with: pip install torch transformers Pillow")

try:
    from scipy.stats import spearmanr

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("❌ Missing scipy")
    print("   Install with: pip install scipy")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available, plots will be skipped")

from config.settings import settings
from validation import (
    CLIPExtractor,
    KNNPredictor,
    PlacePulseLoader,
    ValidationAnalyzer,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_K = 10
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 32


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CLIP + K-NN Validation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python experiments/05_clip_knn_validation.py --input results/analysis_results.json

  # With custom K and caching
  python experiments/05_clip_knn_validation.py --input results/analysis_results.json --k 15 --use-cache

  # Use different CLIP model
  python experiments/05_clip_knn_validation.py --input results/analysis_results.json --model openai/clip-vit-large-patch14

  # Skip plots (faster)
  python experiments/05_clip_knn_validation.py --input results/analysis_results.json --skip-plots
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to VLM analysis results JSON file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/validation/TIMESTAMP/)",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of neighbors for K-NN (default: {DEFAULT_K})",
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached CLIP features if available",
    )

    parser.add_argument(
        "--score-type",
        type=str,
        choices=["objective", "persona"],
        default="objective",
        help="Which VLM scores to use (default: objective)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CLIP_MODEL,
        help=f"CLIP model to use (default: {DEFAULT_CLIP_MODEL})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for CLIP extraction (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip generating visualization plots"
    )

    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of CLIP features even if cache exists",
    )

    parser.add_argument(
        "--anchor-subset",
        type=int,
        default=None,
        help="Use subset of anchor set for testing (default: use all)",
    )

    return parser.parse_args()


# ============================================================================
# VLM RESULTS LOADER
# ============================================================================


def load_vlm_results(input_path: Path) -> List[Dict]:
    """Load VLM evaluation results.

    Args:
        input_path: Path to analysis results JSON

    Returns:
        List of waypoint evaluation results

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is invalid
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"📂 Loading VLM results from: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    # Handle different possible formats
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        # Try to find results in different keys
        if "waypoints" in data:
            results = data["waypoints"]
        elif "results" in data:
            results = data["results"]
        elif "evaluations" in data:
            results = data["evaluations"]
        else:
            raise ValueError(
                f"Unrecognized format: expected list or dict with 'waypoints'/'results'/'evaluations'"
            )
    else:
        raise ValueError(f"Invalid data format: {type(data)}")

    print(f"✓ Loaded {len(results)} waypoint evaluations")

    return results


def extract_vlm_scores(
    vlm_results: List[Dict], score_type: str = "objective"
) -> pd.DataFrame:
    """Extract VLM scores into DataFrame format.

    Args:
        vlm_results: List of VLM evaluation results
        score_type: Type of scores to extract ('objective' or 'persona')

    Returns:
        DataFrame with columns: safe, lively, beautiful, wealthy
    """
    print(f"📊 Extracting {score_type} scores from VLM results...")

    scores = []
    valid_count = 0

    for result in vlm_results:
        # Try different possible structures
        evaluation = result.get("evaluation", result.get("scores", {})) or {}

        if score_type == "objective":
            score_data = (
                evaluation.get("objective_scores") or evaluation.get("scores") or {}
            )
            # Fallback to top-level fields from recent pipelines
            if not score_data:
                score_data = result.get("objective_scores")
        else:
            score_data = evaluation.get("persona_scores") or {}
            if not score_data:
                score_data = (
                    result.get("persona_scores") or result.get("system2_scores") or {}
                )

        # Map dimension names (handle different naming conventions)
        dimension_mapping = {
            "safe": ["safe", "safety", "安全感"],
            "lively": ["lively", "vitality", "活力感"],
            "beautiful": ["beautiful", "beauty", "aesthetic", "美观性"],
            "wealthy": ["wealthy", "wealth", "affluence", "富裕感"],
        }

        row = {}
        for target_dim, possible_names in dimension_mapping.items():
            for name in possible_names:
                if name in score_data:
                    row[target_dim] = score_data[name]
                    break

        if len(row) == 4:  # All dimensions found
            scores.append(row)
            valid_count += 1

    if valid_count == 0:
        raise ValueError(
            f"No valid {score_type} scores found in VLM results. "
            f"Please check the input format."
        )

    scores_df = pd.DataFrame(scores)

    print(f"✓ Extracted {len(scores_df)} complete score sets")
    print(f"  Dimensions: {list(scores_df.columns)}")
    print(f"  Score ranges: {scores_df.min().to_dict()} to {scores_df.max().to_dict()}")

    return scores_df


def get_image_paths(vlm_results: List[Dict]) -> List[Path]:
    """Extract image paths from VLM results.

    Args:
        vlm_results: List of VLM evaluation results

    Returns:
        List of image paths
    """
    image_paths = []

    for result in vlm_results:
        # Try different possible keys
        image_path = (
            result.get("image_path") or result.get("image") or result.get("path")
        )

        if image_path:
            image_paths.append(Path(image_path))

    print(f"✓ Extracted {len(image_paths)} image paths")

    # Verify images exist
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"⚠️  Warning: {len(missing)} images not found")
        if len(missing) <= 5:
            for p in missing:
                print(f"   - {p}")

    return image_paths


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Run the CLIP + K-NN validation pipeline."""
    args = parse_args()

    print("=" * 70)
    print("         CLIP + K-NN Validation Experiment")
    print("=" * 70)
    print()

    # Check dependencies
    if not HAS_CLIP:
        print("❌ Cannot proceed without CLIP dependencies")
        sys.exit(1)

    if not HAS_SCIPY:
        print("❌ Cannot proceed without scipy")
        sys.exit(1)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results" / "validation" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()

    # Save experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(args.input),
        "k": args.k,
        "clip_model": args.model,
        "batch_size": args.batch_size,
        "score_type": args.score_type,
        "use_cache": args.use_cache,
        "force_recompute": args.force_recompute,
        "anchor_subset": args.anchor_subset,
    }

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Step 1: Load VLM evaluation results
    print("=" * 70)
    print("Step 1: Load VLM Evaluation Results")
    print("=" * 70)
    print()

    vlm_results = load_vlm_results(args.input)
    vlm_scores = extract_vlm_scores(vlm_results, score_type=args.score_type)
    image_paths = get_image_paths(vlm_results)

    if len(vlm_scores) != len(image_paths):
        print(
            f"⚠️  Warning: Score count ({len(vlm_scores)}) != "
            f"Image count ({len(image_paths)})"
        )
        # Take minimum
        min_len = min(len(vlm_scores), len(image_paths))
        vlm_scores = vlm_scores.iloc[:min_len]
        image_paths = image_paths[:min_len]
        print(f"   Using first {min_len} samples")

    print()

    # Step 2: Load Place Pulse anchor set
    print("=" * 70)
    print("Step 2: Load Place Pulse 2.0 Anchor Set")
    print("=" * 70)
    print()

    try:
        pp_loader = PlacePulseLoader(settings.place_pulse_dir)
        anchor_scores = pp_loader.load_scores()
        print()

        # Use subset if requested
        if args.anchor_subset and args.anchor_subset < len(anchor_scores):
            print(f"📊 Using subset: {args.anchor_subset} anchors (for testing)")
            anchor_scores = anchor_scores.sample(n=args.anchor_subset, random_state=42)
            print()

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print()
        print("To use this validation pipeline, you need Place Pulse 2.0 dataset:")
        print("  1. Download from: http://pulse.media.mit.edu/data/")
        print(f"  2. Extract to: {settings.place_pulse_dir}")
        print("  3. Ensure scores.csv and images/ directory are present")
        sys.exit(1)

    # Step 3: Extract CLIP features for anchor set
    print("=" * 70)
    print("Step 3: Extract CLIP Features for Anchor Set")
    print("=" * 70)
    print()

    clip_extractor = CLIPExtractor(
        model_name=args.model,
        device="auto",
        batch_size=args.batch_size,
    )
    print()

    cache_path = (
        settings.place_pulse_dir / f"clip_features_{args.model.replace('/', '_')}.npy"
    )

    anchor_image_paths = [
        pp_loader.get_image_path(img_id) for img_id in anchor_scores["image_id"]
    ]

    anchor_features = clip_extractor.extract_and_cache(
        anchor_image_paths,
        cache_path=cache_path,
        force_recompute=args.force_recompute,
    )

    print()

    # Step 4: Extract CLIP features for our images
    print("=" * 70)
    print("Step 4: Extract CLIP Features for Our Images")
    print("=" * 70)
    print()

    print(f"Extracting features for {len(image_paths)} images...")
    our_features = clip_extractor.extract_batch(image_paths, show_progress=True)

    # Verify features
    verification = clip_extractor.verify_features(our_features)
    print(f"\nFeature verification:")
    print(f"  Shape: {verification['shape']}")
    print(f"  Normalized: {verification['normalized']}")
    print(f"  Mean L2 norm: {verification['mean_l2_norm']:.4f}")

    print()

    # Step 5: Run K-NN prediction
    print("=" * 70)
    print("Step 5: K-NN Score Prediction")
    print("=" * 70)
    print()

    predictor = KNNPredictor(
        anchor_features=anchor_features,
        anchor_scores=anchor_scores,
        k=args.k,
        use_faiss=True,
        weighting="softmax",
    )

    print()
    print(f"Predicting scores for {len(our_features)} images...")
    knn_predictions = predictor.predict_batch(our_features, show_progress=True)

    print()
    print("K-NN Prediction Statistics:")
    print(knn_predictions.describe())

    print()

    # Step 6: Run validation analysis
    print("=" * 70)
    print("Step 6: Statistical Validation Analysis")
    print("=" * 70)
    print()

    image_ids = [f"img_{i:03d}" for i in range(len(vlm_scores))]

    analyzer = ValidationAnalyzer(
        vlm_scores=vlm_scores,
        knn_predictions=knn_predictions,
        image_ids=image_ids,
    )

    # Analyze all dimensions
    results_df = analyzer.analyze_all_dimensions()

    print("Validation Results Summary:")
    print(
        results_df[
            ["dimension", "spearman_rho", "spearman_p", "pearson_r", "r2"]
        ].to_string(index=False)
    )

    print()

    # Step 7: Generate report
    print("=" * 70)
    print("Step 7: Generate Validation Report")
    print("=" * 70)
    print()

    report = analyzer.generate_report()
    print(report)

    print()

    # Step 8: Generate plots
    if not args.skip_plots and HAS_MATPLOTLIB:
        print("=" * 70)
        print("Step 8: Generate Visualization Plots")
        print("=" * 70)
        print()

        plots_dir = output_dir / "correlation_plots"
        plots_dir.mkdir(exist_ok=True)

        analyzer.plot_all_dimensions(plots_dir)

        # Also create distribution comparison plots
        from validation.score_normalizer import ScoreNormalizer

        for dim in ["safe", "lively", "beautiful", "wealthy"]:
            normalizer = analyzer.normalizers[dim]
            vlm_vals = vlm_scores[dim].values
            knn_vals = knn_predictions[dim].values

            dist_plot_path = plots_dir / f"{dim}_distribution.png"
            normalizer.plot_distributions(vlm_vals, knn_vals, str(dist_plot_path))

        print()

    elif args.skip_plots:
        print("⏭️  Skipping plots (--skip-plots)")
        print()

    # Step 9: Save results
    print("=" * 70)
    print("Step 9: Save Results")
    print("=" * 70)
    print()

    # Export validation results
    analyzer.export_results(output_dir)

    # Save results DataFrame
    results_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    results_df.to_json(
        output_dir / "validation_metrics.json", orient="records", indent=2
    )

    # Save summary statistics
    summary = {
        "n_samples": len(vlm_scores),
        "n_anchors": len(anchor_scores),
        "k": args.k,
        "clip_model": args.model,
        "score_type": args.score_type,
        "overall_spearman_rho": float(results_df["spearman_rho"].mean()),
        "significant_dimensions": int(results_df["spearman_significant"].sum()),
        "dimensions": results_df.to_dict(orient="records"),
    }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()

    # Final summary
    print("=" * 70)
    print("                  VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print(f"✅ Results saved to: {output_dir}")
    print()
    print("Key Findings:")
    print(f"  • Average Spearman ρ: {summary['overall_spearman_rho']:.3f}")
    print(f"  • Significant dimensions: {summary['significant_dimensions']}/4")
    print()

    best_dim = results_df.loc[results_df["spearman_rho"].idxmax()]
    print(
        f"  • Best: {best_dim['dimension'].capitalize()} "
        f"(ρ={best_dim['spearman_rho']:.3f})"
    )

    worst_dim = results_df.loc[results_df["spearman_rho"].idxmin()]
    print(
        f"  • Worst: {worst_dim['dimension'].capitalize()} "
        f"(ρ={worst_dim['spearman_rho']:.3f})"
    )

    print()
    print("Generated Files:")
    print(f"  • {output_dir / 'validation_report.txt'}")
    print(f"  • {output_dir / 'validation_summary.json'}")
    print(f"  • {output_dir / 'validation_metrics.csv'}")
    print(f"  • {output_dir / 'score_comparison.csv'}")
    if not args.skip_plots and HAS_MATPLOTLIB:
        print(f"  • {output_dir / 'correlation_plots' / 'all_dimensions.png'}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
