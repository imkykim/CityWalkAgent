"""Simple test script for Validation Analyzer.

This demonstrates the full validation pipeline with synthetic data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.validation_analyzer import ValidationAnalyzer


def create_correlated_scores(n_samples: int = 60):
    """Create synthetic VLM and K-NN scores with correlation.

    Simulates realistic scenario where VLM and K-NN scores are
    correlated but have different distributions.

    Args:
        n_samples: Number of samples

    Returns:
        Tuple of (vlm_scores_df, knn_scores_df)
    """
    np.random.seed(42)

    dimensions = ["safe", "lively", "beautiful", "wealthy"]

    # Create underlying "ground truth" quality factors
    # These represent actual walkability characteristics
    quality_factors = {
        "infrastructure": np.random.normal(5, 2, n_samples),
        "social_vitality": np.random.normal(5, 2, n_samples),
        "aesthetics": np.random.normal(5, 2, n_samples),
        "affluence": np.random.normal(5, 2, n_samples),
    }

    # K-NN scores: closer to ground truth, more variance
    knn_scores = {
        "safe": quality_factors["infrastructure"]
        + 0.3 * quality_factors["affluence"]
        + np.random.normal(0, 0.5, n_samples),
        "lively": quality_factors["social_vitality"]
        + np.random.normal(0, 0.6, n_samples),
        "beautiful": 0.5 * quality_factors["aesthetics"]
        + 0.3 * quality_factors["infrastructure"]
        + np.random.normal(0, 0.5, n_samples),
        "wealthy": quality_factors["affluence"]
        + 0.2 * quality_factors["infrastructure"]
        + np.random.normal(0, 0.4, n_samples),
    }

    # VLM scores: correlated with ground truth but with bias
    # - Compressed range (6-8 typical)
    # - Optimistic bias
    # - Lower variance
    vlm_scores = {
        "safe": 6.5
        + 0.4 * (quality_factors["infrastructure"] - 5)
        + 0.2 * (quality_factors["affluence"] - 5)
        + np.random.normal(0, 0.3, n_samples),
        "lively": 6.5
        + 0.4 * (quality_factors["social_vitality"] - 5)
        + np.random.normal(0, 0.4, n_samples),
        "beautiful": 7.0
        + 0.3 * (quality_factors["aesthetics"] - 5)
        + 0.2 * (quality_factors["infrastructure"] - 5)
        + np.random.normal(0, 0.3, n_samples),
        "wealthy": 6.8
        + 0.4 * (quality_factors["affluence"] - 5)
        + 0.1 * (quality_factors["infrastructure"] - 5)
        + np.random.normal(0, 0.3, n_samples),
    }

    # Clip to valid range
    for dim in dimensions:
        knn_scores[dim] = np.clip(knn_scores[dim], 0, 10)
        vlm_scores[dim] = np.clip(vlm_scores[dim], 4, 9)  # VLM rarely uses extremes

    # Convert to DataFrames
    vlm_df = pd.DataFrame(vlm_scores)
    knn_df = pd.DataFrame(knn_scores)

    return vlm_df, knn_df


def test_analyzer():
    """Test the validation analyzer."""
    print("=" * 70)
    print("Testing Validation Analyzer")
    print("=" * 70)
    print()

    # Create synthetic data
    print("Creating synthetic VLM and K-NN scores...")
    n_samples = 60
    vlm_scores, knn_predictions = create_correlated_scores(n_samples)

    image_ids = [f"img_{i:03d}" for i in range(n_samples)]

    print(f"VLM scores shape: {vlm_scores.shape}")
    print(f"K-NN predictions shape: {knn_predictions.shape}")
    print()

    # Test 1: Initialize analyzer
    print("=" * 70)
    print("Test 1: Analyzer Initialization")
    print("=" * 70)

    analyzer = ValidationAnalyzer(
        vlm_scores=vlm_scores,
        knn_predictions=knn_predictions,
        image_ids=image_ids,
    )

    print(f"✓ Analyzer initialized with {analyzer.n_samples} samples")
    print()

    # Test 2: Analyze single dimension
    print("=" * 70)
    print("Test 2: Single Dimension Analysis")
    print("=" * 70)

    result = analyzer.analyze_dimension("safe")

    print("Safety dimension analysis:")
    print(f"  N samples: {result['n_samples']}")
    print(f"  Spearman ρ: {result['spearman_rho']:.3f} (p={result['spearman_p']:.4f})")
    print(f"  Significant: {result['spearman_significant']}")
    print(f"  Pearson r: {result['pearson_r']:.3f}")
    print(f"  MAE: {result['mae']:.2f}")
    print(f"  RMSE: {result['rmse']:.2f}")
    print(f"  R²: {result['r2']:.3f}")
    print(f"  VLM mean: {result['vlm_mean']:.2f}, std: {result['vlm_std']:.2f}")
    print(f"  K-NN mean: {result['knn_mean']:.2f}, std: {result['knn_std']:.2f}")
    print()

    # Test 3: Analyze all dimensions
    print("=" * 70)
    print("Test 3: All Dimensions Analysis")
    print("=" * 70)

    results_df = analyzer.analyze_all_dimensions()

    print("Summary DataFrame:")
    print(results_df[["dimension", "spearman_rho", "pearson_r", "mae", "r2"]].to_string(index=False))
    print()

    # Test 4: Generate report
    print("=" * 70)
    print("Test 4: Validation Report")
    print("=" * 70)

    report = analyzer.generate_report()
    print(report)
    print()

    # Test 5: Plot correlations
    print("=" * 70)
    print("Test 5: Correlation Plots")
    print("=" * 70)

    try:
        import matplotlib

        # Set backend to avoid display issues
        matplotlib.use("Agg")

        output_dir = Path("data/test_cache/validation_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")
        print()

        # Plot single dimension
        print("Plotting single dimension (safety)...")
        analyzer.plot_correlation("safe", output_dir / "safety_correlation.png")
        print()

        # Plot all dimensions
        print("Plotting all dimensions...")
        analyzer.plot_all_dimensions(output_dir)
        print()

    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization tests")
    except Exception as e:
        print(f"⚠ Error generating plots: {e}")

    # Test 6: Export results
    print("=" * 70)
    print("Test 6: Export Results")
    print("=" * 70)

    export_dir = Path("data/test_cache/validation_export")
    print(f"Exporting results to: {export_dir}")

    analyzer.export_results(export_dir)
    print()

    # Verify exported files
    expected_files = [
        "validation_summary.csv",
        "validation_report.txt",
        "score_comparison.csv",
    ]

    print("Checking exported files:")
    for filename in expected_files:
        filepath = export_dir / filename
        exists = "✓" if filepath.exists() else "✗"
        print(f"  {exists} {filename}")

    print()

    # Test 7: Correlation interpretation
    print("=" * 70)
    print("Test 7: Correlation Strength Interpretation")
    print("=" * 70)

    print("\nSpearman correlation interpretation:")
    for _, row in results_df.iterrows():
        rho = row["spearman_rho"]
        dim = row["dimension"]

        if rho >= 0.7:
            strength = "STRONG"
        elif rho >= 0.5:
            strength = "MODERATE-STRONG"
        elif rho >= 0.3:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        sig = "✓" if row["spearman_significant"] else "✗"
        print(f"  {dim.capitalize():<12}: ρ={rho:.3f} ({strength}) {sig}")

    print()

    # Test 8: Distribution warnings
    print("=" * 70)
    print("Test 8: Distribution Analysis")
    print("=" * 70)

    print("\nRange utilization:")
    for _, row in results_df.iterrows():
        vlm_util = row["vlm_range_utilization"] * 100
        knn_util = row["knn_range_utilization"] * 100
        dim = row["dimension"]

        print(f"  {dim.capitalize():<12}: VLM {vlm_util:.0f}%, K-NN {knn_util:.0f}%")

        if vlm_util < 50:
            print(f"    ⚠ VLM shows limited range utilization")

    print()

    # Test 9: Per-dimension normalizers
    print("=" * 70)
    print("Test 9: Per-Dimension Normalization")
    print("=" * 70)

    print("\nTesting normalization for each dimension:")
    for dim in ["safe", "lively", "beautiful", "wealthy"]:
        vlm_values = vlm_scores[dim].values
        normalizer = analyzer.normalizers[dim]

        normalized = normalizer.zscore_normalize(vlm_values)

        print(f"\n{dim.capitalize()}:")
        print(f"  Original VLM: mean={vlm_values.mean():.2f}, std={vlm_values.std():.2f}")
        print(f"  Normalized:   mean={normalized.mean():.2f}, std={normalized.std():.2f}")
        print(f"  Target K-NN:  mean={knn_predictions[dim].mean():.2f}, std={knn_predictions[dim].std():.2f}")

    print()
    print("✓ Validation analyzer test completed successfully!")


if __name__ == "__main__":
    test_analyzer()
