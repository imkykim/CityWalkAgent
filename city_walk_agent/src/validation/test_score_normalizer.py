"""Simple test script for Score Normalizer.

This demonstrates the basic functionality with synthetic data that simulates
typical VLM vs K-NN score distribution differences.
"""

import numpy as np

from src.validation.score_normalizer import ScoreNormalizer


def create_synthetic_scores(n_samples: int = 100):
    """Create synthetic VLM and K-NN scores with different distributions.

    VLM scores typically:
    - Cluster around 6-8 (optimistic bias)
    - Have lower variance
    - Rarely use extreme values

    K-NN/Place Pulse scores:
    - More uniform distribution
    - Higher variance
    - Better range utilization

    Args:
        n_samples: Number of samples

    Returns:
        Tuple of (vlm_scores, knn_scores)
    """
    np.random.seed(42)

    # VLM scores: clustered around 7, std ~0.9
    vlm_scores = np.random.normal(7.0, 0.9, n_samples)
    vlm_scores = np.clip(vlm_scores, 4.5, 9.5)  # Rarely use extremes

    # K-NN scores: more spread out, std ~1.7
    knn_scores = np.random.normal(5.5, 1.7, n_samples)
    knn_scores = np.clip(knn_scores, 1.0, 9.5)

    # Add some correlation so they're measuring similar things
    correlation = 0.6
    shared_component = np.random.normal(0, 1, n_samples)

    vlm_scores = vlm_scores + correlation * shared_component * 0.5
    knn_scores = knn_scores + correlation * shared_component * 1.0

    # Clip again
    vlm_scores = np.clip(vlm_scores, 0, 10)
    knn_scores = np.clip(knn_scores, 0, 10)

    return vlm_scores, knn_scores


def test_normalizer():
    """Test the score normalizer."""
    print("=" * 70)
    print("Testing Score Normalizer")
    print("=" * 70)
    print()

    # Create synthetic data
    print("Creating synthetic VLM and K-NN scores...")
    vlm_scores, knn_scores = create_synthetic_scores(n_samples=100)

    print(f"VLM scores shape: {vlm_scores.shape}")
    print(f"K-NN scores shape: {knn_scores.shape}")
    print()

    # Test 1: Fit normalizer
    print("=" * 70)
    print("Test 1: Fitting Normalizer")
    print("=" * 70)

    normalizer = ScoreNormalizer()
    normalizer.fit(vlm_scores, knn_scores)

    print("✓ Normalizer fitted successfully")
    print()

    # Test 2: Get distribution report
    print("=" * 70)
    print("Test 2: Distribution Report")
    print("=" * 70)

    report = normalizer.get_distribution_report()
    print(report)
    print()

    # Test 3: Z-score normalization
    print("=" * 70)
    print("Test 3: Z-Score Normalization")
    print("=" * 70)

    normalized_vlm = normalizer.zscore_normalize(vlm_scores)

    print("Before normalization:")
    print(f"  VLM mean: {vlm_scores.mean():.2f}, std: {vlm_scores.std():.2f}")
    print(f"  K-NN mean: {knn_scores.mean():.2f}, std: {knn_scores.std():.2f}")

    print("\nAfter normalization:")
    print(f"  Normalized VLM mean: {normalized_vlm.mean():.2f}, std: {normalized_vlm.std():.2f}")
    print(f"  K-NN mean: {knn_scores.mean():.2f}, std: {knn_scores.std():.2f}")

    # Check if normalization worked
    mean_diff = abs(normalized_vlm.mean() - knn_scores.mean())
    std_diff = abs(normalized_vlm.std() - knn_scores.std())

    print(f"\nMean difference: {mean_diff:.4f} (should be ~0)")
    print(f"Std difference: {std_diff:.4f} (should be ~0)")

    if mean_diff < 0.1 and std_diff < 0.1:
        print("✓ Normalization successful!")
    else:
        print("⚠ Normalization may have issues")
    print()

    # Test 4: Rank transformation
    print("=" * 70)
    print("Test 4: Rank Transformation")
    print("=" * 70)

    test_scores = np.array([5.2, 7.8, 6.1, 8.9, 4.3])
    ranks = normalizer.rank_transform(test_scores)

    print("Test scores:", test_scores)
    print("Ranks:", ranks)
    print(f"Rank range: {ranks.min():.0f} to {ranks.max():.0f}")
    print()

    # Test 5: Min-max normalization
    print("=" * 70)
    print("Test 5: Min-Max Normalization")
    print("=" * 70)

    minmax_normalized = normalizer.minmax_normalize(vlm_scores, target_min=0, target_max=10)

    print("Min-max normalized:")
    print(f"  Min: {minmax_normalized.min():.2f} (target: 0.0)")
    print(f"  Max: {minmax_normalized.max():.2f} (target: 10.0)")
    print(f"  Mean: {minmax_normalized.mean():.2f}")
    print()

    # Test 6: Multi-dimensional normalization
    print("=" * 70)
    print("Test 6: Multi-Dimensional Normalization")
    print("=" * 70)

    # Create multi-dimensional scores (e.g., 4 dimensions)
    n_samples = 50
    n_dims = 4

    vlm_multi = np.random.normal(7.0, 0.9, (n_samples, n_dims))
    vlm_multi = np.clip(vlm_multi, 4.5, 9.5)

    knn_multi = np.random.normal(5.5, 1.7, (n_samples, n_dims))
    knn_multi = np.clip(knn_multi, 1.0, 9.5)

    print(f"Multi-dimensional scores shape: {vlm_multi.shape}")

    normalizer_multi = ScoreNormalizer()
    normalizer_multi.fit(vlm_multi, knn_multi)

    normalized_multi = normalizer_multi.zscore_normalize(vlm_multi)

    print("\nBefore normalization:")
    print(f"  VLM mean: {vlm_multi.mean():.2f}, std: {vlm_multi.std():.2f}")
    print(f"  K-NN mean: {knn_multi.mean():.2f}, std: {knn_multi.std():.2f}")

    print("\nAfter normalization:")
    print(f"  Normalized VLM mean: {normalized_multi.mean():.2f}, std: {normalized_multi.std():.2f}")
    print(f"  K-NN mean: {knn_multi.mean():.2f}, std: {knn_multi.std():.2f}")
    print()

    # Test 7: Plot distributions (if matplotlib available)
    print("=" * 70)
    print("Test 7: Distribution Visualization")
    print("=" * 70)

    try:
        import matplotlib

        # Set backend to avoid display issues
        matplotlib.use("Agg")

        from pathlib import Path

        output_dir = Path("data/test_cache")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "distribution_comparison.png"

        print(f"Generating distribution plot: {output_path}")
        normalizer.plot_distributions(vlm_scores, knn_scores, str(output_path))

    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization test")
    except Exception as e:
        print(f"⚠ Error generating plot: {e}")

    print()

    # Test 8: Get distribution stats
    print("=" * 70)
    print("Test 8: Distribution Statistics")
    print("=" * 70)

    stats = normalizer.get_distribution_stats()

    print("VLM Statistics:")
    for key, value in stats["vlm"].items():
        print(f"  {key}: {value:.3f}")

    print("\nK-NN Statistics:")
    for key, value in stats["knn"].items():
        print(f"  {key}: {value:.3f}")

    print()
    print("✓ Score normalizer test completed successfully!")


if __name__ == "__main__":
    test_normalizer()
