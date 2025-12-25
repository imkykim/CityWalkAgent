"""Simple test script for K-NN predictor.

This demonstrates the basic functionality with synthetic data.
"""

import numpy as np
import pandas as pd

from src.validation.knn_predictor import KNNPredictor


def create_synthetic_data(n_samples: int = 1000, feature_dim: int = 512):
    """Create synthetic anchor data for testing.

    Args:
        n_samples: Number of synthetic samples
        feature_dim: Feature dimension

    Returns:
        Tuple of (features, scores_df)
    """
    print(f"Creating synthetic data: {n_samples} samples, {feature_dim} dims")

    # Generate random features and normalize them
    features = np.random.randn(n_samples, feature_dim).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Generate correlated scores (simulate realistic walkability patterns)
    # Create some underlying "quality" factors
    quality_factors = np.random.randn(n_samples, 3)

    # Each dimension is influenced by different quality factors
    scores_df = pd.DataFrame(
        {
            "safe": 5.0 + 2.0 * quality_factors[:, 0] + np.random.randn(n_samples) * 0.5,
            "lively": 5.0 + 2.0 * quality_factors[:, 1] + np.random.randn(n_samples) * 0.5,
            "beautiful": 5.0 + 1.5 * (quality_factors[:, 0] + quality_factors[:, 2]) + np.random.randn(n_samples) * 0.5,
            "wealthy": 5.0 + 2.0 * quality_factors[:, 2] + np.random.randn(n_samples) * 0.5,
        }
    )

    # Clip to 0-10 range
    scores_df = scores_df.clip(0, 10)

    return features, scores_df


def test_predictor():
    """Test the K-NN predictor."""
    print("=" * 70)
    print("Testing K-NN Predictor")
    print("=" * 70)
    print()

    # Create synthetic anchor data
    n_anchors = 1000
    feature_dim = 512
    anchor_features, anchor_scores = create_synthetic_data(n_anchors, feature_dim)

    print(f"Anchor features shape: {anchor_features.shape}")
    print(f"Anchor scores shape: {anchor_scores.shape}")
    print()

    # Test 1: Initialize predictor with different configurations
    print("=" * 70)
    print("Test 1: Predictor Initialization")
    print("=" * 70)

    for weighting in ["uniform", "similarity", "softmax"]:
        print(f"\nInitializing predictor with weighting='{weighting}'...")
        predictor = KNNPredictor(
            anchor_features=anchor_features,
            anchor_scores=anchor_scores,
            k=10,
            use_faiss=True,
            weighting=weighting,
        )
        print()

    # Use softmax weighting for remaining tests
    predictor = KNNPredictor(
        anchor_features=anchor_features,
        anchor_scores=anchor_scores,
        k=10,
        use_faiss=True,
        weighting="softmax",
    )

    # Test 2: Single prediction
    print("=" * 70)
    print("Test 2: Single Prediction")
    print("=" * 70)

    # Use an anchor image as query (should retrieve similar scores)
    query_idx = 42
    query_feature = anchor_features[query_idx]
    actual_scores = anchor_scores.iloc[query_idx]

    print(f"Query: Anchor image #{query_idx}")
    print("\nActual scores:")
    for dim in ["safe", "lively", "beautiful", "wealthy"]:
        print(f"  {dim}: {actual_scores[dim]:.2f}")

    prediction = predictor.predict_single(query_feature)
    print("\nPredicted scores:")
    for dim, score in prediction.items():
        print(f"  {dim}: {score:.2f}")

    print("\nPrediction errors:")
    for dim in ["safe", "lively", "beautiful", "wealthy"]:
        error = abs(prediction[dim] - actual_scores[dim])
        print(f"  {dim}: {error:.2f}")
    print()

    # Test 3: Get neighbors
    print("=" * 70)
    print("Test 3: Neighbor Retrieval")
    print("=" * 70)

    indices, similarities = predictor.get_neighbors(query_feature)
    print(f"Retrieved {len(indices)} neighbors")
    print("\nTop 5 neighbors:")
    for i in range(min(5, len(indices))):
        idx = indices[i]
        sim = similarities[i]
        print(f"  {i+1}. Anchor #{idx}, similarity: {sim:.4f}")

    # Check if query retrieved itself
    if query_idx in indices:
        rank = list(indices).index(query_idx)
        print(f"\n✓ Query image found at rank {rank} (should be 0)")
    print()

    # Test 4: Batch prediction
    print("=" * 70)
    print("Test 4: Batch Prediction")
    print("=" * 70)

    # Create query batch
    n_queries = 50
    query_features, query_scores = create_synthetic_data(n_queries, feature_dim)

    print(f"Predicting for {n_queries} query images...")
    predictions_df = predictor.predict_batch(query_features, show_progress=True)

    print(f"\nPredictions shape: {predictions_df.shape}")
    print("\nPrediction statistics:")
    print(predictions_df.describe())
    print()

    # Test 5: Prediction confidence
    print("=" * 70)
    print("Test 5: Prediction Confidence")
    print("=" * 70)

    confidence = predictor.get_prediction_confidence(query_feature)
    print("Confidence metrics:")
    print(f"  Mean similarity: {confidence['mean_similarity']:.4f}")
    print(f"  Min similarity: {confidence['min_similarity']:.4f}")
    print(f"  Max similarity: {confidence['max_similarity']:.4f}")
    print(f"  Std similarity: {confidence['std_similarity']:.4f}")

    print("\nScore agreement (std dev):")
    for dim in ["safe", "lively", "beautiful", "wealthy"]:
        std = confidence[f"{dim}_score_std"]
        print(f"  {dim}: {std:.2f}")
    print()

    # Test 6: Retrieval quality analysis
    print("=" * 70)
    print("Test 6: Retrieval Quality Analysis")
    print("=" * 70)

    print("Analyzing retrieval quality (self-retrieval test)...")
    quality_metrics = predictor.analyze_retrieval_quality(sample_size=100)

    print("\nRetrieval quality metrics:")
    print(f"  Mean self-retrieval rank: {quality_metrics['self_retrieval_mean_rank']:.2f}")
    print(f"  Top-1 retrieval rate: {quality_metrics['self_retrieval_top1_rate']:.2%}")
    print(f"  Mean absolute error: {quality_metrics['mean_absolute_error']:.2f}")
    print()

    # Test 7: Compare weighting methods
    print("=" * 70)
    print("Test 7: Weighting Method Comparison")
    print("=" * 70)

    weighting_methods = ["uniform", "similarity", "softmax"]
    test_query = anchor_features[0]
    actual = anchor_scores.iloc[0]

    print("Comparing predictions for same query with different weightings:\n")
    print(f"{'Dimension':<12} {'Actual':<8} {'Uniform':<8} {'Similarity':<11} {'Softmax':<8}")
    print("-" * 55)

    predictions_by_method = {}
    for method in weighting_methods:
        pred = KNNPredictor(
            anchor_features, anchor_scores, k=10, weighting=method
        )
        predictions_by_method[method] = pred.predict_single(test_query)

    for dim in ["safe", "lively", "beautiful", "wealthy"]:
        row = f"{dim:<12} {actual[dim]:<8.2f}"
        for method in weighting_methods:
            row += f" {predictions_by_method[method][dim]:<8.2f}"
        print(row)

    print()
    print("✓ K-NN predictor test completed successfully!")


if __name__ == "__main__":
    test_predictor()
