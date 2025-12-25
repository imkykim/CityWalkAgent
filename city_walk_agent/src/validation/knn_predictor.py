"""K-NN based score predictor using CLIP features.

This module implements K-Nearest Neighbors prediction for walkability scores
based on visual similarity using CLIP embeddings.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class KNNPredictor:
    """K-NN predictor for walkability scores based on visual similarity."""

    def __init__(
        self,
        anchor_features: np.ndarray,
        anchor_scores: pd.DataFrame,
        k: int = 10,
        use_faiss: bool = True,
        weighting: str = "softmax",
    ):
        """Initialize predictor with anchor set.

        Args:
            anchor_features: CLIP features for anchor images, shape (N, D)
                           Must be L2 normalized for cosine similarity
            anchor_scores: DataFrame with dimension scores for anchor images
                         Must have columns: safe, lively, beautiful, wealthy
            k: Number of nearest neighbors to use
            use_faiss: Use FAISS for fast search (if available)
            weighting: How to weight neighbors ('uniform', 'similarity', 'softmax')

        Raises:
            ValueError: If features and scores don't match in size
        """
        if len(anchor_features) != len(anchor_scores):
            raise ValueError(
                f"Feature count ({len(anchor_features)}) doesn't match "
                f"score count ({len(anchor_scores)})"
            )

        self.anchor_features = anchor_features.astype(np.float32)
        self.anchor_scores = anchor_scores
        self.k = min(k, len(anchor_features))  # Can't have k > N
        self.weighting = weighting

        # Verify required dimensions exist
        self.dimensions = ["safe", "lively", "beautiful", "wealthy"]
        missing_dims = set(self.dimensions) - set(anchor_scores.columns)
        if missing_dims:
            raise ValueError(f"Missing score dimensions: {missing_dims}")

        # Initialize search index
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        if self.use_faiss:
            self._init_faiss_index()
            print(f"✓ Initialized FAISS index with {len(anchor_features):,} anchors")
        else:
            if use_faiss and not FAISS_AVAILABLE:
                print("Warning: FAISS not available, using numpy (slower)")
            print(f"✓ Initialized numpy-based search with {len(anchor_features):,} anchors")

        print(f"K-NN configuration: k={self.k}, weighting={self.weighting}")

    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search."""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        dimension = self.anchor_features.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Add anchor features to index
        self.index.add(self.anchor_features)

    def get_neighbors(
        self, query_feature: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices and similarities of K nearest neighbors.

        Args:
            query_feature: Query feature vector, shape (D,)

        Returns:
            Tuple of (indices, similarities)
            - indices: Array of neighbor indices, shape (K,)
            - similarities: Array of cosine similarities, shape (K,)
        """
        query_feature = query_feature.astype(np.float32).reshape(1, -1)

        if self.use_faiss:
            # FAISS search returns (similarities, indices)
            similarities, indices = self.index.search(query_feature, self.k)
            return indices[0], similarities[0]
        else:
            # Numpy-based search
            # Compute cosine similarities (dot product with normalized vectors)
            similarities = np.dot(self.anchor_features, query_feature.T).squeeze()

            # Get top-K indices
            indices = np.argsort(similarities)[::-1][: self.k]
            top_similarities = similarities[indices]

            return indices, top_similarities

    def _compute_weights(self, similarities: np.ndarray) -> np.ndarray:
        """Compute weights for neighbors based on similarities.

        Args:
            similarities: Array of cosine similarities, shape (K,)

        Returns:
            Normalized weights, shape (K,)
        """
        if self.weighting == "uniform":
            # Equal weight for all neighbors
            weights = np.ones(len(similarities)) / len(similarities)

        elif self.weighting == "similarity":
            # Weight proportional to similarity
            # Add small epsilon to avoid division by zero
            weights = similarities + 1e-8
            weights = weights / weights.sum()

        elif self.weighting == "softmax":
            # Softmax of similarities (emphasizes most similar)
            # Scale similarities for better softmax behavior
            weights = softmax(similarities * 10)

        else:
            raise ValueError(f"Unknown weighting method: {self.weighting}")

        return weights

    def predict_single(self, query_feature: np.ndarray) -> Dict[str, float]:
        """Predict scores for single query image.

        Args:
            query_feature: Query feature vector, shape (D,)

        Returns:
            Dictionary mapping dimension to predicted score:
            {dimension: predicted_score}
        """
        # Get K nearest neighbors
        indices, similarities = self.get_neighbors(query_feature)

        # Compute weights
        weights = self._compute_weights(similarities)

        # Get neighbor scores
        neighbor_scores = self.anchor_scores.iloc[indices]

        # Compute weighted average for each dimension
        predictions = {}
        for dim in self.dimensions:
            scores = neighbor_scores[dim].values
            predicted_score = np.sum(weights * scores)
            predictions[dim] = float(predicted_score)

        return predictions

    def predict_batch(
        self, query_features: np.ndarray, show_progress: bool = False
    ) -> pd.DataFrame:
        """Predict scores for multiple query images.

        Args:
            query_features: Query feature array, shape (N, D)
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with columns: [safe, lively, beautiful, wealthy]
            Shape: (N, 4)
        """
        predictions = []

        iterator = range(len(query_features))
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="K-NN prediction", unit="image")

        for i in iterator:
            query_feature = query_features[i]
            prediction = self.predict_single(query_feature)
            predictions.append(prediction)

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions, columns=self.dimensions)

        return predictions_df

    def get_prediction_confidence(
        self, query_feature: np.ndarray
    ) -> Dict[str, float]:
        """Compute confidence metrics for a prediction.

        Args:
            query_feature: Query feature vector, shape (D,)

        Returns:
            Dictionary with confidence metrics:
            {
                'mean_similarity': Mean similarity to K neighbors,
                'min_similarity': Minimum similarity to K neighbors,
                'std_similarity': Std dev of similarities,
                'score_agreement': Std dev of neighbor scores (lower = more agreement)
            }
        """
        # Get K nearest neighbors
        indices, similarities = self.get_neighbors(query_feature)

        # Get neighbor scores
        neighbor_scores = self.anchor_scores.iloc[indices]

        # Compute confidence metrics
        confidence = {
            "mean_similarity": float(similarities.mean()),
            "min_similarity": float(similarities.min()),
            "max_similarity": float(similarities.max()),
            "std_similarity": float(similarities.std()),
        }

        # Score agreement per dimension
        for dim in self.dimensions:
            scores = neighbor_scores[dim].values
            confidence[f"{dim}_score_std"] = float(scores.std())
            confidence[f"{dim}_score_range"] = float(scores.max() - scores.min())

        return confidence

    def analyze_retrieval_quality(self, sample_size: int = 100) -> Dict:
        """Analyze retrieval quality using self-retrieval.

        Test by querying with random anchor images and checking if they
        retrieve themselves and similar scores.

        Args:
            sample_size: Number of random anchors to test

        Returns:
            Dictionary with quality metrics
        """
        import random

        sample_indices = random.sample(range(len(self.anchor_features)), sample_size)

        self_retrieval_ranks = []
        score_correlations = []

        for idx in sample_indices:
            query_feature = self.anchor_features[idx]
            actual_scores = self.anchor_scores.iloc[idx]

            # Get neighbors (should include self)
            neighbor_indices, similarities = self.get_neighbors(query_feature)

            # Find rank of self
            if idx in neighbor_indices:
                rank = list(neighbor_indices).index(idx)
                self_retrieval_ranks.append(rank)

            # Predict and compare
            predicted_scores = self.predict_single(query_feature)

            # Compute correlation for each dimension
            for dim in self.dimensions:
                actual = actual_scores[dim]
                predicted = predicted_scores[dim]
                # Simple absolute error
                score_correlations.append(abs(actual - predicted))

        return {
            "self_retrieval_mean_rank": np.mean(self_retrieval_ranks),
            "self_retrieval_top1_rate": sum(r == 0 for r in self_retrieval_ranks)
            / len(self_retrieval_ranks),
            "mean_absolute_error": np.mean(score_correlations),
            "sample_size": sample_size,
        }
