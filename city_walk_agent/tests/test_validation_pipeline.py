"""Integration tests for CLIP + K-NN validation pipeline.

This test suite verifies that all components of the validation pipeline
work correctly together, using synthetic data to avoid dependencies on
the full Place Pulse 2.0 dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from validation import (
    CLIPExtractor,
    KNNPredictor,
    PlacePulseLoader,
    ScoreNormalizer,
    zscore_normalize_knn_to_vlm,
    ValidationAnalyzer,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def synthetic_anchor_data():
    """Generate synthetic Place Pulse-like anchor data.

    Returns:
        Tuple of (features, scores_df)
    """
    n_samples = 1000
    feature_dim = 512

    # Generate random normalized features
    features = np.random.randn(n_samples, feature_dim).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Generate correlated scores
    quality_factors = np.random.randn(n_samples, 3)

    scores_df = pd.DataFrame(
        {
            "image_id": [f"anchor_{i:06d}" for i in range(n_samples)],
            "safe": np.clip(
                5.0 + 2.0 * quality_factors[:, 0] + np.random.randn(n_samples) * 0.5,
                0,
                10,
            ),
            "lively": np.clip(
                5.0 + 2.0 * quality_factors[:, 1] + np.random.randn(n_samples) * 0.5,
                0,
                10,
            ),
            "beautiful": np.clip(
                5.0
                + 1.5 * (quality_factors[:, 0] + quality_factors[:, 2])
                + np.random.randn(n_samples) * 0.5,
                0,
                10,
            ),
            "wealthy": np.clip(
                5.0 + 2.0 * quality_factors[:, 2] + np.random.randn(n_samples) * 0.5,
                0,
                10,
            ),
        }
    )

    return features, scores_df


@pytest.fixture
def synthetic_query_data():
    """Generate synthetic query data (VLM-evaluated images).

    Returns:
        Tuple of (features, vlm_scores_df, knn_scores_df)
    """
    n_samples = 60
    feature_dim = 512

    # Generate query features
    features = np.random.randn(n_samples, feature_dim).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Generate ground truth
    quality_factors = {
        "infrastructure": np.random.normal(5, 2, n_samples),
        "social_vitality": np.random.normal(5, 2, n_samples),
        "aesthetics": np.random.normal(5, 2, n_samples),
        "affluence": np.random.normal(5, 2, n_samples),
    }

    # K-NN scores (simulating human perception)
    knn_scores_df = pd.DataFrame(
        {
            "safe": np.clip(
                quality_factors["infrastructure"]
                + 0.3 * quality_factors["affluence"]
                + np.random.normal(0, 0.5, n_samples),
                0,
                10,
            ),
            "lively": np.clip(
                quality_factors["social_vitality"]
                + np.random.normal(0, 0.6, n_samples),
                0,
                10,
            ),
            "beautiful": np.clip(
                0.5 * quality_factors["aesthetics"]
                + 0.3 * quality_factors["infrastructure"]
                + np.random.normal(0, 0.5, n_samples),
                0,
                10,
            ),
            "wealthy": np.clip(
                quality_factors["affluence"]
                + 0.2 * quality_factors["infrastructure"]
                + np.random.normal(0, 0.4, n_samples),
                0,
                10,
            ),
        }
    )

    # VLM scores (correlated but compressed range)
    vlm_scores_df = pd.DataFrame(
        {
            "safe": np.clip(
                6.5
                + 0.4 * (quality_factors["infrastructure"] - 5)
                + 0.2 * (quality_factors["affluence"] - 5)
                + np.random.normal(0, 0.3, n_samples),
                4,
                9,
            ),
            "lively": np.clip(
                6.5
                + 0.4 * (quality_factors["social_vitality"] - 5)
                + np.random.normal(0, 0.4, n_samples),
                4,
                9,
            ),
            "beautiful": np.clip(
                7.0
                + 0.3 * (quality_factors["aesthetics"] - 5)
                + 0.2 * (quality_factors["infrastructure"] - 5)
                + np.random.normal(0, 0.3, n_samples),
                4,
                9,
            ),
            "wealthy": np.clip(
                6.8
                + 0.4 * (quality_factors["affluence"] - 5)
                + 0.1 * (quality_factors["infrastructure"] - 5)
                + np.random.normal(0, 0.3, n_samples),
                4,
                9,
            ),
        }
    )

    return features, vlm_scores_df, knn_scores_df


# ============================================================================
# COMPONENT TESTS
# ============================================================================


class TestPlacePulseLoader:
    """Tests for Place Pulse data loader."""

    def test_loader_initialization(self, tmp_path):
        """Test loader can be initialized."""
        loader = PlacePulseLoader(tmp_path)
        assert loader.data_dir == tmp_path
        # When no scores file exists, falls back to qscores.tsv path
        assert loader.scores_file in (tmp_path / "scores.csv", tmp_path / "qscores.tsv")

    def test_loader_with_missing_data(self, tmp_path):
        """Test loader handles missing data gracefully."""
        loader = PlacePulseLoader(tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_scores()

    def test_loader_with_valid_data(self, tmp_path):
        """Test loader works with valid Place Pulse 2.0 format data."""
        import pandas as pd
        # Create mock scores file in Place Pulse 2.0 long format
        scores_df = pd.DataFrame(
            {
                "location_id": ["loc1", "loc1", "loc1", "loc1", "loc2", "loc2", "loc2", "loc2"],
                "study_id": [
                    "50a68a51fdc9f05596000002",  # safe
                    "50f62ccfa84ea7c5fdd2e459",  # depressing (will be filtered out)
                    "50f62c41a84ea7c5fdd2e454",  # lively
                    "5217c351ad93a7d3e7b07a64",  # beautiful
                    "50a68a51fdc9f05596000002",  # safe
                    "50f62c41a84ea7c5fdd2e454",  # lively
                    "5217c351ad93a7d3e7b07a64",  # beautiful
                    "50f62cb7a84ea7c5fdd2e458",  # wealthy
                ],
                "trueskill.score": [4.5, 5.0, 6.0, 7.0, 3.5, 5.5, 6.5, 4.0],
            }
        )

        scores_file = tmp_path / "scores.csv"
        scores_df.to_csv(scores_file, index=False)

        # Load and verify (images dir not set up so some rows may be dropped)
        loader = PlacePulseLoader(tmp_path)
        loaded_scores = loader.load_scores()

        assert isinstance(loaded_scores, pd.DataFrame)
        # Check normalized score columns present
        for dim in ["safe", "lively", "beautiful"]:
            assert dim in loaded_scores.columns


class TestCLIPExtractor:
    """Tests for CLIP feature extractor."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent / "data" / "images",
        reason="No test images available",
    )
    def test_extractor_initialization(self):
        """Test CLIP extractor initializes correctly."""
        extractor = CLIPExtractor(
            model_name="ViT-B/32", device="cpu", batch_size=8
        )

        assert extractor.model_name == "ViT-B/32"
        assert extractor.device == "cpu"
        assert extractor.batch_size == 8
        assert extractor.feature_dim == 512

    def test_feature_verification(self):
        """Test feature verification works."""
        extractor = CLIPExtractor(device="cpu")

        # Create synthetic features
        features = np.random.randn(10, 512).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        verification = extractor.verify_features(features)

        assert verification["shape"] == (10, 512)
        assert verification["feature_dim_matches"] is True
        assert verification["normalized"] is True
        assert not verification["has_nan"]
        assert not verification["has_inf"]


class TestKNNPredictor:
    """Tests for K-NN predictor."""

    def test_predictor_initialization(self, synthetic_anchor_data):
        """Test K-NN predictor initializes correctly."""
        features, scores_df = synthetic_anchor_data

        predictor = KNNPredictor(
            anchor_features=features,
            anchor_scores=scores_df,
            k=10,
            use_faiss=False,  # Use numpy for testing
        )

        assert predictor.k == 10
        assert len(predictor.anchor_features) == 1000
        assert len(predictor.anchor_scores) == 1000

    def test_predictor_single_prediction(self, synthetic_anchor_data):
        """Test single image prediction."""
        features, scores_df = synthetic_anchor_data

        predictor = KNNPredictor(
            anchor_features=features, anchor_scores=scores_df, k=10, use_faiss=False
        )

        # Query with first anchor (should get similar scores)
        query_feature = features[0]
        prediction = predictor.predict_single(query_feature)

        # Check prediction structure
        assert set(prediction.keys()) == {"safe", "lively", "beautiful", "wealthy"}

        # All scores should be in valid range
        for score in prediction.values():
            assert 0 <= score <= 10

    def test_predictor_batch_prediction(self, synthetic_anchor_data):
        """Test batch prediction."""
        features, scores_df = synthetic_anchor_data

        predictor = KNNPredictor(
            anchor_features=features, anchor_scores=scores_df, k=10, use_faiss=False
        )

        # Query with multiple features
        query_features = features[:50]
        predictions_df = predictor.predict_batch(query_features, show_progress=False)

        assert len(predictions_df) == 50
        assert set(predictions_df.columns) == {"safe", "lively", "beautiful", "wealthy"}

        # All scores should be in valid range
        for dim in ["safe", "lively", "beautiful", "wealthy"]:
            assert predictions_df[dim].between(0, 10).all()

    def test_predictor_self_retrieval(self, synthetic_anchor_data):
        """Test that querying with anchor retrieves itself."""
        features, scores_df = synthetic_anchor_data

        predictor = KNNPredictor(
            anchor_features=features, anchor_scores=scores_df, k=5, use_faiss=False
        )

        # Query with anchor
        query_idx = 42
        query_feature = features[query_idx]
        indices, similarities = predictor.get_neighbors(query_feature)

        # Should retrieve itself as top match
        assert indices[0] == query_idx
        assert similarities[0] > 0.99  # Near perfect similarity


class TestScoreNormalizer:
    """Tests for score normalizer."""

    def test_normalizer_fit(self):
        """Test normalizer fitting."""
        vlm_scores = np.random.normal(7.0, 0.9, 100)
        knn_scores = np.random.normal(5.5, 1.7, 100)

        normalizer = ScoreNormalizer()
        normalizer.fit(vlm_scores, knn_scores)

        assert normalizer.vlm_stats is not None
        assert normalizer.knn_stats is not None
        assert abs(normalizer.vlm_stats["mean"] - 7.0) < 0.3
        assert abs(normalizer.knn_stats["mean"] - 5.5) < 0.3

    def test_normalizer_zscore_knn_to_vlm(self):
        """Test z-score normalization direction K-NN → VLM."""
        vlm_scores = np.random.normal(7.0, 0.9, 100)
        knn_scores = np.random.normal(5.5, 1.7, 100)

        normalized = zscore_normalize_knn_to_vlm(vlm_scores, knn_scores)

        # Normalized K-NN should align with VLM stats
        assert abs(normalized.mean() - vlm_scores.mean()) < 0.1
        assert abs(normalized.std() - vlm_scores.std()) < 0.1

    def test_normalizer_rank_transform(self):
        """Test rank transformation."""
        normalizer = ScoreNormalizer()
        scores = np.array([5.2, 7.8, 6.1, 8.9, 4.3])

        ranks = normalizer.rank_transform(scores)

        # Check ranks are correct
        assert len(ranks) == 5
        assert ranks.min() == 1
        assert ranks.max() == 5


class TestValidationAnalyzer:
    """Tests for validation analyzer."""

    def test_analyzer_initialization(self, synthetic_query_data):
        """Test analyzer initializes correctly."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(vlm_scores=vlm_scores, knn_predictions=knn_scores)

        assert analyzer.n_samples == 60
        assert len(analyzer.vlm_scores) == 60
        assert len(analyzer.knn_predictions) == 60

    def test_analyzer_dimension_analysis(self, synthetic_query_data):
        """Test single dimension analysis."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(vlm_scores=vlm_scores, knn_predictions=knn_scores)

        result = analyzer.analyze_dimension("safe")

        # Check all required metrics are present
        assert "dimension" in result
        assert "n_samples" in result
        assert "spearman_rho" in result
        assert "spearman_p" in result
        assert "pearson_r" in result
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

        # Check metric ranges
        assert -1 <= result["spearman_rho"] <= 1
        assert 0 <= result["spearman_p"] <= 1
        assert -1 <= result["pearson_r"] <= 1
        assert result["mae"] >= 0
        assert result["rmse"] >= 0

    def test_analyzer_quantile_normalization(self, synthetic_query_data):
        """Ensure quantile normalization is applied and stats are aligned."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(
            vlm_scores=vlm_scores,
            knn_predictions=knn_scores,
            normalization_method="quantile",
        )

        result = analyzer.analyze_dimension("safe")

        assert result["normalization"] == "quantile_matching"
        assert "match_stats" in result
        assert result["normalized_source"] == "vlm"
        assert abs(result["normalized_mean"] - result["knn_mean"]) < 1.0

    def test_analyzer_zscore_direction(self, synthetic_query_data):
        """Ensure z-score normalization now scales K-NN to VLM."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(
            vlm_scores=vlm_scores,
            knn_predictions=knn_scores,
            normalization_method="zscore",
        )

        result = analyzer.analyze_dimension("safe")

        assert result["normalization"] == "zscore_knn_to_vlm"
        assert result["normalized_source"] == "knn"
        assert abs(result["normalized_mean"] - result["vlm_mean"]) < 1.0

    def test_analyzer_all_dimensions(self, synthetic_query_data):
        """Test analysis of all dimensions."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(vlm_scores=vlm_scores, knn_predictions=knn_scores)

        results_df = analyzer.analyze_all_dimensions()

        assert len(results_df) == 4
        assert set(results_df["dimension"]) == {"safe", "lively", "beautiful", "wealthy"}

        # All correlations should be positive (due to synthetic data design)
        assert (results_df["spearman_rho"] > 0).all()

    def test_analyzer_report_generation(self, synthetic_query_data):
        """Test report generation."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(vlm_scores=vlm_scores, knn_predictions=knn_scores)

        report = analyzer.generate_report()

        # Check report contains key sections
        assert "CLIP + K-NN Validation Report" in report
        assert "Spearman Correlation" in report
        assert "INTERPRETATION" in report

        # Check all dimensions are mentioned
        for dim in ["safe", "lively", "beautiful", "wealthy"]:
            assert dim.capitalize() in report


# ============================================================================
# END-TO-END TESTS
# ============================================================================


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_synthetic(self, synthetic_anchor_data, synthetic_query_data):
        """Test complete validation pipeline with synthetic data."""
        # Get synthetic data
        anchor_features, anchor_scores = synthetic_anchor_data
        query_features, vlm_scores, true_knn_scores = synthetic_query_data

        # Step 1: Initialize K-NN predictor
        predictor = KNNPredictor(
            anchor_features=anchor_features,
            anchor_scores=anchor_scores,
            k=10,
            use_faiss=False,
        )

        # Step 2: Predict K-NN scores
        knn_predictions = predictor.predict_batch(query_features, show_progress=False)

        assert len(knn_predictions) == len(vlm_scores)

        # Step 3: Run validation analysis
        analyzer = ValidationAnalyzer(
            vlm_scores=vlm_scores, knn_predictions=knn_predictions
        )

        results_df = analyzer.analyze_all_dimensions()

        # Verify results
        assert len(results_df) == 4
        assert "spearman_rho" in results_df.columns  # Correlation computed

        # Generate report
        report = analyzer.generate_report()
        assert len(report) > 0

    def test_pipeline_with_export(self, synthetic_query_data, tmp_path):
        """Test pipeline with result export."""
        _, vlm_scores, knn_scores = synthetic_query_data

        analyzer = ValidationAnalyzer(vlm_scores=vlm_scores, knn_predictions=knn_scores)

        # Export results
        output_dir = tmp_path / "validation_output"
        analyzer.export_results(output_dir)

        # Verify exported files
        assert (output_dir / "validation_summary.csv").exists()
        assert (output_dir / "validation_report.txt").exists()
        assert (output_dir / "score_comparison.csv").exists()

        # Verify file contents
        summary_df = pd.read_csv(output_dir / "validation_summary.csv")
        assert len(summary_df) == 4

        with open(output_dir / "validation_report.txt") as f:
            report = f.read()
            assert "Spearman Correlation" in report


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance and scaling tests."""

    def test_knn_scaling(self, synthetic_anchor_data):
        """Test K-NN performance with different K values."""
        features, scores_df = synthetic_anchor_data

        k_values = [5, 10, 20, 50]
        query_features = features[:10]

        for k in k_values:
            predictor = KNNPredictor(
                anchor_features=features,
                anchor_scores=scores_df,
                k=k,
                use_faiss=False,
            )

            predictions = predictor.predict_batch(query_features, show_progress=False)
            assert len(predictions) == 10

    def test_batch_processing(self, synthetic_anchor_data):
        """Test batch prediction with different batch sizes."""
        features, scores_df = synthetic_anchor_data

        predictor = KNNPredictor(
            anchor_features=features, anchor_scores=scores_df, k=10, use_faiss=False
        )

        # Test with different query sizes
        for n_queries in [1, 10, 50, 100]:
            query_features = features[:n_queries]
            predictions = predictor.predict_batch(query_features, show_progress=False)
            assert len(predictions) == n_queries


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
