"""Score normalizer for handling distribution differences between VLM and K-NN scores.

This module provides utilities to normalize score distributions, which is critical
because VLM direct scoring and Place Pulse TrueSkill scores have different
statistical properties.
"""

from typing import Dict, Tuple

import numpy as np
from scipy import interpolate, stats


class ScoreNormalizer:
    """Normalizer for handling score distribution differences."""

    def __init__(self):
        """Initialize normalizer."""
        self.vlm_stats = None
        self.knn_stats = None
        self._is_fitted = False

    def fit(self, vlm_scores: np.ndarray, knn_scores: np.ndarray):
        """Compute statistics for both score distributions.

        Args:
            vlm_scores: VLM scores, shape (N,) or (N, D)
            knn_scores: K-NN predicted scores, shape (N,) or (N, D)

        Raises:
            ValueError: If score arrays have different shapes
        """
        vlm_scores = np.asarray(vlm_scores)
        knn_scores = np.asarray(knn_scores)

        if vlm_scores.shape != knn_scores.shape:
            raise ValueError(
                f"VLM and K-NN scores must have same shape. "
                f"Got {vlm_scores.shape} vs {knn_scores.shape}"
            )

        # Compute statistics
        self.vlm_stats = {
            "mean": float(np.mean(vlm_scores)),
            "std": float(np.std(vlm_scores)),
            "min": float(np.min(vlm_scores)),
            "max": float(np.max(vlm_scores)),
            "median": float(np.median(vlm_scores)),
            "q25": float(np.percentile(vlm_scores, 25)),
            "q75": float(np.percentile(vlm_scores, 75)),
        }

        self.knn_stats = {
            "mean": float(np.mean(knn_scores)),
            "std": float(np.std(knn_scores)),
            "min": float(np.min(knn_scores)),
            "max": float(np.max(knn_scores)),
            "median": float(np.median(knn_scores)),
            "q25": float(np.percentile(knn_scores, 25)),
            "q75": float(np.percentile(knn_scores, 75)),
        }

        # Compute skewness
        self.vlm_stats["skewness"] = float(stats.skew(vlm_scores.flatten()))
        self.knn_stats["skewness"] = float(stats.skew(knn_scores.flatten()))

        # Compute score range utilization
        vlm_range = self.vlm_stats["max"] - self.vlm_stats["min"]
        knn_range = self.knn_stats["max"] - self.knn_stats["min"]

        self.vlm_stats["range"] = vlm_range
        self.vlm_stats["range_utilization"] = vlm_range / 10.0  # Assuming 0-10 scale

        self.knn_stats["range"] = knn_range
        self.knn_stats["range_utilization"] = knn_range / 10.0

        self._is_fitted = True

    def rank_transform(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to ranks (1 to N).

        Used for Spearman correlation computation.

        Args:
            scores: Score array, shape (N,)

        Returns:
            Rank array, shape (N,)
        """
        scores = np.asarray(scores)
        return stats.rankdata(scores)

    def minmax_normalize(
        self,
        vlm_scores: np.ndarray,
        target_min: float = 0.0,
        target_max: float = 10.0,
    ) -> np.ndarray:
        """Min-max normalize VLM scores to target range.

        Args:
            vlm_scores: VLM scores to normalize
            target_min: Target minimum value
            target_max: Target maximum value

        Returns:
            Normalized scores in [target_min, target_max]
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before normalizing scores. "
                "Call fit() first."
            )

        vlm_scores = np.asarray(vlm_scores)

        vlm_min = self.vlm_stats["min"]
        vlm_max = self.vlm_stats["max"]

        # Handle edge case
        if vlm_max == vlm_min:
            return np.full_like(vlm_scores, (target_min + target_max) / 2)

        # Min-max normalization
        normalized = (vlm_scores - vlm_min) / (vlm_max - vlm_min)
        normalized = normalized * (target_max - target_min) + target_min

        return normalized

    def get_distribution_report(self) -> str:
        """Generate comparison report of VLM vs K-NN distributions.

        Returns:
            Formatted report string

        Raises:
            RuntimeError: If normalizer hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before generating report. "
                "Call fit() first."
            )

        report = []
        report.append("=" * 70)
        report.append("Distribution Comparison: VLM vs K-NN Predictions")
        report.append("=" * 70)
        report.append("")

        # Statistics table
        report.append(f"{'Statistic':<15} {'VLM':<12} {'K-NN Pred':<12} {'Difference':<12}")
        report.append("-" * 55)

        metrics = ["mean", "std", "min", "max", "median", "q25", "q75"]
        for metric in metrics:
            vlm_val = self.vlm_stats[metric]
            knn_val = self.knn_stats[metric]
            diff = vlm_val - knn_val

            report.append(
                f"{metric.capitalize():<15} {vlm_val:<12.2f} {knn_val:<12.2f} {diff:+12.2f}"
            )

        report.append("")
        report.append(f"{'Skewness':<15} {self.vlm_stats['skewness']:<12.2f} "
                     f"{self.knn_stats['skewness']:<12.2f} "
                     f"{self.vlm_stats['skewness'] - self.knn_stats['skewness']:+12.2f}")

        report.append("")
        report.append("Score Range Utilization:")
        report.append(
            f"  VLM:  [{self.vlm_stats['min']:.1f} - {self.vlm_stats['max']:.1f}] "
            f"({self.vlm_stats['range_utilization']:.1%} of 0-10 scale)"
        )
        report.append(
            f"  K-NN: [{self.knn_stats['min']:.1f} - {self.knn_stats['max']:.1f}] "
            f"({self.knn_stats['range_utilization']:.1%} of 0-10 scale)"
        )

        report.append("")
        report.append("Distribution Characteristics:")

        # Analyze VLM distribution
        if self.vlm_stats["std"] < self.knn_stats["std"]:
            report.append("  • VLM scores show LOWER variance (more clustered)")
        else:
            report.append("  • VLM scores show HIGHER variance (more spread)")

        if abs(self.vlm_stats["skewness"]) > 0.5:
            skew_dir = "negatively" if self.vlm_stats["skewness"] < 0 else "positively"
            report.append(f"  • VLM scores are {skew_dir} skewed")

        if abs(self.knn_stats["skewness"]) > 0.5:
            skew_dir = "negatively" if self.knn_stats["skewness"] < 0 else "positively"
            report.append(f"  • K-NN scores are {skew_dir} skewed")

        if self.vlm_stats["range_utilization"] < 0.5:
            report.append(
                f"  • VLM uses only {self.vlm_stats['range_utilization']:.0%} "
                "of available scale (limited range)"
            )

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    def get_distribution_stats(self) -> Dict:
        """Get distribution statistics as dictionary.

        Returns:
            Dictionary with VLM and K-NN statistics

        Raises:
            RuntimeError: If normalizer hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before getting statistics. "
                "Call fit() first."
            )

        return {
            "vlm": self.vlm_stats.copy(),
            "knn": self.knn_stats.copy(),
        }

    def plot_distributions(
        self, vlm_scores: np.ndarray, knn_scores: np.ndarray, output_path: str
    ):
        """Plot VLM and K-NN score distributions for comparison.

        Args:
            vlm_scores: VLM scores
            knn_scores: K-NN scores
            output_path: Path to save plot
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Histogram comparison
        ax = axes[0, 0]
        ax.hist(vlm_scores.flatten(), bins=30, alpha=0.6, label="VLM", color="blue")
        ax.hist(knn_scores.flatten(), bins=30, alpha=0.6, label="K-NN", color="orange")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Score Distribution Comparison")
        ax.legend()
        ax.grid(alpha=0.3)

        # Box plot comparison
        ax = axes[0, 1]
        ax.boxplot(
            [vlm_scores.flatten(), knn_scores.flatten()],
            labels=["VLM", "K-NN"],
            showmeans=True,
        )
        ax.set_ylabel("Score")
        ax.set_title("Distribution Box Plots")
        ax.grid(alpha=0.3)

        # Q-Q plot
        ax = axes[1, 0]
        sorted_vlm = np.sort(vlm_scores.flatten())
        sorted_knn = np.sort(knn_scores.flatten())
        # Interpolate to same length
        if len(sorted_vlm) != len(sorted_knn):
            min_len = min(len(sorted_vlm), len(sorted_knn))
            sorted_vlm = sorted_vlm[::len(sorted_vlm)//min_len][:min_len]
            sorted_knn = sorted_knn[::len(sorted_knn)//min_len][:min_len]
        ax.scatter(sorted_knn, sorted_vlm, alpha=0.5, s=10)
        # Add diagonal line
        min_val = min(sorted_knn.min(), sorted_vlm.min())
        max_val = max(sorted_knn.max(), sorted_vlm.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel("K-NN Quantiles")
        ax.set_ylabel("VLM Quantiles")
        ax.set_title("Q-Q Plot")
        ax.grid(alpha=0.3)

        # Normalized comparison (K-NN → VLM scale)
        ax = axes[1, 1]
        knn_normalized = zscore_normalize_knn_to_vlm(vlm_scores, knn_scores)
        ax.hist(
            vlm_scores.flatten(),
            bins=30,
            alpha=0.6,
            label="VLM",
            color="blue",
        )
        ax.hist(
            knn_normalized.flatten(),
            bins=30,
            alpha=0.6,
            label="K-NN (normalized)",
            color="green",
        )
        ax.set_xlabel("Score (VLM scale)")
        ax.set_ylabel("Frequency")
        ax.set_title("After Z-Score Normalization (K-NN → VLM)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Distribution plot saved to: {output_path}")


class QuantileMatchingNormalizer:
    """
    Normalize scores using Quantile Matching.

    Transforms source distribution to match target distribution
    while preserving relative rankings.
    """

    def __init__(self):
        self.fitted = False
        self.source_sorted = None
        self.target_sorted = None
        self.score_to_percentile = None
        self.percentile_to_target = None

    def fit(self, source_scores: np.ndarray, target_scores: np.ndarray):
        """
        Fit the quantile matching transformation.

        Args:
            source_scores: VLM scores (to be transformed)
            target_scores: K-NN scores (target distribution)
        """
        self.source_sorted = np.sort(source_scores)
        self.target_sorted = np.sort(target_scores)

        # Create percentile mappings
        n_source = len(self.source_sorted)
        n_target = len(self.target_sorted)

        source_percentiles = np.linspace(0, 100, n_source)
        target_percentiles = np.linspace(0, 100, n_target)

        # Source score → percentile
        self.score_to_percentile = interpolate.interp1d(
            self.source_sorted,
            source_percentiles,
            bounds_error=False,
            fill_value=(0, 100),
        )

        # Percentile → target score
        self.percentile_to_target = interpolate.interp1d(
            target_percentiles,
            self.target_sorted,
            bounds_error=False,
            fill_value=(self.target_sorted[0], self.target_sorted[-1]),
        )

        self.fitted = True

        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores using fitted quantile matching.

        Args:
            scores: Scores to transform

        Returns:
            Transformed scores matching target distribution
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        scores = np.asarray(scores)

        # Source → percentile → target
        percentiles = self.score_to_percentile(scores)
        matched = self.percentile_to_target(percentiles)

        return matched

    def fit_transform(self, source_scores: np.ndarray, target_scores: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(source_scores, target_scores)
        return self.transform(source_scores)

    def get_stats(self) -> Dict:
        """Return distribution statistics for reporting."""
        if not self.fitted:
            return {}

        return {
            "source_min": float(self.source_sorted[0]),
            "source_max": float(self.source_sorted[-1]),
            "source_median": float(np.median(self.source_sorted)),
            "target_min": float(self.target_sorted[0]),
            "target_max": float(self.target_sorted[-1]),
            "target_median": float(np.median(self.target_sorted)),
        }


def quantile_match_scores(vlm_scores: np.ndarray, knn_scores: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for quantile matching.

    Args:
        vlm_scores: VLM scores to transform
        knn_scores: K-NN scores (target distribution)

    Returns:
        Tuple of (matched_scores, stats_dict)
    """
    normalizer = QuantileMatchingNormalizer()
    matched = normalizer.fit_transform(vlm_scores, knn_scores)
    stats_dict = normalizer.get_stats()

    return matched, stats_dict


class ZScoreNormalizer:
    """Z-Score normalization: K-NN → VLM scale."""

    def __init__(self):
        self.fitted = False
        self.knn_mean = None
        self.knn_std = None
        self.vlm_mean = None
        self.vlm_std = None

    def fit(self, vlm_scores: np.ndarray, knn_scores: np.ndarray):
        """
        Fit normalizer parameters.

        Args:
            vlm_scores: VLM scores (target scale, typically 1-10)
            knn_scores: K-NN predicted scores (to be transformed)
        """
        vlm_scores = np.asarray(vlm_scores)
        knn_scores = np.asarray(knn_scores)

        self.vlm_mean = float(np.mean(vlm_scores))
        self.vlm_std = float(np.std(vlm_scores))
        self.knn_mean = float(np.mean(knn_scores))
        self.knn_std = float(np.std(knn_scores))
        self.fitted = True
        return self

    def transform(self, knn_scores: np.ndarray) -> np.ndarray:
        """
        Transform K-NN scores to VLM scale.

        Args:
            knn_scores: K-NN scores to transform

        Returns:
            K-NN scores normalized to VLM scale
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        knn_scores = np.asarray(knn_scores)

        if self.knn_std == 0:
            return np.full_like(knn_scores, self.vlm_mean, dtype=float)

        z = (knn_scores - self.knn_mean) / self.knn_std
        knn_normalized = z * self.vlm_std + self.vlm_mean

        return knn_normalized

    def fit_transform(self, vlm_scores: np.ndarray, knn_scores: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(vlm_scores, knn_scores)
        return self.transform(knn_scores)

    def get_stats(self) -> Dict:
        """Return normalization statistics."""
        if not self.fitted:
            return {}

        return {
            "vlm_mean": self.vlm_mean,
            "vlm_std": self.vlm_std,
            "knn_mean": self.knn_mean,
            "knn_std": self.knn_std,
            "direction": "K-NN → VLM scale",
        }


def zscore_normalize_knn_to_vlm(vlm_scores: np.ndarray, knn_scores: np.ndarray) -> np.ndarray:
    """
    Convenience function: normalize K-NN scores to VLM scale.

    Args:
        vlm_scores: VLM scores (target distribution)
        knn_scores: K-NN scores (to be transformed)

    Returns:
        K-NN scores on VLM scale
    """
    normalizer = ZScoreNormalizer()
    return normalizer.fit_transform(vlm_scores, knn_scores)
