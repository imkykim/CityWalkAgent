"""Validation analyzer for statistical comparison of VLM and K-NN scores.

This module provides comprehensive statistical analysis to validate VLM scores
against K-NN predictions from Place Pulse 2.0 human perception data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .score_normalizer import (
    ScoreNormalizer,
    quantile_match_scores,
    zscore_normalize_knn_to_vlm,
)


class ValidationAnalyzer:
    """Analyzer for validating VLM scores against K-NN predictions."""

    DIMENSIONS = ["safe", "lively", "beautiful", "wealthy"]

    def __init__(
        self,
        vlm_scores: pd.DataFrame,
        knn_predictions: pd.DataFrame,
        image_ids: Optional[List[str]] = None,
        normalization_method: str = "quantile",
    ):
        """Initialize analyzer with score pairs.

        Args:
            vlm_scores: VLM scores DataFrame with columns: safe, lively, beautiful, wealthy
            knn_predictions: K-NN predictions DataFrame with same structure
            image_ids: Optional list of image IDs for reference
            normalization_method: How to normalize scores for secondary metrics
                ('quantile', 'zscore', or 'none')

        Raises:
            ValueError: If DataFrames have mismatched shapes or missing columns
        """
        if len(vlm_scores) != len(knn_predictions):
            raise ValueError(
                f"VLM and K-NN score counts don't match: "
                f"{len(vlm_scores)} vs {len(knn_predictions)}"
            )

        # Verify required dimensions
        missing_vlm = set(self.DIMENSIONS) - set(vlm_scores.columns)
        missing_knn = set(self.DIMENSIONS) - set(knn_predictions.columns)

        if missing_vlm:
            raise ValueError(f"VLM scores missing dimensions: {missing_vlm}")
        if missing_knn:
            raise ValueError(f"K-NN predictions missing dimensions: {missing_knn}")

        self.vlm_scores = vlm_scores[self.DIMENSIONS].copy()
        self.knn_predictions = knn_predictions[self.DIMENSIONS].copy()
        self.image_ids = image_ids or list(range(len(vlm_scores)))
        self.n_samples = len(vlm_scores)
        self.normalization_method = normalization_method

        if normalization_method not in {"quantile", "zscore", "none"}:
            raise ValueError(
                f"Unknown normalization method: {normalization_method}. "
                "Choose from {'quantile', 'zscore', 'none'}"
            )

        # Initialize normalizer for each dimension
        self.normalizers = {}
        for dim in self.DIMENSIONS:
            normalizer = ScoreNormalizer()
            normalizer.fit(
                self.vlm_scores[dim].values,
                self.knn_predictions[dim].values
            )
            self.normalizers[dim] = normalizer

    def analyze_dimension(self, dimension: str) -> Dict[str, Any]:
        """Perform full analysis for a single dimension.

        Args:
            dimension: Dimension to analyze (safe, lively, beautiful, wealthy)

        Returns:
            Dictionary with analysis results including:
            - Spearman correlation (primary metric)
            - Pearson correlation (after normalization)
            - MAE, RMSE, R² (after normalization)
            - Distribution statistics

        Raises:
            ValueError: If dimension is not recognized
        """
        if dimension not in self.DIMENSIONS:
            raise ValueError(
                f"Unknown dimension: {dimension}. "
                f"Must be one of {self.DIMENSIONS}"
            )

        vlm = self.vlm_scores[dimension].values
        knn = self.knn_predictions[dimension].values

        # Primary metric: Spearman correlation (rank-based, no normalization)
        spearman_rho, spearman_p = spearmanr(vlm, knn)

        # Normalize VLM scores for secondary metrics
        normalizer = self.normalizers[dimension]
        normalization = self.normalization_method
        match_stats: Dict[str, Any] = {}
        normalized_scores: np.ndarray
        normalized_source = "none"

        if normalization == "quantile":
            normalized_scores, match_stats = quantile_match_scores(vlm, knn)
            normalization_label = "quantile_matching"
            normalized_source = "vlm"
            metric_true = knn
            metric_pred = normalized_scores
        elif normalization == "zscore":
            normalized_scores = zscore_normalize_knn_to_vlm(vlm, knn)
            normalization_label = "zscore_knn_to_vlm"
            normalized_source = "knn"
            metric_true = vlm
            metric_pred = normalized_scores
        else:
            normalized_scores = knn
            normalization_label = "none"
            normalized_source = "none"
            metric_true = knn
            metric_pred = vlm

        # Secondary metrics: Pearson, MAE, RMSE, R² (after normalization)
        pearson_r, pearson_p = pearsonr(metric_pred, metric_true)
        mae = mean_absolute_error(metric_true, metric_pred)
        rmse = np.sqrt(mean_squared_error(metric_true, metric_pred))
        r2 = r2_score(metric_true, metric_pred)

        # Get distribution statistics
        dist_stats = normalizer.get_distribution_stats()

        results = {
            # Metadata
            "dimension": dimension,
            "n_samples": self.n_samples,
            # Primary metric
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "spearman_significant": bool(spearman_p < 0.05),
            # Secondary metrics (normalized)
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "normalization": normalization_label,
            "normalized_source": normalized_source,
            # Distribution info
            "vlm_mean": float(dist_stats["vlm"]["mean"]),
            "vlm_std": float(dist_stats["vlm"]["std"]),
            "knn_mean": float(dist_stats["knn"]["mean"]),
            "knn_std": float(dist_stats["knn"]["std"]),
            "vlm_range_utilization": float(dist_stats["vlm"]["range_utilization"]),
            "knn_range_utilization": float(dist_stats["knn"]["range_utilization"]),
            "normalized_mean": float(np.mean(normalized_scores)),
            "normalized_std": float(np.std(normalized_scores)),
        }

        if match_stats:
            results["match_stats"] = match_stats

        return results

    def analyze_all_dimensions(self) -> pd.DataFrame:
        """Run analysis for all dimensions.

        Returns:
            DataFrame with one row per dimension and columns for all metrics
        """
        results = []
        for dim in self.DIMENSIONS:
            result = self.analyze_dimension(dim)
            results.append(result)

        return pd.DataFrame(results)

    def generate_report(self, k_value: Optional[int] = None) -> str:
        """Generate human-readable validation report.

        Args:
            k_value: Optional K used for K-NN (for display)

        Returns:
            Formatted report string with interpretation
        """
        # Analyze all dimensions
        results_df = self.analyze_all_dimensions()

        normalization_label_map = {
            "quantile": "Quantile Matching",
            "zscore": "Z-Score Normalization",
            "none": "None",
        }
        normalization_heading_map = {
            "quantile": "Quantile-Matched",
            "zscore": "Z-Score Normalized",
            "none": "Raw Scores",
        }
        normalization_title = normalization_label_map.get(
            self.normalization_method, self.normalization_method
        ).upper()
        normalization_heading = normalization_heading_map.get(
            self.normalization_method, self.normalization_method
        )

        report = []
        report.append("═" * 70)
        report.append("            CLIP + K-NN Validation Report")
        report.append("═" * 70)
        report.append("")
        report.append(f"Normalization: {normalization_title}")
        if k_value is not None:
            report.append(f"K value: {k_value}")
        report.append(f"Samples: {self.n_samples} waypoints")
        report.append("Anchor Set: Place Pulse 2.0")
        report.append("Method: K-NN prediction via CLIP embeddings")
        report.append("")

        # Primary metric: Spearman correlation
        report.append("─" * 70)
        report.append("                PRIMARY METRIC: Spearman Correlation")
        report.append("─" * 70)
        report.append(f"{'Dimension':<15} │ {'ρ (rho)':<10} │ {'p-value':<10} │ {'Significant':<12}")
        report.append("─" * 15 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12)

        for _, row in results_df.iterrows():
            sig = "✓" if row["spearman_significant"] else "✗"
            p_str = "<0.001" if row["spearman_p"] < 0.001 else f"{row['spearman_p']:.3f}"
            report.append(
                f"{row['dimension'].capitalize():<15} │ "
                f"{row['spearman_rho']:>10.3f} │ "
                f"{p_str:>10} │ "
                f"{sig:>12}"
            )

        avg_rho = results_df["spearman_rho"].mean()
        n_significant = results_df["spearman_significant"].sum()
        report.append("─" * 15 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12)
        report.append(
            f"{'Average':<15} │ {avg_rho:>10.3f} │ {'':>10} │ "
            f"{n_significant}/{len(results_df)} ✓"
        )

        report.append("")

        # Secondary metrics
        report.append("─" * 70)
        report.append(f"                SECONDARY METRICS ({normalization_heading})")
        report.append("─" * 70)
        report.append(
            f"{'Dimension':<15} │ {'Pearson r':<10} │ {'MAE':<8} │ {'RMSE':<8} │ {'R²':<8}"
        )
        report.append("─" * 15 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 8)

        for _, row in results_df.iterrows():
            report.append(
                f"{row['dimension'].capitalize():<15} │ "
                f"{row['pearson_r']:>10.3f} │ "
                f"{row['mae']:>8.2f} │ "
                f"{row['rmse']:>8.2f} │ "
                f"{row['r2']:>8.2f}"
            )

        report.append("")

        # Distribution comparison
        report.append("─" * 70)
        report.append("                DISTRIBUTION ALIGNMENT (After Normalization)")
        report.append("─" * 70)
        if self.normalization_method == "zscore":
            left_label = "VLM Mean"
            right_label = "K-NN (norm) Mean"
            left_std_label = "VLM Std"
            right_std_label = "K-NN (norm) Std"
            left_mean_key, right_mean_key = "vlm_mean", "normalized_mean"
            left_std_key, right_std_key = "vlm_std", "normalized_std"
        elif self.normalization_method == "quantile":
            left_label = "VLM (matched) Mean"
            right_label = "K-NN Mean"
            left_std_label = "VLM (matched) Std"
            right_std_label = "K-NN Std"
            left_mean_key, right_mean_key = "normalized_mean", "knn_mean"
            left_std_key, right_std_key = "normalized_std", "knn_std"
        else:
            left_label = "VLM Mean"
            right_label = "K-NN Mean"
            left_std_label = "VLM Std"
            right_std_label = "K-NN Std"
            left_mean_key, right_mean_key = "vlm_mean", "knn_mean"
            left_std_key, right_std_key = "vlm_std", "knn_std"

        report.append(
            f"{'Dimension':<15} │ {left_label:<15} │ {right_label:<17} │ "
            f"{left_std_label:<13} │ {right_std_label:<13}"
        )
        report.append(
            "─" * 15 + "┼" + "─" * 17 + "┼" + "─" * 19 + "┼" + "─" * 15 + "┼" + "─" * 13
        )

        for _, row in results_df.iterrows():
            report.append(
                f"{row['dimension'].capitalize():<15} │ "
                f"{row[left_mean_key]:>15.2f} │ "
                f"{row[right_mean_key]:>17.2f} │ "
                f"{row[left_std_key]:>13.2f} │ "
                f"{row[right_std_key]:>13.2f}"
            )

        report.append("")

        # Interpretation
        report.append("─" * 70)
        report.append("                INTERPRETATION")
        report.append("─" * 70)

        # Overall significance
        if n_significant == len(results_df):
            report.append(
                "✓ All dimensions show statistically significant correlation"
            )
            report.append("  with human perception (p < 0.05)")
        elif n_significant > 0:
            report.append(
                f"△ {n_significant}/{len(results_df)} dimensions show "
                "statistically significant correlation"
            )
        else:
            report.append(
                "✗ No dimensions show statistically significant correlation"
            )

        report.append("")

        # Correlation strength interpretation
        if avg_rho >= 0.7:
            strength = "STRONG"
        elif avg_rho >= 0.5:
            strength = "MODERATE-STRONG"
        elif avg_rho >= 0.3:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        report.append(
            f"✓ Average Spearman ρ = {avg_rho:.3f} indicates {strength} alignment"
        )
        report.append("  between VLM scores and crowdsourced human judgments")
        report.append("")

        # Best and worst dimensions
        best_dim = results_df.loc[results_df["spearman_rho"].idxmax()]
        worst_dim = results_df.loc[results_df["spearman_rho"].idxmin()]

        report.append(
            f"✓ Strongest correlation: {best_dim['dimension'].capitalize()} "
            f"(ρ={best_dim['spearman_rho']:.3f})"
        )
        if best_dim["dimension"] == "wealthy":
            report.append("  - VLM reliably identifies affluence cues")
        elif best_dim["dimension"] == "safe":
            report.append("  - VLM accurately assesses safety perception")

        report.append("")
        report.append(
            f"△ Weakest correlation: {worst_dim['dimension'].capitalize()} "
            f"(ρ={worst_dim['spearman_rho']:.3f})"
        )
        if worst_dim["dimension"] == "lively":
            report.append("  - Temporal/social activity harder to infer from static images")

        report.append("")

        # Range utilization warning
        low_range_dims = results_df[results_df["vlm_range_utilization"] < 0.5]
        if not low_range_dims.empty:
            report.append("⚠ VLM shows limited score range utilization:")
            for _, row in low_range_dims.iterrows():
                util = row["vlm_range_utilization"] * 100
                report.append(
                    f"  - {row['dimension'].capitalize()}: {util:.0f}% of 0-10 scale"
                )
            report.append("")

        report.append("Reference: Prior work (arXiv 2407.19719) achieved R²≈0.44")
        report.append("for Safety using similar CLIP+K-NN approach.")

        report.append("═" * 70)

        return "\n".join(report)

    def plot_correlation(
        self, dimension: str, output_path: Path, show_normalized: bool = True
    ):
        """Create scatter plot for single dimension.

        Args:
            dimension: Dimension to plot
            output_path: Path to save plot
            show_normalized: Whether to show normalized VLM scores
        """
        import matplotlib.pyplot as plt

        vlm = self.vlm_scores[dimension].values
        knn = self.knn_predictions[dimension].values

        # Get analysis results
        result = self.analyze_dimension(dimension)
        normalization = self.normalization_method

        if normalization == "quantile":
            x_scores = knn
            y_scores, _ = quantile_match_scores(vlm, knn)
            x_label = "K-NN Predicted Score"
            y_label = "VLM Score (quantile-matched)"
        elif normalization == "zscore":
            x_scores = zscore_normalize_knn_to_vlm(vlm, knn)
            y_scores = vlm
            x_label = "K-NN Predicted Score (normalized)"
            y_label = "VLM Score"
        else:
            x_scores = knn
            y_scores = vlm
            x_label = "K-NN Predicted Score"
            y_label = "VLM Score"

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(x_scores, y_scores, alpha=0.6, s=50, edgecolors="black", linewidths=0.5)

        # Add regression line
        z = np.polyfit(x_scores, y_scores, 1)
        p = np.poly1d(z)
        if normalization in {"quantile", "zscore"}:
            x_line = np.linspace(1, 10, 100)
        else:
            x_line = np.linspace(x_scores.min(), x_scores.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Regression")

        # Add diagonal reference line
        min_val = min(x_scores.min(), y_scores.min())
        max_val = max(x_scores.max(), y_scores.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k:",
            alpha=0.5,
            linewidth=1,
            label="Perfect correlation",
        )

        # Labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(
            f"{dimension.capitalize()} - VLM vs K-NN Predictions\n"
            f"Spearman ρ = {result['spearman_rho']:.3f} "
            f"(p = {result['spearman_p']:.4f})",
            fontsize=14,
            fontweight="bold",
        )

        if normalization in {"quantile", "zscore"}:
            ax.set_xlim(1, 10)
            ax.set_ylim(1, 10)
            ax.set_aspect("equal", adjustable="box")

        # Add statistics box
        stats_text = (
            f"N = {result['n_samples']}\n"
            f"Spearman ρ = {result['spearman_rho']:.3f}\n"
            f"Pearson r = {result['pearson_r']:.3f}\n"
            f"MAE = {result['mae']:.2f}\n"
            f"RMSE = {result['rmse']:.2f}\n"
            f"R² = {result['r2']:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved correlation plot: {output_path}")

    def plot_quantile_matching(self, dimension: str, output_path: Path):
        """Visualize quantile matching effect for a dimension."""
        import matplotlib.pyplot as plt

        vlm_original = self.vlm_scores[dimension].values
        knn_scores = self.knn_predictions[dimension].values
        vlm_matched, _ = quantile_match_scores(vlm_original, knn_scores)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{dimension.capitalize()}: Quantile Matching Effect", fontsize=14)

        # Before matching
        axes[0, 0].hist(
            vlm_original, bins=20, alpha=0.7, label="VLM (original)", color="blue"
        )
        axes[0, 0].hist(
            knn_scores, bins=20, alpha=0.7, label="K-NN", color="orange"
        )
        axes[0, 0].set_title("Before Quantile Matching")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Score")
        axes[0, 0].set_ylabel("Frequency")

        # After matching
        axes[0, 1].hist(
            vlm_matched, bins=20, alpha=0.7, label="VLM (matched)", color="green"
        )
        axes[0, 1].hist(
            knn_scores, bins=20, alpha=0.7, label="K-NN", color="orange"
        )
        axes[0, 1].set_title("After Quantile Matching")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Score")
        axes[0, 1].set_ylabel("Frequency")

        # Q-Q plot before
        sorted_vlm = np.sort(vlm_original)
        sorted_knn = np.sort(knn_scores)
        if len(sorted_vlm) != len(sorted_knn):
            x = np.linspace(0, 1, min(len(sorted_vlm), len(sorted_knn)))
            sorted_vlm = np.interp(x, np.linspace(0, 1, len(sorted_vlm)), sorted_vlm)
            sorted_knn = np.interp(x, np.linspace(0, 1, len(sorted_knn)), sorted_knn)

        axes[1, 0].scatter(sorted_knn, sorted_vlm, alpha=0.6, s=20)
        min_val = min(sorted_knn.min(), sorted_vlm.min())
        max_val = max(sorted_knn.max(), sorted_vlm.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
        axes[1, 0].set_title("Q-Q Plot (Before)")
        axes[1, 0].set_xlabel("K-NN Quantiles")
        axes[1, 0].set_ylabel("VLM Quantiles")
        axes[1, 0].legend()

        # Q-Q plot after
        sorted_matched = np.sort(vlm_matched)
        axes[1, 1].scatter(sorted_knn, sorted_matched, alpha=0.6, s=20, color="green")
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
        axes[1, 1].set_title("Q-Q Plot (After)")
        axes[1, 1].set_xlabel("K-NN Quantiles")
        axes[1, 1].set_ylabel("VLM Quantiles (Matched)")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved quantile matching plot: {output_path}")

    def plot_all_dimensions(self, output_dir: Path):
        """Create 2x2 subplot with all dimension correlations.

        Args:
            output_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create individual plots
        for dim in self.DIMENSIONS:
            output_path = output_dir / f"{dim}.png"
            self.plot_correlation(dim, output_path)

        # Create combined 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()

        for idx, dim in enumerate(self.DIMENSIONS):
            ax = axes[idx]

            vlm = self.vlm_scores[dim].values
            knn = self.knn_predictions[dim].values

            # Get analysis results
            result = self.analyze_dimension(dim)
            normalization = self.normalization_method

            if normalization == "quantile":
                x_scores = knn
                y_scores, _ = quantile_match_scores(vlm, knn)
            elif normalization == "zscore":
                x_scores = zscore_normalize_knn_to_vlm(vlm, knn)
                y_scores = vlm
            else:
                x_scores = knn
                y_scores = vlm

            # Scatter plot
            ax.scatter(x_scores, y_scores, alpha=0.6, s=30, edgecolors="black", linewidths=0.5)

            # Regression line
            z = np.polyfit(x_scores, y_scores, 1)
            p = np.poly1d(z)
            if normalization in {"quantile", "zscore"}:
                x_line = np.linspace(1, 10, 100)
            else:
                x_line = np.linspace(x_scores.min(), x_scores.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Diagonal reference
            # Labels
            if normalization == "zscore":
                ax.set_xlabel("K-NN Predicted Score (normalized)", fontsize=11)
                ax.set_ylabel("VLM Score", fontsize=11)
            elif normalization == "quantile":
                ax.set_xlabel("K-NN Predicted Score", fontsize=11)
                ax.set_ylabel("VLM Score (quantile-matched)", fontsize=11)
            else:
                ax.set_xlabel("K-NN Predicted Score", fontsize=11)
                ax.set_ylabel("VLM Score", fontsize=11)
            ax.set_title(
                f"{dim.capitalize()}\nρ = {result['spearman_rho']:.3f}, "
                f"R² = {result['r2']:.3f}",
                fontsize=12,
                fontweight="bold",
            )

            ax.grid(alpha=0.3)
            if normalization in {"quantile", "zscore"}:
                ax.set_xlim(1, 10)
                ax.set_ylim(1, 10)
                ax.set_aspect("equal", adjustable="box")

        plt.suptitle(
            "VLM vs K-NN Predictions - All Dimensions",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        combined_path = output_dir / "all_dimensions.png"
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved combined plot: {combined_path}")

    def export_results(self, output_dir: Path, k_value: Optional[int] = None):
        """Export validation results to files.

        Args:
            output_dir: Directory to save results
            k_value: Optional K used for K-NN (for display)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export summary DataFrame
        results_df = self.analyze_all_dimensions()
        results_df.to_csv(output_dir / "validation_summary.csv", index=False)

        # Export full report
        report = self.generate_report(k_value=k_value)
        with open(output_dir / "validation_report.txt", "w") as f:
            f.write(report)

        # Export raw scores
        comparison_df = pd.DataFrame(
            {
                "image_id": self.image_ids,
                **{f"vlm_{dim}": self.vlm_scores[dim] for dim in self.DIMENSIONS},
                **{f"knn_{dim}": self.knn_predictions[dim] for dim in self.DIMENSIONS},
            }
        )
        comparison_df.to_csv(output_dir / "score_comparison.csv", index=False)

        print(f"✓ Exported results to: {output_dir}")
