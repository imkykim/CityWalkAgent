"""
Visualization module for StreetAgent evaluation metrics.
Generates multi-dimensional route analysis visualizations with trend lines.
"""

import sys
from pathlib import Path

# When run as a script from src/utils, the script directory is sys.path[0].
# That would shadow stdlib's logging with our src.utils.logging when matplotlib imports logging.
if __name__ == "__main__":  # only adjust when invoked as a script
    current_dir = Path(__file__).resolve().parent
    if sys.path and sys.path[0] == str(current_dir):
        sys.path.pop(0)
        # Optionally add project root for convenience
        project_root = current_dir.parent.parent
        sys.path.insert(0, str(project_root))

import json
import csv
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

try:
    import matplotlib

    # Non-interactive backend for batch/CLI usage
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization. Install with `pip install matplotlib`."
    ) from exc

import numpy as np
from scipy.signal import savgol_filter

# Shared palette used by pandas-style rendering example; fallbacks use matplotlib cycles.
DEFAULT_SCORE_COLORS = {
    "functional_quality": "#D97706",
    "sensory_complexity": "#60A5FA",
    "spatial_legibility": "#10B981",
    "spatial_sequence": "#EAB308",
    "visual_coherence": "#1D4ED8",
    "safety": "#2E86AB",
    "comfort": "#A23B72",
    "interest": "#F18F01",
    "aesthetics": "#C73E1D",
}
DEFAULT_SCORE_ORDER: Tuple[str, ...] = tuple(DEFAULT_SCORE_COLORS.keys())


class RouteVisualizer:
    """Visualizes evaluation scores along walking routes with trend analysis."""

    DIMENSIONS = {
        "safety": {"color": "#2E86AB", "label": "Safety"},
        "comfort": {"color": "#A23B72", "label": "Comfort"},
        "interest": {"color": "#F18F01", "label": "Interest"},
        "aesthetics": {"color": "#C73E1D", "label": "Aesthetics"},
    }

    def __init__(self, figsize: Tuple[int, int] = (14, 6), dpi: int = 100):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-darkgrid")

    def _build_dim_config(
        self, dim_keys: Sequence[str], custom: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict]:
        """Merge default config with dynamic dimensions."""
        config = {
            k: v.copy() for k, v in (custom or self.DIMENSIONS).items() if k in dim_keys
        }

        palette = plt.cm.tab10.colors + plt.cm.Set3.colors
        for idx, dim in enumerate(dim_keys):
            if dim not in config:
                color = palette[idx % len(palette)]
                config[dim] = {"color": color, "label": dim}
            else:
                # Ensure label and color exist
                config.setdefault(dim, {})
                config[dim].setdefault("color", palette[idx % len(palette)])
                config[dim].setdefault("label", dim)
        return config

    def plot_scores_with_trends(
        self,
        scores: Dict[str, List[float]],
        waypoint_ids: Optional[List[str]] = None,
        title: str = "Scores by Dimension with Average Trend",
        show_legend: bool = True,
        smooth_window: int = 11,
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        visual_change_waypoints: Optional[Sequence[Union[str, int]]] = None,
    ) -> plt.Figure:
        """
        Plot dimension scores with smoothed trend lines.

        Args:
            scores: Dictionary mapping dimension names to score lists
            waypoint_ids: Optional list of waypoint identifiers
            title: Plot title
            show_legend: Whether to show legend
            smooth_window: Window size for Savitzky-Golay smoothing (must be odd)
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimension ids to plot
            visual_change_waypoints: Optional ids/indices for waypoints with visual changes

        Returns:
            Matplotlib figure object
        """
        dim_keys = list(dimensions) if dimensions else list(scores.keys())
        dim_config = self._build_dim_config(dim_keys)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Generate waypoint indices
        n_points = len(next(iter(scores.values())))
        if waypoint_ids is None:
            waypoint_ids = [str(i) for i in range(n_points)]
        x = np.arange(n_points)

        # Plot each dimension
        for dim_key in dim_keys:
            score_list = scores.get(dim_key)
            if score_list is None:
                continue

            config = dim_config[dim_key]

            # Plot raw scores with points and lines
            ax.plot(
                x,
                score_list,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=config["color"],
                alpha=0.7,
                label=config["label"],
            )

            # Add smoothed trend line if enough points
            if n_points >= smooth_window:
                # Ensure odd window size
                window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
                try:
                    smoothed = savgol_filter(score_list, window, 3)
                    ax.plot(
                        x,
                        smoothed,
                        linestyle="--",
                        linewidth=2,
                        color=config["color"],
                        alpha=0.9,
                    )
                except Exception as e:
                    print(f"Warning: Could not smooth {dim_key}: {e}")

        # Formatting
        ax.set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, 10.5)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        # Set x-ticks with rotation for readability
        tick_step = max(1, n_points // 20)  # Show ~20 labels max
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(
            waypoint_ids[::tick_step], rotation=45, ha="right", fontsize=8
        )

        # Highlight waypoints that triggered visual change detections
        if visual_change_waypoints:
            wp_to_idx = {str(wp_id): idx for idx, wp_id in enumerate(waypoint_ids)}
            change_indices: List[int] = []
            for wp in visual_change_waypoints:
                idx = wp_to_idx.get(str(wp))
                if idx is None:
                    try:
                        numeric_idx = int(wp)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= numeric_idx < n_points:
                        change_indices.append(numeric_idx)
                else:
                    change_indices.append(idx)

            for idx in sorted(set(change_indices)):
                ax.axvline(
                    x=idx,
                    color="gray",
                    linestyle="-",
                    alpha=0.35,
                    linewidth=1,
                    zorder=0,
                )

        if show_legend:
            ax.legend(loc="best", framealpha=0.9, fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        return fig

    def plot_dimension_comparison(
        self,
        scores: Dict[str, List[float]],
        waypoint_ids: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """
        Create a 2x2 subplot comparing all dimensions individually.

        Args:
            scores: Dictionary mapping dimension names to score lists
            waypoint_ids: Optional list of waypoint identifiers
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimension ids to plot

        Returns:
            Matplotlib figure object
        """
        dim_keys = list(dimensions) if dimensions else list(scores.keys())
        dim_config = self._build_dim_config(dim_keys)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        axes = axes.flatten()

        n_points = len(next(iter(scores.values())))
        if waypoint_ids is None:
            waypoint_ids = [str(i) for i in range(n_points)]
        x = np.arange(n_points)

        for idx, dim_key in enumerate(dim_keys):
            if idx >= 4:
                continue

            score_list = scores.get(dim_key)
            if score_list is None:
                continue

            ax = axes[idx]
            config = dim_config[dim_key]

            # Plot scores
            ax.plot(
                x,
                score_list,
                marker="o",
                markersize=5,
                linewidth=2,
                color=config["color"],
                alpha=0.7,
            )

            # Add mean line
            mean_score = np.mean(score_list)
            ax.axhline(
                mean_score,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=f"Mean: {mean_score:.2f}",
            )

            # Formatting
            ax.set_title(config["label"], fontsize=12, fontweight="bold")
            ax.set_xlabel("Waypoint ID", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_ylim(0, 10.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

            # Simplified x-ticks
            tick_step = max(1, n_points // 10)
            ax.set_xticks(x[::tick_step])
            ax.set_xticklabels(
                waypoint_ids[::tick_step], rotation=45, ha="right", fontsize=8
            )

        plt.suptitle(
            "Dimension-wise Score Analysis", fontsize=14, fontweight="bold", y=0.995
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved comparison to {save_path}")

        return fig

    def plot_heatmap(
        self,
        scores: Dict[str, List[float]],
        waypoint_ids: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """
        Create a heatmap visualization of scores across dimensions and waypoints.

        Args:
            scores: Dictionary mapping dimension names to score lists
            waypoint_ids: Optional list of waypoint identifiers
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimension ids to plot

        Returns:
            Matplotlib figure object
        """
        dim_keys = list(dimensions) if dimensions else list(scores.keys())
        dim_config = self._build_dim_config(dim_keys)

        # Prepare data matrix
        dimensions = list(dim_config.keys())
        data = np.array([scores[dim] for dim in dimensions if dim in scores])

        n_points = data.shape[1]
        if waypoint_ids is None:
            waypoint_ids = [str(i) for i in range(n_points)]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 4), dpi=self.dpi)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=10)

        # Set ticks and labels
        ax.set_yticks(np.arange(len(dimensions)))
        ax.set_yticklabels([dim_config[d]["label"] for d in dimensions])

        tick_step = max(1, n_points // 20)
        ax.set_xticks(np.arange(0, n_points, tick_step))
        ax.set_xticklabels(
            waypoint_ids[::tick_step], rotation=45, ha="right", fontsize=8
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Score", rotation=270, labelpad=20, fontsize=10)

        ax.set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        ax.set_title(
            "Score Heatmap by Dimension", fontsize=13, fontweight="bold", pad=15
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved heatmap to {save_path}")

        return fig

    def plot_dual_system_comparison(
        self,
        waypoint_results: List[Dict[str, Any]],
        title: str = "System 1 (Raw VLM) vs System 2 (Memory Context)",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """Plot System 1 vs System 2 scores to visualize memory influence."""
        waypoint_ids = [r["waypoint_id"] for r in waypoint_results]

        system1_scores: Dict[str, List[float]] = {}
        system2_scores: Dict[str, List[float]] = {}

        dim_keys = list(dimensions) if dimensions else ["safety", "comfort", "interest", "aesthetics"]
        for dim in dim_keys:
            system1_scores[dim] = [r["system1_scores"].get(dim, 0) for r in waypoint_results]
            system2_scores[dim] = [
                r.get("system2_scores", {}).get(dim, r["system1_scores"].get(dim, 0))
                if r.get("system2_scores")
                else r["system1_scores"].get(dim, 0)
                for r in waypoint_results
            ]

        system2_triggered = [
            i for i, r in enumerate(waypoint_results) if r.get("system2_triggered", False)
        ]

        dim_config = self._build_dim_config(dim_keys)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        axes = axes.flatten()
        x = np.arange(len(waypoint_ids))

        for idx, dim_key in enumerate(dim_keys):
            if idx >= 4:
                continue

            ax = axes[idx]
            config = dim_config[dim_key]

            ax.plot(
                x,
                system1_scores[dim_key],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=config["color"],
                alpha=0.5,
                label="System 1 (Raw VLM)",
                linestyle="--",
            )

            ax.plot(
                x,
                system2_scores[dim_key],
                marker="s",
                markersize=5,
                linewidth=2,
                color=config["color"],
                alpha=0.9,
                label="System 2 (Context)",
                linestyle="-",
            )

            for i in system2_triggered:
                s1_score = system1_scores[dim_key][i]
                s2_score = system2_scores[dim_key][i]
                if abs(s2_score - s1_score) >= 0.5:
                    color = "green" if s2_score > s1_score else "red"
                    ax.annotate(
                        "",
                        xy=(i, s2_score),
                        xytext=(i, s1_score),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            alpha=0.6,
                            lw=1.5,
                        ),
                    )

            ax.set_title(f"{config['label']} - Memory Influence", fontsize=12, fontweight="bold")
            ax.set_xlabel("Waypoint ID", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_ylim(0, 10.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

            tick_step = max(1, len(waypoint_ids) // 10)
            ax.set_xticks(x[::tick_step])
            ax.set_xticklabels(
                [str(wid) for wid in waypoint_ids[::tick_step]],
                rotation=45,
                ha="right",
                fontsize=8,
            )

            s1_mean = np.mean(system1_scores[dim_key])
            s2_mean = np.mean(system2_scores[dim_key])
            delta = s2_mean - s1_mean
            stats_text = f"Δ: {delta:+.2f}\nS1: {s1_mean:.2f}\nS2: {s2_mean:.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved dual-system comparison to {save_path}")

        return fig

    def plot_adjustment_heatmap(
        self,
        waypoint_results: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """Create heatmap showing System 2 score adjustments."""
        dim_keys = list(dimensions) if dimensions else ["safety", "comfort", "interest", "aesthetics"]

        adjustments: List[List[float]] = []
        for dim in dim_keys:
            dim_adjustments = [
                r.get("score_adjustments", {}).get(dim, 0) if r.get("score_adjustments") else 0
                for r in waypoint_results
            ]
            adjustments.append(dim_adjustments)

        adjustments_arr = np.array(adjustments)
        waypoint_ids = [r["waypoint_id"] for r in waypoint_results]

        fig, ax = plt.subplots(figsize=(14, 4), dpi=self.dpi)

        im = ax.imshow(
            adjustments_arr, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=3, interpolation="nearest"
        )

        ax.set_yticks(np.arange(len(dim_keys)))
        ax.set_yticklabels(dim_keys)

        tick_step = max(1, len(waypoint_ids) // 20)
        ax.set_xticks(np.arange(0, len(waypoint_ids), tick_step))
        ax.set_xticklabels(
            [str(wid) for wid in waypoint_ids[::tick_step]],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Score Adjustment (System 2 - System 1)", rotation=270, labelpad=20)

        ax.set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        ax.set_title(
            "System 2 Score Adjustments (Memory Influence)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved adjustment heatmap to {save_path}")

        return fig

    def plot_narrative_timeline(
        self,
        narrative_chapters: List[Dict[str, Any]],
        waypoint_scores: Dict[str, List[float]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Visualize narrative chapters alongside score timeline."""
        fig, (ax_scores, ax_narrative) = plt.subplots(
            2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 2]}, dpi=self.dpi
        )

        waypoint_ids = list(range(len(next(iter(waypoint_scores.values())))))

        for dim, scores in waypoint_scores.items():
            ax_scores.plot(waypoint_ids, scores, marker="o", label=dim, linewidth=2)

        for chapter in narrative_chapters:
            wid = chapter["waypoint_id"]
            ax_scores.axvline(x=wid, color="red", linestyle="--", alpha=0.3)
            ax_scores.text(
                wid,
                9.5,
                f"Ch{chapter['chapter_number']}",
                fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax_scores.set_title("Score Timeline with Narrative Chapters", fontsize=13, fontweight="bold")
        ax_scores.set_ylabel("Score", fontsize=11)
        ax_scores.legend(loc="best")
        ax_scores.grid(True, alpha=0.3)
        ax_scores.set_ylim(0, 10.5)

        ax_narrative.set_xlim(0, len(waypoint_ids))
        ax_narrative.set_ylim(0, len(narrative_chapters) + 1)
        ax_narrative.axis("off")

        for i, chapter in enumerate(narrative_chapters):
            y_pos = len(narrative_chapters) - i
            x_pos = chapter["waypoint_id"]
            ax_narrative.plot([x_pos, x_pos], [0, y_pos], "k-", alpha=0.3)

            text = f"Ch{chapter['chapter_number']}: {chapter.get('key_observation', '')[:60]}..."
            tone_colors = {
                "optimistic": "#10B981",
                "neutral": "#6B7280",
                "cautious": "#F59E0B",
                "concerned": "#EF4444",
            }
            color = tone_colors.get(chapter.get("emotional_tone", "neutral"), "#6B7280")

            ax_narrative.text(
                x_pos + 1,
                y_pos,
                text,
                fontsize=9,
                va="center",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved narrative timeline to {save_path}")

        return fig

    def generate_summary_stats(self, scores: Dict[str, List[float]]) -> Dict:
        """
        Calculate summary statistics for each dimension.

        Args:
            scores: Dictionary mapping dimension names to score lists

        Returns:
            Dictionary containing statistics for each dimension
        """
        stats = {}
        for dim_key, score_list in scores.items():
            arr = np.array(score_list)
            stats[dim_key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
            }
        return stats

    def plot_radar_charts(
        self,
        analyses: Iterable[Mapping],
        output_dir: Path,
        dimensions: Optional[Sequence[str]] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> List[Path]:
        """
        Generate radar charts for waypoints with visual change detection.

        Creates one radar chart per waypoint that has visual_change_detected=True,
        showing all dimension scores in a spider/radar plot.

        Args:
            analyses: Iterable of WaypointAnalysis-like objects or dicts
            output_dir: Directory to save radar chart images
            dimensions: Optional ordered list of dimensions to plot
            figsize: Figure size in inches (width, height)

        Returns:
            List of paths to saved radar chart images
        """
        # Create output directory for radar charts
        radar_dir = output_dir / "radar_charts"
        radar_dir.mkdir(parents=True, exist_ok=True)

        analyses_list = list(analyses)
        if not analyses_list:
            print("No analyses provided for radar charts.")
            return []

        def get(obj, key, default=None):
            if isinstance(obj, Mapping):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Filter for waypoints with visual change detected
        def _has_visual_change(analysis) -> bool:
            detected = get(analysis, "visual_change_detected", None)
            if detected is None:
                detected = get(analysis, "visual_change", False)
            return bool(detected)

        filtered_analyses = [a for a in analyses_list if _has_visual_change(a)]

        if not filtered_analyses:
            print(
                "No waypoints with visual change detection passed. No radar charts generated."
            )
            return []

        print(
            f"Generating radar charts for {len(filtered_analyses)} waypoints with visual changes..."
        )

        # Determine dimensions
        dim_keys = (
            list(dimensions)
            if dimensions
            else sorted(
                {k for a in filtered_analyses for k in get(a, "scores", {}).keys()}
            )
        )
        if not dim_keys:
            print("No dimensions found in analyses.")
            return []

        dim_config = self._build_dim_config(dim_keys)
        saved_paths = []

        # Generate one radar chart per filtered waypoint
        for analysis in filtered_analyses:
            waypoint_id = get(analysis, "waypoint_id", "unknown")
            scores_dict = get(analysis, "scores", {})

            # Prepare data for radar chart
            labels = [dim_config[dim]["label"] for dim in dim_keys if dim in scores_dict]
            values = [scores_dict[dim] for dim in dim_keys if dim in scores_dict]

            if not values:
                continue

            # Number of variables
            num_vars = len(labels)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # The plot is circular, so close the loop by appending the start value
            values += values[:1]
            angles += angles[:1]

            # Create radar chart
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi)

            # Plot data
            ax.plot(angles, values, "o-", linewidth=2, color="#2E86AB", label="Scores")
            ax.fill(angles, values, alpha=0.25, color="#2E86AB")

            # Set labels for each axis
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=10)

            # Set y-axis limits
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, alpha=0.7)

            # Add grid
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

            # Add title
            phash_distance = get(analysis, "phash_distance", None)
            title = f"Waypoint {waypoint_id}"
            if phash_distance is not None:
                title += f" (pHash: {phash_distance:.2f})"
            ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

            # Save figure
            output_path = radar_dir / f"radar_waypoint_{waypoint_id}.png"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            saved_paths.append(output_path)
            print(f"  Saved radar chart: {output_path.name}")

        print(f"Generated {len(saved_paths)} radar charts in {radar_dir}")
        return saved_paths


def plot_dual_system_analysis(
    waypoint_results: List[Dict[str, Any]],
    narrative_chapters: Optional[List[Dict[str, Any]]] = None,
    output_dir: Path = Path("."),
    dimensions: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Generate complete dual-system analysis visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    viz = RouteVisualizer()
    paths: Dict[str, Path] = {}

    comparison_path = output_dir / "system1_vs_system2_comparison.png"
    viz.plot_dual_system_comparison(
        waypoint_results,
        save_path=comparison_path,
        dimensions=dimensions,
    )
    paths["comparison"] = comparison_path

    heatmap_path = output_dir / "system2_adjustments_heatmap.png"
    viz.plot_adjustment_heatmap(
        waypoint_results,
        save_path=heatmap_path,
        dimensions=dimensions,
    )
    paths["heatmap"] = heatmap_path

    waypoint_ids = [r["waypoint_id"] for r in waypoint_results]
    system1_scores: Dict[str, List[float]] = {}
    for dim in dimensions or ["safety", "comfort", "interest", "aesthetics"]:
        system1_scores[dim] = [r["system1_scores"].get(dim, 0) for r in waypoint_results]

    system1_path = output_dir / "system1_scores.png"
    viz.plot_scores_with_trends(
        system1_scores,
        waypoint_ids=[str(wid) for wid in waypoint_ids],
        title="System 1 Scores (Raw VLM Perception)",
        save_path=system1_path,
        dimensions=dimensions,
    )
    paths["system1"] = system1_path

    final_scores: Dict[str, List[float]] = {}
    for dim in dimensions or ["safety", "comfort", "interest", "aesthetics"]:
        final_scores[dim] = [
            r.get("system2_scores", {}).get(dim, r["system1_scores"].get(dim, 0))
            if r.get("system2_scores")
            else r["system1_scores"].get(dim, 0)
            for r in waypoint_results
        ]

    final_path = output_dir / "final_scores.png"
    viz.plot_scores_with_trends(
        final_scores,
        waypoint_ids=[str(wid) for wid in waypoint_ids],
        title="Final Scores (System 2 when triggered)",
        save_path=final_path,
        dimensions=dimensions,
    )
    paths["final_scores"] = final_path

    if narrative_chapters:
        narrative_path = output_dir / "narrative_timeline.png"
        viz.plot_narrative_timeline(
            narrative_chapters=narrative_chapters,
            waypoint_scores=final_scores,
            save_path=narrative_path,
        )
        paths["narrative"] = narrative_path

    return paths

    @classmethod
    def from_jsonl(
        cls, jsonl_path: Path, score_fields: Optional[List[str]] = None
    ) -> Tuple[Dict[str, List[float]], List[str], List[Mapping]]:
        """
        Load scores from JSONL file and create visualizations.

        Args:
            jsonl_path: Path to JSONL evaluation file
            score_fields: List of score field names (default: ['safety', 'comfort', 'interest', 'aesthetics'])

        Returns:
            Tuple of (scores dict, waypoint_ids list, analyses list)
        """
        scores: Dict[str, List[float]] = {}
        waypoint_ids: List[str] = []
        analyses: List[Mapping] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                if idx == 0 and score_fields is None:
                    # Infer score fields from first record (exclude obvious metadata)
                    excluded = {"waypoint_id", "image_id", "timestamp", "gps", "heading"}
                    score_fields = [
                        k for k in data.keys() if k not in excluded and isinstance(data.get(k), (int, float))
                    ]
                    if not score_fields:
                        score_fields = list(cls.DIMENSIONS.keys())
                if score_fields is None:
                    score_fields = list(cls.DIMENSIONS.keys())

                waypoint_id = data.get("waypoint_id", str(len(waypoint_ids)))
                waypoint_ids.append(str(waypoint_id))
                record_scores: Dict[str, float] = {}

                for field in score_fields:
                    score = data.get(field, data.get(f"{field}_score", 0))
                    value = float(score)
                    scores.setdefault(field, []).append(value)
                    record_scores[field] = value

                record = {
                    "waypoint_id": waypoint_id,
                    "scores": record_scores,
                    "visual_change_detected": data.get(
                        "visual_change_detected", data.get("visual_change", False)
                    ),
                }
                if "phash_distance" in data:
                    record["phash_distance"] = data["phash_distance"]
                analyses.append(record)

        return scores, waypoint_ids, analyses


def plot_analysis_results(
    analyses: Iterable[Mapping],
    output_path: Path,
    dimensions: Optional[Sequence[str]] = None,
) -> Path:
    """
    Convenience helper: plot waypoint scores directly from analysis outputs.

    Args:
        analyses: Iterable of WaypointAnalysis-like objects or dicts
        output_path: PNG path to write
        dimensions: Optional ordered list of dimensions to plot
    """
    analyses_list = list(analyses)
    if not analyses_list:
        raise ValueError("No analyses provided for plotting.")

    def get(obj, key, default=None):
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _coerce_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_waypoint_id(raw_wp: object, fallback_idx: int) -> Tuple[float, str]:
        """Return numeric waypoint for plotting plus the display label."""
        label = raw_wp if raw_wp is not None else fallback_idx
        try:
            numeric = float(raw_wp)
        except (TypeError, ValueError):
            numeric = float(fallback_idx)
        return numeric, str(label)

    def _format_tick(label: str, numeric: float) -> str:
        try:
            return f"wp{int(float(label)):03d}"
        except (TypeError, ValueError):
            try:
                return f"wp{int(round(numeric)):03d}"
            except (TypeError, ValueError):
                return str(label)

    discovered_dims: Set[str] = set()
    records: List[Dict] = []
    for idx, analysis in enumerate(analyses_list):
        waypoint = get(analysis, "waypoint_id", idx)
        numeric_wp, label = _coerce_waypoint_id(waypoint, idx)
        scores_dict = get(analysis, "scores", {}) or {}
        discovered_dims.update(scores_dict.keys())
        visual_change = bool(
            get(
                analysis,
                "visual_change_detected",
                get(analysis, "visual_change", False),
            )
        )
        records.append(
            {
                "order": idx,
                "numeric_id": numeric_wp,
                "label": label,
                "scores": scores_dict,
                "visual_change": visual_change,
            }
        )

    if dimensions:
        dim_keys = list(dimensions)
    else:
        ordered_defaults = [dim for dim in DEFAULT_SCORE_ORDER if dim in discovered_dims]
        remaining = sorted(discovered_dims - set(ordered_defaults))
        dim_keys = ordered_defaults + remaining
    if not dim_keys:
        raise ValueError("No dimensions found in analyses.")

    # Ensure chronological ordering using numeric ids (fallbacks keep insertion order)
    records.sort(key=lambda rec: (rec["numeric_id"], rec["order"]))
    x_values = [rec["numeric_id"] for rec in records]
    xtick_labels = [_format_tick(rec["label"], rec["numeric_id"]) for rec in records]
    visual_change_positions = [
        rec["numeric_id"] for rec in records if rec["visual_change"]
    ]

    scores_by_dim: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
    for dim in dim_keys:
        for record in records:
            scores_by_dim[dim].append(
                _coerce_float(record["scores"].get(dim, 0.0), default=0.0)
            )

    avg_scores = [
        float(np.mean([scores_by_dim[dim][idx] for dim in dim_keys]))
        for idx in range(len(records))
    ]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, dim in enumerate(dim_keys):
        color = DEFAULT_SCORE_COLORS.get(dim) or plt.cm.tab10(idx % 10)
        ax.plot(
            x_values,
            scores_by_dim[dim],
            marker="o",
            linewidth=2,
            markersize=4,
            label=dim,
            color=color,
        )

    ax.plot(
        x_values,
        avg_scores,
        linestyle="--",
        linewidth=2.5,
        color="black",
        label="average",
    )

    for wp in visual_change_positions:
        ax.axvline(
            x=wp,
            color="gray",
            linestyle="-",
            alpha=0.35,
            linewidth=1,
            zorder=0,
        )

    ax.set_title("Scores by Dimension (Average included, Visual Change Highlighted)")
    ax.set_xlabel("Waypoint ID")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 10.5)
    ax.set_yticks(range(0, 11, 1))
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.3, alpha=0.25, color="#CFCFCF")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("#1F2933")
        ax.spines[spine].set_visible(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    if x_values:
        ax.set_xticks(x_values)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# CLI helper to visualize existing outputs
# ============================================================================


def _load_analyses_from_json(path: Path) -> List[Mapping]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    raise ValueError("Expected a JSON list of analysis records.")


def _load_analyses_from_csv(path: Path) -> List[Mapping]:
    by_waypoint: Dict[str, Dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wp = row.get("image_id") or row.get("waypoint_id")
            dim = row.get("dimension_id") or row.get("dimension_name")
            score = row.get("score")
            if wp is None or dim is None or score is None:
                continue
            record = by_waypoint.setdefault(wp, {"waypoint_id": wp, "scores": {}})
            try:
                record["scores"][dim] = float(score)
            except ValueError:
                continue

    def _sort_key(item):
        wp = item["waypoint_id"]
        try:
            return int(wp)
        except (TypeError, ValueError):
            return wp

    return [by_waypoint[k] for k in sorted(by_waypoint.keys(), key=_sort_key)]


def _extract_visual_change_waypoints(
    analyses: Iterable[Mapping],
) -> List[str]:
    """Return waypoint ids that include a visual change flag."""
    change_ids: List[str] = []

    for idx, analysis in enumerate(analyses):
        if isinstance(analysis, Mapping):
            wp = analysis.get("waypoint_id", idx)
            change_flag = analysis.get(
                "visual_change_detected", analysis.get("visual_change", False)
            )
        else:
            wp = getattr(analysis, "waypoint_id", idx)
            change_flag = getattr(
                analysis,
                "visual_change_detected",
                getattr(analysis, "visual_change", False),
            )
        if change_flag:
            change_ids.append(str(wp))

    return change_ids


def main():
    """CLI entrypoint: visualize analysis JSON/CSV/JSONL outputs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot analysis metrics from JSON/CSV/JSONL outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", type=Path, help="Path to analysis_results.json/.csv/.jsonl"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to save the plot PNG (default: input with .png)",
    )
    parser.add_argument(
        "-d",
        "--dimensions",
        nargs="+",
        help="Explicit dimension ids to plot (otherwise derived from data)",
    )
    parser.add_argument(
        "--radar-charts",
        action="store_true",
        help="Generate radar charts for waypoints that flagged visual change",
    )
    parser.add_argument(
        "--radar-dir",
        type=Path,
        help="Override directory for radar chart outputs (default: alongside input)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = args.output or input_path.with_suffix(".png")
    generate_radar = bool(args.radar_charts or args.radar_dir)
    radar_dir = args.radar_dir or output_path.parent
    analyses_for_radar: Optional[List[Mapping]] = None
    viz = RouteVisualizer()

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        scores, waypoint_ids, analyses_for_radar = RouteVisualizer.from_jsonl(
            input_path, score_fields=args.dimensions
        )
        change_wp_ids = (
            _extract_visual_change_waypoints(analyses_for_radar)
            if analyses_for_radar
            else None
        )
        viz.plot_scores_with_trends(
            scores,
            waypoint_ids=waypoint_ids,
            title="Waypoint Metrics by Dimension",
            dimensions=args.dimensions,
            visual_change_waypoints=change_wp_ids,
            save_path=output_path,
        )
    elif suffix == ".json":
        analyses = _load_analyses_from_json(input_path)
        analyses_for_radar = analyses
        plot_analysis_results(analyses, output_path, dimensions=args.dimensions)
    elif suffix == ".csv":
        analyses = _load_analyses_from_csv(input_path)
        analyses_for_radar = analyses
        plot_analysis_results(analyses, output_path, dimensions=args.dimensions)
    else:
        raise SystemExit("Unsupported input format. Use .json, .csv, or .jsonl.")

    if generate_radar:
        if not analyses_for_radar:
            print("No analyses available to generate radar charts.")
        else:
            viz.plot_radar_charts(
                analyses_for_radar, radar_dir, dimensions=args.dimensions
            )

    print(f"Saved plot to {output_path}")
    try:
        plt.show()
    except Exception:
        # In headless environments showing may fail; ignore.
        pass


if __name__ == "__main__":
    main()
