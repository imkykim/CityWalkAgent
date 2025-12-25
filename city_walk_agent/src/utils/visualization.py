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
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.config import DEFAULT_FRAMEWORK_ID
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

    # Legacy hardcoded dimensions (kept for backward compatibility)
    DIMENSIONS = {
        "safety": {"color": "#2E86AB", "label": "Safety"},
        "comfort": {"color": "#A23B72", "label": "Comfort"},
        "interest": {"color": "#F18F01", "label": "Interest"},
        "aesthetics": {"color": "#C73E1D", "label": "Aesthetics"},
    }

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 6),
        dpi: int = 100,
        framework_id: str = DEFAULT_FRAMEWORK_ID,
    ):
        """
        Initialize visualizer with framework-agnostic dimension support.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution for saved figures
            framework_id: Evaluation framework ID to load dimensions from (default: DEFAULT_FRAMEWORK_ID)
        """
        self.figsize = figsize
        self.dpi = dpi
        self.framework_id = framework_id

        # Use clean white background style
        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linewidth": 0.5,
                "grid.linestyle": ":",
                "axes.edgecolor": "#cccccc",
                "axes.linewidth": 0.8,
            }
        )

        # Load framework and build dimension configuration
        from src.config import load_framework

        self.framework = load_framework(framework_id)
        self.dimension_config = self._build_dimension_config_from_framework()

    def _build_dimension_config_from_framework(self) -> Dict[str, Dict]:
        """Build dimension configuration from loaded framework.

        Returns:
            Dict mapping dimension IDs to config dicts with 'color' and 'label' keys
        """
        config = {}
        palette = plt.cm.tab10.colors + plt.cm.Set3.colors

        for idx, dim in enumerate(self.framework["dimensions"]):
            dim_id = dim["id"]

            # Try to get color from DEFAULT_SCORE_COLORS, otherwise use palette
            color = DEFAULT_SCORE_COLORS.get(dim_id, palette[idx % len(palette)])

            config[dim_id] = {
                "color": color,
                "label": dim["name_en"],  # Use English name from framework
            }

        return config

    def _build_dim_config(
        self, dim_keys: Sequence[str], custom: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict]:
        """Merge default config with dynamic dimensions.

        Args:
            dim_keys: Dimension IDs to include in config
            custom: Optional custom dimension config to override defaults

        Returns:
            Dict mapping dimension IDs to config dicts with 'color' and 'label' keys
        """
        # Use custom config if provided, otherwise use instance dimension config
        base_config = custom or self.dimension_config

        config = {k: v.copy() for k, v in base_config.items() if k in dim_keys}

        # Fill in missing dimensions with auto-generated colors
        palette = plt.cm.tab10.colors + plt.cm.Set3.colors
        for idx, dim in enumerate(dim_keys):
            if dim not in config:
                color = palette[idx % len(palette)]
                config[dim] = {"color": color, "label": dim}
            else:
                # Ensure label and color exist
                config[dim].setdefault("color", palette[idx % len(palette)])
                config[dim].setdefault("label", dim)
        return config

    def _infer_dimension_keys(
        self,
        dimensions: Optional[Sequence[str]] = None,
        scores: Optional[Mapping[str, Sequence[float]]] = None,
        waypoint_results: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[str]:
        """
        Resolve which dimensions to visualize, preferring explicit args, then data, then framework.

        Args:
            dimensions: Explicit dimension ids to plot
            scores: Optional score mapping to inspect
            waypoint_results: Optional waypoint result records to inspect

        Returns:
            Ordered list of dimension ids to plot
        """
        if dimensions:
            return list(dimensions)

        if scores:
            return list(scores.keys())

        if waypoint_results:
            for record in waypoint_results:
                if isinstance(record, Mapping):
                    for key in ("system2_scores", "system1_scores", "scores"):
                        maybe_scores = record.get(key)
                        if maybe_scores:
                            return list(maybe_scores.keys())
                else:
                    for key in ("system2_scores", "system1_scores", "scores"):
                        maybe_scores = getattr(record, key, None)
                        if maybe_scores:
                            return list(maybe_scores.keys())

        if self.dimension_config:
            return list(self.dimension_config.keys())

        return list(DEFAULT_SCORE_ORDER)

    def plot_scores_with_trends(
        self,
        scores: Dict[str, List[float]],
        waypoint_ids: Optional[List[str]] = None,
        title: str = "Scores by Dimension with Overall Average",
        show_legend: bool = True,
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        visual_change_waypoints: Optional[Sequence[Union[str, int]]] = None,
        system2_triggered_waypoints: Optional[Sequence[Union[str, int]]] = None,
    ) -> plt.Figure:
        """
        Plot dimension scores with overall average line.

        Args:
            scores: Dictionary mapping dimension names to score lists
            waypoint_ids: Optional list of waypoint identifiers
            title: Plot title
            show_legend: Whether to show legend
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimension ids to plot
            visual_change_waypoints: Optional ids/indices for waypoints with visual changes
            system2_triggered_waypoints: Optional ids/indices for waypoints where System 2 was triggered

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
        all_scores_array = []
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
                markersize=3.5,
                linewidth=1.2,
                color=config["color"],
                alpha=0.8,
                label=config["label"],
            )

            # Collect scores for overall average
            all_scores_array.append(score_list)

        # Calculate and plot overall average across all dimensions
        if all_scores_array:
            overall_avg = np.mean(all_scores_array, axis=0)
            ax.plot(
                x,
                overall_avg,
                linestyle="--",
                linewidth=1.8,
                color="#333333",
                alpha=0.9,
                label="Overall Average",
                zorder=10,
            )

        # Formatting
        ax.set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, 10.5)

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

        # Highlight waypoints where System 2 was triggered
        if system2_triggered_waypoints:
            wp_to_idx = {str(wp_id): idx for idx, wp_id in enumerate(waypoint_ids)}
            trigger_indices: List[int] = []
            for wp in system2_triggered_waypoints:
                idx = wp_to_idx.get(str(wp))
                if idx is None:
                    try:
                        numeric_idx = int(wp)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= numeric_idx < n_points:
                        trigger_indices.append(numeric_idx)
                else:
                    trigger_indices.append(idx)

            for idx in sorted(set(trigger_indices)):
                ax.axvline(
                    x=idx,
                    color="gray",
                    linestyle="--",
                    alpha=0.3,
                    linewidth=0.8,
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

    def plot_route_scatter(
        self,
        data_points: List[Dict[str, float]],
        title: str = "Route Scatter Plot with Average Score",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a scatter plot of a route, with GPS path colored by average score.

        Args:
            data_points: List of dictionaries, each with 'lat', 'lon', and 'average_score'.
            title: Plot title.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        if not data_points:
            raise ValueError("No data points provided for scatter plot.")

        lats = [p["lat"] for p in data_points]
        lons = [p["lon"] for p in data_points]
        scores = [p["average_score"] for p in data_points]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)

        # Create scatter plot
        scatter = ax.scatter(
            lons,
            lats,
            c=scores,
            cmap="viridis",
            s=50,
            vmin=0,
            vmax=10,
            alpha=0.8,
        )

        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Average Score", fontsize=11, fontweight="bold")

        # Formatting
        ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved scatter plot to {save_path}")

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

        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )
        for dim in dim_keys:
            system1_scores[dim] = [
                r["system1_scores"].get(dim, 0) for r in waypoint_results
            ]
            system2_scores[dim] = [
                (
                    r.get("system2_scores", {}).get(
                        dim, r["system1_scores"].get(dim, 0)
                    )
                    if r.get("system2_scores")
                    else r["system1_scores"].get(dim, 0)
                )
                for r in waypoint_results
            ]

        system2_triggered = [
            i
            for i, r in enumerate(waypoint_results)
            if r.get("system2_triggered", False)
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
                markersize=3.5,
                linewidth=1.2,
                color=config["color"],
                alpha=0.6,
                label="System 1 (Raw VLM)",
                linestyle="--",
            )

            ax.plot(
                x,
                system2_scores[dim_key],
                marker="s",
                markersize=4,
                linewidth=1.5,
                color=config["color"],
                alpha=0.9,
                label="System 2 (Context)",
                linestyle="-",
            )

            # Add vertical lines for System 2 triggered waypoints
            for i in system2_triggered:
                ax.axvline(
                    x=i,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.4,
                    zorder=0,
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

            ax.set_title(
                f"{config['label']} - Memory Influence", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Waypoint ID", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_ylim(0, 10.5)
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
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )

        adjustments: List[List[float]] = []
        for dim in dim_keys:
            dim_adjustments = [
                (
                    r.get("score_adjustments", {}).get(dim, 0)
                    if r.get("score_adjustments")
                    else 0
                )
                for r in waypoint_results
            ]
            adjustments.append(dim_adjustments)

        adjustments_arr = np.array(adjustments)
        waypoint_ids = [r["waypoint_id"] for r in waypoint_results]

        fig, ax = plt.subplots(figsize=(14, 4), dpi=self.dpi)

        im = ax.imshow(
            adjustments_arr,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-3,
            vmax=3,
            interpolation="nearest",
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
        cbar.set_label(
            "Score Adjustment (System 2 - System 1)", rotation=270, labelpad=20
        )

        # Add vertical lines for System 2 triggered waypoints
        system2_triggered = [
            i
            for i, r in enumerate(waypoint_results)
            if r.get("system2_triggered", False)
        ]
        for idx in system2_triggered:
            ax.axvline(
                x=idx,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
                zorder=10,
            )

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
        narrative_chapters: Optional[List[Dict[str, Any]]],
        waypoint_scores: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """Visualize narrative chapters alongside score timeline."""
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, scores=waypoint_scores
        )
        dim_config = self._build_dim_config(dim_keys)

        fig, (ax_scores, ax_narrative) = plt.subplots(
            2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 2]}, dpi=self.dpi
        )

        waypoint_ids = list(range(len(next(iter(waypoint_scores.values())))))

        for dim in dim_keys:
            scores = waypoint_scores.get(dim)
            if scores is None:
                continue
            config = dim_config.get(dim, {"label": dim, "color": None})
            ax_scores.plot(
                waypoint_ids,
                scores,
                marker="o",
                markersize=3.5,
                label=config["label"],
                linewidth=1.2,
                color=config.get("color"),
                alpha=0.8,
            )

        if narrative_chapters:
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

        ax_scores.set_title(
            "Score Timeline with Narrative Chapters", fontsize=13, fontweight="bold"
        )
        ax_scores.set_ylabel("Score", fontsize=11)
        ax_scores.legend(loc="best")
        ax_scores.set_ylim(0, 10.5)

        ax_narrative.set_xlim(0, len(waypoint_ids))
        if narrative_chapters:
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
                color = tone_colors.get(
                    chapter.get("emotional_tone", "neutral"), "#6B7280"
                )

                ax_narrative.text(
                    x_pos + 1,
                    y_pos,
                    text,
                    fontsize=9,
                    va="center",
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
                )
        else:
            ax_narrative.set_ylim(0, 1)
            ax_narrative.axis("off")
            ax_narrative.text(
                0.5,
                0.5,
                "No narrative chapters available",
                transform=ax_narrative.transAxes,
                fontsize=12,
                ha="center",
                va="center",
                style="italic",
                color="gray",
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
            labels = [
                dim_config[dim]["label"] for dim in dim_keys if dim in scores_dict
            ]
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
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi
            )

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
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.3)

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

    def plot_dual_system_radar_charts(
        self,
        waypoint_results: Sequence[Mapping[str, Any]],
        output_dir: Path,
        dimensions: Optional[Sequence[str]] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> Dict[str, Path]:
        """
        Generate System1, System2/Final, and overlay radar charts for all waypoints.

        Args:
            waypoint_results: Sequence of waypoint result dicts containing system1_scores/system2_scores
            output_dir: Base directory to write radar charts into
            dimensions: Optional ordered list of dimensions to plot
            figsize: Figure size in inches

        Returns:
            Dict of directories for the generated radar chart sets
        """
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )
        dim_config = self._build_dim_config(dim_keys)

        base_dir = output_dir / "radar_charts_dual"
        sys1_dir = base_dir / "system1"
        sys2_dir = base_dir / "system2"
        overlay_dir = base_dir / "overlay"
        for d in (sys1_dir, sys2_dir, overlay_dir):
            d.mkdir(parents=True, exist_ok=True)

        def _prepare_scores(result: Mapping[str, Any], key: str) -> List[float]:
            source = (
                result.get(key)
                if isinstance(result, Mapping)
                else getattr(result, key, None)
            )
            if source is None:
                source = {}
            return [float(source.get(dim, 0.0) or 0.0) for dim in dim_keys]

        def _plot_single(
            path: Path, scores: List[float], label: str, color: str
        ) -> None:
            labels = [dim_config[dim]["label"] for dim in dim_keys]
            values = list(scores) + [scores[0]]
            angles = np.linspace(0, 2 * np.pi, len(dim_keys), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi
            )
            ax.plot(angles, values, "o-", linewidth=2, color=color, label=label)
            ax.fill(angles, values, alpha=0.25, color=color)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, alpha=0.7)
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.3)
            ax.set_title(label, fontsize=13, fontweight="bold", pad=20)
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        def _plot_overlay(
            path: Path, scores1: List[float], scores2: List[float]
        ) -> None:
            labels = [dim_config[dim]["label"] for dim in dim_keys]
            angles = np.linspace(0, 2 * np.pi, len(dim_keys), endpoint=False).tolist()
            angles += angles[:1]

            s1 = list(scores1) + [scores1[0]]
            s2 = list(scores2) + [scores2[0]]

            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi
            )
            ax.plot(angles, s1, "o--", linewidth=1.8, color="#94A3B8", label="System 1")
            ax.fill(angles, s1, alpha=0.20, color="#94A3B8")
            ax.plot(
                angles,
                s2,
                "o-",
                linewidth=2.2,
                color="#EF4444",
                label="System 2 / Final",
            )
            ax.fill(angles, s2, alpha=0.25, color="#EF4444")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, alpha=0.7)
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.3)
            ax.set_title("Objective vs Persona", fontsize=13, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        for wp_result in waypoint_results:
            wp_id = (
                wp_result.get("waypoint_id")
                if isinstance(wp_result, Mapping)
                else getattr(wp_result, "waypoint_id", "unknown")
            )
            sys1_scores = _prepare_scores(wp_result, "system1_scores")
            sys2_scores = _prepare_scores(wp_result, "system2_scores")
            obj_scores = _prepare_scores(wp_result, "objective_scores")
            per_scores = _prepare_scores(wp_result, "persona_scores")

            fallback = (
                wp_result.get("scores", {})
                if isinstance(wp_result, Mapping)
                else getattr(wp_result, "scores", {})
            )
            if not any(sys2_scores) and fallback:
                sys2_scores = [float(fallback.get(dim, 0.0) or 0.0) for dim in dim_keys]

            _plot_single(
                sys1_dir / f"radar_sys1_wp_{wp_id}.png",
                sys1_scores,
                f"Waypoint {wp_id} - System 1",
                "#0EA5E9",
            )
            _plot_single(
                sys2_dir / f"radar_sys2_wp_{wp_id}.png",
                sys2_scores,
                f"Waypoint {wp_id} - System 2/Final",
                "#EF4444",
            )
            _plot_overlay(
                overlay_dir / f"radar_overlay_wp_{wp_id}.png", obj_scores, per_scores
            )

        return {
            "radar_system1_dir": sys1_dir,
            "radar_system2_dir": sys2_dir,
            "radar_overlay_dir": overlay_dir,
        }

    def plot_persona_comparison(
        self,
        waypoint_results: Sequence[Mapping[str, Any]],
        title: str = "Objective vs Persona-Aware Evaluation",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """
        Plot objective scores vs persona-aware scores for dual VLM evaluation.

        This visualization shows the difference between objective (framework-only)
        and persona-aware evaluations in the dual evaluation system using area fill
        to highlight differences between the two evaluation systems.

        Args:
            waypoint_results: Sequence of waypoint result dicts containing:
                - objective_scores: Scores from objective evaluation (research)
                - persona_scores: Scores from persona-aware evaluation (final)
            title: Plot title
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimensions to plot

        Returns:
            Matplotlib figure object
        """
        # Infer dimensions
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )
        dim_config = self._build_dim_config(dim_keys)

        # Extract waypoint IDs
        waypoint_ids = []
        for r in waypoint_results:
            if isinstance(r, Mapping):
                waypoint_ids.append(r.get("waypoint_id", len(waypoint_ids)))
            else:
                waypoint_ids.append(getattr(r, "waypoint_id", len(waypoint_ids)))

        # Prepare score data
        objective_scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
        persona_scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}

        for r in waypoint_results:
            if isinstance(r, Mapping):
                obj_scores = r.get("objective_scores", {})
                per_scores = r.get("persona_scores", {})
                # Fallback to old field names for backward compatibility
                if not obj_scores:
                    obj_scores = r.get("neutral_scores", r.get("system1_scores", {}))
                if not per_scores:
                    per_scores = r.get("scores", {})
            else:
                obj_scores = getattr(r, "objective_scores", {})
                per_scores = getattr(r, "persona_scores", {})
                # Fallback to old field names for backward compatibility
                if not obj_scores:
                    obj_scores = getattr(
                        r, "neutral_scores", getattr(r, "system1_scores", {})
                    )
                if not per_scores:
                    per_scores = getattr(r, "scores", {})

            for dim in dim_keys:
                objective_scores[dim].append(float(obj_scores.get(dim, 0) or 0))
                persona_scores[dim].append(float(per_scores.get(dim, 0) or 0))

        # Create 3x2 subplot for dimensions
        fig, axes = plt.subplots(3, 2, figsize=(16, 16), dpi=self.dpi)
        axes = axes.flatten()
        x = np.arange(len(waypoint_ids))

        # Colors for the two evaluation types (matching reference image style)
        objective_color = "#94A3B8"  # Gray for objective/System 1
        persona_color = "#EF4444"  # Red for persona-aware/System 2
        fill_color = "#EF4444"  # Red fill for difference area

        for idx, dim_key in enumerate(dim_keys[:5]):  # Limit to 5 dimensions
            ax = axes[idx]
            config = dim_config[dim_key]

            obj_data = np.array(objective_scores[dim_key])
            per_data = np.array(persona_scores[dim_key])

            # Plot objective scores (solid line, no markers for cleaner look)
            ax.plot(
                x,
                obj_data,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=objective_color,
                alpha=0.9,
                label="Objective",
                linestyle="-",  # Solid line instead of dashed
            )

            # Plot persona-aware scores (solid line with different marker)
            ax.plot(
                x,
                per_data,
                marker="x",
                markersize=5,
                linewidth=1.5,
                color=persona_color,
                alpha=0.9,
                label="Persona-Aware",
                linestyle="-",  # Solid line
            )

            # Fill area between the two lines to show difference
            ax.fill_between(
                x,
                obj_data,
                per_data,
                color=fill_color,
                alpha=0.15,
                label="Difference",
            )

            ax.set_title(
                f"{config['label']} - Persona Impact", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Waypoint ID", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_ylim(0, 10.5)
            ax.legend(loc="upper right", fontsize=9)

            # X-axis formatting
            tick_step = max(1, len(waypoint_ids) // 10)
            ax.set_xticks(x[::tick_step])
            ax.set_xticklabels(
                [str(wid) for wid in waypoint_ids[::tick_step]],
                rotation=45,
                ha="right",
                fontsize=8,
            )

            # Add statistics text box
            obj_mean = np.mean(obj_data)
            p_mean = np.mean(per_data)
            delta_mean = p_mean - obj_mean
            stats_text = (
                f"Avg Δ: {delta_mean:+.2f}\n"
                f"Objective: {obj_mean:.2f}\n"
                f"Persona: {p_mean:.2f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Hide unused subplot if dimensions < 6
        if len(dim_keys) < 6:
            axes[5].set_visible(False)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved persona comparison to {save_path}")

        return fig

    def plot_persona_neutral_overview(
        self,
        waypoint_results: Sequence[Mapping[str, Any]],
        title: str = "Objective vs Persona Overview",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """
        Plot objective vs persona averages alongside a delta heatmap.

        This combines the line + heatmap view showing the difference between
        objective (research) and persona-aware (final) evaluations.
        """
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )
        if not dim_keys:
            raise ValueError("No dimensions available for persona overview plot.")

        def _get(obj, key, default=None):
            if isinstance(obj, Mapping):
                return obj.get(key, default)
            return getattr(obj, key, default)

        waypoint_ids: List[str] = []
        objective_scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
        persona_scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}

        for idx, result in enumerate(waypoint_results):
            waypoint_id = _get(result, "waypoint_id", idx)
            waypoint_ids.append(str(waypoint_id))

            # Try new field names first, fallback to old for backward compatibility
            objective = (
                _get(result, "objective_scores")
                or _get(result, "neutral_scores")
                or _get(result, "system1_scores")
                or {}
            )
            persona = (
                _get(result, "persona_scores") or _get(result, "scores") or objective
            )

            for dim in dim_keys:
                objective_scores[dim].append(float(objective.get(dim, 0) or 0.0))
                persona_scores[dim].append(float(persona.get(dim, 0) or 0.0))

        n_points = len(waypoint_ids)
        if n_points == 0:
            raise ValueError("No waypoint results provided for persona overview plot.")

        avg_objective = [
            float(np.mean([objective_scores[dim][i] for dim in dim_keys]))
            for i in range(n_points)
        ]
        avg_persona = [
            float(np.mean([persona_scores[dim][i] for dim in dim_keys]))
            for i in range(n_points)
        ]

        delta_matrix = np.array(
            [
                [
                    persona_scores[dim][i] - objective_scores[dim][i]
                    for i in range(n_points)
                ]
                for dim in dim_keys
            ]
        )

        fig, axes = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.25, 1]}
        )

        x = np.arange(n_points)
        axes[0].plot(
            x,
            avg_objective,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color="#94A3B8",
            alpha=0.9,
            label="Objective (Research)",
            linestyle="--",
        )
        axes[0].plot(
            x,
            avg_persona,
            marker="s",
            markersize=4,
            linewidth=1.6,
            color="#EF4444",
            alpha=0.95,
            label="Persona-Aware (Final)",
        )

        axes[0].fill_between(
            x,
            avg_objective,
            avg_persona,
            color="#EF4444",
            alpha=0.12,
            label="Difference Zone",
        )

        tick_step = max(1, n_points // 12)
        axes[0].set_xticks(x[::tick_step])
        axes[0].set_xticklabels(
            [waypoint_ids[i] for i in range(0, n_points, tick_step)],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axes[0].set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Average Score", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, 10.5)
        axes[0].set_title(f"{title} – Averages", fontsize=13, fontweight="bold", pad=12)
        axes[0].legend(loc="best", fontsize=10)

        if delta_matrix.size:
            max_abs = float(np.max(np.abs(delta_matrix))) if delta_matrix.size else 1.0
            max_abs = max(max_abs, 0.5)
            im = axes[1].imshow(
                delta_matrix,
                aspect="auto",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
                interpolation="nearest",
            )
            axes[1].set_yticks(np.arange(len(dim_keys)))
            axes[1].set_yticklabels(dim_keys)

            heat_tick_step = max(1, n_points // 12)
            axes[1].set_xticks(np.arange(0, n_points, heat_tick_step))
            axes[1].set_xticklabels(
                waypoint_ids[::heat_tick_step],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            axes[1].set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
            axes[1].set_title(
                f"{title} – Persona − Neutral",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )

            # Annotate deltas for readability
            for row_idx, dim in enumerate(dim_keys):
                for col_idx in range(n_points):
                    delta_val = delta_matrix[row_idx, col_idx]
                    axes[1].text(
                        col_idx,
                        row_idx,
                        f"{delta_val:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label("Persona Adjustment", rotation=270, labelpad=18)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved persona overview to {save_path}")

        return fig

    def plot_persona_summary_radar(
        self,
        waypoint_results: Sequence[Mapping[str, Any]],
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        figsize: Tuple[int, int] = (10, 10),
    ) -> plt.Figure:
        """
        Create radar chart showing average neutral vs persona scores across all dimensions.

        Args:
            waypoint_results: Sequence of waypoint result dicts with neutral_scores and scores
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimensions to plot
            figsize: Figure size in inches

        Returns:
            Matplotlib figure object
        """
        # Infer dimensions
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )
        dim_config = self._build_dim_config(dim_keys)

        # Calculate average scores across all waypoints
        neutral_avgs = []
        persona_avgs = []

        for dim in dim_keys:
            neutral_vals = []
            persona_vals = []

            for r in waypoint_results:
                if isinstance(r, Mapping):
                    obj_scores = r.get("neutral_scores", r.get("system1_scores", {}))
                    per_scores = r.get("scores", {})
                else:
                    obj_scores = getattr(
                        r, "neutral_scores", getattr(r, "system1_scores", {})
                    )
                    per_scores = getattr(r, "scores", {})

                neutral_vals.append(float(obj_scores.get(dim, 0) or 0))
                persona_vals.append(float(per_scores.get(dim, 0) or 0))

            neutral_avgs.append(np.mean(neutral_vals))
            persona_avgs.append(np.mean(persona_vals))

        # Prepare radar chart data
        labels = [dim_config[dim]["label"] for dim in dim_keys]
        num_vars = len(labels)

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the plot by appending the first value
        neutral_avgs += neutral_avgs[:1]
        persona_avgs += persona_avgs[:1]
        angles += angles[:1]

        # Create radar chart
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi
        )

        # Plot neutral scores
        ax.plot(
            angles,
            neutral_avgs,
            "o--",
            linewidth=2,
            color="#94A3B8",
            label="Neutral (No Persona)",
        )
        ax.fill(angles, neutral_avgs, alpha=0.20, color="#94A3B8")

        # Plot persona scores
        ax.plot(
            angles,
            persona_avgs,
            "o-",
            linewidth=2.5,
            color="#EF4444",
            label="Persona-Aware",
        )
        ax.fill(angles, persona_avgs, alpha=0.25, color="#EF4444")

        # Set axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)

        # Set y-axis limits and labels
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9, alpha=0.7)

        # Add grid
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

        # Add legend
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

        # Add title
        ax.set_title(
            "Average Persona Impact Across All Dimensions",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved persona summary radar to {save_path}")

        return fig

    def plot_persona_delta_distribution(
        self,
        waypoint_results: Sequence[Mapping[str, Any]],
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """
        Plot distribution of persona adjustments (score deltas) across dimensions.

        Shows histograms and box plots of how much persona hints adjust scores
        from neutral baseline, revealing persona bias patterns.

        Args:
            waypoint_results: Sequence of waypoint result dicts with persona_adjustments
            save_path: Optional path to save figure
            dimensions: Optional ordered list of dimensions to plot

        Returns:
            Matplotlib figure object
        """
        # Infer dimensions
        dim_keys = self._infer_dimension_keys(
            dimensions=dimensions, waypoint_results=waypoint_results
        )
        dim_config = self._build_dim_config(dim_keys)

        # Collect adjustments for each dimension
        adjustments_by_dim: Dict[str, List[float]] = {dim: [] for dim in dim_keys}

        for r in waypoint_results:
            if isinstance(r, Mapping):
                adj = r.get("persona_adjustments", {})
            else:
                adj = getattr(r, "persona_adjustments", {})

            if adj:
                for dim in dim_keys:
                    val = adj.get(dim)
                    if val is not None:
                        adjustments_by_dim[dim].append(float(val))

        # Create subplot with histogram and box plot
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Top: Histograms for first 4 dimensions
        for idx, dim in enumerate(dim_keys[:4]):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            config = dim_config[dim]
            deltas = adjustments_by_dim[dim]

            if not deltas:
                ax.text(
                    0.5,
                    0.5,
                    "No persona adjustments",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    style="italic",
                    color="gray",
                )
                ax.set_title(config["label"], fontsize=12, fontweight="bold")
                continue

            # Plot histogram
            ax.hist(
                deltas,
                bins=20,
                color=config["color"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add vertical line at zero
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)

            # Add mean and median lines
            mean_delta = np.mean(deltas)
            median_delta = np.median(deltas)

            ax.axvline(
                x=mean_delta,
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=0.8,
                label=f"Mean: {mean_delta:+.2f}",
            )
            ax.axvline(
                x=median_delta,
                color="blue",
                linestyle="-.",
                linewidth=2,
                alpha=0.8,
                label=f"Median: {median_delta:+.2f}",
            )

            ax.set_title(config["label"], fontsize=12, fontweight="bold")
            ax.set_xlabel("Score Adjustment (Persona - Neutral)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=":")

        # Bottom: Box plot comparing all dimensions
        ax_box = fig.add_subplot(gs[2, :])

        # Prepare data for box plot
        box_data = []
        box_labels = []

        for dim in dim_keys:
            deltas = adjustments_by_dim[dim]
            if deltas:
                box_data.append(deltas)
                box_labels.append(dim_config[dim]["label"])

        if box_data:
            bp = ax_box.boxplot(
                box_data,
                labels=box_labels,
                patch_artist=True,
                notch=True,
                showmeans=True,
            )

            # Color boxes
            for idx, (patch, dim) in enumerate(
                zip(bp["boxes"], dim_keys[: len(box_data)])
            ):
                config = dim_config.get(dim, {})
                color = config.get("color", "#94A3B8")
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Add horizontal line at zero
            ax_box.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)

            ax_box.set_title(
                "Persona Adjustment Distribution by Dimension",
                fontsize=13,
                fontweight="bold",
            )
            ax_box.set_ylabel("Score Adjustment", fontsize=11)
            ax_box.set_xlabel("Dimension", fontsize=11)
            ax_box.grid(True, alpha=0.3, linestyle=":", axis="y")
            ax_box.tick_params(axis="x", rotation=45)

        plt.suptitle(
            "Persona Adjustment Patterns (Persona - Neutral)",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved persona delta distribution to {save_path}")

        return fig

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: Path,
        score_fields: Optional[List[str]] = None,
        framework_id: Optional[str] = None,
    ) -> Tuple[Dict[str, List[float]], List[str], List[Mapping]]:
        """
        Load scores from JSONL file and create visualizations.

        Args:
            jsonl_path: Path to JSONL evaluation file
            score_fields: Explicit score field names (otherwise inferred from data or framework)
            framework_id: Optional framework id to fall back to for dimension names

        Returns:
            Tuple of (scores dict, waypoint_ids list, analyses list)
        """
        scores: Dict[str, List[float]] = {}
        waypoint_ids: List[str] = []
        analyses: List[Mapping] = []
        framework_dims: List[str] = []

        if framework_id:
            try:
                from src.config import load_framework

                framework = load_framework(framework_id)
                framework_dims = [dim["id"] for dim in framework.get("dimensions", [])]
            except Exception:
                framework_dims = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                if idx == 0 and score_fields is None:
                    # Infer score fields from first record (exclude obvious metadata)
                    excluded = {
                        "waypoint_id",
                        "image_id",
                        "timestamp",
                        "gps",
                        "heading",
                    }
                    score_fields = [
                        k
                        for k in data.keys()
                        if k not in excluded and isinstance(data.get(k), (int, float))
                    ]
                    if not score_fields and framework_dims:
                        score_fields = framework_dims
                    if not score_fields:
                        score_fields = list(DEFAULT_SCORE_ORDER)
                if score_fields is None:
                    score_fields = framework_dims or list(DEFAULT_SCORE_ORDER)

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


def plot_dual_system_analysis(
    waypoint_results: List[Dict[str, Any]],
    narrative_chapters: Optional[List[Dict[str, Any]]] = None,
    output_dir: Path = Path("."),
    dimensions: Optional[Sequence[str]] = None,
    framework_id: str = DEFAULT_FRAMEWORK_ID,
    generate_radar_sets: bool = False,
) -> Dict[str, Path]:
    """Generate complete dual-system analysis visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    viz = RouteVisualizer(framework_id=framework_id)
    paths: Dict[str, Path] = {}
    dim_keys = viz._infer_dimension_keys(
        dimensions=dimensions, waypoint_results=waypoint_results
    )

    comparison_path = output_dir / "system1_vs_system2_comparison.png"
    viz.plot_dual_system_comparison(
        waypoint_results,
        save_path=comparison_path,
        dimensions=dim_keys,
    )
    paths["comparison"] = comparison_path

    heatmap_path = output_dir / "system2_adjustments_heatmap.png"
    viz.plot_adjustment_heatmap(
        waypoint_results,
        save_path=heatmap_path,
        dimensions=dim_keys,
    )
    paths["heatmap"] = heatmap_path

    waypoint_ids = [r["waypoint_id"] for r in waypoint_results]

    # Get indices where System 2 was triggered
    system2_triggered_indices = [
        i for i, r in enumerate(waypoint_results) if r.get("system2_triggered", False)
    ]
    system2_triggered_waypoints = [waypoint_ids[i] for i in system2_triggered_indices]

    system1_scores: Dict[str, List[float]] = {}
    for dim in dim_keys:
        system1_scores[dim] = [
            r["system1_scores"].get(dim, 0) for r in waypoint_results
        ]

    system1_path = output_dir / "system1_scores.png"
    viz.plot_scores_with_trends(
        system1_scores,
        waypoint_ids=[str(wid) for wid in waypoint_ids],
        title="System 1 Scores (Raw VLM Perception)",
        save_path=system1_path,
        dimensions=dim_keys,
        system2_triggered_waypoints=system2_triggered_waypoints,
    )
    paths["system1"] = system1_path

    final_scores: Dict[str, List[float]] = {}
    for dim in dim_keys:
        final_scores[dim] = [
            (
                r.get("system2_scores", {}).get(dim, r["system1_scores"].get(dim, 0))
                if r.get("system2_scores")
                else r["system1_scores"].get(dim, 0)
            )
            for r in waypoint_results
        ]

    final_path = output_dir / "final_scores.png"
    viz.plot_scores_with_trends(
        final_scores,
        waypoint_ids=[str(wid) for wid in waypoint_ids],
        title="Final Scores (System 2 when triggered)",
        save_path=final_path,
        dimensions=dim_keys,
        system2_triggered_waypoints=system2_triggered_waypoints,
    )
    paths["final_scores"] = final_path

    narrative_path = output_dir / "narrative_timeline.png"
    viz.plot_narrative_timeline(
        narrative_chapters=narrative_chapters,
        waypoint_scores=final_scores,
        save_path=narrative_path,
        dimensions=dim_keys,
    )
    paths["narrative"] = narrative_path

    if generate_radar_sets:
        radar_dirs = viz.plot_dual_system_radar_charts(
            waypoint_results=waypoint_results,
            output_dir=output_dir,
            dimensions=dim_keys,
        )
        paths.update(radar_dirs)

    return paths


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
        if not scores_dict:
            # Fall back to system 2 (final) scores, then system 1 scores
            scores_dict = (
                get(analysis, "system2_scores", {})
                or get(analysis, "system1_scores", {})
                or {}
            )
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
        ordered_defaults = [
            dim for dim in DEFAULT_SCORE_ORDER if dim in discovered_dims
        ]
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


def regenerate_visualizations(
    output_dir: Path,
    framework_id: str = DEFAULT_FRAMEWORK_ID,
    personality: str = "unknown",
    only: str = "all",
    radar_charts: bool = False,
) -> Dict[str, Path]:
    """
    Regenerate persona/system visualizations using an existing analysis_results.json.

    This replaces the old examples/regenerate_visualizations.py script.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    analysis_path = output_dir / "analysis_results.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"analysis_results.json not found in {output_dir}. Run a demo first."
        )

    analyses = _load_analyses_from_json(analysis_path)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    viz = RouteVisualizer(framework_id=framework_id)
    viz_paths: Dict[str, Path] = {}

    def _get(obj, key, default=None):
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if only in ("persona", "all"):
        dim_keys = viz._infer_dimension_keys(dimensions=None, waypoint_results=analyses)
        waypoint_ids = [
            str(_get(r, "waypoint_id", idx)) for idx, r in enumerate(analyses)
        ]
        system2_triggers = [
            str(_get(r, "waypoint_id", idx))
            for idx, r in enumerate(analyses)
            if _get(r, "system2_triggered", False)
        ]

        objective_scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
        persona_scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}

        for result in analyses:
            # Try new field names first, fallback to old for backward compatibility
            objective = (
                _get(result, "objective_scores")
                or _get(result, "neutral_scores")
                or _get(result, "system1_scores")
                or {}
            )
            persona = (
                _get(result, "persona_scores") or _get(result, "scores") or objective
            )

            for dim in dim_keys:
                objective_scores[dim].append(float(objective.get(dim, 0) or 0.0))
                persona_scores[dim].append(float(persona.get(dim, 0) or 0.0))

        comparison_path = viz_dir / "persona_comparison.png"
        viz.plot_persona_comparison(
            waypoint_results=analyses,
            title=f"Objective vs Persona-Aware Evaluation ({personality})",
            save_path=comparison_path,
            dimensions=dim_keys,
        )
        viz_paths["persona_comparison"] = comparison_path

        overview_path = viz_dir / "persona_objective_overview.png"
        viz.plot_persona_neutral_overview(
            waypoint_results=analyses,
            title=f"Objective vs Persona Overview ({personality})",
            save_path=overview_path,
            dimensions=dim_keys,
        )
        viz_paths["persona_overview"] = overview_path

        radar_path = viz_dir / "persona_summary_radar.png"
        viz.plot_persona_summary_radar(
            waypoint_results=analyses,
            save_path=radar_path,
            dimensions=dim_keys,
        )
        viz_paths["persona_radar"] = radar_path

        delta_path = viz_dir / "persona_delta_distribution.png"
        viz.plot_persona_delta_distribution(
            waypoint_results=analyses,
            save_path=delta_path,
            dimensions=dim_keys,
        )
        viz_paths["persona_delta"] = delta_path

        objective_path = viz_dir / "scores_objective.png"
        viz.plot_scores_with_trends(
            scores=objective_scores,
            waypoint_ids=waypoint_ids,
            title="Objective Evaluation (Research)",
            save_path=objective_path,
            dimensions=dim_keys,
            system2_triggered_waypoints=system2_triggers,
        )
        viz_paths["objective_scores"] = objective_path

        persona_path = viz_dir / "scores_persona_aware.png"
        viz.plot_scores_with_trends(
            scores=persona_scores,
            waypoint_ids=waypoint_ids,
            title=f"Persona-Aware Evaluation ({personality})",
            save_path=persona_path,
            dimensions=dim_keys,
            system2_triggered_waypoints=system2_triggers,
        )
        viz_paths["persona_scores"] = persona_path

    if only in ("system", "all"):
        narrative_json = output_dir / "narrative_chapters.json"
        narrative_chapters = None
        if narrative_json.exists():
            narrative_chapters = json.loads(narrative_json.read_text())

        system_paths = plot_dual_system_analysis(
            waypoint_results=analyses,
            narrative_chapters=narrative_chapters,
            output_dir=viz_dir,
            framework_id=framework_id,
            generate_radar_sets=radar_charts,
        )
        viz_paths.update(system_paths)

    return viz_paths


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
        "input",
        nargs="?",
        type=Path,
        help="Path to analysis_results.json/.csv/.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to save the plot PNG (default: input with .png)",
    )
    parser.add_argument(
        "--regenerate-output-dir",
        type=Path,
        help="Regenerate persona/system visualizations from an output directory (replaces examples/regenerate_visualizations.py)",
    )
    parser.add_argument(
        "--regenerate-only",
        choices=["persona", "system", "all"],
        default="all",
        help="Subset of visualizations to regenerate (with --regenerate-output-dir)",
    )
    parser.add_argument(
        "--personality",
        default="unknown",
        help="Personality label for persona plots (regeneration mode)",
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
    parser.add_argument(
        "-f",
        "--framework-id",
        default=DEFAULT_FRAMEWORK_ID,
        help="Framework id to use for labels/colors when inferring dimensions",
    )
    args = parser.parse_args()

    if args.regenerate_output_dir:
        paths = regenerate_visualizations(
            output_dir=args.regenerate_output_dir,
            framework_id=args.framework_id,
            personality=args.personality,
            only=args.regenerate_only,
            radar_charts=args.radar_charts,
        )

        print("Regenerated visualizations:")
        for name, path in paths.items():
            print(f"  - {name}: {path}")
        return

    if args.input is None:
        raise SystemExit(
            "Either provide an input file to plot or --regenerate-output-dir."
        )

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = args.output or input_path.with_suffix(".png")
    generate_radar = bool(args.radar_charts or args.radar_dir)
    radar_dir = args.radar_dir or output_path.parent
    analyses_for_radar: Optional[List[Mapping]] = None
    framework_id = args.framework_id
    viz = RouteVisualizer(framework_id=framework_id)

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        scores, waypoint_ids, analyses_for_radar = RouteVisualizer.from_jsonl(
            input_path, score_fields=args.dimensions, framework_id=framework_id
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
