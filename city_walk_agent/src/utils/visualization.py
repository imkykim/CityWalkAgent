"""
Visualization module for StreetAgent evaluation metrics.
Refactored for clarity and maintainability.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Fix sys.path for script execution
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    if sys.path and sys.path[0] == str(current_dir):
        sys.path.pop(0)
        sys.path.insert(0, str(current_dir.parent.parent))

from src.config import DEFAULT_FRAMEWORK_ID

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install with `pip install matplotlib`."
    ) from exc

import numpy as np


# =============================================================================
# Constants & Configuration
# =============================================================================


class PlotConfig:
    """Centralized plot configuration."""

    # Colors
    OBJECTIVE_COLOR = "#94A3B8"
    PERSONA_COLOR = "#EF4444"
    AVERAGE_COLOR = "#333333"
    MARKER_COLOR = "gray"

    # Alpha values
    LINE_ALPHA = 0.8
    FILL_ALPHA = 0.15
    MARKER_ALPHA = 0.35

    # Sizes
    FIGSIZE_STANDARD = (14, 6)
    FIGSIZE_LARGE = (16, 12)
    FIGSIZE_RADAR = (8, 8)
    DEFAULT_DPI = 100

    # Axis
    Y_LIMITS = (0, 10.5)
    RADAR_YTICKS = [2, 4, 6, 8, 10]
    MAX_XTICKS = 20

    # Line styles
    MARKER_SIZE = 3.5
    LINE_WIDTH = 1.2


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


# =============================================================================
# Utility Functions
# =============================================================================


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Universal getter for dict or object attributes."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _setup_xaxis(
    ax: plt.Axes,
    waypoint_ids: List[str],
    max_ticks: int = PlotConfig.MAX_XTICKS,
) -> None:
    """Configure x-axis with appropriate tick spacing."""
    n_points = len(waypoint_ids)
    x = np.arange(n_points)
    tick_step = max(1, n_points // max_ticks)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(waypoint_ids[::tick_step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")


def _draw_markers(
    ax: plt.Axes,
    indices: Sequence[int],
    style: str = "solid",
    color: str = PlotConfig.MARKER_COLOR,
) -> None:
    """Draw vertical marker lines at specified indices."""
    linestyle = "-" if style == "solid" else "--"
    alpha = PlotConfig.MARKER_ALPHA if style == "solid" else 0.3
    for idx in indices:
        ax.axvline(
            x=idx, color=color, linestyle=linestyle, alpha=alpha, linewidth=1, zorder=0
        )


def _resolve_indices(
    markers: Sequence[Any],
    waypoint_ids: List[str],
) -> List[int]:
    """Convert waypoint IDs or indices to numeric indices."""
    wp_to_idx = {str(wp): i for i, wp in enumerate(waypoint_ids)}
    n_points = len(waypoint_ids)
    indices = []

    for marker in markers:
        idx = wp_to_idx.get(str(marker))
        if idx is not None:
            indices.append(idx)
        else:
            try:
                numeric = int(marker)
                if 0 <= numeric < n_points:
                    indices.append(numeric)
            except (TypeError, ValueError):
                continue

    return sorted(set(indices))


def _get_scores_from_result(
    result: Mapping,
    dim_keys: List[str],
    mode: str = "persona",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract objective and persona scores from a result record.

    Args:
        result: Single waypoint result
        dim_keys: Dimension keys to extract
        mode: "persona" or "system" for field name selection

    Returns:
        Tuple of (objective_scores, persona_scores) dicts
    """
    if mode == "system":
        obj_source = _get(result, "system1_scores") or {}
        per_source = (
            _get(result, "system2_scores") or _get(result, "scores") or obj_source
        )
    else:
        obj_source = (
            _get(result, "objective_scores")
            or _get(result, "neutral_scores")
            or _get(result, "system1_scores")
            or {}
        )
        per_source = (
            _get(result, "persona_scores") or _get(result, "scores") or obj_source
        )

    obj_scores = {dim: _coerce_float(obj_source.get(dim, 0)) for dim in dim_keys}
    per_scores = {dim: _coerce_float(per_source.get(dim, 0)) for dim in dim_keys}

    return obj_scores, per_scores


def _extract_all_scores(
    waypoint_results: Sequence[Mapping],
    dim_keys: List[str],
    mode: str = "persona",
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[str]]:
    """
    Extract scores from all waypoint results.

    Returns:
        Tuple of (objective_scores, persona_scores, waypoint_ids)
    """
    objective: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
    persona: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
    waypoint_ids: List[str] = []

    for idx, result in enumerate(waypoint_results):
        waypoint_ids.append(str(_get(result, "waypoint_id", idx)))
        obj_scores, per_scores = _get_scores_from_result(result, dim_keys, mode)

        for dim in dim_keys:
            objective[dim].append(obj_scores[dim])
            persona[dim].append(per_scores[dim])

    return objective, persona, waypoint_ids


def _apply_plot_style() -> None:
    """Apply default matplotlib style settings."""
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


# =============================================================================
# Data Loading
# =============================================================================


def load_analyses(path: Path) -> List[Mapping]:
    """Load analysis results from JSON, CSV, or JSONL file."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        raise ValueError("Expected JSON array")

    elif suffix == ".jsonl":
        analyses = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    analyses.append(json.loads(line))
        return analyses

    elif suffix == ".csv":
        by_waypoint: Dict[str, Dict] = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wp = row.get("image_id") or row.get("waypoint_id")
                dim = row.get("dimension_id") or row.get("dimension_name")
                score = row.get("score")
                if not all([wp, dim, score]):
                    continue
                record = by_waypoint.setdefault(wp, {"waypoint_id": wp, "scores": {}})
                try:
                    record["scores"][dim] = float(score)
                except ValueError:
                    continue

        return sorted(by_waypoint.values(), key=lambda x: x["waypoint_id"])

    raise ValueError(f"Unsupported file format: {suffix}")


# =============================================================================
# RouteVisualizer Class
# =============================================================================


class RouteVisualizer:
    """Visualizes evaluation scores along walking routes."""

    def __init__(
        self,
        figsize: Tuple[int, int] = PlotConfig.FIGSIZE_STANDARD,
        dpi: int = PlotConfig.DEFAULT_DPI,
        framework_id: str = DEFAULT_FRAMEWORK_ID,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.framework_id = framework_id

        _apply_plot_style()

        # Load framework dimensions
        from src.config import load_framework

        self.framework = load_framework(framework_id)
        self.dim_config = self._build_dim_config_from_framework()

    def _build_dim_config_from_framework(self) -> Dict[str, Dict]:
        """Build dimension config from framework."""
        config = {}
        palette = plt.cm.tab10.colors + plt.cm.Set3.colors

        for idx, dim in enumerate(self.framework.get("dimensions", [])):
            dim_id = dim["id"]
            config[dim_id] = {
                "color": DEFAULT_SCORE_COLORS.get(dim_id, palette[idx % len(palette)]),
                "label": dim["name_en"],
            }
        return config

    def _get_dim_config(self, dim_keys: Sequence[str]) -> Dict[str, Dict]:
        """Get dimension configuration with fallbacks."""
        palette = plt.cm.tab10.colors + plt.cm.Set3.colors
        config = {}

        for idx, dim in enumerate(dim_keys):
            if dim in self.dim_config:
                config[dim] = self.dim_config[dim].copy()
            else:
                config[dim] = {
                    "color": DEFAULT_SCORE_COLORS.get(dim, palette[idx % len(palette)]),
                    "label": dim,
                }
        return config

    def _infer_dimensions(
        self,
        dimensions: Optional[Sequence[str]] = None,
        scores: Optional[Mapping[str, Sequence[float]]] = None,
        waypoint_results: Optional[Sequence[Mapping]] = None,
    ) -> List[str]:
        """Infer dimension keys from available data."""
        if dimensions:
            return list(dimensions)

        if scores:
            return list(scores.keys())

        if waypoint_results:
            for record in waypoint_results:
                for key in (
                    "system2_scores",
                    "system1_scores",
                    "scores",
                    "objective_scores",
                    "persona_scores",
                ):
                    source = _get(record, key)
                    if source:
                        return list(source.keys())

        return list(self.dim_config.keys())

    # -------------------------------------------------------------------------
    # Core Plot Methods
    # -------------------------------------------------------------------------

    def plot_scores(
        self,
        scores: Dict[str, List[float]],
        waypoint_ids: Optional[List[str]] = None,
        title: str = "Scores by Dimension",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        markers: Optional[Sequence[Any]] = None,
        marker_style: str = "solid",
        show_average: bool = True,
    ) -> plt.Figure:
        """
        Plot dimension scores with optional markers and average line.

        Args:
            scores: Dict mapping dimension names to score lists
            waypoint_ids: Optional waypoint identifiers
            title: Plot title
            save_path: Path to save figure
            dimensions: Dimension keys to plot
            markers: Waypoint IDs/indices to highlight with vertical lines
            marker_style: "solid" or "dashed" for marker lines
            show_average: Whether to show overall average line
        """
        dim_keys = list(dimensions) if dimensions else list(scores.keys())
        dim_config = self._get_dim_config(dim_keys)

        n_points = len(next(iter(scores.values())))
        if waypoint_ids is None:
            waypoint_ids = [str(i) for i in range(n_points)]
        x = np.arange(n_points)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        all_scores = []

        # Plot each dimension
        for dim in dim_keys:
            if dim not in scores:
                continue
            cfg = dim_config[dim]
            ax.plot(
                x,
                scores[dim],
                marker="o",
                markersize=PlotConfig.MARKER_SIZE,
                linewidth=PlotConfig.LINE_WIDTH,
                color=cfg["color"],
                alpha=PlotConfig.LINE_ALPHA,
                label=cfg["label"],
            )
            all_scores.append(scores[dim])

        # Overall average
        if show_average and all_scores:
            avg = np.mean(all_scores, axis=0)
            ax.plot(
                x,
                avg,
                linestyle="--",
                linewidth=1.8,
                color=PlotConfig.AVERAGE_COLOR,
                alpha=0.9,
                label="Overall Average",
                zorder=10,
            )

        # Markers
        if markers:
            indices = _resolve_indices(markers, waypoint_ids)
            _draw_markers(ax, indices, style=marker_style)

        # Formatting
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(*PlotConfig.Y_LIMITS)
        ax.legend(loc="best", framealpha=0.9, fontsize=10)
        _setup_xaxis(ax, waypoint_ids)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Score Heatmap",
        save_path: Optional[Path] = None,
        cmap: str = "RdYlGn",
        vmin: float = 0,
        vmax: float = 10,
        vcenter: float = 6.0,
        colorbar_label: str = "Score",
    ) -> plt.Figure:
        """
        Create a heatmap visualization with diverging colormap.

        Args:
            data: 2D numpy array of values
            row_labels: Labels for y-axis
            col_labels: Labels for x-axis (waypoints)
            title: Plot title
            save_path: Path to save figure
            cmap: Colormap name
            vmin, vmax: Color scale limits
            vcenter: Center value for diverging colormap (default 6.0 for yellow at 5-7 range)
            colorbar_label: Label for colorbar
        """
        from matplotlib.colors import TwoSlopeNorm

        fig, ax = plt.subplots(figsize=(14, 4), dpi=self.dpi)

        # Use diverging normalization to center yellow around vcenter
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        n_cols = len(col_labels)
        tick_step = max(1, n_cols // PlotConfig.MAX_XTICKS)
        ax.set_xticks(np.arange(0, n_cols, tick_step))
        ax.set_xticklabels(col_labels[::tick_step], rotation=45, ha="right", fontsize=8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, rotation=270, labelpad=20)

        ax.set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_radar(
        self,
        scores: Dict[str, float],
        title: str = "Radar Chart",
        save_path: Optional[Path] = None,
        color: str = "#2E86AB",
        figsize: Tuple[int, int] = PlotConfig.FIGSIZE_RADAR,
    ) -> plt.Figure:
        """
        Create a radar/spider chart.

        Args:
            scores: Dict mapping dimension names to single score values
            title: Plot title
            save_path: Path to save figure
            color: Fill color
            figsize: Figure size
        """
        labels = list(scores.keys())
        values = list(scores.values())
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = values + values[:1]
        angles = angles + angles[:1]

        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi
        )

        # Plot with enhanced styling
        ax.plot(angles, values, "o-", linewidth=3, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.3, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=20, fontweight="bold")
        ax.set_ylim(0, 10)
        ax.set_yticks(PlotConfig.RADAR_YTICKS)
        ax.set_yticklabels([])  # Hide radial tick labels
        ax.grid(True, linestyle="-", linewidth=1.0, alpha=0.35, color="#999999")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=25)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_radar_comparison(
        self,
        objective_scores: Dict[str, float],
        persona_scores: Dict[str, float],
        title: str = "Radar Chart Comparison",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = PlotConfig.FIGSIZE_RADAR,
        mode: str = "persona",
    ) -> plt.Figure:
        """
        Create a radar/spider chart comparing objective vs persona scores.

        Args:
            objective_scores: Dict mapping dimension names to objective score values
            persona_scores: Dict mapping dimension names to persona score values
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            mode: "persona" or "system" for labels
        """
        labels = list(objective_scores.keys())
        obj_values = list(objective_scores.values())
        per_values = list(persona_scores.values())
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        obj_values = obj_values + obj_values[:1]
        per_values = per_values + per_values[:1]
        angles = angles + angles[:1]

        label_obj = "System 1" if mode == "system" else "Objective"
        label_per = "System 2" if mode == "system" else "Persona"

        # Use clearer, more contrasting colors
        color_obj = "#3B82F6"  # Bright blue
        color_per = "#F97316"  # Bright orange

        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection="polar"), dpi=self.dpi
        )

        # Plot objective scores with lower opacity for base layer
        ax.plot(
            angles,
            obj_values,
            "s-",
            # "o--",
            linewidth=3,
            color=color_obj,
            label=label_obj,
            markersize=8,
            alpha=0.9,
        )
        ax.fill(angles, obj_values, alpha=0.2, color=color_obj)

        # Plot persona scores on top with higher opacity and solid line
        ax.plot(
            angles,
            per_values,
            "s-",
            linewidth=3,
            color=color_per,
            label=label_per,
            markersize=7,
            alpha=0.99,
        )
        ax.fill(angles, per_values, alpha=0.2, color=color_per)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=20, fontweight="bold")
        ax.set_ylim(0, 10)
        ax.set_yticks(PlotConfig.RADAR_YTICKS)
        ax.set_yticklabels([])  # Hide radial tick labels
        ax.grid(True, linestyle="-", linewidth=1.0, alpha=0.35, color="#999999")
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.12),
            fontsize=14,
            frameon=True,
            framealpha=0.95,
            edgecolor="#CCCCCC",
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=25)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_comparison(
        self,
        waypoint_results: Sequence[Mapping],
        title: str = "Objective vs Persona-Aware",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        mode: str = "persona",
    ) -> plt.Figure:
        """
        Plot comparison between two evaluation modes (objective vs persona/system1 vs system2).

        Args:
            waypoint_results: List of waypoint result dicts
            title: Plot title
            save_path: Path to save figure
            dimensions: Dimension keys to plot
            mode: "persona" or "system" for field selection
        """
        dim_keys = self._infer_dimensions(dimensions, waypoint_results=waypoint_results)
        dim_config = self._get_dim_config(dim_keys)

        objective, persona, waypoint_ids = _extract_all_scores(
            waypoint_results, dim_keys, mode
        )

        # Determine triggered waypoints
        triggered = [
            i
            for i, r in enumerate(waypoint_results)
            if _get(r, "system2_triggered", False)
        ]

        n_dims = min(len(dim_keys), 6)
        rows = (n_dims + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows), dpi=self.dpi)
        axes = axes.flatten() if n_dims > 1 else [axes]
        x = np.arange(len(waypoint_ids))

        label_obj = "System 1" if mode == "system" else "Objective"
        label_per = "System 2" if mode == "system" else "Persona-Aware"

        for idx, dim in enumerate(dim_keys[:n_dims]):
            ax = axes[idx]
            cfg = dim_config[dim]
            obj_data = np.array(objective[dim])
            per_data = np.array(persona[dim])

            ax.plot(
                x,
                obj_data,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=PlotConfig.OBJECTIVE_COLOR,
                alpha=0.9,
                label=label_obj,
                linestyle="--" if mode == "system" else "-",
            )
            ax.plot(
                x,
                per_data,
                marker="x" if mode == "persona" else "s",
                markersize=5 if mode == "persona" else 4,
                linewidth=1.5,
                color=PlotConfig.PERSONA_COLOR,
                alpha=0.9,
                label=label_per,
            )
            ax.fill_between(
                x,
                obj_data,
                per_data,
                color=PlotConfig.PERSONA_COLOR,
                alpha=PlotConfig.FILL_ALPHA,
            )

            # Draw triggered markers
            if triggered:
                _draw_markers(ax, triggered, style="dashed")

            ax.set_title(f"{cfg['label']}", fontsize=16, fontweight="bold")
            ax.set_xlabel("Waypoint ID", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_ylim(*PlotConfig.Y_LIMITS)
            ax.legend(loc="upper right", fontsize=11)

            # Stats box
            delta = np.mean(per_data) - np.mean(obj_data)
            stats = f"Δ: {delta:+.2f}\n{label_obj}: {np.mean(obj_data):.2f}\n{label_per}: {np.mean(per_data):.2f}"
            ax.text(
                0.02,
                0.98,
                stats,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
            )

            _setup_xaxis(ax, waypoint_ids, max_ticks=10)

        # Hide unused subplots
        for idx in range(n_dims, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_route_scatter(
        self,
        waypoint_results: Sequence[Mapping],
        title: str = "Route Scatter Plot with Average Score",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "RdYlGn",
        point_size: int = 75,
        score_type: str = "persona",
        mode: str = "persona",
    ) -> plt.Figure:
        """
        Create GPS scatter plot colored by average score.

        Uses fixed color scale (1-10) for cross-region comparison.
        Default colormap: RdYlGn (Red-Yellow-Green) for intuitive quality visualization.

        Args:
            waypoint_results: List of waypoint results with GPS coordinates
            title: Plot title
            save_path: Path to save figure
            dimensions: Dimension keys for averaging (None = all)
            figsize: Figure size
            cmap: Colormap name (default: RdYlGn - red=low, green=high)
            point_size: Scatter point size
            score_type: "objective" or "persona" - which scores to use
            mode: "persona" or "system" - for field name selection

        Returns:
            Matplotlib figure
        """
        dim_keys = self._infer_dimensions(dimensions, waypoint_results=waypoint_results)

        lats, lons, avg_scores = [], [], []

        for result in waypoint_results:
            # Extract GPS coordinates
            gps = _get(result, "gps") or _get(result, "coordinates")
            lat = None
            lon = None

            # Handle GPS as array [lat, lon]
            if isinstance(gps, (list, tuple)) and len(gps) >= 2:
                lat = gps[0]
                lon = gps[1]
            # Handle GPS as dict
            elif isinstance(gps, dict):
                lat = _get(gps, "lat") or _get(gps, "latitude")
                lon = _get(gps, "lon") or _get(gps, "lng") or _get(gps, "longitude")
            # Fallback to direct fields
            else:
                lat = _get(result, "lat") or _get(result, "latitude")
                lon = _get(result, "lon") or _get(result, "longitude")

            if lat is None or lon is None:
                continue

            # Calculate average score based on score_type
            obj_scores, per_scores = _get_scores_from_result(result, dim_keys, mode)
            scores = obj_scores if score_type == "objective" else per_scores
            avg = np.mean([scores[d] for d in dim_keys if d in scores])

            lats.append(float(lat))
            lons.append(float(lon))
            avg_scores.append(avg)

        if not lats:
            raise ValueError("No valid GPS coordinates found in waypoint results")

        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        # Fixed color scale 1-10 for cross-region comparison
        scatter = ax.scatter(
            lons,
            lats,
            c=avg_scores,
            cmap=cmap,
            s=point_size,
            vmin=1,  # Fixed minimum
            vmax=10,  # Fixed maximum
            alpha=1.0,
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

        # Enhanced colorbar with fixed scale
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.08)
        cbar.set_label("Average Score", fontsize=13, fontweight="bold", labelpad=15)
        cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cbar.ax.tick_params(labelsize=11)

        # Formatting
        ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
        # Keep figure size fixed - adjust data limits to maintain aspect ratio
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="x", rotation=45)

        # Add grid for better spatial reference
        ax.grid(
            True, alpha=0.4, linestyle=":", linewidth=0.8, color="#CCCCCC", zorder=0
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_timeline(
        self,
        scores: Dict[str, List[float]],
        chapters: Optional[List[Dict[str, Any]]] = None,
        title: str = "Score Timeline",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> plt.Figure:
        """
        Visualize scores with optional narrative chapters.

        Args:
            scores: Dict mapping dimensions to score lists
            chapters: Optional narrative chapter data
            title: Plot title
            save_path: Path to save figure
            dimensions: Dimension keys to plot
        """
        dim_keys = list(dimensions) if dimensions else list(scores.keys())
        dim_config = self._get_dim_config(dim_keys)

        has_chapters = bool(chapters)
        height_ratios = [3, 2] if has_chapters else [1]
        n_rows = 2 if has_chapters else 1

        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(16, 10 if has_chapters else 6),
            gridspec_kw={"height_ratios": height_ratios},
            dpi=self.dpi,
        )

        ax_scores = axes[0] if has_chapters else axes
        n_points = len(next(iter(scores.values())))
        x = np.arange(n_points)

        # Plot scores
        for dim in dim_keys:
            if dim not in scores:
                continue
            cfg = dim_config[dim]
            ax_scores.plot(
                x,
                scores[dim],
                marker="o",
                markersize=PlotConfig.MARKER_SIZE,
                label=cfg["label"],
                linewidth=PlotConfig.LINE_WIDTH,
                color=cfg["color"],
                alpha=PlotConfig.LINE_ALPHA,
            )

        # Chapter markers on score plot
        if chapters:
            for ch in chapters:
                wid = ch.get("waypoint_id", 0)
                ax_scores.axvline(x=wid, color="red", linestyle="--", alpha=0.3)
                ax_scores.text(
                    wid,
                    9.5,
                    f"Ch{ch.get('chapter_number', '?')}",
                    fontsize=8,
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        ax_scores.set_title(title, fontsize=13, fontweight="bold")
        ax_scores.set_ylabel("Score", fontsize=11)
        ax_scores.legend(loc="best")
        ax_scores.set_ylim(*PlotConfig.Y_LIMITS)

        # Narrative panel
        if has_chapters:
            ax_narrative = axes[1]
            ax_narrative.set_xlim(0, n_points)
            ax_narrative.axis("off")

            if chapters:
                ax_narrative.set_ylim(0, len(chapters) + 1)
                tone_colors = {
                    "optimistic": "#10B981",
                    "neutral": "#6B7280",
                    "cautious": "#F59E0B",
                    "concerned": "#EF4444",
                }
                for i, ch in enumerate(chapters):
                    y_pos = len(chapters) - i
                    x_pos = ch.get("waypoint_id", 0)
                    ax_narrative.plot([x_pos, x_pos], [0, y_pos], "k-", alpha=0.3)

                    text = f"Ch{ch.get('chapter_number', '?')}: {ch.get('key_observation', '')[:60]}..."
                    color = tone_colors.get(
                        ch.get("emotional_tone", "neutral"), "#6B7280"
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
                ax_narrative.text(
                    0.5,
                    0.5,
                    "No narrative chapters",
                    transform=ax_narrative.transAxes,
                    fontsize=12,
                    ha="center",
                    va="center",
                    style="italic",
                    color="gray",
                )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_overview(
        self,
        waypoint_results: Sequence[Mapping],
        title: str = "Evaluation Overview",
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        mode: str = "persona",
    ) -> plt.Figure:
        """
        Create overview with average line plot and delta heatmap side by side.

        Args:
            waypoint_results: List of waypoint result dicts
            title: Plot title
            save_path: Path to save figure
            dimensions: Dimension keys to plot
            mode: "persona" or "system"
        """
        dim_keys = self._infer_dimensions(dimensions, waypoint_results=waypoint_results)
        objective, persona, waypoint_ids = _extract_all_scores(
            waypoint_results, dim_keys, mode
        )

        n_points = len(waypoint_ids)
        avg_obj = [
            np.mean([objective[d][i] for d in dim_keys]) for i in range(n_points)
        ]
        avg_per = [np.mean([persona[d][i] for d in dim_keys]) for i in range(n_points)]

        delta_matrix = np.array(
            [
                [persona[d][i] - objective[d][i] for i in range(n_points)]
                for d in dim_keys
            ]
        )

        fig, axes = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.25, 1]}
        )
        x = np.arange(n_points)

        label_obj = "System 1" if mode == "system" else "Objective"
        label_per = "System 2" if mode == "system" else "Persona-Aware"

        # Line plot
        axes[0].plot(
            x,
            avg_obj,
            marker="o",
            markersize=PlotConfig.MARKER_SIZE,
            linewidth=1.4,
            color=PlotConfig.OBJECTIVE_COLOR,
            alpha=0.9,
            label=label_obj,
            linestyle="--",
        )
        axes[0].plot(
            x,
            avg_per,
            marker="s",
            markersize=4,
            linewidth=1.6,
            color=PlotConfig.PERSONA_COLOR,
            alpha=0.95,
            label=label_per,
        )
        axes[0].fill_between(
            x, avg_obj, avg_per, color=PlotConfig.PERSONA_COLOR, alpha=0.12
        )

        axes[0].set_ylabel("Average Score", fontsize=11, fontweight="bold")
        axes[0].set_ylim(*PlotConfig.Y_LIMITS)
        axes[0].set_title(f"{title} – Averages", fontsize=13, fontweight="bold", pad=12)
        axes[0].legend(loc="best", fontsize=10)
        _setup_xaxis(axes[0], waypoint_ids, max_ticks=12)

        # Heatmap
        max_abs = max(float(np.max(np.abs(delta_matrix))), 0.5)
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

        tick_step = max(1, n_points // 12)
        axes[1].set_xticks(np.arange(0, n_points, tick_step))
        axes[1].set_xticklabels(
            waypoint_ids[::tick_step], rotation=45, ha="right", fontsize=8
        )
        axes[1].set_xlabel("Waypoint ID", fontsize=11, fontweight="bold")
        axes[1].set_title(f"{title} – Delta", fontsize=13, fontweight="bold", pad=12)

        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Adjustment", rotation=270, labelpad=18)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_summary_radar(
        self,
        waypoint_results: Sequence[Mapping],
        save_path: Optional[Path] = None,
        dimensions: Optional[Sequence[str]] = None,
        mode: str = "persona",
    ) -> plt.Figure:
        """
        Create radar chart comparing average objective vs persona scores.

        Args:
            waypoint_results: List of waypoint result dicts
            save_path: Path to save figure
            dimensions: Dimension keys to plot
            mode: "persona" or "system"
        """
        dim_keys = self._infer_dimensions(dimensions, waypoint_results=waypoint_results)
        dim_config = self._get_dim_config(dim_keys)
        objective, persona, _ = _extract_all_scores(waypoint_results, dim_keys, mode)

        obj_avgs = [np.mean(objective[d]) for d in dim_keys]
        per_avgs = [np.mean(persona[d]) for d in dim_keys]

        labels = [dim_config[d]["label"] for d in dim_keys]
        angles = np.linspace(0, 2 * np.pi, len(dim_keys), endpoint=False).tolist()

        obj_avgs = obj_avgs + obj_avgs[:1]
        per_avgs = per_avgs + per_avgs[:1]
        angles = angles + angles[:1]

        label_obj = "System 1" if mode == "system" else "Objective"
        label_per = "System 2" if mode == "system" else "Persona-Aware"

        # Use clearer, more contrasting colors
        color_obj = "#3B82F6"  # Bright blue
        color_per = "#F97316"  # Bright orange

        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(projection="polar"), dpi=self.dpi
        )

        # Plot objective scores with lower opacity for base layer
        ax.plot(
            angles,
            obj_avgs,
            "o--",
            linewidth=3,
            color=color_obj,
            label=label_obj,
            markersize=8,
            alpha=0.85,
        )
        ax.fill(angles, obj_avgs, alpha=0.12, color=color_obj)

        # Plot persona scores on top with higher opacity
        ax.plot(
            angles,
            per_avgs,
            "s-",
            linewidth=3.5,
            color=color_per,
            label=label_per,
            markersize=7,
            alpha=0.95,
        )
        ax.fill(angles, per_avgs, alpha=0.18, color=color_per)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=20, fontweight="bold")
        ax.set_ylim(0, 10)
        ax.set_yticks(PlotConfig.RADAR_YTICKS)
        ax.set_yticklabels([])  # Hide radial tick labels
        ax.grid(True, linestyle="-", linewidth=1.0, alpha=0.35, color="#999999")
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.12),
            fontsize=14,
            frameon=True,
            framealpha=0.95,
            edgecolor="#CCCCCC",
        )
        ax.set_title("Average Score Comparison", fontsize=16, fontweight="bold", pad=25)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # -------------------------------------------------------------------------
    # Batch Generation
    # -------------------------------------------------------------------------

    def generate_report(
        self,
        waypoint_results: Sequence[Mapping],
        output_dir: Path,
        chapters: Optional[List[Dict]] = None,
        mode: str = "persona",
        include_radar: bool = False,
    ) -> Dict[str, Path]:
        """
        Generate complete visualization report.

        Args:
            waypoint_results: List of waypoint result dicts
            output_dir: Directory to save visualizations
            chapters: Optional narrative chapters
            mode: "persona" or "system"
            include_radar: Whether to generate per-waypoint radar charts

        Returns:
            Dict mapping output names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        dim_keys = self._infer_dimensions(waypoint_results=waypoint_results)
        objective, persona, waypoint_ids = _extract_all_scores(
            waypoint_results, dim_keys, mode
        )

        triggered = [
            waypoint_ids[i]
            for i, r in enumerate(waypoint_results)
            if _get(r, "system2_triggered", False)
        ]

        # 1. Objective scores
        obj_path = output_dir / "scores_objective.png"
        self.plot_scores(
            objective,
            waypoint_ids,
            "Objective Scores",
            save_path=obj_path,
            dimensions=dim_keys,
            markers=triggered,
        )
        paths["objective_scores"] = obj_path

        # 2. Persona/Final scores
        per_path = output_dir / "scores_final.png"
        self.plot_scores(
            persona,
            waypoint_ids,
            "Final Scores",
            save_path=per_path,
            dimensions=dim_keys,
            markers=triggered,
        )
        paths["final_scores"] = per_path

        # 3. Comparison
        comp_path = output_dir / "comparison.png"
        self.plot_comparison(
            waypoint_results, save_path=comp_path, dimensions=dim_keys, mode=mode
        )
        paths["comparison"] = comp_path

        # 4. Overview
        over_path = output_dir / "overview.png"
        self.plot_overview(
            waypoint_results, save_path=over_path, dimensions=dim_keys, mode=mode
        )
        paths["overview"] = over_path

        # 5. Summary radar
        radar_path = output_dir / "summary_radar.png"
        self.plot_summary_radar(
            waypoint_results, save_path=radar_path, dimensions=dim_keys, mode=mode
        )
        paths["summary_radar"] = radar_path

        # 8. Route scatter (GPS map) - Generate both objective and persona versions
        try:
            # Objective scores version
            scatter_obj_path = output_dir / "route_scatter_objective.png"
            self.plot_route_scatter(
                waypoint_results,
                title="Route Map - Objective Scores",
                save_path=scatter_obj_path,
                dimensions=dim_keys,
                score_type="objective",
                mode=mode,
            )
            paths["route_scatter_objective"] = scatter_obj_path

            # Persona scores version
            scatter_per_path = output_dir / "route_scatter_persona.png"
            self.plot_route_scatter(
                waypoint_results,
                title="Route Map - Persona Scores",
                save_path=scatter_per_path,
                dimensions=dim_keys,
                score_type="persona",
                mode=mode,
            )
            paths["route_scatter_persona"] = scatter_per_path
        except ValueError as e:
            # No GPS data available - skip scatter plot
            print(f"Skipping route scatter: {e}")
        except Exception as e:
            # Other errors - log but continue
            print(f"Warning: Could not generate route scatter plot: {e}")

        # 6. Timeline with chapters
        if chapters:
            timeline_path = output_dir / "timeline.png"
            self.plot_timeline(
                persona,
                chapters,
                "Narrative Timeline",
                save_path=timeline_path,
                dimensions=dim_keys,
            )
            paths["timeline"] = timeline_path

        # 7. Heatmap
        heatmap_data = np.array([persona[d] for d in dim_keys])
        heatmap_path = output_dir / "heatmap.png"
        dim_labels = [self._get_dim_config(dim_keys)[d]["label"] for d in dim_keys]
        self.plot_heatmap(
            heatmap_data,
            dim_labels,
            waypoint_ids,
            "Score Heatmap",
            save_path=heatmap_path,
        )
        paths["heatmap"] = heatmap_path

        # 8. Per-waypoint radar charts (optional)
        if include_radar:
            radar_base_dir = output_dir / "radar_charts"
            radar_base_dir.mkdir(exist_ok=True)

            # Create subdirectories for objective and comparison charts
            radar_obj_dir = radar_base_dir / "objective"
            radar_comp_dir = radar_base_dir / "comparison"
            radar_obj_dir.mkdir(exist_ok=True)
            radar_comp_dir.mkdir(exist_ok=True)

            for idx, result in enumerate(waypoint_results):
                wp_id = waypoint_ids[idx]
                obj_scores, per_scores = _get_scores_from_result(result, dim_keys, mode)

                # Create labels for radar charts
                obj_labels = {
                    self._get_dim_config(dim_keys)[d]["label"]: obj_scores[d]
                    for d in dim_keys
                }
                per_labels = {
                    self._get_dim_config(dim_keys)[d]["label"]: per_scores[d]
                    for d in dim_keys
                }

                # Generate objective-only radar chart with bright blue color
                obj_radar_path = radar_obj_dir / f"radar_wp_{wp_id}.png"
                fig_obj = self.plot_radar(
                    obj_labels,
                    f"Waypoint {wp_id} - Objective",
                    save_path=obj_radar_path,
                    color="#3B82F6",  # Bright blue
                )
                plt.close(fig_obj)

                # Generate comparison radar chart
                comp_radar_path = radar_comp_dir / f"radar_wp_{wp_id}.png"
                fig_comp = self.plot_radar_comparison(
                    obj_labels,
                    per_labels,
                    f"Waypoint {wp_id} - Comparison",
                    save_path=comp_radar_path,
                    mode=mode,
                )
                plt.close(fig_comp)

            paths["radar_objective_dir"] = radar_obj_dir
            paths["radar_comparison_dir"] = radar_comp_dir

        return paths


# =============================================================================
# Convenience Functions
# =============================================================================


def plot_analysis_results(
    analyses: Iterable[Mapping],
    output_path: Path,
    dimensions: Optional[Sequence[str]] = None,
    framework_id: str = DEFAULT_FRAMEWORK_ID,
) -> Path:
    """
    Quick plot of analysis results.

    Args:
        analyses: Iterable of analysis records
        output_path: Path to save PNG
        dimensions: Optional dimension keys
        framework_id: Framework ID for config
    """
    analyses_list = list(analyses)
    if not analyses_list:
        raise ValueError("No analyses provided")

    viz = RouteVisualizer(framework_id=framework_id)
    dim_keys = viz._infer_dimensions(dimensions, waypoint_results=analyses_list)

    # Extract data
    scores: Dict[str, List[float]] = {d: [] for d in dim_keys}
    waypoint_ids: List[str] = []
    visual_changes: List[str] = []

    for idx, a in enumerate(analyses_list):
        wp_id = str(_get(a, "waypoint_id", idx))
        waypoint_ids.append(wp_id)

        score_source = (
            _get(a, "scores")
            or _get(a, "system2_scores")
            or _get(a, "system1_scores")
            or {}
        )
        for d in dim_keys:
            scores[d].append(_coerce_float(score_source.get(d, 0)))

        if _get(a, "visual_change_detected") or _get(a, "visual_change"):
            visual_changes.append(wp_id)

    viz.plot_scores(
        scores,
        waypoint_ids,
        title="Scores by Dimension",
        save_path=output_path,
        dimensions=dim_keys,
        markers=visual_changes,
    )
    plt.close()
    return output_path


def regenerate_visualizations(
    output_dir: Path,
    framework_id: str = DEFAULT_FRAMEWORK_ID,
    mode: str = "persona",
    include_radar: bool = False,
) -> Dict[str, Path]:
    """
    Regenerate all visualizations from existing analysis_results.json.

    Args:
        output_dir: Directory containing analysis_results.json
        framework_id: Framework ID
        mode: "persona" or "system"
        include_radar: Generate per-waypoint radar charts

    Returns:
        Dict of generated file paths
    """
    output_dir = Path(output_dir)
    analysis_path = output_dir / "analysis_results.json"

    if not analysis_path.exists():
        raise FileNotFoundError(f"analysis_results.json not found in {output_dir}")

    analyses = load_analyses(analysis_path)

    # Load chapters if available
    chapters_path = output_dir / "narrative_chapters.json"
    chapters = None
    if chapters_path.exists():
        chapters = json.loads(chapters_path.read_text())

    viz_dir = output_dir / "visualizations"
    viz = RouteVisualizer(framework_id=framework_id)

    return viz.generate_report(
        analyses,
        output_dir=viz_dir,
        chapters=chapters,
        mode=mode,
        include_radar=include_radar,
    )


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entrypoint for visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visualizations from analysis results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="Analysis file (JSON/CSV/JSONL)"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output path for single plot")
    parser.add_argument(
        "-f", "--framework", default=DEFAULT_FRAMEWORK_ID, help="Framework ID"
    )
    parser.add_argument("-d", "--dimensions", nargs="+", help="Dimension keys to plot")
    parser.add_argument(
        "--regenerate",
        type=Path,
        metavar="DIR",
        help="Regenerate all visualizations from output directory",
    )
    parser.add_argument(
        "--mode",
        choices=["persona", "system"],
        default="persona",
        help="Comparison mode",
    )
    parser.add_argument("--radar", action="store_true", help="Generate radar charts")

    args = parser.parse_args()

    if args.regenerate:
        paths = regenerate_visualizations(
            args.regenerate,
            framework_id=args.framework,
            mode=args.mode,
            include_radar=args.radar,
        )
        print("Generated visualizations:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return

    if not args.input:
        parser.error("Provide input file or use --regenerate")

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    output_path = args.output or args.input.with_suffix(".png")

    analyses = load_analyses(args.input)
    plot_analysis_results(
        analyses, output_path, dimensions=args.dimensions, framework_id=args.framework
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
