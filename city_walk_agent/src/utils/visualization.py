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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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

    dim_keys = (
        list(dimensions)
        if dimensions
        else sorted({k for a in analyses_list for k in get(a, "scores", {}).keys()})
    )
    if not dim_keys:
        raise ValueError("No dimensions found in analyses.")

    # Build flattened data for plotting
    scores: Dict[str, List[float]] = {dim: [] for dim in dim_keys}
    waypoint_ids: List[str] = []
    numeric_waypoints: List[float] = []
    for analysis in analyses_list:
        wp = get(analysis, "waypoint_id", len(waypoint_ids))
        waypoint_ids.append(str(wp))
        try:
            numeric_waypoints.append(float(wp))
        except (TypeError, ValueError):
            numeric_waypoints.append(len(numeric_waypoints))

        for dim in dim_keys:
            val = get(analysis, "scores", {}).get(dim, 0.0)
            scores[dim].append(float(val))

    # Plot in the requested style
    fig, ax = plt.subplots(figsize=(18, 4))
    color_cycle = ["skyblue", "gold", "orange", "lightgreen", "green"]

    for idx, dim in enumerate(dim_keys):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(
            numeric_waypoints,
            scores[dim],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label=dim,
            color=color,
        )

    # Average line (black dashed)
    avg_scores = [
        float(np.mean([scores[dim][i] for dim in dim_keys])) for i in range(len(waypoint_ids))
    ]
    ax.plot(
        numeric_waypoints,
        avg_scores,
        linestyle="--",
        marker="o",
        linewidth=2,
        color="black",
        markersize=4,
        label="Average",
    )

    ax.grid(which="both", linestyle=":", linewidth=0.3, alpha=0.6)
    ax.set_title("Scores by Dimension (Average included)")
    ax.set_xlabel("Waypoint")
    ax.set_ylabel("Score")

    # y-range similar to example
    ax.set_ylim(3, 9.5)

    if numeric_waypoints:
        xmin, xmax = min(numeric_waypoints), max(numeric_waypoints)
        ax.set_xlim(xmin, xmax)
        try:
            step = max(1, int((xmax - xmin) // 10))  # keep ticks reasonable
            step = 3 if step < 3 else step
            ticks = np.arange(xmin, xmax + 1, step)
            ax.set_xticks(ticks)
        except Exception:
            pass

    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()

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
