#!/usr/bin/env python3
"""
ablation_sequential_vs_aggregate.py — Three-way ablation comparison

Compares three conditions using paired data from the same route images:
  (A) Full system — sequential scores from walk_log.json (STM + LTM + context)
  (B) Independent — context-free VLM scores from ablation_independent/independent_scores.json
  (C) Aggregate   — route-level mean of (B), simulating Walk Score

Usage:
    python scripts/ablation_sequential_vs_aggregate.py \\
        --input-dir outputs/test_runs/20260427_120000/

    # Multiple experiment dirs (will be pooled):
    python scripts/ablation_sequential_vs_aggregate.py \\
        --input-dir outputs/test_runs/20260427_120000/ \\
        --barrier-threshold 5.0 \\
        --recommend-threshold 6.0

Output (written to <input_dir>/ablation_analysis/ or per-experiment):
    ablation_three_way_detail.csv
    ablation_summary.json
    context_effect_scatter.png
    three_way_profile.png
    hidden_barrier_comparison.png
    transition_amplification.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_BARRIER_THRESHOLDS = [4.0, 5.0, 6.0]
DEFAULT_RECOMMEND_THRESHOLD = 6.0
TRANSITION_DELTA = 1.5
DIVERGENCE_THRESHOLD = 1.5

CONDITION_COLORS = {
    "A": "#3b82f6",   # blue — full system
    "B": "#f97316",   # orange — independent
    "C": "#22c55e",   # green — aggregate
}
CONDITION_LABELS = {
    "A": "Full System (A)",
    "B": "Independent (B)",
    "C": "Aggregate of B (C)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _is_experiment_dir(path: Path) -> bool:
    return (path / "walk_log.json").exists()


def _find_experiment_dirs(input_dir: Path) -> List[Path]:
    if _is_experiment_dir(input_dir):
        return [input_dir]
    return [d for d in sorted(input_dir.iterdir()) if d.is_dir() and _is_experiment_dir(d)]


def _load_walk_log(exp_dir: Path) -> Optional[List[Dict[str, Any]]]:
    p = exp_dir / "walk_log.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("route_taken", data.get("steps", []))
    return None


def _load_independent_scores(exp_dir: Path, suffix: str = "independent") -> Optional[Dict[str, Any]]:
    p = exp_dir / f"ablation_{suffix}" / "independent_scores.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_a_scores(walk_log: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Extract step → scores from walk_log (condition A: full system)."""
    result: Dict[int, Dict[str, float]] = {}
    for step in walk_log:
        if not step.get("analyzed", False):
            continue
        scores = step.get("scores", {})
        if not scores:
            continue
        step_idx = step.get("step", step.get("waypoint_id", 0))
        result[step_idx] = {k: float(v) for k, v in scores.items() if v is not None}
    return result


def _extract_b_scores(ind_data: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """Extract step → scores from independent_scores.json (condition B)."""
    result: Dict[int, Dict[str, float]] = {}
    for rec in ind_data.get("scores", []):
        step_idx = rec.get("step", 0)
        scores = rec.get("scores", {})
        if scores:
            result[step_idx] = {k: float(v) for k, v in scores.items()}
    return result


def _get_dimensions(
    a_scores: Dict[int, Dict[str, float]],
    b_scores: Dict[int, Dict[str, float]],
) -> List[str]:
    dims: set = set()
    for s in a_scores.values():
        dims.update(s.keys())
    for s in b_scores.values():
        dims.update(s.keys())
    return sorted(dims)


# ─────────────────────────────────────────────────────────────────────────────
# Paired dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_paired_dataset(
    exp_dir: Path,
    independent_suffix: str = "independent",
) -> Optional[Dict[str, Any]]:
    """
    Load and join A and B scores. Returns dataset dict or None if data missing.
    """
    walk_log = _load_walk_log(exp_dir)
    if walk_log is None:
        print(f"  ⚠  walk_log.json missing: {exp_dir.name}")
        return None

    ind_data = _load_independent_scores(exp_dir, independent_suffix)
    if ind_data is None:
        print(
            f"  ⚠  ablation_{independent_suffix}/independent_scores.json missing: {exp_dir.name}\n"
            f"     Run ablation_independent_scoring.py first."
        )
        return None

    a_scores = _extract_a_scores(walk_log)
    b_scores = _extract_b_scores(ind_data)

    # Inner join on step
    common_steps = sorted(set(a_scores) & set(b_scores))
    if not common_steps:
        print(f"  ⚠  No paired steps found for {exp_dir.name}")
        return None

    dimensions = _get_dimensions(a_scores, b_scores)

    # Build per-step records
    rows: List[Dict[str, Any]] = []
    for step in common_steps:
        a = a_scores[step]
        b = b_scores[step]
        row = {"step": step, "experiment": exp_dir.name}
        for dim in dimensions:
            row[f"A_{dim}"] = a.get(dim)
            row[f"B_{dim}"] = b.get(dim)
        rows.append(row)

    # Compute C = mean(B) per dimension across route
    c_means: Dict[str, float] = {}
    for dim in dimensions:
        vals = [r[f"B_{dim}"] for r in rows if r[f"B_{dim}"] is not None]
        c_means[dim] = float(np.mean(vals)) if vals else 0.0

    # Add C and deltas to rows
    for row in rows:
        for dim in dimensions:
            row[f"C_{dim}"] = c_means[dim]
            a_val = row.get(f"A_{dim}")
            b_val = row.get(f"B_{dim}")
            c_val = c_means[dim]
            row[f"dAB_{dim}"] = (a_val - b_val) if (a_val is not None and b_val is not None) else None
            row[f"dAC_{dim}"] = (a_val - c_val) if a_val is not None else None
            row[f"dBC_{dim}"] = (b_val - c_val) if b_val is not None else None

    return {
        "experiment_id": exp_dir.name,
        "exp_dir": exp_dir,
        "dimensions": dimensions,
        "steps": common_steps,
        "rows": rows,
        "c_means": c_means,
        "a_scores_raw": a_scores,
        "b_scores_raw": b_scores,
        "persona": ind_data.get("persona", "unknown"),
        "framework_id": ind_data.get("framework_id", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or not _HAS_SCIPY:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(x, y)
    return float(rho)


def analyze_context_effect(
    datasets: List[Dict[str, Any]],
    dimensions: List[str],
) -> Dict[str, Any]:
    """Analysis 1: A vs B — how does context shift scores?"""
    results: Dict[str, Any] = {}
    for dim in dimensions:
        a_vals, b_vals, deltas = [], [], []
        for ds in datasets:
            for row in ds["rows"]:
                a = row.get(f"A_{dim}")
                b = row.get(f"B_{dim}")
                if a is not None and b is not None:
                    a_vals.append(a)
                    b_vals.append(b)
                    deltas.append(a - b)

        if not a_vals:
            continue

        bias = float(np.mean(deltas))
        divergence_rate = sum(1 for d in deltas if abs(d) > DIVERGENCE_THRESHOLD) / len(deltas)
        rho = _spearman(a_vals, b_vals)

        results[dim] = {
            "bias": round(bias, 4),
            "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
            "divergence_rate": round(divergence_rate, 4),
            "n_waypoints": len(a_vals),
            "mean_A": round(float(np.mean(a_vals)), 4),
            "mean_B": round(float(np.mean(b_vals)), 4),
            "std_A": round(float(np.std(a_vals)), 4),
            "std_B": round(float(np.std(b_vals)), 4),
        }
    return results


def analyze_information_preservation(
    datasets: List[Dict[str, Any]],
    dimensions: List[str],
    barrier_thresholds: List[float],
) -> Dict[str, Any]:
    """Analysis 2: A vs C and B vs C — aggregation information loss."""
    results: Dict[str, Any] = {}
    for dim in dimensions:
        dim_result: Dict[str, Any] = {}
        a_vals, b_vals, c_vals = [], [], []
        for ds in datasets:
            for row in ds["rows"]:
                a = row.get(f"A_{dim}")
                b = row.get(f"B_{dim}")
                c = row.get(f"C_{dim}")
                if a is not None:
                    a_vals.append(a)
                if b is not None:
                    b_vals.append(b)
                if c is not None:
                    c_vals.append(c)

        if not a_vals:
            continue

        # Information loss: 1 - var(C) / var(A)
        var_a = float(np.var(a_vals)) if len(a_vals) > 1 else 0.0
        var_b = float(np.var(b_vals)) if len(b_vals) > 1 else 0.0
        var_c = float(np.var(c_vals)) if len(c_vals) > 1 else 0.0
        info_loss_A = round(1 - var_c / var_a, 4) if var_a > 0 else 1.0
        info_loss_B = round(1 - var_c / var_b, 4) if var_b > 0 else 1.0

        # Score range ratio
        range_a = (max(a_vals) - min(a_vals)) if a_vals else 0
        range_b = (max(b_vals) - min(b_vals)) if b_vals else 0
        range_c = (max(c_vals) - min(c_vals)) if c_vals else 0
        range_ratio_ac = round(range_a / range_c, 4) if range_c > 0 else float("inf")
        range_ratio_bc = round(range_b / range_c, 4) if range_c > 0 else float("inf")

        dim_result["information_loss_A_vs_C"] = info_loss_A
        dim_result["information_loss_B_vs_C"] = info_loss_B
        dim_result["range_ratio_A_vs_C"] = range_ratio_ac
        dim_result["range_ratio_B_vs_C"] = range_ratio_bc

        # Hidden barrier rates per threshold
        barrier_stats: Dict[str, Dict[str, float]] = {}
        for thresh in barrier_thresholds:
            # A-vs-C: step where A says bad but route mean C says ok
            n_hidden_ac = sum(
                1 for a, c in zip(a_vals, c_vals) if a < thresh and c >= thresh
            )
            # B-vs-C: same comparison but with context-free B
            n_hidden_bc = sum(
                1 for b, c in zip(b_vals, c_vals) if b < thresh and c >= thresh
            )
            n = len(a_vals)
            barrier_stats[f"threshold_{thresh}"] = {
                "hidden_barrier_rate_A_vs_C": round(n_hidden_ac / n, 4) if n else 0,
                "hidden_barrier_rate_B_vs_C": round(n_hidden_bc / n, 4) if n else 0,
            }
        dim_result["barrier_detection"] = barrier_stats
        results[dim] = dim_result
    return results


def analyze_decision_divergence(
    datasets: List[Dict[str, Any]],
    dimensions: List[str],
    barrier_thresholds: List[float],
    recommend_threshold: float,
) -> Dict[str, Any]:
    """Analysis 4: Three-way route recommendation divergence."""
    results: Dict[str, Any] = {}

    for thresh in barrier_thresholds:
        divergence: Dict[str, Any] = {
            "threshold": thresh,
            "n_experiments": len(datasets),
            "A_ne_C": 0, "B_ne_C": 0, "A_ne_B": 0, "all_differ": 0,
        }
        for ds in datasets:
            dims = ds["dimensions"]
            # A: sequential profile has no step below threshold → recommend
            a_rec = all(
                all(
                    (row.get(f"A_{dim}") or 999) >= thresh
                    for dim in dims
                )
                for row in ds["rows"]
            )
            # B: independent profile has no step below threshold → recommend
            b_rec = all(
                all(
                    (row.get(f"B_{dim}") or 999) >= thresh
                    for dim in dims
                )
                for row in ds["rows"]
            )
            # C: aggregate mean above threshold → recommend
            c_rec = all(ds["c_means"].get(dim, 0) >= thresh for dim in dims)

            if a_rec != c_rec:
                divergence["A_ne_C"] += 1
            if b_rec != c_rec:
                divergence["B_ne_C"] += 1
            if a_rec != b_rec:
                divergence["A_ne_B"] += 1
            if a_rec != b_rec != c_rec:
                divergence["all_differ"] += 1

        results[f"threshold_{thresh}"] = divergence

    return results


def analyze_transition_detection(
    datasets: List[Dict[str, Any]],
    dimensions: List[str],
) -> Dict[str, Any]:
    """Analysis 5: A vs B transition detection counts."""
    results: Dict[str, Any] = {}

    for dim in dimensions:
        n_trans_a_total = 0
        n_trans_b_total = 0
        n_sequences = 0

        for ds in datasets:
            rows = ds["rows"]
            if len(rows) < 2:
                continue
            a_seq = [r.get(f"A_{dim}") for r in rows]
            b_seq = [r.get(f"B_{dim}") for r in rows]

            # Filter None
            a_clean = [v for v in a_seq if v is not None]
            b_clean = [v for v in b_seq if v is not None]

            n_trans_a = sum(
                1 for i in range(1, len(a_clean)) if abs(a_clean[i] - a_clean[i-1]) > TRANSITION_DELTA
            )
            n_trans_b = sum(
                1 for i in range(1, len(b_clean)) if abs(b_clean[i] - b_clean[i-1]) > TRANSITION_DELTA
            )
            n_trans_a_total += n_trans_a
            n_trans_b_total += n_trans_b
            n_sequences += 1

        ratio = (n_trans_a_total / n_trans_b_total) if n_trans_b_total > 0 else float("inf")
        results[dim] = {
            "transitions_A": n_trans_a_total,
            "transitions_B": n_trans_b_total,
            "amplification_ratio": round(ratio, 4) if not np.isinf(ratio) else None,
            "n_sequences": n_sequences,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_detail_csv(
    datasets: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Write per-step three-way detail CSV."""
    if not datasets:
        return
    all_dims = sorted({d for ds in datasets for d in ds["dimensions"]})
    header = ["experiment", "step"]
    for dim in all_dims:
        header += [f"score_A_{dim}", f"score_B_{dim}", f"score_C_{dim}",
                   f"delta_AB_{dim}", f"delta_AC_{dim}", f"delta_BC_{dim}"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ds in datasets:
            for row in ds["rows"]:
                record = [row["experiment"], row["step"]]
                for dim in all_dims:
                    record += [
                        row.get(f"A_{dim}", ""),
                        row.get(f"B_{dim}", ""),
                        row.get(f"C_{dim}", ""),
                        row.get(f"dAB_{dim}", ""),
                        row.get(f"dAC_{dim}", ""),
                        row.get(f"dBC_{dim}", ""),
                    ]
                writer.writerow(record)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_context_effect_scatter(
    datasets: List[Dict[str, Any]],
    dimensions: List[str],
    out_path: Path,
) -> None:
    """Scatter of A_score vs B_score per dimension. Diagonal = no context effect."""
    ncols = len(dimensions)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for ax, dim in zip(axes, dimensions):
        a_vals, b_vals = [], []
        for ds in datasets:
            for row in ds["rows"]:
                a = row.get(f"A_{dim}")
                b = row.get(f"B_{dim}")
                if a is not None and b is not None:
                    a_vals.append(a)
                    b_vals.append(b)

        ax.scatter(b_vals, a_vals, alpha=0.6, s=30,
                   color=CONDITION_COLORS["A"], edgecolors="none")
        # Diagonal
        lims = [0, 10]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Independent (B)", fontsize=10)
        ax.set_ylabel("Full System (A)", fontsize=10)
        ax.set_title(dim.capitalize(), fontsize=11, fontweight="bold")
        ax.set_aspect("equal")

        # Annotate mean bias
        if a_vals and b_vals:
            bias = np.mean(np.array(a_vals) - np.array(b_vals))
            ax.text(0.05, 0.92, f"bias={bias:+.2f}", transform=ax.transAxes,
                    fontsize=8, color="dimgray")

    fig.suptitle("Context Effect: Full System (A) vs Independent (B)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_three_way_profile(
    datasets: List[Dict[str, Any]],
    dimensions: List[str],
    out_path: Path,
) -> None:
    """Three overlaid score profiles for the most dramatic route."""
    # Pick experiment with highest A-B variance across all dims
    def drama_score(ds: Dict) -> float:
        vals = []
        for dim in ds["dimensions"]:
            for row in ds["rows"]:
                a = row.get(f"A_{dim}")
                b = row.get(f"B_{dim}")
                if a is not None and b is not None:
                    vals.append(abs(a - b))
        return float(np.mean(vals)) if vals else 0.0

    best_ds = max(datasets, key=drama_score)
    rows = best_ds["rows"]
    steps = [r["step"] for r in rows]
    dims = dimensions

    nrows = len(dims)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        a_seq = [r.get(f"A_{dim}") for r in rows]
        b_seq = [r.get(f"B_{dim}") for r in rows]
        c_val = best_ds["c_means"].get(dim)

        ax.plot(steps, a_seq, color=CONDITION_COLORS["A"], linewidth=2,
                label=CONDITION_LABELS["A"], zorder=3)
        ax.plot(steps, b_seq, color=CONDITION_COLORS["B"], linewidth=1.5,
                linestyle="--", label=CONDITION_LABELS["B"], zorder=2)
        if c_val is not None:
            ax.axhline(c_val, color=CONDITION_COLORS["C"], linewidth=1.5,
                       linestyle=":", label=f"{CONDITION_LABELS['C']} ({c_val:.2f})", zorder=1)

        ax.set_ylabel(dim.capitalize(), fontsize=10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Step", fontsize=10)
    fig.suptitle(
        f"Three-Way Score Profile\n{best_ds['experiment_id']} ({best_ds.get('persona', '')})",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_barrier_comparison(
    info_preservation: Dict[str, Any],
    dimensions: List[str],
    barrier_thresholds: List[float],
    out_path: Path,
) -> None:
    """Grouped bar chart: hidden barrier rates A-vs-C and B-vs-C per dimension."""
    x = np.arange(len(dimensions))
    width = 0.35
    n_thresh = len(barrier_thresholds)
    fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 4), sharey=True)
    if n_thresh == 1:
        axes = [axes]

    for ax, thresh in zip(axes, barrier_thresholds):
        key = f"threshold_{thresh}"
        rates_ac = []
        rates_bc = []
        for dim in dimensions:
            dim_data = info_preservation.get(dim, {})
            bd = dim_data.get("barrier_detection", {}).get(key, {})
            rates_ac.append(bd.get("hidden_barrier_rate_A_vs_C", 0))
            rates_bc.append(bd.get("hidden_barrier_rate_B_vs_C", 0))

        bars_a = ax.bar(x - width / 2, rates_ac, width, label="A vs C",
                        color=CONDITION_COLORS["A"], alpha=0.85)
        bars_b = ax.bar(x + width / 2, rates_bc, width, label="B vs C",
                        color=CONDITION_COLORS["B"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in dimensions], rotation=15, ha="right")
        ax.set_title(f"Barrier threshold = {thresh}", fontsize=10)
        ax.set_ylabel("Hidden Barrier Rate", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Hidden Barrier Rate: Sequential vs Aggregate", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transition_amplification(
    transition_data: Dict[str, Any],
    dimensions: List[str],
    out_path: Path,
) -> None:
    """Bar chart: transition counts A vs B per dimension."""
    x = np.arange(len(dimensions))
    width = 0.35
    counts_a = [transition_data.get(dim, {}).get("transitions_A", 0) for dim in dimensions]
    counts_b = [transition_data.get(dim, {}).get("transitions_B", 0) for dim in dimensions]

    fig, ax = plt.subplots(figsize=(max(6, len(dimensions) * 1.5), 4))
    ax.bar(x - width / 2, counts_a, width, label=CONDITION_LABELS["A"],
           color=CONDITION_COLORS["A"], alpha=0.85)
    ax.bar(x + width / 2, counts_b, width, label=CONDITION_LABELS["B"],
           color=CONDITION_COLORS["B"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions], rotation=15, ha="right")
    ax.set_ylabel("Transition Count (|Δ| > 1.5)", fontsize=9)
    ax.set_title("Transition Detection: Full System (A) vs Independent (B)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate ratio
    for i, (ca, cb) in enumerate(zip(counts_a, counts_b)):
        if cb > 0:
            ratio = ca / cb
            ax.text(i, max(ca, cb) + 0.3, f"×{ratio:.1f}", ha="center", fontsize=8, color="dimgray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    input_dir: Path,
    barrier_thresholds: List[float],
    recommend_threshold: float,
    independent_suffix: str = "independent",
) -> int:
    exp_dirs = _find_experiment_dirs(input_dir)
    if not exp_dirs:
        print(f"❌ No experiment directories found under {input_dir}")
        return 1

    print(f"Found {len(exp_dirs)} experiment(s).")

    # Load all datasets
    datasets: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        ds = build_paired_dataset(exp_dir, independent_suffix)
        if ds is not None:
            datasets.append(ds)
            print(f"  ✓ {exp_dir.name}: {len(ds['steps'])} paired steps")

    if not datasets:
        print("❌ No paired datasets available. Run ablation_independent_scoring.py first.")
        return 1

    # Collect all dimensions
    all_dims = sorted({d for ds in datasets for d in ds["dimensions"]})
    total_waypoints = sum(len(ds["steps"]) for ds in datasets)
    print(f"\nTotal paired waypoints: {total_waypoints}")
    print(f"Dimensions: {all_dims}")

    # Run analyses
    print("\nRunning analyses...")
    context_effect = analyze_context_effect(datasets, all_dims)
    info_preservation = analyze_information_preservation(datasets, all_dims, barrier_thresholds)
    decision_divergence = analyze_decision_divergence(
        datasets, all_dims, barrier_thresholds, recommend_threshold
    )
    transition_detection = analyze_transition_detection(datasets, all_dims)

    # Determine output directory
    if _is_experiment_dir(input_dir):
        out_dir = input_dir / "ablation_analysis"
    else:
        out_dir = input_dir / "ablation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to: {out_dir}")

    # 1. Detail CSV
    detail_csv = out_dir / "ablation_three_way_detail.csv"
    write_detail_csv(datasets, detail_csv)
    print(f"  ✓ {detail_csv.name}")

    # 2. Summary JSON
    summary = {
        "n_experiments": len(datasets),
        "paired_waypoints": total_waypoints,
        "dimensions": all_dims,
        "barrier_thresholds": barrier_thresholds,
        "recommend_threshold": recommend_threshold,
        "context_effect": context_effect,
        "information_preservation": info_preservation,
        "decision_divergence": decision_divergence,
        "transition_detection": transition_detection,
    }
    summary_path = out_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {summary_path.name}")

    # 3. Plots
    if total_waypoints > 0:
        scatter_path = out_dir / "context_effect_scatter.png"
        plot_context_effect_scatter(datasets, all_dims, scatter_path)
        print(f"  ✓ {scatter_path.name}")

        profile_path = out_dir / "three_way_profile.png"
        plot_three_way_profile(datasets, all_dims, profile_path)
        print(f"  ✓ {profile_path.name}")

        barrier_path = out_dir / "hidden_barrier_comparison.png"
        plot_hidden_barrier_comparison(info_preservation, all_dims, barrier_thresholds, barrier_path)
        print(f"  ✓ {barrier_path.name}")

        transition_path = out_dir / "transition_amplification.png"
        plot_transition_amplification(transition_detection, all_dims, transition_path)
        print(f"  ✓ {transition_path.name}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("CONTEXT EFFECT SUMMARY (A vs B)")
    print("-" * 40)
    for dim, stats in context_effect.items():
        bias = stats.get("bias", 0)
        rho = stats.get("spearman_rho")
        dr = stats.get("divergence_rate", 0)
        rho_str = f"{rho:.3f}" if rho is not None else "n/a"
        print(
            f"  {dim:12s}  bias={bias:+.3f}  ρ={rho_str}  "
            f"divergence={dr:.1%}"
        )

    print("\nINFORMATION LOSS (1 - var(C)/var(A))")
    print("-" * 40)
    for dim, stats in info_preservation.items():
        il = stats.get("information_loss_A_vs_C", "?")
        print(f"  {dim:12s}  {il:.3f}")

    print("\nTRANSITION AMPLIFICATION (A / B)")
    print("-" * 40)
    for dim, stats in transition_detection.items():
        ratio = stats.get("amplification_ratio")
        na = stats.get("transitions_A", 0)
        nb = stats.get("transitions_B", 0)
        ratio_str = f"{ratio:.2f}×" if ratio is not None else "∞"
        print(f"  {dim:12s}  A={na}  B={nb}  ratio={ratio_str}")

    print(f"\n✅ Analysis complete → {out_dir}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-way ablation comparison: Full System (A) vs Independent (B) vs Aggregate (C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Experiment dir or run dir",
    )
    parser.add_argument(
        "--barrier-threshold", type=float, nargs="+",
        default=DEFAULT_BARRIER_THRESHOLDS,
        dest="barrier_thresholds",
        help="Barrier thresholds (default: 4.0 5.0 6.0)",
    )
    parser.add_argument(
        "--recommend-threshold", type=float, default=DEFAULT_RECOMMEND_THRESHOLD,
        dest="recommend_threshold",
        help=f"Route recommendation threshold (default: {DEFAULT_RECOMMEND_THRESHOLD})",
    )
    parser.add_argument(
        "--independent-suffix", type=str, default="independent",
        dest="independent_suffix",
        help="Suffix of ablation dir to load (default: independent → ablation_independent/)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    return run_analysis(
        input_dir=input_dir,
        barrier_thresholds=args.barrier_thresholds,
        recommend_threshold=args.recommend_threshold,
        independent_suffix=args.independent_suffix,
    )


if __name__ == "__main__":
    sys.exit(main())
