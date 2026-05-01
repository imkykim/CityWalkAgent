"""
compare_personas.py — Cross-persona qualitative + quantitative comparison
for a single CityWalkAgent experiment run (single route, multi-persona).

Usage:
    python scripts/compare_personas.py <run_dir> [--lang en|ko] [--skip-llm]

Example:
    python scripts/compare_personas.py outputs/test_runs/20260409_160633/

Inputs (expected inside <run_dir>):
    comparison_report.json
    experiment_matrix.json
    <route>_<persona>_<timestamp>/
        walk_log.json
        branch_decisions.json
        score_timeline.csv
        run_metadata.json

Outputs (written to <run_dir>/comparison_analysis/):
    qualitative_report.md
    divergence_data.json
    divergence_plots.png
    route_comparison.png
    route_comparison_map.html
    comparison_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env from city_walk_agent/ (works regardless of cwd)
load_dotenv(Path(__file__).parent.parent / ".env")

# Optional LLM
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", os.environ.get("VLM_API_URL", ""))
LLM_MODEL    = os.environ.get("LLM_TEXT_MODEL", os.environ.get("VLM_MODEL", ""))
LLM_API_KEY  = os.environ.get("LLM_API_KEY",   os.environ.get("VLM_API_KEY", ""))

DIMENSIONS = ["safety", "lively", "beautiful", "wealthy"]

# Distinct palette — matches the color conventions in walk_output_writer.py
PERSONA_COLORS = [
    "#3b82f6",  # blue
    "#f97316",  # orange
    "#22c55e",  # green
    "#a855f7",  # purple
    "#ef4444",  # red
    "#06b6d4",  # cyan
    "#eab308",  # yellow
    "#ec4899",  # pink
]


def _persona_color(idx: int) -> str:
    return PERSONA_COLORS[idx % len(PERSONA_COLORS)]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _avg_score_for_step(s: dict) -> float:
    scores = s.get("scores", {})
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _reasoning_excerpt(s: dict, max_chars: int = 200) -> str:
    """First non-empty dimension reasoning text, truncated."""
    reasoning = s.get("reasoning") or {}
    if isinstance(reasoning, dict):
        for dim in DIMENSIONS:
            text = reasoning.get(dim, "")
            if text:
                return text[:max_chars] + ("…" if len(text) > max_chars else "")
    return ""


class PersonaRun:
    def __init__(
        self,
        persona_key: str,
        persona_display: str,
        folder: Path,
        walk_log: dict,
        branch_decisions: list[dict],
        timeline: pd.DataFrame,
        metadata: dict,
    ):
        self.persona_key = persona_key
        self.persona_display = persona_display
        self.folder = folder
        self.walk_log = walk_log
        self.branch_decisions = branch_decisions
        self.timeline = timeline
        self.metadata = metadata

    @property
    def steps(self) -> list[dict]:
        return self.walk_log.get("route_taken", [])

    @property
    def analyzed_steps(self) -> list[dict]:
        """Steps that were scored by the VLM (have non-empty scores dict)."""
        return [s for s in self.steps if s.get("scores")]

    @property
    def s2_steps(self) -> list[dict]:
        """Steps where System 2 triggered (trigger_reason is non-null)."""
        return [s for s in self.steps if s.get("trigger_reason")]

    @property
    def branch_steps(self) -> list[dict]:
        return [s for s in self.steps if s.get("branch_triggered")]

    @property
    def avg_score(self) -> float:
        scores = [_avg_score_for_step(s) for s in self.analyzed_steps]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def visited_panos(self) -> list[str]:
        return [s["pano_id"] for s in self.steps if s.get("pano_id")]

    @property
    def visited_coords(self) -> list[tuple[float, float]]:
        return [(s["lat"], s["lng"]) for s in self.steps
                if isinstance(s.get("lat"), (int, float))
                and isinstance(s.get("lng"), (int, float))]


def _discover_persona_folders(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir()
                   if p.is_dir() and (p / "walk_log.json").exists()])


def load_run_data(run_dir: Path) -> tuple[dict, dict, list[PersonaRun]]:
    cmp_report = json.loads((run_dir / "comparison_report.json").read_text())
    exp_matrix = json.loads((run_dir / "experiment_matrix.json").read_text())

    persona_list = [e["persona"] for e in exp_matrix["experiments"]]
    folders = _discover_persona_folders(run_dir)

    persona_folders: dict[str, Path] = {}
    for p in persona_list:
        match = next((f for f in folders if p in f.name), None)
        if match:
            persona_folders[p] = match

    missing = set(persona_list) - set(persona_folders.keys())
    if missing:
        print(f"[warn] No folder found for personas: {missing}", file=sys.stderr)

    runs: list[PersonaRun] = []
    for persona_key, folder in persona_folders.items():
        walk_log = json.loads((folder / "walk_log.json").read_text())
        try:
            branches = json.loads((folder / "branch_decisions.json").read_text())
        except FileNotFoundError:
            branches = []
        try:
            timeline = pd.read_csv(folder / "score_timeline.csv")
        except FileNotFoundError:
            timeline = pd.DataFrame()
        try:
            metadata = json.loads((folder / "run_metadata.json").read_text())
        except FileNotFoundError:
            metadata = {}

        runs.append(PersonaRun(
            persona_key=persona_key,
            persona_display=walk_log.get("persona", persona_key),
            folder=folder,
            walk_log=walk_log,
            branch_decisions=branches,
            timeline=timeline,
            metadata=metadata,
        ))

    return cmp_report, exp_matrix, runs


# -----------------------------------------------------------------------------
# Quantitative analysis
# -----------------------------------------------------------------------------

def compute_dimension_divergence(runs: list[PersonaRun]) -> dict:
    records = []
    for r in runs:
        for s in r.analyzed_steps:
            scores = s.get("scores", {})
            records.append({
                "persona": r.persona_key,
                "step": s["step"],
                "pano_id": s.get("pano_id"),
                **{d: scores.get(d) for d in DIMENSIONS},
            })
    df = pd.DataFrame(records)
    if df.empty:
        return {"per_step_std": [], "per_dim_gap": {}, "per_dim_std": {}}

    persona_means = df.groupby("persona")[DIMENSIONS].mean()
    per_dim_gap = {d: float(persona_means[d].max() - persona_means[d].min())
                   for d in DIMENSIONS}
    per_dim_std = {d: float(persona_means[d].std(ddof=0)) for d in DIMENSIONS}
    step_std = df.groupby("step")[DIMENSIONS].std(ddof=0).dropna(how="all")

    return {
        "persona_dim_means": persona_means.round(3).to_dict(),
        "per_dim_gap": {k: round(v, 3) for k, v in per_dim_gap.items()},
        "per_dim_std": {k: round(v, 3) for k, v in per_dim_std.items()},
        "step_std_df": step_std,
    }


def compute_path_divergence(runs: list[PersonaRun], grid_m: float = 20.0) -> dict:
    def to_cells(coords: list[tuple[float, float]]) -> set:
        cell = grid_m / 111_000.0
        return {(round(lat / cell), round(lng / cell)) for lat, lng in coords}

    cells = {r.persona_key: to_cells(r.visited_coords) for r in runs}
    pairs = []
    for i, r1 in enumerate(runs):
        for r2 in runs[i + 1:]:
            a, b = cells[r1.persona_key], cells[r2.persona_key]
            union = len(a | b)
            jaccard = len(a & b) / union if union else 0.0
            pairs.append({
                "persona_a": r1.persona_key,
                "persona_b": r2.persona_key,
                "jaccard": round(jaccard, 3),
                "cells_a": len(a),
                "cells_b": len(b),
                "shared": len(a & b),
            })
    return {"grid_m": grid_m, "pairs": pairs}


def compute_s2_alignment(runs: list[PersonaRun]) -> dict:
    out = {}
    for r in runs:
        s2 = r.s2_steps
        out[r.persona_key] = {
            "s2_count": len(s2),
            "s2_steps": [s["step"] for s in s2],
            "trigger_reasons": dict(Counter(s.get("trigger_reason") for s in s2)),
            "gating_signal_counts": r.walk_log.get("gating_signal_counts", {}),
        }
    return out


def compute_branch_divergence(runs: list[PersonaRun]) -> dict:
    by_pano: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in runs:
        for b in r.branch_decisions:
            pano = b.get("pano_id")
            if not pano:
                continue
            chosen = next((c for c in b.get("candidates", []) if c.get("chosen")), None)
            by_pano[pano][r.persona_key] = {
                "step": b.get("step"),
                "chosen_heading": b.get("chosen_heading"),
                "chosen_direction": chosen.get("direction") if chosen else None,
                "chosen_avg_score": chosen.get("avg_score") if chosen else None,
                "candidate_count": b.get("candidate_count"),
                "recommendation": b.get("recommendation"),
            }

    shared = {p: d for p, d in by_pano.items() if len(d) >= 2}
    divergent = []
    for pano, persona_choices in shared.items():
        headings = {pk: pc["chosen_heading"] for pk, pc in persona_choices.items()}
        if len(set(headings.values())) > 1:
            divergent.append({"pano_id": pano, "choices": persona_choices})
    return {
        "shared_intersections": len(shared),
        "divergent_intersections": len(divergent),
        "divergent_detail": divergent[:10],
    }


# -----------------------------------------------------------------------------
# LLM qualitative analysis
# -----------------------------------------------------------------------------

def _llm_client() -> "OpenAI | None":
    if not _HAS_OPENAI:
        print("[warn] openai SDK not installed; skipping LLM analysis", file=sys.stderr)
        return None
    if not LLM_API_KEY:
        print("[warn] No LLM API key found. Set LLM_API_KEY (or VLM_API_KEY) to enable qualitative analysis.",
              file=sys.stderr)
        return None
    if not LLM_BASE_URL:
        print("[warn] No LLM base URL found. Set LLM_BASE_URL (or VLM_API_URL) to enable qualitative analysis.",
              file=sys.stderr)
        return None
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _call_llm_json(client, system: str, user: str, *, max_tokens: int = 1200) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return None


def _top_reasoning_excerpts(run: PersonaRun, k: int = 3) -> list[dict]:
    analyzed = [s for s in run.analyzed_steps if s.get("reasoning")]
    if not analyzed:
        return []
    scored = [(s, _avg_score_for_step(s)) for s in analyzed]
    scored.sort(key=lambda x: x[1])
    if len(scored) >= 3:
        picks = [scored[0], scored[len(scored) // 2], scored[-1]]
    else:
        picks = scored
    return [{
        "step": s["step"],
        "avg_score": round(avg, 2),
        "scores": s["scores"],
        "reasoning": s["reasoning"],
    } for s, avg in picks[:k]]


def llm_persona_lens_summary(client, run: PersonaRun, lang: str) -> dict | None:
    excerpts = _top_reasoning_excerpts(run, k=3)
    route_report = run.walk_log.get("route_report", {})
    system = (
        "You analyze how a persona's priors shape pedestrian perception of urban "
        "environments. Given reasoning excerpts from a single persona's walk, "
        "identify (a) the environmental features they emphasize, (b) the features "
        "they downweight or ignore, (c) their dominant affective stance. Output "
        f"in {'Korean' if lang == 'ko' else 'English'}. Respond ONLY with JSON."
    )
    user = json.dumps({
        "persona": run.persona_display,
        "avg_score": run.avg_score,
        "dim_means": run.metadata.get("statistics", {}).get("dimension_summary", {}),
        "route_verdicts": route_report.get("dimension_verdicts", {}),
        "persona_suitability": route_report.get("persona_suitability"),
        "reasoning_excerpts": excerpts,
    }, ensure_ascii=False, indent=2)
    schema_hint = (
        'Return JSON with keys: '
        '{"features_emphasized": [str, str, str], '
        '"features_downweighted": [str, str], '
        '"affective_stance": str, '
        '"signature_phrase": str}'
    )
    return _call_llm_json(client, system, user + "\n\n" + schema_hint, max_tokens=700)


def llm_branch_point_comparison(client, divergent: list[dict],
                                 runs: list[PersonaRun], lang: str) -> list[dict]:
    if not divergent:
        return []
    run_by_key = {r.persona_key: r for r in runs}
    results = []
    for div in divergent[:3]:
        pano = div["pano_id"]
        persona_interps = {}
        for pk, choice_info in div["choices"].items():
            run = run_by_key.get(pk)
            if not run:
                continue
            bd = next((b for b in run.branch_decisions if b.get("pano_id") == pano), None)
            if not bd:
                continue
            chosen_cand = next((c for c in bd.get("candidates", []) if c.get("chosen")), None)
            persona_interps[pk] = {
                "chosen_direction": choice_info["chosen_direction"],
                "chosen_heading": choice_info["chosen_heading"],
                "chosen_scores": chosen_cand.get("scores") if chosen_cand else None,
                "chosen_interpretation": chosen_cand.get("interpretation") if chosen_cand else None,
                "recommendation": bd.get("recommendation"),
                "all_candidates": [{
                    "direction": c.get("direction"),
                    "heading": c.get("heading"),
                    "avg_score": c.get("avg_score"),
                    "key_concern": c.get("key_concern"),
                } for c in bd.get("candidates", [])],
            }
        if len(persona_interps) < 2:
            continue
        system = (
            "You compare persona-specific branch decisions at the same urban intersection. "
            "Explain the divergence as a function of persona priors, not random choice. "
            f"Respond in {'Korean' if lang == 'ko' else 'English'} as JSON."
        )
        user = json.dumps({"intersection_pano_id": pano, "persona_choices": persona_interps},
                          ensure_ascii=False, indent=2)
        schema = ('JSON: {"divergence_summary": str, '
                  '"persona_rationales": {"<persona>": "1-sentence why"}, '
                  '"shared_environment_signal": str}')
        out = _call_llm_json(client, system, user + "\n\n" + schema, max_tokens=800)
        if out:
            out["pano_id"] = pano
            results.append(out)
    return results


def llm_synthesis_narrative(client, quant: dict, persona_summaries: dict,
                            branch_comparisons: list[dict], lang: str) -> dict | None:
    system = (
        "You synthesize a cross-persona urban perception study. Your audience "
        "is an architecture thesis committee. Argue the core finding: how five "
        "personas walking the SAME street produce materially different perceived "
        "cities. Avoid summarizing each persona; instead articulate the divergence "
        f"structure. Respond in {'Korean' if lang == 'ko' else 'English'} as JSON."
    )
    user = json.dumps({
        "dimension_gaps": quant.get("dim_divergence", {}).get("per_dim_gap"),
        "dimension_stds": quant.get("dim_divergence", {}).get("per_dim_std"),
        "path_divergence_pairs": quant.get("path_divergence", {}).get("pairs", [])[:6],
        "s2_alignment": quant.get("s2_alignment"),
        "persona_lenses": persona_summaries,
        "branch_divergence_examples": branch_comparisons,
    }, ensure_ascii=False, indent=2)
    schema = ('JSON: {"divergence_thesis": "1-sentence core finding", '
              '"dimension_splits": {"safety": str, "lively": str, "beautiful": str, "wealthy": str}, '
              '"path_story": "2-3 sentences on how branch choices compounded", '
              '"paper_paragraph": "3-4 sentence claim suitable for thesis intro"}')
    return _call_llm_json(client, system, user + "\n\n" + schema, max_tokens=1400)


# -----------------------------------------------------------------------------
# Output 1 — divergence_plots.png (existing)
# -----------------------------------------------------------------------------

def make_plots(runs: list[PersonaRun], quant: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for i, r in enumerate(runs):
        df = r.timeline
        if df.empty or "avg_score" not in df.columns:
            continue
        ax.plot(df["step"], df["avg_score"], marker="o", ms=3,
                label=r.persona_key, alpha=0.8, color=_persona_color(i))
    ax.set_title("Average score trajectory by persona")
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    dim_means = quant["dim_divergence"].get("persona_dim_means", {})
    if dim_means:
        personas = sorted({p for dim_d in dim_means.values() for p in dim_d.keys()})
        x = np.arange(len(personas))
        width = 0.2
        dim_colors = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]
        for i, d in enumerate(DIMENSIONS):
            vals = [dim_means.get(d, {}).get(p, 0) for p in personas]
            ax.bar(x + (i - 1.5) * width, vals, width, label=d, color=dim_colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(personas, rotation=30, ha="right", fontsize=8)
        ax.set_title("Dimension score by persona")
        ax.set_ylabel("Mean score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    s2_align = quant.get("s2_alignment", {})
    for i, (pk, info) in enumerate(s2_align.items()):
        steps = info.get("s2_steps", [])
        ax.scatter(steps, [i] * len(steps), s=60, alpha=0.8,
                   label=pk, color=_persona_color(i))
    ax.set_yticks(range(len(s2_align)))
    ax.set_yticklabels(list(s2_align.keys()), fontsize=8)
    ax.set_xlabel("Step")
    ax.set_title("System-2 trigger timeline")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    pairs = quant.get("path_divergence", {}).get("pairs", [])
    if pairs:
        personas = sorted({p for pair in pairs for p in (pair["persona_a"], pair["persona_b"])})
        n = len(personas)
        mat = np.ones((n, n))
        idx = {p: i for i, p in enumerate(personas)}
        for pair in pairs:
            ii, jj = idx[pair["persona_a"]], idx[pair["persona_b"]]
            mat[ii, jj] = mat[jj, ii] = pair["jaccard"]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn_r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(personas, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(personas, fontsize=8)
        for ii in range(n):
            for jj in range(n):
                ax.text(jj, ii, f"{mat[ii,jj]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        ax.set_title("Path overlap (Jaccard, grid cells)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Output 2 — route_comparison.png
# -----------------------------------------------------------------------------

def make_route_png(runs: list[PersonaRun], cmp_report: dict, out_path: Path) -> None:
    """Lat/lng polyline plot of all persona routes, no basemap."""
    route_name = _route_name(cmp_report)

    # Shared intersections (pano_id visited by ≥2 personas)
    pano_counts: Counter = Counter()
    pano_coords: dict[str, tuple[float, float]] = {}
    for r in runs:
        for s in r.steps:
            pid = s.get("pano_id")
            if pid and isinstance(s.get("lat"), (int, float)):
                pano_counts[pid] += 1
                pano_coords[pid] = (s["lat"], s["lng"])
    shared_panos = {pid for pid, cnt in pano_counts.items() if cnt >= 2}

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)

    for i, r in enumerate(runs):
        coords = r.visited_coords
        if len(coords) < 2:
            continue
        lats, lngs = zip(*coords)
        color = _persona_color(i)

        # Full polyline
        ax.plot(lngs, lats, "-", color=color, linewidth=1.8, alpha=0.75,
                label=r.persona_key, zorder=2)

        # Analyzed steps: filled dot; skipped: faded tiny dot
        for s in r.steps:
            if not isinstance(s.get("lat"), (int, float)):
                continue
            if s.get("scores"):
                ax.plot(s["lng"], s["lat"], "o", color=color,
                        markersize=3.5, alpha=0.85, zorder=3)
            else:
                ax.plot(s["lng"], s["lat"], "o", color=color,
                        markersize=1.5, alpha=0.25, zorder=2)

        # Start marker
        ax.plot(lngs[0], lats[0], "^", color=color,
                markersize=10, zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        # End marker
        ax.plot(lngs[-1], lats[-1], "s", color=color,
                markersize=8, zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Shared intersection highlights
    for pid in shared_panos:
        lat, lng = pano_coords[pid]
        ax.plot(lng, lat, "o", color="white", markersize=11, zorder=6)
        ax.plot(lng, lat, "o", color="black", markersize=11,
                markerfacecolor="none", markeredgewidth=1.5, zorder=7)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(f"Route Comparison — {route_name}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)

    # Legend: personas + shared-intersection marker
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    shared_handle = Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="none", markeredgecolor="black",
                           markeredgewidth=1.5, markersize=9, label="Shared intersection")
    handles.append(shared_handle)
    labels.append("Shared intersection (≥2 personas)")

    legend = ax.legend(handles, labels, fontsize=8, loc="best",
                       framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Output 3 — route_comparison_map.html  (Leaflet) + per-persona maps
# -----------------------------------------------------------------------------

_MAP_JS_TEMPLATE = """\
const map = L.map('map').setView(RAW.center, 16);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '© OpenStreetMap contributors © CARTO',
  subdomains: 'abcd', maxZoom: 20
}).addTo(map);

function scoreColor(avg) {
  if (avg >= 8) return '#22c55e';
  if (avg >= 6) return '#84cc16';
  if (avg >= 5) return '#eab308';
  if (avg >= 4) return '#f97316';
  return '#ef4444';
}

function renderPersonaLayer(p, group) {
  // Polyline colored by per-step avg score
  for (let i = 1; i < p.markers.length; i++) {
    const prev = p.markers[i-1], curr = p.markers[i];
    if (curr.avg != null) {
      L.polyline([[prev.lat, prev.lng],[curr.lat, curr.lng]], {
        color: scoreColor(curr.avg), weight: 4, opacity: 0.85
      }).addTo(group);
    } else if (prev.lat != null) {
      L.polyline([[prev.lat, prev.lng],[curr.lat, curr.lng]], {
        color: p.color, weight: 2, opacity: 0.3
      }).addTo(group);
    }
  }

  // Step markers
  p.markers.forEach(function(m) {
    if (!m.has_score && !m.is_s2) {
      L.circleMarker([m.lat, m.lng], {
        radius: 2.5, color: p.color, fillColor: p.color,
        fillOpacity: 0.2, weight: 0
      }).addTo(group);
      return;
    }
    const radius = m.avg != null ? 3 + m.avg * 0.7 : 5;
    const opts = m.is_s2
      ? { radius, color: '#111', fillColor: p.color, fillOpacity: 0.9, weight: 2, dashArray: '4,2' }
      : { radius, color: scoreColor(m.avg ?? 5), fillColor: scoreColor(m.avg ?? 5), fillOpacity: 0.75, weight: 1 };
    const dimRows = Object.entries(m.scores)
      .filter(([_, v]) => v != null)
      .map(([d, v]) => `<tr><td>${d}</td><td>${v}</td></tr>`)
      .join('');
    const popup = `
      <b>Step ${m.step}</b> &nbsp; <span style="color:${p.color}">${p.key}</span>
      ${m.is_s2 ? ' &nbsp; <b>★ S2</b> (' + m.trigger_reason + ')' : ''}
      <br><small>${m.pano_id}</small>
      <table class="popup-table" style="margin-top:4px">
        <tr><td>avg</td><td><b>${m.avg ?? '—'}</b></td></tr>
        ${dimRows}
      </table>
      ${m.excerpt ? '<div style="margin-top:6px;font-size:10px;color:#555;max-width:260px">' + m.excerpt + '</div>' : ''}
    `;
    L.circleMarker([m.lat, m.lng], opts)
      .bindPopup(popup, {maxWidth: 300})
      .addTo(group);
  });
}
"""

_SCORE_LEGEND_HTML = """\
<div style="background:#fff;padding:10px 14px;border-radius:6px;
    box-shadow:0 1px 4px rgba(0,0,0,.12);font-size:11px;line-height:1.8">
  <b>Score</b><br>
  <span style="color:#22c55e">●</span> 8–10 Excellent<br>
  <span style="color:#84cc16">●</span> 6–8 Good<br>
  <span style="color:#eab308">●</span> 5–6 Average<br>
  <span style="color:#f97316">●</span> 4–5 Below avg<br>
  <span style="color:#ef4444">●</span> &lt;4 Poor<br>
  <span style="font-size:10px;color:#555">– – S2 triggered<br>· · Skipped step<br>⚡ Divergent</span>
</div>"""


def _build_persona_markers(r: "PersonaRun", color: str) -> list[dict]:
    markers = []
    for s in r.steps:
        if not isinstance(s.get("lat"), (int, float)):
            continue
        has_score = bool(s.get("scores"))
        is_s2 = bool(s.get("trigger_reason"))
        avg = round(_avg_score_for_step(s), 2) if has_score else None
        scores = s.get("scores", {})
        markers.append({
            "lat": s["lat"],
            "lng": s["lng"],
            "step": s.get("step"),
            "pano_id": s.get("pano_id", ""),
            "has_score": has_score,
            "is_s2": is_s2,
            "trigger_reason": s.get("trigger_reason", ""),
            "avg": avg,
            "scores": {d: scores.get(d) for d in DIMENSIONS},
            "excerpt": _reasoning_excerpt(s, 200),
        })
    return markers


def _build_divergent_markers(branch_div: dict, runs: list["PersonaRun"]) -> list[dict]:
    divergent_markers = []
    for div in branch_div.get("divergent_detail", []):
        lat, lng = None, None
        for r in runs:
            for s in r.steps:
                if s.get("pano_id") == div["pano_id"]:
                    lat, lng = s.get("lat"), s.get("lng")
                    break
            if lat is not None:
                break
        if lat is None:
            continue
        choices_text = "; ".join(
            f"{pk}: {info.get('chosen_direction', '?')} (hdg {info.get('chosen_heading', '?')}°)"
            for pk, info in div["choices"].items()
        )
        divergent_markers.append({
            "lat": lat, "lng": lng,
            "pano_id": div["pano_id"],
            "choices_text": choices_text,
        })
    return divergent_markers


def _map_center(runs: list["PersonaRun"]) -> tuple[float, float]:
    all_lats = [s[0] for r in runs for s in r.visited_coords]
    all_lngs = [s[1] for r in runs for s in r.visited_coords]
    return (
        float(np.mean(all_lats)) if all_lats else 0.0,
        float(np.mean(all_lngs)) if all_lngs else 0.0,
    )


def _score_trend_sparkline_html(r: "PersonaRun", color: str) -> str:
    """Inline SVG sparkline of avg score over steps."""
    pts = [(s.get("step", i), round(_avg_score_for_step(s), 2))
           for i, s in enumerate(r.analyzed_steps)]
    if len(pts) < 2:
        return ""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    W, H = 160, 36
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 1.0, 10.0
    def px(x): return (x - x_min) / max(x_max - x_min, 1) * W
    def py(y): return H - (y - y_min) / (y_max - y_min) * H
    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
    avg = sum(ys) / len(ys)
    return (
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'style="display:block;overflow:visible">'
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linejoin="round"/>'
        f'<text x="{W}" y="10" text-anchor="end" '
        f'style="font-size:9px;fill:{color};font-weight:600">avg {avg:.2f}</text>'
        f'</svg>'
    )


def _make_single_persona_map_html(
    r: "PersonaRun",
    color: str,
    divergent_markers: list[dict],
    run_id: str,
    route_name: str,
    all_persona_links: list[tuple[str, str, str]],  # [(key, color, filename), ...]
) -> str:
    """Generate a standalone Leaflet HTML for a single persona."""
    center_lat, center_lng = _map_center([r])
    markers = _build_persona_markers(r, color)

    # Timeline data for mini chart
    timeline_pts = [
        {"step": m["step"], "avg": m["avg"]}
        for m in markers if m["avg"] is not None
    ]

    data_js = json.dumps({
        "persona": {"key": r.persona_key, "display": r.persona_display, "color": color},
        "markers": markers,
        "divergent_markers": [d for d in divergent_markers],
        "center": [center_lat, center_lng],
        "timeline": timeline_pts,
    }, ensure_ascii=False)

    nav_links = "".join(
        f'<a href="{fname}" style="display:inline-flex;align-items:center;gap:4px;'
        f'padding:3px 8px;border-radius:4px;font-size:11px;text-decoration:none;'
        f'{"background:#f3f4f6;color:#374151" if pkey != r.persona_key else "background:" + pcolor + "22;color:" + pcolor + ";font-weight:600"};">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{pcolor};display:inline-block"></span>'
        f'{pkey}</a>'
        for pkey, pcolor, fname in all_persona_links
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{r.persona_display} — {route_name}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fff; }}
  #titlebar {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 1001;
    background: #fff; border-bottom: 1px solid #e5e7eb;
    padding: 8px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  #titlebar h1 {{ font-size: 14px; font-weight: 600; color: #111; white-space: nowrap; }}
  .nav-bar {{ display: flex; gap: 4px; flex-wrap: wrap; margin-left: auto; }}
  #map {{ position: fixed; top: 45px; bottom: 240px; left: 0; right: 0; }}
  #chart-panel {{
    position: fixed; bottom: 0; left: 0; right: 0; height: 232px;
    background: #fff; border-top: 1px solid #e5e7eb; padding: 10px 16px 8px;
    display: flex; flex-direction: column; gap: 4px;
  }}
  #chart-panel h2 {{ font-size: 11px; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }}
  #chart-wrap {{ flex: 1; min-height: 0; }}
  .popup-table {{ border-collapse: collapse; width: 100%; font-size: 11px; }}
  .popup-table td {{ padding: 2px 4px; }}
  .popup-table tr:nth-child(even) {{ background: #f9fafb; }}
  .legend-box {{
    background: #fff; padding: 8px 12px; border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12); font-size: 11px; line-height: 1.8;
  }}
</style>
</head>
<body>
<div id="titlebar">
  <h1><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
      background:{color};margin-right:6px;vertical-align:middle"></span>
    {r.persona_display}</h1>
  <span style="font-size:11px;color:#9ca3af">{route_name} · {run_id}</span>
  <div class="nav-bar">{nav_links}
    <a href="route_comparison_map.html" style="display:inline-flex;align-items:center;gap:4px;
        padding:3px 8px;border-radius:4px;font-size:11px;text-decoration:none;
        background:#f3f4f6;color:#374151">All →</a>
  </div>
</div>
<div id="map"></div>
<div id="chart-panel">
  <h2>Score Trend — {r.persona_display}</h2>
  <div id="chart-wrap"><canvas id="score-chart"></canvas></div>
</div>
<script>
const RAW = {data_js};
{_MAP_JS_TEMPLATE}

const layerGroup = L.layerGroup();
const p = {{...RAW.persona, markers: RAW.markers}};
renderPersonaLayer(p, layerGroup);
layerGroup.addTo(map);

// Start / end markers
if (RAW.markers.length > 0) {{
  const first = RAW.markers[0], last = RAW.markers[RAW.markers.length - 1];
  L.circleMarker([first.lat, first.lng], {{
    radius: 8, color: RAW.persona.color, fillColor: '#fff', fillOpacity: 1, weight: 3
  }}).bindTooltip('Start').addTo(map);
  L.circleMarker([last.lat, last.lng], {{
    radius: 8, color: RAW.persona.color, fillColor: RAW.persona.color, fillOpacity: 0.9, weight: 3
  }}).bindTooltip('End').addTo(map);
}}

// Divergent branch markers
RAW.divergent_markers.forEach(function(d) {{
  const icon = L.divIcon({{
    html: '<div style="font-size:18px;line-height:1;filter:drop-shadow(0 1px 2px rgba(0,0,0,.4))">⚡</div>',
    className: '', iconAnchor: [9, 9]
  }});
  L.marker([d.lat, d.lng], {{icon}})
    .bindPopup(`<b>Divergent intersection</b><br><small>${{d.pano_id}}</small><br><br>${{d.choices_text}}`)
    .addTo(map);
}});

// Score legend
const legend = L.control({{position: 'bottomleft'}});
legend.onAdd = function() {{
  const div = L.DomUtil.create('div', 'legend-box');
  div.innerHTML = `{_SCORE_LEGEND_HTML}`;
  return div;
}};
legend.addTo(map);

// Score trend chart
(function() {{
  const pts = RAW.timeline;
  if (!pts.length) return;
  new Chart(document.getElementById('score-chart'), {{
    type: 'line',
    data: {{
      datasets: [{{
        label: RAW.persona.display,
        data: pts.map(p => ({{ x: p.step, y: p.avg }})),
        borderColor: RAW.persona.color,
        backgroundColor: RAW.persona.color + '22',
        borderWidth: 2, pointRadius: 3, tension: 0.3,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => 'avg: ' + ctx.parsed.y }} }} }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Step', font: {{ size: 10 }} }}, grid: {{ color: '#f3f4f6' }} }},
        y: {{ min: 1, max: 10, title: {{ display: true, text: 'Score', font: {{ size: 10 }} }}, grid: {{ color: '#f3f4f6' }} }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""


def _build_branch_markers_data(r: "PersonaRun") -> list[dict]:
    """Per-step branch decision data for map rendering."""
    out = []
    for s in r.steps:
        if not s.get("branch_triggered"):
            continue
        if not isinstance(s.get("lat"), (int, float)):
            continue
        candidates = []
        for c in s.get("branch_candidates", []):
            c_scores = c.get("scores", {})
            c_vals = [v for v in c_scores.values() if isinstance(v, (int, float))]
            candidates.append({
                "direction": c.get("direction", "?"),
                "heading": c.get("heading", 0),
                "avg_score": round(sum(c_vals) / len(c_vals), 2) if c_vals else None,
                "chosen": bool(c.get("chosen")),
            })
        out.append({
            "lat": s["lat"],
            "lng": s["lng"],
            "step": s.get("step"),
            "candidate_count": s.get("candidate_count"),
            "recommendation": (s.get("recommendation") or "")[:100],
            "candidates": candidates,
        })
    return out


def make_route_map_html(runs: list[PersonaRun], cmp_report: dict,
                        branch_div: dict, out_path: Path) -> None:
    """Write route_comparison_map.html — grid of one Leaflet map per persona."""
    run_id = cmp_report.get("run_id", "")
    route_name = _route_name(cmp_report)
    center_lat, center_lng = _map_center(runs)

    persona_layers: list[dict] = []
    for i, r in enumerate(runs):
        color = _persona_color(i)
        markers = _build_persona_markers(r, color)
        dest = r.metadata.get("parameters", {}).get("destination")
        persona_layers.append({
            "key": r.persona_key,
            "display": r.persona_display,
            "color": color,
            "markers": markers,
            "branches": _build_branch_markers_data(r),
            "destination": dest,  # {lat, lng} or None
            "nav_route": r.walk_log.get("planner_waypoints", []),  # current nav route
            "nav_route_history": r.walk_log.get("planner_route_history", []),  # past routes before reroute
        })

    data_js = json.dumps({
        "persona_layers": persona_layers,
        "center": [center_lat, center_lng],
        "run_id": run_id,
        "route_name": route_name,
    }, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{route_name} — Route Comparison</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               background: #f3f4f6; overflow: hidden; }}

  /* ── Top bar ── */
  #titlebar {{
    position: fixed; top: 0; left: 0; right: 0; height: 44px; z-index: 2000;
    background: white; border-bottom: 1px solid #e5e7eb;
    display: flex; align-items: center; padding: 0 16px; gap: 14px;
  }}
  #titlebar h1 {{ font-size: 13px; font-weight: 600; color: #111; white-space: nowrap; }}
  #titlebar .meta {{ font-size: 11px; color: #9ca3af; white-space: nowrap; }}
  .slider-group {{
    display: flex; align-items: center; gap: 6px;
    margin-left: auto; flex-shrink: 0;
  }}
  .slider-group + .slider-group {{ margin-left: 10px; }}
  .slider-label {{ font-size: 11px; color: #6b7280; white-space: nowrap; }}
  .slider-val {{ font-size: 11px; font-weight: 600; color: #111; width: 18px; text-align: right; }}
  input[type=range] {{
    -webkit-appearance: none; width: 80px; height: 4px;
    background: #e5e7eb; border-radius: 2px; outline: none; cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: #6366f1; cursor: pointer;
  }}

  /* ── Map grid (includes overlay cell as first item) ── */
  #grid {{
    position: fixed; top: 44px; bottom: 0; left: 0; right: 0;
    display: grid; gap: 2px; background: #e5e7eb;
  }}

  /* ── Individual map cell ── */
  .map-cell {{
    position: relative; background: white; overflow: hidden;
  }}
  .cell-label {{
    position: absolute; top: 8px; left: 8px; z-index: 500;
    background: white; border-radius: 6px;
    padding: 3px 9px 3px 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,.15);
    display: flex; align-items: center; gap: 5px;
    font-size: 12px; font-weight: 600; color: #111;
    pointer-events: none;
  }}
  .cell-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}

  /* ── Shared legend style ── */
  .map-legend {{
    background: white; padding: 8px 12px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,.15); font-size: 11px; line-height: 1.8;
  }}
  .popup-table {{ border-collapse: collapse; width: 100%; font-size: 11px; }}
  .popup-table td {{ padding: 2px 4px; }}
  .popup-table tr:nth-child(even) {{ background: #f9fafb; }}
</style>
</head>
<body>
<div id="titlebar">
  <h1>{route_name}</h1>
  <span class="meta">{run_id} · {len(runs)} personas</span>
  <div class="slider-group">
    <span class="slider-label">Waypoint</span>
    <input type="range" id="sl-wp" min="1" max="12" step="0.5" value="3">
    <span class="slider-val" id="sl-wp-val">3</span>
  </div>
  <div class="slider-group">
    <span class="slider-label">Branch</span>
    <input type="range" id="sl-br" min="2" max="16" step="0.5" value="8">
    <span class="slider-val" id="sl-br-val">8</span>
  </div>
  <div class="slider-group">
    <span class="slider-label">Offset</span>
    <input type="range" id="sl-off" min="0" max="12" step="0.5" value="4">
    <span class="slider-val" id="sl-off-val">4</span>
  </div>
</div>
<div id="grid"></div>

<script>
const RAW = {data_js};

function scoreColor(avg) {{
  if (avg >= 8) return '#22c55e';
  if (avg >= 6) return '#84cc16';
  if (avg >= 5) return '#eab308';
  if (avg >= 4) return '#f97316';
  return '#ef4444';
}}

// ── Slider state ─────────────────────────────────────────────────────────────
const slWp  = document.getElementById('sl-wp');
const slBr  = document.getElementById('sl-br');
const slOff = document.getElementById('sl-off');
const slWpV = document.getElementById('sl-wp-val');
const slBrV = document.getElementById('sl-br-val');
const slOffV = document.getElementById('sl-off-val');
let wpScale  = parseFloat(slWp.value);
let brScale  = parseFloat(slBr.value);
let offScale = parseFloat(slOff.value);  // meters of perpendicular offset per persona slot

// ── Parallel-offset helper ────────────────────────────────────────────────────
// Shifts a [lat,lng] coordinate by `meters` perpendicular to the segment [A→B].
// Used to visually separate overlapping routes in the overlay map.
function offsetCoord(lat, lng, perpMeters) {{
  // 1 degree lat ≈ 111320 m; 1 degree lng ≈ 111320 * cos(lat) m
  const mPerDegLat = 111320;
  const mPerDegLng = 111320 * Math.cos(lat * Math.PI / 180);
  return [lat + perpMeters / mPerDegLat, lng + perpMeters / mPerDegLng];
}}

// Apply perpendicular offset to a polyline coords array.
// segDir: [dLat, dLng] direction of the segment (un-normalised).
// perpMeters: signed distance (positive = left of travel direction).
function applyOffset(lat, lng, dLat, dLng, perpMeters) {{
  if (perpMeters === 0 || (dLat === 0 && dLng === 0)) return [lat, lng];
  // Perpendicular direction (rotate 90° CCW): (-dLng, dLat) in lat/lng space
  // Normalise accounting for degree-to-metre scaling
  const mPerLat = 111320;
  const mPerLng = 111320 * Math.cos(lat * Math.PI / 180);
  const dy = dLat * mPerLat;   // segment in metres
  const dx = dLng * mPerLng;
  const len = Math.sqrt(dy * dy + dx * dx);
  if (len < 1e-9) return [lat, lng];
  // Perpendicular unit vector (rotate 90° left): (-dx, dy) / len
  const px = -dx / len;  // in metres, lng direction
  const py =  dy / len;  // in metres, lat direction
  return [lat + (perpMeters * py) / mPerLat,
          lng + (perpMeters * px) / mPerLng];
}}

// Compute a single global perpendicular offset vector for a marker array.
// Uses the overall start→end bearing of the route so all points shift by
// the same (dLat, dLng) — no kinking at segment joints.
function globalOffsetCoords(markers, perpMeters) {{
  const valid = markers.filter(m => m.lat != null && m.lng != null);
  if (valid.length < 2 || perpMeters === 0) return valid.map(m => [m.lat, m.lng]);
  const first = valid[0], last = valid[valid.length - 1];
  const mPerLat = 111320;
  const mPerLng = 111320 * Math.cos(first.lat * Math.PI / 180);
  const dy = (last.lat - first.lat) * mPerLat;
  const dx = (last.lng - first.lng) * mPerLng;
  const len = Math.sqrt(dy * dy + dx * dx);
  // If route is nearly a point, fall back to pure north offset
  const py = len > 1 ? -dx / len : 1;  // perp unit vector (lat direction)
  const px = len > 1 ?  dy / len : 0;  // perp unit vector (lng direction)
  const dLat = (perpMeters * py) / mPerLat;
  const dLng = (perpMeters * px) / mPerLng;
  return valid.map(m => [m.lat + dLat, m.lng + dLng]);
}}

// overlayMap is initialised after the overlay cell div is added to the DOM (see below)
let overlayMap, overlayRouteGroup;

function drawOverlay() {{
  overlayRouteGroup.clearLayers();
  const total = RAW.persona_layers.length;
  const step = offScale;  // metres per slot
  RAW.persona_layers.forEach(function(p, idx) {{
    const slot = idx - (total - 1) / 2;  // centred: -2,-1,0,1,2 for 5 personas
    const perpM = slot * step;
    const ms = p.markers.filter(m => m.lat != null && m.lng != null);
    if (ms.length < 2) return;

    // Shift all points by the same global offset vector — no kinking
    const coords = globalOffsetCoords(ms, perpM);

    // Draw as a single continuous polyline per persona
    // Use persona colour; thickness reflects whether steps are scored
    L.polyline(coords, {{
      color: p.color, weight: 4, opacity: 0.82,
    }}).addTo(overlayRouteGroup);

    // Start / end markers at offset positions
    L.circleMarker(coords[0], {{
      radius: 6, color: p.color, fillColor: '#fff', fillOpacity: 1, weight: 2.5
    }}).bindTooltip(`${{p.display || p.key}} — Start`).addTo(overlayRouteGroup);
    L.circleMarker(coords[coords.length - 1], {{
      radius: 6, color: p.color, fillColor: p.color, fillOpacity: 0.9, weight: 2.5
    }}).bindTooltip(`${{p.display || p.key}} — End`).addTo(overlayRouteGroup);
  }});

  // ── Destination (shared — draw once from first persona that has it) ───────
  const destPersona = RAW.persona_layers.find(p => p.destination && p.destination.lat != null);
  if (destPersona) {{
    const d = destPersona.destination;
    const destIcon = L.divIcon({{
      className: '',
      html: `<div style="
        width:22px;height:22px;border-radius:50% 50% 50% 0;
        background:#ef4444;border:2px solid white;
        transform:rotate(-45deg);
        box-shadow:0 2px 6px rgba(0,0,0,.35)"></div>`,
      iconSize: [22, 22], iconAnchor: [11, 22],
    }});
    L.marker([d.lat, d.lng], {{ icon: destIcon }})
      .bindTooltip('Destination')
      .addTo(overlayRouteGroup);
  }}
}}

// ── Grid layout (n personas + 1 overlay cell = n+1 total) ────────────────────
const n = RAW.persona_layers.length;
const total_cells = n + 1;
const cols = total_cells <= 2 ? total_cells : total_cells <= 4 ? 2 : 3;
const rows = Math.ceil(total_cells / cols);
const grid = document.getElementById('grid');
grid.style.gridTemplateColumns = `repeat(${{cols}}, 1fr)`;
grid.style.gridTemplateRows = `repeat(${{rows}}, 1fr)`;

// ── Overlay cell (first in grid) ──────────────────────────────────────────────
(function() {{
  const cell = document.createElement('div');
  cell.className = 'map-cell';
  const mapDiv = document.createElement('div');
  mapDiv.id = 'overlay-map';
  mapDiv.style.cssText = 'position:absolute;inset:0;';
  cell.appendChild(mapDiv);
  const lbl = document.createElement('div');
  lbl.className = 'cell-label';
  lbl.innerHTML = `<span class="cell-dot" style="background:linear-gradient(135deg,#3b82f6,#a855f7)"></span>All Routes`;
  cell.appendChild(lbl);
  grid.appendChild(cell);

  // Initialise Leaflet NOW that the div is in the DOM
  overlayMap = L.map(mapDiv, {{ zoomControl: false, attributionControl: false }})
    .setView(RAW.center, 16);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    subdomains: 'abcd', maxZoom: 20
  }}).addTo(overlayMap);
  L.control.zoom({{ position: 'topright' }}).addTo(overlayMap);

  const legendCtrl = L.control({{ position: 'bottomleft' }});
  legendCtrl.onAdd = function() {{
    const div = L.DomUtil.create('div', 'map-legend');
    div.innerHTML = RAW.persona_layers.map(function(p) {{
      return `<span style="color:${{p.color}}">&#9644;</span> ${{p.display || p.key}}`;
    }}).join('<br>');
    return div;
  }};
  legendCtrl.addTo(overlayMap);

  overlayRouteGroup = L.layerGroup().addTo(overlayMap);
}})();

// ── Per-map state (map instance + marker LayerGroup) ──────────────────────────
const mapStates = [];  // {{ map, markerGroup, p }}

RAW.persona_layers.forEach(function(p, idx) {{
  // Cell
  const cell = document.createElement('div');
  cell.className = 'map-cell';
  const mapDiv = document.createElement('div');
  mapDiv.id = 'map-' + idx;
  mapDiv.style.cssText = 'position:absolute;inset:0;';
  cell.appendChild(mapDiv);
  const lbl = document.createElement('div');
  lbl.className = 'cell-label';
  lbl.innerHTML = `<span class="cell-dot" style="background:${{p.color}}"></span>${{p.display || p.key}}`;
  cell.appendChild(lbl);
  grid.appendChild(cell);

  // Leaflet map
  const map = L.map(mapDiv, {{ zoomControl: false, attributionControl: false }})
    .setView(RAW.center, 16);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    subdomains: 'abcd', maxZoom: 20
  }}).addTo(map);
  L.control.zoom({{ position: 'topright' }}).addTo(map);

  // ── Static layers: polylines (never change size) ──────────────────────────
  for (let i = 1; i < p.markers.length; i++) {{
    const prev = p.markers[i-1], curr = p.markers[i];
    if (curr.avg != null) {{
      L.polyline([[prev.lat, prev.lng],[curr.lat, curr.lng]], {{
        color: scoreColor(curr.avg), weight: 5, opacity: 0.85
      }}).addTo(map);
    }} else if (prev.lat != null) {{
      L.polyline([[prev.lat, prev.lng],[curr.lat, curr.lng]], {{
        color: '#94a3b8', weight: 2, opacity: 0.4, dashArray: '4,6'
      }}).addTo(map);
    }}
  }}

  // Branch arrows (static — only circle size changes)
  (p.branches || []).forEach(function(b) {{
    b.candidates.forEach(function(c) {{
      const rad = c.heading * Math.PI / 180;
      const len = 0.0003;
      const cosLat = Math.cos(b.lat * Math.PI / 180);
      L.polyline([[b.lat, b.lng],
        [b.lat + len * Math.cos(rad), b.lng + len * Math.sin(rad) / cosLat]], {{
        color: c.avg_score != null ? scoreColor(c.avg_score) : '#94a3b8',
        weight: c.chosen ? 4 : 2, opacity: c.chosen ? 0.9 : 0.4,
        dashArray: c.chosen ? null : '5,4',
      }}).addTo(map);
    }});
  }});

  // ── Navigation routes (dashed, faint) ────────────────────────────────────
  function drawNavRoute(coords, map, opacity) {{
    if (!coords || coords.length < 2) return;
    L.polyline(coords, {{
      color: '#6366f1', weight: 2, opacity: opacity,
      dashArray: '6,5', interactive: false,
    }}).addTo(map);
  }}
  (p.nav_route_history || []).forEach(function(wps) {{
    drawNavRoute(wps, map, 0.25);
  }});
  drawNavRoute(p.nav_route || [], map, 0.45);

  // ── Destination marker ────────────────────────────────────────────────────
  if (p.destination && p.destination.lat != null) {{
    const destIcon = L.divIcon({{
      className: '',
      html: `<div style="
        width:22px;height:22px;border-radius:50% 50% 50% 0;
        background:#ef4444;border:2px solid white;
        transform:rotate(-45deg);
        box-shadow:0 2px 6px rgba(0,0,0,.35)"></div>`,
      iconSize: [22, 22], iconAnchor: [11, 22],
    }});
    L.marker([p.destination.lat, p.destination.lng], {{ icon: destIcon }})
      .bindTooltip('Destination')
      .addTo(map);
  }}

  // Score legend (first map only)
  if (idx === 0) {{
    const legend = L.control({{position: 'bottomleft'}});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'map-legend');
      div.innerHTML =
        '<b style="font-size:11px">Score</b><br>' +
        '<span style="color:#22c55e">●</span> 8–10 Excellent<br>' +
        '<span style="color:#84cc16">●</span> 6–8 Good<br>' +
        '<span style="color:#eab308">●</span> 5–6 Average<br>' +
        '<span style="color:#f97316">●</span> 4–5 Below avg<br>' +
        '<span style="color:#ef4444">●</span> &lt;4 Poor<br>' +
        '<span style="color:#94a3b8">●</span> Unscored<br>' +
        '<span style="color:#7c3aed">●</span> Branch';
      return div;
    }};
    legend.addTo(map);
  }}

  // Marker group (replaced on slider change)
  const markerGroup = L.layerGroup().addTo(map);
  mapStates.push({{ map, markerGroup, p }});
}});

// ── Draw markers with current scale values ────────────────────────────────────
function drawMarkers() {{
  mapStates.forEach(function({{ map, markerGroup, p }}) {{
    markerGroup.clearLayers();

    // Waypoint markers
    p.markers.forEach(function(m) {{
      if (!m.has_score && !m.is_s2) return;
      const baseR = m.avg != null ? 1 + m.avg * 0.3 : 3;  // base: ~4–4 at avg5
      const radius = baseR * (wpScale / 3);               // scale=3 → 1× natural size
      const opts = m.is_s2
        ? {{ radius, color: '#111', fillColor: p.color, fillOpacity: 0.9, weight: 1.5, dashArray: '4,2' }}
        : {{ radius, color: scoreColor(m.avg ?? 5), fillColor: scoreColor(m.avg ?? 5), fillOpacity: 0.75, weight: 1 }};
      const dimRows = Object.entries(m.scores)
        .filter(([_, v]) => v != null)
        .map(([d, v]) => `<tr><td>${{d}}</td><td>${{v}}</td></tr>`).join('');
      const popup = `<b>Step ${{m.step}}</b><br><small>${{m.pano_id}}</small>
        <table class="popup-table" style="margin-top:4px">
          <tr><td>avg</td><td><b>${{m.avg ?? '—'}}</b></td></tr>${{dimRows}}
        </table>
        ${{m.excerpt ? `<div style="margin-top:5px;font-size:10px;color:#555;max-width:240px">${{m.excerpt}}</div>` : ''}}`;
      L.circleMarker([m.lat, m.lng], opts).bindPopup(popup, {{maxWidth:280}}).addTo(markerGroup);
    }});

    // Start / End markers (scale with waypoint slider)
    const seR = Math.max(4, wpScale * 1.5);
    if (p.markers.length > 0) {{
      const first = p.markers[0];
      L.circleMarker([first.lat, first.lng], {{
        radius: seR, color: p.color, fillColor: '#fff', fillOpacity: 1, weight: 2.5
      }}).bindTooltip('Start').addTo(markerGroup);
    }}
    if (p.markers.length > 1) {{
      const last = p.markers[p.markers.length - 1];
      L.circleMarker([last.lat, last.lng], {{
        radius: seR, color: p.color, fillColor: p.color, fillOpacity: 0.9, weight: 2.5
      }}).bindTooltip('End').addTo(markerGroup);
    }}

    // Branch circles (separate slider)
    (p.branches || []).forEach(function(b) {{
      const popup = `<b>Step ${{b.step}}</b> — Branch<br>` +
        `Candidates: ${{b.candidate_count ?? b.candidates.length}}<br>` +
        (b.recommendation ? `<div style="font-size:10px;color:#555;margin-top:3px">${{b.recommendation}}</div>` : '');
      L.circleMarker([b.lat, b.lng], {{
        radius: brScale, color: '#7c3aed', fillColor: '#7c3aed', fillOpacity: 0.7, weight: 2
      }}).bindPopup(popup, {{maxWidth:260}}).addTo(markerGroup);
    }});
  }});
}}

// ── Slider events ─────────────────────────────────────────────────────────────
slWp.addEventListener('input', function() {{
  wpScale = parseFloat(this.value);
  slWpV.textContent = this.value;
  drawMarkers();
}});
slBr.addEventListener('input', function() {{
  brScale = parseFloat(this.value);
  slBrV.textContent = this.value;
  drawMarkers();
}});
slOff.addEventListener('input', function() {{
  offScale = parseFloat(this.value);
  slOffV.textContent = this.value;
  drawOverlay();
}});

drawMarkers();
drawOverlay();

// Force Leaflet to recalculate sizes after layout paint
requestAnimationFrame(function() {{
  overlayMap.invalidateSize();
  mapStates.forEach(function(s) {{ s.map.invalidateSize(); }});
}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  [map] grid map ({len(runs)} personas) → {out_path}")


# -----------------------------------------------------------------------------
# Output 4 — comparison_dashboard.html
# -----------------------------------------------------------------------------

def _find_pano_image_b64(pano_id: str, runs: list[PersonaRun]) -> str | None:
    """Find a saved image for a pano_id across all persona folders.
    Matches full pano_id or a prefix (filenames may truncate long IDs).
    Returns base64 data-URI string or None."""
    import base64
    # Truncated safe filename prefix used by image_collector (first 8 chars typically)
    prefix = pano_id[:8]
    for r in runs:
        img_dir = r.folder / "images"
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.iterdir()):
            fname = img_file.name
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if pano_id in fname or prefix in fname:
                try:
                    b64 = base64.b64encode(img_file.read_bytes()).decode()
                    mime = "image/jpeg" if img_file.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    return f"data:{mime};base64,{b64}"
                except Exception:
                    pass
    return None


def _build_branch_decisions_by_pano(runs: list[PersonaRun]) -> dict[str, dict]:
    """For each pano_id with a branch decision, collect candidate info from the
    first persona that has it, plus per-persona chosen direction."""
    result: dict[str, dict] = {}
    # Collect candidates (same per pano regardless of persona)
    for r in runs:
        for bd in r.branch_decisions:
            pano = bd.get("pano_id")
            if not pano or pano in result:
                continue
            result[pano] = {
                "lat": bd.get("lat"),
                "lng": bd.get("lng"),
                "candidates": [
                    {
                        "direction": c.get("direction"),
                        "heading": c.get("heading"),
                        "avg_score": c.get("avg_score"),
                        "chosen": bool(c.get("chosen")),
                        "next_pano_id": None,
                    }
                    for c in bd.get("candidates", [])
                ],
                "persona_choices": {},
            }

    # Build (branch_pano, direction) → next_pano_id by examining each persona's walk_log
    # When a persona takes a branch step, steps[i+1].pano_id is the chosen direction's first pano
    direction_to_next: dict[tuple[str, str], str] = {}
    for r in runs:
        steps = r.steps
        for i, s in enumerate(steps):
            if not s.get("branch_triggered"):
                continue
            pano = s.get("pano_id")
            if not pano or pano not in result:
                continue
            for bd in r.branch_decisions:
                if bd.get("pano_id") != pano:
                    continue
                chosen_c = next((c for c in bd.get("candidates", []) if c.get("chosen")), None)
                if chosen_c is None:
                    break
                direction = chosen_c.get("direction")
                if direction and i + 1 < len(steps):
                    next_pano = steps[i + 1].get("pano_id")
                    if next_pano:
                        direction_to_next[(pano, direction)] = next_pano
                break

    # Back-fill next_pano_id into candidates
    for pano, info in result.items():
        for c in info["candidates"]:
            key = (pano, c["direction"])
            if key in direction_to_next:
                c["next_pano_id"] = direction_to_next[key]

    # Fill per-persona choices
    for r in runs:
        for bd in r.branch_decisions:
            pano = bd.get("pano_id")
            if not pano or pano not in result:
                continue
            chosen = next((c for c in bd.get("candidates", []) if c.get("chosen")), None)
            direction = chosen.get("direction") if chosen else None
            result[pano]["persona_choices"][r.persona_key] = {
                "direction": direction,
                "heading": bd.get("chosen_heading"),
                "next_pano_id": direction_to_next.get((pano, direction)) if direction else None,
            }
    return result


def make_dashboard_html(data_dump: dict, runs: list[PersonaRun],
                        cmp_report: dict, out_path: Path) -> None:
    run_id = data_dump.get("run_id", "")
    route_name = _route_name(cmp_report)
    generated_at = cmp_report.get("generated_at", "")[:10]
    n_personas = len(data_dump.get("personas", []))

    # Embed per-step trajectory data (step, avg_score, reasoning excerpt) per persona
    trajectory_data: list[dict] = []
    for i, r in enumerate(runs):
        points = []
        for s in r.analyzed_steps:
            points.append({
                "step": s.get("step"),
                "avg": round(_avg_score_for_step(s), 2),
                "excerpt": _reasoning_excerpt(s, 180),
            })
        trajectory_data.append({
            "key": r.persona_key,
            "display": r.persona_display,
            "color": _persona_color(i),
            "points": points,
        })

    # S2 trigger timeline with trigger_reason colors
    s2_data: list[dict] = []
    trigger_reason_palette = {
        "intersection": "#7c3aed",
        "heading_delta": "#3b82f6",
        "distance_milestone": "#f97316",
        "score_drop": "#ef4444",
        "phash_change": "#22c55e",
    }
    for i, r in enumerate(runs):
        events = []
        for s in r.s2_steps:
            tr = s.get("trigger_reason", "other")
            events.append({
                "step": s.get("step"),
                "reason": tr,
                "color": trigger_reason_palette.get(tr, "#6b7280"),
            })
        s2_data.append({
            "key": r.persona_key,
            "color": _persona_color(i),
            "events": events,
        })

    # Embed divergent intersection images (base64) if --save-image was used
    branch_div = data_dump.get("quantitative", {}).get("branch_divergence", {})
    divergent_images: dict[str, str] = {}
    for div in branch_div.get("divergent_detail", []):
        pano = div.get("pano_id", "")
        if pano and pano not in divergent_images:
            img = _find_pano_image_b64(pano, runs)
            if img:
                divergent_images[pano] = img

    branch_decisions_by_pano = _build_branch_decisions_by_pano(runs)

    # Inline all data as a single JS const
    js_data = json.dumps({
        "run_id": run_id,
        "route_name": route_name,
        "generated_at": generated_at,
        "personas": data_dump.get("personas", []),
        "quantitative": data_dump.get("quantitative", {}),
        "persona_lenses": data_dump.get("persona_lenses", {}),
        "branch_comparisons": data_dump.get("branch_comparisons", []),
        "synthesis": data_dump.get("synthesis"),
        "trajectory": trajectory_data,
        "s2_timeline": s2_data,
        "persona_colors": {r.persona_key: _persona_color(i) for i, r in enumerate(runs)},
        "divergent_images": divergent_images,
        "divergent_detail": branch_div.get("divergent_detail", []),
        "branch_decisions_by_pano": branch_decisions_by_pano,
        "gmaps_key": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
    }, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{route_name} — Comparison Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .section {{ border-top: 1px solid #e5e7eb; padding-top: 2rem; margin-top: 2rem; }}
  .dim-chip {{ display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 500; margin: 2px; }}
  canvas {{ max-height: 320px; }}
  #s2-strip {{ overflow-x: auto; }}
  .s2-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
  .s2-label {{ width: 120px; font-size: 12px; color: #6b7280; flex-shrink: 0; }}
  .s2-track {{ position: relative; height: 18px; flex: 1; background: #f3f4f6; border-radius: 4px; }}
  .s2-dot {{
    position: absolute; top: 50%; transform: translate(-50%,-50%);
    width: 10px; height: 10px; border-radius: 50%;
  }}
</style>
</head>
<body class="bg-white text-gray-900 antialiased">

<div class="max-w-5xl mx-auto px-6 py-10">

  <!-- Header -->
  <div class="flex items-start justify-between mb-8">
    <div>
      <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-1">CityWalkAgent</p>
      <h1 class="text-2xl font-bold tracking-tight" id="hdr-route"></h1>
      <p class="text-sm text-gray-500 mt-1">
        Run <code id="hdr-run" class="text-xs bg-gray-100 px-1 py-0.5 rounded"></code>
        &nbsp;·&nbsp; <span id="hdr-personas"></span> personas
        &nbsp;·&nbsp; <span id="hdr-date"></span>
      </p>
    </div>
    <a href="route_comparison_map.html"
       class="text-xs border border-gray-300 rounded px-3 py-1.5 hover:bg-gray-50 whitespace-nowrap">
      Open map →
    </a>
  </div>

  <!-- Thesis card (hidden if no synthesis) -->
  <div id="thesis-card" class="mb-10 hidden">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-3">Divergence Thesis</p>
    <p id="thesis-text" class="text-xl font-semibold leading-snug text-center max-w-3xl mx-auto mb-4"></p>
    <blockquote id="paper-para"
      class="border-l-2 border-gray-300 pl-4 text-sm text-gray-600 italic max-w-2xl mx-auto"></blockquote>
  </div>

  <!-- Quant strip -->
  <div class="section">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Dimension Divergence</p>
    <div id="dim-cards" class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-2"></div>
  </div>

  <!-- Score trajectory -->
  <div class="section">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Score Trajectory</p>
    <canvas id="chart-trajectory"></canvas>
  </div>

  <!-- Dimension bar chart -->
  <div class="section">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Dimension Scores by Persona</p>
    <canvas id="chart-dims"></canvas>
  </div>

  <!-- Persona lens cards -->
  <div class="section">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Persona Lenses</p>
    <div id="lens-cards" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
  </div>

  <!-- S2 trigger timeline -->
  <div class="section">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">System-2 Trigger Timeline</p>
    <div id="s2-strip"></div>
    <div class="flex flex-wrap gap-3 mt-3 text-xs text-gray-500" id="s2-legend"></div>
  </div>

  <!-- Branch divergence -->
  <div id="branch-section" class="section hidden">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Branch Decision Divergence</p>
    <div id="branch-cards" class="space-y-4"></div>
  </div>

  <!-- Divergent intersection street views -->
  <div id="divimg-section" class="section hidden">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Divergent Intersections — Street View</p>
    <div id="divimg-cards" class="space-y-8"></div>
  </div>

  <!-- Dimension splits -->
  <div id="splits-section" class="section hidden">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Dimension Splits</p>
    <div id="splits-cards" class="grid grid-cols-2 md:grid-cols-4 gap-3"></div>
  </div>

  <!-- Path overlap table -->
  <div class="section">
    <p class="text-xs font-medium tracking-widest text-gray-400 uppercase mb-4">Path Overlap (Jaccard)</p>
    <div class="overflow-x-auto">
      <table id="jaccard-table" class="text-xs w-full border-collapse"></table>
    </div>
    <p class="text-xs text-gray-400 mt-2">Values &lt; 0.4 highlighted. 1.0 = identical path.</p>
  </div>

  <div class="mt-12 pt-6 border-t border-gray-100 text-xs text-gray-400">
    Generated by CityWalkAgent · <a href="https://github.com/imkykim/CityWalkAgent" class="underline">imkykim/CityWalkAgent</a>
  </div>

</div>

<script>
const DATA = {js_data};

// ── Header ──────────────────────────────────────────────────────────────────
document.getElementById('hdr-route').textContent = DATA.route_name;
document.getElementById('hdr-run').textContent = DATA.run_id;
document.getElementById('hdr-personas').textContent = DATA.personas.length;
document.getElementById('hdr-date').textContent = DATA.generated_at;

// ── Thesis card ─────────────────────────────────────────────────────────────
if (DATA.synthesis) {{
  document.getElementById('thesis-card').classList.remove('hidden');
  document.getElementById('thesis-text').textContent = DATA.synthesis.divergence_thesis || '';
  document.getElementById('paper-para').textContent = DATA.synthesis.paper_paragraph || '';
}}

// ── Dimension cards ─────────────────────────────────────────────────────────
const DIMS = ['safety','lively','beautiful','wealthy'];
const dimGap = (DATA.quantitative.dim_divergence || {{}}).per_dim_gap || {{}};
const dimContainer = document.getElementById('dim-cards');
DIMS.forEach(d => {{
  const gap = dimGap[d] ?? 0;
  const pct = Math.min(gap / 4, 1);  // normalize: 4 = max possible gap
  const r = Math.round(220 * pct), g = Math.round(220 * (1 - pct));
  const color = `rgb(${{r}},${{g}},60)`;
  dimContainer.innerHTML += `
    <div class="border border-gray-200 rounded p-4 text-center">
      <p class="text-xs text-gray-400 uppercase tracking-wide mb-1">${{d}}</p>
      <p class="text-2xl font-bold" style="color:${{color}}">${{gap.toFixed(2)}}</p>
      <p class="text-xs text-gray-400 mt-1">max − min gap</p>
    </div>`;
}});

// ── Score trajectory chart ───────────────────────────────────────────────────
(function() {{
  const datasets = DATA.trajectory.map(p => ({{
    label: p.key,
    data: p.points.map(pt => ({{ x: pt.step, y: pt.avg, excerpt: pt.excerpt }})),
    borderColor: p.color,
    backgroundColor: p.color + '22',
    borderWidth: 2,
    pointRadius: 3,
    tension: 0.3,
  }}));
  new Chart(document.getElementById('chart-trajectory'), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y}}`,
            afterLabel: ctx => {{
              const excerpt = ctx.raw.excerpt;
              if (!excerpt) return '';
              return excerpt.length > 80 ? excerpt.slice(0, 80) + '…' : excerpt;
            }}
          }}
        }},
        legend: {{ labels: {{ font: {{ size: 11 }} }} }}
      }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Step', font: {{ size: 11 }} }},
               grid: {{ color: '#f3f4f6' }} }},
        y: {{ min: 1, max: 10, title: {{ display: true, text: 'Avg score', font: {{ size: 11 }} }},
               grid: {{ color: '#f3f4f6' }} }}
      }}
    }}
  }});
}})();

// ── Dimension bar chart ──────────────────────────────────────────────────────
(function() {{
  const dimMeans = (DATA.quantitative.dim_divergence || {{}}).persona_dim_means || {{}};
  const personas = DATA.personas;
  const dimColors = ['#3b82f6','#f97316','#22c55e','#a855f7'];
  const datasets = DIMS.map((d, i) => ({{
    label: d,
    data: personas.map(p => (dimMeans[d] || {{}})[p] ?? 0),
    backgroundColor: dimColors[i] + 'cc',
    borderColor: dimColors[i],
    borderWidth: 1,
  }}));
  new Chart(document.getElementById('chart-dims'), {{
    type: 'bar',
    data: {{ labels: personas, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }},
        y: {{ min: 0, max: 10, grid: {{ color: '#f3f4f6' }} }}
      }}
    }}
  }});
}})();

// ── Persona lens cards ───────────────────────────────────────────────────────
(function() {{
  const container = document.getElementById('lens-cards');
  DATA.trajectory.forEach(p => {{
    const lens = (DATA.persona_lenses || {{}})[p.key] || {{}};
    const emphasized = (lens.features_emphasized || []).map(f =>
      `<span class="dim-chip bg-gray-100 text-gray-700">${{f}}</span>`).join('');
    const downweighted = (lens.features_downweighted || []).map(f =>
      `<span class="text-xs text-gray-400">${{f}}</span>`).join(', ');
    const avgScore = p.points.length
      ? (p.points.reduce((s,pt) => s + pt.avg, 0) / p.points.length).toFixed(2)
      : '—';
    container.innerHTML += `
      <div class="border border-gray-200 rounded p-4">
        <div class="flex items-center gap-2 mb-2">
          <span style="background:${{p.color}}" class="w-2.5 h-2.5 rounded-full flex-shrink-0"></span>
          <span class="font-semibold text-sm">${{p.display || p.key}}</span>
          <span class="ml-auto text-xs text-gray-400">avg ${{avgScore}}</span>
        </div>
        ${{emphasized ? `<div class="mb-2">${{emphasized}}</div>` : ''}}
        ${{downweighted ? `<p class="text-xs text-gray-400 mb-2">↓ ${{downweighted}}</p>` : ''}}
        ${{lens.affective_stance ? `<p class="text-xs text-gray-600 mb-1">${{lens.affective_stance}}</p>` : ''}}
        ${{lens.signature_phrase ? `<p class="text-xs italic text-gray-500">"${{lens.signature_phrase}}"</p>` : ''}}
      </div>`;
  }});
}})();

// ── S2 trigger timeline ──────────────────────────────────────────────────────
(function() {{
  const strip = document.getElementById('s2-strip');
  const maxStep = Math.max(...DATA.s2_timeline.flatMap(p =>
    p.events.map(e => e.step || 0)), 1);

  DATA.s2_timeline.forEach(p => {{
    const dotsHtml = p.events.map(e => {{
      const pct = (e.step / maxStep * 100).toFixed(1);
      return `<div class="s2-dot" style="left:${{pct}}%;background:${{e.color}}" title="Step ${{e.step}}: ${{e.reason}}"></div>`;
    }}).join('');
    strip.innerHTML += `
      <div class="s2-row">
        <span class="s2-label">
          <span style="color:${{p.color}}">■</span> ${{p.key}}
        </span>
        <div class="s2-track">${{dotsHtml}}</div>
      </div>`;
  }});

  // Trigger reason legend
  const reasons = {{}};
  DATA.s2_timeline.forEach(p => p.events.forEach(e => reasons[e.reason] = e.color));
  const legEl = document.getElementById('s2-legend');
  Object.entries(reasons).forEach(([r, c]) => {{
    legEl.innerHTML += `<span><span style="color:${{c}}">●</span> ${{r}}</span> `;
  }});
}})();

// ── Branch divergence ────────────────────────────────────────────────────────
(function() {{
  const bcs = DATA.branch_comparisons || [];
  if (bcs.length === 0) return;
  document.getElementById('branch-section').classList.remove('hidden');
  const container = document.getElementById('branch-cards');
  bcs.forEach(bc => {{
    const rationales = Object.entries(bc.persona_rationales || {{}})
      .map(([p, why]) => `<li><b>${{p}}</b>: ${{why}}</li>`).join('');
    container.innerHTML += `
      <div class="border border-gray-200 rounded p-4">
        <p class="text-xs font-mono text-gray-400 mb-1">${{bc.pano_id}}</p>
        <p class="text-sm font-medium mb-2">${{bc.divergence_summary || ''}}</p>
        ${{bc.shared_environment_signal
          ? `<p class="text-xs text-gray-500 mb-2">Shared signal: ${{bc.shared_environment_signal}}</p>`
          : ''}}
        <ul class="text-xs text-gray-600 space-y-0.5 list-none pl-0">${{rationales}}</ul>
      </div>`;
  }});
}})();

// ── Divergent intersection Street Views ───────────────────────────────────────
(function() {{
  const GMAPS_KEY = DATA.gmaps_key || '';
  const detail = DATA.divergent_detail || [];
  const bdByPano = DATA.branch_decisions_by_pano || {{}};

  // Collect all panos that have branch decision data with lat/lng
  const panoSet = detail.filter(d => {{
    const bd = bdByPano[d.pano_id];
    return bd && bd.lat != null && bd.candidates && bd.candidates.length > 0;
  }}).map(d => d.pano_id);

  if (panoSet.length === 0) return;
  document.getElementById('divimg-section').classList.remove('hidden');
  const container = document.getElementById('divimg-cards');

  panoSet.forEach(pano => {{
    const divInfo = detail.find(d => d.pano_id === pano);
    const bd = bdByPano[pano];
    const candidates = bd.candidates || [];
    const personaChoices = bd.persona_choices || {{}};

    // Which personas chose each direction?
    const dirPersonas = {{}};
    Object.entries(personaChoices).forEach(([persona, info]) => {{
      const dir = info.direction;
      if (!dir) return;
      (dirPersonas[dir] = dirPersonas[dir] || []).push(persona);
    }});

    // ── One Street View iframe per candidate direction ────────────────────
    const svPanels = candidates.map(c => {{
      const dir = c.direction;
      const heading = c.heading;
      const chosen = c.chosen;
      const choosers = dirPersonas[dir] || [];
      const scoreStr = c.avg_score != null ? c.avg_score.toFixed(1) : '—';

      // Persona dots HTML
      const dotHtml = choosers.map(p => {{
        const col = DATA.persona_colors[p] || '#6b7280';
        return `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:${{col}};border:1.5px solid white;margin-right:2px;
          box-shadow:0 1px 3px rgba(0,0,0,.25)"></span>`;
      }}).join('');

      // Direction label style — chosen gets accent border
      const borderStyle = chosen
        ? 'border: 2px solid #2563eb;'
        : 'border: 2px solid #e5e7eb;';

      // Street View embed URL — prefer next_pano_id (first pano after entering this road)
      // so the view shows looking forward from inside the chosen road, not just the intersection
      const nextPano = c.next_pano_id;
      const svUrl = GMAPS_KEY
        ? nextPano
          ? `https://www.google.com/maps/embed/v1/streetview?key=${{GMAPS_KEY}}&pano=${{encodeURIComponent(nextPano)}}&heading=${{heading}}&pitch=0&fov=90`
          : `https://www.google.com/maps/embed/v1/streetview?key=${{GMAPS_KEY}}&pano=${{encodeURIComponent(pano)}}&heading=${{heading}}&pitch=0&fov=90`
        : '';

      const label = `
        <div style="position:absolute;top:10px;left:10px;z-index:10;
          background:${{chosen ? '#2563eb' : 'rgba(0,0,0,0.65)'}};
          color:white;border-radius:6px;padding:4px 9px;
          font-size:13px;font-weight:700;pointer-events:none;
          box-shadow:0 2px 6px rgba(0,0,0,.3)">
          ${{dir}}
        </div>`;

      const scoreTag = `
        <div style="position:absolute;top:10px;right:10px;z-index:10;
          background:rgba(255,255,255,0.92);border-radius:6px;
          padding:3px 8px;font-size:11px;font-weight:600;color:#111;
          pointer-events:none;box-shadow:0 1px 4px rgba(0,0,0,.15)">
          ${{scoreStr}}
        </div>`;

      const personaBar = choosers.length > 0 ? `
        <div style="position:absolute;bottom:10px;left:10px;z-index:10;
          background:rgba(255,255,255,0.92);border-radius:20px;
          padding:3px 8px;display:flex;align-items:center;gap:4px;
          font-size:10px;color:#374151;pointer-events:none;
          box-shadow:0 1px 4px rgba(0,0,0,.15)">
          ${{dotHtml}}
          <span>${{choosers.map(p=>p.replace('_',' ')).join(', ')}}</span>
        </div>` : '';

      const iframeOrMsg = svUrl
        ? `<iframe src="${{svUrl}}" width="100%" height="300"
            style="display:block;border:none;" loading="lazy"
            allowfullscreen referrerpolicy="no-referrer-when-downgrade"></iframe>`
        : `<div style="height:300px;display:flex;align-items:center;justify-content:center;
            background:#f3f4f6;color:#9ca3af;font-size:12px">
            No API key — set GOOGLE_MAPS_API_KEY</div>`;

      return `
        <div style="flex:1;min-width:0;position:relative;${{borderStyle}}border-radius:10px;overflow:hidden;">
          ${{iframeOrMsg}}
          ${{label}}
          ${{scoreTag}}
          ${{personaBar}}
        </div>`;
    }}).join('');

    // ── Caption row ──────────────────────────────────────────────────────
    const captionBadges = divInfo
      ? Object.entries(divInfo.choices || {{}}).map(([persona, info]) => {{
          const col = DATA.persona_colors[persona] || '#6b7280';
          return `<span style="display:inline-flex;align-items:center;gap:3px;
            margin:2px 4px 2px 0;font-size:11px;padding:2px 8px;
            border-radius:9999px;background:${{col}}18;color:${{col}};font-weight:500">
            ${{persona.replace('_',' ')}}&nbsp;→&nbsp;<b>${{info.chosen_direction || '?'}}</b>
          </span>`;
        }}).join('') : '';

    container.innerHTML += `
      <div>
        <p class="text-xs font-mono text-gray-400 mb-3">${{pano}}</p>
        <div style="display:flex;gap:8px;align-items:stretch;">
          ${{svPanels}}
        </div>
        <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:2px;">
          ${{captionBadges}}
        </div>
      </div>`;
  }});
}})();

// ── Dimension splits ─────────────────────────────────────────────────────────
(function() {{
  const syn = DATA.synthesis;
  if (!syn || !syn.dimension_splits) return;
  document.getElementById('splits-section').classList.remove('hidden');
  const container = document.getElementById('splits-cards');
  DIMS.forEach(d => {{
    container.innerHTML += `
      <div class="border border-gray-200 rounded p-4">
        <p class="text-xs text-gray-400 uppercase tracking-wide mb-1">${{d}}</p>
        <p class="text-xs text-gray-700">${{syn.dimension_splits[d] || '—'}}</p>
      </div>`;
  }});
}})();

// ── Path overlap table ───────────────────────────────────────────────────────
(function() {{
  const pairs = ((DATA.quantitative.path_divergence || {{}}).pairs) || [];
  if (pairs.length === 0) return;
  const table = document.getElementById('jaccard-table');
  let html = '<thead><tr class="text-left border-b border-gray-200">'
    + '<th class="py-1 pr-4 font-medium">Persona A</th>'
    + '<th class="py-1 pr-4 font-medium">Persona B</th>'
    + '<th class="py-1 pr-4 font-medium">Jaccard</th>'
    + '<th class="py-1 font-medium">Shared cells</th>'
    + '</tr></thead><tbody>';
  pairs.forEach(p => {{
    const divergent = p.jaccard < 0.4;
    html += `<tr class="${{divergent ? 'bg-red-50' : ''}} border-b border-gray-100">
      <td class="py-1 pr-4">${{p.persona_a}}</td>
      <td class="py-1 pr-4">${{p.persona_b}}</td>
      <td class="py-1 pr-4 ${{divergent ? 'font-semibold text-red-600' : ''}}">${{p.jaccard.toFixed(3)}}</td>
      <td class="py-1">${{p.shared}}</td>
    </tr>`;
  }});
  table.innerHTML = html + '</tbody>';
}})();
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _route_name(cmp_report: dict) -> str:
    """Best-effort route name from comparison_report."""
    by_route = cmp_report.get("by_route", {})
    if by_route:
        first_route = next(iter(by_route.values()), {})
        name = first_route.get("route_name", "")
        if name:
            return name
    return cmp_report.get("run_id", "Unknown Route")


# -----------------------------------------------------------------------------
# Report assembly (markdown)
# -----------------------------------------------------------------------------

def render_markdown(run_dir: Path, cmp_report: dict, runs: list[PersonaRun],
                    quant: dict, lens_summaries: dict,
                    branch_comparisons: list[dict],
                    synthesis: dict | None, lang: str) -> str:
    H = "##"
    lines: list[str] = []
    lines.append("# Cross-Persona Comparative Analysis")
    lines.append(f"*Run: `{run_dir.name}` — {len(runs)} personas*\n")

    if synthesis:
        lines.append(f"{H} Divergence Thesis")
        lines.append(f"**{synthesis.get('divergence_thesis', '—')}**\n")
        lines.append(f"{H} Paper Paragraph (draft)")
        lines.append(f"> {synthesis.get('paper_paragraph', '—')}\n")

    lines.append(f"{H} Quantitative Divergence")
    dd = quant["dim_divergence"]
    lines.append("| Dimension | Persona gap (max − min) | Persona std |")
    lines.append("|---|---|---|")
    for d in DIMENSIONS:
        lines.append(f"| {d} | {dd['per_dim_gap'].get(d, 0):.2f} | {dd['per_dim_std'].get(d, 0):.2f} |")
    lines.append("")

    lines.append(f"{H} Per-Persona Lens")
    for r in runs:
        lines.append(f"### {r.persona_display} (`{r.persona_key}`)")
        lines.append(f"- Avg score: **{r.avg_score:.2f}** · "
                     f"S2 triggers: {len(r.s2_steps)} · "
                     f"Branch decisions: {len(r.branch_decisions)} · "
                     f"Reroutes: {r.walk_log.get('reroute_count', 0)}")
        lens = lens_summaries.get(r.persona_key)
        if lens:
            lines.append(f"- **Emphasizes**: {', '.join(lens.get('features_emphasized', []))}")
            lines.append(f"- **Downweights**: {', '.join(lens.get('features_downweighted', []))}")
            lines.append(f"- **Stance**: {lens.get('affective_stance', '—')}")
            if lens.get("signature_phrase"):
                lines.append(f"- *\"{lens['signature_phrase']}\"*")
        lines.append("")

    if branch_comparisons:
        lines.append(f"{H} Branch Decision Divergence")
        for bc in branch_comparisons:
            lines.append(f"### Intersection `{bc['pano_id']}`")
            lines.append(f"- {bc.get('divergence_summary', '—')}")
            lines.append(f"- Shared environment signal: {bc.get('shared_environment_signal', '—')}")
            for pk, why in bc.get("persona_rationales", {}).items():
                lines.append(f"  - **{pk}**: {why}")
            lines.append("")

    if synthesis:
        lines.append(f"{H} Dimension-by-Dimension Splits")
        splits = synthesis.get("dimension_splits", {})
        for d in DIMENSIONS:
            lines.append(f"- **{d}**: {splits.get(d, '—')}")
        lines.append("")
        lines.append(f"{H} Path Story")
        lines.append(synthesis.get("path_story", "—"))
        lines.append("")

    lines.append(f"{H} Path Overlap (pairwise Jaccard)")
    lines.append("| A | B | Jaccard | Shared cells |")
    lines.append("|---|---|---|---|")
    for p in quant["path_divergence"]["pairs"]:
        lines.append(f"| {p['persona_a']} | {p['persona_b']} | {p['jaccard']:.2f} | {p['shared']} |")
    lines.append("")
    lines.append("![Plots](./divergence_plots.png)\n")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Quantitative only; no LLM calls")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    assert run_dir.exists(), f"Not found: {run_dir}"

    print(f"[load] {run_dir}")
    cmp_report, exp_matrix, runs = load_run_data(run_dir)
    print(f"[load] {len(runs)} persona runs: {[r.persona_key for r in runs]}")

    if len(runs) < 2:
        print("[warn] Fewer than 2 personas found — comparison is degenerate.")

    # Quantitative
    print("[quant] dimension divergence...")
    dim_div = compute_dimension_divergence(runs)
    dim_div.pop("step_std_df", None)

    print("[quant] path divergence...")
    path_div = compute_path_divergence(runs)

    print("[quant] s2 alignment...")
    s2_align = compute_s2_alignment(runs)

    print("[quant] branch divergence...")
    branch_div = compute_branch_divergence(runs)

    quant = {
        "dim_divergence": dim_div,
        "path_divergence": path_div,
        "s2_alignment": s2_align,
        "branch_divergence": branch_div,
    }

    # Qualitative (LLM)
    lens_summaries: dict = {}
    branch_comparisons: list[dict] = []
    synthesis: dict | None = None

    if not args.skip_llm:
        client = _llm_client()
        if client:
            print(f"[llm] persona lenses ({len(runs)} calls)...")
            for r in runs:
                out = llm_persona_lens_summary(client, r, args.lang)
                if out:
                    lens_summaries[r.persona_key] = out

            print("[llm] branch-point divergence comparisons...")
            branch_comparisons = llm_branch_point_comparison(
                client, branch_div.get("divergent_detail", []), runs, args.lang)

            print("[llm] synthesis narrative...")
            synthesis = llm_synthesis_narrative(
                client, quant, lens_summaries, branch_comparisons, args.lang)

    out_dir = run_dir / "comparison_analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"[plot] → {out_dir}/divergence_plots.png")
    make_plots(runs, quant, out_dir / "divergence_plots.png")

    print(f"[write] → {out_dir}/divergence_data.json")
    data_dump = {
        "run_id": cmp_report.get("run_id"),
        "personas": [r.persona_key for r in runs],
        "quantitative": quant,
        "persona_lenses": lens_summaries,
        "branch_comparisons": branch_comparisons,
        "synthesis": synthesis,
    }
    (out_dir / "divergence_data.json").write_text(
        json.dumps(data_dump, ensure_ascii=False, indent=2))

    print(f"[write] → {out_dir}/qualitative_report.md")
    md = render_markdown(run_dir, cmp_report, runs, quant,
                         lens_summaries, branch_comparisons, synthesis, args.lang)
    (out_dir / "qualitative_report.md").write_text(md)

    print(f"[plot] → {out_dir}/route_comparison.png")
    make_route_png(runs, cmp_report, out_dir / "route_comparison.png")

    print(f"[write] → {out_dir}/route_comparison_map.html")
    make_route_map_html(runs, cmp_report, branch_div, out_dir / "route_comparison_map.html")

    print(f"[write] → {out_dir}/comparison_dashboard.html")
    make_dashboard_html(data_dump, runs, cmp_report, out_dir / "comparison_dashboard.html")

    print("[done]")


if __name__ == "__main__":
    main()
