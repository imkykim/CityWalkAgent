"""
Walk Output Writer — structured experiment outputs for autonomous_walk mode.

Generates from a single walk result:
  1. run_metadata.json    — experiment parameters, timing, summary stats
  2. walk_log.json        — full step-by-step log (existing, unchanged)
  3. score_timeline.csv   — tabular step×dimension scores for easy analysis
  4. branch_decisions.json — extracted branch-only decisions with full context
  5. s2_reasoning.json    — System 2 trigger events with reasoning
  6. route_map.html       — interactive Folium map with scored route + branches
  7. score_chart.png      — 4D score line chart over steps
  8. summary_radar.png    — average scores radar chart

Usage (standalone from orchestrator):
    writer = WalkOutputWriter(output_dir, walk_result, run_params)
    writer.write_all()

Integration (in orchestrator.autonomous_walk):
    from src.utils.walk_output_writer import WalkOutputWriter
    writer = WalkOutputWriter(output_dir, result, params)
    writer.write_all()
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _safe_json_default(obj: Any) -> Any:
    """JSON serializer fallback for non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class WalkOutputWriter:
    """Write structured experiment outputs from an autonomous walk result."""

    def __init__(
        self,
        output_dir: Path,
        walk_result: Dict[str, Any],
        run_params: Dict[str, Any],
        framework_id: str = "place_pulse_2.0",
    ):
        """
        Args:
            output_dir: Directory to write outputs. Auto-created.
            walk_result: The dict returned by autonomous_walk().
            run_params: Experiment parameters dict with keys like:
                start_lat, start_lng, dest_lat, dest_lng,
                persona, max_steps, lookahead_depth,
                start_time (datetime), end_time (datetime), etc.
            framework_id: Evaluation framework ID for viz labels.
        """
        self.output_dir = Path(output_dir)
        self.result = walk_result
        self.params = run_params
        self.framework_id = framework_id
        self.route_taken: List[Dict] = walk_result.get("route_taken", [])

        # Infer dimension keys from first step with scores
        self.dim_keys: List[str] = []
        for step in self.route_taken:
            scores = step.get("scores", {})
            if scores:
                self.dim_keys = sorted(scores.keys())
                break

    # ── public API ───────────────────────────────────────────────────────

    def write_all(self) -> Dict[str, Path]:
        """Write all outputs. Returns dict mapping output name → file path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        paths["run_metadata"] = self._write_run_metadata()
        paths["walk_log"] = self._write_walk_log()
        paths["score_timeline"] = self._write_score_timeline()
        paths["branch_decisions"] = self._write_branch_decisions()
        paths["s2_reasoning"] = self._write_s2_reasoning()

        # Visualizations (graceful degradation if matplotlib unavailable)
        try:
            paths["route_map"] = self._write_route_map()
        except Exception as e:
            print(f"  ⚠ route_map skipped: {e}")

        try:
            paths["score_chart"] = self._write_score_chart()
        except Exception as e:
            print(f"  ⚠ score_chart skipped: {e}")

        try:
            paths["summary_radar"] = self._write_summary_radar()
        except Exception as e:
            print(f"  ⚠ summary_radar skipped: {e}")

        return paths

    # ── 1. run_metadata.json ─────────────────────────────────────────────

    def _write_run_metadata(self) -> Path:
        """Experiment parameters + summary statistics."""
        steps = self.route_taken
        total_steps = len(steps)

        # Score statistics
        all_avg_scores: List[float] = []
        dim_all: Dict[str, List[float]] = {d: [] for d in self.dim_keys}
        branch_count = 0
        s2_trigger_count = 0

        for s in steps:
            scores = s.get("scores", {})
            vals = [v for v in scores.values() if isinstance(v, (int, float))]
            if vals:
                all_avg_scores.append(sum(vals) / len(vals))
            for d in self.dim_keys:
                if d in scores and isinstance(scores[d], (int, float)):
                    dim_all[d].append(scores[d])
            if s.get("branch_triggered"):
                branch_count += 1
            if s.get("trigger_reason"):
                s2_trigger_count += 1

        dim_summary = {}
        for d in self.dim_keys:
            vals = dim_all[d]
            if vals:
                dim_summary[d] = {
                    "mean": round(np.mean(vals), 3),
                    "std": round(np.std(vals), 3),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                }

        # Distance traveled (sum of step-to-step geodesic)
        total_distance_m = 0.0
        for i in range(1, len(steps)):
            prev = steps[i - 1]
            curr = steps[i]
            try:
                from geopy.distance import geodesic
                total_distance_m += geodesic(
                    (prev["lat"], prev["lng"]),
                    (curr["lat"], curr["lng"]),
                ).meters
            except Exception:
                total_distance_m += 20.0  # fallback ~20m per step

        metadata = {
            "experiment": {
                "timestamp": self.params.get("start_time", datetime.now().isoformat()),
                "duration_seconds": self.params.get("duration_seconds"),
                "framework_id": self.framework_id,
            },
            "parameters": {
                "start": {
                    "lat": self.params.get("start_lat"),
                    "lng": self.params.get("start_lng"),
                },
                "destination": {
                    "lat": self.params.get("dest_lat"),
                    "lng": self.params.get("dest_lng"),
                },
                "persona": self.params.get("persona"),
                "max_steps": self.params.get("max_steps"),
                "lookahead_depth": self.params.get("lookahead_depth"),
                "save_images": self.params.get("save_images", False),
            },
            "outcome": {
                "arrived": self.result.get("arrived", False),
                "total_steps": total_steps,
                "final_distance_m": self.result.get("final_distance_m"),
                "total_distance_walked_m": round(total_distance_m, 1),
            },
            "statistics": {
                "branch_decisions": branch_count,
                "s2_triggers": s2_trigger_count,
                "s2_trigger_rate": round(s2_trigger_count / total_steps, 3) if total_steps else 0,
                "avg_score_overall": round(np.mean(all_avg_scores), 3) if all_avg_scores else None,
                "score_std_overall": round(np.std(all_avg_scores), 3) if all_avg_scores else None,
                "dimension_summary": dim_summary,
            },
            "memory_debug_summary": {
                "snapshot_count": len(
                    (self.result.get("memory_debug") or {}).get("snapshots", [])
                ),
                "episode_count": len(
                    (self.result.get("memory_debug") or {}).get("episodes", [])
                ),
            },
        }

        path = self.output_dir / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=_safe_json_default)
        return path

    # ── 2. walk_log.json (existing behavior, preserved) ──────────────────

    def _write_walk_log(self) -> Path:
        """Full walk result — same as current behavior."""
        # Strip image_base64 from the persisted log to save disk space
        clean_result = {**self.result}
        clean_route = []
        for step in clean_result.get("route_taken", []):
            s = {k: v for k, v in step.items() if k != "image_base64"}
            clean_route.append(s)
        clean_result["route_taken"] = clean_route

        path = self.output_dir / "walk_log.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, indent=2, ensure_ascii=False, default=_safe_json_default)
        return path

    # ── 3. score_timeline.csv ────────────────────────────────────────────

    def _write_score_timeline(self) -> Path:
        """Flat CSV: step, lat, lng, dim1, dim2, ..., avg, is_intersection, trigger."""
        path = self.output_dir / "score_timeline.csv"
        fieldnames = [
            "step", "pano_id", "lat", "lng", "heading",
            "dist_to_dest_m", "dest_bearing",
        ] + self.dim_keys + [
            "avg_score", "is_intersection", "branch_triggered",
            "trigger_reason", "confidence", "visit_count",
            "phash_distance", "visual_change",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for step in self.route_taken:
                scores = step.get("scores", {})
                vals = [v for v in scores.values() if isinstance(v, (int, float))]
                row = {
                    "step": step.get("step"),
                    "pano_id": step.get("pano_id"),
                    "lat": step.get("lat"),
                    "lng": step.get("lng"),
                    "heading": step.get("heading"),
                    "dist_to_dest_m": step.get("dist_to_dest_m"),
                    "dest_bearing": step.get("dest_bearing"),
                    "avg_score": round(sum(vals) / len(vals), 3) if vals else "",
                    "is_intersection": step.get("is_intersection", False),
                    "branch_triggered": step.get("branch_triggered", False),
                    "trigger_reason": step.get("trigger_reason", ""),
                    "confidence": step.get("confidence", ""),
                    "visit_count": step.get("visit_count", 0),
                    "phash_distance": step.get("phash_distance", ""),
                    "visual_change": step.get("visual_change", ""),
                }
                for d in self.dim_keys:
                    row[d] = round(scores.get(d, 0), 3) if d in scores else ""
                writer.writerow(row)

        return path

    # ── 4. branch_decisions.json ─────────────────────────────────────────

    def _write_branch_decisions(self) -> Path:
        """Extract only branch decision steps with full candidate data."""
        branches = []
        for step in self.route_taken:
            if not step.get("branch_triggered"):
                continue
            candidates = step.get("branch_candidates", [])
            branches.append({
                "step": step.get("step"),
                "pano_id": step.get("pano_id"),
                "lat": step.get("lat"),
                "lng": step.get("lng"),
                "is_intersection": step.get("is_intersection"),
                "trigger_reason": step.get("trigger_reason"),
                "dist_to_dest_m": step.get("dist_to_dest_m"),
                "dest_bearing": step.get("dest_bearing"),
                "dest_cardinal": step.get("dest_cardinal"),
                "chosen_heading": step.get("heading"),
                "confidence": step.get("confidence"),
                "recommendation": step.get("recommendation"),
                "candidate_count": step.get("candidate_count"),
                "candidates": candidates,
            })

        path = self.output_dir / "branch_decisions.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(branches, f, indent=2, ensure_ascii=False, default=_safe_json_default)
        return path

    # ── 5. s2_reasoning.json ─────────────────────────────────────────────

    def _write_s2_reasoning(self) -> Path:
        """Extract System 2 trigger events: reasoning, scores, context."""
        s2_events = []
        for step in self.route_taken:
            trigger = step.get("trigger_reason")
            if not trigger:
                continue
            s2_events.append({
                "step": step.get("step"),
                "pano_id": step.get("pano_id"),
                "lat": step.get("lat"),
                "lng": step.get("lng"),
                "trigger_reason": trigger,
                "is_intersection": step.get("is_intersection"),
                "scores": step.get("scores", {}),
                "reasoning": step.get("reasoning", {}),
                "recommendation": step.get("recommendation"),
                "confidence": step.get("confidence"),
                "phash_distance": step.get("phash_distance"),
                "visual_change": step.get("visual_change"),
                "dist_to_dest_m": step.get("dist_to_dest_m"),
            })

        path = self.output_dir / "s2_reasoning.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s2_events, f, indent=2, ensure_ascii=False, default=_safe_json_default)
        return path

    # ── 6. route_map.html (Folium) ───────────────────────────────────────

    def _write_route_map(self) -> Path:
        """Interactive Folium map: planner route + scored walked route + branch markers."""
        import folium

        steps = self.route_taken

        def _valid_point(lat: Any, lng: Any) -> bool:
            return isinstance(lat, (int, float)) and isinstance(lng, (int, float))

        planned_route_raw = self.result.get("planner_waypoints", []) or []
        planned_points: List[List[float]] = []
        for pt in planned_route_raw:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2 and _valid_point(pt[0], pt[1]):
                planned_points.append([float(pt[0]), float(pt[1])])

        # Previous routes (before reroutes)
        route_history_raw = self.result.get("planner_route_history", []) or []
        route_history: List[List[List[float]]] = []
        for route in route_history_raw:
            pts = []
            for pt in route:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2 and _valid_point(pt[0], pt[1]):
                    pts.append([float(pt[0]), float(pt[1])])
            if len(pts) >= 2:
                route_history.append(pts)

        step_points = [
            [float(s["lat"]), float(s["lng"])]
            for s in steps
            if isinstance(s, dict) and _valid_point(s.get("lat"), s.get("lng"))
        ]
        anchor_points = step_points + planned_points
        if not anchor_points and _valid_point(self.params.get("start_lat"), self.params.get("start_lng")):
            anchor_points.append([float(self.params["start_lat"]), float(self.params["start_lng"])])
        if not anchor_points and _valid_point(self.params.get("dest_lat"), self.params.get("dest_lng")):
            anchor_points.append([float(self.params["dest_lat"]), float(self.params["dest_lng"])])
        if not anchor_points:
            raise ValueError("No coordinates to map")

        center_lat = np.mean([p[0] for p in anchor_points])
        center_lng = np.mean([p[1] for p in anchor_points])
        m = folium.Map(location=[center_lat, center_lng], zoom_start=16, tiles="cartodbpositron")

        # Color helper
        def _score_color(avg: float) -> str:
            if avg >= 8:
                return "#22c55e"  # green
            elif avg >= 6:
                return "#84cc16"  # lime
            elif avg >= 5:
                return "#eab308"  # yellow
            elif avg >= 4:
                return "#f97316"  # orange
            else:
                return "#ef4444"  # red

        # Previous routes before reroute — shown as dashed lines with labels
        for hist_idx, hist_pts in enumerate(route_history):
            nav_label = f"Navigation {hist_idx + 1}"
            folium.PolyLine(
                hist_pts,
                color="#94a3b8",
                weight=3,
                opacity=0.35,
                dash_array="4,10",
                tooltip=f"{nav_label} (superseded)",
            ).add_to(m)
            # Label at midpoint
            mid = hist_pts[len(hist_pts) // 2]
            folium.Marker(
                mid,
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;color:#94a3b8;white-space:nowrap;'
                         f'background:rgba(255,255,255,0.75);padding:1px 4px;border-radius:3px;'
                         f'border:1px solid #cbd5e1">{nav_label}</div>',
                    icon_size=(90, 20),
                    icon_anchor=(45, 10),
                ),
            ).add_to(m)

        # Current planner route (Directions waypoints): light overlay
        nav_current_label = f"Navigation {len(route_history) + 1}" if route_history else "Navigation"
        if len(planned_points) >= 2:
            folium.PolyLine(
                planned_points,
                color="#94a3b8",
                weight=4,
                opacity=0.45,
                dash_array="6,8",
                tooltip=f"{nav_current_label} (active)",
            ).add_to(m)
            # Label at midpoint
            mid = planned_points[len(planned_points) // 2]
            folium.Marker(
                mid,
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;color:#64748b;white-space:nowrap;'
                         f'background:rgba(255,255,255,0.85);padding:1px 4px;border-radius:3px;'
                         f'border:1px solid #94a3b8;font-weight:600">{nav_current_label}</div>',
                    icon_size=(100, 20),
                    icon_anchor=(50, 10),
                ),
            ).add_to(m)

        # Route segments colored by score
        for i in range(1, len(steps)):
            prev = steps[i - 1]
            curr = steps[i]
            if not (_valid_point(prev.get("lat"), prev.get("lng")) and _valid_point(curr.get("lat"), curr.get("lng"))):
                continue
            scores = curr.get("scores", {})
            vals = [v for v in scores.values() if isinstance(v, (int, float))]
            avg = sum(vals) / len(vals) if vals else 5.0
            color = _score_color(avg)

            folium.PolyLine(
                [[prev["lat"], prev["lng"]], [curr["lat"], curr["lng"]]],
                color=color, weight=5, opacity=0.85,
            ).add_to(m)

        # Start marker
        start_lat = self.params.get("start_lat")
        start_lng = self.params.get("start_lng")
        if not _valid_point(start_lat, start_lng) and step_points:
            start_lat, start_lng = step_points[0]
        if _valid_point(start_lat, start_lng):
            folium.Marker(
                [start_lat, start_lng],
                icon=folium.Icon(color="blue", icon="play", prefix="fa"),
                popup="Start",
            ).add_to(m)

        # End / destination marker
        if _valid_point(self.params.get("dest_lat"), self.params.get("dest_lng")):
            folium.Marker(
                [self.params["dest_lat"], self.params["dest_lng"]],
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                popup="Destination",
            ).add_to(m)

        # Last actual position
        if step_points:
            last = steps[-1]
            if _valid_point(last.get("lat"), last.get("lng")):
                folium.Marker(
                    [last["lat"], last["lng"]],
                    icon=folium.Icon(
                        color="green" if self.result.get("arrived") else "orange",
                        icon="stop" if self.result.get("arrived") else "pause",
                        prefix="fa",
                    ),
                    popup=f"Final (dist={last.get('dist_to_dest_m', '?')}m)",
                ).add_to(m)

        # Branch decision points
        for step in steps:
            if not step.get("branch_triggered"):
                continue
            candidates = step.get("branch_candidates", [])
            chosen = next((c for c in candidates if c.get("chosen")), None)
            popup_lines = [
                f"<b>Step {step['step']}</b> — Branch Decision",
                f"Candidates: {step.get('candidate_count', '?')}",
            ]
            if chosen:
                popup_lines.append(
                    f"Chosen: {chosen['direction']} "
                    f"(avg {chosen.get('avg_score', '?')})"
                )
            if step.get("recommendation"):
                popup_lines.append(f"Reason: {step['recommendation'][:80]}...")

            folium.CircleMarker(
                [step["lat"], step["lng"]],
                radius=8, color="#7c3aed", fill=True,
                fill_color="#7c3aed", fill_opacity=0.7,
                popup=folium.Popup("<br>".join(popup_lines), max_width=300),
            ).add_to(m)

            # Draw candidate direction arrows
            for c in candidates:
                heading_rad = np.radians(c.get("heading", 0))
                arrow_len = 0.0003  # ~30m in degrees
                end_lat = step["lat"] + arrow_len * np.cos(heading_rad)
                end_lng = step["lng"] + arrow_len * np.sin(heading_rad) / np.cos(
                    np.radians(step["lat"])
                )
                c_scores = c.get("scores", {})
                c_vals = [v for v in c_scores.values() if isinstance(v, (int, float))]
                c_avg = sum(c_vals) / len(c_vals) if c_vals else 5.0

                folium.PolyLine(
                    [[step["lat"], step["lng"]], [end_lat, end_lng]],
                    color=_score_color(c_avg),
                    weight=4 if c.get("chosen") else 2,
                    opacity=0.9 if c.get("chosen") else 0.4,
                    dash_array="6" if not c.get("chosen") else None,
                ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                    background:white; padding:10px 14px; border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,.15); font-size:12px;">
            <b>Score Legend</b><br>
            <span style="color:#22c55e">●</span> 8–10 Excellent<br>
            <span style="color:#84cc16">●</span> 6–8 Good<br>
            <span style="color:#eab308">●</span> 5–6 Average<br>
            <span style="color:#f97316">●</span> 4–5 Below avg<br>
            <span style="color:#ef4444">●</span> &lt;4 Poor<br>
            <span style="color:#94a3b8">●</span> Planner navigation<br>
            <span style="color:#7c3aed">●</span> Branch decision
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        path = self.output_dir / "route_map.html"
        m.save(str(path))
        return path

    # ── 7. score_chart.png ───────────────────────────────────────────────

    def _write_score_chart(self) -> Path:
        """Line chart of 4D scores + avg over steps, branch points marked."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.route_taken or not self.dim_keys:
            raise ValueError("No score data to chart")

        steps_x = [s.get("step", i) for i, s in enumerate(self.route_taken)]
        dim_series: Dict[str, List[float]] = {d: [] for d in self.dim_keys}
        avg_series: List[float] = []
        branch_steps: List[int] = []

        for i, step in enumerate(self.route_taken):
            scores = step.get("scores", {})
            vals = []
            for d in self.dim_keys:
                v = scores.get(d)
                if isinstance(v, (int, float)):
                    dim_series[d].append(v)
                    vals.append(v)
                else:
                    dim_series[d].append(np.nan)
            avg_series.append(np.nanmean(vals) if vals else np.nan)
            if step.get("branch_triggered"):
                branch_steps.append(steps_x[i])

        # Color palette
        colors = {
            "safe": "#3b82f6",
            "lively": "#f97316",
            "beautiful": "#22c55e",
            "wealthy": "#a855f7",
            "safety": "#3b82f6",
        }

        fig, ax = plt.subplots(figsize=(14, 5), dpi=120)

        for d in self.dim_keys:
            ax.plot(
                steps_x, dim_series[d],
                marker=".", markersize=3, linewidth=1.2,
                alpha=0.7, label=d.capitalize(),
                color=colors.get(d, None),
            )
        ax.plot(
            steps_x, avg_series,
            linewidth=2, linestyle="--", color="#111", alpha=0.8,
            label="Average",
        )

        # Mark branch decisions
        for bx in branch_steps:
            ax.axvline(x=bx, color="#7c3aed", alpha=0.3, linewidth=1, linestyle=":")

        ax.set_xlabel("Step")
        ax.set_ylabel("Score (1–10)")
        ax.set_title(
            f"Score Timeline — {self.params.get('persona', 'objective')} "
            f"({len(self.route_taken)} steps)",
            fontweight="bold",
        )
        ax.set_ylim(1, 10)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "score_chart.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 8. summary_radar.png ─────────────────────────────────────────────

    def _write_summary_radar(self) -> Path:
        """Radar chart of average dimension scores."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.dim_keys:
            raise ValueError("No dimensions for radar")

        # Compute averages
        dim_avgs: Dict[str, float] = {}
        for d in self.dim_keys:
            vals = []
            for step in self.route_taken:
                v = step.get("scores", {}).get(d)
                if isinstance(v, (int, float)):
                    vals.append(v)
            dim_avgs[d] = np.mean(vals) if vals else 0.0

        labels = [d.capitalize() for d in self.dim_keys]
        values = [dim_avgs[d] for d in self.dim_keys]
        values += values[:1]  # close the polygon

        angles = np.linspace(0, 2 * np.pi, len(self.dim_keys), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=120)
        ax.plot(angles, values, "o-", linewidth=2, color="#3b82f6")
        ax.fill(angles, values, alpha=0.15, color="#3b82f6")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_title(
            f"Average Scores — {self.params.get('persona', 'objective')}",
            fontsize=13, fontweight="bold", pad=20,
        )
        plt.tight_layout()

        path = self.output_dir / "summary_radar.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path


# ── Convenience function for orchestrator integration ────────────────────

def save_walk_outputs(
    output_dir: Path,
    walk_result: Dict[str, Any],
    run_params: Dict[str, Any],
    framework_id: str = "place_pulse_2.0",
) -> Dict[str, Path]:
    """One-liner for orchestrator: save all walk outputs.

    Args:
        output_dir: Target directory.
        walk_result: Return value of autonomous_walk().
        run_params: Dict with start_lat, start_lng, dest_lat, dest_lng,
                    persona, max_steps, lookahead_depth, start_time, duration_seconds, etc.
        framework_id: Evaluation framework.

    Returns:
        Dict mapping output name → file path.
    """
    writer = WalkOutputWriter(output_dir, walk_result, run_params, framework_id)
    return writer.write_all()
