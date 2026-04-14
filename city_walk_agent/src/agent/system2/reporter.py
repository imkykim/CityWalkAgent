"""Reporter — Record & Report (Step 4 of System 2)

Three phases, called at different points in the walk lifecycle:
- generate_snapshot(): Before S2 pipeline (per trigger) — no LLM
- generate_episode(): After S2 pipeline (per trigger) — LLM 1 call
- report_route(): After walk completes (once) — LLM 1 call
"""

from __future__ import annotations
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger
from .llm_client import call_llm


class Reporter:
    """Record & Report — generates snapshots, episodes, and route reports."""

    def __init__(self, framework_id: str) -> None:
        self.framework_id = framework_id
        self.logger = get_logger(self.__class__.__name__)

    # ================================================================
    # Phase 1: Snapshot generation (before Interpreter)
    # ================================================================

    def generate_snapshot(
        self,
        stm_context: Dict[str, Any],
        last_snapshot_waypoint: int,
        current_waypoint_id: int,
    ) -> Dict[str, Any]:
        """Generate route snapshot from STM data since last snapshot.

        Pure statistics — no LLM call.

        Returns:
            {
                "span_start": int,
                "span_end": int,
                "trend": "stable"|"improving"|"declining"|"volatile",
                "avg": {dim: float},
                "trajectory": float,
                "worst_dimension": str|None,
                "barrier_segments": [(start_wp, end_wp), ...],
                "gating_signal_counts": {signal: int},
                "urgency_tier_counts": {tier: int},
                "reroute_count": int,
            }
        """
        telemetry = stm_context.get("telemetry", {}) if isinstance(stm_context, dict) else {}
        gating_signal_counts = telemetry.get("gating_signal_counts", {}) if isinstance(telemetry, dict) else {}
        urgency_tier_counts = telemetry.get("urgency_tier_counts", {}) if isinstance(telemetry, dict) else {}
        reroute_count = telemetry.get("reroute_count", 0) if isinstance(telemetry, dict) else 0

        if not stm_context or not stm_context.get("recent_scores"):
            return {
                "span_start": last_snapshot_waypoint,
                "span_end": current_waypoint_id,
                "trend": "stable",
                "avg": {},
                "trajectory": 0.0,
                "worst_dimension": None,
                "barrier_segments": [],
                "gating_signal_counts": gating_signal_counts,
                "urgency_tier_counts": urgency_tier_counts,
                "reroute_count": reroute_count,
            }

        waypoint_ids = stm_context.get("waypoint_ids", [])
        recent_scores = stm_context["recent_scores"]

        # Filter to items since last snapshot
        span_items = [
            (wid, scores)
            for wid, scores in zip(waypoint_ids, recent_scores)
            if wid >= last_snapshot_waypoint
        ]
        if not span_items:
            span_items = list(zip(waypoint_ids, recent_scores))

        span_ids = [wid for wid, _ in span_items]
        all_scores = [scores for _, scores in span_items]
        n = len(all_scores)

        wp_avgs = [sum(s.values()) / len(s) for s in all_scores if s]

        if len(wp_avgs) >= 2:
            mid = max(1, n // 2)
            first_half_avg = sum(wp_avgs[:mid]) / mid
            second_half_avg = sum(wp_avgs[mid:]) / max(1, n - mid)
            diff = second_half_avg - first_half_avg
            if abs(diff) < 0.3:
                trend = "stable"
            elif diff > 0:
                trend = "improving"
            else:
                trend = "declining"
            try:
                if statistics.stdev(wp_avgs) > 1.5:
                    trend = "volatile"
            except Exception:
                pass
            trajectory = round(wp_avgs[-1] - wp_avgs[0], 2)
        else:
            trend = "stable"
            trajectory = 0.0

        dimensions = list(all_scores[0].keys()) if all_scores else []
        per_dim_avg: Dict[str, float] = {}
        for dim in dimensions:
            dim_scores = [s.get(dim, 0.0) for s in all_scores]
            per_dim_avg[dim] = round(sum(dim_scores) / len(dim_scores), 2)

        worst_dim = min(per_dim_avg, key=per_dim_avg.get) if per_dim_avg else None

        # Barrier segments: consecutive waypoints with avg < 4.5
        barrier_segments: List[Tuple[int, int]] = []
        seg_start = None
        for i, avg in enumerate(wp_avgs):
            wp_id = span_ids[i]
            if avg < 4.5:
                if seg_start is None:
                    seg_start = wp_id
            else:
                if seg_start is not None and i >= 1:
                    end_id = span_ids[i - 1]
                    if end_id != seg_start:
                        barrier_segments.append((seg_start, end_id))
                seg_start = None
        if seg_start is not None:
            barrier_segments.append((seg_start, span_ids[-1]))

        return {
            "span_start": span_ids[0] if span_ids else last_snapshot_waypoint,
            "span_end": current_waypoint_id,
            "trend": trend,
            "avg": per_dim_avg,
            "trajectory": trajectory,
            "worst_dimension": worst_dim,
            "barrier_segments": barrier_segments,
            "gating_signal_counts": gating_signal_counts,
            "urgency_tier_counts": urgency_tier_counts,
            "reroute_count": reroute_count,
        }

    # ================================================================
    # Phase 2: Episode generation (after Interpreter + Decider)
    # ================================================================

    def generate_episode(
        self,
        waypoint_id: int,
        interpretation: Dict[str, Any],
        decision: Dict[str, Any],
        system1_scores: Dict[str, float],
        personality: Any,
        trigger_reason: Any,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured episode record from S2 pipeline results.

        Single LLM call to produce narrative + significance_tag + persona_reaction.

        Returns:
            {
                "waypoint_id": int,
                "significance": str,
                "avoid": bool,
                "interpretation": str,
                "key_concern": str|None,
                "score_change_reason": str|None,
                "scores": {dim: float},
                "trigger_reason": str,
                "narrative": str,
                "significance_tag": str,
                "persona_reaction": str,
                "snapshot_trend": str,
            }
        """
        episode: Dict[str, Any] = {
            "waypoint_id": waypoint_id,
            "significance": decision.get("significance", "medium"),
            "avoid": decision.get("avoid", False),
            "interpretation": interpretation.get("text", ""),
            "key_concern": interpretation.get("key_concern"),
            "score_change_reason": interpretation.get("score_change_reason"),
            "scores": system1_scores,
            "trigger_reason": getattr(trigger_reason, "value", str(trigger_reason)) if trigger_reason else "unknown",
            "snapshot_trend": snapshot.get("trend", "stable"),
        }

        persona_name = getattr(personality, "name", "Unknown")
        avg_score = sum(system1_scores.values()) / len(system1_scores) if system1_scores else 5.0

        prompt = f"""You are recording a walking experience episode for a specific persona.

Persona: {persona_name}
Waypoint: {waypoint_id}
Average score: {avg_score:.1f}/10
Trend in this segment: {snapshot.get("trend", "stable")}
Interpretation: {interpretation.get("text", "")}
Key concern: {interpretation.get("key_concern", "none")}
Decision: {"AVOID" if decision.get("avoid") else "ACCEPTABLE"} (significance: {decision.get("significance", "medium")})

Generate a concise episode record. Respond ONLY with valid JSON:
{{
  "narrative": "1-2 sentences describing this moment in the walk, written as a walking journal entry",
  "significance_tag": "one of: transition, barrier, highlight, routine",
  "persona_reaction": "max 15 chars, first-person casual reaction from the persona, e.g. 'Love this path!' or '여기 위험해...'"
}}"""

        result = call_llm(prompt, max_tokens=200)
        if result:
            episode["narrative"] = result.get("narrative", "")
            episode["significance_tag"] = result.get("significance_tag", "routine")
            episode["persona_reaction"] = result.get("persona_reaction", "")
        else:
            episode["narrative"] = interpretation.get("text", "")[:100]
            episode["significance_tag"] = "barrier" if decision.get("avoid") else "routine"
            episode["persona_reaction"] = ""

        return episode

    # ================================================================
    # Phase 3: Route-level report (after walk completes)
    # ================================================================

    def report_route(
        self,
        snapshots: List[Dict[str, Any]],
        episodes: List[Dict[str, Any]],
        route_stats: Dict[str, Any],
        planner_summary: Dict[str, Any],
        personality: Any,
        waypoints_index: Optional[Dict[int, Dict[str, Any]]] = None,
        walk_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive route report from accumulated data.

        Single LLM call. Called once at end of autonomous_walk().

        Returns:
            {
                "executive_summary": str,
                "score_narrative": str,
                "segment_highlights": [{"waypoint_id": int, "type": str, "description": str}],
                "dimension_verdicts": {dim: str},
                "recommendation": str,
                "recommendation_reason": str,
                "navigation_note": str,
                "persona_suitability": str,
            }
        """
        persona_name = getattr(personality, "name", "Unknown")
        persona_desc = getattr(personality, "description", "")

        snapshot_lines = []
        for s in snapshots:
            traj = s.get("trajectory", 0.0)
            traj_str = f"+{traj:.1f}" if traj >= 0 else f"{traj:.1f}"
            dim_str = " ".join(f"{k}={v:.1f}" for k, v in s.get("avg", {}).items())
            barrier = " ⚠barrier" if s.get("barrier_segments") else ""
            snapshot_lines.append(
                f"WP{s['span_start']}-{s['span_end']}: {s['trend'].upper()} {traj_str} [{dim_str}]{barrier}"
            )
        snapshots_text = "\n".join(snapshot_lines) if snapshot_lines else "No snapshots recorded."

        sorted_eps = sorted(
            episodes,
            key=lambda e: {"high": 0, "medium": 1, "low": 2}.get(e.get("significance", "low"), 2),
        )
        episode_lines = []
        for ep in sorted_eps[:5]:
            avoid_str = " ⚠AVOID" if ep.get("avoid") else ""
            episode_lines.append(
                f"WP{ep['waypoint_id']} [{ep.get('significance_tag', 'routine').upper()}{avoid_str}]: "
                f"{ep.get('narrative', ep.get('interpretation', ''))[:80]}"
            )
        episodes_text = "\n".join(episode_lines) if episode_lines else "No episodes recorded."

        stats_text = (
            f"Steps: {route_stats.get('steps', '?')} (analyzed: {route_stats.get('analyzed_steps', '?')}, "
            f"skip rate: {route_stats.get('skip_rate', 0):.0%})\n"
            f"Arrived: {'Yes' if route_stats.get('arrived') else 'No'} "
            f"(final distance: {route_stats.get('final_distance_m', '?')}m)\n"
            f"Duration: {route_stats.get('duration_seconds', 0):.0f}s"
        )

        dim_avgs = route_stats.get("dimension_avgs", {})
        dim_avg_text = " ".join(f"{k}={v:.1f}" for k, v in dim_avgs.items()) if dim_avgs else "N/A"

        nav_text = (
            f"Re-routes: {planner_summary.get('reroute_count', 0)}, "
            f"Route coverage: {planner_summary.get('route_coverage_pct', 0):.0%}, "
            f"Has Directions route: {'Yes' if planner_summary.get('has_route') else 'No'}"
        )

        prompt = f"""You are writing a comprehensive walking route evaluation report for a specific persona.

Persona: {persona_name}
Description: {persona_desc}

== Route Statistics ==
{stats_text}
Overall dimension averages: {dim_avg_text}

== Navigation ==
{nav_text}

== Segment Snapshots ==
{snapshots_text}

== Key Episodes ==
{episodes_text}

Write a comprehensive evaluation. Respond ONLY with valid JSON:
{{
  "executive_summary": "3-4 sentences: overall impression of this walk from the persona's perspective",
  "score_narrative": "2-3 sentences: how the walking experience quality changed throughout the route",
  "segment_highlights": [
    {{"waypoint_id": <int>, "type": "highlight|barrier|transition", "description": "one sentence"}}
  ],
  "dimension_verdicts": {{
    "safety": "one sentence verdict",
    "lively": "one sentence verdict",
    "beautiful": "one sentence verdict",
    "wealthy": "one sentence verdict"
  }},
  "recommendation": "recommended|acceptable|avoid",
  "recommendation_reason": "one sentence",
  "navigation_note": "one sentence about the navigation experience",
  "persona_suitability": "one sentence: why this route is or isn't suitable for this persona"
}}"""

        self.logger.info("Generating route-level report via LLM")
        result = call_llm(prompt, max_tokens=800)

        if not (result and "executive_summary" in result):
            self.logger.warning("Route report LLM failed — using fallback")
            result = self._report_route_fallback(route_stats, dim_avgs, snapshots, episodes)
        else:
            self.logger.info(f"Route report generated: recommendation={result.get('recommendation')}")

        # Build snapshots (frontend shape)
        snapshots_out = [
            {
                "span_start": s.get("span_start"),
                "span_end": s.get("span_end"),
                "trend": s.get("trend", "stable"),
                "trajectory": s.get("trajectory", 0.0),
                "barrier_segments": [list(b) for b in s.get("barrier_segments", [])],
            }
            for s in snapshots
        ]

        # Build key_moments by joining segment_highlights ↔ episodes ↔ waypoints_index
        key_moments = []
        for highlight in result.get("segment_highlights", []):
            try:
                wp_id = int(highlight.get("waypoint_id"))
            except (TypeError, ValueError):
                continue
            ep = next((e for e in episodes if e.get("waypoint_id") == wp_id), {})
            wp = (waypoints_index or {}).get(wp_id, {})
            image_url = None
            if walk_id and wp.get("image_filename"):
                image_url = f"/api/walk/{walk_id}/image/{wp['image_filename']}"
            key_moments.append({
                "waypoint_id": wp_id,
                "type": highlight.get("type", "routine"),
                "description": highlight.get("description", ""),
                "narrative": ep.get("narrative", ""),
                "persona_reaction": ep.get("persona_reaction", ""),
                "image_url": image_url,
                "gps": wp.get("gps"),
                "scores": wp.get("scores", {}),
            })

        # Build efficiency block
        efficiency = {
            "steps_total": route_stats.get("steps", 0),
            "steps_analyzed": route_stats.get("analyzed_steps", 0),
            "skip_rate": route_stats.get("skip_rate", 0.0),
            "reroute_count": planner_summary.get("reroute_count", 0),
            "duration_seconds": route_stats.get("duration_seconds", 0),
            "distance_m": route_stats.get("distance_m", 0),
        }

        # Remove segment_highlights (folded into key_moments) and merge new fields
        result.pop("segment_highlights", None)
        result["snapshots"] = snapshots_out
        result["key_moments"] = key_moments
        result["efficiency"] = efficiency
        return result

    def _report_route_fallback(
        self,
        route_stats: Dict,
        dim_avgs: Dict,
        snapshots: List[Dict],
        episodes: List[Dict],
    ) -> Dict[str, Any]:
        """Heuristic fallback when LLM fails for route report."""
        overall_avg = sum(dim_avgs.values()) / len(dim_avgs) if dim_avgs else 5.0
        rec = "recommended" if overall_avg >= 7 else "acceptable" if overall_avg >= 5 else "avoid"
        barrier_count = sum(1 for e in episodes if e.get("avoid"))
        return {
            "executive_summary": f"Walk completed in {route_stats.get('steps', '?')} steps. Average score: {overall_avg:.1f}/10.",
            "score_narrative": f"The route showed {'consistent' if not snapshots else snapshots[-1].get('trend', 'stable')} quality.",
            "segment_highlights": [
                {
                    "waypoint_id": e["waypoint_id"],
                    "type": e.get("significance_tag", "routine"),
                    "description": e.get("narrative", "")[:60],
                }
                for e in episodes[:3]
            ],
            "dimension_verdicts": {dim: f"Average: {score:.1f}/10" for dim, score in dim_avgs.items()},
            "recommendation": rec,
            "recommendation_reason": f"Overall average {overall_avg:.1f}/10 with {barrier_count} barrier segments.",
            "navigation_note": f"Route completed with {route_stats.get('steps', '?')} steps.",
            "persona_suitability": "See dimension verdicts for details.",
        }
