"""Decider — WHAT action to take? (Step 2 of System 2)

Responsibilities:
- Accept / reject / modify route recommendation
- Identify segments to avoid based on persona priorities
- Score confidence based on evidence quality
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger
from .models import System1Evidence, InterpretResult, DecideResult
from .llm_client import call_llm


class Decider:
    """Makes route decisions from interpreted System 1 evidence.

    TODO: Implement using:
    - Persona-specific thresholds (parent_with_kids stricter on safety)
    - Barrier severity classification (dealbreaker vs acceptable)
    - Confidence scoring based on evidence quality
    """

    def __init__(
        self,
        framework_id: str,
        vlm_provider: str = "claude",
        vlm_api_key: Optional[str] = None,
    ) -> None:
        self.framework_id = framework_id
        self.vlm_provider = vlm_provider
        self.vlm_api_key  = vlm_api_key
        self.logger = get_logger(self.__class__.__name__)

    def decide(
        self,
        evidence:         System1Evidence,
        interpret_result: InterpretResult,
        personality:      Optional[Any] = None,
    ) -> DecideResult:
        """Make a route decision from evidence and interpretation.

        Args:
            evidence:         Raw System 1 evidence.
            interpret_result: Output from Interpreter.
            personality:      Agent personality (EnhancedPersonalityConfig).

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Decider.decide() — not yet implemented.")

    def decide_waypoint(
        self,
        waypoint_id: int,
        interpretation: Dict[str, Any],
        system1_scores: Dict[str, float],
        ltm_patterns: Optional[List[Dict]],
        personality: Any,
        dimension_ids: List[str],
        dimensions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Waypoint-level decision via LLM.

        Args:
            waypoint_id: Current waypoint ID.
            interpretation: Output from Interpreter.interpret_waypoint().
            system1_scores: Per-dimension scores from System 1.
            ltm_patterns: Long-term memory patterns (optional).
            personality: Agent personality object.
            dimension_ids: Ordered list of dimension IDs.
            dimensions: Mapping of dimension ID → display name.

        Returns:
            {"significance": str, "avoid": bool, "reason": str|None,
             "concerns": list, "opportunities": list}
        """
        persona_name = getattr(personality, "name", "Unknown Persona")
        persona_desc = getattr(personality, "description", "")

        scores_text = "\n".join(
            f"  - {dimensions.get(dim_id, dim_id)}: {system1_scores.get(dim_id, 0.0):.1f}/10"
            for dim_id in dimension_ids
        )

        ltm_text = _format_ltm_patterns(ltm_patterns if ltm_patterns else None)
        self.logger.debug(
            f"[Decider] waypoint={waypoint_id} ltm_text preview: "
            f"{ltm_text[:120].replace(chr(10), ' | ')}"
        )

        key_concern_text = interpretation.get("key_concern") or "(none identified)"

        prompt = f"""You are making a routing decision for a specific persona at a street waypoint.

Persona: {persona_name}
Description: {persona_desc}

Waypoint interpretation:
{interpretation.get("text", "(none)")}

Key concern for this persona: {key_concern_text}

Current System 1 scores:
{scores_text}

Long-term memory patterns:
{ltm_text}

Make a judgment about this waypoint for the persona. Apply these criteria:
- significance=high: if a core dimension for this persona changed sharply, OR average score < 3.5 or > 8.5
- significance=medium: moderate changes or moderate scores
- significance=low: no significant changes, scores within normal range
- avoid=true: if a priority dimension for this persona falls below their acceptable threshold
- Ensure the judgment reflects this persona's specific priorities (different personas judge the same environment differently)

Respond ONLY with valid JSON matching this exact schema:
{{
  "significance": "high|medium|low",
  "avoid": true or false,
  "reason": "one sentence justification for the decision, or null",
  "concerns": ["list of concerns for this persona"],
  "opportunities": ["list of positive aspects for this persona"]
}}"""

        self.logger.debug(f"[Decider] waypoint={waypoint_id} calling LLM")
        result = call_llm(prompt, max_tokens=512)
        if result and "significance" in result:
            self.logger.debug(f"[Decider] LLM ok | significance={result.get('significance')} avoid={result.get('avoid')}")
            return {
                "significance": result.get("significance", "medium"),
                "avoid": result.get("avoid", False),
                "reason": result.get("reason"),
                "concerns": result.get("concerns", []),
                "opportunities": result.get("opportunities", []),
            }

        self.logger.warning(f"[Decider] LLM failed → fallback heuristic")
        return self._decide_fallback(system1_scores=system1_scores)

    def decide_branch(
        self,
        candidates: List[Dict[str, Any]],
        ltm_patterns: Optional[Dict],
        personality: Any,
        dimension_ids: List[str],
        dimensions: Dict[str, str],
        destination_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Choose the best direction at a branch point.

        Args:
            candidates: List of dicts, one per direction:
                {
                    "direction": str,          # "A", "B", "C" or heading label
                    "heading": float,          # compass heading in degrees
                    "pano_id": str,            # Street View pano ID
                    "scores": Dict[str, float],
                    "interpretation": str,     # from Interpreter.interpret_waypoint()["text"]
                    "key_concern": str | None,
                }
            ltm_patterns: LTM context from MemoryManager (snapshots + episodes)
            personality: Agent personality object
            dimension_ids: Ordered dimension IDs
            dimensions: dimension_id → display name

        Returns:
            {
                "chosen_direction": str,   # matches candidate["direction"]
                "chosen_heading": float,
                "chosen_pano_id": str,
                "reason": str,
                "confidence": float,       # 0.0–1.0
                "ranking": List[str],      # all directions ranked best→worst
            }
        """
        persona_name = getattr(personality, "name", "Unknown Persona")
        persona_desc = getattr(personality, "description", "")

        ltm_text = _format_ltm_patterns(ltm_patterns)

        dest_line = f"Destination: {destination_context}" if destination_context else ""

        candidate_lines = []
        for c in candidates:
            scores_str = "  ".join(
                f"{dimensions.get(d, d)}={c['scores'].get(d, 0):.1f}"
                for d in dimension_ids
            )

            # look-ahead trend (only shown when depth > 1)
            trend = c.get("score_trend", "")
            avg_scores = c.get("avg_scores", {})
            waypoint_scores = c.get("waypoint_scores", [])
            trend_str = ""
            if len(waypoint_scores) > 1:
                avg_str = "  ".join(
                    f"{dimensions.get(d, d)}={v:.1f}"
                    for d, v in avg_scores.items()
                )
                trend_icon = (
                    "↑" if trend == "improving"
                    else "↓" if trend == "declining"
                    else "→"
                )
                trend_str = (
                    f"\n    {trend_icon} Trend: {trend.upper()} over {len(waypoint_scores)} waypoints"
                    f"\n    Avg: {avg_str}"
                )

            concern_str = (
                f"\n    Key concern: {c['key_concern']}"
                if c.get("key_concern") else ""
            )
            visit_count = c.get("visit_count", 0)
            visit_str = (
                f"\n    ⚠ Previously visited ({visit_count} times)"
                if visit_count > 0 else ""
            )
            candidate_lines.append(
                f"[{c['direction']}] heading={c['heading']:.0f}°  {scores_str}\n"
                f"    \"{c['interpretation'][:120]}\""
                f"{trend_str}{concern_str}{visit_str}"
            )
        candidates_text = "\n\n".join(candidate_lines)

        direction_labels = [c["direction"] for c in candidates]
        directions_str = " | ".join(direction_labels)

        prompt = f"""You are {persona_name} navigating a city.
{persona_desc}

{dest_line}

Route context:
{ltm_text}

Your options:
{candidates_text}

Which direction do you choose?

Respond ONLY with valid JSON:
{{
  "chosen_direction": "(one of: {directions_str})",
  "reason": "one sentence from {persona_name}'s perspective",
  "confidence": 0.0 to 1.0,
  "ranking": ["best", "second", ...] using direction labels
}}"""

        self.logger.debug(f"[Decider.branch] {len(candidates)} candidates calling LLM")
        result = call_llm(prompt, max_tokens=256)

        if result and "chosen_direction" in result:
            chosen_dir = result["chosen_direction"]
            chosen = next((c for c in candidates if c["direction"] == chosen_dir), candidates[0])
            self.logger.debug(
                f"[Decider.branch] chosen={chosen_dir} confidence={result.get('confidence')}"
            )
            return {
                "chosen_direction": chosen_dir,
                "chosen_heading": chosen["heading"],
                "chosen_pano_id": chosen.get("pano_id", ""),
                "reason": result.get("reason", ""),
                "confidence": result.get("confidence", 0.5),
                "ranking": result.get("ranking", direction_labels),
            }

        # Fallback: pick highest average score
        self.logger.warning("[Decider.branch] LLM failed → fallback: highest avg score")
        best = max(
            candidates,
            key=lambda c: sum(c["scores"].values()) / len(c["scores"]) if c["scores"] else 0,
        )
        return {
            "chosen_direction": best["direction"],
            "chosen_heading": best["heading"],
            "chosen_pano_id": best.get("pano_id", ""),
            "reason": "Selected by highest average score (fallback).",
            "confidence": 0.3,
            "ranking": [best["direction"]],
        }

    def _decide_fallback(self, system1_scores: Dict[str, float]) -> Dict[str, Any]:
        """Heuristic fallback when LLM call fails."""
        avg = sum(system1_scores.values()) / len(system1_scores) if system1_scores else 5.0
        avoid = avg < 4.0
        significance = "high" if avg < 3.5 or avg > 8.5 else "medium"
        return {
            "significance": significance,
            "avoid": avoid,
            "reason": "Low quality threshold exceeded" if avoid else None,
            "concerns": [],
            "opportunities": [],
        }


def _format_ltm_patterns(ltm_patterns) -> str:
    """Format route context for Decider prompt."""
    if not ltm_patterns:
        return "No route history available."

    if isinstance(ltm_patterns, dict):
        lines = []

        snapshots = ltm_patterns.get("snapshots", [])
        if snapshots:
            lines.append("Route history (snapshots):")
            for s in snapshots:
                barrier_str = (
                    f"  ⚠barrier WP{s['barrier_segments']}"
                    if s.get("barrier_segments")
                    else ""
                )
                traj = s.get("trajectory", 0.0)
                traj_str = f"+{traj:.1f}" if traj >= 0 else f"{traj:.1f}"
                dim_str = " ".join(
                    f"{k}={v:.1f}" for k, v in s.get("avg", {}).items()
                )
                lines.append(
                    f"  WP{s['span_start']}–{s['span_end']}: {s['trend'].upper()} "
                    f"{traj_str}  [{dim_str}]{barrier_str}"
                )

        episodes = ltm_patterns.get("reasoning_episodes", [])
        if episodes:
            lines.append(f"Prior S2 episodes ({len(episodes)}):")
            for ep in episodes[-3:]:
                avoid_str = " ⚠AVOID" if ep.get("avoid") else ""
                lines.append(
                    f"  WP{ep['waypoint_id']} [{ep['significance'].upper()}{avoid_str}]: "
                    f"{ep['interpretation'][:80]}"
                )
                if ep.get("key_concern"):
                    lines.append(f"    concern: {ep['key_concern']}")

        return "\n".join(lines) if lines else "No route history available."

    # Legacy list format fallback
    if isinstance(ltm_patterns, list):
        if not ltm_patterns:
            return "No prior waypoint reasoning in this route."
        lines = ["Prior route patterns:"]
        for p in ltm_patterns[:5]:
            lines.append(f"  - {p.get('pattern_type', '?')}: {p.get('description', '')}")
        return "\n".join(lines)

    return "No route context available."
