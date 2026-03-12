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

        ltm_text = _format_ltm_patterns(ltm_patterns[:3] if ltm_patterns else None)

        prompt = f"""You are making a routing decision for a specific persona at a street waypoint.

Persona: {persona_name}
Description: {persona_desc}

Waypoint interpretation:
{interpretation.get("text", "(none)")}

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


def _format_ltm_patterns(patterns: Optional[List[Dict[str, Any]]]) -> str:
    """Format long-term memory patterns into human-readable summary."""
    if not patterns:
        return "No relevant past patterns found"
    lines = ["**Relevant patterns from past routes:**"]
    for pattern in patterns:
        pattern_name = pattern.get("pattern_type", "unknown")
        description = pattern.get("description", "")
        frequency = pattern.get("frequency", 0)
        lines.append(f"- **{pattern_name}** (seen {frequency}x): {description}")
    return "\n".join(lines)
