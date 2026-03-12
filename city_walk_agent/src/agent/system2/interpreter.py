"""Interpreter — WHY did scores change? (Step 1 of System 2)

Responsibilities:
- Explain score trajectories per dimension
- Detect where persona scoring diverged from objective scoring
- Classify environmental transitions with causal analysis
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger
from .models import System1Evidence, InterpretResult
from .llm_client import call_llm


class Interpreter:
    """Analyzes System 1 evidence to explain score changes and persona divergence.

    TODO: Implement using VLM calls with:
    - Score trajectory analysis (which dimensions changed and where)
    - Persona vs objective divergence detection
    - Environmental transition classification
    - Causal explanation generation
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

    def interpret(
        self,
        evidence:    System1Evidence,
        personality: Optional[Any] = None,
    ) -> InterpretResult:
        """Interpret System 1 evidence and explain score changes.

        Args:
            evidence:    Aggregated System 1 perception results.
            personality: Agent personality (EnhancedPersonalityConfig).

        Returns:
            InterpretResult with explanations and divergence points.

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Interpreter.interpret() — not yet implemented.")

    def interpret_waypoint(
        self,
        waypoint_id: int,
        system1_scores: Dict[str, float],
        system1_reasoning: Dict[str, str],
        stm_context: Dict[str, Any],
        trigger_reason: Any,
        personality: Any,
        dimension_ids: List[str],
        dimensions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Waypoint-level interpretation via LLM.

        Args:
            waypoint_id: Current waypoint ID.
            system1_scores: Per-dimension scores from System 1.
            system1_reasoning: Per-dimension reasoning from System 1.
            stm_context: Short-term memory context dict.
            trigger_reason: TriggerReason enum value.
            personality: Agent personality object.
            dimension_ids: Ordered list of dimension IDs.
            dimensions: Mapping of dimension ID → display name.

        Returns:
            {"text": str, "score_change_reason": str|None, "persona_divergence": None}
        """
        persona_name = getattr(personality, "name", "Unknown Persona")
        persona_desc = getattr(personality, "description", "")

        # Build scores text
        scores_text = "\n".join(
            f"  - {dimensions.get(dim_id, dim_id)}: {system1_scores.get(dim_id, 0.0):.1f}/10"
            f" - {system1_reasoning.get(dim_id, 'No reasoning provided')}"
            for dim_id in dimension_ids
        )

        # Build STM recent context
        recent = stm_context.get("recent_scores", [])[-3:]
        trend = stm_context.get("trend", "unknown")
        if recent:
            recent_lines = []
            for idx, s in enumerate(recent):
                avg = sum(s.values()) / len(s) if s else 0
                recent_lines.append(f"  - [{idx + 1}] avg={avg:.1f}, scores={s}")
            recent_text = "\n".join(recent_lines)
        else:
            recent_text = "  (no prior context)"

        trigger_text = _explain_trigger(trigger_reason) if trigger_reason else "Unknown trigger"

        prompt = f"""You are analyzing a street waypoint from the perspective of a specific persona.

Persona: {persona_name}
Description: {persona_desc}

Current waypoint ID: {waypoint_id}
Trigger reason: {trigger_text}

System 1 scores and reasoning (per dimension):
{scores_text}

Recent waypoint scores (most recent last, up to 3):
{recent_text}
Trend: {trend}

Based on this information, interpret what is happening at this waypoint from the persona's perspective.

Respond ONLY with valid JSON matching this exact schema:
{{
  "text": "2-3 sentences describing what happened at this waypoint from the persona's perspective",
  "score_change_reason": "reason for score change compared to previous waypoints, or null if no significant change",
  "key_concern": "the single most important concern for this persona at this waypoint, or null"
}}"""

        self.logger.debug(f"[Interpreter] waypoint={waypoint_id} persona={getattr(personality, 'name', '?')} calling LLM")
        result = call_llm(prompt, max_tokens=512)
        if result and "text" in result:
            self.logger.debug(f"[Interpreter] LLM ok | text={result.get('text','')[:60]}")
            return {
                "text": result.get("text", ""),
                "score_change_reason": result.get("score_change_reason"),
                "persona_divergence": None,
            }

        self.logger.warning(f"[Interpreter] LLM failed → fallback heuristic")
        return self._interpret_fallback(
            waypoint_id=waypoint_id,
            system1_scores=system1_scores,
            stm_context=stm_context,
        )

    def _interpret_fallback(
        self,
        waypoint_id: int,
        system1_scores: Dict[str, float],
        stm_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Heuristic fallback when LLM call fails."""
        recent = stm_context.get("recent_scores", [])
        text = f"Waypoint {waypoint_id}: "
        if recent:
            avg_recent = sum(sum(s.values()) / len(s) for s in recent if s) / len(recent)
            avg_current = sum(system1_scores.values()) / len(system1_scores) if system1_scores else 5.0
            delta = avg_current - avg_recent
            if delta > 1.0:
                text += "Noticeable quality improvement."
            elif delta < -1.0:
                text += "Quality decline detected."
            else:
                text += "Consistent with recent conditions."
        else:
            text += "No prior context — first triggered evaluation."
        return {"text": text, "score_change_reason": None, "persona_divergence": None}


def _explain_trigger(trigger_reason: Any) -> str:
    """Explain why System 2 was triggered."""
    value = getattr(trigger_reason, "value", str(trigger_reason))
    explanations = {
        "visual_change": "Major visual change detected (pHash distance > 20)",
        "score_volatility": "Significant score change from previous waypoint",
        "distance_milestone": "Regular evaluation checkpoint (600m)",
        "exceptional_moment": "Manually flagged as exceptional",
    }
    return explanations.get(value, "Unknown trigger")
