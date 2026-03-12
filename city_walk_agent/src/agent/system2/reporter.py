"""Reporter — OUTPUT comprehensive evaluation report. (Step 4 of System 2)

Responsibilities:
- Executive summary for stakeholders
- Per-dimension findings and recommendations
- Urban planning improvement suggestions
- Persona suitability matrix (all personas, not just current)
- Bilingual output (Chinese + English)
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from src.utils.logging import get_logger
from .models import (
    System1Evidence,
    InterpretResult,
    DecideResult,
    PlanResult,
    ReportResult,
)
from .llm_client import call_llm


class Reporter:
    """Generates comprehensive evaluation reports from the full pipeline.

    TODO: Implement using VLM narrative generation:
    - Synthesise all pipeline outputs into structured prose
    - Urban planning format for planners and stakeholders
    - Persona suitability matrix across all available personas
    - Bilingual output (Chinese + English) for thesis presentation
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

    def report(
        self,
        evidence:         System1Evidence,
        interpret_result: InterpretResult,
        decide_result:    DecideResult,
        plan_result:      PlanResult,
        personality:      Optional[Any] = None,
    ) -> ReportResult:
        """Generate a comprehensive evaluation report.

        Args:
            evidence:         Raw System 1 evidence.
            interpret_result: Output from Interpreter.
            decide_result:    Output from Decider.
            plan_result:      Output from Planner.
            personality:      Agent personality (EnhancedPersonalityConfig).

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Reporter.report() — not yet implemented.")

    def report_waypoint(
        self,
        waypoint_id: int,
        interpretation: Dict[str, Any],
        decision: Dict[str, Any],
        plan: Dict[str, Any],
        personality: Any,
    ) -> Dict[str, Any]:
        """Waypoint-level report via LLM.

        Args:
            waypoint_id: Current waypoint ID.
            interpretation: Output from Interpreter.interpret_waypoint().
            decision: Output from Decider.decide_waypoint().
            plan: Output from _plan() stub.
            personality: Agent personality object.

        Returns:
            {"message": str, "confidence": float}
        """
        persona_name = getattr(personality, "name", "Unknown Persona")
        avoid = decision.get("avoid", False)
        reason = decision.get("reason", "")
        concerns = decision.get("concerns", [])

        key_concern_text = interpretation.get("key_concern") or "(none identified)"

        prompt = f"""You are generating a concise, actionable walking recommendation for a specific persona.

Persona: {persona_name}
Waypoint ID: {waypoint_id}

Situation summary:
{interpretation.get("text", "(none)")}

Key concern: {key_concern_text}

Decision: {"AVOID this segment" if avoid else "This segment is acceptable"}
Reason: {reason or "(none)"}
Concerns: {", ".join(concerns) if concerns else "none"}

Generate a single, specific sentence of walking advice from the persona's perspective.

Examples:
- parent_with_kids + avoid=true: "이 구간은 인도가 좁고 차량 통행이 많아 유아차 이동이 위험합니다."
- runner + avoid=false: "평탄한 보도와 충분한 공간으로 페이스 유지에 적합한 구간입니다."

Write in English. Be specific about what the persona should do or expect.

Respond ONLY with valid JSON matching this exact schema:
{{
  "message": "one concrete action-oriented sentence for the persona",
  "confidence": 0.0 to 1.0
}}"""

        self.logger.debug(f"[Reporter] waypoint={waypoint_id} calling LLM")
        result = call_llm(prompt, max_tokens=256)
        if result and "message" in result:
            self.logger.debug(f"[Reporter] LLM ok | message={result.get('message','')[:60]}")
            confidence = float(result.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            return {"message": result["message"], "confidence": confidence}

        self.logger.warning(f"[Reporter] LLM failed → fallback heuristic")
        return self._report_fallback(avoid=avoid)

    def _report_fallback(self, avoid: bool) -> Dict[str, Any]:
        """Heuristic fallback when LLM call fails."""
        msg = "Consider an alternative route." if avoid else "Conditions are acceptable."
        return {"message": msg, "confidence": 0.6}
