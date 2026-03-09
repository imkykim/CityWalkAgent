"""PersonaReasoner — System 2 waypoint-level reasoning.

Design principle: REASON, don't perceive or act.
Pipeline: Interpret → Decide → Plan → Report
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.config import (
    ENHANCED_PERSONALITIES,
    get_enhanced_personality,
    EnhancedPersonalityConfig,
)
from src.core import DEFAULT_FRAMEWORK_ID, settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TriggerReason(Enum):
    """Reasons for triggering waypoint-level reasoning."""

    VISUAL_CHANGE = "visual_change"
    SCORE_VOLATILITY = "score_volatility"
    DISTANCE_MILESTONE = "distance_milestone"
    USER_REQUEST = "user_request"
    EXCEPTIONAL_MOMENT = "exceptional_moment"


@dataclass
class ReasoningResult:
    """Result of System 2 persona-aware reasoning (Interpret → Decide → Plan → Report)."""

    waypoint_id: int
    trigger_reason: TriggerReason
    # Step 1: Interpret
    interpretation: str
    score_change_reason: Optional[str]
    persona_divergence: Optional[str]
    # Step 2: Decide
    significance: str                    # "high" | "medium" | "low"
    avoid_recommendation: bool
    decision_reason: Optional[str]
    # Step 3: Plan
    prediction: Optional[str]
    alternative_suggestion: Optional[str]
    # Step 4: Report
    recommendation: Optional[str]
    confidence: float
    # Memory
    used_stm_context: bool = False
    used_ltm_patterns: bool = False
    personality_factor: str = "unknown"
    # Metadata
    system1_scores: Dict[str, float] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    memory_influence: Dict[str, Any] = field(default_factory=dict)


class PersonaReasoner:
    """Waypoint-level System 2 reasoning with trigger conditions.

    Pipeline: Interpret → Decide → Plan → Report

    Usage::

        reasoner = PersonaReasoner(
            framework_id="place_pulse_2.0",
            distance_trigger_meters=600.0,
            score_delta_threshold=1.5,
        )

        trigger = reasoner.should_trigger(
            waypoint_id=5,
            visual_change=True,
            score_delta=2.3,
            distance_from_last=650,
        )

        if trigger:
            result = reasoner.reason(
                waypoint_id=5,
                trigger_reason=trigger,
                current_image_path=Path("waypoint_5.jpg"),
                system1_scores={"safety": 8.5, "comfort": 7.2},
                system1_reasoning={"safety": "Wide sidewalk..."},
                stm_context=stm.get_context(),
                ltm_patterns=None,
                personality=agent.personality,
                route_metadata={"route_id": "R123", "length_km": 2.5},
            )
    """

    def __init__(
        self,
        framework_id: str = DEFAULT_FRAMEWORK_ID,
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        distance_trigger_meters: float = 600.0,
        score_delta_threshold: float = 1.5,
    ):
        """Initialize PersonaReasoner.

        Args:
            framework_id: Evaluation framework to use.
            llm_api_url: API URL for LLM calls (defaults to settings.qwen_vlm_api_url).
            llm_api_key: API key for LLM.
            distance_trigger_meters: Distance threshold for milestone triggers.
            score_delta_threshold: Score change threshold for volatility triggers.
        """
        from src.core import load_framework

        self.framework_id = framework_id
        self.framework = load_framework(framework_id)
        self.dimensions = {d["id"]: d["name_en"] for d in self.framework["dimensions"]}
        self.dimension_ids = list(self.dimensions.keys())

        self.llm_api_url = self._prepare_chat_endpoint(
            llm_api_url or settings.qwen_vlm_api_url
        )
        self.llm_api_key = llm_api_key or settings.qwen_vlm_api_key

        # Trigger thresholds
        self.distance_trigger_meters = distance_trigger_meters
        self.score_delta_threshold = score_delta_threshold

        # State tracking
        self.last_trigger_waypoint: Optional[int] = None
        self.reasoning_history: List[ReasoningResult] = []

        self.logger = get_logger(f"{__name__}.PersonaReasoner")

        self.logger.debug(
            "PersonaReasoner initialized",
            distance_trigger=distance_trigger_meters,
            score_delta_threshold=score_delta_threshold,
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def should_trigger(
        self,
        waypoint_id: int,
        visual_change: bool,
        score_delta: float,
        distance_from_last: float,
        is_exceptional: bool = False,
    ) -> Optional[TriggerReason]:
        """Determine if waypoint-level reasoning should be triggered.

        Args:
            waypoint_id: Current waypoint ID.
            visual_change: Whether pHash detected visual change.
            score_delta: Absolute change in average score from previous waypoint.
            distance_from_last: Distance in meters since last trigger.
            is_exceptional: Whether this is a manually flagged exceptional moment.

        Returns:
            TriggerReason if trigger condition met, None otherwise.
        """
        if is_exceptional:
            self.logger.debug("Trigger: Exceptional moment", waypoint_id=waypoint_id)
            return TriggerReason.EXCEPTIONAL_MOMENT

        if visual_change:
            self.logger.debug("Trigger: Visual change", waypoint_id=waypoint_id)
            return TriggerReason.VISUAL_CHANGE

        if score_delta >= self.score_delta_threshold:
            self.logger.debug(
                "Trigger: Score volatility",
                waypoint_id=waypoint_id,
                delta=score_delta,
            )
            return TriggerReason.SCORE_VOLATILITY

        if distance_from_last >= self.distance_trigger_meters:
            self.logger.debug(
                "Trigger: Distance milestone",
                waypoint_id=waypoint_id,
                distance=distance_from_last,
            )
            return TriggerReason.DISTANCE_MILESTONE

        return None

    def reason(
        self,
        waypoint_id: int,
        trigger_reason: TriggerReason,
        current_image_path: Path,
        system1_scores: Dict[str, float],
        system1_reasoning: Dict[str, str],
        stm_context: Dict[str, Any],
        ltm_patterns: Optional[List[Dict[str, Any]]],
        personality: Any,
        route_metadata: Dict[str, Any],
    ) -> ReasoningResult:
        """System 2: Interpret → Decide → Plan → Report

        Note: System 1 scores are FINAL. No score revision.

        Args:
            waypoint_id: Current waypoint ID.
            trigger_reason: Why System 2 was triggered.
            current_image_path: Path to waypoint image.
            system1_scores: Final scores from System 1.
            system1_reasoning: Per-dimension reasoning from System 1.
            stm_context: Short-term memory context dict.
            ltm_patterns: Long-term memory patterns (optional).
            personality: Agent personality object.
            route_metadata: Route-level info dict.

        Returns:
            ReasoningResult with interpretation, decision, plan, and report.
        """
        start = time.time()

        interpretation = self._interpret(
            waypoint_id=waypoint_id,
            system1_scores=system1_scores,
            system1_reasoning=system1_reasoning,
            stm_context=stm_context,
            trigger_reason=trigger_reason,
            personality=personality,
        )
        decision = self._decide(
            waypoint_id=waypoint_id,
            interpretation=interpretation,
            system1_scores=system1_scores,
            ltm_patterns=ltm_patterns,
            personality=personality,
        )
        plan = self._plan(
            waypoint_id=waypoint_id,
            decision=decision,
            stm_context=stm_context,
            route_metadata=route_metadata,
            personality=personality,
        )
        report = self._report(
            waypoint_id=waypoint_id,
            interpretation=interpretation,
            decision=decision,
            plan=plan,
            personality=personality,
        )

        result = ReasoningResult(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            interpretation=interpretation.get("text", ""),
            score_change_reason=interpretation.get("score_change_reason"),
            persona_divergence=interpretation.get("persona_divergence"),
            significance=decision.get("significance", "medium"),
            avoid_recommendation=decision.get("avoid", False),
            decision_reason=decision.get("reason"),
            prediction=plan.get("prediction"),
            alternative_suggestion=plan.get("alternative"),
            recommendation=report.get("message"),
            confidence=report.get("confidence", 0.5),
            used_stm_context=bool(stm_context),
            used_ltm_patterns=bool(ltm_patterns),
            personality_factor=getattr(personality, "name", "unknown"),
            system1_scores=system1_scores.copy(),
            processing_time_seconds=time.time() - start,
            memory_influence={
                "stm_waypoints": len(stm_context.get("recent_scores", [])),
                "ltm_patterns": len(ltm_patterns) if ltm_patterns else 0,
            },
        )

        self.reasoning_history.append(result)

        self.logger.debug(
            "System 2 reasoning complete",
            waypoint_id=waypoint_id,
            trigger=trigger_reason.value,
            significance=result.significance,
        )

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics and summary of reasoning history.

        Returns:
            Dictionary containing:
            - total_reasoned: Total number of reasoning episodes
            - trigger_breakdown: Count by trigger reason
            - significance_breakdown: Count by significance level
            - avg_confidence: Average confidence across all results
            - high_significance_waypoints: List of high-significance waypoint IDs
        """
        if not self.reasoning_history:
            return {
                "total_reasoned": 0,
                "trigger_breakdown": {},
                "significance_breakdown": {},
                "avg_confidence": 0.0,
                "high_significance_waypoints": [],
            }

        trigger_breakdown: Dict[str, int] = {}
        for result in self.reasoning_history:
            reason = result.trigger_reason.value
            trigger_breakdown[reason] = trigger_breakdown.get(reason, 0) + 1

        significance_breakdown: Dict[str, int] = {}
        for result in self.reasoning_history:
            sig = result.significance
            significance_breakdown[sig] = significance_breakdown.get(sig, 0) + 1

        confidences = [r.confidence for r in self.reasoning_history]
        avg_confidence = sum(confidences) / len(confidences)

        high_sig_waypoints = [
            r.waypoint_id
            for r in self.reasoning_history
            if r.significance == "high"
        ]

        summary = {
            "total_reasoned": len(self.reasoning_history),
            "trigger_breakdown": trigger_breakdown,
            "significance_breakdown": significance_breakdown,
            "avg_confidence": avg_confidence,
            "high_significance_waypoints": high_sig_waypoints,
        }

        self.logger.debug(
            "Reasoning summary generated",
            total_reasoned=summary["total_reasoned"],
            avg_confidence=summary["avg_confidence"],
        )

        return summary

    # ========================================================================
    # System 2 Pipeline Stubs
    # ========================================================================

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> Optional[Dict]:
        """text-only LLM call → JSON 파싱 반환. 실패 시 None."""
        import httpx
        import json

        url = self._prepare_chat_endpoint(
            getattr(settings, "qwen_vlm_api_url", None)
            or getattr(settings, "qwen_api_url", None)
        )
        api_key = (
            getattr(settings, "qwen_vlm_api_key", None)
            or getattr(settings, "qwen_api_key", None)
        )
        model = (
            getattr(settings, "qwen_vlm_model", None)
            or getattr(settings, "qwen_model", None)
        )

        if not url or not api_key:
            self.logger.warning("LLM credentials not found in settings")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                clean = content.strip()
                # JSON 펜스 제거
                if clean.startswith("```"):
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                clean = clean.strip()
                return json.loads(clean)
        except Exception as e:
            self.logger.warning(f"_call_llm failed: {e}")
            return None

    def _interpret(self, **kwargs) -> Dict[str, Any]:
        """Step 1: Interpret — why is this waypoint noteworthy?"""
        waypoint_id = kwargs["waypoint_id"]
        system1_scores = kwargs.get("system1_scores", {})
        system1_reasoning = kwargs.get("system1_reasoning", {})
        stm_context = kwargs.get("stm_context", {})
        trigger_reason = kwargs.get("trigger_reason")
        personality = kwargs.get("personality")

        enhanced = self._get_enhanced_config(personality)
        persona_name = getattr(personality, "name", "Unknown Persona")
        persona_desc = getattr(personality, "description", "")

        thresholds_text = ""
        if enhanced and enhanced.thresholds:
            thresholds_text = ", ".join(
                f"{k}={v}" for k, v in enhanced.thresholds.items()
            )

        scores_text = self._format_system1_results(system1_scores, system1_reasoning)

        recent = stm_context.get("recent_scores", [])[-3:]
        trend = stm_context.get("trend", "unknown")
        recent_text = ""
        if recent:
            for i, s in enumerate(recent):
                avg = sum(s.values()) / len(s) if s else 0
                recent_text += f"  - [{i+1}] avg={avg:.1f}, scores={s}\n"
        else:
            recent_text = "  (no prior context)\n"

        trigger_text = self._explain_trigger(trigger_reason) if trigger_reason else "Unknown trigger"

        prompt = f"""You are analyzing a street waypoint from the perspective of a specific persona.

Persona: {persona_name}
Description: {persona_desc}
Decision thresholds: {thresholds_text if thresholds_text else "default"}

Current waypoint ID: {waypoint_id}
Trigger reason: {trigger_text}

System 1 scores and reasoning (per dimension):
{scores_text}

Recent waypoint scores (most recent last, up to 3):
{recent_text}Trend: {trend}

Based on this information, interpret what is happening at this waypoint from the persona's perspective.

Respond ONLY with valid JSON matching this exact schema:
{{
  "text": "2-3 sentences describing what happened at this waypoint from the persona's perspective",
  "score_change_reason": "reason for score change compared to previous waypoints, or null if no significant change",
  "key_concern": "the single most important concern for this persona at this waypoint, or null"
}}"""

        result = self._call_llm(prompt, max_tokens=512)
        if result and "text" in result:
            return {
                "text": result.get("text", ""),
                "score_change_reason": result.get("score_change_reason"),
                "persona_divergence": None,
            }

        # Fallback to heuristic
        return self._interpret_fallback(**kwargs)

    def _interpret_fallback(self, **kwargs) -> Dict[str, Any]:
        """Heuristic fallback for _interpret."""
        scores = kwargs.get("system1_scores", {})
        stm = kwargs.get("stm_context", {})
        recent = stm.get("recent_scores", [])
        text = f"Waypoint {kwargs['waypoint_id']}: "
        if recent:
            avg_recent = sum(sum(s.values()) / len(s) for s in recent if s) / len(recent)
            avg_current = sum(scores.values()) / len(scores) if scores else 5.0
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

    def _decide(self, **kwargs) -> Dict[str, Any]:
        """Step 2: Decide — what judgment is needed at this point?"""
        interpretation = kwargs.get("interpretation", {})
        system1_scores = kwargs.get("system1_scores", {})
        ltm_patterns = kwargs.get("ltm_patterns")
        personality = kwargs.get("personality")

        enhanced = self._get_enhanced_config(personality)
        persona_name = getattr(personality, "name", "Unknown Persona")
        persona_desc = getattr(personality, "description", "")

        thresholds_text = ""
        if enhanced and enhanced.thresholds:
            thresholds_text = ", ".join(
                f"{k}={v}" for k, v in enhanced.thresholds.items()
            )

        scores_text = "\n".join(
            f"  - {dim}: {score:.1f}/10"
            for dim, score in system1_scores.items()
        )

        ltm_text = self._format_ltm_patterns(ltm_patterns[:3] if ltm_patterns else None)

        prompt = f"""You are making a routing decision for a specific persona at a street waypoint.

Persona: {persona_name}
Description: {persona_desc}
Decision thresholds: {thresholds_text if thresholds_text else "default"}

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

        result = self._call_llm(prompt, max_tokens=512)
        if result and "significance" in result:
            return {
                "significance": result.get("significance", "medium"),
                "avoid": result.get("avoid", False),
                "reason": result.get("reason"),
                "concerns": result.get("concerns", []),
                "opportunities": result.get("opportunities", []),
            }

        # Fallback to heuristic
        return self._decide_fallback(**kwargs)

    def _decide_fallback(self, **kwargs) -> Dict[str, Any]:
        """Heuristic fallback for _decide."""
        scores = kwargs.get("system1_scores", {})
        avg = sum(scores.values()) / len(scores) if scores else 5.0
        avoid = avg < 4.0
        significance = "high" if avg < 3.5 or avg > 8.5 else "medium"
        return {
            "significance": significance,
            "avoid": avoid,
            "reason": "Low quality threshold exceeded" if avoid else None,
            "concerns": [],
            "opportunities": [],
        }

    def _plan(self, **kwargs) -> Dict[str, Any]:
        """Step 3: Plan — route planning and forward prediction.

        TODO: Implement LLM call for route-level planning.
        """
        return {"prediction": None, "alternative": None}

    def _report(self, **kwargs) -> Dict[str, Any]:
        """Step 4: Report — generate final user-facing message."""
        waypoint_id = kwargs.get("waypoint_id")
        interpretation = kwargs.get("interpretation", {})
        decision = kwargs.get("decision", {})
        personality = kwargs.get("personality")

        persona_name = getattr(personality, "name", "Unknown Persona")
        avoid = decision.get("avoid", False)
        reason = decision.get("reason", "")
        concerns = decision.get("concerns", [])

        prompt = f"""You are generating a concise, actionable walking recommendation for a specific persona.

Persona: {persona_name}
Waypoint ID: {waypoint_id}

Situation summary:
{interpretation.get("text", "(none)")}

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

        result = self._call_llm(prompt, max_tokens=256)
        if result and "message" in result:
            confidence = float(result.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            return {"message": result["message"], "confidence": confidence}

        # Fallback to heuristic
        return self._report_fallback(**kwargs)

    def _report_fallback(self, **kwargs) -> Dict[str, Any]:
        """Heuristic fallback for _report."""
        avoid = kwargs.get("decision", {}).get("avoid", False)
        msg = "Consider an alternative route." if avoid else "Conditions are acceptable."
        return {"message": msg, "confidence": 0.6}

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_enhanced_config(
        self,
        personality: Any,
    ) -> Optional[EnhancedPersonalityConfig]:
        """Get enhanced personality config if available."""
        if personality is None:
            return None

        personality_id = getattr(personality, "personality_id", None)
        if personality_id:
            normalized_id = personality_id.lower().replace(" ", "_")
            if normalized_id in ENHANCED_PERSONALITIES:
                try:
                    return get_enhanced_personality(normalized_id)
                except ValueError:
                    pass

        personality_name = personality.name.lower().replace(" ", "_")

        name_mappings = {
            "safety_guardian": "parent_with_kids",
            "scenic_explorer": "photographer",
            "balanced_navigator": "homebuyer",
            "comfort_seeker": "elderly_walker",
            "urban_explorer": "photographer",
            "street_photographer": "photographer",
            "prospective_homebuyer": "homebuyer",
            "urban_runner": "runner",
            "parent_with_young_children": "parent_with_kids",
        }

        try:
            if personality_name in ENHANCED_PERSONALITIES:
                return get_enhanced_personality(personality_name)
            elif personality_name in name_mappings:
                return get_enhanced_personality(name_mappings[personality_name])
        except ValueError:
            pass

        return None

    def _format_stm_context(self, stm_context: Dict[str, Any]) -> str:
        """Format short-term memory into human-readable summary."""
        recent_scores = stm_context.get("recent_scores", [])
        recent_summaries = stm_context.get("recent_summaries", [])
        waypoint_ids = stm_context.get("waypoint_ids", [])

        lines = ["**Recent waypoints (most recent last):**"]
        for wid, scores, summary in zip(waypoint_ids, recent_scores, recent_summaries):
            if scores:
                avg = sum(scores.values()) / len(scores)
                lines.append(f"- Waypoint {wid}: Avg={avg:.1f} | {summary}")
        return "\n".join(lines)

    def _format_ltm_patterns(self, patterns: Optional[List[Dict[str, Any]]]) -> str:
        """Format long-term memory patterns into human-readable summary."""
        if not patterns:
            return "No relevant past patterns found"

        lines = ["**Relevant patterns from past routes:**"]
        for pattern in patterns[:3]:
            pattern_name = pattern.get("pattern_type", "unknown")
            description = pattern.get("description", "")
            frequency = pattern.get("frequency", 0)
            lines.append(f"- **{pattern_name}** (seen {frequency}x): {description}")
        return "\n".join(lines)

    def _explain_trigger(self, trigger_reason: TriggerReason) -> str:
        """Explain why System 2 was triggered."""
        explanations = {
            TriggerReason.VISUAL_CHANGE: "Major visual change detected (pHash distance > 20)",
            TriggerReason.SCORE_VOLATILITY: "Significant score change from previous waypoint",
            TriggerReason.DISTANCE_MILESTONE: "Regular evaluation checkpoint (600m)",
            TriggerReason.EXCEPTIONAL_MOMENT: "Manually flagged as exceptional",
        }
        return explanations.get(trigger_reason, "Unknown trigger")

    def _format_system1_results(
        self, scores: Dict[str, float], reasoning: Dict[str, str]
    ) -> str:
        """Format System 1 results for prompt."""
        lines = []
        for dim_id in self.dimension_ids:
            score = scores.get(dim_id, 0.0)
            reason = reasoning.get(dim_id, "No reasoning provided")
            dim_name = self.dimensions[dim_id]
            lines.append(f"- {dim_name}: {score:.1f}/10 - {reason}")
        return "\n".join(lines)

    def _generate_dimension_json_template(self) -> str:
        """Generate JSON template for current framework dimensions."""
        fields = [f'"{dim_id}": <float 1-10>' for dim_id in self.dimension_ids]
        return "{\n    " + ",\n    ".join(fields) + "\n  }"

    def _generate_reasoning_fields(self) -> str:
        """Generate reasoning fields for all dimensions."""
        fields = [
            f'"{dim_id}": "<Explain: kept same / adjusted because...>"'
            for dim_id in self.dimension_ids
        ]
        return ",\n    ".join(fields)

    def _format_dimension_descriptions(self) -> str:
        """Format dimension descriptions from framework for prompt."""
        lines = []
        for dim in self.framework["dimensions"]:
            lines.append(f"- **{dim['name_en']}** ({dim['id']}): {dim['description']}")
        return "\n".join(lines)

    def _get_dimension_count(self) -> int:
        """Get number of dimensions in current framework."""
        return len(self.dimension_ids)

    def _create_fallback_result(
        self,
        waypoint_id: int,
        trigger_reason: TriggerReason,
        system1_scores: Dict[str, float],
        error: str = "Reasoning unavailable",
    ) -> ReasoningResult:
        """Create fallback result when reasoning fails."""
        return ReasoningResult(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            interpretation=f"System 2 reasoning failed: {error}.",
            score_change_reason=None,
            persona_divergence=None,
            significance="low",
            avoid_recommendation=False,
            decision_reason=None,
            prediction=None,
            alternative_suggestion=None,
            recommendation="Continue with System 1 evaluation",
            confidence=0.0,
            used_stm_context=False,
            used_ltm_patterns=False,
            personality_factor="none",
            system1_scores=system1_scores.copy(),
            processing_time_seconds=0.0,
            memory_influence={},
        )

    @staticmethod
    def _prepare_chat_endpoint(url: Optional[str]) -> Optional[str]:
        """Ensure API endpoints target the chat completions route."""
        if not url:
            return None
        normalized = url.rstrip("/")
        if "/chat" in normalized or "/completions" in normalized:
            return normalized
        return f"{normalized}/chat/completions"
