"""Thinking capability - agent's reasoning and decision-making.

This module provides two thinking mechanisms:
1. ThinkingCapability: Route-level stateless reasoning (original, backward compatible)
2. ThinkingModule: Waypoint-level reasoning with LLM integration (new)

Design principle: REASON, don't perceive or act.
"""

import json
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.agent.config.constants import (
    CONFIDENCE_BARRIER_WEIGHT,
    CONFIDENCE_SCORE_WEIGHT,
    CONFIDENCE_VOLATILITY_WEIGHT,
    EXCELLENT_SCORE_THRESHOLD,
    LOW_VOLATILITY_THRESHOLD,
)
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# New Components for Waypoint-Level Reasoning
# ============================================================================


class TriggerReason(Enum):
    """Reasons for triggering waypoint-level thinking."""

    VISUAL_CHANGE = "visual_change"
    SCORE_VOLATILITY = "score_volatility"
    DISTANCE_MILESTONE = "distance_milestone"
    USER_REQUEST = "user_request"
    EXCEPTIONAL_MOMENT = "exceptional_moment"


@dataclass
class ThinkingResult:
    """Result of System 2 waypoint-level reasoning with VLM-revised scores."""

    waypoint_id: int
    trigger_reason: TriggerReason

    # === Interpretation (existing) ===
    interpretation: str
    significance: str
    pattern_detected: Optional[str]
    prediction: Optional[str]
    recommendation: Optional[str]
    confidence: float
    used_vlm: bool  # Keep for backward compatibility

    # === NEW: VLM-Revised Scores ===
    revised_scores: Optional[Dict[str, float]] = None
    score_adjustments: Optional[Dict[str, float]] = None
    revision_reasoning: Optional[Dict[str, str]] = None

    # === NEW: Memory Influence Tracking ===
    memory_influence: Dict[str, Any] = field(default_factory=dict)
    used_stm_context: bool = False
    used_ltm_patterns: bool = False
    personality_factor: str = "unknown"

    # === NEW: Metadata ===
    vlm_model_used: str = "none"
    system1_scores: Dict[str, float] = field(default_factory=dict)
    processing_time_seconds: float = 0.0


class ThinkingModule:
    """Waypoint-level reasoning with trigger conditions and LLM integration.

    Unlike ThinkingCapability (route-level, stateless), ThinkingModule:
    - Operates at waypoint level during route traversal
    - Maintains thinking history for context
    - Uses LLM for deep interpretations
    - Triggers selectively based on conditions
    - Optionally uses VLM for visual deep dives

    Usage::

        thinking = ThinkingModule(
            llm_api_url=settings.qwen_vlm_api_url,
            llm_api_key=settings.qwen_vlm_api_key,
            enable_vlm_deep_dive=False,
            distance_trigger_meters=600.0,
            score_delta_threshold=1.5
        )

        # During route traversal
        trigger = thinking.should_trigger(
            waypoint_id=5,
            visual_change=True,
            score_delta=2.3,
            distance_from_last=650,
            is_exceptional=False
        )

        if trigger:
            result = thinking.think_waypoint(
                waypoint_id=5,
                trigger_reason=trigger,
                stm_context=short_term_memory.get_context(),
                current_scores={"safety": 8.5, "comfort": 7.2},
                current_reasoning={"safety": "Wide sidewalk..."},
                current_image_path=Path("waypoint_5.jpg"),
                route_metadata={"route_id": "R123", "length_km": 2.5}
            )
    """

    def __init__(
        self,
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        vlm_api_url: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        enable_vlm_deep_dive: bool = False,
        enable_score_revision: bool = True,
        distance_trigger_meters: float = 600.0,
        score_delta_threshold: float = 1.5,
    ):
        """Initialize thinking module with LLM/VLM configuration.

        Args:
            llm_api_url: API URL for LLM calls (defaults to settings.qwen_vlm_api_url).
            llm_api_key: API key for LLM.
            vlm_api_url: API URL for VLM deep dives (optional).
            vlm_api_key: API key for VLM (optional).
            enable_vlm_deep_dive: Whether to enable VLM deep dives (expensive).
            enable_score_revision: Enable System 2 score revision.
            distance_trigger_meters: Distance threshold for milestone triggers.
            score_delta_threshold: Score change threshold for volatility triggers.
        """
        self.llm_api_url = self._prepare_chat_endpoint(
            llm_api_url or settings.qwen_vlm_api_url
        )
        self.llm_api_key = llm_api_key or settings.qwen_vlm_api_key
        self.vlm_api_url = self._prepare_chat_endpoint(
            vlm_api_url or settings.qwen_vlm_api_url
        )
        self.vlm_api_key = vlm_api_key or settings.qwen_vlm_api_key
        self.enable_vlm_deep_dive = enable_vlm_deep_dive
        self.enable_score_revision = enable_score_revision

        # Trigger thresholds
        self.distance_trigger_meters = distance_trigger_meters
        self.score_delta_threshold = score_delta_threshold

        # State tracking
        self.last_trigger_waypoint: Optional[int] = None
        self.thinking_history: List[ThinkingResult] = []

        # Logger
        self.logger = get_logger(f"{__name__}.ThinkingModule")

        self.logger.info(
            "ThinkingModule initialized",
            distance_trigger=distance_trigger_meters,
            score_delta_threshold=score_delta_threshold,
            vlm_enabled=enable_vlm_deep_dive,
            score_revision=enable_score_revision,
        )

    def should_trigger(
        self,
        waypoint_id: int,
        visual_change: bool,
        score_delta: float,
        distance_from_last: float,
        is_exceptional: bool = False,
    ) -> Optional[TriggerReason]:
        """Determine if waypoint-level thinking should be triggered.

        Checks multiple conditions and returns the trigger reason if any
        condition is met.

        Args:
            waypoint_id: Current waypoint ID.
            visual_change: Whether pHash detected visual change.
            score_delta: Absolute change in average score from previous waypoint.
            distance_from_last: Distance in meters since last trigger.
            is_exceptional: Whether this is a manually flagged exceptional moment.

        Returns:
            TriggerReason if trigger condition met, None otherwise.
        """
        # Exceptional moments always trigger
        if is_exceptional:
            self.logger.debug(
                "Trigger: Exceptional moment", waypoint_id=waypoint_id
            )
            return TriggerReason.EXCEPTIONAL_MOMENT

        # Visual change trigger
        if visual_change:
            self.logger.debug("Trigger: Visual change", waypoint_id=waypoint_id)
            return TriggerReason.VISUAL_CHANGE

        # Score volatility trigger
        if score_delta >= self.score_delta_threshold:
            self.logger.debug(
                "Trigger: Score volatility",
                waypoint_id=waypoint_id,
                delta=score_delta,
            )
            return TriggerReason.SCORE_VOLATILITY

        # Distance milestone trigger
        if distance_from_last >= self.distance_trigger_meters:
            self.logger.debug(
                "Trigger: Distance milestone",
                waypoint_id=waypoint_id,
                distance=distance_from_last,
            )
            return TriggerReason.DISTANCE_MILESTONE

        # No trigger
        return None

    def think_waypoint(
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
    ) -> ThinkingResult:
        """Perform System 2 reasoning with VLM-based score revision.

        This is the main entry point for System 2 evaluation:
        1. Build rich multi-modal prompt (image + context)
        2. Call VLM API with image + text
        3. Parse revised scores + interpretation
        4. Calculate adjustments
        5. Return ThinkingResult

        Args:
            waypoint_id: Current waypoint ID
            trigger_reason: Why System 2 was triggered
            current_image_path: Path to waypoint image
            system1_scores: Original VLM scores
            system1_reasoning: Original reasoning
            stm_context: Short-term memory context
            ltm_patterns: Relevant long-term memory patterns
            personality: Agent personality configuration
            route_metadata: Route-level info

        Returns:
            ThinkingResult with revised scores and reasoning
        """
        import time

        start_time = time.time()

        if not self.enable_score_revision:
            return self._create_fallback_result(
                waypoint_id=waypoint_id,
                trigger_reason=trigger_reason,
                system1_scores=system1_scores,
                error="Score revision disabled",
            )

        try:
            prompt = self._build_vlm_revision_prompt(
                waypoint_id=waypoint_id,
                system1_scores=system1_scores,
                system1_reasoning=system1_reasoning,
                stm_context=stm_context,
                ltm_patterns=ltm_patterns,
                personality=personality,
                trigger_reason=trigger_reason,
            )

            vlm_response = self._call_vlm_with_image(
                image_path=current_image_path,
                prompt=prompt,
            )

            parsed = self._parse_vlm_response(vlm_response)

            adjustments = {
                dim: parsed["revised_scores"][dim] - system1_scores[dim]
                for dim in system1_scores.keys()
            }

            result = ThinkingResult(
                waypoint_id=waypoint_id,
                trigger_reason=trigger_reason,
                interpretation=parsed.get("interpretation", "No interpretation"),
                significance=parsed.get("significance", "medium"),
                pattern_detected=parsed.get("pattern_detected"),
                prediction=parsed.get("prediction"),
                recommendation=parsed.get("recommendation"),
                confidence=float(parsed.get("confidence", 0.7)),
                used_vlm=True,
                revised_scores=parsed["revised_scores"],
                score_adjustments=adjustments,
                revision_reasoning=parsed.get("revision_reasoning", {}),
                memory_influence=parsed.get("memory_influence", {}),
                used_stm_context=parsed.get("memory_influence", {}).get("stm_impact", "none") != "none",
                used_ltm_patterns=parsed.get("memory_influence", {}).get("ltm_impact", "none") != "none",
                personality_factor=parsed.get("memory_influence", {}).get("personality_impact", "unknown"),
                vlm_model_used="qwen-vl-max",
                system1_scores=system1_scores.copy(),
                processing_time_seconds=time.time() - start_time,
            )

            self.thinking_history.append(result)

            self.logger.info(
                "System 2 evaluation complete",
                waypoint_id=waypoint_id,
                trigger=trigger_reason.value,
                adjustments={k: f"{v:+.1f}" for k, v in adjustments.items()},
                significance=result.significance,
            )

            return result

        except Exception as e:
            self.logger.error(
                "System 2 evaluation failed",
                waypoint_id=waypoint_id,
                error=str(e),
            )

            return self._create_fallback_result(
                waypoint_id=waypoint_id,
                trigger_reason=trigger_reason,
                system1_scores=system1_scores,
                error=str(e),
            )

    def _create_fallback_result(
        self,
        waypoint_id: int,
        trigger_reason: TriggerReason,
        system1_scores: Dict[str, float],
        error: str,
    ) -> ThinkingResult:
        """Create fallback result when VLM call fails."""
        return ThinkingResult(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            interpretation=f"System 2 evaluation failed: {error}. Using System 1 scores.",
            significance="low",
            pattern_detected=None,
            prediction=None,
            recommendation="Continue with System 1 evaluation",
            confidence=0.0,
            used_vlm=False,
            revised_scores=system1_scores.copy(),
            score_adjustments={dim: 0.0 for dim in system1_scores.keys()},
            revision_reasoning={dim: "Fallback - System 2 failed" for dim in system1_scores.keys()},
            memory_influence={},
            used_stm_context=False,
            used_ltm_patterns=False,
            personality_factor="none",
            vlm_model_used="none",
            system1_scores=system1_scores.copy(),
            processing_time_seconds=0.0,
        )

    def _call_vlm_with_image(
        self,
        image_path: Path,
        prompt: str,
    ) -> Dict[str, Any]:
        """Call VLM API with image and text prompt."""
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": "qwen-vl-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.vlm_api_key}",
        }

        response = requests.post(
            self.vlm_api_url,
            headers=headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def _parse_vlm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VLM API response and extract structured data."""
        content = response["choices"][0]["message"]["content"]
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in VLM response")

        parsed = json.loads(json_str)

        for field in ["revised_scores", "adjustments", "revision_reasoning"]:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        return parsed

    def _deep_dive_with_vlm(
        self,
        image_path: Path,
        llm_interpretation: str,
        scores: Dict[str, float],
    ) -> str:
        """Perform deep visual analysis using VLM.

        This is an optional expensive operation that uses VLM to
        perform detailed visual analysis of the waypoint image.

        Args:
            image_path: Path to waypoint image.
            llm_interpretation: LLM's text-based interpretation.
            scores: Current waypoint scores.

        Returns:
            VLM's visual insight string.
        """
        try:
            if not self.vlm_api_url or not self.vlm_api_key:
                return "VLM not configured"

            # Prepare VLM request with image
            # Note: This is a simplified implementation
            # Real implementation would need to encode image and handle multimodal input

            prompt = f"""Based on this street view image, provide additional insight.

Current text-based analysis: {llm_interpretation}
Current scores: {scores}

What visual details support or contradict this analysis?
Provide 1-2 sentences of visual insight."""

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.vlm_api_key}",
            }

            # Simplified payload (real implementation needs image encoding)
            payload = {
                "model": settings.qwen_vlm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 200,
            }

            response = requests.post(
                self.vlm_api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            response_data = response.json()
            insight = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            self.logger.debug("VLM deep dive successful")
            return insight

        except Exception as e:
            self.logger.warning(f"VLM deep dive failed: {e}")
            return f"VLM analysis unavailable: {str(e)}"

    def _build_vlm_revision_prompt(
        self,
        waypoint_id: int,
        system1_scores: Dict[str, float],
        system1_reasoning: Dict[str, str],
        stm_context: Dict[str, Any],
        ltm_patterns: Optional[List[Dict[str, Any]]],
        personality: Any,
        trigger_reason: TriggerReason,
    ) -> str:
        """Build multi-modal prompt for VLM to revise scores with memory context."""

        # Format STM context
        stm_summary = self._format_stm_context(stm_context)

        # Format LTM patterns
        ltm_summary = (
            self._format_ltm_patterns(ltm_patterns)
            if ltm_patterns
            else "No relevant past patterns"
        )

        # Personality description
        personality_desc = f"""
**Agent Personality: {personality.name if personality else 'Default'}**
- Safety Priority: {personality.safety_priority if personality else 5}/10
- Risk Tolerance: {personality.risk_tolerance if personality else 'medium'}
"""

        prompt = f"""# TASK: Re-evaluate this street view waypoint with full context

You are a walking experience evaluator analyzing **Waypoint #{waypoint_id}**.

## SYSTEM 1 EVALUATION (Initial VLM Perception - Single Image Only)
The initial evaluation saw ONLY this image without any context:

**Scores (1-10 scale):**
{json.dumps(system1_scores, indent=2)}

**Initial Reasoning:**
{json.dumps(system1_reasoning, indent=2)}

---

## YOUR TASK (System 2 - Image + Memory Context)
Now re-evaluate this SAME image, but with FULL CONTEXT:

### 1. SHORT-TERM MEMORY (What happened just before this)
{stm_summary}

**Key patterns in recent waypoints:**
- Score trend: {stm_context.get('trend', 'unknown')}
- Average of last 5 waypoints: {self._format_score_averages(stm_context.get('statistics', {}))}
- Volatility: {stm_context.get('volatility', 'unknown')}

### 2. LONG-TERM MEMORY (Similar situations from past routes)
{ltm_summary}

### 3. AGENT PERSPECTIVE
{personality_desc}

**This waypoint was triggered for System 2 reasoning because:**
{self._explain_trigger(trigger_reason)}

---

## EVALUATION INSTRUCTIONS

**Re-score the image considering:**
1. **Sequential Context**: This is part of a walking journey, not an isolated moment
2. **Cumulative Effects**: How do previous waypoints influence perception here?
3. **Pattern Recognition**: Does this match known patterns from memory?
4. **Agent Perspective**: How would this {personality.name if personality else 'agent'} specifically perceive this?
5. **Predictive Factors**: What does this signal about upcoming sections?

**Important Guidelines:**
- You may KEEP System 1 scores if context doesn't change perception
- ADJUST scores UP if context reveals hidden positives (e.g., transition to better area)
- ADJUST scores DOWN if context reveals hidden risks (e.g., part of declining trend, cumulative stress)
- Focus on dimensions most affected by memory/context (usually Safety and Comfort)
- Interest and Aesthetics may be less influenced by sequential context

---

## OUTPUT FORMAT (JSON)

```json
{{
  "revised_scores": {{
    "safety": <float 1-10>,
    "comfort": <float 1-10>,
    "interest": <float 1-10>,
    "aesthetics": <float 1-10>
  }},

  "adjustments": {{
    "safety": <delta from System 1, can be 0>,
    "comfort": <delta>,
    "interest": <delta>,
    "aesthetics": <delta>
  }},

  "revision_reasoning": {{
    "safety": "<Explain: kept same / adjusted because...>",
    "comfort": "<Why changed or stayed>",
    "interest": "<Reasoning>",
    "aesthetics": "<Reasoning>"
  }},

  "interpretation": "<Overall understanding of this waypoint in context>",

  "memory_influence": {{
    "stm_impact": "<How recent history affected scores: high/medium/low/none>",
    "ltm_impact": "<How past patterns affected scores: high/medium/low/none>",
    "personality_impact": "<How agent personality influenced perception: high/medium/low/none>",
    "key_factors": ["<factor1>", "<factor2>"]
  }},

  "confidence": <float 0-1>,
  "pattern_detected": "<pattern name or null>",
  "prediction": "<What to expect in next waypoints>",
  "significance": "<high/medium/low>"
}
```

**CRITICAL**: Base your evaluation on what you SEE in the image, enriched by context. Don't hallucinate details not visible.
"""

        return prompt

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

    def _format_score_averages(self, stats: Dict[str, Any]) -> str:
        """Format score statistics into readable text."""
        if not stats:
            return "No statistics available"

        avg = stats.get("average_scores", {})
        return ", ".join([f"{dim}={score:.1f}" for dim, score in avg.items()])

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

    @staticmethod
    def _prepare_chat_endpoint(url: Optional[str]) -> Optional[str]:
        """Ensure API endpoints target the chat completions route."""
        if not url:
            return None

        normalized = url.rstrip("/")

        # Assume explicit endpoint if caller already provided chat/completions path
        if "/chat" in normalized or "/completions" in normalized:
            return normalized

        return f"{normalized}/chat/completions"

    def get_thinking_summary(self) -> Dict[str, Any]:
        """Get statistics and summary of thinking history.

        Returns:
            Dictionary containing:
            - total_thoughts: Total number of thinking episodes
            - trigger_breakdown: Count by trigger reason
            - significance_breakdown: Count by significance level
            - avg_confidence: Average confidence across all thoughts
            - high_significance_waypoints: List of high-significance waypoint IDs
            - patterns_detected: List of unique patterns detected
        """
        if not self.thinking_history:
            return {
                "total_thoughts": 0,
                "trigger_breakdown": {},
                "significance_breakdown": {},
                "avg_confidence": 0.0,
                "high_significance_waypoints": [],
                "patterns_detected": [],
            }

        # Count by trigger reason
        trigger_breakdown = {}
        for result in self.thinking_history:
            reason = result.trigger_reason.value
            trigger_breakdown[reason] = trigger_breakdown.get(reason, 0) + 1

        # Count by significance
        significance_breakdown = {}
        for result in self.thinking_history:
            sig = result.significance
            significance_breakdown[sig] = significance_breakdown.get(sig, 0) + 1

        # Calculate average confidence
        confidences = [r.confidence for r in self.thinking_history]
        avg_confidence = sum(confidences) / len(confidences)

        # High significance waypoints
        high_sig_waypoints = [
            r.waypoint_id
            for r in self.thinking_history
            if r.significance == "high"
        ]

        # Unique patterns detected
        patterns = list(
            set(
                r.pattern_detected
                for r in self.thinking_history
                if r.pattern_detected
            )
        )

        summary = {
            "total_thoughts": len(self.thinking_history),
            "trigger_breakdown": trigger_breakdown,
            "significance_breakdown": significance_breakdown,
            "avg_confidence": avg_confidence,
            "high_significance_waypoints": high_sig_waypoints,
            "patterns_detected": patterns,
        }

        self.logger.debug(
            "Thinking summary generated",
            total_thoughts=summary["total_thoughts"],
            avg_confidence=summary["avg_confidence"],
        )

        return summary


# ============================================================================
# Original ThinkingCapability (Route-Level, Backward Compatible)
# ============================================================================


class ThinkingCapability:
    """Reasoning capability for making decisions about routes.

    This is a stateless reasoning engine that applies personality weights
    and decision thresholds to observations to produce recommendations.

    Example:
        ```python
        thinker = ThinkingCapability()
        decision = thinker.think(
            observation=observation,
            sequential_analysis=analysis,
            personality_weights={"safety": 2.0, "comfort": 1.2},
            decision_thresholds={"min_overall_score": 6.5},
            explanation_style="safety"
        )
        # decision contains: recommendation, confidence, explanation
        ```
    """

    def __init__(self):
        """Initialize thinking capability.

        The capability is stateless - all decision context is provided
        to the think() method.
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("ThinkingCapability initialized")

    def think(
        self,
        observation: Dict[str, Any],
        sequential_analysis: Dict[str, Any],
        personality_weights: Dict[str, float],
        decision_thresholds: Dict[str, float],
        explanation_style: str = "balanced",
    ) -> Dict[str, Any]:
        """Make a decision about a route based on observation and personality.

        This is the main reasoning method. It applies personality-driven
        logic to observations and produces a recommendation.

        Args:
            observation: Observation dict from ObservationCapability.
            sequential_analysis: Analysis dict from SequentialAnalyzer.
            personality_weights: Weight for each dimension (e.g., {"safety": 2.0}).
            decision_thresholds: Decision criteria (e.g., {"min_overall_score": 6.5}).
            explanation_style: Style of explanation ("safety", "scenic", "balanced", "technical").

        Returns:
            Dict[str, Any] containing:
                - recommendation: "accept" | "reject"
                - confidence: float (0-1)
                - weighted_score: float
                - explanation: str (human-readable)
                - concerns: List[str]
                - highlights: List[str]
                - key_factors: Dict with detailed reasoning
        """
        route_id = observation.get("route_info", {}).get("route_id", "unknown")

        self.logger.debug(
            "Thinking about route",
            route_id=route_id,
            explanation_style=explanation_style,
        )

        # Extract key information
        dimension_stats = observation.get("dimension_stats", {})
        volatility = sequential_analysis.get("volatility", 0.0)
        hidden_barriers = sequential_analysis.get("hidden_barriers", [])
        pattern_type = sequential_analysis.get("pattern_type", "unknown")

        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(
            dimension_stats, personality_weights
        )

        # Identify concerns and highlights
        concerns = self._identify_concerns(
            dimension_stats,
            personality_weights,
            decision_thresholds,
            volatility,
            hidden_barriers,
        )

        highlights = self._identify_highlights(
            dimension_stats, personality_weights, volatility
        )

        # Make recommendation
        recommendation = self._make_recommendation(
            weighted_score,
            decision_thresholds,
            concerns,
            hidden_barriers,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            weighted_score,
            decision_thresholds,
            volatility,
            len(hidden_barriers),
            recommendation,
        )

        # Generate explanation
        explanation = self._generate_explanation(
            recommendation=recommendation,
            weighted_score=weighted_score,
            concerns=concerns,
            highlights=highlights,
            volatility=volatility,
            pattern_type=pattern_type,
            style=explanation_style,
        )

        # Build key factors for transparency
        key_factors = {
            "weighted_score": weighted_score,
            "volatility": volatility,
            "num_barriers": len(hidden_barriers),
            "pattern_type": pattern_type,
            "dimension_averages": {
                dim: stats.get("avg", 0.0)
                for dim, stats in dimension_stats.items()
            },
            "applied_weights": personality_weights,
            "thresholds_used": decision_thresholds,
        }

        decision = {
            "recommendation": recommendation,
            "confidence": confidence,
            "weighted_score": weighted_score,
            "explanation": explanation,
            "concerns": concerns,
            "highlights": highlights,
            "key_factors": key_factors,
        }

        self.logger.info(
            "Decision made",
            route_id=route_id,
            recommendation=recommendation,
            confidence=round(confidence, 2),
            weighted_score=round(weighted_score, 2),
        )

        return decision

    def _calculate_weighted_score(
        self,
        dimension_stats: Dict[str, Dict[str, float]],
        personality_weights: Dict[str, float],
    ) -> float:
        """Calculate overall weighted score.

        Args:
            dimension_stats: Statistics for each dimension.
            personality_weights: Weight for each dimension.

        Returns:
            Weighted average score.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for dimension_id, weight in personality_weights.items():
            if dimension_id in dimension_stats:
                avg_score = dimension_stats[dimension_id].get("avg", 0.0)
                weighted_sum += avg_score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _identify_concerns(
        self,
        dimension_stats: Dict[str, Dict[str, float]],
        personality_weights: Dict[str, float],
        decision_thresholds: Dict[str, float],
        volatility: float,
        hidden_barriers: List[Dict[str, Any]],
    ) -> List[str]:
        """Identify concerns about the route.

        Args:
            dimension_stats: Statistics for each dimension.
            personality_weights: Weight for each dimension.
            decision_thresholds: Decision criteria.
            volatility: Route volatility.
            hidden_barriers: List of detected barriers.

        Returns:
            List of concern descriptions.
        """
        concerns = []

        # Check for low scores in important dimensions
        for dimension_id, weight in personality_weights.items():
            if dimension_id not in dimension_stats:
                continue

            avg_score = dimension_stats[dimension_id].get("avg", 0.0)
            threshold_key = f"min_{dimension_id}_score"

            # Check dimension-specific threshold
            if threshold_key in decision_thresholds:
                min_threshold = decision_thresholds[threshold_key]
                if avg_score < min_threshold:
                    concerns.append(
                        f"Low {dimension_id} score ({avg_score:.1f} < {min_threshold})"
                    )

            # Check if important dimension (high weight) has low score
            elif weight >= 1.5 and avg_score < 6.5:
                concerns.append(
                    f"{dimension_id.capitalize()} below expectations ({avg_score:.1f})"
                )

        # Check volatility
        max_volatility = decision_thresholds.get("max_volatility", 999)
        if volatility > max_volatility:
            concerns.append(
                f"High volatility ({volatility:.1f} > {max_volatility})"
            )

        # Check hidden barriers
        max_barriers = decision_thresholds.get("max_barriers", 999)
        if len(hidden_barriers) > max_barriers:
            concerns.append(
                f"Too many barriers ({len(hidden_barriers)} > {max_barriers})"
            )

        return concerns

    def _identify_highlights(
        self,
        dimension_stats: Dict[str, Dict[str, float]],
        personality_weights: Dict[str, float],
        volatility: float,
    ) -> List[str]:
        """Identify positive highlights about the route.

        Args:
            dimension_stats: Statistics for each dimension.
            personality_weights: Weight for each dimension.
            volatility: Route volatility.

        Returns:
            List of highlight descriptions.
        """
        highlights = []

        # Check for excellent scores in important dimensions
        for dimension_id, weight in personality_weights.items():
            if dimension_id not in dimension_stats:
                continue

            avg_score = dimension_stats[dimension_id].get("avg", 0.0)

            # Highlight if important dimension has excellent score
            if weight >= 1.5 and avg_score >= EXCELLENT_SCORE_THRESHOLD:
                highlights.append(
                    f"Excellent {dimension_id} ({avg_score:.1f}/10)"
                )

        # Check for low volatility (consistency)
        if volatility <= LOW_VOLATILITY_THRESHOLD:
            highlights.append(f"Very consistent experience (volatility: {volatility:.1f})")

        # Check for universally high scores
        all_scores = [
            stats.get("avg", 0.0)
            for stats in dimension_stats.values()
        ]
        if all_scores and min(all_scores) >= 7.0:
            highlights.append("All dimensions score well")

        return highlights

    def _make_recommendation(
        self,
        weighted_score: float,
        decision_thresholds: Dict[str, float],
        concerns: List[str],
        hidden_barriers: List[Dict[str, Any]],
    ) -> str:
        """Make accept/reject recommendation.

        Args:
            weighted_score: Overall weighted score.
            decision_thresholds: Decision criteria.
            concerns: List of identified concerns.
            hidden_barriers: List of detected barriers.

        Returns:
            "accept" or "reject"
        """
        # Check minimum overall score
        min_overall = decision_thresholds.get("min_overall_score", 6.5)
        if weighted_score < min_overall:
            return "reject"

        # Check if there are critical concerns
        max_barriers = decision_thresholds.get("max_barriers", 999)
        if len(hidden_barriers) > max_barriers:
            return "reject"

        # Check for primary dimension failures
        if any("min_primary_score" in str(c) for c in concerns):
            return "reject"

        # Otherwise accept
        return "accept"

    def _calculate_confidence(
        self,
        weighted_score: float,
        decision_thresholds: Dict[str, float],
        volatility: float,
        num_barriers: int,
        recommendation: str,
    ) -> float:
        """Calculate confidence in the decision.

        Confidence is based on:
        - How far the weighted score is from the threshold (margin)
        - Route volatility (lower is better)
        - Number of barriers (fewer is better)

        Args:
            weighted_score: Overall weighted score.
            decision_thresholds: Decision criteria.
            volatility: Route volatility.
            num_barriers: Number of detected barriers.
            recommendation: The recommendation made.

        Returns:
            Confidence value between 0 and 1.
        """
        # Score component: how far from threshold
        min_overall = decision_thresholds.get("min_overall_score", 6.5)
        score_margin = abs(weighted_score - min_overall)

        if recommendation == "accept":
            # For accept: higher scores = higher confidence
            score_confidence = min(score_margin / 2.0, 1.0)
        else:
            # For reject: lower scores = higher confidence
            score_confidence = min(score_margin / 2.0, 1.0)

        # Volatility component: lower volatility = higher confidence
        max_volatility = decision_thresholds.get("max_volatility", 2.0)
        volatility_confidence = max(0.0, 1.0 - (volatility / max_volatility))

        # Barrier component: fewer barriers = higher confidence
        max_barriers = decision_thresholds.get("max_barriers", 2)
        barrier_confidence = max(0.0, 1.0 - (num_barriers / max(max_barriers, 1)))

        # Weighted combination
        confidence = (
            CONFIDENCE_SCORE_WEIGHT * score_confidence
            + CONFIDENCE_VOLATILITY_WEIGHT * volatility_confidence
            + CONFIDENCE_BARRIER_WEIGHT * barrier_confidence
        )

        return min(1.0, max(0.0, confidence))

    def _generate_explanation(
        self,
        recommendation: str,
        weighted_score: float,
        concerns: List[str],
        highlights: List[str],
        volatility: float,
        pattern_type: str,
        style: str,
    ) -> str:
        """Generate human-readable explanation.

        Args:
            recommendation: The recommendation made.
            weighted_score: Overall weighted score.
            concerns: List of concerns.
            highlights: List of highlights.
            volatility: Route volatility.
            pattern_type: Route pattern classification.
            style: Explanation style.

        Returns:
            Human-readable explanation string.
        """
        if style == "safety":
            return self._explain_safety_focus(
                recommendation, weighted_score, concerns, highlights
            )
        elif style == "scenic":
            return self._explain_scenic_focus(
                recommendation, weighted_score, highlights, pattern_type
            )
        elif style == "technical":
            return self._explain_technical(
                recommendation, weighted_score, volatility, concerns, highlights
            )
        else:  # balanced
            return self._explain_balanced(
                recommendation, weighted_score, concerns, highlights
            )

    def _explain_safety_focus(
        self,
        recommendation: str,
        weighted_score: float,
        concerns: List[str],
        highlights: List[str],
    ) -> str:
        """Generate safety-focused explanation."""
        if recommendation == "accept":
            msg = f"✅ Route approved (score: {weighted_score:.1f}/10). "
            if highlights:
                msg += f"Strengths: {', '.join(highlights[:2])}. "
            msg += "Safe for walking."
        else:
            msg = f"⚠️ Route not recommended (score: {weighted_score:.1f}/10). "
            if concerns:
                msg += f"Safety concerns: {', '.join(concerns[:2])}. "
            msg += "Consider alternative route."

        return msg

    def _explain_scenic_focus(
        self,
        recommendation: str,
        weighted_score: float,
        highlights: List[str],
        pattern_type: str,
    ) -> str:
        """Generate scenic-focused explanation."""
        if recommendation == "accept":
            msg = f"✨ Great route! (score: {weighted_score:.1f}/10). "
            if highlights:
                msg += f"{', '.join(highlights[:2])}. "
            msg += f"Pattern: {pattern_type}."
        else:
            msg = f"Route score: {weighted_score:.1f}/10. "
            msg += "May not meet your expectations for scenic experience."

        return msg

    def _explain_balanced(
        self,
        recommendation: str,
        weighted_score: float,
        concerns: List[str],
        highlights: List[str],
    ) -> str:
        """Generate balanced explanation."""
        if recommendation == "accept":
            msg = f"Route score: {weighted_score:.1f}/10. "
            if highlights:
                msg += f"Highlights: {', '.join(highlights[:2])}. "
            if concerns:
                msg += f"Minor concerns: {', '.join(concerns[:1])}."
            else:
                msg += "Overall good quality."
        else:
            msg = f"Route score: {weighted_score:.1f}/10 (below threshold). "
            if concerns:
                msg += f"Issues: {', '.join(concerns[:2])}."

        return msg

    def _explain_technical(
        self,
        recommendation: str,
        weighted_score: float,
        volatility: float,
        concerns: List[str],
        highlights: List[str],
    ) -> str:
        """Generate technical explanation with details."""
        msg = f"Decision: {recommendation.upper()}. "
        msg += f"Weighted score: {weighted_score:.2f}/10. "
        msg += f"Volatility: {volatility:.2f}. "

        if highlights:
            msg += f"Positive factors: {len(highlights)}. "
        if concerns:
            msg += f"Concerns: {len(concerns)}. "

        msg += f"Details: {', '.join(concerns[:2]) if concerns else 'No major issues'}."

        return msg
