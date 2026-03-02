"""ThinkingModule — waypoint-level reasoning with LLM integration.

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

from src.agent.config import (
    ENHANCED_PERSONALITIES,
    get_enhanced_personality,
    EnhancedPersonalityConfig,
)
from src.config import DEFAULT_FRAMEWORK_ID, settings
from src.utils.logging import get_logger

logger = get_logger(__name__)



class TriggerReason(Enum):
    """Reasons for triggering waypoint-level thinking."""

    VISUAL_CHANGE = "visual_change"
    SCORE_VOLATILITY = "score_volatility"
    DISTANCE_MILESTONE = "distance_milestone"
    USER_REQUEST = "user_request"
    EXCEPTIONAL_MOMENT = "exceptional_moment"


@dataclass
class ThinkingResult:
    """Result of System 2 waypoint-level reasoning with VLM-revised scores.

    System 2 receives persona_scores from System 1 (not objective scores).
    Revisions are made to persona-aware scores based on memory and context.
    """

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

    # === VLM-Revised Scores ===
    revised_scores: Optional[Dict[str, float]] = None  # Revisions of persona_scores
    score_adjustments: Optional[Dict[str, float]] = None  # Differences from System 1 persona scores
    revision_reasoning: Optional[Dict[str, str]] = None

    # === Memory Influence Tracking ===
    memory_influence: Dict[str, Any] = field(default_factory=dict)
    used_stm_context: bool = False
    used_ltm_patterns: bool = False
    personality_factor: str = "unknown"

    # === Metadata ===
    vlm_model_used: str = "none"
    system1_scores: Dict[str, float] = field(default_factory=dict)  # System 1 persona scores
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
        framework_id: str = DEFAULT_FRAMEWORK_ID,
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        vlm_api_url: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        enable_score_revision: bool = True,
        distance_trigger_meters: float = 600.0,
        score_delta_threshold: float = 1.5,
        framework_dimensions: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize thinking module with LLM/VLM configuration.

        Args:
            framework_id: Evaluation framework to use (default: DEFAULT_FRAMEWORK_ID).
            llm_api_url: API URL for LLM calls (defaults to settings.qwen_vlm_api_url).
            llm_api_key: API key for LLM.
            vlm_api_url: API URL for VLM deep dives (optional).
            vlm_api_key: API key for VLM (optional).
            enable_score_revision: Enable System 2 score revision.
            distance_trigger_meters: Distance threshold for milestone triggers.
            score_delta_threshold: Score change threshold for volatility triggers.
            framework_dimensions: Framework dimension definitions for semantic mapping.
        """
        # Load framework to get dimensions
        from src.config import load_framework

        self.framework_id = framework_id
        self.framework = load_framework(framework_id)
        self.dimensions = {d["id"]: d["name_en"] for d in self.framework["dimensions"]}
        self.dimension_ids = list(self.dimensions.keys())

        self.llm_api_url = self._prepare_chat_endpoint(
            llm_api_url or settings.qwen_vlm_api_url
        )
        self.llm_api_key = llm_api_key or settings.qwen_vlm_api_key
        self.vlm_api_url = self._prepare_chat_endpoint(
            vlm_api_url or settings.qwen_vlm_api_url
        )
        self.vlm_api_key = vlm_api_key or settings.qwen_vlm_api_key
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
            score_revision=enable_score_revision,
        )

    def _get_enhanced_config(
        self,
        personality: Any,
    ) -> Optional[EnhancedPersonalityConfig]:
        """Get enhanced personality config if available.

        Falls back to None for basic personalities, enabling graceful degradation.
        """
        if personality is None:
            return None

        # Prefer explicit personality_id when available
        personality_id = getattr(personality, "personality_id", None)
        if personality_id:
            normalized_id = personality_id.lower().replace(" ", "_")
            if normalized_id in ENHANCED_PERSONALITIES:
                try:
                    return get_enhanced_personality(normalized_id)
                except ValueError:
                    pass

        # Try to find enhanced config by matching name
        personality_name = personality.name.lower().replace(" ", "_")

        # Common mappings
        name_mappings = {
            "safety_guardian": "parent_with_kids",  # Map safety to parent
            "scenic_explorer": "photographer",
            "balanced_navigator": "homebuyer",
            "comfort_seeker": "elderly_walker",
            "urban_explorer": "photographer",
            "street_photographer": "photographer",
            "prospective_homebuyer": "homebuyer",
            "urban_runner": "runner",
            "parent_with_young_children": "parent_with_kids",
        }

        # Check direct match or mapped match
        try:
            if personality_name in ENHANCED_PERSONALITIES:
                return get_enhanced_personality(personality_name)
            elif personality_name in name_mappings:
                return get_enhanced_personality(name_mappings[personality_name])
        except ValueError:
            pass

        return None

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
            self.logger.debug("Trigger: Exceptional moment", waypoint_id=waypoint_id)
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
        """Generate episodic narrative for a waypoint using STM + LTM context.

        Note: System 1 scores are final. This method does NOT revise scores.
        It only generates narrative context for memory and reporting.
        """
        import time

        start_time = time.time()

        # System 1 scores are final — no adjustments
        final_scores = system1_scores.copy()
        adjustments = {dim: 0.0 for dim in system1_scores.keys()}

        # Generate simple interpretation based on context
        interpretation = f"Waypoint {waypoint_id}: Using System 1 scores as final. "
        if stm_context and stm_context.get("recent_scores"):
            avg_recent = sum(
                sum(s.values()) / len(s) for s in stm_context["recent_scores"] if s
            ) / len(stm_context["recent_scores"])
            avg_current = sum(system1_scores.values()) / len(system1_scores)
            if avg_current > avg_recent:
                interpretation += "Quality improving from recent context."
            elif avg_current < avg_recent:
                interpretation += "Quality declining from recent context."
            else:
                interpretation += "Consistent with recent context."
        else:
            interpretation += "No prior context available."

        result = ThinkingResult(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            interpretation=interpretation,
            significance="medium",
            pattern_detected=None,
            prediction=None,
            recommendation=None,
            confidence=0.8,
            used_vlm=False,  # No VLM used for score revision
            revised_scores=final_scores,  # System 1 scores kept as-is
            score_adjustments=adjustments,  # All zeros
            revision_reasoning={dim: "No revision - System 1 score kept" for dim in system1_scores.keys()},
            memory_influence={
                "stm_impact": "none",
                "ltm_impact": "none",
                "personality_impact": "none",
                "key_factors": ["Score revision disabled - narrative only"]
            },
            used_stm_context=stm_context is not None,
            used_ltm_patterns=ltm_patterns is not None and len(ltm_patterns) > 0,
            personality_factor="narrative_only",
            vlm_model_used="none",
            system1_scores=system1_scores.copy(),
            processing_time_seconds=time.time() - start_time,
        )

        self.thinking_history.append(result)

        self.logger.info(
            "System 2 narrative-only mode",
            waypoint_id=waypoint_id,
            trigger=trigger_reason.value,
            note="Score revision disabled - System 1 scores are final",
        )

        return result

    def _create_fallback_result(
        self,
        waypoint_id: int,
        trigger_reason: TriggerReason,
        system1_scores: Dict[str, float],
        error: str = "VLM revision unavailable",
    ) -> ThinkingResult:
        """Create fallback result when VLM call fails - framework-agnostic."""

        # Ensure all framework dimensions are present in fallback
        fallback_scores = {
            dim_id: system1_scores.get(dim_id, 5.0) for dim_id in self.dimension_ids
        }

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
            revised_scores=fallback_scores,
            score_adjustments={dim_id: 0.0 for dim_id in self.dimension_ids},
            revision_reasoning={
                dim_id: f"{error} - kept System 1 score"
                for dim_id in self.dimension_ids
            },
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
            "model": settings.qwen_vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
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
            timeout=60,
        )

        response.raise_for_status()
        return response.json()

    def _parse_vlm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VLM API response and extract structured data.

        Validates that all framework dimensions are present in the response.
        """
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

        # Validate required top-level fields
        for field in ["revised_scores", "adjustments", "revision_reasoning"]:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        # Validate all framework dimensions are present
        for dim_id in self.dimension_ids:
            if dim_id not in parsed.get("revised_scores", {}):
                self.logger.warning(f"Missing dimension {dim_id} in revised_scores")
            if dim_id not in parsed.get("adjustments", {}):
                self.logger.warning(f"Missing dimension {dim_id} in adjustments")

        return parsed

    # ========================================================================
    # Narrative generation
    # ========================================================================

    def generate_narrative_chapter(
        self,
        waypoint_id: int,
        visual_description: str,
        system1_scores: Dict[str, float],
        system2_scores: Dict[str, float],
        score_adjustments: Dict[str, float],
        stm_context: Dict[str, Any],
        narrative_context: Dict[str, Any],
        personality: Any,
        trigger_reason: TriggerReason,
    ):
        """Generate narrative chapter using LLM."""
        from datetime import datetime
        from src.agent.capabilities.long_term_memory import NarrativeChapter

        prompt = self._build_narrative_prompt(
            waypoint_id=waypoint_id,
            visual_description=visual_description,
            system1_scores=system1_scores,
            system2_scores=system2_scores,
            score_adjustments=score_adjustments,
            stm_context=stm_context,
            narrative_context=narrative_context,
            personality=personality,
            trigger_reason=trigger_reason,
        )

        try:
            response = self._call_llm_api(prompt)
            parsed = self._parse_narrative_response(response)

            chapter = NarrativeChapter(
                waypoint_id=waypoint_id,
                chapter_number=narrative_context.get("total_chapters", 0) + 1,
                timestamp=datetime.now().isoformat(),
                image_path=Path(""),
                visual_description=visual_description,
                system1_scores=system1_scores,
                system2_scores=system2_scores,
                score_adjustments=score_adjustments,
                narrative_text=parsed.get("narrative_text", ""),
                key_observation=parsed.get("key_observation", ""),
                emotional_tone=parsed.get("emotional_tone", "neutral"),
                references_previous=parsed.get("references_previous", []),
                patterns_mentioned=parsed.get("patterns_mentioned", []),
                predictions_made=parsed.get("predictions_made"),
            )

            return chapter

        except Exception as e:
            self.logger.error(f"Narrative generation failed: {e}")
            return NarrativeChapter(
                waypoint_id=waypoint_id,
                chapter_number=narrative_context.get("total_chapters", 0) + 1,
                timestamp=datetime.now().isoformat(),
                image_path=Path(""),
                visual_description=visual_description,
                system1_scores=system1_scores,
                system2_scores=system2_scores,
                score_adjustments=score_adjustments,
                narrative_text="Failed to generate narrative.",
                key_observation="Error",
                emotional_tone="neutral",
            )

    def _build_narrative_prompt(
        self,
        waypoint_id: int,
        visual_description: str,
        system1_scores: Dict[str, float],
        system2_scores: Dict[str, float],
        score_adjustments: Dict[str, float],
        stm_context: Dict[str, Any],
        narrative_context: Dict[str, Any],
        personality: Any,
        trigger_reason: TriggerReason,
    ) -> str:
        """Build LLM prompt for narrative chapter generation."""

        previous_narrative = self._format_previous_chapters(
            narrative_context.get("recent_chapters", [])
        )

        prompt = f"""# TASK: Write the Next Chapter of a Walking Journey Narrative

You are documenting a pedestrian's journey through an urban environment. You've been observing and analyzing the route, and now you need to write the next chapter of the narrative story.

## Current Waypoint (ID: {waypoint_id})
- Visual description: {visual_description}
- Trigger reason: {self._explain_trigger(trigger_reason)}

## Scores
- System 1 (initial): {json.dumps(system1_scores)}
- System 2 (context-aware): {json.dumps(system2_scores)}
- Adjustments: {json.dumps(score_adjustments)}

## Short-Term Memory (recent context)
{self._format_stm_context(stm_context)}

## Previous Narrative Chapters (most recent first)
{previous_narrative}

## Personality
- Name: {personality.name if personality else 'Default'}
- Tone preference: {getattr(personality, 'explanation_style', 'balanced')}

### OUTPUT FORMAT (JSON)
{{
  "narrative_text": "<2-3 sentence story fragment in first person>",
  "key_observation": "<main takeaway about this waypoint>",
  "emotional_tone": "<optimistic|cautious|concerned|neutral>",
  "references_previous": [<waypoint_ids you referenced>],
  "patterns_mentioned": ["<patterns referenced>"],
  "predictions_made": "<optional forward-looking note>"
}}
"""

        return prompt

    def _format_previous_chapters(self, chapters: List[Dict[str, Any]]) -> str:
        """Format previous narrative chapters for the prompt."""
        if not chapters:
            return "No previous chapters."

        lines = []
        for ch in reversed(chapters):
            lines.append(
                f"- Chapter {ch.get('chapter_number')}, Waypoint {ch.get('waypoint_id')}: {ch.get('key_observation', '')}"
            )
        return "\n".join(lines)

    def _parse_narrative_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response for narrative chapter."""
        content = response["choices"][0]["message"]["content"]
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in narrative response")

        return json.loads(json_str)

    def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Lightweight helper to call LLM for narrative generation."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

        payload = {
            "model": settings.qwen_vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a vivid urban narrator. Respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 600,
        }

        response = requests.post(
            self.llm_api_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 200,
            }

            response = requests.post(
                self.vlm_api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            response_data = response.json()
            insight = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            self.logger.debug("VLM deep dive successful")
            return insight

        except Exception as e:
            self.logger.warning(f"VLM deep dive failed: {e}")
            return f"VLM analysis unavailable: {str(e)}"

    # ========================================================================
    # Framework-Agnostic Helper Methods
    # ========================================================================

    def _generate_dimension_json_template(self) -> str:
        """Generate JSON template for current framework dimensions.

        Returns:
            String like: {"safety": <float 1-10>, "comfort": <float 1-10>, ...}
        """
        fields = [f'"{dim_id}": <float 1-10>' for dim_id in self.dimension_ids]
        return "{\n    " + ",\n    ".join(fields) + "\n  }"

    def _generate_reasoning_fields(self) -> str:
        """Generate reasoning fields for all dimensions.

        Returns:
            String like: "safety": "<Explain...>", "comfort": "<Why...>", ...
        """
        fields = [
            f'"{dim_id}": "<Explain: kept same / adjusted because...>"'
            for dim_id in self.dimension_ids
        ]
        return ",\n    ".join(fields)

    def _format_dimension_descriptions(self) -> str:
        """Format dimension descriptions from framework for prompt.

        Returns:
            Multi-line string with dimension descriptions
        """
        lines = []
        for dim in self.framework["dimensions"]:
            lines.append(f"- **{dim['name_en']}** ({dim['id']}): {dim['description']}")
        return "\n".join(lines)

    def _get_dimension_count(self) -> int:
        """Get number of dimensions in current framework."""
        return len(self.dimension_ids)

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
        """Build multi-modal prompt for VLM to revise scores with STRONG personality-specific guidance."""

        # Format STM context
        stm_summary = self._format_stm_context(stm_context)

        # Format LTM patterns
        ltm_summary = (
            self._format_ltm_patterns(ltm_patterns)
            if ltm_patterns
            else "No relevant past patterns"
        )

        # Get enhanced personality config
        enhanced_config = self._get_enhanced_config(personality)

        # Build personality section
        if enhanced_config:
            persona_prompt = enhanced_config.vlm_persona_prompt

            personality_section = f"""
## YOUR PERSONA
{persona_prompt}

## EVALUATION GUIDANCE
You have already completed a persona-aware System 1 evaluation. Now, using memory and context:
- Review your initial persona-aware scores with deeper reflection
- Consider how patterns from memory might inform your assessment
- Let your persona priorities naturally guide any revisions
- Only adjust scores if memory/context reveals new insights

Evaluate honestly from your perspective.
"""
        else:
            # Fallback for basic personalities
            personality_desc = ""
            if personality:
                personality_desc = (
                    f"You are a {personality.name}. {personality.description}"
                )

            personality_section = f"""
## AGENT PERSPECTIVE
{personality_desc if personality_desc else "Balanced evaluation across all dimensions."}
"""

        # Build trigger explanation
        trigger_explanation = self._explain_trigger(trigger_reason)

        # NEW: Dynamic dimension info
        dimension_descriptions = self._format_dimension_descriptions()
        dimension_json = self._generate_dimension_json_template()
        reasoning_fields = self._generate_reasoning_fields()
        system1_formatted = self._format_system1_results(
            system1_scores, system1_reasoning
        )

        prompt = f"""# SYSTEM 2 RE-EVALUATION WITH PERSONALITY CONTEXT

{personality_section}

---

## SYSTEM 1 EVALUATION (Initial VLM Perception)
The initial evaluation saw ONLY this image without context:

**Scores (1-10 scale):**
{json.dumps(system1_scores, indent=2)}

**Initial Reasoning:**
{json.dumps(system1_reasoning, indent=2)}

---

## CONTEXT FROM MEMORY

### SHORT-TERM MEMORY (Recent waypoints)
{stm_summary}

### LONG-TERM MEMORY (Similar situations)
{ltm_summary}

---

## YOUR TASK

Re-evaluate this image through your persona's lens.

**This waypoint was triggered for deeper analysis because:**
{trigger_explanation}

**EVALUATION APPROACH:**
1. Consider what you would notice and care about in this scene
2. Your scores may differ from System 1 if your priorities lead to different judgments
3. Focus more on dimensions that matter to your persona
4. Be honest about dimensions that are less relevant to you
5. Let your perspective naturally guide your assessment - no need to force changes

---

## OUTPUT FORMAT (JSON)

```json
{{
  "revised_scores": {{
    <dimension_id>: <float 1-10>,
    ...
  }},

  "adjustments": {{
    <dimension_id>: <delta from System 1>,
    ...
  }},

  "revision_reasoning": {{
    <dimension_id>: "<Why changed or kept same - reference your persona>",
    ...
  }},

  "detected_features": ["<feature1>", "<feature2>", ...],

  "interpretation": "<Overall understanding from your persona's perspective>",

  "memory_influence": {{
    "stm_impact": "<high/medium/low/none>",
    "ltm_impact": "<high/medium/low/none>",
    "personality_impact": "<high/medium/low/none>",
    "key_factors": ["<factor1>", "<factor2>"]
  }},

  "confidence": <float 0-1>,
  "pattern_detected": "<pattern or null>",
  "prediction": "<What to expect ahead>",
  "significance": "<high/medium/low>"
}}
```
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
            r.waypoint_id for r in self.thinking_history if r.significance == "high"
        ]

        # Unique patterns detected
        patterns = list(
            set(r.pattern_detected for r in self.thinking_history if r.pattern_detected)
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
