"""Memory Manager - Central hub for coordinating all memory systems.

The MemoryManager serves as the integration layer between:
- Short-term memory (STM): Real-time context during route traversal
- Long-term memory (LTM): Persistent storage of key moments and patterns
- Agent attributes: Personality, profile, and status
- System 2 reasoning: ThinkingModule activation and context preparation

Key responsibilities:
1. Apply attention gate to filter what enters STM
2. Decide when to trigger expensive System 2 (LLM) reasoning
3. Package complete context for ThinkingModule
4. Consolidate STM → LTM when routes complete
5. Use personality to influence memory behavior

Design principles:
- Lazy loading: Only instantiate components when needed
- Personality-driven: Agent attributes influence all decisions
- Selective triggering: ~50% reduction at attention gate, ~20% System 2
- Context-rich: Provide STM + LTM + attributes for reasoning

Usage::

    # Create memory manager
    memory_manager = MemoryManager(agent_id="walker_001")

    # Configure with agent attributes
    memory_manager.set_agent_attributes(
        personality=personality,
        profile={"name": "Walker", "goal": "Find safe routes"},
        status={"mode": "active", "energy": 1.0}
    )

    # Process waypoints during traversal
    for waypoint_analysis in route_waypoints:
        trigger_result = memory_manager.process_waypoint(waypoint_analysis)

        if trigger_result is not None:
            # System 2 triggered - perform deep reasoning
            thinking_result = thinking_module.reason(trigger_result)

    # Consolidate at route end
    route_summary = memory_manager.complete_route(
        route_data=route_data,
        thinking_history=thinking_results
    )
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.capabilities.long_term_memory import LongTermMemory, RouteSummary
from src.agent.capabilities.short_term_memory import MemoryItem, ShortTermMemory
from src.agent.capabilities.thinking import TriggerReason
from src.agent.config import AgentPersonality
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Central hub coordinating all memory systems and agent attributes.

    The MemoryManager integrates Short-Term Memory (STM), Long-Term Memory (LTM),
    and agent attributes (personality, profile, status) to provide unified memory
    services for the agent's reasoning system.

    Key features:
    - **Attention Gate**: Filters waypoints based on significance before STM entry
    - **System 2 Triggering**: Decides when to activate expensive LLM reasoning
    - **Context Preparation**: Packages STM + LTM + attributes for ThinkingModule
    - **Route Consolidation**: Automatically consolidates STM → LTM on completion
    - **Personality Integration**: Uses agent personality to tune all thresholds

    Attributes:
        agent_id: Unique identifier for the agent
        storage_dir: Optional custom directory for LTM storage
        personality: Agent's personality configuration
        profile: Agent metadata (name, id, description, goal)
        status: Dynamic agent state (energy, mode, etc.)
    """

    def __init__(self, agent_id: str, storage_dir: Optional[Path] = None) -> None:
        """Initialize memory manager with agent ID.

        Args:
            agent_id: Unique identifier for this agent (used for LTM storage)
            storage_dir: Optional custom directory for LTM files

        Side Effects:
            Sets up lazy-loading infrastructure for STM and LTM
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir

        # Lazy-loaded memory components
        self._stm: Optional[ShortTermMemory] = None
        self._ltm: Optional[LongTermMemory] = None

        # Agent attributes (set via set_agent_attributes)
        self.personality: Optional[AgentPersonality] = None
        self.profile: Dict[str, Any] = {}
        self.status: Dict[str, Any] = {}

        # Statistics tracking
        self._waypoints_processed = 0
        self._attention_gate_passes = 0
        self._system2_triggers = 0

        # Track System 2 results for sequential context
        self._system2_results: Dict[int, Any] = {}

        self.logger = get_logger(f"{__name__}.{agent_id}")
        self.logger.info("MemoryManager initialized", agent_id=agent_id)

    # ========================================================================
    # Properties: Lazy-loaded Memory Components
    # ========================================================================

    @property
    def stm(self) -> ShortTermMemory:
        """Lazy-load Short-Term Memory.

        Returns:
            ShortTermMemory instance with sliding window
        """
        if self._stm is None:
            self._stm = ShortTermMemory(window_size=5)
            self.logger.debug("ShortTermMemory lazy-loaded")
        return self._stm

    @property
    def episodic_ltm(self) -> LongTermMemory:
        """Lazy-load Long-Term Memory (Episodic).

        Returns:
            LongTermMemory instance for persistent storage
        """
        if self._ltm is None:
            self._ltm = LongTermMemory(
                agent_id=self.agent_id, storage_dir=self.storage_dir
            )
            self.logger.debug("LongTermMemory lazy-loaded")
        return self._ltm

    # ========================================================================
    # Agent Attributes Integration
    # ========================================================================

    def set_agent_attributes(
        self,
        personality: AgentPersonality,
        profile: Dict[str, Any],
        status: Dict[str, Any],
    ) -> None:
        """Configure agent attributes to influence memory operations.

        Agent attributes affect:
        - Attention gate thresholds (personality-based)
        - System 2 trigger sensitivity (personality-based)
        - LTM consolidation behavior

        Args:
            personality: Agent's personality configuration with weights/thresholds
            profile: Static agent metadata (name, id, description, goal)
            status: Dynamic agent state (energy, mode, etc.)

        Side Effects:
            Stores attributes and logs configuration

        Example::

            memory_manager.set_agent_attributes(
                personality=safety_personality,
                profile={"name": "SafeWalker", "goal": "Find safest routes"},
                status={"mode": "active", "energy": 0.8}
            )
        """
        self.personality = personality
        self.profile = profile
        self.status = status

        self.logger.info(
            "Agent attributes configured",
            personality=personality.name,
            profile_keys=list(profile.keys()),
            status=status,
        )

    # ========================================================================
    # Attention Gate: Decide What Enters STM
    # ========================================================================

    def _passes_attention_gate(
        self, waypoint_analysis: Any  # WaypointAnalysis type hint
    ) -> bool:
        """Determine if waypoint is significant enough to enter STM.

        The attention gate filters waypoints based on:
        1. **Visual Change**: pHash detected environmental shift
        2. **Score Anomaly**: Scores differ significantly from STM average
        3. **Low Scores**: Any dimension below personality threshold
        4. **Distance Milestone**: Every 10th waypoint (for continuity)

        Personality influence:
        - **Cautious**: Lower thresholds (more sensitive)
        - **Scenic**: Emphasizes aesthetic anomalies
        - **Balanced**: Standard thresholds

        Args:
            waypoint_analysis: WaypointAnalysis object from ContinuousAnalyzer

        Returns:
            True if waypoint should be added to STM, False otherwise
        """
        # Trigger 1: Visual change detected
        if waypoint_analysis.visual_change_detected:
            self.logger.debug(
                "Attention gate: PASS (visual change)",
                waypoint_id=waypoint_analysis.waypoint_id,
                phash_distance=waypoint_analysis.phash_distance,
            )
            return True

        # Trigger 2: Distance milestone (every 10 waypoints)
        if waypoint_analysis.waypoint_id % 10 == 0:
            self.logger.debug(
                "Attention gate: PASS (milestone)",
                waypoint_id=waypoint_analysis.waypoint_id,
            )
            return True

        # Get current waypoint average score
        current_scores = (
            getattr(waypoint_analysis, "persona_scores", None)
            or getattr(waypoint_analysis, "scores", {})
        )
        if not current_scores:
            return False

        current_avg = sum(current_scores.values()) / len(current_scores)

        # Trigger 3: Score anomaly (compare to STM average)
        stm_context = self.stm.get_context()
        if stm_context and stm_context.get("recent_scores"):
            recent_scores = stm_context["recent_scores"]

            # Calculate STM average
            stm_avgs = []
            for score_dict in recent_scores:
                if score_dict:
                    avg = sum(score_dict.values()) / len(score_dict)
                    stm_avgs.append(avg)

            if stm_avgs:
                stm_mean = sum(stm_avgs) / len(stm_avgs)
                score_delta = abs(current_avg - stm_mean)

                # Personality-based threshold
                delta_threshold = self._get_attention_delta_threshold()

                if score_delta >= delta_threshold:
                    self.logger.debug(
                        "Attention gate: PASS (score anomaly)",
                        waypoint_id=waypoint_analysis.waypoint_id,
                        delta=score_delta,
                        threshold=delta_threshold,
                    )
                    return True

        # Trigger 4: Low scores in any dimension
        low_score_threshold = self._get_low_score_threshold()

        for dim_id, score in current_scores.items():
            if score < low_score_threshold:
                self.logger.debug(
                    "Attention gate: PASS (low score)",
                    waypoint_id=waypoint_analysis.waypoint_id,
                    dimension=dim_id,
                    score=score,
                    threshold=low_score_threshold,
                )
                return True

        # No triggers - filter out
        self.logger.debug(
            "Attention gate: FILTER",
            waypoint_id=waypoint_analysis.waypoint_id,
            avg_score=current_avg,
        )
        return False

    def _get_attention_delta_threshold(self) -> float:
        """Get score delta threshold based on personality.

        Returns:
            Delta threshold for attention gate (1.0-2.0)
        """
        if self.personality is None:
            return 1.5  # Default

        # Map personality to sensitivity
        personality_name = self.personality.name.lower()

        if "cautious" in personality_name or "safety" in personality_name:
            return 1.0  # More sensitive
        elif "scenic" in personality_name or "explorer" in personality_name:
            return 1.2  # Moderately sensitive
        else:
            return 1.5  # Standard

    def _get_low_score_threshold(self) -> float:
        """Get low score threshold based on personality.

        Returns:
            Minimum acceptable score (5.0-6.5)
        """
        if self.personality is None:
            return 5.5  # Default

        # Check decision thresholds first
        if self.personality.decision_thresholds:
            min_overall = self.personality.decision_thresholds.get(
                "min_overall_score", 6.5
            )
            return min_overall - 1.0  # Slightly below threshold

        # Fallback to personality-based
        personality_name = self.personality.name.lower()

        if "cautious" in personality_name or "safety" in personality_name:
            return 6.0  # Higher threshold
        elif "scenic" in personality_name:
            return 5.5  # Moderate
        else:
            return 5.0  # Standard

    # ========================================================================
    # Context Preparation for System 2
    # ========================================================================

    def prepare_context_for_thinking(
        self,
        waypoint_analysis: Any,  # WaypointAnalysis type hint
        trigger_reason: Optional[TriggerReason] = None,
    ) -> Dict[str, Any]:
        """Package complete context for ThinkingModule reasoning.

        Now includes System 2 results from previous waypoints in STM context.
        """
        stm_context = self.stm.get_context()
        current_gps = waypoint_analysis.gps
        past_experience = self._retrieve_relevant_ltm(current_gps)

        recent_context: List[Dict[str, Any]] = []
        if stm_context and stm_context.get("recent_scores"):
            waypoint_ids = stm_context.get("waypoint_ids", [])
            recent_scores = stm_context["recent_scores"]
            recent_summaries = stm_context.get("recent_summaries", [])

            for wid, scores, summary in zip(
                waypoint_ids, recent_scores, recent_summaries
            ):
                system2_info = ""
                if wid in self._system2_results:
                    result = self._system2_results[wid]
                    adjustments = getattr(result, "score_adjustments", {}) or {}
                    avg_adjustment = (
                        sum(adjustments.values()) / len(adjustments)
                        if adjustments
                        else 0.0
                    )
                    system2_info = (
                        f" [System 2: {getattr(result, 'significance', 'medium')}, "
                        f"adjusted by {avg_adjustment:+.1f}]"
                    )

                recent_context.append(
                    {
                        "waypoint_id": wid,
                        "scores": scores,
                        "summary": summary + system2_info,
                    }
                )

        personality_info: Dict[str, Any] = {}
        if self.personality:
            personality_info = {
                "name": self.personality.name,
                "description": self.personality.description,
                "decision_thresholds": self.personality.decision_thresholds,
                "explanation_style": self.personality.explanation_style,
                "personality_id": getattr(self.personality, "personality_id", None),
            }

        persona_scores = (
            getattr(waypoint_analysis, "persona_scores", None)
            or getattr(waypoint_analysis, "scores", {})
        )
        objective_scores = getattr(
            waypoint_analysis, "objective_scores", getattr(waypoint_analysis, "neutral_scores", {})
        )
        persona_reasoning = getattr(
            waypoint_analysis, "persona_reasoning", getattr(waypoint_analysis, "reasoning", {})
        )
        objective_reasoning = getattr(
            waypoint_analysis, "objective_reasoning", getattr(waypoint_analysis, "neutral_reasoning", {})
        )

        context = {
            "trigger_reason": trigger_reason or TriggerReason.VISUAL_CHANGE,
            "waypoint_analysis": {
                "waypoint_id": waypoint_analysis.waypoint_id,
                "scores": persona_scores,  # Backward compatible key (uses persona scores)
                "persona_scores": persona_scores,
                "objective_scores": objective_scores,
                "reasoning": persona_reasoning,  # Backward compatible key
                "persona_reasoning": persona_reasoning,
                "objective_reasoning": objective_reasoning,
                "gps": waypoint_analysis.gps,
                "visual_change": waypoint_analysis.visual_change_detected,
                "phash_distance": waypoint_analysis.phash_distance,
            },
            "stm_context": stm_context,
            "recent_context": recent_context,
            "ltm_patterns": past_experience,
            "personality": personality_info,
            "agent_profile": self.profile,
            "current_status": self.status,
            "route_metadata": {
                "waypoints_processed": self._waypoints_processed,
                "system2_triggers": self._system2_triggers,
            },
            "image_path": waypoint_analysis.image_path,
        }

        return context

    def _retrieve_relevant_ltm(
        self, current_gps: Tuple[float, float], radius_meters: float = 500.0
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant experiences from LTM based on GPS proximity.

        Args:
            current_gps: (latitude, longitude) of current waypoint
            radius_meters: Search radius in meters

        Returns:
            List of relevant LTM moments (limited to 3-5 most relevant)
        """
        # For now, return empty list
        # TODO: Implement GPS-based LTM retrieval when needed
        # This would query episodic_ltm for moments within radius_meters
        return []

    # ========================================================================
    # Waypoint Processing Pipeline
    # ========================================================================

    def process_waypoint(
        self,
        waypoint_analysis: Any,  # WaypointAnalysis type hint
        triggered: bool = False,
        trigger_reason: Optional[TriggerReason] = None,
    ) -> Optional[Dict[str, Any]]:
        """Main processing pipeline for each waypoint.

        Flow:
        1. Check attention gate
        2. Add to STM if passes
        3. If caller indicates trigger, prepare context
        4. Track statistics

        Args:
            waypoint_analysis: VLM evaluation result from ContinuousAnalyzer

        Returns:
            None if no thinking needed, or context dict if System 2 triggered

        Example::

            context = memory_manager.process_waypoint(waypoint_analysis)

            if context is not None:
                # System 2 triggered - perform deep reasoning
                thinking_result = thinking_module.reason(context)
        """
        self._waypoints_processed += 1

        persona_scores = (
            getattr(waypoint_analysis, "persona_scores", None)
            or getattr(waypoint_analysis, "scores", {})
        )

        # Step 1: Apply attention gate
        if not self._passes_attention_gate(waypoint_analysis):
            self.logger.debug(
                "Waypoint filtered by attention gate",
                waypoint_id=waypoint_analysis.waypoint_id,
            )
            return None

        # Step 2: Add to STM
        self._attention_gate_passes += 1

        self.stm.add(
            waypoint_id=waypoint_analysis.waypoint_id,
            scores=persona_scores,
            summary=f"Waypoint {waypoint_analysis.waypoint_id}",
            image_path=(
                waypoint_analysis.image_path
                if waypoint_analysis.visual_change_detected
                else None
            ),
            gps=waypoint_analysis.gps,
            timestamp=waypoint_analysis.timestamp,
        )

        self.logger.debug(
            "Waypoint added to STM",
            waypoint_id=waypoint_analysis.waypoint_id,
            stm_size=self.stm.get_memory_size(),
        )

        # Step 3: Return context only if triggered by controller
        if not triggered:
            return None

        self._system2_triggers += 1

        context = self.prepare_context_for_thinking(
            waypoint_analysis,
            trigger_reason=trigger_reason or TriggerReason.VISUAL_CHANGE,
        )

        self.logger.info(
            "System 2 reasoning triggered",
            waypoint_id=waypoint_analysis.waypoint_id,
            total_triggers=self._system2_triggers,
            trigger_rate=f"{self._system2_triggers}/{self._waypoints_processed}",
        )

        return context

    def update_with_system2_result(
        self, waypoint_id: int, thinking_result: Any  # ThinkingResult
    ) -> None:
        """Update STM with System 2 revised scores.

        Note: Accesses STM._memory directly as a temporary workaround.
        TODO: Add public API to ShortTermMemory for updating items.
        """
        self._system2_results[waypoint_id] = thinking_result

        if getattr(thinking_result, "revised_scores", None):
            # Access internal deque (workaround - should use public API)
            for item in self.stm._memory:
                if item.waypoint_id == waypoint_id:
                    item.scores = thinking_result.revised_scores
                    item.summary = f"Waypoint {waypoint_id} (System 2: {thinking_result.significance})"
                    adjustments = (
                        getattr(thinking_result, "score_adjustments", {}) or {}
                    )
                    self.logger.debug(
                        "Updated STM with System 2 scores",
                        waypoint_id=waypoint_id,
                        adjustments={k: f"{v:+.1f}" for k, v in adjustments.items()},
                    )
                    break

    # ========================================================================
    # Route Consolidation: STM → LTM
    # ========================================================================

    def complete_route(
        self,
        route_data: Dict[str, Any],
        thinking_history: List[Any],  # List[ThinkingResult]
    ) -> RouteSummary:
        """Consolidate STM → LTM when route analysis completes.

        Steps:
        1. Extract key moments from STM
        2. Detect patterns across waypoints
        3. Generate route summary
        4. Store in episodic LTM
        5. Clear STM for next route

        Args:
            route_data: Complete route information (route_id, length_km, etc.)
            thinking_history: All System 2 reasoning results from ThinkingModule

        Returns:
            RouteSummary object with complete route characterization

        Side Effects:
            - Clears STM
            - Writes to LTM files
            - Logs consolidation statistics
        """
        route_id = route_data.get("route_id", "unknown")
        route_length_km = route_data.get("length_km", 0.0)

        self.logger.info(
            "Starting route consolidation",
            route_id=route_id,
            stm_size=self.stm.get_memory_size(),
            thinking_episodes=len(thinking_history),
        )

        # Step 1: Get all STM contents
        stm_context = self.stm.get_context()

        # Convert STM to analysis format for LTM
        all_analyses = []
        if stm_context and stm_context.get("recent_scores"):
            recent_scores = stm_context["recent_scores"]
            recent_summaries = stm_context.get("recent_summaries", [])
            waypoint_ids = stm_context.get("waypoint_ids", [])

            for i, scores in enumerate(recent_scores):
                all_analyses.append(
                    {
                        "waypoint_id": waypoint_ids[i] if i < len(waypoint_ids) else i,
                        "scores": scores,
                        "summary": (
                            recent_summaries[i] if i < len(recent_summaries) else ""
                        ),
                    }
                )

        # Step 2: Add candidate moments from thinking history
        for thinking_result in thinking_history:
            # Extract significance from thinking result
            significance = getattr(thinking_result, "significance", "medium")
            confidence = getattr(thinking_result, "confidence", 0.5)
            waypoint_id = getattr(thinking_result, "waypoint_id", 0)

            # Find corresponding scores from STM
            scores = {}
            for analysis in all_analyses:
                if analysis["waypoint_id"] == waypoint_id:
                    scores = analysis["scores"]
                    break

            # Add to LTM candidate moments
            # Note: We would need image_path, gps, etc. from original waypoint_analysis
            # For now, skip if we don't have complete info
            if scores:
                self.logger.debug(
                    "Candidate moment identified",
                    waypoint_id=waypoint_id,
                    significance=significance,
                )

        # Step 3: Extract patterns
        thinking_texts = [getattr(t, "interpretation", "") for t in thinking_history]

        self.episodic_ltm.extract_patterns(
            all_analyses=all_analyses, thinking_history=thinking_texts
        )

        # Step 4: Generate route summary
        route_summary = self.episodic_ltm.generate_route_summary(
            route_id=route_id,
            total_waypoints=len(all_analyses),
            length_km=route_length_km,
            all_analyses=all_analyses,
        )

        # Step 5: Clear STM for next route
        self.stm.clear()

        # Log statistics
        self.logger.info(
            "Route consolidation complete",
            route_id=route_id,
            recommendation=route_summary.overall_recommendation,
            patterns=len(route_summary.patterns),
            barriers=len(route_summary.major_barriers),
            waypoints_processed=self._waypoints_processed,
            attention_gate_pass_rate=f"{self._attention_gate_passes}/{self._waypoints_processed}",
            system2_trigger_rate=f"{self._system2_triggers}/{self._waypoints_processed}",
        )

        # Reset statistics for next route
        self._waypoints_processed = 0
        self._attention_gate_passes = 0
        self._system2_triggers = 0

        return route_summary

    # ========================================================================
    # Statistics and Introspection
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory manager statistics.

        Returns:
            Dictionary containing:
            - waypoints_processed: Total waypoints seen
            - attention_gate_passes: Waypoints that entered STM
            - system2_triggers: Times ThinkingModule was triggered
            - stm_size: Current STM occupancy
            - ltm_stats: LTM statistics
        """
        ltm_stats = self.episodic_ltm.get_statistics()

        return {
            "waypoints_processed": self._waypoints_processed,
            "attention_gate_passes": self._attention_gate_passes,
            "system2_triggers": self._system2_triggers,
            "attention_gate_pass_rate": (
                self._attention_gate_passes / self._waypoints_processed
                if self._waypoints_processed > 0
                else 0.0
            ),
            "system2_trigger_rate": (
                self._system2_triggers / self._waypoints_processed
                if self._waypoints_processed > 0
                else 0.0
            ),
            "stm_size": self.stm.get_memory_size(),
            "stm_full": self.stm.is_full(),
            "ltm_stats": ltm_stats,
        }
