"""
Continuous analyzer - real-time waypoint analysis engine

This is a companion to SequentialAnalyzer:
- SequentialAnalyzer: Post-hoc batch analysis of completed routes
- ContinuousAnalyzer: Real-time sequential analysis during route traversal

ContinuousAnalyzer receives visual change detection results from CognitiveController
and focuses on VLM evaluation with temporal context tracking and short-term memory
integration for adaptive route analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.evaluator import Evaluator
from src.evaluation.vlm_client import VLMConfig
from src.config import settings, load_framework
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WaypointAnalysis:
    """Analysis results for a single waypoint during continuous traversal.

    Uses dual evaluation system with independent objective and persona scores.
    """

    waypoint_id: int
    image_path: Path

    # Dual evaluation results (independent)
    objective_scores: Dict[str, float]  # Objective evaluation scores (research)
    objective_reasoning: Dict[str, str]  # Objective evaluation reasoning
    persona_scores: Dict[str, float]  # Persona-aware scores (final/decision-making)
    persona_reasoning: Dict[str, str]  # Persona-aware reasoning

    # Metadata
    timestamp: str
    gps: Tuple[float, float]
    heading: float
    visual_change_detected: bool
    phash_distance: Optional[float]


class ContinuousAnalyzer:
    """
    Real-time waypoint analyzer that receives visual change detection from CognitiveController

    Key differences from SequentialAnalyzer:
    - Processes waypoints sequentially as they arrive (simulates real-time)
    - Receives visual change detection results from CognitiveController
    - Maintains sliding context window of recent waypoints for temporal reasoning
    - Performs VLM evaluation with temporal context

    The Evaluator performs N VLM calls per image (1 per dimension),
    enriched with:
    - Pre-computed visual change detection (from CognitiveController)
    - Change point detection for multi-image evaluation
    - Short-term memory (analysis history)

    Usage:
        ```python
        analyzer = ContinuousAnalyzer(
            framework_id="walkability_v1",
            context_window=3
        )
        # Visual change info comes from CognitiveController
        results = analyzer.analyze_route(image_paths, metadata_list)
        stats = analyzer.get_statistics()
        change_points = analyzer.get_change_points()
        ```
    """

    def __init__(
        self,
        framework_id: str,
        context_window: int = 3,
        enable_multi_image: bool = True,
        persona: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ):
        """
        Initialize continuous analyzer with VLM evaluator

        Args:
            framework_id: ID of evaluation framework to use
            context_window: Number of previous waypoints to maintain in memory
            enable_multi_image: Enable/disable multi-image comparison feature
            persona: Optional EnhancedPersonalityConfig for persona-aware evaluation
            memory_manager: Optional MemoryManager for STM integration
        """
        logger.info(f"Initializing ContinuousAnalyzer with framework: {framework_id}")

        # Initialize Evaluator (reuse existing VLM evaluation logic)
        framework = load_framework(framework_id)
        vlm_config = VLMConfig(
            api_key=settings.qwen_vlm_api_key,
            model=settings.qwen_vlm_model,
            api_url=settings.qwen_vlm_api_url,
        )
        self.evaluator = Evaluator(vlm_config, framework, max_concurrent=5)

        # Multi-image evaluation configuration
        self.enable_multi_image = enable_multi_image
        self.multi_image_evaluations: int = 0

        # History for temporal context
        self.analysis_history: List[WaypointAnalysis] = []
        self.context_window = context_window

        # Statistics tracking
        self.phash_distances: List[float] = []  # Track distances for statistics

        # Store persona for dual evaluation
        self.persona = persona

        # Memory manager for STM integration (Phase 2)
        self.memory_manager = memory_manager

        logger.info(
            f"Analyzer initialized - context_window={context_window}, "
            f"multi_image: {'enabled' if enable_multi_image else 'disabled'}, "
            f"persona_aware: {persona is not None}, "
            f"stm_enabled: {memory_manager is not None}"
        )

    def analyze_waypoint(
        self,
        waypoint_id: int,
        image_path: Path,
        metadata: Dict,
        visual_change_detected: bool = False,
        phash_distance: Optional[float] = None,
    ) -> WaypointAnalysis:
        """
        Analyze single waypoint using dual evaluation (objective + persona).

        This method performs parallel dual evaluation if a persona is configured.
        Both objective and persona evaluations are independent - no weight-based adjustments.

        Args:
            waypoint_id: Sequential ID of this waypoint
            image_path: Path to waypoint image
            metadata: Metadata dict with timestamp, lat, lon, heading
            visual_change_detected: Whether visual change was detected (from CognitiveController)
            phash_distance: pHash distance from previous waypoint (from CognitiveController)

        Returns:
            WaypointAnalysis object with dual evaluation results
        """
        logger.debug(f"Analyzing waypoint {waypoint_id}: {image_path}")

        # Track pHash distance for statistics
        if phash_distance is not None:
            self.phash_distances.append(phash_distance)

        if visual_change_detected:
            distance_display = (
                f"{phash_distance:.2f}" if phash_distance is not None else "N/A"
            )
            logger.info(
                f"Visual change detected at waypoint {waypoint_id} "
                f"(phash_distance={distance_display})"
            )

        # Build previous context with STM integration (Phase 2)
        previous_context = None
        stm_context = None

        # Retrieve STM context if memory_manager is available
        if self.memory_manager:
            stm_context = self.memory_manager.stm.get_context()
            logger.debug(
                f"Retrieved STM context for waypoint {waypoint_id}: "
                f"{len(stm_context.get('recent_scores', []))} recent waypoints"
            )

        if self.analysis_history:
            prev_analysis = self.analysis_history[-1]
            previous_context = {
                "waypoint_id": prev_analysis.waypoint_id,
                "scores": prev_analysis.persona_scores,  # Use persona scores for context
                "reasoning": prev_analysis.persona_reasoning,
                "summary": self._summarize_analysis(prev_analysis),
                "phash_distance": phash_distance,
                "visual_change_detected": visual_change_detected,
                "stm_context": stm_context,  # Include STM context for temporal awareness
            }
        elif stm_context:
            # No previous single analysis, but STM has context from earlier waypoints
            previous_context = {
                "stm_context": stm_context,
                "visual_change_detected": visual_change_detected,
            }

        # Perform dual evaluation if persona is configured
        if self.persona:
            # Multi-image evaluation trigger logic (Phase 2)
            # When visual_change_detected = True: use multi-image VLM call (previous + current)
            # When visual_change_detected = False: use single-image + text context from STM
            use_multi_image = self.enable_multi_image and visual_change_detected and self.analysis_history

            if use_multi_image:
                logger.info(
                    f"🔶 Dual evaluation with MULTI-IMAGE for waypoint {waypoint_id} "
                    f"(persona: {self.persona.name}, visual change detected)"
                )
                self.multi_image_evaluations += 1
            else:
                context_type = "STM text context" if stm_context else "no prior context"
                logger.info(
                    f"🔶 Dual evaluation with SINGLE-IMAGE for waypoint {waypoint_id} "
                    f"(persona: {self.persona.name}, using {context_type})"
                )

            # Single call returns both objective and persona results
            # Note: Multi-image evaluation is controlled by previous_context containing image info
            dual_results = self.evaluator.evaluate_image(
                str(image_path),
                previous_context=previous_context,
                persona=self.persona,
                evaluation_mode="dual"
            )

            # Extract objective and persona scores from DualEvaluationResult objects
            objective_scores = {}
            objective_reasoning = {}
            persona_scores = {}
            persona_reasoning = {}

            for result in dual_results:
                # result is a DualEvaluationResult object
                objective_scores[result.dimension_id] = result.objective_score
                objective_reasoning[result.dimension_id] = result.objective_reasoning
                persona_scores[result.dimension_id] = result.persona_score
                persona_reasoning[result.dimension_id] = result.persona_reasoning

            # Log significant differences between objective and persona scores
            score_differences = {
                dim: round(persona_scores[dim] - objective_scores[dim], 2)
                for dim in objective_scores
            }
            significant_diffs = {
                dim: diff for dim, diff in score_differences.items()
                if abs(diff) >= 0.5
            }
            if significant_diffs:
                logger.info(
                    f"   Objective vs Persona differences (WP {waypoint_id}): {significant_diffs}"
                )

        else:
            # No persona - run objective evaluation only
            # Apply same multi-image logic
            use_multi_image = self.enable_multi_image and visual_change_detected and self.analysis_history

            if use_multi_image:
                logger.info(
                    f"Objective-only evaluation with MULTI-IMAGE for waypoint {waypoint_id} "
                    f"(visual change detected)"
                )
                self.multi_image_evaluations += 1
            else:
                logger.info(f"Objective-only evaluation with SINGLE-IMAGE for waypoint {waypoint_id}")

            eval_results = self.evaluator.evaluate_image(
                str(image_path),
                previous_context=previous_context,
                evaluation_mode="objective"
            )

            # Extract scores and reasoning
            objective_scores = {r["dimension_id"]: r["score"] for r in eval_results}
            objective_reasoning = {r["dimension_id"]: r["reasoning"] for r in eval_results}

            # No persona - use objective scores for persona fields as well
            persona_scores = objective_scores.copy()
            persona_reasoning = objective_reasoning.copy()

        # Build WaypointAnalysis object with dual evaluation results
        analysis = WaypointAnalysis(
            waypoint_id=waypoint_id,
            image_path=image_path,
            objective_scores=objective_scores,
            objective_reasoning=objective_reasoning,
            persona_scores=persona_scores,
            persona_reasoning=persona_reasoning,
            timestamp=metadata.get("timestamp", ""),
            gps=(metadata.get("lat", 0.0), metadata.get("lon", 0.0)),
            heading=metadata.get("heading", 0.0),
            visual_change_detected=visual_change_detected,
            phash_distance=phash_distance,
        )

        logger.debug(
            f"Waypoint {waypoint_id} analyzed - "
            f"avg_objective={sum(objective_scores.values()) / len(objective_scores):.2f}, "
            f"avg_persona={sum(persona_scores.values()) / len(persona_scores):.2f}, "
            f"visual_change={visual_change_detected}"
        )

        return analysis

    def analyze_route(
        self, image_paths: List[Path], waypoint_metadata: List[Dict]
    ) -> List[WaypointAnalysis]:
        """
        Analyze all waypoints sequentially with temporal context

        This is the main analysis method that processes each waypoint by calling
        analyze_waypoint() and maintaining the analysis history.

        Args:
            image_paths: Ordered list of paths to waypoint images
            waypoint_metadata: List of metadata dicts with keys:
                - timestamp: ISO format timestamp
                - lat, lon: GPS coordinates
                - heading: Compass heading in degrees

        Returns:
            List of WaypointAnalysis objects, one per waypoint in order

        Raises:
            ValueError: If image_paths and waypoint_metadata lengths don't match
        """
        logger.info(f"Starting route analysis: {len(image_paths)} waypoints")

        if len(image_paths) != len(waypoint_metadata):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs "
                f"{len(waypoint_metadata)} metadata entries"
            )

        results: List[WaypointAnalysis] = []

        for i, (img_path, metadata) in enumerate(zip(image_paths, waypoint_metadata)):
            logger.debug(f"Processing waypoint {i}/{len(image_paths)}: {img_path}")

            # Analyze waypoint using extracted method
            analysis = self.analyze_waypoint(
                waypoint_id=i, image_path=img_path, metadata=metadata
            )

            # Update analysis history
            self.analysis_history.append(analysis)

            results.append(analysis)

        logger.info(f"Route analysis complete: {len(results)} waypoints processed")
        return results

    def get_statistics(self) -> Dict:
        """
        Return statistics about visual changes and dual evaluation across the route.

        Returns:
            Dictionary containing:
            - total_waypoints: Total number of analyzed waypoints
            - visual_changes_detected: Count of waypoints with visual changes
            - avg_phash_distance: Average pHash distance across waypoints
            - phash_distances: List of all pHash distances (for plotting)
            - objective_score_stats: Stats for objective evaluation
            - persona_score_stats: Stats for persona evaluation
            - avg_score_difference: Average difference between persona and objective
        """
        if not self.analysis_history:
            return {}

        # Calculate objective score statistics
        all_objective_scores = []
        all_persona_scores = []
        score_differences = []

        for analysis in self.analysis_history:
            obj_scores = list(analysis.objective_scores.values())
            per_scores = list(analysis.persona_scores.values())

            all_objective_scores.extend(obj_scores)
            all_persona_scores.extend(per_scores)

            # Calculate per-waypoint difference
            avg_obj = sum(obj_scores) / len(obj_scores) if obj_scores else 0
            avg_per = sum(per_scores) / len(per_scores) if per_scores else 0
            score_differences.append(avg_per - avg_obj)

        stats = {
            "total_waypoints": len(self.analysis_history),
            "visual_changes_detected": sum(
                1 for a in self.analysis_history if a.visual_change_detected
            ),
            "avg_phash_distance": (
                sum(self.phash_distances) / len(self.phash_distances)
                if self.phash_distances
                else 0
            ),
            "phash_distances": self.phash_distances,
            "objective_score_stats": {
                "avg": sum(all_objective_scores) / len(all_objective_scores) if all_objective_scores else 0,
                "min": min(all_objective_scores) if all_objective_scores else 0,
                "max": max(all_objective_scores) if all_objective_scores else 0,
            },
            "persona_score_stats": {
                "avg": sum(all_persona_scores) / len(all_persona_scores) if all_persona_scores else 0,
                "min": min(all_persona_scores) if all_persona_scores else 0,
                "max": max(all_persona_scores) if all_persona_scores else 0,
            },
            "avg_score_difference": sum(score_differences) / len(score_differences) if score_differences else 0,
        }

        logger.info(
            f"Statistics: {stats['total_waypoints']} waypoints, "
            f"{stats['visual_changes_detected']} changes, "
            f"avg_objective={stats['objective_score_stats']['avg']:.2f}, "
            f"avg_persona={stats['persona_score_stats']['avg']:.2f}, "
            f"avg_diff={stats['avg_score_difference']:.2f}"
        )

        return stats

    def get_context_window(self) -> List[WaypointAnalysis]:
        """
        Get the current context window of recent analyses

        This represents the "short-term memory" available for
        adaptive decision-making during route traversal.

        Returns:
            List of recent WaypointAnalysis objects (up to context_window size)
        """
        return self.analysis_history[-self.context_window :].copy()

    def get_change_points(self) -> List[WaypointAnalysis]:
        """
        Get all waypoints where visual changes were detected

        Useful for identifying key transition points in the route:
        - Street type changes (residential → commercial)
        - Infrastructure transitions (wide → narrow sidewalk)
        - Neighborhood boundaries

        Returns:
            List of WaypointAnalysis objects where visual_change_detected=True
        """
        change_points = [
            analysis
            for analysis in self.analysis_history
            if analysis.visual_change_detected
        ]

        logger.debug(f"Found {len(change_points)} change points")
        return change_points

    def _summarize_analysis(self, analysis: WaypointAnalysis) -> str:
        """
        Create concise summary of waypoint analysis for context.

        Uses persona scores as the primary decision-making scores.

        Args:
            analysis: WaypointAnalysis object to summarize

        Returns:
            Human-readable summary string
        """
        if not analysis.persona_scores:
            return "No evaluation data"

        avg_score = sum(analysis.persona_scores.values()) / len(analysis.persona_scores)

        # Find highest and lowest scoring dimensions
        best_dim = max(analysis.persona_scores.items(), key=lambda x: x[1])
        worst_dim = min(analysis.persona_scores.items(), key=lambda x: x[1])

        summary = (
            f"Average score: {avg_score:.1f}/10. "
            f"Best: {best_dim[0]} ({best_dim[1]:.1f}), "
            f"Worst: {worst_dim[0]} ({worst_dim[1]:.1f})"
        )

        return summary

    def reset(self) -> None:
        """Reset analyzer state for new route analysis.

        Clears:
        - Analysis history
        - pHash distance statistics
        - Multi-image evaluation counter

        This ensures each route is analyzed independently without
        cross-contamination from previous routes.
        """
        self.analysis_history.clear()
        self.phash_distances.clear()
        self.multi_image_evaluations = 0
        logger.debug("ContinuousAnalyzer state reset for new route")
