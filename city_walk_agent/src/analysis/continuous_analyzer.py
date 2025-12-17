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
    """Analysis results for a single waypoint during continuous traversal"""

    waypoint_id: int
    image_path: Path
    scores: Dict[str, float]  # Final scores (with persona if applied)
    reasoning: Dict[str, str]  # Final reasoning
    timestamp: str
    gps: Tuple[float, float]
    heading: float
    visual_change_detected: bool
    phash_distance: Optional[float]

    # Dual VLM evaluation fields
    neutral_scores: Optional[Dict[str, float]] = None  # Scores without persona
    persona_adjustments: Optional[Dict[str, float]] = (
        None  # Difference (persona - neutral)
    )
    neutral_reasoning: Optional[Dict[str, str]] = None  # Reasoning without persona
    persona_applied: bool = False  # Flag indicating if persona was used


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
        multi_image_threshold: float = 30.0,
        enable_multi_image: bool = True,
        personality: Optional[Any] = None,
        persona_hint: Optional[str] = None,
    ):
        """
        Initialize continuous analyzer with VLM evaluator

        Args:
            framework_id: ID of evaluation framework to use
            context_window: Number of previous waypoints to maintain in memory
            multi_image_threshold: pHash threshold to trigger multi-image evaluation
            enable_multi_image: Enable/disable multi-image comparison feature
            personality: Optional personality object for persona-aware evaluation
            persona_hint: Optional persona hint to prepend to evaluation prompts
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
        self.multi_image_threshold = multi_image_threshold
        self.enable_multi_image = enable_multi_image
        self.multi_image_evaluations: int = 0

        # History for temporal context
        self.analysis_history: List[WaypointAnalysis] = []
        self.context_window = context_window

        # Statistics tracking
        self.phash_distances: List[float] = []  # Track distances for statistics

        # Store personality for persona-aware evaluation
        self.personality = personality
        self.persona_hint = persona_hint

        logger.info(
            f"Analyzer initialized - context_window={context_window}, "
            f"multi_image: {'enabled' if enable_multi_image else 'disabled'}, "
            f"multi_image_threshold={multi_image_threshold}, "
            f"persona_aware: {persona_hint is not None}"
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
        Analyze single waypoint with visual change info from CognitiveController

        This method:
        1. Receives pre-computed visual change detection from CognitiveController
        2. Determines whether to use multi-image or single-image evaluation
        3. Calls appropriate evaluator method
        4. Returns WaypointAnalysis object

        Args:
            waypoint_id: Sequential ID of this waypoint
            image_path: Path to waypoint image
            metadata: Metadata dict with timestamp, lat, lon, heading
            visual_change_detected: Whether visual change was detected (from CognitiveController)
            phash_distance: pHash distance from previous waypoint (from CognitiveController)

        Returns:
            WaypointAnalysis object with scores, reasoning, and metadata
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

        # 2. Build previous context if available
        previous_context = None
        prev_img_path = None
        if self.analysis_history:
            prev_analysis = self.analysis_history[-1]
            prev_img_path = prev_analysis.image_path
            previous_context = {
                "waypoint_id": prev_analysis.waypoint_id,
                "scores": prev_analysis.scores,
                "reasoning": prev_analysis.reasoning,
                "summary": self._summarize_analysis(prev_analysis),
                "phash_distance": phash_distance,
                "visual_change_detected": visual_change_detected,
            }

        # 3. Determine evaluation mode
        use_multi_image = (
            self.enable_multi_image
            and visual_change_detected
            and phash_distance is not None
            and phash_distance >= self.multi_image_threshold
            and prev_img_path is not None  # Need previous waypoint
        )

        # 4. Choose evaluation method based on trigger logic
        if use_multi_image:
            logger.info(
                f"Using multi-image evaluation for waypoint {waypoint_id} "
                f"(pHash distance: {phash_distance:.1f})"
            )

            comparison_results = self.evaluator.evaluate_with_comparison(
                previous_image_path=prev_img_path,
                current_image_path=image_path,
                previous_context=previous_context,
            )

            # Convert comparison results to standard format
            scores = {
                dim_id: res["score"] for dim_id, res in comparison_results.items()
            }
            reasoning = {
                dim_id: res["reasoning"] for dim_id, res in comparison_results.items()
            }

            # For multi-image, run persona-aware evaluation separately if hint provided
            neutral_scores = scores.copy()
            neutral_reasoning = reasoning.copy()
            persona_adjustments = None
            persona_applied = False

            if self.persona_hint:
                logger.info(
                    f"🔶 Dual VLM (multi-image mode): Making persona-aware call for waypoint {waypoint_id}"
                )
                persona_results = self.evaluator.evaluate_image(
                    str(image_path),
                    previous_context=previous_context,
                    persona_hint=self.persona_hint,
                )
                persona_scores = {
                    r["dimension_id"]: r["score"] for r in persona_results
                }
                persona_reasoning = {
                    r["dimension_id"]: r["reasoning"] for r in persona_results
                }

                persona_adjustments = {
                    dim: round(
                        persona_scores.get(dim, neutral_scores[dim])
                        - neutral_scores[dim],
                        2,
                    )
                    for dim in neutral_scores
                }
                persona_applied = True

                # Log significant adjustments
                significant_adjustments = {
                    dim: adj
                    for dim, adj in persona_adjustments.items()
                    if abs(adj) >= 0.5
                }
                if significant_adjustments:
                    logger.info(
                        f"   Persona adjustments (WP {waypoint_id}, multi-img): {significant_adjustments}"
                    )

                # Final scores reflect persona-aware evaluation
                scores = persona_scores
                reasoning = persona_reasoning
            else:
                logger.warning(
                    f"⚠️ Multi-image mode: No persona hint for waypoint {waypoint_id}"
                )

            self.multi_image_evaluations += 1

        else:
            # Standard single-image evaluation with dual VLM calls (neutral + persona)
            logger.debug(
                f"Using single-image evaluation with context for waypoint {waypoint_id}"
            )

            # === VLM Call 1: Neutral (no persona) ===
            eval_results = self.evaluator.evaluate_image(
                str(image_path),
                previous_context=previous_context,
                persona_hint=None,  # No persona for neutral evaluation
            )

            neutral_scores = {r["dimension_id"]: r["score"] for r in eval_results}
            neutral_reasoning = {
                r["dimension_id"]: r["reasoning"] for r in eval_results
            }

            # === VLM Call 2: With Persona (if persona_hint provided) ===
            if self.persona_hint:
                logger.info(
                    f"🔶 Dual VLM: Making persona-aware call for waypoint {waypoint_id}"
                )
                persona_results = self.evaluator.evaluate_image(
                    str(image_path),
                    previous_context=previous_context,
                    persona_hint=self.persona_hint,
                )
                persona_scores = {
                    r["dimension_id"]: r["score"] for r in persona_results
                }
                persona_reasoning = {
                    r["dimension_id"]: r["reasoning"] for r in persona_results
                }

                # Calculate adjustments
                persona_adjustments = {
                    dim: round(persona_scores[dim] - neutral_scores[dim], 2)
                    for dim in neutral_scores
                }
                persona_applied = True

                # Log significant adjustments
                significant_adjustments = {
                    dim: adj
                    for dim, adj in persona_adjustments.items()
                    if abs(adj) >= 0.5
                }
                if significant_adjustments:
                    logger.info(
                        f"   Persona adjustments (WP {waypoint_id}): {significant_adjustments}"
                    )

                # Final scores = persona scores
                scores = persona_scores
                reasoning = persona_reasoning
            else:
                logger.warning(
                    f"⚠️ No persona hint available for waypoint {waypoint_id} - "
                    f"using neutral scores only"
                )
                # No persona - use neutral as final
                persona_scores = neutral_scores
                persona_reasoning = neutral_reasoning
                persona_adjustments = None
                persona_applied = False

                scores = neutral_scores
                reasoning = neutral_reasoning

        # 5. Build WaypointAnalysis object with dual score fields
        analysis = WaypointAnalysis(
            waypoint_id=waypoint_id,
            image_path=image_path,
            scores=scores,  # Final = persona (or neutral if no persona)
            reasoning=reasoning,
            timestamp=metadata.get("timestamp", ""),
            gps=(metadata.get("lat", 0.0), metadata.get("lon", 0.0)),
            heading=metadata.get("heading", 0.0),
            visual_change_detected=visual_change_detected,
            phash_distance=phash_distance,
            # NEW: Dual score fields
            neutral_scores=neutral_scores,
            persona_adjustments=persona_adjustments,
            neutral_reasoning=neutral_reasoning,
            persona_applied=persona_applied,
        )

        logger.debug(
            f"Waypoint {waypoint_id} analyzed - "
            f"avg_score={sum(scores.values()) / len(scores):.2f}, "
            f"visual_change={visual_change_detected}, "
            f"mode={'multi-image' if use_multi_image else 'single-image'}"
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
        Return statistics about visual changes across the route

        Useful for understanding route characteristics:
        - How many significant visual transitions occurred?
        - What's the typical pHash distance between waypoints?
        - Are changes frequent (urban) or rare (monotonous)?

        Returns:
            Dictionary containing:
            - total_waypoints: Total number of analyzed waypoints
            - visual_changes_detected: Count of waypoints with visual changes
            - avg_phash_distance: Average pHash distance across waypoints
            - phash_distances: List of all pHash distances (for plotting)
            - multi_image_evaluations: Count of multi-image comparisons performed
        """
        if not self.analysis_history:
            return {}

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
            "multi_image_evaluations": self.multi_image_evaluations,
        }

        logger.info(
            f"Statistics: {stats['visual_changes_detected']} changes in "
            f"{stats['total_waypoints']} waypoints "
            f"(avg distance={stats['avg_phash_distance']:.2f}), "
            f"multi-image evals: {stats['multi_image_evaluations']}"
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
        Create concise summary of waypoint analysis for context

        Args:
            analysis: WaypointAnalysis object to summarize

        Returns:
            Human-readable summary string
        """
        if not analysis.scores:
            return "No evaluation data"

        avg_score = sum(analysis.scores.values()) / len(analysis.scores)

        # Find highest and lowest scoring dimensions
        best_dim = max(analysis.scores.items(), key=lambda x: x[1])
        worst_dim = min(analysis.scores.items(), key=lambda x: x[1])

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
