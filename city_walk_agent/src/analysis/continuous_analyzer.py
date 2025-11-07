"""
Continuous analyzer - real-time waypoint analysis engine

This is a companion to SequentialAnalyzer:
- SequentialAnalyzer: Post-hoc batch analysis of completed routes
- ContinuousAnalyzer: Real-time sequential analysis during route traversal

Both wrap the existing Evaluator (N VLM calls per image), but ContinuousAnalyzer
adds pHash-based visual change detection, temporal context tracking, and
short-term memory integration for adaptive route analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import imagehash
from PIL import Image

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
    scores: Dict[str, float]  # {dimension_id: score}
    reasoning: Dict[str, str]  # {dimension_id: reasoning text}
    timestamp: str
    gps: Tuple[float, float]
    heading: float
    visual_change_detected: bool
    phash_distance: Optional[float]


class ContinuousAnalyzer:
    """
    Real-time waypoint analyzer with visual change detection

    Key differences from SequentialAnalyzer:
    - Processes waypoints sequentially as they arrive (simulates real-time)
    - Adds pHash-based visual change detection to identify significant transitions
    - Maintains sliding context window of recent waypoints for temporal reasoning
    - Uses adaptive threshold learning from observed pHash variance

    The Evaluator still performs N VLM calls per image (1 per dimension),
    but we enrich this with:
    - Visual similarity tracking (pHash distance)
    - Change point detection (adaptive thresholding)
    - Short-term memory (analysis history)

    Usage:
        ```python
        analyzer = ContinuousAnalyzer(
            framework_id="walkability_v1",
            context_window=3,
            adaptive_threshold=True
        )
        results = analyzer.analyze_route(image_paths, metadata_list)
        stats = analyzer.get_statistics()
        change_points = analyzer.get_change_points()
        ```
    """

    def __init__(
        self,
        framework_id: str,
        context_window: int = 3,
        phash_threshold: float = 15.0,
        adaptive_threshold: bool = True
    ):
        """
        Initialize continuous analyzer with VLM evaluator and change detection

        Args:
            framework_id: ID of evaluation framework to use
            context_window: Number of previous waypoints to maintain in memory
            phash_threshold: Fixed threshold for pHash distance (if not adaptive)
            adaptive_threshold: Use adaptive threshold based on running statistics
        """
        logger.info(f"Initializing ContinuousAnalyzer with framework: {framework_id}")

        # Initialize Evaluator (reuse existing VLM evaluation logic)
        framework = load_framework(framework_id)
        vlm_config = VLMConfig(
            api_key=settings.qwen_vlm_api_key,
            model=settings.qwen_vlm_model,
            api_url=settings.qwen_vlm_api_url
        )
        self.evaluator = Evaluator(vlm_config, framework, max_concurrent=5)

        # pHash tracking for visual change detection
        self.last_phash: Optional[imagehash.ImageHash] = None
        self.phash_distances: List[float] = []
        self.phash_threshold = phash_threshold
        self.adaptive_threshold = adaptive_threshold

        # History for temporal context
        self.analysis_history: List[WaypointAnalysis] = []
        self.context_window = context_window

        logger.info(
            f"Analyzer initialized - context_window={context_window}, "
            f"phash_threshold={phash_threshold}, adaptive={adaptive_threshold}"
        )

    def analyze_route(
        self,
        image_paths: List[Path],
        waypoint_metadata: List[Dict]
    ) -> List[WaypointAnalysis]:
        """
        Analyze all waypoints sequentially with temporal context

        This is the main analysis method that processes each waypoint by:
        1. Computing pHash for visual change detection
        2. Calling Evaluator for dimension scoring (N VLM calls per image)
        3. Building WaypointAnalysis with enriched metadata
        4. Updating analysis history for context window

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

            # 1. Compute pHash for visual change detection
            current_phash = self._compute_phash(img_path)
            visual_change, phash_dist = self._detect_visual_change(current_phash)

            if visual_change:
                logger.info(
                    f"Visual change detected at waypoint {i} "
                    f"(phash_distance={phash_dist:.2f})"
                )

            # 2. Call existing Evaluator (N VLM calls internally)
            eval_results = self.evaluator.evaluate_image(str(img_path))

            # 3. Convert to WaypointAnalysis format
            scores = {r['dimension_id']: r['score'] for r in eval_results}
            reasoning = {r['dimension_id']: r['reasoning'] for r in eval_results}

            analysis = WaypointAnalysis(
                waypoint_id=i,
                image_path=img_path,
                scores=scores,
                reasoning=reasoning,
                timestamp=metadata.get('timestamp', ''),
                gps=(metadata.get('lat', 0.0), metadata.get('lon', 0.0)),
                heading=metadata.get('heading', 0.0),
                visual_change_detected=visual_change,
                phash_distance=phash_dist
            )

            # 4. Update state for next iteration
            self.last_phash = current_phash
            self.analysis_history.append(analysis)

            # Maintain context window (sliding window)
            if len(self.analysis_history) > self.context_window:
                self.analysis_history.pop(0)

            results.append(analysis)

            logger.debug(
                f"Waypoint {i} analyzed - "
                f"avg_score={sum(scores.values()) / len(scores):.2f}, "
                f"visual_change={visual_change}"
            )

        logger.info(f"Route analysis complete: {len(results)} waypoints processed")
        return results

    def _compute_phash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute perceptual hash (pHash) for an image

        pHash provides a robust fingerprint that's resilient to minor
        variations but sensitive to significant visual changes.

        Args:
            image_path: Path to the image file

        Returns:
            ImageHash object representing the perceptual hash

        Raises:
            Exception: If image cannot be loaded or processed
        """
        try:
            img = Image.open(image_path)
            return imagehash.phash(img)
        except Exception as e:
            logger.error(f"Failed to compute pHash for {image_path}: {e}")
            raise

    def _detect_visual_change(
        self,
        current_phash: imagehash.ImageHash
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect visual change based on pHash distance

        Uses either fixed or adaptive threshold:
        - Fixed: Use phash_threshold directly
        - Adaptive: Use mean + 1.5 * std of observed distances (after 5+ samples)

        The Hamming distance between pHashes indicates visual dissimilarity.
        Higher values = more different images.

        Args:
            current_phash: pHash of the current image

        Returns:
            Tuple of (change_detected, phash_distance)
            - change_detected: True if distance exceeds threshold
            - phash_distance: Hamming distance, or None for first image
        """
        # First image has no reference
        if self.last_phash is None:
            return False, None

        # Compute Hamming distance between hashes
        distance = float(current_phash - self.last_phash)
        self.phash_distances.append(distance)

        # Determine threshold
        if self.adaptive_threshold and len(self.phash_distances) >= 5:
            # Adaptive: Statistical approach using mean + 1.5 * std
            mean = sum(self.phash_distances) / len(self.phash_distances)
            variance = sum(
                (x - mean) ** 2 for x in self.phash_distances
            ) / len(self.phash_distances)
            std = variance ** 0.5
            threshold = mean + 1.5 * std

            logger.debug(
                f"Adaptive threshold: {threshold:.2f} "
                f"(mean={mean:.2f}, std={std:.2f})"
            )
        else:
            # Fixed threshold
            threshold = self.phash_threshold

        change_detected = distance > threshold
        return change_detected, distance

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
                if self.phash_distances else 0
            ),
            "phash_distances": self.phash_distances
        }

        logger.info(
            f"Statistics: {stats['visual_changes_detected']} changes in "
            f"{stats['total_waypoints']} waypoints "
            f"(avg distance={stats['avg_phash_distance']:.2f})"
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
        return self.analysis_history.copy()

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
            analysis for analysis in self.analysis_history
            if analysis.visual_change_detected
        ]

        logger.debug(f"Found {len(change_points)} change points")
        return change_points
