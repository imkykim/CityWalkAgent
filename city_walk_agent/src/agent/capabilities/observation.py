"""Observation capability - agent's visual perception through VLM.

This capability handles the perception phase of the agent's cognitive loop.
It uses VLM evaluation to observe and assess walking routes, extracting
structured dimension scores without making decisions.

Design principle: OBSERVE, don't decide.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.config.constants import MAX_CONCURRENT_VLM_CALLS
from src.config import load_framework, settings
from src.evaluation.evaluator import Evaluator
from src.evaluation.vlm_client import VLMConfig
from src.utils.logging import get_logger


class ObservationCapability:
    """Visual perception capability using VLM evaluation.

    This capability translates raw route data (images, waypoints) into
    structured observations with dimension scores and statistics.

    Example:
        ```python
        observer = ObservationCapability()
        observation = observer.observe_route(route_data)
        # observation contains: scores, stats, framework info
        ```
    """

    def __init__(
        self,
        framework_id: Optional[str] = None,
        vlm_config: Optional[VLMConfig] = None,
        max_concurrent: Optional[int] = None,
    ):
        """Initialize observation capability.

        Args:
            framework_id: Evaluation framework to use. Defaults to settings.
            vlm_config: VLM client configuration. Creates from settings if None.
            max_concurrent: Max concurrent VLM calls. Uses constant if None.
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Load framework
        self.framework_id = framework_id or settings.default_framework_id
        self.framework = load_framework(self.framework_id)

        # Create VLM config if not provided
        if vlm_config is None:
            vlm_config = VLMConfig(
                api_key=settings.qwen_vlm_api_key,
                model=settings.qwen_vlm_model,
                api_url=settings.qwen_vlm_api_url,
            )

        # Set max concurrent
        self.max_concurrent = max_concurrent or MAX_CONCURRENT_VLM_CALLS

        # Initialize evaluator
        self.evaluator = Evaluator(
            vlm_config=vlm_config,
            framework=self.framework,
            max_concurrent=self.max_concurrent,
            show_progress=False,  # Agents operate silently
        )

        self.logger.info(
            "ObservationCapability initialized",
            framework_id=self.framework_id,
            max_concurrent=self.max_concurrent,
        )

        # Route data storage (for collected Street View metadata)
        self.route_dir: Optional[Path] = None
        self.metadata: List[Dict[str, Any]] = []
        self.num_waypoints: int = 0

    def load_route_data(self, route_dir: str) -> None:
        """
        Load collected images and metadata for a route.
        
        Loads collection_metadata.json produced by ImageCollector.
        
        Args:
            route_dir: Path to route directory containing collection_metadata.json
        
        Sets:
            self.route_dir: Path object
            self.metadata: List of metadata dicts (one per waypoint)
            self.num_waypoints: Total waypoints in route
            
        Raises:
            FileNotFoundError: If collection_metadata.json not found
        """
        self.route_dir = Path(route_dir)
        
        # Load JSON metadata from ImageCollector
        json_path = self.route_dir / "collection_metadata.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {json_path}. "
                f"Ensure images have been collected with ImageCollector."
            )
        
        self.logger.info("Loading collection metadata", path=str(json_path))
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert ImageCollector format to list of waypoint metadata
        self.metadata = []
        for result in data.get('results', []):
            waypoint_meta = {
                'waypoint_index': result.get('waypoint_id', 0),
                'lat': result.get('lat'),
                'lon': result.get('lon'),
                'heading': result.get('metadata', {}).get('headings', [0])[0] if result.get('metadata') else 0,
                'image_path': result.get('image_path'),
                'success': result.get('download_success', False),
                'timestamp': result.get('timestamp'),
                # Add corner detection fields if present
                'is_corner': result.get('is_corner', False),
                'angle_change': result.get('angle_change', 0.0),
            }
            self.metadata.append(waypoint_meta)
        
        self.num_waypoints = len(self.metadata)
        self.logger.info(
            "Route data loaded",
            num_waypoints=self.num_waypoints,
            route_dir=str(self.route_dir)
        )

    def observe_waypoint(self, waypoint_index: int) -> Dict[str, Any]:
        """
        Analyze what agent sees at a specific waypoint.

        Returns basic observation data for now. VLM evaluation can be added
        later for detailed perception analysis.

        Args:
            waypoint_index: Which waypoint to observe (0 to num_waypoints-1)

        Returns:
            Dict containing:
                - waypoint_index: int
                - location: {'lat': float, 'lon': float}
                - heading: float (0-360 degrees)
                - image_path: str (absolute path to image)
                - is_corner: bool (if corner detection was enabled)
                - angle_change: float (if corner detection was enabled)
                - vlm_evaluation: Dict (placeholder for future VLM analysis)

        Raises:
            ValueError: If route data not loaded or waypoint_index out of bounds

        Example:
            >>> observer = ObservationCapability()
            >>> observer.load_route_data("/data/routes/route_001")
            >>> obs = observer.observe_waypoint(0)
            >>> print(f"At ({obs['location']['lat']}, {obs['location']['lon']})")
            >>> print(f"Heading: {obs['heading']}°")
        """
        if not self.metadata:
            raise ValueError(
                "No route data loaded. Call load_route_data() first."
            )

        if waypoint_index < 0 or waypoint_index >= self.num_waypoints:
            raise ValueError(
                f"waypoint_index {waypoint_index} out of bounds "
                f"(0 to {self.num_waypoints - 1})"
            )

        # Get metadata for this waypoint
        waypoint_meta = self.metadata[waypoint_index]

        # Build absolute image path
        image_path = waypoint_meta.get('image_path')
        if image_path and self.route_dir:
            # Convert relative path to absolute
            abs_image_path = str(self.route_dir / image_path)
        else:
            abs_image_path = None

        # Build observation
        observation = {
            'waypoint_index': waypoint_index,
            'location': {
                'lat': waypoint_meta.get('lat'),
                'lon': waypoint_meta.get('lon'),
            },
            'heading': waypoint_meta.get('heading'),
            'image_path': abs_image_path,
            'success': waypoint_meta.get('success', False),
        }

        # Add corner detection info if available
        if 'is_corner' in waypoint_meta:
            observation['is_corner'] = waypoint_meta['is_corner']
            observation['angle_change'] = waypoint_meta.get('angle_change', 0.0)

        # Placeholder for future VLM evaluation
        # TODO: Integrate VLM evaluation for detailed perception analysis
        # Could use self.evaluator.evaluate_images([abs_image_path])
        observation['vlm_evaluation'] = None

        self.logger.debug(
            "Waypoint observed",
            waypoint_index=waypoint_index,
            heading=waypoint_meta.get('heading'),
            has_image=abs_image_path is not None,
        )

        return observation

    def observe_route(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """Observe a complete route with images.

        This is the main observation method. It evaluates all images along
        a route and returns structured observations.

        Args:
            route_data: Route information containing:
                - route_id: str
                - image_paths: List[str]
                - waypoints: List[Tuple[float, float]] (optional)
                - distance: float (optional)
                - interval: int (optional)

        Returns:
            Dict[str, Any] containing:
                - route_info: Basic route metadata
                - dimension_scores: Scores grouped by dimension
                - dimension_stats: Statistical summary per dimension
                - raw_evaluations: Raw evaluation results
                - framework_id: Framework used
                - observation_timestamp: When observation was made
        """
        route_id = route_data.get("route_id", "unknown")
        image_paths = route_data.get("image_paths", [])

        if not image_paths:
            self.logger.warning("No images to observe", route_id=route_id)
            return self._empty_observation(route_id)

        self.logger.debug(
            "Observing route",
            route_id=route_id,
            num_images=len(image_paths),
        )

        # Evaluate all images
        raw_evaluations = self.evaluator.evaluate_images(image_paths)

        # Group scores by dimension
        dimension_scores = self._group_scores_by_dimension(raw_evaluations)

        # Calculate statistics for each dimension
        dimension_stats = self._calculate_dimension_statistics(dimension_scores)

        # Build observation
        observation = {
            "route_info": {
                "route_id": route_id,
                "route": route_data.get("route"),
                "waypoints": route_data.get("waypoints", []),
                "distance": route_data.get("distance"),
                "interval": route_data.get("interval"),
                "num_images": len(image_paths),
            },
            "dimension_scores": dimension_scores,
            "dimension_stats": dimension_stats,
            "raw_evaluations": raw_evaluations,
            "framework_id": self.framework_id,
            "observation_timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            "Route observation complete",
            route_id=route_id,
            num_evaluations=len(raw_evaluations),
            dimensions=len(dimension_scores),
        )

        return observation

    def observe_images(
        self, image_paths: List[str], image_id_prefix: str = "img"
    ) -> Dict[str, Any]:
        """Observe a list of images directly (no route context).

        Useful for ad-hoc image evaluation without route information.

        Args:
            image_paths: List of image file paths to evaluate.
            image_id_prefix: Prefix for generated image IDs.

        Returns:
            Dict[str, Any]: Observation dict similar to observe_route().
        """
        if not image_paths:
            self.logger.warning("No images provided for observation")
            return self._empty_observation("direct_observation")

        self.logger.debug("Observing images", num_images=len(image_paths))

        # Evaluate images
        raw_evaluations = self.evaluator.evaluate_images(image_paths)

        # Group and calculate statistics
        dimension_scores = self._group_scores_by_dimension(raw_evaluations)
        dimension_stats = self._calculate_dimension_statistics(dimension_scores)

        observation = {
            "route_info": {
                "route_id": f"{image_id_prefix}_observation",
                "num_images": len(image_paths),
            },
            "dimension_scores": dimension_scores,
            "dimension_stats": dimension_stats,
            "raw_evaluations": raw_evaluations,
            "framework_id": self.framework_id,
            "observation_timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            "Image observation complete",
            num_images=len(image_paths),
            num_evaluations=len(raw_evaluations),
        )

        return observation

    def get_framework_info(self) -> Dict[str, Any]:
        """Get information about the current evaluation framework.

        Returns:
            Dict containing framework metadata:
                - framework_id
                - name
                - dimensions (list of dimension definitions)
                - num_dimensions
        """
        return {
            "framework_id": self.framework_id,
            "name": self.framework.get("name", "Unknown"),
            "dimensions": self.framework.get("dimensions", []),
            "num_dimensions": len(self.framework.get("dimensions", [])),
        }

    def _group_scores_by_dimension(
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Group evaluation scores by dimension.

        Args:
            evaluations: List of evaluation results from VLM.

        Returns:
            Dict mapping dimension_id to list of scores.
        """
        scores_by_dimension = {}

        for eval_result in evaluations:
            dimension_id = eval_result.get("dimension_id")
            score = eval_result.get("score")

            if dimension_id is not None and score is not None:
                if dimension_id not in scores_by_dimension:
                    scores_by_dimension[dimension_id] = []
                scores_by_dimension[dimension_id].append(score)

        return scores_by_dimension

    def _calculate_dimension_statistics(
        self, dimension_scores: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical summary for each dimension.

        Args:
            dimension_scores: Scores grouped by dimension.

        Returns:
            Dict mapping dimension_id to stats dict with:
                min, max, avg, std, count
        """
        stats = {}

        for dimension_id, scores in dimension_scores.items():
            if not scores:
                stats[dimension_id] = {
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0,
                    "std": 0.0,
                    "count": 0,
                }
                continue

            stats[dimension_id] = {
                "min": min(scores),
                "max": max(scores),
                "avg": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "count": len(scores),
            }

        return stats

    def _empty_observation(self, route_id: str) -> Dict[str, Any]:
        """Create an empty observation when no images are available.

        Args:
            route_id: Route identifier.

        Returns:
            Empty observation dict with proper structure.
        """
        return {
            "route_info": {
                "route_id": route_id,
                "num_images": 0,
            },
            "dimension_scores": {},
            "dimension_stats": {},
            "raw_evaluations": [],
            "framework_id": self.framework_id,
            "observation_timestamp": datetime.now().isoformat(),
        }
