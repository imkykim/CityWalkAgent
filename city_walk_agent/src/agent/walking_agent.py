"""WalkingAgent - orchestrates all capabilities for route analysis.

This is the main concrete agent implementation that uses ObservationCapability,
ThinkingCapability, and AgentMemory to analyze walking routes with personality-driven
decision making.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.capabilities import (
    ActionCapability,
    AnalysisCapability,
    AgentMemory,
    ObservationCapability,
    ThinkingCapability,
)
from src.agent.config import AgentPersonality, get_preset
from src.pipeline import WalkingAgentPipeline
from src.config import settings
from src.utils.data_models import Route, Waypoint
from src.utils.logging import get_logger


class WalkingAgent(BaseAgent):
    """Walking route analysis agent with personality-driven decision making.

    This agent orchestrates the full cognitive pipeline:
    1. Observe: Use VLM to evaluate route images
    2. Analyze: Sequential pattern analysis and barrier detection
    3. Think: Apply personality weights and make decisions
    4. Act: Generate formatted recommendations
    5. Remember: Store experiences for learning

    Example:
        ```python
        # Create from preset
        agent = WalkingAgent.from_preset("safety", "sagai_2025")

        # Analyze a route
        result = agent.run(
            start=(40.7589, -73.9851),
            end=(40.7614, -73.9776)
        )

        # Check recommendation
        print(result["decision"]["recommendation"])  # "accept" or "reject"
        print(result["result"]["message"])  # Human-readable message
        ```
    """

    def __init__(
        self,
        agent_id: str,
        personality: AgentPersonality,
        framework_id: Optional[str] = None,
        enable_memory: bool = True,
    ):
        """Initialize WalkingAgent with personality.

        Args:
            agent_id: Unique identifier for this agent.
            personality: AgentPersonality with weights and thresholds.
            framework_id: Evaluation framework (defaults to personality's framework).
            enable_memory: Whether to enable persistent memory.
        """
        # Create metadata from personality
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=personality.name,
            description=personality.description,
            primary_goal=f"Analyze routes with {personality.name} preferences",
            constraints={
                "explanation_style": personality.explanation_style,
                "enable_memory": enable_memory,
            },
        )

        # Initialize base agent
        super().__init__(metadata=metadata)

        # Store personality configuration
        self.personality = personality
        self.framework_id = framework_id or "sagai_2025"
        self.enable_memory = enable_memory

        # Initialize state with personality weights
        self.state.preferences = personality.dimension_weights.copy()

        # Lazy-loaded capabilities
        self._observer: Optional[ObservationCapability] = None
        self._analyzer: Optional[AnalysisCapability] = None
        self._thinker: Optional[ThinkingCapability] = None
        self._actor: Optional[ActionCapability] = None
        self._pipeline: Optional[WalkingAgentPipeline] = None

        self.logger.info(
            "WalkingAgent created",
            agent_id=agent_id,
            personality=personality.name,
            framework_id=self.framework_id,
        )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        framework_id: str,
        agent_id: Optional[str] = None,
        use_semantic: bool = True,
    ) -> "WalkingAgent":
        """Create agent from a personality preset.

        Args:
            preset_name: Personality identifier (e.g., "safety", "scenic", "balanced").
            framework_id: Evaluation framework to use.
            agent_id: Optional agent ID (defaults to f"{preset_name}_agent").
            use_semantic: Whether to use semantic mapping vs framework-specific config.

        Returns:
            Configured WalkingAgent instance.

        Example:
            ```python
            # Using semantic mapping (adapts to any framework)
            agent = WalkingAgent.from_preset("safety", "sagai_2025", use_semantic=True)

            # Using framework-specific config
            agent = WalkingAgent.from_preset("safety", "sagai_2025", use_semantic=False)
            ```
        """
        # Load personality
        personality = get_preset(preset_name, framework_id, use_semantic)

        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"{preset_name}_agent_{framework_id}"

        # Create agent
        return cls(
            agent_id=agent_id,
            personality=personality,
            framework_id=framework_id,
        )

    def run(
        self,
        start: Optional[Tuple[float, float]] = None,
        end: Optional[Tuple[float, float]] = None,
        *,
        route_folder: Optional[Union[str, Path]] = None,
        metadata_filename: str = "collection_metadata.json",
        **kwargs
    ) -> Dict[str, Any]:
        """Run the agent either by generating a route or from a pre-collected folder."""
        if route_folder is not None:
            if (start is not None) or (end is not None):
                raise ValueError("Provide either start/end or route_folder, but not both.")

            return self.run_from_folder(
                route_folder=route_folder,
                metadata_filename=metadata_filename,
                **kwargs,
            )

        if start is None or end is None:
            raise ValueError("Both start and end must be provided when route_folder is not supplied.")

        return super().run(start=start, end=end, **kwargs)

    def run_from_folder(
        self,
        route_folder: Union[str, Path],
        *,
        metadata_filename: str = "collection_metadata.json",
        interval: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the agent workflow using a pre-existing folder of route imagery."""
        folder_path = Path(route_folder).expanduser().resolve()
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Route folder does not exist: {folder_path}")

        metadata_path = folder_path / metadata_filename
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Route folder is missing metadata file '{metadata_filename}' in {folder_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        waypoint_entries: List[Dict[str, Any]] = metadata.get("results") or []
        if not waypoint_entries:
            raise ValueError(f"No waypoint entries found in {metadata_path}")

        waypoint_entries = sorted(
            waypoint_entries,
            key=lambda entry: entry.get("waypoint_id", 0),
        )

        resolved_waypoints: List[Waypoint] = []
        resolved_image_paths: List[str] = []

        for entry in waypoint_entries:
            sequence_id = entry.get("waypoint_id", len(resolved_waypoints))

            image_path = entry.get("image_path")
            image_candidate: Optional[Path]
            if image_path:
                image_candidate = Path(image_path)
                if not image_candidate.is_absolute():
                    image_candidate = folder_path / image_candidate
            else:
                waypoint_dir = folder_path / f"waypoint_{sequence_id:03d}"
                if waypoint_dir.is_dir():
                    image_candidate = next(waypoint_dir.glob("*.jpg"), None)
                else:
                    image_candidate = None

            if image_candidate is None or not image_candidate.exists():
                raise FileNotFoundError(
                    f"Could not locate image for waypoint {sequence_id} in {folder_path}"
                )

            image_str = str(image_candidate.resolve())
            resolved_image_paths.append(image_str)

            waypoint = Waypoint(
                lat=entry["lat"],
                lon=entry["lon"],
                sequence_id=sequence_id,
                heading=entry.get("heading"),
                timestamp=None,
                image_path=image_str,
            )
            resolved_waypoints.append(waypoint)

        if not resolved_waypoints:
            raise ValueError(f"Failed to build waypoints from metadata {metadata_path}")

        route_id = metadata.get("route_id") or folder_path.name
        derived_interval = (
            interval
            or metadata.get("interval")
            or metadata.get("interval_meters")
            or kwargs.get("interval")
            or settings.default_sampling_interval
        )

        route = Route(
            route_id=route_id,
            start_lat=resolved_waypoints[0].lat,
            start_lon=resolved_waypoints[0].lon,
            end_lat=resolved_waypoints[-1].lat,
            end_lon=resolved_waypoints[-1].lon,
            waypoints=resolved_waypoints,
            route_name=metadata.get("route_name"),
            interval_meters=int(derived_interval),
        )

        override_data = {
            "route_id": route.route_id,
            "route": route,
            "image_paths": resolved_image_paths,
            "waypoints": resolved_waypoints,
            "distance": route.total_distance,
            "interval": route.interval_meters,
            "from_cache": True,
            "source": "route_folder",
            "folder_path": str(folder_path),
            "metadata_file": str(metadata_path),
        }

        kwargs.setdefault("interval", route.interval_meters)

        return super().run(
            start=(route.start_lat, route.start_lon),
            end=(route.end_lat, route.end_lon),
            _route_data_override=override_data,
            route_folder=str(folder_path),
            **kwargs,
        )

    @property
    def observer(self) -> ObservationCapability:
        """Lazy-load observation capability."""
        if self._observer is None:
            self._observer = ObservationCapability(framework_id=self.framework_id)
            self.logger.debug("ObservationCapability initialized")
        return self._observer

    @property
    def analyzer(self) -> AnalysisCapability:
        """Lazy-load analysis capability."""
        if self._analyzer is None:
            self._analyzer = AnalysisCapability()
            self.logger.debug("AnalysisCapability initialized")
        return self._analyzer

    @property
    def thinker(self) -> ThinkingCapability:
        """Lazy-load thinking capability."""
        if self._thinker is None:
            self._thinker = ThinkingCapability()
            self.logger.debug("ThinkingCapability initialized")
        return self._thinker

    @property
    def actor(self) -> ActionCapability:
        """Lazy-load action capability."""
        if self._actor is None:
            self._actor = ActionCapability()
            self.logger.debug("ActionCapability initialized")
        return self._actor

    @property
    def pipeline(self) -> WalkingAgentPipeline:
        """Lazy-load pipeline for route data."""
        if self._pipeline is None:
            self._pipeline = WalkingAgentPipeline(framework_id=self.framework_id)
            self.logger.debug("Pipeline initialized")
        return self._pipeline

    def perceive(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """Observe route using VLM evaluation.

        This implements post-hoc position tracking: as we process the route,
        we update state to reflect which waypoint we've analyzed.

        Args:
            route_data: Route information with image_paths, route_id, etc.

        Returns:
            Observation dict with dimension scores and statistics.
        """
        self.logger.debug("Perceiving route", route_id=route_data.get("route_id"))

        # Track position as we observe waypoints (post-hoc tracking)
        waypoints = route_data.get("waypoints", [])
        if waypoints:
            # Update position tracking as we process waypoints
            for idx, waypoint in enumerate(waypoints):
                # Update state to reflect current observation position
                self.state.current_location = (waypoint.lat, waypoint.lon)
                self.state.waypoint_index = idx

                self.logger.debug(
                    "Processing waypoint",
                    waypoint_index=idx,
                    total=len(waypoints),
                    location=(waypoint.lat, waypoint.lon)
                )

            # Final position is the last waypoint
            self.logger.debug(
                "Position tracking complete",
                final_waypoint=self.state.waypoint_index,
                total_waypoints=len(waypoints)
            )

        # Use ObservationCapability to evaluate route
        observation = self.observer.observe_route(route_data)

        self.logger.debug(
            "Perception complete",
            route_id=route_data.get("route_id"),
            num_dimensions=len(observation.get("dimension_stats", {})),
        )

        return observation

    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Apply personality-driven reasoning to make a decision.

        Args:
            perception: Observation dict from perceive().

        Returns:
            Decision dict with recommendation, confidence, explanation.
        """
        route_id = perception.get("route_info", {}).get("route_id", "unknown")
        self.logger.debug("Reasoning about route", route_id=route_id)

        # Get route info for analysis
        route_info = perception.get("route_info", {})

        # Use AnalysisCapability for sequential pattern analysis
        sequential_analysis = self.analyzer.analyze_patterns(
            observation=perception,
            route_info=route_info,
        )

        # Use ThinkingCapability to make decision
        decision = self.thinker.think(
            observation=perception,
            sequential_analysis=sequential_analysis,
            personality_weights=self.personality.dimension_weights,
            decision_thresholds=self.personality.decision_thresholds,
            explanation_style=self.personality.explanation_style,
        )

        # Add sequential analysis to decision for transparency
        decision["sequential_analysis"] = sequential_analysis

        self.logger.debug(
            "Reasoning complete",
            route_id=route_id,
            recommendation=decision["recommendation"],
            confidence=round(decision["confidence"], 2),
        )

        return decision

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action using ActionCapability.

        Args:
            decision: Decision dict from reason().

        Returns:
            Action result with formatted message.
        """
        return self.actor.execute(decision)

    def _get_route_data(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        **kwargs
    ) -> Dict[str, Any]:
        """Get route data using the pipeline.

        Implements cache checking: if route has been evaluated before,
        load cached results instead of regenerating.

        Args:
            start: Starting coordinate (lat, lon).
            end: Destination coordinate (lat, lon).
            **kwargs: Additional options (route_id, interval, etc.).

        Returns:
            Route data dict with route_id, image_paths, waypoints, etc.
        """
        override = kwargs.pop("_route_data_override", None)
        kwargs.pop("route_folder", None)
        if override is not None:
            self.logger.info(
                "Using provided route data override",
                route_id=override.get("route_id"),
                source=override.get("source", "unknown"),
            )
            return override

        interval = kwargs.get("interval", 50)

        # Generate route_id from coordinates if not provided
        route_id = kwargs.get("route_id")
        if route_id is None:
            # Generate deterministic route_id from coordinates
            route_id = f"route_{start[0]:.6f}_{start[1]:.6f}_{end[0]:.6f}_{end[1]:.6f}_{interval}"

        # Check cache first
        cached_data = self.pipeline.load_cached_route(route_id)
        if cached_data is not None:
            self.logger.info("Using cached route data", route_id=route_id)
            # Mark as from cache
            cached_data["from_cache"] = True
            return cached_data

        # Cache miss - generate new route
        self.logger.debug("Cache miss - generating new route", start=start, end=end)
        route = self.pipeline.route_generator.create_simple_route(
            start_lat=start[0],
            start_lon=start[1],
            end_lat=end[0],
            end_lon=end[1],
            interval_meters=interval,
        )
        route_id = route.route_id

        # Collect images if needed
        image_dir = self.pipeline.route_generator.get_route_dir(route_id)
        image_paths = list(image_dir.glob("*.jpg"))

        if not image_paths:
            self.logger.debug("Collecting images for route", route_id=route_id)
            # Use ImageCollector to get images
            from src.data_collection.image_collector import ImageCollector

            collector = ImageCollector()
            image_paths = collector.collect_route_images(route)

        # Build route data
        route_data = {
            "route_id": route_id,
            "route": route,
            "image_paths": [str(p) for p in image_paths],
            "waypoints": route.waypoints,
            "distance": route.total_distance,
            "interval": interval,
            "from_cache": False,  # Fresh data, not cached
        }

        self.logger.debug(
            "Route data prepared",
            route_id=route_id,
            num_images=len(image_paths),
        )

        return route_data
