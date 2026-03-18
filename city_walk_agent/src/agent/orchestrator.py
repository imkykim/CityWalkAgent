"""CityWalkAgent - orchestrates all capabilities for route analysis.

This is the main concrete agent implementation that uses ContinuousAnalyzer,
ShortTermMemory, PersonaReasoner, and LongTermMemory to analyze walking routes
with personality-driven decision making.

Analysis mode: run_with_memory() / run_with_memory_from_folder()
   - Real-time sequential analysis during route traversal
   - Uses ContinuousAnalyzer → ShortTermMemory → PersonaReasoner → LongTermMemory
   - Includes pHash detection, triggered reasoning, and moment curation
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from geopy.distance import geodesic

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.memory import (
    LongTermMemory,
    MemoryManager,
    ShortTermMemory,
)
from src.agent.system2 import PersonaReasoner, TriggerReason
from src.agent.system1 import CognitiveController
from src.agent.config import AgentPersonality, get_preset
from src.core import DEFAULT_FRAMEWORK_ID, settings
from src.utils.data_models import Route, Waypoint
from src.utils.logging import get_logger


def _calc_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2 (degrees, 0=N clockwise)."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _bearing_to_cardinal(b: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(b / 45) % 8]



def _closest_link(links: List[Dict], target_heading: float) -> Dict:
    return min(
        links,
        key=lambda l: abs(
            ((l.get("heading") or l.get("yawDeg") or 0) - target_heading + 180) % 360 - 180
        ),
    )


class CityWalkAgent(BaseAgent):
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
        agent = CityWalkAgent.from_preset("safety", "sagai_2025")

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
        """Initialize CityWalkAgent with personality.

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
        self.framework_id = framework_id or DEFAULT_FRAMEWORK_ID
        self.enable_memory = enable_memory
        self._cognitive_controller: Optional[CognitiveController] = None

        # System 2
        self._interpreter = None
        self._decider = None
        self._planner = None
        self._reporter = None

        # Configurable thresholds (can be overridden via set_thresholds())
        self._phash_threshold: int = 40  # Default for System 2 trigger

        # Initialize state (preferences no longer used in dual evaluation system)
        self.state.preferences = {}

        # Lazy-loaded memory system components
        self._continuous_analyzer = None
        self._short_term_memory = None
        self._persona_reasoner = None
        self._memory_manager = None

        self.logger.info(f"Agent: {personality.name} | framework={self.framework_id}")

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        framework_id: str = DEFAULT_FRAMEWORK_ID,
        agent_id: Optional[str] = None,
    ) -> "CityWalkAgent":
        """Create agent from a personality preset.

        Args:
            preset_name: Personality identifier (e.g., "homebuyer", "runner", "photographer").
            framework_id: Evaluation framework to use.
            agent_id: Optional agent ID (defaults to f"{preset_name}_agent").

        Returns:
            Configured CityWalkAgent instance.

        Example:
            ```python
            agent = CityWalkAgent.from_preset("homebuyer", "streetagent_5d")
            ```
        """
        # Load personality
        personality = get_preset(preset_name, framework_id)

        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"{preset_name}_agent_{framework_id}"

        # Create agent
        return cls(
            agent_id=agent_id,
            personality=personality,
            framework_id=framework_id,
        )

    def set_thresholds(
        self,
        phash_threshold: Optional[int] = None,
    ) -> None:
        """Configure visual change detection threshold.

        IMPORTANT: Must be called BEFORE accessing continuous_analyzer or cognitive
        properties, as these are lazy-loaded and will use the configured threshold.

        Args:
            phash_threshold: Threshold for CognitiveController (System 2 trigger).
                Lower values = more sensitive to visual changes.
                This threshold controls both visual change detection AND multi-image evaluation.
                Default: 30 (if not set)

        Example:
            ```python
            agent = CityWalkAgent.from_preset("balanced", "streetagent_5d")

            # Configure threshold BEFORE using the agent
            agent.set_thresholds(phash_threshold=30)

            # Now run analysis - will use configured threshold
            result = agent.run_with_memory(start=(lat1, lon1), end=(lat2, lon2))
            ```
        """
        if phash_threshold is not None:
            self._phash_threshold = phash_threshold
            self.logger.debug(f"pHash threshold set to {phash_threshold}")

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

            # Resolve image path - try multiple strategies
            image_path = entry.get("image_path")
            image_candidate: Optional[Path] = None

            # Strategy 1: If metadata provides a path, try it
            if image_path:
                candidate = Path(image_path)

                # If it's absolute, check if it exists
                if candidate.is_absolute():
                    if candidate.exists():
                        image_candidate = candidate
                    else:
                        # Absolute path doesn't exist, try using just the filename
                        # in the current folder (images may have been moved)
                        image_candidate = folder_path / candidate.name
                else:
                    # Relative path - resolve relative to folder_path
                    image_candidate = folder_path / candidate

            # Strategy 2: If still not found, look for waypoint pattern in folder
            if image_candidate is None or not image_candidate.exists():
                # Try pattern: waypoint_NNN_*.jpg
                pattern_candidates = list(
                    folder_path.glob(f"waypoint_{sequence_id:03d}_*.jpg")
                )
                if pattern_candidates:
                    image_candidate = pattern_candidates[0]

            # Strategy 3: Look in waypoint subdirectory
            if image_candidate is None or not image_candidate.exists():
                waypoint_dir = folder_path / f"waypoint_{sequence_id:03d}"
                if waypoint_dir.is_dir():
                    jpg_files = list(waypoint_dir.glob("*.jpg"))
                    if jpg_files:
                        image_candidate = jpg_files[0]

            # Final check
            if image_candidate is None or not image_candidate.exists():
                raise FileNotFoundError(
                    f"Could not locate image for waypoint {sequence_id} in {folder_path}. "
                    f"Expected pattern: waypoint_{sequence_id:03d}_*.jpg"
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

    # ========================================================================
    # Memory System Properties
    # ========================================================================

    @property
    def continuous_analyzer(self):
        """Lazy-load continuous analyzer for waypoint-level analysis.

        Returns:
            ContinuousAnalyzer instance configured with agent's framework and persona.
        """
        if self._continuous_analyzer is None:
            from src.agent.system1 import ContinuousAnalyzer

            # Get enhanced persona configuration if available
            enhanced_persona = self._get_enhanced_persona()

            if not enhanced_persona:
                self.logger.warning(
                    "No persona configured — using objective evaluation only"
                )

            self._continuous_analyzer = ContinuousAnalyzer(
                framework_id=self.framework_id,
                context_window=3,
                enable_multi_image=True,
                persona=enhanced_persona,
                memory_manager=self.memory_manager,  # Pass memory_manager for STM integration
            )
        return self._continuous_analyzer

    @property
    def short_term_memory(self) -> ShortTermMemory:
        """Lazy-load short-term memory for temporal context.

        Returns:
            ShortTermMemory instance with sliding window.
        """
        if self._short_term_memory is None:
            self._short_term_memory = ShortTermMemory(window_size=5)
            self.logger.debug("ShortTermMemory initialized")
        return self._short_term_memory

    @property
    def persona_reasoner(self) -> PersonaReasoner:
        """Lazy-load persona reasoner for waypoint-level System 2 reasoning.

        Returns:
            PersonaReasoner instance configured for triggered reasoning.
        """
        if self._persona_reasoner is None:
            self._persona_reasoner = PersonaReasoner(
                framework_id=self.framework_id,
                distance_trigger_meters=800.0,
                score_delta_threshold=2.0,
            )
            self.logger.debug(
                "PersonaReasoner initialized", framework_id=self.framework_id
            )
        return self._persona_reasoner

    @property
    def memory_manager(self) -> MemoryManager:
        """Lazy-load memory manager for unified memory coordination.

        Returns:
            MemoryManager instance configured with agent attributes
        """
        if self._memory_manager is None:
            self._memory_manager = MemoryManager(agent_id=self.metadata.agent_id)

            # Set agent attributes
            self._memory_manager.set_agent_attributes(
                personality=self.personality,
                profile={
                    "agent_id": self.metadata.agent_id,
                    "name": self.metadata.name,
                    "goal": self.metadata.primary_goal,
                },
                status={"mode": "active"},
            )

            self.logger.debug("MemoryManager initialized")
        return self._memory_manager

    @property
    def cognitive(self) -> CognitiveController:
        """Lazy-load cognitive controller for trigger logic."""
        if self._cognitive_controller is None:
            self._cognitive_controller = CognitiveController(
                phash_threshold=self._phash_threshold
            )
            self.logger.debug(
                "CognitiveController initialized", phash_threshold=self._phash_threshold
            )
        return self._cognitive_controller

    # ========================================================================
    # System 2 Properties
    # ========================================================================

    @property
    def interpreter(self):
        """Lazy-load System 2 Interpreter."""
        if self._interpreter is None:
            from src.agent.system2 import Interpreter

            self._interpreter = Interpreter(framework_id=self.framework_id)
            self.logger.debug("System2 Interpreter initialized")
        return self._interpreter

    @property
    def decider(self):
        """Lazy-load System 2 Decider."""
        if self._decider is None:
            from src.agent.system2 import Decider

            self._decider = Decider(framework_id=self.framework_id)
            self.logger.debug("System2 Decider initialized")
        return self._decider

    @property
    def planner(self):
        """Lazy-load System 2 Planner."""
        if self._planner is None:
            from src.agent.system2 import Planner

            self._planner = Planner(framework_id=self.framework_id)
            self.logger.debug("System2 Planner initialized")
        return self._planner

    @property
    def reporter(self):
        """Lazy-load System 2 Reporter."""
        if self._reporter is None:
            from src.agent.system2 import Reporter

            self._reporter = Reporter(framework_id=self.framework_id)
            self.logger.debug("System2 Reporter initialized")
        return self._reporter

    def _get_prev_avg_score(self, stm_context: Optional[Dict]) -> float:
        """STM context에서 직전 waypoint 평균 점수 반환. 없으면 0.0."""
        if not stm_context:
            return 0.0
        recent = stm_context.get("recent_scores", [])
        if not recent:
            return 0.0
        last = recent[-1]
        if not last:
            return 0.0
        return sum(last.values()) / len(last)

    def _get_enhanced_persona(self):
        """Get EnhancedPersonalityConfig from current personality.

        Returns:
            EnhancedPersonalityConfig if available, None otherwise.
        """
        if self.personality is None:
            self.logger.warning("No personality set - cannot get enhanced persona")
            return None

        try:
            from src.agent.config import (
                ENHANCED_PERSONALITIES,
                get_enhanced_personality,
            )

            # Convert personality name to personality_id format (lowercase, underscores)
            raw_personality_id = (
                getattr(self.personality, "personality_id", None)
                or self.personality.name
            )
            personality_id = raw_personality_id.lower().replace(" ", "_")

            self.logger.debug(
                f"Resolving enhanced persona - raw_id: '{raw_personality_id}', "
                f"converted: '{personality_id}'"
            )

            # If the name does not directly match, try to map common display names
            if personality_id not in ENHANCED_PERSONALITIES:
                name_map = {
                    "street_photographer": "photographer",
                    "prospective_homebuyer": "homebuyer",
                    "urban_runner": "runner",
                    "parent_with_young_children": "parent_with_kids",
                }
                original_id = personality_id
                personality_id = name_map.get(personality_id, personality_id)
                if original_id != personality_id:
                    self.logger.debug(f"Mapped '{original_id}' -> '{personality_id}'")

            # Get enhanced personality configuration
            self.logger.debug(f"Loading enhanced personality: '{personality_id}'")
            enhanced = get_enhanced_personality(personality_id)

            self.logger.debug(f"Enhanced persona loaded: '{enhanced.name}'")
            return enhanced

        except (ImportError, ValueError, AttributeError) as e:
            # Fallback: no enhanced persona available
            self.logger.warning(
                f"Could not load enhanced personality: {e}. "
                "Will use objective evaluation only."
            )
            return None

    # ========================================================================
    # Traditional Methods (Backward Compatible)
    # ========================================================================

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
                    location=(waypoint.lat, waypoint.lon),
                )

            # Final position is the last waypoint
            self.logger.debug(
                "Position tracking complete",
                final_waypoint=self.state.waypoint_index,
                total_waypoints=len(waypoints),
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
        self, start: Tuple[float, float], end: Tuple[float, float], **kwargs
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
        route = self.pipeline.route_generator.create_google_maps_route(
            origin=start,
            destination=end,
            interval_meters=interval,
            mode="walking",
            route_name=None,
            route_id=route_id,
        )
        route_id = route.route_id

        # Collect images if needed
        image_dir = self.pipeline.route_generator.get_route_dir(route_id)
        image_paths = list(image_dir.glob("*.jpg"))

        if not image_paths:
            self.logger.debug("Collecting images for route", route_id=route_id)
            # Use ImageCollector to get images
            from src.data_collection.image_collector import ImageCollector

            collector = ImageCollector(api_key=settings.google_maps_api_key)
            # Use directional collection with advanced features
            results = collector.collect_google_street_view_images_static(
                route,
                use_route_direction=True,
                all_around=False,
                fov=90,
                pitch=0,
                lookahead_distance=2,
                detect_corners=True,
                corner_threshold=30.0,
            )
            image_paths = [
                Path(result["image_path"])
                for result in results
                if result.get("image_path")
            ]

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

    # ========================================================================
    # New Memory System Method
    # ========================================================================

    def _validate_and_reconstruct_waypoints(
        self,
        image_paths: List[Path],
        waypoints: List[Any],
        metadata: List[Dict[str, Any]],
    ) -> Tuple[List[Path], List[Any], List[Dict[str, Any]]]:
        """Validate waypoint images and reconstruct sequence for missing files.

        This method checks if all waypoint image files exist. If any images are
        missing (e.g., manually deleted for data curation), it reconstructs the
        waypoint sequence using only valid images while preserving all GPS and
        metadata information.

        Args:
            image_paths: List of Path objects pointing to waypoint images.
            waypoints: List of Waypoint objects with GPS coordinates.
            metadata: List of metadata dicts for each waypoint.

        Returns:
            Tuple of (valid_image_paths, valid_waypoints, valid_metadata)
            containing only waypoints with existing image files.

        Raises:
            ValueError: If all waypoint images are missing.

        Example:
            Input: 100 waypoints, 2 images missing
            Output: 98 waypoints with renumbered sequence
            Logs: Warnings for missing files, reconstruction summary
        """
        total_count = len(image_paths)
        self.logger.debug("Validating waypoint images", total_waypoints=total_count)

        valid_image_paths: List[Path] = []
        valid_waypoints: List[Any] = []
        valid_metadata: List[Dict[str, Any]] = []
        missing_count = 0

        for idx, (img_path, waypoint, meta) in enumerate(
            zip(image_paths, waypoints, metadata)
        ):
            if not img_path.exists():
                self.logger.warning(
                    f"Missing image file, skipping waypoint {idx}", path=str(img_path)
                )
                missing_count += 1
            else:
                valid_image_paths.append(img_path)
                valid_waypoints.append(waypoint)
                valid_metadata.append(meta)

        # Check if all images are missing
        if not valid_image_paths:
            raise ValueError(
                f"No valid waypoint images found. All {total_count} images are missing."
            )

        valid_count = len(valid_image_paths)

        # Log summary if any images were missing
        if missing_count > 0:
            retention_rate = (valid_count / total_count) * 100
            self.logger.warning(
                "Waypoint sequence reconstructed",
                original_count=total_count,
                valid_count=valid_count,
                missing_count=missing_count,
                retention_rate=f"{retention_rate:.1f}%",
            )
        else:
            self.logger.debug("All waypoint images validated", total_count=total_count)

        return valid_image_paths, valid_waypoints, valid_metadata

    def run_with_memory(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        interval: int = 50,
        output_dir: Optional[Path] = None,
        skip_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Run agent with full memory system integration.

        This method provides waypoint-level analysis with the complete memory pipeline:
        1. ContinuousAnalyzer - per-waypoint VLM evaluation with pHash detection
        2. ShortTermMemory - sliding window context for recent waypoints
        3. PersonaReasoner - triggered System 2 reasoning at key moments (optional)
        4. LongTermMemory - curated key moments and pattern extraction

        This is separate from run() to maintain backward compatibility. Use this
        method when you want deep waypoint-level analysis with memory formation.

        Args:
            start: Starting GPS coordinates (lat, lon).
            end: Destination GPS coordinates (lat, lon).
            interval: Sampling interval in meters (default: 50).
            output_dir: Optional directory to save memory artifacts.
            skip_reasoning: If True, skip Phase 3 (PersonaReasoner) and use only Phase 2 scores.

        Returns:
            Dictionary containing:
            - analysis_results: List of waypoint result dicts (System 1/2 scores)
            - reasoning_results: List of ReasoningResult objects
            - long_term_memory: LongTermMemory instance
            - route_summary: RouteSummary object
            - statistics: Dict with pipeline statistics
            - route_data: Basic route information

        Example:
            ```python
            agent = CityWalkAgent.from_preset("balanced", "sagai_2025")

            result = agent.run_with_memory(
                start=(40.7589, -73.9851),
                end=(40.7614, -73.9776),
                interval=50,
                output_dir=Path("outputs/memory_analysis")
            )

            # Access results
            print(f"Analyzed {len(result['analysis_results'])} waypoints")
            print(f"Key moments: {len(result['long_term_memory'].get_key_images())}")
            print(f"Recommendation: {result['route_summary'].overall_recommendation}")
            ```
        """
        self.logger.info(
            "Starting memory pipeline run",
            start=start,
            end=end,
            interval=interval,
        )

        # ====================================================================
        # Phase 1: Get Route Data
        # ====================================================================
        self.logger.debug("Phase 1: Collecting route data")

        route_data = self._get_route_data(start, end, interval=interval)
        route_id = route_data["route_id"]
        route = route_data["route"]
        image_paths = [Path(p) for p in route_data["image_paths"]]
        waypoints = route_data["waypoints"]

        # Ensure images and metadata align
        if len(image_paths) != len(waypoints):
            min_len = min(len(image_paths), len(waypoints))
            self.logger.warning(
                "Image/metadata count mismatch; trimming to align",
                images=len(image_paths),
                waypoints=len(waypoints),
                used=min_len,
            )
            image_paths = image_paths[:min_len]
            waypoints = waypoints[:min_len]

        # Calculate route length
        route_length_km = route.total_distance / 1000.0

        self.logger.info(
            "Route data collected",
            route_id=route_id,
            num_waypoints=len(waypoints),
            length_km=route_length_km,
        )

        # Prepare metadata
        metadata = [
            {
                "filename": img_path.name,
                "lat": wp.lat,
                "lon": wp.lon,
                "heading": wp.heading or 0.0,
                "timestamp": wp.timestamp or "",
            }
            for img_path, wp in zip(image_paths, waypoints)
        ]

        # Validate images and reconstruct waypoint sequence
        image_paths, waypoints, metadata = self._validate_and_reconstruct_waypoints(
            image_paths, waypoints, metadata
        )

        # Update route data with validated waypoints
        route_data["image_paths"] = [str(p) for p in image_paths]
        route_data["waypoints"] = waypoints

        # Recalculate route length if waypoints were removed
        if len(waypoints) < len(route.waypoints):
            new_distance = 0.0
            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i + 1]
                new_distance += geodesic((wp1.lat, wp1.lon), (wp2.lat, wp2.lon)).meters

            route_length_km = new_distance / 1000.0

            self.logger.info(
                "Route metrics updated after validation",
                valid_waypoints=len(waypoints),
                length_km=route_length_km,
            )

        # ====================================================================
        # Phase 2+3: Interleaved analysis and memory per waypoint
        # STM is populated immediately after each analysis so the next
        # waypoint's VLM call receives prior context.
        # ====================================================================
        if skip_reasoning:
            self.logger.debug(
                "Phase 2+3: Continuous analysis (reasoning SKIPPED — system1-only mode)"
            )
        else:
            self.logger.debug(
                "Phase 2+3: Interleaved continuous analysis + memory + reasoning"
            )

        # Reset state for new route analysis
        self.cognitive.reset()
        self.continuous_analyzer.reset()
        self.logger.debug(
            "CognitiveController and ContinuousAnalyzer state reset for new route"
        )

        analysis_results = []
        reasoning_results = []
        memory_manager = self.memory_manager
        waypoint_results: List[Dict[str, Any]] = []

        for i, (img_path, meta) in enumerate(zip(image_paths, metadata)):
            # Step A: CognitiveController detects visual change
            is_first_waypoint = i == 0
            visual_change_result = self.cognitive.detect_visual_change(
                image_path=img_path, force=is_first_waypoint
            )

            self.logger.debug(
                f"Visual change detection for waypoint {i}",
                changed=visual_change_result.changed,
                distance=visual_change_result.phash_distance,
                reason=visual_change_result.reason,
            )

            # Step B: ContinuousAnalyzer analyzes waypoint (STM already populated by prior iterations)
            analysis = self.continuous_analyzer.analyze_waypoint(
                waypoint_id=i,
                image_path=img_path,
                metadata=meta,
                visual_change_detected=visual_change_result.changed,
                phash_distance=visual_change_result.phash_distance,
            )

            self.continuous_analyzer.analysis_history.append(analysis)
            analysis_results.append(analysis)

            # Step C: PersonaReasoner.should_trigger()로 종합 판단
            stm_context_for_trigger = self.memory_manager.stm.get_context()

            # score_delta 계산
            current_scores = analysis.persona_scores or analysis.scores
            current_avg = (
                sum(current_scores.values()) / len(current_scores)
                if current_scores
                else 5.0
            )
            prev_avg = self._get_prev_avg_score(stm_context_for_trigger)
            score_delta = abs(current_avg - prev_avg)

            # distance_from_last: lat/lon으로 직접 계산
            if i == 0:
                distance_from_last = 0.0
            else:
                prev_meta = metadata[i - 1]
                try:
                    distance_from_last = geodesic(
                        (prev_meta.get("lat", 0), prev_meta.get("lon", 0)),
                        (meta.get("lat", 0), meta.get("lon", 0)),
                    ).meters
                except Exception:
                    distance_from_last = 0.0

            # PersonaReasoner.should_trigger()로 종합 판단
            trigger_reason = self.persona_reasoner.should_trigger(
                waypoint_id=i,
                visual_change=visual_change_result.changed,
                score_delta=score_delta,
                distance_from_last=distance_from_last,
                is_exceptional=(i == 0),
            )
            should_reason = trigger_reason is not None

            self.logger.debug(
                f"System 2 trigger decision for waypoint {i}",
                trigger=trigger_reason.value if trigger_reason else None,
                visual_change=visual_change_result.changed,
                score_delta=round(score_delta, 2),
                distance_from_last=distance_from_last,
            )

            # Process waypoint through memory manager (attention gate + STM)
            context = memory_manager.process_waypoint(
                analysis,
                triggered=should_reason,
                trigger_reason=trigger_reason,
            )

            reasoning_result = None

            # Only run PersonaReasoner if skip_reasoning is False
            if not skip_reasoning and context is not None:
                # Ensure route metadata includes route info
                route_meta = context.get("route_metadata", {}) or {}
                route_meta.setdefault("route_id", route_id)
                route_meta.setdefault("length_km", route_length_km)

                try:
                    reasoning_result = self.persona_reasoner.reason(
                        waypoint_id=analysis.waypoint_id,
                        trigger_reason=context["trigger_reason"],
                        current_image_path=context["image_path"],
                        system1_scores=analysis.persona_scores,
                        system1_reasoning=analysis.persona_reasoning,
                        stm_context=context["stm_context"],
                        ltm_patterns=context.get("ltm_patterns"),
                        personality=self.personality,
                        route_metadata=route_meta,
                        waypoints_since_trigger=context.get("waypoints_since_trigger", 0),
                    )

                    reasoning_results.append(reasoning_result)

                    # Update STM with reasoning result for sequential context
                    memory_manager.update_with_system2_result(
                        waypoint_id=analysis.waypoint_id,
                        reasoning_result=reasoning_result,
                    )

                    # TODO: Narrative will be handled by PersonaReasoner._report()
                    # when System 2 is complete.

                    self.logger.info(
                        f"WP {analysis.waypoint_id:>3}   S2 {context['trigger_reason'].value:<20}"
                        f" significance={reasoning_result.significance}"
                        f" avoid={reasoning_result.avoid_recommendation}"
                        f" | {reasoning_result.interpretation[:80]}"
                    )
                    self.logger.info(
                        f"WP {analysis.waypoint_id:>3}   S2 rec: {reasoning_result.recommendation}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Reasoning failed at waypoint {analysis.waypoint_id}: {e}"
                    )

            ts_value = analysis.timestamp
            if hasattr(ts_value, "isoformat"):
                ts_value = ts_value.isoformat()

            waypoint_results.append(
                {
                    "waypoint_id": analysis.waypoint_id,
                    "image_path": str(analysis.image_path),
                    "gps": analysis.gps,
                    "timestamp": ts_value,
                    "system1_scores": analysis.persona_scores,
                    "system1_reasoning": analysis.persona_reasoning,
                    "system2_triggered": reasoning_result is not None,
                    "system2_scores": None,
                    "system2_reasoning": (
                        reasoning_result.recommendation if reasoning_result else None
                    ),
                    "reasoning_interpretation": (
                        reasoning_result.interpretation if reasoning_result else None
                    ),
                    "reasoning_significance": (
                        reasoning_result.significance if reasoning_result else None
                    ),
                    "memory_influence": (
                        reasoning_result.memory_influence if reasoning_result else None
                    ),
                    # Dual VLM evaluation fields (objective vs persona-aware)
                    "objective_scores": analysis.objective_scores,
                    "objective_reasoning": analysis.objective_reasoning,
                    "persona_scores": analysis.persona_scores,
                    "persona_reasoning": analysis.persona_reasoning,
                }
            )

        analysis_stats = self.continuous_analyzer.get_statistics()
        self.logger.info(
            "Continuous analysis complete",
            total_waypoints=analysis_stats["total_waypoints"],
            visual_changes=analysis_stats["visual_changes_detected"],
        )

        reasoning_summary = self.persona_reasoner.get_summary()
        self.logger.info(
            "Reasoning complete",
            episodes=len(reasoning_results),
            avg_confidence=reasoning_summary.get("avg_confidence", 0),
        )

        # ====================================================================
        # Phase 4: Long-Term Memory Formation
        # ====================================================================
        self.logger.info("Phase 4: Forming long-term memory")

        # Use agent's long-term memory
        ltm = self.memory

        # Add candidate moments
        for wp in waypoint_results:
            if not wp.get("system2_triggered"):
                continue

            reasoning_result = next(
                (t for t in reasoning_results if t.waypoint_id == wp["waypoint_id"]),
                None,
            )
            if not reasoning_result:
                continue

            ltm.add_candidate_moment(
                waypoint_id=wp["waypoint_id"],
                image_path=Path(wp["image_path"]),
                scores=wp["system1_scores"],
                summary=reasoning_result.interpretation[:200],
                significance=reasoning_result.significance,
                gps=wp.get("gps", (0.0, 0.0)),
                timestamp=wp.get("timestamp"),
                thinking_confidence=reasoning_result.confidence,
                visual_change_detected=wp.get("system2_triggered", False),
                score_delta=None,
            )

        # Curate moments
        ltm.curate_moments(route_length_km=route_length_km)

        # Extract patterns
        analysis_dicts = [
            {
                "waypoint_id": w["waypoint_id"],
                "scores": w["system1_scores"],
                "summary": "",
            }
            for w in waypoint_results
        ]
        reasoning_history = [t.interpretation for t in reasoning_results]

        ltm.extract_patterns(
            all_analyses=analysis_dicts, thinking_history=reasoning_history
        )

        # Generate route summary
        route_summary = ltm.generate_route_summary(
            route_id=route_id,
            total_waypoints=len(analysis_results),
            length_km=route_length_km,
            all_analyses=analysis_dicts,
        )

        self.logger.info(
            "Long-term memory formed",
            key_moments=len(ltm.get_key_images()),
            patterns=len(route_summary.patterns),
        )

        # ====================================================================
        # Phase 5: Save Outputs (if output_dir provided)
        # ====================================================================
        complete_narrative = ltm.get_complete_narrative()
        narrative_chapters = ltm.narrative_chapters

        if output_dir:
            import shutil

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Saving outputs", output_dir=str(output_dir))

            # Save analysis results with dual scores when available
            analysis_file = output_dir / "analysis_results.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(waypoint_results, f, indent=2)

            # Save System 1 only results
            system1_results = [
                {
                    "waypoint_id": wp["waypoint_id"],
                    "image_path": wp["image_path"],
                    "gps": wp["gps"],
                    "timestamp": wp["timestamp"],
                    "scores": wp["system1_scores"],
                    "reasoning": wp["system1_reasoning"],
                }
                for wp in waypoint_results
            ]
            system1_file = output_dir / "analysis_results_system1.json"
            with open(system1_file, "w", encoding="utf-8") as f:
                json.dump(system1_results, f, indent=2)

            # Save System 2 only results (only waypoints where system2 was triggered)
            system2_results = [
                {
                    "waypoint_id": wp["waypoint_id"],
                    "image_path": wp["image_path"],
                    "gps": wp["gps"],
                    "timestamp": wp["timestamp"],
                    "reasoning": wp["system2_reasoning"],
                    "interpretation": wp["reasoning_interpretation"],
                    "significance": wp["reasoning_significance"],
                    "memory_influence": wp["memory_influence"],
                }
                for wp in waypoint_results
                if wp["system2_triggered"]
            ]
            system2_file = output_dir / "analysis_results_system2.json"
            with open(system2_file, "w", encoding="utf-8") as f:
                json.dump(system2_results, f, indent=2)

            # Save reasoning results
            reasoning_file = output_dir / "reasoning_results.json"
            with open(reasoning_file, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "waypoint_id": t.waypoint_id,
                            "trigger_reason": t.trigger_reason.value,
                            "interpretation": t.interpretation,
                            "significance": t.significance,
                            "confidence": t.confidence,
                        }
                        for t in reasoning_results
                    ],
                    f,
                    indent=2,
                )

            # Save narrative
            narrative_file = output_dir / "narrative.md"
            with open(narrative_file, "w", encoding="utf-8") as f:
                f.write(complete_narrative)

            # Save complete narrative markdown
            narrative_md = output_dir / "complete_narrative.md"
            with open(narrative_md, "w", encoding="utf-8") as f:
                f.write(complete_narrative)

            # Copy key images
            key_images_dir = output_dir / "key_images"
            key_images_dir.mkdir(exist_ok=True)

            for img_path in ltm.get_key_images():
                if img_path.exists():
                    shutil.copy2(img_path, key_images_dir / img_path.name)

            self.logger.info(
                "Outputs saved",
                output_dir=str(output_dir),
                system1_waypoints=len(system1_results),
                system2_waypoints=len(system2_results),
            )

        # ====================================================================
        # Build final result
        # ====================================================================
        result = {
            "analysis_results": waypoint_results,
            "reasoning_results": reasoning_results,
            "narrative_chapters": narrative_chapters,
            "complete_narrative": complete_narrative,
            "long_term_memory": ltm,
            "route_summary": route_summary,
            "statistics": {
                "analysis": analysis_stats,
                "reasoning": reasoning_summary,
                "dual_system": self._compute_dual_system_statistics(waypoint_results),
                "route_length_km": route_length_km,
                "total_waypoints": len(waypoint_results),
            },
            "route_data": {
                "route_id": route_id,
                "start": start,
                "end": end,
                "interval": interval,
            },
        }

        self.logger.info(
            "Memory pipeline complete",
            route_id=route_id,
            recommendation=route_summary.overall_recommendation,
        )

        return result

    def run_with_memory_from_folder(
        self,
        route_folder: Union[str, Path],
        *,
        metadata_filename: str = "collection_metadata.json",
        output_dir: Optional[Path] = None,
        skip_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Run memory pipeline using pre-existing folder of route imagery.

        This method loads waypoint images and metadata from an existing route folder
        and processes them through the full memory system (continuous analysis,
        short-term memory, PersonaReasoner, and long-term memory).

        Args:
            route_folder: Path to folder containing waypoint images and metadata.
            metadata_filename: Name of the JSON metadata file (default: "collection_metadata.json").
            output_dir: Optional directory to save memory artifacts.
            skip_reasoning: If True, skip Phase 3 (PersonaReasoner) and use only Phase 2 scores.

        Returns:
            Dictionary containing analysis results, thinking results, narrative, etc.
            Same format as run_with_memory().

        Example:
            ```python
            agent = CityWalkAgent.from_preset("safety", "streetagent_5d")
            result = agent.run_with_memory_from_folder(
                route_folder="data/images/singapore/",
                output_dir=Path("outputs/singapore_analysis")
            )
            ```
        """
        folder_path = Path(route_folder).expanduser().resolve()
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Route folder does not exist: {folder_path}")

        metadata_path = folder_path / metadata_filename
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Route folder is missing metadata file '{metadata_filename}' in {folder_path}"
            )

        self.logger.debug("Loading route from folder", folder_path=str(folder_path))

        # Load metadata
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        waypoint_entries: List[Dict[str, Any]] = metadata.get("results") or []
        if not waypoint_entries:
            raise ValueError(f"No waypoint entries found in {metadata_path}")

        # Sort by waypoint_id
        waypoint_entries = sorted(
            waypoint_entries,
            key=lambda entry: entry.get("waypoint_id", 0),
        )

        # Build waypoint list and image paths
        resolved_waypoints: List[Waypoint] = []
        resolved_image_paths: List[Path] = []

        for entry in waypoint_entries:
            sequence_id = entry.get("waypoint_id", len(resolved_waypoints))

            # Resolve image path - try multiple strategies
            image_path = entry.get("image_path")
            image_candidate: Optional[Path] = None

            # Strategy 1: If metadata provides a path, try it
            if image_path:
                candidate = Path(image_path)

                # If it's absolute, check if it exists
                if candidate.is_absolute():
                    if candidate.exists():
                        image_candidate = candidate
                    else:
                        # Absolute path doesn't exist, try using just the filename
                        # in the current folder (images may have been moved)
                        image_candidate = folder_path / candidate.name
                else:
                    # Relative path - resolve relative to folder_path
                    image_candidate = folder_path / candidate

            # Strategy 2: If still not found, look for waypoint pattern in folder
            if image_candidate is None or not image_candidate.exists():
                # Try pattern: waypoint_NNN_*.jpg
                pattern_candidates = list(
                    folder_path.glob(f"waypoint_{sequence_id:03d}_*.jpg")
                )
                if pattern_candidates:
                    image_candidate = pattern_candidates[0]

            # Strategy 3: Look in waypoint subdirectory
            if image_candidate is None or not image_candidate.exists():
                waypoint_dir = folder_path / f"waypoint_{sequence_id:03d}"
                if waypoint_dir.is_dir():
                    jpg_files = list(waypoint_dir.glob("*.jpg"))
                    if jpg_files:
                        image_candidate = jpg_files[0]

            # Final check
            if image_candidate is None or not image_candidate.exists():
                raise FileNotFoundError(
                    f"Could not locate image for waypoint {sequence_id} in {folder_path}. "
                    f"Expected pattern: waypoint_{sequence_id:03d}_*.jpg"
                )

            resolved_image_paths.append(image_candidate)

            waypoint = Waypoint(
                lat=entry["lat"],
                lon=entry["lon"],
                sequence_id=sequence_id,
                heading=entry.get("heading"),
                timestamp=None,
                image_path=str(image_candidate.resolve()),
            )
            resolved_waypoints.append(waypoint)

        if not resolved_waypoints:
            raise ValueError(f"Failed to build waypoints from metadata {metadata_path}")

        route_id = metadata.get("route_id") or folder_path.name
        interval = (
            metadata.get("interval")
            or metadata.get("interval_meters")
            or settings.default_sampling_interval
        )

        # Build route object
        route = Route(
            route_id=route_id,
            start_lat=resolved_waypoints[0].lat,
            start_lon=resolved_waypoints[0].lon,
            end_lat=resolved_waypoints[-1].lat,
            end_lon=resolved_waypoints[-1].lon,
            waypoints=resolved_waypoints,
            route_name=metadata.get("route_name"),
            interval_meters=int(interval),
        )

        self.logger.debug(
            "Route loaded", route_id=route_id, waypoints=len(resolved_waypoints)
        )

        # Prepare metadata for continuous analyzer
        waypoint_metadata = [
            {
                "filename": img_path.name,
                "lat": wp.lat,
                "lon": wp.lon,
                "heading": wp.heading or 0.0,
                "timestamp": wp.timestamp or "",
            }
            for img_path, wp in zip(resolved_image_paths, resolved_waypoints)
        ]

        # Validate images and reconstruct waypoint sequence
        resolved_image_paths, resolved_waypoints, waypoint_metadata = (
            self._validate_and_reconstruct_waypoints(
                resolved_image_paths, resolved_waypoints, waypoint_metadata
            )
        )

        # Calculate route length
        route_length_km = 0.0
        for i in range(len(resolved_waypoints) - 1):
            wp1 = resolved_waypoints[i]
            wp2 = resolved_waypoints[i + 1]
            route_length_km += geodesic((wp1.lat, wp1.lon), (wp2.lat, wp2.lon)).meters
        route_length_km /= 1000.0

        enhanced_persona = self._get_enhanced_persona()
        persona_name = (
            getattr(enhanced_persona, "name", "objective")
            if enhanced_persona is not None
            else "objective"
        )
        self.logger.info(
            f"Route: {len(resolved_waypoints)} waypoints, {route_length_km:.2f} km"
            f" | persona={persona_name}"
        )

        # ====================================================================
        # Phase 2+3: Interleaved analysis and memory per waypoint
        # STM is populated immediately after each analysis so the next
        # waypoint's VLM call receives prior context.
        # ====================================================================
        if skip_reasoning:
            self.logger.debug(
                "Phase 2+3: Continuous analysis (reasoning SKIPPED — system1-only mode)"
            )
        else:
            self.logger.debug(
                "Phase 2+3: Interleaved continuous analysis + memory + reasoning"
            )

        # Reset state for new route analysis
        self.cognitive.reset()
        self.continuous_analyzer.reset()
        self.logger.debug(
            "CognitiveController and ContinuousAnalyzer state reset for new route"
        )

        analysis_results = []
        reasoning_results = []
        memory_manager = self.memory_manager
        waypoint_results: List[Dict[str, Any]] = []

        for i, (img_path, meta) in enumerate(
            zip(resolved_image_paths, waypoint_metadata)
        ):
            # Step A: CognitiveController detects visual change
            is_first_waypoint = i == 0
            visual_change_result = self.cognitive.detect_visual_change(
                image_path=img_path, force=is_first_waypoint
            )

            self.logger.debug(
                f"Visual change detection for waypoint {i}",
                changed=visual_change_result.changed,
                distance=visual_change_result.phash_distance,
                reason=visual_change_result.reason,
            )

            # Step B: ContinuousAnalyzer analyzes waypoint (STM already populated by prior iterations)
            analysis = self.continuous_analyzer.analyze_waypoint(
                waypoint_id=i,
                image_path=img_path,
                metadata=meta,
                visual_change_detected=visual_change_result.changed,
                phash_distance=visual_change_result.phash_distance,
            )

            self.continuous_analyzer.analysis_history.append(analysis)
            analysis_results.append(analysis)

            # Step C: PersonaReasoner.should_trigger()로 종합 판단
            stm_context_for_trigger = self.memory_manager.stm.get_context()

            # score_delta 계산
            current_scores = analysis.persona_scores or analysis.scores
            current_avg = (
                sum(current_scores.values()) / len(current_scores)
                if current_scores
                else 5.0
            )
            prev_avg = self._get_prev_avg_score(stm_context_for_trigger)
            score_delta = abs(current_avg - prev_avg)

            # distance_from_last: lat/lon으로 직접 계산
            if i == 0:
                distance_from_last = 0.0
            else:
                prev_meta = waypoint_metadata[i - 1]
                try:
                    distance_from_last = geodesic(
                        (prev_meta.get("lat", 0), prev_meta.get("lon", 0)),
                        (meta.get("lat", 0), meta.get("lon", 0)),
                    ).meters
                except Exception:
                    distance_from_last = 0.0

            # PersonaReasoner.should_trigger()로 종합 판단
            trigger_reason = self.persona_reasoner.should_trigger(
                waypoint_id=i,
                visual_change=visual_change_result.changed,
                score_delta=score_delta,
                distance_from_last=distance_from_last,
                is_exceptional=(i == 0),
            )
            should_reason = trigger_reason is not None

            self.logger.debug(
                f"System 2 trigger decision for waypoint {i}",
                trigger=trigger_reason.value if trigger_reason else None,
                visual_change=visual_change_result.changed,
                score_delta=round(score_delta, 2),
                distance_from_last=distance_from_last,
            )

            # Process waypoint through memory manager (attention gate + STM)
            context = memory_manager.process_waypoint(
                analysis,
                triggered=should_reason,
                trigger_reason=trigger_reason,
            )

            reasoning_result = None

            # Only run PersonaReasoner if skip_reasoning is False
            if not skip_reasoning and context is not None:
                # Ensure route metadata includes route info
                route_meta = context.get("route_metadata", {}) or {}
                route_meta.setdefault("route_id", route_id)
                route_meta.setdefault("length_km", route_length_km)

                try:
                    reasoning_result = self.persona_reasoner.reason(
                        waypoint_id=analysis.waypoint_id,
                        trigger_reason=context["trigger_reason"],
                        current_image_path=context["image_path"],
                        system1_scores=analysis.persona_scores,
                        system1_reasoning=analysis.persona_reasoning,
                        stm_context=context["stm_context"],
                        ltm_patterns=context.get("ltm_patterns"),
                        personality=self.personality,
                        route_metadata=route_meta,
                        waypoints_since_trigger=context.get("waypoints_since_trigger", 0),
                    )

                    reasoning_results.append(reasoning_result)

                    # Update STM with reasoning result for sequential context
                    memory_manager.update_with_system2_result(
                        waypoint_id=analysis.waypoint_id,
                        reasoning_result=reasoning_result,
                    )

                    # TODO: Narrative will be handled by PersonaReasoner._report()
                    # when System 2 is complete.

                    self.logger.info(
                        f"WP {analysis.waypoint_id:>3}   S2 {context['trigger_reason'].value:<20}"
                        f" significance={reasoning_result.significance}"
                        f" avoid={reasoning_result.avoid_recommendation}"
                        f" | {reasoning_result.interpretation[:80]}"
                    )
                    self.logger.info(
                        f"WP {analysis.waypoint_id:>3}   S2 rec: {reasoning_result.recommendation}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Reasoning failed at waypoint {analysis.waypoint_id}: {e}"
                    )

            ts_value = analysis.timestamp
            if hasattr(ts_value, "isoformat"):
                ts_value = ts_value.isoformat()

            waypoint_results.append(
                {
                    "waypoint_id": analysis.waypoint_id,
                    "image_path": str(analysis.image_path),
                    "gps": analysis.gps,
                    "timestamp": ts_value,
                    "system1_scores": analysis.persona_scores,
                    "system1_reasoning": analysis.persona_reasoning,
                    "system2_triggered": reasoning_result is not None,
                    "system2_scores": None,
                    "system2_reasoning": (
                        reasoning_result.recommendation if reasoning_result else None
                    ),
                    "reasoning_interpretation": (
                        reasoning_result.interpretation if reasoning_result else None
                    ),
                    "reasoning_significance": (
                        reasoning_result.significance if reasoning_result else None
                    ),
                    "memory_influence": (
                        reasoning_result.memory_influence if reasoning_result else None
                    ),
                    # Dual VLM evaluation fields (objective vs persona-aware)
                    "objective_scores": analysis.objective_scores,
                    "objective_reasoning": analysis.objective_reasoning,
                    "persona_scores": analysis.persona_scores,
                    "persona_reasoning": analysis.persona_reasoning,
                }
            )

        analysis_stats = self.continuous_analyzer.get_statistics()
        self.logger.info(
            "Continuous analysis complete",
            total_waypoints=analysis_stats["total_waypoints"],
            visual_changes=analysis_stats["visual_changes_detected"],
        )

        reasoning_summary = self.persona_reasoner.get_summary()
        self.logger.info(
            "Reasoning complete",
            episodes=len(reasoning_results),
            avg_confidence=reasoning_summary.get("avg_confidence", 0),
        )

        # Phase 4: Long-Term Memory Formation
        self.logger.info("Phase 4: Forming long-term memory")

        # Use agent's long-term memory
        ltm = self.memory

        # Add candidate moments
        for wp in waypoint_results:
            if not wp.get("system2_triggered"):
                continue

            reasoning_result = next(
                (t for t in reasoning_results if t.waypoint_id == wp["waypoint_id"]),
                None,
            )
            if not reasoning_result:
                continue

            ltm.add_candidate_moment(
                waypoint_id=wp["waypoint_id"],
                image_path=Path(wp["image_path"]),
                scores=wp["system1_scores"],
                summary=reasoning_result.interpretation[:200],
                significance=reasoning_result.significance,
                gps=wp.get("gps", (0.0, 0.0)),
                timestamp=wp.get("timestamp"),
                thinking_confidence=reasoning_result.confidence,
                visual_change_detected=wp.get("system2_triggered", False),
                score_delta=None,
            )

        # Curate moments
        ltm.curate_moments(route_length_km=route_length_km)

        # Extract patterns
        analysis_dicts = [
            {
                "waypoint_id": w["waypoint_id"],
                "scores": w["system1_scores"],
                "summary": "",
            }
            for w in waypoint_results
        ]
        reasoning_history = [t.interpretation for t in reasoning_results]

        ltm.extract_patterns(
            all_analyses=analysis_dicts, thinking_history=reasoning_history
        )

        # Generate route summary
        route_summary = ltm.generate_route_summary(
            route_id=route_id,
            total_waypoints=len(analysis_results),
            length_km=route_length_km,
            all_analyses=analysis_dicts,
        )

        self.logger.info(
            "Long-term memory formed",
            key_moments=len(ltm.get_key_images()),
            patterns=len(route_summary.patterns),
        )

        # Phase 5: Save Outputs (if output_dir provided)
        complete_narrative = ltm.get_complete_narrative()
        narrative_chapters = ltm.narrative_chapters

        if output_dir:
            import shutil

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Saving outputs", output_dir=str(output_dir))

            # Save analysis results with dual scores when available
            analysis_file = output_dir / "analysis_results.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(waypoint_results, f, indent=2)

            # Save System 1 only results
            system1_results = [
                {
                    "waypoint_id": wp["waypoint_id"],
                    "image_path": wp["image_path"],
                    "gps": wp["gps"],
                    "timestamp": wp["timestamp"],
                    "scores": wp["system1_scores"],
                    "reasoning": wp["system1_reasoning"],
                }
                for wp in waypoint_results
            ]
            system1_file = output_dir / "analysis_results_system1.json"
            with open(system1_file, "w", encoding="utf-8") as f:
                json.dump(system1_results, f, indent=2)

            # Save System 2 only results (only waypoints where system2 was triggered)
            system2_results = [
                {
                    "waypoint_id": wp["waypoint_id"],
                    "image_path": wp["image_path"],
                    "gps": wp["gps"],
                    "timestamp": wp["timestamp"],
                    "reasoning": wp["system2_reasoning"],
                    "interpretation": wp["reasoning_interpretation"],
                    "significance": wp["reasoning_significance"],
                    "memory_influence": wp["memory_influence"],
                }
                for wp in waypoint_results
                if wp["system2_triggered"]
            ]
            system2_file = output_dir / "analysis_results_system2.json"
            with open(system2_file, "w", encoding="utf-8") as f:
                json.dump(system2_results, f, indent=2)

            # Save reasoning results
            reasoning_file = output_dir / "reasoning_results.json"
            with open(reasoning_file, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "waypoint_id": t.waypoint_id,
                            "trigger_reason": t.trigger_reason.value,
                            "interpretation": t.interpretation,
                            "significance": t.significance,
                            "confidence": t.confidence,
                        }
                        for t in reasoning_results
                    ],
                    f,
                    indent=2,
                )

            # Save narrative
            narrative_file = output_dir / "narrative.md"
            with open(narrative_file, "w", encoding="utf-8") as f:
                f.write(complete_narrative)

            # Save complete narrative markdown
            narrative_md = output_dir / "complete_narrative.md"
            with open(narrative_md, "w", encoding="utf-8") as f:
                f.write(complete_narrative)

            # Copy key images
            key_images_dir = output_dir / "key_images"
            key_images_dir.mkdir(exist_ok=True)

            for img_path in ltm.get_key_images():
                if img_path.exists():
                    shutil.copy2(img_path, key_images_dir / img_path.name)

            # Save memory debug snapshot (before complete_route resets state)
            memory_debug = {
                "stm_final": {
                    "window_size": memory_manager.stm.window_size,
                    "items": [
                        {
                            "waypoint_id": item.waypoint_id,
                            "scores": item.scores,
                            "summary": item.summary,
                        }
                        for item in memory_manager.stm.get_all_items()
                    ],
                },
                "ltm_snapshots": memory_manager._route_snapshots,
                "ltm_reasoning_episodes": memory_manager._route_reasoning_log,
                "stats": {
                    "waypoints_since_trigger": memory_manager._waypoints_since_trigger,
                    "total_snapshots": len(memory_manager._route_snapshots),
                    "total_episodes": len(memory_manager._route_reasoning_log),
                },
            }
            memory_debug_file = output_dir / "memory_debug.json"
            with open(memory_debug_file, "w", encoding="utf-8") as f:
                json.dump(memory_debug, f, indent=2, ensure_ascii=False)

            self.logger.info(
                "Outputs saved",
                output_dir=str(output_dir),
                system1_waypoints=len(system1_results),
                system2_waypoints=len(system2_results),
            )

        # Build final result
        result = {
            "analysis_results": waypoint_results,
            "reasoning_results": reasoning_results,
            "narrative_chapters": narrative_chapters,
            "complete_narrative": complete_narrative,
            "long_term_memory": ltm,
            "route_summary": route_summary,
            "statistics": {
                "analysis": analysis_stats,
                "reasoning": reasoning_summary,
                "dual_system": self._compute_dual_system_statistics(waypoint_results),
                "route_length_km": route_length_km,
                "total_waypoints": len(waypoint_results),
            },
            "route_data": {
                "route_id": route_id,
                "start": (route.start_lat, route.start_lon),
                "end": (route.end_lat, route.end_lon),
                "interval": interval,
            },
        }

        self.logger.info(
            "Memory pipeline complete",
            route_id=route_id,
            recommendation=route_summary.overall_recommendation,
        )

        return result

    def synthesize_route(
        self,
        analysis_results: List[Any],
        route_metadata: Dict[str, Any],
        candidate_routes: Optional[List[Any]] = None,
        reasoning_results: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """경로 완주 후 전체 결과를 종합하는 route-level System 2 파이프라인.

        waypoint-level PersonaReasoner.reason()과 구분되는 경로 전체 요약 단계.

        Args:
            analysis_results:  Output from run_with_memory() or run_with_memory_from_folder().
            route_metadata:    Route-level info dict (route_id, length_km, etc.).
            candidate_routes:  Optional pre-generated alternative routes for Planner.
            reasoning_results: Optional PersonaReasoner waypoint-level results.

        Returns:
            Dict with keys: interpret, decide, plan, report.

        Raises:
            NotImplementedError: Until System 2 stages are implemented.
        """
        from src.agent.system2.models import System1Evidence

        enhanced_persona = self._get_enhanced_persona()

        evidence = System1Evidence(
            waypoint_results=analysis_results,
            route_metadata=route_metadata,
            reasoning_results=(
                reasoning_results if reasoning_results is not None else []
            ),
        )

        interpret_result = self.interpreter.interpret(evidence, enhanced_persona)
        decide_result = self.decider.decide(
            evidence, interpret_result, enhanced_persona
        )
        plan_result = self.planner.plan(
            evidence, decide_result, candidate_routes, enhanced_persona
        )
        report_result = self.reporter.report(
            evidence, interpret_result, decide_result, plan_result, enhanced_persona
        )

        return {
            "interpret": interpret_result,
            "decide": decide_result,
            "plan": plan_result,
            "report": report_result,
        }

    def _compute_dual_system_statistics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics comparing System 1 vs System 2."""
        import statistics as stats

        system2_results = [r for r in results if r.get("system2_triggered")]

        if not system2_results:
            return {
                "total_waypoints": len(results),
                "system2_triggered": 0,
                "system2_trigger_rate": 0.0,
            }

        all_adjustments: List[float] = []
        for r in system2_results:
            adjustments = r.get("score_adjustments") or {}
            if adjustments:
                avg_adj = sum(adjustments.values()) / len(adjustments)
                all_adjustments.append(avg_adj)

        return {
            "total_waypoints": len(results),
            "system2_triggered": len(system2_results),
            "system2_trigger_rate": len(system2_results) / len(results),
            "avg_score_adjustment": (
                stats.mean(all_adjustments) if all_adjustments else 0.0
            ),
            "max_adjustment": max(all_adjustments) if all_adjustments else 0.0,
            "min_adjustment": min(all_adjustments) if all_adjustments else 0.0,
            "negative_adjustments": sum(1 for adj in all_adjustments if adj < 0),
            "positive_adjustments": sum(1 for adj in all_adjustments if adj > 0),
            "no_change": sum(1 for adj in all_adjustments if abs(adj) < 0.1),
        }

    def branch_decision(
        self,
        branch_pano_id: str,
        candidate_headings: List[float],
        memory_manager: Any,
        destination_context: Optional[str] = None,
        visit_counts: Optional[Dict[str, int]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate candidate directions at a branch point and choose the best.

        Args:
            branch_pano_id: pano_id of the branch point (current position)
            candidate_headings: List of compass headings to evaluate (2–5 directions)
            memory_manager: Current route MemoryManager (provides LTM context)
            output_dir: Optional directory to save branch decision log

        Returns:
            {
                "chosen_direction": str,
                "chosen_heading": float,
                "chosen_pano_id": str,
                "reason": str,
                "confidence": float,
                "ranking": List[str],
                "candidates": List[Dict],  # full per-direction analysis
            }
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import string
        import tempfile

        import requests

        if not (2 <= len(candidate_headings) <= 5):
            raise ValueError(
                f"candidate_headings must have 2–5 entries, got {len(candidate_headings)}"
            )

        direction_labels = list(string.ascii_uppercase[: len(candidate_headings)])

        def fetch_and_analyze(label: str, heading: float) -> Dict[str, Any]:
            image_url = (
                f"https://maps.googleapis.com/maps/api/streetview"
                f"?size=640x640&pano={branch_pano_id}&heading={heading}"
                f"&pitch=0&fov=90&key={settings.google_maps_api_key}"
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            try:
                resp = requests.get(image_url, timeout=15)
                resp.raise_for_status()
                tmp.write(resp.content)
                tmp.flush()
                image_path = Path(tmp.name)
            finally:
                tmp.close()

            try:
                analysis = self.continuous_analyzer.analyze_waypoint(
                    waypoint_id=0,
                    image_path=image_path,
                    metadata={"heading": heading, "lat": 0, "lon": 0},
                    visual_change_detected=True,
                    phash_distance=None,
                )
            finally:
                image_path.unlink(missing_ok=True)

            return {
                "direction": label,
                "heading": heading,
                "pano_id": branch_pano_id,
                "scores": analysis.persona_scores,
                "system1_reasoning": analysis.persona_reasoning,
                "visit_count": (visit_counts or {}).get(branch_pano_id, 0),
            }

        self.logger.info(
            f"Branch decision: {len(candidate_headings)} candidates at pano={branch_pano_id}"
        )

        with ThreadPoolExecutor(max_workers=len(candidate_headings)) as executor:
            futures = {
                executor.submit(fetch_and_analyze, label, heading): label
                for label, heading in zip(direction_labels, candidate_headings)
            }
            direction_results = []
            for future in as_completed(futures):
                direction_results.append(future.result())

        direction_results.sort(key=lambda x: direction_labels.index(x["direction"]))

        ltm_patterns = {
            "snapshots": list(memory_manager._route_snapshots),
            "reasoning_episodes": list(memory_manager._route_reasoning_log),
        }

        def run_interpreter(dr: Dict) -> Dict:
            interp = self.persona_reasoner.interpreter.interpret_waypoint(
                waypoint_id=0,
                system1_scores=dr["scores"],
                system1_reasoning=dr["system1_reasoning"],
                stm_context=memory_manager.stm.get_context(),
                trigger_reason=None,
                personality=self.personality,
                dimension_ids=self.persona_reasoner.dimension_ids,
                dimensions=self.persona_reasoner.dimensions,
                waypoints_since_trigger=memory_manager._waypoints_since_trigger,
                ltm_patterns=ltm_patterns,
            )
            return {**dr, "interpretation": interp.get("text", ""), "key_concern": interp.get("key_concern")}

        with ThreadPoolExecutor(max_workers=len(direction_results)) as executor:
            candidates = list(executor.map(run_interpreter, direction_results))

        decision = self.persona_reasoner.decider.decide_branch(
            candidates=candidates,
            ltm_patterns=ltm_patterns,
            personality=self.personality,
            dimension_ids=self.persona_reasoner.dimension_ids,
            dimensions=self.persona_reasoner.dimensions,
            destination_context=destination_context,
        )

        self.logger.info(
            f"Branch decision: chosen={decision['chosen_direction']} "
            f"heading={decision['chosen_heading']} confidence={decision['confidence']}"
        )

        result = {**decision, "candidates": candidates}

        if output_dir:
            branch_log = {
                "branch_pano_id": branch_pano_id,
                "candidate_headings": candidate_headings,
                "candidates": [
                    {
                        "direction": c["direction"],
                        "heading": c["heading"],
                        "scores": c["scores"],
                        "interpretation": c["interpretation"],
                        "key_concern": c["key_concern"],
                    }
                    for c in candidates
                ],
                "decision": decision,
            }
            log_path = Path(output_dir) / "branch_decisions.json"
            existing = []
            if log_path.exists():
                with open(log_path) as f:
                    existing = json.load(f)
            existing.append(branch_log)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        return result

    async def autonomous_walk(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        max_steps: int = 60,
        arrival_threshold_m: float = 50.0,
        output_dir: Optional[Path] = None,
        step_callback=None,
    ) -> Dict[str, Any]:
        """Walk autonomously from start to destination.

        Each step:
        1. Fetch pano metadata → links
        2. Analyze current position (System 1)
        3. Check arrival
        4. Filter candidate headings toward destination
        5. branch_decision() if 2+ candidates, else take single heading
        6. Move to next pano
        """
        import asyncio
        import tempfile

        import httpx

        from src.agent.memory.memory_manager import MemoryManager
        from src.agent.system2.persona_reasoner import TriggerReason
        from demo.server import MapTilesSession

        loop = asyncio.get_event_loop()
        nav = MapTilesSession(settings.google_maps_api_key)

        memory_manager = MemoryManager(agent_id=f"walk_{int(time.time())}")
        if self.personality:
            memory_manager.set_agent_attributes(
                personality=self.personality,
                profile={"name": "Autonomous Walker"},
                status={"mode": "autonomous"},
            )
        self.continuous_analyzer.memory_manager = memory_manager

        visit_counts: Dict[str, int] = {}
        route_taken: List[Dict] = []
        arrived = False
        current_lat, current_lng = start_lat, start_lng
        prev_avg_score: float = 0.0
        distance_from_last_trigger: float = 0.0
        trigger_reason = None

        pano_id = await nav.coord_to_pano_id(start_lat, start_lng)
        if not pano_id:
            raise ValueError(f"No Street View pano at ({start_lat}, {start_lng})")

        self.logger.info(
            f"autonomous_walk start | pano={pano_id[:12]} "
            f"dest=({dest_lat:.4f},{dest_lng:.4f}) max_steps={max_steps}"
        )

        for step in range(max_steps):
            # Hard stop: same pano visited 5+ times
            if visit_counts.get(pano_id, 0) >= 5:
                self.logger.warning(
                    f"Step {step}: pano {pano_id[:12]} visited 5+ times — stuck, stopping"
                )
                break

            # 1. Metadata
            try:
                metadata = await nav.get_metadata(pano_id)
            except Exception as e:
                self.logger.warning(f"Step {step}: metadata error — {e}")
                break

            current_lat = metadata.get("lat", current_lat)
            current_lng = metadata.get("lng", current_lng)
            links = metadata.get("links", [])

            if not links:
                self.logger.warning(f"Step {step}: no links — stopping")
                break

            # 2. Arrival check
            dist_m = geodesic(
                (current_lat, current_lng), (dest_lat, dest_lng)
            ).meters

            if dist_m <= arrival_threshold_m:
                arrived = True
                self.logger.info(f"Step {step}: ARRIVED — dist={dist_m:.0f}m")
                break

            # 3. Destination context
            dest_bearing = _calc_bearing(current_lat, current_lng, dest_lat, dest_lng)
            cardinal = _bearing_to_cardinal(dest_bearing)
            dest_context = f"{cardinal} ({dest_bearing:.0f}°), {dist_m:.0f}m away"

            # 4. Analyze current pano (System 1)
            analysis = None
            image_path = None
            try:
                sv_url = (
                    f"https://maps.googleapis.com/maps/api/streetview"
                    f"?pano={pano_id}&size=640x640"
                    f"&heading={links[0].get('heading', 0):.0f}"
                    f"&fov=90&pitch=0&key={settings.google_maps_api_key}"
                )
                async with httpx.AsyncClient(timeout=15.0) as client:
                    img_resp = await client.get(sv_url)
                    img_resp.raise_for_status()

                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.write(img_resp.content)
                tmp.flush()
                tmp.close()
                image_path = Path(tmp.name)

                analysis = await loop.run_in_executor(
                    None,
                    lambda: self.continuous_analyzer.analyze_waypoint(
                        waypoint_id=step,
                        image_path=image_path,
                        metadata={"heading": links[0].get("heading", 0),
                                  "lat": current_lat, "lon": current_lng},
                        visual_change_detected=True,
                        phash_distance=None,
                    ),
                )
            except Exception as e:
                self.logger.warning(f"Step {step}: analysis error — {e}")
            finally:
                if image_path is not None:
                    image_path.unlink(missing_ok=True)

            # 5. Update memory (S2 trigger logic)
            trigger_reason = None
            if analysis:
                current_avg = (
                    sum(analysis.persona_scores.values()) / len(analysis.persona_scores)
                    if analysis.persona_scores else 0.0
                )
                score_delta = abs(current_avg - prev_avg_score)

                trigger_reason = self.persona_reasoner.should_trigger(
                    waypoint_id=step,
                    visual_change=True,
                    score_delta=score_delta,
                    distance_from_last=distance_from_last_trigger,
                )

                if trigger_reason is None and len(links) >= 3:
                    trigger_reason = TriggerReason.VISUAL_CHANGE
                    self.logger.debug(f"Step {step}: intersection trigger")

                memory_manager.process_waypoint(
                    analysis,
                    triggered=trigger_reason is not None,
                    trigger_reason=trigger_reason,
                )

                if trigger_reason is not None:
                    distance_from_last_trigger = 0.0

                prev_avg_score = current_avg

            # 6. All links as candidates
            candidate_headings = [
                float(l.get("heading") or l.get("yawDeg") or 0)
                for l in links
            ]

            # 7. Choose direction
            chosen_heading = candidate_headings[0]
            recommendation = None

            if len(candidate_headings) >= 2 and analysis:
                try:
                    branch_result = await loop.run_in_executor(
                        None,
                        lambda: self.branch_decision(
                            branch_pano_id=pano_id,
                            candidate_headings=candidate_headings,
                            memory_manager=memory_manager,
                            destination_context=dest_context,
                            visit_counts=visit_counts,
                        ),
                    )
                    chosen_heading = branch_result["chosen_heading"]
                    recommendation = branch_result.get("reason")
                    self.logger.info(
                        f"Step {step:>3} | branch → {branch_result['chosen_direction']} "
                        f"({chosen_heading:.0f}°) conf={branch_result['confidence']:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"Step {step}: branch error — {e}")

            # 8. Find next pano (prefer unvisited)
            unvisited_links = [
                l for l in links
                if visit_counts.get(l.get("panoId") or l.get("id"), 0) == 0
            ]
            target_links = unvisited_links if unvisited_links else links
            next_link = _closest_link(target_links, chosen_heading)
            next_pano_id = next_link.get("panoId") or next_link.get("id")

            if not next_pano_id:
                self.logger.warning(f"Step {step}: no next pano — stopping")
                break

            # 9. Record step
            step_result = {
                "step": step,
                "pano_id": pano_id,
                "lat": current_lat,
                "lng": current_lng,
                "heading": round(chosen_heading, 1),
                "dist_to_dest_m": round(dist_m, 1),
                "dest_bearing": round(dest_bearing, 1),
                "dest_cardinal": cardinal,
                "scores": analysis.persona_scores if analysis else {},
                "recommendation": recommendation,
                "is_intersection": len(links) >= 3,
                "candidate_count": len(candidate_headings),
                "visit_count": visit_counts.get(pano_id, 0),
                "trigger_reason": trigger_reason.value if trigger_reason else None,
            }
            route_taken.append(step_result)
            visit_counts[pano_id] = visit_counts.get(pano_id, 0) + 1
            distance_from_last_trigger += 20.0

            self.logger.info(
                f"Step {step:>3} | dist={dist_m:.0f}m {cardinal}({dest_bearing:.0f}°) "
                f"→ heading={chosen_heading:.0f}° | next={next_pano_id[:12]}"
            )

            if step_callback:
                await step_callback(step_result)

            pano_id = next_pano_id

        final_dist = geodesic(
            (current_lat, current_lng), (dest_lat, dest_lng)
        ).meters

        result = {
            "arrived": arrived,
            "steps": len(route_taken),
            "final_distance_m": round(final_dist, 1),
            "route_taken": route_taken,
            "persona": getattr(self.personality, "name", "objective"),
            "memory_debug": {
                "snapshots": memory_manager._route_snapshots,
                "episodes": memory_manager._route_reasoning_log,
            },
        }

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "walk_log.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Walk log saved → {output_dir}/walk_log.json")

        self.logger.info(
            f"Walk complete | arrived={arrived} steps={len(route_taken)} "
            f"final_dist={final_dist:.0f}m"
        )
        return result
