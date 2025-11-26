"""WalkingAgent - orchestrates all capabilities for route analysis.

This is the main concrete agent implementation that uses ObservationCapability,
ThinkingCapability, and LongTermMemory to analyze walking routes with personality-driven
decision making.

Two Analysis Modes:
1. run(): Traditional route-level analysis (backward compatible)
   - Post-hoc batch evaluation of completed routes
   - Uses ObservationCapability → AnalysisCapability → ThinkingCapability

2. run_with_memory(): Waypoint-level analysis with full memory system (new)
   - Real-time sequential analysis during route traversal
   - Uses ContinuousAnalyzer → ShortTermMemory → ThinkingModule → LongTermMemory
   - Includes pHash detection, triggered reasoning, and moment curation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.capabilities import (
    ActionCapability,
    AnalysisCapability,
    LongTermMemory,
    MemoryManager,
    ObservationCapability,
    ShortTermMemory,
    ThinkingCapability,
    ThinkingModule,
    TriggerReason,
)
from src.agent.cognitive_controller import CognitiveController
from src.agent.config import AgentPersonality, get_preset
from src.pipeline import WalkingAgentPipeline
from src.config import settings
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
        self.framework_id = framework_id or "streetagent_5d"
        self.enable_memory = enable_memory
        self._cognitive_controller: Optional[CognitiveController] = None

        # Initialize state with personality weights
        self.state.preferences = personality.dimension_weights.copy()

        # Lazy-loaded capabilities (traditional)
        self._observer: Optional[ObservationCapability] = None
        self._analyzer: Optional[AnalysisCapability] = None
        self._thinker: Optional[ThinkingCapability] = None
        self._actor: Optional[ActionCapability] = None
        self._pipeline: Optional[WalkingAgentPipeline] = None

        # Lazy-loaded memory system components (new)
        self._continuous_analyzer = None
        self._short_term_memory = None
        self._thinking_module = None
        self._memory_manager = None

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
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the agent either by generating a route or from a pre-collected folder."""
        if route_folder is not None:
            if (start is not None) or (end is not None):
                raise ValueError(
                    "Provide either start/end or route_folder, but not both."
                )

            return self.run_from_folder(
                route_folder=route_folder,
                metadata_filename=metadata_filename,
                **kwargs,
            )

        if start is None or end is None:
            raise ValueError(
                "Both start and end must be provided when route_folder is not supplied."
            )

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

    # ========================================================================
    # Memory System Properties (New)
    # ========================================================================

    @property
    def continuous_analyzer(self):
        """Lazy-load continuous analyzer for waypoint-level analysis.

        Returns:
            ContinuousAnalyzer instance configured with agent's framework.
        """
        if self._continuous_analyzer is None:
            from src.analysis import ContinuousAnalyzer

            self._continuous_analyzer = ContinuousAnalyzer(
                framework_id=self.framework_id,
                context_window=3,
                adaptive_threshold=True,
            )
            self.logger.debug("ContinuousAnalyzer initialized")
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
    def thinking_module(self) -> ThinkingModule:
        """Lazy-load thinking module for waypoint-level reasoning.

        Returns:
            ThinkingModule instance configured for triggered LLM reasoning.
        """
        if self._thinking_module is None:
            self._thinking_module = ThinkingModule(
                framework_id=self.framework_id,
                distance_trigger_meters=600.0,
                score_delta_threshold=1.5,
                enable_vlm_deep_dive=False,
            )
            self.logger.debug(
                "ThinkingModule initialized", framework_id=self.framework_id
            )
        return self._thinking_module

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
            self._cognitive_controller = CognitiveController(phash_threshold=20)
            self.logger.debug("CognitiveController initialized")
        return self._cognitive_controller

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
                pitch=-5,
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

    def run_with_memory(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        interval: int = 50,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run agent with full memory system integration.

        This method provides waypoint-level analysis with the complete memory pipeline:
        1. ContinuousAnalyzer - per-waypoint VLM evaluation with pHash detection
        2. ShortTermMemory - sliding window context for recent waypoints
        3. ThinkingModule - triggered LLM-based reasoning at key moments
        4. LongTermMemory - curated key moments and pattern extraction

        This is separate from run() to maintain backward compatibility. Use this
        method when you want deep waypoint-level analysis with memory formation.

        Args:
            start: Starting GPS coordinates (lat, lon).
            end: Destination GPS coordinates (lat, lon).
            interval: Sampling interval in meters (default: 50).
            output_dir: Optional directory to save memory artifacts.

        Returns:
            Dictionary containing:
            - analysis_results: List of waypoint result dicts (System 1/2 scores)
            - thinking_results: List of ThinkingResult objects
            - long_term_memory: LongTermMemory instance
            - route_summary: RouteSummary object
            - statistics: Dict with pipeline statistics
            - route_data: Basic route information

        Example:
            ```python
            agent = WalkingAgent.from_preset("balanced", "sagai_2025")

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

        # ====================================================================
        # Phase 2: Continuous Analysis
        # ====================================================================
        self.logger.info("Phase 2: Running continuous analysis")

        analysis_results = self.continuous_analyzer.analyze_route(
            image_paths=image_paths, waypoint_metadata=metadata
        )

        # Get analysis statistics
        analysis_stats = self.continuous_analyzer.get_statistics()

        self.logger.info(
            "Continuous analysis complete",
            total_waypoints=analysis_stats["total_waypoints"],
            visual_changes=analysis_stats["visual_changes_detected"],
        )

        # ====================================================================
        # Phase 3: Thinking with Short-Term Memory
        # ====================================================================
        self.logger.info("Phase 3: Processing with short-term memory and thinking")

        thinking_results = []
        memory_manager = self.memory_manager
        waypoint_results: List[Dict[str, Any]] = []

        for analysis in analysis_results:
            should_think, reason = self.cognitive.should_trigger_thinking(
                current_image=analysis.image_path,
                waypoint=analysis,
                force=analysis.waypoint_id == 0,
            )

            trigger_reason = (
                TriggerReason.VISUAL_CHANGE
                if reason == "visual_change"
                else TriggerReason.EXCEPTIONAL_MOMENT
            )

            # Process waypoint through memory manager (attention gate + STM)
            context = memory_manager.process_waypoint(
                analysis,
                triggered=should_think,
                trigger_reason=trigger_reason,
            )

            thinking_result = None

            if context is not None:
                # Ensure route metadata includes route info
                route_meta = context.get("route_metadata", {}) or {}
                route_meta.setdefault("route_id", route_id)
                route_meta.setdefault("length_km", route_length_km)

                try:
                    thinking_result = self.thinking_module.think_waypoint(
                        waypoint_id=analysis.waypoint_id,
                        trigger_reason=context["trigger_reason"],
                        current_image_path=context["image_path"],
                        system1_scores=analysis.scores,
                        system1_reasoning=analysis.reasoning,
                        stm_context=context["stm_context"],
                        ltm_patterns=context.get("ltm_patterns"),
                        personality=self.personality,
                        route_metadata=route_meta,
                    )

                    thinking_results.append(thinking_result)

                    # Update STM with System 2 revisions for sequential context
                    memory_manager.update_with_system2_result(
                        waypoint_id=analysis.waypoint_id,
                        thinking_result=thinking_result,
                    )

                    # Generate narrative chapter using updated context
                    try:
                        narrative_context = (
                            memory_manager.episodic_ltm.get_narrative_context()
                        )
                        visual_description = thinking_result.interpretation

                        narrative_chapter = (
                            self.thinking_module.generate_narrative_chapter(
                                waypoint_id=analysis.waypoint_id,
                                visual_description=visual_description,
                                system1_scores=analysis.scores,
                                system2_scores=thinking_result.revised_scores,
                                score_adjustments=thinking_result.score_adjustments,
                                stm_context=context["stm_context"],
                                narrative_context=narrative_context,
                                personality=self.personality,
                                trigger_reason=context["trigger_reason"],
                            )
                        )

                        narrative_chapter.image_path = analysis.image_path
                        memory_manager.episodic_ltm.add_narrative_chapter(
                            narrative_chapter
                        )
                    except Exception as e:
                        self.logger.warning(f"Narrative generation failed: {e}")

                except Exception as e:
                    self.logger.warning(
                        f"Thinking failed at waypoint {analysis.waypoint_id}: {e}"
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
                    "system1_scores": analysis.scores,
                    "system1_reasoning": analysis.reasoning,
                    "system2_triggered": thinking_result is not None,
                    "system2_scores": (
                        thinking_result.revised_scores if thinking_result else None
                    ),
                    "system2_reasoning": (
                        thinking_result.revision_reasoning if thinking_result else None
                    ),
                    "score_adjustments": (
                        thinking_result.score_adjustments if thinking_result else None
                    ),
                    "thinking_interpretation": (
                        thinking_result.interpretation if thinking_result else None
                    ),
                    "thinking_significance": (
                        thinking_result.significance if thinking_result else None
                    ),
                    "memory_influence": (
                        thinking_result.memory_influence if thinking_result else None
                    ),
                }
            )

        thinking_summary = self.thinking_module.get_thinking_summary()

        self.logger.info(
            "Thinking complete",
            episodes=len(thinking_results),
            avg_confidence=thinking_summary.get("avg_confidence", 0),
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

            thinking_result = next(
                (t for t in thinking_results if t.waypoint_id == wp["waypoint_id"]),
                None,
            )
            if not thinking_result:
                continue

            ltm.add_candidate_moment(
                waypoint_id=wp["waypoint_id"],
                image_path=Path(wp["image_path"]),
                scores=wp.get("system2_scores") or wp["system1_scores"],
                summary=thinking_result.interpretation[:200],
                significance=thinking_result.significance,
                gps=wp.get("gps", (0.0, 0.0)),
                timestamp=wp.get("timestamp"),
                thinking_confidence=thinking_result.confidence,
                visual_change_detected=wp.get("system2_triggered", False),
                score_delta=None,
            )

        # Curate moments
        ltm.curate_moments(route_length_km=route_length_km)

        # Extract patterns
        analysis_dicts = [
            {
                "waypoint_id": w["waypoint_id"],
                "scores": w.get("system2_scores") or w["system1_scores"],
                "summary": "",
            }
            for w in waypoint_results
        ]
        thinking_history = [t.interpretation for t in thinking_results]

        ltm.extract_patterns(
            all_analyses=analysis_dicts, thinking_history=thinking_history
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

            # Save thinking results
            thinking_file = output_dir / "thinking_results.json"
            with open(thinking_file, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "waypoint_id": t.waypoint_id,
                            "trigger_reason": t.trigger_reason.value,
                            "interpretation": t.interpretation,
                            "significance": t.significance,
                            "confidence": t.confidence,
                        }
                        for t in thinking_results
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

            self.logger.info("Outputs saved", output_dir=str(output_dir))

        # ====================================================================
        # Build final result
        # ====================================================================
        result = {
            "analysis_results": waypoint_results,
            "thinking_results": thinking_results,
            "narrative_chapters": narrative_chapters,
            "complete_narrative": complete_narrative,
            "long_term_memory": ltm,
            "route_summary": route_summary,
            "statistics": {
                "analysis": analysis_stats,
                "thinking": thinking_summary,
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
