"""WalkingAgent - orchestrates all capabilities for route analysis.

This is the main concrete agent implementation that uses ObservationCapability,
ThinkingCapability, and AgentMemory to analyze walking routes with personality-driven
decision making.
"""

from typing import Any, Dict, Optional, Tuple

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.capabilities import (
    AgentMemory,
    ObservationCapability,
    ThinkingCapability,
)
from src.agent.personalities import AgentPersonality, get_preset
from src.analysis import SequentialAnalyzer
from src.pipeline import WalkingAgentPipeline
from src.utils.logging import get_logger


class WalkingAgent(BaseAgent):
    """Walking route analysis agent with personality-driven decision making.

    This agent orchestrates the full pipeline:
    1. Observe: Use VLM to evaluate route images
    2. Think: Apply personality weights and make decisions
    3. Act: Generate formatted recommendations
    4. Remember: Store experiences for learning

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
        self._thinker: Optional[ThinkingCapability] = None
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

    @property
    def observer(self) -> ObservationCapability:
        """Lazy-load observation capability."""
        if self._observer is None:
            self._observer = ObservationCapability(framework_id=self.framework_id)
            self.logger.debug("ObservationCapability initialized")
        return self._observer

    @property
    def thinker(self) -> ThinkingCapability:
        """Lazy-load thinking capability."""
        if self._thinker is None:
            self._thinker = ThinkingCapability()
            self.logger.debug("ThinkingCapability initialized")
        return self._thinker

    @property
    def pipeline(self) -> WalkingAgentPipeline:
        """Lazy-load pipeline for route data."""
        if self._pipeline is None:
            self._pipeline = WalkingAgentPipeline(framework_id=self.framework_id)
            self.logger.debug("Pipeline initialized")
        return self._pipeline

    def perceive(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """Observe route using VLM evaluation.

        Args:
            route_data: Route information with image_paths, route_id, etc.

        Returns:
            Observation dict with dimension scores and statistics.
        """
        self.logger.debug("Perceiving route", route_id=route_data.get("route_id"))

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

        # Get sequential analysis from route evaluations
        raw_evaluations = perception.get("raw_evaluations", [])

        # Create a simple sequential analysis
        # In a full implementation, this would use SequentialAnalyzer
        # For now, calculate basic stats
        sequential_analysis = self._analyze_sequence(raw_evaluations, perception)

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
        """Format decision into actionable output.

        Args:
            decision: Decision dict from reason().

        Returns:
            Action dict with formatted message and details.
        """
        recommendation = decision.get("recommendation", "unknown")
        confidence = decision.get("confidence", 0.0)
        explanation = decision.get("explanation", "")

        # Format message based on recommendation
        if recommendation == "accept":
            action_type = "approve"
            icon = "✅"
        else:
            action_type = "reject"
            icon = "⚠️"

        message = f"{icon} {explanation}"

        # Build action result
        result = {
            "action_type": action_type,
            "message": message,
            "confidence": confidence,
            "recommendation": recommendation,
            "concerns": decision.get("concerns", []),
            "highlights": decision.get("highlights", []),
            "weighted_score": decision.get("weighted_score", 0.0),
        }

        self.logger.debug("Action formatted", action_type=action_type)

        return result

    def _get_route_data(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        **kwargs
    ) -> Dict[str, Any]:
        """Get route data using the pipeline.

        Args:
            start: Starting coordinate (lat, lon).
            end: Destination coordinate (lat, lon).
            **kwargs: Additional options (route_id, interval, etc.).

        Returns:
            Route data dict with route_id, image_paths, waypoints, etc.
        """
        # Generate or load route
        route_id = kwargs.get("route_id")
        interval = kwargs.get("interval", 50)

        if route_id is None:
            # Generate new route
            self.logger.debug("Generating new route", start=start, end=end)
            route = self.pipeline.route_generator.generate_route(
                start=start,
                end=end,
                interval=interval,
            )
            route_id = route.route_id
        else:
            # Load existing route
            self.logger.debug("Loading existing route", route_id=route_id)
            route = self.pipeline.route_generator.load_route(route_id)

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
        }

        self.logger.debug(
            "Route data prepared",
            route_id=route_id,
            num_images=len(image_paths),
        )

        return route_data

    def _analyze_sequence(
        self,
        evaluations: list,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create sequential analysis from evaluations.

        Args:
            evaluations: Raw evaluation results.
            observation: Observation dict with stats.

        Returns:
            Sequential analysis dict with volatility, barriers, pattern.
        """
        dimension_stats = observation.get("dimension_stats", {})

        # Calculate overall volatility (average of dimension std devs)
        volatilities = [
            stats.get("std", 0.0)
            for stats in dimension_stats.values()
        ]
        overall_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.0

        # Simple barrier detection: check for large drops
        hidden_barriers = []
        for dim_id, stats in dimension_stats.items():
            if stats.get("min", 10) < 5.0:  # Very low score
                hidden_barriers.append({
                    "dimension": dim_id,
                    "severity": "high",
                    "min_score": stats.get("min", 0.0),
                })

        # Classify pattern based on volatility
        if overall_volatility < 1.0:
            pattern_type = "stable"
        elif overall_volatility < 2.0:
            pattern_type = "moderate"
        else:
            pattern_type = "volatile"

        return {
            "volatility": overall_volatility,
            "hidden_barriers": hidden_barriers,
            "pattern_type": pattern_type,
            "dimension_volatilities": {
                dim: stats.get("std", 0.0)
                for dim, stats in dimension_stats.items()
            },
        }
