"""
Foundational agent architecture for CityWalkAgent route analysis.

This module implements a hierarchical agent pattern inspired by the VIRL framework,
adapted for route evaluation and analysis tasks. The architecture follows the
Platform → Agent → Capability pattern observed in VIRL.

Key architectural patterns from VIRL:
- AgentTemplate: Maintains state (status, intention, background)
- Platform: Provides external capabilities (APIs, navigation, perception)
- TaskTemplate: Orchestrates agent + platform + memory
- Memory: Stores and retrieves experiences

CityWalkAgent adaptation:
- AgentState: Tracks route evaluation state and learned preferences
- AgentMetadata: Immutable agent identity and configuration
- BaseAgent: Abstract agent with perceive → reason → act → remember workflow
- Pipeline integration: Uses existing WalkingAgentPipeline as the "platform"

The key difference from direct pipeline usage is the cognitive loop:
instead of directly calling evaluation functions, agents perceive route data,
reason about it using learned preferences, act on decisions, and remember
experiences to improve future evaluations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.utils.logging import get_logger


@dataclass
class AgentState:
    """
    Agent state tracking for route evaluation and navigation.

    Inspired by VIRL's AgentTemplate status dictionary, but structured
    for route analysis tasks. Tracks spatial state, evaluation history,
    and learned preferences using exponential moving averages.

    Attributes:
        current_location: Agent's current GPS coordinates (lat, lon)
        destination: Target destination coordinates (lat, lon)
        evaluated_routes: List of route IDs already evaluated
        preferences: Learned dimension weights (dimension_id -> weight)
            Updated via exponential moving average from feedback
        memory_count: Number of experiences stored in memory
        last_update: Timestamp of last state modification
    """

    current_location: Optional[Tuple[float, float]] = None
    destination: Optional[Tuple[float, float]] = None
    evaluated_routes: List[str] = field(default_factory=list)
    preferences: Dict[str, float] = field(default_factory=dict)
    memory_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    def update_preferences(self, feedback: Dict[str, float]) -> None:
        """
        Update preference weights using exponential moving average.

        Similar to how VIRL agents update their status dictionary, but uses
        EMA to smoothly incorporate feedback while retaining learned patterns.

        Formula: new_weight = 0.7 * old_weight + 0.3 * feedback_weight

        Args:
            feedback: Dictionary of dimension_id -> feedback_weight
                Higher weights indicate stronger preference for that dimension

        Example:
            >>> state = AgentState(preferences={"safety": 5.0})
            >>> state.update_preferences({"safety": 8.0, "comfort": 6.0})
            >>> state.preferences["safety"]  # 0.7 * 5.0 + 0.3 * 8.0 = 5.9
            >>> state.preferences["comfort"]  # 0.0 * 0.7 + 6.0 * 0.3 = 1.8
        """
        alpha = 0.3  # Weight for new feedback (higher = faster adaptation)

        for dimension, feedback_value in feedback.items():
            old_value = self.preferences.get(dimension, 0.0)
            new_value = (1 - alpha) * old_value + alpha * feedback_value
            self.preferences[dimension] = new_value

        self.last_update = datetime.now()


@dataclass
class AgentMetadata:
    """
    Immutable agent identity and configuration.

    Inspired by VIRL's agent initialization pattern (name, background, intention),
    but structured as immutable metadata for agent tracking and logging.

    Unlike VIRL's mutable intention, CityWalkAgent uses primary_goal as a stable
    identifier, with runtime goals handled through the state object.

    Attributes:
        agent_id: Unique identifier for this agent instance
        name: Human-readable agent name
        description: Brief description of agent's purpose
        primary_goal: Main objective (e.g., "evaluate_route", "optimize_path")
        constraints: Configuration constraints (max_routes, time_budget, etc.)
        created_at: Agent creation timestamp
    """

    agent_id: str
    name: str
    description: str
    primary_goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    Abstract base agent implementing the perceive → reason → act → remember cycle.

    Architecture pattern inspired by VIRL's hierarchical design:
    - Platform (WalkingAgentPipeline): Provides route generation, image collection,
      and VLM evaluation capabilities
    - Agent (this class): Cognitive loop for perception, reasoning, and action
    - Memory: Stores experiences for learning and retrieval

    Key differences from VIRL:
    1. VIRL uses Platform for external APIs (Google Maps, Street View)
       CityWalkAgent uses Pipeline for route evaluation

    2. VIRL's perceive/act focuses on visual navigation and object detection
       CityWalkAgent's perceive/act focuses on route quality assessment

    3. VIRL tasks orchestrate agent + platform + chatbot
       CityWalkAgent agents orchestrate pipeline + memory + preferences

    Subclasses must implement:
    - perceive(): Process route data into structured perceptions
    - reason(): Make decisions based on perceptions and preferences
    - act(): Execute decisions and return results

    The run() method implements the full workflow:
    1. Update state (location, destination)
    2. Get route data via _get_route_data()
    3. perceive(route_data) → perception
    4. reason(perception) → decision
    5. act(decision) → result
    6. remember(experience)
    7. Return complete response
    """

    def __init__(
        self,
        metadata: AgentMetadata,
        initial_state: Optional[AgentState] = None
    ) -> None:
        """
        Initialize base agent with metadata and state.

        Uses lazy loading for heavy components (_pipeline, _memory) following
        the VIRL pattern of initializing capabilities on-demand.

        Args:
            metadata: Immutable agent configuration
            initial_state: Optional starting state (creates fresh if None)
        """
        self.metadata = metadata
        self.state = initial_state or AgentState()
        self.logger = get_logger(f"agent.{metadata.agent_id}")

        # Lazy-loaded components (initialized on first access)
        self._pipeline = None
        self._memory: List[Dict[str, Any]] = []

        self.logger.info(
            "Agent initialized",
            agent_id=metadata.agent_id,
            name=metadata.name,
            goal=metadata.primary_goal
        )

    @abstractmethod
    def perceive(self, route_data: Any) -> Dict[str, Any]:
        """
        Process raw route data into structured perceptions.

        Inspired by VIRL's perception modules (detector, recognizer, mm_llm),
        but adapted for route analysis instead of visual navigation.

        VIRL example: Perceive visual features, detect objects, recognize text
        CityWalkAgent: Perceive route segments, extract dimension scores,
                      identify barriers or points of interest

        Args:
            route_data: Raw route information (coordinates, images, metadata)

        Returns:
            Structured perception dictionary with processed features
            Example: {
                "segments": [...],
                "dimension_scores": {...},
                "barriers": [...],
                "total_distance": 1500.0
            }

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @abstractmethod
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on perceptions and agent preferences.

        Similar to VIRL's chatbot-based reasoning for navigation decisions,
        but uses learned preferences and evaluation criteria instead of LLM.

        VIRL example: Reason about navigation direction based on visual cues
        CityWalkAgent: Reason about route quality based on dimension scores
                      and learned preferences

        Args:
            perception: Structured perception from perceive()

        Returns:
            Decision dictionary with action plan
            Example: {
                "evaluation": "accept",
                "recommended_adjustments": [...],
                "confidence": 0.85
            }

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Any:
        """
        Execute decisions and return action results.

        Analogous to VIRL's navigation actions (move, check_surrounding),
        but focused on evaluation actions rather than physical movement.

        VIRL example: Navigate to coordinates, capture street view images
        CityWalkAgent: Generate evaluation report, suggest route modifications,
                      trigger re-evaluation with different parameters

        Args:
            decision: Decision from reason()

        Returns:
            Action result (type depends on action)
            Could be evaluation report, modified route, or status update

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    def remember(self, experience: Dict[str, Any]) -> None:
        """
        Store experience in memory for learning and retrieval.

        Inspired by VIRL's Memory class that stores views by geocode,
        adapted to store route evaluation experiences.

        VIRL stores: obj_id → [view1, view2, ...] with geocode-based retrieval
        CityWalkAgent stores: route evaluations with preference-based learning

        Future enhancement: Could implement geocode-based retrieval like VIRL
        for retrieving similar routes by spatial proximity.

        Args:
            experience: Dictionary containing:
                - route_id: Route identifier
                - perception: What was perceived
                - decision: What was decided
                - result: What happened
                - feedback: Optional user/system feedback
        """
        experience["timestamp"] = datetime.now().isoformat()
        experience["agent_id"] = self.metadata.agent_id
        self._memory.append(experience)
        self.state.memory_count = len(self._memory)

        # Update preferences if feedback is provided
        if "feedback" in experience and isinstance(experience["feedback"], dict):
            self.state.update_preferences(experience["feedback"])

        self.logger.debug(
            "Experience stored",
            route_id=experience.get("route_id"),
            memory_count=self.state.memory_count
        )

    def run(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete agent workflow for route evaluation.

        This method implements the core cognitive loop inspired by VIRL's
        task execution pattern, but structured for route analysis.

        VIRL workflow (RouteOptimizer example):
        1. Initialize navigator with platform
        2. Loop: navigate() → check surroundings → update memory
        3. Visualize trajectory

        CityWalkAgent workflow:
        1. Update agent state (location, destination)
        2. Get route data using _get_route_data()
        3. perceive(route_data) → perception
        4. reason(perception) → decision
        5. act(decision) → result
        6. remember(experience)
        7. Return complete response

        Args:
            start: Starting coordinates (latitude, longitude)
            end: Destination coordinates (latitude, longitude)
            **kwargs: Additional parameters for route generation/evaluation
                - framework_id: Evaluation framework to use
                - sampling_interval: Distance between image samples
                - route_id: Reuse existing route instead of generating new

        Returns:
            Complete response dictionary with:
                - route_id: Generated or provided route identifier
                - perception: Structured perception output
                - decision: Reasoning output
                - result: Action result
                - state: Current agent state snapshot
                - metadata: Agent metadata

        Example:
            >>> agent = MyAgent(metadata)
            >>> response = agent.run(
            ...     start=(40.7128, -74.0060),  # New York
            ...     end=(40.7580, -73.9855),    # Times Square
            ...     framework_id="walkability"
            ... )
            >>> print(response["decision"]["evaluation"])
        """
        # Step 1: Update agent state
        self.state.current_location = start
        self.state.destination = end
        self.state.last_update = datetime.now()

        self.logger.info(
            "Starting agent run",
            start=start,
            end=end,
            kwargs=kwargs
        )

        # Step 2: Get route data
        route_data = self._get_route_data(start, end, **kwargs)
        route_id = route_data.get("route_id", "unknown")

        if route_id not in self.state.evaluated_routes:
            self.state.evaluated_routes.append(route_id)

        # Step 3: Perceive
        self.logger.debug("Perceiving route data", route_id=route_id)
        perception = self.perceive(route_data)

        # Step 4: Reason
        self.logger.debug("Reasoning about perception", route_id=route_id)
        decision = self.reason(perception)

        # Step 5: Act
        self.logger.debug("Executing decision", route_id=route_id)
        result = self.act(decision)

        # Step 6: Remember
        experience = {
            "route_id": route_id,
            "start": start,
            "end": end,
            "perception": perception,
            "decision": decision,
            "result": result,
            "feedback": kwargs.get("feedback")  # Optional user feedback
        }
        self.remember(experience)

        # Step 7: Return complete response
        response = {
            "route_id": route_id,
            "perception": perception,
            "decision": decision,
            "result": result,
            "state": {
                "current_location": self.state.current_location,
                "destination": self.state.destination,
                "evaluated_routes": self.state.evaluated_routes,
                "preferences": self.state.preferences,
                "memory_count": self.state.memory_count
            },
            "metadata": {
                "agent_id": self.metadata.agent_id,
                "name": self.metadata.name,
                "goal": self.metadata.primary_goal
            }
        }

        self.logger.info(
            "Agent run completed",
            route_id=route_id,
            memory_count=self.state.memory_count
        )

        return response

    def _get_route_data(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get route data using the pipeline (lazy-loaded).

        This method serves as the "platform" interface, similar to how VIRL
        agents use Platform.get_streetview_from_geocode() or Platform.get_route().

        Override this method in subclasses to customize route data retrieval
        or use alternative data sources (cached routes, external APIs, etc.).

        Args:
            start: Starting coordinates
            end: Destination coordinates
            **kwargs: Additional parameters

        Returns:
            Route data dictionary with at minimum:
                - route_id: Unique identifier
                - coordinates: List of waypoints
                - metadata: Any additional route information
        """
        # Placeholder implementation - subclasses should override
        # In production, would use WalkingAgentPipeline
        self.logger.warning(
            "Using placeholder _get_route_data - override in subclass",
            start=start,
            end=end
        )

        return {
            "route_id": f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start": start,
            "end": end,
            "coordinates": [start, end],
            "metadata": kwargs
        }

    def get_state(self) -> AgentState:
        """
        Get current agent state.

        Returns:
            Current AgentState instance
        """
        return self.state

    def update_goal(self, new_goal: str) -> None:
        """
        Update agent's primary goal.

        Similar to VIRL's AgentTemplate.update_intention(), but modifies
        the metadata goal rather than maintaining separate intention.

        Args:
            new_goal: New primary goal string
        """
        old_goal = self.metadata.primary_goal
        self.metadata.primary_goal = new_goal
        self.state.last_update = datetime.now()

        self.logger.info(
            "Goal updated",
            old_goal=old_goal,
            new_goal=new_goal
        )
