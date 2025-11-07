"""Core abstractions for CityWalkAgent's cognitive agent loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.agent.capabilities import LongTermMemory


@dataclass
class AgentState:
    """Track position, evaluation history, and learned preferences.

    Attributes:
        current_location: Latest GPS coordinates `(lat, lon)`.
        destination: Target GPS coordinates `(lat, lon)`.
        evaluated_routes: Route identifiers the agent has inspected.
        preferences: Learned dimension weights keyed by dimension id.
        memory_count: Number of stored experiences.
        last_update: Timestamp of the last state mutation.
        waypoint_index: Current waypoint position (0-indexed).
        total_waypoints: Total number of waypoints in current route.
    """

    current_location: Optional[Tuple[float, float]] = None
    destination: Optional[Tuple[float, float]] = None
    evaluated_routes: List[str] = field(default_factory=list)
    preferences: Dict[str, float] = field(default_factory=dict)
    memory_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    waypoint_index: int = 0
    total_waypoints: int = 0

    def update_preferences(self, feedback: Dict[str, float]) -> None:
        """Blend new feedback into preferences via exponential smoothing.

        Formula:
            new_weight = 0.7 * old_weight + 0.3 * feedback_weight

        Args:
            feedback: Mapping of `dimension_id -> feedback_weight`.
        """
        alpha = 0.3  # Weight for new feedback (higher = faster adaptation)

        for dimension, feedback_value in feedback.items():
            old_value = self.preferences.get(dimension, 0.0)
            new_value = (1 - alpha) * old_value + alpha * feedback_value
            self.preferences[dimension] = new_value

        self.last_update = datetime.now()


@dataclass
class AgentMetadata:
    """Immutable identity, description, goal, and constraint metadata.

    Attributes:
        agent_id: Unique identifier for this agent.
        name: Human-readable agent label.
        description: Short synopsis of agent responsibilities.
        primary_goal: Canonical objective phrase.
        constraints: Optional runtime limits (budgets, caps, etc.).
        created_at: Timestamp of agent creation.
    """

    agent_id: str
    name: str
    description: str
    primary_goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """Abstract perceive → reason → act → remember loop for routes."""

    def __init__(
        self,
        metadata: AgentMetadata,
        initial_state: Optional[AgentState] = None
    ) -> None:
        """Attach metadata, seed optional state, and prep lazy components.

        Args:
            metadata: Static configuration describing the agent.
            initial_state: Pre-populated state; if omitted a fresh one is created.
        """
        self.metadata = metadata
        self.state = initial_state or AgentState()
        self.logger = get_logger(f"agent.{metadata.agent_id}")

        # Lazy-loaded components (initialized on first access)
        self._pipeline = None
        self._memory_system: Optional[LongTermMemory] = None

        self.logger.info(
            "Agent initialized",
            agent_id=metadata.agent_id,
            name=metadata.name,
            goal=metadata.primary_goal
        )

    @property
    def memory(self) -> LongTermMemory:
        """
        Lazy-load persistent memory system.

        Inspired by VIRL's pattern of initializing heavy components on demand.
        The LongTermMemory provides JSONL-based persistent storage for experiences,
        replacing the simple in-memory list.

        Returns:
            LongTermMemory instance for this agent
        """
        if self._memory_system is None:
            from src.agent.capabilities import LongTermMemory
            self._memory_system = LongTermMemory(self.metadata.agent_id)
            self.logger.debug("Memory system initialized")
        return self._memory_system

    @abstractmethod
    def perceive(self, route_data: Any) -> Dict[str, Any]:
        """Transform route data into structured perception information.

        Extract structured observations from raw route data including
        segments, dimension scores, barriers, and distances.

        Args:
            route_data: Raw route information (coordinates, images, metadata).

        Returns:
            Dict[str, Any]: Structured perception containing segments,
                dimension_scores, barriers, total_distance, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Derive an action plan from perception and preferences.

        Args:
            perception: Structured signals generated by `perceive`.

        Returns:
            Dict[str, Any]: Decision payload ready for `act`.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Any:
        """Carry out the decision and return the resulting payload.

        Args:
            decision: Action plan emitted by `reason`.

        Returns:
            Any: Result of executing the decision (implementation specific).
        """
        raise NotImplementedError

    def remember(self, experience: Dict[str, Any]) -> None:
        """Persist experience to disk, update counters, and learn from feedback.

        Uses the persistent LongTermMemory system to store experiences in JSONL format,
        similar to VIRL's Memory.add() but with append-only file storage instead of
        in-memory dict + pickle checkpoint.

        Args:
            experience: Collected data spanning perception, decision, result, and
                optional feedback metadata. Must contain 'route_id'.

        Side effects:
            - Stores experience to disk via LongTermMemory
            - Updates state.memory_count
            - Updates state.preferences if feedback provided
        """
        # Store experience using persistent memory system
        self.memory.store(experience)

        # Update state counter from memory statistics
        stats = self.memory.get_statistics()
        self.state.memory_count = stats["total_experiences"]

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
        """Execute the full loop and return decision, result, and state.

        Args:
            start: Starting coordinate `(lat, lon)`.
            end: Destination coordinate `(lat, lon)`.
            **kwargs: Optional overrides (e.g., framework_id, sampling_interval,
                route_id, feedback).

        Returns:
            Dict[str, Any]: Aggregated response containing route id, perception,
            decision, action result, state snapshot, and metadata.
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

        # Update waypoint tracking from route data
        waypoints = route_data.get("waypoints", [])
        self.state.total_waypoints = len(waypoints)
        self.state.waypoint_index = 0  # Reset to start

        self.logger.debug(
            "Route waypoints loaded",
            route_id=route_id,
            total_waypoints=self.state.total_waypoints
        )

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
                "memory_count": self.state.memory_count,
                "waypoint_index": self.state.waypoint_index,
                "total_waypoints": self.state.total_waypoints,
                "progress_percent": (
                    (self.state.waypoint_index / self.state.total_waypoints * 100)
                    if self.state.total_waypoints > 0 else 0.0
                )
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
        """Fetch route data; subclasses should provide real implementations.

        Args:
            start: Starting coordinate `(lat, lon)`.
            end: Destination coordinate `(lat, lon)`.
            **kwargs: Implementation-specific options.

        Returns:
            Dict[str, Any]: Route description containing identifiers, coordinates,
            and metadata.
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
        """Return the live AgentState object.

        Returns:
            AgentState: Mutable state reference for the agent.
        """
        return self.state

    def update_goal(self, new_goal: str) -> None:
        """Set a new primary goal and refresh timestamps/logs.

        Args:
            new_goal: Replacement value for `metadata.primary_goal`.
        """
        old_goal = self.metadata.primary_goal
        self.metadata.primary_goal = new_goal
        self.state.last_update = datetime.now()

        self.logger.info(
            "Goal updated",
            old_goal=old_goal,
            new_goal=new_goal
        )
