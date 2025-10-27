"""
Placeholder base classes for agent architecture

These are skeleton implementations to define the future agent interface.
DO NOT implement actual functionality here - this is for architectural planning only.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentState:
    """
    Agent state representation

    Future implementation will track:
    - Current location and destination
    - Active route and navigation progress
    - Memory and learned patterns
    - User preferences
    - Analysis history
    """
    current_location: Optional[Tuple[float, float]] = None
    destination: Optional[Tuple[float, float]] = None
    active_route_id: Optional[str] = None
    memory: Dict[str, Any] = None
    preferences: Dict[str, float] = None
    last_update: datetime = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = {}
        if self.preferences is None:
            self.preferences = {}
        if self.last_update is None:
            self.last_update = datetime.now()


class BaseAgent(ABC):
    """
    Abstract base class for walking agents

    Future implementations will provide:
    - Route planning with experience optimization
    - Adaptive sampling strategies
    - Preference learning
    - Explanation generation
    """

    @abstractmethod
    def plan_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Any:  # Will return Route object
        """
        Plan optimal walking route

        Args:
            start: (lat, lon) start coordinates
            end: (lat, lon) end coordinates
            constraints: Optional planning constraints
                - max_distance: Maximum route length
                - min_quality: Minimum acceptable quality score
                - avoid_areas: Areas to avoid
                - preferred_dimensions: Which dimensions to prioritize

        Returns:
            Planned Route object

        Future implementation will:
        1. Query existing routes from memory
        2. Predict experience quality using learned patterns
        3. Optimize for user preferences
        4. Generate route with adaptive sampling intervals
        """
        raise NotImplementedError("Agent architecture not yet implemented")

    @abstractmethod
    def evaluate_route(self, route: Any) -> Any:  # Route -> RouteAnalysis
        """
        Evaluate route quality

        Args:
            route: Route object to evaluate

        Returns:
            RouteAnalysis object

        Future implementation will:
        1. Trigger VLM evaluation if needed
        2. Run sequential analysis
        3. Store experience in memory
        4. Update learned patterns
        """
        raise NotImplementedError("Agent architecture not yet implemented")

    @abstractmethod
    def explain_decision(self, decision: Any) -> str:
        """
        Generate human-readable explanation for decision

        Args:
            decision: Decision object (route choice, recommendation, etc.)

        Returns:
            Natural language explanation string

        Future implementation will:
        1. Extract key factors from analysis
        2. Generate explanation template
        3. Translate to user's language
        4. Include relevant metrics and insights
        """
        raise NotImplementedError("Agent architecture not yet implemented")

    @abstractmethod
    def update_preferences(
        self,
        feedback: Dict[str, float]
    ) -> None:
        """
        Update user preferences based on feedback

        Args:
            feedback: Dict mapping dimension to preference weight
                Example: {"safety": 1.5, "aesthetics": 0.8}

        Future implementation will:
        1. Incorporate user feedback
        2. Adjust dimension weights
        3. Update scoring functions
        4. Retrain if needed
        """
        raise NotImplementedError("Agent architecture not yet implemented")


class AgentMemory(ABC):
    """
    Abstract base class for agent memory system

    Future implementations will provide:
    - Experience storage and retrieval
    - Pattern recognition
    - Similarity matching
    - Knowledge transfer
    """

    @abstractmethod
    def store_experience(
        self,
        route: Any,
        analysis: Any
    ) -> None:
        """
        Store route experience in memory

        Args:
            route: Route object
            analysis: RouteAnalysis object
        """
        raise NotImplementedError("Agent memory not yet implemented")

    @abstractmethod
    def retrieve_similar_routes(
        self,
        route: Any,
        k: int = 5
    ) -> List[Tuple[Any, Any]]:
        """
        Retrieve similar routes from memory

        Args:
            route: Query route
            k: Number of similar routes to retrieve

        Returns:
            List of (Route, RouteAnalysis) tuples
        """
        raise NotImplementedError("Agent memory not yet implemented")

    @abstractmethod
    def learn_patterns(self) -> Dict[str, Any]:
        """
        Learn patterns from stored experiences

        Returns:
            Dict with learned patterns and statistics
        """
        raise NotImplementedError("Agent memory not yet implemented")


class DecisionMaker(ABC):
    """
    Abstract base class for agent decision-making

    Future implementations will provide:
    - Multi-criteria optimization
    - Trade-off analysis
    - Risk assessment
    - Preference integration
    """

    @abstractmethod
    def choose_route(
        self,
        candidate_routes: List[Any],
        preferences: Dict[str, float]
    ) -> Any:
        """
        Choose best route from candidates

        Args:
            candidate_routes: List of Route objects
            preferences: User preferences

        Returns:
            Best Route object
        """
        raise NotImplementedError("Decision making not yet implemented")

    @abstractmethod
    def recommend_interventions(
        self,
        analysis: Any
    ) -> List[Dict[str, Any]]:
        """
        Recommend urban interventions based on analysis

        Args:
            analysis: RouteAnalysis object

        Returns:
            List of intervention recommendations
        """
        raise NotImplementedError("Decision making not yet implemented")
