"""
Agent architecture for CityWalkAgent

Provides base classes and concrete implementations for intelligent
route analysis agents.

Status: Implemented (v0.1.0)
"""

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.capabilities import (
    ActionCapability,
    AnalysisCapability,
    KeyMoment,
    LongTermMemory,
    MemoryItem,
    ObservationCapability,
    RoutePattern,
    RouteSummary,
    ShortTermMemory,
    ThinkingCapability,
    ThinkingModule,
    ThinkingResult,
    TriggerReason,
)
from src.agent.cognitive_controller import CognitiveController
from src.agent.config import (
    AgentPersonality,
    create_neutral_personality,
    get_preset,
    list_presets,
)
from src.agent.walking_agent import WalkingAgent

__all__ = [
    # Base classes
    "AgentMetadata",
    "AgentState",
    "BaseAgent",
    # Walking agent
    "WalkingAgent",
    # Personality system
    "AgentPersonality",
    "create_neutral_personality",
    "get_preset",
    "list_presets",
    # Core capabilities
    "ActionCapability",
    "AnalysisCapability",
    "ObservationCapability",
    "ThinkingCapability",
    # Long-term memory system
    "LongTermMemory",
    "KeyMoment",
    "RoutePattern",
    "RouteSummary",
    # Short-term memory system
    "ShortTermMemory",
    "MemoryItem",
    # Thinking module (waypoint-level reasoning)
    "ThinkingModule",
    "TriggerReason",
    "ThinkingResult",
    # Cognitive controller
    "CognitiveController",
]

__version__ = "0.1.0"
