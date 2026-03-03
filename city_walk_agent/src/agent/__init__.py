"""
Agent architecture for CityWalkAgent

Provides base classes and concrete implementations for intelligent
route analysis agents.

Status: Implemented (v0.1.0)
"""

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.memory import (
    KeyMoment,
    LongTermMemory,
    MemoryItem,
    RoutePattern,
    RouteSummary,
    ShortTermMemory,
)
from src.agent.system2 import PersonaReasoner, ReasoningResult, TriggerReason
from src.agent.system1 import CognitiveController
from src.agent.config import (
    AgentPersonality,
    create_neutral_personality,
    get_preset,
    list_presets,
)
from src.agent.orchestrator import CityWalkAgent

__all__ = [
    # Base classes
    "AgentMetadata",
    "AgentState",
    "BaseAgent",
    # Walking agent
    "CityWalkAgent",
    # Personality system
    "AgentPersonality",
    "create_neutral_personality",
    "get_preset",
    "list_presets",
    # Long-term memory system
    "LongTermMemory",
    "KeyMoment",
    "RoutePattern",
    "RouteSummary",
    # Short-term memory system
    "ShortTermMemory",
    "MemoryItem",
    # Persona reasoner (System 2 waypoint-level reasoning)
    "PersonaReasoner",
    "TriggerReason",
    "ReasoningResult",
    # Cognitive controller
    "CognitiveController",
]

__version__ = "0.1.0"
