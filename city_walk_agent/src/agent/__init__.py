"""
Agent architecture for CityWalkAgent

Provides base classes and concrete implementations for intelligent
route analysis agents.

Status: Implemented (v0.1.0)
"""

from src.agent.base import AgentMetadata, AgentState, BaseAgent
from src.agent.capabilities import (
    AgentMemory,
    ObservationCapability,
    ThinkingCapability,
)
from src.agent.personalities import (
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
    # Capabilities
    "AgentMemory",
    "ObservationCapability",
    "ThinkingCapability",
]

__version__ = "0.1.0"
