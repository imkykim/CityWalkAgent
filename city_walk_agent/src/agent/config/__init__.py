"""Agent configuration package.

Public API:
    from src.agent.config import get_preset, AgentPersonality

    personality = get_preset("homebuyer", "sagai_2025")
    agent = WalkingAgent(personality=personality)
"""

# All imports now come from personalities.py
from src.agent.config.personalities import (
    AgentPersonality,
    EnhancedPersonalityConfig,
    ENHANCED_PERSONALITIES,
    PERSONALITIES,
    create_neutral_personality,
    get_enhanced_personality,
    get_preset,
    list_enhanced_personalities,
    list_presets,
)

# Import constants
from src.agent.config.constants import (
    DEFAULT_FRAMEWORK_ID,
)

__all__ = [
    # Primary API
    "AgentPersonality",
    "get_preset",
    "list_presets",
    "create_neutral_personality",
    # Enhanced personality access
    "EnhancedPersonalityConfig",
    "get_enhanced_personality",
    "list_enhanced_personalities",
    "ENHANCED_PERSONALITIES",
    "PERSONALITIES",  # Alias for backward compatibility
    # Constants
    "DEFAULT_FRAMEWORK_ID",
]
