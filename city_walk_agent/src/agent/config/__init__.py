"""Agent configuration package.

This package consolidates all agent configuration functionality:
- Personality definitions and presets
- Semantic-to-framework weight mapping
- Configuration validation utilities
- Agent constants

Users should import from this package rather than submodules:
    from src.agent.config import get_preset, AgentPersonality

The implementation details (whether functions come from personalities.py,
constants.py, etc.) are hidden from users.
"""

# Import personality configuration functions and classes
from src.agent.config.personalities import (
    AgentPersonality,
    create_neutral_personality,
    get_available_frameworks_for_personality,
    get_available_personalities,
    get_personality_for_framework,
    get_preset,
    get_primary_dimensions,
    list_presets,
    map_semantic_to_weights,
    normalize_weights,
    validate_dimension_weights,
)

# Import constants (make them accessible through the package)
from src.agent.config.constants import (
    FRAMEWORK_SPECIFIC_PERSONALITIES,
    MAX_WEIGHT,
    MIN_WEIGHT,
    SEMANTIC_PERSONALITIES,
    WEIGHT_NEUTRAL_VALUE,
    WEIGHT_THRESHOLD_PRIMARY,
)

__all__ = [
    # High-level personality API
    "AgentPersonality",
    "get_preset",
    "list_presets",
    "create_neutral_personality",
    # Configuration helper functions
    "get_personality_for_framework",
    "validate_dimension_weights",
    "map_semantic_to_weights",
    "normalize_weights",
    "get_primary_dimensions",
    "get_available_personalities",
    "get_available_frameworks_for_personality",
    # Constants
    "SEMANTIC_PERSONALITIES",
    "FRAMEWORK_SPECIFIC_PERSONALITIES",
    "WEIGHT_NEUTRAL_VALUE",
    "WEIGHT_THRESHOLD_PRIMARY",
    "MIN_WEIGHT",
    "MAX_WEIGHT",
]
