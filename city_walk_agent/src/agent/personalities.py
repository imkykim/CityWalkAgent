"""High-level agent personality API.

This module provides a convenient interface for working with agent personalities.
It wraps the lower-level configuration functions in src.agent.config and provides
a clean dataclass-based API.

Usage:
    # Get a preset personality
    personality = get_preset("safety", "sagai_2025")

    # List available presets
    presets = list_presets()

    # Create a neutral personality for any framework
    neutral = create_neutral_personality("custom_framework_id")
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from src.agent.config import (
    get_available_personalities,
    get_personality_for_framework,
    map_semantic_to_weights,
    normalize_weights,
)
from src.agent.constants import WEIGHT_NEUTRAL_VALUE
from src.config import load_framework


@dataclass
class AgentPersonality:
    """Represents a complete agent personality configuration.

    Attributes:
        name: Human-readable personality name.
        description: Brief explanation of personality behavior.
        dimension_weights: Mapping of dimension IDs to weight values.
        decision_thresholds: Threshold values for decision-making.
        explanation_style: Style of explanations (safety, scenic, balanced, technical).
    """

    name: str
    description: str
    dimension_weights: Dict[str, float]
    decision_thresholds: Dict[str, float]
    explanation_style: str


def get_preset(
    preset_name: str, framework_id: str, use_semantic: bool = True
) -> AgentPersonality:
    """Get a preset personality configuration for a specific framework.

    Args:
        preset_name: Personality identifier (safety, scenic, balanced, etc.).
        framework_id: Target evaluation framework identifier.
        use_semantic: When True, use semantic mapping to adapt to framework.
            When False, use framework-specific preset if available.

    Returns:
        AgentPersonality: Configured personality instance.

    Raises:
        ValueError: If preset_name is invalid or framework_id not found.

    Examples:
        >>> safety = get_preset("safety", "sagai_2025")
        >>> safety.dimension_weights
        {'safety': 2.0, 'comfort': 1.2, 'interest': 0.8, 'aesthetics': 0.6}

        >>> # Semantic mapping adapts to any framework
        >>> explorer = get_preset("explorer", "custom_framework_10dim", use_semantic=True)
    """
    # Get personality configuration
    personality_config = get_personality_for_framework(
        preset_name, framework_id, use_semantic
    )

    # Load framework to get dimension definitions
    framework = load_framework(framework_id)
    framework_dimensions = framework.get("dimensions", [])

    if not framework_dimensions:
        raise ValueError(f"Framework '{framework_id}' has no dimensions defined")

    # Extract weights based on configuration type
    if use_semantic and "semantic_config" in personality_config:
        # Map semantic configuration to actual weights
        semantic_config = personality_config["semantic_config"]
        weight_distribution = personality_config["weight_distribution"]

        dimension_weights = map_semantic_to_weights(
            semantic_config, framework_dimensions, weight_distribution
        )

        # Normalize weights so average = 1.0
        dimension_weights = normalize_weights(dimension_weights)
    else:
        # Use framework-specific weights directly
        dimension_weights = personality_config.get("weights", {})

        # Ensure all framework dimensions have weights
        framework_dim_ids = {d["id"] for d in framework_dimensions}
        for dim_id in framework_dim_ids:
            if dim_id not in dimension_weights:
                dimension_weights[dim_id] = WEIGHT_NEUTRAL_VALUE

    # Extract other personality attributes
    name = personality_config.get("name", preset_name.title())
    description = personality_config.get("description", "")
    decision_thresholds = personality_config.get("thresholds", {})
    explanation_style = personality_config.get("explanation_style", "balanced")

    return AgentPersonality(
        name=name,
        description=description,
        dimension_weights=dimension_weights,
        decision_thresholds=decision_thresholds,
        explanation_style=explanation_style,
    )


def list_presets() -> List[str]:
    """Return the list of available personality preset names.

    Returns:
        List[str]: Personality identifiers that can be used with get_preset().

    Examples:
        >>> list_presets()
        ['safety', 'scenic', 'balanced', 'comfort', 'explorer', 'technical']
    """
    return get_available_personalities()


def create_neutral_personality(framework_id: str) -> AgentPersonality:
    """Create a balanced personality with equal weights for all dimensions.

    This is useful for creating a baseline agent or for frameworks where
    no preset personalities are defined.

    Args:
        framework_id: Target evaluation framework identifier.

    Returns:
        AgentPersonality: Neutral personality with all weights = 1.0.

    Raises:
        ValueError: If framework_id not found.

    Examples:
        >>> neutral = create_neutral_personality("sagai_2025")
        >>> neutral.dimension_weights
        {'safety': 1.0, 'comfort': 1.0, 'interest': 1.0, 'aesthetics': 1.0}
    """
    # Load framework to get dimension definitions
    framework = load_framework(framework_id)
    framework_dimensions = framework.get("dimensions", [])

    if not framework_dimensions:
        raise ValueError(f"Framework '{framework_id}' has no dimensions defined")

    # Create equal weights for all dimensions
    dimension_weights = {
        dim["id"]: WEIGHT_NEUTRAL_VALUE for dim in framework_dimensions
    }

    # Use default thresholds (framework-agnostic)
    decision_thresholds = {
        "min_overall_score": 6.5,
        "max_volatility": 2.0,
        "max_barriers": 2,
    }

    return AgentPersonality(
        name="Neutral Observer",
        description="Balanced evaluation across all dimensions",
        dimension_weights=dimension_weights,
        decision_thresholds=decision_thresholds,
        explanation_style="balanced",
    )
