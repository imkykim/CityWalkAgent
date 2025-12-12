"""Agent personality configuration and mapping functions.

This module provides both low-level utility functions for working with
personality configurations and a high-level dataclass-based API.

The module is organized into two main sections:
1. Configuration Helper Functions - for semantic mapping and validation
2. High-Level Personality API - AgentPersonality dataclass and presets

Usage:
    # Get a preset personality
    personality = get_preset("safety", "sagai_2025")

    # List available presets
    presets = list_presets()

    # Create a neutral personality for any framework
    neutral = create_neutral_personality("custom_framework_id")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agent.config.constants import (
    FRAMEWORK_SPECIFIC_PERSONALITIES,
    MAX_WEIGHT,
    MIN_WEIGHT,
    PERSONALITY_ENHANCEMENT_MAP,
    SEMANTIC_PERSONALITIES,
    WEIGHT_NEUTRAL_VALUE,
    WEIGHT_THRESHOLD_PRIMARY,
)
from src.agent.config.enhanced_personalities import (
    get_enhanced_personality,
    list_enhanced_personalities,
)
from src.config import load_framework


def _resolve_enhanced_personality_id(preset_name: str) -> Optional[str]:
    """Return enhanced personality id for a given preset name if available."""
    normalized = preset_name.lower()

    # Direct enhanced personality
    if normalized in list_enhanced_personalities():
        return normalized

    # Legacy preset mapped to enhanced personality
    mapped = PERSONALITY_ENHANCEMENT_MAP.get(normalized)
    if mapped:
        return mapped

    return None


# ============================================================================
# Configuration Helper Functions
# ============================================================================


def get_personality_for_framework(
    personality_type: str, framework_id: str, use_semantic: bool = True
) -> Dict[str, Any]:
    """Return a framework-aware personality configuration.

    Args:
        personality_type: Personality key such as `safety` or `balanced`.
        framework_id: Target framework identifier.
        use_semantic: When True, return the semantic personality template;
            otherwise return the framework-specific preset.

    Returns:
        Dict[str, Any]: Personality definition including weights/thresholds or
        semantic mappings.

    Raises:
        ValueError: If the requested combination is not available.
    """
    valid_personalities = [
        "safety",
        "scenic",
        "balanced",
        "comfort",
        "explorer",
        "technical",
    ]

    if personality_type not in valid_personalities:
        raise ValueError(
            f"Unknown personality type: {personality_type}. "
            f"Valid types: {', '.join(valid_personalities)}"
        )

    if use_semantic:
        # Return semantic configuration (will be mapped at runtime)
        if personality_type not in SEMANTIC_PERSONALITIES:
            raise ValueError(f"No semantic personality defined for: {personality_type}")
        return SEMANTIC_PERSONALITIES[personality_type]
    else:
        # Return framework-specific configuration
        if personality_type not in FRAMEWORK_SPECIFIC_PERSONALITIES:
            raise ValueError(
                f"No framework-specific personality for: {personality_type}"
            )

        if framework_id not in FRAMEWORK_SPECIFIC_PERSONALITIES[personality_type]:
            raise ValueError(
                f"Personality '{personality_type}' not defined for framework "
                f"'{framework_id}'. Use semantic mapping instead by setting "
                f"use_semantic=True."
            )

        return FRAMEWORK_SPECIFIC_PERSONALITIES[personality_type][framework_id]


def validate_dimension_weights(
    weights: Dict[str, float], framework_dimensions: List[Dict[str, Any]]
) -> bool:
    """Confirm weights cover framework dimensions and stay in range.

    Args:
        weights: Mapping of dimension ids to weight values.
        framework_dimensions: Framework dimension definitions containing `id` keys.

    Returns:
        bool: True when weights are valid, otherwise False.
    """
    # Extract dimension IDs from framework
    framework_dim_ids = {d["id"] for d in framework_dimensions}

    # Check all weights are for valid dimensions
    weight_dim_ids = set(weights.keys())
    if not weight_dim_ids.issubset(framework_dim_ids):
        # Weights reference dimensions not in framework
        return False

    # Check all framework dimensions have weights
    if not framework_dim_ids.issubset(weight_dim_ids):
        # Missing weights for some dimensions
        return False

    # Check weight ranges
    for weight in weights.values():
        if not (MIN_WEIGHT <= weight <= MAX_WEIGHT):
            return False

    return True


def map_semantic_to_weights(
    semantic_config: Dict[str, List[str]],
    framework_dimensions: List[Dict[str, Any]],
    weight_distribution: Dict[str, float],
) -> Dict[str, float]:
    """Translate semantic focus definitions into framework-specific weights.

    Args:
        semantic_config: Keyword groups describing priority levels.
        framework_dimensions: Framework dimensions with ids and optional names.
        weight_distribution: Weight values for priority buckets plus `unmapped`.

    Returns:
        Dict[str, float]: Weight mapping keyed by dimension id.
    """
    weights = {}

    for dim in framework_dimensions:
        dim_id = dim["id"]

        # Collect all possible names for this dimension (ID + various name fields)
        dim_names = [dim_id.lower()]

        # Add name variations if present
        for name_field in ["name", "name_en", "name_cn", "dimension_name"]:
            if name_field in dim:
                dim_names.append(dim[name_field].lower())

        # Check which priority level this dimension matches
        assigned = False

        priority_levels = [
            "primary_focus",
            "secondary_focus",
            "tertiary_focus",
            "low_priority",
        ]

        for priority_level in priority_levels:
            keywords = semantic_config.get(priority_level, [])

            # Check if dimension matches any keyword
            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Check if keyword appears in any of the dimension's names
                for dim_name in dim_names:
                    if keyword_lower in dim_name or dim_name in keyword_lower:
                        # Map priority level to weight
                        priority_key = priority_level.replace("_focus", "").replace(
                            "_priority", ""
                        )
                        weights[dim_id] = weight_distribution.get(
                            priority_key, WEIGHT_NEUTRAL_VALUE
                        )
                        assigned = True
                        break

                if assigned:
                    break

            if assigned:
                break

        # If not matched, use unmapped weight
        if not assigned:
            weights[dim_id] = weight_distribution.get("unmapped", WEIGHT_NEUTRAL_VALUE)

    return weights


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Rescale weights so their average equals 1.0.

    Args:
        weights: Mapping of dimension ids to raw weights.

    Returns:
        Dict[str, float]: Normalized weight mapping.
    """
    total_weight = sum(weights.values())
    num_dimensions = len(weights)

    if total_weight == 0:
        # Avoid division by zero - return equal weights
        return {dim_id: WEIGHT_NEUTRAL_VALUE for dim_id in weights}

    # Normalize so sum equals number of dimensions (average = 1.0)
    normalization_factor = num_dimensions / total_weight

    return {dim_id: weight * normalization_factor for dim_id, weight in weights.items()}


def get_primary_dimensions(
    weights: Dict[str, float], threshold: float = WEIGHT_THRESHOLD_PRIMARY
) -> List[str]:
    """Return dimension ids whose weight meets or exceeds the threshold.

    Args:
        weights: Mapping of dimension ids to weights.
        threshold: Minimum weight regarded as primary.

    Returns:
        List[str]: Dimension ids flagged as primary.
    """
    return [dim_id for dim_id, weight in weights.items() if weight >= threshold]


def get_available_personalities() -> List[str]:
    """Return the list of supported personality identifiers."""
    enhanced = list_enhanced_personalities()
    legacy = [p for p in SEMANTIC_PERSONALITIES.keys() if p not in enhanced]
    return enhanced + legacy


def get_available_frameworks_for_personality(personality_type: str) -> List[str]:
    """Return frameworks with explicit presets for the given personality.

    Args:
        personality_type: Personality identifier.

    Returns:
        List[str]: Framework ids that offer specific configurations.
    """
    if personality_type not in FRAMEWORK_SPECIFIC_PERSONALITIES:
        return []

    return list(FRAMEWORK_SPECIFIC_PERSONALITIES[personality_type].keys())


# ============================================================================
# High-Level Personality API
# ============================================================================


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
    personality_id: Optional[str] = None


def get_preset(
    preset_name: str, framework_id: str, use_semantic: bool = True
) -> AgentPersonality:
    """Get a preset personality configuration for a specific framework.

    Args:
        preset_name: Personality identifier (homebuyer, runner, parent_with_kids,
            photographer, elderly_walker, or legacy presets like safety/scenic).
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
        >>> # Enhanced personas are supported directly
        >>> photo = get_preset("photographer", "streetagent_5d")
    """
    # Prefer enhanced personalities (including legacy aliases) when available
    enhanced_id = _resolve_enhanced_personality_id(preset_name)
    resolved_personality_id = enhanced_id or preset_name

    if enhanced_id:
        use_semantic = True  # Enhanced personas are framework-agnostic
        enhanced_config = get_enhanced_personality(enhanced_id)
        personality_config = {
            "name": enhanced_config.name,
            "description": enhanced_config.description,
            "semantic_config": enhanced_config.semantic_config,
            "weight_distribution": enhanced_config.weight_distribution,
            "thresholds": enhanced_config.thresholds,
            "explanation_style": enhanced_config.explanation_style,
        }
    else:
        try:
            personality_config = get_personality_for_framework(
                preset_name, framework_id, use_semantic
            )
        except ValueError as e:
            available = ", ".join(get_available_personalities())
            raise ValueError(
                f"Unknown personality type: {preset_name}. "
                f"Valid types: {available}"
            ) from e

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
        personality_id=resolved_personality_id,
    )


def list_presets() -> List[str]:
    """Return the list of available personality preset names.

    Returns:
        List[str]: Personality identifiers that can be used with get_preset().

    Examples:
        >>> list_presets()
        ['homebuyer', 'runner', 'parent_with_kids', 'photographer', 'elderly_walker', 'safety', 'scenic', 'balanced', 'comfort', 'explorer', 'technical']
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
        personality_id="neutral",
    )
