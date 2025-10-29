"""
Agent configuration helper functions.

This module provides utility functions for working with agent personalities,
dimension weights, and framework configurations. These functions operate on
the data defined in src.agent.constants.

Functions moved from src.config.agent_constants:
- get_personality_for_framework()
- validate_dimension_weights()
- map_semantic_to_weights()
- normalize_weights()
- get_primary_dimensions()
- get_available_personalities()
- get_available_frameworks_for_personality()
"""

from typing import Any, Dict, List

from src.agent.constants import (
    FRAMEWORK_SPECIFIC_PERSONALITIES,
    MAX_WEIGHT,
    MIN_WEIGHT,
    SEMANTIC_PERSONALITIES,
    WEIGHT_NEUTRAL_VALUE,
    WEIGHT_THRESHOLD_PRIMARY,
)


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
    return list(SEMANTIC_PERSONALITIES.keys())


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
