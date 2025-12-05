"""Score transformation module for personality-driven score adjustments.

This module applies post-VLM transformations to strengthen personality
impact on final scores. Transformations are framework-agnostic, using
semantic matching to map rules to actual dimension IDs.

Transformation pipeline:
1. Extract features from VLM reasoning (keyword detection)
2. Apply feature modifiers (dimension-specific adjustments)
3. Apply sensitivity multipliers (amplify deviations from neutral)
4. Apply attention floor penalties (extra penalty for below-floor scores)
5. Apply keyword modifiers (general adjustments from concern/boost words)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.agent.config.scoring_rules import PersonalityScoringRules
from src.utils.logging import get_logger

# Neutral score - deviations from this are amplified
NEUTRAL_SCORE = 5.5

# Score bounds
MIN_SCORE = 1.0
MAX_SCORE = 10.0


class ScoreTransformer:
    """Transforms VLM scores based on personality scoring rules.

    The transformer operates in a framework-agnostic manner by matching
    semantic keywords in scoring rules to actual dimension IDs/names.
    """

    def __init__(
        self,
        scoring_rules: PersonalityScoringRules,
        dimension_mapping: Dict[str, str],
    ):
        """Initialize score transformer.

        Args:
            scoring_rules: Personality-specific scoring rules
            dimension_mapping: Maps dimension_id -> list of semantic keywords
                Example: {"spatial_sequence": ["spatial", "sequence", "rhythm"]}
        """
        self.rules = scoring_rules
        self.dimension_mapping = dimension_mapping
        self.logger = get_logger(f"{__name__}.ScoreTransformer")

        # Build reverse mapping: keyword -> dimension_id
        self._keyword_to_dim = self._build_keyword_mapping(dimension_mapping)

    def _build_keyword_mapping(
        self,
        dimension_mapping: Dict[str, str]
    ) -> Dict[str, str]:
        """Build reverse mapping from keywords to dimension IDs."""
        keyword_to_dim = {}

        for dim_id, keywords_or_name in dimension_mapping.items():
            # Add dimension ID itself as keyword
            keyword_to_dim[dim_id.lower()] = dim_id

            # If it's a string (name), split into keywords
            if isinstance(keywords_or_name, str):
                words = keywords_or_name.lower().replace("_", " ").split()
                for word in words:
                    if len(word) > 2:  # Skip short words
                        keyword_to_dim[word] = dim_id

            # If it's a list of keywords
            elif isinstance(keywords_or_name, list):
                for kw in keywords_or_name:
                    keyword_to_dim[kw.lower()] = dim_id

        return keyword_to_dim

    def transform(
        self,
        scores: Dict[str, float],
        vlm_reasoning: Dict[str, str],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Extract features for logging but don't modify scores.

        The VLM handles all scoring adjustments through persona prompts.
        This method only provides metadata for debugging and analysis.

        Args:
            scores: VLM scores {dimension_id: score}
            vlm_reasoning: VLM reasoning per dimension {dimension_id: reasoning_text}

        Returns:
            Tuple of:
                - Original scores (unchanged)
                - Metadata with detected features for logging
        """
        all_reasoning = " ".join(vlm_reasoning.values()).lower()

        # Feature detection for logging and analysis only
        detected_features = self._detect_features(all_reasoning)

        # Detect concern keywords
        detected_concerns = [
            kw for kw in self.rules.concern_keywords
            if kw.lower() in all_reasoning
        ]

        # Detect boost keywords
        detected_boosts = [
            kw for kw in self.rules.boost_keywords
            if kw.lower() in all_reasoning
        ]

        metadata = {
            "original_scores": scores.copy(),
            "detected_features": detected_features,
            "detected_concerns": detected_concerns,
            "detected_boosts": detected_boosts,
            "transformation_applied": False,  # No longer modifying scores
            "total_adjustment_per_dim": {dim: 0.0 for dim in scores},
            "note": "VLM handles all scoring through persona prompts"
        }

        self.logger.debug(
            "Feature detection complete (no score modification)",
            features=len(detected_features),
            concerns=len(detected_concerns),
            boosts=len(detected_boosts),
        )

        # Return scores unchanged - VLM already applied persona perspective
        return scores.copy(), metadata

    def _detect_features(self, reasoning_text: str) -> List[str]:
        """Detect features from VLM reasoning text."""
        detected = []

        for feature_name in self.rules.feature_modifiers.keys():
            # Convert feature name to search terms
            search_terms = feature_name.lower().replace("_", " ").split()

            # Check if all terms appear in reasoning
            if all(term in reasoning_text for term in search_terms):
                detected.append(feature_name)

        return detected

    def _apply_feature_modifiers(
        self,
        scores: Dict[str, float],
        detected_features: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Apply feature-based score adjustments."""
        transformed = scores.copy()
        adjustments = {}

        for feature in detected_features:
            modifiers = self.rules.feature_modifiers.get(feature, {})
            feature_adj = {}

            for keyword, modifier in modifiers.items():
                # Find matching dimension ID
                dim_id = self._find_dimension_for_keyword(keyword, scores.keys())

                if dim_id and dim_id in transformed:
                    old_score = transformed[dim_id]
                    new_score = self._clamp_score(old_score + modifier)
                    transformed[dim_id] = new_score
                    feature_adj[dim_id] = modifier

            if feature_adj:
                adjustments[feature] = feature_adj

        return transformed, adjustments

    def _apply_sensitivity_multipliers(
        self,
        scores: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply sensitivity multipliers to amplify deviations from neutral."""
        transformed = scores.copy()
        adjustments = {}

        for keyword, multiplier in self.rules.sensitivity_multipliers.items():
            dim_id = self._find_dimension_for_keyword(keyword, scores.keys())

            if dim_id and dim_id in transformed:
                original = transformed[dim_id]
                deviation = original - NEUTRAL_SCORE
                amplified_deviation = deviation * multiplier
                new_score = self._clamp_score(NEUTRAL_SCORE + amplified_deviation)

                adjustment = new_score - original
                if abs(adjustment) > 0.01:
                    transformed[dim_id] = new_score
                    adjustments[dim_id] = round(adjustment, 2)

        return transformed, adjustments

    def _apply_attention_floors(
        self,
        scores: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply additional penalties for scores below attention floors."""
        transformed = scores.copy()
        penalties = {}

        for keyword, floor in self.rules.attention_floors.items():
            dim_id = self._find_dimension_for_keyword(keyword, scores.keys())

            if dim_id and dim_id in transformed:
                current = transformed[dim_id]

                if current < floor:
                    # Apply penalty proportional to how far below floor
                    shortfall = floor - current
                    penalty = shortfall * 0.5  # 50% additional penalty
                    new_score = self._clamp_score(current - penalty)

                    if abs(penalty) > 0.01:
                        transformed[dim_id] = new_score
                        penalties[dim_id] = round(-penalty, 2)

        return transformed, penalties

    def _apply_keyword_modifiers(
        self,
        scores: Dict[str, float],
        reasoning_text: str,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply general adjustments based on concern/boost keywords."""
        transformed = scores.copy()
        adjustments = {}

        concern_count = 0
        boost_count = 0

        # Count concern keywords
        for keyword in self.rules.concern_keywords:
            if keyword.lower() in reasoning_text:
                concern_count += 1

        # Count boost keywords
        for keyword in self.rules.boost_keywords:
            if keyword.lower() in reasoning_text:
                boost_count += 1

        # Calculate net adjustment
        net_modifier = (boost_count - concern_count) * self.rules.keyword_modifier_strength

        if abs(net_modifier) > 0.01:
            # Apply to all dimensions, weighted by sensitivity
            for dim_id in transformed:
                # Find sensitivity for this dimension
                sensitivity = 1.0
                for keyword, sens in self.rules.sensitivity_multipliers.items():
                    if self._find_dimension_for_keyword(keyword, [dim_id]) == dim_id:
                        sensitivity = sens
                        break

                # Higher sensitivity = more affected by keywords
                dim_modifier = net_modifier * (sensitivity / 1.5)

                if abs(dim_modifier) > 0.01:
                    old = transformed[dim_id]
                    transformed[dim_id] = self._clamp_score(old + dim_modifier)
                    adjustments[dim_id] = round(dim_modifier, 2)

        return transformed, adjustments

    def _find_dimension_for_keyword(
        self,
        keyword: str,
        available_dims: List[str],
    ) -> Optional[str]:
        """Find the dimension ID that matches a semantic keyword."""
        keyword_lower = keyword.lower()

        # Direct match in reverse mapping
        if keyword_lower in self._keyword_to_dim:
            matched = self._keyword_to_dim[keyword_lower]
            if matched in available_dims:
                return matched

        # Fuzzy match: check if keyword appears in any dimension ID
        for dim_id in available_dims:
            if keyword_lower in dim_id.lower():
                return dim_id
            # Also check the dimension name from mapping
            if dim_id in self.dimension_mapping:
                dim_name = self.dimension_mapping[dim_id]
                if isinstance(dim_name, str) and keyword_lower in dim_name.lower():
                    return dim_id

        return None

    def _clamp_score(self, score: float) -> float:
        """Clamp score to valid range."""
        return max(MIN_SCORE, min(MAX_SCORE, score))


def create_dimension_mapping(framework_dimensions: List[Dict[str, Any]]) -> Dict[str, str]:
    """Create dimension mapping from framework definition.

    Args:
        framework_dimensions: List of dimension dicts from framework config
            Each should have 'id', 'name_en', and optionally 'description'

    Returns:
        Dict mapping dimension_id to searchable name/keywords
    """
    mapping = {}

    for dim in framework_dimensions:
        dim_id = dim["id"]

        # Collect all name variations
        names = [dim_id]

        if "name_en" in dim:
            names.append(dim["name_en"])
        if "name_cn" in dim:
            names.append(dim["name_cn"])
        if "description" in dim:
            # Extract key terms from description
            desc_words = dim["description"].lower().split()[:10]
            names.extend(desc_words)

        # Join into searchable string
        mapping[dim_id] = " ".join(names)

    return mapping
