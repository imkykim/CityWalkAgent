"""Lightweight keyword analytics for enhanced personalities.

The VLM persona prompt now handles score shaping directly. This module
keeps minimal keyword detection for observability only and never
modifies scores.
"""

from typing import Any, Dict, List, Tuple

from src.agent.config.scoring_rules import PersonalityScoringRules
from src.utils.logging import get_logger


class ScoreTransformer:
    """Detects persona keywords from VLM reasoning (no score changes)."""

    def __init__(
        self,
        scoring_rules: PersonalityScoringRules,
        dimension_mapping: Dict[str, str],
    ):
        """Initialize score transformer."""
        self.rules = scoring_rules
        self.dimension_mapping = dimension_mapping
        self.logger = get_logger(f"{__name__}.ScoreTransformer")

    def transform(
        self,
        scores: Dict[str, float],
        vlm_reasoning: Dict[str, str],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        self.logger.debug("ScoreTransformer initialized (analytics-only)")

    def transform(
        self,
        scores: Dict[str, float],
        vlm_reasoning: Dict[str, str],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Return scores unchanged with keyword detection metadata."""
        all_reasoning = " ".join(vlm_reasoning.values()).lower()

        detected_concerns = [
            kw for kw in self.rules.concern_keywords if kw.lower() in all_reasoning
        ]
        detected_boosts = [
            kw for kw in self.rules.boost_keywords if kw.lower() in all_reasoning
        ]

        metadata = {
            "original_scores": scores.copy(),
            "detected_features": [],  # Kept for backward compatibility
            "detected_concerns": detected_concerns,
            "detected_boosts": detected_boosts,
            "transformation_applied": False,
            "total_adjustment_per_dim": {dim: 0.0 for dim in scores},
            "note": "VLM persona prompt handles scoring; transformer is analytics-only.",
        }

        self.logger.debug(
            "Keyword detection complete (no score modification)",
            concerns=len(detected_concerns),
            boosts=len(detected_boosts),
        )

        return scores.copy(), metadata


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
