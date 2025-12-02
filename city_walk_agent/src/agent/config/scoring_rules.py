"""Personality-specific scoring rules for enhanced score differentiation.

This module provides framework-agnostic scoring rules that amplify
personality impact through:
1. Feature-based modifiers (detected from VLM reasoning)
2. Sensitivity multipliers (amplify deviations for priority dimensions)
3. Attention floors (stricter thresholds for important dimensions)
4. Keyword detection (concern/boost triggers from VLM output)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PersonalityScoringRules:
    """Explicit scoring rules for personality-driven adjustments.

    These rules are applied AFTER VLM evaluation to strengthen
    personality impact on final scores.

    Attributes:
        feature_modifiers: Map detected features to score adjustments.
            Key = feature name (framework-agnostic semantic description)
            Value = Dict of dimension_keyword -> adjustment value
            Example: {"shops_nearby": {"comfort": 1.0, "interest": 0.5}}

        sensitivity_multipliers: Amplify score deviations from neutral (5.5).
            Key = dimension keyword (matched semantically)
            Value = multiplier (1.0 = no change, 2.0 = double deviation)

        attention_floors: Minimum acceptable scores for priority dimensions.
            If score < floor, apply additional penalty.
            Key = dimension keyword
            Value = floor score (e.g., 7.0)

        concern_keywords: Keywords in VLM reasoning that trigger penalties.
            These apply general negative modifiers when detected.

        boost_keywords: Keywords in VLM reasoning that trigger bonuses.
            These apply general positive modifiers when detected.

        keyword_modifier_strength: How much concern/boost keywords affect scores.
            Default: 0.5 per keyword match.
    """

    feature_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sensitivity_multipliers: Dict[str, float] = field(default_factory=dict)
    attention_floors: Dict[str, float] = field(default_factory=dict)
    concern_keywords: List[str] = field(default_factory=list)
    boost_keywords: List[str] = field(default_factory=list)
    keyword_modifier_strength: float = 0.5


@dataclass
class EnhancedPersonalityConfig:
    """Complete enhanced personality configuration.

    Extends the base personality with scoring rules and VLM persona prompt.

    Attributes:
        personality_id: Unique identifier (e.g., "homebuyer", "runner")
        name: Human-readable name
        description: Brief description of persona behavior
        semantic_config: Priority mapping for dimension weights
        weight_distribution: Weight values for priority levels
        scoring_rules: Post-VLM scoring transformation rules
        vlm_persona_prompt: Detailed VLM instruction for this persona
        thresholds: Decision thresholds
        explanation_style: Output explanation style
    """

    personality_id: str
    name: str
    description: str
    semantic_config: Dict[str, List[str]]
    weight_distribution: Dict[str, float]
    scoring_rules: PersonalityScoringRules
    vlm_persona_prompt: str
    thresholds: Dict[str, float] = field(default_factory=dict)
    explanation_style: str = "balanced"
