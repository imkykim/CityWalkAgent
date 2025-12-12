"""Personality-specific scoring rules (keyword based).

Enhanced personalities now rely on persona prompts for scoring shifts.
We keep lightweight keyword tracking for analytics only (no score math).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PersonalityScoringRules:
    """Explicit scoring rules for personality-driven adjustments.

    These rules are used for lightweight keyword analytics only. The VLM
    persona prompt is responsible for shaping the actual scores.

    Attributes:
        concern_keywords: Keywords in VLM reasoning that signal risks/penalties.
        boost_keywords: Keywords in VLM reasoning that signal positives/bonuses.
        keyword_modifier_strength: Reserved field for future use.
    """

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
        system1_persona_hint: Short hint prepended to evaluator prompts (for dual VLM calls)
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
    system1_persona_hint: Optional[str] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    explanation_style: str = "balanced"
