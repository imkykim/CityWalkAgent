# Enhanced Personality System Implementation Guide

## Overview

This guide strengthens personality impact on System 2 scores through a **framework-agnostic** design. The current system has minimal personality effect because:

1. Weights are normalized (average = 1.0), creating subtle differences
2. VLM prompts mention personality generically without specific behavioral guidance
3. No post-processing amplification of personality-relevant scores

**Solution**: Multi-layer enhancement with semantic feature detection, persona-specific VLM prompts, and post-VLM score transformation.

---

## Step 1: Add Scoring Rules Dataclass

**File**: `src/agent/config/scoring_rules.py` (NEW FILE)

```bash
touch src/agent/config/scoring_rules.py
```

```python
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
```

---

## Step 2: Create Enhanced Personality Definitions

**File**: `src/agent/config/enhanced_personalities.py` (NEW FILE)

```bash
touch src/agent/config/enhanced_personalities.py
```

```python
"""Enhanced personality definitions with strong scoring differentiation.

Framework-agnostic personalities that use semantic keywords to map
to any evaluation framework's dimensions.

Key design principles:
1. Semantic keywords match dimension names/descriptions across frameworks
2. Scoring rules use semantic terms, not hardcoded dimension IDs
3. VLM persona prompts give concrete behavioral guidance
4. Feature modifiers detect semantic concepts from VLM reasoning
"""

from src.agent.config.scoring_rules import (
    EnhancedPersonalityConfig,
    PersonalityScoringRules,
)


# ============================================================================
# ENHANCED PERSONALITY DEFINITIONS
# ============================================================================

ENHANCED_PERSONALITIES: dict[str, EnhancedPersonalityConfig] = {

    # -------------------------------------------------------------------------
    # HOMEBUYER - Evaluating neighborhood for family residence
    # -------------------------------------------------------------------------
    "homebuyer": EnhancedPersonalityConfig(
        personality_id="homebuyer",
        name="Prospective Homebuyer",
        description="Evaluating neighborhood livability for family residence",

        # Semantic mapping - works with any framework
        semantic_config={
            "primary_focus": [
                "safety", "secure", "protection", "安全",
                "comfort", "ease", "convenience", "舒适",
                "functional", "amenity", "service", "功能",
            ],
            "secondary_focus": [
                "legibility", "navigation", "wayfinding", "清晰",
                "coherence", "order", "organization", "秩序",
            ],
            "tertiary_focus": [
                "interest", "engagement", "趣味",
                "aesthetics", "beauty", "visual", "美观",
            ],
            "low_priority": [
                "complexity", "mystery", "novelty",
            ],
        },

        # Stronger weight spread for more differentiation
        weight_distribution={
            "primary": 2.5,
            "secondary": 1.4,
            "tertiary": 0.8,
            "low": 0.5,
            "unmapped": 0.7,
        },

        # Scoring rules for post-VLM transformation
        scoring_rules=PersonalityScoringRules(
            # Feature modifiers - semantic keywords detected in VLM reasoning
            # Maps to dimension KEYWORDS not IDs (framework-agnostic)
            feature_modifiers={
                "shops_visible": {
                    "comfort": 1.2, "functional": 1.2, "interest": 0.5
                },
                "cafe_restaurant": {
                    "comfort": 1.0, "interest": 0.8, "functional": 0.8
                },
                "school_nearby": {
                    "safety": 1.5, "comfort": 1.0, "functional": 1.5
                },
                "park_greenery": {
                    "comfort": 1.5, "aesthetics": 1.2, "safety": 0.5
                },
                "playground": {
                    "comfort": 1.5, "interest": 1.0, "safety": 0.5
                },
                "heavy_traffic": {
                    "safety": -2.0, "comfort": -1.5
                },
                "narrow_sidewalk": {
                    "safety": -1.5, "comfort": -1.2
                },
                "poor_lighting": {
                    "safety": -1.5
                },
                "residential_quiet": {
                    "comfort": 1.5, "safety": 0.8
                },
                "industrial_zone": {
                    "comfort": -1.5, "aesthetics": -1.0, "safety": -0.5
                },
                "construction": {
                    "safety": -1.5, "comfort": -1.2
                },
            },

            # Sensitivity multipliers - semantic dimension keywords
            sensitivity_multipliers={
                "safety": 1.8,
                "secure": 1.8,
                "protection": 1.8,
                "comfort": 1.5,
                "ease": 1.5,
                "functional": 1.4,
                "interest": 0.8,
                "aesthetics": 0.7,
                "complexity": 0.6,
            },

            # Attention floors - strict thresholds for priority dimensions
            attention_floors={
                "safety": 7.0,
                "secure": 7.0,
                "comfort": 6.5,
                "functional": 6.5,
            },

            # Concern keywords - trigger general negative adjustment
            concern_keywords=[
                "traffic", "construction", "noise", "industrial",
                "abandoned", "graffiti", "dark", "narrow", "unsafe",
                "poor condition", "no sidewalk", "dangerous",
            ],

            # Boost keywords - trigger general positive adjustment
            boost_keywords=[
                "tree-lined", "park", "school", "playground", "shop",
                "cafe", "restaurant", "residential", "quiet", "clean",
                "well-maintained", "family", "safe", "convenient",
            ],

            keyword_modifier_strength=0.6,
        ),

        # Detailed VLM persona prompt
        vlm_persona_prompt="""
You are evaluating this location as a PROSPECTIVE HOMEBUYER considering moving to this neighborhood with your family.

YOUR EVALUATION PRIORITIES (in strict order):
1. SAFETY: Would I feel safe walking here with my family at any time of day?
2. DAILY CONVENIENCE: Would errands (shopping, dining, schools) be accessible?
3. LIVABILITY: Is this a pleasant residential environment?

CRITICAL SCORING ADJUSTMENTS - Apply these actively:

BONUSES (increase scores):
- Visible shops/cafes/restaurants: +1 to +2 for comfort/functional dimensions
- Schools or playgrounds visible: +1.5 for safety, +1 for comfort
- Parks or green spaces: +1.5 for comfort, +1 for aesthetics
- Quiet residential streets: +1.5 for comfort
- Clean, well-maintained: +1 across comfort dimensions

PENALTIES (decrease scores):
- Heavy vehicle traffic: -2 for safety, -1.5 for comfort (SIGNIFICANT)
- Narrow or missing sidewalks: -1.5 for safety, -1 for comfort
- Industrial/commercial zones: -1.5 for comfort, -1 for aesthetics
- Construction zones: -1.5 for safety and comfort
- Poor lighting/dark areas: -1.5 for safety

YOUR SCORING PHILOSOPHY:
- Be STRICT on safety - a homebuyer won't compromise on family safety
- Be GENEROUS when you see family-friendly amenities
- Aesthetic beauty is secondary to practical livability
- Your scores should differ from System 1 by ±1 to ±3 points based on these priorities
""",

        thresholds={
            "min_overall_score": 6.5,
            "min_primary_score": 7.0,
            "max_volatility": 2.0,
            "max_barriers": 1,
        },

        explanation_style="balanced",
    ),

    # -------------------------------------------------------------------------
    # RUNNER - Evaluating route for regular running/jogging
    # -------------------------------------------------------------------------
    "runner": EnhancedPersonalityConfig(
        personality_id="runner",
        name="Urban Runner",
        description="Evaluating route suitability for regular running and jogging",

        semantic_config={
            "primary_focus": [
                "safety", "secure", "安全",
                "comfort", "ease", "smooth", "舒适",
                "spatial", "openness", "width", "空间",
            ],
            "secondary_focus": [
                "aesthetics", "visual", "景观",
                "coherence", "continuity", "连续",
            ],
            "tertiary_focus": [
                "interest", "variety", "趣味",
            ],
            "low_priority": [
                "functional", "amenity", "service", "shops",
                "complexity", "mystery",
            ],
        },

        weight_distribution={
            "primary": 2.5,
            "secondary": 1.3,
            "tertiary": 0.7,
            "low": 0.4,
            "unmapped": 0.6,
        },

        scoring_rules=PersonalityScoringRules(
            feature_modifiers={
                "wide_path": {
                    "comfort": 2.0, "safety": 1.0, "spatial": 1.5
                },
                "park_trail": {
                    "comfort": 2.5, "aesthetics": 1.5, "safety": 1.0
                },
                "dedicated_running_path": {
                    "comfort": 3.0, "safety": 1.5
                },
                "smooth_surface": {
                    "comfort": 1.5, "safety": 1.0
                },
                "uneven_surface": {
                    "safety": -2.0, "comfort": -2.0
                },
                "cobblestone": {
                    "safety": -1.5, "comfort": -1.5
                },
                "crowded_pedestrians": {
                    "comfort": -1.5, "spatial": -1.0
                },
                "tree_shade": {
                    "comfort": 1.5, "aesthetics": 1.0
                },
                "waterfront": {
                    "aesthetics": 2.0, "comfort": 1.0, "interest": 1.0
                },
                "heavy_traffic": {
                    "safety": -1.5, "comfort": -1.0
                },
                "steps_stairs": {
                    "comfort": -1.5
                },
                "air_pollution": {
                    "comfort": -2.0, "safety": -1.0
                },
            },

            sensitivity_multipliers={
                "comfort": 2.0,  # Very sensitive to running comfort
                "ease": 2.0,
                "spatial": 1.8,
                "safety": 1.6,
                "aesthetics": 1.2,
                "interest": 0.6,
                "functional": 0.4,
            },

            attention_floors={
                "comfort": 6.5,
                "spatial": 6.0,
                "safety": 6.0,
            },

            concern_keywords=[
                "uneven", "cracked", "cobblestone", "crowded",
                "narrow", "steps", "stairs", "obstacles", "traffic",
                "pollution", "smog", "congested",
            ],

            boost_keywords=[
                "park", "promenade", "wide", "smooth", "dedicated",
                "path", "trail", "tree-lined", "waterfront", "shade",
                "open", "spacious",
            ],

            keyword_modifier_strength=0.7,
        ),

        vlm_persona_prompt="""
You are evaluating this location as an URBAN RUNNER planning regular jogging routes.

YOUR EVALUATION PRIORITIES (in strict order):
1. RUNNING SURFACE: Is the ground smooth, even, safe for running at speed?
2. PATH WIDTH: Is there room to run without dodging pedestrians/obstacles?
3. SAFETY: Good visibility, no tripping hazards, safe from vehicle traffic?
4. ENVIRONMENT: Shade availability, air quality, scenic views during run?

CRITICAL SCORING ADJUSTMENTS - Apply these actively:

BONUSES (increase scores):
- Wide, smooth paths/sidewalks: +2 for comfort, +1 for safety
- Park paths or dedicated running trails: +2.5 for comfort, +1.5 for aesthetics
- Tree-lined routes with shade: +1.5 for comfort
- Waterfront/scenic promenades: +2 for aesthetics, +1 for interest
- Open, uncrowded spaces: +1.5 for spatial dimensions

PENALTIES (decrease scores):
- Uneven/cracked/cobblestone surfaces: -2 for safety AND comfort (CRITICAL)
- Crowded with pedestrians: -1.5 for comfort
- Narrow paths requiring constant maneuvering: -1.5 for comfort
- Steps or stairs: -1.5 for comfort (breaks running rhythm)
- Near heavy traffic: -1 for safety and comfort
- Visible air pollution: -2 for comfort

YOUR SCORING PHILOSOPHY:
- Running surface quality is PARAMOUNT - more important than anything else
- A beautiful cafe is IRRELEVANT if the sidewalk is too narrow to run
- Shops and amenities are LOW PRIORITY for a runner
- Your scores should differ significantly from System 1 based on running needs
""",

        thresholds={
            "min_overall_score": 6.0,
            "min_primary_score": 6.5,
            "max_volatility": 2.5,
            "max_barriers": 2,
        },

        explanation_style="balanced",
    ),

    # -------------------------------------------------------------------------
    # PARENT_WITH_KIDS - Extreme safety focus for family with young children
    # -------------------------------------------------------------------------
    "parent_with_kids": EnhancedPersonalityConfig(
        personality_id="parent_with_kids",
        name="Parent with Young Children",
        description="Evaluating route safety and family-friendliness with stroller",

        semantic_config={
            "primary_focus": [
                "safety", "secure", "protection", "安全",
            ],
            "secondary_focus": [
                "comfort", "ease", "accessibility", "舒适",
                "functional", "amenity", "功能",
            ],
            "tertiary_focus": [
                "interest", "engagement", "趣味",
                "legibility", "navigation", "清晰",
            ],
            "low_priority": [
                "aesthetics", "beauty", "美观",
                "complexity", "mystery",
            ],
        },

        weight_distribution={
            "primary": 3.0,  # MAXIMUM safety weight
            "secondary": 1.4,
            "tertiary": 0.6,
            "low": 0.3,
            "unmapped": 0.5,
        },

        scoring_rules=PersonalityScoringRules(
            feature_modifiers={
                "playground": {
                    "interest": 2.5, "comfort": 1.5, "safety": 0.5
                },
                "park": {
                    "safety": 1.0, "comfort": 1.5, "interest": 1.5
                },
                "school_nearby": {
                    "safety": 1.0, "comfort": 1.0
                },
                "crosswalk": {
                    "safety": 1.5
                },
                "wide_sidewalk": {
                    "safety": 1.0, "comfort": 2.0  # Stroller needs width
                },
                "heavy_traffic": {
                    "safety": -3.0,  # SEVERE penalty
                    "comfort": -1.5
                },
                "no_sidewalk": {
                    "safety": -3.0,  # DEAL BREAKER
                    "comfort": -2.0
                },
                "narrow_sidewalk": {
                    "safety": -2.0,
                    "comfort": -2.0  # Stroller problem
                },
                "construction": {
                    "safety": -2.5,
                    "comfort": -1.5
                },
                "steps_stairs": {
                    "comfort": -2.5,  # Cannot use with stroller
                    "safety": -1.0
                },
                "uneven_surface": {
                    "safety": -1.5,
                    "comfort": -2.0  # Stroller wheels
                },
                "isolated_area": {
                    "safety": -2.0
                },
                "poor_lighting": {
                    "safety": -2.0
                },
            },

            sensitivity_multipliers={
                "safety": 2.5,  # MAXIMUM sensitivity
                "secure": 2.5,
                "protection": 2.5,
                "comfort": 1.6,
                "accessibility": 1.8,
                "interest": 0.6,
                "aesthetics": 0.4,
            },

            attention_floors={
                "safety": 7.5,  # HIGHEST threshold
                "secure": 7.5,
                "comfort": 6.5,
            },

            concern_keywords=[
                "traffic", "vehicle", "no sidewalk", "construction",
                "narrow", "steps", "stairs", "unsafe", "dark",
                "isolated", "dangerous", "hazard", "obstacle",
                "uneven", "broken", "hole",
            ],

            boost_keywords=[
                "playground", "park", "school", "crosswalk", "crossing",
                "wide", "protected", "safe", "family", "stroller",
                "accessible", "flat", "smooth",
            ],

            keyword_modifier_strength=0.8,  # Strong keyword impact
        ),

        vlm_persona_prompt="""
You are evaluating this location as a PARENT WITH YOUNG CHILDREN pushing a stroller.

YOUR #1 PRIORITY IS CHILD SAFETY - Everything else is secondary.

EVALUATION CRITERIA (in STRICT order):
1. SAFETY (CRITICAL): Traffic danger? Safe crossings? Hazards a child could encounter?
2. STROLLER ACCESSIBILITY: Can I push a stroller here? Steps? Narrow paths?
3. KID-FRIENDLINESS: Playgrounds? Parks? Things to keep children entertained?

CRITICAL SCORING ADJUSTMENTS - APPLY AGGRESSIVELY:

SEVERE PENALTIES (DEAL BREAKERS):
- Heavy vehicle traffic without protection: -3 for safety (CRITICAL)
- No sidewalk or very narrow sidewalk: -3 for safety, -2 for comfort
- Steps/stairs (stroller blocker): -2.5 for comfort, -1 for safety
- Construction zones: -2.5 for safety
- Dark or isolated areas: -2 for safety

BONUSES:
- Playgrounds visible: +2.5 for interest, +1.5 for comfort
- Parks and green spaces: +1.5 for safety, +1.5 for comfort
- Clear crosswalks: +1.5 for safety
- Wide, protected sidewalks: +2 for comfort, +1 for safety
- Family-friendly environment: +1 for all positive dimensions

YOUR SCORING PHILOSOPHY:
- You are EXTREMELY RISK-AVERSE
- Even MINOR safety concerns should SIGNIFICANTLY reduce scores
- A beautiful street means NOTHING if it's dangerous for children
- Stroller accessibility is as important as safety
- Aesthetics are nearly irrelevant compared to child safety
- Your safety scores should be much more extreme than System 1
""",

        thresholds={
            "min_overall_score": 7.0,  # High threshold
            "min_primary_score": 7.5,  # Very high for safety
            "max_volatility": 1.5,  # Low tolerance for inconsistency
            "max_barriers": 0,  # Zero tolerance
        },

        explanation_style="safety",
    ),

    # -------------------------------------------------------------------------
    # PHOTOGRAPHER - Seeking photogenic locations
    # -------------------------------------------------------------------------
    "photographer": EnhancedPersonalityConfig(
        personality_id="photographer",
        name="Street Photographer",
        description="Seeking photogenic urban scenes and interesting compositions",

        semantic_config={
            "primary_focus": [
                "aesthetics", "visual", "beauty", "美观",
                "complexity", "interest", "variety", "趣味",
                "spatial", "sequence", "composition", "构图",
            ],
            "secondary_focus": [
                "coherence", "harmony", "秩序",
                "mystery", "intrigue", "神秘",
            ],
            "tertiary_focus": [
                "legibility", "landmark", "地标",
            ],
            "low_priority": [
                "safety", "secure", "安全",
                "comfort", "ease", "舒适",
                "functional", "amenity", "功能",
            ],
        },

        weight_distribution={
            "primary": 2.5,
            "secondary": 1.6,
            "tertiary": 1.0,
            "low": 0.5,
            "unmapped": 0.8,
        },

        scoring_rules=PersonalityScoringRules(
            feature_modifiers={
                "dramatic_lighting": {
                    "aesthetics": 2.5, "interest": 1.5
                },
                "interesting_architecture": {
                    "aesthetics": 2.0, "interest": 1.5, "complexity": 1.0
                },
                "layered_depth": {
                    "spatial": 2.0, "aesthetics": 1.5
                },
                "street_life": {
                    "interest": 2.0, "complexity": 1.5
                },
                "unique_character": {
                    "aesthetics": 2.0, "mystery": 1.5
                },
                "visual_contrast": {
                    "aesthetics": 1.5, "complexity": 1.5
                },
                "monotonous": {
                    "aesthetics": -2.0, "interest": -2.0
                },
                "generic_buildings": {
                    "aesthetics": -1.5, "interest": -1.0
                },
                "visually_cluttered": {
                    "aesthetics": -1.0, "coherence": -1.5
                },
                "harsh_lighting": {
                    "aesthetics": -1.0
                },
            },

            sensitivity_multipliers={
                "aesthetics": 2.0,
                "visual": 2.0,
                "complexity": 1.8,
                "interest": 1.8,
                "spatial": 1.6,
                "mystery": 1.5,
                "safety": 0.5,
                "comfort": 0.5,
                "functional": 0.4,
            },

            attention_floors={
                "aesthetics": 6.0,
                "interest": 5.5,
            },

            concern_keywords=[
                "boring", "monotonous", "generic", "bland", "uniform",
                "featureless", "plain", "uninteresting", "flat",
            ],

            boost_keywords=[
                "dramatic", "interesting", "unique", "character", "texture",
                "contrast", "layers", "depth", "light", "shadow",
                "architectural", "historic", "street life", "vibrant",
            ],

            keyword_modifier_strength=0.7,
        ),

        vlm_persona_prompt="""
You are evaluating this location as a STREET PHOTOGRAPHER looking for photogenic scenes.

YOUR EVALUATION PRIORITIES:
1. VISUAL INTEREST: Compelling compositions? Interesting subjects?
2. AESTHETICS: Beautiful or dramatic visual qualities?
3. COMPLEXITY: Rich details and layers worth capturing?
4. LIGHT & SHADOW: Interesting lighting conditions?

CRITICAL SCORING ADJUSTMENTS:

BONUSES (what makes a great photo location):
- Dramatic lighting/shadow play: +2.5 for aesthetics
- Interesting architecture: +2 for aesthetics, +1.5 for interest
- Visual depth with foreground/background: +2 for spatial dimensions
- Street life and activity: +2 for interest
- Unique character or atmosphere: +2 for aesthetics

PENALTIES (photographically uninteresting):
- Monotonous, repetitive scenes: -2 for aesthetics AND interest
- Generic, featureless buildings: -1.5 for aesthetics
- Visual clutter without composition: -1.5 for coherence
- Flat, harsh lighting: -1 for aesthetics

YOUR SCORING PHILOSOPHY:
- Safety and comfort are nearly IRRELEVANT to you
- A dangerous alley with amazing light is MORE interesting than a safe boring street
- You're looking for visual stories, not practical walkability
- Complexity and mystery are HIGHLY valued
- Your aesthetics scores should diverge significantly from System 1
""",

        thresholds={
            "min_overall_score": 5.5,  # Lower threshold - will accept some risks
            "min_primary_score": 6.0,
            "max_volatility": 3.5,  # High tolerance for variety
            "max_barriers": 3,
        },

        explanation_style="scenic",
    ),

    # -------------------------------------------------------------------------
    # ELDERLY_WALKER - Accessibility and stability focus
    # -------------------------------------------------------------------------
    "elderly_walker": EnhancedPersonalityConfig(
        personality_id="elderly_walker",
        name="Elderly Walker",
        description="Evaluating route for safe, comfortable walking with mobility considerations",

        semantic_config={
            "primary_focus": [
                "safety", "secure", "安全",
                "comfort", "ease", "舒适",
            ],
            "secondary_focus": [
                "legibility", "navigation", "wayfinding", "清晰",
                "functional", "amenity", "rest", "功能",
            ],
            "tertiary_focus": [
                "aesthetics", "beauty", "美观",
                "interest", "趣味",
            ],
            "low_priority": [
                "complexity", "mystery",
                "novelty",
            ],
        },

        weight_distribution={
            "primary": 2.8,
            "secondary": 1.5,
            "tertiary": 0.6,
            "low": 0.3,
            "unmapped": 0.5,
        },

        scoring_rules=PersonalityScoringRules(
            feature_modifiers={
                "benches_seating": {
                    "comfort": 2.5, "functional": 2.0
                },
                "smooth_surface": {
                    "safety": 2.0, "comfort": 2.0
                },
                "good_lighting": {
                    "safety": 1.5
                },
                "gentle_slope": {
                    "comfort": 1.5, "safety": 1.0
                },
                "uneven_surface": {
                    "safety": -2.5, "comfort": -2.0  # Trip hazard
                },
                "steep_slope": {
                    "safety": -2.0, "comfort": -2.0
                },
                "steps_stairs": {
                    "safety": -2.0, "comfort": -2.5
                },
                "crowded": {
                    "comfort": -1.5, "safety": -1.0
                },
                "no_handrails": {
                    "safety": -1.5
                },
                "long_distance_no_rest": {
                    "comfort": -2.0
                },
                "poor_signage": {
                    "legibility": -1.5, "comfort": -1.0
                },
            },

            sensitivity_multipliers={
                "safety": 2.2,
                "comfort": 2.2,
                "ease": 2.2,
                "legibility": 1.6,
                "functional": 1.5,
                "aesthetics": 0.7,
                "interest": 0.6,
                "complexity": 0.4,
            },

            attention_floors={
                "safety": 7.0,
                "comfort": 7.0,
                "ease": 7.0,
            },

            concern_keywords=[
                "uneven", "cracked", "steep", "steps", "stairs",
                "crowded", "obstacle", "trip", "hazard", "slippery",
                "no handrail", "dark", "confusing",
            ],

            boost_keywords=[
                "bench", "seating", "rest area", "smooth", "flat",
                "well-lit", "handrail", "clear", "accessible",
                "gentle", "wide", "uncrowded",
            ],

            keyword_modifier_strength=0.7,
        ),

        vlm_persona_prompt="""
You are evaluating this location as an ELDERLY WALKER with mobility considerations.

YOUR EVALUATION PRIORITIES:
1. SURFACE SAFETY: Even surfaces? Trip hazards? Slippery conditions?
2. PHYSICAL COMFORT: Can I walk here without strain? Are there rest spots?
3. ACCESSIBILITY: Steps? Steep slopes? Distance between rest areas?
4. CLARITY: Easy to navigate? Clear signage?

CRITICAL SCORING ADJUSTMENTS:

SEVERE PENALTIES:
- Uneven/cracked surfaces: -2.5 for safety AND comfort (trip hazard)
- Steps or stairs without alternative: -2.5 for comfort, -2 for safety
- Steep slopes: -2 for safety and comfort
- No seating over long distances: -2 for comfort

BONUSES:
- Benches or rest areas: +2.5 for comfort, +2 for functional
- Smooth, even surfaces: +2 for safety and comfort
- Good lighting: +1.5 for safety
- Clear wayfinding: +1.5 for legibility

YOUR SCORING PHILOSOPHY:
- Trip hazards are your GREATEST concern
- Rest availability is CRITICAL for comfort
- Visual interest is nice but secondary to physical safety
- You need MORE conservative scores than System 1
- Even small surface irregularities matter
""",

        thresholds={
            "min_overall_score": 7.0,
            "min_primary_score": 7.5,
            "max_volatility": 1.5,
            "max_barriers": 0,
        },

        explanation_style="safety",
    ),
}


def get_enhanced_personality(personality_id: str) -> EnhancedPersonalityConfig:
    """Get enhanced personality configuration by ID.

    Args:
        personality_id: Personality identifier (e.g., "homebuyer", "runner")

    Returns:
        EnhancedPersonalityConfig for the requested personality

    Raises:
        ValueError: If personality_id not found
    """
    if personality_id not in ENHANCED_PERSONALITIES:
        available = list(ENHANCED_PERSONALITIES.keys())
        raise ValueError(
            f"Unknown personality: {personality_id}. "
            f"Available: {', '.join(available)}"
        )
    return ENHANCED_PERSONALITIES[personality_id]


def list_enhanced_personalities() -> list[str]:
    """List available enhanced personality IDs."""
    return list(ENHANCED_PERSONALITIES.keys())
```

---

## Step 3: Create Score Transformation Module

**File**: `src/agent/capabilities/score_transformer.py` (NEW FILE)

```bash
touch src/agent/capabilities/score_transformer.py
```

```python
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

logger = get_logger(__name__)

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
        self.logger = logger.bind(component="ScoreTransformer")

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
        """Apply full transformation pipeline to scores.

        Args:
            scores: Original VLM scores {dimension_id: score}
            vlm_reasoning: VLM reasoning per dimension {dimension_id: reasoning_text}

        Returns:
            Tuple of:
                - Transformed scores {dimension_id: transformed_score}
                - Transformation metadata for logging/debugging
        """
        transformed = scores.copy()
        metadata = {
            "original_scores": scores.copy(),
            "detected_features": [],
            "feature_adjustments": {},
            "sensitivity_adjustments": {},
            "floor_penalties": {},
            "keyword_adjustments": {},
            "total_adjustment_per_dim": {},
        }

        # Combine all reasoning text for feature detection
        all_reasoning = " ".join(vlm_reasoning.values()).lower()

        # Step 1: Detect features from reasoning
        detected_features = self._detect_features(all_reasoning)
        metadata["detected_features"] = detected_features

        # Step 2: Apply feature modifiers
        transformed, feature_adj = self._apply_feature_modifiers(
            transformed, detected_features
        )
        metadata["feature_adjustments"] = feature_adj

        # Step 3: Apply sensitivity multipliers
        transformed, sens_adj = self._apply_sensitivity_multipliers(transformed)
        metadata["sensitivity_adjustments"] = sens_adj

        # Step 4: Apply attention floor penalties
        transformed, floor_pen = self._apply_attention_floors(transformed)
        metadata["floor_penalties"] = floor_pen

        # Step 5: Apply keyword modifiers
        transformed, kw_adj = self._apply_keyword_modifiers(
            transformed, all_reasoning
        )
        metadata["keyword_adjustments"] = kw_adj

        # Calculate total adjustments
        for dim_id in scores:
            original = scores[dim_id]
            final = transformed[dim_id]
            metadata["total_adjustment_per_dim"][dim_id] = round(final - original, 2)

        self.logger.debug(
            "Score transformation complete",
            features_detected=len(detected_features),
            dimensions_adjusted=sum(
                1 for adj in metadata["total_adjustment_per_dim"].values()
                if abs(adj) > 0.1
            ),
        )

        return transformed, metadata

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
```

---

## Step 4: Integrate with ThinkingModule

**File**: `src/agent/capabilities/thinking.py`

Add imports at the top:

```python
from src.agent.config.enhanced_personalities import (
    get_enhanced_personality,
    EnhancedPersonalityConfig,
)
from src.agent.capabilities.score_transformer import (
    ScoreTransformer,
    create_dimension_mapping,
)
```

Modify the `ThinkingModule` class:

### 4.1 Update `__init__` method

```python
def __init__(
    self,
    enable_score_revision: bool = True,
    framework_dimensions: Optional[List[Dict[str, Any]]] = None,
):
    """Initialize ThinkingModule.

    Args:
        enable_score_revision: Whether to enable System 2 score revision
        framework_dimensions: Framework dimension definitions for semantic mapping
    """
    self.enable_score_revision = enable_score_revision
    self.framework_dimensions = framework_dimensions or []
    self.thinking_history: List[ThinkingResult] = []
    self.logger = get_logger(__name__).bind(component="ThinkingModule")

    # Create dimension mapping for score transformation
    self.dimension_mapping = create_dimension_mapping(self.framework_dimensions)
```

### 4.2 Add method to get enhanced personality

```python
def _get_enhanced_config(
    self,
    personality: Optional[AgentPersonality],
) -> Optional[EnhancedPersonalityConfig]:
    """Get enhanced personality config if available.

    Falls back to None for basic personalities, enabling graceful degradation.
    """
    if personality is None:
        return None

    # Try to find enhanced config by matching name
    personality_name = personality.name.lower().replace(" ", "_")

    # Common mappings
    name_mappings = {
        "safety_guardian": "parent_with_kids",  # Map safety to parent
        "scenic_explorer": "photographer",
        "balanced_navigator": "homebuyer",
        "comfort_seeker": "homebuyer",
        "urban_explorer": "photographer",
    }

    # Check direct match or mapped match
    try:
        if personality_name in ENHANCED_PERSONALITIES:
            return get_enhanced_personality(personality_name)
        elif personality_name in name_mappings:
            return get_enhanced_personality(name_mappings[personality_name])
    except ValueError:
        pass

    return None
```

### 4.3 Update `_build_vlm_revision_prompt` method

Replace the existing method with this enhanced version:

````python
def _build_vlm_revision_prompt(
    self,
    waypoint_id: int,
    system1_scores: Dict[str, float],
    system1_reasoning: Dict[str, str],
    stm_context: Dict[str, Any],
    ltm_patterns: List[Dict[str, Any]],
    personality: Optional[AgentPersonality],
    trigger_reason: TriggerReason,
) -> str:
    """Build VLM prompt with STRONG personality-specific guidance."""

    # Format memory context
    stm_summary = self._format_stm_context(stm_context)
    ltm_summary = self._format_ltm_patterns(ltm_patterns)

    # Get enhanced personality config
    enhanced_config = self._get_enhanced_config(personality)

    # Build personality section
    if enhanced_config:
        persona_prompt = enhanced_config.vlm_persona_prompt

        # Add scoring rule examples
        feature_examples = []
        for feature, mods in list(enhanced_config.scoring_rules.feature_modifiers.items())[:6]:
            mod_str = ", ".join(f"{k}: {v:+.1f}" for k, v in mods.items())
            feature_examples.append(f"  - {feature.replace('_', ' ')}: {mod_str}")

        scoring_guidance = "\n".join(feature_examples) if feature_examples else "N/A"

        personality_section = f"""
## YOUR PERSONA
{persona_prompt}

## EXPLICIT SCORING ADJUSTMENTS TO APPLY
When you observe these features, apply these adjustments:
{scoring_guidance}

## KEYWORDS THAT SHOULD TRIGGER CONCERNS
{', '.join(enhanced_config.scoring_rules.concern_keywords[:10])}

## KEYWORDS THAT SHOULD TRIGGER BONUSES
{', '.join(enhanced_config.scoring_rules.boost_keywords[:10])}
"""
    else:
        # Fallback for basic personalities
        personality_desc = ""
        if personality:
            personality_desc = f"You are a {personality.name}. {personality.description}"

        personality_section = f"""
## AGENT PERSPECTIVE
{personality_desc if personality_desc else "Balanced evaluation across all dimensions."}
"""

    # Build trigger explanation
    trigger_explanation = self._explain_trigger(trigger_reason)

    prompt = f"""
# SYSTEM 2 RE-EVALUATION WITH PERSONALITY CONTEXT

{personality_section}

---

## SYSTEM 1 EVALUATION (Initial VLM Perception)
The initial evaluation saw ONLY this image without context:

**Scores (1-10 scale):**
{json.dumps(system1_scores, indent=2)}

**Initial Reasoning:**
{json.dumps(system1_reasoning, indent=2)}

---

## CONTEXT FROM MEMORY

### SHORT-TERM MEMORY (Recent waypoints)
{stm_summary}

### LONG-TERM MEMORY (Similar situations)
{ltm_summary}

---

## YOUR TASK

Re-evaluate this image with your persona's priorities.

**This waypoint was triggered for deeper analysis because:**
{trigger_explanation}

**IMPORTANT INSTRUCTIONS:**
1. Your scores SHOULD DIFFER from System 1 based on your persona's priorities
2. Apply the specific adjustments listed above when you detect relevant features
3. Be MORE EXTREME than System 1 for dimensions you care about
4. Be LESS DETAILED about dimensions you don't prioritize
5. Expected adjustment range: ±0.5 to ±3.0 points per dimension

---

## OUTPUT FORMAT (JSON)

```json
{{
  "revised_scores": {{
    <dimension_id>: <float 1-10>,
    ...
  }},

  "adjustments": {{
    <dimension_id>: <delta from System 1>,
    ...
  }},

  "revision_reasoning": {{
    <dimension_id>: "<Why changed or kept same - reference your persona>",
    ...
  }},

  "detected_features": ["<feature1>", "<feature2>", ...],

  "interpretation": "<Overall understanding from your persona's perspective>",

  "memory_influence": {{
    "stm_impact": "<high/medium/low/none>",
    "ltm_impact": "<high/medium/low/none>",
    "personality_impact": "<high/medium/low> - SHOULD BE HIGH",
    "key_factors": ["<factor1>", "<factor2>"]
  }},

  "confidence": <float 0-1>,
  "pattern_detected": "<pattern or null>",
  "prediction": "<What to expect ahead>",
  "significance": "<high/medium/low>"
}}
````

**CRITICAL**: Your personality_impact should almost always be "high" - you are NOT a neutral observer.
"""
return prompt

````

### 4.4 Update `evaluate_with_context` to apply transformation

Modify the method to apply post-VLM transformation:

```python
def evaluate_with_context(
    self,
    waypoint_id: int,
    trigger_reason: TriggerReason,
    current_image_path: Path,
    system1_scores: Dict[str, float],
    system1_reasoning: Dict[str, str],
    stm_context: Dict[str, Any],
    ltm_patterns: List[Dict[str, Any]],
    personality: Optional[AgentPersonality] = None,
    route_metadata: Optional[Dict[str, Any]] = None,
) -> ThinkingResult:
    """Evaluate waypoint with full context and personality transformation."""
    import time
    start_time = time.time()

    if not self.enable_score_revision:
        return self._create_fallback_result(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            system1_scores=system1_scores,
            error="Score revision disabled",
        )

    try:
        # Build enhanced prompt
        prompt = self._build_vlm_revision_prompt(
            waypoint_id=waypoint_id,
            system1_scores=system1_scores,
            system1_reasoning=system1_reasoning,
            stm_context=stm_context,
            ltm_patterns=ltm_patterns,
            personality=personality,
            trigger_reason=trigger_reason,
        )

        # Call VLM
        vlm_response = self._call_vlm_with_image(
            image_path=current_image_path,
            prompt=prompt,
        )

        # Parse response
        parsed = self._parse_vlm_response(vlm_response)
        vlm_scores = parsed.get("revised_scores", system1_scores)
        vlm_reasoning = parsed.get("revision_reasoning", {})

        # =====================================================
        # NEW: Apply personality score transformation
        # =====================================================
        enhanced_config = self._get_enhanced_config(personality)
        transformation_metadata = {}

        if enhanced_config and self.dimension_mapping:
            transformer = ScoreTransformer(
                scoring_rules=enhanced_config.scoring_rules,
                dimension_mapping=self.dimension_mapping,
            )

            final_scores, transformation_metadata = transformer.transform(
                scores=vlm_scores,
                vlm_reasoning=vlm_reasoning,
            )

            self.logger.debug(
                "Applied personality transformation",
                waypoint_id=waypoint_id,
                personality=enhanced_config.personality_id,
                adjustments=transformation_metadata.get("total_adjustment_per_dim", {}),
            )
        else:
            final_scores = vlm_scores

        # Calculate total adjustments from System 1
        adjustments = {
            dim: final_scores[dim] - system1_scores[dim]
            for dim in system1_scores.keys()
        }

        # Build memory influence with transformation info
        memory_influence = parsed.get("memory_influence", {})
        memory_influence["transformation_applied"] = bool(enhanced_config)
        memory_influence["detected_features"] = (
            transformation_metadata.get("detected_features", []) +
            parsed.get("detected_features", [])
        )

        result = ThinkingResult(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            interpretation=parsed.get("interpretation", "No interpretation"),
            significance=parsed.get("significance", "medium"),
            pattern_detected=parsed.get("pattern_detected"),
            prediction=parsed.get("prediction"),
            recommendation=parsed.get("recommendation"),
            confidence=float(parsed.get("confidence", 0.7)),
            used_vlm=True,
            revised_scores=final_scores,  # Use transformed scores
            score_adjustments=adjustments,
            revision_reasoning=vlm_reasoning,
            memory_influence=memory_influence,
            used_stm_context=memory_influence.get("stm_impact", "none") != "none",
            used_ltm_patterns=memory_influence.get("ltm_impact", "none") != "none",
            personality_factor=memory_influence.get("personality_impact", "unknown"),
            vlm_model_used="qwen-vl-max",
            system1_scores=system1_scores.copy(),
            processing_time_seconds=time.time() - start_time,
        )

        self.thinking_history.append(result)

        self.logger.info(
            "System 2 evaluation complete",
            waypoint_id=waypoint_id,
            trigger=trigger_reason.value,
            adjustments={k: f"{v:+.1f}" for k, v in adjustments.items()},
            personality=enhanced_config.personality_id if enhanced_config else "basic",
            significance=result.significance,
        )

        return result

    except Exception as e:
        self.logger.error(
            "System 2 evaluation failed",
            waypoint_id=waypoint_id,
            error=str(e),
        )
        return self._create_fallback_result(
            waypoint_id=waypoint_id,
            trigger_reason=trigger_reason,
            system1_scores=system1_scores,
            error=str(e),
        )
````

---

## Step 5: Update WalkingAgent Integration

**File**: `src/agent/walking_agent.py`

Update the agent initialization to pass framework dimensions:

```python
def _initialize_thinking_module(self) -> ThinkingModule:
    """Initialize thinking module with framework dimensions."""
    return ThinkingModule(
        enable_score_revision=True,
        framework_dimensions=self.framework.get("dimensions", []),
    )
```

---

## Step 6: Update Constants for New Personalities

**File**: `src/agent/config/constants.py`

Add mapping from basic personality names to enhanced:

```python
# Mapping from basic personality presets to enhanced personalities
PERSONALITY_ENHANCEMENT_MAP = {
    "safety": "parent_with_kids",
    "scenic": "photographer",
    "balanced": "homebuyer",
    "comfort": "elderly_walker",
    "explorer": "photographer",
    "technical": None,  # No enhancement
}
```

---

## Step 7: Update `__init__.py` Files

**File**: `src/agent/config/__init__.py`

```python
from src.agent.config.personalities import (
    AgentPersonality,
    get_preset,
    list_presets,
    create_neutral_personality,
)
from src.agent.config.scoring_rules import (
    PersonalityScoringRules,
    EnhancedPersonalityConfig,
)
from src.agent.config.enhanced_personalities import (
    ENHANCED_PERSONALITIES,
    get_enhanced_personality,
    list_enhanced_personalities,
)

__all__ = [
    "AgentPersonality",
    "get_preset",
    "list_presets",
    "create_neutral_personality",
    "PersonalityScoringRules",
    "EnhancedPersonalityConfig",
    "ENHANCED_PERSONALITIES",
    "get_enhanced_personality",
    "list_enhanced_personalities",
]
```

**File**: `src/agent/capabilities/__init__.py`

```python
from src.agent.capabilities.score_transformer import (
    ScoreTransformer,
    create_dimension_mapping,
)

# Add to __all__
__all__ = [
    # ... existing exports ...
    "ScoreTransformer",
    "create_dimension_mapping",
]
```

---

## Step 8: Testing

Create a test file:

**File**: `tests/test_enhanced_personalities.py`

```python
"""Tests for enhanced personality system."""

import pytest

from src.agent.config.enhanced_personalities import (
    ENHANCED_PERSONALITIES,
    get_enhanced_personality,
    list_enhanced_personalities,
)
from src.agent.capabilities.score_transformer import (
    ScoreTransformer,
    create_dimension_mapping,
)


class TestEnhancedPersonalities:
    """Test enhanced personality configurations."""

    def test_list_personalities(self):
        """Should list all enhanced personalities."""
        personalities = list_enhanced_personalities()
        assert len(personalities) >= 5
        assert "homebuyer" in personalities
        assert "runner" in personalities
        assert "parent_with_kids" in personalities

    def test_get_personality(self):
        """Should retrieve enhanced personality by ID."""
        homebuyer = get_enhanced_personality("homebuyer")
        assert homebuyer.name == "Prospective Homebuyer"
        assert homebuyer.scoring_rules is not None
        assert len(homebuyer.scoring_rules.feature_modifiers) > 0

    def test_invalid_personality_raises(self):
        """Should raise ValueError for unknown personality."""
        with pytest.raises(ValueError):
            get_enhanced_personality("nonexistent")


class TestScoreTransformer:
    """Test score transformation logic."""

    @pytest.fixture
    def streetagent_5d_dimensions(self):
        """Sample StreetAgent 5D dimensions."""
        return [
            {"id": "spatial_sequence", "name_en": "Spatial Sequence"},
            {"id": "visual_coherence", "name_en": "Visual Coherence"},
            {"id": "sensory_complexity", "name_en": "Sensory Complexity"},
            {"id": "spatial_legibility", "name_en": "Spatial Legibility"},
            {"id": "functional_quality", "name_en": "Functional Quality"},
        ]

    @pytest.fixture
    def homebuyer_transformer(self, streetagent_5d_dimensions):
        """Create transformer for homebuyer personality."""
        config = get_enhanced_personality("homebuyer")
        mapping = create_dimension_mapping(streetagent_5d_dimensions)
        return ScoreTransformer(
            scoring_rules=config.scoring_rules,
            dimension_mapping=mapping,
        )

    def test_feature_detection(self, homebuyer_transformer):
        """Should detect features from reasoning text."""
        reasoning = {
            "functional_quality": "I can see several shops visible along the street."
        }
        transformed, metadata = homebuyer_transformer.transform(
            scores={"functional_quality": 6.0},
            vlm_reasoning=reasoning,
        )
        assert "shops_visible" in metadata["detected_features"]

    def test_positive_adjustment(self, homebuyer_transformer):
        """Should apply positive adjustment for valued features."""
        # Homebuyer values functional quality + shops
        scores = {
            "spatial_sequence": 6.0,
            "visual_coherence": 6.0,
            "sensory_complexity": 6.0,
            "spatial_legibility": 6.0,
            "functional_quality": 6.0,
        }
        reasoning = {
            "functional_quality": "Multiple shops and cafes visible, good amenities."
        }

        transformed, metadata = homebuyer_transformer.transform(scores, reasoning)

        # Functional quality should increase
        assert transformed["functional_quality"] > scores["functional_quality"]

    def test_negative_adjustment(self, homebuyer_transformer):
        """Should apply negative adjustment for concern features."""
        scores = {
            "spatial_sequence": 7.0,
            "visual_coherence": 7.0,
            "sensory_complexity": 7.0,
            "spatial_legibility": 7.0,
            "functional_quality": 7.0,
        }
        reasoning = {
            "spatial_legibility": "Heavy traffic on the main road, narrow sidewalk."
        }

        transformed, metadata = homebuyer_transformer.transform(scores, reasoning)

        # Should have negative adjustments
        total_adj = sum(metadata["total_adjustment_per_dim"].values())
        assert total_adj < 0  # Net negative

    def test_sensitivity_amplification(self, homebuyer_transformer):
        """Should amplify deviations for high-sensitivity dimensions."""
        # Start with extreme score
        scores = {
            "spatial_sequence": 3.0,  # Very low
            "visual_coherence": 5.5,  # Neutral
            "sensory_complexity": 5.5,
            "spatial_legibility": 5.5,
            "functional_quality": 8.0,  # High
        }

        transformed, metadata = homebuyer_transformer.transform(
            scores, vlm_reasoning={}
        )

        # High sensitivity dims should have larger changes
        # functional_quality has high sensitivity for homebuyer
        func_adj = metadata["sensitivity_adjustments"].get("functional_quality", 0)

        # The adjustment should be non-zero for non-neutral scores
        assert func_adj != 0 or scores["functional_quality"] == 5.5


class TestFrameworkAgnostic:
    """Test that system works with different frameworks."""

    def test_sagai_4d_mapping(self):
        """Should work with SAGAI 4D framework."""
        sagai_dims = [
            {"id": "safety", "name_en": "Safety"},
            {"id": "comfort", "name_en": "Comfort"},
            {"id": "interest", "name_en": "Interest"},
            {"id": "aesthetics", "name_en": "Aesthetics"},
        ]

        config = get_enhanced_personality("parent_with_kids")
        mapping = create_dimension_mapping(sagai_dims)
        transformer = ScoreTransformer(
            scoring_rules=config.scoring_rules,
            dimension_mapping=mapping,
        )

        scores = {"safety": 5.0, "comfort": 6.0, "interest": 7.0, "aesthetics": 7.0}
        reasoning = {"safety": "Heavy traffic, no crosswalk visible."}

        transformed, metadata = transformer.transform(scores, reasoning)

        # Safety should decrease significantly for parent
        assert transformed["safety"] < scores["safety"]

    def test_ewing_handy_5d_mapping(self):
        """Should work with Ewing & Handy 5D framework."""
        ewing_dims = [
            {"id": "imageability", "name_en": "Imageability"},
            {"id": "enclosure", "name_en": "Enclosure"},
            {"id": "human_scale", "name_en": "Human Scale"},
            {"id": "transparency", "name_en": "Transparency"},
            {"id": "complexity", "name_en": "Complexity"},
        ]

        config = get_enhanced_personality("photographer")
        mapping = create_dimension_mapping(ewing_dims)
        transformer = ScoreTransformer(
            scoring_rules=config.scoring_rules,
            dimension_mapping=mapping,
        )

        scores = {
            "imageability": 6.0,
            "enclosure": 6.0,
            "human_scale": 6.0,
            "transparency": 6.0,
            "complexity": 6.0,
        }
        reasoning = {
            "complexity": "Very monotonous scene, generic buildings."
        }

        transformed, metadata = transformer.transform(scores, reasoning)

        # Photographer should penalize monotonous scenes
        assert transformed["complexity"] < scores["complexity"]
```

Run tests:

```bash
cd city_walk_agent
pytest tests/test_enhanced_personalities.py -v
```

---

## Expected Results

After implementation, running the same route with different personalities should produce significantly different scores:

| Personality  | spatial_sequence | visual_coherence | sensory_complexity | spatial_legibility | functional_quality |
| ------------ | ---------------- | ---------------- | ------------------ | ------------------ | ------------------ |
| Neutral      | 6.5              | 7.0              | 6.0                | 7.5                | 6.5                |
| Homebuyer    | 5.8              | 6.5              | 5.2                | 7.0                | **8.2** (+1.7)     |
| Runner       | 5.0              | 6.8              | 4.5                | **8.5** (+1.0)     | 4.0 (-2.5)         |
| Parent       | **4.2** (-2.3)   | 6.0              | 4.0                | 7.8                | 7.0                |
| Photographer | 7.5              | **8.5** (+1.5)   | **8.0** (+2.0)     | 6.0                | 3.5 (-3.0)         |

The key changes:

- **Homebuyer**: Boosts functional_quality when shops detected
- **Runner**: Boosts spatial_legibility (clear paths), penalizes functional (shops irrelevant)
- **Parent**: Severely penalizes any safety concerns
- **Photographer**: Boosts complexity/aesthetics, ignores functional

---

## CLI Commands Summary

```bash
# Step 1: Create new files
touch src/agent/config/scoring_rules.py
touch src/agent/config/enhanced_personalities.py
touch src/agent/capabilities/score_transformer.py

# Step 2: Edit files (use your editor)
# - Copy code from this guide into each file

# Step 3: Update existing files
# - thinking.py: Add imports, update methods
# - walking_agent.py: Update initialization
# - constants.py: Add mapping
# - __init__.py files: Add exports

# Step 4: Run tests
cd city_walk_agent
pytest tests/test_enhanced_personalities.py -v

# Step 5: Test with real route
python -c "
from src.agent import WalkingAgent
from src.agent.config import get_enhanced_personality, list_enhanced_personalities

# List available enhanced personalities
print('Enhanced personalities:', list_enhanced_personalities())

# Create agents with different personalities and compare
for personality_id in ['homebuyer', 'runner', 'parent_with_kids']:
    print(f'\\n=== {personality_id.upper()} ===')
    config = get_enhanced_personality(personality_id)
    print(f'Description: {config.description}')
    print(f'VLM prompt preview: {config.vlm_persona_prompt[:200]}...')
"
```

This implementation maintains framework-agnostic design by using semantic keyword matching rather than hardcoded dimension IDs.
