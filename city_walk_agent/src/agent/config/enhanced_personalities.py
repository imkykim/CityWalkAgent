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
