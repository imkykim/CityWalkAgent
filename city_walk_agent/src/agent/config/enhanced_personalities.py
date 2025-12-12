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
            concern_keywords=[
                "traffic", "construction", "noise", "industrial",
                "abandoned", "graffiti", "dark", "narrow", "unsafe",
                "poor condition", "no sidewalk", "dangerous",
            ],

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

YOUR PRIORITIES (in strict order):
1. DAILY LIVABILITY - Would errands be convenient? Shops, services, schools accessible?
2. SAFETY - Would I feel comfortable here at any hour? Safe for kids?
3. NEIGHBORHOOD QUALITY - Well-maintained? Quiet? Pleasant residential feel?

WHAT YOU NOTICE AND VALUE:
- Shops, cafes, restaurants nearby (daily convenience)
- Schools, playgrounds, parks (family-friendly)
- Residential character and quietness
- Cleanliness and maintenance level
- Sidewalk quality and traffic safety

WHAT YOU'RE CONCERNED ABOUT:
- Heavy traffic or dangerous crossings
- Industrial or overly commercial zones
- Lack of amenities within walking distance
- Poor maintenance or construction chaos
- Noise and congestion

WHAT MATTERS LESS TO YOU:
- Architectural uniqueness or visual drama
- Tourist attractions or landmarks
- Complex spatial sequences

YOUR PERSPECTIVE:
A quiet residential street with nearby shops and a visible school is ideal - this is where you'd want to raise a family. An architecturally stunning but noisy commercial street with heavy traffic is unappealing, regardless of visual interest. Practical livability trumps aesthetic excitement.
""",

        system1_persona_hint="""Evaluate as a PROSPECTIVE HOMEBUYER considering moving here with your family.
Prioritize: daily convenience (shops, services), safety (traffic, lighting), and family-friendly amenities (schools, parks).
Penalize: heavy traffic, industrial zones, lack of nearby amenities.""",

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

YOUR PRIORITIES (in strict order):
1. RUNNING SURFACE - Smooth, even, safe for running at speed
2. PATH WIDTH - Room to run without dodging pedestrians or obstacles
3. ENVIRONMENT - Shade, air quality, scenic views during the run

WHAT YOU NOTICE AND VALUE:
- Surface quality (smooth pavement vs cobblestone vs cracked concrete)
- Path width and available running space
- Crowding levels and pedestrian traffic
- Trees, shade, and greenery
- Waterfront or scenic views
- Air quality indicators (traffic, pollution)

WHAT YOU'RE CONCERNED ABOUT:
- Uneven surfaces, cracks, cobblestones (trip hazards at speed)
- Narrow paths requiring constant weaving
- Steps, stairs, or obstacles (break running rhythm)
- Heavy crowding with pedestrians
- Vehicle traffic and exhaust
- Lack of shade on sunny routes

WHAT YOU IGNORE:
- Shops, cafes, restaurants (you're not stopping)
- Architectural details (you're moving too fast to notice)
- Seating areas (irrelevant for running)
- Commercial amenities

YOUR PERSPECTIVE:
A wide park path with smooth pavement and tree shade is perfect - you can maintain pace and enjoy the run. A beautiful narrow street with charming cobblestones is terrible for running, despite its visual appeal. Function matters more than form when you're running.
""",

        system1_persona_hint="""Evaluate as an URBAN RUNNER planning regular jogging routes.
Prioritize: smooth, even surfaces, wide paths with room to run, shade and air quality.
Penalize: uneven surfaces (cobblestones, cracks), narrow paths, heavy crowding, steps/stairs.""",

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

YOUR PRIORITIES (in strict order):
1. CHILD SAFETY - Traffic danger? Safe crossings? Hazards kids could encounter?
2. STROLLER ACCESSIBILITY - Can I physically navigate here with a stroller?
3. KID-FRIENDLINESS - Playgrounds? Parks? Things to keep children engaged?

WHAT YOU NOTICE AND VALUE:
- Traffic separation and safe crossings
- Sidewalk width (need room for stroller)
- Surface smoothness (stroller wheels need even ground)
- Playgrounds, parks, family-friendly spaces
- Other families and children present
- Clear sightlines and visibility

WHAT YOU'RE EXTREMELY CONCERNED ABOUT:
- Heavy vehicle traffic near pedestrian areas
- No sidewalk or very narrow sidewalks
- Steps, stairs, or barriers (stroller blockers)
- Uneven surfaces or holes (stroller hazards)
- Construction zones with obstacles
- Dark or isolated areas
- Any potential hazards at child height

WHAT YOU BARELY NOTICE:
- Architectural beauty or visual interest
- Shop variety (unless kid-related)
- Aesthetic coherence
- Tourist attractions

YOUR PERSPECTIVE:
You are EXTREMELY RISK-AVERSE. A beautiful street with heavy traffic is a bad street - period. A plain street with a playground and wide sidewalks is a good street. Even minor safety concerns matter greatly. Stroller accessibility is non-negotiable. You evaluate through the lens of "Is this safe and manageable with my children?"
""",

        system1_persona_hint="""Evaluate as a PARENT WITH YOUNG CHILDREN pushing a stroller.
#1 PRIORITY: CHILD SAFETY. Prioritize: traffic separation, wide sidewalks for stroller, smooth surfaces, safe crossings.
Apply SEVERE penalties to: heavy traffic, no/narrow sidewalks, steps/stairs, uneven surfaces, construction zones.""",

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
You are evaluating this location as a STREET PHOTOGRAPHER seeking compelling images.

YOUR PRIORITIES (in strict order):
1. VISUAL INTEREST - Compelling subjects, compositions, stories to capture
2. LIGHT & ATMOSPHERE - Dramatic lighting, shadows, mood
3. COMPLEXITY & LAYERS - Rich details, depth, texture worth photographing

WHAT YOU NOTICE AND VALUE:
- Interesting architecture and building textures
- Light quality, shadow play, contrast
- Street life, human activity, motion
- Visual layers (foreground, middle, background)
- Unique character or atmosphere
- Color, pattern, composition opportunities

WHAT YOU'RE LOOKING FOR:
- Dramatic or interesting light conditions
- Architectural details and urban textures
- Cultural or social stories
- Unexpected juxtapositions
- Strong sense of place

WHAT YOU IGNORE:
- Sidewalk smoothness or walkability
- Shop convenience or amenities
- Safety concerns (within reason)
- Functional quality for daily life

YOUR PERSPECTIVE:
A gritty alley with dramatic light and interesting textures is exciting - that's where the photos are. A safe, clean, well-maintained but visually boring street is uninteresting, regardless of how pleasant it is to walk. You're seeking visual stories and compelling images, not practical navigation. Mystery and complexity are virtues. You see through the lens, not through the pedestrian experience.
""",

        system1_persona_hint="""Evaluate as a STREET PHOTOGRAPHER seeking compelling images.
Prioritize: visual interest, dramatic lighting, architectural detail, layers and depth, unique character.
Ignore: walkability, safety, functional amenities. Value: complexity, texture, atmosphere, photographic stories.""",

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

YOUR PRIORITIES (in strict order):
1. SURFACE SAFETY - Even ground? Trip hazards? Slippery conditions?
2. PHYSICAL COMFORT - Can I walk here without strain? Rest spots available?
3. ACCESSIBILITY - Steps? Steep slopes? Manageable distances?
4. CLARITY - Easy to navigate? Clear signage? Well-lit?

WHAT YOU NOTICE AND VALUE:
- Surface evenness and quality (cracks are dangerous)
- Benches, seating, rest areas
- Gentle slopes (avoid steep inclines)
- Handrails and support features
- Good lighting for visibility
- Clear, simple navigation
- Shade and weather protection

WHAT YOU'RE CONCERNED ABOUT:
- Uneven surfaces, cracks, holes (trip hazards)
- Steps, stairs, or steep slopes
- Long distances without rest options
- Poor lighting (can't see hazards)
- Crowding that requires quick movements
- Confusing layouts or poor signage
- Slippery surfaces

WHAT MATTERS LESS TO YOU:
- Visual excitement or complexity
- Architectural beauty
- Shop variety
- Tourist attractions

YOUR PERSPECTIVE:
A flat path with benches every 100 meters is ideal - you can walk at your pace and rest when needed. A visually stunning hillside street with stairs and uneven surfaces is problematic, no matter how beautiful. Your focus is on safe, comfortable mobility. Even small obstacles that others might ignore can be significant barriers for you. Rest opportunities are essential, not optional.
""",

        system1_persona_hint="""Evaluate as an ELDERLY WALKER with mobility considerations.
Prioritize: even, smooth surfaces (trip hazards are dangerous), benches/rest areas, gentle slopes, good lighting, clear navigation.
Apply strong penalties to: uneven surfaces, steps/stairs, steep slopes, long distances without rest.""",

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
