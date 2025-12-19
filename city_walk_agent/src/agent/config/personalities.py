"""Agent personality configuration.

All personality definitions and utility functions consolidated in one place.

Main entry points:
- get_preset(): Get personality by ID
- list_presets(): List available personalities  
- create_neutral_personality(): Create baseline with equal weights

Usage:
    from src.agent.config import get_preset
    
    personality = get_preset("homebuyer", "sagai_2025")
    agent = WalkingAgent(personality=personality)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import load_framework


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class EnhancedPersonalityConfig:
    """Complete enhanced personality configuration.

    Attributes:
        personality_id: Unique identifier (e.g., "homebuyer", "runner")
        name: Human-readable name
        description: Brief description of persona behavior
        vlm_persona_prompt: Detailed VLM instruction for this persona
        system1_persona_hint: Short hint for System 1 evaluation
        thresholds: Decision thresholds (min_score, max_volatility, etc.)
        explanation_style: Output explanation style
    """

    personality_id: str
    name: str
    description: str
    vlm_persona_prompt: str
    system1_persona_hint: Optional[str] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    explanation_style: str = "balanced"


@dataclass
class AgentPersonality:
    """Runtime personality configuration for WalkingAgent.

    Attributes:
        name: Human-readable personality name
        description: Brief explanation of personality behavior
        decision_thresholds: Threshold values for decision-making
        explanation_style: Style of explanations (safety, scenic, balanced, technical)
        personality_id: Optional personality identifier
    """

    name: str
    description: str
    decision_thresholds: Dict[str, float]
    explanation_style: str
    personality_id: Optional[str] = None


# ============================================================================
# PERSONALITY DEFINITIONS
# ============================================================================

ENHANCED_PERSONALITIES: dict[str, EnhancedPersonalityConfig] = {

    # -------------------------------------------------------------------------
    # HOMEBUYER - Evaluating neighborhood for family residence
    # -------------------------------------------------------------------------
    "homebuyer": EnhancedPersonalityConfig(
        personality_id="homebuyer",
        name="Prospective Homebuyer",
        description="Evaluating neighborhood livability for family residence",

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


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


def list_enhanced_personalities() -> List[str]:
    """List available enhanced personality IDs."""
    return list(ENHANCED_PERSONALITIES.keys())


# Weight utility functions removed - no longer needed in dual evaluation system


# ============================================================================
# PUBLIC API
# ============================================================================


def get_preset(preset_name: str, framework_id: str) -> AgentPersonality:
    """Get a preset personality configuration.

    Args:
        preset_name: Personality ID (homebuyer, runner, parent_with_kids,
            photographer, elderly_walker)
        framework_id: Target evaluation framework identifier (for validation)

    Returns:
        AgentPersonality: Configured personality instance

    Raises:
        ValueError: If preset_name is invalid or framework_id not found

    Examples:
        >>> personality = get_preset("homebuyer", "sagai_2025")
        >>> personality.personality_id
        'homebuyer'

        >>> photo = get_preset("photographer", "streetagent_5d")
        >>> photo.personality_id
        'photographer'
    """
    # Get enhanced personality configuration
    enhanced_config = get_enhanced_personality(preset_name)

    # Load framework to validate it exists
    framework = load_framework(framework_id)
    if not framework.get("dimensions", []):
        raise ValueError(f"Framework '{framework_id}' has no dimensions defined")

    # Build personality instance (no weight computation needed)
    return AgentPersonality(
        name=enhanced_config.name,
        description=enhanced_config.description,
        decision_thresholds=enhanced_config.thresholds,
        explanation_style=enhanced_config.explanation_style,
        personality_id=preset_name,
    )


def list_presets() -> List[str]:
    """Return the list of available personality preset names.

    Returns:
        List[str]: Enhanced personality IDs that can be used with get_preset()

    Examples:
        >>> list_presets()
        ['homebuyer', 'runner', 'parent_with_kids', 'photographer', 'elderly_walker']
    """
    return list_enhanced_personalities()


def create_neutral_personality(framework_id: str) -> AgentPersonality:
    """Create a balanced personality for objective evaluation.

    This is useful for creating a baseline agent or for frameworks where
    no preset personalities are defined.

    Args:
        framework_id: Target evaluation framework identifier

    Returns:
        AgentPersonality: Neutral personality for objective evaluation

    Raises:
        ValueError: If framework_id not found

    Examples:
        >>> neutral = create_neutral_personality("sagai_2025")
        >>> neutral.personality_id
        'neutral'
    """
    # Load framework to validate it exists
    framework = load_framework(framework_id)
    if not framework.get("dimensions", []):
        raise ValueError(f"Framework '{framework_id}' has no dimensions defined")

    # Use default thresholds (framework-agnostic)
    decision_thresholds = {
        "min_overall_score": 6.5,
        "max_volatility": 2.0,
        "max_barriers": 2,
    }

    return AgentPersonality(
        name="Neutral Observer",
        description="Balanced evaluation across all dimensions",
        decision_thresholds=decision_thresholds,
        explanation_style="balanced",
        personality_id="neutral",
    )


# Alias for backward compatibility
PERSONALITIES = ENHANCED_PERSONALITIES
