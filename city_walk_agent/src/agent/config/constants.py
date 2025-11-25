"""Agent defaults, thresholds, and personality presets.

Design principle:
- Never hardcode dimension names; always read from the active framework.

The module provides two configuration styles:
1. Framework-specific presets for known evaluation schemas.
2. Semantic personalities that adapt via keyword mapping.
"""

from typing import Any, Dict, List

from src.config.constants import DEFAULT_FRAMEWORK_ID as GLOBAL_DEFAULT_FRAMEWORK_ID


# ============================================================================
# FRAMEWORK & EVALUATION
# ============================================================================

# Default evaluation framework for all agents
DEFAULT_FRAMEWORK_ID = GLOBAL_DEFAULT_FRAMEWORK_ID

# Dimension validation - framework must have at least this many
MIN_DIMENSIONS_REQUIRED = 2
MAX_DIMENSIONS_SUPPORTED = 20


# ============================================================================
# AGENT DEFAULTS
# ============================================================================

# Default personality when none specified
DEFAULT_PERSONALITY = "balanced"

# Default agent behavior
ENABLE_MEMORY_BY_DEFAULT = True
MAX_MEMORY_ENTRIES = 1000
MAX_CONCURRENT_VLM_CALLS = 5


# ============================================================================
# DECISION THRESHOLDS (Framework-agnostic defaults)
# ============================================================================

# Minimum overall weighted score to accept route (0-10 scale)
DEFAULT_MIN_OVERALL_SCORE = 6.5

# Maximum acceptable volatility (standard deviation of scores)
DEFAULT_MAX_VOLATILITY = 2.0

# Maximum number of hidden barriers before rejecting route
DEFAULT_MAX_BARRIERS = 2

# Minimum score for ANY dimension when it's heavily weighted (weight > 1.5)
# This is a fallback - specific thresholds can be defined per dimension
DEFAULT_MIN_IMPORTANT_DIMENSION_SCORE = 7.0


# ============================================================================
# DIMENSION WEIGHTS (Framework-agnostic)
# ============================================================================

# Neutral weight value (applied to all dimensions equally)
WEIGHT_NEUTRAL_VALUE = 1.0

# Weight ranges for validation
MIN_WEIGHT = 0.1  # Dimension can't be ignored completely
MAX_WEIGHT = 3.0  # Dimension can't dominate too much

# Weight importance thresholds
WEIGHT_THRESHOLD_IMPORTANT = 1.5  # Above this = important dimension
WEIGHT_THRESHOLD_PRIMARY = 2.0  # Above this = primary focus


# ============================================================================
# PERSONALITY CONFIGURATION - APPROACH 1: FRAMEWORK-SPECIFIC PRESETS
# ============================================================================

# Explicit weights for known frameworks
# Format: {personality_type: {framework_id: config}}

FRAMEWORK_SPECIFIC_PERSONALITIES: Dict[str, Dict[str, Any]] = {
    # Safety-focused personalities by framework
    "safety": {
        "sagai_2025": {
            "name": "Safety Guardian",
            "description": "Prioritizes pedestrian safety",
            "weights": {
                "safety": 2.0,
                "comfort": 1.2,
                "interest": 0.8,
                "aesthetics": 0.6,
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "min_safety_score": 7.0,
                "max_volatility": 2.0,
                "max_barriers": 1,
            },
            "explanation_style": "safety",
        },
        "streetagent_5d": {
            "name": "Safety Guardian",
            "description": "Prioritizes pedestrian safety and functional quality",
            "weights": {
                "functional_quality": 2.2,    # Primary: sidewalk condition, hazards
                "spatial_legibility": 1.5,    # Secondary: clear wayfinding = safety
                "visual_coherence": 1.2,      # Tertiary: organized environment
                "sensory_complexity": 0.7,    # Low: overstimulation can reduce safety awareness
                "spatial_sequence": 0.6,      # Low: sequence less critical than function
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "min_functional_quality": 7.5,  # High threshold for safety
                "min_spatial_legibility": 7.0,
                "max_volatility": 1.5,
                "max_barriers": 1,
            },
            "explanation_style": "safety",
        },
    },
    # Scenic-focused personalities by framework
    "scenic": {
        "sagai_2025": {
            "name": "Scenic Explorer",
            "description": "Prioritizes beauty and landmarks",
            "weights": {
                "aesthetics": 2.0,
                "interest": 1.8,
                "comfort": 1.0,
                "safety": 1.2,
            },
            "thresholds": {
                "min_overall_score": 6.0,
                "min_aesthetics_score": 6.5,
                "min_interest_score": 6.0,
                "max_volatility": 2.5,
                "max_barriers": 2,
            },
            "explanation_style": "scenic",
        },
        "streetagent_5d": {
            "name": "Aesthetic Explorer",
            "description": "Prioritizes visual beauty and sensory richness",
            "weights": {
                "sensory_complexity": 2.0,     # Primary: rich visual stimuli
                "visual_coherence": 1.8,       # Secondary: harmonious composition
                "spatial_sequence": 1.5,       # Tertiary: interesting unfolding
                "spatial_legibility": 1.0,     # Moderate: some mystery is OK
                "functional_quality": 1.2,     # Moderate: beauty with comfort
            },
            "thresholds": {
                "min_overall_score": 6.0,
                "min_sensory_complexity": 6.5,
                "min_visual_coherence": 6.5,
                "max_volatility": 2.5,
                "max_barriers": 2,
            },
            "explanation_style": "scenic",
        },
    },
    # Balanced personalities by framework
    "balanced": {
        "sagai_2025": {
            "name": "Balanced Navigator",
            "description": "Balances all dimensions equally",
            "weights": {
                "safety": 1.0,
                "comfort": 1.0,
                "interest": 1.0,
                "aesthetics": 1.0,
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "max_volatility": 2.0,
                "max_barriers": 2,
            },
            "explanation_style": "balanced",
        },
        "streetagent_5d": {
            "name": "Balanced Navigator",
            "description": "Equal consideration across all dimensions",
            "weights": {
                "functional_quality": 1.0,
                "spatial_legibility": 1.0,
                "visual_coherence": 1.0,
                "sensory_complexity": 1.0,
                "spatial_sequence": 1.0,
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "max_volatility": 2.0,
                "max_barriers": 2,
            },
            "explanation_style": "balanced",
        },
    },
    # Comfort-focused personalities by framework
    "comfort": {
        "sagai_2025": {
            "name": "Comfort Seeker",
            "description": "Prioritizes smooth walking conditions",
            "weights": {
                "comfort": 2.0,
                "safety": 1.5,
                "aesthetics": 0.8,
                "interest": 0.7,
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "min_comfort_score": 7.0,
                "max_volatility": 1.5,
                "max_barriers": 1,
            },
            "explanation_style": "balanced",
        },
        "streetagent_5d": {
            "name": "Comfort Seeker",
            "description": "Prioritizes ease, amenities, and smooth experience",
            "weights": {
                "functional_quality": 2.0,     # Primary: amenities, width, surface
                "visual_coherence": 1.5,       # Secondary: calm, organized
                "spatial_sequence": 1.2,       # Tertiary: gentle rhythm
                "spatial_legibility": 1.3,     # Moderate: avoid confusion stress
                "sensory_complexity": 0.8,     # Low: avoid overstimulation
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "min_functional_quality": 7.0,
                "max_volatility": 1.5,
                "max_barriers": 1,
            },
            "explanation_style": "balanced",
        },
    },
    # Explorer personalities by framework
    "explorer": {
        "sagai_2025": {
            "name": "Urban Explorer",
            "description": "Seeks interesting experiences",
            "weights": {
                "interest": 2.0,
                "aesthetics": 1.5,
                "comfort": 1.0,
                "safety": 1.3,
            },
            "thresholds": {
                "min_overall_score": 6.0,
                "min_interest_score": 6.5,
                "max_volatility": 3.0,
                "max_barriers": 2,
            },
            "explanation_style": "balanced",
        },
        "streetagent_5d": {
            "name": "Urban Explorer",
            "description": "Seeks discovery, variety, and engaging sequences",
            "weights": {
                "spatial_sequence": 2.0,       # Primary: unfolding discovery
                "sensory_complexity": 1.8,     # Secondary: richness and variety
                "spatial_legibility": 1.0,     # Moderate: some mystery
                "visual_coherence": 1.2,       # Moderate: too much order is boring
                "functional_quality": 1.3,     # Moderate: needs to be walkable
            },
            "thresholds": {
                "min_overall_score": 6.0,
                "min_spatial_sequence": 6.5,
                "min_sensory_complexity": 6.5,
                "max_volatility": 3.0,
                "max_barriers": 2,
            },
            "explanation_style": "balanced",
        },
    },
    # Technical analysis personalities by framework
    "technical": {
        "sagai_2025": {
            "name": "Technical Analyst",
            "description": "Detailed technical analysis",
            "weights": {
                "safety": 1.0,
                "comfort": 1.0,
                "interest": 1.0,
                "aesthetics": 1.0,
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "max_volatility": 2.0,
                "max_barriers": 2,
            },
            "explanation_style": "technical",
        },
        "streetagent_5d": {
            "name": "Technical Analyst",
            "description": "Comprehensive evaluation across all dimensions",
            "weights": {
                "functional_quality": 1.0,
                "spatial_legibility": 1.0,
                "visual_coherence": 1.0,
                "sensory_complexity": 1.0,
                "spatial_sequence": 1.0,
            },
            "thresholds": {
                "min_overall_score": 6.5,
                "max_volatility": 2.0,
                "max_barriers": 2,
            },
            "explanation_style": "technical",
        },
    },
}


# ============================================================================
# PERSONALITY CONFIGURATION - APPROACH 2: SEMANTIC MAPPING
# ============================================================================

# Intent-based personality definitions that adapt to any framework
# These use semantic tags that get mapped to actual dimensions at runtime

SEMANTIC_PERSONALITIES: Dict[str, Dict[str, Any]] = {
    "safety": {
        "name": "Safety Guardian",
        "description": "Prioritizes pedestrian safety",
        "semantic_config": {
            "primary_focus": ["safety", "安全性", "安全", "secure", "protection"],
            "secondary_focus": ["comfort", "舒适度", "舒适", "ease", "convenience"],
            "tertiary_focus": ["interest", "趣味性", "engagement"],
            "low_priority": ["aesthetics", "美观性", "美学", "visual", "beauty"],
        },
        "weight_distribution": {
            "primary": 2.0,
            "secondary": 1.2,
            "tertiary": 0.8,
            "low_priority": 0.6,
            "unmapped": 1.0,  # Dimensions that don't match any keyword
        },
        "thresholds": {
            "min_overall_score": 6.5,
            "min_primary_score": 7.0,  # Min score for primary focus dimensions
            "max_volatility": 2.0,
            "max_barriers": 1,
        },
        "explanation_style": "safety",
    },
    "scenic": {
        "name": "Scenic Explorer",
        "description": "Prioritizes beauty and landmarks",
        "semantic_config": {
            "primary_focus": [
                "aesthetics",
                "美观性",
                "美学",
                "visual",
                "beauty",
                "景观",
            ],
            "secondary_focus": [
                "interest",
                "趣味性",
                "points_of_interest",
                "landmarks",
                "景点",
            ],
            "tertiary_focus": ["comfort", "舒适度", "ease"],
            "low_priority": [],
        },
        "weight_distribution": {
            "primary": 2.0,
            "secondary": 1.8,
            "tertiary": 1.0,
            "low_priority": 0.8,
            "unmapped": 1.2,
        },
        "thresholds": {
            "min_overall_score": 6.0,
            "min_primary_score": 6.5,
            "min_secondary_score": 6.0,
            "max_volatility": 2.5,
            "max_barriers": 2,
        },
        "explanation_style": "scenic",
    },
    "balanced": {
        "name": "Balanced Navigator",
        "description": "Balances all dimensions equally",
        "semantic_config": {
            "primary_focus": [],  # Empty = all dimensions equally
            "secondary_focus": [],
            "tertiary_focus": [],
            "low_priority": [],
        },
        "weight_distribution": {
            "primary": 1.0,
            "secondary": 1.0,
            "tertiary": 1.0,
            "low_priority": 1.0,
            "unmapped": 1.0,
        },
        "thresholds": {
            "min_overall_score": 6.5,
            "max_volatility": 2.0,
            "max_barriers": 2,
        },
        "explanation_style": "balanced",
    },
    "comfort": {
        "name": "Comfort Seeker",
        "description": "Prioritizes smooth walking",
        "semantic_config": {
            "primary_focus": ["comfort", "舒适度", "ease", "smooth", "便利"],
            "secondary_focus": ["safety", "安全性", "secure"],
            "tertiary_focus": [],
            "low_priority": ["interest", "趣味性"],
        },
        "weight_distribution": {
            "primary": 2.0,
            "secondary": 1.5,
            "tertiary": 1.0,
            "low_priority": 0.7,
            "unmapped": 0.8,
        },
        "thresholds": {
            "min_overall_score": 6.5,
            "min_primary_score": 7.0,
            "max_volatility": 1.5,
            "max_barriers": 1,
        },
        "explanation_style": "balanced",
    },
    "explorer": {
        "name": "Urban Explorer",
        "description": "Seeks interesting experiences",
        "semantic_config": {
            "primary_focus": ["interest", "趣味性", "engagement", "variety", "探索"],
            "secondary_focus": ["aesthetics", "美观性", "beauty", "景观"],
            "tertiary_focus": ["comfort", "舒适度"],
            "low_priority": [],
        },
        "weight_distribution": {
            "primary": 2.0,
            "secondary": 1.5,
            "tertiary": 1.0,
            "low_priority": 0.8,
            "unmapped": 1.3,
        },
        "thresholds": {
            "min_overall_score": 6.0,
            "min_primary_score": 6.5,
            "max_volatility": 3.0,
            "max_barriers": 2,
        },
        "explanation_style": "balanced",
    },
    "technical": {
        "name": "Technical Analyst",
        "description": "Detailed analysis",
        "semantic_config": {
            "primary_focus": [],
            "secondary_focus": [],
            "tertiary_focus": [],
            "low_priority": [],
        },
        "weight_distribution": {
            "primary": 1.0,
            "secondary": 1.0,
            "tertiary": 1.0,
            "low_priority": 1.0,
            "unmapped": 1.0,
        },
        "thresholds": {
            "min_overall_score": 6.5,
            "max_volatility": 2.0,
            "max_barriers": 2,
        },
        "explanation_style": "technical",
    },
}


# ============================================================================
# EXPLANATION STYLES
# ============================================================================

EXPLANATION_STYLES = ["safety", "scenic", "balanced", "technical"]
DEFAULT_EXPLANATION_STYLE = "balanced"


# ============================================================================
# OBSERVATION (PERCEPTION) SETTINGS
# ============================================================================

OBSERVATION_BATCH_SIZE = 32

# TODO: Implement timeout logic for observation operations
# OBSERVATION_TIMEOUT_SECONDS = 30

# TODO: Implement retry logic for failed observations
# OBSERVATION_RETRY_ATTEMPTS = 3


# ============================================================================
# THINKING (REASONING) SETTINGS
# ============================================================================

# Confidence calculation weights
CONFIDENCE_SCORE_WEIGHT = 0.5
CONFIDENCE_VOLATILITY_WEIGHT = 0.3
CONFIDENCE_BARRIER_WEIGHT = 0.2

# Thresholds for highlighting dimensions
EXCELLENT_SCORE_THRESHOLD = 7.5
LOW_VOLATILITY_THRESHOLD = 1.0


# ============================================================================
# MEMORY SETTINGS
# ============================================================================

MEMORY_FILE_FORMAT = "jsonl"
MEMORY_INDEX_FORMAT = "json"

# TODO: Implement file size monitoring and rotation
# MEMORY_MAX_FILE_SIZE_MB = 100

# TODO: Implement automatic cleanup of old memory entries
# MEMORY_AUTO_CLEANUP_ENABLED = False
