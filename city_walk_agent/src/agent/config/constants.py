"""Essential constants for personality configuration."""

from src.config.constants import DEFAULT_FRAMEWORK_ID as GLOBAL_DEFAULT_FRAMEWORK_ID


# Default evaluation framework for all agents
DEFAULT_FRAMEWORK_ID = GLOBAL_DEFAULT_FRAMEWORK_ID

# Confidence calculation weights (used by thinking module)
CONFIDENCE_SCORE_WEIGHT = 0.5
CONFIDENCE_VOLATILITY_WEIGHT = 0.3
CONFIDENCE_BARRIER_WEIGHT = 0.2

# Thresholds for highlighting dimensions (used by thinking module)
EXCELLENT_SCORE_THRESHOLD = 7.5
LOW_VOLATILITY_THRESHOLD = 1.0

# Observation settings (used by observation module)
MAX_CONCURRENT_VLM_CALLS = 5
