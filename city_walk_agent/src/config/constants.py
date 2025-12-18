"""Project-wide constants for CityWalkAgent configuration."""

from typing import Final

# API Configuration
DEFAULT_VLM_TIMEOUT: Final[int] = 60  # seconds
DEFAULT_MAX_CONCURRENT: Final[int] = 5
DEFAULT_RETRY_ATTEMPTS: Final[int] = 3

# Route Configuration
DEFAULT_SAMPLING_INTERVAL: Final[int] = 30  # meters
MIN_SAMPLING_INTERVAL: Final[int] = 5
MAX_SAMPLING_INTERVAL: Final[int] = 50

# VLM Processing
OPTIMAL_BATCH_SIZE: Final[int] = 32  # images per batch
MAX_IMAGE_SIZE_MB: Final[int] = 5

# Scoring
MIN_SCORE: Final[int] = 1
MAX_SCORE: Final[int] = 10
SCORE_DECIMAL_PLACES: Final[int] = 1
DEFAULT_FRAMEWORK_ID: Final[str] = "streetagent_5d"

# Multi-Image Evaluation
DEFAULT_PHASH_THRESHOLD: Final[float] = 15.0  # Visual change detection and multi-image trigger threshold
DEFAULT_ENABLE_MULTI_IMAGE: Final[bool] = True  # Enable multi-image evaluation
DEFAULT_CONTEXT_WINDOW: Final[int] = 3  # Waypoint context window size
