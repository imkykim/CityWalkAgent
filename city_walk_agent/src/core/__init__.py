"""Core infrastructure shared by System 1 and System 2."""

from src.core.constants import (
    DEFAULT_FRAMEWORK_ID,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_SAMPLING_INTERVAL,
    DEFAULT_VLM_TIMEOUT,
    DEFAULT_PHASH_THRESHOLD,
    DEFAULT_ENABLE_MULTI_IMAGE,
    DEFAULT_CONTEXT_WINDOW,
    MAX_IMAGE_SIZE_MB,
    MAX_SCORE, MIN_SCORE,
    MIN_SAMPLING_INTERVAL,
    OPTIMAL_BATCH_SIZE,
    SCORE_DECIMAL_PLACES,
)
from src.core.frameworks.manager import (
    FrameworkManager,
    get_framework_manager,
    load_framework,
    list_frameworks,
)
from src.core.settings import settings

__all__ = [
    "DEFAULT_FRAMEWORK_ID", "DEFAULT_MAX_CONCURRENT", "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_SAMPLING_INTERVAL", "DEFAULT_VLM_TIMEOUT", "DEFAULT_PHASH_THRESHOLD",
    "DEFAULT_ENABLE_MULTI_IMAGE", "DEFAULT_CONTEXT_WINDOW",
    "MAX_IMAGE_SIZE_MB", "MAX_SCORE", "MIN_SCORE",
    "MIN_SAMPLING_INTERVAL", "OPTIMAL_BATCH_SIZE", "SCORE_DECIMAL_PLACES",
    "FrameworkManager", "get_framework_manager", "load_framework", "list_frameworks",
    "settings",
]
