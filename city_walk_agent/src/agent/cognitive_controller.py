"""Cognitive controller for visual change gating and System 2 triggering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import imagehash
from PIL import Image

from src.utils.logging import get_logger


class CognitiveController:
    """Handle perceptual hashing and trigger decisions for System 2."""

    def __init__(self, phash_threshold: int = 10, hash_size: int = 8) -> None:
        """Initialize with pHash threshold for visual change detection."""
        self.phash_threshold = phash_threshold
        self.hash_size = hash_size
        self.last_phash: Any = None
        self.logger = get_logger(__name__ + ".CognitiveController")

    def should_trigger_thinking(
        self,
        current_image: Path,
        waypoint: Any,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """Return (should_trigger, reason) based on perceptual change."""
        if force:
            return True, "force"

        if not current_image or not Path(current_image).exists():
            return False, "no_image"

        try:
            current_hash = self._compute_phash(current_image)
        except Exception as e:
            self.logger.warning("Failed to compute pHash", error=str(e))
            return False, "phash_error"

        # First image establishes baseline
        if self.last_phash is None:
            self.last_phash = current_hash
            return False, "baseline"

        distance = self.last_phash - current_hash
        self.logger.debug(
            "pHash distance computed",
            waypoint_id=getattr(waypoint, "waypoint_id", None),
            distance=distance,
            threshold=self.phash_threshold,
        )

        if distance >= self.phash_threshold:
            self.last_phash = current_hash
            return True, "visual_change"

        return False, "no_change"

    def _compute_phash(self, image: Path):
        """Compute perceptual hash for an image."""
        with Image.open(image) as img:
            return imagehash.phash(img, hash_size=self.hash_size)

    def reset(self) -> None:
        """Reset state for new route."""
        self.last_phash = None
