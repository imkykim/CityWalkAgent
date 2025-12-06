"""Cognitive controller for visual change gating and System 2 triggering.

This class is the authoritative source for pHash computation and visual change detection.
It provides the detect_visual_change() method that ContinuousAnalyzer uses to determine
whether a waypoint represents a significant visual transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import imagehash
from PIL import Image

from src.utils.logging import get_logger


@dataclass
class VisualChangeResult:
    """Result of visual change detection for a single waypoint."""
    changed: bool
    phash_distance: Optional[float]
    should_trigger: bool
    reason: str


class CognitiveController:
    """Handle perceptual hashing and trigger decisions for System 2."""

    def __init__(self, phash_threshold: int = 10, hash_size: int = 8) -> None:
        """Initialize with pHash threshold for visual change detection."""
        self.phash_threshold = phash_threshold
        self.hash_size = hash_size
        self.last_phash: Any = None
        self.logger = get_logger(__name__ + ".CognitiveController")

    def detect_visual_change(
        self,
        image_path: Path,
        force: bool = False,
    ) -> VisualChangeResult:
        """Detect visual change between current image and previous waypoint.

        This is the authoritative method for visual change detection. It:
        1. Computes pHash for the current image
        2. Compares with the last waypoint's pHash
        3. Determines if visual change occurred based on threshold
        4. Updates internal state for next comparison

        Args:
            image_path: Path to the current waypoint image
            force: If True, always trigger (e.g., for first waypoint)

        Returns:
            VisualChangeResult with change detection outcome

        Example:
            ```python
            controller = CognitiveController(phash_threshold=15)
            result = controller.detect_visual_change(Path("waypoint_001.jpg"))
            if result.should_trigger:
                # Trigger System 2 thinking
                pass
            ```
        """
        if force:
            return VisualChangeResult(
                changed=True,
                phash_distance=None,
                should_trigger=True,
                reason="force"
            )

        if not image_path or not Path(image_path).exists():
            return VisualChangeResult(
                changed=False,
                phash_distance=None,
                should_trigger=False,
                reason="no_image"
            )

        try:
            current_hash = self._compute_phash(image_path)
        except Exception as e:
            self.logger.warning("Failed to compute pHash", error=str(e))
            return VisualChangeResult(
                changed=False,
                phash_distance=None,
                should_trigger=False,
                reason="phash_error"
            )

        # First image establishes baseline
        if self.last_phash is None:
            self.last_phash = current_hash
            return VisualChangeResult(
                changed=False,
                phash_distance=None,
                should_trigger=False,
                reason="baseline"
            )

        # Compute distance
        distance = float(self.last_phash - current_hash)

        self.logger.debug(
            "pHash distance computed",
            distance=distance,
            threshold=self.phash_threshold,
        )

        # Always update last_phash to compare consecutive images
        self.last_phash = current_hash

        # Determine if change is significant
        changed = distance >= self.phash_threshold

        return VisualChangeResult(
            changed=changed,
            phash_distance=distance,
            should_trigger=changed,
            reason="visual_change" if changed else "no_change"
        )

    def should_trigger_thinking(
        self,
        current_image: Path,
        waypoint: Any,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """Return (should_trigger, reason) based on perceptual change.

        This method wraps detect_visual_change() for backward compatibility.
        New code should use detect_visual_change() directly.

        Args:
            current_image: Path to current waypoint image
            waypoint: Waypoint object (for logging only)
            force: Force trigger regardless of visual change

        Returns:
            Tuple of (should_trigger, reason)
        """
        result = self.detect_visual_change(current_image, force=force)
        return result.should_trigger, result.reason

    def _compute_phash(self, image: Path):
        """Compute perceptual hash for an image."""
        with Image.open(image) as img:
            return imagehash.phash(img, hash_size=self.hash_size)

    def reset(self) -> None:
        """Reset state for new route."""
        self.last_phash = None
