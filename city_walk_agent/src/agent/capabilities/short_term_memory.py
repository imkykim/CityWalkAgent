"""Short-term memory for real-time route traversal context.

Maintains a sliding window of recent waypoint observations to support
adaptive decision-making during route execution. Unlike LongTermMemory which
provides persistent storage, ShortTermMemory is ephemeral and designed
for temporal reasoning within a single route traversal session.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    """Single waypoint memory entry with evaluation results and metadata."""

    waypoint_id: int
    scores: Dict[str, float]  # {dimension_id: score}
    summary: str
    image_path: Optional[Path]
    gps: Tuple[float, float]
    timestamp: str


class ShortTermMemory:
    """Sliding window memory for recent waypoint observations.

    Maintains temporal context during route traversal by storing the most
    recent N waypoints with their evaluation scores, summaries, and metadata.
    Provides trend analysis and context summarization for adaptive agent
    decision-making.

    Key differences from LongTermMemory:
    - Ephemeral: Does not persist to disk
    - Fixed size: Uses circular buffer (deque) with max size
    - Real-time: Designed for in-flight route decisions
    - Analytical: Computes trends and patterns across recent history

    Usage::

        memory = ShortTermMemory(window_size=5)

        # During route traversal
        memory.add(
            waypoint_id=0,
            scores={"safety": 8.5, "comfort": 7.2},
            summary="Wide sidewalk, good lighting",
            image_path=Path("waypoint_0.jpg"),
            gps=(37.7749, -122.4194),
            timestamp="2024-01-01T12:00:00"
        )

        # Get context for decision-making
        context = memory.get_context()
        trend = memory.get_summary_text()

        # Clear for next route
        memory.clear()
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize short-term memory with fixed window size.

        Args:
            window_size: Maximum number of recent waypoints to retain.
                Older waypoints are automatically evicted when capacity is reached.

        Side Effects:
            Initializes an empty circular buffer (deque) for memory storage.
        """
        self.window_size = window_size
        self._memory: deque[MemoryItem] = deque(maxlen=window_size)

        logger.info(
            "ShortTermMemory initialized",
            window_size=window_size
        )

    @property
    def memory(self) -> deque[MemoryItem]:
        """Direct access to the underlying memory buffer (ordered oldest→newest)."""
        return self._memory

    def add(
        self,
        waypoint_id: int,
        scores: Dict[str, float],
        summary: str,
        image_path: Optional[Path] = None,
        gps: Optional[Tuple[float, float]] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """Add a new waypoint observation to memory.

        Automatically evicts the oldest waypoint if window is at capacity.

        Args:
            waypoint_id: Sequential waypoint identifier (0-indexed).
            scores: Evaluation scores keyed by dimension ID.
            summary: Human-readable summary of waypoint characteristics.
            image_path: Optional path to waypoint street view image.
            gps: Optional (latitude, longitude) tuple.
            timestamp: Optional ISO format timestamp. Defaults to current time.

        Side Effects:
            Appends to internal deque, potentially evicting oldest item.
            Logs addition at debug level.
        """
        # Default values
        if gps is None:
            gps = (0.0, 0.0)
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        item = MemoryItem(
            waypoint_id=waypoint_id,
            scores=scores,
            summary=summary,
            image_path=image_path,
            gps=gps,
            timestamp=timestamp
        )

        self._memory.append(item)

        logger.debug(
            "Waypoint added to short-term memory",
            waypoint_id=waypoint_id,
            memory_size=len(self._memory),
            avg_score=sum(scores.values()) / len(scores) if scores else 0.0
        )

    def get_context(self) -> Dict:
        """Get complete context from recent waypoint history.

        Returns structured context suitable for agent decision-making,
        including score history, summaries, trend analysis, and metadata.

        Returns:
            Dictionary containing:
            - recent_scores: List of score dicts per waypoint (oldest first)
            - recent_summaries: List of summary strings per waypoint
            - trend: Trend classification ("improving"|"declining"|"stable"|"volatile")
            - image_paths: List of image paths (oldest first)
            - waypoint_ids: List of waypoint IDs (oldest first)
            - gps_coordinates: List of (lat, lon) tuples (oldest first)
            - window_size: Current memory occupancy

            Empty dict if no waypoints in memory.
        """
        if not self._memory:
            logger.debug("get_context called on empty memory")
            return {}

        context = {
            "recent_scores": [item.scores for item in self._memory],
            "recent_summaries": [item.summary for item in self._memory],
            "trend": self._compute_trend(),
            "image_paths": [item.image_path for item in self._memory],
            "waypoint_ids": [item.waypoint_id for item in self._memory],
            "gps_coordinates": [item.gps for item in self._memory],
            "window_size": len(self._memory)
        }

        logger.debug(
            "Context retrieved",
            window_size=context["window_size"],
            trend=context["trend"]
        )

        return context

    def get_score_trends(self) -> Dict[str, List[float]]:
        """Get per-dimension score trends across recent waypoints.

        Useful for understanding how individual dimensions (safety, comfort,
        accessibility, etc.) are evolving during route traversal.

        Returns:
            Dictionary mapping dimension ID to list of scores (oldest first).
            Empty dict if no waypoints in memory.

        Example::

            trends = memory.get_score_trends()
            # {"safety": [8.0, 8.2, 7.5], "comfort": [7.0, 7.5, 8.0]}
        """
        if not self._memory:
            return {}

        # Collect all unique dimension IDs
        all_dimensions = set()
        for item in self._memory:
            all_dimensions.update(item.scores.keys())

        # Build trend lists per dimension
        trends = {}
        for dimension in all_dimensions:
            trends[dimension] = [
                item.scores.get(dimension, 0.0)
                for item in self._memory
            ]

        logger.debug(
            "Score trends retrieved",
            dimensions=len(trends),
            window_size=len(self._memory)
        )

        return trends

    def clear(self) -> None:
        """Clear all waypoints from memory.

        Typically called when starting a new route traversal session.

        Side Effects:
            Removes all items from the internal deque.
        """
        previous_size = len(self._memory)
        self._memory.clear()

        logger.info(
            "Short-term memory cleared",
            previous_size=previous_size
        )

    def get_summary_text(self) -> str:
        """Generate natural language summary for LLM consumption.

        Produces a concise textual summary of recent waypoint history,
        including trend analysis and key observations. Designed to be
        included in LLM prompts for context-aware decision-making.

        Returns:
            Human-readable summary string describing recent waypoint history.
            Returns "No recent waypoint history." if memory is empty.

        Example Output::

            Recent waypoint history (last 3 waypoints):
            - Waypoint 0: Wide sidewalk, good lighting (avg: 7.8/10)
            - Waypoint 1: Narrow path, heavy traffic (avg: 5.2/10)
            - Waypoint 2: Park entrance, very pleasant (avg: 9.1/10)

            Trend: improving (scores increased by 1.3 points)
        """
        if not self._memory:
            return "No recent waypoint history."

        # Build waypoint summaries
        lines = [f"Recent waypoint history (last {len(self._memory)} waypoints):"]

        for item in self._memory:
            avg_score = sum(item.scores.values()) / len(item.scores) if item.scores else 0.0
            lines.append(
                f"- Waypoint {item.waypoint_id}: {item.summary} "
                f"(avg: {avg_score:.1f}/10)"
            )

        # Add trend analysis
        trend = self._compute_trend()
        trend_explanation = self._get_trend_explanation(trend)
        lines.append(f"\nTrend: {trend_explanation}")

        summary_text = "\n".join(lines)

        logger.debug(
            "Summary text generated",
            length=len(summary_text),
            trend=trend
        )

        return summary_text

    def _compute_trend(self) -> Literal["improving", "declining", "stable", "volatile"]:
        """Classify the score trend across recent waypoints.

        Analyzes average scores per waypoint to determine if the route
        experience is improving, declining, stable, or volatile.

        Algorithm:
        1. Calculate average score per waypoint
        2. Compare first half vs second half means
        3. Check standard deviation for volatility (>1.5 = volatile)
        4. Classify based on difference:
           - improving: diff > 0.5
           - declining: diff < -0.5
           - stable: |diff| <= 0.5
           - volatile: std > 1.5 (overrides other classifications)

        Returns:
            Trend classification: "improving", "declining", "stable", or "volatile"
            Returns "stable" if insufficient data (< 2 waypoints).
        """
        if len(self._memory) < 2:
            return "stable"

        # Calculate average score per waypoint
        waypoint_averages = []
        for item in self._memory:
            if item.scores:
                avg = sum(item.scores.values()) / len(item.scores)
                waypoint_averages.append(avg)

        if len(waypoint_averages) < 2:
            return "stable"

        # Check for volatility (high standard deviation)
        try:
            std_dev = statistics.stdev(waypoint_averages)
            if std_dev > 1.5:
                logger.debug(
                    "Volatile trend detected",
                    std_dev=std_dev,
                    waypoint_averages=waypoint_averages
                )
                return "volatile"
        except statistics.StatisticsError:
            # Single value or empty list
            return "stable"

        # Compare first half vs second half
        mid = len(waypoint_averages) // 2
        first_half = waypoint_averages[:mid]
        second_half = waypoint_averages[mid:]

        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        difference = second_mean - first_mean

        logger.debug(
            "Trend computed",
            first_mean=first_mean,
            second_mean=second_mean,
            difference=difference,
            std_dev=std_dev if len(waypoint_averages) > 1 else 0.0
        )

        # Classify trend
        if difference > 0.5:
            return "improving"
        elif difference < -0.5:
            return "declining"
        else:
            return "stable"

    def _get_trend_explanation(self, trend: str) -> str:
        """Generate human-readable explanation of trend.

        Args:
            trend: Trend classification from _compute_trend()

        Returns:
            Natural language explanation of the trend with supporting details.
        """
        if len(self._memory) < 2:
            return "stable (insufficient data)"

        # Calculate statistics for explanation
        waypoint_averages = []
        for item in self._memory:
            if item.scores:
                avg = sum(item.scores.values()) / len(item.scores)
                waypoint_averages.append(avg)

        if not waypoint_averages:
            return "stable (no scores)"

        mid = len(waypoint_averages) // 2
        first_half = waypoint_averages[:mid]
        second_half = waypoint_averages[mid:]

        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        difference = second_mean - first_mean

        # Build explanation
        if trend == "improving":
            return f"improving (scores increased by {difference:.1f} points)"
        elif trend == "declining":
            return f"declining (scores decreased by {abs(difference):.1f} points)"
        elif trend == "volatile":
            std_dev = statistics.stdev(waypoint_averages)
            return f"volatile (high variation, std={std_dev:.1f})"
        else:  # stable
            return f"stable (consistent scores around {first_mean:.1f}/10)"

    def get_memory_size(self) -> int:
        """Get current number of waypoints in memory.

        Returns:
            Number of waypoints currently stored (0 to window_size).
        """
        return len(self._memory)

    def is_full(self) -> bool:
        """Check if memory window is at capacity.

        Returns:
            True if memory contains window_size waypoints, False otherwise.
        """
        return len(self._memory) == self.window_size

    def get_latest_waypoint(self) -> Optional[MemoryItem]:
        """Get the most recently added waypoint.

        Returns:
            Most recent MemoryItem, or None if memory is empty.
        """
        if not self._memory:
            return None
        return self._memory[-1]

    def get_oldest_waypoint(self) -> Optional[MemoryItem]:
        """Get the oldest waypoint in the current window.

        Returns:
            Oldest MemoryItem in memory, or None if memory is empty.
        """
        if not self._memory:
            return None
        return self._memory[0]
