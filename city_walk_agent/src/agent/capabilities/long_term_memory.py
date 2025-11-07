"""JSONL-backed persistent memory for CityWalkAgent experiences."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict, is_dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.config import settings
from src.utils.logging import get_logger


@dataclass
class KeyMoment:
    """Significant waypoint moment worth preserving in long-term memory."""

    waypoint_id: int
    image_path: Path
    scores: Dict[str, float]  # {dimension_id: score}
    summary: str
    significance: str  # "high_score" | "barrier" | "transition" | "anomaly"
    gps: Tuple[float, float]
    timestamp: str
    significance_score: float  # 0-10, higher = more important


@dataclass
class RoutePattern:
    """Detected pattern across a sequence of waypoints."""

    pattern_type: str  # "improving" | "declining" | "volatile" | "barrier_cluster"
    start_waypoint: int
    end_waypoint: int
    description: str
    affected_dimensions: List[str]


@dataclass
class RouteSummary:
    """Comprehensive summary of a completed route traversal."""

    route_id: str
    total_waypoints: int
    length_km: float
    avg_scores: Dict[str, float]  # {dimension_id: avg_score}
    score_volatility: Dict[str, float]  # {dimension_id: volatility}
    dominant_characteristics: List[str]  # Top 3 defining features
    major_barriers: List[Dict[str, Any]]  # Significant barriers detected
    highlights: List[str]  # Key positive moments
    patterns: List[RoutePattern]
    overall_recommendation: str  # "recommended" | "acceptable" | "avoid"
    recommendation_reason: str
    target_users: List[str]  # ["accessibility_focused", "scenic_priority", etc.]


class LongTermMemory:
    """Persist agent experiences with append-only JSONL storage.

    File layout::

        agent_memory/
            {agent_id}_memory.jsonl  # Sequential log of experiences
            {agent_id}_index.json    # Lightweight lookup index
            {agent_id}_moments.json  # Curated key moments
            {agent_id}_patterns.json # Extracted patterns

    Stores perception/decision/result triplets per route while keeping a
    compact index for fast retrieval and statistics. Also maintains curated
    key moments and patterns for route summarization.
    """

    def __init__(
        self,
        agent_id: str,
        storage_dir: Optional[Path] = None
    ) -> None:
        """Set up storage paths, logger, and index.

        Args:
            agent_id: Unique identifier for the owning agent.
            storage_dir: Optional override for memory root directory. Defaults
                to `settings.data_dir / "agent_memory"`.

        Side Effects:
            Ensures the storage directory exists and loads/initialises the index.
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir or (settings.data_dir / "agent_memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.memory_file = self.storage_dir / f"{agent_id}_memory.jsonl"
        self.index_file = self.storage_dir / f"{agent_id}_index.json"
        self.moments_file = self.storage_dir / f"{agent_id}_moments.json"
        self.patterns_file = self.storage_dir / f"{agent_id}_patterns.json"

        # Logger
        self.logger = get_logger(f"memory.{agent_id}")

        # Load or initialize index
        self.index = self._load_index()

        # In-memory storage for current session
        self._candidate_moments: List[KeyMoment] = []
        self._extracted_patterns: List[RoutePattern] = []

        self.logger.info(
            "LongTermMemory initialized",
            agent_id=agent_id,
            storage_dir=str(self.storage_dir),
            total_experiences=self.index["total_experiences"]
        )

    def store(self, experience: Dict[str, Any]) -> None:
        """Append an experience and update the index.

        Args:
            experience: Dictionary containing at least `route_id`, plus optional
                perception, decision, result, and timestamp fields.

        Raises:
            ValueError: If `route_id` is missing.
            IOError: If the underlying log cannot be written.
        """
        # Validate required fields
        if "route_id" not in experience:
            raise ValueError("Experience must contain 'route_id'")

        # Add metadata
        stored_at = datetime.now().isoformat()
        experience_with_metadata = {
            **experience,
            "stored_at": stored_at,
            "agent_id": self.agent_id
        }

        # Ensure timestamp exists
        if "timestamp" not in experience_with_metadata:
            experience_with_metadata["timestamp"] = stored_at

        route_id = experience["route_id"]

        try:
            # Append to JSONL file
            with open(self.memory_file, "a", encoding="utf-8") as f:
                json.dump(
                    self._to_json_compatible(experience_with_metadata),
                    f,
                    ensure_ascii=False
                )
                f.write("\n")

            # Update index
            if route_id not in self.index["routes"]:
                self.index["routes"][route_id] = []

            self.index["routes"][route_id].append(stored_at)
            self.index["total_experiences"] += 1
            self._save_index()

            self.logger.debug(
                "Experience stored",
                route_id=route_id,
                total_experiences=self.index["total_experiences"]
            )

        except IOError as e:
            self.logger.error(
                "Failed to store experience",
                route_id=route_id,
                error=str(e)
            )
            raise

    def retrieve(
        self,
        route_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch experiences, optionally filtered by route.

        Args:
            route_id: Restrict results to the specified route id.
            limit: Maximum number of records to return (newest first).

        Returns:
            List[Dict[str, Any]]: Experience payloads ordered from newest to oldest.
        """
        if not self.memory_file.exists():
            self.logger.debug(
                "No memory file found",
                route_id=route_id
            )
            return []

        experiences = []

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        experience = json.loads(line)

                        # Filter by route_id if specified
                        if route_id is None or experience.get("route_id") == route_id:
                            experiences.append(experience)

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            "Skipping corrupted memory line",
                            line_number=line_num,
                            error=str(e)
                        )
                        continue

            # Return most recent first
            experiences.reverse()

            # Apply limit
            if limit > 0:
                experiences = experiences[:limit]

            self.logger.debug(
                "Experiences retrieved",
                route_id=route_id,
                count=len(experiences),
                limit=limit
            )

            return experiences

        except IOError as e:
            self.logger.error(
                "Failed to retrieve experiences",
                route_id=route_id,
                error=str(e)
            )
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Summarise memory usage and per-route counts.

        Returns:
            Dict[str, Any]: Aggregate statistics including totals, unique route
            counts, file sizes, and per-route experience tallies.
        """
        # Calculate file sizes
        memory_size_kb = 0.0
        if self.memory_file.exists():
            memory_size_kb = self.memory_file.stat().st_size / 1024

        index_size_kb = 0.0
        if self.index_file.exists():
            index_size_kb = self.index_file.stat().st_size / 1024

        # Calculate per-route experience counts
        route_counts = {
            route_id: len(timestamps)
            for route_id, timestamps in self.index["routes"].items()
        }

        statistics = {
            "agent_id": self.agent_id,
            "total_experiences": self.index["total_experiences"],
            "unique_routes": len(self.index["routes"]),
            "memory_file_size_kb": round(memory_size_kb, 2),
            "index_file_size_kb": round(index_size_kb, 2),
            "created_at": self.index["created_at"],
            "routes": route_counts
        }

        self.logger.debug(
            "Statistics retrieved",
            total_experiences=statistics["total_experiences"],
            unique_routes=statistics["unique_routes"]
        )

        return statistics

    # ========================================================================
    # New Methods: Key Moments and Route Summarization
    # ========================================================================

    def add_candidate_moment(
        self,
        waypoint_id: int,
        image_path: Path,
        scores: Dict[str, float],
        summary: str,
        significance: str,
        gps: Tuple[float, float],
        timestamp: str,
        thinking_confidence: float,
        visual_change_detected: bool,
        score_delta: Optional[float] = None
    ) -> bool:
        """Add a candidate moment for potential long-term storage.

        Collects significant waypoints during route traversal. These are
        later curated to select the most important moments worth preserving.

        Args:
            waypoint_id: Sequential waypoint identifier.
            image_path: Path to street view image.
            scores: Evaluation scores by dimension.
            summary: Human-readable waypoint description.
            significance: Type of significance ("high_score"|"barrier"|"transition"|"anomaly").
            gps: (latitude, longitude) tuple.
            timestamp: ISO format timestamp.
            thinking_confidence: Agent's confidence in this observation (0-1).
            visual_change_detected: Whether pHash detected visual change.
            score_delta: Score change from previous waypoint (if applicable).

        Returns:
            True if moment was added, False if rejected (low significance).

        Side Effects:
            Appends to internal _candidate_moments list.
        """
        # Calculate significance score
        significance_score = self._calculate_significance_score(
            scores=scores,
            significance=significance,
            thinking_confidence=thinking_confidence,
            visual_change_detected=visual_change_detected,
            score_delta=score_delta
        )

        # Filter out low-significance moments
        if significance_score < 3.0:
            self.logger.debug(
                "Candidate moment rejected",
                waypoint_id=waypoint_id,
                significance_score=significance_score
            )
            return False

        moment = KeyMoment(
            waypoint_id=waypoint_id,
            image_path=image_path,
            scores=scores,
            summary=summary,
            significance=significance,
            gps=gps,
            timestamp=timestamp,
            significance_score=significance_score
        )

        self._candidate_moments.append(moment)

        self.logger.debug(
            "Candidate moment added",
            waypoint_id=waypoint_id,
            significance=significance,
            significance_score=significance_score
        )

        return True

    def curate_moments(self, route_length_km: float) -> None:
        """Select most important moments from candidates and persist to disk.

        Uses adaptive selection: keep top N moments based on route length.
        Longer routes → more moments preserved.

        Args:
            route_length_km: Total route length for adaptive thresholding.

        Side Effects:
            Writes curated moments to {agent_id}_moments.json.
            Clears _candidate_moments buffer.
        """
        if not self._candidate_moments:
            self.logger.info("No candidate moments to curate")
            return

        # Determine how many moments to keep (adaptive)
        # Rule: 1 moment per 0.5km, minimum 3, maximum 20
        target_count = max(3, min(20, int(route_length_km / 0.5)))

        # Sort by significance_score descending
        sorted_moments = sorted(
            self._candidate_moments,
            key=lambda m: m.significance_score,
            reverse=True
        )

        # Select top N
        curated = sorted_moments[:target_count]

        # Load existing moments
        existing_moments = []
        if self.moments_file.exists():
            try:
                with open(self.moments_file, "r", encoding="utf-8") as f:
                    existing_moments = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    "Failed to load existing moments",
                    error=str(e)
                )

        # Append new curated moments
        for moment in curated:
            existing_moments.append(self._to_json_compatible(moment))

        # Save back to disk
        try:
            with open(self.moments_file, "w", encoding="utf-8") as f:
                json.dump(existing_moments, f, indent=2, ensure_ascii=False)

            self.logger.info(
                "Moments curated",
                candidates=len(self._candidate_moments),
                curated=len(curated),
                total_moments=len(existing_moments)
            )

        except IOError as e:
            self.logger.error(
                "Failed to save curated moments",
                error=str(e)
            )

        # Clear candidate buffer
        self._candidate_moments.clear()

    def extract_patterns(
        self,
        all_analyses: List[Dict[str, Any]],
        thinking_history: List[str]
    ) -> None:
        """Analyze waypoint sequence to extract route patterns.

        Identifies trends, barrier clusters, and other sequential patterns
        across the route traversal.

        Args:
            all_analyses: List of waypoint analysis dicts with scores/summaries.
            thinking_history: Agent's thinking log for context.

        Side Effects:
            Updates _extracted_patterns and persists to {agent_id}_patterns.json.
        """
        if len(all_analyses) < 3:
            self.logger.debug("Insufficient data for pattern extraction")
            return

        patterns = []

        # Extract trend patterns (improving/declining sequences)
        trend_patterns = self._extract_trend_patterns(all_analyses)
        patterns.extend(trend_patterns)

        # Extract barrier clusters (consecutive low scores)
        barrier_patterns = self._extract_barrier_patterns(all_analyses)
        patterns.extend(barrier_patterns)

        # Extract volatility patterns (high variance regions)
        volatile_patterns = self._extract_volatile_patterns(all_analyses)
        patterns.extend(volatile_patterns)

        self._extracted_patterns = patterns

        # Load existing patterns
        existing_patterns = []
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    existing_patterns = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    "Failed to load existing patterns",
                    error=str(e)
                )

        # Append new patterns
        for pattern in patterns:
            existing_patterns.append(self._to_json_compatible(pattern))

        # Save to disk
        try:
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(existing_patterns, f, indent=2, ensure_ascii=False)

            self.logger.info(
                "Patterns extracted",
                new_patterns=len(patterns),
                total_patterns=len(existing_patterns)
            )

        except IOError as e:
            self.logger.error(
                "Failed to save patterns",
                error=str(e)
            )

    def generate_route_summary(
        self,
        route_id: str,
        total_waypoints: int,
        length_km: float,
        all_analyses: List[Dict[str, Any]]
    ) -> RouteSummary:
        """Generate comprehensive route summary from traversal data.

        Args:
            route_id: Unique route identifier.
            total_waypoints: Number of waypoints traversed.
            length_km: Total route length.
            all_analyses: List of all waypoint analyses.

        Returns:
            RouteSummary object with complete route characterization.
        """
        # Calculate average scores per dimension
        avg_scores = self._calculate_avg_scores(all_analyses)

        # Calculate volatility per dimension
        score_volatility = self._calculate_score_volatility(all_analyses)

        # Identify dominant characteristics
        dominant_characteristics = self._identify_dominant_characteristics(
            avg_scores, score_volatility
        )

        # Extract major barriers
        major_barriers = self._extract_major_barriers(all_analyses)

        # Extract highlights
        highlights = self._extract_highlights(all_analyses)

        # Generate recommendation
        overall_recommendation, recommendation_reason = self._generate_recommendation(
            avg_scores, score_volatility, major_barriers
        )

        # Identify target users
        target_users = self._identify_target_users(avg_scores, self._extracted_patterns)

        summary = RouteSummary(
            route_id=route_id,
            total_waypoints=total_waypoints,
            length_km=length_km,
            avg_scores=avg_scores,
            score_volatility=score_volatility,
            dominant_characteristics=dominant_characteristics,
            major_barriers=major_barriers,
            highlights=highlights,
            patterns=self._extracted_patterns,
            overall_recommendation=overall_recommendation,
            recommendation_reason=recommendation_reason,
            target_users=target_users
        )

        self.logger.info(
            "Route summary generated",
            route_id=route_id,
            recommendation=overall_recommendation,
            barriers=len(major_barriers)
        )

        return summary

    def get_key_images(self) -> List[Path]:
        """Retrieve image paths for all curated key moments.

        Returns:
            List of Path objects to significant waypoint images.
        """
        if not self.moments_file.exists():
            return []

        try:
            with open(self.moments_file, "r", encoding="utf-8") as f:
                moments_data = json.load(f)

            image_paths = [
                Path(moment["image_path"])
                for moment in moments_data
                if "image_path" in moment
            ]

            self.logger.debug(
                "Key images retrieved",
                count=len(image_paths)
            )

            return image_paths

        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(
                "Failed to load key images",
                error=str(e)
            )
            return []

    def get_narrative(self) -> str:
        """Generate natural language narrative from curated moments and patterns.

        Returns:
            Human-readable story of the route experience.
        """
        if not self.moments_file.exists():
            return "No route memory available."

        try:
            with open(self.moments_file, "r", encoding="utf-8") as f:
                moments_data = json.load(f)

            if not moments_data:
                return "No significant moments recorded."

            # Build narrative from moments
            lines = ["Route Experience Narrative:", ""]

            for i, moment in enumerate(moments_data[-10:], 1):  # Last 10 moments
                waypoint_id = moment.get("waypoint_id", "?")
                summary = moment.get("summary", "No description")
                significance = moment.get("significance", "unknown")

                lines.append(f"{i}. Waypoint {waypoint_id} ({significance}): {summary}")

            # Add pattern summary if available
            if self.patterns_file.exists():
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    patterns_data = json.load(f)

                if patterns_data:
                    lines.append("")
                    lines.append("Notable Patterns:")
                    for pattern in patterns_data[-5:]:  # Last 5 patterns
                        pattern_type = pattern.get("pattern_type", "unknown")
                        description = pattern.get("description", "No description")
                        lines.append(f"- {pattern_type}: {description}")

            narrative = "\n".join(lines)

            self.logger.debug(
                "Narrative generated",
                length=len(narrative)
            )

            return narrative

        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(
                "Failed to generate narrative",
                error=str(e)
            )
            return "Error loading route memory."

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_significance_score(
        self,
        scores: Dict[str, float],
        significance: str,
        thinking_confidence: float,
        visual_change_detected: bool,
        score_delta: Optional[float]
    ) -> float:
        """Calculate significance score for a moment (0-10)."""
        base_score = sum(scores.values()) / len(scores) if scores else 5.0

        # Adjust based on significance type
        significance_multiplier = {
            "high_score": 1.2,
            "barrier": 1.5,
            "transition": 1.3,
            "anomaly": 1.4
        }.get(significance, 1.0)

        # Factor in confidence
        confidence_factor = thinking_confidence

        # Boost for visual changes
        visual_bonus = 1.0 if visual_change_detected else 0.0

        # Boost for large score deltas
        delta_bonus = 0.0
        if score_delta is not None:
            delta_bonus = min(2.0, abs(score_delta) / 2.0)

        significance_score = (
            (base_score / 10.0) * significance_multiplier * confidence_factor * 5.0
            + visual_bonus
            + delta_bonus
        )

        return min(10.0, significance_score)

    def _calculate_avg_scores(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average score per dimension across all waypoints."""
        dimension_scores = {}

        for analysis in all_analyses:
            scores = analysis.get("scores", {})
            for dim_id, score in scores.items():
                if dim_id not in dimension_scores:
                    dimension_scores[dim_id] = []
                dimension_scores[dim_id].append(score)

        return {
            dim_id: sum(scores) / len(scores)
            for dim_id, scores in dimension_scores.items()
        }

    def _calculate_score_volatility(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate score volatility (std dev) per dimension."""
        dimension_scores = {}

        for analysis in all_analyses:
            scores = analysis.get("scores", {})
            for dim_id, score in scores.items():
                if dim_id not in dimension_scores:
                    dimension_scores[dim_id] = []
                dimension_scores[dim_id].append(score)

        volatility = {}
        for dim_id, scores in dimension_scores.items():
            if len(scores) >= 2:
                volatility[dim_id] = statistics.stdev(scores)
            else:
                volatility[dim_id] = 0.0

        return volatility

    def _identify_dominant_characteristics(
        self,
        avg_scores: Dict[str, float],
        volatility: Dict[str, float]
    ) -> List[str]:
        """Identify top 3 defining route characteristics."""
        characteristics = []

        # High score dimensions
        for dim_id, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            if score >= 7.0:
                characteristics.append(f"High {dim_id} (avg: {score:.1f})")

        # High volatility dimensions
        for dim_id, vol in sorted(volatility.items(), key=lambda x: x[1], reverse=True)[:2]:
            if vol >= 1.5:
                characteristics.append(f"Volatile {dim_id} (std: {vol:.1f})")

        return characteristics[:3]  # Top 3

    def _extract_major_barriers(self, all_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract waypoints with significant barriers (low scores)."""
        barriers = []

        for i, analysis in enumerate(all_analyses):
            scores = analysis.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 5.0

            if avg_score < 4.0:  # Barrier threshold
                barriers.append({
                    "waypoint_id": i,
                    "avg_score": avg_score,
                    "summary": analysis.get("summary", "No description")
                })

        return barriers

    def _extract_highlights(self, all_analyses: List[Dict[str, Any]]) -> List[str]:
        """Extract positive highlights from route."""
        highlights = []

        for analysis in all_analyses:
            scores = analysis.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 5.0

            if avg_score >= 8.0:
                summary = analysis.get("summary", "Great waypoint")
                highlights.append(summary)

        return highlights[:5]  # Top 5

    def _generate_recommendation(
        self,
        avg_scores: Dict[str, float],
        volatility: Dict[str, float],
        barriers: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Generate overall route recommendation."""
        overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 5.0
        overall_volatility = sum(volatility.values()) / len(volatility) if volatility else 0.0

        if len(barriers) >= 3:
            return "avoid", f"Multiple barriers detected ({len(barriers)} waypoints)"
        elif overall_avg >= 7.0 and overall_volatility < 1.0:
            return "recommended", f"Consistently good experience (avg: {overall_avg:.1f})"
        elif overall_avg >= 6.0:
            return "acceptable", f"Decent route with some issues (avg: {overall_avg:.1f})"
        else:
            return "avoid", f"Below acceptable standards (avg: {overall_avg:.1f})"

    def _identify_target_users(
        self,
        avg_scores: Dict[str, float],
        patterns: List[RoutePattern]
    ) -> List[str]:
        """Identify user types who would prefer this route."""
        target_users = []

        # Check dimension strengths
        for dim_id, score in avg_scores.items():
            if score >= 8.0:
                if "accessibility" in dim_id.lower():
                    target_users.append("accessibility_focused")
                elif "scenic" in dim_id.lower() or "aesthetics" in dim_id.lower():
                    target_users.append("scenic_priority")
                elif "safety" in dim_id.lower():
                    target_users.append("safety_conscious")

        # Check patterns
        for pattern in patterns:
            if pattern.pattern_type == "improving":
                target_users.append("progressive_experience_seekers")

        return list(set(target_users))  # Deduplicate

    def _extract_trend_patterns(self, all_analyses: List[Dict[str, Any]]) -> List[RoutePattern]:
        """Extract improving/declining trend patterns."""
        patterns = []

        if len(all_analyses) < 5:
            return patterns

        # Calculate waypoint averages
        waypoint_avgs = []
        for analysis in all_analyses:
            scores = analysis.get("scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 5.0
            waypoint_avgs.append(avg)

        # Check for improving trend
        mid = len(waypoint_avgs) // 2
        first_half_avg = sum(waypoint_avgs[:mid]) / mid
        second_half_avg = sum(waypoint_avgs[mid:]) / (len(waypoint_avgs) - mid)

        if second_half_avg - first_half_avg > 1.0:
            patterns.append(RoutePattern(
                pattern_type="improving",
                start_waypoint=0,
                end_waypoint=len(all_analyses) - 1,
                description=f"Route improves significantly ({first_half_avg:.1f} → {second_half_avg:.1f})",
                affected_dimensions=list(all_analyses[0].get("scores", {}).keys())
            ))
        elif first_half_avg - second_half_avg > 1.0:
            patterns.append(RoutePattern(
                pattern_type="declining",
                start_waypoint=0,
                end_waypoint=len(all_analyses) - 1,
                description=f"Route declines significantly ({first_half_avg:.1f} → {second_half_avg:.1f})",
                affected_dimensions=list(all_analyses[0].get("scores", {}).keys())
            ))

        return patterns

    def _extract_barrier_patterns(self, all_analyses: List[Dict[str, Any]]) -> List[RoutePattern]:
        """Extract barrier cluster patterns (consecutive low scores)."""
        patterns = []
        in_cluster = False
        cluster_start = 0

        for i, analysis in enumerate(all_analyses):
            scores = analysis.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 5.0

            if avg_score < 4.5:  # Low score threshold
                if not in_cluster:
                    cluster_start = i
                    in_cluster = True
            else:
                if in_cluster and i - cluster_start >= 2:  # At least 2 consecutive
                    patterns.append(RoutePattern(
                        pattern_type="barrier_cluster",
                        start_waypoint=cluster_start,
                        end_waypoint=i - 1,
                        description=f"Cluster of {i - cluster_start} consecutive poor waypoints",
                        affected_dimensions=list(scores.keys())
                    ))
                in_cluster = False

        return patterns

    def _extract_volatile_patterns(self, all_analyses: List[Dict[str, Any]]) -> List[RoutePattern]:
        """Extract high-volatility regions."""
        patterns = []

        if len(all_analyses) < 5:
            return patterns

        # Use sliding window to find volatile regions
        window_size = 5
        for i in range(len(all_analyses) - window_size + 1):
            window_scores = []
            for j in range(i, i + window_size):
                scores = all_analyses[j].get("scores", {})
                avg = sum(scores.values()) / len(scores) if scores else 5.0
                window_scores.append(avg)

            if len(window_scores) >= 2:
                std = statistics.stdev(window_scores)
                if std > 2.0:  # High volatility threshold
                    patterns.append(RoutePattern(
                        pattern_type="volatile",
                        start_waypoint=i,
                        end_waypoint=i + window_size - 1,
                        description=f"High score variation (std: {std:.1f})",
                        affected_dimensions=list(all_analyses[i].get("scores", {}).keys())
                    ))

        return patterns

    def _to_json_compatible(self, value: Any) -> Any:
        """Recursively convert values into JSON-serialisable structures."""
        if isinstance(value, BaseModel):
            if hasattr(value, "model_dump"):
                raw_data = value.model_dump()
            elif hasattr(value, "dict"):
                raw_data = value.dict()
            else:
                raw_data = value.__dict__
            return {
                key: self._to_json_compatible(val)
                for key, val in raw_data.items()
            }

        if is_dataclass(value):
            return {
                key: self._to_json_compatible(val)
                for key, val in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                key: self._to_json_compatible(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._to_json_compatible(item) for item in value]

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "value"):
            enum_value = getattr(value, "value")
            if not isinstance(enum_value, (dict, list, tuple, set)):
                return enum_value

        return value

    def _load_index(self) -> Dict[str, Any]:
        """Load index from disk or bootstrap a new one.

        Returns:
            Dict[str, Any]: Index structure containing totals, route timestamp
            lists, and metadata.

        Side Effects:
            Creates a fresh index if none exists or if loading fails.
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)

                self.logger.debug(
                    "Index loaded",
                    total_experiences=index.get("total_experiences", 0)
                )

                return index

            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    "Failed to load index, creating new",
                    error=str(e)
                )

        # Create new index
        index = {
            "agent_id": self.agent_id,
            "total_experiences": 0,
            "routes": {},
            "created_at": datetime.now().isoformat()
        }

        self.logger.debug("New index created")

        return index

    def _save_index(self) -> None:
        """Persist the index structure to disk.

        Raises:
            IOError: If the index file cannot be written.
        """
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)

            self.logger.debug("Index saved")

        except IOError as e:
            self.logger.error(
                "Failed to save index",
                error=str(e)
            )
            raise

    # ========================================================================
    # Future Interface Stubs (Not Yet Implemented)
    # ========================================================================

    def find_similar_routes(self, route_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find routes similar to current one based on features.

        TODO: Implement similarity search using route features like:
        - Distance range
        - Geographic proximity
        - Dimension score patterns
        - Environmental characteristics

        Args:
            route_features: Dictionary of route characteristics to match.

        Returns:
            List of similar route experiences. Currently returns empty list.
        """
        self.logger.debug("find_similar_routes not yet implemented")
        return []

    def analyze_preferences(self) -> Dict[str, float]:
        """Learn personality adjustments from historical experiences.

        TODO: Implement preference learning by analyzing:
        - Routes that were accepted vs rejected
        - Feedback patterns across dimensions
        - Evolution of preferences over time
        - Correlation between decisions and outcomes

        Returns:
            Dictionary of learned preference adjustments per dimension.
            Currently returns empty dict.
        """
        self.logger.debug("analyze_preferences not yet implemented")
        return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate decision accuracy and performance metrics.

        TODO: Implement performance analysis including:
        - Decision accuracy (if feedback provided)
        - Confidence calibration
        - Consistency across similar routes
        - Improvement trends over time

        Returns:
            Dictionary of performance metrics and statistics.
            Currently returns empty dict.
        """
        self.logger.debug("get_performance_metrics not yet implemented")
        return {}
