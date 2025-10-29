"""JSONL-backed persistent memory for CityWalkAgent experiences."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.config import settings
from src.utils.logging import get_logger


class AgentMemory:
    """Persist agent experiences with append-only JSONL storage.

    File layout::

        agent_memory/
            {agent_id}_memory.jsonl  # Sequential log of experiences
            {agent_id}_index.json    # Lightweight lookup index

    Stores perception/decision/result triplets per route while keeping a
    compact index for fast retrieval and statistics.
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

        # Logger
        self.logger = get_logger(f"memory.{agent_id}")

        # Load or initialize index
        self.index = self._load_index()

        self.logger.info(
            "Memory initialized",
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
