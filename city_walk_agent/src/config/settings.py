"""Application settings management for CityWalkAgent."""

import json
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import Field

from .constants import DEFAULT_FRAMEWORK_ID, DEFAULT_SAMPLING_INTERVAL

# Load environment variables from .env file
load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - compatibility shim
    from pydantic import BaseSettings


def _framework_dimensions(framework_id: str, frameworks_dir: Path) -> List[str]:
    """Resolve framework dimension identifiers with graceful fallback."""
    fallback = ["safety", "comfort", "interest", "aesthetics"]
    framework_path = frameworks_dir / f"{framework_id}.json"

    if not framework_path.exists():
        return fallback

    try:
        with framework_path.open("r", encoding="utf-8") as framework_file:
            framework = json.load(framework_file)
    except (OSError, json.JSONDecodeError):
        return fallback

    dimensions = [
        dimension.get("id")
        for dimension in framework.get("dimensions", [])
        if isinstance(dimension, dict) and dimension.get("id")
    ]

    return dimensions or fallback


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Required API Keys
    google_maps_api_key: str = Field(env="GOOGLE_MAPS_API_KEY")
    qwen_vlm_api_key: str = Field(env="QWEN_VLM_API_KEY")
    mapillary_api_key: Optional[str] = Field(default=None, env="MAPILLARY_API_KEY")

    # Qwen VLM Configuration
    qwen_vlm_api_url: str = Field(env="QWEN_VLM_API_URL")
    qwen_vlm_model: str = Field(
        default="Qwen3-VL-30B-A3B-Instruct-FP8", env="QWEN_VLM_MODEL"
    )

    # Project Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data"
    )
    images_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "images"
    )
    results_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "results"
    )

    # ZenSVI Integration
    zensvi_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "ZenSVI"
    )

    # VIRL Integration
    virl_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "VIRL"
    )

    # Evaluation Settings
    default_framework_id: str = Field(
        default=DEFAULT_FRAMEWORK_ID,
        env="DEFAULT_FRAMEWORK_ID",
        description="Framework identifier to use for default prompts and experiments",
    )
    default_dimensions: List[str] = Field(default_factory=list)
    default_model: str = Field(default="claude-3-sonnet")
    default_sampling_interval: int = Field(default=DEFAULT_SAMPLING_INTERVAL)

    # Analysis Thresholds
    volatility_threshold: float = Field(default=2.0)
    barrier_threshold: float = Field(default=3.0)

    # Processing Settings
    max_workers: int = Field(default=4)
    batch_size: int = Field(default=32)

    # Experiment Settings
    experiment_output_format: str = Field(default="json")
    save_intermediate_results: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables

    def model_post_init(self, __context: Any) -> None:
        self.ensure_directories()

        if not self.default_dimensions:
            self.default_dimensions = _framework_dimensions(
                self.default_framework_id, self.frameworks_dir
            )

    @property
    def frameworks_dir(self) -> Path:
        """Directory containing framework configuration files."""
        return self.project_root / "src" / "config" / "framework_configs"

    def ensure_directories(self) -> None:
        """Ensure core data directories exist on disk."""
        for directory in (self.data_dir, self.images_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
