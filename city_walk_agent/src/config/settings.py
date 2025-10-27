import os
from pathlib import Path
from typing import List, Any
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


def _framework_dimensions(framework_id: str) -> List[str]:
    """
    Load dimension IDs from the requested framework definition.

    Falls back to the legacy four-dimension set if frameworks cannot be read.
    """
    from .frameworks import get_framework_manager

    try:
        framework = get_framework_manager().load_framework(framework_id)
        dimensions = [dim["id"] for dim in framework.get("dimensions", [])]
        return dimensions or ["safety", "comfort", "interest", "aesthetics"]
    except Exception:
        # Provide legacy fallback to keep initialization resilient
        return ["safety", "comfort", "interest", "aesthetics"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Required API Keys
    google_maps_api_key: str = Field(env="GOOGLE_MAPS_API_KEY")
    qwen_vlm_api_key: str = Field(env="QWEN_VLM_API_KEY")

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
        default="sagai_2025",
        env="DEFAULT_FRAMEWORK_ID",
        description="Framework identifier to use for default prompts and experiments",
    )
    default_dimensions: List[str] = Field(default_factory=list)
    default_model: str = Field(default="claude-3-sonnet")
    default_sampling_interval: int = Field(default=20)  # meters

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
        """
        Populate default dimensions from the configured framework unless provided.
        """
        if not self.default_dimensions:
            self.default_dimensions = _framework_dimensions(self.default_framework_id)


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.images_dir.mkdir(exist_ok=True)
settings.results_dir.mkdir(exist_ok=True)
