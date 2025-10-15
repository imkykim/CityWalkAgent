import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys
    google_maps_api_key: Optional[str] = Field(default=None, env="GOOGLE_MAPS_API_KEY")
    mapillary_api_key: Optional[str] = Field(default=None, env="MAPILLARY_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Project Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    images_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "images")
    results_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "results")

    # ZenSVI Integration
    zensvi_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "ZenSVI")

    # VIRL Integration
    virl_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "VIRL")

    # Evaluation Settings
    default_dimensions: List[str] = Field(default=["safety", "comfort", "interest", "aesthetics"])
    default_model: str = Field(default="claude-3-sonnet")
    default_sampling_interval: int = Field(default=10)  # meters

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


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.images_dir.mkdir(exist_ok=True)
settings.results_dir.mkdir(exist_ok=True)