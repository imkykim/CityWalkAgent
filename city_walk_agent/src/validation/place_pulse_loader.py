"""Place Pulse 2.0 dataset loader for validation.

This module provides utilities to load and process the Place Pulse 2.0 dataset,
which contains ~110,988 street view images with crowdsourced perception scores.
"""

import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


class PlacePulseLoader:
    """Loader for Place Pulse 2.0 dataset."""

    # Place Pulse 2.0 uses TrueSkill scores which typically range from ~-3 to +3
    # We normalize these to 0-10 scale for consistency with VLM scores
    DIMENSIONS = ["safe", "lively", "beautiful", "wealthy"]

    def __init__(self, data_dir: Path):
        """Initialize with path to Place Pulse data directory.

        Args:
            data_dir: Path to directory containing Place Pulse data
                     Should contain 'scores.csv' and 'images/' subdirectory
        """
        self.data_dir = Path(data_dir)
        self.scores_file = self.data_dir / "scores.csv"
        self.images_dir = self.data_dir / "images"
        self._scores_df = None

    def load_scores(self) -> pd.DataFrame:
        """Load and return scores DataFrame.

        Returns:
            DataFrame with columns: image_id, safe, lively, beautiful, wealthy
            Scores are normalized to 0-10 range.

        Raises:
            FileNotFoundError: If scores file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.scores_file.exists():
            raise FileNotFoundError(
                f"Place Pulse scores file not found: {self.scores_file}\n"
                "Please download Place Pulse 2.0 dataset from:\n"
                "http://pulse.media.mit.edu/data/"
            )

        # Load scores
        df = pd.read_csv(self.scores_file)

        # Verify required columns exist
        required_cols = ["image_id"] + self.DIMENSIONS
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in scores file: {missing_cols}"
            )

        # Select only required columns
        df = df[required_cols].copy()

        # Normalize scores to 0-10 range
        # Assuming TrueSkill scores are roughly in [-3, +3] range
        # We'll use a more robust min-max normalization per dimension
        for dim in self.DIMENSIONS:
            min_score = df[dim].min()
            max_score = df[dim].max()
            if max_score > min_score:
                df[dim] = 10 * (df[dim] - min_score) / (max_score - min_score)
            else:
                # If all scores are the same, set to middle value
                df[dim] = 5.0

        # Cache the loaded scores
        self._scores_df = df

        # Print dataset statistics
        print(f"Place Pulse 2.0 loaded: {len(df):,} images")
        print(f"Dimensions: {', '.join(self.DIMENSIONS)}")
        print(f"Score ranges: 0.0 - 10.0")

        mean_scores = ", ".join(
            [f"{dim}={df[dim].mean():.1f}" for dim in self.DIMENSIONS]
        )
        print(f"Mean scores: {mean_scores}")

        return df

    def get_image_path(self, image_id: str) -> Path:
        """Return full path to image file.

        Args:
            image_id: Image identifier

        Returns:
            Path to image file
        """
        # Assuming images are stored as {image_id}.jpg
        # Adjust extension if needed (.png, etc.)
        return self.images_dir / f"{image_id}.jpg"

    def get_all_image_paths(self) -> List[Path]:
        """Return paths to all images in the dataset.

        Returns:
            List of Path objects to all images

        Raises:
            RuntimeError: If scores haven't been loaded yet
        """
        if self._scores_df is None:
            raise RuntimeError(
                "Must call load_scores() before getting image paths"
            )

        return [
            self.get_image_path(image_id)
            for image_id in self._scores_df["image_id"]
        ]

    def get_score_statistics(self) -> Dict[str, Dict]:
        """Return per-dimension statistics.

        Returns:
            Dictionary mapping dimension to statistics dict:
            {dimension: {mean, std, min, max, count}}

        Raises:
            RuntimeError: If scores haven't been loaded yet
        """
        if self._scores_df is None:
            raise RuntimeError(
                "Must call load_scores() before getting statistics"
            )

        stats = {}
        for dim in self.DIMENSIONS:
            stats[dim] = {
                "mean": float(self._scores_df[dim].mean()),
                "std": float(self._scores_df[dim].std()),
                "min": float(self._scores_df[dim].min()),
                "max": float(self._scores_df[dim].max()),
                "count": int(self._scores_df[dim].count()),
            }

        return stats

    def sample_images(self, n: int, seed: int = 42) -> List[str]:
        """Return n random image_ids for testing.

        Args:
            n: Number of images to sample
            seed: Random seed for reproducibility

        Returns:
            List of image IDs

        Raises:
            RuntimeError: If scores haven't been loaded yet
            ValueError: If n is larger than dataset size
        """
        if self._scores_df is None:
            raise RuntimeError(
                "Must call load_scores() before sampling images"
            )

        if n > len(self._scores_df):
            raise ValueError(
                f"Cannot sample {n} images from dataset of size {len(self._scores_df)}"
            )

        random.seed(seed)
        return random.sample(list(self._scores_df["image_id"]), n)
