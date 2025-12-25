"""Place Pulse 2.0 dataset loader for validation.

This module provides utilities to load and process the Place Pulse 2.0 dataset,
which contains ~110,988 street view images with crowdsourced perception scores.

The dataset uses a long format where each image appears 6 times (once per dimension).
This loader pivots the data to wide format for easier analysis.
"""

import random
from pathlib import Path
from typing import Dict, List
import pandas as pd


class PlacePulseLoader:
    """Loader for Place Pulse 2.0 dataset.

    The dataset contains TrueSkill scores for 6 perception dimensions.
    Scores are normalized to 0-10 scale for consistency with VLM scores.
    """

    # Official study_id to dimension mapping from Place Pulse 2.0
    # Source: https://github.com/predictionmachines/PlacePulse/blob/master/PlacePulseDataset.py
    STUDY_ID_MAP = {
        '50a68a51fdc9f05596000002': 'safe',
        '50f62cb7a84ea7c5fdd2e458': 'wealthy',
        '50f62c41a84ea7c5fdd2e454': 'lively',
        '5217c351ad93a7d3e7b07a64': 'beautiful',
        '50f62c68a84ea7c5fdd2e456': 'boring',
        '50f62ccfa84ea7c5fdd2e459': 'depressing',
    }

    # Dimensions we care about for walkability evaluation
    DIMENSIONS = ["safe", "lively", "beautiful", "wealthy"]

    def __init__(self, data_dir: Path):
        """Initialize with path to Place Pulse data directory.

        Args:
            data_dir: Path to directory containing Place Pulse data
                     Should contain 'scores.csv' (or 'qscores.tsv') and 'images/' subdirectory
        """
        self.data_dir = Path(data_dir)
        # Try both .csv and .tsv versions
        self.scores_file = self.data_dir / "scores.csv"
        if not self.scores_file.exists():
            self.scores_file = self.data_dir / "qscores.tsv"
        self.images_dir = self.data_dir / "images"
        self._scores_df = None
        self._location_to_filename = None

        # Build lookup immediately so downstream code can filter to available images
        self._build_location_to_filename_map()

    def load_scores(self) -> pd.DataFrame:
        """Load and return scores DataFrame.

        The raw data is in long format (each image appears 6 times, once per dimension).
        This method pivots it to wide format with one row per image.

        Returns:
            DataFrame with columns: location_id, safe, lively, beautiful, wealthy
            Scores are normalized to 0-10 range.

        Raises:
            FileNotFoundError: If scores file doesn't exist
            ValueError: If required columns are missing or data format is incorrect
        """
        if not self.scores_file.exists():
            raise FileNotFoundError(
                f"Place Pulse scores file not found: {self.scores_file}\n"
                "Please download Place Pulse 2.0 dataset from:\n"
                "http://pulse.media.mit.edu/data/\n"
                "Looking for either 'scores.csv' or 'qscores.tsv'"
            )

        print(f"Loading scores from: {self.scores_file.name}")

        # Detect delimiter based on file extension
        delimiter = '\t' if self.scores_file.suffix == '.tsv' else ','

        # Load raw scores in long format
        df_long = pd.read_csv(self.scores_file, delimiter=delimiter)

        # Verify required columns exist
        required_cols = ['location_id', 'study_id', 'trueskill.score']
        missing_cols = set(required_cols) - set(df_long.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in scores file: {missing_cols}\n"
                f"Found columns: {list(df_long.columns)}"
            )

        print(f"Loaded {len(df_long):,} rows in long format")

        # Map study_id hashes to dimension names
        df_long['dimension'] = df_long['study_id'].map(self.STUDY_ID_MAP)

        # Check if all study_ids were mapped
        unmapped = df_long['dimension'].isna().sum()
        if unmapped > 0:
            unmapped_ids = df_long[df_long['dimension'].isna()]['study_id'].unique()
            raise ValueError(
                f"Found {unmapped} rows with unmapped study_ids: {unmapped_ids}"
            )

        # Filter to only the dimensions we care about
        df_long = df_long[df_long['dimension'].isin(self.DIMENSIONS)]

        # Pivot from long to wide format
        # Each location_id should have exactly one row with all dimensions
        df_wide = df_long.pivot_table(
            index='location_id',
            columns='dimension',
            values='trueskill.score',
            aggfunc='first'  # In case of duplicates, take first
        ).reset_index()

        # Keep only images that actually exist on disk
        available_ids = set(self._location_to_filename.keys())
        before_filter = len(df_wide)
        df_wide = df_wide[df_wide["location_id"].isin(available_ids)].copy()
        dropped_missing_images = before_filter - len(df_wide)
        if dropped_missing_images > 0:
            print(f"Dropped {dropped_missing_images:,} locations without image files")

        # Rename location_id to image_id for consistency with existing code
        df_wide = df_wide.rename(columns={'location_id': 'image_id'})

        # Verify all dimensions are present
        missing_dims = set(self.DIMENSIONS) - set(df_wide.columns)
        if missing_dims:
            raise ValueError(
                f"Missing dimensions after pivot: {missing_dims}\n"
                f"Available columns: {list(df_wide.columns)}"
            )

        # Select and order columns
        df_wide = df_wide[['image_id'] + self.DIMENSIONS].copy()

        # Check for missing scores
        for dim in self.DIMENSIONS:
            missing_count = df_wide[dim].isna().sum()
            if missing_count > 0:
                print(f"Warning: {missing_count} images missing '{dim}' scores")

        # Drop rows with any missing scores
        initial_count = len(df_wide)
        df_wide = df_wide.dropna()
        dropped_count = initial_count - len(df_wide)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} images with missing scores")

        # Normalize scores to 0-10 range using min-max normalization
        print("Normalizing scores to 0-10 range...")
        for dim in self.DIMENSIONS:
            min_score = df_wide[dim].min()
            max_score = df_wide[dim].max()
            if max_score > min_score:
                df_wide[dim] = 10 * (df_wide[dim] - min_score) / (max_score - min_score)
            else:
                # If all scores are the same, set to middle value
                df_wide[dim] = 5.0

        # Cache the loaded scores
        self._scores_df = df_wide

        # Print dataset statistics
        print(f"\n{'='*70}")
        print(f"Place Pulse 2.0 loaded successfully")
        print(f"{'='*70}")
        print(f"Total images: {len(df_wide):,}")
        print(f"Dimensions: {', '.join(self.DIMENSIONS)}")
        print(f"\nScore statistics (normalized 0-10):")
        for dim in self.DIMENSIONS:
            mean = df_wide[dim].mean()
            std = df_wide[dim].std()
            min_val = df_wide[dim].min()
            max_val = df_wide[dim].max()
            print(f"  {dim:10s} mean={mean:5.2f}, std={std:5.2f}, range=[{min_val:5.2f}, {max_val:5.2f}]")
        print(f"{'='*70}\n")

        return df_wide

    def _build_location_to_filename_map(self):
        """Build mapping from location_id to actual image filename.

        Image files are named: {lat}_{lon}_{location_id}_{city}.JPG
        We need to extract the location_id from each filename.
        """
        if self._location_to_filename is not None:
            return  # Already built

        print("Building location_id to filename mapping...")
        self._location_to_filename = {}

        # Accept common jpg/JPG variants
        patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        for pattern in patterns:
            for img_path in self.images_dir.glob(pattern):
                filename = img_path.name
                # Parse filename: {lat}_{lon}_{location_id}_{city}.jpg
                parts = img_path.stem.split("_")
                if len(parts) >= 4:
                    # location_id is the 3rd part (after lat and lon)
                    location_id = parts[2]
                    self._location_to_filename[location_id] = img_path

        print(f"Mapped {len(self._location_to_filename):,} location_ids to filenames")

    def get_image_path(self, image_id: str) -> Path:
        """Return full path to image file.

        Args:
            image_id: Image identifier (location_id)

        Returns:
            Path to image file

        Raises:
            ValueError: If image_id not found in dataset
        """
        # Build mapping if not already done
        if self._location_to_filename is None:
            self._build_location_to_filename_map()

        # Look up filename
        if image_id not in self._location_to_filename:
            raise ValueError(
                f"Image ID '{image_id}' not found in images directory.\n"
                f"Available image count: {len(self._location_to_filename)}"
            )

        return self._location_to_filename[image_id]

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

        # Build mapping once for efficiency
        if self._location_to_filename is None:
            self._build_location_to_filename_map()

        image_paths = []
        for image_id in self._scores_df["image_id"]:
            try:
                image_paths.append(self.get_image_path(image_id))
            except ValueError:
                # Skip images that don't have corresponding files
                pass

        return image_paths

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
