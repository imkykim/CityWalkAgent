"""Simple test script for Place Pulse loader.

This demonstrates the basic functionality without requiring the full dataset.
"""

from pathlib import Path

from src.config.settings import settings
from src.validation.place_pulse_loader import PlacePulseLoader


def test_loader():
    """Test the Place Pulse loader with available data."""
    print("=" * 70)
    print("Testing Place Pulse 2.0 Data Loader")
    print("=" * 70)
    print()

    # Initialize loader
    loader = PlacePulseLoader(settings.place_pulse_dir)
    print(f"Data directory: {settings.place_pulse_dir}")
    print(f"Scores file: {loader.scores_file}")
    print(f"Images directory: {loader.images_dir}")
    print()

    # Check if data exists
    if not loader.scores_file.exists():
        print("⚠️  Place Pulse 2.0 dataset not found!")
        print()
        print("To download:")
        print("  1. Visit: http://pulse.media.mit.edu/data/")
        print("  2. Download the dataset")
        print(f"  3. Extract to: {settings.place_pulse_dir}")
        print("  4. Ensure scores.csv and images/ directory are present")
        return

    # Load scores
    try:
        scores_df = loader.load_scores()
        print()

        # Print statistics
        print("=" * 70)
        print("Dataset Statistics")
        print("=" * 70)
        stats = loader.get_score_statistics()

        for dim, dim_stats in stats.items():
            print(f"\n{dim.capitalize()}:")
            print(f"  Mean: {dim_stats['mean']:.2f}")
            print(f"  Std:  {dim_stats['std']:.2f}")
            print(f"  Min:  {dim_stats['min']:.2f}")
            print(f"  Max:  {dim_stats['max']:.2f}")

        # Sample a few images
        print()
        print("=" * 70)
        print("Sample Images")
        print("=" * 70)
        sample_ids = loader.sample_images(n=5)

        for i, image_id in enumerate(sample_ids, 1):
            image_path = loader.get_image_path(image_id)
            exists = "✓" if image_path.exists() else "✗"
            print(f"{i}. {image_id}: {exists} {image_path}")

        print()
        print("✓ Place Pulse loader test completed successfully!")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


if __name__ == "__main__":
    test_loader()
