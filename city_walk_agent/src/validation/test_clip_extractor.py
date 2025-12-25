"""Simple test script for CLIP feature extractor.

This demonstrates the basic functionality with a small sample of images.
"""

from pathlib import Path

from src.validation.clip_extractor import CLIPExtractor


def test_extractor():
    """Test the CLIP feature extractor."""
    print("=" * 70)
    print("Testing CLIP Feature Extractor")
    print("=" * 70)
    print()

    # Initialize extractor
    extractor = CLIPExtractor(
        model_name="openai/clip-vit-base-patch32",
        device="auto",
        batch_size=8,
    )
    print()

    # Find some test images
    print("Looking for test images...")
    test_images_dir = Path("data/images")

    if not test_images_dir.exists():
        print(f"⚠️  Test images directory not found: {test_images_dir}")
        print("Please ensure you have some images in the data/images directory")
        return

    # Get first few images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    test_images = []

    for ext in image_extensions:
        test_images.extend(list(test_images_dir.glob(ext)))
        if len(test_images) >= 10:
            break

    test_images = test_images[:10]

    if not test_images:
        print(f"⚠️  No images found in {test_images_dir}")
        return

    print(f"Found {len(test_images)} test images")
    print()

    # Test single image extraction
    print("=" * 70)
    print("Test 1: Single Image Extraction")
    print("=" * 70)

    test_image = test_images[0]
    print(f"Extracting features from: {test_image.name}")

    features = extractor.extract_single(test_image)
    print(f"Feature shape: {features.shape}")
    print(f"Feature dimension: {features.shape[0]}")
    print(f"L2 norm: {float(np.linalg.norm(features)):.4f}")
    print()

    # Verify features
    import numpy as np

    verification = extractor.verify_features(features.reshape(1, -1))
    print("Verification:")
    print(f"  Shape matches: {verification['feature_dim_matches']}")
    print(f"  Normalized: {verification['normalized']}")
    print(f"  Mean L2 norm: {verification['mean_l2_norm']:.4f}")
    print(f"  Has NaN: {verification['has_nan']}")
    print(f"  Has Inf: {verification['has_inf']}")
    print()

    # Test batch extraction
    print("=" * 70)
    print("Test 2: Batch Extraction")
    print("=" * 70)

    print(f"Extracting features from {len(test_images)} images...")
    batch_features = extractor.extract_batch(test_images, show_progress=True)

    print(f"\nBatch features shape: {batch_features.shape}")
    print(
        f"Expected: ({len(test_images)}, {extractor.feature_dim})"
    )
    print()

    # Verify batch features
    batch_verification = extractor.verify_features(batch_features)
    print("Batch Verification:")
    print(f"  Shape: {batch_verification['shape']}")
    print(f"  Dimension matches: {batch_verification['feature_dim_matches']}")
    print(f"  Normalized: {batch_verification['normalized']}")
    print(f"  Mean L2 norm: {batch_verification['mean_l2_norm']:.4f}")
    print(f"  Std L2 norm: {batch_verification['std_l2_norm']:.4f}")
    print()

    # Test caching
    print("=" * 70)
    print("Test 3: Caching")
    print("=" * 70)

    cache_path = Path("data/test_cache/clip_features.npy")
    print(f"Cache path: {cache_path}")
    print()

    # First extraction (will create cache)
    print("First extraction (creating cache)...")
    cached_features = extractor.extract_and_cache(
        test_images, cache_path, force_recompute=True
    )
    print()

    # Second extraction (will load from cache)
    print("Second extraction (loading from cache)...")
    loaded_features = extractor.extract_and_cache(
        test_images, cache_path, force_recompute=False
    )
    print()

    # Verify they match
    features_match = np.allclose(cached_features, loaded_features)
    print(f"Cached and loaded features match: {features_match}")
    print()

    # Test similarity computation
    print("=" * 70)
    print("Test 4: Cosine Similarity")
    print("=" * 70)

    # Compute similarity between first image and all others
    query_features = batch_features[0]
    similarities = np.dot(batch_features, query_features)

    print(f"Query image: {test_images[0].name}")
    print("\nTop 5 most similar images:")

    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    for rank, idx in enumerate(sorted_indices[:5], 1):
        print(
            f"  {rank}. {test_images[idx].name}: {similarities[idx]:.4f}"
        )

    print()
    print("✓ CLIP extractor test completed successfully!")


if __name__ == "__main__":
    import numpy as np

    test_extractor()
