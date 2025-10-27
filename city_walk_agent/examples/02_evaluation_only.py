#!/usr/bin/env python3
"""
Example 2: Evaluation only (no full pipeline)

Demonstrates using just the evaluation engine to evaluate images.

This example shows how to:
1. Load a framework
2. Initialize the VLM evaluator
3. Evaluate images directly
4. Access dimension scores
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import settings, load_framework
from evaluation import Evaluator
from evaluation.vlm_client import VLMConfig


def main():
    """Run evaluation-only example"""
    print("="*70)
    print("CityWalkAgent - Evaluation Only Example")
    print("="*70)

    # Load framework
    print("\n1. Loading framework...")
    framework = load_framework("sagai_2025")
    print(f"   Framework: {framework['framework_name_cn']}")
    print(f"   Dimensions: {framework['num_dimensions']}")

    # Create VLM config
    print("\n2. Configuring VLM...")
    vlm_config = VLMConfig(
        api_key=settings.qwen_vlm_api_key,
        model=settings.qwen_vlm_model,
        api_url=settings.qwen_vlm_api_url
    )

    # Initialize evaluator
    evaluator = Evaluator(
        vlm_config=vlm_config,
        framework=framework,
        max_concurrent=5
    )

    # Example image paths (replace with actual images)
    image_paths = [
        "data/images/gsv/example_route/waypoint_000.jpg",
        "data/images/gsv/example_route/waypoint_001.jpg",
        "data/images/gsv/example_route/waypoint_002.jpg",
    ]

    print(f"\n3. Evaluating {len(image_paths)} images...")

    # Evaluate images
    results = evaluator.evaluate_images(image_paths)

    # Display results
    print(f"\n4. Results:")
    for i, result in enumerate(results, 1):
        print(f"\n   Image {result['image_id']}:")
        print(f"     Dimension: {result['dimension_name']}")
        print(f"     Score:     {result['score']:.1f}/10")
        print(f"     Reasoning: {result['reasoning'][:80]}...")

        if i >= 6:  # Show first 6 results
            print(f"\n   ... and {len(results) - 6} more results")
            break

    # Show statistics
    print(f"\n5. Statistics:")
    stats = evaluator.get_statistics()
    print(f"   Total API calls: {stats['vlm_stats']['total_calls']}")
    print(f"   Avg time:        {stats['vlm_stats']['avg_time']:.2f}s")

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
