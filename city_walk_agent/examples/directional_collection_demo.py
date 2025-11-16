"""
Directional Image Collection Demo (Production)

Shows how to collect north-aligned Street View images using the production
ImageCollector with route-based directional alignment enabled. The previous
experimental DirectionalImageCollector has been removed; this demo reflects
the supported code path.

Run:
    python examples/directional_collection_demo.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.data_collection.route_generator import RouteGenerator
from src.data_collection.image_collector import ImageCollector


def demo_production_directional_collection() -> None:
    """Demonstrate production-ready directional collection using ImageCollector."""

    print("=== Production Approach: ImageCollector ===\n")

    # Generate a routed path using Google Maps directions
    route_gen = RouteGenerator()
    route = route_gen.create_google_maps_route(
        origin=(40.758896, -73.985130),
        destination=(40.785091, -73.968285),
        interval_meters=30,
        route_name="demo_production",
    )

    output_dir = Path("data/routes/demo_production")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect with directional alignment
    collector = ImageCollector(api_key=settings.google_maps_api_key)
    results = collector.collect_google_street_view_images_static(
        route,
        output_dir=str(output_dir),
        use_route_direction=True,  # Enable directional alignment
        all_around=False,  # Single direction only
        fov=90,
        pitch=-5,
    )

    print(f"Collected {len(results)} directional images")
    print(f"Output: {output_dir}")


def main() -> None:
    print("=" * 80)
    print("  Directional Image Collection Demo (Production)")
    print("=" * 80)
    print("Note: The experimental DirectionalImageCollector has been removed;")
    print("      this demo shows the supported production flow.\n")

    demo_production_directional_collection()

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
