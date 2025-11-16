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
    """Demonstrate production-ready directional collection with advanced features."""

    print("=== Directional Image Collection with Advanced Features ===\n")

    # Generate a routed path using Google Maps directions
    route_gen = RouteGenerator()
    route = route_gen.create_google_maps_route(
        origin=(40.758896, -73.985130),
        destination=(40.785091, -73.968285),
        interval_meters=30,
        route_name="demo_advanced",
    )

    output_dir = Path("data/routes/demo_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect with all directional features enabled
    collector = ImageCollector(api_key=settings.google_maps_api_key)
    
    print("Features enabled:")
    print("  ✓ Walking-direction alignment (use_route_direction=True)")
    print("  ✓ Lookahead for smooth corners (lookahead_distance=2)")
    print("  ✓ Corner/intersection detection (detect_corners=True)")
    print("  ✓ Single forward-facing view (all_around=False)")
    print()
    
    results = collector.collect_google_street_view_images_static(
        route,
        output_dir=str(output_dir),
        use_route_direction=True,   # Calculate heading from route
        all_around=False,           # Single direction only
        fov=90,                     # Human field of view
        pitch=-5,                   # Natural downward tilt
        lookahead_distance=2,       # Look 2 waypoints ahead
        detect_corners=True,        # Detect intersections
        corner_threshold=30.0       # 30° = corner
    )

    successful = sum(1 for r in results if r.get('download_success'))
    corners = sum(1 for r in results if r.get('is_corner', False))
    
    print(f"Collected {successful}/{len(results)} directional images")
    print(f"Detected {corners} corners along the route")
    print(f"Output: {output_dir}")
    print()
    
    # Show sample corner detections
    corner_results = [r for r in results if r.get('is_corner')]
    if corner_results:
        print("Sample corners detected:")
        for r in corner_results[:3]:
            print(f"  Waypoint {r['waypoint_id']}: {r.get('angle_change', 0):.1f}° turn")


def main() -> None:
    print("=" * 80)
    print("  Directional Image Collection Demo")
    print("=" * 80)
    print()

    demo_production_directional_collection()

    print("\n" + "=" * 80)
    print("Features integrated into ImageCollector:")
    print("  • Walking-direction alignment")
    print("  • Lookahead for smooth corners")
    print("  • Corner/intersection detection")
    print("=" * 80)


if __name__ == "__main__":
    main()
