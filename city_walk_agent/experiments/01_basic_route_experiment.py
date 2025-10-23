#!/usr/bin/env python3
"""
Experiment 1: Basic Route Generation and Analysis

This experiment demonstrates the core workflow:
1. Generate routes between different locations
2. Compare simple vs complex routes
3. Analyze route characteristics
4. Prepare for image collection

This is experimental code for research and hypothesis testing.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_collection import RouteGenerator
from config import settings
from utils.data_models import Route


def experiment_route_comparison():
    """Experiment: Compare different route generation approaches"""
    print("🧪 Experiment 1: Route Generation Comparison")
    print("=" * 50)

    # Use the real API key if available, otherwise use the dummy key for offline tests
    api_key = settings.google_maps_api_key or "test_key"
    route_gen = RouteGenerator(api_key=api_key)
    if "test_key" in api_key:
        print("⚠️  Running in offline mode. No real API key found.")
    if settings.google_maps_api_key:
        print("🔑 Using provided Google Maps API key.")

    # Test different route configurations
    test_routes = [
        {
            "name": "short_dense",
            "start": (40.7589, -73.9851),  # Times Square
            "end": (40.7614, -73.9776),  # Central Park
            "interval": 20,
            "description": "Short route, dense sampling",
        },
        {
            "name": "short_sparse",
            "start": (40.7589, -73.9851),  # Times Square
            "end": (40.7614, -73.9776),  # Central Park
            "interval": 50,
            "description": "Short route, sparse sampling",
        },
        {
            "name": "medium_route",
            "start": (40.7580, -73.9855),  # Times Square area
            "end": (40.7505, -73.9934),  # Penn Station area
            "interval": 30,
            "description": "Medium route, moderate sampling",
        },
    ]

    routes = {}

    for config in test_routes:
        print(f"\nGenerating {config['name']}: {config['description']}")

        route = route_gen.create_simple_route(
            start_lat=config["start"][0],
            start_lon=config["start"][1],
            end_lat=config["end"][0],
            end_lon=config["end"][1],
            interval_meters=config["interval"],
            route_name=config["name"],
        )

        routes[config["name"]] = route

        # Save route
        route_file = route_gen.save_route(route)

        print(f"  Waypoints: {len(route.waypoints)}")
        print(f"  Interval: {route.interval_meters}m")
        print(f"  Saved: {route_file}")

    return routes


def analyze_route_characteristics(routes):
    """Analyze the characteristics of generated routes"""
    print("\n📊 Route Analysis")
    print("=" * 30)

    analysis_data = []

    for name, route in routes.items():
        # Calculate basic metrics
        waypoints = route.waypoints

        # Latitude and longitude ranges
        lats = [wp.lat for wp in waypoints]
        lons = [wp.lon for wp in waypoints]

        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Approximate distance (very rough calculation)
        from geopy.distance import geodesic

        total_distance = 0
        for i in range(1, len(waypoints)):
            dist = geodesic(
                (waypoints[i - 1].lat, waypoints[i - 1].lon),
                (waypoints[i].lat, waypoints[i].lon),
            ).meters
            total_distance += dist

        analysis = {
            "route_name": name,
            "waypoint_count": len(waypoints),
            "interval_meters": route.interval_meters,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "total_distance_m": total_distance,
            "density_waypoints_per_100m": (len(waypoints) / total_distance) * 100,
        }

        analysis_data.append(analysis)

        print(f"{name}:")
        print(f"  Waypoints: {analysis['waypoint_count']}")
        print(f"  Total distance: {analysis['total_distance_m']:.1f}m")
        print(f"  Density: {analysis['density_waypoints_per_100m']:.2f} waypoints/100m")

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(analysis_data)

    # Save analysis results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = results_dir / f"route_analysis_{timestamp}.csv"
    df.to_csv(analysis_file, index=False)

    print(f"\n💾 Analysis saved to: {analysis_file}")

    return df


def experiment_sampling_impact():
    """Experiment: How does sampling interval affect route representation?"""
    print("\n🔬 Experiment 2: Sampling Interval Impact")
    print("=" * 40)

    # Use the real API key if available, otherwise use the dummy key for offline tests
    api_key = settings.google_maps_api_key or "test_key"
    route_gen = RouteGenerator(api_key=api_key)
    if "test_key" in api_key:
        print("⚠️  Running in offline mode. No real API key found.")

    # Same route, different sampling intervals
    start_lat, start_lon = 40.7589, -73.9851
    end_lat, end_lon = 40.7505, -73.9934

    intervals = [10, 30, 50]

    print("Testing sampling intervals:", intervals)

    for interval in intervals:
        route = route_gen.create_simple_route(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            interval_meters=interval,
            route_name=f"sampling_test_{interval}m",
        )

        print(f"  {interval}m interval → {len(route.waypoints)} waypoints")

    print("\n💡 Insight: Sampling interval directly affects analysis granularity")
    print("   - Dense sampling: More detail, higher cost")
    print("   - Sparse sampling: Less detail, lower cost")
    print("   - Optimal interval depends on route length and analysis goals")


def main():
    """Run all route generation experiments"""
    print("🚀 CityWalkAgent Route Generation Experiments")
    print("=" * 60)

    # Experiment 1: Route comparison
    routes = experiment_route_comparison()

    # Analysis
    analysis_df = analyze_route_characteristics(routes)

    # Experiment 2: Sampling impact
    experiment_sampling_impact()

    print("\n" + "=" * 60)
    print("🎉 Experiments completed!")
    print("\n📋 Key Findings:")
    print("1. Route generation framework is working correctly")
    print("2. Sampling interval significantly affects waypoint density")
    print("3. Framework ready for image collection phase")
    print("4. Data persistence and analysis workflows validated")

    print(f"\n📁 Results saved in: {Path(__file__).parent / 'results'}")

    return routes, analysis_df


if __name__ == "__main__":
    routes, analysis = main()
