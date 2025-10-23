#!/usr/bin/env python3
"""
Experiment 2: Image Collection Testing and Comparison

This experiment tests the image collection functionality:
1. Load existing routes from Experiment 01
2. Test Google Street View image collection
3. Test Mapillary image collection
4. Validate image quality and completeness
5. Compare collection methods (speed, coverage, cost)
6. Test parallel processing optimization

This is experimental code for research and hypothesis testing.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import traceback

# Load environment variables from .env file
load_dotenv()

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_collection import ImageCollector
from config import settings
from utils.data_models import Route

# Constants
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2


def load_latest_routes() -> Dict[str, Route]:
    """
    Load the most recent routes from experiment 01

    Returns:
        Dictionary mapping route names to Route objects
    """
    print("📂 Loading routes from previous experiment...")

    routes_dir = settings.data_dir / "routes"
    if not routes_dir.exists():
        raise FileNotFoundError(f"Routes directory not found: {routes_dir}")

    # Group route files by base name (without timestamp)
    route_files = {}
    for route_file in routes_dir.glob("*.json"):
        # Parse filename: name_route_timestamp.json
        # Handle case where filename has multiple "route" instances
        stem = route_file.stem
        if "_route_" in stem:
            # Find the last occurrence of _route_
            last_route_idx = stem.rfind("_route_")
            base_name = stem[:last_route_idx]
            timestamp_str = stem[last_route_idx + 7 :]  # Skip "_route_"

            try:
                timestamp = int(timestamp_str)
                if (
                    base_name not in route_files
                    or timestamp > route_files[base_name][1]
                ):
                    route_files[base_name] = (route_file, timestamp)
            except ValueError:
                print(f"  ⚠️  Skipping file with invalid timestamp: {route_file.name}")

    # Load the most recent version of each route
    routes = {}
    for base_name, (route_file, timestamp) in route_files.items():
        try:
            with open(route_file, "r") as f:
                route_data = json.load(f)
                # Use Pydantic's model_validate (v2) or parse_obj (v1)
                try:
                    route = Route.model_validate(route_data)  # Pydantic v2
                except AttributeError:
                    route = Route.parse_obj(route_data)  # Pydantic v1
                routes[base_name] = route
                print(f"  ✅ Loaded {base_name}: {len(route.waypoints)} waypoints")
        except Exception as e:
            print(f"  ❌ Failed to load {route_file}: {e}")

    if not routes:
        raise ValueError("No valid routes found!")

    print(f"\n📊 Loaded {len(routes)} routes")
    return routes


def experiment_gsv_collection(
    route: Route, max_workers: int = 4, buffer: int = 50
) -> Dict[str, Any]:
    """
    Experiment: Google Street View image collection

    Args:
        route: Route to collect images for
        max_workers: Number of parallel workers
        buffer: Search radius from waypoint in meters

    Returns:
        Collection results with timing and statistics
    """
    print(f"\n🖼️  Experiment: Google Street View Collection")
    print(f"   Route: {route.route_id}")
    print(f"   Waypoints: {len(route.waypoints)}")
    print(f"   Workers: {max_workers}")
    print("=" * 60)

    if not settings.google_maps_api_key:
        print("⏭️  Skipping: No Google Maps API key found")
        return {
            "platform": "google_street_view",
            "skipped": True,
            "reason": "No API key",
        }

    try:
        # Initialize collector
        collector = ImageCollector(max_workers=max_workers)

        # Create output directory
        output_dir = settings.images_dir / "gsv" / route.route_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect images with timing
        start_time = time.time()

        print(f"📥 Starting download to {output_dir}...")
        results = collector.collect_google_street_view_images(
            route=route, output_dir=str(output_dir), buffer=buffer
        )

        end_time = time.time()
        duration = end_time - start_time

        # Calculate statistics
        successful = sum(1 for r in results if r.get("download_success", False))
        failed = len(results) - successful
        success_rate = (successful / len(results) * 100) if results else 0

        print(f"\n✅ Collection completed in {duration:.2f}s")
        print(f"   Success: {successful}/{len(results)} ({success_rate:.1f}%)")
        print(f"   Failed: {failed}")
        print(f"   Speed: {duration/len(results):.2f}s per waypoint")

        return {
            "platform": "google_street_view",
            "route_id": route.route_id,
            "total_waypoints": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate,
            "duration_seconds": duration,
            "seconds_per_waypoint": duration / len(results) if results else 0,
            "output_dir": str(output_dir),
            "max_workers": max_workers,
            "buffer": buffer,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"❌ Error during GSV collection: {e}")
        traceback.print_exc()
        return {
            "platform": "google_street_view",
            "route_id": route.route_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }


def experiment_mapillary_collection(
    route: Route, max_workers: int = 4, buffer: int = 50
) -> Dict[str, Any]:
    """
    Experiment: Mapillary image collection

    Args:
        route: Route to collect images for
        max_workers: Number of parallel workers
        buffer: Search radius from waypoint in meters

    Returns:
        Collection results with timing and statistics
    """
    print(f"\n🖼️  Experiment: Mapillary Collection")
    print(f"   Route: {route.route_id}")
    print(f"   Waypoints: {len(route.waypoints)}")
    print(f"   Workers: {max_workers}")
    print("=" * 60)

    if not settings.mapillary_api_key:
        print("⏭️  Skipping: No Mapillary API key found")
        return {"platform": "mapillary", "skipped": True, "reason": "No API key"}

    try:
        # Initialize collector
        collector = ImageCollector(
            api_key=settings.mapillary_api_key, max_workers=max_workers
        )

        # Create output directory
        output_dir = settings.images_dir / "mapillary" / route.route_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect images with timing
        start_time = time.time()

        print(f"📥 Starting download to {output_dir}...")
        results = collector.collect_mapillary_images(
            route=route, output_dir=str(output_dir), buffer=buffer
        )

        end_time = time.time()
        duration = end_time - start_time

        # Calculate statistics
        successful = sum(1 for r in results if r.get("images_downloaded", 0) > 0)
        failed = len(results) - successful
        success_rate = (successful / len(results) * 100) if results else 0
        total_images = sum(r.get("images_downloaded", 0) for r in results)

        print(f"\n✅ Collection completed in {duration:.2f}s")
        print(
            f"   Waypoints with images: {successful}/{len(results)} ({success_rate:.1f}%)"
        )
        print(f"   Total images: {total_images}")
        print(f"   Failed: {failed}")
        print(f"   Speed: {duration/len(results):.2f}s per waypoint")

        return {
            "platform": "mapillary",
            "route_id": route.route_id,
            "total_waypoints": len(results),
            "waypoints_with_images": successful,
            "failed": failed,
            "success_rate": success_rate,
            "total_images": total_images,
            "avg_images_per_waypoint": total_images / len(results) if results else 0,
            "duration_seconds": duration,
            "seconds_per_waypoint": duration / len(results) if results else 0,
            "output_dir": str(output_dir),
            "max_workers": max_workers,
            "buffer": buffer,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"❌ Error during Mapillary collection: {e}")
        traceback.print_exc()
        return {
            "platform": "mapillary",
            "route_id": route.route_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }


def validate_image_collection(
    output_dir: Path, platform: str, expected_count: int
) -> Dict[str, Any]:
    """
    Validate collected images for quality and completeness

    Args:
        output_dir: Directory containing images
        platform: Platform name (gsv or mapillary)
        expected_count: Expected number of waypoints

    Returns:
        Validation results
    """
    print(f"\n🔍 Validating {platform} images in {output_dir}")

    if not output_dir.exists():
        return {
            "platform": platform,
            "valid": False,
            "error": "Output directory does not exist",
        }

    # Find all image files
    image_files = list(output_dir.glob("**/*.jpg")) + list(output_dir.glob("**/*.png"))

    validation_results = {
        "platform": platform,
        "output_dir": str(output_dir),
        "expected_waypoints": expected_count,
        "total_images": len(image_files),
        "valid_images": 0,
        "corrupted_images": 0,
        "missing_waypoints": 0,
        "image_stats": {
            "total_size_mb": 0,
            "avg_size_kb": 0,
            "min_size_kb": float("inf"),
            "max_size_kb": 0,
            "dimensions": [],
        },
        "issues": [],
    }

    total_size_bytes = 0
    valid_count = 0

    for img_file in image_files:
        try:
            # Check file size
            file_size = img_file.stat().st_size
            file_size_kb = file_size / 1024
            total_size_bytes += file_size

            # Try to open and validate image
            with Image.open(img_file) as img:
                width, height = img.size

                # Check if image is valid (not corrupted)
                img.verify()

                valid_count += 1
                validation_results["image_stats"]["dimensions"].append(
                    {"width": width, "height": height, "file": img_file.name}
                )

                # Update size stats
                validation_results["image_stats"]["min_size_kb"] = min(
                    validation_results["image_stats"]["min_size_kb"], file_size_kb
                )
                validation_results["image_stats"]["max_size_kb"] = max(
                    validation_results["image_stats"]["max_size_kb"], file_size_kb
                )

        except Exception as e:
            validation_results["corrupted_images"] += 1
            validation_results["issues"].append(
                {"file": str(img_file), "error": str(e)}
            )

    validation_results["valid_images"] = valid_count
    validation_results["image_stats"]["total_size_mb"] = total_size_bytes / (
        1024 * 1024
    )

    if valid_count > 0:
        validation_results["image_stats"]["avg_size_kb"] = (
            total_size_bytes / valid_count / 1024
        )

    # Calculate coverage
    validation_results["coverage_percentage"] = (
        (valid_count / expected_count * 100) if expected_count > 0 else 0
    )
    validation_results["missing_waypoints"] = max(0, expected_count - valid_count)

    # Print summary
    print(f"   Total images: {len(image_files)}")
    print(f"   Valid: {valid_count}")
    print(f"   Corrupted: {validation_results['corrupted_images']}")
    print(f"   Coverage: {validation_results['coverage_percentage']:.1f}%")
    print(f"   Total size: {validation_results['image_stats']['total_size_mb']:.2f} MB")
    print(f"   Avg size: {validation_results['image_stats']['avg_size_kb']:.2f} KB")

    if validation_results["issues"]:
        print(f"   ⚠️  {len(validation_results['issues'])} issues found")

    return validation_results


def compare_collection_methods(
    gsv_results: Dict[str, Any], mapillary_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compare GSV and Mapillary collection methods

    Args:
        gsv_results: Google Street View collection results
        mapillary_results: Mapillary collection results

    Returns:
        Comparison DataFrame
    """
    print("\n📊 Comparing Collection Methods")
    print("=" * 60)

    comparison_data = []

    # GSV metrics
    if not gsv_results.get("skipped", False) and "error" not in gsv_results:
        comparison_data.append(
            {
                "Platform": "Google Street View",
                "Total Waypoints": gsv_results.get("total_waypoints", 0),
                "Successful": gsv_results.get("successful", 0),
                "Success Rate (%)": gsv_results.get("success_rate", 0),
                "Duration (s)": gsv_results.get("duration_seconds", 0),
                "Speed (s/waypoint)": gsv_results.get("seconds_per_waypoint", 0),
                "Workers": gsv_results.get("max_workers", 0),
                "Images per Waypoint": 1.0,  # GSV typically provides 1 image per waypoint
            }
        )

    # Mapillary metrics
    if not mapillary_results.get("skipped", False) and "error" not in mapillary_results:
        comparison_data.append(
            {
                "Platform": "Mapillary",
                "Total Waypoints": mapillary_results.get("total_waypoints", 0),
                "Successful": mapillary_results.get("waypoints_with_images", 0),
                "Success Rate (%)": mapillary_results.get("success_rate", 0),
                "Duration (s)": mapillary_results.get("duration_seconds", 0),
                "Speed (s/waypoint)": mapillary_results.get("seconds_per_waypoint", 0),
                "Workers": mapillary_results.get("max_workers", 0),
                "Images per Waypoint": mapillary_results.get(
                    "avg_images_per_waypoint", 0
                ),
            }
        )

    if not comparison_data:
        print("⚠️  No valid results to compare")
        return pd.DataFrame()

    df = pd.DataFrame(comparison_data)

    print("\n" + df.to_string(index=False))

    # Analysis
    print("\n💡 Analysis:")
    if len(comparison_data) == 2:
        gsv_speed = comparison_data[0]["Speed (s/waypoint)"]
        mly_speed = comparison_data[1]["Speed (s/waypoint)"]

        if gsv_speed > 0 and mly_speed > 0:
            if gsv_speed < mly_speed:
                speedup = mly_speed / gsv_speed
                print(f"   - GSV is {speedup:.2f}x faster than Mapillary")
            else:
                speedup = gsv_speed / mly_speed
                print(f"   - Mapillary is {speedup:.2f}x faster than GSV")

        gsv_success = comparison_data[0]["Success Rate (%)"]
        mly_success = comparison_data[1]["Success Rate (%)"]
        print(f"   - GSV success rate: {gsv_success:.1f}%")
        print(f"   - Mapillary success rate: {mly_success:.1f}%")

        mly_images = comparison_data[1]["Images per Waypoint"]
        print(
            f"   - Mapillary provides {mly_images:.1f} images per waypoint on average"
        )

    return df


def experiment_parallel_processing(
    route: Route, worker_counts: List[int] = [1, 2, 4, 8]
) -> Dict[str, Any]:
    """
    Experiment: Test different parallel processing settings

    Args:
        route: Route to test with
        worker_counts: List of worker counts to test

    Returns:
        Performance results for each worker count
    """
    print("\n⚡ Experiment: Parallel Processing Optimization")
    print(f"   Testing worker counts: {worker_counts}")
    print("=" * 60)

    # Use a small subset of the route for testing
    test_waypoints = min(10, len(route.waypoints))
    start_wp = route.waypoints[0]
    end_wp = route.waypoints[test_waypoints - 1]

    test_route = Route(
        route_id=f"{route.route_id}_parallel_test",
        start_lat=start_wp.lat,
        start_lon=start_wp.lon,
        end_lat=end_wp.lat,
        end_lon=end_wp.lon,
        waypoints=route.waypoints[:test_waypoints],
        route_name=f"{route.route_name}_test" if route.route_name else None,
        interval_meters=route.interval_meters,
    )

    print(f"   Using {test_waypoints} waypoints for testing")

    results = []

    for workers in worker_counts:
        print(f"\n🔧 Testing with {workers} worker(s)...")

        result = experiment_gsv_collection(
            route=test_route,
            max_workers=workers,
            buffer=30,  # Smaller buffer for faster testing
        )

        if "error" not in result and not result.get("skipped", False):
            results.append(
                {
                    "workers": workers,
                    "duration": result.get("duration_seconds", 0),
                    "speed": result.get("seconds_per_waypoint", 0),
                    "success_rate": result.get("success_rate", 0),
                }
            )

            print(f"   Duration: {result.get('duration_seconds', 0):.2f}s")
            print(
                f"   Speed: {result.get('seconds_per_waypoint', 0):.2f}s per waypoint"
            )

    if results:
        # Find optimal worker count
        optimal = min(results, key=lambda x: x["speed"])

        print("\n📈 Results:")
        for r in results:
            marker = " ⭐" if r == optimal else ""
            print(f"   {r['workers']} workers: {r['speed']:.2f}s per waypoint{marker}")

        print(
            f"\n✅ Optimal: {optimal['workers']} workers ({optimal['speed']:.2f}s per waypoint)"
        )

        return {
            "test_waypoints": test_waypoints,
            "results": results,
            "optimal_workers": optimal["workers"],
            "optimal_speed": optimal["speed"],
        }
    else:
        print("⚠️  No valid results from parallel processing tests")
        return {"error": "No valid results"}


def main():
    """Run all image collection experiments"""
    print("🚀 CityWalkAgent Image Collection Experiments")
    print("=" * 60)

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load routes from experiment 01
    try:
        routes = load_latest_routes()
    except Exception as e:
        print(f"❌ Failed to load routes: {e}")
        return

    # Select a test route (use the shortest one for efficiency)
    test_route_name = "short_sparse"  # Smallest route for quick testing
    if test_route_name not in routes:
        test_route_name = list(routes.keys())[0]

    test_route = routes[test_route_name]
    print(f"\n🎯 Selected test route: {test_route_name}")
    print(f"   Waypoints: {len(test_route.waypoints)}")

    # Experiment 1: Google Street View Collection
    gsv_results = experiment_gsv_collection(route=test_route, max_workers=4, buffer=50)

    # Experiment 2: Mapillary Collection
    mapillary_results = experiment_mapillary_collection(
        route=test_route, max_workers=4, buffer=50
    )

    # Validate collections
    validation_results = {}

    if not gsv_results.get("skipped", False) and "error" not in gsv_results:
        gsv_output = Path(gsv_results["output_dir"])
        validation_results["gsv"] = validate_image_collection(
            output_dir=gsv_output,
            platform="Google Street View",
            expected_count=gsv_results["total_waypoints"],
        )

    if not mapillary_results.get("skipped", False) and "error" not in mapillary_results:
        mly_output = Path(mapillary_results["output_dir"])
        validation_results["mapillary"] = validate_image_collection(
            output_dir=mly_output,
            platform="Mapillary",
            expected_count=mapillary_results["total_waypoints"],
        )

    # Compare methods
    comparison_df = compare_collection_methods(gsv_results, mapillary_results)

    if not comparison_df.empty:
        comparison_file = results_dir / f"collection_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n💾 Comparison saved to: {comparison_file}")

    # # Experiment 3: Parallel Processing (only if GSV available)
    # parallel_results = None
    # if not gsv_results.get("skipped", False):
    #     parallel_results = experiment_parallel_processing(
    #         route=test_route, worker_counts=[1, 2, 4, 8]
    #     )

    # Save comprehensive log
    log_data = {
        "experiment": "02_image_collection",
        "timestamp": timestamp,
        "test_route": test_route_name,
        "gsv_results": gsv_results,
        "mapillary_results": mapillary_results,
        "validation_results": validation_results,
        # "parallel_processing": parallel_results,
    }

    log_file = results_dir / f"image_collection_log_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n💾 Detailed log saved to: {log_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Experiments Completed!")
    print("\n📋 Key Findings:")

    findings = []

    if not gsv_results.get("skipped", False) and "error" not in gsv_results:
        findings.append(
            f"1. GSV: {gsv_results.get('success_rate', 0):.1f}% success rate, {gsv_results.get('seconds_per_waypoint', 0):.2f}s per waypoint"
        )

    if not mapillary_results.get("skipped", False) and "error" not in mapillary_results:
        findings.append(
            f"2. Mapillary: {mapillary_results.get('success_rate', 0):.1f}% success rate, {mapillary_results.get('avg_images_per_waypoint', 0):.1f} images/waypoint"
        )

    if validation_results:
        findings.append(
            f"3. Image validation completed for {len(validation_results)} platform(s)"
        )

    # if parallel_results and "error" not in parallel_results:
    #     findings.append(
    #         f"4. Optimal parallelization: {parallel_results.get('optimal_workers', 'N/A')} workers"
    #     )

    if not findings:
        findings.append("No successful experiments - check API keys and configuration")

    for finding in findings:
        print(finding)

    print(f"\n📁 Results saved in: {results_dir}")

    return {
        "routes": routes,
        "gsv_results": gsv_results,
        "mapillary_results": mapillary_results,
        "validation_results": validation_results,
        # "parallel_results": parallel_results,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    experiment_results = main()
