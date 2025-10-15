import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd
from datetime import datetime

try:
    from ..utils.data_models import Route, Waypoint
    from ..config.settings import settings
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from utils.data_models import Route, Waypoint
    from config.settings import settings


class ImageCollector:
    """Collect street view images using ZenSVI library"""

    def __init__(self, api_key: Optional[str] = None, max_workers: int = 4):
        """
        Initialize ImageCollector

        Args:
            api_key: API key for street view service (Mapillary, Google, etc.)
            max_workers: Number of parallel download workers
        """
        self.api_key = api_key or settings.mapillary_api_key
        self.max_workers = max_workers
        self.zensvi_path = settings.zensvi_root

        # Add ZenSVI to Python path if not already there
        if str(self.zensvi_path / "src") not in sys.path:
            sys.path.insert(0, str(self.zensvi_path / "src"))

        try:
            # Import ZenSVI components
            from zensvi.download import MLYDownloader, GSVDownloader
            self.MLYDownloader = MLYDownloader
            self.GSVDownloader = GSVDownloader
        except ImportError as e:
            raise ImportError(f"Could not import ZenSVI from {self.zensvi_path}. Error: {e}")

    def collect_mapillary_images(
        self,
        route: Route,
        output_dir: Optional[str] = None,
        image_size: int = 2048,
        max_distance: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect Mapillary images for all waypoints in a route

        Args:
            route: Route object containing waypoints
            output_dir: Directory to save images (default: data/images/route_id)
            image_size: Size of images to download
            max_distance: Maximum distance from waypoint to consider images (meters)

        Returns:
            List of download results with metadata
        """
        if not self.api_key:
            raise ValueError("Mapillary API key is required for image collection")

        # Set up output directory
        if not output_dir:
            output_dir = settings.images_dir / route.route_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Mapillary downloader
        downloader = self.MLYDownloader(
            key=self.api_key,
            max_workers=self.max_workers
        )

        results = []

        # Collect images for each waypoint
        for waypoint in route.waypoints:
            try:
                print(f"Collecting images for waypoint {waypoint.sequence_id} at ({waypoint.lat}, {waypoint.lon})")

                # Create waypoint-specific directory
                waypoint_dir = output_dir / f"waypoint_{waypoint.sequence_id:03d}"
                waypoint_dir.mkdir(exist_ok=True)

                # Download images near this waypoint
                download_result = downloader.download_images_from_bbox(
                    bbox=[waypoint.lon - 0.001, waypoint.lat - 0.001,
                          waypoint.lon + 0.001, waypoint.lat + 0.001],
                    output_dir=str(waypoint_dir),
                    image_size=image_size,
                    max_workers=self.max_workers
                )

                # Update waypoint with image path if images were downloaded
                image_files = list(waypoint_dir.glob("*.jpg"))
                if image_files:
                    # Use the first (closest) image
                    waypoint.image_path = str(image_files[0])

                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "images_downloaded": len(image_files),
                    "image_path": str(image_files[0]) if image_files else None,
                    "download_result": download_result,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

            except Exception as e:
                print(f"Error collecting images for waypoint {waypoint.sequence_id}: {e}")
                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "images_downloaded": 0,
                    "image_path": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

        # Save collection metadata
        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "route_id": route.route_id,
                "collection_timestamp": datetime.now().isoformat(),
                "total_waypoints": len(route.waypoints),
                "results": results
            }, f, indent=2)

        return results

    def collect_google_street_view_images(
        self,
        route: Route,
        output_dir: Optional[str] = None,
        image_size: str = "640x640",
        fov: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Collect Google Street View images for all waypoints in a route

        Args:
            route: Route object containing waypoints
            output_dir: Directory to save images
            image_size: Size of images (e.g., "640x640")
            fov: Field of view in degrees

        Returns:
            List of download results with metadata
        """
        if not settings.google_maps_api_key:
            raise ValueError("Google Maps API key is required for Street View images")

        # Set up output directory
        if not output_dir:
            output_dir = settings.images_dir / route.route_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Google Street View downloader
        downloader = self.GSVDownloader(
            key=settings.google_maps_api_key
        )

        results = []

        # Create DataFrame for batch download
        waypoints_df = pd.DataFrame([
            {
                "id": f"waypoint_{wp.sequence_id:03d}",
                "lat": wp.lat,
                "lon": wp.lon,
                "heading": wp.heading or 0
            }
            for wp in route.waypoints
        ])

        try:
            # Batch download images
            download_result = downloader.download_images(
                input_csv_file=waypoints_df,
                output_dir=str(output_dir),
                image_size=image_size,
                fov=fov,
                max_workers=self.max_workers
            )

            # Update waypoints with image paths
            for waypoint in route.waypoints:
                image_filename = f"waypoint_{waypoint.sequence_id:03d}.jpg"
                image_path = output_dir / image_filename

                if image_path.exists():
                    waypoint.image_path = str(image_path)

                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "image_path": str(image_path) if image_path.exists() else None,
                    "download_success": image_path.exists(),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

        except Exception as e:
            print(f"Error in batch download: {e}")
            # Fall back to individual downloads
            results = self._download_gsv_individually(route, output_dir, image_size, fov)

        # Save collection metadata
        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "route_id": route.route_id,
                "collection_timestamp": datetime.now().isoformat(),
                "total_waypoints": len(route.waypoints),
                "platform": "google_street_view",
                "results": results
            }, f, indent=2)

        return results

    def _download_gsv_individually(
        self,
        route: Route,
        output_dir: Path,
        image_size: str,
        fov: int
    ) -> List[Dict[str, Any]]:
        """Fallback method to download GSV images individually"""
        downloader = self.GSVDownloader(key=settings.google_maps_api_key)
        results = []

        for waypoint in route.waypoints:
            try:
                image_filename = f"waypoint_{waypoint.sequence_id:03d}.jpg"
                image_path = output_dir / image_filename

                # Download single image
                success = downloader.download_single_image(
                    lat=waypoint.lat,
                    lon=waypoint.lon,
                    output_path=str(image_path),
                    size=image_size,
                    fov=fov,
                    heading=waypoint.heading or 0
                )

                if success:
                    waypoint.image_path = str(image_path)

                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "image_path": str(image_path) if success else None,
                    "download_success": success,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

            except Exception as e:
                print(f"Error downloading image for waypoint {waypoint.sequence_id}: {e}")
                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "image_path": None,
                    "download_success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

        return results

    def validate_images(self, route: Route) -> Dict[str, Any]:
        """
        Validate that all waypoints have valid images

        Args:
            route: Route object to validate

        Returns:
            Validation summary
        """
        total_waypoints = len(route.waypoints)
        waypoints_with_images = 0
        missing_images = []

        for waypoint in route.waypoints:
            if waypoint.image_path and Path(waypoint.image_path).exists():
                waypoints_with_images += 1
            else:
                missing_images.append(waypoint.sequence_id)

        return {
            "total_waypoints": total_waypoints,
            "waypoints_with_images": waypoints_with_images,
            "missing_images": missing_images,
            "coverage_percentage": (waypoints_with_images / total_waypoints) * 100,
            "validation_timestamp": datetime.now().isoformat()
        }