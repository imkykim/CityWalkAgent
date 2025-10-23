import sys
import os
import math
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

import requests

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
            from zensvi.download import KVDownloader

            self.KVDownloader = KVDownloader
        except ImportError as e:
            raise ImportError(
                f"Could not import ZenSVI from {self.zensvi_path}. Error: {e}"
            )

    def collect_mapillary_images(
        self,
        route: Route,
        output_dir: Optional[str] = None,
        image_size: int = 2048,
        max_distance: int = 50,
        buffer: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect Mapillary images for all waypoints in a route

        Args:
            route: Route object containing waypoints
            output_dir: Directory to save images (default: data/images/route_id)
            image_size: Size of images to download
            max_distance: Legacy search distance in meters (used if buffer not provided)
            buffer: Preferred search distance in meters for compatibility with experiments

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

        results = []

        # Collect images for each waypoint
        for waypoint in route.waypoints:
            try:
                print(
                    f"Collecting images for waypoint {waypoint.sequence_id} at ({waypoint.lat}, {waypoint.lon})"
                )

                # Create waypoint-specific directory
                waypoint_dir = output_dir / f"waypoint_{waypoint.sequence_id:03d}"
                waypoint_dir.mkdir(exist_ok=True)

                candidates = self._fetch_mapillary_candidates(
                    lat=waypoint.lat,
                    lon=waypoint.lon,
                    image_size=image_size,
                    max_distance=buffer if buffer is not None else max_distance,
                )

                image_metadata = None
                image_path = waypoint_dir / f"waypoint_{waypoint.sequence_id:03d}.jpg"
                downloaded = False

                if not candidates:
                    search_limit = buffer if buffer is not None else max_distance
                    print(
                        f"  ⚠️ No Mapillary candidates found within ~{search_limit}m for waypoint {waypoint.sequence_id}"
                    )

                for candidate in candidates:
                    try:
                        if image_path.exists():
                            image_path.unlink()
                        self._download_image(
                            candidate["image_url"],
                            image_path,
                            timeout=20,
                        )
                        waypoint.image_path = str(image_path)
                        image_metadata = candidate
                        downloaded = True
                        distance = candidate.get("distance_m")
                        radius_used = candidate.get("radius")
                        distance_str = (
                            f"~{distance:.1f}m" if distance is not None else "unknown distance"
                        )
                        radius_str = f"{radius_used}m" if radius_used is not None else "unknown radius"
                        print(
                            f"  ✓ Mapillary image {candidate.get('image_id')} selected at {distance_str} (search radius {radius_str})"
                        )
                        break
                    except Exception as download_error:
                        print(
                            f"Warning: failed to download Mapillary image {candidate.get('image_id')}: {download_error}"
                        )
                        if image_path.exists():
                            image_path.unlink(missing_ok=True)

                if not downloaded:
                    image_path = None
                    waypoint.image_path = None
                    print(
                        f"  ❌ Mapillary download failed for waypoint {waypoint.sequence_id} after exhausting candidates"
                    )
                elif not image_path.exists():
                    downloaded = False
                    image_metadata = None
                    waypoint.image_path = None
                    image_path = None

                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "images_downloaded": 1 if downloaded else 0,
                    "image_path": str(image_path) if image_path else None,
                    "mapillary_image_id": (
                        image_metadata.get("image_id") if image_metadata else None
                    ),
                    "captured_at": (
                        image_metadata.get("captured_at") if image_metadata else None
                    ),
                    "distance_m": (
                        image_metadata.get("distance_m") if image_metadata else None
                    ),
                    "search_radius_m": (
                        image_metadata.get("radius") if image_metadata else None
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

            except Exception as e:
                print(
                    f"Error collecting images for waypoint {waypoint.sequence_id}: {e}"
                )
                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "images_downloaded": 0,
                    "image_path": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

        # Save collection metadata
        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "route_id": route.route_id,
                    "collection_timestamp": datetime.now().isoformat(),
                    "total_waypoints": len(route.waypoints),
                    "results": results,
                },
                f,
                indent=2,
            )
        return results

    def _fetch_mapillary_candidates(
        self,
        lat: float,
        lon: float,
        image_size: int,
        max_distance: int,
    ) -> List[Dict[str, Any]]:
        """Return candidate Mapillary images near a coordinate, expanding the search radius if needed."""

        allowed_sizes = {256, 1024, 2048}
        if image_size not in allowed_sizes:
            raise ValueError(
                f"Mapillary only supports image_size values {sorted(allowed_sizes)}, got {image_size}."
            )

        base_distance = max(max_distance, 10)
        field_name = f"thumb_{image_size}_url"

        search_radii: List[int] = []
        for candidate_radius in (
            base_distance,
            int(base_distance * 1.5),
            base_distance * 2,
            300,
            400,
            500,
        ):
            radius_value = max(10, int(candidate_radius))
            if radius_value not in search_radii:
                search_radii.append(radius_value)

        candidates: List[Dict[str, Any]] = []
        seen_ids = set()

        for radius_m in search_radii:
            delta_lat, delta_lon = self._meters_to_degree_offsets(lat, radius_m)
            bbox = [
                lon - delta_lon,
                lat - delta_lat,
                lon + delta_lon,
                lat + delta_lat,
            ]

            params = {
                "fields": f"id,{field_name},captured_at,computed_geometry",
                "access_token": self.api_key,
                "limit": 10,
                "bbox": ",".join(map(str, bbox)),
            }

            try:
                response = requests.get(
                    "https://graph.mapillary.com/images",
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
            except requests.RequestException as api_error:
                print(
                    f"Warning: Mapillary request failed for radius {radius_m}m at ({lat}, {lon}): {api_error}"
                )
                continue

            data = response.json().get("data", [])
            for image_data in data:
                image_id = image_data.get("id")
                image_url = image_data.get(field_name)
                if not image_url or not image_id:
                    continue
                if image_id in seen_ids:
                    continue

                coordinates = None
                try:
                    coordinates = image_data.get("computed_geometry", {}).get("coordinates")
                except AttributeError:
                    coordinates = None

                distance = None
                if coordinates and len(coordinates) >= 2:
                    distance = self._haversine_distance(
                        lat,
                        lon,
                        coordinates[1],
                        coordinates[0],
                    )

                seen_ids.add(image_id)
                candidates.append(
                    {
                        "image_id": image_id,
                        "image_url": image_url,
                        "captured_at": image_data.get("captured_at"),
                        "radius": radius_m,
                        "distance_m": distance,
                    }
                )

            if candidates:
                break

        if not candidates:
            fallback_radii = [base_distance, 200, 400, 800]
            for radius_m in fallback_radii:
                params = {
                    "fields": f"id,{field_name},captured_at,computed_geometry",
                    "access_token": self.api_key,
                    "limit": 5,
                    "closeto": f"{lon},{lat}",
                }

                try:
                    response = requests.get(
                        "https://graph.mapillary.com/images",
                        params=params,
                        timeout=15,
                    )
                    response.raise_for_status()
                except requests.RequestException as api_error:
                    print(
                        f"Warning: Mapillary closeto request failed near ({lat}, {lon}): {api_error}"
                    )
                    continue

                data = response.json().get("data", [])
                for image_data in data:
                    image_id = image_data.get("id")
                    image_url = image_data.get(field_name)
                    if not image_id or not image_url or image_id in seen_ids:
                        continue
                    coordinates = None
                    try:
                        coordinates = image_data.get("computed_geometry", {}).get("coordinates")
                    except AttributeError:
                        coordinates = None
                    distance = None
                    if coordinates and len(coordinates) >= 2:
                        distance = self._haversine_distance(
                            lat,
                            lon,
                            coordinates[1],
                            coordinates[0],
                        )
                    seen_ids.add(image_id)
                    candidates.append(
                        {
                            "image_id": image_id,
                            "image_url": image_url,
                            "captured_at": image_data.get("captured_at"),
                            "radius": radius_m,
                            "distance_m": distance,
                        }
                    )

                if candidates:
                    break

        candidates.sort(
            key=lambda c: (
                float("inf") if c.get("distance_m") is None else c["distance_m"],
                c.get("radius", float("inf")),
            )
        )

        return candidates

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute distance in meters between two lat/lon points."""

        radius_earth = 6_371_000  # meters
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius_earth * c

    @staticmethod
    def _meters_to_degree_offsets(lat: float, distance_meters: float) -> tuple:
        """Convert a radial distance in meters to latitude/longitude degree offsets."""

        if distance_meters <= 0:
            return 0.0005, 0.0005  # default small search box (~55m)

        meters_per_deg_lat = 111_320
        lat_offset = distance_meters / meters_per_deg_lat

        cos_lat = math.cos(math.radians(lat))
        cos_lat = max(cos_lat, 1e-6)
        meters_per_deg_lon = meters_per_deg_lat * cos_lat
        lon_offset = distance_meters / meters_per_deg_lon

        return lat_offset, lon_offset

    @staticmethod
    def _download_image(url: str, output_path: Path, timeout: int = 20) -> None:
        """Download image from URL to destination path."""

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(output_path, "wb") as image_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    image_file.write(chunk)

    def _download_gsv_image(
        self,
        lat: float,
        lon: float,
        heading: float,
        image_size: str,
        fov: int,
        radius: int,
        output_path: Path,
        timeout: int = 20,
    ) -> Dict[str, Any]:
        """Fetch metadata and download a single Google Street View image, retrying with larger radii."""

        metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        image_url = "https://maps.googleapis.com/maps/api/streetview"

        base_radius = max(radius, 10)
        search_radii = []
        for multiplier in (1, 1.5, 2):
            candidate_radius = max(10, int(base_radius * multiplier))
            if candidate_radius not in search_radii:
                search_radii.append(candidate_radius)

        last_metadata: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        for current_radius in search_radii:
            metadata_params = {
                "location": f"{lat},{lon}",
                "radius": current_radius,
                "key": settings.google_maps_api_key,
                "source": "outdoor",
            }

            try:
                metadata_response = requests.get(
                    metadata_url, params=metadata_params, timeout=timeout
                )
                metadata_response.raise_for_status()
                metadata = metadata_response.json()
                last_metadata = metadata
            except requests.RequestException as request_error:
                last_error = f"metadata_request_failed:{request_error}"
                continue

            if metadata.get("status") != "OK":
                last_error = metadata.get("status", "UNKNOWN_STATUS")
                continue

            pano_id = metadata.get("pano_id")

            image_params = {
                "size": image_size,
                "location": f"{lat},{lon}",
                "fov": fov,
                "heading": heading,
                "pitch": 0,
                "key": settings.google_maps_api_key,
                "source": "outdoor",
            }

            if pano_id:
                image_params["pano"] = pano_id

            try:
                image_response = requests.get(
                    image_url, params=image_params, stream=True, timeout=timeout
                )
                image_response.raise_for_status()
            except requests.RequestException as image_error:
                last_error = f"image_request_failed:{image_error}"
                continue

            content_type = image_response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                last_error = f"unexpected_content_type:{content_type}"
                continue

            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass

            with open(output_path, "wb") as image_file:
                for chunk in image_response.iter_content(chunk_size=8192):
                    if chunk:
                        image_file.write(chunk)

            metadata["search_radius_m"] = current_radius
            return {"success": True, "metadata": metadata}

        if output_path.exists() and output_path.stat().st_size == 0:
            try:
                output_path.unlink()
            except OSError:
                pass

        return {"success": False, "metadata": last_metadata, "error": last_error}

    def collect_kartaview_images(
        self,
        route: Route,
        output_dir: Optional[str] = None,
        buffer: int = 50,
        metadata_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Collect KartaView images for all waypoints in a route
        Args:
            route: Route object containing waypoints
            output_dir: Directory to save images (default: data/images/route_id)
            buffer: Search radius from waypoint in meters
            metadata_only: If True, only download metadata without images
        Returns:
            List of download results with metadata
        """
        # Set up output directory
        if not output_dir:
            output_dir = settings.images_dir / route.route_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize KartaView downloader
        downloader = self.KVDownloader(max_workers=self.max_workers)

        results = []

        # Collect images for each waypoint
        for waypoint in route.waypoints:
            try:
                print(
                    f"Collecting KartaView images for waypoint {waypoint.sequence_id} at ({waypoint.lat}, {waypoint.lon})"
                )

                # Create waypoint-specific directory
                waypoint_dir = output_dir / f"waypoint_{waypoint.sequence_id:03d}"
                waypoint_dir.mkdir(exist_ok=True)

                # Download images near this waypoint
                downloader.download_svi(
                    str(waypoint_dir),
                    lat=waypoint.lat,
                    lon=waypoint.lon,
                    buffer=buffer,
                    metadata_only=metadata_only,
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
                    "metadata_only": metadata_only,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

            except Exception as e:
                print(
                    f"Error collecting KartaView images for waypoint {waypoint.sequence_id}: {e}"
                )
                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "images_downloaded": 0,
                    "image_path": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

        # Save collection metadata
        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "route_id": route.route_id,
                    "collection_timestamp": datetime.now().isoformat(),
                    "total_waypoints": len(route.waypoints),
                    "metadata_only": metadata_only,
                    "results": results,
                },
                f,
                indent=2,
            )

        return results

    def collect_google_street_view_images(
        self,
        route: Route,
        output_dir: Optional[str] = None,
        image_size: str = "640x640",
        fov: int = 90,
        buffer: Optional[int] = None,
        clean_output: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Collect Google Street View images for all waypoints in a route

        Args:
            route: Route object containing waypoints
            output_dir: Directory to save images
            image_size: Image size string accepted by Google Street View Static API
            fov: Field of view in degrees
            buffer: Optional radius in meters when searching for nearby panoramas
            clean_output: If True, delete existing JPEGs in the output directory before downloading

        Returns:
            List of download results with metadata
        """
        if not settings.google_maps_api_key:
            raise ValueError("Google Maps API key is required for Street View images")

        if not output_dir:
            output_dir = settings.images_dir / route.route_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if clean_output:
            for existing_file in output_dir.glob("*.jpg"):
                try:
                    existing_file.unlink()
                except Exception as cleanup_error:
                    print(
                        f"Warning: failed to remove existing GSV image {existing_file}: {cleanup_error}"
                    )

        radius = buffer if buffer is not None else 50

        results = []

        for waypoint in route.waypoints:
            try:
                print(
                    f"Collecting GSV image for waypoint {waypoint.sequence_id} at ({waypoint.lat}, {waypoint.lon})"
                )

                image_filename = f"waypoint_{waypoint.sequence_id:03d}.jpg"
                image_path = output_dir / image_filename

                download_info = self._download_gsv_image(
                    lat=waypoint.lat,
                    lon=waypoint.lon,
                    heading=waypoint.heading or 0,
                    image_size=image_size,
                    fov=fov,
                    radius=radius,
                    output_path=image_path,
                )

                if download_info["success"]:
                    waypoint.image_path = str(image_path)

                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "image_path": str(image_path) if download_info["success"] else None,
                    "download_success": download_info["success"],
                    "gsv_pano_id": (
                        download_info.get("metadata", {}).get("pano_id")
                        if download_info.get("metadata")
                        else None
                    ),
                    "metadata": download_info.get("metadata"),
                    "error": download_info.get("error"),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

            except Exception as e:
                print(
                    f"Error downloading image for waypoint {waypoint.sequence_id}: {e}"
                )
                result = {
                    "waypoint_id": waypoint.sequence_id,
                    "lat": waypoint.lat,
                    "lon": waypoint.lon,
                    "image_path": None,
                    "download_success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "route_id": route.route_id,
                    "collection_timestamp": datetime.now().isoformat(),
                    "total_waypoints": len(route.waypoints),
                    "platform": "google_street_view",
                    "results": results,
                },
                f,
                indent=2,
            )

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
            "validation_timestamp": datetime.now().isoformat(),
        }
