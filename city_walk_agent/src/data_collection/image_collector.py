import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from src.config import settings
from src.utils.data_models import Route, Waypoint
from src.utils.logging import get_logger


class ImageCollector:
    """Collect street view images using ZenSVI library."""

    def __init__(self, api_key: Optional[str] = None, max_workers: int = 4) -> None:
        """Initialize the collector with API credentials and worker limits."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.api_key = api_key or settings.mapillary_api_key
        self.max_workers = max_workers
        self.zensvi_path = settings.zensvi_root
        self.virl_path = settings.virl_root

        # Disable ZenSVI proxy usage by default to avoid slow, unreliable proxy hops
        os.environ.setdefault("ZENSVI_DISABLE_PROXIES", "1")

        zensvi_src = self.zensvi_path / "src"
        if str(zensvi_src) not in sys.path:
            sys.path.insert(0, str(zensvi_src))
            self.logger.debug("ZenSVI path appended", path=str(zensvi_src))

        virl_pkg_root = self.virl_path
        if virl_pkg_root.exists() and str(virl_pkg_root) not in sys.path:
            sys.path.insert(0, str(virl_pkg_root))
            self.logger.debug("VIRL path appended", path=str(virl_pkg_root))

        self._kv_downloader_cls = None
        self._virl_google_api = None

    def collect_route_images(
        self,
        route: Route,
        prefer_mapillary: bool = False,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Collect images for a route using available providers.

        Args:
            route: Route to collect images for.
            prefer_mapillary: Whether to try Mapillary before Google Street View. Defaults to False to try GSV first.
            output_dir: Optional override for the image output directory.

        Returns:
            List of image paths.
        """
        image_dir = Path(output_dir) if output_dir else settings.images_dir / route.route_id
        image_dir.mkdir(parents=True, exist_ok=True)

        provider_order = ["mapillary", "gsv"] if prefer_mapillary else ["gsv", "mapillary"]
        attempt_notes: List[str] = []

        for provider in provider_order:
            if provider == "mapillary":
                if not settings.mapillary_api_key:
                    attempt_notes.append("mapillary: missing API key")
                    self.logger.debug(
                        "Skipping Mapillary collection",
                        route_id=route.route_id,
                        reason="missing API key"
                    )
                    continue

                try:
                    results = self.collect_mapillary_images(route, output_dir=image_dir)
                except Exception as error:
                    attempt_notes.append(f"mapillary: {error}")
                    self.logger.warning(
                        "Mapillary collection failed",
                        route_id=route.route_id,
                        error=str(error)
                    )
                    continue

                image_paths = [
                    Path(result["image_path"])
                    for result in results
                    if result.get("image_path")
                ]

                if image_paths:
                    self.logger.info(
                        "Mapillary images collected",
                        route_id=route.route_id,
                        count=len(image_paths)
                    )
                    return sorted(image_paths)

                attempt_notes.append("mapillary: no images returned")
                continue

            if provider == "gsv":
                if not settings.google_maps_api_key:
                    attempt_notes.append("gsv: missing API key")
                    self.logger.debug(
                        "Skipping Google Street View collection",
                        route_id=route.route_id,
                        reason="missing API key"
                    )
                    continue

                try:
                    results = self.collect_google_street_view_images_static(
                        route,
                        output_dir=image_dir,
                        clean_output=True
                    )
                except Exception as error:
                    attempt_notes.append(f"gsv: {error}")
                    self.logger.warning(
                        "Google Street View collection failed",
                        route_id=route.route_id,
                        error=str(error)
                    )
                    continue

                image_paths = [
                    Path(result["image_path"])
                    for result in results
                    if result.get("image_path")
                ]

                if image_paths:
                    self.logger.info(
                        "Google Street View images collected",
                        route_id=route.route_id,
                        count=len(image_paths)
                    )
                    return sorted(image_paths)

                attempt_notes.append("gsv: no images returned")

        if attempt_notes:
            failed_summary = "; ".join(attempt_notes)
            raise RuntimeError(
                f"Image collection failed for route {route.route_id}: {failed_summary}"
            )

        raise RuntimeError(
            f"Image collection failed for route {route.route_id}: no providers available"
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
                self.logger.info(
                    "Collecting Mapillary images",
                    route_id=route.route_id,
                    waypoint_id=waypoint.sequence_id,
                    latitude=waypoint.lat,
                    longitude=waypoint.lon
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
                    self.logger.warning(
                        "No Mapillary candidates found",
                        route_id=route.route_id,
                        waypoint_id=waypoint.sequence_id,
                        search_radius_m=search_limit
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
                        self.logger.info(
                            "Mapillary image selected",
                            route_id=route.route_id,
                            waypoint_id=waypoint.sequence_id,
                            image_id=candidate.get("image_id"),
                            distance=distance_str,
                            search_radius=radius_str
                        )
                        break
                    except Exception as download_error:
                        self.logger.warning(
                            "Failed to download Mapillary image",
                            route_id=route.route_id,
                            waypoint_id=waypoint.sequence_id,
                            image_id=candidate.get("image_id"),
                            error=str(download_error)
                        )
                        if image_path.exists():
                            image_path.unlink(missing_ok=True)

                if not downloaded:
                    image_path = None
                    waypoint.image_path = None
                    self.logger.error(
                        "Mapillary download failed",
                        route_id=route.route_id,
                        waypoint_id=waypoint.sequence_id
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
                self.logger.error(
                    "Error collecting Mapillary images",
                    route_id=route.route_id,
                    waypoint_id=waypoint.sequence_id,
                    error=str(e)
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
                self.logger.warning(
                    "Mapillary API request failed",
                    latitude=lat,
                    longitude=lon,
                    radius_m=radius_m,
                    error=str(api_error)
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
                    self.logger.warning(
                        "Mapillary closeto request failed",
                        latitude=lat,
                        longitude=lon,
                        radius_m=radius_m,
                        error=str(api_error)
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
    def _meters_to_degree_offsets(lat: float, distance_meters: float) -> Tuple[float, float]:
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

    def _get_kv_downloader_cls(self):
        """Lazily import the KartaView downloader to avoid heavy dependencies when unused."""

        if self._kv_downloader_cls is not None:
            return self._kv_downloader_cls

        try:
            from importlib import import_module

            kv_module = import_module("zensvi.download.kv")
            self._kv_downloader_cls = getattr(kv_module, "KVDownloader")
            return self._kv_downloader_cls
        except ImportError as error:
            self.logger.error(
                "Failed to import ZenSVI KartaView downloader",
                zensvi_path=str(self.zensvi_path),
                error=str(error)
            )
            raise ImportError(
                f"Could not import ZenSVI KVDownloader from {self.zensvi_path}. Error: {error}"
            ) from error

    def _get_gsv_downloader_cls(self):
        """Lazily import the GSV downloader to avoid geopandas dependency unless required."""

        if self._gsv_downloader_cls is not None:
            return self._gsv_downloader_cls

        try:
            from importlib import import_module

            gsv_module = import_module("zensvi.download.gsv")
            self._gsv_downloader_cls = getattr(gsv_module, "GSVDownloader")
            return self._gsv_downloader_cls
        except ImportError as error:
            self.logger.error(
                "Failed to import ZenSVI GSV downloader",
                zensvi_path=str(self.zensvi_path),
                error=str(error)
            )
            raise ImportError(
                f"Could not import ZenSVI GSVDownloader from {self.zensvi_path}. Error: {error}"
            ) from error

    @staticmethod
    def _compose_panorama_image(images: List[Image.Image]) -> Image.Image:
        """Create a simple panorama by stitching images horizontally."""

        if not images:
            raise ValueError("No images provided for panorama composition")

        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        panorama = Image.new("RGB", (total_width, max_height))

        offset = 0
        for img in images:
            panorama.paste(img, (offset, 0))
            offset += img.width

        return panorama

    def _get_virl_google_api(self):
        """Instantiate VIRL's Google Maps helper for Static Street View downloads."""

        if self._virl_google_api is not None:
            return self._virl_google_api

        virl_root = self.virl_path
        if not virl_root.exists():
            raise FileNotFoundError(
                f"VIRL root directory not found at {virl_root}. Cannot use VIRL Street View utilities."
            )

        try:
            from virl.platform.google_map_apis import GoogleMapAPI
        except ImportError as error:
            self.logger.error(
                "Failed to import VIRL GoogleMapAPI",
                virl_path=str(virl_root),
                error=str(error)
            )
            raise ImportError(
                f"Could not import VIRL from {virl_root}. Error: {error}"
            ) from error

        offline_cfg = SimpleNamespace(
            ENABLED=False,
            PANORAMA_DIR="None",
            GPS_TO_PANO_PATH="None",
        )

        if settings.google_maps_api_key and not os.environ.get("GOOGLE_MAP_API_KEY"):
            os.environ["GOOGLE_MAP_API_KEY"] = settings.google_maps_api_key

        api = GoogleMapAPI(offline_cfg=offline_cfg)
        if not api.key:
            raise ValueError(
                "VIRL GoogleMapAPI did not receive a Google Maps API key. "
                "Ensure GOOGLE_MAPS_API_KEY is configured."
            )

        self._virl_google_api = api
        return api

    def _get_streetview_metadata(
        self,
        lat: float,
        lon: float,
        radius: int,
        source: str = "outdoor",
        timeout: int = 15,
    ) -> Optional[Dict[str, Any]]:
        """Fetch metadata for the closest Street View panorama."""

        metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "source": source,
            "key": settings.google_maps_api_key,
        }

        try:
            response = requests.get(metadata_url, params=params, timeout=timeout)
            response.raise_for_status()
            metadata = response.json()
        except requests.RequestException as error:
            self.logger.warning(
                "Street View metadata request failed",
                latitude=lat,
                longitude=lon,
                error=str(error)
            )
            return None

        if metadata.get("status") != "OK":
            self.logger.debug(
                "Street View metadata returned non-OK status",
                latitude=lat,
                longitude=lon,
                status=metadata.get("status"),
                error_message=metadata.get("error_message"),
            )
            return metadata

        return metadata

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

        # Initialize KartaView downloader lazily
        downloader_cls = self._get_kv_downloader_cls()
        downloader = downloader_cls(max_workers=self.max_workers)

        results = []

        # Collect images for each waypoint
        for waypoint in route.waypoints:
            try:
                self.logger.info(
                    "Collecting KartaView images",
                    route_id=route.route_id,
                    waypoint_id=waypoint.sequence_id,
                    latitude=waypoint.lat,
                    longitude=waypoint.lon
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
                self.logger.error(
                    "Error collecting KartaView images",
                    route_id=route.route_id,
                    waypoint_id=waypoint.sequence_id,
                    error=str(e)
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


    def collect_google_street_view_images_static(
        self,
        route: Route,
        output_dir: Optional[str] = None,
        size: Tuple[int, int] = (640, 640),
        fov: int = 90,
        pitch: int = 0,
        headings: Optional[List[int]] = None,
        all_around: bool = True,
        buffer: int = 50,
        clean_output: bool = True,
        source: str = "outdoor",
    ) -> List[Dict[str, Any]]:
        """
        Collect Street View images using VIRL's GoogleMapAPI helper with the static API.

        Args:
            route: Route containing waypoints.
            output_dir: Directory to save images (defaults to images_dir/<route_id>).
            size: Tuple(width, height) for Static API requests (max 640x640 per Google).
            fov: Field of view per capture (10-120 degrees).
            pitch: Camera pitch angle.
            headings: Optional list of heading angles to capture. If None and all_around=True, capture a 360 sweep.
            all_around: Whether to automatically cover 360 degrees using the provided fov.
            buffer: Metadata search radius in meters.
            clean_output: If True, clears existing JPEGs before download.
            source: Street View source parameter ("default" or "outdoor").

        Returns:
            List of metadata dictionaries describing the downloads.
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
                    self.logger.warning(
                        "Failed to remove existing Street View image",
                        file=str(existing_file),
                        error=str(cleanup_error)
                    )

        api = self._get_virl_google_api()
        results: List[Dict[str, Any]] = []

        fov = max(10, min(120, int(fov)))
        heading_step = max(1, fov)

        for waypoint in route.waypoints:
            heading_plan: List[int]
            if headings:
                heading_plan = [int(h) % 360 for h in headings]
            elif all_around:
                heading_plan = list(range(0, 360, heading_step))
            else:
                heading_value = waypoint.heading if waypoint.heading is not None else 0
                heading_plan = [int(heading_value) % 360]

            try:
                self.logger.info(
                    "Collecting Static Street View images",
                    route_id=route.route_id,
                    waypoint_id=waypoint.sequence_id,
                    latitude=waypoint.lat,
                    longitude=waypoint.lon,
                    headings=heading_plan,
                )

                heading_images: List[Image.Image] = []
                heading_entries: List[Dict[str, Any]] = []

                for heading in heading_plan:
                    street_view_image = api.get_streetview_from_geocode(
                        (waypoint.lat, waypoint.lon),
                        size=size,
                        heading=heading,
                        pitch=pitch,
                        fov=fov,
                        source=source,
                        idx=heading,
                    )
                    pil_image = street_view_image.image.copy()
                    heading_path = output_dir / f"waypoint_{waypoint.sequence_id:03d}_heading_{heading:03d}.jpg"
                    pil_image.save(heading_path)
                    heading_images.append(pil_image)
                    heading_entries.append(
                        {
                            "heading": heading,
                            "image_path": str(heading_path),
                        }
                    )

                panorama_path: Optional[Path] = None
                if all_around and len(heading_images) > 1:
                    panorama_path = output_dir / f"waypoint_{waypoint.sequence_id:03d}_panorama.jpg"
                    panorama_image = self._compose_panorama_image(heading_images)
                    panorama_image.save(panorama_path)

                primary_image_path = (
                    str(panorama_path)
                    if panorama_path
                    else (heading_entries[0]["image_path"] if heading_entries else None)
                )
                if primary_image_path:
                    waypoint.image_path = primary_image_path

                metadata_raw = self._get_streetview_metadata(waypoint.lat, waypoint.lon, buffer)
                metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
                metadata_payload = {
                    **metadata,
                    "headings": heading_plan,
                    "fov": fov,
                    "pitch": pitch,
                    "source": source,
                    "size": size,
                }

                results.append(
                    {
                        "waypoint_id": waypoint.sequence_id,
                        "lat": waypoint.lat,
                        "lon": waypoint.lon,
                        "image_path": primary_image_path,
                        "download_success": bool(heading_entries),
                        "gsv_pano_id": metadata.get("pano_id"),
                        "metadata": metadata_payload,
                        "heading_images": heading_entries,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as error:
                self.logger.error(
                    "Error collecting Static Street View images",
                    route_id=route.route_id,
                    waypoint_id=waypoint.sequence_id,
                    error=str(error)
                )
                results.append(
                    {
                        "waypoint_id": waypoint.sequence_id,
                        "lat": waypoint.lat,
                        "lon": waypoint.lon,
                        "image_path": None,
                        "download_success": False,
                        "error": str(error),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, "w") as metadata_stream:
            json.dump(
                {
                    "route_id": route.route_id,
                    "collection_timestamp": datetime.now().isoformat(),
                    "total_waypoints": len(route.waypoints),
                    "platform": "google_street_view_static",
                    "fov": fov,
                    "pitch": pitch,
                    "size": size,
                    "all_around": all_around,
                    "buffer_m": buffer,
                    "results": results,
                },
                metadata_stream,
                indent=2,
            )

        success_count = sum(1 for result in results if result.get("download_success"))
        self.logger.info(
            "Completed Static Street View collection",
            route_id=route.route_id,
            success=success_count,
            total=len(results),
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
