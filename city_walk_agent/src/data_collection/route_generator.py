"""Route generation utilities for CityWalkAgent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import googlemaps
except ModuleNotFoundError as exc:
    googlemaps = None  # type: ignore
    GOOGLEMAPS_IMPORT_ERROR = exc
else:
    GOOGLEMAPS_IMPORT_ERROR = None
import numpy as np
from geopy.distance import geodesic

from src.config import DEFAULT_SAMPLING_INTERVAL, settings
from src.utils.data_models import Route, Waypoint
from src.utils.logging import get_logger


class RouteGenerator:
    """Generate routes and waypoints using Google Maps API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize generator with optional Google Maps API key."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.api_key = api_key or settings.google_maps_api_key
        self.gmaps: Optional[googlemaps.Client] = None

        if googlemaps is None:
            self.logger.warning(
                "googlemaps package not installed; Google Maps routing disabled. "
                "Install the 'googlemaps' dependency to enable full functionality."
            )
            return

        if self.api_key and self.api_key != "test_key":
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
                self.logger.info("Google Maps client initialized")
            except ValueError as error:
                self.logger.warning(
                    "Invalid Google Maps API key",
                    error=str(error)
                )

    def get_routes_dir(self) -> Path:
        """Return the directory where route definitions are stored."""
        routes_dir = settings.data_dir / "routes"
        routes_dir.mkdir(parents=True, exist_ok=True)
        return routes_dir

    def get_route_file(self, route_id: str) -> Path:
        """Return path to the JSON file for a route."""
        return self.get_routes_dir() / f"{route_id}.json"

    def get_route_dir(self, route_id: str, create: bool = True) -> Path:
        """
        Return the directory for assets associated with a route.

        Args:
            route_id: Route identifier.
            create: Whether to ensure the directory exists.
        """
        route_dir = settings.images_dir / route_id
        if create:
            route_dir.mkdir(parents=True, exist_ok=True)
        return route_dir

    def create_simple_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        interval_meters: int = DEFAULT_SAMPLING_INTERVAL,
        route_name: Optional[str] = None,
    ) -> Route:
        """
        Create a simple straight-line route between two points

        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude
            end_lon: Ending longitude
            interval_meters: Distance between waypoints in meters
            route_name: Optional name for the route

        Returns:
            Route object with waypoints
        """
        start_point = (start_lat, start_lon)
        end_point = (end_lat, end_lon)

        # Calculate total distance
        total_distance = geodesic(start_point, end_point).meters

        # Calculate number of waypoints
        num_points = max(2, int(total_distance / interval_meters) + 1)

        # Generate waypoints using linear interpolation
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)

        waypoints = []
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            waypoint = Waypoint(
                lat=lat, lon=lon, sequence_id=i, timestamp=datetime.now()
            )
            waypoints.append(waypoint)

        route_id = f"route_{int(datetime.now().timestamp())}"
        if route_name:
            route_id = f"{route_name}_{route_id}"

        route = Route(
            route_id=route_id,
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            waypoints=waypoints,
            route_name=route_name,
            interval_meters=interval_meters,
        )

        return route

    def create_google_maps_route(
        self,
        origin: str,
        destination: str,
        interval_meters: int = DEFAULT_SAMPLING_INTERVAL,
        mode: str = "walking",
        route_name: Optional[str] = None,
    ) -> Route:
        """
        Create a route using Google Maps Directions API

        Args:
            origin: Starting address or coordinates
            destination: Ending address or coordinates
            interval_meters: Distance between waypoints in meters
            mode: Travel mode (walking, driving, etc.)
            route_name: Optional name for the route

        Returns:
            Route object with waypoints along the actual path
        """
        if not self.gmaps:
            self.logger.error("Google Maps client not available")
            raise ValueError(
                "Google Maps client is unavailable. Ensure the 'googlemaps' package "
                "is installed and a valid API key is configured."
            )

        # Get directions from Google Maps
        directions = self.gmaps.directions(
            origin=origin, destination=destination, mode=mode
        )

        if not directions:
            self.logger.warning(
                "No Google Maps route found",
                origin=origin,
                destination=destination
            )
            raise ValueError(f"No route found from {origin} to {destination}")

        # Extract the main route
        route_data = directions[0]
        legs = route_data["legs"]

        # Decode the polyline to get detailed path coordinates
        polyline_points = []
        for leg in legs:
            for step in leg["steps"]:
                # Decode the polyline for this step
                decoded = googlemaps.convert.decode_polyline(step["polyline"]["points"])
                polyline_points.extend(decoded)

        # Sample waypoints along the path at specified intervals
        waypoints = self._sample_waypoints_along_path(polyline_points, interval_meters)

        # Get start and end coordinates
        start_location = legs[0]["start_location"]
        end_location = legs[-1]["end_location"]

        route_id = f"gmaps_route_{int(datetime.now().timestamp())}"
        if route_name:
            route_id = f"{route_name}_{route_id}"

        route = Route(
            route_id=route_id,
            start_lat=start_location["lat"],
            start_lon=start_location["lng"],
            end_lat=end_location["lat"],
            end_lon=end_location["lng"],
            waypoints=waypoints,
            route_name=route_name,
            description=f"Google Maps route from {origin} to {destination}",
            interval_meters=interval_meters,
        )

        return route

    def _sample_waypoints_along_path(
        self,
        path_points: Sequence[Dict[str, float]],
        interval_meters: int
    ) -> List[Waypoint]:
        """
        Sample waypoints along a path at regular intervals

        Args:
            path_points: List of lat/lng dictionaries from Google Maps
            interval_meters: Distance between waypoints in meters

        Returns:
            List of Waypoint objects
        """
        if not path_points:
            return []

        waypoints = []
        current_distance = 0
        sequence_id = 0

        # Always include the first point
        first_point = path_points[0]
        waypoints.append(
            Waypoint(
                lat=first_point["lat"],
                lon=first_point["lng"],
                sequence_id=sequence_id,
                timestamp=datetime.now(),
            )
        )
        sequence_id += 1

        # Sample points along the path
        for i in range(1, len(path_points)):
            prev_point = path_points[i - 1]
            curr_point = path_points[i]

            # Calculate distance between consecutive points
            segment_distance = geodesic(
                (prev_point["lat"], prev_point["lng"]),
                (curr_point["lat"], curr_point["lng"]),
            ).meters

            current_distance += segment_distance

            # Add waypoint if we've traveled the required interval
            if current_distance >= interval_meters:
                waypoints.append(
                    Waypoint(
                        lat=curr_point["lat"],
                        lon=curr_point["lng"],
                        sequence_id=sequence_id,
                        timestamp=datetime.now(),
                    )
                )
                sequence_id += 1
                current_distance = 0  # Reset distance counter

        # Always include the last point if it wasn't already added
        last_point = path_points[-1]
        if (
            waypoints[-1].lat != last_point["lat"]
            or waypoints[-1].lon != last_point["lng"]
        ):
            waypoints.append(
                Waypoint(
                    lat=last_point["lat"],
                    lon=last_point["lng"],
                    sequence_id=sequence_id,
                    timestamp=datetime.now(),
                )
            )

        return waypoints

    def save_route(self, route: Route, filepath: Optional[Path] = None) -> Path:
        """Persist a route definition to disk."""
        target_path = Path(filepath) if filepath else self.get_route_file(route.route_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(route.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info("Route saved", route_id=route.route_id, path=str(target_path))
        return target_path

    @staticmethod
    def load_route(identifier: Path | str) -> Route:
        """
        Load route from JSON file

        Args:
            identifier: Route identifier or path to JSON file

        Returns:
            Route object
        """
        path = Path(identifier)
        candidate_paths = []

        if path.is_file():
            candidate_paths.append(path)
        else:
            if path.suffix:
                candidate_paths.append(path)
                if not path.is_absolute():
                    candidate_paths.append(settings.data_dir / "routes" / path.name)
            else:
                candidate_paths.append(path.with_suffix(".json"))
                candidate_paths.append(settings.data_dir / "routes" / f"{path.name}.json")

        resolved_path: Optional[Path] = None

        for candidate in candidate_paths:
            if candidate.exists():
                resolved_path = candidate
                break
        else:
            raise FileNotFoundError(f"Route file not found for identifier: {identifier}")

        route_data = resolved_path.read_text(encoding="utf-8")
        return Route.model_validate_json(route_data)
