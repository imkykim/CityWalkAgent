import googlemaps
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from geopy.distance import geodesic
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


class RouteGenerator:
    """Generate routes and waypoints using Google Maps API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Google Maps API key"""
        self.api_key = api_key or settings.google_maps_api_key
        self.gmaps = None

        # Only initialize Google Maps client if we have a valid key
        if self.api_key and self.api_key != "test_key":
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
            except ValueError:
                print(f"Warning: Invalid Google Maps API key provided. Google Maps functionality will be disabled.")

    def create_simple_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        interval_meters: int = 10,
        route_name: Optional[str] = None
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
                lat=lat,
                lon=lon,
                sequence_id=i,
                timestamp=datetime.now()
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
            interval_meters=interval_meters
        )

        return route

    def create_google_maps_route(
        self,
        origin: str,
        destination: str,
        interval_meters: int = 10,
        mode: str = "walking",
        route_name: Optional[str] = None
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
            raise ValueError("Google Maps API key is required for this functionality")

        # Get directions from Google Maps
        directions = self.gmaps.directions(
            origin=origin,
            destination=destination,
            mode=mode
        )

        if not directions:
            raise ValueError(f"No route found from {origin} to {destination}")

        # Extract the main route
        route_data = directions[0]
        legs = route_data['legs']

        # Decode the polyline to get detailed path coordinates
        polyline_points = []
        for leg in legs:
            for step in leg['steps']:
                # Decode the polyline for this step
                decoded = googlemaps.convert.decode_polyline(step['polyline']['points'])
                polyline_points.extend(decoded)

        # Sample waypoints along the path at specified intervals
        waypoints = self._sample_waypoints_along_path(polyline_points, interval_meters)

        # Get start and end coordinates
        start_location = legs[0]['start_location']
        end_location = legs[-1]['end_location']

        route_id = f"gmaps_route_{int(datetime.now().timestamp())}"
        if route_name:
            route_id = f"{route_name}_{route_id}"

        route = Route(
            route_id=route_id,
            start_lat=start_location['lat'],
            start_lon=start_location['lng'],
            end_lat=end_location['lat'],
            end_lon=end_location['lng'],
            waypoints=waypoints,
            route_name=route_name,
            description=f"Google Maps route from {origin} to {destination}",
            interval_meters=interval_meters
        )

        return route

    def _sample_waypoints_along_path(
        self,
        path_points: List[dict],
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
        waypoints.append(Waypoint(
            lat=first_point['lat'],
            lon=first_point['lng'],
            sequence_id=sequence_id,
            timestamp=datetime.now()
        ))
        sequence_id += 1

        # Sample points along the path
        for i in range(1, len(path_points)):
            prev_point = path_points[i - 1]
            curr_point = path_points[i]

            # Calculate distance between consecutive points
            segment_distance = geodesic(
                (prev_point['lat'], prev_point['lng']),
                (curr_point['lat'], curr_point['lng'])
            ).meters

            current_distance += segment_distance

            # Add waypoint if we've traveled the required interval
            if current_distance >= interval_meters:
                waypoints.append(Waypoint(
                    lat=curr_point['lat'],
                    lon=curr_point['lng'],
                    sequence_id=sequence_id,
                    timestamp=datetime.now()
                ))
                sequence_id += 1
                current_distance = 0  # Reset distance counter

        # Always include the last point if it wasn't already added
        last_point = path_points[-1]
        if (waypoints[-1].lat != last_point['lat'] or
            waypoints[-1].lon != last_point['lng']):
            waypoints.append(Waypoint(
                lat=last_point['lat'],
                lon=last_point['lng'],
                sequence_id=sequence_id,
                timestamp=datetime.now()
            ))

        return waypoints

    def save_route(self, route: Route, filepath: Optional[str] = None) -> str:
        """
        Save route to JSON file

        Args:
            route: Route object to save
            filepath: Optional custom filepath

        Returns:
            Path where route was saved
        """
        if not filepath:
            filepath = settings.data_dir / "routes" / f"{route.route_id}.json"

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save route as JSON
        with open(filepath, 'w') as f:
            f.write(route.model_dump_json(indent=2))

        return str(filepath)

    @staticmethod
    def load_route(filepath: str) -> Route:
        """
        Load route from JSON file

        Args:
            filepath: Path to route JSON file

        Returns:
            Route object
        """
        with open(filepath, 'r') as f:
            route_data = f.read()

        return Route.model_validate_json(route_data)