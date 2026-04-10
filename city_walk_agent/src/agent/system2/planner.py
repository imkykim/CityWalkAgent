"""Planner — Stateful navigation manager (Step 3 of System 2)

Responsibilities:
- Fetch and cache Directions API walking route
- Compute per-step bearing toward next route waypoint
- Detect route deviation and trigger re-route
- Provide navigation context string for branch_decision prompt
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

from geopy.distance import geodesic

from src.utils.logging import get_logger
from .models import System1Evidence, DecideResult, PlanResult


def _calc_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2 (degrees, 0=N clockwise)."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _bearing_to_cardinal(b: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(b / 45) % 8]


class Planner:
    """Stateful navigation manager — owns Directions API route and bearing computation."""

    def __init__(self, framework_id: str) -> None:
        self.framework_id = framework_id
        self.route_waypoints: List[Tuple[float, float]] = []
        self.wp_index: int = 0
        self.reroute_threshold_m: float = 80.0
        self.reroute_count: int = 0
        self.total_route_distance_m: float = 0.0
        self.urgency_override: Optional[str] = None  # None | "explore" | "navigate" | "converge"
        self.logger = get_logger(self.__class__.__name__)

    def reset(self) -> None:
        """Clear state from any previous walk."""
        self.route_waypoints = []
        self.wp_index = 0
        self.reroute_count = 0
        self.total_route_distance_m = 0.0
        self.urgency_override = None

    def set_urgency_override(self, mode: Optional[str]) -> None:
        """Force a specific urgency tier for the entire walk.

        Args:
            mode: One of "explore", "navigate", "converge", or None for auto.

        Raises:
            ValueError: If mode is not in the allowed set.
        """
        if mode is not None and mode not in ("explore", "navigate", "converge"):
            raise ValueError(
                f"urgency_override must be one of 'explore', 'navigate', 'converge', or None; got {mode!r}"
            )
        self.urgency_override = mode
        if mode:
            self.logger.info(f"Urgency override active: {mode}")

    async def init_route(
        self,
        start: Tuple[float, float],
        dest: Tuple[float, float],
        api_key: str,
    ) -> None:
        """Call Directions API at walk start. Store decoded polyline waypoints.
        On failure, log warning and leave route_waypoints empty (fallback to dest_bearing)."""
        try:
            self.route_waypoints = await self._fetch_walking_route(start, dest, api_key)
            self.logger.info(f"Directions API: {len(self.route_waypoints)} waypoints")
        except Exception as e:
            self.logger.warning(f"Directions API failed, falling back to dest_bearing: {e}")
            self.route_waypoints = []

    def get_navigation_context(
        self,
        current_lat: float,
        current_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> Dict[str, Any]:
        """Per-step navigation info.

        Returns:
            {
                "wp_bearing": float | None,
                "dest_bearing": float,
                "dest_cardinal": str,
                "dist_to_dest_m": float,
                "urgency_tier": str,
                "nearest_wp_dist_m": float | None,
                "dest_context": str,
                "has_route": bool,
            }
        Also internally advances wp_index as agent progresses.
        """
        dest_bearing = _calc_bearing(current_lat, current_lng, dest_lat, dest_lng)
        dest_cardinal = _bearing_to_cardinal(dest_bearing)
        dist_to_dest_m = geodesic((current_lat, current_lng), (dest_lat, dest_lng)).meters
        urgency_tier = self.urgency_override or self._get_urgency_tier(dist_to_dest_m)

        wp_bearing: Optional[float] = None
        nearest_wp_dist_m: Optional[float] = None
        dest_context = (
            f"Destination: {dest_cardinal} ({dest_bearing:.0f}°), {dist_to_dest_m:.0f}m away. "
            f"Urgency: {urgency_tier}"
        )

        if self.route_waypoints:
            wp_bearing, self.wp_index, nearest_wp_dist_m = self._get_wp_bearing(
                current_lat, current_lng,
            )
            wp_cardinal = _bearing_to_cardinal(wp_bearing)
            dest_context = (
                f"Reference walking route bearing: {wp_cardinal} ({wp_bearing:.0f}°). "
                f"Destination: {dest_cardinal} ({dest_bearing:.0f}°), {dist_to_dest_m:.0f}m away. "
                f"Urgency: {urgency_tier}"
            )

        return {
            "wp_bearing": wp_bearing,
            "dest_bearing": dest_bearing,
            "dest_cardinal": dest_cardinal,
            "dist_to_dest_m": dist_to_dest_m,
            "urgency_tier": urgency_tier,
            "nearest_wp_dist_m": nearest_wp_dist_m,
            "dest_context": dest_context,
            "has_route": bool(self.route_waypoints),
        }

    async def check_and_reroute(
        self,
        current_lat: float,
        current_lng: float,
        dest_lat: float,
        dest_lng: float,
        api_key: str,
    ) -> bool:
        """Check if nearest_wp_dist exceeds threshold; if so, re-fetch route.
        Returns True if re-routed. Increments self.reroute_count."""
        try:
            self.route_waypoints = await self._fetch_walking_route(
                (current_lat, current_lng), (dest_lat, dest_lng), api_key,
            )
            self.wp_index = 0
            self.reroute_count += 1
            self.logger.info(
                f"Re-routed: {len(self.route_waypoints)} new waypoints "
                f"(reroute #{self.reroute_count})"
            )
            return True
        except Exception as e:
            self.logger.warning(f"Re-route failed: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Navigation summary for route-level Reporter."""
        total = len(self.route_waypoints)
        return {
            "reroute_count": self.reroute_count,
            "total_waypoints_in_route": total,
            "wp_index_reached": self.wp_index,
            "route_coverage_pct": round(self.wp_index / total, 3) if total > 0 else 0.0,
            "has_route": bool(self.route_waypoints),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_wp_bearing(
        self,
        current_lat: float,
        current_lng: float,
    ) -> Tuple[float, int, float]:
        """Find nearest remaining waypoint, return bearing to the next one.

        Returns:
            (bearing, updated_wp_index, nearest_wp_dist_m)
        """
        current = (current_lat, current_lng)
        nearest_idx = self.wp_index
        min_dist = float("inf")
        for i in range(self.wp_index, len(self.route_waypoints)):
            d = geodesic(current, self.route_waypoints[i]).meters
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        target_idx = min(nearest_idx + 1, len(self.route_waypoints) - 1)
        bearing = _calc_bearing(
            current_lat, current_lng,
            self.route_waypoints[target_idx][0],
            self.route_waypoints[target_idx][1],
        )
        return bearing, target_idx, min_dist

    @staticmethod
    def _get_urgency_tier(dist_m: float) -> str:
        if dist_m <= 150:
            return "converge"
        elif dist_m <= 500:
            return "navigate"
        else:
            return "explore"

    @staticmethod
    async def _fetch_walking_route(
        start: Tuple[float, float],
        dest: Tuple[float, float],
        api_key: str,
    ) -> List[Tuple[float, float]]:
        """Google Directions API walking route → decoded polyline waypoints."""
        import httpx as _httpx
        import polyline as pl

        url = (
            f"https://maps.googleapis.com/maps/api/directions/json"
            f"?origin={start[0]},{start[1]}"
            f"&destination={dest[0]},{dest[1]}"
            f"&mode=walking"
            f"&key={api_key}"
        )
        async with _httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK" or not data.get("routes"):
            return []
        encoded = data["routes"][0]["overview_polyline"]["points"]
        return pl.decode(encoded)  # [(lat, lng), ...]

    # ------------------------------------------------------------------ #
    # Legacy stubs — not yet used; keep for compatibility                  #
    # ------------------------------------------------------------------ #

    def plan(
        self,
        evidence: System1Evidence,
        decide_result: DecideResult,
        candidate_routes: Optional[List[Any]] = None,
        personality: Optional[Any] = None,
    ) -> PlanResult:
        """Generate an optimal route plan.

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Planner.plan() — not yet implemented.")
