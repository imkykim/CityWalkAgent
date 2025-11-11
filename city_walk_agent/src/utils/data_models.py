import math
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from geopy.distance import geodesic as _geodesic_distance
except ImportError:
    _geodesic_distance = None

from pydantic import BaseModel, Field

from src.config import settings


class WalkabilityScore(BaseModel):
    """Individual walkability dimension score"""
    dimension: str = Field(description="Walkability dimension (safety, comfort, interest, aesthetics)")
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Confidence in score")
    reasoning: Optional[str] = Field(default=None, description="Explanation for the score")


class Waypoint(BaseModel):
    """Single waypoint with GPS coordinates and metadata"""
    lat: float = Field(description="Latitude")
    lon: float = Field(description="Longitude")
    sequence_id: int = Field(description="Order in the route")
    heading: Optional[float] = Field(default=None, description="Camera heading in degrees")
    timestamp: Optional[datetime] = Field(default=None, description="When waypoint was created")
    image_path: Optional[str] = Field(default=None, description="Path to street view image")


class EvaluationResult(BaseModel):
    """Evaluation result for a single waypoint"""
    waypoint: Waypoint
    scores: List[WalkabilityScore]
    overall_score: Optional[float] = Field(default=None, description="Overall walkability score")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(description="Model used for evaluation")
    prompt_version: str = Field(description="Version of prompt used")


class Route(BaseModel):
    """Complete route with metadata"""
    route_id: str = Field(description="Unique route identifier")
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    waypoints: List[Waypoint]
    route_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    interval_meters: int = Field(default=10, description="Sampling interval in meters")

    @property
    def total_distance(self) -> float:
        """Total route distance in meters."""
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for prev, curr in zip(self.waypoints, self.waypoints[1:]):
            total += _compute_distance_meters(
                (prev.lat, prev.lon),
                (curr.lat, curr.lon)
            )
        return total


class SequentialAnalysis(BaseModel):
    """Results of sequential analysis"""
    route_id: str
    analysis_type: str = Field(description="Type of sequential analysis performed")

    # Basic metrics
    mean_score: float
    median_score: float
    std_deviation: float
    min_score: float
    max_score: float

    # Sequential-specific metrics
    volatility: float = Field(description="Measure of score variation along route")
    hidden_barriers: List[Dict[str, Any]] = Field(description="Detected sudden score drops")
    transition_analysis: Dict[str, float] = Field(description="Analysis of score transitions")

    # Comparison with aggregate
    aggregate_score: float = Field(description="Traditional average score")
    sequential_score: float = Field(description="Sequential-weighted score")
    recommendation_difference: Optional[str] = Field(default=None, description="How recommendations differ")

    analysis_timestamp: datetime = Field(default_factory=datetime.now)


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run"""
    experiment_id: str
    prompt_version: str
    dimensions: List[str] = Field(default_factory=lambda: settings.default_dimensions)
    model_name: str = Field(default="claude-3-sonnet")
    sampling_interval: int = Field(default=10, description="Meters between waypoints")
    volatility_threshold: float = Field(default=2.0, description="Threshold for detecting volatility")
    barrier_threshold: float = Field(default=3.0, description="Score drop threshold for hidden barriers")
    created_at: datetime = Field(default_factory=datetime.now)


def _compute_distance_meters(point_a, point_b) -> float:
    """Compute distance between two (lat, lon) points in meters."""
    if _geodesic_distance:
        return _geodesic_distance(point_a, point_b).meters
    return _haversine_distance(point_a, point_b)


def _haversine_distance(point_a, point_b) -> float:
    """Haversine fallback when geopy is unavailable."""
    lat1, lon1 = point_a
    lat2, lon2 = point_b
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
