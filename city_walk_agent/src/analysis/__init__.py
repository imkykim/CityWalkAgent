"""
Sequential analysis engine for CityWalkAgent

This module implements the core innovation of CityWalkAgent: analyzing walking
experiences as sequential patterns rather than isolated point measurements.

Key components:
- SequentialAnalyzer: Post-hoc batch analysis engine
- ContinuousAnalyzer: Real-time waypoint analysis with visual change detection
- Metrics: Volatility, barriers, transitions
- Aggregator: Traditional aggregate scoring (for comparison)
- Visualizer: Timeline graphs and heatmaps
- Comparator: Compare sequential vs aggregate methods
"""

from .sequential_analyzer import SequentialAnalyzer
from .continuous_analyzer import ContinuousAnalyzer, WaypointAnalysis
from .metrics import (
    calculate_volatility,
    detect_hidden_barriers,
    analyze_transitions,
    Barrier,
    TransitionAnalysis
)
from .aggregator import AggregateAnalyzer
from .comparator import MethodComparator

__all__ = [
    "SequentialAnalyzer",
    "ContinuousAnalyzer",
    "WaypointAnalysis",
    "calculate_volatility",
    "detect_hidden_barriers",
    "analyze_transitions",
    "Barrier",
    "TransitionAnalysis",
    "AggregateAnalyzer",
    "MethodComparator",
]
