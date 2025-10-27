"""
Sequential analysis engine for CityWalkAgent

This module implements the core innovation of CityWalkAgent: analyzing walking
experiences as sequential patterns rather than isolated point measurements.

Key components:
- SequentialAnalyzer: Main sequential analysis engine
- Metrics: Volatility, barriers, transitions
- Aggregator: Traditional aggregate scoring (for comparison)
- Visualizer: Timeline graphs and heatmaps
- Comparator: Compare sequential vs aggregate methods
"""

from .sequential_analyzer import SequentialAnalyzer
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
    "calculate_volatility",
    "detect_hidden_barriers",
    "analyze_transitions",
    "Barrier",
    "TransitionAnalysis",
    "AggregateAnalyzer",
    "MethodComparator",
]
