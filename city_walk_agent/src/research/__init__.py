"""Research tools for method comparison and paper validation.

Not part of the main pipeline — used in experiments/ scripts.
"""

from src.research.sequential_analyzer import SequentialAnalyzer
from src.research.aggregator import AggregateAnalyzer
from src.research.comparator import MethodComparator
from src.research.metrics import (
    calculate_volatility,
    detect_hidden_barriers,
    analyze_transitions,
    Barrier,
    TransitionAnalysis,
)

__all__ = [
    "SequentialAnalyzer", "AggregateAnalyzer", "MethodComparator",
    "calculate_volatility", "detect_hidden_barriers", "analyze_transitions",
    "Barrier", "TransitionAnalysis",
]
