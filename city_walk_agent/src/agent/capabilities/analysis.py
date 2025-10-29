"""Analysis capability - wraps sequential analyzer for cognitive pipeline.

This capability provides sequential analysis of route observations,
detecting patterns, volatility, and hidden barriers.
"""

from typing import Any, Dict, Optional

from src.analysis.sequential_analyzer import SequentialAnalyzer
from src.utils.logging import get_logger


class AnalysisCapability:
    """Capability for sequential analysis of route observations.

    This wraps the SequentialAnalyzer to provide pattern analysis
    as part of the agent's cognitive pipeline:
    Observe → Analyze → Think → Act → Remember

    Example:
        ```python
        analyzer = AnalysisCapability()
        analysis = analyzer.analyze_patterns(observation, route_info)
        ```
    """

    def __init__(self):
        """Initialize analysis capability."""
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("AnalysisCapability initialized")

    def analyze_patterns(
        self,
        observation: Dict[str, Any],
        route_info: Dict[str, Any],
        volatility_threshold: float = 2.0,
        barrier_threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """Analyze patterns in route observations using sequential analysis.

        Args:
            observation: Observation dict from ObservationCapability with:
                - raw_evaluations: List of evaluation dicts
                - dimension_stats: Statistics per dimension
            route_info: Route information with:
                - route: Route object with waypoints
                - route_id: Route identifier
            volatility_threshold: Threshold for volatile classification (default: 2.0)
            barrier_threshold: Minimum score drop for barrier detection (default: 3.0)

        Returns:
            Analysis dict with:
                - volatility: Overall volatility score
                - barriers: List of detected barriers
                - pattern_type: Route pattern classification
                - transition_analysis: Transition statistics
                - dimension_volatilities: Per-dimension volatility
                - sequential_score: Adjusted score accounting for patterns
                - aggregate_score: Traditional average score
        """
        route = route_info.get("route")
        route_id = route_info.get("route_id", "unknown")
        raw_evaluations = observation.get("raw_evaluations", [])

        if not route or not raw_evaluations:
            self.logger.warning(
                "Missing route or evaluations for analysis",
                route_id=route_id,
                has_route=route is not None,
                num_evaluations=len(raw_evaluations),
            )
            return self._empty_analysis()

        self.logger.debug(
            "Analyzing route patterns",
            route_id=route_id,
            num_evaluations=len(raw_evaluations),
        )

        # Use SequentialAnalyzer for full analysis
        analyzer = SequentialAnalyzer(
            route=route,
            evaluations=raw_evaluations,
            volatility_threshold=volatility_threshold,
            barrier_threshold=barrier_threshold,
        )

        # Run full analysis
        full_analysis = analyzer.full_analysis()

        # Extract key metrics for decision making
        analysis = {
            "volatility": full_analysis.volatility,
            "barriers": full_analysis.hidden_barriers,
            "pattern_type": full_analysis.pattern_type,
            "transition_analysis": full_analysis.transition_analysis,
            "dimension_volatilities": full_analysis.dimension_volatilities,
            "sequential_score": full_analysis.sequential_score,
            "aggregate_score": full_analysis.aggregate_score,
            "key_insight": full_analysis.key_insight,
            "recommendation": full_analysis.recommendation,
        }

        self.logger.debug(
            "Pattern analysis complete",
            route_id=route_id,
            pattern_type=analysis["pattern_type"],
            volatility=round(analysis["volatility"], 2),
            num_barriers=len(analysis["barriers"]),
        )

        return analysis

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis when data is missing.

        Returns:
            Empty analysis dict with safe default values.
        """
        return {
            "volatility": 0.0,
            "barriers": [],
            "pattern_type": "unknown",
            "transition_analysis": {},
            "dimension_volatilities": {},
            "sequential_score": 0.0,
            "aggregate_score": 0.0,
            "key_insight": "Insufficient data for analysis",
            "recommendation": "Unable to analyze route",
        }
