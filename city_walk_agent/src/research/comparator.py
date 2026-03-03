"""
Method comparator - sequential vs aggregate

Provides direct comparison between sequential and aggregate methods
"""

from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from .sequential_analyzer import SequentialAnalyzer, RouteAnalysis
from .aggregator import AggregateAnalyzer


@dataclass
class MethodComparison:
    """Comparison results between sequential and aggregate methods"""
    route_id: str

    # Scores
    aggregate_score: float
    sequential_score: float
    score_difference: float

    # Analysis insights
    sequential_pattern: str
    aggregate_quality: str

    # Sequential-specific findings
    volatility: float
    num_barriers: int
    worst_barrier_drop: float

    # Comparison insights
    recommendation_change: str  # "yes" or "no"
    key_differences: List[str]
    which_method_better: str  # "sequential", "aggregate", or "similar"

    # Full analyses
    sequential_analysis: Dict[str, Any]
    aggregate_analysis: Dict[str, Any]


class MethodComparator:
    """
    Compare sequential and aggregate analysis methods

    This is the KEY component for demonstrating that sequential
    analysis reveals patterns invisible to aggregate methods.

    Usage:
        ```python
        comparator = MethodComparator(route, evaluations)
        comparison = comparator.compare()
        ```
    """

    def __init__(
        self,
        route: Any,
        evaluations: List[Dict[str, Any]],
        volatility_threshold: float = 2.0,
        barrier_threshold: float = 3.0
    ):
        """
        Initialize comparator

        Args:
            route: Route object
            evaluations: List of evaluation results
            volatility_threshold: Volatility threshold
            barrier_threshold: Barrier detection threshold
        """
        self.route = route
        self.evaluations = evaluations

        # Initialize both analyzers
        self.sequential_analyzer = SequentialAnalyzer(
            route,
            evaluations,
            volatility_threshold,
            barrier_threshold
        )
        self.aggregate_analyzer = AggregateAnalyzer(evaluations)

    def compare(self) -> MethodComparison:
        """
        Run full comparison between methods

        Returns:
            MethodComparison object with all results
        """
        # Run both analyses
        sequential_analysis = self.sequential_analyzer.full_analysis()
        aggregate_summary = self.aggregate_analyzer.get_summary()

        # Extract key metrics
        aggregate_score = aggregate_summary["overall_average"]
        sequential_score = sequential_analysis.sequential_score
        score_difference = aggregate_score - sequential_score

        # Determine if recommendation would change
        # (significant difference in ranking)
        recommendation_change = "yes" if abs(score_difference) >= 1.0 else "no"

        # Identify key differences
        key_differences = self._identify_key_differences(
            sequential_analysis,
            aggregate_summary
        )

        # Determine which method provides better insight
        which_better = self._determine_better_method(
            sequential_analysis,
            aggregate_summary
        )

        # Get worst barrier drop
        worst_barrier_drop = 0.0
        if sequential_analysis.hidden_barriers:
            worst_barrier_drop = max(
                b["score_drop"] for b in sequential_analysis.hidden_barriers
            )

        return MethodComparison(
            route_id=self.route.route_id,
            aggregate_score=aggregate_score,
            sequential_score=sequential_score,
            score_difference=score_difference,
            sequential_pattern=sequential_analysis.pattern_type,
            aggregate_quality=aggregate_summary["quality_classification"],
            volatility=sequential_analysis.volatility,
            num_barriers=len(sequential_analysis.hidden_barriers),
            worst_barrier_drop=worst_barrier_drop,
            recommendation_change=recommendation_change,
            key_differences=key_differences,
            which_method_better=which_better,
            sequential_analysis=asdict(sequential_analysis),
            aggregate_analysis=aggregate_summary
        )

    def _identify_key_differences(
        self,
        sequential_analysis: RouteAnalysis,
        aggregate_summary: Dict[str, Any]
    ) -> List[str]:
        """Identify key differences between methods"""
        differences = []

        # Barrier detection
        if sequential_analysis.hidden_barriers:
            num_barriers = len(sequential_analysis.hidden_barriers)
            differences.append(
                f"Sequential detected {num_barriers} hidden barrier(s) invisible to aggregate"
            )

            worst = max(
                sequential_analysis.hidden_barriers,
                key=lambda b: b["score_drop"]
            )
            differences.append(
                f"Worst barrier: {worst['score_drop']:.1f} point drop in {worst['dimension']}"
            )

        # Volatility
        if sequential_analysis.volatility > 2.0:
            differences.append(
                f"High volatility ({sequential_analysis.volatility:.2f}) "
                "indicates inconsistent experience (missed by aggregate)"
            )

        # Pattern recognition
        if sequential_analysis.pattern_type in ["barrier_present", "volatile"]:
            differences.append(
                f"Sequential identifies problematic pattern: {sequential_analysis.pattern_type}"
            )

        # Score adjustment
        score_diff = abs(
            sequential_analysis.aggregate_score - sequential_analysis.sequential_score
        )
        if score_diff >= 1.0:
            differences.append(
                f"Sequential adjusts score by {score_diff:.1f} points to account for patterns"
            )

        if not differences:
            differences.append(
                "Both methods agree - route has consistent quality"
            )

        return differences

    def _determine_better_method(
        self,
        sequential_analysis: RouteAnalysis,
        aggregate_summary: Dict[str, Any]
    ) -> str:
        """
        Determine which method provides better insight

        Returns:
            "sequential", "aggregate", or "similar"
        """
        # If barriers detected, sequential is better
        if sequential_analysis.hidden_barriers:
            return "sequential"

        # If high volatility, sequential is better
        if sequential_analysis.volatility > 2.0:
            return "sequential"

        # If pattern is problematic, sequential is better
        if sequential_analysis.pattern_type in ["barrier_present", "volatile", "declining"]:
            return "sequential"

        # If scores are very similar and pattern is stable, methods are similar
        score_diff = abs(
            sequential_analysis.aggregate_score - sequential_analysis.sequential_score
        )
        if score_diff < 0.5 and sequential_analysis.pattern_type in ["stable_high", "stable_low"]:
            return "similar"

        # Default: sequential provides more nuanced insight
        return "sequential"

    def generate_comparison_report(self) -> str:
        """
        Generate human-readable comparison report

        Returns:
            Report string
        """
        comparison = self.compare()

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║  Sequential vs Aggregate Analysis Comparison                     ║
╚══════════════════════════════════════════════════════════════════╝

Route: {comparison.route_id}

──────────────────────────────────────────────────────────────────
SCORES
──────────────────────────────────────────────────────────────────
Aggregate Score:   {comparison.aggregate_score:.2f}/10
Sequential Score:  {comparison.sequential_score:.2f}/10
Difference:        {comparison.score_difference:+.2f} points

──────────────────────────────────────────────────────────────────
PATTERN ANALYSIS
──────────────────────────────────────────────────────────────────
Sequential Pattern: {comparison.sequential_pattern}
Aggregate Quality:  {comparison.aggregate_quality}
Volatility:         {comparison.volatility:.2f}
Barriers Detected:  {comparison.num_barriers}

──────────────────────────────────────────────────────────────────
KEY DIFFERENCES
──────────────────────────────────────────────────────────────────
"""

        for i, diff in enumerate(comparison.key_differences, 1):
            report += f"{i}. {diff}\n"

        report += f"""
──────────────────────────────────────────────────────────────────
CONCLUSION
──────────────────────────────────────────────────────────────────
Better Method:          {comparison.which_method_better}
Recommendation Change:  {comparison.recommendation_change}

Sequential Analysis Insight:
{comparison.sequential_analysis['key_insight']}

Recommendation:
{comparison.sequential_analysis['recommendation']}

"""

        return report

    @staticmethod
    def compare_multiple_routes(
        routes: List[Any],
        evaluations_by_route: Dict[str, List[Dict[str, Any]]]
    ) -> List[MethodComparison]:
        """
        Compare methods across multiple routes

        Args:
            routes: List of Route objects
            evaluations_by_route: Dict mapping route_id to evaluations

        Returns:
            List of MethodComparison objects
        """
        comparisons = []

        for route in routes:
            if route.route_id in evaluations_by_route:
                evaluations = evaluations_by_route[route.route_id]
                comparator = MethodComparator(route, evaluations)
                comparison = comparator.compare()
                comparisons.append(comparison)

        return comparisons
