"""
Sequential analyzer - main analysis engine

This implements the core innovation of analyzing walking experiences as
continuous sequences rather than isolated points.
"""

import statistics
from typing import List, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

from .metrics import (
    calculate_volatility,
    detect_hidden_barriers,
    analyze_transitions,
    calculate_sequential_score,
    classify_route_pattern,
    Barrier,
    TransitionAnalysis
)


@dataclass
class RouteAnalysis:
    """Complete sequential analysis for one route"""
    route_id: str
    aggregate_score: float
    sequential_score: float
    volatility: float
    pattern_type: str
    hidden_barriers: List[Dict[str, Any]]  # JSON-serializable
    transition_analysis: Dict[str, Any]  # JSON-serializable
    dimension_volatilities: Dict[str, float]
    key_insight: str
    recommendation: str


class SequentialAnalyzer:
    """
    Core analyzer implementing sequential walking experience analysis

    This is the heart of CityWalkAgent's innovation - analyzing walking
    experiences as continuous sequences rather than isolated points.

    Usage:
        ```python
        analyzer = SequentialAnalyzer(route, evaluations)
        analysis = analyzer.full_analysis()
        ```
    """

    def __init__(
        self,
        route: Any,  # Route object
        evaluations: List[Dict[str, Any]],
        volatility_threshold: float = 2.0,
        barrier_threshold: float = 3.0
    ):
        """
        Initialize analyzer with route and evaluation results

        Args:
            route: Route object with waypoints
            evaluations: List of evaluation dicts from VLM
            volatility_threshold: Threshold for volatile classification
            barrier_threshold: Minimum score drop for barrier detection
        """
        self.route = route
        self.evaluations = evaluations
        self.volatility_threshold = volatility_threshold
        self.barrier_threshold = barrier_threshold

        # Organize evaluations by waypoint and dimension
        self.scores_by_waypoint = defaultdict(dict)
        self.scores_by_dimension = defaultdict(list)

        self._organize_scores()

    def _organize_scores(self):
        """Organize evaluation results by waypoint and dimension"""
        for eval_result in self.evaluations:
            # Extract waypoint ID from image_id (e.g., "waypoint_5" -> 5)
            wp_id = eval_result.get("image_id", "")
            seq_match = wp_id.split("_")

            if len(seq_match) >= 2:
                try:
                    seq_num = int(seq_match[-1])
                    dimension = eval_result["dimension_id"]
                    score = eval_result["score"]

                    self.scores_by_waypoint[seq_num][dimension] = score
                    self.scores_by_dimension[dimension].append({
                        "sequence": seq_num,
                        "score": score
                    })
                except (ValueError, KeyError):
                    continue

        # Sort dimension scores by sequence
        for dim in self.scores_by_dimension:
            self.scores_by_dimension[dim].sort(key=lambda x: x["sequence"])

    def calculate_overall_volatility(self) -> float:
        """
        Calculate overall volatility across all waypoints

        Returns:
            Volatility score
        """
        if not self.scores_by_waypoint:
            return 0.0

        # Calculate average score per waypoint
        waypoint_averages = []
        for seq_num in sorted(self.scores_by_waypoint.keys()):
            scores = list(self.scores_by_waypoint[seq_num].values())
            if scores:
                waypoint_averages.append(sum(scores) / len(scores))

        if len(waypoint_averages) < 2:
            return 0.0

        return calculate_volatility(waypoint_averages, method="diff_std")

    def calculate_dimension_volatility(self, dimension: str) -> float:
        """
        Calculate volatility for specific dimension

        Args:
            dimension: Dimension ID

        Returns:
            Volatility score for dimension
        """
        scores = self.scores_by_dimension.get(dimension, [])
        if len(scores) < 2:
            return 0.0

        score_values = [s["score"] for s in scores]
        return calculate_volatility(score_values, method="diff_std")

    def detect_all_barriers(self) -> List[Barrier]:
        """
        Detect hidden barriers across all dimensions

        Returns:
            List of all detected barriers
        """
        all_barriers = []

        for dimension, score_data in self.scores_by_dimension.items():
            if len(score_data) < 2:
                continue

            scores = [s["score"] for s in score_data]
            sequences = [s["sequence"] for s in score_data]

            barriers = detect_hidden_barriers(
                scores=scores,
                sequences=sequences,
                waypoints=self.route.waypoints,
                dimension=dimension,
                route_id=self.route.route_id,
                threshold=self.barrier_threshold
            )

            all_barriers.extend(barriers)

        # Sort by severity
        all_barriers.sort(key=lambda b: b.score_drop, reverse=True)
        return all_barriers

    def analyze_all_transitions(self) -> TransitionAnalysis:
        """
        Analyze transitions across route

        Returns:
            TransitionAnalysis object
        """
        # Get overall scores per waypoint
        waypoint_scores = []
        for seq_num in sorted(self.scores_by_waypoint.keys()):
            scores = list(self.scores_by_waypoint[seq_num].values())
            if scores:
                waypoint_scores.append(sum(scores) / len(scores))

        if len(waypoint_scores) < 2:
            return TransitionAnalysis(
                positive_transitions=0,
                negative_transitions=0,
                smooth_transitions=0,
                moderate_transitions=0,
                abrupt_transitions=0,
                severe_transitions=0,
                average_transition_magnitude=0.0,
                max_drop=0.0,
                max_improvement=0.0,
                transition_volatility=0.0
            )

        return analyze_transitions(waypoint_scores)

    def calculate_aggregate_score(self) -> float:
        """
        Calculate traditional aggregate score (simple average)

        Returns:
            Aggregate score
        """
        all_scores = []
        for wp_scores in self.scores_by_waypoint.values():
            all_scores.extend(wp_scores.values())

        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def calculate_route_sequential_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate sequential score accounting for patterns

        Args:
            weights: Custom penalty weights

        Returns:
            Sequential score
        """
        base_score = self.calculate_aggregate_score()
        volatility = self.calculate_overall_volatility()
        barriers = self.detect_all_barriers()
        transitions = self.analyze_all_transitions()

        barrier_severity = sum(b.score_drop for b in barriers)

        return calculate_sequential_score(
            base_score=base_score,
            volatility=volatility,
            num_barriers=len(barriers),
            barrier_severity=barrier_severity,
            num_severe_transitions=transitions.severe_transitions,
            weights=weights
        )

    def classify_pattern(self) -> str:
        """
        Classify route pattern type

        Returns:
            Pattern type string
        """
        avg_score = self.calculate_aggregate_score()
        volatility = self.calculate_overall_volatility()
        barriers = self.detect_all_barriers()

        # Calculate trend
        waypoint_averages = []
        for seq_num in sorted(self.scores_by_waypoint.keys()):
            scores = list(self.scores_by_waypoint[seq_num].values())
            if scores:
                waypoint_averages.append(sum(scores) / len(scores))

        trend = 0.0
        if len(waypoint_averages) >= 2:
            mid = len(waypoint_averages) // 2
            first_half_avg = sum(waypoint_averages[:mid]) / mid
            second_half_avg = sum(waypoint_averages[mid:]) / (len(waypoint_averages) - mid)
            trend = second_half_avg - first_half_avg

        return classify_route_pattern(
            avg_score=avg_score,
            volatility=volatility,
            num_barriers=len(barriers),
            score_trend=trend,
            volatility_threshold=self.volatility_threshold
        )

    def compare_with_aggregate(self) -> Dict[str, Any]:
        """
        Compare sequential vs aggregate scoring

        Returns:
            Comparison dictionary with insights
        """
        aggregate_score = self.calculate_aggregate_score()
        sequential_score = self.calculate_route_sequential_score()
        difference = aggregate_score - sequential_score

        barriers = self.detect_all_barriers()
        volatility = self.calculate_overall_volatility()
        pattern = self.classify_pattern()

        # Generate insights
        key_insights = []

        if barriers:
            key_insights.append(
                f"{len(barriers)} hidden barrier(s) detected that aggregate misses"
            )
            worst_barrier = max(barriers, key=lambda b: b.score_drop)
            key_insights.append(
                f"Worst barrier: {worst_barrier.score_drop:.1f} point drop in {worst_barrier.dimension}"
            )

        if volatility > self.volatility_threshold:
            key_insights.append(
                f"High volatility ({volatility:.2f}) indicates inconsistent experience"
            )

        if pattern == "barrier_present":
            key_insights.append(
                "Pattern: Route has dangerous segments despite good average"
            )
        elif pattern == "volatile":
            key_insights.append(
                "Pattern: Unpredictable experience uncomfortable for pedestrians"
            )
        elif pattern == "stable_high":
            key_insights.append(
                "Pattern: Consistently pleasant - aggregate method accurate here"
            )

        if abs(difference) >= 1.0:
            key_insights.append(
                f"Sequential analysis adjusts score by {abs(difference):.1f} points"
            )

        return {
            "aggregate_score": aggregate_score,
            "sequential_score": sequential_score,
            "difference": difference,
            "pattern": pattern,
            "volatility": volatility,
            "barrier_count": len(barriers),
            "key_insights": key_insights
        }

    def full_analysis(self) -> RouteAnalysis:
        """
        Run complete sequential analysis

        Returns:
            RouteAnalysis object with all results
        """
        comparison = self.compare_with_aggregate()
        barriers = self.detect_all_barriers()
        transitions = self.analyze_all_transitions()

        # Calculate dimension-specific volatilities
        dimension_volatilities = {}
        for dimension in self.scores_by_dimension.keys():
            dimension_volatilities[dimension] = self.calculate_dimension_volatility(dimension)

        # Generate key insight
        if barriers:
            worst_barrier = max(barriers, key=lambda b: b.score_drop)
            key_insight = (
                f"{len(barriers)} barrier(s) found. Worst: {worst_barrier.score_drop:.1f} "
                f"point drop at waypoint {worst_barrier.waypoint_sequence}"
            )
        elif comparison["volatility"] > self.volatility_threshold:
            key_insight = (
                f"High volatility ({comparison['volatility']:.2f}) - inconsistent experience"
            )
        else:
            key_insight = (
                f"Pattern: {comparison['pattern']} - aggregate method fairly accurate"
            )

        # Generate recommendation
        if barriers:
            recommendation = (
                f"Priority: Fix barrier at waypoint {barriers[0].waypoint_sequence}"
            )
        elif comparison["pattern"] == "volatile":
            recommendation = (
                "Investigate cause of high volatility - multiple interventions needed"
            )
        elif comparison["pattern"] == "stable_low":
            recommendation = "Comprehensive improvement needed - consistently poor"
        else:
            recommendation = "Route acceptable - monitor for degradation"

        return RouteAnalysis(
            route_id=self.route.route_id,
            aggregate_score=comparison["aggregate_score"],
            sequential_score=comparison["sequential_score"],
            volatility=comparison["volatility"],
            pattern_type=comparison["pattern"],
            hidden_barriers=[asdict(b) for b in barriers],  # Convert to dict
            transition_analysis=asdict(transitions),  # Convert to dict
            dimension_volatilities=dimension_volatilities,
            key_insight=key_insight,
            recommendation=recommendation
        )
