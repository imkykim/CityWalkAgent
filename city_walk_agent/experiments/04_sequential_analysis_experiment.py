#!/usr/bin/env python3
"""
Experiment 04: Sequential Walking Experience Analysis

Core Hypothesis: "Sequential analysis reveals patterns invisible to aggregate scoring methods."

This experiment proves that analyzing walking experiences as sequential patterns
(rather than point-based averages) provides superior insights for urban planning.

Example:
    Route A: [7→7→7→7→7] - Aggregate: 7.0, Sequential: 7.0 (consistently pleasant)
    Route B: [9→8→3→2→8] - Aggregate: 6.0, Sequential: 4.5 (hidden barriers detected)

Traditional methods rate B as "worse" but miss the dangerous mid-route segment.
Sequential analysis detects the 6-point drop and flags it for intervention.
"""

import sys
import json
import math
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import settings
from utils.data_models import Route, Waypoint, EvaluationResult, SequentialAnalysis

# ============================================================================
# CONFIGURATION
# ============================================================================

VOLATILITY_THRESHOLD = 2.0  # Score variation threshold for "volatile" classification
BARRIER_THRESHOLD = 3.0  # Minimum score drop to count as "hidden barrier"
BARRIER_DURATION = 2  # Minimum waypoints a barrier must persist

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Barrier:
    """Detected hidden barrier"""

    route_id: str
    waypoint_sequence: int
    score_drop: float
    before_score: float
    after_score: float
    dimension: str
    lat: float
    lon: float
    duration: int  # How many waypoints barrier persists
    recovery_time: Optional[int] = None  # Waypoints until score recovers
    severity_level: str = "moderate"  # low, moderate, high, severe


@dataclass
class TransitionAnalysis:
    """Analysis of score transitions along route"""

    positive_transitions: int
    negative_transitions: int
    smooth_transitions: int  # ±0.5
    moderate_transitions: int  # ±0.5-2.0
    abrupt_transitions: int  # ±2.0-3.0
    severe_transitions: int  # ±3.0+
    average_transition_magnitude: float
    max_drop: float
    max_improvement: float
    transition_volatility: float


@dataclass
class RouteAnalysis:
    """Complete sequential analysis for one route"""

    route_id: str
    aggregate_score: float
    sequential_score: float
    volatility: float
    pattern_type: str
    hidden_barriers: List[Barrier]
    transition_analysis: TransitionAnalysis
    dimension_volatilities: Dict[str, float]
    key_insight: str
    recommendation: str


# ============================================================================
# SEQUENTIAL ANALYZER
# ============================================================================


class SequentialAnalyzer:
    """
    Core analyzer implementing sequential walking experience analysis.

    This is the heart of CityWalkAgent's innovation - analyzing walking
    experiences as continuous sequences rather than isolated points.
    """

    debug_scores_logged = False

    def __init__(self, route: Route, evaluations: List[Dict]):
        """
        Initialize analyzer with route and evaluation results.

        Args:
            route: Route object with waypoints
            evaluations: List of evaluation dicts from VLM (experiment 03 output)
        """
        self.route = route
        self.evaluations = evaluations

        # Organize evaluations by waypoint and dimension
        self.scores_by_waypoint = defaultdict(dict)
        self.scores_by_dimension = defaultdict(list)

        for eval_result in evaluations:
            wp_id = eval_result.get("image_id", "")
            # Extract sequence number from image_id (e.g., "waypoint_5" -> 5)
            seq_match = wp_id.split("_")
            if len(seq_match) >= 2:
                try:
                    seq_num = int(seq_match[-1])
                    dimension = eval_result["dimension_id"]
                    score = eval_result["score"]

                    self.scores_by_waypoint[seq_num][dimension] = score
                    self.scores_by_dimension[dimension].append(
                        {"sequence": seq_num, "score": score}
                    )
                except (ValueError, KeyError):
                    continue

        # Sort dimension scores by sequence
        for dim in self.scores_by_dimension:
            self.scores_by_dimension[dim].sort(key=lambda x: x["sequence"])

        if not SequentialAnalyzer.debug_scores_logged:
            SequentialAnalyzer.debug_scores_logged = True
            print("\n[DEBUG] scores_by_waypoint (sequence -> dimension scores):")
            for seq_num in sorted(self.scores_by_waypoint.keys()):
                print(f"  {seq_num}: {self.scores_by_waypoint[seq_num]}")

            print("\n[DEBUG] scores_by_dimension (dimension -> ordered scores):")
            for dim, score_list in self.scores_by_dimension.items():
                ordered = [(item["sequence"], item["score"]) for item in score_list]
                print(f"  {dim}: {ordered}")

    def calculate_volatility(self, dimension: Optional[str] = None) -> float:
        """
        Measure score variation along route.

        High volatility = inconsistent, uncomfortable walking experience
        Low volatility = predictable, consistent experience

        Formula: Standard deviation of score differences between consecutive waypoints
        volatility = std([score[i+1] - score[i] for i in range(n-1)])

        Args:
            dimension: Specific dimension to analyze, or None for overall

        Returns:
            Volatility score (higher = more volatile)
        """
        if dimension:
            scores = self.scores_by_dimension.get(dimension, [])
            if len(scores) < 2:
                return 0.0
            score_values = [s["score"] for s in scores]
        else:
            # Overall volatility: average across all waypoints
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
            score_values = waypoint_averages

        # Calculate differences between consecutive waypoints
        differences = []
        for i in range(len(score_values) - 1):
            diff = abs(score_values[i + 1] - score_values[i])
            differences.append(diff)

        if not differences:
            return 0.0

        # Volatility is std of differences
        return statistics.stdev(differences) if len(differences) > 1 else differences[0]

    def detect_hidden_barriers(
        self, threshold: float = BARRIER_THRESHOLD
    ) -> List[Barrier]:
        """
        Find sudden score drops that aggregate methods miss.

        A "hidden barrier" is a segment where walkability suddenly deteriorates,
        which would be masked by averaging. Example: dangerous crossing,
        sudden loss of sidewalk, threatening area.

        Args:
            threshold: Minimum score drop to count as barrier

        Returns:
            List of detected barriers with locations and severity
        """
        barriers = []

        # Analyze each dimension separately
        for dimension, score_data in self.scores_by_dimension.items():
            if len(score_data) < 2:
                continue

            scores = [s["score"] for s in score_data]
            sequences = [s["sequence"] for s in score_data]

            i = 0
            while i < len(scores) - 1:
                score_drop = scores[i] - scores[i + 1]

                if score_drop >= threshold:
                    # Found a barrier - determine duration
                    duration = 1
                    j = i + 1

                    # Check how long low score persists
                    while j < len(scores) - 1 and scores[j] < scores[i] - threshold:
                        duration += 1
                        j += 1

                    # Calculate recovery time
                    recovery_time = None
                    for k in range(j, len(scores)):
                        if (
                            scores[k] >= scores[i] - 1.0
                        ):  # Score recovers to near original
                            recovery_time = k - i
                            break

                    # Get waypoint location
                    seq_num = sequences[i + 1]
                    waypoint = None
                    for wp in self.route.waypoints:
                        if wp.sequence_id == seq_num:
                            waypoint = wp
                            break

                    # Determine severity
                    if score_drop >= 5.0:
                        severity = "severe"
                    elif score_drop >= 4.0:
                        severity = "high"
                    elif score_drop >= 3.0:
                        severity = "moderate"
                    else:
                        severity = "low"

                    barrier = Barrier(
                        route_id=self.route.route_id,
                        waypoint_sequence=seq_num,
                        score_drop=score_drop,
                        before_score=scores[i],
                        after_score=scores[i + 1],
                        dimension=dimension,
                        lat=waypoint.lat if waypoint else 0.0,
                        lon=waypoint.lon if waypoint else 0.0,
                        duration=duration,
                        recovery_time=recovery_time,
                        severity_level=severity,
                    )
                    barriers.append(barrier)

                    # Skip past this barrier
                    i = j
                else:
                    i += 1

        # Sort by severity (score drop)
        barriers.sort(key=lambda b: b.score_drop, reverse=True)
        return barriers

    def analyze_transitions(self) -> TransitionAnalysis:
        """
        Analyze how experience changes along route.

        Transitions reveal the dynamics of walking experience:
        - Smooth transitions: comfortable, expected changes
        - Abrupt transitions: jarring, uncomfortable shifts

        Returns:
            Comprehensive transition analysis
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
                transition_volatility=0.0,
            )

        # Analyze all transitions
        positive = 0
        negative = 0
        smooth = 0
        moderate = 0
        abrupt = 0
        severe = 0

        transitions = []
        max_drop = 0.0
        max_improvement = 0.0

        for i in range(len(waypoint_scores) - 1):
            change = waypoint_scores[i + 1] - waypoint_scores[i]
            abs_change = abs(change)
            transitions.append(abs_change)

            # Direction
            if change > 0:
                positive += 1
                max_improvement = max(max_improvement, change)
            elif change < 0:
                negative += 1
                max_drop = max(max_drop, abs(change))

            # Magnitude classification
            if abs_change <= 0.5:
                smooth += 1
            elif abs_change <= 2.0:
                moderate += 1
            elif abs_change <= 3.0:
                abrupt += 1
            else:
                severe += 1

        avg_magnitude = sum(transitions) / len(transitions) if transitions else 0.0
        volatility = statistics.stdev(transitions) if len(transitions) > 1 else 0.0

        return TransitionAnalysis(
            positive_transitions=positive,
            negative_transitions=negative,
            smooth_transitions=smooth,
            moderate_transitions=moderate,
            abrupt_transitions=abrupt,
            severe_transitions=severe,
            average_transition_magnitude=avg_magnitude,
            max_drop=max_drop,
            max_improvement=max_improvement,
            transition_volatility=volatility,
        )

    def calculate_sequential_score(
        self, weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate route score that accounts for sequential patterns.

        Unlike aggregate average, this penalizes:
        - High volatility (uncomfortable, unpredictable)
        - Hidden barriers (dangerous segments)
        - Abrupt transitions (jarring experiences)

        Args:
            weights: Custom weights for penalties (optional)

        Returns:
            Sequential score (adjusted for experience quality)
        """
        # Default weights
        if weights is None:
            weights = {
                "base_score": 1.0,
                "volatility_penalty": 0.3,
                "barrier_penalty": 0.5,
                "transition_penalty": 0.2,
            }

        # Calculate base score (aggregate average)
        all_scores = []
        for wp_scores in self.scores_by_waypoint.values():
            all_scores.extend(wp_scores.values())

        if not all_scores:
            return 0.0

        base_score = sum(all_scores) / len(all_scores)

        # Calculate penalties
        volatility = self.calculate_volatility()
        barriers = self.detect_hidden_barriers()
        transitions = self.analyze_transitions()

        # Volatility penalty (normalized to 0-1 scale)
        volatility_penalty = min(volatility / 5.0, 1.0) * weights["volatility_penalty"]

        # Barrier penalty (each barrier reduces score)
        barrier_count = len(barriers)
        barrier_severity = (
            sum(b.score_drop for b in barriers) / 10.0 if barriers else 0.0
        )
        barrier_penalty = min(barrier_severity, 2.0) * weights["barrier_penalty"]

        # Transition penalty (severe transitions are uncomfortable)
        transition_penalty = (
            transitions.severe_transitions * 0.1 + transitions.abrupt_transitions * 0.05
        ) * weights["transition_penalty"]

        # Calculate final score
        sequential_score = (
            base_score - volatility_penalty - barrier_penalty - transition_penalty
        )

        # Ensure score stays in valid range
        return max(1.0, min(10.0, sequential_score))

    def classify_route_pattern(self) -> str:
        """
        Classify route into pattern types for easier interpretation.

        Pattern types:
        - stable_high: Consistently good (low volatility, high scores)
        - stable_low: Consistently poor (low volatility, low scores)
        - improving: Scores trend upward over route
        - declining: Scores trend downward over route
        - volatile: High variation, unpredictable experience
        - barrier_present: Contains significant drops

        Returns:
            Pattern type string
        """
        # Get overall metrics
        volatility = self.calculate_volatility()
        barriers = self.detect_hidden_barriers()

        # Calculate average score
        all_scores = []
        for wp_scores in self.scores_by_waypoint.values():
            all_scores.extend(wp_scores.values())
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Calculate trend (improving vs declining)
        waypoint_averages = []
        for seq_num in sorted(self.scores_by_waypoint.keys()):
            scores = list(self.scores_by_waypoint[seq_num].values())
            if scores:
                waypoint_averages.append(sum(scores) / len(scores))

        trend = 0.0
        if len(waypoint_averages) >= 2:
            # Simple linear trend: compare first half vs second half
            mid = len(waypoint_averages) // 2
            first_half_avg = sum(waypoint_averages[:mid]) / mid
            second_half_avg = sum(waypoint_averages[mid:]) / (
                len(waypoint_averages) - mid
            )
            trend = second_half_avg - first_half_avg

        # Classification logic
        if barriers and len(barriers) >= 2:
            return "barrier_present"
        elif volatility > VOLATILITY_THRESHOLD * 1.5:
            return "volatile"
        elif trend > 1.0:
            return "improving"
        elif trend < -1.0:
            return "declining"
        elif volatility < VOLATILITY_THRESHOLD * 0.5:
            if avg_score >= 7.0:
                return "stable_high"
            else:
                return "stable_low"
        else:
            # Default to stable with score qualifier
            if avg_score >= 7.0:
                return "stable_high"
            else:
                return "stable_low"

    def compare_with_aggregate(self) -> Dict[str, Any]:
        """
        Direct comparison with traditional aggregate method.

        This shows the KEY INSIGHT: sequential analysis reveals what
        aggregate scoring misses.

        Returns:
            Comparison dict with insights
        """
        # Aggregate method: simple average
        all_scores = []
        for wp_scores in self.scores_by_waypoint.values():
            all_scores.extend(wp_scores.values())
        aggregate_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Sequential method: weighted score
        sequential_score = self.calculate_sequential_score()

        # Difference
        difference = aggregate_score - sequential_score

        # Determine if ranking would change
        recommendation_change = "yes" if abs(difference) >= 1.0 else "no"

        # Generate insights
        barriers = self.detect_hidden_barriers()
        volatility = self.calculate_volatility()
        pattern = self.classify_route_pattern()

        key_insights = []

        if barriers:
            key_insights.append(
                f"{len(barriers)} hidden barrier(s) detected that aggregate misses"
            )
            worst_barrier = max(barriers, key=lambda b: b.score_drop)
            key_insights.append(
                f"Worst barrier: {worst_barrier.score_drop:.1f} point drop in {worst_barrier.dimension}"
            )

        if volatility > VOLATILITY_THRESHOLD:
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
                f"Sequential analysis reduces score by {difference:.1f} points - significant!"
            )

        return {
            "aggregate_score": aggregate_score,
            "sequential_score": sequential_score,
            "difference": difference,
            "recommendation_change": recommendation_change,
            "key_insights": key_insights,
            "pattern": pattern,
            "volatility": volatility,
            "barrier_count": len(barriers),
        }

    def full_analysis(self) -> RouteAnalysis:
        """
        Run complete sequential analysis.

        Returns:
            Complete RouteAnalysis object
        """
        comparison = self.compare_with_aggregate()
        barriers = self.detect_hidden_barriers()
        transitions = self.analyze_transitions()

        # Calculate dimension-specific volatilities
        dimension_volatilities = {}
        for dimension in self.scores_by_dimension.keys():
            dimension_volatilities[dimension] = self.calculate_volatility(dimension)

        # Generate key insight
        if barriers:
            worst_barrier = max(barriers, key=lambda b: b.score_drop)
            key_insight = (
                f"{len(barriers)} barrier(s) found. Worst: {worst_barrier.score_drop:.1f} "
                f"point drop at waypoint {worst_barrier.waypoint_sequence}"
            )
        elif comparison["volatility"] > VOLATILITY_THRESHOLD:
            key_insight = f"High volatility ({comparison['volatility']:.2f}) - inconsistent experience"
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
            hidden_barriers=barriers,
            transition_analysis=transitions,
            dimension_volatilities=dimension_volatilities,
            key_insight=key_insight,
            recommendation=recommendation,
        )


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================


def load_evaluation_results(results_dir: Path) -> Tuple[List[Dict], Optional[str]]:
    """
    Load VLM evaluation results from experiment 03

    Returns:
        Tuple of (evaluation results, route_id that was evaluated)
    """
    print("\n📂 Loading evaluation results...")

    # Find most recent evaluation results
    result_dirs = sorted(
        results_dir.glob("vlm_*"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not result_dirs:
        print("❌ No evaluation results found")
        print(f"   Expected in: {results_dir}")
        print("   Run experiment 03 first")
        return [], None

    # Try to load full_results.json from most recent run
    for result_dir in result_dirs[:5]:  # Check last 5 runs
        full_results_file = result_dir / "full_results.json"
        if full_results_file.exists():
            try:
                with open(full_results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)

                # Try to determine which route was evaluated by looking at image paths
                # Check experiment 03's load_test_images logic - it uses most recent route
                route_id = infer_route_id_from_evaluations(results)

                print(f"   ✓ Loaded {len(results)} evaluations from {result_dir.name}")
                if route_id:
                    print(f"   ✓ Detected route: {route_id}")
                else:
                    print(
                        f"   ⚠️  Could not detect route - will try to match with all routes"
                    )

                return results, route_id
            except Exception as e:
                print(f"   ✗ Failed to load {full_results_file.name}: {e}")
                continue

    print("❌ No valid evaluation results found")
    return [], None


def infer_route_id_from_evaluations(evaluations: List[Dict]) -> Optional[str]:
    """
    Try to infer which route was evaluated by checking image directories.

    Since experiment 03 uses images from a route, we can check which route
    has images matching the evaluation image_ids.
    """
    if not evaluations:
        return None

    # Get sample image_id from evaluations
    sample_image_ids = set()
    for eval_result in evaluations[:10]:
        image_id = eval_result.get("image_id", "")
        if image_id:
            sample_image_ids.add(image_id)

    if not sample_image_ids:
        return None

    # Check GSV image directories for matching images
    images_dir = settings.images_dir / "gsv"
    if not images_dir.exists():
        return None

    # Check each route directory
    for route_dir in images_dir.iterdir():
        if not route_dir.is_dir():
            continue

        # Check if this route has images matching our sample
        route_images = set()
        for img_file in route_dir.glob("waypoint_*.jpg"):
            # Extract waypoint number from filename
            img_id = img_file.stem  # e.g., "waypoint_000"
            route_images.add(img_id)

        # If we find a match, this is our route
        if sample_image_ids.intersection(route_images):
            return route_dir.name

    return None


def load_routes() -> List[Route]:
    """Load routes from data directory"""
    print("\n📂 Loading routes...")

    routes_dir = settings.data_dir / "routes"
    if not routes_dir.exists():
        print(f"❌ Routes directory not found: {routes_dir}")
        return []

    routes = []
    route_files = sorted(routes_dir.glob("*.json"))

    for route_file in route_files:
        try:
            with open(route_file, "r") as f:
                route_data = json.load(f)
                try:
                    route = Route.model_validate(route_data)
                except AttributeError:
                    route = Route.parse_obj(route_data)
                routes.append(route)
        except Exception as e:
            print(f"   ✗ Failed to load {route_file.name}: {e}")
            continue

    print(f"   ✓ Loaded {len(routes)} routes")
    return routes


def experiment_aggregate_vs_sequential(route: Route, evaluations: List[Dict]) -> Dict:
    """
    Core experiment proving sequential analysis superiority.

    Compares aggregate vs sequential scoring to show:
    1. How often they disagree
    2. What insights sequential reveals
    3. Impact on route ranking

    Args:
        route: Single route to analyze
        evaluations: VLM evaluations for this route

    Returns:
        Comparison dictionary
    """
    print("\n" + "=" * 70)
    print("📈 Experiment: Aggregate vs Sequential Comparison")
    print("=" * 70)

    print(f"\n  Analyzing route: {route.route_id}")
    print(f"  Waypoints: {len(route.waypoints)}")
    print(f"  Evaluations: {len(evaluations)}")

    # Run sequential analysis
    analyzer = SequentialAnalyzer(route, evaluations)
    comparison = analyzer.compare_with_aggregate()

    print(f"\n  📊 Results:")
    print(f"    Aggregate:  {comparison['aggregate_score']:.2f}")
    print(f"    Sequential: {comparison['sequential_score']:.2f}")
    print(f"    Difference: {comparison['difference']:.2f}")
    print(f"    Pattern:    {comparison['pattern']}")
    print(f"    Volatility: {comparison['volatility']:.2f}")
    print(f"    Barriers:   {comparison['barrier_count']}")

    if comparison["key_insights"]:
        print(f"\n  💡 Key Insights:")
        for insight in comparison["key_insights"]:
            print(f"    • {insight}")

    result = {"route_id": route.route_id, **comparison}

    return result


def experiment_hidden_barrier_detection(
    route: Route, evaluations: List[Dict]
) -> List[Barrier]:
    """
    Detect and analyze hidden barriers in route.

    Provides actionable intelligence: specific GPS coordinates
    of problem segments that need intervention.
    """
    print("\n" + "=" * 70)
    print("🔍 Experiment: Hidden Barrier Detection")
    print("=" * 70)

    analyzer = SequentialAnalyzer(route, evaluations)
    barriers = analyzer.detect_hidden_barriers()

    if barriers:
        print(f"\n  🚨 {len(barriers)} barrier(s) detected in route")
        for i, barrier in enumerate(barriers, 1):
            print(f"\n  Barrier {i}:")
            print(f"    Waypoint: {barrier.waypoint_sequence}")
            print(f"    Score Drop: {barrier.score_drop:.1f} points")
            print(f"    Severity: {barrier.severity_level}")
            print(f"    Dimension: {barrier.dimension}")
            print(
                f"    Before: {barrier.before_score:.1f} → After: {barrier.after_score:.1f}"
            )
            print(f"    Location: ({barrier.lat:.6f}, {barrier.lon:.6f})")
            if barrier.recovery_time:
                print(f"    Recovery: {barrier.recovery_time} waypoints")
    else:
        print("\n  ✅ No significant barriers detected")

    all_barriers = barriers

    return all_barriers


def experiment_volatility_analysis(
    route: Route, evaluations: List[Dict]
) -> Dict[str, Any]:
    """
    Analyze volatility patterns in route.

    Questions answered:
    1. What's typical volatility for this route?
    2. Which dimensions most volatile?
    3. How does it affect experience?
    """
    print("\n" + "=" * 70)
    print("📊 Experiment: Volatility Analysis")
    print("=" * 70)

    analyzer = SequentialAnalyzer(route, evaluations)
    overall_volatility = analyzer.calculate_volatility()

    print(f"\n  Overall Volatility: {overall_volatility:.2f}")

    # Volatility classification
    if overall_volatility < VOLATILITY_THRESHOLD * 0.5:
        vol_class = "Low (Consistent experience)"
    elif overall_volatility < VOLATILITY_THRESHOLD:
        vol_class = "Medium (Some variation)"
    else:
        vol_class = "High (Unpredictable experience)"

    print(f"  Classification: {vol_class}")

    # Dimension-specific volatilities
    dimension_volatilities = {}
    print(f"\n  Dimension-Specific Volatility:")
    for dimension in analyzer.scores_by_dimension.keys():
        dim_volatility = analyzer.calculate_volatility(dimension)
        dimension_volatilities[dimension] = dim_volatility
        print(f"    {dimension}: {dim_volatility:.2f}")

    # Transition analysis
    transitions = analyzer.analyze_transitions()
    print(f"\n  Transition Analysis:")
    print(f"    Smooth transitions: {transitions.smooth_transitions}")
    print(f"    Moderate transitions: {transitions.moderate_transitions}")
    print(f"    Abrupt transitions: {transitions.abrupt_transitions}")
    print(f"    Severe transitions: {transitions.severe_transitions}")

    return {
        "overall_volatility": overall_volatility,
        "classification": vol_class,
        "dimension_volatilities": dimension_volatilities,
        "transition_analysis": transitions,
    }


def experiment_route_classification(route: Route, evaluations: List[Dict]) -> str:
    """
    Classify route into pattern type.

    Shows what type of quality pattern this route exhibits.
    """
    print("\n" + "=" * 70)
    print("🏷️  Experiment: Route Pattern Classification")
    print("=" * 70)

    analyzer = SequentialAnalyzer(route, evaluations)
    pattern = analyzer.classify_route_pattern()

    pattern_descriptions = {
        "stable_high": "Consistently good - maintain quality",
        "stable_low": "Consistently poor - comprehensive improvement needed",
        "improving": "Trending better - good destination points",
        "declining": "Trending worse - investigate degradation",
        "volatile": "Unpredictable - multiple interventions needed",
        "barrier_present": "Hidden dangers - targeted fixes possible",
    }

    print(f"\n  Pattern: {pattern}")
    print(f"  Description: {pattern_descriptions.get(pattern, 'Unknown pattern')}")

    return pattern


# ============================================================================
# REPORTING
# ============================================================================


def generate_sequential_analysis_report(
    aggregate_comparison: Dict,
    barriers: List[Barrier],
    volatility_data: Dict[str, Any],
    pattern: str,
    route_id: str,
    timestamp: str,
) -> Path:
    """
    Generate comprehensive analysis report.

    This is the deliverable proving sequential analysis superiority.
    """
    print("\n" + "=" * 70)
    print("📝 Generating Sequential Analysis Report")
    print("=" * 70)

    # Create results directory
    report_dir = RESULTS_DIR / f"sequential_analysis_{route_id}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate vs Sequential CSV (single route)
    csv_file = report_dir / "aggregate_vs_sequential.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(
            "route_id,aggregate_score,sequential_score,difference,pattern,barrier_count,volatility\n"
        )
        f.write(
            f"{aggregate_comparison['route_id']},{aggregate_comparison['aggregate_score']:.2f},"
            f"{aggregate_comparison['sequential_score']:.2f},{aggregate_comparison['difference']:.2f},"
            f"{aggregate_comparison['pattern']},{aggregate_comparison['barrier_count']},"
            f"{aggregate_comparison['volatility']:.2f}\n"
        )

    print(f"  ✓ Saved: {csv_file.name}")

    # 2. Hidden Barriers CSV
    barriers_file = report_dir / "hidden_barriers.csv"
    with open(barriers_file, "w", encoding="utf-8") as f:
        f.write(
            "route_id,waypoint_seq,lat,lon,score_drop,dimension,severity,duration,recovery_time\n"
        )
        for b in barriers:
            f.write(
                f"{b.route_id},{b.waypoint_sequence},{b.lat:.6f},{b.lon:.6f},"
                f"{b.score_drop:.2f},{b.dimension},{b.severity_level},{b.duration},"
                f"{b.recovery_time if b.recovery_time else 'N/A'}\n"
            )

    print(f"  ✓ Saved: {barriers_file.name}")

    # 3. Comprehensive JSON report
    # Convert volatility_data to JSON-serializable format
    volatility_json = {
        "overall_volatility": volatility_data.get("overall_volatility", 0),
        "classification": volatility_data.get("classification", ""),
        "dimension_volatilities": volatility_data.get("dimension_volatilities", {}),
        "transition_analysis": (
            asdict(volatility_data["transition_analysis"])
            if "transition_analysis" in volatility_data
            else {}
        ),
    }

    report = {
        "timestamp": timestamp,
        "route_id": route_id,
        "executive_summary": {
            "route_analyzed": route_id,
            "aggregate_score": aggregate_comparison["aggregate_score"],
            "sequential_score": aggregate_comparison["sequential_score"],
            "difference": aggregate_comparison["difference"],
            "total_barriers_detected": len(barriers),
            "pattern_type": pattern,
            "key_finding": f"Sequential analysis adjusts score by {abs(aggregate_comparison['difference']):.1f} points",
        },
        "aggregate_vs_sequential": aggregate_comparison,
        "barriers": [asdict(b) for b in barriers],
        "volatility_analysis": volatility_json,
        "pattern_classification": pattern,
    }

    report_file = report_dir / "sequential_analysis_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved: {report_file.name}")
    print(f"\n📁 Report saved to: {report_dir}")

    return report_dir


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run sequential analysis experiments"""
    print("🧪 CityWalkAgent Sequential Analysis Experiment")
    print("=" * 70)
    print("Proving sequential analysis superiority over aggregate methods")
    print("=" * 70)

    # Load evaluation data
    evaluations, evaluated_route_id = load_evaluation_results(RESULTS_DIR)
    if not evaluations:
        print("\n❌ No evaluation results found. Run experiment 03 first.")
        return

    # Load routes
    routes = load_routes()
    if not routes:
        print("\n❌ No routes found. Run experiment 01 first.")
        return

    # Find the route that was evaluated
    target_route = None
    if evaluated_route_id:
        for route in routes:
            if route.route_id == evaluated_route_id:
                target_route = route
                break

    if not target_route:
        print(f"\n❌ Could not find evaluated route: {evaluated_route_id}")
        print("   Available routes:")
        for r in routes[:10]:
            print(f"      - {r.route_id}")
        return

    print(f"\n✅ Ready to analyze:")
    print(f"   Route: {target_route.route_id}")
    print(f"   Waypoints: {len(target_route.waypoints)}")
    print(f"   Evaluations: {len(evaluations)}")

    # Run experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Experiment 1: Aggregate vs Sequential
    aggregate_comparison = experiment_aggregate_vs_sequential(target_route, evaluations)

    # Experiment 2: Hidden Barriers
    barriers = experiment_hidden_barrier_detection(target_route, evaluations)

    # Experiment 3: Volatility Analysis
    volatility_data = experiment_volatility_analysis(target_route, evaluations)

    # Experiment 4: Route Classification
    pattern = experiment_route_classification(target_route, evaluations)

    # Generate report
    generate_sequential_analysis_report(
        aggregate_comparison,
        barriers,
        volatility_data,
        pattern,
        target_route.route_id,
        timestamp,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 Sequential Analysis Complete!")
    print("=" * 70)

    print("\n📋 Key Findings:")
    print(f"  • Route: {target_route.route_id}")
    print(f"  • Aggregate score: {aggregate_comparison['aggregate_score']:.2f}")
    print(f"  • Sequential score: {aggregate_comparison['sequential_score']:.2f}")
    print(f"  • Difference: {aggregate_comparison['difference']:.2f} points")
    print(f"  • Pattern: {pattern}")

    if barriers:
        print(
            f"  • {len(barriers)} hidden barrier(s) detected (invisible to aggregate)"
        )
        worst = max(barriers, key=lambda b: b.score_drop)
        print(
            f"  • Worst barrier: {worst.score_drop:.1f} point drop in {worst.dimension}"
        )
    else:
        print(f"  • No significant barriers detected")

    print("\n📊 Next Steps:")
    print("  1. Review aggregate_vs_sequential.csv for comparison")
    print("  2. Check hidden_barriers.csv for intervention priorities")
    print("  3. Examine sequential_analysis_report.json for full details")
    print("  4. Run experiment 03 on more routes to compare multiple routes")

    print("\n💡 Academic Contribution:")
    print("  This analysis demonstrates AI-powered continuous pedestrian")
    print("  experience assessment, revealing patterns invisible to traditional")
    print("  point-based walkability scoring methods.")


if __name__ == "__main__":
    main()
