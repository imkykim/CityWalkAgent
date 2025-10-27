"""
Sequential analysis metrics

Core algorithms for volatility detection, hidden barriers, and transitions
"""

import statistics
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class Barrier:
    """Detected hidden barrier in walking route"""
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


def calculate_volatility(
    scores: List[float],
    method: str = "diff_std"
) -> float:
    """
    Calculate volatility of score sequence

    Volatility measures the unpredictability and variation in walking experience.
    High volatility = inconsistent, uncomfortable experience
    Low volatility = predictable, consistent experience

    Args:
        scores: Ordered list of scores along route
        method: Calculation method
            - "diff_std": Standard deviation of differences (default)
            - "std": Simple standard deviation
            - "range": Max - min

    Returns:
        Volatility score (higher = more volatile)
    """
    if len(scores) < 2:
        return 0.0

    if method == "diff_std":
        # Standard deviation of differences between consecutive points
        differences = [abs(scores[i+1] - scores[i]) for i in range(len(scores)-1)]
        return statistics.stdev(differences) if len(differences) > 1 else differences[0]

    elif method == "std":
        # Simple standard deviation
        return statistics.stdev(scores)

    elif method == "range":
        # Range (max - min)
        return max(scores) - min(scores)

    else:
        raise ValueError(f"Unknown volatility method: {method}")


def detect_hidden_barriers(
    scores: List[float],
    sequences: List[int],
    waypoints: List[Any],
    dimension: str,
    route_id: str,
    threshold: float = 3.0,
    min_duration: int = 1
) -> List[Barrier]:
    """
    Detect hidden barriers (sudden score drops) along route

    A "hidden barrier" is a segment where walkability suddenly deteriorates,
    which would be masked by aggregate averaging. Examples:
    - Dangerous crossing
    - Sudden loss of sidewalk
    - Threatening area
    - Poor lighting section

    Args:
        scores: Ordered list of scores
        sequences: Sequence IDs corresponding to scores
        waypoints: Waypoint objects with GPS coordinates
        dimension: Dimension being analyzed
        route_id: Route identifier
        threshold: Minimum score drop to count as barrier
        min_duration: Minimum waypoints a barrier must persist

    Returns:
        List of detected Barrier objects
    """
    if len(scores) < 2:
        return []

    barriers = []
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

            # Only record if meets minimum duration
            if duration >= min_duration:
                # Calculate recovery time
                recovery_time = None
                for k in range(j, len(scores)):
                    if scores[k] >= scores[i] - 1.0:  # Score recovers to near original
                        recovery_time = k - i
                        break

                # Get waypoint location
                seq_num = sequences[i + 1]
                waypoint = None
                for wp in waypoints:
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
                    route_id=route_id,
                    waypoint_sequence=seq_num,
                    score_drop=score_drop,
                    before_score=scores[i],
                    after_score=scores[i + 1],
                    dimension=dimension,
                    lat=waypoint.lat if waypoint else 0.0,
                    lon=waypoint.lon if waypoint else 0.0,
                    duration=duration,
                    recovery_time=recovery_time,
                    severity_level=severity
                )
                barriers.append(barrier)

            # Skip past this barrier
            i = j
        else:
            i += 1

    # Sort by severity (score drop)
    barriers.sort(key=lambda b: b.score_drop, reverse=True)
    return barriers


def analyze_transitions(scores: List[float]) -> TransitionAnalysis:
    """
    Analyze how experience changes along route

    Transitions reveal the dynamics of walking experience:
    - Smooth transitions: comfortable, expected changes
    - Abrupt transitions: jarring, uncomfortable shifts

    Args:
        scores: Ordered list of scores along route

    Returns:
        TransitionAnalysis object with comprehensive metrics
    """
    if len(scores) < 2:
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

    for i in range(len(scores) - 1):
        change = scores[i + 1] - scores[i]
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
        transition_volatility=volatility
    )


def calculate_sequential_score(
    base_score: float,
    volatility: float,
    num_barriers: int,
    barrier_severity: float,
    num_severe_transitions: int,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate route score accounting for sequential patterns

    Unlike aggregate average, this penalizes:
    - High volatility (uncomfortable, unpredictable)
    - Hidden barriers (dangerous segments)
    - Abrupt transitions (jarring experiences)

    Args:
        base_score: Aggregate average score
        volatility: Volatility metric
        num_barriers: Number of detected barriers
        barrier_severity: Sum of barrier score drops
        num_severe_transitions: Number of severe transitions
        weights: Custom penalty weights

    Returns:
        Sequential score (adjusted for experience quality)
    """
    # Default weights
    if weights is None:
        weights = {
            "volatility_penalty": 0.3,
            "barrier_penalty": 0.5,
            "transition_penalty": 0.2
        }

    # Volatility penalty (normalized to 0-1 scale)
    volatility_penalty = min(volatility / 5.0, 1.0) * weights["volatility_penalty"]

    # Barrier penalty
    barrier_penalty = min(barrier_severity / 10.0, 2.0) * weights["barrier_penalty"]

    # Transition penalty
    transition_penalty = num_severe_transitions * 0.1 * weights["transition_penalty"]

    # Calculate final score
    sequential_score = base_score - volatility_penalty - barrier_penalty - transition_penalty

    # Ensure score stays in valid range
    return max(1.0, min(10.0, sequential_score))


def classify_route_pattern(
    avg_score: float,
    volatility: float,
    num_barriers: int,
    score_trend: float,
    volatility_threshold: float = 2.0
) -> str:
    """
    Classify route into pattern types

    Pattern types:
    - stable_high: Consistently good (low volatility, high scores)
    - stable_low: Consistently poor (low volatility, low scores)
    - improving: Scores trend upward over route
    - declining: Scores trend downward over route
    - volatile: High variation, unpredictable experience
    - barrier_present: Contains significant drops

    Args:
        avg_score: Average score across route
        volatility: Volatility metric
        num_barriers: Number of detected barriers
        score_trend: Trend metric (positive = improving, negative = declining)
        volatility_threshold: Threshold for volatile classification

    Returns:
        Pattern type string
    """
    # Classification logic
    if num_barriers >= 2:
        return "barrier_present"
    elif volatility > volatility_threshold * 1.5:
        return "volatile"
    elif score_trend > 1.0:
        return "improving"
    elif score_trend < -1.0:
        return "declining"
    elif volatility < volatility_threshold * 0.5:
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
