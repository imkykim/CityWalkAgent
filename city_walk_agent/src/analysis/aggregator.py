"""
Traditional aggregate analyzer

Implements traditional point-based aggregation for comparison with sequential methods
"""

import statistics
from typing import List, Dict, Any
from collections import defaultdict


class AggregateAnalyzer:
    """
    Traditional aggregate scoring analyzer

    Implements conventional point-based walkability assessment for
    comparison with sequential methods. This is what most existing
    research uses.

    Methods:
    - Simple averaging across all points
    - Dimension-level aggregation
    - Basic statistical summaries
    """

    def __init__(self, evaluations: List[Dict[str, Any]]):
        """
        Initialize with evaluation results

        Args:
            evaluations: List of evaluation result dicts
        """
        self.evaluations = evaluations
        self.scores_by_dimension = self._organize_by_dimension()

    def _organize_by_dimension(self) -> Dict[str, List[float]]:
        """Organize scores by dimension"""
        scores = defaultdict(list)

        for eval_result in self.evaluations:
            dimension = eval_result.get("dimension_id", "")
            score = eval_result.get("score", 0.0)
            scores[dimension].append(score)

        return dict(scores)

    def calculate_overall_average(self) -> float:
        """
        Calculate overall average score (traditional method)

        Returns:
            Simple mean of all scores
        """
        all_scores = []
        for scores in self.scores_by_dimension.values():
            all_scores.extend(scores)

        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def calculate_dimension_averages(self) -> Dict[str, float]:
        """
        Calculate average score per dimension

        Returns:
            Dict mapping dimension to average score
        """
        averages = {}

        for dimension, scores in self.scores_by_dimension.items():
            averages[dimension] = sum(scores) / len(scores) if scores else 0.0

        return averages

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistical summary

        Returns:
            Dict with statistical metrics
        """
        all_scores = []
        for scores in self.scores_by_dimension.values():
            all_scores.extend(scores)

        if not all_scores:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "count": 0
            }

        return {
            "mean": statistics.mean(all_scores),
            "median": statistics.median(all_scores),
            "std": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            "min": min(all_scores),
            "max": max(all_scores),
            "range": max(all_scores) - min(all_scores),
            "count": len(all_scores)
        }

    def calculate_dimension_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics per dimension

        Returns:
            Dict mapping dimension to stats dict
        """
        dimension_stats = {}

        for dimension, scores in self.scores_by_dimension.items():
            if not scores:
                dimension_stats[dimension] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0
                }
            else:
                dimension_stats[dimension] = {
                    "mean": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }

        return dimension_stats

    def classify_route_quality(self, threshold_good: float = 7.0, threshold_poor: float = 4.0) -> str:
        """
        Classify route quality based on aggregate score

        Args:
            threshold_good: Score threshold for "good" classification
            threshold_poor: Score threshold for "poor" classification

        Returns:
            Quality classification string
        """
        avg_score = self.calculate_overall_average()

        if avg_score >= threshold_good:
            return "good"
        elif avg_score <= threshold_poor:
            return "poor"
        else:
            return "moderate"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive aggregate analysis summary

        Returns:
            Summary dict with all aggregate metrics
        """
        return {
            "method": "aggregate",
            "overall_average": self.calculate_overall_average(),
            "dimension_averages": self.calculate_dimension_averages(),
            "statistics": self.calculate_statistics(),
            "dimension_statistics": self.calculate_dimension_statistics(),
            "quality_classification": self.classify_route_quality()
        }
