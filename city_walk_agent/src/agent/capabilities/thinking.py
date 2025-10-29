"""Thinking capability - agent's reasoning and decision-making.

This capability handles the reasoning phase of the agent's cognitive loop.
It takes observations and applies personality-driven logic to make decisions
and generate explanations.

Design principle: REASON, don't perceive or act.
"""

import statistics
from typing import Any, Dict, List, Optional

from src.agent.constants import (
    CONFIDENCE_BARRIER_WEIGHT,
    CONFIDENCE_SCORE_WEIGHT,
    CONFIDENCE_VOLATILITY_WEIGHT,
    EXCELLENT_SCORE_THRESHOLD,
    LOW_VOLATILITY_THRESHOLD,
)
from src.utils.logging import get_logger


class ThinkingCapability:
    """Reasoning capability for making decisions about routes.

    This is a stateless reasoning engine that applies personality weights
    and decision thresholds to observations to produce recommendations.

    Example:
        ```python
        thinker = ThinkingCapability()
        decision = thinker.think(
            observation=observation,
            sequential_analysis=analysis,
            personality_weights={"safety": 2.0, "comfort": 1.2},
            decision_thresholds={"min_overall_score": 6.5},
            explanation_style="safety"
        )
        # decision contains: recommendation, confidence, explanation
        ```
    """

    def __init__(self):
        """Initialize thinking capability.

        The capability is stateless - all decision context is provided
        to the think() method.
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("ThinkingCapability initialized")

    def think(
        self,
        observation: Dict[str, Any],
        sequential_analysis: Dict[str, Any],
        personality_weights: Dict[str, float],
        decision_thresholds: Dict[str, float],
        explanation_style: str = "balanced",
    ) -> Dict[str, Any]:
        """Make a decision about a route based on observation and personality.

        This is the main reasoning method. It applies personality-driven
        logic to observations and produces a recommendation.

        Args:
            observation: Observation dict from ObservationCapability.
            sequential_analysis: Analysis dict from SequentialAnalyzer.
            personality_weights: Weight for each dimension (e.g., {"safety": 2.0}).
            decision_thresholds: Decision criteria (e.g., {"min_overall_score": 6.5}).
            explanation_style: Style of explanation ("safety", "scenic", "balanced", "technical").

        Returns:
            Dict[str, Any] containing:
                - recommendation: "accept" | "reject"
                - confidence: float (0-1)
                - weighted_score: float
                - explanation: str (human-readable)
                - concerns: List[str]
                - highlights: List[str]
                - key_factors: Dict with detailed reasoning
        """
        route_id = observation.get("route_info", {}).get("route_id", "unknown")

        self.logger.debug(
            "Thinking about route",
            route_id=route_id,
            explanation_style=explanation_style,
        )

        # Extract key information
        dimension_stats = observation.get("dimension_stats", {})
        volatility = sequential_analysis.get("volatility", 0.0)
        hidden_barriers = sequential_analysis.get("hidden_barriers", [])
        pattern_type = sequential_analysis.get("pattern_type", "unknown")

        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(
            dimension_stats, personality_weights
        )

        # Identify concerns and highlights
        concerns = self._identify_concerns(
            dimension_stats,
            personality_weights,
            decision_thresholds,
            volatility,
            hidden_barriers,
        )

        highlights = self._identify_highlights(
            dimension_stats, personality_weights, volatility
        )

        # Make recommendation
        recommendation = self._make_recommendation(
            weighted_score,
            decision_thresholds,
            concerns,
            hidden_barriers,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            weighted_score,
            decision_thresholds,
            volatility,
            len(hidden_barriers),
            recommendation,
        )

        # Generate explanation
        explanation = self._generate_explanation(
            recommendation=recommendation,
            weighted_score=weighted_score,
            concerns=concerns,
            highlights=highlights,
            volatility=volatility,
            pattern_type=pattern_type,
            style=explanation_style,
        )

        # Build key factors for transparency
        key_factors = {
            "weighted_score": weighted_score,
            "volatility": volatility,
            "num_barriers": len(hidden_barriers),
            "pattern_type": pattern_type,
            "dimension_averages": {
                dim: stats.get("avg", 0.0)
                for dim, stats in dimension_stats.items()
            },
            "applied_weights": personality_weights,
            "thresholds_used": decision_thresholds,
        }

        decision = {
            "recommendation": recommendation,
            "confidence": confidence,
            "weighted_score": weighted_score,
            "explanation": explanation,
            "concerns": concerns,
            "highlights": highlights,
            "key_factors": key_factors,
        }

        self.logger.info(
            "Decision made",
            route_id=route_id,
            recommendation=recommendation,
            confidence=round(confidence, 2),
            weighted_score=round(weighted_score, 2),
        )

        return decision

    def _calculate_weighted_score(
        self,
        dimension_stats: Dict[str, Dict[str, float]],
        personality_weights: Dict[str, float],
    ) -> float:
        """Calculate overall weighted score.

        Args:
            dimension_stats: Statistics for each dimension.
            personality_weights: Weight for each dimension.

        Returns:
            Weighted average score.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for dimension_id, weight in personality_weights.items():
            if dimension_id in dimension_stats:
                avg_score = dimension_stats[dimension_id].get("avg", 0.0)
                weighted_sum += avg_score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _identify_concerns(
        self,
        dimension_stats: Dict[str, Dict[str, float]],
        personality_weights: Dict[str, float],
        decision_thresholds: Dict[str, float],
        volatility: float,
        hidden_barriers: List[Dict[str, Any]],
    ) -> List[str]:
        """Identify concerns about the route.

        Args:
            dimension_stats: Statistics for each dimension.
            personality_weights: Weight for each dimension.
            decision_thresholds: Decision criteria.
            volatility: Route volatility.
            hidden_barriers: List of detected barriers.

        Returns:
            List of concern descriptions.
        """
        concerns = []

        # Check for low scores in important dimensions
        for dimension_id, weight in personality_weights.items():
            if dimension_id not in dimension_stats:
                continue

            avg_score = dimension_stats[dimension_id].get("avg", 0.0)
            threshold_key = f"min_{dimension_id}_score"

            # Check dimension-specific threshold
            if threshold_key in decision_thresholds:
                min_threshold = decision_thresholds[threshold_key]
                if avg_score < min_threshold:
                    concerns.append(
                        f"Low {dimension_id} score ({avg_score:.1f} < {min_threshold})"
                    )

            # Check if important dimension (high weight) has low score
            elif weight >= 1.5 and avg_score < 6.5:
                concerns.append(
                    f"{dimension_id.capitalize()} below expectations ({avg_score:.1f})"
                )

        # Check volatility
        max_volatility = decision_thresholds.get("max_volatility", 999)
        if volatility > max_volatility:
            concerns.append(
                f"High volatility ({volatility:.1f} > {max_volatility})"
            )

        # Check hidden barriers
        max_barriers = decision_thresholds.get("max_barriers", 999)
        if len(hidden_barriers) > max_barriers:
            concerns.append(
                f"Too many barriers ({len(hidden_barriers)} > {max_barriers})"
            )

        return concerns

    def _identify_highlights(
        self,
        dimension_stats: Dict[str, Dict[str, float]],
        personality_weights: Dict[str, float],
        volatility: float,
    ) -> List[str]:
        """Identify positive highlights about the route.

        Args:
            dimension_stats: Statistics for each dimension.
            personality_weights: Weight for each dimension.
            volatility: Route volatility.

        Returns:
            List of highlight descriptions.
        """
        highlights = []

        # Check for excellent scores in important dimensions
        for dimension_id, weight in personality_weights.items():
            if dimension_id not in dimension_stats:
                continue

            avg_score = dimension_stats[dimension_id].get("avg", 0.0)

            # Highlight if important dimension has excellent score
            if weight >= 1.5 and avg_score >= EXCELLENT_SCORE_THRESHOLD:
                highlights.append(
                    f"Excellent {dimension_id} ({avg_score:.1f}/10)"
                )

        # Check for low volatility (consistency)
        if volatility <= LOW_VOLATILITY_THRESHOLD:
            highlights.append(f"Very consistent experience (volatility: {volatility:.1f})")

        # Check for universally high scores
        all_scores = [
            stats.get("avg", 0.0)
            for stats in dimension_stats.values()
        ]
        if all_scores and min(all_scores) >= 7.0:
            highlights.append("All dimensions score well")

        return highlights

    def _make_recommendation(
        self,
        weighted_score: float,
        decision_thresholds: Dict[str, float],
        concerns: List[str],
        hidden_barriers: List[Dict[str, Any]],
    ) -> str:
        """Make accept/reject recommendation.

        Args:
            weighted_score: Overall weighted score.
            decision_thresholds: Decision criteria.
            concerns: List of identified concerns.
            hidden_barriers: List of detected barriers.

        Returns:
            "accept" or "reject"
        """
        # Check minimum overall score
        min_overall = decision_thresholds.get("min_overall_score", 6.5)
        if weighted_score < min_overall:
            return "reject"

        # Check if there are critical concerns
        max_barriers = decision_thresholds.get("max_barriers", 999)
        if len(hidden_barriers) > max_barriers:
            return "reject"

        # Check for primary dimension failures
        if any("min_primary_score" in str(c) for c in concerns):
            return "reject"

        # Otherwise accept
        return "accept"

    def _calculate_confidence(
        self,
        weighted_score: float,
        decision_thresholds: Dict[str, float],
        volatility: float,
        num_barriers: int,
        recommendation: str,
    ) -> float:
        """Calculate confidence in the decision.

        Confidence is based on:
        - How far the weighted score is from the threshold (margin)
        - Route volatility (lower is better)
        - Number of barriers (fewer is better)

        Args:
            weighted_score: Overall weighted score.
            decision_thresholds: Decision criteria.
            volatility: Route volatility.
            num_barriers: Number of detected barriers.
            recommendation: The recommendation made.

        Returns:
            Confidence value between 0 and 1.
        """
        # Score component: how far from threshold
        min_overall = decision_thresholds.get("min_overall_score", 6.5)
        score_margin = abs(weighted_score - min_overall)

        if recommendation == "accept":
            # For accept: higher scores = higher confidence
            score_confidence = min(score_margin / 2.0, 1.0)
        else:
            # For reject: lower scores = higher confidence
            score_confidence = min(score_margin / 2.0, 1.0)

        # Volatility component: lower volatility = higher confidence
        max_volatility = decision_thresholds.get("max_volatility", 2.0)
        volatility_confidence = max(0.0, 1.0 - (volatility / max_volatility))

        # Barrier component: fewer barriers = higher confidence
        max_barriers = decision_thresholds.get("max_barriers", 2)
        barrier_confidence = max(0.0, 1.0 - (num_barriers / max(max_barriers, 1)))

        # Weighted combination
        confidence = (
            CONFIDENCE_SCORE_WEIGHT * score_confidence
            + CONFIDENCE_VOLATILITY_WEIGHT * volatility_confidence
            + CONFIDENCE_BARRIER_WEIGHT * barrier_confidence
        )

        return min(1.0, max(0.0, confidence))

    def _generate_explanation(
        self,
        recommendation: str,
        weighted_score: float,
        concerns: List[str],
        highlights: List[str],
        volatility: float,
        pattern_type: str,
        style: str,
    ) -> str:
        """Generate human-readable explanation.

        Args:
            recommendation: The recommendation made.
            weighted_score: Overall weighted score.
            concerns: List of concerns.
            highlights: List of highlights.
            volatility: Route volatility.
            pattern_type: Route pattern classification.
            style: Explanation style.

        Returns:
            Human-readable explanation string.
        """
        if style == "safety":
            return self._explain_safety_focus(
                recommendation, weighted_score, concerns, highlights
            )
        elif style == "scenic":
            return self._explain_scenic_focus(
                recommendation, weighted_score, highlights, pattern_type
            )
        elif style == "technical":
            return self._explain_technical(
                recommendation, weighted_score, volatility, concerns, highlights
            )
        else:  # balanced
            return self._explain_balanced(
                recommendation, weighted_score, concerns, highlights
            )

    def _explain_safety_focus(
        self,
        recommendation: str,
        weighted_score: float,
        concerns: List[str],
        highlights: List[str],
    ) -> str:
        """Generate safety-focused explanation."""
        if recommendation == "accept":
            msg = f"✅ Route approved (score: {weighted_score:.1f}/10). "
            if highlights:
                msg += f"Strengths: {', '.join(highlights[:2])}. "
            msg += "Safe for walking."
        else:
            msg = f"⚠️ Route not recommended (score: {weighted_score:.1f}/10). "
            if concerns:
                msg += f"Safety concerns: {', '.join(concerns[:2])}. "
            msg += "Consider alternative route."

        return msg

    def _explain_scenic_focus(
        self,
        recommendation: str,
        weighted_score: float,
        highlights: List[str],
        pattern_type: str,
    ) -> str:
        """Generate scenic-focused explanation."""
        if recommendation == "accept":
            msg = f"✨ Great route! (score: {weighted_score:.1f}/10). "
            if highlights:
                msg += f"{', '.join(highlights[:2])}. "
            msg += f"Pattern: {pattern_type}."
        else:
            msg = f"Route score: {weighted_score:.1f}/10. "
            msg += "May not meet your expectations for scenic experience."

        return msg

    def _explain_balanced(
        self,
        recommendation: str,
        weighted_score: float,
        concerns: List[str],
        highlights: List[str],
    ) -> str:
        """Generate balanced explanation."""
        if recommendation == "accept":
            msg = f"Route score: {weighted_score:.1f}/10. "
            if highlights:
                msg += f"Highlights: {', '.join(highlights[:2])}. "
            if concerns:
                msg += f"Minor concerns: {', '.join(concerns[:1])}."
            else:
                msg += "Overall good quality."
        else:
            msg = f"Route score: {weighted_score:.1f}/10 (below threshold). "
            if concerns:
                msg += f"Issues: {', '.join(concerns[:2])}."

        return msg

    def _explain_technical(
        self,
        recommendation: str,
        weighted_score: float,
        volatility: float,
        concerns: List[str],
        highlights: List[str],
    ) -> str:
        """Generate technical explanation with details."""
        msg = f"Decision: {recommendation.upper()}. "
        msg += f"Weighted score: {weighted_score:.2f}/10. "
        msg += f"Volatility: {volatility:.2f}. "

        if highlights:
            msg += f"Positive factors: {len(highlights)}. "
        if concerns:
            msg += f"Concerns: {len(concerns)}. "

        msg += f"Details: {', '.join(concerns[:2]) if concerns else 'No major issues'}."

        return msg
