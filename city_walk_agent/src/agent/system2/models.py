"""Data models for the System 2 reasoning pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class System1Evidence:
    """Aggregated System 1 output passed into System 2.

    Attributes:
        waypoint_results:  Per-waypoint analysis dicts from ContinuousAnalyzer.
        route_metadata:    Route-level info (route_id, length_km, framework_id, personality).
        visual_changes:    Waypoint IDs where visual change was detected.
        barriers:          Detected hidden barrier dicts.
        overall_stats:     Aggregate statistics (avg scores, volatility, etc.).
        reasoning_results: Waypoint-level PersonaReasoner results.
    """
    waypoint_results:  List[Dict[str, Any]]
    route_metadata:    Dict[str, Any]
    visual_changes:    List[int]            = field(default_factory=list)
    barriers:          List[Dict[str, Any]] = field(default_factory=list)
    overall_stats:     Dict[str, Any]       = field(default_factory=dict)
    reasoning_results: List[Any]            = field(default_factory=list)


@dataclass
class InterpretResult:
    """Output of Interpreter: WHY did scores change?"""
    score_change_explanations: Dict[str, str]       = field(default_factory=dict)
    persona_divergence_points: List[Dict[str, Any]] = field(default_factory=list)
    key_transitions:           List[Dict[str, Any]] = field(default_factory=list)
    interpretation_summary:    str                  = ""


@dataclass
class DecideResult:
    """Output of Decider: WHAT action to take?"""
    recommendation:     str                  = "accept"  # "accept" | "reject" | "modify"
    confidence:         float                = 0.0
    avoid_segments:     List[Dict[str, Any]] = field(default_factory=list)
    decision_reasoning: str                  = ""
    concerns:           List[str]            = field(default_factory=list)
    opportunities:      List[str]            = field(default_factory=list)


@dataclass
class PlanResult:
    """Output of Planner: HOW to walk?"""
    primary_route:                List[int]            = field(default_factory=list)
    alternative_routes:           List[Dict[str, Any]] = field(default_factory=list)
    route_comparison:             Dict[str, Any]       = field(default_factory=dict)
    persona_priority_explanation: str                  = ""


@dataclass
class ReportResult:
    """Output of Reporter: final comprehensive evaluation."""
    executive_summary:          str              = ""
    dimension_analysis:         Dict[str, str]   = field(default_factory=dict)
    urban_planning_suggestions: List[str]        = field(default_factory=list)
    target_user_suitability:    Dict[str, Any]   = field(default_factory=dict)
    full_narrative:             str              = ""
