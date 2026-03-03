"""Tests for dual-system evaluation."""

import pytest

from src.agent.system2.persona_reasoner import ReasoningResult, TriggerReason


def test_reasoning_result_creation():
    """ReasoningResult stores core reasoning fields."""
    result = ReasoningResult(
        waypoint_id=5,
        trigger_reason=TriggerReason.VISUAL_CHANGE,
        interpretation="Test interpretation",
        score_change_reason="Scores dropped due to construction",
        persona_divergence="Persona rates safety lower than objective",
        significance="high",
        avoid_recommendation=False,
        decision_reason="No need to avoid",
        prediction="Expect improvement",
        alternative_suggestion=None,
        recommendation="Continue",
        confidence=0.85,
        used_stm_context=True,
        used_ltm_patterns=False,
        personality_factor="high",
        system1_scores={"safety": 8.0, "comfort": 7.0},
        processing_time_seconds=2.5,
        memory_influence={"stm_impact": "high"},
    )

    assert result.waypoint_id == 5
    assert result.trigger_reason == TriggerReason.VISUAL_CHANGE
    assert result.interpretation == "Test interpretation"
    assert result.significance == "high"
    assert result.used_stm_context is True
    assert result.confidence == 0.85


def test_reasoning_result_defaults():
    """ReasoningResult has sensible defaults for optional fields."""
    result = ReasoningResult(
        waypoint_id=1,
        trigger_reason=TriggerReason.SCORE_VOLATILITY,
        interpretation="Basic interpretation",
        score_change_reason=None,
        persona_divergence=None,
        significance="medium",
        avoid_recommendation=False,
        decision_reason=None,
        prediction=None,
        alternative_suggestion=None,
        recommendation=None,
        confidence=0.5,
    )

    assert result.used_stm_context is False
    assert result.used_ltm_patterns is False
    assert result.personality_factor == "unknown"
    assert result.system1_scores == {}
    assert result.processing_time_seconds == 0.0
    assert result.memory_influence == {}


def test_narrative_chapter_creation():
    """NarrativeChapter creation keeps provided fields."""
    from pathlib import Path
    from src.agent.memory.long_term_memory import NarrativeChapter

    chapter = NarrativeChapter(
        waypoint_id=10,
        chapter_number=1,
        timestamp="2025-01-01T12:00:00",
        image_path=Path("/test/image.jpg"),
        visual_description="A busy street",
        system1_scores={"safety": 8.0},
        system2_scores={"safety": 7.5},
        score_adjustments={"safety": -0.5},
        narrative_text="As I walk down this street...",
        key_observation="Busy but manageable",
        emotional_tone="cautious",
    )

    assert chapter.waypoint_id == 10
    assert chapter.chapter_number == 1
    assert chapter.emotional_tone == "cautious"
