"""Tests for dual-system evaluation."""

from pathlib import Path

import pytest

from src.agent.capabilities.thinking import ThinkingResult, TriggerReason


def test_thinking_result_with_revised_scores():
    """ThinkingResult stores revised scores and flags context usage."""
    result = ThinkingResult(
        waypoint_id=5,
        trigger_reason=TriggerReason.VISUAL_CHANGE,
        interpretation="Test interpretation",
        significance="high",
        pattern_detected="transition",
        prediction="Expect improvement",
        recommendation="Continue",
        confidence=0.85,
        used_vlm=True,
        revised_scores={"safety": 7.5, "comfort": 8.0},
        score_adjustments={"safety": -0.5, "comfort": 1.0},
        revision_reasoning={"safety": "Adjusted down", "comfort": "Adjusted up"},
        memory_influence={"stm_impact": "high"},
        used_stm_context=True,
        used_ltm_patterns=False,
        personality_factor="high",
        vlm_model_used="qwen-vl-max",
        system1_scores={"safety": 8.0, "comfort": 7.0},
        processing_time_seconds=2.5,
    )

    assert result.waypoint_id == 5
    assert result.revised_scores == {"safety": 7.5, "comfort": 8.0}
    assert result.score_adjustments == {"safety": -0.5, "comfort": 1.0}
    assert result.used_stm_context is True


def test_narrative_chapter_creation():
    """NarrativeChapter creation keeps provided fields."""
    from src.agent.capabilities.long_term_memory import NarrativeChapter

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
