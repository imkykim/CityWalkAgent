from unittest.mock import patch, MagicMock
from src.agent.system2.decider import Decider


def make_candidate(direction, heading, scores, interpretation, key_concern=None):
    return {
        "direction": direction,
        "heading": heading,
        "pano_id": "test_pano",
        "scores": scores,
        "interpretation": interpretation,
        "key_concern": key_concern,
    }


def test_decide_branch_returns_valid_direction():
    decider = Decider(framework_id="place_pulse_2.0")
    candidates = [
        make_candidate("A", 0,   {"safety": 8.0, "lively": 3.0}, "Wide safe path"),
        make_candidate("B", 90,  {"safety": 5.0, "lively": 6.0}, "Construction zone", "Stroller blocked"),
        make_candidate("C", 180, {"safety": 7.5, "lively": 5.0}, "Residential street"),
    ]

    personality = MagicMock()
    personality.name = "Parent with Young Children"
    personality.description = "Prioritizes safety and stroller access"

    with patch("src.agent.system2.decider.call_llm") as mock_llm:
        mock_llm.return_value = {
            "chosen_direction": "A",
            "reason": "Safest path for stroller",
            "confidence": 0.9,
            "ranking": ["A", "C", "B"],
        }
        result = decider.decide_branch(
            candidates=candidates,
            ltm_patterns=None,
            personality=personality,
            dimension_ids=["safety", "lively"],
            dimensions={"safety": "Safety", "lively": "Lively"},
        )

    assert result["chosen_direction"] == "A"
    assert result["chosen_heading"] == 0
    assert result["confidence"] == 0.9
    assert result["ranking"] == ["A", "C", "B"]


def test_decide_branch_fallback_on_llm_failure():
    decider = Decider(framework_id="place_pulse_2.0")
    candidates = [
        make_candidate("A", 0,   {"safety": 4.0, "lively": 3.0}, "Bad path"),
        make_candidate("B", 90,  {"safety": 8.0, "lively": 7.0}, "Great path"),
    ]

    personality = MagicMock()
    personality.name = "Test"
    personality.description = ""

    with patch("src.agent.system2.decider.call_llm") as mock_llm:
        mock_llm.return_value = None  # LLM failure

        result = decider.decide_branch(
            candidates=candidates,
            ltm_patterns=None,
            personality=personality,
            dimension_ids=["safety", "lively"],
            dimensions={"safety": "Safety", "lively": "Lively"},
        )

    # Fallback: highest avg score = B (7.5) over A (3.5)
    assert result["chosen_direction"] == "B"
    assert result["confidence"] == 0.3
