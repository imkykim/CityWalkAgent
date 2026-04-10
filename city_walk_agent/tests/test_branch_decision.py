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
            wp_bearing=None,
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
            wp_bearing=None,
        )

    # Fallback: highest avg score = B (7.5) over A (3.5)
    assert result["chosen_direction"] == "B"
    assert result["confidence"] == 0.3


def test_decide_branch_prompt_grants_deviation_permission():
    """Ensure the prompt explicitly tells the persona it can deviate from the route."""
    decider = Decider(framework_id="place_pulse_2.0")
    candidates = [
        make_candidate("A", 0,   {"safety": 7.0, "lively": 5.0}, "Path A"),
        make_candidate("B", 90,  {"safety": 6.0, "lively": 7.0}, "Path B"),
    ]
    personality = MagicMock()
    personality.name = "Tourist"
    personality.description = "Loves scenic routes"

    captured_prompts = []

    def fake_llm(prompt, **kwargs):
        captured_prompts.append(prompt)
        return {
            "chosen_direction": "A",
            "reason": "test",
            "confidence": 0.5,
            "ranking": ["A", "B"],
        }

    with patch("src.agent.system2.decider.call_llm", side_effect=fake_llm):
        decider.decide_branch(
            candidates=candidates,
            ltm_patterns=None,
            personality=personality,
            dimension_ids=["safety", "lively"],
            dimensions={"safety": "Safety", "lively": "Lively"},
            destination_context="Reference walking route bearing: NE (45°). Destination: NE (50°), 400m away. Urgency: navigate",
            wp_bearing=45.0,
        )

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    # Permission language must be present
    assert "FREE to deviate" in prompt
    assert "encouraged" in prompt
    # Old normative language must be gone
    assert "correct road" not in prompt
    assert "recommended walking path" not in prompt
    assert "best balances quality AND progress" not in prompt
