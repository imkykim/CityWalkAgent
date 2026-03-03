"""Integration tests for framework-agnostic functionality.

Tests that PersonaReasoner, RouteVisualizer, and CityWalkAgent work correctly
with all supported evaluation frameworks.
"""

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

from src.agent.system2.persona_reasoner import PersonaReasoner, TriggerReason
from src.utils.visualization import RouteVisualizer
from src.agent.orchestrator import CityWalkAgent
from src.agent.config import get_preset
from src.core import DEFAULT_FRAMEWORK_ID, load_framework


# All supported frameworks
FRAMEWORKS = [
    DEFAULT_FRAMEWORK_ID,
    "sagai_2025",
    "streetagent_5d",
    "ewing_handy_5d",
    "kaplan_4d",
    "phenomenology_3d",
]


class TestFrameworkLoading:
    """Test that all frameworks load correctly."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_framework_loads(self, framework_id):
        """Test that framework can be loaded."""
        framework = load_framework(framework_id)

        assert framework is not None
        assert "framework_id" in framework
        assert "framework_name" in framework
        assert "dimensions" in framework
        assert len(framework["dimensions"]) > 0
        assert framework["framework_id"] == framework_id

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_framework_has_valid_dimensions(self, framework_id):
        """Test that framework dimensions are valid."""
        framework = load_framework(framework_id)

        for dim in framework["dimensions"]:
            assert "id" in dim
            assert "name_en" in dim
            assert "description" in dim
            assert isinstance(dim["id"], str)
            assert isinstance(dim["name_en"], str)
            assert len(dim["id"]) > 0
            assert len(dim["name_en"]) > 0


class TestPersonaReasonerFrameworks:
    """Test PersonaReasoner with different frameworks."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_persona_reasoner_initializes(self, framework_id):
        """Test that PersonaReasoner initializes with framework."""
        reasoner = PersonaReasoner(framework_id=framework_id)

        assert reasoner.framework_id == framework_id
        assert reasoner.framework is not None
        assert reasoner.dimensions is not None
        assert reasoner.dimension_ids is not None
        assert len(reasoner.dimension_ids) > 0

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_fallback_result_fields(self, framework_id):
        """Test that fallback result has required fields."""
        reasoner = PersonaReasoner(framework_id=framework_id)
        framework = load_framework(framework_id)

        system1_scores = {dim["id"]: 7.5 for dim in framework["dimensions"]}

        fallback = reasoner._create_fallback_result(
            waypoint_id=1,
            trigger_reason=TriggerReason.VISUAL_CHANGE,
            system1_scores=system1_scores,
            error="Test error"
        )

        assert fallback.waypoint_id == 1
        assert fallback.trigger_reason == TriggerReason.VISUAL_CHANGE
        assert fallback.system1_scores == system1_scores
        assert fallback.confidence == 0.0
        assert fallback.significance == "low"


class TestRouteVisualizerFrameworks:
    """Test RouteVisualizer with different frameworks."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_visualizer_initializes(self, framework_id):
        """Test that RouteVisualizer initializes with framework."""
        viz = RouteVisualizer(framework_id=framework_id)

        assert viz.framework_id == framework_id
        assert viz.framework is not None
        assert viz.dim_config is not None
        assert len(viz.dim_config) > 0

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_dimension_config_has_all_dimensions(self, framework_id):
        """Test that dimension config includes all framework dimensions."""
        viz = RouteVisualizer(framework_id=framework_id)
        framework = load_framework(framework_id)

        # Check that all dimensions have color and label
        for dim in framework["dimensions"]:
            assert dim["id"] in viz.dim_config
            assert "color" in viz.dim_config[dim["id"]]
            assert "label" in viz.dim_config[dim["id"]]

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_get_dim_config_uses_framework_dimensions(self, framework_id):
        """Test that _get_dim_config uses framework dimensions."""
        viz = RouteVisualizer(framework_id=framework_id)
        framework = load_framework(framework_id)

        dim_keys = [dim["id"] for dim in framework["dimensions"]]
        config = viz._get_dim_config(dim_keys)

        # Check that all dimensions are configured
        assert len(config) == len(framework["dimensions"])
        for dim_id in dim_keys:
            assert dim_id in config
            assert "color" in config[dim_id]
            assert "label" in config[dim_id]

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_plot_with_framework_dimensions(self, framework_id):
        """Test that plotting works with framework dimensions."""
        viz = RouteVisualizer(framework_id=framework_id)
        framework = load_framework(framework_id)

        # Create mock score data for all dimensions
        scores = {dim["id"]: [7.0, 7.5, 8.0, 7.8, 8.2] for dim in framework["dimensions"]}
        waypoint_ids = ["wp_0", "wp_1", "wp_2", "wp_3", "wp_4"]

        # Test plotting (should not raise errors)
        fig = viz.plot_scores(
            scores=scores,
            waypoint_ids=waypoint_ids,
            title=f"Test {framework_id}",
            save_path=None
        )

        assert fig is not None


class TestCityWalkAgentFrameworks:
    """Test CityWalkAgent with different frameworks."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_agent_initializes_with_framework(self, framework_id):
        """Test that CityWalkAgent initializes with framework."""
        personality = get_preset("homebuyer", framework_id)

        agent = CityWalkAgent(
            agent_id="test_agent",
            personality=personality,
            framework_id=framework_id
        )

        assert agent.framework_id == framework_id
        assert agent.personality is not None

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_persona_reasoner_uses_framework(self, framework_id):
        """Test that CityWalkAgent's PersonaReasoner uses correct framework."""
        personality = get_preset("homebuyer", framework_id)

        agent = CityWalkAgent(
            agent_id="test_agent",
            personality=personality,
            framework_id=framework_id
        )

        # Access persona_reasoner to trigger lazy loading
        reasoner = agent.persona_reasoner

        assert reasoner.framework_id == framework_id
        assert reasoner.framework is not None

        framework = load_framework(framework_id)
        assert len(reasoner.dimension_ids) == len(framework["dimensions"])

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_agent_from_preset(self, framework_id):
        """Test creating agent from preset with framework."""
        agent = CityWalkAgent.from_preset(
            preset_name="homebuyer",
            framework_id=framework_id,
            agent_id=f"test_{framework_id}"
        )

        assert agent.framework_id == framework_id
        assert agent.personality is not None
        assert agent.persona_reasoner.framework_id == framework_id


class TestFrameworkSwitching:
    """Test switching between frameworks."""

    def test_different_agents_different_frameworks(self):
        """Test that different agents can use different frameworks simultaneously."""
        agent1 = CityWalkAgent.from_preset("homebuyer", "sagai_2025", "agent1")
        agent2 = CityWalkAgent.from_preset("homebuyer", "streetagent_5d", "agent2")

        assert agent1.framework_id == "sagai_2025"
        assert agent2.framework_id == "streetagent_5d"

        assert len(agent1.persona_reasoner.dimension_ids) == 4  # sagai_2025 has 4 dimensions
        assert len(agent2.persona_reasoner.dimension_ids) == 5  # streetagent_5d has 5 dimensions

    def test_visualizers_with_different_frameworks(self):
        """Test that multiple visualizers can use different frameworks."""
        viz1 = RouteVisualizer(framework_id="sagai_2025")
        viz2 = RouteVisualizer(framework_id="phenomenology_3d")

        assert viz1.framework_id == "sagai_2025"
        assert viz2.framework_id == "phenomenology_3d"

        assert len(viz1.dim_config) == 4  # sagai_2025 has 4 dimensions
        assert len(viz2.dim_config) == 3  # phenomenology_3d has 3 dimensions


class TestBackwardCompatibility:
    """Test that default framework follows DEFAULT_FRAMEWORK_ID."""

    def test_persona_reasoner_defaults_to_default_framework(self):
        """Test that PersonaReasoner defaults to configured default."""
        reasoner = PersonaReasoner()
        assert reasoner.framework_id == DEFAULT_FRAMEWORK_ID

    def test_visualizer_defaults_to_default_framework(self):
        """Test that RouteVisualizer defaults to configured default."""
        viz = RouteVisualizer()
        assert viz.framework_id == DEFAULT_FRAMEWORK_ID

    def test_agent_defaults_to_default_framework(self):
        """Test that CityWalkAgent defaults to default framework when not specified."""
        personality = get_preset("homebuyer", DEFAULT_FRAMEWORK_ID)
        agent = CityWalkAgent(agent_id="test", personality=personality)
        assert agent.framework_id == DEFAULT_FRAMEWORK_ID


class TestDimensionCounts:
    """Test that dimension counts match framework definitions."""

    def test_sagai_2025_has_4_dimensions(self):
        """Test that sagai_2025 has 4 dimensions."""
        framework = load_framework("sagai_2025")
        assert len(framework["dimensions"]) == 4

    def test_streetagent_5d_has_5_dimensions(self):
        """Test that streetagent_5d has 5 dimensions."""
        framework = load_framework("streetagent_5d")
        assert len(framework["dimensions"]) == 5

    def test_place_pulse_has_4_dimensions(self):
        """Test that place_pulse_2.0 has 4 dimensions."""
        framework = load_framework("place_pulse_2.0")
        assert len(framework["dimensions"]) == 4

    def test_ewing_handy_5d_has_5_dimensions(self):
        """Test that ewing_handy_5d has 5 dimensions."""
        framework = load_framework("ewing_handy_5d")
        assert len(framework["dimensions"]) == 5

    def test_kaplan_4d_has_4_dimensions(self):
        """Test that kaplan_4d has 4 dimensions."""
        framework = load_framework("kaplan_4d")
        assert len(framework["dimensions"]) == 4

    def test_phenomenology_3d_has_3_dimensions(self):
        """Test that phenomenology_3d has 3 dimensions."""
        framework = load_framework("phenomenology_3d")
        assert len(framework["dimensions"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
