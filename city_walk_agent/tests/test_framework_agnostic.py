"""Integration tests for framework-agnostic functionality.

Tests that ThinkingModule, RouteVisualizer, and WalkingAgent work correctly
with all supported evaluation frameworks.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.agent.capabilities.thinking import ThinkingModule, TriggerReason
from src.utils.visualization import RouteVisualizer
from src.agent.walking_agent import WalkingAgent
from src.agent.config import get_preset
from src.config import load_framework


# All supported frameworks
FRAMEWORKS = [
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


class TestThinkingModuleFrameworks:
    """Test ThinkingModule with different frameworks."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_thinking_module_initializes(self, framework_id):
        """Test that ThinkingModule initializes with framework."""
        thinking = ThinkingModule(framework_id=framework_id)

        assert thinking.framework_id == framework_id
        assert thinking.framework is not None
        assert thinking.dimensions is not None
        assert thinking.dimension_ids is not None
        assert len(thinking.dimension_ids) > 0

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_dimension_json_template_generation(self, framework_id):
        """Test that dimension JSON template is generated correctly."""
        thinking = ThinkingModule(framework_id=framework_id)
        framework = load_framework(framework_id)

        template = thinking._generate_dimension_json_template()

        # Check that all framework dimensions are in the template
        for dim in framework["dimensions"]:
            assert dim["id"] in template

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_reasoning_fields_generation(self, framework_id):
        """Test that reasoning fields are generated for all dimensions."""
        thinking = ThinkingModule(framework_id=framework_id)
        framework = load_framework(framework_id)

        fields = thinking._generate_reasoning_fields()

        # Check that all framework dimensions have reasoning fields
        for dim in framework["dimensions"]:
            assert dim["id"] in fields

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_dimension_descriptions_formatting(self, framework_id):
        """Test that dimension descriptions are formatted correctly."""
        thinking = ThinkingModule(framework_id=framework_id)
        framework = load_framework(framework_id)

        descriptions = thinking._format_dimension_descriptions()

        # Check that all framework dimensions are described
        for dim in framework["dimensions"]:
            assert dim["name_en"] in descriptions or dim["id"] in descriptions

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_fallback_result_has_all_dimensions(self, framework_id):
        """Test that fallback result includes all framework dimensions."""
        thinking = ThinkingModule(framework_id=framework_id)
        framework = load_framework(framework_id)

        # Create system1_scores with all dimensions
        system1_scores = {dim["id"]: 7.5 for dim in framework["dimensions"]}

        fallback = thinking._create_fallback_result(
            waypoint_id=1,
            trigger_reason=TriggerReason.VISUAL_CHANGE,
            system1_scores=system1_scores,
            error="Test error"
        )

        # Check that fallback has all dimensions
        assert len(fallback.revised_scores) == len(framework["dimensions"])
        assert len(fallback.score_adjustments) == len(framework["dimensions"])
        assert len(fallback.revision_reasoning) == len(framework["dimensions"])

        for dim in framework["dimensions"]:
            assert dim["id"] in fallback.revised_scores
            assert dim["id"] in fallback.score_adjustments
            assert dim["id"] in fallback.revision_reasoning
            assert fallback.score_adjustments[dim["id"]] == 0.0


class TestRouteVisualizerFrameworks:
    """Test RouteVisualizer with different frameworks."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_visualizer_initializes(self, framework_id):
        """Test that RouteVisualizer initializes with framework."""
        viz = RouteVisualizer(framework_id=framework_id)

        assert viz.framework_id == framework_id
        assert viz.framework is not None
        assert viz.dimension_config is not None
        assert len(viz.dimension_config) > 0

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_dimension_config_has_all_dimensions(self, framework_id):
        """Test that dimension config includes all framework dimensions."""
        viz = RouteVisualizer(framework_id=framework_id)
        framework = load_framework(framework_id)

        # Check that all dimensions have color and label
        for dim in framework["dimensions"]:
            assert dim["id"] in viz.dimension_config
            assert "color" in viz.dimension_config[dim["id"]]
            assert "label" in viz.dimension_config[dim["id"]]

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_build_dim_config_uses_framework_dimensions(self, framework_id):
        """Test that _build_dim_config uses framework dimensions."""
        viz = RouteVisualizer(framework_id=framework_id)
        framework = load_framework(framework_id)

        dim_keys = [dim["id"] for dim in framework["dimensions"]]
        config = viz._build_dim_config(dim_keys)

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
        fig = viz.plot_scores_with_trends(
            scores=scores,
            waypoint_ids=waypoint_ids,
            title=f"Test {framework_id}",
            save_path=None
        )

        assert fig is not None


class TestWalkingAgentFrameworks:
    """Test WalkingAgent with different frameworks."""

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_agent_initializes_with_framework(self, framework_id):
        """Test that WalkingAgent initializes with framework."""
        personality = get_preset("balanced", framework_id)

        agent = WalkingAgent(
            agent_id="test_agent",
            personality=personality,
            framework_id=framework_id
        )

        assert agent.framework_id == framework_id
        assert agent.personality is not None

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_thinking_module_uses_framework(self, framework_id):
        """Test that WalkingAgent's ThinkingModule uses correct framework."""
        personality = get_preset("balanced", framework_id)

        agent = WalkingAgent(
            agent_id="test_agent",
            personality=personality,
            framework_id=framework_id
        )

        # Access thinking_module to trigger lazy loading
        thinking = agent.thinking_module

        assert thinking.framework_id == framework_id
        assert thinking.framework is not None

        framework = load_framework(framework_id)
        assert len(thinking.dimension_ids) == len(framework["dimensions"])

    @pytest.mark.parametrize("framework_id", FRAMEWORKS)
    def test_agent_from_preset(self, framework_id):
        """Test creating agent from preset with framework."""
        agent = WalkingAgent.from_preset(
            preset_name="balanced",
            framework_id=framework_id,
            agent_id=f"test_{framework_id}"
        )

        assert agent.framework_id == framework_id
        assert agent.personality is not None
        assert agent.thinking_module.framework_id == framework_id


class TestFrameworkSwitching:
    """Test switching between frameworks."""

    def test_different_agents_different_frameworks(self):
        """Test that different agents can use different frameworks simultaneously."""
        agent1 = WalkingAgent.from_preset("balanced", "sagai_2025", "agent1")
        agent2 = WalkingAgent.from_preset("balanced", "streetagent_5d", "agent2")

        assert agent1.framework_id == "sagai_2025"
        assert agent2.framework_id == "streetagent_5d"

        assert len(agent1.thinking_module.dimension_ids) == 4  # sagai_2025 has 4 dimensions
        assert len(agent2.thinking_module.dimension_ids) == 5  # streetagent_5d has 5 dimensions

    def test_visualizers_with_different_frameworks(self):
        """Test that multiple visualizers can use different frameworks."""
        viz1 = RouteVisualizer(framework_id="sagai_2025")
        viz2 = RouteVisualizer(framework_id="phenomenology_3d")

        assert viz1.framework_id == "sagai_2025"
        assert viz2.framework_id == "phenomenology_3d"

        assert len(viz1.dimension_config) == 4  # sagai_2025 has 4 dimensions
        assert len(viz2.dimension_config) == 3  # phenomenology_3d has 3 dimensions


class TestBackwardCompatibility:
    """Test that default framework is streetagent_5d."""

    def test_thinking_module_defaults_to_streetagent_5d(self):
        """Test that ThinkingModule defaults to streetagent_5d."""
        thinking = ThinkingModule()
        assert thinking.framework_id == "streetagent_5d"

    def test_visualizer_defaults_to_streetagent_5d(self):
        """Test that RouteVisualizer defaults to streetagent_5d."""
        viz = RouteVisualizer()
        assert viz.framework_id == "streetagent_5d"

    def test_agent_defaults_to_streetagent_5d(self):
        """Test that WalkingAgent defaults to streetagent_5d when not specified."""
        personality = get_preset("balanced", "streetagent_5d")
        agent = WalkingAgent(agent_id="test", personality=personality)
        assert agent.framework_id == "streetagent_5d"


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
