"""
Test suite for CityWalkAgent capabilities and run loop.

Tests cover:
- Agent initialization and configuration
- Memory capability
- Full run loop integration
- State management
- Cache functionality
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.agent.orchestrator import CityWalkAgent
from src.agent.base import AgentState, AgentMetadata
from src.agent.config import AgentPersonality, get_preset
from src.agent.memory import LongTermMemory


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_route_data() -> Dict[str, Any]:
    """Create mock route data for testing."""
    # Create mock waypoints
    mock_waypoint = Mock()
    mock_waypoint.lat = 37.7749
    mock_waypoint.lon = -122.4194
    mock_waypoint.image_path = "/path/to/image.jpg"

    waypoints = [mock_waypoint] * 5

    return {
        "route_id": "test_route_123",
        "route": Mock(
            route_id="test_route_123",
            waypoints=waypoints,
            total_distance=1000.0,
            interval_meters=50,
        ),
        "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        "waypoints": waypoints,
        "distance": 1000.0,
        "interval": 50,
        "from_cache": False,
    }


@pytest.fixture
def mock_observation() -> Dict[str, Any]:
    """Create mock observation data."""
    return {
        "route_info": {
            "route_id": "test_route_123",
            "num_images": 5,
        },
        "dimension_scores": {
            "safety": [7.5, 8.0, 7.8, 7.9, 8.1],
            "comfort": [6.5, 7.0, 6.8, 7.2, 7.5],
        },
        "dimension_stats": {
            "safety": {"mean": 7.86, "std": 0.2, "min": 7.5, "max": 8.1},
            "comfort": {"mean": 7.0, "std": 0.35, "min": 6.5, "max": 7.5},
        },
        "raw_evaluations": [
            {"image_id": "waypoint_0", "dimension_id": "safety", "score": 7.5},
            {"image_id": "waypoint_1", "dimension_id": "safety", "score": 8.0},
        ],
        "framework_id": "sagai_2025",
        "observation_timestamp": "2025-10-29T12:00:00",
    }



@pytest.fixture
def test_personality() -> AgentPersonality:
    """Create a test personality configuration."""
    return AgentPersonality(
        name="Test Safety",
        description="Test safety-focused personality",
        decision_thresholds={
            "min_overall_score": 6.5,
            "max_volatility": 2.0,
            "max_barriers": 2,
        },
        explanation_style="safety",
        personality_id="test_safety",
    )


@pytest.fixture
def test_agent(test_personality) -> CityWalkAgent:
    """Create a test CityWalkAgent instance."""
    return CityWalkAgent(
        agent_id="test_agent",
        personality=test_personality,
        framework_id="sagai_2025",
        enable_memory=True,
    )


# ============================================================================
# Agent Initialization Tests
# ============================================================================


class TestAgentInitialization:
    """Test agent initialization and configuration."""

    def test_agent_creation_with_personality(self, test_personality):
        """Test creating agent with custom personality."""
        agent = CityWalkAgent(
            agent_id="test_agent_1",
            personality=test_personality,
            framework_id="sagai_2025",
        )

        assert agent.metadata.agent_id == "test_agent_1"
        assert agent.metadata.name == "Test Safety"
        assert agent.personality == test_personality
        assert agent.framework_id == "sagai_2025"

    def test_agent_creation_from_preset(self):
        """Test creating agent from preset personality."""
        agent = CityWalkAgent.from_preset("homebuyer", "sagai_2025")

        assert "homebuyer" in agent.metadata.agent_id
        assert agent.personality.name == "Prospective Homebuyer"
        assert agent.framework_id == "sagai_2025"

    def test_agent_state_initialized(self, test_agent):
        """Test that agent state is properly initialized."""
        assert isinstance(test_agent.state, AgentState)
        assert test_agent.state.current_location is None
        assert test_agent.state.destination is None
        assert test_agent.state.waypoint_index == 0
        assert test_agent.state.total_waypoints == 0
        assert test_agent.state.memory_count == 0
        assert isinstance(test_agent.state.preferences, dict)



class TestMemoryCapability:
    """Test LongTermMemory functionality."""

    def test_memory_initialization(self, test_agent):
        """Test memory capability initialization."""
        memory = test_agent.memory

        assert memory is not None
        assert isinstance(memory, LongTermMemory)
        assert memory.agent_id == test_agent.metadata.agent_id

    def test_memory_store_and_retrieve(self, test_agent):
        """Test storing and retrieving experiences."""
        memory = test_agent.memory

        # Store an experience
        experience = {
            "route_id": "test_route_123",
            "start": (37.7749, -122.4194),
            "end": (37.7849, -122.4094),
            "perception": {"test": "data"},
            "decision": {"recommendation": "accept"},
            "result": {"action_type": "approve"},
        }

        memory.store(experience)

        # Retrieve experiences
        experiences = memory.retrieve(route_id="test_route_123", limit=10)

        assert len(experiences) > 0
        assert experiences[0]["route_id"] == "test_route_123"

    def test_memory_statistics(self, test_agent):
        """Test memory statistics."""
        memory = test_agent.memory
        stats = memory.get_statistics()

        assert "agent_id" in stats
        assert "total_experiences" in stats
        assert "unique_routes" in stats
        assert stats["agent_id"] == test_agent.metadata.agent_id



# ============================================================================
# Agent Run Loop Tests
# ============================================================================


class TestAgentRunLoop:
    """Test the complete agent run loop."""

    @patch.object(CityWalkAgent, "_get_route_data")
    @patch.object(CityWalkAgent, "perceive")
    @patch.object(CityWalkAgent, "reason")
    @patch.object(CityWalkAgent, "act")
    @patch.object(CityWalkAgent, "remember")
    def test_run_full_loop(
        self,
        mock_remember,
        mock_act,
        mock_reason,
        mock_perceive,
        mock_get_route,
        test_agent,
        mock_route_data,
        mock_observation,
    ):
        """Test complete run loop execution."""
        # Setup mocks
        mock_get_route.return_value = mock_route_data
        mock_perceive.return_value = mock_observation
        mock_reason.return_value = {
            "recommendation": "accept",
            "confidence": 0.85,
            "explanation": "Safe route",
        }
        mock_act.return_value = {
            "action_type": "approve",
            "message": "✅ Safe route",
        }

        # Run agent
        start = (37.7749, -122.4194)
        end = (37.7849, -122.4094)
        result = test_agent.run(start=start, end=end)

        # Verify all methods called
        mock_get_route.assert_called_once()
        mock_perceive.assert_called_once()
        mock_reason.assert_called_once()
        mock_act.assert_called_once()
        mock_remember.assert_called_once()

        # Check result structure
        assert "route_id" in result
        assert "perception" in result
        assert "decision" in result
        assert "result" in result
        assert "state" in result
        assert "metadata" in result

    @patch.object(CityWalkAgent, "_get_route_data")
    @patch.object(CityWalkAgent, "perceive")
    @patch.object(CityWalkAgent, "reason")
    @patch.object(CityWalkAgent, "act")
    def test_run_updates_state(
        self,
        mock_act,
        mock_reason,
        mock_perceive,
        mock_get_route,
        test_agent,
        mock_route_data,
    ):
        """Test that run updates agent state correctly."""
        # Setup mocks
        mock_get_route.return_value = mock_route_data
        mock_perceive.return_value = {"route_info": {"route_id": "test"}}
        mock_reason.return_value = {"recommendation": "accept"}
        mock_act.return_value = {"action_type": "approve"}

        start = (37.7749, -122.4194)
        end = (37.7849, -122.4094)

        # Initial state
        assert test_agent.state.current_location is None
        assert test_agent.state.destination is None

        # Run agent
        result = test_agent.run(start=start, end=end)

        # Check state updated
        assert test_agent.state.current_location == start
        assert test_agent.state.destination == end
        assert test_agent.state.total_waypoints == len(mock_route_data["waypoints"])
        assert "test_route_123" in test_agent.state.evaluated_routes

        # Check state in result
        assert result["state"]["current_location"] == start
        assert result["state"]["destination"] == end
        assert result["state"]["total_waypoints"] == len(mock_route_data["waypoints"])

    @patch.object(CityWalkAgent, "_get_route_data")
    def test_run_with_route_id(self, mock_get_route, test_agent, mock_route_data):
        """Test running with explicit route_id."""
        mock_get_route.return_value = mock_route_data

        # Mock other methods to avoid full execution
        with patch.multiple(
            test_agent,
            perceive=Mock(return_value={}),
            reason=Mock(return_value={"recommendation": "accept"}),
            act=Mock(return_value={"action_type": "approve"}),
        ):
            start = (37.7749, -122.4194)
            end = (37.7849, -122.4094)

            result = test_agent.run(start=start, end=end, route_id="custom_route_123")

            # Verify route_id passed through
            mock_get_route.assert_called_once()
            call_kwargs = mock_get_route.call_args[1]
            assert call_kwargs["route_id"] == "custom_route_123"


# ============================================================================
# Integration Tests
# ============================================================================


class TestAgentIntegration:
    """Integration tests for complete agent functionality."""

    def test_agent_from_preset_to_run(self):
        """Test creating agent from preset and verifying it can run (mocked)."""
        agent = CityWalkAgent.from_preset("homebuyer", "sagai_2025")

        # Verify agent created correctly
        assert agent is not None
        assert agent.personality.name == "Prospective Homebuyer"
        assert agent.framework_id == "sagai_2025"

        # Verify memory accessible
        assert agent.memory is not None

    def test_multiple_runs_update_state(self, test_agent):
        """Test that multiple runs correctly update state."""
        # Mock all the methods
        with patch.multiple(
            test_agent,
            _get_route_data=Mock(
                return_value={
                    "route_id": "route_1",
                    "route": Mock(waypoints=[Mock()] * 5),
                    "waypoints": [Mock()] * 5,
                    "image_paths": [],
                    "distance": 1000,
                    "interval": 50,
                }
            ),
            perceive=Mock(return_value={}),
            reason=Mock(return_value={"recommendation": "accept"}),
            act=Mock(return_value={"action_type": "approve"}),
        ):
            # First run
            test_agent.run(start=(37.7749, -122.4194), end=(37.7849, -122.4094))
            assert len(test_agent.state.evaluated_routes) == 1
            first_memory_count = test_agent.state.memory_count

            # Second run (different route)
            test_agent._get_route_data.return_value["route_id"] = "route_2"
            test_agent.run(start=(37.7849, -122.4094), end=(37.7949, -122.3994))

            assert len(test_agent.state.evaluated_routes) == 2
            assert test_agent.state.memory_count > first_memory_count


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
