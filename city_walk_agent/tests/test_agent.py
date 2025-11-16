"""
Test suite for WalkingAgent capabilities and run loop.

Tests cover:
- Agent initialization and configuration
- Individual capabilities (Observer, Analyzer, Thinker, Memory)
- Cognitive pipeline (perceive, reason, act, remember)
- Full run loop integration
- State management
- Cache functionality
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from src.agent.walking_agent import WalkingAgent
from src.agent.base import AgentState, AgentMetadata
from src.agent.config import AgentPersonality, get_preset
from src.agent.capabilities import (
    ObservationCapability,
    AnalysisCapability,
    ThinkingCapability,
    LongTermMemory,
)


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
def mock_analysis() -> Dict[str, Any]:
    """Create mock sequential analysis data."""
    return {
        "volatility": 1.5,
        "barriers": [],
        "pattern_type": "stable_high",
        "transition_analysis": {
            "positive_transitions": 3,
            "negative_transitions": 1,
        },
        "dimension_volatilities": {
            "safety": 0.2,
            "comfort": 0.35,
        },
        "sequential_score": 7.8,
        "aggregate_score": 7.9,
        "key_insight": "Consistently safe route",
        "recommendation": "Route acceptable",
    }


@pytest.fixture
def test_personality() -> AgentPersonality:
    """Create a test personality configuration."""
    return AgentPersonality(
        name="Test Safety",
        description="Test safety-focused personality",
        dimension_weights={
            "safety": 2.0,
            "comfort": 1.2,
            "interest": 0.8,
            "aesthetics": 0.6,
        },
        decision_thresholds={
            "min_overall_score": 6.5,
            "max_volatility": 2.0,
            "max_barriers": 2,
        },
        explanation_style="safety",
    )


@pytest.fixture
def test_agent(test_personality) -> WalkingAgent:
    """Create a test WalkingAgent instance."""
    return WalkingAgent(
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
        agent = WalkingAgent(
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
        agent = WalkingAgent.from_preset("safety", "sagai_2025")

        assert "safety" in agent.metadata.agent_id.lower()
        assert agent.personality.name == "Safety Guardian"
        assert agent.framework_id == "sagai_2025"

    def test_agent_state_initialized(self, test_agent):
        """Test that agent state is properly initialized."""
        assert isinstance(test_agent.state, AgentState)
        assert test_agent.state.current_location is None
        assert test_agent.state.destination is None
        assert test_agent.state.waypoint_index == 0
        assert test_agent.state.total_waypoints == 0
        assert test_agent.state.memory_count == 0
        assert len(test_agent.state.preferences) > 0

    def test_capabilities_lazy_loaded(self, test_agent):
        """Test that capabilities are lazy-loaded."""
        # Before access, capabilities should be None
        assert test_agent._observer is None
        assert test_agent._analyzer is None
        assert test_agent._thinker is None

        # After access, capabilities should be initialized
        observer = test_agent.observer
        assert test_agent._observer is not None
        assert isinstance(test_agent._observer, ObservationCapability)

        analyzer = test_agent.analyzer
        assert test_agent._analyzer is not None
        assert isinstance(test_agent._analyzer, AnalysisCapability)

        thinker = test_agent.thinker
        assert test_agent._thinker is not None
        assert isinstance(test_agent._thinker, ThinkingCapability)


# ============================================================================
# Capability Tests
# ============================================================================


class TestObservationCapability:
    """Test ObservationCapability functionality."""

    def test_observer_initialization(self, test_agent):
        """Test observer capability initialization."""
        observer = test_agent.observer

        assert observer is not None
        assert observer.framework_id == "sagai_2025"
        assert hasattr(observer, "observe_route")

    @patch.object(ObservationCapability, "observe_route")
    def test_observer_observe_route(self, mock_observe, test_agent, mock_route_data):
        """Test observer observes route correctly."""
        # Setup mock return value
        mock_observe.return_value = {
            "route_info": {"route_id": "test_route_123"},
            "dimension_scores": {},
            "dimension_stats": {},
        }

        # Call observe_route
        observation = test_agent.observer.observe_route(mock_route_data)

        # Verify
        mock_observe.assert_called_once_with(mock_route_data)
        assert "route_info" in observation
        assert observation["route_info"]["route_id"] == "test_route_123"


class TestAnalysisCapability:
    """Test AnalysisCapability functionality."""

    def test_analyzer_initialization(self, test_agent):
        """Test analyzer capability initialization."""
        analyzer = test_agent.analyzer

        assert analyzer is not None
        assert isinstance(analyzer, AnalysisCapability)
        assert hasattr(analyzer, "analyze_patterns")

    @patch.object(AnalysisCapability, "analyze_patterns")
    def test_analyzer_analyze_patterns(
        self, mock_analyze, test_agent, mock_observation, mock_route_data
    ):
        """Test analyzer analyzes patterns correctly."""
        # Setup mock return value
        mock_analyze.return_value = {
            "volatility": 1.5,
            "barriers": [],
            "pattern_type": "stable_high",
        }

        # Call analyze_patterns
        analysis = test_agent.analyzer.analyze_patterns(
            observation=mock_observation,
            route_info={"route": mock_route_data["route"], "route_id": "test_route_123"},
        )

        # Verify
        mock_analyze.assert_called_once()
        assert "volatility" in analysis
        assert "pattern_type" in analysis


class TestThinkingCapability:
    """Test ThinkingCapability functionality."""

    def test_thinker_initialization(self, test_agent):
        """Test thinker capability initialization."""
        thinker = test_agent.thinker

        assert thinker is not None
        assert isinstance(thinker, ThinkingCapability)
        assert hasattr(thinker, "think")

    @patch.object(ThinkingCapability, "think")
    def test_thinker_make_decision(
        self, mock_think, test_agent, mock_observation, mock_analysis
    ):
        """Test thinker makes decisions correctly."""
        # Setup mock return value
        mock_think.return_value = {
            "recommendation": "accept",
            "confidence": 0.85,
            "explanation": "Safe and comfortable route",
            "weighted_score": 7.8,
        }

        # Call think
        decision = test_agent.thinker.think(
            observation=mock_observation,
            sequential_analysis=mock_analysis,
            personality_weights=test_agent.personality.dimension_weights,
            decision_thresholds=test_agent.personality.decision_thresholds,
            explanation_style=test_agent.personality.explanation_style,
        )

        # Verify
        mock_think.assert_called_once()
        assert "recommendation" in decision
        assert decision["recommendation"] in ["accept", "reject"]
        assert "confidence" in decision
        assert 0.0 <= decision["confidence"] <= 1.0


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

    def test_memory_placeholder_methods(self, test_agent):
        """Test memory placeholder methods return empty results."""
        memory = test_agent.memory

        # Test placeholder methods
        similar = memory.find_similar_routes({"distance": 1000})
        assert isinstance(similar, list)
        assert len(similar) == 0

        preferences = memory.analyze_preferences()
        assert isinstance(preferences, dict)
        assert len(preferences) == 0

        metrics = memory.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) == 0


# ============================================================================
# Cognitive Pipeline Tests
# ============================================================================


class TestCognitivePipeline:
    """Test the full cognitive pipeline (perceive, reason, act, remember)."""

    @patch.object(WalkingAgent, "_get_route_data")
    @patch.object(ObservationCapability, "observe_route")
    def test_perceive_method(
        self, mock_observe, mock_get_route, test_agent, mock_route_data, mock_observation
    ):
        """Test perceive method transforms route data to observations."""
        mock_observe.return_value = mock_observation

        # Call perceive
        observation = test_agent.perceive(mock_route_data)

        # Verify
        mock_observe.assert_called_once()
        assert "route_info" in observation
        assert "dimension_scores" in observation
        assert "dimension_stats" in observation

        # Check that position tracking updated state
        assert test_agent.state.waypoint_index >= 0

    @patch.object(AnalysisCapability, "analyze_patterns")
    @patch.object(ThinkingCapability, "think")
    def test_reason_method(
        self,
        mock_think,
        mock_analyze,
        test_agent,
        mock_observation,
        mock_analysis,
    ):
        """Test reason method makes decisions from observations."""
        # Setup mocks
        mock_analyze.return_value = mock_analysis
        mock_think.return_value = {
            "recommendation": "accept",
            "confidence": 0.85,
            "explanation": "Safe route",
            "weighted_score": 7.8,
        }

        # Call reason
        decision = test_agent.reason(mock_observation)

        # Verify
        mock_analyze.assert_called_once()
        mock_think.assert_called_once()

        assert "recommendation" in decision
        assert "confidence" in decision
        assert "sequential_analysis" in decision
        assert decision["sequential_analysis"] == mock_analysis

    def test_act_method(self, test_agent):
        """Test act method formats decisions into actions."""
        # Test with accept recommendation
        decision_accept = {
            "recommendation": "accept",
            "confidence": 0.85,
            "explanation": "Safe and comfortable route",
            "concerns": [],
            "highlights": ["High safety scores", "Low volatility"],
            "weighted_score": 7.8,
        }

        result = test_agent.act(decision_accept)

        assert result["action_type"] == "approve"
        assert "✅" in result["message"]
        assert result["recommendation"] == "accept"
        assert result["confidence"] == 0.85

        # Test with reject recommendation
        decision_reject = {
            "recommendation": "reject",
            "confidence": 0.75,
            "explanation": "Safety concerns detected",
            "concerns": ["Low safety score in segment 3"],
            "highlights": [],
            "weighted_score": 5.2,
        }

        result = test_agent.act(decision_reject)

        assert result["action_type"] == "reject"
        assert "⚠️" in result["message"]
        assert result["recommendation"] == "reject"

    @patch.object(LongTermMemory, "store")
    def test_remember_method(self, mock_store, test_agent):
        """Test remember method stores experiences."""
        experience = {
            "route_id": "test_route_123",
            "start": (37.7749, -122.4194),
            "end": (37.7849, -122.4094),
            "perception": {"test": "data"},
            "decision": {"recommendation": "accept"},
            "result": {"action_type": "approve"},
        }

        # Call remember
        test_agent.remember(experience)

        # Verify
        mock_store.assert_called_once_with(experience)


# ============================================================================
# Agent Run Loop Tests
# ============================================================================


class TestAgentRunLoop:
    """Test the complete agent run loop."""

    @patch.object(WalkingAgent, "_get_route_data")
    @patch.object(WalkingAgent, "perceive")
    @patch.object(WalkingAgent, "reason")
    @patch.object(WalkingAgent, "act")
    @patch.object(WalkingAgent, "remember")
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

    @patch.object(WalkingAgent, "_get_route_data")
    @patch.object(WalkingAgent, "perceive")
    @patch.object(WalkingAgent, "reason")
    @patch.object(WalkingAgent, "act")
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

    @patch.object(WalkingAgent, "_get_route_data")
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
# Cache Functionality Tests
# ============================================================================


class TestCacheFunctionality:
    """Test agent cache functionality."""

    @patch("src.pipeline.WalkingAgentPipeline.load_cached_route")
    def test_cache_hit(self, mock_load_cache, test_agent, mock_route_data):
        """Test cache hit returns cached data."""
        # Setup cache hit
        mock_load_cache.return_value = mock_route_data

        # Call _get_route_data
        start = (37.7749, -122.4194)
        end = (37.7849, -122.4094)
        result = test_agent._get_route_data(start, end)

        # Verify cache was checked
        mock_load_cache.assert_called_once()

        # Verify from_cache flag set
        assert result["from_cache"] is True
        assert result["route_id"] == "test_route_123"

    @patch("src.pipeline.WalkingAgentPipeline.load_cached_route")
    def test_cache_miss(self, mock_load_cache, test_agent):
        """Test cache miss generates new route."""
        # Setup cache miss
        mock_load_cache.return_value = None

        # Setup route generation
        mock_route = Mock()
        mock_route.route_id = "new_route_456"
        mock_route.waypoints = [Mock()] * 5
        mock_route.total_distance = 1000.0

        # Create a fully mocked route_generator
        mock_route_generator = MagicMock()
        mock_route_generator.create_google_maps_route.return_value = mock_route

        # Mock image directory
        mock_dir = Mock()
        mock_dir.glob.return_value = [Path("image1.jpg")]
        mock_route_generator.get_route_dir.return_value = mock_dir

        # Replace the route_generator
        test_agent.pipeline._route_generator = mock_route_generator

        start = (37.7749, -122.4194)
        end = (37.7849, -122.4094)
        result = test_agent._get_route_data(start, end)

        # Verify cache was checked
        mock_load_cache.assert_called_once()

        # Verify from_cache flag set correctly
        assert result["from_cache"] is False
        assert result["route_id"] == "new_route_456"

    def test_deterministic_route_id(self, test_agent):
        """Test that same coordinates generate same route_id."""
        start = (37.7749, -122.4194)
        end = (37.7849, -122.4094)
        interval = 50

        # Generate expected route_id format
        expected_route_id = f"route_{start[0]:.6f}_{start[1]:.6f}_{end[0]:.6f}_{end[1]:.6f}_{interval}"

        # Mock load_cached_route to return None (cache miss)
        with patch("src.pipeline.WalkingAgentPipeline.load_cached_route") as mock_load:
            mock_load.return_value = None

            # Setup mock route
            mock_route = Mock()
            mock_route.route_id = "generated_id"
            mock_route.waypoints = []
            mock_route.total_distance = 1000.0

            # Create a fully mocked route_generator
            mock_route_generator = MagicMock()
            mock_route_generator.create_google_maps_route.return_value = mock_route

            # Mock image directory with existing images (to skip ImageCollector)
            mock_dir = Mock()
            mock_dir.glob.return_value = [Path("existing_image1.jpg"), Path("existing_image2.jpg")]
            mock_route_generator.get_route_dir.return_value = mock_dir

            # Replace the route_generator
            test_agent.pipeline._route_generator = mock_route_generator

            # First call
            test_agent._get_route_data(start, end, interval=interval)

            # Check that cache was called with deterministic route_id
            first_call_route_id = mock_load.call_args[0][0]

            # Second call with same coordinates
            test_agent._get_route_data(start, end, interval=interval)

            # Check that cache was called with same route_id
            second_call_route_id = mock_load.call_args[0][0]

            # Both should generate the same deterministic route_id
            assert first_call_route_id == second_call_route_id
            assert first_call_route_id == expected_route_id


# ============================================================================
# Integration Tests
# ============================================================================


class TestAgentIntegration:
    """Integration tests for complete agent functionality."""

    def test_agent_from_preset_to_run(self):
        """Test creating agent from preset and verifying it can run (mocked)."""
        agent = WalkingAgent.from_preset("safety", "sagai_2025")

        # Verify agent created correctly
        assert agent is not None
        assert agent.personality.name == "Safety Guardian"
        assert agent.framework_id == "sagai_2025"

        # Verify capabilities accessible
        assert agent.observer is not None
        assert agent.analyzer is not None
        assert agent.thinker is not None
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
