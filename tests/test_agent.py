"""Tests for WalkingAgent and its capabilities."""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.walking_agent import WalkingAgent
from src.agent.config import AgentPersonality, DimensionWeights
from src.agent.capabilities import (
    ObservationCapability,
    AnalysisCapability,
    ThinkingCapability,
    AgentMemory,
    ActionCapability,
)


@pytest.fixture
def test_agent():
    """Create a test agent with default personality."""
    personality = AgentPersonality(
        name="test_agent",
        description="Test agent for unit tests",
        dimension_weights=DimensionWeights(
            safety=0.4,
            comfort=0.3,
            efficiency=0.2,
            scenic=0.1,
        ),
        decision_thresholds={
            "accept_threshold": 6.5,
            "reject_threshold": 5.0,
        },
        explanation_style="balanced",
    )

    agent = WalkingAgent(
        agent_id="test_agent_001",
        personality=personality,
        framework_id="sagai_2025",
        enable_memory=False,
    )

    return agent


class TestActionCapability:
    """Test ActionCapability functionality."""

    def test_actor_initialization(self, test_agent):
        """Test action capability initialization."""
        actor = test_agent.actor

        assert actor is not None
        assert isinstance(actor, ActionCapability)
        assert hasattr(actor, "execute")

    def test_actor_execute_accept(self, test_agent):
        """Test action execution for accept recommendation."""
        decision = {
            "recommendation": "accept",
            "confidence": 0.85,
            "explanation": "Safe and comfortable route",
            "concerns": [],
            "highlights": ["High safety scores"],
            "weighted_score": 7.8,
        }

        result = test_agent.actor.execute(decision)

        assert result["action_type"] == "approve"
        assert "✅" in result["message"]
        assert result["recommendation"] == "accept"
        assert result["confidence"] == 0.85

    def test_actor_execute_reject(self, test_agent):
        """Test action execution for reject recommendation."""
        decision = {
            "recommendation": "reject",
            "confidence": 0.70,
            "explanation": "Safety concerns detected",
            "concerns": ["Low safety score"],
            "highlights": [],
            "weighted_score": 5.2,
        }

        result = test_agent.actor.execute(decision)

        assert result["action_type"] == "reject"
        assert "⚠️" in result["message"]
        assert result["recommendation"] == "reject"


class TestWalkingAgent:
    """Test WalkingAgent methods."""

    def test_act_method(self, test_agent):
        """Test act method uses ActionCapability."""
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


class TestCapabilityInitialization:
    """Test lazy loading of all capabilities."""

    def test_observer_lazy_loading(self, test_agent):
        """Test observer capability is lazy loaded."""
        # Access observer
        observer = test_agent.observer
        assert observer is not None
        assert isinstance(observer, ObservationCapability)

        # Verify same instance is returned
        assert test_agent.observer is observer

    def test_analyzer_lazy_loading(self, test_agent):
        """Test analyzer capability is lazy loaded."""
        analyzer = test_agent.analyzer
        assert analyzer is not None
        assert isinstance(analyzer, AnalysisCapability)

        # Verify same instance is returned
        assert test_agent.analyzer is analyzer

    def test_thinker_lazy_loading(self, test_agent):
        """Test thinker capability is lazy loaded."""
        thinker = test_agent.thinker
        assert thinker is not None
        assert isinstance(thinker, ThinkingCapability)

        # Verify same instance is returned
        assert test_agent.thinker is thinker

    def test_actor_lazy_loading(self, test_agent):
        """Test actor capability is lazy loaded."""
        actor = test_agent.actor
        assert actor is not None
        assert isinstance(actor, ActionCapability)

        # Verify same instance is returned
        assert test_agent.actor is actor
