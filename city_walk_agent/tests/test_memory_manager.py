"""Tests for MemoryManager - Central memory coordination hub.

Tests cover:
1. Initialization and lazy loading
2. Agent attributes configuration
3. Attention gate filtering
4. System 2 trigger logic
5. Context preparation
6. Waypoint processing pipeline
7. Route consolidation
8. Integration with CityWalkAgent
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Dict, Tuple

from src.agent.memory.memory_manager import MemoryManager
from src.agent.config import AgentPersonality


# ============================================================================
# Test Fixtures and Mock Data
# ============================================================================

@dataclass
class MockWaypointAnalysis:
    """Mock WaypointAnalysis for testing."""
    waypoint_id: int
    image_path: Path
    scores: Dict[str, float]
    reasoning: Dict[str, str]
    timestamp: str
    gps: Tuple[float, float]
    heading: float
    visual_change_detected: bool
    phash_distance: float


@pytest.fixture
def mock_personality():
    """Create a mock AgentPersonality for testing."""
    return AgentPersonality(
        name="Safety-Focused",
        description="Prioritizes safety above all",
        decision_thresholds={
            "min_overall_score": 6.5,
            "max_volatility": 2.0,
            "max_barriers": 2
        },
        explanation_style="safety",
        personality_id="test_safety",
    )


@pytest.fixture
def memory_manager(tmp_path):
    """Create a MemoryManager instance for testing."""
    return MemoryManager(
        agent_id="test_agent",
        storage_dir=tmp_path
    )


@pytest.fixture
def configured_memory_manager(memory_manager, mock_personality):
    """Create a configured MemoryManager with agent attributes."""
    memory_manager.set_agent_attributes(
        personality=mock_personality,
        profile={"name": "TestAgent", "goal": "Test routes"},
        status={"mode": "active", "energy": 1.0}
    )
    return memory_manager


def create_waypoint_analysis(
    waypoint_id: int,
    scores: Dict[str, float],
    visual_change: bool = False,
    phash_distance: float = 0.0,
    gps: Tuple[float, float] = (37.7749, -122.4194)
) -> MockWaypointAnalysis:
    """Helper to create mock waypoint analysis."""
    return MockWaypointAnalysis(
        waypoint_id=waypoint_id,
        image_path=Path(f"waypoint_{waypoint_id}.jpg"),
        scores=scores,
        reasoning={k: f"Score {v}" for k, v in scores.items()},
        timestamp="2024-01-01T12:00:00",
        gps=gps,
        heading=0.0,
        visual_change_detected=visual_change,
        phash_distance=phash_distance
    )


# ============================================================================
# Test 1: Initialization
# ============================================================================

def test_memory_manager_initialization():
    """Test MemoryManager initializes correctly."""
    mm = MemoryManager(agent_id="test_agent")

    assert mm.agent_id == "test_agent"
    assert mm.personality is None
    assert mm.profile == {}
    assert mm.status == {}
    assert mm._stm is None
    assert mm._ltm is None


def test_lazy_loading_stm(memory_manager):
    """Test STM is lazy-loaded on first access."""
    assert memory_manager._stm is None

    stm = memory_manager.stm
    assert stm is not None
    assert memory_manager._stm is stm

    # Second access returns same instance
    stm2 = memory_manager.stm
    assert stm2 is stm


def test_lazy_loading_ltm(memory_manager):
    """Test LTM is lazy-loaded on first access."""
    assert memory_manager._ltm is None

    ltm = memory_manager.episodic_ltm
    assert ltm is not None
    assert memory_manager._ltm is ltm

    # Second access returns same instance
    ltm2 = memory_manager.episodic_ltm
    assert ltm2 is ltm


# ============================================================================
# Test 2: Agent Attributes Configuration
# ============================================================================

def test_set_agent_attributes(memory_manager, mock_personality):
    """Test agent attributes can be set."""
    profile = {"name": "TestAgent", "goal": "Test"}
    status = {"mode": "active"}

    memory_manager.set_agent_attributes(
        personality=mock_personality,
        profile=profile,
        status=status
    )

    assert memory_manager.personality == mock_personality
    assert memory_manager.profile == profile
    assert memory_manager.status == status


def test_personality_influences_thresholds(configured_memory_manager):
    """Test personality affects decision thresholds."""
    # Safety personality should have stricter thresholds
    delta_threshold = configured_memory_manager._get_attention_delta_threshold()
    assert delta_threshold <= 1.5  # More sensitive

    low_score_threshold = configured_memory_manager._get_low_score_threshold()
    assert low_score_threshold >= 5.5  # Higher bar


# ============================================================================
# Test 3: Attention Gate Logic
# ============================================================================

def test_attention_gate_visual_change(configured_memory_manager):
    """Test attention gate passes waypoints with visual change."""
    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0, "comfort": 7.0},
        visual_change=True,
        phash_distance=25.0
    )

    assert configured_memory_manager._passes_attention_gate(waypoint) is True


def test_attention_gate_milestone(configured_memory_manager):
    """Test attention gate passes milestone waypoints."""
    # Every 10th waypoint should pass
    waypoint = create_waypoint_analysis(
        waypoint_id=10,
        scores={"safety": 7.0, "comfort": 7.0},
        visual_change=False
    )

    assert configured_memory_manager._passes_attention_gate(waypoint) is True


def test_attention_gate_low_score(configured_memory_manager):
    """Test attention gate passes waypoints with low scores."""
    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 4.0, "comfort": 7.0},  # Low safety score
        visual_change=False
    )

    assert configured_memory_manager._passes_attention_gate(waypoint) is True


def test_attention_gate_score_anomaly(configured_memory_manager):
    """Test attention gate passes waypoints with score anomalies."""
    # Add some waypoints to STM first
    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0}
        )
        configured_memory_manager.stm.add(
            waypoint_id=wp.waypoint_id,
            scores=wp.scores,
            summary="test",
            gps=wp.gps
        )

    # Now create waypoint with significantly different score
    anomaly_waypoint = create_waypoint_analysis(
        waypoint_id=4,
        scores={"safety": 3.0, "comfort": 3.0},  # Much lower
        visual_change=False
    )

    assert configured_memory_manager._passes_attention_gate(anomaly_waypoint) is True


def test_attention_gate_filters_normal(configured_memory_manager):
    """Test attention gate filters normal waypoints."""
    # Add some waypoints to STM
    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0}
        )
        configured_memory_manager.stm.add(
            waypoint_id=wp.waypoint_id,
            scores=wp.scores,
            summary="test",
            gps=wp.gps
        )

    # Create similar waypoint
    normal_waypoint = create_waypoint_analysis(
        waypoint_id=4,
        scores={"safety": 7.1, "comfort": 7.2},  # Similar scores
        visual_change=False
    )

    # Should be filtered out
    assert configured_memory_manager._passes_attention_gate(normal_waypoint) is False


# ============================================================================
# Test 4: System 2 Integration (Triggering moved to external controller)
# ============================================================================

def test_system2_context_with_trigger(configured_memory_manager):
    """Test that triggered waypoints get System 2 context prepared."""
    from src.agent.system2.persona_reasoner import TriggerReason

    # Add waypoints to STM
    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0},
            visual_change=True
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)

    # Process waypoint with trigger flag
    trigger_waypoint = create_waypoint_analysis(
        waypoint_id=4,
        scores={"safety": 3.0, "comfort": 3.0},
        visual_change=True
    )

    context = configured_memory_manager.process_waypoint(
        trigger_waypoint,
        triggered=True,
        trigger_reason=TriggerReason.SCORE_VOLATILITY
    )

    # Should return context when triggered
    assert context is not None
    assert "waypoint_analysis" in context
    assert "trigger_reason" in context
    assert context["trigger_reason"] == TriggerReason.SCORE_VOLATILITY


def test_system2_no_context_without_trigger(configured_memory_manager):
    """Test that non-triggered waypoints don't get System 2 context."""
    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0, "comfort": 7.0},
        visual_change=True
    )

    # Process without trigger flag
    context = configured_memory_manager.process_waypoint(waypoint, triggered=False)

    # Should not return context
    assert context is None


def test_system2_result_update(configured_memory_manager):
    """Test updating STM with System 2 results."""
    from unittest.mock import Mock

    # Add waypoint to STM
    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0, "comfort": 7.0},
        visual_change=True
    )
    configured_memory_manager.process_waypoint(waypoint, triggered=False)

    # Create mock thinking result
    thinking_result = Mock()
    thinking_result.waypoint_id = 1
    thinking_result.revised_scores = {"safety": 6.5, "comfort": 7.5}
    thinking_result.score_adjustments = {"safety": -0.5, "comfort": 0.5}
    thinking_result.significance = "high"

    # Update with System 2 result
    configured_memory_manager.update_with_system2_result(1, thinking_result)

    # Verify the result was stored
    assert 1 in configured_memory_manager._system2_results


def test_system2_statistics_tracking(configured_memory_manager):
    """Test that System 2 triggers are tracked in statistics."""
    from src.agent.system2.persona_reasoner import TriggerReason

    # Process several waypoints
    for i in range(5):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0},
            visual_change=True
        )
        # Trigger every other waypoint
        configured_memory_manager.process_waypoint(
            wp,
            triggered=(i % 2 == 0),
            trigger_reason=TriggerReason.VISUAL_CHANGE
        )

    stats = configured_memory_manager.get_statistics()

    # Should have 3 triggers (waypoints 0, 2, 4)
    assert stats["system2_triggers"] == 3
    assert stats["waypoints_processed"] == 5


# ============================================================================
# Test 5: Context Preparation
# ============================================================================

def test_prepare_context_structure(configured_memory_manager):
    """Test context preparation returns correct structure."""
    from src.agent.system2.persona_reasoner import TriggerReason

    # Add some waypoints to STM
    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0}
        )
        configured_memory_manager.stm.add(
            waypoint_id=wp.waypoint_id,
            scores=wp.scores,
            summary="test",
            gps=wp.gps
        )

    # Create current waypoint
    current = create_waypoint_analysis(
        waypoint_id=4,
        scores={"safety": 5.0, "comfort": 5.0},
        visual_change=True
    )

    context = configured_memory_manager.prepare_context_for_reasoning(
        current,
        trigger_reason=TriggerReason.VISUAL_CHANGE
    )

    # Check structure (updated for new format)
    assert "waypoint_analysis" in context
    assert "recent_context" in context
    assert "ltm_patterns" in context
    assert "personality" in context
    assert "agent_profile" in context
    assert "current_status" in context
    assert "trigger_reason" in context
    assert "stm_context" in context

    # Check current waypoint (now in waypoint_analysis)
    assert context["waypoint_analysis"]["waypoint_id"] == 4
    assert context["waypoint_analysis"]["visual_change"] is True

    # Check recent context
    assert len(context["recent_context"]) == 3

    # Check personality info
    assert context["personality"]["name"] == "Safety-Focused"


def test_prepare_context_includes_personality(configured_memory_manager):
    """Test prepared context includes personality information."""
    from src.agent.system2.persona_reasoner import TriggerReason

    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0, "comfort": 7.0}
    )

    context = configured_memory_manager.prepare_context_for_reasoning(
        waypoint,
        trigger_reason=TriggerReason.VISUAL_CHANGE
    )

    personality = context["personality"]
    assert personality["name"] == "Safety-Focused"
    assert "decision_thresholds" in personality


# ============================================================================
# Test 6: Waypoint Processing Pipeline
# ============================================================================

def test_process_waypoint_filters_normal(configured_memory_manager):
    """Test process_waypoint returns None for non-triggered waypoints (STM still written)."""
    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0, "comfort": 7.0},
        visual_change=False
    )

    result = configured_memory_manager.process_waypoint(waypoint)

    assert result is None
    # STM receives ALL waypoints now regardless of attention gate
    assert configured_memory_manager.stm.get_memory_size() == 1


def test_process_waypoint_adds_to_stm(configured_memory_manager):
    """Test process_waypoint adds significant waypoints to STM."""
    waypoint = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0, "comfort": 7.0},
        visual_change=True  # Should pass attention gate
    )

    result = configured_memory_manager.process_waypoint(waypoint)

    # Should be added to STM
    assert configured_memory_manager.stm.get_memory_size() == 1


def test_process_waypoint_triggers_system2(configured_memory_manager):
    """Test process_waypoint triggers System 2 when triggered flag is set."""
    from src.agent.system2.persona_reasoner import TriggerReason

    # Add some waypoints first
    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0},
            visual_change=True
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)

    # Create waypoint and trigger System 2 externally
    trigger_waypoint = create_waypoint_analysis(
        waypoint_id=4,
        scores={"safety": 3.0, "comfort": 3.0},
        visual_change=True
    )

    result = configured_memory_manager.process_waypoint(
        trigger_waypoint,
        triggered=True,
        trigger_reason=TriggerReason.SCORE_VOLATILITY
    )

    # Should return context for System 2
    assert result is not None
    assert "waypoint_analysis" in result
    assert "personality" in result
    assert "trigger_reason" in result


def test_process_waypoint_statistics(configured_memory_manager):
    """Test process_waypoint tracks statistics correctly."""
    # Process various waypoints
    waypoints = [
        create_waypoint_analysis(1, {"safety": 7.0}, visual_change=True),
        create_waypoint_analysis(2, {"safety": 7.0}, visual_change=False),
        create_waypoint_analysis(3, {"safety": 7.0}, visual_change=True),
        create_waypoint_analysis(4, {"safety": 3.0}, visual_change=True),  # Should trigger
    ]

    for wp in waypoints:
        configured_memory_manager.process_waypoint(wp)

    stats = configured_memory_manager.get_statistics()

    assert stats["waypoints_processed"] == 4
    assert stats["attention_gate_passes"] >= 2  # At least visual changes
    assert stats["attention_gate_pass_rate"] > 0


# ============================================================================
# Test 7: Route Consolidation
# ============================================================================

def test_complete_route_clears_stm(configured_memory_manager):
    """Test complete_route clears STM after consolidation."""
    # Add waypoints to STM
    for i in range(5):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0},
            visual_change=True
        )
        configured_memory_manager.process_waypoint(wp)

    assert configured_memory_manager.stm.get_memory_size() > 0

    # Complete route
    route_data = {
        "route_id": "test_route",
        "length_km": 2.5
    }

    summary = configured_memory_manager.complete_route(
        route_data=route_data,
        reasoning_history=[]
    )

    # STM should be cleared
    assert configured_memory_manager.stm.get_memory_size() == 0

    # Should return summary
    assert summary is not None
    assert summary.route_id == "test_route"


def test_complete_route_generates_summary(configured_memory_manager):
    """Test complete_route generates route summary."""
    # Process route
    for i in range(10):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0 - i * 0.3, "comfort": 7.0},  # Declining safety
            visual_change=(i % 2 == 0)
        )
        configured_memory_manager.process_waypoint(wp)

    route_data = {
        "route_id": "declining_route",
        "length_km": 1.5
    }

    summary = configured_memory_manager.complete_route(
        route_data=route_data,
        reasoning_history=[]
    )

    assert summary.route_id == "declining_route"
    assert summary.total_waypoints >= 0
    assert summary.length_km == 1.5
    assert summary.overall_recommendation in ["recommended", "acceptable", "avoid"]


# ============================================================================
# Test 8: Integration with CityWalkAgent
# ============================================================================

def test_walking_agent_has_memory_manager():
    """Test CityWalkAgent can access memory_manager property."""
    from src.agent.orchestrator import CityWalkAgent
    from src.agent.config import get_preset

    # Create agent
    personality = get_preset("homebuyer", "sagai_2025")
    agent = CityWalkAgent(
        agent_id="test_walker",
        personality=personality
    )

    # Access memory manager
    mm = agent.memory_manager

    assert mm is not None
    assert mm.agent_id == "test_walker"
    assert mm.personality is not None
    assert mm.personality.name is not None


def test_memory_manager_persists_across_calls():
    """Test MemoryManager is reused across multiple accesses."""
    from src.agent.orchestrator import CityWalkAgent
    from src.agent.config import get_preset

    personality = get_preset("homebuyer", "sagai_2025")
    agent = CityWalkAgent(
        agent_id="test_walker",
        personality=personality
    )

    mm1 = agent.memory_manager
    mm2 = agent.memory_manager

    assert mm1 is mm2  # Same instance


# ============================================================================
# Test 9: Statistics and Introspection
# ============================================================================

def test_get_statistics(configured_memory_manager):
    """Test get_statistics returns complete information."""
    # Process some waypoints
    for i in range(10):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0},
            visual_change=(i % 3 == 0)
        )
        configured_memory_manager.process_waypoint(wp)

    stats = configured_memory_manager.get_statistics()

    assert "waypoints_processed" in stats
    assert "attention_gate_passes" in stats
    assert "system2_triggers" in stats
    assert "attention_gate_pass_rate" in stats
    assert "system2_trigger_rate" in stats
    assert "stm_size" in stats
    assert "ltm_stats" in stats

    assert stats["waypoints_processed"] == 10
    assert 0 <= stats["attention_gate_pass_rate"] <= 1.0


def test_statistics_reset_after_route_completion(configured_memory_manager):
    """Test statistics reset after route completion."""
    # Process waypoints
    for i in range(5):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0},
            visual_change=True
        )
        configured_memory_manager.process_waypoint(wp)

    # Complete route
    route_data = {"route_id": "test", "length_km": 1.0}
    configured_memory_manager.complete_route(route_data, [])

    # Statistics should be reset
    stats = configured_memory_manager.get_statistics()
    assert stats["waypoints_processed"] == 0
    assert stats["attention_gate_passes"] == 0
    assert stats["system2_triggers"] == 0


# ============================================================================
# Test 10: Within-route reasoning log (unchanged)
# ============================================================================

def test_route_reasoning_log_accumulates(memory_manager):
    """Test _route_reasoning_log accumulates System 2 results within a route."""
    class FakeResult:
        waypoint_id = 3
        significance = "high"
        avoid_recommendation = True
        interpretation = "Safety issue detected"
        key_concern = "Narrow sidewalk"
        score_change_reason = "Safety dropped"
        system1_scores = {"safety": 3.0}
        trigger_reason = None

    memory_manager.update_with_system2_result(3, FakeResult())
    assert len(memory_manager._route_reasoning_log) == 1
    entry = memory_manager._route_reasoning_log[0]
    assert entry["waypoint_id"] == 3
    assert entry["significance"] == "high"
    assert entry["avoid"] is True
    assert entry["key_concern"] == "Narrow sidewalk"


def test_route_reasoning_log_cleared_after_complete_route(configured_memory_manager):
    """Test _route_reasoning_log is reset after complete_route."""
    class FakeResult:
        significance = "medium"
        avoid_recommendation = False
        interpretation = "Normal"
        key_concern = None
        score_change_reason = None
        system1_scores = {}
        trigger_reason = None

    configured_memory_manager.update_with_system2_result(1, FakeResult())
    assert len(configured_memory_manager._route_reasoning_log) == 1

    configured_memory_manager.complete_route({"route_id": "r1", "length_km": 1.0}, [])
    assert len(configured_memory_manager._route_reasoning_log) == 0


# ============================================================================
# Test 11: STM receives ALL waypoints (no gate blocking)
# ============================================================================

def test_stm_receives_all_waypoints_regardless_of_score(configured_memory_manager):
    """STM should receive every waypoint, even those that would have failed the gate."""
    # Normal-scoring waypoints that have no visual change, no low scores, no anomaly
    # The first one won't have STM context so can't compute anomaly — add 3 similar first
    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 7.0},
            visual_change=True,
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)

    # This waypoint would have been filtered by the old gate (similar scores, no visual change)
    normal_wp = create_waypoint_analysis(
        waypoint_id=4,
        scores={"safety": 7.1, "comfort": 7.2},
        visual_change=False,
    )
    configured_memory_manager.process_waypoint(normal_wp, triggered=False)

    # STM window is 5; all 4 should be in STM
    assert configured_memory_manager.stm.get_memory_size() == 4


def test_synthesize_summary_uses_persona_reasoning(memory_manager):
    """Test _synthesize_summary extracts text from persona_reasoning."""
    wp = create_waypoint_analysis(
        waypoint_id=1,
        scores={"safety": 7.0},
    )
    # Attach persona_reasoning
    wp.persona_reasoning = {
        "safety": "Wide sidewalk with good lighting. No obstacles.",
        "lively": "Busy commercial street. Many pedestrians.",
    }
    summary = memory_manager._synthesize_summary(wp)
    assert "Wide sidewalk" in summary
    assert len(summary) <= 120


def test_synthesize_summary_fallback(memory_manager):
    """Test _synthesize_summary falls back to waypoint ID when no reasoning."""
    wp = create_waypoint_analysis(waypoint_id=5, scores={"safety": 7.0})
    summary = memory_manager._synthesize_summary(wp)
    assert "5" in summary


# ============================================================================
# Test 12: Snapshot-based route LTM
# ============================================================================

def test_route_snapshots_generated_at_s2_trigger(configured_memory_manager):
    """Test _route_snapshots has one entry after an S2 trigger."""
    from src.agent.system2.persona_reasoner import TriggerReason

    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i,
            scores={"safety": 7.0, "comfort": 6.0},
            visual_change=True,
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)

    trigger_wp = create_waypoint_analysis(
        waypoint_id=3,
        scores={"safety": 4.0, "comfort": 4.0},
        visual_change=True,
    )
    context = configured_memory_manager.process_waypoint(
        trigger_wp, triggered=True, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    assert context is not None
    assert len(configured_memory_manager._route_snapshots) == 1
    snap = configured_memory_manager._route_snapshots[0]
    assert snap["span_end"] == 3
    assert "trend" in snap
    assert "avg" in snap


def test_route_snapshots_cleared_after_complete_route(configured_memory_manager):
    """Test _route_snapshots is reset after complete_route."""
    from src.agent.system2.persona_reasoner import TriggerReason

    wp = create_waypoint_analysis(
        waypoint_id=0, scores={"safety": 5.0}, visual_change=True
    )
    configured_memory_manager.process_waypoint(
        wp, triggered=True, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    assert len(configured_memory_manager._route_snapshots) == 1

    configured_memory_manager.complete_route({"route_id": "r1", "length_km": 1.0}, [])
    assert len(configured_memory_manager._route_snapshots) == 0


def test_ltm_patterns_has_snapshots_format(configured_memory_manager):
    """Test prepare_context_for_reasoning returns ltm_patterns with snapshots key."""
    from src.agent.system2.persona_reasoner import TriggerReason

    wp = create_waypoint_analysis(
        waypoint_id=1, scores={"safety": 7.0, "comfort": 7.0}, visual_change=True
    )
    context = configured_memory_manager.prepare_context_for_reasoning(
        wp, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    ltm = context["ltm_patterns"]
    assert isinstance(ltm, dict)
    assert "snapshots" in ltm
    assert "reasoning_episodes" in ltm


# ============================================================================
# Test 13: waypoints_since_trigger tracking
# ============================================================================

def test_waypoints_since_trigger_increments(configured_memory_manager):
    """Test _waypoints_since_trigger increments with each process_waypoint call."""
    for i in range(4):
        wp = create_waypoint_analysis(
            waypoint_id=i, scores={"safety": 7.0}, visual_change=True
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)
    assert configured_memory_manager._waypoints_since_trigger == 4


def test_waypoints_since_trigger_resets_on_s2(configured_memory_manager):
    """Test _waypoints_since_trigger resets to 0 when S2 is triggered."""
    from src.agent.system2.persona_reasoner import TriggerReason

    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i, scores={"safety": 7.0}, visual_change=True
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)

    assert configured_memory_manager._waypoints_since_trigger == 3

    trigger_wp = create_waypoint_analysis(
        waypoint_id=3, scores={"safety": 4.0}, visual_change=True
    )
    configured_memory_manager.process_waypoint(
        trigger_wp, triggered=True, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    assert configured_memory_manager._waypoints_since_trigger == 0


def test_waypoints_since_trigger_in_context(configured_memory_manager):
    """Test waypoints_since_trigger is included in the prepared context."""
    from src.agent.system2.persona_reasoner import TriggerReason

    for i in range(3):
        wp = create_waypoint_analysis(
            waypoint_id=i, scores={"safety": 7.0}, visual_change=True
        )
        configured_memory_manager.process_waypoint(wp, triggered=False)

    trigger_wp = create_waypoint_analysis(
        waypoint_id=3, scores={"safety": 4.0}, visual_change=True
    )
    context = configured_memory_manager.process_waypoint(
        trigger_wp, triggered=True, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    assert "waypoints_since_trigger" in context
    # 3 non-triggered + 1 triggered = 4 increments before reset
    assert context["waypoints_since_trigger"] == 4


def test_snapshot_span_covers_since_last_trigger(configured_memory_manager):
    """Test _last_snapshot_waypoint advances after each S2 trigger."""
    from src.agent.system2.persona_reasoner import TriggerReason

    wp1 = create_waypoint_analysis(waypoint_id=0, scores={"safety": 7.0}, visual_change=True)
    configured_memory_manager.process_waypoint(
        wp1, triggered=True, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    assert configured_memory_manager._last_snapshot_waypoint == 0

    wp2 = create_waypoint_analysis(waypoint_id=5, scores={"safety": 4.0}, visual_change=True)
    configured_memory_manager.process_waypoint(
        wp2, triggered=True, trigger_reason=TriggerReason.VISUAL_CHANGE
    )
    assert configured_memory_manager._last_snapshot_waypoint == 5
    assert len(configured_memory_manager._route_snapshots) == 2


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
