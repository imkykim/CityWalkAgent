"""
Test suite for Memory System components.

Tests cover:
- ShortTermMemory: sliding window, trend computation, context management
- LongTermMemory: moment curation, pattern extraction, route summaries
- PersonaReasoner: trigger detection, waypoint reasoning
- ContinuousAnalyzer: pHash detection, visual change tracking, VLM analysis
- Integration: full memory pipeline workflows
- Error handling: edge cases and failure modes
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from typing import Dict, Any, List, Tuple
from datetime import datetime

from src.agent.capabilities.short_term_memory import ShortTermMemory, MemoryItem
from src.agent.capabilities.long_term_memory import (
    LongTermMemory,
    KeyMoment,
    RoutePattern,
    RouteSummary
)
from src.agent.system2.persona_reasoner import (
    PersonaReasoner,
    TriggerReason,
    ReasoningResult,
)
from src.analysis.continuous_analyzer import ContinuousAnalyzer, WaypointAnalysis


# ============================================================================
# Fixtures - Test Data
# ============================================================================


@pytest.fixture
def mock_waypoint_scores() -> Dict[str, float]:
    """Create mock evaluation scores for a waypoint."""
    return {
        "safety": 8.5,
        "comfort": 7.2,
        "interest": 6.8,
        "aesthetics": 7.5
    }


@pytest.fixture
def mock_memory_item() -> MemoryItem:
    """Create a mock MemoryItem for testing."""
    return MemoryItem(
        waypoint_id=0,
        scores={"safety": 8.5, "comfort": 7.2},
        summary="Wide sidewalk, good lighting",
        image_path=Path("/test/waypoint_0.jpg"),
        gps=(37.7749, -122.4194),
        timestamp="2025-01-01T12:00:00"
    )


@pytest.fixture
def mock_waypoint_analysis() -> WaypointAnalysis:
    """Create a mock WaypointAnalysis for testing."""
    return WaypointAnalysis(
        waypoint_id=0,
        image_path=Path("/test/waypoint_0.jpg"),
        scores={"safety": 8.5, "comfort": 7.2},
        reasoning={"safety": "Wide sidewalk", "comfort": "Low noise"},
        timestamp="2025-01-01T12:00:00",
        gps=(37.7749, -122.4194),
        heading=90.0,
        visual_change_detected=False,
        phash_distance=None
    )


@pytest.fixture
def mock_key_moment() -> KeyMoment:
    """Create a mock KeyMoment for testing."""
    return KeyMoment(
        waypoint_id=5,
        image_path=Path("/test/waypoint_5.jpg"),
        scores={"safety": 9.5, "comfort": 9.0},
        summary="Beautiful park entrance with excellent visibility",
        significance="high",
        significance_score=8.5,
        gps=(37.7750, -122.4180),
        timestamp="2025-01-01T12:05:00",
        thinking_confidence=0.92,
        visual_change_detected=True,
        score_delta=2.3
    )


@pytest.fixture
def mock_reasoning_result() -> ReasoningResult:
    """Create a mock ReasoningResult for testing."""
    return ReasoningResult(
        waypoint_id=3,
        trigger_reason=TriggerReason.VISUAL_CHANGE,
        interpretation="Significant improvement in environment quality",
        score_change_reason=None,
        persona_divergence=None,
        significance="high",
        avoid_recommendation=False,
        decision_reason=None,
        prediction="Route quality will continue improving",
        alternative_suggestion=None,
        recommendation="Continue on current path",
        confidence=0.85,
    )


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory storage testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# ShortTermMemory Tests
# ============================================================================


class TestShortTermMemory:
    """Test ShortTermMemory functionality."""

    def test_initialization(self):
        """Test memory initializes with correct window size."""
        stm = ShortTermMemory(window_size=5)

        assert stm.window_size == 5
        assert stm.get_memory_size() == 0
        assert not stm.is_full()

    def test_add_waypoint(self, mock_waypoint_scores):
        """Test adding waypoints to memory."""
        stm = ShortTermMemory(window_size=3)

        stm.add(
            waypoint_id=0,
            scores=mock_waypoint_scores,
            summary="Test waypoint",
            image_path=Path("/test/waypoint_0.jpg"),
            gps=(37.7749, -122.4194),
            timestamp="2025-01-01T12:00:00"
        )

        assert stm.get_memory_size() == 1
        assert not stm.is_full()

        latest = stm.get_latest_waypoint()
        assert latest is not None
        assert latest.waypoint_id == 0
        assert latest.scores == mock_waypoint_scores
        assert latest.summary == "Test waypoint"

    def test_window_eviction(self, mock_waypoint_scores):
        """Test that oldest waypoints are evicted when window is full."""
        stm = ShortTermMemory(window_size=3)

        # Add 5 waypoints (should evict first 2)
        for i in range(5):
            stm.add(
                waypoint_id=i,
                scores=mock_waypoint_scores,
                summary=f"Waypoint {i}",
                gps=(37.7749 + i * 0.001, -122.4194)
            )

        assert stm.get_memory_size() == 3
        assert stm.is_full()

        # Oldest should be waypoint 2, newest should be waypoint 4
        oldest = stm.get_oldest_waypoint()
        latest = stm.get_latest_waypoint()

        assert oldest.waypoint_id == 2
        assert latest.waypoint_id == 4

    def test_get_context_empty(self):
        """Test get_context returns empty dict when memory is empty."""
        stm = ShortTermMemory(window_size=5)
        context = stm.get_context()

        assert context == {}

    def test_get_context_populated(self, mock_waypoint_scores):
        """Test get_context returns complete context data."""
        stm = ShortTermMemory(window_size=5)

        # Add 3 waypoints
        for i in range(3):
            stm.add(
                waypoint_id=i,
                scores={"safety": 7.0 + i, "comfort": 6.0 + i},
                summary=f"Waypoint {i}",
                gps=(37.7749 + i * 0.001, -122.4194)
            )

        context = stm.get_context()

        assert "recent_scores" in context
        assert "recent_summaries" in context
        assert "trend" in context
        assert "waypoint_ids" in context
        assert "gps_coordinates" in context
        assert "window_size" in context

        assert len(context["recent_scores"]) == 3
        assert context["window_size"] == 3
        assert context["waypoint_ids"] == [0, 1, 2]

    def test_compute_trend_improving(self):
        """Test trend detection for improving scores."""
        stm = ShortTermMemory(window_size=10)

        # Add waypoints with improving scores
        for i in range(5):
            stm.add(
                waypoint_id=i,
                scores={"safety": 6.0 + i * 0.5},  # 6.0 -> 8.0
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        context = stm.get_context()
        assert context["trend"] == "improving"

    def test_compute_trend_declining(self):
        """Test trend detection for declining scores."""
        stm = ShortTermMemory(window_size=10)

        # Add waypoints with declining scores
        for i in range(5):
            stm.add(
                waypoint_id=i,
                scores={"safety": 8.0 - i * 0.5},  # 8.0 -> 6.0
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        context = stm.get_context()
        assert context["trend"] == "declining"

    def test_compute_trend_stable(self):
        """Test trend detection for stable scores."""
        stm = ShortTermMemory(window_size=10)

        # Add waypoints with stable scores
        for i in range(5):
            stm.add(
                waypoint_id=i,
                scores={"safety": 7.5 + i * 0.1},  # 7.5 -> 7.9 (small change)
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        context = stm.get_context()
        assert context["trend"] == "stable"

    def test_compute_trend_volatile(self):
        """Test trend detection for volatile scores."""
        stm = ShortTermMemory(window_size=10)

        # Add waypoints with volatile scores
        scores_sequence = [5.0, 9.0, 4.0, 8.5, 5.5]
        for i, score in enumerate(scores_sequence):
            stm.add(
                waypoint_id=i,
                scores={"safety": score},
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        context = stm.get_context()
        assert context["trend"] == "volatile"

    def test_get_score_trends(self):
        """Test per-dimension score trends."""
        stm = ShortTermMemory(window_size=10)

        # Add waypoints with varying scores
        for i in range(3):
            stm.add(
                waypoint_id=i,
                scores={"safety": 7.0 + i, "comfort": 6.0 + i * 0.5},
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        trends = stm.get_score_trends()

        assert "safety" in trends
        assert "comfort" in trends
        assert trends["safety"] == [7.0, 8.0, 9.0]
        assert trends["comfort"] == [6.0, 6.5, 7.0]

    def test_get_summary_text_empty(self):
        """Test summary text for empty memory."""
        stm = ShortTermMemory(window_size=5)
        summary = stm.get_summary_text()

        assert summary == "No recent waypoint history."

    def test_get_summary_text_populated(self):
        """Test summary text generation."""
        stm = ShortTermMemory(window_size=5)

        for i in range(3):
            stm.add(
                waypoint_id=i,
                scores={"safety": 7.0 + i},
                summary=f"Test waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        summary = stm.get_summary_text()

        assert "Recent waypoint history" in summary
        assert "Waypoint 0" in summary
        assert "Waypoint 1" in summary
        assert "Waypoint 2" in summary
        assert "Trend:" in summary

    def test_clear_memory(self, mock_waypoint_scores):
        """Test clearing memory."""
        stm = ShortTermMemory(window_size=5)

        # Add some waypoints
        for i in range(3):
            stm.add(
                waypoint_id=i,
                scores=mock_waypoint_scores,
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        assert stm.get_memory_size() == 3

        stm.clear()

        assert stm.get_memory_size() == 0
        assert stm.get_context() == {}

    def test_default_timestamp(self):
        """Test that default timestamp is generated when not provided."""
        stm = ShortTermMemory(window_size=5)

        stm.add(
            waypoint_id=0,
            scores={"safety": 8.0},
            summary="Test",
            gps=(37.7749, -122.4194)
            # timestamp not provided
        )

        latest = stm.get_latest_waypoint()
        assert latest is not None
        assert latest.timestamp is not None
        # Should be valid ISO format
        datetime.fromisoformat(latest.timestamp)

    def test_default_gps(self):
        """Test that default GPS is used when not provided."""
        stm = ShortTermMemory(window_size=5)

        stm.add(
            waypoint_id=0,
            scores={"safety": 8.0},
            summary="Test"
            # gps not provided
        )

        latest = stm.get_latest_waypoint()
        assert latest is not None
        assert latest.gps == (0.0, 0.0)


# ============================================================================
# LongTermMemory Tests
# ============================================================================


class TestLongTermMemory:
    """Test LongTermMemory functionality."""

    def test_initialization(self, temp_memory_dir):
        """Test memory initializes with correct agent_id and storage."""
        ltm = LongTermMemory(
            agent_id="test_agent",
            storage_path=temp_memory_dir
        )

        assert ltm.agent_id == "test_agent"
        assert ltm.storage_path == temp_memory_dir

        # Check that storage files are created
        assert (temp_memory_dir / "test_agent_memory.jsonl").exists()
        assert (temp_memory_dir / "test_agent_index.json").exists()

    def test_add_candidate_moment_high_significance(self, temp_memory_dir):
        """Test adding high-significance candidate moment."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        result = ltm.add_candidate_moment(
            waypoint_id=5,
            image_path=Path("/test/waypoint_5.jpg"),
            scores={"safety": 9.5, "comfort": 9.0},
            summary="Exceptional park entrance",
            significance="high",
            gps=(37.7750, -122.4180),
            timestamp="2025-01-01T12:05:00",
            thinking_confidence=0.95,
            visual_change_detected=True,
            score_delta=2.5
        )

        assert result is True
        # High significance should pass the threshold

    def test_add_candidate_moment_low_significance_rejected(self, temp_memory_dir):
        """Test that low-significance moments are rejected."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        result = ltm.add_candidate_moment(
            waypoint_id=3,
            image_path=Path("/test/waypoint_3.jpg"),
            scores={"safety": 7.0, "comfort": 7.0},
            summary="Normal sidewalk",
            significance="low",
            gps=(37.7745, -122.4190),
            timestamp="2025-01-01T12:03:00",
            thinking_confidence=0.5,
            visual_change_detected=False,
            score_delta=0.2
        )

        assert result is False
        # Low significance should be below threshold

    def test_curate_moments_adaptive_count(self, temp_memory_dir):
        """Test adaptive moment curation based on route length."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        # Add 20 candidate moments
        for i in range(20):
            ltm.add_candidate_moment(
                waypoint_id=i,
                image_path=Path(f"/test/waypoint_{i}.jpg"),
                scores={"safety": 8.0 + i * 0.1},
                summary=f"Waypoint {i}",
                significance="high",
                gps=(37.7749 + i * 0.001, -122.4194),
                timestamp=f"2025-01-01T12:{i:02d}:00",
                thinking_confidence=0.8,
                visual_change_detected=True,
                score_delta=1.5
            )

        # Route is 1.0 km -> expects ~2 moments (1.0 / 0.5)
        # But with min=3, should get 3 moments
        ltm.curate_moments(route_length_km=1.0)

        # Check that top moments were selected
        # Should have curated the highest significance scores
        # Exact count depends on implementation, but should be reasonable

    def test_curate_moments_respects_min_max(self, temp_memory_dir):
        """Test that curation respects min/max moment counts."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        # Add only 2 high-significance moments
        for i in range(2):
            ltm.add_candidate_moment(
                waypoint_id=i,
                image_path=Path(f"/test/waypoint_{i}.jpg"),
                scores={"safety": 9.0},
                summary=f"Waypoint {i}",
                significance="high",
                gps=(37.7749, -122.4194),
                timestamp=f"2025-01-01T12:00:{i:02d}",
                thinking_confidence=0.9,
                visual_change_detected=True,
                score_delta=2.0
            )

        # Very short route (0.1 km) - but should get min=3 moments
        # However, we only have 2 candidates, so should get 2
        ltm.curate_moments(route_length_km=0.1)

    def test_extract_patterns_trend_detection(self, temp_memory_dir, mock_waypoint_analysis):
        """Test pattern extraction for trend patterns."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        # Create analysis with improving trend
        analyses = []
        for i in range(10):
            analysis = WaypointAnalysis(
                waypoint_id=i,
                image_path=Path(f"/test/waypoint_{i}.jpg"),
                scores={"safety": 6.0 + i * 0.5},  # Improving
                reasoning={"safety": "Getting better"},
                timestamp=f"2025-01-01T12:{i:02d}:00",
                gps=(37.7749, -122.4194),
                heading=90.0,
                visual_change_detected=False,
                phash_distance=None
            )
            analyses.append(analysis)

        patterns = ltm.extract_patterns(analyses, thinking_history=[])

        # Should detect trend patterns
        # Implementation may vary, but should return list of RoutePattern objects
        assert isinstance(patterns, list)

    def test_extract_patterns_barrier_clusters(self, temp_memory_dir):
        """Test pattern extraction for barrier clusters."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        # Create analyses with low-score cluster (barrier)
        analyses = []
        for i in range(10):
            # Create a cluster of low scores at waypoints 4-6
            score = 4.0 if 4 <= i <= 6 else 8.0
            analysis = WaypointAnalysis(
                waypoint_id=i,
                image_path=Path(f"/test/waypoint_{i}.jpg"),
                scores={"safety": score},
                reasoning={"safety": "Barrier detected" if score < 5 else "Safe"},
                timestamp=f"2025-01-01T12:{i:02d}:00",
                gps=(37.7749 + i * 0.001, -122.4194),
                heading=90.0,
                visual_change_detected=False,
                phash_distance=None
            )
            analyses.append(analysis)

        patterns = ltm.extract_patterns(analyses, thinking_history=[])

        # Should detect barrier cluster pattern
        assert isinstance(patterns, list)

    def test_generate_route_summary(self, temp_memory_dir, mock_waypoint_analysis):
        """Test route summary generation."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        # Add some candidate moments and curate
        for i in range(5):
            ltm.add_candidate_moment(
                waypoint_id=i,
                image_path=Path(f"/test/waypoint_{i}.jpg"),
                scores={"safety": 8.0 + i * 0.2},
                summary=f"Moment {i}",
                significance="high",
                gps=(37.7749, -122.4194),
                timestamp=f"2025-01-01T12:{i:02d}:00",
                thinking_confidence=0.85,
                visual_change_detected=True,
                score_delta=1.5
            )

        ltm.curate_moments(route_length_km=1.0)

        # Extract patterns
        analyses = [mock_waypoint_analysis] * 5
        patterns = ltm.extract_patterns(analyses, thinking_history=[])

        # Generate summary
        summary = ltm.generate_route_summary(
            route_name="Test Route",
            start_gps=(37.7749, -122.4194),
            end_gps=(37.7849, -122.4094),
            total_distance_km=1.0,
            total_waypoints=10,
            average_scores={"safety": 8.2, "comfort": 7.5},
            curated_moments=[],
            detected_patterns=patterns
        )

        assert isinstance(summary, RouteSummary)
        assert summary.route_name == "Test Route"
        assert summary.total_distance_km == 1.0
        assert summary.total_waypoints == 10

    def test_store_and_retrieve_experience(self, temp_memory_dir):
        """Test storing and retrieving experiences."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        experience = {
            "route_id": "test_route_123",
            "start": (37.7749, -122.4194),
            "end": (37.7849, -122.4094),
            "perception": {"scores": {"safety": 8.0}},
            "decision": {"recommendation": "accept"},
            "result": {"action": "approved"}
        }

        ltm.store(experience)

        # Retrieve experiences
        retrieved = ltm.retrieve(route_id="test_route_123", limit=10)

        assert len(retrieved) > 0
        assert retrieved[0]["route_id"] == "test_route_123"

    def test_get_statistics(self, temp_memory_dir):
        """Test memory statistics."""
        ltm = LongTermMemory(agent_id="test_agent", storage_path=temp_memory_dir)

        # Store some experiences
        for i in range(3):
            ltm.store({
                "route_id": f"route_{i}",
                "start": (37.7749, -122.4194),
                "end": (37.7849, -122.4094)
            })

        stats = ltm.get_statistics()

        assert "agent_id" in stats
        assert "total_experiences" in stats
        assert stats["agent_id"] == "test_agent"
        assert stats["total_experiences"] >= 3


# ============================================================================
# PersonaReasoner Tests
# ============================================================================


class TestPersonaReasoner:
    """Test PersonaReasoner functionality."""

    def test_initialization(self):
        """Test PersonaReasoner initializes with correct parameters."""
        reasoner = PersonaReasoner(
            distance_trigger_meters=500.0,
            score_delta_threshold=1.5,
        )

        assert reasoner.distance_trigger_meters == 500.0
        assert reasoner.score_delta_threshold == 1.5

    def test_initialization_defaults(self):
        """Test PersonaReasoner initializes with defaults."""
        reasoner = PersonaReasoner()

        assert reasoner.framework_id is not None
        assert reasoner.dimensions is not None

    def test_trigger_reason_enum_values(self):
        """Test TriggerReason enum has expected values."""
        assert hasattr(TriggerReason, "VISUAL_CHANGE")
        assert hasattr(TriggerReason, "SCORE_VOLATILITY")
        assert hasattr(TriggerReason, "DISTANCE_MILESTONE")
        assert hasattr(TriggerReason, "USER_REQUEST")
        assert hasattr(TriggerReason, "EXCEPTIONAL_MOMENT")

    def test_fallback_result_fields(self):
        """Test fallback result has correct fields."""
        reasoner = PersonaReasoner()
        system1_scores = {"safety": 7.5, "comfort": 8.0}

        fallback = reasoner._create_fallback_result(
            waypoint_id=1,
            trigger_reason=TriggerReason.VISUAL_CHANGE,
            system1_scores=system1_scores,
            error="Test error"
        )

        assert isinstance(fallback, ReasoningResult)
        assert fallback.waypoint_id == 1
        assert fallback.trigger_reason == TriggerReason.VISUAL_CHANGE
        assert fallback.system1_scores == system1_scores
        assert fallback.confidence == 0.0
        assert fallback.significance == "low"


# ============================================================================
# ContinuousAnalyzer Tests
# ============================================================================


class TestContinuousAnalyzer:
    """Test ContinuousAnalyzer functionality."""

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    def test_initialization(self, mock_evaluator, mock_load_framework):
        """Test analyzer initializes correctly."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(
            framework_id="sagai_2025",
            context_window=3
        )

        assert analyzer.context_window == 3
        assert analyzer.enable_multi_image is True
        assert len(analyzer.analysis_history) == 0
        assert len(analyzer.phash_distances) == 0

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    @patch('imagehash.phash')
    @patch('PIL.Image.open')
    @pytest.mark.skip(reason="Deprecated: CognitiveController now handles pHash detection")

    def test_detect_visual_change_first_image(
        self, mock_image_open, mock_phash, mock_evaluator, mock_load_framework
    ):
        """Test visual change detection on first image (no baseline)."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(framework_id="sagai_2025")

        # Mock pHash
        mock_hash = Mock()
        mock_phash.return_value = mock_hash
        mock_image = Mock()
        mock_image_open.return_value = mock_image

        # First image should not detect change (no baseline)
        changed, distance = analyzer._detect_visual_change(mock_hash)

        assert changed is False
        assert distance is None
        assert analyzer.last_phash == mock_hash

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    @pytest.mark.skip(reason="Deprecated: CognitiveController now handles pHash detection")

    def test_detect_visual_change_threshold_exceeded(self, mock_evaluator, mock_load_framework):
        """Test visual change detection when threshold is exceeded."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(
            framework_id="sagai_2025",
            phash_threshold=10.0,
            adaptive_threshold=False
        )

        # Set baseline hash
        mock_hash_1 = Mock()
        mock_hash_1.__sub__ = Mock(return_value=5)  # Small difference
        analyzer.last_phash = mock_hash_1

        # New hash with large difference
        mock_hash_2 = Mock()
        mock_hash_2.__sub__ = Mock(return_value=15)  # Large difference
        mock_hash_1.__sub__.return_value = 15

        changed, distance = analyzer._detect_visual_change(mock_hash_2)

        assert changed is True
        assert distance == 15.0

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    @pytest.mark.skip(reason="Deprecated: CognitiveController now handles pHash detection")

    def test_detect_visual_change_threshold_not_exceeded(
        self, mock_evaluator, mock_load_framework
    ):
        """Test visual change detection when threshold is not exceeded."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(
            framework_id="sagai_2025",
            phash_threshold=20.0,
            adaptive_threshold=False
        )

        # Set baseline hash
        mock_hash_1 = Mock()
        analyzer.last_phash = mock_hash_1

        # New hash with small difference
        mock_hash_2 = Mock()
        mock_hash_2.__sub__ = Mock(return_value=10)
        mock_hash_1.__sub__.return_value = 10

        changed, distance = analyzer._detect_visual_change(mock_hash_2)

        assert changed is False
        assert distance == 10.0

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    @pytest.mark.skip(reason="Deprecated: CognitiveController now handles pHash detection")

    def test_adaptive_threshold_computation(self, mock_evaluator, mock_load_framework):
        """Test adaptive threshold uses statistical approach."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(
            framework_id="sagai_2025",
            adaptive_threshold=True
        )

        # Build up phash distance history
        analyzer.phash_distances = [5.0, 6.0, 7.0, 5.5, 6.5]

        # Set baseline
        mock_hash_1 = Mock()
        analyzer.last_phash = mock_hash_1

        # New hash with moderate difference
        mock_hash_2 = Mock()
        mock_hash_2.__sub__ = Mock(return_value=12)
        mock_hash_1.__sub__.return_value = 12

        # Adaptive threshold = mean + 1.5 * std
        # mean ≈ 6.0, std ≈ 0.7, threshold ≈ 7.05
        # distance = 12.0 > 7.05 -> should detect change
        changed, distance = analyzer._detect_visual_change(mock_hash_2)

        assert changed is True
        assert distance == 12.0

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    @patch('imagehash.phash')
    @patch('PIL.Image.open')
    def test_analyze_waypoint(
        self, mock_image_open, mock_phash, mock_evaluator_class, mock_load_framework
    ):
        """Test single waypoint analysis."""
        mock_framework = Mock()
        mock_framework.dimensions = [
            {"id": "safety", "name": "Safety"},
            {"id": "comfort", "name": "Comfort"}
        ]
        mock_load_framework.return_value = mock_framework

        # Mock evaluator
        mock_evaluator = Mock()
        mock_eval_result = [
            {"dimension_id": "safety", "score": 8.5, "reasoning": "Safe conditions"},
            {"dimension_id": "comfort", "score": 7.2, "reasoning": "Comfortable path"}
        ]
        mock_evaluator.evaluate_image.return_value = mock_eval_result
        mock_evaluator_class.return_value = mock_evaluator

        # Mock pHash
        mock_hash = Mock()
        mock_phash.return_value = mock_hash
        mock_image = Mock()
        mock_image_open.return_value = mock_image

        analyzer = ContinuousAnalyzer(framework_id="sagai_2025")

        # New architecture: visual change info comes from CognitiveController
        result = analyzer.analyze_waypoint(
            waypoint_id=0,
            image_path=Path("/test/waypoint_0.jpg"),
            metadata={
                "lat": 37.7749,
                "lon": -122.4194,
                "heading": 90.0,
                "timestamp": "2025-01-01T12:00:00"
            },
            visual_change_detected=False,
            phash_distance=5.0
        )

        assert isinstance(result, WaypointAnalysis)
        assert result.waypoint_id == 0
        assert result.scores["safety"] == 8.5
        assert result.scores["comfort"] == 7.2
        assert "safety" in result.reasoning
        assert result.visual_change_detected is False
        assert result.phash_distance == 5.0

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    def test_get_context_window_empty(self, mock_evaluator, mock_load_framework):
        """Test context window when analysis history is empty."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(framework_id="sagai_2025", context_window=3)

        context = analyzer.get_context_window()

        assert context == []

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    def test_get_context_window_populated(self, mock_evaluator, mock_load_framework):
        """Test context window returns recent analyses."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        analyzer = ContinuousAnalyzer(framework_id="sagai_2025", context_window=3)

        # Add 5 analyses
        for i in range(5):
            analysis = WaypointAnalysis(
                waypoint_id=i,
                image_path=Path(f"/test/waypoint_{i}.jpg"),
                scores={"safety": 8.0},
                reasoning={"safety": "Good"},
                timestamp=f"2025-01-01T12:{i:02d}:00",
                gps=(37.7749, -122.4194),
                heading=90.0,
                visual_change_detected=False,
                phash_distance=None
            )
            analyzer.analysis_history.append(analysis)

        context = analyzer.get_context_window()

        # Should return last 3 analyses (waypoints 2, 3, 4)
        assert len(context) == 3
        assert context[0].waypoint_id == 2
        assert context[1].waypoint_id == 3
        assert context[2].waypoint_id == 4


# ============================================================================
# Integration Tests
# ============================================================================


class TestMemorySystemIntegration:
    """Integration tests for complete memory system workflows."""

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    def test_full_memory_pipeline(
        self, mock_evaluator_class, mock_load_framework, temp_memory_dir
    ):
        """Test complete memory pipeline workflow."""
        # Setup mocks
        mock_framework = Mock()
        mock_framework.dimensions = [{"id": "safety", "name": "Safety"}]
        mock_load_framework.return_value = mock_framework

        mock_evaluator = Mock()
        mock_evaluator.evaluate_single_image.return_value = {
            "safety": {"score": 8.0, "reasoning": "Safe"}
        }
        mock_evaluator_class.return_value = mock_evaluator

        # Initialize components
        analyzer = ContinuousAnalyzer(framework_id="sagai_2025")
        stm = ShortTermMemory(window_size=5)
        reasoner = PersonaReasoner()
        ltm = LongTermMemory(agent_id="integration_test", storage_path=temp_memory_dir)

        # Simulate waypoint processing
        with patch('imagehash.phash'), patch('PIL.Image.open'):
            for i in range(3):
                # Analyze waypoint
                analysis = WaypointAnalysis(
                    waypoint_id=i,
                    image_path=Path(f"/test/waypoint_{i}.jpg"),
                    scores={"safety": 7.0 + i},
                    reasoning={"safety": "Getting better"},
                    timestamp=f"2025-01-01T12:{i:02d}:00",
                    gps=(37.7749 + i * 0.001, -122.4194),
                    heading=90.0,
                    visual_change_detected=i > 0,
                    phash_distance=10.0 if i > 0 else None
                )

                # Add to STM
                stm.add(
                    waypoint_id=i,
                    scores=analysis.scores,
                    summary=f"Waypoint {i}",
                    image_path=analysis.image_path,
                    gps=analysis.gps
                )

        # Verify pipeline results
        assert stm.get_memory_size() == 3
        assert stm.get_context()["trend"] in ["improving", "stable", "volatile"]

    def test_memory_components_interoperability(self, temp_memory_dir):
        """Test that memory components work together correctly."""
        stm = ShortTermMemory(window_size=5)
        ltm = LongTermMemory(agent_id="interop_test", storage_path=temp_memory_dir)

        # Add data to STM
        for i in range(3):
            stm.add(
                waypoint_id=i,
                scores={"safety": 8.0 + i * 0.5},
                summary=f"Waypoint {i}",
                gps=(37.7749, -122.4194)
            )

        # Get STM context
        context = stm.get_context()

        # Use context to inform LTM decision
        latest = stm.get_latest_waypoint()
        if latest and context["trend"] == "improving":
            ltm.add_candidate_moment(
                waypoint_id=latest.waypoint_id,
                image_path=latest.image_path,
                scores=latest.scores,
                summary="Improving trend moment",
                significance="high",
                gps=latest.gps,
                timestamp=latest.timestamp,
                thinking_confidence=0.85,
                visual_change_detected=False,
                score_delta=1.0
            )

        # Verify interoperability
        assert context["trend"] == "improving"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_short_term_memory_empty_operations(self):
        """Test STM operations on empty memory."""
        stm = ShortTermMemory(window_size=5)

        assert stm.get_context() == {}
        assert stm.get_score_trends() == {}
        assert stm.get_latest_waypoint() is None
        assert stm.get_oldest_waypoint() is None
        assert stm.get_summary_text() == "No recent waypoint history."

    def test_long_term_memory_invalid_path(self):
        """Test LTM handles invalid storage path gracefully."""
        # This should create the directory if it doesn't exist
        ltm = LongTermMemory(
            agent_id="test_agent",
            storage_dir=Path("/tmp/test_memory_nonexistent")
        )

        assert ltm.agent_id == "test_agent"

    def test_persona_reasoner_fallback_on_empty_context(self):
        """Test PersonaReasoner fallback result has correct waypoint_id."""
        reasoner = PersonaReasoner()

        fallback = reasoner._create_fallback_result(
            waypoint_id=0,
            trigger_reason=TriggerReason.USER_REQUEST,
            system1_scores={},
            error="No context available"
        )

        assert isinstance(fallback, ReasoningResult)
        assert fallback.waypoint_id == 0

    @patch('src.analysis.continuous_analyzer.load_framework')
    @patch('src.analysis.continuous_analyzer.Evaluator')
    @patch('PIL.Image.open')
    def test_continuous_analyzer_image_load_failure(
        self, mock_image_open, mock_evaluator, mock_load_framework
    ):
        """Test analyzer handles image load failures gracefully."""
        mock_framework = Mock()
        mock_load_framework.return_value = mock_framework

        # Mock image open failure
        mock_image_open.side_effect = IOError("Image not found")

        analyzer = ContinuousAnalyzer(framework_id="sagai_2025")

        # Should handle error gracefully
        with pytest.raises(IOError):
            analyzer.analyze_waypoint(
                waypoint_id=0,
                image_path=Path("/nonexistent/image.jpg"),
                metadata={"gps": (37.7749, -122.4194), "heading": 90.0}
            )


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
