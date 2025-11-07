"""Agent capabilities package.

This package provides modular capabilities for agents following the
Observe-Analyze-Think-Act pattern:

- LongTermMemory: Persistent JSONL storage for experiences and learning
- ShortTermMemory: Sliding window memory for real-time route context
- Observation: Visual perception through VLM evaluation
- Analysis: Sequential pattern analysis and barrier detection
- Thinking: Reasoning and decision-making logic
- Action: Route execution and navigation capabilities
- (Future: Planning, Communication, etc.)

Each capability is designed to be:
- Composable: Can be used independently or together
- Testable: Clear interfaces with minimal dependencies
- Reusable: Shared across different agent types
"""

from src.agent.capabilities.action import ActionCapability
from src.agent.capabilities.analysis import AnalysisCapability
from src.agent.capabilities.long_term_memory import (
    LongTermMemory,
    KeyMoment,
    RoutePattern,
    RouteSummary
)
from src.agent.capabilities.short_term_memory import ShortTermMemory, MemoryItem
from src.agent.capabilities.observation import ObservationCapability
from src.agent.capabilities.thinking import (
    ThinkingCapability,
    ThinkingModule,
    TriggerReason,
    ThinkingResult
)

__all__ = [
    # Core capabilities
    "ActionCapability",
    "AnalysisCapability",
    "ObservationCapability",
    "ThinkingCapability",
    # Long-term memory
    "LongTermMemory",
    "KeyMoment",
    "RoutePattern",
    "RouteSummary",
    # Short-term memory
    "ShortTermMemory",
    "MemoryItem",
    # Thinking module (waypoint-level)
    "ThinkingModule",
    "TriggerReason",
    "ThinkingResult",
]
