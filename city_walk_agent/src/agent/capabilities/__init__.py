"""Agent capabilities package.

This package provides modular capabilities for agents following the
Observe-Analyze-Think-Act pattern:

- Memory: Persistent storage for experiences and learning
- Observation: Visual perception through VLM evaluation
- Analysis: Sequential pattern analysis and barrier detection
- Thinking: Reasoning and decision-making logic
- (Future: Planning, Communication, etc.)

Each capability is designed to be:
- Composable: Can be used independently or together
- Testable: Clear interfaces with minimal dependencies
- Reusable: Shared across different agent types
"""

from src.agent.capabilities.action import ActionCapability
from src.agent.capabilities.analysis import AnalysisCapability
from src.agent.capabilities.memory import AgentMemory
from src.agent.capabilities.observation import ObservationCapability
from src.agent.capabilities.thinking import ThinkingCapability

__all__ = [
    "ActionCapability",
    "AnalysisCapability",
    "AgentMemory",
    "ObservationCapability",
    "ThinkingCapability",
]
