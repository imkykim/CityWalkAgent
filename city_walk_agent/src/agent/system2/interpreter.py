"""Interpreter — WHY did scores change? (Step 1 of System 2)

Responsibilities:
- Explain score trajectories per dimension
- Detect where persona scoring diverged from objective scoring
- Classify environmental transitions with causal analysis
"""

from __future__ import annotations
from typing import Any, Optional

from src.utils.logging import get_logger
from .models import System1Evidence, InterpretResult


class Interpreter:
    """Analyzes System 1 evidence to explain score changes and persona divergence.

    TODO: Implement using VLM calls with:
    - Score trajectory analysis (which dimensions changed and where)
    - Persona vs objective divergence detection
    - Environmental transition classification
    - Causal explanation generation
    """

    def __init__(
        self,
        framework_id: str,
        vlm_provider: str = "claude",
        vlm_api_key: Optional[str] = None,
    ) -> None:
        self.framework_id = framework_id
        self.vlm_provider = vlm_provider
        self.vlm_api_key  = vlm_api_key
        self.logger = get_logger(self.__class__.__name__)

    def interpret(
        self,
        evidence:    System1Evidence,
        personality: Optional[Any] = None,
    ) -> InterpretResult:
        """Interpret System 1 evidence and explain score changes.

        Args:
            evidence:    Aggregated System 1 perception results.
            personality: Agent personality (EnhancedPersonalityConfig).

        Returns:
            InterpretResult with explanations and divergence points.

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Interpreter.interpret() — not yet implemented.")
