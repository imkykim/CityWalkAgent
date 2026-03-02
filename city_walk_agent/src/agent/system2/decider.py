"""Decider — WHAT action to take? (Step 2 of System 2)

Responsibilities:
- Accept / reject / modify route recommendation
- Identify segments to avoid based on persona priorities
- Score confidence based on evidence quality
"""

from __future__ import annotations
from typing import Any, Optional

from src.utils.logging import get_logger
from .models import System1Evidence, InterpretResult, DecideResult


class Decider:
    """Makes route decisions from interpreted System 1 evidence.

    TODO: Implement using:
    - Persona-specific thresholds (parent_with_kids stricter on safety)
    - Barrier severity classification (dealbreaker vs acceptable)
    - Confidence scoring based on evidence quality
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

    def decide(
        self,
        evidence:         System1Evidence,
        interpret_result: InterpretResult,
        personality:      Optional[Any] = None,
    ) -> DecideResult:
        """Make a route decision from evidence and interpretation.

        Args:
            evidence:         Raw System 1 evidence.
            interpret_result: Output from Interpreter.
            personality:      Agent personality (EnhancedPersonalityConfig).

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Decider.decide() — not yet implemented.")
