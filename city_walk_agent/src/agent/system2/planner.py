"""Planner — HOW to walk? (Step 3 of System 2)

Responsibilities:
- Compare multiple route candidates by persona priorities
- Select primary route and rank alternatives
- Integrate avoid-segments from Decider
"""

from __future__ import annotations
from typing import Any, List, Optional

from src.utils.logging import get_logger
from .models import System1Evidence, DecideResult, PlanResult


class Planner:
    """Generates persona-optimised route plans.

    TODO: Implement using:
    - Multi-route scoring weighted by persona dimension priorities
    - Avoid-segment integration from DecideResult
    - Alternative ranking by persona-weighted composite score
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

    def plan(
        self,
        evidence:         System1Evidence,
        decide_result:    DecideResult,
        candidate_routes: Optional[List[Any]] = None,
        personality:      Optional[Any] = None,
    ) -> PlanResult:
        """Generate an optimal route plan.

        Args:
            evidence:         Raw System 1 evidence.
            decide_result:    Output from Decider (avoid segments, etc.).
            candidate_routes: Optional pre-generated route alternatives.
            personality:      Agent personality (EnhancedPersonalityConfig).

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Planner.plan() — not yet implemented.")
