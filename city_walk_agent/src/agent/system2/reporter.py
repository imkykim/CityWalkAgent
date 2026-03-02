"""Reporter — OUTPUT comprehensive evaluation report. (Step 4 of System 2)

Responsibilities:
- Executive summary for stakeholders
- Per-dimension findings and recommendations
- Urban planning improvement suggestions
- Persona suitability matrix (all personas, not just current)
- Bilingual output (Chinese + English)
"""

from __future__ import annotations
from typing import Any, Optional

from src.utils.logging import get_logger
from .models import (
    System1Evidence,
    InterpretResult,
    DecideResult,
    PlanResult,
    ReportResult,
)


class Reporter:
    """Generates comprehensive evaluation reports from the full pipeline.

    TODO: Implement using VLM narrative generation:
    - Synthesise all pipeline outputs into structured prose
    - Urban planning format for planners and stakeholders
    - Persona suitability matrix across all available personas
    - Bilingual output (Chinese + English) for thesis presentation
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

    def report(
        self,
        evidence:         System1Evidence,
        interpret_result: InterpretResult,
        decide_result:    DecideResult,
        plan_result:      PlanResult,
        personality:      Optional[Any] = None,
    ) -> ReportResult:
        """Generate a comprehensive evaluation report.

        Args:
            evidence:         Raw System 1 evidence.
            interpret_result: Output from Interpreter.
            decide_result:    Output from Decider.
            plan_result:      Output from Planner.
            personality:      Agent personality (EnhancedPersonalityConfig).

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError("Reporter.report() — not yet implemented.")
