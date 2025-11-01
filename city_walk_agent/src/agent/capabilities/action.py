"""Action capability - agent's action execution and output formatting.

Design principle: EXECUTE and FORMAT, don't decide.
"""

from typing import Any, Dict
from src.utils.logging import get_logger


class ActionCapability:
    """Capability for executing actions based on decisions.

    Takes decisions from ThinkingCapability and formats them into
    actionable outputs (messages, reports, etc.).

    Example:
        actor = ActionCapability()
        result = actor.execute(decision)
    """

    def __init__(self):
        """Initialize action capability."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("ActionCapability initialized")

    def execute(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action by formatting decision into output.

        Args:
            decision: Decision dict from ThinkingCapability with:
                - recommendation: "accept" or "reject"
                - confidence: float (0-1)
                - explanation: str
                - concerns: List[str]
                - highlights: List[str]
                - weighted_score: float

        Returns:
            Action result dict with:
                - action_type: "approve" or "reject"
                - message: formatted message string
                - confidence: float
                - recommendation: str
                - concerns: List[str]
                - highlights: List[str]
                - weighted_score: float
        """
        recommendation = decision.get("recommendation", "unknown")
        confidence = decision.get("confidence", 0.0)
        explanation = decision.get("explanation", "")

        # Format message based on recommendation
        if recommendation == "accept":
            action_type = "approve"
            icon = "✅"
        else:
            action_type = "reject"
            icon = "⚠️"

        message = f"{icon} {explanation}"

        # Build action result
        result = {
            "action_type": action_type,
            "message": message,
            "confidence": confidence,
            "recommendation": recommendation,
            "concerns": decision.get("concerns", []),
            "highlights": decision.get("highlights", []),
            "weighted_score": decision.get("weighted_score", 0.0),
        }

        self.logger.debug("Action executed", action_type=action_type)

        return result
