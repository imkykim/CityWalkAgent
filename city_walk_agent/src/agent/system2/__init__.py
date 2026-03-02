"""System 2: Persona-aware Reasoning Engine.

Four-stage pipeline called sequentially from WalkingAgent.run_system2():

    System1Evidence
        → Interpreter  (WHY:  explain score changes, persona divergence)
        → Decider      (WHAT: route recommendation, avoid segments)
        → Planner      (HOW:  optimal route, multi-route comparison)
        → Reporter     (OUT:  comprehensive evaluation report)

Each stage receives the outputs of all previous stages so context
accumulates through the pipeline.
"""

from .interpreter import Interpreter
from .decider     import Decider
from .planner     import Planner
from .reporter    import Reporter
from .models import (
    System1Evidence,
    InterpretResult,
    DecideResult,
    PlanResult,
    ReportResult,
)

__all__ = [
    "Interpreter",
    "Decider",
    "Planner",
    "Reporter",
    "System1Evidence",
    "InterpretResult",
    "DecideResult",
    "PlanResult",
    "ReportResult",
]
