"""System 2: Persona-aware Reasoning Engine.

Four-stage pipeline called sequentially from CityWalkAgent.run_system2():

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
from .llm_client  import call_llm
from .models import (
    System1Evidence,
    InterpretResult,
    DecideResult,
    PlanResult,
    ReportResult,
)
from .persona_reasoner import PersonaReasoner, ReasoningResult, TriggerReason

__all__ = [
    "Interpreter",
    "Decider",
    "Planner",
    "Reporter",
    "call_llm",
    "System1Evidence",
    "InterpretResult",
    "DecideResult",
    "PlanResult",
    "ReportResult",
    "PersonaReasoner",
    "ReasoningResult",
    "TriggerReason",
]
