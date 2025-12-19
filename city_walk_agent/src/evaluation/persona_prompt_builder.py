"""PersonaPromptBuilder for dual evaluation system.

Builds both objective and persona-aware prompts for VLM evaluation.
Objective prompts use the framework's dimension definitions directly.
Persona prompts incorporate the persona's perspective autonomously via VLM interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from src.agent.config.personalities import EnhancedPersonalityConfig


@dataclass
class DualPrompts:
    """Container for both objective and persona prompts for a dimension.

    Attributes:
        dimension_id: The dimension identifier
        objective_prompt: Original framework prompt (unbiased evaluation)
        persona_prompt: Persona-aware prompt (VLM interprets through persona lens)
    """

    dimension_id: str
    objective_prompt: str
    persona_prompt: str


class PersonaPromptBuilder:
    """Builds objective and persona-aware prompts for dual evaluation.

    The builder creates two types of prompts:
    - Objective: Uses the framework's dimension definition directly
    - Persona: Wraps the dimension with persona context for VLM interpretation

    The VLM autonomously interprets what each dimension means for the persona's
    specific needs, without hardcoded weights or keyword matching.
    """

    def __init__(
        self,
        framework: Dict,
        persona: Optional[EnhancedPersonalityConfig] = None
    ):
        """Initialize the prompt builder.

        Args:
            framework: The evaluation framework dictionary with dimensions
            persona: Optional persona configuration for persona-aware prompts
        """
        self.framework = framework
        self.persona = persona
        self._dimension_cache = {
            dim["id"]: dim for dim in framework.get("dimensions", [])
        }

    def build_objective_prompt(self, dimension_id: str) -> str:
        """Build objective evaluation prompt (framework definition only).

        Args:
            dimension_id: The dimension to build a prompt for

        Returns:
            The original VLM prompt from the framework dimension

        Raises:
            ValueError: If dimension_id not found in framework
        """
        dimension = self._dimension_cache.get(dimension_id)
        if not dimension:
            raise ValueError(
                f"Dimension '{dimension_id}' not found in framework. "
                f"Available: {list(self._dimension_cache.keys())}"
            )

        return dimension.get("vlm_prompt", "")

    def build_persona_prompt(self, dimension_id: str) -> str:
        """Build persona-aware evaluation prompt.

        The VLM interprets the dimension through the persona's lens autonomously.
        No hardcoded weights or keywords - the VLM decides what matters for this persona.

        Args:
            dimension_id: The dimension to build a prompt for

        Returns:
            Persona-contextualized prompt for VLM evaluation

        Raises:
            ValueError: If dimension_id not found or no persona configured
        """
        if not self.persona:
            raise ValueError(
                "Cannot build persona prompt: no persona configured. "
                "Initialize with a persona or use build_objective_prompt() instead."
            )

        dimension = self._dimension_cache.get(dimension_id)
        if not dimension:
            raise ValueError(
                f"Dimension '{dimension_id}' not found in framework. "
                f"Available: {list(self._dimension_cache.keys())}"
            )

        # Extract dimension information
        dim_name = dimension.get("name", dimension_id)
        dim_description = dimension.get("description", "")

        # Build persona-aware prompt using template
        prompt = f"""# YOUR IDENTITY
{self.persona.system1_persona_hint}

# DIMENSION: {dim_name}
{dim_description}

# EVALUATION APPROACH
As a {self.persona.name}, consider what "{dim_name}" means for YOUR specific needs.
Your priorities should naturally guide how you interpret and weight this dimension.

Score based on YOUR needs, not objective architectural merit.

# OUTPUT (JSON)
{{"score": <1-10>, "reasoning": "<2-3 sentences from YOUR perspective>"}}
"""

        return prompt

    def build_dual_prompts(self, dimension_id: str) -> DualPrompts:
        """Build both objective and persona prompts for a dimension.

        Args:
            dimension_id: The dimension to build prompts for

        Returns:
            DualPrompts containing both objective and persona prompts

        Raises:
            ValueError: If dimension_id not found or no persona configured
        """
        objective = self.build_objective_prompt(dimension_id)
        persona = self.build_persona_prompt(dimension_id)

        return DualPrompts(
            dimension_id=dimension_id,
            objective_prompt=objective,
            persona_prompt=persona,
        )

    def get_dimension_ids(self) -> list[str]:
        """Get all available dimension IDs from the framework.

        Returns:
            List of dimension identifiers
        """
        return list(self._dimension_cache.keys())

    def has_persona(self) -> bool:
        """Check if a persona is configured.

        Returns:
            True if persona is configured, False otherwise
        """
        return self.persona is not None
