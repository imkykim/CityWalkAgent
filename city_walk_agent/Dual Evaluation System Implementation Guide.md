# Dual Evaluation System Implementation Guide

## Overview

Implement parallel Objective + Persona-aware evaluation in System 1, removing weight-based scoring in favor of VLM-driven persona interpretation.

**Key Principles:**

- VLM interprets dimensions through persona lens autonomously (no hardcoded interpretations)
- Objective and Persona evaluations run in parallel (independent)
- Persona scores are final scores (no weighted calculation)
- Objective scores kept for research/comparison only

---

## Phase 1: Simplify EnhancedPersonalityConfig

### Step 1.1: Remove weight-related fields from dataclass

**File:** `src/agent/config/personalities.py`

Remove from `EnhancedPersonalityConfig`:

- `semantic_config: Dict[str, List[str]]` - DELETE
- `weight_distribution: Dict[str, float]` - DELETE

Keep:

- `personality_id`
- `name`
- `description`
- `vlm_persona_prompt` (for System 2)
- `system1_persona_hint` (for System 1)
- `thresholds`
- `explanation_style`

### Step 1.2: Update all 5 personality definitions

Remove `semantic_config` and `weight_distribution` blocks from:

- homebuyer
- runner
- parent_with_kids
- photographer
- elderly_walker

### Step 1.3: Remove weight utility functions

**File:** `src/agent/config/personalities.py`

Delete or deprecate:

- `map_semantic_to_weights()` - DELETE
- `normalize_weights()` - DELETE
- `get_primary_dimensions()` - DELETE or keep for other uses

### Step 1.4: Simplify AgentPersonality dataclass

Remove:

- `dimension_weights: Dict[str, float]` - DELETE

Keep:

- `name`
- `description`
- `decision_thresholds`
- `explanation_style`
- `personality_id`

### Step 1.5: Simplify get_preset() function

Remove all weight calculation logic:

- Remove `map_semantic_to_weights()` call
- Remove `normalize_weights()` call
- Remove `dimension_weights` from return

New logic: Simply return persona config without weight computation

### Step 1.6: Update **init**.py exports

Remove from exports:

- `map_semantic_to_weights`
- `normalize_weights`
- `get_primary_dimensions`

### Step 1.7: Clean up constants.py

Remove if unused elsewhere:

- `WEIGHT_NEUTRAL_VALUE`
- `MIN_WEIGHT`
- `MAX_WEIGHT`

---

## Phase 2: Create PersonaPromptBuilder

### Step 2.1: Create new file

**File:** `src/evaluation/persona_prompt_builder.py`

### Step 2.2: Implement PersonaPromptBuilder class

Key methods:

- `__init__(framework, persona=None)` - Initialize with framework and optional persona
- `build_objective_prompt(dimension_id)` - Return original dimension.vlm_prompt
- `build_persona_prompt(dimension_id)` - Build VLM-interpreted persona prompt
- `build_dual_prompts(dimension_id)` - Return both prompts as tuple/dataclass

### Step 2.3: Implement persona prompt template

Template structure (VLM interprets autonomously):

```
# YOUR IDENTITY
{persona.system1_persona_hint}

# DIMENSION: {dimension.name}
{dimension.description}

# EVALUATION APPROACH
As a {persona.name}, consider what "{dimension.name}" means for YOUR specific needs.
Your priorities should naturally guide how you interpret and weight this dimension.

Score based on YOUR needs, not objective architectural merit.

# OUTPUT (JSON)
{"score": <1-10>, "reasoning": "<2-3 sentences from YOUR perspective>"}
```

### Step 2.4: Add DualPrompts dataclass

```python
@dataclass
class DualPrompts:
    dimension_id: str
    objective_prompt: str
    persona_prompt: str
```

---

## Phase 3: Update Evaluator for Dual Evaluation

### Step 3.1: Modify evaluate_image() signature

**File:** `src/evaluation/evaluator.py`

Change parameters:

- Remove: `persona_hint: Optional[str]`
- Add: `persona: Optional[EnhancedPersonalityConfig]`
- Add: `evaluation_mode: Literal["objective", "persona", "dual"]`

### Step 3.2: Integrate PersonaPromptBuilder

- Import PersonaPromptBuilder
- Create builder instance with framework and persona
- Use appropriate prompt based on evaluation_mode

### Step 3.3: Implement dual evaluation logic

When `evaluation_mode="dual"`:

- Build both objective and persona prompts
- Make 2 VLM calls per dimension (can be parallelized later)
- Return both score sets

### Step 3.4: Create new return dataclass

```python
@dataclass
class DualEvaluationResult:
    dimension_id: str
    objective_score: float
    objective_reasoning: str
    persona_score: float
    persona_reasoning: str
```

### Step 3.5: Add evaluate_image_dual() method (optional)

Alternative: Create separate method specifically for dual evaluation to keep backward compatibility with existing evaluate_image()

---

## Phase 4: Update ContinuousAnalyzer

### Step 4.1: Remove sequential dual-call logic

**File:** `src/analysis/continuous_analyzer.py`

Remove current pattern:

```python
# OLD: Sequential
eval_results = evaluator.evaluate_image(..., persona_hint=None)
if self.persona_hint:
    persona_results = evaluator.evaluate_image(..., persona_hint=self.persona_hint)
```

### Step 4.2: Implement parallel dual evaluation

New pattern:

```python
# NEW: Single call returns both
dual_results = evaluator.evaluate_image(
    image_path,
    persona=self.persona,
    evaluation_mode="dual"
)
# dual_results contains both objective_scores and persona_scores
```

### Step 4.3: Update WaypointAnalysis dataclass

Rename/clarify fields:

- `scores` → `persona_scores` (final scores for decision making)
- `neutral_scores` → `objective_scores` (for research)
- Remove `persona_adjustments` (no longer meaningful - they're independent)
- Remove `persona_applied` flag (always true if persona exists)

New structure:

```python
@dataclass
class WaypointAnalysis:
    waypoint_id: int
    image_path: Path

    # Dual evaluation results
    objective_scores: Dict[str, float]
    objective_reasoning: Dict[str, str]
    persona_scores: Dict[str, float]  # This is the "final" score
    persona_reasoning: Dict[str, str]

    # Metadata
    timestamp: str
    gps: Tuple[float, float]
    heading: float
    visual_change_detected: bool
    phash_distance: Optional[float]
```

### Step 4.4: Update analyze_waypoint() method

- Use new evaluator interface
- Store both objective and persona results
- Remove adjustment calculations

### Step 4.5: Update statistics methods

- `get_statistics()` should report both objective and persona score distributions
- Add comparison metrics (how much do they differ on average?)

---

## Phase 5: Update ThinkingModule (System 2)

### Step 5.1: Review current implementation

**File:** `src/agent/capabilities/thinking.py`

Check references to:

- `scoring_rules` (should be removed - was deleted in previous refactor)
- `boost_keywords` / `concern_keywords` (remove if still referenced)
- Weight-based calculations

### Step 5.2: Update System 2 prompt building

- `_build_vlm_revision_prompt()` should use `vlm_persona_prompt` directly
- Remove any keyword-based sections
- Ensure System 2 receives persona_scores from System 1 (not objective)

### Step 5.3: Update ThinkingResult

- Clarify that `revised_scores` are revisions of persona_scores
- Add field for objective_scores if needed for comparison

---

## Phase 6: Update Visualization

### Step 6.1: Update plot_persona_comparison()

**File:** `src/utils/visualization.py`

Rename/update:

- "Neutral vs Persona" → "Objective vs Persona"
- Update data extraction to use new field names

### Step 6.2: Update radar plot methods

- `plot_waypoint_radars()` should handle both score types
- Clear labeling: "Objective (Research)" vs "Persona (Final)"

### Step 6.3: Add new comparison visualizations

Consider adding:

- Scatter plot: Objective vs Persona scores per dimension
- Distribution plot: How much do scores differ by persona type?
- Heatmap: Dimension × Persona deviation from objective

### Step 6.4: Update any hardcoded field names

Search for and update:

- `neutral_scores` → `objective_scores`
- `persona_adjustments` → remove or replace

---

## Phase 7: Update WalkingAgent Integration

### Step 7.1: Update \_get_persona_hint() method

**File:** `src/agent/walking_agent.py`

Review if this method is still needed or should return full persona object instead of just hint string.

### Step 7.2: Update ContinuousAnalyzer initialization

Pass full persona object instead of just hint:

```python
# OLD
self._continuous_analyzer = ContinuousAnalyzer(
    persona_hint=self._get_persona_hint()
)

# NEW
self._continuous_analyzer = ContinuousAnalyzer(
    persona=get_enhanced_personality(self.personality.personality_id)
)
```

### Step 7.3: Update run_with_memory() method

Ensure it uses persona_scores for decision making, not objective_scores

---

## Phase 8: Update Examples and Tests

### Step 8.1: Update dual_system_demo.py

**File:** `examples/dual_system_demo.py`

- Update terminology (neutral → objective)
- Update data access patterns
- Verify demo still works

### Step 8.2: Update test files

Search for tests that reference:

- `semantic_config`
- `weight_distribution`
- `dimension_weights`
- `neutral_scores`
- `persona_adjustments`

Update or remove as needed.

### Step 8.3: Create new test for dual evaluation

Test that:

- Both objective and persona scores are generated
- Persona prompt includes persona hint
- Objective prompt is unchanged from framework definition

---

## Phase 9: Documentation

### Step 9.1: Update SYSTEM_ARCHITECTURE.md

- Document new dual evaluation flow
- Explain objective vs persona distinction
- Remove weight-related documentation

### Step 9.2: Update docstrings

All modified functions/classes should have updated docstrings reflecting new behavior.

### Step 9.3: Create migration notes

Document breaking changes:

- `AgentPersonality.dimension_weights` removed
- `map_semantic_to_weights()` removed
- Field name changes in WaypointAnalysis

---

## Validation Checklist

After each phase, verify:

```bash
# Phase 1: Personality config
python -c "from src.agent.config import get_preset, EnhancedPersonalityConfig; p = get_preset('homebuyer', 'streetagent_5d'); print(p)"

# Phase 2: Prompt builder
python -c "from src.evaluation.persona_prompt_builder import PersonaPromptBuilder; print('OK')"

# Phase 3: Evaluator
python -c "from src.evaluation.evaluator import Evaluator; print('OK')"

# Phase 4: ContinuousAnalyzer
python -c "from src.analysis.continuous_analyzer import ContinuousAnalyzer; print('OK')"

# Phase 5: ThinkingModule
python -c "from src.agent.capabilities.thinking import ThinkingModule; print('OK')"

# Phase 6: Visualization
python -c "from src.utils.visualization import RouteVisualizer; print('OK')"

# Full integration test
python examples/dual_system_demo.py --help
```

---

## Summary of Deleted Code

| Item                        | Location                  | Reason                 |
| --------------------------- | ------------------------- | ---------------------- |
| `semantic_config`           | EnhancedPersonalityConfig | Weight system removed  |
| `weight_distribution`       | EnhancedPersonalityConfig | Weight system removed  |
| `dimension_weights`         | AgentPersonality          | Weight system removed  |
| `map_semantic_to_weights()` | personalities.py          | No longer needed       |
| `normalize_weights()`       | personalities.py          | No longer needed       |
| `get_primary_dimensions()`  | personalities.py          | Optional - may keep    |
| `persona_adjustments`       | WaypointAnalysis          | Independent evaluation |
| `persona_applied`           | WaypointAnalysis          | Always true if persona |
| Sequential dual-call logic  | ContinuousAnalyzer        | Replaced with parallel |

---

## Summary of New Code

| Item                        | Location                   | Purpose                 |
| --------------------------- | -------------------------- | ----------------------- |
| `PersonaPromptBuilder`      | persona_prompt_builder.py  | Build dual prompts      |
| `DualPrompts`               | persona_prompt_builder.py  | Container dataclass     |
| `DualEvaluationResult`      | evaluator.py               | Dual result container   |
| `evaluation_mode` param     | Evaluator.evaluate_image() | Control evaluation type |
| `objective_scores` field    | WaypointAnalysis           | Research scores         |
| `objective_reasoning` field | WaypointAnalysis           | Research reasoning      |

---

## Execution Order

1. **Phase 1** first - simplifies data structures
2. **Phase 2** next - creates new prompt builder
3. **Phase 3** - updates evaluator to use new builder
4. **Phase 4** - updates analyzer to use new evaluator
5. **Phase 5** - ensures System 2 works with new System 1 output
6. **Phase 6** - updates visualizations
7. **Phase 7** - integrates into WalkingAgent
8. **Phase 8** - updates examples and tests
9. **Phase 9** - documentation

Each phase should be independently testable before moving to next.
