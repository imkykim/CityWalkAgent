# StreetAgent System Architecture

**Last Updated**: 2025-11-25  
**Status**: Production Ready

---

## Overview

StreetAgent is an AI-powered pedestrian experience analysis platform that uses a **dual-system cognitive architecture** to evaluate urban walkability through continuous street-level imagery analysis.

---

## Core Architecture

### Dual-System Cognitive Model

```
┌──────────────────┐
│ WalkingAgent     │
└────────┬─────────┘
         │
         ├─> CognitiveController (pHash, trigger decisions)
         │   ├─> Triggers multi-image evaluation (Layer 2)
         │   └─> Triggers System 2 reasoning
         │
         ├─> MemoryManager
         │   ├─> Attention Gate (filter waypoints)
         │   ├─> STM (recent context)
         │   ├─> LTM (narrative, patterns)
         │   └─> Sequential Context (_system2_results)
         │
         └─> ThinkingModule
             ├─> VLM Score Revision (System 2)
             ├─> Narrative Generation
             └─> Memory Integration
```

---

## Component Roles

### 1. **WalkingAgent** - Main Orchestrator

**File**: `src/agent/walking_agent.py`

**Purpose**: Coordinates the entire evaluation pipeline

**Responsibilities**:

- Route generation and waypoint sampling
- Image collection from Google Street View/Mapillary
- Orchestrates System 1 and System 2 analysis
- Manages memory systems (STM, LTM)
- Outputs results (JSON, visualizations, narratives)

**Key Properties**:

- `framework_id`: Evaluation framework (e.g., "place_pulse_2.0")
- `personality`: Agent personality (e.g., "safety", "scenic")
- `memory_manager`: Lazy-loaded memory coordination
- `thinking_module`: Lazy-loaded System 2 reasoning

---

### 2. **CognitiveController** - Evaluation Strategy Controller

**File**: `src/agent/cognitive_controller.py`

**Purpose**: Decides evaluation strategy based on visual and score changes

**Two Types of Triggers**:

#### A. Multi-Image Evaluation Trigger (Layer 2 Sequential)

**When**: Significant visual change detected

**Conditions**:

- pHash distance > threshold (typically 15-20)
- Indicates meaningful environmental transition

**Action**: Use **2-image VLM comparison** instead of single-image evaluation

**Example**:

```
pHash distance = 18 → Trigger multi-image
→ Send both previous AND current image to VLM
→ VLM directly compares visual changes
→ More accurate transition analysis
```

#### B. System 2 Reasoning Trigger (Layer 3 Sequential)

**When**: Deep contextual reasoning needed

**Conditions**:

- **Major Visual Change**: pHash distance > 20
- **Score Anomaly**: Scores deviate significantly from patterns
- **Low Quality**: Scores fall below minimum thresholds
- **Distance Milestones**: Every N meters (e.g., 500m intervals)

**Action**: Activate **ThinkingModule** for LLM-based contextual re-evaluation

**Example**:

```
pHash distance = 25 AND score dropped 2.5 points
→ Trigger System 2
→ ThinkingModule gets full context (STM + LTM + previous S2)
→ Revises scores with deep reasoning
→ Updates STM
```

**Process Flow**:

```
1. Analyze waypoint
2. Compute pHash distance
3. Check visual change threshold
   ├─> If 15-20: Trigger multi-image evaluation
   └─> If 20+: Also consider System 2
4. Check score patterns
   └─> If anomaly detected: Trigger System 2
5. Execute triggered evaluation strategy
```

**Why Separate Controller**:

- Decouples triggering logic from memory management
- Enables flexible triggering strategies
- Can use different controllers for different use cases

---

### 3. **MemoryManager** - Memory Coordination Hub

**File**: `src/agent/capabilities/memory_manager.py`

**Purpose**: Manages information flow between perception, memory, and reasoning

#### Three-Layer Memory Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      MemoryManager                              │
│                    (Coordination Hub)                           │
└────────┬────────────────────────────┬────────────────────┬──────┘
         │                            │                    │
         ▼                            ▼                    ▼
┌──────────────────┐       ┌──────────────────┐  ┌────────────────┐
│ Short-Term       │       │ Long-Term        │  │ Sequential     │
│ Memory (STM)     │       │ Memory (LTM)     │  │ Context        │
│                  │       │                  │  │                │
│ - Last 5-10      │       │ - Key moments    │  │ - System 2     │
│   waypoints      │       │ - Patterns       │  │   results      │
│ - Scores         │       │ - Narratives     │  │ - Previous     │
│ - Summaries      │       │ - Route summary  │  │   reasoning    │
│ - Visual changes │       │ - Statistics     │  │ - Continuity   │
└──────────────────┘       └──────────────────┘  └────────────────┘
```

#### 3.1 Attention Gate

**Function**: Filters which waypoints enter short-term memory

**Filtering Criteria**:

- Visual change exceeds threshold (pHash)
- Score volatility (change from previous)
- Low scores (below quality threshold)
- Milestones (every N waypoints)

**Pass Rate**: ~40-50% of waypoints (configurable)

#### 3.2 Short-Term Memory (STM)

**Function**: Maintains recent walking experience context

**Contents**:

- Last N waypoints (default: 5-10)
- Scores (System 1 or revised System 2)
- Summaries of each waypoint
- Visual change indicators

**Updates**: When System 2 revises scores, STM is updated retroactively

#### 3.3 Long-Term Memory (LTM)

**Function**: Stores route-level patterns and narratives

**Contents**:

- Key moments (significant waypoints)
- Extracted patterns (recurring themes)
- Narrative chapters (episodic stories)
- Route statistics

#### 3.4 Sequential Context

**Function**: Tracks System 2 results for continuity

**Storage**: `_system2_results` dictionary maps waypoint_id → ThinkingResult

**Purpose**: Next System 2 evaluation gets previous System 2 context, creating rich sequential reasoning

**Example Flow**:

```
Waypoint 1: System 1 only → STM contains System 1 scores
Waypoint 5: System 2 triggered → Gets STM + previous System 2 results → Revises scores → Updates STM
Waypoint 10: System 2 triggered → Gets updated STM + Waypoint 5's System 2 results → Continues reasoning
```

---

### 4. **ThinkingModule** - System 2 Reasoning

**File**: `src/agent/capabilities/thinking.py`

**Purpose**: Slow, context-aware reasoning using Vision-Language Models (VLM)

#### 4.1 VLM Score Revision

**Function**: Re-evaluates waypoint with memory context

**Inputs**:

- Current waypoint image
- System 1 scores (initial perception)
- STM context (recent experience)
- LTM patterns (past insights)
- Previous System 2 results (sequential reasoning)
- Personality traits
- Trigger reason

**Process**:

1. Build comprehensive prompt with all context
2. Send to VLM (e.g., Qwen Vision-Language Model)
3. Parse revised scores and reasoning
4. Calculate adjustments (delta from System 1)
5. Return ThinkingResult with all metadata

**Output Structure** (`ThinkingResult`):

```python
@dataclass
class ThinkingResult:
    revised_scores: Dict[str, float]           # System 2 scores
    score_adjustments: Dict[str, float]        # Delta from System 1
    revision_reasoning: Dict[str, str]         # Per-dimension explanation
    memory_influence: Dict[str, str]           # How STM/LTM affected revision
    used_stm_context: bool                     # Flag
    used_ltm_patterns: bool                    # Flag
    personality_factor: float                  # Personality influence
    vlm_model_used: str                        # Model identifier
    system1_scores: Dict[str, float]           # Original scores
    processing_time_seconds: float             # Performance metric
```

#### 4.2 Narrative Generation

**Function**: Creates episodic walking experience stories

**Progressive Narrative System**:

- Generates chapter-by-chapter narrative
- Each chapter references previous chapters
- Builds cumulative walking story
- Tracks emotional tone progression

**Narrative Context**:

- Last 3 chapters (for continuity)
- Current waypoint scores
- Route segment characteristics
- Emotional trajectory

**Output**: Stored in LTM as `NarrativeChapter` objects

---

### 5. **ContinuousAnalyzer** - System 1 Perception

**File**: `src/analysis/continuous_analyzer.py`

**Purpose**: Fast, initial perception of each waypoint

**Process**:

1. Receives street-level image
2. Applies evaluation framework (e.g., place_pulse_2.0)
3. VLM generates initial scores (System 1)
4. Computes pHash for visual change detection
5. Returns WaypointAnalysis

**Evaluation Modes** (controlled by CognitiveController):

- **Single-image + text context** (default, ~85-90% of waypoints)
- **Multi-image comparison** (when visual change detected, ~10-15% of waypoints)

**Speed**:

- Single-image: ~0.1 seconds per image
- Multi-image: ~0.15 seconds per evaluation

---

### 6. **RouteVisualizer** - Framework-Agnostic Visualization

**File**: `src/utils/visualization.py`

**Purpose**: Generates visual analysis of route evaluation

**Key Methods**:

#### `plot_dual_system_comparison()`

Shows System 1 vs System 2 scores with adjustment arrows

#### `plot_narrative_timeline()`

Timeline with narrative chapter markers

#### `plot_dual_system_analysis()`

Complete dashboard with:

- Dual-system score comparison
- Adjustment heatmap
- Statistics boxes
- Narrative timeline

**Framework Support**: Dynamically adapts to any framework's dimensions

---

## Sequential Context Integration

### Three-Layer Sequential Processing

StreetAgent implements a sophisticated sequential awareness system with three layers:

#### Layer 1: Previous Waypoint Context (All System 1 Evaluations)

**What**: Every waypoint (except first) receives text context about its predecessor

**Context Structure**:

```python
previous_context = {
    "waypoint_id": 45,
    "scores": {"functional_quality": 7.5, ...},
    "reasoning": {"functional_quality": "Wide sidewalk...", ...},
    "phash_distance": 12.5,
    "visual_change_detected": True
}
```

**How it works**:

- ContinuousAnalyzer builds this context from previous waypoint
- Prepends to VLM prompt as text
- VLM evaluates current waypoint with knowledge of predecessor

**Benefits**:

- VLM understands progression, not just absolute state
- Detects relative changes and transitions
- ~0% cost increase (just longer prompt)

---

#### Layer 2: Multi-Image Comparison (Triggered by CognitiveController)

**Trigger**: CognitiveController detects significant visual change (pHash > threshold)

**What**: VLM receives **both images** (previous + current) for direct visual comparison

**Comparison Prompt Structure**:

```markdown
# Sequential Waypoint Comparison

**Previous Waypoint (ID: 45)**

- Previous score: 7.5/10
- Visual change detected: Yes (pHash: 18.5)

**Images:**
Image 1 (Previous): First image
Image 2 (Current): Second image

**Task:**

1. Compare images directly
2. Identify what changed
3. Determine if change improves/worsens quality
4. Score the CURRENT waypoint
```

**Process**:

```
CognitiveController: pHash = 18 > threshold (15)
→ Trigger multi-image evaluation
→ ContinuousAnalyzer uses evaluate_with_comparison()
→ VLM receives 2 images + comparison prompt
→ Returns scores based on visual comparison
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    ContinuousAnalyzer                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │  Waypoint N  │───>│ pHash Check  │───>│Visual Change│     │
│  └──────────────┘    └──────────────┘    │  Detected?  │     │
│                                           └─────┬───────┘     │
│                                                 │             │
│                                    ┌────────────┴─────────┐   │
│                                    │                      │   │
│                              Yes   │                 No   │   │
│                                    ▼                      ▼   │
│                      ┌──────────────────────┐  ┌──────────────┐
│                      │Multi-Image Evaluation│  │Single-Image  │
│                      │   (2 images to VLM)  │  │  Evaluation  │
│                      └──────────────────────┘  │(with context)│
│                                                └──────────────┘
└─────────────────────────────────────────────────────────────────┘
```

**Benefits**:

- More accurate than text-only context
- Detects subtle visual changes VLM might miss with text alone
- Better transition understanding

**Cost**: +5-15% (triggered on ~10-15% of waypoints only)

---

#### Layer 3: System 2 Sequential Context (Triggered by CognitiveController)

**Trigger**: CognitiveController decides deep reasoning needed

**What**: ThinkingModule receives full context including previous System 2 results

**Why This Matters**:

```
Without: Each System 2 evaluation is isolated
With: System 2 evaluations form a reasoning chain

Example:
Waypoint 40 (S2): "Construction detected, temporary barrier"
Waypoint 45 (S2): "Construction continues, pedestrian detour"
Waypoint 50 (S2): "Construction resolved, normal conditions restored"
```

**Process**:

```
CognitiveController: Major change + score anomaly detected
→ Trigger System 2
→ MemoryManager prepares context:
  - Current waypoint
  - STM (with revised scores from previous S2)
  - Previous System 2 results (waypoint 40, 45)
  - LTM patterns
→ ThinkingModule reasons with full context
→ Returns ThinkingResult
→ MemoryManager updates STM + stores in sequential context
```

**Benefits**:

- Tracks persistent issues across multiple evaluations
- Detects resolution of problems
- Maintains narrative coherence
- Builds on previous deep reasoning

---

## Visual Change Detection

### pHash (Perceptual Hash)

**What**: Fingerprint of image content for similarity comparison

**Distance Scale**:

- **0-5**: Nearly identical
- **5-15**: Similar scene (few steps forward)
- **15-25**: Noticeable change (different block) → **Triggers Layer 2**
- **25-40**: Significant change (neighborhood transition) → **May trigger System 2**
- **40+**: Completely different environment → **Likely triggers System 2**

**Usage**:

- **CognitiveController** uses pHash to decide:
  - Whether to trigger multi-image evaluation (Layer 2)
  - Whether visual change warrants System 2 reasoning (Layer 3)
- Informs attention gate in MemoryManager
- Provides context to VLM in all evaluations

### Adaptive Thresholding

**Problem**: Fixed threshold doesn't work for all route types

**Solution**: Dynamically adjust based on route characteristics

**Algorithm**:

```python
threshold = mean(recent_distances) + 1.5 * std(recent_distances)
```

**Benefits**:

- Urban areas: Higher threshold (more variation is normal)
- Suburban areas: Lower threshold (less variation)
- Adapts automatically

---

## System Flow

### Complete Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│ 1. ROUTE GENERATION & IMAGE COLLECTION                         │
│    WalkingAgent → ZenSVI → Google Street View/Mapillary        │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. SYSTEM 1 ANALYSIS (Fast Perception)                         │
│    ContinuousAnalyzer → Compute pHash                          │
│    CognitiveController → Check visual change threshold         │
│      ├─> If pHash > 15: Multi-image VLM comparison (Layer 2)  │
│      └─> Else: Single-image VLM + text context (Layer 1)      │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. MEMORY PROCESSING                                            │
│    MemoryManager → Attention Gate → STM (if passes)            │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. SYSTEM 2 TRIGGER DECISION                                    │
│    CognitiveController → Check System 2 triggers                │
│      - Major visual change (pHash > 20)                        │
│      - Score anomaly detected                                   │
│      - Persistent low scores                                    │
│      - Transitions detected                                     │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼ (if triggered)
┌────────────────────────────────────────────────────────────────┐
│ 5. SYSTEM 2 REASONING (Contextual Re-evaluation) - Layer 3     │
│    ThinkingModule → VLM + Full Context → Revised Scores        │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ 6. MEMORY UPDATE                                                │
│    MemoryManager → Update STM with System 2 scores              │
│                 → Store in Sequential Context                   │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ 7. ROUTE CONSOLIDATION                                          │
│    MemoryManager → LTM → Extract patterns + Generate narrative │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ 8. OUTPUT GENERATION                                            │
│    - analysis_results.json (both System 1 & 2 scores)          │
│    - narrative.jsonl (chapters)                                 │
│    - complete_narrative.md (compiled story)                     │
│    - Visualizations (dual-system plots)                         │
└────────────────────────────────────────────────────────────────┘
```

---

## CognitiveController Decision Logic

### Decision Tree

```
New Waypoint Arrives
    │
    ├─> Compute pHash distance from previous
    │
    ├─> Decision 1: Evaluation Mode (Layer 1 vs Layer 2)
    │   │
    │   ├─> pHash < 15: Single-image + text context (Layer 1)
    │   │   └─> Standard System 1 evaluation
    │   │
    │   └─> pHash ≥ 15: Multi-image comparison (Layer 2)
    │       └─> Send 2 images to VLM for direct comparison
    │
    ├─> Get evaluation results (scores, reasoning)
    │
    ├─> Decision 2: System 2 Activation (Layer 3)
    │   │
    │   ├─> Check triggers:
    │   │   ├─> pHash > 20? (Major visual change)
    │   │   ├─> Score delta > threshold? (Anomaly)
    │   │   ├─> 3+ consecutive low scores? (Persistent issue)
    │   │   └─> STM trend "volatile"? (Transition)
    │   │
    │   ├─> If ANY trigger = True:
    │   │   └─> Activate ThinkingModule (System 2)
    │   │
    │   └─> Else:
    │       └─> Continue with System 1 scores
    │
    └─> Return results to WalkingAgent
```

### Trigger Statistics Target

**Layer 2 (Multi-image)**: ~10-15% of waypoints

- Triggered by visual changes
- Modest cost increase (~1.5x per evaluation)
- Net cost: +5-15%

**Layer 3 (System 2)**: ~15-25% of waypoints

- Triggered by major changes or anomalies
- Expensive LLM reasoning
- Net cost: +50-100% (but only on 20% of waypoints)

**Combined Strategy**: Smart resource allocation

- Most waypoints (70-80%): Layer 1 only (cheapest)
- Some waypoints (10-15%): Layer 1 + Layer 2 (moderate)
- Few waypoints (15-25%): Layer 1 + Layer 3 (expensive)
- Rare waypoints (~5%): All three layers (most expensive)

---

## Attention Gate Details

### Purpose

**First-stage filter** that determines which waypoints enter Short-Term Memory.

### Architecture

```
Every Waypoint (100%)
        ↓
Attention Gate (~50% pass)
        ↓
Short-Term Memory
        ↓
CognitiveController Evaluation
├─> Layer 2 Multi-image (~10-15%)
└─> Layer 3 System 2 (~15-25%)
```

### Filtering Criteria

#### 1. Visual Change

**Trigger**: pHash distance > threshold

**Personality Influence**: Scenic personalities have lower thresholds (more sensitive)

#### 2. Score Anomaly

**Trigger**: Current scores deviate from STM average

**Calculation**:

```python
deviation = abs(current_score - stm_average)
if deviation > threshold: passes_gate = True
```

#### 3. Low Quality Score

**Trigger**: Any dimension below minimum threshold

**Thresholds by Personality**:

- Safety Guardian: 6.5/10
- Comfort Seeker: 6.0/10
- Balanced Navigator: 5.5/10
- Urban Explorer: 5.0/10

#### 4. Distance Milestones

**Trigger**: Every Nth waypoint (default: every 10th)

**Purpose**: Ensures STM has regular updates even in monotonous areas

---

## Framework System

### Framework-Agnostic Design

StreetAgent supports **any evaluation framework** through dynamic configuration loading.

**Supported Frameworks**:

1. **place_pulse_2.0** (4D, default): safety, lively, beautiful, wealthy
2. **sagai_2025** (4D): safety, comfort, interest, aesthetics
3. **streetagent_5d** (5D): functional_quality, spatial_legibility, visual_coherence, sensory_complexity, spatial_sequence
4. **ewing_handy_5d** (5D): imageability, enclosure, human_scale, transparency, complexity
5. **kaplan_4d** (4D): coherence, complexity, legibility, mystery
6. **phenomenology_3d** (3D): atmosphere, embodiment, place_identity

**Default**: `place_pulse_2.0` (empirically validated perceptual assessment)

### Component Framework Integration

**ThinkingModule**:

- Loads framework dimensions
- Generates framework-specific prompts
- Validates scores against framework

**RouteVisualizer**:

- Dynamically builds dimension colors/labels
- Adapts plot layouts to dimension count
- Framework-agnostic legends

**WalkingAgent**:

- Passes framework_id to all components
- Ensures consistency across pipeline

```
┌────────────────────────────────────────────────┐
│ Framework Configuration (JSON)                 │
│ - framework_id, name, dimensions, criteria     │
└─────────────────┬──────────────────────────────┘
                  │
                  │ load_framework()
                  │
    ┌─────────────┴──────────────┬─────────────────────┬──────────────────┐
    │                            │                     │                  │
    ▼                            ▼                     ▼                  ▼
┌─────────────┐          ┌──────────────┐     ┌──────────────┐   ┌──────────────┐
│ Thinking    │          │ Route        │     │ Walking      │   │ Memory       │
│ Module      │          │ Visualizer   │     │ Agent        │   │ Manager      │
└─────────────┘          └──────────────┘     └──────────────┘   └──────────────┘
│                        │                     │                  │
├─ Dynamic prompts      ├─ Dynamic colors     ├─ Framework ID    ├─ Semantic
├─ Dimension templates  ├─ Dynamic labels     ├─ Personality     │   matching
├─ VLM integration      ├─ Framework plots    └─ Integration     └─ Agnostic
└─ Fallback handling    └─ Legend generation                         thresholds
```

---

## Personality System

### 6 Personalities (StreetAgent 5D)

#### 1. **Safety Guardian** (`safety`)

**Focus**: Pedestrian safety and functional quality

**Weights**: functional_quality (2.2), spatial_legibility (1.5), sensory_complexity (0.7)

#### 2. **Aesthetic Explorer** (`scenic`)

**Focus**: Visual beauty and sensory richness

**Weights**: sensory_complexity (2.0), visual_coherence (1.8), spatial_sequence (1.5)

#### 3. **Balanced Navigator** (`balanced`)

**Focus**: Equal consideration across all dimensions

**Weights**: All dimensions = 1.0

#### 4. **Comfort Seeker** (`comfort`)

**Focus**: Ease, amenities, smooth experience

**Weights**: functional_quality (2.0), visual_coherence (1.5), sensory_complexity (0.8)

#### 5. **Urban Explorer** (`explorer`)

**Focus**: Discovery, variety, engaging sequences

**Weights**: spatial_sequence (2.0), sensory_complexity (1.8), functional_quality (1.3)

#### 6. **Technical Analyst** (`technical`)

**Focus**: Comprehensive objective evaluation

**Weights**: All dimensions = 1.0

---

## Key Design Principles

### 1. Separation of Concerns

- **MemoryManager**: Manages memory, doesn't decide evaluation strategy
- **CognitiveController**: Decides evaluation strategy, doesn't manage memory
- **ThinkingModule**: Performs deep reasoning, doesn't manage triggers or memory

### 2. Sequential Context

System 2 evaluations build on previous System 2 reasoning, not just System 1 scores

### 3. Framework Agnostic

Components adapt to any evaluation framework dynamically

### 4. Lazy Loading

Components (ThinkingModule, MemoryManager) are initialized only when needed

### 5. Progressive Narrative

Narrative builds chapter-by-chapter with continuity, not as single summary

### 6. Adaptive Resource Allocation

CognitiveController intelligently allocates expensive operations (multi-image, System 2) only when beneficial

---

## Performance Characteristics

### System 1 (ContinuousAnalyzer)

- **Single-image**: ~0.1 seconds per image (85-90% of waypoints)
- **Multi-image**: ~0.15 seconds per evaluation (10-15% of waypoints)
- **Coverage**: 100% of waypoints
- **Purpose**: Fast initial perception

### System 2 (ThinkingModule)

- **Speed**: ~2-5 seconds per evaluation
- **Coverage**: ~20% of waypoints (triggered by CognitiveController)
- **Purpose**: Deep contextual reasoning

### Memory Manager

- **Overhead**: <10ms per waypoint
- **Attention Gate Pass Rate**: ~40-50%
- **STM Size**: 5-10 waypoints (sliding window)

### Overall Pipeline

- **1km route**: ~100 waypoints = 400 images
- **Processing Time**: ~4 minutes
- **Layer 2 (Multi-image)**: ~10 waypoints (10%)
- **Layer 3 (System 2)**: ~20 waypoints (20%)
- **Both Layer 2 + 3**: ~5 waypoints (5%)

---

## Output Structure

### Files Generated

#### `analysis_results.json`

```json
{
  "route_id": "route_001",
  "framework_id": "streetagent_5d",
  "personality": "balanced",
  "waypoints": [
    {
      "waypoint_id": 1,
      "system1_scores": { "functional_quality": 7.0, ... },
      "evaluation_mode": "single_image",
      "system2_triggered": false
    },
    {
      "waypoint_id": 5,
      "system1_scores": { "functional_quality": 6.5, ... },
      "evaluation_mode": "multi_image",
      "visual_change_detected": true,
      "phash_distance": 18.5,
      "system2_triggered": false
    },
    {
      "waypoint_id": 10,
      "system1_scores": { "functional_quality": 6.0, ... },
      "system2_scores": { "functional_quality": 6.5, ... },
      "evaluation_mode": "multi_image",
      "system2_triggered": true,
      "adjustments": { "functional_quality": +0.5, ... }
    }
  ]
}
```

#### `narrative.jsonl`

```jsonl
{"chapter_id": 1, "waypoint_range": [1, 20], "narrative": "...", "emotional_tone": "positive"}
{"chapter_id": 2, "waypoint_range": [21, 40], "narrative": "...", "emotional_tone": "neutral"}
```

#### `complete_narrative.md`

Compiled walking experience story with all chapters

#### Visualizations

- Dual-system comparison plots
- Score adjustment heatmaps
- Narrative timeline
- Statistics dashboards

---

## Usage Example

```python
from src.agent.walking_agent import WalkingAgent

# Create agent with personality and framework
agent = WalkingAgent.from_preset(
    personality_name="balanced",
    framework_id="streetagent_5d",
    agent_id="route_analysis_001"
)

# Run analysis on route
results = await agent.run_with_memory(
    start_coords=(37.7749, -122.4194),
    end_coords=(37.7849, -122.4094),
    interval_meters=10
)

# Results include:
# - System 1 scores (all waypoints)
# - Multi-image evaluations (visual change points)
# - System 2 scores (triggered waypoints)
# - Progressive narrative chapters
# - Visualizations
```

---

## Demo Script

**Location**: `examples/dual_system_demo.py`

Demonstrates complete dual-system pipeline with multi-image evaluation, System 2 triggering, score revision, and narrative generation.

---

**Status**: Production Ready | **Version**: 1.0 | **Last Updated**: 2025-11-25
