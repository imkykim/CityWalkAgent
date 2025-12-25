# StreetAgent Framework Guide

**Last Updated**: 2025-11-25  
**Status**: Production Ready

---

## Overview

StreetAgent uses a **framework-agnostic architecture** that supports multiple walkability evaluation frameworks. Each framework defines a unique set of dimensions for assessing urban pedestrian experience.

---

## Default Framework

**Current Default**: `place_pulse_2.0`

All components (WalkingAgent, ThinkingModule, RouteVisualizer) default to `place_pulse_2.0` unless explicitly specified otherwise.

Default dimensions: safety, lively, beautiful, wealthy (Place Pulse 2.0).

---

## Supported Frameworks

### 1. **StreetAgent 5D** (`streetagent_5d`)

**Dimensions**: 5
**Focus**: Comprehensive spatial and experiential walkability

#### Dimensions

##### 1.1 **Functional Quality**

**What it measures**: Practical walkability, amenities, surface conditions

**Criteria**:

- Sidewalk width and condition
- Street furniture (benches, lighting, trash bins)
- Accessibility features (ramps, tactile paving)
- Hazards and obstructions
- Crossings and intersections
- Shade and weather protection

**Scale**:

- **1-3**: Poor conditions, major hazards
- **4-6**: Adequate but limited amenities
- **7-9**: Excellent conditions, comprehensive amenities
- **10**: Ideal pedestrian infrastructure

---

##### 1.2 **Spatial Legibility**

**What it measures**: Wayfinding, orientation, clarity of spatial structure

**Criteria**:

- Clear pathways and navigation
- Visible landmarks
- Intersection legibility
- Signage and markers
- Spatial organization
- Visual hierarchy

**Scale**:

- **1-3**: Confusing, disorienting
- **4-6**: Navigable with effort
- **7-9**: Clear, intuitive wayfinding
- **10**: Perfect spatial clarity

---

##### 1.3 **Visual Coherence**

**What it measures**: Aesthetic harmony, visual organization

**Criteria**:

- Building facade harmony
- Color palette consistency
- Architectural style coherence
- Visual rhythm
- Compositional balance
- Scale consistency

**Scale**:

- **1-3**: Chaotic, clashing elements
- **4-6**: Somewhat organized
- **7-9**: Harmonious, unified
- **10**: Perfect aesthetic coherence

---

##### 1.4 **Sensory Complexity**

**What it measures**: Visual richness, diversity, stimulation

**Criteria**:

- Variety of visual elements
- Texture and detail density
- Color diversity
- Pattern complexity
- Sensory engagement
- Visual interest

**Scale**:

- **1-3**: Monotonous, understimulating
- **4-6**: Moderate variety
- **7-9**: Rich, engaging
- **10**: Exceptionally stimulating

---

##### 1.5 **Spatial Sequence**

**What it measures**: Progressive revelation, journey experience

**Criteria**:

- View unfolding dynamics
- Spatial progression
- Surprise and discovery
- Rhythm of experience
- Journey narrative
- Anticipation and reveal

**Scale**:

- **1-3**: Static, no progression
- **4-6**: Some variation
- **7-9**: Engaging unfolding
- **10**: Perfect experiential sequence

---

### 2. **SAGAI 2025** (`sagai_2025`)

**Dimensions**: 4
**Focus**: Traditional walkability assessment

#### Dimensions

##### 2.1 **Safety**

Pedestrian security, traffic safety, crime prevention, lighting

##### 2.2 **Comfort**

Weather protection, seating, noise levels, crowd density, air quality

##### 2.3 **Interest**

Visual appeal, landmarks, activity, diversity, cultural richness

##### 2.4 **Aesthetics**

Beauty, cleanliness, maintenance, architectural quality

---

### 3. **Ewing & Handy 5D** (`ewing_handy_5d`)

**Dimensions**: 5
**Focus**: Urban design qualities

#### Dimensions

##### 3.1 **Imageability**

Memorable, distinct, recognizable environment

##### 3.2 **Enclosure**

Spatial definition by buildings, trees, walls

##### 3.3 **Human Scale**

Building proportions comfortable for pedestrians

##### 3.4 **Transparency**

Visual permeability, ability to see activity inside buildings

##### 3.5 **Complexity**

Visual richness, variety of elements

---

### 4. **Kaplan & Kaplan 4D** (`kaplan_4d`)

**Dimensions**: 4
**Focus**: Environmental preference theory

#### Dimensions

##### 4.1 **Coherence**

Organization, ease of comprehension

##### 4.2 **Complexity**

Visual richness, number of elements

##### 4.3 **Legibility**

Ease of wayfinding, clear structure

##### 4.4 **Mystery**

Promise of more information, hidden views

---

### 5. **Phenomenology 3D** (`phenomenology_3d`)

**Dimensions**: 3
**Focus**: Experiential and philosophical qualities

#### Dimensions

##### 5.1 **Atmosphere**

Emotional and sensory mood of the space

##### 5.2 **Embodiment**

Physical and bodily engagement with environment

##### 5.3 **Place Identity**

Sense of place, cultural meaning, distinctiveness

---

## Framework Comparison

| Framework            | Dimensions | Focus                   | Best For                        |
| -------------------- | ---------- | ----------------------- | ------------------------------- |
| **streetagent_5d**   | 5          | Spatial + Experiential  | Comprehensive urban analysis    |
| **sagai_2025**       | 4          | Traditional walkability | Safety/comfort focused projects |
| **ewing_handy_5d**   | 5          | Urban design            | Architectural/planning research |
| **kaplan_4d**        | 4          | Cognitive psychology    | Preference studies              |
| **phenomenology_3d** | 3          | Experiential            | Philosophical/cultural research |

---

## StreetAgent 5D - Personality Weights

### How Personalities Interpret Dimensions

Different personalities assign different importance (weights) to each dimension, reflecting diverse walking priorities.

#### Weight Legend

- **2.0+**: Primary focus
- **1.5-2.0**: Secondary focus
- **1.0-1.5**: Moderate consideration
- **0.5-1.0**: Lower priority
- **<0.5**: Minimal consideration

---

### Personality Weight Matrix

| Dimension              | Safety  | Scenic  | Balanced | Comfort | Explorer | Technical |
| ---------------------- | ------- | ------- | -------- | ------- | -------- | --------- |
| **Functional Quality** | **2.2** | 1.2     | 1.0      | **2.0** | 1.3      | 1.0       |
| **Spatial Legibility** | **1.5** | 1.0     | 1.0      | 1.3     | 1.0      | 1.0       |
| **Visual Coherence**   | 1.2     | **1.8** | 1.0      | **1.5** | 1.2      | 1.0       |
| **Sensory Complexity** | 0.7     | **2.0** | 1.0      | 0.8     | **1.8**  | 1.0       |
| **Spatial Sequence**   | 0.6     | **1.5** | 1.0      | 1.2     | **2.0**  | 1.0       |

---

### Personality Profiles

#### **Safety Guardian** (`safety`)

**Philosophy**: "Safety first, everything else follows"

**Weight Rationale**:

- **High functional_quality (2.2)**: Good infrastructure = safer walking
- **High spatial_legibility (1.5)**: Clear navigation reduces risk
- **Low sensory_complexity (0.7)**: Overstimulation = distraction = danger

**Typical User**: Parents with children, elderly, safety-conscious commuters

---

#### **Aesthetic Explorer** (`scenic`)

**Philosophy**: "Beauty enriches the journey"

**Weight Rationale**:

- **High sensory_complexity (2.0)**: Seeks rich visual stimuli
- **High visual_coherence (1.8)**: Values harmonious composition
- **High spatial_sequence (1.5)**: Appreciates unfolding views

**Typical User**: Tourists, photographers, architectural enthusiasts

---

#### **Balanced Navigator** (`balanced`)

**Philosophy**: "Consider everything equally"

**Weight Rationale**:

- All dimensions = 1.0 (no preferences)
- Objective, comprehensive assessment

**Typical User**: Urban planners, researchers, general public

---

#### **Comfort Seeker** (`comfort`)

**Philosophy**: "Make walking easy and pleasant"

**Weight Rationale**:

- **High functional_quality (2.0)**: Needs good conditions
- **High visual_coherence (1.5)**: Prefers calm, organized spaces
- **Low sensory_complexity (0.8)**: Avoids overwhelming stimuli

**Typical User**: People with mobility issues, stressed commuters

---

#### **Urban Explorer** (`explorer`)

**Philosophy**: "Discovery makes walking exciting"

**Weight Rationale**:

- **High spatial_sequence (2.0)**: Seeks progressive discovery
- **High sensory_complexity (1.8)**: Values variety and novelty
- **Moderate spatial_legibility (1.0)**: Mystery is acceptable

**Typical User**: Urban adventurers, flâneurs, experience seekers

---

#### **Technical Analyst** (`technical`)

**Philosophy**: "Objective evaluation across all criteria"

**Weight Rationale**:

- All dimensions = 1.0 (no bias)
- Comprehensive technical assessment

**Typical User**: Researchers, policy makers, consultants

---

## How Frameworks Work in StreetAgent

### Component Integration

#### 1. **ThinkingModule**

- Loads framework dimensions dynamically
- Generates framework-specific evaluation prompts
- Validates VLM scores against framework dimensions

**Code Example**:

```python
thinking = ThinkingModule(framework_id="streetagent_5d")
# Automatically loads 5 dimensions from framework config
```

---

#### 2. **RouteVisualizer**

- Dynamically builds dimension colors and labels
- Adapts plot layouts to dimension count (3D, 4D, 5D)
- Framework-agnostic legends

**Code Example**:

```python
viz = RouteVisualizer(framework_id="ewing_handy_5d")
# Plots will show: imageability, enclosure, human_scale, transparency, complexity
```

---

#### 3. **WalkingAgent**

- Passes framework_id to all components
- Ensures consistency across pipeline

**Code Example**:

```python
agent = WalkingAgent.from_preset(
    personality_name="balanced",
    framework_id="kaplan_4d"
)
# Entire pipeline uses Kaplan's 4D framework
```

---

### Framework Selection Strategy

#### Use **streetagent_5d** when:

- Need comprehensive spatial and experiential assessment
- Want to capture journey progression (spatial_sequence)
- Focus on both objective and subjective qualities
- Rich personality differentiation desired

#### Use **sagai_2025** when:

- Traditional walkability metrics sufficient
- Safety and comfort are primary concerns
- Comparing with existing SAGAI research
- Simpler 4D analysis preferred

#### Use **ewing_handy_5d** when:

- Urban design research focus
- Architectural qualities emphasized
- Academic urban planning projects
- Comparing with Ewing & Handy literature

#### Use **kaplan_4d** when:

- Cognitive/preference research
- Mystery and discovery important
- Environmental psychology studies
- Comparing with Kaplan & Kaplan work

#### Use **phenomenology_3d** when:

- Philosophical/cultural analysis
- Sense of place critical
- Qualitative experiential research
- Minimal quantitative metrics desired

---

## Usage Examples

### Default Framework (Place Pulse 2.0)

```python
from src.agent.walking_agent import WalkingAgent

# Uses place_pulse_2.0 by default
agent = WalkingAgent.from_preset("balanced")
assert agent.framework_id == "place_pulse_2.0"
assert len(agent.thinking_module.dimension_ids) == 4
```

---

### Explicit Framework Selection

```python
# Safety-focused with SAGAI 4D
safety_agent = WalkingAgent.from_preset("safety", "sagai_2025")

# Aesthetic with Ewing & Handy
scenic_agent = WalkingAgent.from_preset("scenic", "ewing_handy_5d")

# Exploratory with Kaplan
explorer_agent = WalkingAgent.from_preset("explorer", "kaplan_4d")
```

---

### Multi-Framework Comparison

```python
frameworks = ["place_pulse_2.0", "streetagent_5d", "sagai_2025", "kaplan_4d"]
agents = {
    fw: WalkingAgent.from_preset("balanced", fw, f"agent_{fw}")
    for fw in frameworks
}

# Run same route with different frameworks
results = {}
for fw, agent in agents.items():
    results[fw] = await agent.run_with_memory(start, end)

# Compare dimensional perspectives
```

---

### Framework-Specific Visualization

```python
from src.utils.visualization import RouteVisualizer

# Visualize with phenomenology framework
viz = RouteVisualizer(framework_id="phenomenology_3d")
viz.plot_scores_with_trends({
    "atmosphere": [7.0, 7.5, 8.0],
    "embodiment": [6.5, 7.0, 7.5],
    "place_identity": [8.0, 8.5, 9.0]
})
```

---

## Framework Configuration Format

**Location**: `src/frameworks/*.json`

**Structure**:

```json
{
  "framework_id": "streetagent_5d",
  "name": "StreetAgent 5D Framework",
  "version": "1.0",
  "description": "Comprehensive spatial and experiential walkability assessment",
  "dimensions": [
    {
      "id": "functional_quality",
      "name": "Functional Quality",
      "description": "Practical walkability, amenities, surface conditions",
      "weight": 1.0
    }
  ],
  "criteria": {
    "functional_quality": [
      "Sidewalk width and condition",
      "Street furniture availability",
      "Accessibility features"
    ]
  },
  "scale": {
    "min": 1,
    "max": 10,
    "interpretation": {
      "1-3": "Poor",
      "4-6": "Adequate",
      "7-9": "Good",
      "10": "Excellent"
    }
  }
}
```

---

## Creating Custom Frameworks

### Steps

1. **Create JSON file**: `src/frameworks/my_framework.json`
2. **Define dimensions**: At least 3, maximum 10 recommended
3. **Specify criteria**: List evaluation criteria per dimension
4. **Add personality weights**: In `src/agent/config/constants.py`
5. **Test**: Use `test_framework_agnostic.py` pattern

### Example Custom Framework

```json
{
  "framework_id": "accessibility_3d",
  "name": "Accessibility-Focused Framework",
  "dimensions": [
    {
      "id": "physical_access",
      "name": "Physical Accessibility",
      "description": "Wheelchair access, ramps, curb cuts"
    },
    {
      "id": "sensory_access",
      "name": "Sensory Accessibility",
      "description": "Visual/audio aids, tactile paving"
    },
    {
      "id": "cognitive_access",
      "name": "Cognitive Accessibility",
      "description": "Clear signage, simple navigation"
    }
  ]
}
```

---

## Backward Compatibility

### Legacy Code

All existing code using `sagai_2025` continues to work unchanged. Framework-agnostic design maintains full backward compatibility.

### Migration Path

```python
# Old (still works)
agent = WalkingAgent.from_preset("balanced", "sagai_2025")

# New (uses place_pulse_2.0)
agent = WalkingAgent.from_preset("balanced", "place_pulse_2.0")

# Or rely on default
agent = WalkingAgent.from_preset("balanced")  # Uses place_pulse_2.0
```

---

## Framework Selection Decision Tree

```
START: What's your primary goal?
│
├─ Comprehensive spatial + experiential analysis with empirical grounding?
│  └─> place_pulse_2.0
│
├─ Traditional safety/comfort focus?
│  └─> sagai_2025
│
├─ Urban design/architectural research?
│  └─> ewing_handy_5d
│
├─ Environmental preference studies?
│  └─> kaplan_4d
│
└─ Philosophical/cultural analysis?
   └─> phenomenology_3d
```

---

## Best Practices

### 1. **Consistency Within Projects**

Use the same framework throughout a project for comparability

### 2. **Document Framework Choice**

Always document why a particular framework was selected

### 3. **Personality-Framework Pairing**

- Safety Guardian → sagai_2025 or streetagent_5d
- Aesthetic Explorer → streetagent_5d or ewing_handy_5d
- Urban Explorer → kaplan_4d or streetagent_5d

### 4. **Multi-Framework Analysis**

For comprehensive research, evaluate with multiple frameworks to capture different perspectives

### 5. **Framework Validation**

Always validate custom frameworks with pilot studies before large-scale deployment

---

**Status**: Production Ready | **Version**: 1.0 | **Last Updated**: 2025-11-25
