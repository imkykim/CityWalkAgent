# CityWalkAgent Examples

This directory contains example scripts demonstrating how to use the CityWalkAgent production framework.

## Available Examples

### 1. Simple Pipeline Usage (`01_simple_pipeline_usage.py`)

**What it demonstrates:**
- Full end-to-end pipeline usage
- Route generation from coordinates
- VLM evaluation
- Sequential analysis
- Results export

**Usage:**
```bash
python examples/01_simple_pipeline_usage.py
```

**Requirements:**
- VLM API key configured in `.env`
- Google Maps API key (for image collection)

**Output:**
- Route data
- Evaluation results
- Sequential analysis
- Method comparison
- Summary reports

---

### 2. Evaluation Only (`02_evaluation_only.py`)

**What it demonstrates:**
- Using just the evaluation engine
- Direct image evaluation without full pipeline
- Custom framework selection
- Accessing dimension scores

**Usage:**
```bash
python examples/02_evaluation_only.py
```

**Requirements:**
- VLM API key
- Existing street view images

**Use cases:**
- Evaluating pre-collected images
- Testing different frameworks
- Quick walkability assessment

---

### 3. Analysis Only (`03_analysis_only.py`)

**What it demonstrates:**
- Using just the analysis engine
- Loading existing evaluation results
- Sequential vs aggregate comparison
- Barrier detection and transitions

**Usage:**
```bash
python examples/03_analysis_only.py
```

**Requirements:**
- Existing route file
- Existing evaluation results

**Use cases:**
- Re-analyzing with different thresholds
- Comparing analysis methods
- Research and experimentation

---

### 4. Agent Analysis (`04_agent_analysis.py`)

**What it demonstrates:**
- WalkingAgent with different personalities
- Full cognitive pipeline: Observe → Analyze → Think → Act → Remember
- Real route analysis (not mocked)
- Personality-driven decision making
- Agent state and memory management
- Cache functionality for efficiency

**Usage:**
```bash
python examples/04_agent_analysis.py
```

**Requirements:**
- VLM API key (Claude recommended)
- Google Maps API key (optional, uses simple routes if not available)

**Use cases:**
- Comparing how different agent personalities evaluate the same route
- Understanding the complete agent cognitive pipeline
- Demonstrating personality-based reasoning
- Testing agent memory and state management

**Output:**
- Agent configurations and weights
- Full cognitive pipeline execution details
- Personality-based decisions and explanations
- Agent comparison table
- Memory statistics

---

## Quick Start

### Option 1: Full Pipeline (Recommended)

```python
from src.pipeline import WalkingAgentPipeline

# Initialize
pipeline = WalkingAgentPipeline(
    framework_id="sagai_2025",
    vlm_provider="claude",
    vlm_api_key="your-key"
)

# Analyze route
result = pipeline.analyze_route(
    start=(40.7589, -73.9851),
    end=(40.7614, -73.9776),
    interval_meters=10
)

# Access results
print(f"Aggregate: {result.sequential_analysis['aggregate_score']:.2f}")
print(f"Sequential: {result.sequential_analysis['sequential_score']:.2f}")
```

### Option 2: Evaluation Only

```python
from src.evaluation import Evaluator
from src.evaluation.vlm_client import VLMConfig, VLMProvider
from src.config import load_framework

# Load framework
framework = load_framework("sagai_2025")

# Configure VLM
vlm_config = VLMConfig(
    provider=VLMProvider.CLAUDE,
    api_key="your-key",
    model="claude-sonnet-4-20250514"
)

# Evaluate
evaluator = Evaluator(vlm_config, framework)
results = evaluator.evaluate_images(image_paths)
```

### Option 3: Analysis Only

```python
from src.analysis import SequentialAnalyzer

# Load route and evaluations (from files)
# ...

# Analyze
analyzer = SequentialAnalyzer(route, evaluations)
analysis = analyzer.full_analysis()

print(f"Pattern: {analysis.pattern_type}")
print(f"Barriers: {len(analysis.hidden_barriers)}")
```

### Option 4: Agent-Based Analysis

```python
from src.agent import WalkingAgent

# Create agent with personality
agent = WalkingAgent.from_preset("safety", "sagai_2025")

# Analyze route (full cognitive pipeline)
result = agent.run(
    start=(40.7589, -73.9851),
    end=(40.7614, -73.9776),
    interval=50
)

# Access agent's decision
print(f"Recommendation: {result['decision']['recommendation']}")
print(f"Confidence: {result['decision']['confidence']:.1%}")
print(f"Explanation: {result['result']['message']}")

# Check agent state
print(f"Memory count: {result['state']['memory_count']}")
print(f"Routes evaluated: {len(result['state']['evaluated_routes'])}")
```

---

## Common Configuration

### Environment Variables

Create `.env` file in `city_walk_agent/`:

```bash
# VLM APIs
OPENAI_API_KEY=your-openai-key
QWEN_VLM_API_URL=your-qwen-url
QWEN_VLM_API_KEY=your-qwen-key

# Data Collection
GOOGLE_MAPS_API_KEY=your-google-key
MAPILLARY_API_KEY=your-mapillary-key
```

### Framework Selection

Available frameworks:
- `sagai_2025` - SAGAI baseline (4D: safety, comfort, interest, aesthetics)
- `streetagent_5d` - StreetAgent (5D: spatial_sequence, visual_coherence, sensory_complexity, spatial_legibility, functional_quality)
- `ewing_handy_5d` - Ewing & Handy (5D: imageability, enclosure, human_scale, transparency, complexity)
- `kaplan_4d` - Kaplan (4D: coherence, legibility, complexity_mystery, affordance)
- `phenomenology_3d` - Phenomenology (3D: material_atmosphere, spatial_emotion, sensory_experience)

### VLM Provider Selection

Supported providers:
- `claude` - Anthropic Claude (Recommended: Sonnet 4.5)
- `openai` - OpenAI GPT-4V
- `qwen` - Qwen VLM (for testing)

---

## Advanced Usage

### Custom Analysis Parameters

```python
# Custom volatility and barrier thresholds
analyzer = SequentialAnalyzer(
    route,
    evaluations,
    volatility_threshold=2.5,  # Default: 2.0
    barrier_threshold=3.5      # Default: 3.0
)
```

### Batch Processing Multiple Routes

```python
routes = [route1, route2, route3]
results = []

for route in routes:
    result = pipeline.analyze_existing_route(route_file)
    results.append(result)
```

### Cost Estimation

```python
from src.utils import get_cost_tracker

# Before running
tracker = get_cost_tracker()
estimated_cost = tracker.estimate_cost(
    provider="claude",
    model="claude-sonnet-4-20250514",
    num_images=50,
    num_dimensions=4
)
print(f"Estimated cost: ${estimated_cost:.2f}")

# After running
tracker.print_summary()
```

---

## Troubleshooting

### Missing API Keys

Error: `API key required for claude`

**Solution:** Add API key to `.env` or pass explicitly:
```python
pipeline = WalkingAgentPipeline(
    framework_id="sagai_2025",
    vlm_provider="claude",
    vlm_api_key="your-key-here"
)
```

### Framework Not Found

Error: `Framework 'xyz' not found`

**Solution:** Check available frameworks:
```python
from src.config import list_frameworks
frameworks = list_frameworks()
for f in frameworks:
    print(f"{f['id']}: {f['name']}")
```

### No Images Found

Error: `No images found for route`

**Solution:** Enable image collection:
```python
result = pipeline.analyze_route(
    start=start,
    end=end,
    collect_images=True  # Enable collection
)
```

---

## Next Steps

After running examples:
1. Review output files in `data/results/pipeline/`
2. Examine `sequential_analysis.json` for detailed metrics
3. Read `summary_report.txt` for human-readable summary
4. Visualize results (see experiment notebooks)
5. Try different frameworks and compare

## Support

For issues or questions:
- Check project documentation
- Review experiment notebooks in `experiments/`
- See API documentation in source code
