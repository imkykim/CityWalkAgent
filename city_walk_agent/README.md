# CityWalkAgent - Proof of Concept

A sequential walking experience analysis system that evaluates pedestrian experiences along complete routes rather than individual points.

## Project Structure

```
city_walk_agent/
├── src/
│   ├── config/          # Configuration and settings
│   ├── data_collection/ # Route generation and image collection
│   ├── evaluation/      # AI-powered walkability evaluation (TODO)
│   ├── analysis/        # Sequential analysis algorithms (TODO)
│   └── utils/           # Common utilities and data models
├── experiments/         # Research scripts and hypothesis testing
├── tests/              # Unit and integration tests
├── data/               # Route data, images, and results
└── requirements.txt    # Python dependencies
```

### Directory Purposes

**`tests/`** - Code validation and quality assurance
- Unit tests for individual components
- Integration tests for workflows
- Framework validation (`test_framework.py`)
- Run before commits and releases

**`experiments/`** - Research and proof-of-concept work
- Exploratory analysis scripts
- Hypothesis testing code
- Prototype implementations
- Results and findings documentation

## Getting Started

### Prerequisites

1. **API Keys Required:**
   - Google Maps API key (for route generation and Street View)
   - Mapillary API key (alternative image source)
   - Anthropic API key (for Claude evaluation)

2. **Dependencies:**
   - ZenSVI library (for image collection)
   - VIRL library (for VLM analysis)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your API keys:
```bash
GOOGLE_MAPS_API_KEY=your_google_api_key
MAPILLARY_API_KEY=your_mapillary_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Quick Start

1. **Test the framework:**
```bash
python tests/test_framework.py
```

2. **Test API connections:**
```bash
python tests/test_api_connections.py
```

3. **Run experiments:**
```bash
python experiments/01_basic_route_experiment.py
```

4. **Basic usage:**
```python
from src.data_collection import RouteGenerator

# Generate a route (no API key needed for simple routes)
route_gen = RouteGenerator(api_key="test_key")
route = route_gen.create_simple_route(
    start_lat=40.7589, start_lon=-73.9851,
    end_lat=40.7614, end_lon=-73.9776,
    interval_meters=20
)
```

## Core Components

### 1. Route Generator
- Creates routes between two points
- Supports Google Maps integration for realistic paths
- Generates waypoints at specified intervals

### 2. Image Collector
- Downloads street view images at each waypoint
- Supports multiple platforms (Google Street View, Mapillary)
- Handles batch processing and error recovery

### 3. VLM Evaluation (Coming Next)
- AI-powered walkability assessment
- Multi-dimensional scoring (safety, comfort, interest, aesthetics)
- Structured output for analysis

### 4. Sequential Analysis (Coming Next)
- Volatility detection
- Hidden barrier identification
- Comparison with traditional aggregate methods

## Experimental Design

This PoC focuses on validating the core hypothesis:
**Sequential analysis provides more meaningful insights than aggregate point scores**

Key experiments:
1. Compare routes with similar aggregate scores but different sequential patterns
2. Detect "hidden barriers" missed by traditional methods
3. Test different evaluation prompts and dimensions

## Next Steps

The current framework provides the foundation for:
1. VLM-based evaluation engine
2. Sequential analysis algorithms
3. Comparative analysis tools
4. Visualization and reporting

## Architecture

Based on the successful patterns from:
- **ZenSVI**: Street view image collection
- **VIRL**: Vision-language model analysis pipeline
- **Modular design**: Easy experimentation and iteration