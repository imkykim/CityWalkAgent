# CityWalkAgent

> Sequential, route-level pedestrian experience analysis using a dual-process cognitive architecture.

CityWalkAgent is an autonomous pedestrian navigation and urban environment analysis platform. It treats walking journeys as continuous narratives rather than aggregate point scores, enabling detection of *hidden barriers* and within-route variance that traditional averaging methods miss.

**Project website:** https://imkykim.github.io/CityWalkAgent/
**Live demo:** see [Quickstart](#quickstart) below
**Thesis:** Master's thesis, School of Architecture, Tsinghua University (2026). Supervisor: Prof. Huang Weixin (黄蔚欣).

<!-- ![demo screenshot](documentation/screenshots/demo.png) -->

## Key ideas

- **Dual-process architecture (Kahneman):** A fast System 1 scores every waypoint via a vision-language model; a slower System 2 reasons about the route at intersections and significant scene changes.
- **Persona as interpretive lens:** Six personas (Prospective Homebuyer, Urban Runner, Parent with Young Children, Street Photographer, Elderly Walker, Tourist) reinterpret the same street differently — not by post-hoc score adjustment but by reframing the VLM's evaluation prompt.
- **Validated against Place Pulse 2.0:** Spearman ρ between 0.57 and 0.85 across four perceptual dimensions (Safety, Lively, Beautiful, Wealthy) using CLIP + K-NN against 1.1M+ human pairwise judgments.

For the full architecture, see [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md).
For validation methodology and results, see [`documentation/VALIDATION.md`](documentation/VALIDATION.md).

## Quickstart

```bash
# 1. Clone
git clone https://github.com/imkykim/CityWalkAgent.git
cd CityWalkAgent/city_walk_agent

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Open .env and fill in GOOGLE_MAPS_API_KEY, QWEN_VLM_API_URL, QWEN_VLM_API_KEY

# 4. Run the demo
python -m uvicorn demo.server:app --reload --port 8000

# 5. Open http://localhost:8000
```

The demo lets you click waypoints on a map, see Street View imagery analyzed by the VLM in real time, and watch the agent autonomously walk a route with live System 1 / System 2 reasoning. Two pre-loaded routes are included: Singapore (Marina Bay) and Hong Kong (Sham Shui Po).

For demo-specific options (personas, autonomous walk parameters), see [`city_walk_agent/demo/README.md`](city_walk_agent/demo/README.md).

## Repository structure

```
CityWalkAgent/
├── README.md                      ← you are here
├── LICENSE                        (MIT)
├── CITATION.cff
├── docs/                          ← project website (GitHub Pages)
│   ├── index.html
│   └── assets/
├── documentation/                 ← developer/reviewer documentation
│   ├── ARCHITECTURE.md            ← System 1/2, gating, persona model
│   ├── VALIDATION.md              ← Place Pulse 2.0 evaluation
│   └── screenshots/
└── city_walk_agent/               ← Python package
    ├── .env.example
    ├── requirements.txt
    ├── main.py
    ├── demo/                      ← FastAPI server + web UI
    ├── src/
    │   ├── core/                  (settings, frameworks, constants)
    │   ├── agent/                 (orchestrator, system1/2, memory)
    │   ├── data_collection/       (Street View image fetchers)
    │   └── validation/            (CLIP + KNN against Place Pulse 2.0)
    └── tests/
```

## Requirements

- Python 3.10+
- A Google Maps Platform API key with Maps JavaScript API, Street View Static API, Map Tiles API, and Directions API enabled
- An OpenAI-compatible vision endpoint (tested with Qwen3-VL-30B-A3B; OpenAI GPT-4o, etc. also work)
- (Optional, for validation) Place Pulse 2.0 dataset — see [`documentation/VALIDATION.md`](documentation/VALIDATION.md)

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{kim2026citywalkagent,
  author  = {Kim, Kyungdeok},
  title   = {CityWalkAgent: Sequential Pedestrian Experience Analysis with a Dual-Process Cognitive Architecture},
  school  = {Tsinghua University, School of Architecture},
  year    = {2026},
  address = {Beijing, China}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This work was developed as a master's thesis at Tsinghua University under the supervision of Prof. Huang Weixin (黄蔚欣), with TAs Huang Zilong (黄梓龙), Wang Wenda (王文达), and Lu Jiachen (鲁佳辰). Validation uses the Place Pulse 2.0 dataset from MIT Media Lab.
