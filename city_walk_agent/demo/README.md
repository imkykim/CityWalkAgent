# CityWalk VLM Demo

Interactive web app: click a waypoint on the map → Street View image is analyzed by VLM in real time → scores and reasoning appear instantly.

## Setup

```bash
# from project root
cd city_walk_agent

pip install fastapi uvicorn httpx python-dotenv
```

This demo reads `GOOGLE_MAPS_API_KEY`, `QWEN_VLM_API_URL`, and `QWEN_VLM_API_KEY` from `city_walk_agent/.env`. See the root [README](../../README.md#quickstart) for setup. If `GOOGLE_MAPS_API_KEY` is missing, only local pre-collected routes (Singapore, Hong Kong) will work — live Street View fetching will be disabled.

## Run

```bash
cd city_walk_agent
python -m uvicorn demo.server:app --reload --port 8000
```

Open: http://localhost:8000

## Routes available

| Button | Folder |
|--------|--------|
| Singapore | `data/images/singapore/` (56 waypoints) |
| Hong Kong | `data/images/hongkong/` (62 waypoints) |

## Features

- **Persona selector** — objective, homebuyer, runner, parent_with_kids, photographer, …
- **Click-to-analyze** — POST /analyze triggers real-time VLM evaluation (5–15 s)
- **Score bars** — 4 Place Pulse dimensions with animated color-coded bars
- **Reasoning accordion** — per-dimension text, collapsible
- **Cache** — same waypoint + persona re-clicked instantly without re-analyzing
- **Persona change** — clears cache, resets all marker colors

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves `index.html` |
| GET | `/route/{singapore\|hongkong}` | Returns waypoint list |
| POST | `/analyze` | Runs VLM analysis, returns scores + reasoning + image |
| GET | `/personas` | Lists available persona IDs |
