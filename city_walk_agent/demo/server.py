"""
demo/server.py — FastAPI backend for CityWalk interactive demo

Serves waypoint data, runs VLM analysis on demand, and serves the frontend.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── settings & project imports ────────────────────────────────────────────────
from src.core import settings, load_framework
from src.core.evaluation.evaluator import Evaluator
from src.core.evaluation.vlm_client import VLMConfig
from src.agent.config.personalities import ENHANCED_PERSONALITIES

# ── constants ─────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
FRAMEWORKS_DIR = PROJECT_ROOT / "src" / "core" / "frameworks" / "configs"

ROUTE_FOLDERS: Dict[str, Path] = {
    "singapore": IMAGES_DIR / "singapore",
    "hongkong": IMAGES_DIR / "hongkong",
}

GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ── app ───────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Open browser when server starts
    async def _open():
        await asyncio.sleep(1.0)
        subprocess.Popen(["open", "http://localhost:8000"])
    asyncio.create_task(_open())
    yield

app = FastAPI(title="CityWalk Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = DEMO_DIR / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── request/response models ───────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    heading: float = 0.0
    persona: str = "objective"
    image_path: Optional[str] = None
    waypoint_id: Optional[int] = None


class AnalyzeResponse(BaseModel):
    scores: Dict[str, float]
    reasoning: Dict[str, str]
    image_base64: str
    persona: str
    waypoint_id: int
    processing_time_sec: float


# ── helper: build evaluator ───────────────────────────────────────────────────
def _build_evaluator() -> Evaluator:
    framework = load_framework("place_pulse_2.0")
    vlm_config = VLMConfig(
        api_key=settings.qwen_vlm_api_key,
        model=settings.qwen_vlm_model,
        api_url=settings.qwen_vlm_api_url,
    )
    return Evaluator(vlm_config, framework, max_concurrent=4)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    index_path = DEMO_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/route/{route_id}")
async def get_route(route_id: str):
    """Return waypoint list from collection_metadata.json."""
    folder = ROUTE_FOLDERS.get(route_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Unknown route: {route_id}. Available: {list(ROUTE_FOLDERS)}")

    metadata_path = folder / "collection_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"collection_metadata.json not found in {folder}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    results = metadata.get("results", [])

    waypoints = []
    for entry in sorted(results, key=lambda e: e.get("waypoint_id", 0)):
        # Resolve image path: prefer local folder, fall back to stored absolute path
        img_path = _resolve_image_path(entry, folder)
        # Extract heading: from heading_images, filename, or entry field
        heading = _extract_heading(entry, img_path)
        waypoints.append({
            "id": entry.get("waypoint_id", 0),
            "lat": entry.get("lat"),
            "lon": entry.get("lon"),
            "heading": heading,
            "image_path": str(img_path) if img_path else None,
        })

    return {"route_id": route_id, "waypoints": waypoints}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_waypoint(req: AnalyzeRequest):
    """Run VLM analysis on a waypoint image."""
    t0 = time.time()

    # 1. Obtain image bytes
    image_bytes = await _get_image_bytes(req)

    # 2. Write to a temp file so Evaluator can read it (it expects a file path)
    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # 3. Build persona object (None = objective)
        persona = None
        if req.persona and req.persona != "objective":
            persona = ENHANCED_PERSONALITIES.get(req.persona)

        # 4. Run evaluation
        # "persona" mode when persona is set (single VLM call per dimension),
        # "objective" mode otherwise. "dual" is intentionally avoided — it doubles
        # API calls and System 2 reasoning is not yet implemented in the demo.
        evaluator = _build_evaluator()
        mode = "persona" if persona else "objective"
        loop = asyncio.get_event_loop()
        eval_results = await loop.run_in_executor(
            None,
            lambda: evaluator.evaluate_image(
                tmp_path,
                persona=persona,
                evaluation_mode=mode,
            ),
        )

        # 5. Extract scores/reasoning
        # Both "objective" and "persona" modes return plain dicts with "score"/"reasoning".
        scores: Dict[str, float] = {}
        reasoning: Dict[str, str] = {}

        for r in eval_results:
            scores[r["dimension_id"]] = r["score"]
            reasoning[r["dimension_id"]] = r["reasoning"]

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return AnalyzeResponse(
            scores=scores,
            reasoning=reasoning,
            image_base64=image_b64,
            persona=req.persona,
            waypoint_id=req.waypoint_id or 0,
            processing_time_sec=round(time.time() - t0, 2),
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/image")
async def get_image(image_path: str):
    """Return raw image bytes for instant display before VLM analysis."""
    p = Path(image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/personas")
async def get_personas():
    """Return available persona IDs."""
    return ["objective"] + list(ENHANCED_PERSONALITIES.keys())


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_image_path(entry: dict, folder: Path) -> Optional[Path]:
    """Try multiple strategies to find the waypoint image."""
    wp_id = entry.get("waypoint_id", 0)

    # Strategy 1: waypoint_NNN_heading_HHH.jpg in the named route folder (singapore/hongkong)
    candidates = list(folder.glob(f"waypoint_{wp_id:03d}_heading_*.jpg"))
    if candidates:
        return candidates[0]

    # Strategy 2: stored absolute path (may point to old location)
    for key in ("image_path",):
        stored = entry.get(key)
        if stored:
            p = Path(stored)
            if p.exists():
                return p
            # Filename only, relative to this folder
            rel = folder / p.name
            if rel.exists():
                return rel

    # Strategy 3: heading_images list
    for hi in entry.get("heading_images", []):
        p = Path(hi.get("image_path", ""))
        if p.exists():
            return p

    # Strategy 4: waypoint_NNN/ subdirectory
    sub = folder / f"waypoint_{wp_id:03d}"
    if sub.is_dir():
        imgs = list(sub.glob("*.jpg"))
        if imgs:
            return imgs[0]

    return None


def _extract_heading(entry: dict, path: Optional[Path]) -> float:
    """Extract heading from entry metadata or filename."""
    # Try heading_images list first
    hi_list = entry.get("heading_images", [])
    if hi_list:
        return float(hi_list[0].get("heading", 0))

    # Direct field
    if "heading" in entry:
        return float(entry["heading"])

    # From filename: waypoint_000_heading_214.jpg
    if path:
        parts = path.stem.split("_heading_")
        if len(parts) == 2:
            try:
                return float(parts[1])
            except ValueError:
                pass
    return 0.0


async def _get_image_bytes(req: AnalyzeRequest) -> bytes:
    """Fetch image bytes from local file or Google Street View API."""
    # Local file path provided
    if req.image_path:
        p = Path(req.image_path)
        if p.exists():
            return p.read_bytes()
        raise HTTPException(status_code=400, detail=f"Image file not found: {req.image_path}")

    # Fall back to Google Street View
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="No image_path provided and GOOGLE_MAPS_API_KEY not set. "
                   "Provide image_path or set the API key in .env.",
        )

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{req.lat},{req.lon}",
        "heading": str(req.heading),
        "key": GOOGLE_API_KEY,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Street View API error")
        return resp.content


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("demo.server:app", host="0.0.0.0", port=8000, reload=True)
