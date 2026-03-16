"""
demo/server.py — FastAPI backend for CityWalk interactive demo

Serves waypoint data, runs VLM analysis on demand, and serves the frontend.
Also provides Street View navigation via Google Map Tiles API (merged from nav_server.py).
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
from src.agent.config.personalities import ENHANCED_PERSONALITIES, get_preset
from src.agent.orchestrator import CityWalkAgent
from src.agent.memory.memory_manager import MemoryManager

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


# ── Map Tiles session (merged from nav_server.py) ─────────────────────────────

class MapTilesSession:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._token: Optional[str] = None
        self._expiry: float = 0

    async def get_token(self) -> str:
        if self._token and time.time() < self._expiry - 3600:
            return self._token
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://tile.googleapis.com/v1/createSession",
                params={"key": self.api_key},
                json={"mapType": "streetview", "language": "en", "region": "US"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        self._token = data["session"]
        self._expiry = float(data["expiry"])
        return self._token

    async def coord_to_pano_id(self, lat: float, lng: float) -> Optional[str]:
        token = await self.get_token()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://tile.googleapis.com/v1/streetview/panoIds",
                params={"session": token, "key": self.api_key},
                json={"locations": [{"lat": lat, "lng": lng}], "radius": 50},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        pano_ids = data.get("panoIds", [])
        if not pano_ids or pano_ids[0] == "":
            return None
        return pano_ids[0]

    async def get_metadata(self, pano_id: str) -> dict:
        token = await self.get_token()
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://tile.googleapis.com/v1/streetview/metadata",
                params={"session": token, "key": self.api_key, "panoId": pano_id},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()


_nav_session: Optional[MapTilesSession] = None


def _get_nav_session() -> MapTilesSession:
    global _nav_session
    if _nav_session is None:
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_MAPS_API_KEY not set. Add it to .env.",
            )
        _nav_session = MapTilesSession(GOOGLE_API_KEY)
    return _nav_session


def _make_sv_url(pano_id: str, heading: float = 0) -> str:
    return (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?pano={pano_id}&size=640x640&heading={heading}&fov=90&pitch=0&key={GOOGLE_API_KEY}"
    )


def _build_pano_response(pano_id: str, metadata: dict, heading: float = 0) -> dict:
    return {
        "pano_id": pano_id,
        "lat": metadata.get("lat", 0),
        "lng": metadata.get("lng", 0),
        "links": metadata.get("links", []),
        "is_intersection": len(metadata.get("links", [])) >= 3,
        "link_count": len(metadata.get("links", [])),
        "street_view_image_url": _make_sv_url(pano_id, heading),
        "date": metadata.get("date"),
        "copyright": metadata.get("copyright"),
    }


# ── request/response models ───────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    heading: float = 0.0
    persona: str = "objective"
    image_path: Optional[str] = None
    pano_id: Optional[str] = None      # for navigation mode
    waypoint_id: Optional[int] = None


class NavStartBody(BaseModel):
    lat: float
    lng: float
    persona: str = "objective"


class NavNavigateBody(BaseModel):
    pano_id: str
    heading: Optional[float] = None


class AnalyzeResponse(BaseModel):
    scores: Dict[str, float]
    reasoning: Dict[str, str]
    image_base64: str
    persona: str
    waypoint_id: int
    processing_time_sec: float


# ── session store ─────────────────────────────────────────────────────────────

@dataclass
class NavSession:
    session_id: str
    agent: CityWalkAgent
    memory_manager: MemoryManager
    waypoint_counter: int = 0
    created_at: float = field(default_factory=time.time)


_sessions: Dict[str, NavSession] = {}


def _create_session(persona_id: str) -> NavSession:
    """Create a new navigation session with Orchestrator + MemoryManager."""
    import uuid
    session_id = str(uuid.uuid4())[:8]

    personality = get_preset(persona_id, "place_pulse_2.0") if persona_id != "objective" else get_preset("homebuyer", "place_pulse_2.0")

    agent = CityWalkAgent(
        agent_id=f"demo_{session_id}",
        personality=personality,
        framework_id="place_pulse_2.0",
    )

    memory_manager = MemoryManager(agent_id=f"demo_{session_id}")
    memory_manager.set_agent_attributes(
        personality=personality,
        profile={"name": "Demo Walker"},
        status={"mode": "explore"},
    )

    # Inject memory_manager into continuous_analyzer
    agent.continuous_analyzer.memory_manager = memory_manager

    session = NavSession(
        session_id=session_id,
        agent=agent,
        memory_manager=memory_manager,
    )
    _sessions[session_id] = session

    # Evict sessions older than 2 hours
    cutoff = time.time() - 7200
    stale = [k for k, v in _sessions.items() if v.created_at < cutoff]
    for k in stale:
        del _sessions[k]

    return session


def _get_session(session_id: str) -> NavSession:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired.")
    return _sessions[session_id]


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


@app.get("/api/maps-key")
async def maps_key():
    """Return Google Maps JS API key for frontend StreetViewPanorama."""
    return {"key": GOOGLE_API_KEY or ""}


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
    """Fetch image bytes from local file, pano ID, or lat/lon Street View."""
    # Local file path provided
    if req.image_path:
        p = Path(req.image_path)
        if p.exists():
            return p.read_bytes()
        raise HTTPException(status_code=400, detail=f"Image file not found: {req.image_path}")

    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="No image_path provided and GOOGLE_MAPS_API_KEY not set.",
        )

    # Navigation mode: fetch by pano_id
    if req.pano_id:
        url = _make_sv_url(req.pano_id, req.heading)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="Street View API error (pano)")
            return resp.content

    # Fall back to lat/lon Street View
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


# ── navigation endpoints (merged from nav_server.py) ──────────────────────────

@app.get("/api/nav/status")
async def nav_status():
    """Check Google Maps API session validity."""
    if not GOOGLE_API_KEY:
        return {"api_key_set": False, "session_ok": False, "token_preview": "", "error": "API key not set"}
    try:
        session = _get_nav_session()
        token = await session.get_token()
        return {"api_key_set": True, "session_ok": True, "token_preview": token[:20] + "...", "error": None}
    except httpx.HTTPError as e:
        return {"api_key_set": True, "session_ok": False, "token_preview": "", "error": str(e)}
    except Exception as e:
        return {"api_key_set": True, "session_ok": False, "token_preview": "", "error": str(e)}


@app.post("/api/nav/start")
async def nav_start(body: NavStartBody):
    """{ lat, lng, persona? } → PanoResponse + session_id"""
    nav = _get_nav_session()
    try:
        pano_id = await nav.coord_to_pano_id(body.lat, body.lng)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Map Tiles API error: {e}")
    if pano_id is None:
        raise HTTPException(status_code=404, detail="No Street View panorama found near these coordinates.")
    try:
        metadata = await nav.get_metadata(pano_id)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Metadata fetch error: {e}")

    session = _create_session(body.persona)
    response = _build_pano_response(pano_id, metadata, heading=0)
    response["session_id"] = session.session_id
    return response


@app.post("/api/nav/navigate")
async def nav_navigate(body: NavNavigateBody):
    """{ pano_id, heading? } → PanoResponse"""
    session = _get_nav_session()
    try:
        metadata = await session.get_metadata(body.pano_id)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Map Tiles API error: {e}")
    heading = body.heading if body.heading is not None else 0
    return _build_pano_response(body.pano_id, metadata, heading=heading)


# ── stateful nav endpoints ────────────────────────────────────────────────────

class NavAnalyzeBody(BaseModel):
    session_id: str
    pano_id: str
    heading: float = 0.0
    waypoint_id: Optional[int] = None


@app.post("/api/nav/analyze")
async def nav_analyze(body: NavAnalyzeBody):
    """Analyze a single waypoint and update MemoryManager state."""
    import tempfile
    session = _get_session(body.session_id)

    # Fetch image via existing helper
    image_bytes = await _get_image_bytes(AnalyzeRequest(
        lat=0, lon=0,
        heading=body.heading,
        pano_id=body.pano_id,
        persona=getattr(session.agent.personality, "personality_id", "objective"),
    ))

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        tmp.write(image_bytes)
        tmp.flush()
        image_path = Path(tmp.name)
    finally:
        tmp.close()

    try:
        waypoint_id = body.waypoint_id if body.waypoint_id is not None else session.waypoint_counter
        session.waypoint_counter += 1

        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            None,
            lambda: session.agent.continuous_analyzer.analyze_waypoint(
                waypoint_id=waypoint_id,
                image_path=image_path,
                metadata={"heading": body.heading, "lat": 0, "lon": 0},
                visual_change_detected=True,
                phash_distance=None,
            ),
        )

        session.memory_manager.process_waypoint(analysis, triggered=False)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        scores = analysis.persona_scores or analysis.objective_scores or {}
        reasoning = analysis.persona_reasoning or analysis.objective_reasoning or {}

        return {
            "scores": scores,
            "reasoning": reasoning,
            "image_base64": image_b64,
            "waypoint_id": waypoint_id,
            "session_id": body.session_id,
            "ltm_snapshot_count": len(session.memory_manager._route_snapshots),
        }
    finally:
        image_path.unlink(missing_ok=True)


class NavBranchBody(BaseModel):
    session_id: str
    branch_pano_id: str
    candidate_headings: List[float]


@app.post("/api/nav/branch")
async def nav_branch(body: NavBranchBody):
    """Evaluate candidate directions at a branch point and choose the best."""
    session = _get_session(body.session_id)

    if len(body.candidate_headings) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 candidate headings.")
    if len(body.candidate_headings) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 candidate headings.")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: session.agent.branch_decision(
                branch_pano_id=body.branch_pano_id,
                candidate_headings=body.candidate_headings,
                memory_manager=session.memory_manager,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Branch decision failed: {e}")

    return {
        "session_id": body.session_id,
        "branch_pano_id": body.branch_pano_id,
        "chosen_direction": result["chosen_direction"],
        "chosen_heading": result["chosen_heading"],
        "chosen_pano_id": result["chosen_pano_id"],
        "reason": result["reason"],
        "confidence": result["confidence"],
        "ranking": result["ranking"],
        "candidates": [
            {
                "direction": c["direction"],
                "heading": c["heading"],
                "scores": c["scores"],
                "interpretation": c["interpretation"],
                "key_concern": c["key_concern"],
            }
            for c in result["candidates"]
        ],
    }


@app.post("/api/nav/session/reset")
async def nav_session_reset(body: dict):
    """Clear a session's memory (start fresh on same route)."""
    session_id = body.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"status": "ok"}


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("demo.server:app", host="0.0.0.0", port=8000, reload=True)
