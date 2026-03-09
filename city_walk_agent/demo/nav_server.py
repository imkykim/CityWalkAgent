# 실행: cd city_walk_agent && python -m uvicorn demo.nav_server:app --port 8001
# 접속: http://localhost:8001

import time
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# Load .env from city_walk_agent directory
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

THIS_DIR = Path(__file__).parent


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
                json={"mapType": "streetview", "language": "ko", "region": "KR"},
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

    async def get_links(self, pano_id: str) -> list[dict]:
        metadata = await self.get_metadata(pano_id)
        return metadata.get("links", [])


_session: Optional[MapTilesSession] = None


def get_session() -> MapTilesSession:
    global _session
    if _session is None:
        if not API_KEY:
            raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다. .env 파일에 GOOGLE_MAPS_API_KEY를 설정하세요.")
        _session = MapTilesSession(API_KEY)
    return _session


def make_street_view_url(pano_id: str, heading: float = 0) -> str:
    return (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?pano={pano_id}&size=640x640&heading={heading}&fov=90&pitch=0&key={API_KEY}"
    )


def build_pano_response(pano_id: str, metadata: dict, heading: float = 0) -> dict:
    lat = metadata.get("lat", 0)
    lng = metadata.get("lng", 0)
    links = metadata.get("links", [])
    date = metadata.get("date")
    copyright_ = metadata.get("copyright")
    return {
        "pano_id": pano_id,
        "lat": lat,
        "lng": lng,
        "links": links,
        "is_intersection": len(links) >= 3,
        "link_count": len(links),
        "street_view_image_url": make_street_view_url(pano_id, heading),
        "date": date,
        "copyright": copyright_,
    }


@app.get("/")
async def root():
    return FileResponse(THIS_DIR / "nav.html")


@app.get("/api/status")
async def status():
    api_key_set = bool(API_KEY)
    if not api_key_set:
        return {"api_key_set": False, "session_ok": False, "token_preview": "", "error": "API 키 없음"}
    try:
        session = get_session()
        token = await session.get_token()
        return {
            "api_key_set": True,
            "session_ok": True,
            "token_preview": token[:20] + "...",
            "error": None,
        }
    except httpx.HTTPError as e:
        return {"api_key_set": True, "session_ok": False, "token_preview": "", "error": str(e)}
    except Exception as e:
        return {"api_key_set": True, "session_ok": False, "token_preview": "", "error": str(e)}


class StartBody(BaseModel):
    lat: float
    lng: float


class NavigateBody(BaseModel):
    pano_id: str
    heading: Optional[float] = None


@app.post("/api/start")
async def start(body: StartBody):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")
    session = get_session()
    try:
        pano_id = await session.coord_to_pano_id(body.lat, body.lng)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Map Tiles API 오류: {e}")
    if pano_id is None:
        raise HTTPException(status_code=404, detail="해당 좌표 근처에서 Street View 파노라마를 찾을 수 없습니다.")
    try:
        metadata = await session.get_metadata(pano_id)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"메타데이터 조회 오류: {e}")
    return build_pano_response(pano_id, metadata, heading=0)


@app.post("/api/navigate")
async def navigate(body: NavigateBody):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")
    session = get_session()
    try:
        metadata = await session.get_metadata(body.pano_id)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Map Tiles API 오류: {e}")
    heading = body.heading if body.heading is not None else 0
    return build_pano_response(body.pano_id, metadata, heading=heading)


if __name__ == "__main__":
    uvicorn.run("demo.nav_server:app", host="0.0.0.0", port=8001, reload=True)
