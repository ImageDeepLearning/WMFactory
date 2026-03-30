from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .paths import WORLDFM_END_ROOT
from .scene_registry import list_scenes_api
from .ws_frames import websocket_frames
from .ws_session_stub import websocket_session_stub

# Local dev: run from WMFactory root, e.g.
#   uvicorn worldfmEnd.server:app --host 0.0.0.0 --port 8889
# Point worldfmFrontend getApiBase/getWsBase to this host or add rsbuild dev proxy.

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="WorldFM Demo Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = WORLDFM_END_ROOT / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/scenes")
def api_scenes() -> Dict[str, Any]:
    return list_scenes_api()


@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    """
    transport: tcp | udp (udp maps to WebRTC in frontend; we only support tcp here).
    Override with env WORLDFMEND_TRANSPORT=tcp|udp — default tcp so clients use /ws/frames.
    """
    t = (os.getenv("WORLDFMEND_TRANSPORT") or "tcp").lower().strip()
    if t not in ("tcp", "udp"):
        t = "tcp"
    return {"transport": t}


@app.websocket("/ws/frames")
async def ws_frames_route(ws: WebSocket) -> None:
    await websocket_frames(ws)


@app.websocket("/ws/session")
async def ws_session_route(ws: WebSocket) -> None:
    await websocket_session_stub(ws)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "worldfmEnd"}
