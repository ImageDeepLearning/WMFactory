from __future__ import annotations

import json

from fastapi import WebSocket


async def websocket_session_stub(ws: WebSocket) -> None:
    """WebRTC signaling is not implemented; tell client to use TCP (?transport=tcp)."""
    await ws.accept()
    try:
        await ws.send_json(
            {
                "type": "error",
                "error": "webrtc_disabled",
                "message": "WebRTC is not enabled on this server. Use TCP: set transport=tcp in URL or /api/config.",
            }
        )
    finally:
        await ws.close()
