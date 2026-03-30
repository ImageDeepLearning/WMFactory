from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image

from .control_adapter import (
    control_payload_to_action,
    precache_control_kind,
    precache_ring_step,
    precache_virtual_camera_delta_deg,
)
from .precache_frames import (
    list_precache_frame_paths,
    load_keyframes_manifest,
    nearest_keyframe_index,
    path_index_for_output_index,
)
from .scene_registry import scene_by_name
from .worldfm_integration import get_svc

logger = logging.getLogger(__name__)

_active_conn_id: Optional[str] = None
_active_lock = asyncio.Lock()


def _image_file_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _frame_png_b64_to_jpeg_bytes(frame_b64: str) -> bytes:
    raw = base64.b64decode(frame_b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    q = int(os.getenv("WORLDFMEND_JPEG_QUALITY", "85"))
    im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def _png_path_to_jpeg_bytes(path: Path) -> bytes:
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    q = int(os.getenv("WORLDFMEND_JPEG_QUALITY", "85"))
    im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def _load_start_session(init_b64: str, cache_key: Optional[str] = None) -> dict[str, Any]:
    svc = get_svc()
    svc.load()
    return svc.start_session(init_b64, cache_key=cache_key)


def _step_session(session_id: str, action: dict[str, Any]) -> dict[str, Any]:
    return get_svc().step(session_id, action)


def _reset_session(
    session_id: str, init_b64: str, cache_key: Optional[str] = None
) -> dict[str, Any]:
    return get_svc().reset_session(session_id, init_b64, cache_key=cache_key)


async def websocket_frames(ws: WebSocket) -> None:
    global _active_conn_id

    await ws.accept()
    conn_id = str(uuid.uuid4())
    session_id: Optional[str] = None
    init_b64: Optional[str] = None
    timeup_task: Optional[asyncio.Task[None]] = None
    precache_paths: list[Path] = []
    precache_idx: int = 0
    precache_keyframes: Optional[list[dict[str, Any]]] = None
    virt_yaw: float = 0.0
    virt_pitch: float = 0.0
    scene_name: Optional[str] = None

    async def clear_active_if_me() -> None:
        global _active_conn_id
        nonlocal session_id, timeup_task, precache_paths, precache_idx, precache_keyframes, virt_yaw, virt_pitch, scene_name
        async with _active_lock:
            if _active_conn_id == conn_id:
                _active_conn_id = None
        if timeup_task and not timeup_task.done():
            timeup_task.cancel()
        session_id = None
        precache_paths = []
        precache_idx = 0
        precache_keyframes = None
        virt_yaw = 0.0
        virt_pitch = 0.0
        scene_name = None

    def schedule_timeup() -> None:
        nonlocal timeup_task
        max_sec = float(os.getenv("WORLDFM_MAX_SESSION_SEC", "0") or "0")
        if max_sec <= 0:
            return

        async def _fire() -> None:
            try:
                await asyncio.sleep(max_sec)
                try:
                    await ws.send_json({"type": "timeup"})
                except Exception:
                    pass
                await ws.close()
            except asyncio.CancelledError:
                pass

        timeup_task = asyncio.create_task(_fire())

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] != "websocket.receive":
                continue

            if "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                mtype = data.get("type")

                if mtype == "ping":
                    await ws.send_json({"type": "pong"})
                    continue

                if mtype == "start":
                    scene = str(data.get("scene") or "")
                    sc = scene_by_name(scene)
                    if sc is None:
                        await ws.send_json(
                            {
                                "type": "error",
                                "error": "unknown_scene",
                                "message": f"Unknown scene: {scene}",
                            }
                        )
                        continue

                    async with _active_lock:
                        if _active_conn_id is not None and _active_conn_id != conn_id:
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "error": "server_busy",
                                    "message": "Another session is active; disconnect it first",
                                }
                            )
                            continue
                        _active_conn_id = conn_id

                    init_b64 = _image_file_to_b64(sc.init_image)
                    pc_paths = (
                        list_precache_frame_paths(sc.precache_dir)
                        if sc.precache_dir
                        else []
                    )
                    if pc_paths:
                        await ws.send_json(
                            {
                                "type": "queue_status",
                                "status": "waiting",
                                "position": 1,
                                "message": "Loading pre-baked frames…",
                            }
                        )
                        await ws.send_json(
                            {
                                "type": "queue_status",
                                "status": "active",
                                "position": 1,
                                "message": "Session ready",
                            }
                        )
                        session_id = f"precache:{scene}"
                        scene_name = scene
                        precache_paths = pc_paths
                        kf = (
                            load_keyframes_manifest(sc.precache_dir)
                            if sc.precache_dir
                            else None
                        )
                        precache_keyframes = kf
                        precache_idx = 0
                        if precache_keyframes:
                            k0 = precache_keyframes[0]
                            virt_yaw = float(k0.get("yaw_deg", 0.0))
                            virt_pitch = float(k0.get("pitch_deg", 0.0))
                            ni = nearest_keyframe_index(
                                precache_keyframes, virt_yaw, virt_pitch
                            )
                            oi = int(precache_keyframes[ni].get("index", 0))
                            precache_idx = path_index_for_output_index(
                                precache_paths, oi
                            )
                            jpeg = await asyncio.to_thread(
                                _png_path_to_jpeg_bytes, precache_paths[precache_idx]
                            )
                        else:
                            virt_yaw = 0.0
                            virt_pitch = 0.0
                            jpeg = await asyncio.to_thread(
                                _png_path_to_jpeg_bytes, precache_paths[0]
                            )
                        await ws.send_bytes(jpeg)
                        schedule_timeup()
                        continue
                    try:
                        scene_name = scene
                        start_data = await asyncio.to_thread(
                            _load_start_session, init_b64, scene
                        )
                    except Exception as exc:
                        await clear_active_if_me()
                        logger.exception("start_session failed")
                        await ws.send_json(
                            {
                                "type": "error",
                                "error": "server",
                                "message": str(exc),
                            }
                        )
                        continue

                    session_id = str(start_data.get("session_id") or "")
                    frame_b64 = str(start_data.get("frame_base64") or "")
                    if not session_id or not frame_b64:
                        await clear_active_if_me()
                        await ws.send_json(
                            {
                                "type": "error",
                                "error": "server",
                                "message": "Invalid start_session response",
                            }
                        )
                        continue

                    jpeg = await asyncio.to_thread(_frame_png_b64_to_jpeg_bytes, frame_b64)
                    await ws.send_bytes(jpeg)
                    schedule_timeup()
                    continue

                if mtype == "control":
                    payload = {k: v for k, v in data.items() if k != "type"}
                    if not session_id or not init_b64:
                        continue

                    if session_id.startswith("precache:") and precache_paths:
                        if payload.get("recenter"):
                            if precache_keyframes:
                                k0 = precache_keyframes[0]
                                virt_yaw = float(k0.get("yaw_deg", 0.0))
                                virt_pitch = float(k0.get("pitch_deg", 0.0))
                                ni = nearest_keyframe_index(
                                    precache_keyframes, virt_yaw, virt_pitch
                                )
                                oi = int(precache_keyframes[ni].get("index", 0))
                                precache_idx = path_index_for_output_index(
                                    precache_paths, oi
                                )
                                jpeg = await asyncio.to_thread(
                                    _png_path_to_jpeg_bytes,
                                    precache_paths[precache_idx],
                                )
                            else:
                                precache_idx = 0
                                jpeg = await asyncio.to_thread(
                                    _png_path_to_jpeg_bytes, precache_paths[0]
                                )
                            await ws.send_bytes(jpeg)
                            continue
                        kind = precache_control_kind(payload)
                        if kind is None:
                            continue
                        if precache_keyframes:
                            if kind == "look":
                                dy, dp = precache_virtual_camera_delta_deg(payload)
                                virt_yaw += dy
                                virt_pitch += dp
                                ni = nearest_keyframe_index(
                                    precache_keyframes, virt_yaw, virt_pitch
                                )
                                oi = int(precache_keyframes[ni].get("index", 0))
                                precache_idx = path_index_for_output_index(
                                    precache_paths, oi
                                )
                            else:
                                step = precache_ring_step(payload)
                                if step is None:
                                    continue
                                n = len(precache_paths)
                                precache_idx = (precache_idx + step) % n
                            jpeg = await asyncio.to_thread(
                                _png_path_to_jpeg_bytes, precache_paths[precache_idx]
                            )
                        else:
                            step = precache_ring_step(payload)
                            if step is None:
                                continue
                            n = len(precache_paths)
                            precache_idx = (precache_idx + step) % n
                            jpeg = await asyncio.to_thread(
                                _png_path_to_jpeg_bytes, precache_paths[precache_idx]
                            )
                        await ws.send_bytes(jpeg)
                        continue

                    act = control_payload_to_action(payload)
                    if act is None:
                        continue

                    if act.get("__recenter__"):
                        try:
                            reset_data = await asyncio.to_thread(
                                _reset_session, session_id, init_b64, scene_name
                            )
                        except Exception as exc:
                            logger.exception("reset_session failed")
                            await ws.send_json(
                                {"type": "error", "error": "server", "message": str(exc)}
                            )
                            continue
                        session_id = str(reset_data.get("session_id") or session_id)
                        frame_b64 = str(reset_data.get("frame_base64") or "")
                        if frame_b64:
                            jpeg = await asyncio.to_thread(
                                _frame_png_b64_to_jpeg_bytes, frame_b64
                            )
                            await ws.send_bytes(jpeg)
                        continue

                    act.pop("__recenter__", None)
                    try:
                        step_data = await asyncio.to_thread(_step_session, session_id, act)
                    except Exception as exc:
                        logger.exception("step failed")
                        await ws.send_json(
                            {"type": "error", "error": "server", "message": str(exc)}
                        )
                        continue
                    frame_b64 = str(step_data.get("frame_base64") or "")
                    if frame_b64:
                        jpeg = await asyncio.to_thread(
                            _frame_png_b64_to_jpeg_bytes, frame_b64
                        )
                        await ws.send_bytes(jpeg)
                    continue

            elif "bytes" in msg and msg["bytes"] is not None:
                continue

    except WebSocketDisconnect:
        pass
    finally:
        await clear_active_if_me()
