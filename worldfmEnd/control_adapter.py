from __future__ import annotations

import math
import os
from typing import Any, Dict, Literal, Optional

PrecacheControlKind = Literal["look", "strafe", "forward"]


def precache_control_kind(payload: Dict[str, Any]) -> Optional[PrecacheControlKind]:
    """Which axis dominated (same magnitudes as precache_ring_step). None if no input."""
    if payload.get("recenter"):
        return None
    if "move_local" not in payload and "look_delta" not in payload:
        return None
    ml = payload.get("move_local") or [0.0, 0.0, 0.0]
    ld = payload.get("look_delta") or [0.0, 0.0]
    try:
        mx = float(ml[0])
        mz = float(ml[2])
        lx = float(ld[0])
    except (TypeError, ValueError, IndexError):
        return None
    thresh = float(os.getenv("WORLDFMEND_MOVE_THRESH", "0.02"))
    look_eps = float(os.getenv("WORLDFMEND_PRECACHE_LOOK_EPS", "1e-5"))
    cand: list[tuple[float, PrecacheControlKind]] = []
    if abs(lx) > look_eps:
        cand.append((abs(lx), "look"))
    if abs(mx) > thresh:
        cand.append((abs(mx), "strafe"))
    if abs(mz) > thresh:
        cand.append((abs(mz), "forward"))
    if not cand:
        return None
    return max(cand, key=lambda x: x[0])[1]


def precache_virtual_camera_delta_deg(payload: Dict[str, Any]) -> tuple[float, float]:
    """
    Same camera deltas as WorldFMRuntimeService._apply_action, in degrees
    (matches keyframes.json yaw_deg / pitch_deg).
    """
    ld = payload.get("look_delta") or [0.0, 0.0]
    try:
        lx = float(ld[0])
        ly = float(ld[1])
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)
    look_scale = float(os.getenv("WORLDFMEND_LOOK_SCALE", "1.0"))
    yaw_step = math.radians(float(os.getenv("WM_WORLDFM_YAW_DEG", "3.0")))
    pitch_step = math.radians(float(os.getenv("WM_WORLDFM_PITCH_DEG", "2.0")))
    invert_pitch = os.getenv("WM_WORLDFM_INVERT_PITCH", "1") == "1"
    camera_dx = lx * look_scale
    camera_dy = ly * look_scale
    d_yaw_rad = camera_dx * yaw_step
    if invert_pitch:
        d_pitch_rad = -camera_dy * pitch_step
    else:
        d_pitch_rad = camera_dy * pitch_step
    return (math.degrees(d_yaw_rad), math.degrees(d_pitch_rad))


def precache_ring_step(payload: Dict[str, Any]) -> Optional[int]:
    """
    Map control to a signed step on the precached frame ring (not sequential +1).
    Dominant axis among horizontal look, strafe, forward/back by magnitude.
    """
    if payload.get("recenter"):
        return None
    if "move_local" not in payload and "look_delta" not in payload:
        return None
    ml = payload.get("move_local") or [0.0, 0.0, 0.0]
    ld = payload.get("look_delta") or [0.0, 0.0]
    try:
        mx = float(ml[0])
        mz = float(ml[2])
        lx = float(ld[0])
    except (TypeError, ValueError, IndexError):
        return None
    thresh = float(os.getenv("WORLDFMEND_MOVE_THRESH", "0.02"))
    look_eps = float(os.getenv("WORLDFMEND_PRECACHE_LOOK_EPS", "1e-5"))
    cand: list[tuple[float, int]] = []
    if abs(lx) > look_eps:
        cand.append((abs(lx), 1 if lx > 0 else -1))
    if abs(mx) > thresh:
        cand.append((abs(mx), 1 if mx > 0 else -1))
    if abs(mz) > thresh:
        cand.append((abs(mz), 1 if mz > 0 else -1))
    if not cand:
        return None
    return max(cand, key=lambda x: x[0])[1]


def control_payload_to_action(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Map worldfmFrontend control payloads to WorldFMRuntimeService step() action dict.
    Returns None if this message should be ignored (no step).
    Special key __recenter__ means caller should reset_session instead of step.
    """
    if payload.get("recenter"):
        return {"__recenter__": True}

    if "move_local" not in payload and "look_delta" not in payload:
        # Optional keys like render_focal_delta — ignore for inference
        return None

    ml = payload.get("move_local") or [0.0, 0.0, 0.0]
    ld = payload.get("look_delta") or [0.0, 0.0]
    try:
        mx = float(ml[0])
        mz = float(ml[2])
        lx = float(ld[0])
        ly = float(ld[1])
    except (TypeError, ValueError, IndexError):
        return None

    look_scale = float(os.getenv("WORLDFMEND_LOOK_SCALE", "1.0"))
    thresh = float(os.getenv("WORLDFMEND_MOVE_THRESH", "0.02"))

    out = {
        "camera_dx": lx * look_scale,
        "camera_dy": ly * look_scale,
        "w": mz > thresh,
        "s": mz < -thresh,
        "a": mx < -thresh,
        "d": mx > thresh,
        "shift": bool(payload.get("shift", False)),
    }
    if not (
        out["w"]
        or out["s"]
        or out["a"]
        or out["d"]
        or abs(out["camera_dx"]) > 1e-6
        or abs(out["camera_dy"]) > 1e-6
    ):
        return None
    return out
