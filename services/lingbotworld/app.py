from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import imageio
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
LINGBOT_ROOT = Path(os.getenv("WM_LINGBOTWORLD_ROOT", str(ROOT / "models" / "lingbot-world"))).resolve()
PREQUANT_ROOT = Path(
    os.getenv(
        "WM_LINGBOTWORLD_PREQUANT_ROOT",
        str(LINGBOT_ROOT / "lingbot-world-base-cam-nf4"),
    )
).resolve()
EXAMPLE_ROOT = Path(
    os.getenv(
        "WM_LINGBOTWORLD_EXAMPLE_ROOT",
        str(LINGBOT_ROOT / "examples" / "00"),
    )
).resolve()

if str(PREQUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(PREQUANT_ROOT))

from generate_prequant import WanI2V_PreQuant  # type: ignore  # noqa: E402


class LoadRequest(BaseModel):
    model_id: Optional[str] = "lingbot-world"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


@dataclass
class SessionState:
    session_id: str
    session_dir: Path
    current_frame_path: Path
    current_video_path: Optional[Path]
    current_frame_b64: str
    current_pose: np.ndarray
    current_seed_image: Image.Image
    started_at: float
    step_count: int = 0
    last_action: Optional[Dict[str, Any]] = None
    last_motion: Optional[Dict[str, float]] = None


@dataclass
class Runtime:
    pipeline: Optional[WanI2V_PreQuant] = None
    loaded: bool = False
    session: Optional[SessionState] = None
    device: str = "cuda:0"


class LingBotWorldRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        self.runtime = Runtime()
        self._lock = threading.Lock()
        self.session_root = Path(
            os.getenv(
                "WM_LINGBOTWORLD_SESSION_DIR",
                str(ROOT / "outputs" / "lingbotworld" / "sessions"),
            )
        ).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)

        self.prompt = os.getenv(
            "WM_LINGBOTWORLD_PROMPT",
            (
                "Continue this first-person world naturally with coherent egomotion, "
                "stable geometry, realistic scene persistence, and smooth camera motion."
            ),
        )
        self.size = os.getenv("WM_LINGBOTWORLD_SIZE", "480*832")
        size_h, size_w = self.size.split("*", 1)
        self.height = int(size_h)
        self.width = int(size_w)
        self.max_area = self.height * self.width
        self.frame_num = int(os.getenv("WM_LINGBOTWORLD_FRAME_NUM", "17"))
        self.sampling_steps = int(os.getenv("WM_LINGBOTWORLD_SAMPLING_STEPS", "10"))
        self.guide_scale = float(os.getenv("WM_LINGBOTWORLD_GUIDE_SCALE", "5.0"))
        self.shift = float(os.getenv("WM_LINGBOTWORLD_SHIFT", "5.0"))
        self.seed = int(os.getenv("WM_LINGBOTWORLD_SEED", "42"))
        self.t5_cpu = os.getenv("WM_LINGBOTWORLD_T5_CPU", "1") == "1"

        self.camera_deadzone = float(os.getenv("WM_LINGBOTWORLD_CAMERA_DEADZONE", "0.08"))
        self.forward_step = float(os.getenv("WM_LINGBOTWORLD_FORWARD_STEP", "0.08"))
        self.strafe_step = float(os.getenv("WM_LINGBOTWORLD_STRAFE_STEP", "0.06"))
        self.vertical_step = float(os.getenv("WM_LINGBOTWORLD_VERTICAL_STEP", "0.04"))
        self.yaw_step_deg = float(os.getenv("WM_LINGBOTWORLD_YAW_DEG", "3.0"))
        self.pitch_step_deg = float(os.getenv("WM_LINGBOTWORLD_PITCH_DEG", "2.5"))
        self.invert_yaw = os.getenv("WM_LINGBOTWORLD_INVERT_YAW", "1") == "1"
        self.invert_pitch = os.getenv("WM_LINGBOTWORLD_INVERT_PITCH", "0") == "1"

        intrinsics = np.load(EXAMPLE_ROOT / "intrinsics.npy")
        self.base_intrinsics = intrinsics[0].astype(np.float32)

    def _log(self, message: str) -> None:
        print(f"[service][lingbot-world] {message}", flush=True)

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "lingbot-world",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            self._log("load requested")
            if self.runtime.loaded and self.runtime.pipeline is not None:
                return {
                    "model_id": "lingbot-world",
                    "status": "already_loaded",
                    "device": self.runtime.device,
                    "sampling_steps": self.sampling_steps,
                    "frame_num": self.frame_num,
                    "size": self.size,
                    "variant": "cam-nf4",
                }

            pipeline = WanI2V_PreQuant(
                checkpoint_dir=str(PREQUANT_ROOT),
                device_id=0,
                t5_cpu=self.t5_cpu,
            )
            self.runtime.pipeline = pipeline
            self.runtime.loaded = True
            self.runtime.device = str(pipeline.device)
            self._log(f"load done device={self.runtime.device}")
            return {
                "model_id": "lingbot-world",
                "status": "loaded",
                "device": self.runtime.device,
                "sampling_steps": self.sampling_steps,
                "frame_num": self.frame_num,
                "size": self.size,
                "variant": "cam-nf4",
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for LingBot-World start")
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            session_id = str(uuid.uuid4())
            session_dir = self.session_root / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            current_frame_path = session_dir / "current.png"
            image, frame_b64 = self._write_current_image(init_image_bytes, current_frame_path)
            self.runtime.session = SessionState(
                session_id=session_id,
                session_dir=session_dir,
                current_frame_path=current_frame_path,
                current_video_path=None,
                current_frame_b64=frame_b64,
                current_pose=np.eye(4, dtype=np.float32),
                current_seed_image=image,
                started_at=time.time(),
            )
            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64)

        with self._lock:
            if init_image_bytes is not None:
                image, frame_b64 = self._write_current_image(init_image_bytes, session.current_frame_path)
                session.current_seed_image = image
                session.current_frame_b64 = frame_b64
            session.current_pose = np.eye(4, dtype=np.float32)
            session.current_video_path = None
            session.step_count = 0
            session.started_at = time.time()
            session.last_action = None
            session.last_motion = None
            self._log(f"reset_session done session_id={session_id}")
            return {"session_id": session.session_id, "frame_base64": session.current_frame_b64}

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if not self.runtime.loaded or self.runtime.pipeline is None:
            self.load()

        with self._lock:
            t0 = time.perf_counter()
            motion = self._motion_from_action(action)
            poses = self._build_chunk_poses(session.current_pose, motion)
            intrinsics = np.repeat(self.base_intrinsics[None, :], poses.shape[0], axis=0)

            chunk_dir = session.session_dir / f"chunk_{session.step_count:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            np.save(chunk_dir / "poses.npy", poses.astype(np.float32))
            np.save(chunk_dir / "intrinsics.npy", intrinsics.astype(np.float32))

            video = self.runtime.pipeline.generate(
                input_prompt=self.prompt,
                img=session.current_seed_image,
                action_path=str(chunk_dir),
                max_area=self.max_area,
                frame_num=self.frame_num,
                shift=self.shift,
                sampling_steps=self.sampling_steps,
                guide_scale=self.guide_scale,
                seed=self.seed + session.step_count,
            )

            video_path = chunk_dir / "chunk.mp4"
            self._save_video(video, video_path, fps=16)
            next_image, frame_b64 = self._extract_last_frame(video)
            next_image.save(session.current_frame_path, format="PNG")

            session.current_seed_image = next_image
            session.current_frame_b64 = frame_b64
            session.current_video_path = video_path
            session.current_pose = poses[-1].astype(np.float32)
            session.step_count += 1
            session.last_action = dict(action)
            session.last_motion = dict(motion)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            movement_key = self._movement_key(action)
            camera_key = self._camera_key(action)
            self._log(
                f"step done session_id={session.session_id} step={session.step_count} "
                f"move={movement_key} camera={camera_key} latency_ms={latency_ms}"
            )
            return {
                "session_id": session.session_id,
                "frame_base64": frame_b64,
                "reward": 0.0,
                "ended": False,
                "truncated": False,
                "extra": {
                    "latency_ms": latency_ms,
                    "step_count": session.step_count,
                    "movement_key": movement_key,
                    "camera_key": camera_key,
                    "motion": motion,
                    "video_path": str(video_path),
                    "sampling_steps": self.sampling_steps,
                    "frame_num": self.frame_num,
                },
            }

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _write_current_image(self, init_image_bytes: bytes, current_frame_path: Path) -> tuple[Image.Image, str]:
        image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        image.save(current_frame_path, format="PNG")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return image, frame_b64

    def _movement_key(self, action: Dict[str, Any]) -> str:
        forward = bool(action.get("w")) and not bool(action.get("s"))
        backward = bool(action.get("s")) and not bool(action.get("w"))
        left = bool(action.get("a")) and not bool(action.get("d"))
        right = bool(action.get("d")) and not bool(action.get("a"))

        if forward and left:
            return "W+A"
        if forward and right:
            return "W+D"
        if backward and left:
            return "S+A"
        if backward and right:
            return "S+D"
        if forward:
            return "W"
        if backward:
            return "S"
        if left:
            return "A"
        if right:
            return "D"
        return "None"

    def _camera_key(self, action: Dict[str, Any]) -> str:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        if self.invert_yaw:
            dx = -dx
        if self.invert_pitch:
            dy = -dy
        horiz = ""
        vert = ""
        if dx >= self.camera_deadzone:
            horiz = "→"
        elif dx <= -self.camera_deadzone:
            horiz = "←"
        if dy >= self.camera_deadzone:
            vert = "↓"
        elif dy <= -self.camera_deadzone:
            vert = "↑"
        return f"{vert}{horiz}" or "·"

    def _motion_from_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        if abs(dx) <= self.camera_deadzone:
            dx = 0.0
        if abs(dy) <= self.camera_deadzone:
            dy = 0.0

        speed_scale = 2.0 if bool(action.get("shift", False)) else 1.0
        vertical = 0.0
        if bool(action.get("space", False)) and not bool(action.get("ctrl", False)):
            vertical += self.vertical_step
        if bool(action.get("ctrl", False)) and not bool(action.get("space", False)):
            vertical -= self.vertical_step

        motion: Dict[str, float] = {}
        forward = (
            (1.0 if bool(action.get("w", False)) else 0.0)
            - (1.0 if bool(action.get("s", False)) else 0.0)
        ) * self.forward_step * speed_scale
        right = (
            (1.0 if bool(action.get("d", False)) else 0.0)
            - (1.0 if bool(action.get("a", False)) else 0.0)
        ) * self.strafe_step * speed_scale
        yaw_input = -dx if self.invert_yaw else dx
        yaw = yaw_input * np.deg2rad(self.yaw_step_deg)
        pitch_input = -dy if self.invert_pitch else dy
        pitch = pitch_input * np.deg2rad(self.pitch_step_deg)

        if abs(forward) > 1e-8:
            motion["forward"] = float(forward)
        if abs(right) > 1e-8:
            motion["right"] = float(right)
        if abs(vertical) > 1e-8:
            motion["vertical"] = float(vertical)
        if abs(yaw) > 1e-8:
            motion["yaw"] = float(yaw)
        if abs(pitch) > 1e-8:
            motion["pitch"] = float(pitch)
        return motion

    def _build_chunk_poses(self, start_pose: np.ndarray, motion: Dict[str, float]) -> np.ndarray:
        pose = start_pose.astype(np.float64).copy()
        poses = [pose.copy()]
        for _ in range(self.frame_num - 1):
            pose = self._advance_pose(pose, motion)
            poses.append(pose.copy())
        return np.stack(poses, axis=0)

    def _advance_pose(self, pose: np.ndarray, motion: Dict[str, float]) -> np.ndarray:
        updated = pose.copy()
        rot = Rotation.from_matrix(updated[:3, :3])
        yaw = motion.get("yaw", 0.0)
        pitch = motion.get("pitch", 0.0)
        if abs(yaw) > 0.0:
            rot = rot * Rotation.from_rotvec(np.array([0.0, yaw, 0.0], dtype=np.float64))
        if abs(pitch) > 0.0:
            rot = rot * Rotation.from_rotvec(np.array([pitch, 0.0, 0.0], dtype=np.float64))

        rotation_matrix = rot.as_matrix()
        local_translation = np.array(
            [
                motion.get("right", 0.0),
                -motion.get("vertical", 0.0),
                motion.get("forward", 0.0),
            ],
            dtype=np.float64,
        )
        updated[:3, :3] = rotation_matrix
        updated[:3, 3] = updated[:3, 3] + rotation_matrix @ local_translation
        return updated

    def _extract_last_frame(self, video: torch.Tensor) -> tuple[Image.Image, str]:
        frame = video[:, -1, :, :].detach().float().clamp(-1, 1)
        frame = frame.add(1.0).div(2.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(frame)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return image, base64.b64encode(buf.getvalue()).decode("utf-8")

    def _save_video(self, frames: torch.Tensor, output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames_np = ((frames + 1) / 2 * 255).clamp(0, 255).byte()
        frames_np = frames_np.permute(1, 2, 3, 0).cpu().numpy()
        imageio.mimwrite(output_path, frames_np, fps=fps, codec="libx264")


svc = LingBotWorldRuntimeService()
app = FastAPI(title="WMFactory LingBot-World Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return svc.health()


@app.post("/load")
def load(_: LoadRequest) -> Dict[str, Any]:
    try:
        return svc.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/start")
def start(req: StartRequest) -> Dict[str, Any]:
    try:
        return svc.start_session(req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/step")
def step(req: StepRequest) -> Dict[str, Any]:
    try:
        return svc.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return svc.reset_session(req.session_id, req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
