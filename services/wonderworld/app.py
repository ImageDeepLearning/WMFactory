from __future__ import annotations

import base64
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import socketio
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
WONDER_ROOT = ROOT / "models" / "WonderWorld"


class LoadRequest(BaseModel):
    model_id: Optional[str] = "wonderworld"


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
    backend_port: int
    fallback_frame_b64: str
    yaw: float = 0.0
    pitch: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Runtime:
    loaded: bool = False
    session: Optional[SessionState] = None


class WonderWorldRuntimeService:
    def __init__(self) -> None:
        self.runtime = Runtime()

        self.model_python = os.getenv(
            "WM_WONDERWORLD_PYTHON",
            str(ROOT / "venvs" / "WonderWorld" / "bin" / "python"),
        )
        self.model_dir = Path(os.getenv("WM_WONDERWORLD_MODEL_DIR", str(WONDER_ROOT))).resolve()
        self.model_port = int(os.getenv("WM_WONDERWORLD_MODEL_PORT", "17777"))
        self.model_start_timeout = float(os.getenv("WM_WONDERWORLD_START_TIMEOUT", "7200"))
        self.model_step_timeout = float(os.getenv("WM_WONDERWORLD_STEP_TIMEOUT", "3"))
        self.model_health_timeout = float(os.getenv("WM_WONDERWORLD_HEALTH_TIMEOUT", "600"))
        self.move_step = float(os.getenv("WM_WONDERWORLD_MOVE_STEP", "0.03"))
        self.yaw_step = float(os.getenv("WM_WONDERWORLD_YAW_STEP", "0.06"))
        self.pitch_step = float(os.getenv("WM_WONDERWORLD_PITCH_STEP", "0.05"))
        self.max_pitch = float(os.getenv("WM_WONDERWORLD_MAX_PITCH", "1.2"))
        self.step_log_every = int(os.getenv("WM_WONDERWORLD_STEP_LOG_EVERY", "20"))
        self._step_counter = 0

        self.style_prompt = os.getenv("WM_WONDERWORLD_STYLE_PROMPT", "DSLR 35mm landscape")
        self.content_prompt = os.getenv(
            "WM_WONDERWORLD_CONTENT_PROMPT",
            "Interactive world, building, road, sky",
        )
        self.negative_prompt = os.getenv("WM_WONDERWORLD_NEGATIVE_PROMPT", "text")
        self.background_prompt = os.getenv("WM_WONDERWORLD_BACKGROUND_PROMPT", "")

        self._proc: Optional[subprocess.Popen[Any]] = None
        self._proc_lock = threading.Lock()

        self._sio: Optional[socketio.Client] = None
        self._frame_lock = threading.Lock()
        self._frame_cond = threading.Condition(self._frame_lock)
        self._latest_frame_png_b64: Optional[str] = None
        self._latest_frame_idx = 0
        self._connected = False

    def _log(self, message: str) -> None:
        print(f"[service][wonderworld] {message}", flush=True)

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "wonderworld",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
            "backend_alive": self._backend_alive(),
            "socket_connected": self._connected,
        }

    def load(self) -> Dict[str, Any]:
        self.runtime.loaded = True
        return {
            "model_id": "wonderworld",
            "status": "loaded",
            "device": "cuda",
            "backend_port": self.model_port,
        }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        if not self.runtime.loaded:
            self.load()

        init_bytes = self._decode_image(init_image_base64)
        if init_bytes is None:
            raise RuntimeError("init_image_base64 is required for wonderworld start")

        session_id = str(uuid.uuid4())
        self._prepare_runtime_example(init_bytes)

        self._restart_backend()
        self._connect_socket()

        fallback = self._image_bytes_to_png_b64(init_bytes)
        first_frame = self._wait_for_new_frame(prev_idx=-1, timeout=min(self.model_start_timeout, 180.0))
        if first_frame is None:
            self._log("start_session fallback: no streamed frame yet, using init image")
            first_frame = fallback

        self.runtime.session = SessionState(
            session_id=session_id,
            backend_port=self.model_port,
            fallback_frame_b64=fallback,
        )
        self._log(f"start_session done session_id={session_id}")
        return {"session_id": session_id, "frame_base64": first_frame}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        self._require_session(session_id)
        return self.start_session(init_image_base64)

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if self._sio is None:
            raise RuntimeError("WonderWorld socket is not connected")

        self._apply_action(session, action)
        view_matrix = self._compose_view_matrix(session)

        with self._frame_lock:
            prev_idx = self._latest_frame_idx

        self._step_counter += 1
        if self._step_counter % max(1, self.step_log_every) == 0:
            self._log(f"step #{self._step_counter} pose=({session.x:.3f},{session.y:.3f},{session.z:.3f})")

        self._sio.emit("render-pose", view_matrix)
        frame_b64 = self._wait_for_new_frame(prev_idx=prev_idx, timeout=self.model_step_timeout)
        if frame_b64 is None:
            with self._frame_lock:
                frame_b64 = self._latest_frame_png_b64
        if frame_b64 is None:
            frame_b64 = session.fallback_frame_b64

        return {
            "session_id": session_id,
            "frame_base64": frame_b64,
            "reward": 0.0,
            "ended": False,
            "truncated": False,
            "extra": {
                "yaw": float(session.yaw),
                "pitch": float(session.pitch),
                "position": [float(session.x), float(session.y), float(session.z)],
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

    def _prepare_runtime_example(self, init_image_bytes: bytes) -> None:
        images_dir = self.model_dir / "examples" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / "wmfactory_input.png"

        img = Image.open(io.BytesIO(init_image_bytes)).convert("RGB").resize((512, 512), resample=Image.BICUBIC)
        img.save(image_path)

        # Reuse available sky assets to avoid SyncDiffusion branch that may require unavailable checkpoints.
        # This keeps the original WonderWorld pipeline unchanged while making startup robust on hf-mirror.
        src_sky_dir = self.model_dir / "examples" / "sky_images" / "real_campus_2"
        dst_sky_dir = self.model_dir / "examples" / "sky_images" / "wmfactory_input"
        dst_sky_dir.mkdir(parents=True, exist_ok=True)
        for name in ("sky_0.png", "sky_1.png", "finished_3dgs_sky_tanh.ply"):
            src = src_sky_dir / name
            dst = dst_sky_dir / name
            if src.exists() and not dst.exists():
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

        examples_yaml = self.model_dir / "examples" / "examples.yaml"
        text = examples_yaml.read_text(encoding="utf-8")
        block = (
            "- name: wmfactory_input\n"
            "  image_filepath: examples/images/wmfactory_input.png\n"
            f"  style_prompt: {json.dumps(self.style_prompt)}\n"
            f"  content_prompt: {json.dumps(self.content_prompt)}\n"
            f"  negative_prompt: {json.dumps(self.negative_prompt)}\n"
            f"  background: {json.dumps(self.background_prompt)}\n"
        )

        marker = "- name: wmfactory_input\n"
        if marker in text:
            prefix = text.split(marker, 1)[0]
            rest = text.split(marker, 1)[1]
            lines = rest.splitlines()
            i = 0
            while i < len(lines) and not lines[i].startswith("- name:"):
                i += 1
            rest_tail = "\n".join(lines[i:])
            new_text = prefix.rstrip() + "\n" + block + ("\n" + rest_tail if rest_tail else "")
        else:
            new_text = text.rstrip() + "\n\n" + block
        examples_yaml.write_text(new_text.rstrip() + "\n", encoding="utf-8")

        config_path = self.model_dir / "config" / "wmfactory_runtime.yaml"
        config = {
            "runs_dir": "output/wmfactory_runtime",
            "example_name": "wmfactory_input",
            "seed": 1,
            "depth_conditioning": True,
            "use_gpt": False,
            "debug": False,
            "depth_model": "marigold",
            "camera_speed": 0.001,
            "fg_depth_range": 0.015,
            "depth_shift": 0.001,
            "sky_hard_depth": 0.02,
            "init_focal_length": 960,
            "gen_sky_image": False,
            "gen_sky": False,
            "gen_layer": True,
            "load_gen": False,
            "stable_diffusion_checkpoint": "runwayml/stable-diffusion-inpainting",
        }
        # Avoid adding a YAML dependency in gateway env.
        lines = [f"{k}: {json.dumps(v) if isinstance(v, str) else str(v)}" for k, v in config.items()]
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _backend_alive(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def _restart_backend(self) -> None:
        self._stop_backend()

        cmd = [
            self.model_python,
            "run.py",
            "--example_config",
            "config/wmfactory_runtime.yaml",
            "--port",
            str(self.model_port),
        ]
        env = os.environ.copy()
        env["http_proxy"] = ""
        env["https_proxy"] = ""
        env["HTTP_PROXY"] = ""
        env["HTTPS_PROXY"] = ""
        env.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        env.setdefault("OPENAI_API_KEY", "dummy")

        with self._proc_lock:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.model_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                env=env,
            )
            proc = self._proc

        assert proc is not None
        t = threading.Thread(target=self._stream_child_logs, args=(proc,), daemon=True)
        t.start()

        self._wait_port_ready(self.model_port, timeout=self.model_health_timeout)

    def _stream_child_logs(self, proc: subprocess.Popen[Any]) -> None:
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    self._log(f"[model] {line}")
        except Exception as exc:
            self._log(f"model log stream stopped: {exc}")

    def _stop_backend(self) -> None:
        self._disconnect_socket()
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    def _wait_port_ready(self, port: int, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return
            with self._proc_lock:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(f"WonderWorld process exited early with code {self._proc.returncode}")
            time.sleep(0.3)
        raise RuntimeError(f"Timeout waiting WonderWorld backend port {port}")

    def _connect_socket(self) -> None:
        self._disconnect_socket()

        sio = socketio.Client(logger=False, engineio_logger=False, reconnection=False)

        @sio.event
        def connect() -> None:
            self._connected = True
            self._log("socket connected")

        @sio.event
        def disconnect() -> None:
            self._connected = False
            self._log("socket disconnected")

        @sio.on("frame")
        def on_frame(data: Any) -> None:
            try:
                if isinstance(data, str):
                    payload = data.encode("latin1", errors="ignore")
                else:
                    payload = bytes(data)
                arr = np.frombuffer(payload, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    return
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ok, png = cv2.imencode(".png", rgb)
                if not ok:
                    return
                b64 = base64.b64encode(png.tobytes()).decode("utf-8")
                with self._frame_cond:
                    self._latest_frame_png_b64 = b64
                    self._latest_frame_idx += 1
                    self._frame_cond.notify_all()
            except Exception as exc:
                self._log(f"frame decode failed: {exc}")

        sio.connect(f"http://127.0.0.1:{self.model_port}", wait_timeout=30)
        sio.emit("start", {})
        self._sio = sio

    def _disconnect_socket(self) -> None:
        sio = self._sio
        self._sio = None
        self._connected = False
        if sio is not None:
            try:
                sio.disconnect()
            except Exception:
                pass

    def _wait_for_new_frame(self, prev_idx: int, timeout: float) -> Optional[str]:
        deadline = time.time() + timeout
        with self._frame_cond:
            while time.time() < deadline:
                if self._latest_frame_idx > prev_idx and self._latest_frame_png_b64 is not None:
                    return self._latest_frame_png_b64
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._frame_cond.wait(timeout=min(0.2, remaining))
            return None

    def _apply_action(self, session: SessionState, action: Dict[str, Any]) -> None:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        session.yaw += dx * self.yaw_step
        session.pitch = float(np.clip(session.pitch - dy * self.pitch_step, -self.max_pitch, self.max_pitch))

        forward = np.array([np.sin(session.yaw), 0.0, np.cos(session.yaw)], dtype=np.float32)
        right = np.array([forward[2], 0.0, -forward[0]], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)

        if bool(action.get("w", False)):
            move += forward
        if bool(action.get("s", False)):
            move -= forward
        if bool(action.get("a", False)):
            move -= right
        if bool(action.get("d", False)):
            move += right
        if bool(action.get("space", False)):
            move[1] += 1.0
        if bool(action.get("ctrl", False)):
            move[1] -= 1.0

        norm = float(np.linalg.norm(move))
        if norm > 1e-6:
            speed = self.move_step * (2.0 if bool(action.get("shift", False)) else 1.0)
            move = move / norm * speed
            session.x += float(move[0])
            session.y += float(move[1])
            session.z += float(move[2])

    def _compose_view_matrix(self, session: SessionState) -> list[float]:
        cy = float(np.cos(session.yaw))
        sy = float(np.sin(session.yaw))
        cp = float(np.cos(session.pitch))
        sp = float(np.sin(session.pitch))

        r_yaw = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            dtype=np.float32,
        )
        r_pitch = np.array(
            [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
            dtype=np.float32,
        )
        r = r_yaw @ r_pitch
        negate_xy = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = r @ negate_xy
        m[3, 0] = session.x
        m[3, 1] = session.y
        m[3, 2] = session.z
        return m.reshape(-1).astype(float).tolist()

    def _image_bytes_to_png_b64(self, data: bytes) -> str:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


svc = WonderWorldRuntimeService()
app = FastAPI(title="WMFactory WonderWorld Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return svc.health()


@app.post("/load")
def load(req: LoadRequest) -> Dict[str, Any]:
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
