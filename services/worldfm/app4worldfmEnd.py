from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
WORLDFM_ROOT = ROOT / "models" / "worldfm"

if str(WORLDFM_ROOT) not in sys.path:
    sys.path.insert(0, str(WORLDFM_ROOT))

import run_pipeline as worldfm_pipeline  # type: ignore
from run_pipeline import (  # type: ignore
    DEFAULT_CFG,
    setup_external_repos,
    step1_panogen,
    step2_moge_pipeline,
    step3_init,
    step3_render_one,
    step4_infer_one,
    step4_init,
)


class LoadRequest(BaseModel):
    model_id: Optional[str] = "worldfm"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None
    cache_key: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None
    cache_key: Optional[str] = None


class ProgressRequest(BaseModel):
    request_id: Optional[str] = None


@dataclass
class SessionState:
    session_id: str
    session_dir: Path
    K: np.ndarray
    c2w: np.ndarray
    yaw: float
    pitch: float
    pp_result: Any
    renderer: Any
    cond_db: Any
    rcfg: Any
    render_size: int
    cache_key: Optional[str] = None
    cache_export_idx: int = 0


@dataclass
class Runtime:
    cfg: Any
    device: torch.device
    svc: Optional[Any]
    wcfg: Optional[Any]
    session: Optional[SessionState] = None


class WorldFMRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        self.runtime: Optional[Runtime] = None
        self._runtime_root = ROOT / "outputs" / "worldfm" / "sessions"
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._export_root = Path(
            os.getenv("WM_WORLDFM_EXPORT_ROOT", str(WORLDFM_ROOT / "outputs")).strip()
            or str(WORLDFM_ROOT / "outputs")
        )

        # action -> pose increments (per /sessions/step call)
        self.move_step = float(os.getenv("WM_WORLDFM_MOVE_STEP", "0.03"))
        self.yaw_step = math.radians(float(os.getenv("WM_WORLDFM_YAW_DEG", "3.0")))
        self.pitch_step = math.radians(float(os.getenv("WM_WORLDFM_PITCH_DEG", "2.0")))
        self.max_pitch = math.radians(float(os.getenv("WM_WORLDFM_MAX_PITCH_DEG", "60.0")))
        self.default_focal_ratio = float(os.getenv("WM_WORLDFM_FOCAL_RATIO", "0.625"))
        self.max_input_side = int(os.getenv("WM_WORLDFM_MAX_INPUT_SIDE", "1280"))
        # Control direction conventions:
        # - frontend camera_dy is negative when dragging upward.
        # - some scenes/c2w conventions may mirror A/D.
        self.invert_pitch = os.getenv("WM_WORLDFM_INVERT_PITCH", "1") == "1"
        self.invert_ad = os.getenv("WM_WORLDFM_INVERT_AD", "1") == "1"
        self._progress_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._start_inflight = False
        self._progress: Dict[str, Any] = {
            "request_id": None,
            "phase": "idle",
            "message": "idle",
            "active": False,
            "session_id": None,
            "updated_at": time.time(),
        }
        # Monkey-patch pipeline logger to avoid BrokenPipe crashing service.
        worldfm_pipeline._log = self._pipeline_log

    def load(self) -> Dict[str, Any]:
        self._log("load requested")
        if self.runtime is not None:
            return {
                "model_id": "worldfm",
                "status": "already_loaded",
                "device": str(self.runtime.device),
            }

        cfg = self._load_cfg()
        gpu_index = int(cfg.pipeline.gpu_index)
        if torch.cuda.is_available() and gpu_index >= 0:
            torch.cuda.set_device(gpu_index)
            device = torch.device(f"cuda:{gpu_index}")
        else:
            device = torch.device("cpu")

        setup_external_repos(
            hw_path=str(cfg.submodules.hw_path),
            moge_path=str(cfg.submodules.moge_path),
        )
        # Keep /load lightweight. Step4 model will be lazily initialized on first /sessions/step.
        self.runtime = Runtime(cfg=cfg, device=device, svc=None, wcfg=None)

        self._log(f"load done device={device}")
        return {
            "model_id": "worldfm",
            "status": "loaded",
            "device": str(device),
            "worldfm_step": int(cfg.worldfm.step),
            "render_size": int(cfg.render.render_size),
            "step4_status": "lazy",
        }

    def start_session(self, init_image_base64: Optional[str], cache_key: Optional[str] = None) -> Dict[str, Any]:
        self._log("start_session requested")
        with self._start_lock:
            if self._start_inflight:
                raise RuntimeError("start/reset already in progress, please poll /sessions/progress")
            self._start_inflight = True
        try:
            if self.runtime is None:
                self._log("runtime not loaded, auto-loading before start")
                self.load()
            session_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            self._set_progress(
                request_id=request_id,
                phase="queued",
                message="start request accepted",
                active=True,
                session_id=session_id,
            )
            frame = self._build_session(
                session_id, init_image_base64, request_id=request_id, cache_key=cache_key
            )
            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": frame}
        finally:
            with self._start_lock:
                self._start_inflight = False

    def reset_session(
        self, session_id: str, init_image_base64: Optional[str], cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._start_lock:
            if self._start_inflight:
                raise RuntimeError("start/reset already in progress, please poll /sessions/progress")
            self._start_inflight = True
        try:
            runtime = self._require_runtime()
            prev = self._require_session(runtime, session_id)
            prev_cache = prev.cache_key
            request_id = str(uuid.uuid4())
            self._set_progress(
                request_id=request_id,
                phase="queued",
                message="reset request accepted",
                active=True,
                session_id=session_id,
            )
            ck = cache_key if cache_key is not None else prev_cache
            frame = self._build_session(
                session_id, init_image_base64, request_id=request_id, cache_key=ck
            )
            return {"session_id": session_id, "frame_base64": frame}
        finally:
            with self._start_lock:
                self._start_inflight = False

    @torch.inference_mode()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)
        self._ensure_step4(runtime)
        self._apply_action(session, action)
        frame = self._render_current_frame(runtime, session)
        self._maybe_export_frame_png(session, frame)
        return {
            "session_id": session.session_id,
            "frame_base64": frame,
            "reward": 0.0,
            "ended": False,
            "truncated": False,
            "extra": {
                "yaw_deg": float(np.degrees(session.yaw)),
                "pitch_deg": float(np.degrees(session.pitch)),
                "position": session.c2w[:3, 3].astype(float).tolist(),
            },
        }

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "worldfm",
            "ready": self.runtime is not None,
            "session_id": None if self.runtime is None or self.runtime.session is None else self.runtime.session.session_id,
        }

    def progress(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        with self._progress_lock:
            p = dict(self._progress)
        if request_id and p.get("request_id") and request_id != p.get("request_id"):
            return {
                "request_id": request_id,
                "phase": "unknown",
                "message": "request_id not found",
                "active": False,
                "session_id": None,
                "updated_at": time.time(),
            }
        return p

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _log(self, message: str) -> None:
        text = f"[service][worldfm] {message}"
        try:
            print(text, flush=True)
        except BrokenPipeError:
            # Service can outlive parent process that owned stdout pipe.
            pass
        except OSError:
            pass

    def _pipeline_log(self, step: str, message: str) -> None:
        text = f"[WorldFM][{step}] {message}"
        try:
            print(text, flush=True)
        except BrokenPipeError:
            pass
        except OSError:
            pass


    def _sanitize_cache_key(self, key: str) -> str:
        s = "".join(c if c.isalnum() or c in "._-" else "_" for c in key.strip())[:128]
        return s or "scene"

    def _maybe_export_frame_png(self, session: SessionState, frame_base64: str) -> None:
        if not session.cache_key:
            return
        if os.getenv("WM_WORLDFM_EXPORT_FRAMES", "1").strip().lower() in ("0", "false", "no", "off"):
            return
        try:
            sub = self._sanitize_cache_key(session.cache_key)
            out_dir = self._export_root / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = session.cache_export_idx
            out_path = out_dir / f"output_{idx:04d}.png"
            raw = base64.b64decode(frame_base64)
            out_path.write_bytes(raw)
            session.cache_export_idx = idx + 1
            self._log(f"exported frame -> {out_path}")
            if os.getenv("WM_WORLDFM_EXPORT_KEYFRAMES", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            ):
                kf_path = out_dir / "keyframes.json"
                yaw_deg = float(np.degrees(session.yaw))
                pitch_deg = float(np.degrees(session.pitch))
                entry = {"index": idx, "yaw_deg": yaw_deg, "pitch_deg": pitch_deg}
                rows: list[Dict[str, Any]] = []
                if kf_path.is_file():
                    try:
                        prev = json.loads(kf_path.read_text(encoding="utf-8"))
                        if isinstance(prev, list):
                            rows = [
                                r
                                for r in prev
                                if isinstance(r, dict) and int(r.get("index", -1)) != idx
                            ]
                    except (json.JSONDecodeError, OSError, ValueError):
                        rows = []
                rows.append(entry)
                rows.sort(key=lambda r: int(r.get("index", 0)))
                kf_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log(f"frame export skipped: {exc}")

    def _set_progress(
        self,
        *,
        request_id: Optional[str] = None,
        phase: str,
        message: str,
        active: bool,
        session_id: Optional[str],
    ) -> None:
        with self._progress_lock:
            if request_id is not None:
                self._progress["request_id"] = request_id
            self._progress["phase"] = phase
            self._progress["message"] = message
            self._progress["active"] = active
            self._progress["session_id"] = session_id
            self._progress["updated_at"] = time.time()

    def _build_session(
        self,
        session_id: str,
        init_image_base64: Optional[str],
        *,
        request_id: str,
        cache_key: Optional[str] = None,
    ) -> str:
        runtime = self._require_runtime()
        init_image = self._decode_image(init_image_base64)
        if init_image is None:
            self._set_progress(
                request_id=request_id,
                phase="error",
                message="init_image_base64 is required",
                active=False,
                session_id=session_id,
            )
            raise RuntimeError("init_image_base64 is required for worldfm start/reset")

        if runtime.session is not None:
            self._cleanup_session(runtime.session)
        # Critical for OOM prevention: start/reset (step1-3) must not overlap with step4 weights.
        self._unload_step4(runtime)

        session_dir = self._runtime_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        input_path = session_dir / "input.png"
        input_path.write_bytes(init_image)
        if self.max_input_side > 0:
            self._downscale_input_if_needed(input_path)

        try:
            self._set_progress(
                request_id=request_id,
                phase="step1",
                message="running step1 panorama generation",
                active=True,
                session_id=session_id,
            )
            panorama = step1_panogen(str(input_path), session_dir, cfg=runtime.cfg)

            self._set_progress(
                request_id=request_id,
                phase="step2",
                message="running step2 depth and condition preprocessing",
                active=True,
                session_id=session_id,
            )
            pp_result = step2_moge_pipeline(panorama, session_dir, cfg=runtime.cfg)

            self._set_progress(
                request_id=request_id,
                phase="step3",
                message="running step3 renderer and condition db initialization",
                active=True,
                session_id=session_id,
            )
            renderer, cond_db, rcfg, render_size = step3_init(pp_result, cfg=runtime.cfg)

            K = self._default_intrinsics(render_size)
            yaw, pitch = 0.0, 0.0
            c2w = self._c2w_from_yaw_pitch(yaw, pitch)

            self._set_progress(
                request_id=request_id,
                phase="ready",
                message="session ready (step1-3 complete, waiting for step input)",
                active=True,
                session_id=session_id,
            )
            session = SessionState(
                session_id=session_id,
                session_dir=session_dir,
                K=K,
                c2w=c2w,
                yaw=yaw,
                pitch=pitch,
                pp_result=pp_result,
                renderer=renderer,
                cond_db=cond_db,
                rcfg=rcfg,
                render_size=render_size,
                cache_key=cache_key,
                cache_export_idx=0,
            )
            runtime.session = session
            # Start returns step3 render preview; step4 runs on /sessions/step only.
            render_u8, _ = step3_render_one(
                session.renderer,
                session.cond_db,
                session.pp_result,
                session.K,
                session.c2w,
                rcfg=session.rcfg,
                render_size=session.render_size,
            )
            frame = self._frame_from_render_u8(render_u8)
            self._maybe_export_frame_png(session, frame)
            self._set_progress(
                request_id=request_id,
                phase="ready",
                message="session is ready",
                active=False,
                session_id=session_id,
            )
            return frame
        except Exception as exc:
            self._set_progress(
                request_id=request_id,
                phase="error",
                message=str(exc),
                active=False,
                session_id=session_id,
            )
            raise

    def _load_cfg(self):
        cfg = OmegaConf.create(DEFAULT_CFG)
        cfg_path = os.getenv("WM_WORLDFM_CONFIG", "").strip()
        if cfg_path:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))

        model_path = os.getenv("WM_WORLDFM_MODEL_PATH", "").strip()
        if model_path:
            cfg.worldfm.model_path = model_path

        vae_path = os.getenv("WM_WORLDFM_VAE_PATH", "").strip()
        if vae_path:
            cfg.worldfm.vae_path = vae_path

        step = os.getenv("WM_WORLDFM_STEP", "").strip()
        if step:
            cfg.worldfm.step = int(step)

        gpu_index = os.getenv("WM_WORLDFM_GPU_INDEX", "").strip()
        if gpu_index:
            cfg.pipeline.gpu_index = int(gpu_index)

        # Low-memory defaults for step1 (keeps behavior stable on 24GB cards).
        if os.getenv("WM_WORLDFM_PANOGEN_FP8", "1") == "1":
            cfg.panogen.fp8_attention = True
            cfg.panogen.fp8_gemm = True
        if os.getenv("WM_WORLDFM_PANOGEN_CACHE", "0") == "0":
            cfg.panogen.cache = False

        # Normalize checkpoint paths against models/worldfm root, not service cwd.
        model_path_cfg = str(cfg.worldfm.model_path)
        model_path_p = Path(model_path_cfg)
        if not model_path_p.is_absolute():
            cfg.worldfm.model_path = str((WORLDFM_ROOT / model_path_p).resolve())

        vae_path_cfg = str(cfg.worldfm.vae_path)
        vae_path_p = Path(vae_path_cfg)
        if not vae_path_p.is_absolute():
            cfg.worldfm.vae_path = str((WORLDFM_ROOT / vae_path_p).resolve())

        return cfg

    def _require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise RuntimeError("Model is not loaded. Call /load first.")
        return self.runtime

    def _require_session(self, runtime: Runtime, session_id: str) -> SessionState:
        if runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return runtime.session

    def _cleanup_session(self, session: SessionState) -> None:
        try:
            del session.renderer
            del session.cond_db
            del session.pp_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _unload_step4(self, runtime: Runtime) -> None:
        if runtime.svc is None and runtime.wcfg is None:
            return
        self._log("unloading step4 runtime before step1-3")
        try:
            if runtime.svc is not None:
                del runtime.svc
            runtime.svc = None
            runtime.wcfg = None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _default_intrinsics(self, render_size: int) -> np.ndarray:
        s = float(render_size)
        f = self.default_focal_ratio * s
        c = s / 2.0
        return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _basis_from_yaw_pitch(self, yaw: float, pitch: float):
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        f = np.array([cyaw * cp, sp, syaw * cp], dtype=np.float64)
        f = f / max(1e-8, np.linalg.norm(f))
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(f, up)
        r = r / max(1e-8, np.linalg.norm(r))
        u = np.cross(r, f)
        u = u / max(1e-8, np.linalg.norm(u))
        return r, u, f

    def _c2w_from_yaw_pitch(self, yaw: float, pitch: float, pos: Optional[np.ndarray] = None) -> np.ndarray:
        r, u, f = self._basis_from_yaw_pitch(yaw, pitch)
        z = -f
        M = np.eye(4, dtype=np.float64)
        M[:3, 0] = r
        M[:3, 1] = u
        M[:3, 2] = z
        A = np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64))
        c2w = A @ M @ A
        if pos is not None:
            c2w[:3, 3] = pos.astype(np.float64)
        return c2w

    def _apply_action(self, session: SessionState, action: Dict[str, Any]) -> None:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)

        session.yaw += dx * self.yaw_step
        if self.invert_pitch:
            session.pitch -= dy * self.pitch_step
        else:
            session.pitch += dy * self.pitch_step
        session.pitch = float(np.clip(session.pitch, -self.max_pitch, self.max_pitch))

        pos = session.c2w[:3, 3].astype(np.float64).copy()
        _, _, f = self._basis_from_yaw_pitch(session.yaw, session.pitch)
        f_flat = np.array([f[0], 0.0, f[2]], dtype=np.float64)
        if np.linalg.norm(f_flat) < 1e-8:
            f_flat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        f_flat = f_flat / np.linalg.norm(f_flat)
        right = np.array([-f_flat[2], 0.0, f_flat[0]], dtype=np.float64)

        move = np.zeros(3, dtype=np.float64)
        if bool(action.get("w", False)):
            move += f_flat
        if bool(action.get("s", False)):
            move -= f_flat
        if self.invert_ad:
            if bool(action.get("a", False)):
                move += right
            if bool(action.get("d", False)):
                move -= right
        else:
            if bool(action.get("a", False)):
                move -= right
            if bool(action.get("d", False)):
                move += right
        if bool(action.get("space", False)):
            move[1] += 1.0
        if bool(action.get("ctrl", False)):
            move[1] -= 1.0

        speed = self.move_step * (2.0 if bool(action.get("shift", False)) else 1.0)
        norm = float(np.linalg.norm(move))
        if norm > 1e-8:
            pos += move / norm * speed

        session.c2w = self._c2w_from_yaw_pitch(session.yaw, session.pitch, pos=pos)

    def _render_current_frame(self, runtime: Runtime, session: SessionState) -> str:
        if runtime.svc is None or runtime.wcfg is None:
            raise RuntimeError("step4 inference runtime is not initialized")
        render_u8, cond_nearest = step3_render_one(
            session.renderer,
            session.cond_db,
            session.pp_result,
            session.K,
            session.c2w,
            rcfg=session.rcfg,
            render_size=session.render_size,
        )
        frame_rgb = step4_infer_one(runtime.svc, render_u8, cond_nearest, wcfg=runtime.wcfg)
        return self._frame_to_base64(frame_rgb)

    def _ensure_step4(self, runtime: Runtime) -> None:
        if runtime.svc is not None and runtime.wcfg is not None:
            return
        self._log("initializing step4 runtime on first step")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        svc, wcfg = step4_init(cfg=runtime.cfg)
        runtime.svc, runtime.wcfg = svc, wcfg

    def _frame_from_render_u8(self, render_rgb_u8: torch.Tensor) -> str:
        arr = render_rgb_u8.detach().cpu().numpy().astype(np.uint8)
        return self._frame_to_base64(arr)

    def _frame_to_base64(self, frame_rgb: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _downscale_input_if_needed(self, input_path: Path) -> None:
        try:
            img = Image.open(input_path).convert("RGB")
            w, h = img.size
            m = max(w, h)
            if m <= self.max_input_side:
                return
            s = float(self.max_input_side) / float(m)
            nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            img = img.resize((nw, nh), resample=Image.BICUBIC)
            img.save(input_path)
            self._log(f"downscaled input from {w}x{h} to {nw}x{nh} for step1 memory")
        except Exception as exc:
            self._log(f"input downscale skipped due to error: {exc}")


svc = WorldFMRuntimeService()
app = FastAPI(title="WMFactory WorldFM Service")


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
        return svc.start_session(req.init_image_base64, cache_key=req.cache_key)
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
        return svc.reset_session(req.session_id, req.init_image_base64, cache_key=req.cache_key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/progress")
def progress(req: ProgressRequest) -> Dict[str, Any]:
    try:
        return svc.progress(req.request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
