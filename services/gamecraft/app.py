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

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
GAMECRAFT_ROOT = Path(os.getenv("WM_GAMECRAFT_ROOT", str(ROOT / "models" / "Hunyuan-GameCraft-1.0"))).resolve()

os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("CPU_OFFLOAD", "1")
os.environ.setdefault("MODEL_BASE", str(GAMECRAFT_ROOT / "weights" / "stdmodels"))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.getenv("WM_GAMECRAFT_MASTER_PORT", "29612"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("WM_GAMECRAFT_CUDA_VISIBLE_DEVICES", "1"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if str(GAMECRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(GAMECRAFT_ROOT))


class LoadRequest(BaseModel):
    model_id: Optional[str] = "gamecraft"


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
    init_image_bytes: bytes
    seed_frame_b64: str
    last_frame_b64: str
    raw_ref_images: list[Image.Image]
    ref_latents: torch.Tensor
    last_latents: torch.Tensor
    started_at: float
    step_count: int = 0
    last_action: Optional[Dict[str, Any]] = None


@dataclass
class Runtime:
    session: Optional[SessionState] = None
    loaded: bool = False
    device: str = "cuda:0"


@dataclass
class ResolvedAction:
    action_id: Optional[str]
    action_speed: float
    movement_key: str
    camera_key: str
    source: str
    reason: str


class CropResize:
    def __init__(self, size: tuple[int, int] = (704, 1216)) -> None:
        self.target_h, self.target_w = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = max(self.target_w / w, self.target_h / h)
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        resized_img = resize_transform(img)
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)


class GameCraftRuntimeService:
    def __init__(self) -> None:
        self.runtime = Runtime(device="cuda:0")
        self._lock = threading.Lock()
        self.prompt = os.getenv(
            "WM_GAMECRAFT_PROMPT",
            "A high-quality realistic first-person gameplay video with coherent motion and stable scene geometry.",
        )
        self.negative_prompt = os.getenv(
            "WM_GAMECRAFT_NEG_PROMPT",
            "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
        )
        self.move_speed = float(os.getenv("WM_GAMECRAFT_MOVE_SPEED", "0.2"))
        self.camera_deadzone = float(os.getenv("WM_GAMECRAFT_CAMERA_DEADZONE", "0.18"))
        self.rotate_speed_min = float(os.getenv("WM_GAMECRAFT_ROTATE_SPEED_MIN", "4.0"))
        self.rotate_speed_max = float(os.getenv("WM_GAMECRAFT_ROTATE_SPEED_MAX", "12.0"))
        self.seed = int(os.getenv("WM_GAMECRAFT_SEED", "250160"))
        self.output_dir = Path(os.getenv("WM_GAMECRAFT_OUTPUT_DIR", str(ROOT / "outputs" / "gamecraft"))).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gamecraft = None
        self.parse_args = None
        self.initialize_distributed = None
        self.nccl_info = None
        self.ref_image_transform = transforms.Compose(
            [
                CropResize((704, 1216)),
                transforms.CenterCrop((704, 1216)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _log(self, message: str) -> None:
        print(f"[service][gamecraft] {message}", flush=True)

    def _lazy_imports(self) -> None:
        if self.gamecraft is not None:
            return
        from diffusers.hooks import apply_group_offloading
        from hymm_sp.config import parse_args
        from hymm_sp.modules.parallel_states import initialize_distributed, nccl_info
        from hymm_sp.sample_inference import HunyuanVideoSampler

        self.gamecraft = {
            "apply_group_offloading": apply_group_offloading,
            "HunyuanVideoSampler": HunyuanVideoSampler,
        }
        self.parse_args = parse_args
        self.initialize_distributed = initialize_distributed
        self.nccl_info = nccl_info

    def _default_args(self) -> Any:
        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            return self.parse_args()
        finally:
            sys.argv = argv_backup

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "gamecraft",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if self.runtime.loaded:
                args = self.runtime_args
                return {
                    "model_id": "gamecraft",
                    "status": "already_loaded",
                    "device": self.runtime.device,
                    "infer_steps": args.infer_steps,
                    "cfg_scale": args.cfg_scale,
                    "video_size": [704, 1216],
                }

            self._lazy_imports()
            if not torch.distributed.is_initialized():
                self.initialize_distributed(self.seed)

            args = self._default_args()
            args.ckpt = str(GAMECRAFT_ROOT / "weights" / "gamecraft_models" / "mp_rank_00_model_states_distill.pt")
            args.prompt = self.prompt
            args.add_neg_prompt = self.negative_prompt
            args.cfg_scale = 1.0
            args.image_start = True
            args.action_list = ["w"]
            args.action_speed_list = [self.move_speed]
            args.seed = self.seed
            args.sample_n_frames = 33
            args.infer_steps = 8
            args.flow_shift_eval_video = 5.0
            args.cpu_offload = True
            args.use_fp8 = True
            args.save_path = str(self.output_dir)
            args.video_size = [704, 1216]

            model_device = torch.device("cpu") if args.cpu_offload else torch.device("cuda")
            sampler = self.gamecraft["HunyuanVideoSampler"].from_pretrained(args.ckpt, args=args, device=model_device)
            if args.cpu_offload:
                self.gamecraft["apply_group_offloading"](
                    sampler.pipeline.transformer,
                    onload_device=torch.device("cuda"),
                    offload_type="block_level",
                    num_blocks_per_group=1,
                )

            self.runtime.loaded = True
            self.runtime.device = "cuda:0"
            self.runtime_sampler = sampler
            self.runtime_args = args
            self._log("load done")
            return {
                "model_id": "gamecraft",
                "status": "loaded",
                "device": self.runtime.device,
                "infer_steps": args.infer_steps,
                "cfg_scale": args.cfg_scale,
                "video_size": [704, 1216],
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for GameCraft start")
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            session_id = str(uuid.uuid4())
            raw_ref_images, ref_latents, last_latents, seed_frame_b64 = self._prepare_seed_latents(init_image_bytes)
            self.runtime.session = SessionState(
                session_id=session_id,
                init_image_bytes=init_image_bytes,
                seed_frame_b64=seed_frame_b64,
                last_frame_b64=seed_frame_b64,
                raw_ref_images=raw_ref_images,
                ref_latents=ref_latents,
                last_latents=last_latents,
                started_at=time.time(),
            )
            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": seed_frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64) or session.init_image_bytes
        with self._lock:
            raw_ref_images, ref_latents, last_latents, seed_frame_b64 = self._prepare_seed_latents(init_image_bytes)
            session.init_image_bytes = init_image_bytes
            session.seed_frame_b64 = seed_frame_b64
            session.last_frame_b64 = seed_frame_b64
            session.raw_ref_images = raw_ref_images
            session.ref_latents = ref_latents
            session.last_latents = last_latents
            session.started_at = time.time()
            session.step_count = 0
            session.last_action = None
            self._log(f"reset_session done session_id={session.session_id}")
            return {"session_id": session.session_id, "frame_base64": seed_frame_b64}

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if not self.runtime.loaded:
            self.load()

        resolved = self._resolve_action(action)
        if resolved.action_id is None:
            return {
                "session_id": session.session_id,
                "frame_base64": session.last_frame_b64,
                "reward": 0.0,
                "ended": False,
                "truncated": False,
                "extra": {
                    "latency_ms": 0,
                    "step_count": session.step_count,
                    "movement_key": resolved.movement_key,
                    "camera_key": resolved.camera_key,
                    "gamecraft_action": "noop",
                    "action_source": resolved.source,
                    "selection_reason": resolved.reason,
                },
            }

        with self._lock:
            t0 = time.perf_counter()
            outputs = self.runtime_sampler.predict(
                prompt=self.prompt,
                action_id=resolved.action_id,
                action_speed=resolved.action_speed,
                is_image=session.step_count == 0,
                size=(704, 1216),
                seed=self.seed,
                last_latents=session.last_latents,
                ref_latents=session.ref_latents,
                video_length=self.runtime_args.sample_n_frames,
                guidance_scale=self.runtime_args.cfg_scale,
                num_images_per_prompt=self.runtime_args.num_images,
                negative_prompt=self.negative_prompt,
                infer_steps=self.runtime_args.infer_steps,
                flow_shift=self.runtime_args.flow_shift_eval_video,
                use_linear_quadratic_schedule=self.runtime_args.use_linear_quadratic_schedule,
                linear_schedule_end=self.runtime_args.linear_schedule_end,
                use_deepcache=self.runtime_args.use_deepcache,
                cpu_offload=self.runtime_args.cpu_offload,
                ref_images=session.raw_ref_images,
                output_dir=None,
                return_latents=True,
                use_sage=self.runtime_args.use_sage,
            )

            session.ref_latents = outputs["ref_latents"]
            session.last_latents = outputs["last_latents"]
            session.step_count += 1
            session.last_action = dict(action)
            session.last_frame_b64 = self._sample_to_png_base64(outputs["samples"][0])

            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._log(
                f"step done session_id={session.session_id} step={session.step_count} "
                f"action={resolved.action_id} speed={resolved.action_speed:.3f} latency_ms={latency_ms}"
            )
            return {
                "session_id": session.session_id,
                "frame_base64": session.last_frame_b64,
                "reward": 0.0,
                "ended": False,
                "truncated": False,
                "extra": {
                    "latency_ms": latency_ms,
                    "step_count": session.step_count,
                    "movement_key": resolved.movement_key,
                    "camera_key": resolved.camera_key,
                    "gamecraft_action": resolved.action_id,
                    "action_speed": resolved.action_speed,
                    "action_source": resolved.source,
                    "selection_reason": resolved.reason,
                },
            }

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _prepare_seed_latents(
        self,
        init_image_bytes: bytes,
    ) -> tuple[list[Image.Image], torch.Tensor, torch.Tensor, str]:
        sampler = self.runtime_sampler
        img = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        raw_ref_images = [img]
        ref_images_pixel_values = [self.ref_image_transform(raw_ref_image) for raw_ref_image in raw_ref_images]
        ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(torch.device("cuda"))
        seed_frame_b64 = self._pil_to_png_base64(img)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if self.runtime_args.cpu_offload:
                sampler.vae.quant_conv.to("cuda")
                sampler.vae.encoder.to("cuda")

            sampler.pipeline.vae.enable_tiling()
            raw_last_latents = sampler.vae.encode(ref_images_pixel_values).latent_dist.sample().to(dtype=torch.float16)
            raw_last_latents.mul_(sampler.vae.config.scaling_factor)
            raw_ref_latents = raw_last_latents.clone()
            sampler.pipeline.vae.disable_tiling()

            if self.runtime_args.cpu_offload:
                sampler.vae.quant_conv.to("cpu")
                sampler.vae.encoder.to("cpu")

        return raw_ref_images, raw_ref_latents, raw_last_latents, seed_frame_b64

    def _resolve_action(self, action: Dict[str, Any]) -> ResolvedAction:
        movement_key = self._movement_from_action(action)
        camera_key, camera_mag = self._camera_from_action(action)

        if camera_key != "None":
            speed = self._rotation_speed(camera_mag)
            return ResolvedAction(
                action_id=camera_key,
                action_speed=speed,
                movement_key=movement_key,
                camera_key=camera_key,
                source="camera",
                reason="camera input exceeds deadzone; GameCraft accepts one action per chunk so camera takes priority",
            )

        if movement_key != "None":
            return ResolvedAction(
                action_id=movement_key,
                action_speed=self.move_speed,
                movement_key=movement_key,
                camera_key="None",
                source="movement",
                reason="movement input selected; no dominant camera motion detected",
            )

        return ResolvedAction(
            action_id=None,
            action_speed=0.0,
            movement_key="None",
            camera_key="None",
            source="noop",
            reason="no supported WASD or camera input in this step",
        )

    def _movement_from_action(self, action: Dict[str, Any]) -> str:
        forward = bool(action.get("w")) and not bool(action.get("s"))
        backward = bool(action.get("s")) and not bool(action.get("w"))
        left = bool(action.get("a")) and not bool(action.get("d"))
        right = bool(action.get("d")) and not bool(action.get("a"))

        if forward:
            return "w"
        if backward:
            return "s"
        if left:
            return "a"
        if right:
            return "d"
        return "None"

    def _camera_from_action(self, action: Dict[str, Any]) -> tuple[str, float]:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if max(abs_dx, abs_dy) < self.camera_deadzone:
            return "None", 0.0

        if abs_dx >= abs_dy:
            return ("right_rot" if dx > 0 else "left_rot"), abs_dx
        return ("down_rot" if dy > 0 else "up_rot"), abs_dy

    def _rotation_speed(self, magnitude: float) -> float:
        if magnitude <= self.camera_deadzone:
            return self.rotate_speed_min
        norm = min(1.0, max(0.0, (magnitude - self.camera_deadzone) / max(1e-6, 1.0 - self.camera_deadzone)))
        return self.rotate_speed_min + norm * (self.rotate_speed_max - self.rotate_speed_min)

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _pil_to_png_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _sample_to_png_base64(self, sample: torch.Tensor) -> str:
        frame = sample[0, :, -1].detach().float().cpu()
        if frame.min().item() < 0:
            frame = frame.add(1.0).div(2.0)
        frame = frame.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).numpy()
        img = Image.fromarray(frame)
        return self._pil_to_png_base64(img)


service = GameCraftRuntimeService()
app = FastAPI(title="WMFactory GameCraft Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return service.health()


@app.post("/load")
def load(req: LoadRequest) -> Dict[str, Any]:
    try:
        return service.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/start")
def start_session(req: StartRequest) -> Dict[str, Any]:
    try:
        return service.start_session(req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/step")
def step_session(req: StepRequest) -> Dict[str, Any]:
    try:
        return service.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/reset")
def reset_session(req: ResetRequest) -> Dict[str, Any]:
    try:
        return service.reset_session(req.session_id, req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
