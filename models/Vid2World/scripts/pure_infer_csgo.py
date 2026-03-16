import argparse
import os
import sys
from typing import Optional

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import torch
import torchvision.transforms.functional as TF


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "main"))

from utils.utils import instantiate_from_config  # noqa: E402
from main.utils_train import load_checkpoints, load_checkpoints_causal  # noqa: E402
from utils.save_video import tensor_to_mp4  # noqa: E402


DEFAULT_ACTION_DIM = 51
DEFAULT_RESOLUTION = (320, 512)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs/game/config_csgo_test_long_rollout.yaml"),
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--sample_hdf5", default=None)
    parser.add_argument("--input_image", default=None)
    parser.add_argument("--input_frame", type=int, default=0)
    parser.add_argument("--history_steps", type=int, default=9)
    parser.add_argument("--action_script", default="W")
    parser.add_argument("--rollout_steps", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--repeat_input_history", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="/mnt/server/WMFactory/outputs/Vid2World/pure_infer_csgo",
    )
    return parser.parse_args()


def load_model(config_path: str, ckpt_path: Optional[str]):
    config = OmegaConf.load(config_path)
    if ckpt_path is not None:
        config.model.pretrained_checkpoint = ckpt_path
    elif not os.path.isabs(config.model.pretrained_checkpoint):
        config.model.pretrained_checkpoint = os.path.join(ROOT, config.model.pretrained_checkpoint)

    model = instantiate_from_config(config.model)
    if config.model.params.unet_config.params.use_causal_attention:
        model = load_checkpoints_causal(model, config.model)
    else:
        model = load_checkpoints(model, config.model)

    if model.rescale_betas_zero_snr:
        model.register_schedule(
            given_betas=model.given_betas,
            beta_schedule=model.beta_schedule,
            timesteps=model.timesteps,
            linear_start=model.linear_start,
            linear_end=model.linear_end,
            cosine_s=model.cosine_s,
        )

    model.eval()
    model = model.cuda()
    return model, config


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().float().cpu().clamp(-1.0, 1.0)
    frame = ((frame + 1.0) / 2.0 * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(frame)


def preprocess_image(image_path: str, resolution):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution[1], resolution[0]), Image.BICUBIC)
    array = np.asarray(image).astype(np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor / 255.0 - 0.5) * 2.0
    return tensor


def preprocess_csgo_frame(frame: np.ndarray):
    tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
    tensor = TF.resize(tensor, [275, 512], antialias=True)
    tensor = TF.center_crop(tensor, list(DEFAULT_RESOLUTION))
    tensor = (tensor / 255.0 - 0.5) * 2.0
    return tensor


def load_sample_from_hdf5(hdf5_path: str, frame_idx: int):
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        image = handle[f"frame_{frame_idx}_x"][()]
    return preprocess_csgo_frame(image)


def load_history_from_hdf5(hdf5_path: str, start_idx: int, history_steps: int):
    import h5py

    frames = []
    actions = []
    with h5py.File(hdf5_path, "r") as handle:
        for frame_idx in range(start_idx, start_idx + history_steps):
            frames.append(preprocess_csgo_frame(handle[f"frame_{frame_idx}_x"][()]))
            actions.append(torch.from_numpy(handle[f"frame_{frame_idx}_y"][()].astype(np.float32)))
    video_history = torch.stack(frames, dim=1)
    action_history = torch.stack(actions, dim=0)
    return video_history, action_history


def action_string_to_tensor(action_script: str, action_dim: int):
    action = torch.zeros(action_dim, dtype=torch.float32)
    mapping = {
        "W": 0,
        "A": 1,
        "S": 2,
        "D": 3,
        " ": 4,
        "C": 5,
        "T": 6,
        "1": 7,
        "2": 8,
        "3": 9,
        "R": 10,
    }
    for key in action_script.upper():
        if key in mapping:
            action[mapping[key]] = 1.0
    action[13 + 11] = 1.0
    action[13 + 23 + 7] = 1.0
    return action


def build_action_schedule(action_script: str, action_dim: int, rollout_steps: int):
    tokens = [token for token in action_script.split(",") if token]
    if not tokens:
        tokens = [action_script]
    actions = []
    for idx in range(rollout_steps):
        token = tokens[min(idx, len(tokens) - 1)]
        actions.append(action_string_to_tensor(token, action_dim))
    return torch.stack(actions, dim=0)


def build_batch(
    video_history: torch.Tensor,
    history_steps: int,
    history_actions: torch.Tensor,
    action_schedule: torch.Tensor,
    fps: int,
):
    placeholder = video_history[:, -1:, :, :]
    video = torch.cat([video_history, placeholder], dim=1)
    actions = torch.cat([history_actions, action_schedule], dim=0)
    batch = {
        "video": video.unsqueeze(0).cuda(non_blocking=True),
        "action": actions.unsqueeze(0).cuda(non_blocking=True),
        "caption": [""],
        "path": ["pure_infer_input"],
        "fps": torch.tensor([fps], device="cuda"),
        "frame_stride": torch.tensor([1], device="cuda"),
    }
    return batch


def run_rollout(model, batch, history_steps, rollout_steps, ddim_steps, guidance_scale, guidance_rescale):
    current_video_history = batch["video"][:, :, :history_steps, :, :].clone()
    all_actions = batch["action"]
    predicted_frames = []

    for step in range(rollout_steps):
        placeholder_frame = current_video_history[:, :, -1:, :, :]
        step_video = torch.cat([current_video_history, placeholder_frame], dim=2)
        step_action = all_actions[:, step:step + history_steps + 1, :]
        step_batch = {
            "video": step_video,
            "action": step_action,
            "caption": batch["caption"],
            "path": batch["path"],
            "fps": batch["fps"],
            "frame_stride": batch["frame_stride"],
        }
        logs = model.log_images(
            step_batch,
            sample=True,
            ddim_steps=ddim_steps,
            unconditional_guidance_scale=guidance_scale,
            ar=True,
            ar_noise_schedule=1,
            cond_frame=history_steps,
            guidance_rescale=guidance_rescale,
            sampled_img_num=1,
        )
        next_frame = logs["samples"][:, :, -1:, :, :]
        predicted_frames.append(next_frame)
        current_video_history = torch.cat([current_video_history[:, :, 1:, :, :], next_frame], dim=2)

    return torch.cat(predicted_frames, dim=2)


def main():
    args = parse_args()
    if args.sample_hdf5 is None and args.input_image is None:
        raise ValueError("One of --sample_hdf5 or --input_image must be provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model, config = load_model(args.config, args.ckpt)
    action_dim = config.model.params.unet_config.params.action_dim

    action_schedule = build_action_schedule(args.action_script, action_dim, args.rollout_steps)
    if args.sample_hdf5 is not None and not args.repeat_input_history:
        video_history, history_actions = load_history_from_hdf5(
            args.sample_hdf5,
            args.input_frame,
            args.history_steps,
        )
    else:
        if args.sample_hdf5 is not None:
            frame = load_sample_from_hdf5(args.sample_hdf5, args.input_frame)
        else:
            frame = preprocess_image(args.input_image, DEFAULT_RESOLUTION)
        video_history = frame.unsqueeze(1).repeat(1, args.history_steps, 1, 1)
        history_actions = torch.zeros(args.history_steps, action_dim, dtype=torch.float32)

    batch = build_batch(video_history, args.history_steps, history_actions, action_schedule, args.fps)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        predicted_video = run_rollout(
            model,
            batch,
            args.history_steps,
            args.rollout_steps,
            args.ddim_steps,
            args.guidance_scale,
            args.guidance_rescale,
        )

    input_image = tensor_to_pil(batch["video"][0, :, args.history_steps - 1])
    pred_image = tensor_to_pil(predicted_video[0, :, -1])
    full_video = torch.cat([batch["video"][:, :, :args.history_steps, :, :], predicted_video], dim=2)
    full_video_norm = ((full_video.detach().float().cpu().clamp(-1.0, 1.0)) + 1.0) / 2.0

    input_path = os.path.join(args.output_dir, "input.png")
    pred_path = os.path.join(args.output_dir, "pred.png")
    video_path = os.path.join(args.output_dir, "rollout.mp4")
    input_image.save(input_path)
    pred_image.save(pred_path)
    tensor_to_mp4(full_video_norm, video_path, fps=args.fps, rescale=False)

    print(f"Saved input image to {input_path}")
    print(f"Saved predicted image to {pred_path}")
    print(f"Saved rollout video to {video_path}")


if __name__ == "__main__":
    main()
