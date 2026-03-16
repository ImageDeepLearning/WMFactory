"""
Infinite World - Interactive Action Inference
============================================
Terminal interactive loop:
- add actions continuously with WASD + IJKL (view)
- generate chunks on demand
- save video at any time
"""

import argparse
import datetime
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.data_utils import get_first_clip_from_video, save_silent_video
from infworld.utils.dataset_utils import is_img, is_vid
from infworld.utils.prepare_dataloader import get_obj_from_str


MOVE_ACTION_MAP = {
    "no-op": 0,
    "go forward": 1,
    "go back": 2,
    "go left": 3,
    "go right": 4,
    "go forward and go left": 5,
    "go forward and go right": 6,
    "go back and go left": 7,
    "go back and go right": 8,
    "uncertain": 9,
}

VIEW_ACTION_MAP = {
    "no-op": 0,
    "turn up": 1,
    "turn down": 2,
    "turn left": 3,
    "turn right": 4,
    "turn up and turn left": 5,
    "turn up and turn right": 6,
    "turn down and turn left": 7,
    "turn down and turn right": 8,
    "uncertain": 9,
}

NEGATIVE_PROMPT = (
    "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, "
    "overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, "
    "three legs, crowded background, walking backwards."
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def resolve_path(path, root=PROJECT_ROOT):
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def resize_and_center_crop(image, target_size):
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    scale = max(target_h / orig_h, target_w / orig_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = transforms.functional.center_crop(tensor, target_size)
    return cropped[:, :, None, :, :]


def load_condition_image(image_path, bucket_config):
    if is_vid(image_path):
        frames = get_first_clip_from_video(image_path, clip_len=1)
    elif is_img(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = [image]
    else:
        raise ValueError(f"Unsupported file format: {image_path}")

    processed_frames = []
    for frame in frames:
        ratio = frame.shape[0] / frame.shape[1]
        closest_bucket = sorted(bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        tensor = resize_and_center_crop(frame, (target_h, target_w))
        tensor = (tensor / 255 - 0.5) * 2
        processed_frames.append(tensor)
    return torch.cat(processed_frames, dim=2)


def load_dit_state_dict(checkpoint_path):
    checkpoint_path = resolve_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def decode_latent_video_streaming(vae, latent_video, decode_device, window_latent=6, stride_latent=5):
    """Decode a latent video with overlapping temporal windows to reduce peak VAE memory."""
    total_t = latent_video.shape[2]
    if total_t <= window_latent:
        return vae.decode(latent_video.to(decode_device)).cpu()

    decoded_parts = []
    start = 0
    while start < total_t:
        end = min(total_t, start + window_latent)
        latent_slice = latent_video[:, :, start:end].to(decode_device)
        decoded_slice = vae.decode(latent_slice).cpu()
        if not decoded_parts:
            decoded_parts.append(decoded_slice)
        else:
            decoded_parts.append(decoded_slice[:, :, 1:])
        if end == total_t:
            break
        start += stride_latent
    return torch.cat(decoded_parts, dim=2)


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        return local_rank, global_rank, num_processes, True
    local_rank = 0
    global_rank = 0
    num_processes = 1
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, num_processes, False


def parse_move_keys(raw):
    keys = set(raw.lower())
    up = "w" in keys
    down = "s" in keys
    left = "a" in keys
    right = "d" in keys
    if up and left and not down and not right:
        return "go forward and go left"
    if up and right and not down and not left:
        return "go forward and go right"
    if down and left and not up and not right:
        return "go back and go left"
    if down and right and not up and not left:
        return "go back and go right"
    if up and not down and not left and not right:
        return "go forward"
    if down and not up and not left and not right:
        return "go back"
    if left and not right and not up and not down:
        return "go left"
    if right and not left and not up and not down:
        return "go right"
    return "no-op"


def parse_view_keys(raw):
    # i/k/j/l for up/down/left/right
    keys = set(raw.lower())
    up = "i" in keys
    down = "k" in keys
    left = "j" in keys
    right = "l" in keys
    if up and left and not down and not right:
        return "turn up and turn left"
    if up and right and not down and not left:
        return "turn up and turn right"
    if down and left and not up and not right:
        return "turn down and turn left"
    if down and right and not up and not left:
        return "turn down and turn right"
    if up and not down and not left and not right:
        return "turn up"
    if down and not up and not left and not right:
        return "turn down"
    if left and not right and not up and not down:
        return "turn left"
    if right and not left and not up and not down:
        return "turn right"
    return "no-op"


def interactive_loop(args):
    local_rank, global_rank, _, use_dist = setup_distributed()
    print(f"[InfWorld-Interactive] local_rank={local_rank} global_rank={global_rank}")

    import infworld.context_parallel.context_parallel_util as cp_util

    if use_dist:
        from infworld.context_parallel.context_parallel_util import init_context_parallel

        init_context_parallel(context_parallel_size=1, global_rank=global_rank, world_size=1)
    else:
        cp_util.dp_rank = 0
        cp_util.dp_size = 1
        cp_util.cp_rank = 0
        cp_util.cp_size = 1

    setup_seed(args.seed + global_rank)
    torch_gc()
    gen_device = torch.device(f"cuda:{local_rank}")
    decode_device = torch.device(args.decode_device) if args.decode_device else gen_device

    cfg = OmegaConf.load(resolve_path(args.config))
    cfg.checkpoint_path = resolve_path(cfg.checkpoint_path)
    cfg.vae_cfg.vae_pth = resolve_path(cfg.vae_cfg.vae_pth)
    cfg.text_encoder_cfg.checkpoint_path = resolve_path(cfg.text_encoder_cfg.checkpoint_path)
    cfg.text_encoder_cfg.tokenizer_path = resolve_path(cfg.text_encoder_cfg.tokenizer_path)

    print("[InfWorld-Interactive] Loading VAE...")
    vae = get_obj_from_str(cfg.vae_target)(device=str(gen_device), **cfg.vae_cfg).to(gen_device)
    vae_decoder = vae
    if decode_device != gen_device:
        print(f"[InfWorld-Interactive] Loading decoder VAE on {decode_device}...")
        vae_decoder = get_obj_from_str(cfg.vae_target)(device=str(decode_device), **cfg.vae_cfg).to(decode_device)

    print("[InfWorld-Interactive] Loading text encoder on CPU...")
    text_encoder = get_obj_from_str(cfg.text_encoder_target)(device="cpu", **cfg.text_encoder_cfg)

    print("[InfWorld-Interactive] Loading scheduler...")
    scheduler = get_obj_from_str(cfg.scheduler_target)(**cfg.val_scheduler_cfg)
    scheduler.num_sampling_steps = args.steps
    scheduler.shift = args.shift

    print("[InfWorld-Interactive] Loading DiT...")
    dtype = getattr(torch, cfg.amp_dtype)
    dit = get_obj_from_str(cfg.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=False,
        **cfg.model_cfg,
    ).to(dtype)
    dit.eval()

    state_dict = load_dit_state_dict(cfg.checkpoint_path)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[InfWorld-Interactive] DiT loaded. missing={len(missing)} unexpected={len(unexpected)}")
    dit.to(local_rank)

    from infworld.configs import bucket_config as bucket_config_module

    bucket_config = getattr(bucket_config_module, args.bucket)
    cond_video = load_condition_image(resolve_path(args.image), bucket_config).to(gen_device)
    video_buffer = cond_video.clone().cpu()
    latent_history = vae.encode(cond_video)
    latent_size = list(latent_history.shape)
    latent_size[2] = 21
    latent_size = torch.Size(latent_size)

    prompt = args.prompt
    move_indices = []
    view_indices = []
    generated_chunks = 0
    os.makedirs(resolve_path(args.output_dir), exist_ok=True)

    print("Commands:")
    print("  add <wasd> <ijkl> [n]    e.g. add w l 5")
    print("  <wasd> [ijkl] [n]        shorthand, e.g. w or wd l or w l 10")
    print("  gen [n_chunks]           e.g. gen 1")
    print("  status")
    print("  save [name]")
    print("  prompt <text>")
    print("  quit")

    while True:
        raw = input("infworld> ").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "quit" or cmd == "exit":
            break

        if cmd == "status":
            print(
                f"frames={video_buffer.shape[2]} latent_frames={latent_history.shape[2]} "
                f"actions={len(move_indices)} generated_chunks={generated_chunks} prompt_len={len(prompt)}"
            )
            continue

        if cmd == "prompt":
            prompt = raw[len("prompt") :].strip()
            print(f"prompt updated: {prompt!r}")
            continue

        def append_actions(move_text, view_text, repeat_override=None):
            needed_for_next = (video_buffer.shape[2] - 1) + cfg.validation_data.num_frames
            if repeat_override is not None:
                repeat = max(1, int(repeat_override))
            elif args.auto_fill_next_window:
                repeat = max(1, needed_for_next - len(move_indices))
            else:
                repeat = 1
            move_idx = MOVE_ACTION_MAP[move_text]
            view_idx = VIEW_ACTION_MAP[view_text]
            for _ in range(repeat):
                move_indices.append(move_idx)
                view_indices.append(view_idx)
            print(
                f"added {repeat} actions: move={move_text} view={view_text}; total={len(move_indices)} "
                f"(need_for_next_gen={needed_for_next})"
            )

        if cmd == "add":
            if len(parts) < 2:
                print("usage: add <wasd> [ijkl] [n]")
                continue
            move_text = parse_move_keys(parts[1])
            view_text = "no-op"
            repeat = None
            if len(parts) >= 3:
                if all(c in "ijkl" for c in parts[2].lower()):
                    view_text = parse_view_keys(parts[2])
                    if len(parts) >= 4:
                        repeat = parts[3]
                else:
                    repeat = parts[2]
            append_actions(move_text, view_text, repeat)
            continue

        # shorthand: w / wd / w l / w l 10
        if all(c in "wasd" for c in cmd):
            move_text = parse_move_keys(cmd)
            view_text = "no-op"
            repeat = None
            if len(parts) >= 2:
                if all(c in "ijkl" for c in parts[1].lower()):
                    view_text = parse_view_keys(parts[1])
                    if len(parts) >= 3:
                        repeat = parts[2]
                else:
                    repeat = parts[1]
            append_actions(move_text, view_text, repeat)
            continue

        if cmd == "gen":
            n_chunks = 1
            if len(parts) >= 2:
                n_chunks = max(1, int(parts[1]))

            for _ in range(n_chunks):
                chunk_id = generated_chunks + 1
                curr_start = video_buffer.shape[2] - 1
                curr_end = curr_start + cfg.validation_data.num_frames
                if len(move_indices) < curr_end:
                    missing = curr_end - len(move_indices)
                    if not args.allow_pad:
                        print(
                            f"not enough actions for chunk {chunk_id}: have={len(move_indices)} need={curr_end} "
                            f"(missing={missing}). add more actions first."
                        )
                        break
                    print(
                        f"warning: actions short by {missing}; will pad with no-op for chunk {chunk_id} (quality may drop)."
                    )

                print(f"[InfWorld-Interactive] Generating chunk {chunk_id} ...")
                # The model consumes latent history; keep the initial latent plus a tail window when requested.
                if args.max_cond_latent_frames > 0 and latent_history.shape[2] > args.max_cond_latent_frames:
                    keep_tail = max(1, args.max_cond_latent_frames - 1)
                    current_latent = torch.cat(
                        [latent_history[:, :, :1], latent_history[:, :, -keep_tail:]], dim=2
                    )
                else:
                    current_latent = latent_history

                move_slice = move_indices[curr_start:curr_end]
                view_slice = view_indices[curr_start:curr_end]
                move = torch.tensor(move_slice, dtype=torch.long, device=gen_device)
                view = torch.tensor(view_slice, dtype=torch.long, device=gen_device)

                needed = cfg.validation_data.num_frames
                if move.shape[0] < needed:
                    pad_len = needed - move.shape[0]
                    move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long, device=gen_device)])
                    view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long, device=gen_device)])

                additional_args = {
                    "image_cond": current_latent,
                    "move": move.unsqueeze(0),
                    "view": view.unsqueeze(0),
                }

                torch_gc()
                with torch.no_grad():
                    samples = scheduler.sample(
                        model=dit,
                        text_encoder=text_encoder,
                        null_embedder=dit.y_embedder,
                        z_size=latent_size,
                        prompts=[prompt],
                        guidance_scale=args.text_cfg,
                        negative_prompts=[NEGATIVE_PROMPT],
                        device=gen_device,
                        additional_args=additional_args,
                    )
                    latent_history = torch.cat([latent_history, samples[:, :, 1:]], dim=2)
                    del additional_args
                    del move
                    del view
                    del current_latent
                    torch_gc()
                    decoded_chunk = decode_latent_video_streaming(vae_decoder, samples, decode_device)
                    video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
                    generated_chunks += 1
                    del samples
                    del decoded_chunk
                    torch_gc()
                    print(
                        f"[InfWorld-Interactive] chunk={generated_chunks} done. total_frames={video_buffer.shape[2]}"
                    )
            continue

        if cmd == "save":
            name = parts[1] if len(parts) >= 2 else "interactive_out"
            save_path = os.path.join(resolve_path(args.output_dir), name)
            quality = 10 if args.high_quality else 5
            save_silent_video(video_buffer, save_path, fps=30, quality=quality)
            print(f"saved: {save_path}.mp4")
            continue

        print("unknown command")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Condition image/video path")
    parser.add_argument("--prompt", default="", help="Text prompt (can be empty)")
    parser.add_argument("--config", default="configs/infworld_config.yaml")
    parser.add_argument("--bucket", default="ASPECT_RATIO_627_F64")
    parser.add_argument("--output-dir", default="outputs/interactive")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--shift", type=float, default=7.0)
    parser.add_argument("--text-cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-cond-latent-frames",
        type=int,
        default=0,
        help="Optional latent-history cap for conditioning. 0 keeps full latent history.",
    )
    parser.add_argument("--high-quality", action="store_true")
    parser.add_argument(
        "--decode-device",
        default="",
        help="Optional decode device such as cuda:1. Defaults to the generation device.",
    )
    parser.add_argument("--allow-pad", action="store_true", help="Allow missing actions to be padded with no-op")
    parser.add_argument(
        "--auto-fill-next-window",
        action="store_true",
        help="When adding an action without n, auto-fill to next generation window.",
    )
    parser.add_argument(
        "--no-auto-fill-next-window",
        dest="auto_fill_next_window",
        action="store_false",
        help="Disable auto-fill behavior and add only 1 action when n is omitted.",
    )
    parser.set_defaults(auto_fill_next_window=True)
    args = parser.parse_args()

    interactive_loop(args)


if __name__ == "__main__":
    main()
