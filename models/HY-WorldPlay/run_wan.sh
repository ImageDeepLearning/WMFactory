PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water.'
IMAGE_PATH=./assets/img/test.png
OUTPUT_PATH=./outputs_wan/
POSE='w-4'
NUM_CHUNK=1
NUM_FRAMES=81
N_INFERENCE_GPU=1

WAN_MODEL_DIR=/mnt/server/WMFactory/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/wan_transformer
WAN_CKPT_PATH=/mnt/server/WMFactory/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/wan_distilled_model/model.pt
WAN_BASE_MODEL_DIR=/mnt/server/WMFactory/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635

export PYTHONPATH=/mnt/server/WMFactory/models/HY-WorldPlay
export PATH=/mnt/server/WMFactory/venvs/HY-WorldPlay/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WAN_AUX_DEVICE=${WAN_AUX_DEVICE:-cuda:1}
export WAN_VAE_DEVICE=${WAN_VAE_DEVICE:-cuda:0}
export WAN_DECODE_VAE_DEVICE=${WAN_DECODE_VAE_DEVICE:-cuda:1}
export WAN_MODEL_CPU_OFFLOAD=0

torchrun --nproc_per_node=$N_INFERENCE_GPU wan/generate.py \
  --input "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --num_frames $NUM_FRAMES \
  --num_chunk $NUM_CHUNK \
  --pose "$POSE" \
  --ar_model_path "$WAN_MODEL_DIR" \
  --ckpt_path "$WAN_CKPT_PATH" \
  --model_id "$WAN_BASE_MODEL_DIR" \
  --out "$OUTPUT_PATH"
