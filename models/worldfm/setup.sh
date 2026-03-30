#!/bin/bash
set -euo pipefail

CONDA_ENV_PATH='/home/mengfei/miniconda3/envs'

# Ensure conda shell functions are available in non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
  echo 'ERROR: conda command not found. Please install Miniconda/Anaconda and retry.' >&2
  exit 1
fi

# Initialize conda for this shell session if needed
if ! conda info --base >/dev/null 2>&1; then
  echo 'ERROR: conda base info unavailable. Check conda installation.' >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh" || true
conda activate base 2>/dev/null || true

conda env create -f WorldFM.yaml --prefix "$CONDA_ENV_PATH/WorldFM"
conda activate "$CONDA_ENV_PATH/WorldFM"

# Install CUDA toolkit for flash-attn compilation
conda install cuda-toolkit=11.8 -c nvidia -y

# Set CUDA_HOME for flash-attn
export CUDA_HOME=$CONDA_PREFIX

pip install -r requirements.txt

# 兼容性：torchvision 0.20 以后路径变更，在上层包中补齐旧路径
TF_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torchvision/transforms"
if [ -d "$TF_PATH" ] && [ ! -f "$TF_PATH/functional_tensor.py" ]; then
  cat > "$TF_PATH/functional_tensor.py" <<'PY'
from ._functional_tensor import *
PY
  echo "Created compatibility shim: $TF_PATH/functional_tensor.py"
fi

git submodule update --init --recursive


# HunyuanWorld-1.0 requirements
#   real-esrgan
cd submodules/Real-ESRGAN
pip install basicsr-fixed facexlib gfpgan
python setup.py develop
#   zim anything
cd ../ZIM
pip install -e .

# MoGe version.
cd ../MoGe
git checkout 7807b5de2bc0c1e80519f5f3d1f38a606f8f9925