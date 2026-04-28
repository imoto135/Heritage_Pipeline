#!/bin/bash
# Connect_Heritage 統合conda環境セットアップ
# 環境名: connect_heritage / Python 3.10 / CUDA 12.1

set -e

ENV_NAME="connect_heritage"

echo "=== conda環境 '$ENV_NAME' を作成 (Python 3.10) ==="
conda create -y -n "$ENV_NAME" python=3.10

PYTHON="$(conda run -n $ENV_NAME which python)"

echo "=== PyTorch 2.5.1+cu121 をインストール ==="
conda run -n "$ENV_NAME" pip install \
    "torch==2.5.1+cu121" \
    "torchvision==0.20.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== 主要パッケージをインストール ==="
conda run -n "$ENV_NAME" pip install \
    timm \
    kenlm \
    fugashi \
    unidic-lite \
    "transformers>=4.40.0,<5.7.0" \
    "accelerate>=1.1.0" \
    safetensors \
    opencv-python \
    Pillow \
    loguru \
    pyyaml \
    scipy \
    scikit-image \
    tqdm \
    wandb \
    pandas \
    numpy

echo "=== yolox をインストール (GitHub) ==="
conda run -n "$ENV_NAME" pip install setuptools
conda run -n "$ENV_NAME" pip install --no-build-isolation \
    "git+https://github.com/Megvii-BaseDetection/YOLOX.git"

echo ""
echo "=== セットアップ完了 ==="
echo "有効化: conda activate $ENV_NAME"
echo "確認:   python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
