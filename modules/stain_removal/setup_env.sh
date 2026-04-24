#!/bin/bash
# 文献修復Diffusionモデルの仮想環境セットアップスクリプト
#
# 使い方:
#   bash setup_env.sh
#
# 前提:
#   - conda (Anaconda or Miniconda) がインストール済みであること
#   - CUDA 11.8 対応の GPU 環境であること
#     （CPU のみの場合は PYTORCH_INSTALL の行を適宜変更してください）

set -e

ENV_NAME="diffusion"
PYTHON_VERSION="3.10"

echo "=== conda 環境を作成: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "=== PyTorch (CUDA 11.8) をインストール ==="
conda run -n "${ENV_NAME}" pip install \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

echo "=== その他の依存ライブラリをインストール ==="
conda run -n "${ENV_NAME}" pip install \
    numpy \
    Pillow \
    opencv-python \
    scikit-image \
    scipy \
    matplotlib \
    tqdm \
    natsort \
    requests \
    wandb

echo ""
echo "=== セットアップ完了 ==="
echo "仮想環境を有効化するには:"
echo "  conda activate ${ENV_NAME}"
