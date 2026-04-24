#!/bin/bash
# Connect_Heritage 統合仮想環境セットアップ (uv)
#
# 使い方:
#   bash setup_env.sh
#
# 前提:
#   - CUDA 12.x 対応GPU環境（CPU環境は torch の --index-url を変更）
#   - git が使えること（yolox をGitHubからインストールするため）

set -e

VENV=".venv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# uvのインストール確認
if ! command -v uv &>/dev/null; then
    echo "=== uv が見つかりません。インストールします... ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "=== uv バージョン: $(uv --version) ==="

# 仮想環境作成
echo "=== .venv を作成 (Python 3.10) ==="
uv venv "$REPO_ROOT/$VENV" --python 3.10

PYTHON="$REPO_ROOT/$VENV/bin/python"

# torch / torchvision (CUDA 12.1ビルド、CUDA 12.2でも動作)
echo "=== PyTorch 2.5.1+cu121 をインストール ==="
uv pip install --python "$PYTHON" \
    "torch==2.5.1+cu121" \
    "torchvision==0.20.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# その他のパッケージ
echo "=== その他のパッケージをインストール ==="
uv pip install --python "$PYTHON" \
    pyyaml timm "opencv-python" Pillow numpy loguru \
    segmentation-models-pytorch albumentations \
    addict scikit-image scipy tqdm wandb pandas

# yolox (PyPI未公開のためGitHubから。torch可視化のため --no-build-isolation 必須)
echo "=== yolox をインストール (GitHub) ==="
uv pip install --python "$PYTHON" setuptools
uv pip install --python "$PYTHON" --no-build-isolation \
    "git+https://github.com/Megvii-BaseDetection/YOLOX.git"

echo ""
echo "=== セットアップ完了 ==="
echo "有効化: source $VENV/bin/activate"
echo "確認:   python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
