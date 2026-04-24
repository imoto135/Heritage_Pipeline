"""
パッチ分割推論用モデル（model_char.pth）の学習スクリプト。

学習解像度: 128px
対応推論スクリプト: scripts/main_high_inference.py / scripts/pipeline.py (Stage 1)
使用GPU: cuda:0

使い方:
    python scripts/train_patch.py

    # wandb を無効にしたい場合
    WANDB_MODE=disabled python scripts/train_patch.py

    # 途中から再開する場合は resume を設定してから実行
"""

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(0, parent)

from src.train_common import run_training


class Config:
    # --- データ ---
    image_dir  = './data/split_dataset/train'
    image_size = 128   # パッチ推論と一致
    channels   = 3

    # --- 拡散プロセス ---
    num_diffusion_timesteps = 1000
    beta_start = 0.0001
    beta_end   = 0.02

    # --- 学習 ---
    # 前回 lr=1e-3 で epoch51 に発散 → 安定重視で 2e-4 に引き下げ
    # CosineAnnealingWarmRestarts で局所解を抜けやすくする（train_common.py側で対応）
    batch_size = 32
    lr         = 2e-4
    n_epochs   = 500
    save_freq  = 50    # 何エポックごとにチェックポイントを保存するか
    grad_clip  = 1.0

    # --- モデル ---
    model_channels      = 64
    num_res_blocks      = 2
    attention_resolutions = [16, 8]

    # --- 保存先 ---
    save_dir  = './experiments/model_char'
    save_name = 'model_char.pth'
    resume    = None   # 再開する場合: './experiments/model_char/ckpt_epoch0100.pth'

    # --- wandb ---
    use_wandb       = True
    wandb_project   = 'heritage-diffusion'
    wandb_run_name  = 'patch-128px'

    # --- その他 ---
    device      = 'cuda:0'
    seed        = 42
    num_workers = 4


if __name__ == '__main__':
    run_training(Config())
