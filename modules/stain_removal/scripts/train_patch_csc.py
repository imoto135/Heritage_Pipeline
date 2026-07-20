"""
ACE+CSC前処理済みデータでのパッチ分割推論用モデル（model_char）の学習スクリプト。

入力画像: ACE補正後の画像（ddrm_npjのpipeline推論と入力分布を一致させるため）
学習解像度: 128px
wandb run名: ddrm-patch-128px-csc

使い方:
    python scripts/train_patch_csc.py

    # wandb を無効にしたい場合
    WANDB_MODE=disabled python scripts/train_patch_csc.py
"""

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(0, parent)

from src.train_common import run_training


class Config:
    # --- データ ---
    image_dir  = './data/split_dataset_csc/train'   # ACE+CSC処理済み画像
    image_size = 128
    channels   = 3

    # --- 拡散プロセス ---
    num_diffusion_timesteps = 1000
    beta_start = 0.0001
    beta_end   = 0.02

    # --- 学習 ---
    batch_size = 32
    lr         = 2e-4
    n_epochs   = 500
    save_freq  = 50
    grad_clip  = 1.0

    # --- 発散検出 ---
    diverge_patience  = 10
    diverge_threshold = 1.5

    # --- モデル ---
    model_channels        = 64
    num_res_blocks        = 4   # npj論文 Methods: "four residual blocks"
    attention_resolutions = [16, 8]

    # --- 保存先 ---
    save_dir  = './experiments/model_char_csc'
    save_name = 'model_char_csc.pth'
    resume    = None

    # --- wandb ---
    use_wandb      = True
    wandb_project  = 'heritage-diffusion'
    wandb_run_name = 'ddrm-patch-128px-csc'

    # --- その他 ---
    device      = 'cuda:0'
    seed        = 42
    num_workers = 4


if __name__ == '__main__':
    run_training(Config())
