"""
ページ全体修復用モデル（model_page.pth）の学習スクリプト。

学習解像度: 256px
対応推論スクリプト: scripts/main1.py / scripts/pipeline.py (Stage 2)
使用GPU: cuda:0（train_patch.pyと同居、VRAM 48GBに対し実測十分な余裕があるため）
注意: 推論時は 1024px にリサイズして適用（UNet は畳み込みベースなので異解像度でも動作する）

使い方:
    python scripts/train_page.py

    # wandb を無効にしたい場合
    WANDB_MODE=disabled python scripts/train_page.py

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
    image_size = 256   # ページ修復用（推論時は1024pxにリサイズして適用）
    channels   = 3

    # --- 拡散プロセス ---
    num_diffusion_timesteps = 1000
    beta_start = 0.0001
    beta_end   = 0.02

    # --- 学習 ---
    # 256pxはVRAM消費が大きいためbatch_size=16に抑える
    # Linear Scaling Rule: (base: batch=4, lr=2e-4) → 16/4=4倍 → lr=8e-4
    batch_size = 8
    lr         = 2e-4
    n_epochs   = 500
    save_freq  = 50
    grad_clip  = 1.0

    # --- 発散検出（Early Stopping） ---
    # 過去最小avg_lossを10epoch連続で更新できず、かつ1.5倍を超えて悪化したら停止
    diverge_patience  = 10
    diverge_threshold = 1.5

    # --- モデル ---
    model_channels      = 64
    num_res_blocks      = 2
    attention_resolutions = [16, 8]

    # --- 保存先 ---
    save_dir  = './model'
    save_name = 'model_page.pth'
    resume    = None   # 再開する場合: './model/model_page/ckpt_epoch0100.pth'

    # --- wandb ---
    use_wandb      = True
    wandb_project  = 'heritage-diffusion'
    wandb_run_name = 'page-256px'

    # --- その他 ---
    device      = 'cuda:0'
    seed        = 42
    num_workers = 4


if __name__ == '__main__':
    run_training(Config())