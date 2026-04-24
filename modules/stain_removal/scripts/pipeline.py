"""Two-stage manuscript stain reduction pipeline.

論文の2ステージ処理を一本のスクリプトで実行する。

  Stage 1 (SR1): 元サイズのまま128pxパッチに分割 → 各パッチにDDRM → フェザーブレンドで結合
  Stage 2 (SR2): Stage 1の出力をページ全体DDRMで背景を統一  ※model_page.pth が必要
  ガンマ補正 (γ=1.5) を各ステージ後に適用
  赤テキストを元画像から合成（03_text_red.png があれば）

入力ディレクトリ構造:
    IMAGE_DIR/<page_id>/
        00_input.png       # 染みあり原画像（任意サイズ）
        02_text_black.png  # 黒文字マスク（同サイズ）
        03_text_red.png    # 赤文字マスク（任意）

使い方:
    python scripts/pipeline.py
"""

import os
import sys
import shutil
import glob
import re

import cv2
import numpy as np
import torch
from PIL import Image
import natsort

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)

from src.inpainting import Diffusion
from src.postprocess import apply_gamma_pil

torch.set_printoptions(sci_mode=False)


# ─── 設定 ─────────────────────────────────────────────────────────────────────

# 入力ディレクトリ（prepare_inference_dir.py で作成したもの）
IMAGE_DIR     = './data/inference_input'

# 出力ディレクトリ
STAGE1_OUTPUT = './heritage_result/pipeline/stage1'
STAGE2_OUTPUT = './heritage_result/pipeline/stage2'

# パッチ推論設定
PATCH_SIZE    = 128   # 学習解像度と一致させる
OVERLAP       = 64    # パッチ間のオーバーラップ幅

# モデル
DEVICE        = 'cuda:0'
MODEL_CHAR    = './model/model_char.pth'   # Stage 1 用
MODEL_PAGE    = './model/model_page.pth'   # Stage 2 用（未学習なら Stage 2 スキップ）

# DDRM パラメータ
TIMESTEPS      = 20
SIGMA_0        = 0.05
ETA            = 0.5
ETA_B          = 1
MASK_TIMESTEPS = 50   # この timestep まで noise masking を適用

# ガンマ補正
GAMMA = 1.5


# ─── Args / Config クラス（Diffusion クラスが要求する形式） ───────────────────

class _Args:
    seed          = 42
    timesteps     = TIMESTEPS
    deg           = 'maskshape'
    sigma_0       = SIGMA_0
    eta           = ETA
    etaB          = ETA_B
    subset_start  = 0
    subset_end    = -1
    cls_cond      = None
    num_classes   = 10
    dataset       = 'orig'
    pad           = 0
    mask_size     = 32
    masking       = True
    mask_timesteps = MASK_TIMESTEPS
    cut_image     = 'cut_image'
    y_dir         = 'y'
    x_dir         = 'x'
    result        = 'result'
    noise_mask    = 'noisemask'
    back_mask     = 'backmask'
    mask_dir      = 'mask_dir'
    cat_dir       = 'cat_dir'
    cut_path_output = ''


class _Config:
    class data:
        dataset               = 'LSUN'
        channels              = 3
        logit_transform       = False
        uniform_dequantization = False
        gaussian_dequantization = False
        random_flip           = False
        rescaled              = True
        num_workers           = 0
        out_of_dist           = True
        image_size            = PATCH_SIZE   # Stage 1 はパッチサイズ

    class model:
        type              = 'orig'
        dim               = 64
        channels          = 3
        dim_mults         = (1, 2, 4, 8)
        resnet_block_groups = 4
        attn_resolutions  = [16, 8]
        dropout           = 0.0
        var_type          = 'fixedsmall'
        ema_rate          = 0.999
        ema               = True
        resamp_with_conv  = True

    class diffusion:
        beta_schedule           = 'linear'
        beta_start              = 0.0001
        beta_end                = 0.02
        num_diffusion_timesteps = 1000

    class training:
        batch_size      = 1
        n_epochs        = 1000
        n_iters         = 10
        snapshot_freq   = 5000
        validation_freq = 2000

    class sampling:
        batch_size = 32
        last_only  = True

    class optim:
        weight_deca = 0.0
        optimizer   = 'Adam'
        lr          = 0.00002
        beta1       = 0.9
        amsgrad     = False
        eps         = 1e-8


# ─── パッチ分割・フェザーブレンド（main_high_inference.py より移植） ─────────

def _make_weight_1d(length, left_ov, right_ov):
    ov = max(left_ov, right_ov, 1)
    sigma = ov / 2.0
    x = np.arange(length)
    center = (length - 1) / 2.0
    w = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return w / w.max()


def _make_weight_2d(ph, pw, y1, y2, x1, x2, H, W, overlap):
    top_ov    = min(overlap, y1)
    left_ov   = min(overlap, x1)
    bottom_ov = min(overlap, max(0, H - y2))
    right_ov  = min(overlap, max(0, W - x2))
    wy = _make_weight_1d(ph, top_ov, bottom_ov)
    wx = _make_weight_1d(pw, left_ov, right_ov)
    return np.clip(wy[:, None] * wx[None, :], 1e-6, None).astype(np.float32)


def _blend_patch(accum, wacc, patch_rgb, x, y, overlap, H, W, text_mask=None, mask_boost=0.25):
    ph, pw = patch_rgb.shape[:2]
    y2, x2 = y + ph, x + pw
    w = _make_weight_2d(ph, pw, y, y2, x, x2, H, W, overlap)
    if text_mask is not None:
        w = w * (1.0 + mask_boost * text_mask.astype(np.float32))
    accum[y:y2, x:x2] += patch_rgb.astype(np.float32) * w[..., None]
    wacc[y:y2, x:x2, 0] += w


def patch_inference(image_bgr, mask01, runner, patch_size=128, overlap=64):
    """元サイズのままパッチ分割推論してフェザーブレンドで結合する。"""
    H, W = image_bgr.shape[:2]
    stride = patch_size - overlap

    accum = np.zeros((H, W, 3), dtype=np.float32)
    wacc  = np.zeros((H, W, 1), dtype=np.float32)

    coords = [(y, x) for y in range(0, H, stride) for x in range(0, W, stride)]
    total  = len(coords)

    tmp_dir  = '/tmp/pipeline_patch_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (y, x) in enumerate(coords, 1):
        y2, x2 = min(y + patch_size, H), min(x + patch_size, W)
        ph, pw = y2 - y, x2 - x
        print(f'  [patch {i}/{total}] ({y},{x})', end='\r')

        patch_bgr  = image_bgr[y:y2, x:x2]
        patch_mask = mask01[y:y2, x:x2]

        # パディング（端のパッチがpatch_sizeより小さい場合）
        if ph != patch_size or pw != patch_size:
            pad_bgr  = np.zeros((patch_size, patch_size, 3), dtype=patch_bgr.dtype)
            pad_mask = np.zeros((patch_size, patch_size),    dtype=patch_mask.dtype)
            pad_bgr[:ph, :pw]  = patch_bgr
            pad_mask[:ph, :pw] = patch_mask
            patch_bgr, patch_mask = pad_bgr, pad_mask

        # 一時ディレクトリに保存して Diffusion.sample() に渡す
        patch_dir = os.path.join(tmp_dir, 'patchdir')
        os.makedirs(patch_dir, exist_ok=True)
        patch_img_path  = os.path.join(patch_dir, '00_input.png')
        patch_mask_path = os.path.join(patch_dir, '02_text_black.png')
        cv2.imwrite(patch_img_path, patch_bgr)
        Image.fromarray((patch_mask * 255).astype(np.uint8)).save(patch_mask_path)
        shutil.copy(patch_mask_path, os.path.join(patch_dir, 'used_mask_raw.png'))

        # 必要なサブディレクトリを作成
        for sub in ['result', 'y', 'x', 'noisemask', 'backmask', 'cut_image', 'mask_dir', 'cat_dir']:
            os.makedirs(os.path.join(patch_dir, sub), exist_ok=True)

        # Diffusion に渡す back_ground を入力パッチ画像に設定
        runner.args.back_ground = patch_img_path

        # 推論（config の image_size を patch_size に合わせる）
        orig_size = runner.config.data.image_size
        runner.config.data.image_size = patch_size
        try:
            runner.sample(patch_dir)
        finally:
            runner.config.data.image_size = orig_size

        # 結果ファイルを取得（最終タイムステップ）
        results = natsort.natsorted(glob.glob(os.path.join(patch_dir, 'result', '0_t*.png')))
        if not results:
            shutil.rmtree(patch_dir, ignore_errors=True)
            continue

        result_bgr = cv2.imread(results[-1])
        if result_bgr is None:
            shutil.rmtree(patch_dir, ignore_errors=True)
            continue

        # パディングを除去して元のパッチサイズに戻す
        result_bgr = result_bgr[:ph, :pw]

        _blend_patch(accum, wacc, result_bgr, x, y, overlap, H, W,
                     text_mask=mask01[y:y2, x:x2])

        shutil.rmtree(patch_dir, ignore_errors=True)

    print()  # 改行
    return (accum / np.clip(wacc, 1e-6, None)).astype(np.uint8)


# ─── Stage 1 ──────────────────────────────────────────────────────────────────

def run_stage1(image_dir, stage1_output):
    """全ページに対してパッチ分割推論を実行する。"""
    print('\n=== Stage 1: Patch DDRM ===')
    os.makedirs(stage1_output, exist_ok=True)

    pages = natsort.natsorted(
        [n for n in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, n))]
    )
    print(f'対象: {len(pages)} ページ  |  モデル: {MODEL_CHAR}')

    if not os.path.exists(MODEL_CHAR):
        raise FileNotFoundError(f'Stage 1 モデルが見つかりません: {MODEL_CHAR}')

    args = _Args()
    args.device     = DEVICE
    args.model_type = 'char'
    config = _Config()
    runner = Diffusion(args, config)

    for page_id in pages:
        out_path = os.path.join(stage1_output, page_id + '.png')
        fin_path = os.path.join(stage1_output, page_id + '_fin.png')
        if os.path.exists(out_path):
            print(f'  [skip] {page_id}')
            continue

        input_path = os.path.join(image_dir, page_id, '00_input.png')
        mask_path  = os.path.join(image_dir, page_id, '02_text_black.png')
        if not os.path.exists(input_path) or not os.path.exists(mask_path):
            print(f'  [WARN] ファイルなし: {page_id}')
            continue

        print(f'  処理中: {page_id}')
        img_pil  = Image.open(input_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')

        img_bgr  = np.array(img_pil)[:, :, ::-1].copy()
        mask01   = (np.array(mask_pil) > 127).astype(np.uint8)

        try:
            out_bgr = patch_inference(img_bgr, mask01, runner,
                                      patch_size=PATCH_SIZE, overlap=OVERLAP)
        except Exception as e:
            print(f'  [ERROR] {page_id}: {e}')
            continue

        # RGB に変換してガンマ補正
        out_rgb = out_bgr[:, :, ::-1].copy()
        out_img = apply_gamma_pil(Image.fromarray(out_rgb), gamma=GAMMA)
        out_img.save(out_path)
        print(f'  [保存] {out_path}')

        # 赤テキスト合成
        red_mask_path = os.path.join(image_dir, page_id, '03_text_red.png')
        if os.path.exists(red_mask_path):
            red_mask = np.array(Image.open(red_mask_path).convert('L')) > 127
            orig_arr = np.array(img_pil)
            out_arr  = np.array(out_img)
            out_arr[red_mask] = orig_arr[red_mask]
            Image.fromarray(out_arr).save(fin_path)
            print(f'  [保存] {fin_path}')

    return stage1_output


# ─── Stage 2 ──────────────────────────────────────────────────────────────────

def run_stage2(stage1_output, image_dir, stage2_output):
    """Stage 1 の出力に対してページ全体 DDRM を適用する。"""
    print('\n=== Stage 2: Page DDRM ===')
    os.makedirs(stage2_output, exist_ok=True)

    if not os.path.exists(MODEL_PAGE):
        print(f'  [SKIP] Stage 2 モデルが見つかりません: {MODEL_PAGE}')
        print('  model_page.pth を学習後に再実行してください。')
        return

    stage1_files = natsort.natsorted(glob.glob(os.path.join(stage1_output, '*.png')))
    # _fin.png は除外
    stage1_files = [f for f in stage1_files if not f.endswith('_fin.png')]
    print(f'対象: {len(stage1_files)} ページ  |  モデル: {MODEL_PAGE}')

    args = _Args()
    args.device     = DEVICE
    args.model_type = 'page'
    config = _Config()
    config.data.image_size = 256  # Stage 2 モデルの学習解像度に合わせる
    runner = Diffusion(args, config)

    for stage1_path in stage1_files:
        page_id = os.path.splitext(os.path.basename(stage1_path))[0]
        out_path = os.path.join(stage2_output, page_id + '.png')
        if os.path.exists(out_path):
            print(f'  [skip] {page_id}')
            continue

        mask_path = os.path.join(image_dir, page_id, '02_text_black.png')
        if not os.path.exists(mask_path):
            print(f'  [WARN] マスクなし: {page_id}')
            continue

        print(f'  処理中: {page_id}')

        # Stage 1 出力を一時ディレクトリに配置して sample() に渡す
        tmp_dir = os.path.join('/tmp/pipeline_page_tmp', page_id)
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy(stage1_path, os.path.join(tmp_dir, '00_input.png'))
        shutil.copy(mask_path,   os.path.join(tmp_dir, 'used_mask_raw.png'))
        for sub in ['result', 'y', 'x', 'noisemask', 'backmask', 'cut_image', 'mask_dir', 'cat_dir']:
            os.makedirs(os.path.join(tmp_dir, sub), exist_ok=True)

        args.back_ground = os.path.join(tmp_dir, '00_input.png')

        try:
            runner.sample(tmp_dir)
        except Exception as e:
            print(f'  [ERROR] {page_id}: {e}')
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        results = natsort.natsorted(glob.glob(os.path.join(tmp_dir, 'result', '0_t*.png')))
        if not results:
            print(f'  [WARN] 結果ファイルなし: {page_id}')
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        out_img = apply_gamma_pil(Image.open(results[-1]).convert('RGB'), gamma=GAMMA)
        out_img.save(out_path)
        print(f'  [保存] {out_path}')

        # 赤テキスト合成
        red_mask_path = os.path.join(image_dir, page_id, '03_text_red.png')
        orig_path     = os.path.join(image_dir, page_id, '00_input.png')
        if os.path.exists(red_mask_path) and os.path.exists(orig_path):
            red_mask = np.array(Image.open(red_mask_path).convert('L')) > 127
            orig_arr = np.array(Image.open(orig_path).convert('RGB'))
            out_arr  = np.array(out_img)
            # マスクと出力サイズが違う場合はリサイズ
            if red_mask.shape != out_arr.shape[:2]:
                red_mask = np.array(Image.fromarray(red_mask.astype(np.uint8) * 255)
                                    .resize((out_arr.shape[1], out_arr.shape[0]), Image.LANCZOS)) > 127
                orig_arr = np.array(Image.fromarray(orig_arr)
                                    .resize((out_arr.shape[1], out_arr.shape[0]), Image.LANCZOS))
            out_arr[red_mask] = orig_arr[red_mask]
            fin_path = os.path.join(stage2_output, page_id + '_fin.png')
            Image.fromarray(out_arr).save(fin_path)
            print(f'  [保存] {fin_path}')

        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    print(f'入力: {IMAGE_DIR}')
    print(f'Stage 1 出力: {STAGE1_OUTPUT}')
    print(f'Stage 2 出力: {STAGE2_OUTPUT}')

    run_stage1(IMAGE_DIR, STAGE1_OUTPUT)
    run_stage2(STAGE1_OUTPUT, IMAGE_DIR, STAGE2_OUTPUT)

    print('\n=== 完了 ===')


if __name__ == '__main__':
    main()
