"""
DDRM推論パイプライン（ddrm_npj モジュール用）。

処理フロー:
  1. 文字マスク生成: --mask_dir指定時はgenerate_mask_csc.pyのCSC黒文字マスク
     （npj提案手法）、未指定時は大津二値化（npj論文中の比較ベースライン）
  2. DDRM パッチ推論（文字領域を修復）
  3. --mask_dir指定時は推論後に03_text_red.pngで赤文字を元画像から合成
  4. wandbに進捗・サンプル画像をログ

使い方（npj提案手法・CSCマスク使用）:
    cd modules/ddrm_npj
    python scripts/pipeline_ddrm.py \
        --input_dir  ../../data/test/200003803 \
        --output_dir ../../tmp_work/ddrm_result/200003803 \
        --mask_dir   ./data/csc_output/200003803 \
        --ckpt       ../../modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth \
        --device     cuda:0 \
        --wandb_run_name  ddrm-infer-csc-gpu0

使い方（Otsuベースライン・従来動作）:
    python scripts/pipeline_ddrm.py \
        --input_dir  ../../data/test/200003803 \
        --output_dir ../../tmp_work/ddrm_result_otsu/200003803 \
        --ckpt       ../../modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth \
        --device     cuda:0 \
        --wandb_run_name  ddrm-infer-otsu-gpu0

stain_removal モジュールとは完全に独立しており、互いに影響しない。
"""

import argparse
import logging
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

# ddrm_npj ルートを sys.path に追加
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from codes.Inpainting import Diffusion

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Config / Args
# ──────────────────────────────────────────────

class _Config:
    class data:
        dataset = "LSUN"
        category = "church_outdoor"
        image_size = 128
        channels = 3
        logit_transform = False
        uniform_dequantization = False
        gaussian_dequantization = False
        random_flip = False
        rescaled = True
        num_workers = 0
        out_of_dist = True

    class model:
        type = "orig"
        dim = 64
        channels = 3
        dim_mults = (1, 2, 4, 8)
        resnet_block_groups = 4
        attn_resolutions = [16]
        dropout = 0.0
        var_type = 'fixedsmall'
        ema_rate = 0.999
        ema = True
        resamp_with_conv = True

    class diffusion:
        beta_schedule = 'linear'
        beta_start = 0.0001
        beta_end = 0.02
        num_diffusion_timesteps = 1000

    class training:
        batch_size = 64
        n_epochs = 1000
        n_iters = 10
        snapshot_freq = 5000
        validation_freq = 2000

    class sampling:
        batch_size = 64
        last_only = True

    class optim:
        weight_deca = 0.000
        optimizer = "Adam"
        lr = 0.00002
        beta1 = 0.9
        amsgrad = False
        eps = 1e-8


class _Args:
    def __init__(self, ckpt: str, device: str, back_ground: str):
        self.seed = 999
        self.timesteps = 20
        self.deg = 'maskshape'
        self.sigma_0 = 0.05
        self.eta = 0.5
        self.etaB = 1
        self.subset_start = 0
        self.subset_end = -1
        self.model_type = 'page'
        self.cls_cond = None
        self.num_classes = 10
        self.dataset = 'orig'
        self.pad = 0
        self.mask_size = 32
        self.device = device
        self.ckpt = ckpt
        self.back_ground = back_ground
        self.masking = True
        self.mask_timesteps = 50
        self.image_folder = ''
        self.image_name = ''
        self.cut_path_output = './result/char/0_-1.png'
        self.cut_image = "cut_image"
        self.y_dir = "y"
        self.x_dir = "x"
        self.result = "result"
        self.noise_mask = "noisemask"
        self.back_mask = "backmask"
        self.mask_dir = 'mask_dir'
        self.cat_dir = 'cat_dir'


# ──────────────────────────────────────────────
# 大津二値化マスク生成
# ──────────────────────────────────────────────

def otsu_text_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    大津二値化で黒文字マスクを生成する（npj論文中の比較ベースライン）。
    Returns: uint8 (H,W)、文字=1 背景=0
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.uint8)


def csc_text_mask(mask_page_dir: str) -> np.ndarray | None:
    """
    generate_mask_csc.py が出力したCSC黒文字マスクを読み込む（npj提案手法）。
    マスク規約: 0（黒）=文字, 255（白）=背景・対象外
    Returns: uint8 (H,W)、文字=1 背景=0。ファイルがなければNone
    """
    path = os.path.join(mask_page_dir, '02_text_black.png')
    if not os.path.exists(path):
        return None
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (gray < 127).astype(np.uint8)


def load_red_mask(mask_page_dir: str) -> np.ndarray | None:
    """
    generate_mask_csc.py が出力したCSC赤文字マスクを読み込む。
    マスク規約: 0（黒）=赤文字, 255（白）=対象外
    Returns: bool (H,W)、赤文字=True。ファイルがなければNone
    """
    path = os.path.join(mask_page_dir, '03_text_red.png')
    if not os.path.exists(path):
        return None
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return gray < 127


# ──────────────────────────────────────────────
# フェザーブレンド（パッチ合成）
# ──────────────────────────────────────────────

def _make_weight_1d(length: int, left_ov: int, right_ov: int) -> np.ndarray:
    ov = max(left_ov, right_ov, 1)
    sigma = ov / 2.0
    x = np.arange(length, dtype=np.float32)
    center = (length - 1) / 2.0
    w = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return w / w.max()


def _make_weight_2d(ph, pw, y1, y2, x1, x2, H, W, overlap) -> np.ndarray:
    top_ov    = min(overlap, y1)
    left_ov   = min(overlap, x1)
    bottom_ov = min(overlap, max(0, H - y2))
    right_ov  = min(overlap, max(0, W - x2))
    wy = _make_weight_1d(ph, top_ov, bottom_ov)
    wx = _make_weight_1d(pw, left_ov, right_ov)
    return np.clip(wy[:, None] * wx[None, :], 1e-6, None).astype(np.float32)


def _blend_patch(accum, wacc, patch_bgr, x, y, overlap, H, W):
    ph, pw = patch_bgr.shape[:2]
    y2, x2 = y + ph, x + pw
    w = _make_weight_2d(ph, pw, y, y2, x, x2, H, W, overlap)
    accum[y:y2, x:x2] += patch_bgr.astype(np.float32) * w[..., None]
    wacc[y:y2, x:x2]  += w
    return accum, wacc


# ──────────────────────────────────────────────
# パッチ推論
# ──────────────────────────────────────────────

def _patch_inference(image_bgr: np.ndarray, mask01: np.ndarray,
                     runner: Diffusion, tmp_dir: str,
                     patch_size: int = 128, overlap: int = 32) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    stride = patch_size - overlap
    accum = np.zeros((H, W, 3), dtype=np.float32)
    wacc  = np.zeros((H, W),    dtype=np.float32)

    coords = [(y, x) for y in range(0, H, stride) for x in range(0, W, stride)]
    total  = len(coords)

    for i, (y, x) in enumerate(coords, 1):
        y2, x2 = min(y + patch_size, H), min(x + patch_size, W)
        ph, pw = y2 - y, x2 - x

        patch_bgr  = image_bgr[y:y2, x:x2]
        patch_mask = mask01[y:y2, x:x2]

        if ph != patch_size or pw != patch_size:
            pad_bgr  = np.zeros((patch_size, patch_size, 3), dtype=patch_bgr.dtype)
            pad_mask = np.zeros((patch_size, patch_size),    dtype=patch_mask.dtype)
            pad_bgr[:ph, :pw]  = patch_bgr
            pad_mask[:ph, :pw] = patch_mask
            patch_bgr, patch_mask = pad_bgr, pad_mask

        patch_dir = os.path.join(tmp_dir, f'p_{y}_{x}')
        os.makedirs(patch_dir, exist_ok=True)
        cv2.imwrite(os.path.join(patch_dir, '00_input.png'), patch_bgr)
        Image.fromarray((patch_mask * 255).astype(np.uint8)).save(
            os.path.join(patch_dir, 'used_mask_raw.png'))

        for sub in [runner.args.y_dir, runner.args.x_dir, runner.args.result,
                    runner.args.noise_mask, runner.args.back_mask]:
            os.makedirs(os.path.join(patch_dir, sub), exist_ok=True)

        orig_size = runner.config.data.image_size
        runner.config.data.image_size = patch_size
        runner.sample(patch_dir)
        runner.config.data.image_size = orig_size

        result_path = os.path.join(patch_dir, 'result', f'0_t{runner.args.timesteps}.png')
        if not os.path.exists(result_path):
            logger.warning(f'推論結果なし: {result_path}')
            continue

        res_bgr = cv2.imread(result_path)[:ph, :pw]
        res_bgr = cv2.resize(res_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        accum, wacc = _blend_patch(accum, wacc, res_bgr, x, y, overlap, H, W)

        if i % 100 == 0:
            logger.info(f'  パッチ {i}/{total}')

    out = (accum / np.clip(wacc[..., None], 1e-6, None)).astype(np.uint8)
    return out


# ──────────────────────────────────────────────
# 1ページの処理
# ──────────────────────────────────────────────

def process_page(img_path: str, out_dir: str, runner: Diffusion, tmp_dir: str,
                 patch_size: int, overlap: int, wb=None, mask_dir: str = None) -> None:
    page_id = os.path.splitext(os.path.basename(img_path))[0]
    page_out = os.path.join(out_dir, page_id)
    result_path = os.path.join(page_out, 'result', f'{page_id}.png')

    if os.path.exists(result_path):
        logger.info(f'スキップ（処理済み）: {page_id}')
        return

    logger.info(f'処理中: {page_id}')
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        logger.error(f'読み込み失敗: {img_path}')
        return

    os.makedirs(os.path.join(page_out, 'result'), exist_ok=True)

    # 文字マスク: --mask_dir指定時はCSC黒文字マスク（npj提案手法）、
    # なければ大津二値化（npj論文中の比較ベースライン）
    mask01 = None
    mask_page_dir = os.path.join(mask_dir, page_id) if mask_dir else None
    if mask_page_dir:
        mask01 = csc_text_mask(mask_page_dir)
        if mask01 is None:
            logger.warning(f'CSCマスクが見つかりません: {mask_page_dir} → Otsuにフォールバック')
    if mask01 is None:
        mask01 = otsu_text_mask(img_bgr)

    # パッチ推論
    page_tmp = os.path.join(tmp_dir, page_id)
    shutil.rmtree(page_tmp, ignore_errors=True)
    os.makedirs(page_tmp, exist_ok=True)

    out_bgr = _patch_inference(img_bgr, mask01, runner, page_tmp, patch_size, overlap)

    # 赤文字合成（npj提案手法: 拡散に含めず推論後に元画像から合成）
    if mask_page_dir:
        red_mask = load_red_mask(mask_page_dir)
        if red_mask is not None:
            if red_mask.shape != out_bgr.shape[:2]:
                red_mask = cv2.resize(red_mask.astype(np.uint8),
                                      (out_bgr.shape[1], out_bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
            out_bgr[red_mask] = img_bgr[red_mask]
            logger.info('  赤文字合成: 完了')

    cv2.imwrite(result_path, out_bgr)
    logger.info(f'  保存: {result_path}')

    # wandbにサンプル画像をlog
    if wb is not None:
        try:
            import wandb
            wb.log({
                'page_id': page_id,
                'input':  wandb.Image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
                'output': wandb.Image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)),
                'mask':   wandb.Image((mask01 * 255).astype(np.uint8)),
            })
        except Exception as e:
            logger.warning(f'wandb log失敗: {e}')

    shutil.rmtree(page_tmp, ignore_errors=True)


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DDRM推論パイプライン（大津二値化マスク）')
    parser.add_argument('--input_dir',   required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--ckpt',        required=True,
                        help='DDRMモデルcheckpointパス (.pth)')
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--back_ground', default='./background/white.png')
    parser.add_argument('--mask_dir',    default=None,
                        help='generate_mask_csc.pyの出力ディレクトリ（npj提案手法のCSCマスクを使用。'
                             '未指定時は大津二値化ベースラインにフォールバック）')
    parser.add_argument('--patch_size',  type=int, default=128)
    parser.add_argument('--overlap',     type=int, default=32)
    parser.add_argument('--tmp_dir',     default=None)
    parser.add_argument('--wandb_project',  default='heritage-diffusion')
    parser.add_argument('--wandb_run_name', default=None,
                        help='wandb run名（例: ddrm-infer-otsu-gpu0）')
    parser.add_argument('--no_wandb', action='store_true',
                        help='wandbを無効化する')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # wandb初期化
    wb = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f'ddrm-infer-{os.path.basename(args.input_dir)}'
            wb = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    'input_dir':  args.input_dir,
                    'ckpt':       args.ckpt,
                    'device':     args.device,
                    'patch_size': args.patch_size,
                    'overlap':    args.overlap,
                    'mask':       'csc' if args.mask_dir else 'otsu',
                },
            )
            logger.info(f'wandb run: {run_name}')
        except Exception as e:
            logger.warning(f'wandb初期化失敗（無効化して続行）: {e}')
            wb = None

    # 入力画像一覧
    exts = ('.jpg', '.jpeg', '.png')
    img_paths = sorted(
        p for p in (os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    )
    if not img_paths:
        logger.error(f'画像が見つかりません: {args.input_dir}')
        return

    logger.info(f'対象画像数: {len(img_paths)}')
    if wb:
        wb.config.update({'total_images': len(img_paths)})

    # Diffusionランナー構築
    config = _Config()
    config.data.image_size = args.patch_size
    ddrm_args = _Args(ckpt=args.ckpt, device=args.device, back_ground=args.back_ground)
    runner = Diffusion(ddrm_args, config)

    for i, img_path in enumerate(img_paths, 1):
        try:
            process_page(img_path, args.output_dir, runner, tmp_dir,
                         args.patch_size, args.overlap, wb=wb, mask_dir=args.mask_dir)
            if wb:
                wb.log({'pages_done': i})
        except Exception as e:
            logger.error(f'エラー ({img_path}): {e}', exc_info=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if wb:
        wb.finish()
    logger.info('完了。')


if __name__ == '__main__':
    main()
