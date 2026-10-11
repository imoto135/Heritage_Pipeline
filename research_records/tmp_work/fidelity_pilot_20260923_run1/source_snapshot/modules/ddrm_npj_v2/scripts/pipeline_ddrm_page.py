"""
DDRM Stage2 ページ推論パイプライン（stain_removal論文の2段階手法、ddrm_npj_v2用）。

処理フロー（論文 Section IV-B）:
  Stage1（別スクリプト pipeline_ddrm.py）:
    入力画像 → パッチ分割 → patchモデル推論 → feather blending → patch結果画像

  Stage2（本スクリプト）:
    patch結果画像 → 256×256にリサイズ
    → 文字マスク（--mask_dir指定時はCSC黒文字マスク=npj提案手法、
       未指定時は大津二値化=比較ベースライン）
    → pageモデル推論 → 元解像度に戻す → ガンマ補正（γ=1.5）
    → --mask_dir指定時は03_text_red.pngで赤文字を元画像から合成

使い方（npj提案手法・CSCマスク使用）:
    cd modules/ddrm_npj_v2
    python scripts/pipeline_ddrm_page.py \
        --patch_result_dir ../../tmp_work/ddrm_result_v2/200003803 \
        --output_dir       ../../tmp_work/ddrm_page_result_v2/200003803 \
        --mask_dir         ../../data/split_dataset_csc_mask_v2/test/200003803 \
        --ckpt             ../../modules/stain_removal/experiments/model_page_csc/model_page_csc/ckpt_epoch0500.pth \
        --device           cuda:0 \
        --wandb_run_name   ddrm-page-infer-csc-v2-gpu0-200003803

使い方（Otsuベースライン・従来動作）:
    python scripts/pipeline_ddrm_page.py \
        --patch_result_dir ../../tmp_work/ddrm_result/200003803 \
        --output_dir       ../../tmp_work/ddrm_page_result_otsu/200003803 \
        --ckpt             ../../modules/stain_removal/model/model_page/ckpt_epoch0400.pth \
        --device           cuda:0 \
        --wandb_run_name   ddrm-page-infer-otsu-ep400-gpu0-200003803
"""

import argparse
import logging
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from codes.Inpainting import Diffusion

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PAGE_INFER_SIZE = 256  # pageモデルの学習サイズ（256px）
GAMMA = 1.5            # 論文 Section IV-C


class _Config:
    class data:
        dataset = "LSUN"
        category = "church_outdoor"
        image_size = PAGE_INFER_SIZE
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
        batch_size = 1
        n_epochs = 1000
        n_iters = 10
        snapshot_freq = 5000
        validation_freq = 2000

    class sampling:
        batch_size = 1
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


def gamma_correction(img_bgr: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """ガンマ補正: y = x^gamma（論文 Section IV-C）"""
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return lut[img_bgr]


def otsu_text_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.uint8)


def csc_text_mask(mask_page_dir: str, size: int) -> np.ndarray | None:
    """
    generate_mask_csc.py が出力したCSC黒文字マスクを読み込み、pageモデルの
    学習解像度(size)にリサイズする（npj提案手法）。
    マスク規約: 0（黒）=文字, 255（白）=背景・対象外
    Returns: uint8 (size,size)、文字=1 背景=0。ファイルがなければNone
    """
    path = os.path.join(mask_page_dir, '02_text_black.png')
    if not os.path.exists(path):
        return None
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
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


def process_page(patch_result_path: str, out_dir: str, runner: Diffusion,
                 tmp_dir: str, wb=None, mask_dir: str = None) -> None:
    page_id = os.path.splitext(os.path.basename(patch_result_path))[0]
    result_path = os.path.join(out_dir, page_id, 'result', f'{page_id}.png')

    if os.path.exists(result_path):
        logger.info(f'スキップ（処理済み）: {page_id}')
        return

    logger.info(f'処理中: {page_id}')
    img_bgr = cv2.imread(patch_result_path)
    if img_bgr is None:
        logger.error(f'読み込み失敗: {patch_result_path}')
        return

    orig_h, orig_w = img_bgr.shape[:2]

    # pageモデルの学習サイズにリサイズ
    img_resized = cv2.resize(img_bgr, (PAGE_INFER_SIZE, PAGE_INFER_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)

    # 文字マスク: --mask_dir指定時はCSC黒文字マスク（npj提案手法）、
    # なければ大津二値化（比較ベースライン）
    mask01 = None
    mask_page_dir = os.path.join(mask_dir, page_id) if mask_dir else None
    if mask_page_dir:
        mask01 = csc_text_mask(mask_page_dir, PAGE_INFER_SIZE)
        if mask01 is None:
            logger.warning(f'CSCマスクが見つかりません: {mask_page_dir} → Otsuにフォールバック')
    if mask01 is None:
        mask01 = otsu_text_mask(img_resized)

    # 作業ディレクトリ準備
    page_tmp = os.path.join(tmp_dir, page_id)
    shutil.rmtree(page_tmp, ignore_errors=True)
    os.makedirs(page_tmp, exist_ok=True)
    for sub in [runner.args.y_dir, runner.args.x_dir, runner.args.result,
                runner.args.noise_mask, runner.args.back_mask]:
        os.makedirs(os.path.join(page_tmp, sub), exist_ok=True)

    cv2.imwrite(os.path.join(page_tmp, '00_input.png'), img_resized)
    Image.fromarray((mask01 * 255).astype(np.uint8)).save(
        os.path.join(page_tmp, 'used_mask_raw.png'))

    # pageモデルで推論
    runner.config.data.image_size = PAGE_INFER_SIZE
    runner.sample(page_tmp)

    result_tmp = os.path.join(page_tmp, 'result', f'0_t{runner.args.timesteps}.png')
    if not os.path.exists(result_tmp):
        logger.warning(f'推論結果なし: {result_tmp}')
        return

    out_bgr = cv2.imread(result_tmp)
    # 元解像度に戻す
    out_bgr = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    # ガンマ補正（論文 Section IV-C, γ=1.5）
    out_bgr = gamma_correction(out_bgr, GAMMA)

    # 赤文字合成（npj提案手法: 拡散に含めず推論後に元画像から合成）
    if mask_page_dir:
        red_mask = load_red_mask(mask_page_dir)
        orig_path = os.path.join(mask_page_dir, '00_input.png')
        if red_mask is not None and os.path.exists(orig_path):
            orig_bgr = cv2.imread(orig_path)
            if red_mask.shape != out_bgr.shape[:2]:
                red_mask = cv2.resize(red_mask.astype(np.uint8),
                                      (out_bgr.shape[1], out_bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
                orig_bgr = cv2.resize(orig_bgr, (out_bgr.shape[1], out_bgr.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)
            out_bgr[red_mask] = orig_bgr[red_mask]
            logger.info('  赤文字合成: 完了')

    os.makedirs(os.path.join(out_dir, page_id, 'result'), exist_ok=True)
    cv2.imwrite(result_path, out_bgr)
    logger.info(f'  保存: {result_path}')

    if wb is not None:
        try:
            import wandb
            wb.log({
                'page_id':       page_id,
                'patch_output':  wandb.Image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
                'page_output':   wandb.Image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)),
                'mask':          wandb.Image((mask01 * 255).astype(np.uint8)),
            })
        except Exception as e:
            logger.warning(f'wandb log失敗: {e}')

    shutil.rmtree(page_tmp, ignore_errors=True)


def collect_patch_results(patch_result_dir: str) -> list[str]:
    """patch推論の出力ディレクトリから結果PNGを収集する。
    構造: {patch_result_dir}/{page_id}/result/{page_id}.png
    """
    paths = []
    if not os.path.isdir(patch_result_dir):
        return paths
    for page_id in sorted(os.listdir(patch_result_dir)):
        p = os.path.join(patch_result_dir, page_id, 'result', f'{page_id}.png')
        if os.path.isfile(p):
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description='DDRM Stage2 page推論（patch結果 → pageモデル → ガンマ補正）')
    parser.add_argument('--patch_result_dir', required=True,
                        help='pipeline_ddrm.pyの出力ディレクトリ（文書単位）')
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--ckpt',             required=True)
    parser.add_argument('--device',           default='cuda:0')
    parser.add_argument('--back_ground',      default='./background/white.png')
    parser.add_argument('--mask_dir',         default=None,
                        help='generate_mask_csc.pyの出力ディレクトリ（npj提案手法のCSCマスクを使用。'
                             '未指定時は大津二値化ベースラインにフォールバック）')
    parser.add_argument('--tmp_dir',          default=None)
    parser.add_argument('--wandb_project',    default='heritage-diffusion')
    parser.add_argument('--wandb_run_name',   default=None)
    parser.add_argument('--no_wandb',         action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = (args.wandb_run_name
                        or f'ddrm-page-infer-{os.path.basename(args.patch_result_dir)}')
            wb = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    'patch_result_dir': args.patch_result_dir,
                    'ckpt':             args.ckpt,
                    'device':           args.device,
                    'page_infer_size':  PAGE_INFER_SIZE,
                    'gamma':            GAMMA,
                    'mask':             'csc' if args.mask_dir else 'otsu',
                    'mode':             'stage2-page',
                },
            )
            logger.info(f'wandb run: {run_name}')
        except Exception as e:
            logger.warning(f'wandb初期化失敗（無効化して続行）: {e}')

    img_paths = collect_patch_results(args.patch_result_dir)
    if not img_paths:
        logger.error(f'patch推論結果が見つかりません: {args.patch_result_dir}')
        return

    logger.info(f'対象画像数: {len(img_paths)}')

    config = _Config()
    ddrm_args = _Args(ckpt=args.ckpt, device=args.device, back_ground=args.back_ground)
    runner = Diffusion(ddrm_args, config)

    for i, img_path in enumerate(img_paths, 1):
        try:
            process_page(img_path, args.output_dir, runner, tmp_dir, wb=wb, mask_dir=args.mask_dir)
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
