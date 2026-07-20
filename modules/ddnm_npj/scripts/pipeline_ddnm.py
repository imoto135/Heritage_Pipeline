"""
DDNM推論パイプライン v5。

設計（stain_removal/scripts/pipeline_csc.py の Stage 1 と同じ方向）:
  - 入力: ACE済み画像 (00_input.png)。モデルはACE済み画像で学習されているため、
    生画像を入れると生成領域とkept領域のトーンが乖離しマス目状の破綻が起きる
  - kept  = 墨文字 (02_text_black の黒) + 赤印章 (hue/彩度検出)
  - missing = それ以外すべて（背景+劣化）→ モデルが背景を再生成して染みを除去
  - y_0 = ACE済み入力パッチ（keptピクセルのみが数学的に使われる）
  - DDNM range-space correction: x0 + H†(H*y_0 - H*x0) で kept を入力へ固定
  - モデル入力はxtのまま（mask適用なし。学習が無条件DDPMのため）
  - 後処理: ガンマ補正(1.5) → 赤印章をACE入力から合成

使い方:
    cd modules/ddnm_npj
    python scripts/pipeline_ddnm.py \
        --csc_dir    ../stain_removal/data/split_dataset_csc_mask/test/200010454 \
        --output_dir ../../tmp_work/ddnm_test/v5 \
        --ckpt       ../stain_removal/experiments/model_char_csc/model_char_csc/ckpt_epoch0500.pth \
        --device     cuda:0
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DDRM_ROOT = os.path.join(os.path.dirname(_MODULE_ROOT), 'ddrm_npj')
if _DDRM_ROOT not in sys.path:
    sys.path.insert(0, _DDRM_ROOT)

if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from functions.denoising import ddnm_steps, compute_alpha
from functions.svd_replacement import Inpainting

_STAIN_ROOT = os.path.join(os.path.dirname(_MODULE_ROOT), 'stain_removal')
if _STAIN_ROOT not in sys.path:
    sys.path.append(_STAIN_ROOT)
from src.models.unet.unet import UNetModel
from src.postprocess import apply_gamma_pil

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_beta_schedule(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    return torch.from_numpy(betas).float()


def load_model(ckpt_path: str, device: str,
               model_channels: int = 64,
               num_res_blocks: int = 4,
               attention_resolutions: tuple = (16, 8),
               channels: int = 3) -> torch.nn.Module:
    model = UNetModel(
        in_channels=channels,
        model_channels=model_channels,
        out_channels=channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()
    logger.info(f'モデルロード完了: {ckpt_path}')
    return model


def load_mask(image_bgr: np.ndarray, mask_path: str = None,
              otsu_thresh: int = None) -> np.ndarray:
    """
    劣化マスクを読み込む。missing=1 の領域がインペインティング対象。
    文字・印章・背景は kept=0 として入力からピン留めされる。

    mask_path: CSCマスク（04_degradation.png、黒=劣化）。
    mask_path が None のときは Otsu フォールバック（暗部=missing）だが、
    これは文字も missing にしてしまうため stain removal には不適。CSCマスク推奨。

    Returns: uint8 (H,W)、劣化=1(missing) それ以外=0(kept)
    """
    # 彩度が高いピクセル（赤印章など）はmissingから除外して保存する
    # 赤印章の彩度はmin=103/p10=162、墨文字はp95=124 → 150で両者を安全に分離できる
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturated = hsv[:, :, 1] > 150

    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 04_degradation.png は白(>127)=劣化（generate_csc_masks.py は
        # degradation_mask を反転せずそのまま書き出す。text系のみ255-maskで反転）
        missing = (mask > 127) & ~saturated
        return missing.astype(np.uint8)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if otsu_thresh is not None:
        binary = (gray < otsu_thresh).astype(np.uint8) * 255
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_mask = (binary > 0) & ~saturated
    return text_mask.astype(np.uint8)


def compute_page_otsu(image_bgr: np.ndarray) -> int:
    """ページ全体でOtsu閾値を計算する。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return int(thresh)


def detect_red_seal(image_bgr: np.ndarray) -> np.ndarray:
    """
    赤印章・朱書きを hue+彩度 で検出する（ACE済み画像用）。
    ACEは彩度ノイズを増幅するため彩度のみでは墨と分離できず、hueを併用する。
    実測: hue≤12|≥168 & sat>180 で赤100%検出・墨誤検出1.5%。
    Returns: bool (H,W)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    return ((h <= 12) | (h >= 168)) & (s > 180)


def build_missing_mask_csc(ace_bgr: np.ndarray, text_black_path: str) -> np.ndarray:
    """
    pipeline_csc.py Stage 1 と同じ方向のマスクを作る。
    kept = 墨文字 (02_text_black の黒) + 赤印章 → missing = それ以外（背景+劣化）
    Returns: uint8 (H,W)、missing=1
    """
    tb = cv2.imread(text_black_path, cv2.IMREAD_GRAYSCALE)
    text_keep = tb < 127
    red_keep = detect_red_seal(ace_bgr)
    missing = ~(text_keep | red_keep)
    return missing.astype(np.uint8)


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


def infer_patch(patch_bgr: np.ndarray, mask01: np.ndarray,
                model: torch.nn.Module, betas: torch.Tensor,
                device: str, patch_size: int,
                bg_patch_bgr: np.ndarray = None,
                timesteps: int = 20, sigma_0: float = 0.05,
                eta: float = 0.85,
                use_shape_guidance: bool = False,
                mask_timesteps: int = 50,
                final_correction: bool = True,
                missing_pull: float = 0.0,
                full_kept: bool = False) -> np.ndarray:
    """
    DDNM で1パッチを修復する。

    Args:
        patch_bgr:    入力BGRパッチ (patch_size, patch_size, 3)
        mask01:       文字=1(missing) 背景=0(kept) のマスク
        model:        DDPMモデル
        betas:        betaスケジュール
        device:       デバイス
        patch_size:   パッチサイズ
        bg_patch_bgr: 背景参照BGRパッチ。Noneなら入力パッチの背景部分を白で埋めたものを使用
        timesteps:    DDIMステップ数
        sigma_0:      観測ノイズ水準
        eta:          ノイズスケール
        use_shape_guidance: True で DDRM同様に xt*mask（文字ゼロ化）をモデル入力に適用。
                      文字形状に墨を再生成させるガイダンス（npj論文のnoise masking）
        mask_timesteps: 形状ガイダンスを適用するタイムステップの下限
        missing_pull: missing側(汚れ)のx0予測を周辺背景色にソフトブレンドする強さ(0〜1)。
                      full_kept=Falseのときのみ使用
        full_kept: True で missing_indicesを空にし、全ピクセルをkept scapeにする。
                      range-space correctionが全画素をbg_patch_bgr(=汚れをinpaintで
                      埋めたクリーンな参照)へ強くピン留めする。DDRMのkept(背景)が
                      y_0に強収束してきれいになるのと同じ原理を汚れ領域にも適用する。

    Returns:
        修復済みBGRパッチ (patch_size, patch_size, 3)
    """
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # 背景参照画像 y_0: [-1,1]
    if bg_patch_bgr is not None:
        bg_rgb = cv2.cvtColor(bg_patch_bgr, cv2.COLOR_BGR2RGB)
        y_0 = to_tensor(Image.fromarray(bg_rgb)).unsqueeze(0).to(device)
    else:
        # 入力パッチの背景部分を使う: 文字ピクセルを白(255)で埋める
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        bg_fill = patch_rgb.copy()
        bg_fill[mask01 == 1] = 255  # 文字を白で埋める
        y_0 = to_tensor(Image.fromarray(bg_fill)).unsqueeze(0).to(device)

    if full_kept:
        # 全ピクセルkept化: missing_indicesを空にする
        missing = torch.tensor([], dtype=torch.long, device=device)
    else:
        # missing_indices: 文字ピクセル(mask01==1) → pixel-major フラットインデックス
        mask_flat = torch.from_numpy(mask01).reshape(-1).float()
        missing_pix = torch.nonzero(mask_flat == 1).long().reshape(-1)
        missing_r = missing_pix * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0).to(device)

    H_funcs = Inpainting(3, patch_size, missing, device)

    skip = 1000 // timesteps
    seq = range(0, 1000, skip)

    x = torch.randn(1, 3, patch_size, patch_size, device=device)

    # 形状ガイダンス用マスク: 背景=1 文字=0（DDRMと同方向）
    guidance_mask = None
    if use_shape_guidance:
        m = torch.from_numpy((1 - mask01).astype(np.float32))
        guidance_mask = m.reshape(1, 1, patch_size, patch_size).repeat(1, 3, 1, 1).to(device)

    # missing_pull用: missing領域をinpaintで周辺色から滑らかに埋めたローカル背景マップ
    pull_target = None
    if missing_pull > 0 and not full_kept:
        pull_bgr = cv2.inpaint(patch_bgr, (mask01 * 255).astype(np.uint8),
                              inpaintRadius=15, flags=cv2.INPAINT_TELEA)
        pull_rgb = cv2.cvtColor(pull_bgr, cv2.COLOR_BGR2RGB)
        pull_target = to_tensor(Image.fromarray(pull_rgb)).unsqueeze(0).to(device)

    betas_dev = betas.to(device)
    xs, _ = ddnm_steps(x, seq, model, betas_dev, H_funcs, y_0, sigma_0, device,
                       eta=eta, mask=guidance_mask, mask_timesteps=mask_timesteps,
                       final_correction=final_correction,
                       missing_pull=missing_pull, pull_target=pull_target)

    x_out = xs[-1].squeeze(0).clamp(-1, 1)
    x_out = (x_out + 1) / 2
    x_out = x_out.permute(1, 2, 0).cpu().numpy()
    x_out = (x_out * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(x_out, cv2.COLOR_RGB2BGR)


def infer_page(image_bgr: np.ndarray,
               model: torch.nn.Module, betas: torch.Tensor,
               device: str, patch_size: int = 128, overlap: int = 32,
               bg_image_bgr: np.ndarray = None,
               mask01: np.ndarray = None,
               timesteps: int = 20, sigma_0: float = 0.05,
               eta: float = 0.85,
               fixed_bg_patch: np.ndarray = None,
               use_shape_guidance: bool = False,
               mask_timesteps: int = 50,
               final_correction: bool = True,
               missing_pull: float = 0.0,
               full_kept: bool = False) -> np.ndarray:
    """
    fixed_bg_patch: (patch_size, patch_size, 3)。指定時は全パッチで同じ背景参照を使う
                    （ddrm_npj本家と同じ: isemonogatariをパッチサイズにリサイズしたもの）
    full_kept: infer_patchに転送。Trueならmissing_indicesを空にして全画素をkept化する。
    """
    H, W = image_bgr.shape[:2]

    # マスクがなければページ全体でOtsu計算してパッチ単位の誤検出を防ぐ
    if mask01 is None:
        otsu_thresh = compute_page_otsu(image_bgr)
        logger.info(f'  Otsu閾値: {otsu_thresh}')
        mask01 = load_mask(image_bgr, otsu_thresh=otsu_thresh)
        logger.info(f'  テキスト割合: {100*mask01.mean():.1f}%')

    stride = patch_size - overlap
    accum = np.zeros((H, W, 3), dtype=np.float32)
    wacc  = np.zeros((H, W),    dtype=np.float32)

    coords = [(y, x) for y in range(0, H, stride) for x in range(0, W, stride)]
    total  = len(coords)

    for i, (y, x) in enumerate(coords, 1):
        y2 = min(y + patch_size, H)
        x2 = min(x + patch_size, W)
        ph, pw = y2 - y, x2 - x

        patch_bgr  = image_bgr[y:y2, x:x2]
        patch_mask = mask01[y:y2, x:x2]

        bg_patch = None
        if fixed_bg_patch is not None:
            bg_patch = fixed_bg_patch
        elif bg_image_bgr is not None:
            bg_patch = bg_image_bgr[y:y2, x:x2]

        if ph != patch_size or pw != patch_size:
            pad_bgr  = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
            pad_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            pad_bgr[:ph, :pw]  = patch_bgr
            pad_mask[:ph, :pw] = patch_mask
            patch_bgr, patch_mask = pad_bgr, pad_mask

            if bg_patch is not None and fixed_bg_patch is None:
                pad_bg = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
                pad_bg[:ph, :pw] = bg_patch
                bg_patch = pad_bg

        res = infer_patch(patch_bgr, patch_mask, model, betas, device,
                          patch_size, bg_patch, timesteps, sigma_0, eta,
                          use_shape_guidance=use_shape_guidance,
                          mask_timesteps=mask_timesteps,
                          final_correction=final_correction,
                          missing_pull=missing_pull,
                          full_kept=full_kept)
        res = res[:ph, :pw]

        w2d = _make_weight_2d(ph, pw, y, y+ph, x, x+pw, H, W, overlap)
        accum[y:y+ph, x:x+pw] += res.astype(np.float32) * w2d[..., None]
        wacc[y:y+ph, x:x+pw]  += w2d

        if i % 50 == 0 or i == total:
            logger.info(f'  パッチ {i}/{total}')

    return (accum / np.clip(wacc[..., None], 1e-6, None)).astype(np.uint8)


def run_csc_mode(args, model, betas):
    """
    pipeline_csc.py Stage 1 相当のDDNM推論。
    csc_dir/<image_stem>/ に 00_input.png (ACE済み) と 02_text_black.png がある前提。
    """
    from PIL import Image as PILImage

    page_dirs = sorted([
        os.path.join(args.csc_dir, d)
        for d in os.listdir(args.csc_dir)
        if os.path.isdir(os.path.join(args.csc_dir, d))
    ])
    logger.info(f'{len(page_dirs)} ページを処理します (CSCモード)')

    for page_dir in page_dirs:
        stem = os.path.basename(page_dir)
        out_path = os.path.join(args.output_dir, stem + '.png')
        if os.path.exists(out_path):
            logger.info(f'スキップ（処理済み）: {stem}')
            continue

        input_path = os.path.join(page_dir, '00_input.png')
        tb_path    = os.path.join(page_dir, '02_text_black.png')
        if not (os.path.exists(input_path) and os.path.exists(tb_path)):
            logger.warning(f'必要ファイルなし: {page_dir}')
            continue

        logger.info(f'処理中: {stem}')
        ace_bgr = cv2.imread(input_path)

        mask01 = build_missing_mask_csc(ace_bgr, tb_path)
        logger.info(f'  missing={100*mask01.mean():.1f}% (kept=墨文字+赤印章)')

        result = infer_page(ace_bgr, model, betas, args.device,
                            args.patch_size, args.overlap,
                            None, mask01,
                            args.timesteps, args.sigma_0, args.eta)

        # ガンマ補正（pipeline_csc.py と同じ後処理）
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        gamma_img = apply_gamma_pil(PILImage.fromarray(result_rgb), gamma=args.gamma)
        out_arr = np.array(gamma_img)

        # 赤印章をACE入力から合成（ガンマの影響を受けない完全な色再現）
        red_keep = detect_red_seal(ace_bgr)
        ace_rgb = cv2.cvtColor(ace_bgr, cv2.COLOR_BGR2RGB)
        out_arr[red_keep] = ace_rgb[red_keep]

        cv2.imwrite(out_path, cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR))
        logger.info(f'  保存: {out_path}')


def build_self_background(img_bgr: np.ndarray, bg_mask01: np.ndarray) -> np.ndarray:
    """
    その文献自身のCSC背景クラスから紙の質感を保った参照画像を作る。
    背景クラス以外（文字・劣化）をOpenCVのinpaintで背景色に埋める。
    別文献参照(isemonogatari等)と違い、この文献固有の紙質感・色みを保持できる。
    """
    non_bg = (1 - bg_mask01).astype(np.uint8) * 255
    # inpaintは欠損領域が広いと粗くなるため、まず軽く膨張させた不要域を大きめの半径で埋める
    return cv2.inpaint(img_bgr, non_bg, inpaintRadius=15, flags=cv2.INPAINT_TELEA)


def run_ddrm_style_mode(args, model, betas):
    """
    v6: ddrm_npj本家 (pipeline_ddrm.py + codes/Inpainting.py) と同じ設計のDDNM版。

    - 入力: 生画像（ACEなし）
    - マスク: ページ全体Otsuの文字マスク → missing=文字、kept=背景
      （彩度>150の赤印章はmissingから除外し、最後に入力から合成）
    - y_0: 背景参照。--back_ground指定時はその画像（別文献の参照）をページサイズに
      リサイズしてパッチ位置クロップで使用。--mask_dir指定時（CSC背景クラスあり）は
      その文献自身の背景をinpaintで補完した自己参照を使う（紙質感を保持）。
      いずれもパッチ位置で切り出すためオーバーラップの継ぎ目が出ない。
    - 形状ガイダンス: xt*mask (i>=mask_timesteps) で文字形状に墨を再生成させる
    - range-space correction が背景を参照画像にハード固定
      → ページ全体で背景が均一（マス目破綻なし）
    """
    img_paths = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(args.ext)
    ])
    logger.info(f'{len(img_paths)} 枚を処理します (DDRM-styleモード, '
               f'{"degradation_only" if args.degradation_only else "text_guided"})')

    ext_bg = None
    if args.back_ground:
        ext_bg = cv2.imread(args.back_ground)
        if ext_bg is None:
            raise FileNotFoundError(f'背景参照画像が読めません: {args.back_ground}')
        logger.info(f'背景参照(外部): {args.back_ground} ({ext_bg.shape})')

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, stem + '.png')
        if os.path.exists(out_path):
            logger.info(f'スキップ（処理済み）: {stem}')
            continue

        logger.info(f'処理中: {stem}')
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            logger.error(f'読み込み失敗: {img_path}')
            continue

        bg_mask01 = None
        if args.degradation_only:
            # missing=劣化のみ。kept=文字+背景（そのまま保存し、汚れだけ周辺に馴染ませる）
            deg_path = os.path.join(args.mask_dir, stem, '04_degradation.png')
            if not os.path.exists(deg_path):
                logger.warning(f'劣化マスクが見つかりません: {deg_path} → スキップ')
                continue
            deg = cv2.imread(deg_path, cv2.IMREAD_GRAYSCALE)
            hsv_in = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask01 = ((deg > 127) & (hsv_in[:, :, 1] <= 150)).astype(np.uint8)
            logger.info(f'  CSC劣化マスク使用: missing割合={100*mask01.mean():.1f}%')
            bg_path = os.path.join(args.mask_dir, stem, '05_background.png')
            if os.path.exists(bg_path):
                bg_mask01 = (cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        else:
            # 文字マスク: CSCマスクがあれば優先（GMM分離で染みを文字と誤検出しない）、
            # なければページ全体Otsu。いずれも彩度>150の赤印章は除外
            mask01 = None
            if args.mask_dir:
                tb_path = os.path.join(args.mask_dir, stem, '02_text_black.png')
                bg_path = os.path.join(args.mask_dir, stem, '05_background.png')
                if os.path.exists(tb_path):
                    tb = cv2.imread(tb_path, cv2.IMREAD_GRAYSCALE)
                    hsv_in = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    mask01 = ((tb < 127) & (hsv_in[:, :, 1] <= 150)).astype(np.uint8)
                    logger.info(f'  CSC文字マスク使用: 文字割合={100*mask01.mean():.1f}%')
                if os.path.exists(bg_path):
                    bg_mask01 = (cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            if mask01 is None:
                otsu_thresh = compute_page_otsu(img_bgr)
                mask01 = load_mask(img_bgr, otsu_thresh=otsu_thresh)
                logger.info(f'  Otsu閾値={otsu_thresh}, 文字割合={100*mask01.mean():.1f}%')

        # 背景参照を決定
        # degradation_only: kept=文字+背景の両方を入力からそのままピン留めしたいので、
        # y_0は常に入力画像そのもの（文字位置に紙色を仕込んだ自己背景を使うと文字が消える）
        H, W = img_bgr.shape[:2]
        if args.degradation_only:
            if args.full_kept:
                # 全画素kept化: y_0 = 文字はそのまま・劣化(mask01)だけinpaintで埋めた画像。
                # range-space correctionが全画素をこの「汚れなし版」に強くピン留めするため、
                # DDRMのkept(背景)がy_0に収束してきれいになるのと同じ原理が汚れにも効く。
                #
                # 埋め色の暗化を防ぐ3点セット（いずれも欠くと埋め色が背景より暗くなる）:
                # 1. 文字も一緒にマスク: 劣化の大半は文字近傍（裏写り）で、黒が埋め色に混入する
                # 2. 赤も一緒にマスク: 赤(gray~128)に隣接する劣化の埋め色に赤の暗さが混入する
                #    （赤は最後に入力からオーバーレイされるためマスクに含めて問題ない）
                # 3. 劣化マスクをdilate: CSCマスクは裏写りの細い筋を部分的にしか覆っておらず、
                #    マスク外の未検出染みピクセルが埋めソースになる（入力相関0.54→0に改善）
                tb_path_fk = os.path.join(args.mask_dir, stem, '02_text_black.png')
                tr_path_fk = os.path.join(args.mask_dir, stem, '03_text_red.png')
                ink_fk = None
                if os.path.exists(tb_path_fk):
                    ink_fk = cv2.imread(tb_path_fk, cv2.IMREAD_GRAYSCALE) < 127
                hsv_fk = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                red_fk = hsv_fk[:, :, 1] > 150
                if os.path.exists(tr_path_fk):
                    red_fk = red_fk | (cv2.imread(tr_path_fk, cv2.IMREAD_GRAYSCALE) < 127)

                kernel_fk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                deg_dilated = cv2.dilate(mask01, kernel_fk)

                inpaint_mask = (deg_dilated == 1) | red_fk
                if ink_fk is not None:
                    inpaint_mask = inpaint_mask | ink_fk
                page_bg = cv2.inpaint(img_bgr, inpaint_mask.astype(np.uint8) * 255,
                                      inpaintRadius=15, flags=cv2.INPAINT_TELEA)

                if args.enhance:
                    # 修復強調（DDRM出力の見た目に寄せる）:
                    # ① 背景フラットフィールド補正: 低周波照明マップで割って
                    #    明るい紙色(bg_target)に統一（照明ムラ・経年変色も除去）
                    # bg_target<=0 は自動: 元の紙のトーン(中央値)を維持したまま均一化
                    # （GT比較ベンチマークでは原本トーン維持が必須）
                    gray_bg = cv2.cvtColor(page_bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    illum = cv2.GaussianBlur(gray_bg, (0, 0), sigmaX=50)
                    bg_target = args.bg_target if args.bg_target > 0 else float(np.median(gray_bg))
                    gain = bg_target / np.clip(illum, 1, None)
                    page_bg = np.clip(page_bg.astype(np.float32) * gain[..., None],
                                      0, 255).astype(np.uint8)

                    # ②残存シミの掃討: この時点のpage_bgは文字も赤も書き戻し前
                    # ＝「bg_targetより明確に暗いピクセルはすべて残存シミ」と断定できる。
                    # CSCマスクの取りこぼし・dilate範囲外のハローもここで全て捕まえる
                    if args.residual_thresh > 0:
                        for _ in range(2):
                            gray_now = cv2.cvtColor(page_bg, cv2.COLOR_BGR2GRAY)
                            resid = (gray_now < bg_target - args.residual_thresh)
                            if resid.sum() == 0:
                                break
                            resid = cv2.dilate(resid.astype(np.uint8), kernel_fk)
                            page_bg = cv2.inpaint(page_bg, resid * 255,
                                                  inpaintRadius=15, flags=cv2.INPAINT_TELEA)
                        logger.info(f'  残存シミ掃討: thresh={args.residual_thresh}')

                    # ③ 文字を濃くして書き戻す（DDRMの濃い文字再生成に相当）
                    # ただしCSC文字マスクにはシミ片の誤分類が混じるため、連結成分ごとに
                    # 「本物の墨」（平均gray≤125 or 濃い芯min<90を持つ）だけを書き戻す。
                    # 薄い茶色のシミ片成分は書き戻さない（=y_0の埋め色のまま消える）
                    if ink_fk is not None:
                        gray_in_fk = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        n_lab, labels_fk = cv2.connectedComponents(
                            ink_fk.astype(np.uint8), connectivity=8)
                        ink_write = np.zeros_like(ink_fk)
                        dropped = 0
                        for li in range(1, n_lab):
                            comp = labels_fk == li
                            vals = gray_in_fk[comp]
                            if vals.mean() <= 125 or vals.min() < 90:
                                ink_write |= comp
                            else:
                                dropped += comp.sum()
                        page_bg[ink_write] = np.clip(
                            img_bgr[ink_write].astype(np.float32) * args.text_gain,
                            0, 255).astype(np.uint8)
                        logger.info(f'  文字書き戻し: シミ片除外={dropped}px')
                    logger.info(f'  enhance: 背景→{bg_target:.0f}, 文字gain={args.text_gain}')
                else:
                    if ink_fk is not None:
                        page_bg[ink_fk] = img_bgr[ink_fk]
                page_bg[red_fk] = img_bgr[red_fk]
                logger.info('  full_kept: 劣化(dilate5px)+文字+赤を除外したinpaint y_0を使用')
            else:
                page_bg = img_bgr
        elif ext_bg is not None:
            page_bg = cv2.resize(ext_bg, (W, H), interpolation=cv2.INTER_LINEAR)
        elif bg_mask01 is not None:
            page_bg = build_self_background(img_bgr, bg_mask01)
            logger.info(f'  自己文献背景を使用 (CSC背景クラス={100*bg_mask01.mean():.1f}%)')
        else:
            page_bg = img_bgr

        # degradation_onlyでは文字は既にkept=保存対象なので形状ガイダンス不要
        use_guidance = not args.degradation_only

        if args.y0_only:
            # アブレーション: 拡散モデルを通さず古典処理のy_0をそのまま出力
            result = page_bg.copy()
        else:
            result = infer_page(img_bgr, model, betas, args.device,
                                args.patch_size, args.overlap,
                                page_bg, mask01,
                                args.timesteps, args.sigma_0, args.eta,
                                use_shape_guidance=use_guidance,
                                mask_timesteps=args.mask_timesteps,
                                final_correction=not args.no_final_correction,
                                missing_pull=args.missing_pull,
                                full_kept=args.full_kept)

        # 赤印章を入力から合成（拡散過程に含めず後からオーバーレイ。npj論文の設計）
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        red_keep = hsv[:, :, 1] > 150
        result[red_keep] = img_bgr[red_keep]

        cv2.imwrite(out_path, result)
        logger.info(f'  保存: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='DDNM推論パイプライン v6')
    parser.add_argument('--csc_dir',    default=None,
                        help='CSCマスクディレクトリ (<stem>/00_input.png 構造)。指定時はCSCモード')
    parser.add_argument('--degradation_only', action='store_true',
                        help='missing=劣化領域のみ、kept=文字+背景（汚れだけを周辺に馴染ませる。'
                             '--mask_dirにCSCマスク(04_degradation.png)が必要）')
    parser.add_argument('--ddrm_style', action='store_true',
                        help='ddrm_npj本家と同じ設計で推論（生画像+Otsu+背景参照+形状ガイダンス）')
    parser.add_argument('--input_dir',  default=None, help='入力画像ディレクトリ')
    parser.add_argument('--output_dir', required=True, help='出力ディレクトリ')
    parser.add_argument('--ckpt',       required=True, help='DDPMチェックポイント (.pth)')
    parser.add_argument('--mask_dir',   default=None,  help='劣化マスクディレクトリ（旧モード用）')
    parser.add_argument('--back_ground', default=None, help='背景参照画像')
    parser.add_argument('--device',     default='cuda:0')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--overlap',    type=int, default=32)
    parser.add_argument('--timesteps',  type=int, default=20)
    parser.add_argument('--sigma_0',    type=float, default=0.05)
    parser.add_argument('--eta',        type=float, default=0.85)
    parser.add_argument('--gamma',      type=float, default=1.5)
    parser.add_argument('--mask_timesteps', type=int, default=50,
                        help='形状ガイダンス(xt*mask)を適用するtの下限')
    parser.add_argument('--missing_pull', type=float, default=0.0,
                        help='missing側(汚れ)のx0予測を周辺背景色にソフトブレンドする強さ(0〜1)。'
                             '0だと汚れが別の色に置換されるだけで消えきらない')
    parser.add_argument('--enhance', action='store_true',
                        help='full_kept時に修復を強調: 背景をフラットフィールド補正で'
                             'bg_targetの明るさに統一し、文字をtext_gainで濃くする')
    parser.add_argument('--bg_target', type=float, default=203,
                        help='enhance時の背景目標輝度（DDRM出力実測=203）')
    parser.add_argument('--text_gain', type=float, default=0.5,
                        help='enhance時の文字濃度係数（小さいほど濃い。0.5で入力75→37相当）')
    parser.add_argument('--residual_thresh', type=float, default=0,
                        help='enhance時の残存シミ掃討閾値。bg_target-この値より暗い背景画素を'
                             'シミとみなし再inpaint（0=無効、推奨10）')
    parser.add_argument('--y0_only', action='store_true',
                        help='アブレーション: 拡散モデルを通さずy_0(古典処理)をそのまま出力')
    parser.add_argument('--full_kept', action='store_true',
                        help='missing_indicesを空にし全画素をkept scapeにする。'
                             'degradation_onlyと併用時、y_0=劣化領域をinpaintで埋めた画像へ'
                             '全画素を強くピン留めする（DDRMのkept背景と同じ収束原理）')
    parser.add_argument('--no_final_correction', action='store_true',
                        help='最終出力を未補正のx0予測にする（DDRM同様、境界をモデルが均す）')
    parser.add_argument('--num_res_blocks', type=int, default=4,
                        help='UNetのresidual block数（model_char=2, model_char_csc=4）')
    parser.add_argument('--ext',        default='.jpg')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.ckpt, args.device, num_res_blocks=args.num_res_blocks)
    betas = get_beta_schedule()

    if args.ddrm_style:
        if not args.input_dir:
            parser.error('--ddrm_style には --input_dir が必要です')
        if not args.back_ground and not args.mask_dir:
            parser.error('--ddrm_style には --back_ground（外部背景参照）か '
                        '--mask_dir（自己文献のCSC背景を使用）のいずれかが必要です')
        run_ddrm_style_mode(args, model, betas)
        return

    if args.csc_dir:
        run_csc_mode(args, model, betas)
        return

    if not args.input_dir:
        parser.error('--csc_dir か --input_dir のどちらかを指定してください')

    # ───── 旧モード（生画像 + 劣化マスク/Otsu）─────
    bg_image_bgr = None
    if args.back_ground:
        bg_image_bgr = cv2.imread(args.back_ground)
        if bg_image_bgr is None:
            logger.error(f'背景画像の読み込み失敗: {args.back_ground}')
        else:
            logger.info(f'背景参照画像: {args.back_ground} ({bg_image_bgr.shape})')

    img_paths = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(args.ext)
    ])
    logger.info(f'{len(img_paths)} 枚を処理します')

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, stem + '.png')
        if os.path.exists(out_path):
            logger.info(f'スキップ（処理済み）: {stem}')
            continue

        logger.info(f'処理中: {stem}')
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            logger.error(f'読み込み失敗: {img_path}')
            continue

        # CSCマスク or None（Noneならinfer_page内でページOtsu計算）
        mask01 = None
        if args.mask_dir:
            mask_path = os.path.join(args.mask_dir, stem, '04_degradation.png')
            if os.path.exists(mask_path):
                mask01 = load_mask(img_bgr, mask_path)
                logger.info(f'  CSC劣化マスク使用: missing={100*mask01.mean():.1f}%')
            else:
                logger.warning(f'  CSCマスクが見つかりません: {mask_path} → Otsuフォールバック')

        # 背景参照画像を入力画像サイズに合わせてリサイズ
        page_bg = None
        if bg_image_bgr is not None:
            H, W = img_bgr.shape[:2]
            page_bg = cv2.resize(bg_image_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        result = infer_page(img_bgr, model, betas, args.device,
                            args.patch_size, args.overlap,
                            page_bg, mask01,
                            args.timesteps, args.sigma_0, args.eta)
        cv2.imwrite(out_path, result)
        logger.info(f'  保存: {out_path}')


if __name__ == '__main__':
    main()
