import traceback
import logging
import sys
import os
import torch
from src.inpainting import Diffusion
from src.random_crop import resize_image
import glob
import time
import re
import shutil
import cv2
import torchvision.utils as tvu
from PIL import Image
import torchvision.transforms as T
import numpy as np

torch.set_printoptions(sci_mode=False)

# =========================
# 引数・設定
# =========================
class Args():
    def __init__(self):
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
        self.num_classes=10
        self.dataset = 'orig'
        self.pad = 0
        self.mask_size = 32
        self.device = "cuda:0"

        # 入出力
        self.image_name = './徒然草/turezuregusa_ace_ver2'
        self.cut_path_output = './result/char/0_-1.png'
        self.back_ground = './background/image.png'
        self.masking = True
        self.mask_timesteps = 50
        self.image_folder = './heritage_result/patch_base/RESULT_TUREZUREGUSA_patch_base_ver2'
        self.cut_image ="cut_image"
        self.y_dir = "y"
        self.x_dir = "x"
        self.result ="result"
        self.noise_mask = "noisemask"
        self.back_mask = "backmask"
        self.mask_dir = 'mask_dir'
        self.cat_dir = 'cat_dir'

class Config():
    class data():
        dataset = "LSUN"
        category = "church_outdoor"
        image_size =128
        channels = 3
        logit_transform = False
        uniform_dequantization = False
        gaussian_dequantization= False
        random_flip = False
        rescaled = True
        num_workers= 0
        out_of_dist= True
    class model:
        type = "orig"
        dim = 64
        channels = 3
        dim_mults=(1, 2, 4,8 )
        resnet_block_groups=4
        attn_resolutions = [16, ]
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
        batch_size =64
        last_only = True
    class optim:
        weight_deca = 0.000
        optimizer = "Adam"
        lr = 0.00002
        beta1 = 0.9
        amsgrad = False
        eps = 0.00000001

# =========================
# ユーティリティ
# =========================
def extract_sort_key(name):
    match = re.match(r'^(\d+)_(\d+)_(\d+)$', name)
    if match:
        return (int(match.group(2)), int(match.group(3)))
    else:
        return (float('inf'), float('inf'))

# ---- フェザー用の重み生成 ----
def _cosine_ramp(n: int):
    """0→1→0 のなだらかなランプ（ハニング系）。nはオーバーラップ幅。"""
    if n <= 0:
        return np.ones(1, dtype=np.float32)
    x = np.linspace(0, np.pi, 2*n, dtype=np.float32)  # 長さ 2n
    r = 0.5 * (1 - np.cos(x))  # 0→1→0
    # 使うのは 0→1 の半分 or 1→0 の半分を端に貼る
    return r

def _make_weight_1d(length: int, left_ov: int, right_ov: int):
    """端でコサイン・フェザー。中央=1、端で0に滑らかに。"""
    # ガウス型重み: 端で0, 中央で1に近い値
    w = np.ones(length, dtype=np.float32)
    # 有効なオーバーラップ幅
    ov = max(left_ov, right_ov, 1)
    sigma = ov / 2.0
    x = np.arange(length)
    center = (length - 1) / 2.0
    w = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    w /= w.max()  # 最大値で正規化
    return w

def _make_weight_2d(patch_h, patch_w, y1, y2, x1, x2, H, W, overlap):
    """画像端での実効オーバーラップを考慮した2D重み."""
    top_ov    = min(overlap, y1)
    left_ov   = min(overlap, x1)
    bottom_ov = min(overlap, max(0, H - y2))
    right_ov  = min(overlap, max(0, W - x2))

    wy = _make_weight_1d(patch_h, top_ov, bottom_ov)   # (ph,)
    wx = _make_weight_1d(patch_w, left_ov, right_ov)   # (pw,)
    w2d = wy[:, None] * wx[None, :]                    # (ph,pw)
    return np.clip(w2d, 1e-6, None).astype(np.float32)

# =========================
# ブレンディング（重み付き累積）
# =========================
def blend_patch(accum, wacc, patch_rgb, x, y, overlap, img_H, img_W, text_mask=None, mask_boost=0.25):
    """
    accum:   (H,W,3) float32  画素*重みの累積
    wacc:    (H,W,1) float32  重みの累積
    patch_rgb: (ph,pw,3) uint8 or float32 (RGB)
    x,y:     左上
    overlap: オーバーラップ幅
    img_H,W: 元画像サイズ
    text_mask: (ph,pw) 0/1（テキスト優先したい場合）
    """
    ph, pw = patch_rgb.shape[:2]
    y1, x1 = y, x
    y2, x2 = y + ph, x + pw

    w = _make_weight_2d(ph, pw, y1, y2, x1, x2, img_H, img_W, overlap)  # (ph,pw)

    if text_mask is not None:
        w = w * (1.0 + mask_boost * text_mask.astype(np.float32))

    w3 = w[..., None]  # (ph,pw,1)
    patch_f = patch_rgb.astype(np.float32)

    accum[y1:y2, x1:x2, :] += patch_f * w3
    wacc[y1:y2, x1:x2, 0]   += w
    return accum, wacc

# =========================
# パッチ推論＋フェザーブレンド合成
# =========================
def patch_inference(image_bgr, mask01, runner, patch_size=128, overlap=64):
    """
    image_bgr: (H,W,3) uint8, BGR (cv2読込想定)
    mask01:    (H,W) uint8 {0,1} （テキスト=1 など）
    """
    H, W = image_bgr.shape[:2]
    stride = patch_size - overlap

    accum = np.zeros((H, W, 3), dtype=np.float32)
    wacc  = np.zeros((H, W, 1), dtype=np.float32)

    # パッチ座標
    patch_coords = [(y, x) for y in range(0, H, stride) for x in range(0, W, stride)]
    total_patches = len(patch_coords)

    for i, (y, x) in enumerate(patch_coords, 1):
        print(f"[patch_inference] {i}/{total_patches} patch ({y},{x}) 推論中...")
        y1, x1 = y, x
        y2, x2 = min(y1 + patch_size, H), min(x1 + patch_size, W)

        patch_bgr = image_bgr[y1:y2, x1:x2]
        patch_mask = mask01[y1:y2, x1:x2]
        ph, pw = patch_bgr.shape[:2]

        # パディング
        need_pad = (ph != patch_size) or (pw != patch_size)
        if need_pad:
            pad_patch = np.zeros((patch_size, patch_size, 3), dtype=patch_bgr.dtype)
            pad_mask  = np.zeros((patch_size, patch_size), dtype=patch_mask.dtype)
            pad_patch[:ph, :pw] = patch_bgr
            pad_mask[:ph, :pw]  = patch_mask
            patch_bgr, patch_mask = pad_patch, pad_mask

        # runner.sample が使う一時ディレクトリ
        tmp_dir = './tmp_patch_infer'
        os.makedirs(tmp_dir, exist_ok=True)
        patch_img_path = os.path.join(tmp_dir, 'patch.png')
        patch_mask_path = os.path.join(tmp_dir, 'mask.png')

        # OpenCVはBGRなのでそのまま保存OK
        cv2.imwrite(patch_img_path, patch_bgr)
        Image.fromarray((patch_mask * 255).astype(np.uint8)).save(patch_mask_path)

        patch_dir = os.path.join(tmp_dir, 'patchdir')
        os.makedirs(patch_dir, exist_ok=True)
        shutil.copy(patch_img_path, os.path.join(patch_dir, '00_input.png'))
        shutil.copy(patch_mask_path, os.path.join(patch_dir, '02_text_black.png'))
        shutil.copy(patch_mask_path, os.path.join(patch_dir, 'used_mask_raw.png'))

        # 推論（パッチサイズに合わせる）
        orig_size = runner.config.data.image_size
        runner.config.data.image_size = patch_size
        runner.sample(patch_dir)
        runner.config.data.image_size = orig_size

        result_path = os.path.join(patch_dir, 'result', f'0_t{runner.args.timesteps - 1}.png')
        if not os.path.exists(result_path):
            continue

        # 推論結果読込（BGR）
        result_patch_bgr = cv2.imread(result_path)
        # パディングしていたらクロップ
        result_patch_bgr = result_patch_bgr[:ph, :pw]
        # 端のパッチは小さい可能性があるのでリサイズ
        result_patch_bgr = cv2.resize(result_patch_bgr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)

        # ブレンドはRGBで行ってもBGRで行っても同じなので、このままBGRでOK
        # テキスト重み補正用マスク（0/1, 元サイズ）
        pmask_small = (mask01[y1:y2, x1:x2]).astype(np.uint8)

        accum, wacc = blend_patch(
            accum, wacc,
            result_patch_bgr, x1, y1, overlap,
            img_H=H, img_W=W,
            text_mask=pmask_small,   # 文字を少し優先
            mask_boost=0.25
        )

    # 正規化
    out_bgr = (accum / np.clip(wacc, 1e-6, None)).astype(np.uint8)
    return out_bgr

# =========================
# メイン
# =========================
def main(args=None):
    if args is None:
        args = Args()
    config = Config()

    os.makedirs(args.image_folder, exist_ok=True)

    try:
        # サブフォルダ名（y_x_idx）で並び替え
        images = sorted(
            [name for name in os.listdir(args.image_name) if os.path.isdir(os.path.join(args.image_name, name))],
            key=extract_sort_key
        )

        # 進捗チェック：処理済みと未処理をカウント
        total_images = len(images)
        processed_count = 0
        for image_path in images:
            image_name_path = os.path.join(args.image_folder, image_path)
            dir_base_name = os.path.basename(image_name_path)
            result_check_path = os.path.join(image_name_path, "result", f"{dir_base_name}.png")
            fin_check_path = os.path.join(image_name_path, "result", f"{dir_base_name}_fin.png")
            
            if os.path.exists(result_check_path) and os.path.exists(fin_check_path):
                processed_count += 1
        
        remaining_count = total_images - processed_count
        print(f"[PROGRESS] 全{total_images}件中、{processed_count}件処理済み、{remaining_count}件残り")
        
        if remaining_count == 0:
            print("[INFO] 全ての処理が完了しています。")
            return

        for image_path in images:
            image_name_path = os.path.join(args.image_folder, image_path)
            image_name = os.path.join(args.image_name, image_path, '00_input.png')
            mask_path1 = os.path.join(args.image_name, image_path, '02_text_black.png')

            # 処理済みかチェック（結果ファイルが存在するかで判定）
            dir_base_name = os.path.basename(image_name_path)
            result_check_path = os.path.join(image_name_path, "result", f"{dir_base_name}.png")
            fin_check_path = os.path.join(image_name_path, "result", f"{dir_base_name}_fin.png")
            
            # 両方のファイルが存在する場合はスキップ
            if os.path.exists(result_check_path) and os.path.exists(fin_check_path):
                print(f"[SKIP] {image_path} は既に処理済みです。")
                continue
            elif os.path.exists(result_check_path):
                print(f"[SKIP] {image_path} は部分的に処理済みです（{dir_base_name}.png存在）。")
                continue

            print(f"[PROCESSING] {image_path} を処理中...")

            # 入力画像
            img_pil = Image.open(image_name).convert("RGB")
            mask_img = Image.open(mask_path1).convert("L")

            width, height = img_pil.size
            os.makedirs(image_name_path, exist_ok=True)

            # コピー（元の構成に合わせて保持）
            shutil.copy(image_name, image_name_path)
            shutil.copy(mask_path1, image_name_path)

            # Numpy化
            img_np_rgb = np.array(img_pil, dtype=np.uint8)       # RGB
            img_np_bgr = img_np_rgb[:, :, ::-1].copy()           # BGR
            mask_np = (np.array(mask_img) > 128).astype(np.uint8)  # 0/1

            # 保存用マスク
            mask_save_name = "used_mask_raw.png"
            Image.fromarray((mask_np * 255).astype(np.uint8)).save(os.path.join(image_name_path, mask_save_name))

            # ページ用出力フォルダ
            folders_page = [args.y_dir, args.x_dir, args.result, args.noise_mask, args.back_mask]
            for folder in folders_page:
                os.makedirs(os.path.join(image_name_path, folder), exist_ok=True)

            # ランナー
            runner = Diffusion(args, config)

            # ---- パッチ推論＋フェザー合成（元サイズで） ----
            out_bgr = patch_inference(img_np_bgr, mask_np, runner, patch_size=128, overlap=64)

            # 出力保存（OpenCVはBGR）
            dir_base_name = os.path.basename(image_name_path)
            out_path = os.path.join(image_name_path, "result", f"{dir_base_name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, out_bgr)
            print(f"[SAVED] メイン結果を保存: {out_path}")

            # ---- 赤文字(03_redtext_opened.png)を元入力から復帰して合成 ----
            redtext_path = os.path.join(args.image_name, image_path, '03_text_red.png')
            if os.path.exists(redtext_path):
                redtext_img = Image.open(redtext_path).convert("L")
                input_img = Image.open(image_name).convert("RGB")
                out_img_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

                red_mask = (np.array(redtext_img) > 128)
                input_np = np.array(input_img)
                out_np = np.array(out_img_rgb)

                # 赤文字=1の領域は入力から復帰
                out_np[red_mask] = input_np[red_mask]

                fin_path = os.path.join(image_name_path, "result", f"{dir_base_name}_fin.png")
                Image.fromarray(out_np).save(fin_path)
                print(f"[SAVED] 最終結果を保存: {fin_path}")
            
            print(f"[COMPLETED] {image_path} の処理が完了しました。")

    except Exception:
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
