#!/usr/bin/env python3
"""
YOLOXで検出したcropをUNet++ + NAFNetで修復し、元ページ画像に貼り戻す。

実行例（1ページのみ）:
  conda run -n nafnet2 python scripts/run_restoration_pipeline.py --pages 200003803_00010_1

全ページ一括:
  conda run -n nafnet2 python scripts/run_restoration_pipeline.py
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules", "restoration", "nafnet"))
sys.path.insert(0, os.path.join(ROOT, "modules", "restoration", "unet++"))

import segmentation_models_pytorch as smp
from infer_withmask import load_model as load_nafnet, process_image


# ---------------------------------------------------------------------------
# パディング / アンパディング
# ---------------------------------------------------------------------------

def pad_to_128(img_bgr: np.ndarray, size: int = 128):
    """cropを128×128にアスペクト比保持でパディング。逆変換用の情報も返す。"""
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    top = (size - new_h) // 2
    left = (size - new_w) // 2

    # 背景色: Otsuで背景マスクを作って中央値
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bg_mask = thresh > 0
    if np.any(bg_mask):
        pad_color = np.median(img_bgr[bg_mask], axis=0).astype(int).tolist()
    else:
        pad_color = [255, 255, 255]

    bottom = size - new_h - top
    right = size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, left, top, h, w


def unpad_from_128(padded_bgr: np.ndarray, orig_h: int, orig_w: int,
                   scale: float, offset_x: int, offset_y: int) -> np.ndarray:
    """pad_to_128の逆変換。パディング領域を除去して元のcropサイズに戻す。"""
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    cropped = padded_bgr[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
    if cropped.shape[:2] == (orig_h, orig_w):
        return cropped
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# UNet++ マスク生成
# ---------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PaddedCropDataset(Dataset):
    """既に128×128のパディング済みcrop画像を受け取り、正規化のみ行うDataset。"""
    def __init__(self, padded_imgs: dict):
        self.stems = sorted(padded_imgs.keys())
        self.imgs = padded_imgs

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_bgr = self.imgs[stem]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - _MEAN) / _STD          # HWC float32
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))  # CHW
        return stem, tensor


def load_unetpp(weights: str, device: torch.device):
    model = smp.UnetPlusPlus(
        encoder_name="se_resnext50_32x4d",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def generate_masks(model, padded_imgs: dict, device: torch.device,
                   batch_size: int = 16) -> dict:
    """UNet++でソフトマスクを生成。入出力ともに128×128。"""
    dataset = PaddedCropDataset(padded_imgs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    masks = {}
    with torch.no_grad():
        for stems, tensors in loader:
            tensors = tensors.to(device, dtype=torch.float)
            probs = torch.sigmoid(model(tensors))
            for stem, prob in zip(stems, probs):
                mask_np = (prob.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                masks[stem] = mask_np
    return masks


# ---------------------------------------------------------------------------
# NAFNet 修復
# ---------------------------------------------------------------------------

def restore_crops(nafnet_model, padded_imgs: dict, masks: dict,
                  device: torch.device) -> dict:
    """NAFNetで修復。128×128入力 → 128×128出力。"""
    restored = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for stem, img_bgr in padded_imgs.items():
            if stem not in masks:
                continue
            lq_path = tmpdir / f"{stem}_lq.png"
            mask_path = tmpdir / f"{stem}_mask.png"
            cv2.imwrite(str(lq_path), img_bgr)
            cv2.imwrite(str(mask_path), masks[stem])

            result = process_image(nafnet_model, lq_path, mask_path,
                                   device=str(device))
            if result is not None:
                restored[stem] = result
    return restored


# ---------------------------------------------------------------------------
# 貼り戻し
# ---------------------------------------------------------------------------

def paste_to_page(page_img: np.ndarray, restored_crops: dict,
                  masks: dict, bboxes: list, pad_info: dict,
                  blend_mode: str = "mask") -> np.ndarray:
    """修復済みcropをページ画像の対応するbbox位置に貼り戻す。"""
    result = page_img.astype(np.float32)

    for i, bbox in enumerate(bboxes):
        stem = f"crop_{i:04d}"
        if stem not in restored_crops:
            continue

        info = pad_info[stem]
        scale, off_x, off_y, orig_h, orig_w = info

        # パディング除去して元cropサイズに復元
        restored_full = unpad_from_128(restored_crops[stem], orig_h, orig_w,
                                       scale, off_x, off_y)

        x1 = max(0, bbox["x1"])
        y1 = max(0, bbox["y1"])
        x2 = min(page_img.shape[1], bbox["x2"])
        y2 = min(page_img.shape[0], bbox["y2"])

        roi_h, roi_w = y2 - y1, x2 - x1
        if roi_h <= 0 or roi_w <= 0:
            continue

        # ROIサイズと一致しない場合はリサイズ
        if restored_full.shape[:2] != (roi_h, roi_w):
            restored_full = cv2.resize(restored_full, (roi_w, roi_h),
                                       interpolation=cv2.INTER_CUBIC)

        if blend_mode == "mask" and stem in masks:
            mask_128 = masks[stem]
            # マスクも元cropサイズ→ROIサイズに変換
            mask_full = unpad_from_128(
                np.stack([mask_128] * 3, axis=-1), orig_h, orig_w,
                scale, off_x, off_y
            )[:, :, 0]
            if mask_full.shape != (roi_h, roi_w):
                mask_full = cv2.resize(mask_full, (roi_w, roi_h),
                                       interpolation=cv2.INTER_LINEAR)
            mask_f = mask_full.astype(np.float32) / 255.0
            mask_f = mask_f[:, :, np.newaxis]
            blended = (mask_f * restored_full.astype(np.float32)
                       + (1 - mask_f) * result[y1:y2, x1:x2])
            result[y1:y2, x1:x2] = blended
        else:
            result[y1:y2, x1:x2] = restored_full.astype(np.float32)

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# ページ処理
# ---------------------------------------------------------------------------

def process_page(page_name: str, args, unetpp_model, nafnet_model,
                 device: torch.device) -> bool:
    out_dir = Path(args.output_dir) / page_name
    result_path = out_dir / "restored_page.png"
    if result_path.exists():
        return False  # スキップ

    bbox_json = Path(args.yolo_dir) / page_name / "bboxes.json"
    crops_dir = Path(args.yolo_dir) / page_name / "crops"
    input_img_path = Path(args.input_dir) / page_name / "00_input.png"

    if not bbox_json.exists() or not input_img_path.exists():
        logger.warning("Missing files for: %s", page_name)
        return False

    with open(bbox_json) as f:
        bboxes = json.load(f)

    if len(bboxes) == 0:
        logger.info("No detections for: %s", page_name)
        return False

    page_img = cv2.imread(str(input_img_path))
    if page_img is None:
        logger.warning("Failed to read page image: %s", input_img_path)
        return False

    # crop読み込み & パディング
    padded_imgs = {}
    pad_info = {}
    for i, _ in enumerate(bboxes):
        crop_path = crops_dir / f"crop_{i:04d}.png"
        if not crop_path.exists():
            continue
        crop = cv2.imread(str(crop_path))
        if crop is None or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        stem = f"crop_{i:04d}"
        padded, scale, off_x, off_y, orig_h, orig_w = pad_to_128(crop)
        padded_imgs[stem] = padded
        pad_info[stem] = (scale, off_x, off_y, orig_h, orig_w)

    if not padded_imgs:
        logger.info("No valid crops for: %s", page_name)
        return False

    # マスク生成
    masks = generate_masks(unetpp_model, padded_imgs, device, args.batch_size)

    # 修復
    restored = restore_crops(nafnet_model, padded_imgs, masks, device)

    # 修復済みcropを保存
    crops_out = out_dir / "crops"
    crops_out.mkdir(parents=True, exist_ok=True)
    for stem, img in restored.items():
        info = pad_info[stem]
        scale, off_x, off_y, orig_h, orig_w = info
        crop_restored = unpad_from_128(img, orig_h, orig_w, scale, off_x, off_y)
        cv2.imwrite(str(crops_out / f"{stem}.png"), crop_restored)

    # ページに貼り戻し
    page_restored = paste_to_page(page_img, restored, masks, bboxes,
                                  pad_info, args.blend_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(result_path), page_restored)
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Restoration pipeline: crop → UNet++ → NAFNet → paste")
    parser.add_argument("--unetpp_weights", default="models/unet++/best_model.pth")
    parser.add_argument("--nafnet_config",
                        default="modules/restoration/nafnet/options/Kuzushiji/gtmask.yml")
    parser.add_argument("--nafnet_checkpoint",
                        default="models/nafnet/experiments/"
                                "NAFNet_Kuzushiji_Mask_CharbPercep_archived_20260228_143625/"
                                "models/net_g_200000.pth")
    parser.add_argument("--yolo_dir", default="output/yolo_detection")
    parser.add_argument("--input_dir", default="data/inference_input")
    parser.add_argument("--output_dir", default="output/restoration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--blend_mode", default="mask", choices=["mask", "copy"])
    parser.add_argument("--pages", nargs="*", default=None,
                        help="処理するページ名を指定（省略時は全ページ）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading UNet++ ...")
    unetpp_model = load_unetpp(args.unetpp_weights, device)

    logger.info("Loading NAFNet ...")
    nafnet_model = load_nafnet(args.nafnet_config, args.nafnet_checkpoint,
                               device=str(device))

    if args.pages:
        page_names = args.pages
    else:
        page_names = sorted([
            d for d in os.listdir(args.yolo_dir)
            if os.path.isdir(os.path.join(args.yolo_dir, d))
        ])

    total = len(page_names)
    done = skipped = 0
    for i, page_name in enumerate(page_names):
        result = process_page(page_name, args, unetpp_model, nafnet_model, device)
        if result:
            done += 1
            logger.info("[%d/%d] Done: %s", i + 1, total, page_name)
        else:
            skipped += 1

    logger.info("Finished. Processed: %d, Skipped: %d", done, skipped)


if __name__ == "__main__":
    main()
