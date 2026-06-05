#!/usr/bin/env python3
"""
UNet++で生成したマスクをもとにNAFNetで修復し、元ページ画像に貼り戻す。
run_unetpp_masks.py の出力（output/restoration/<page>/padded/, masks/）が前提。

実行例（1ページのみ）:
  conda run -n nafnet2 python scripts/run_nafnet_restore.py --pages 200003803_00010_1

全ページ一括:
  conda run -n nafnet2 python scripts/run_nafnet_restore.py
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
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules", "restoration", "nafnet"))

from infer_withmask import load_model as load_nafnet, process_image


# ---------------------------------------------------------------------------
# アンパディング
# ---------------------------------------------------------------------------

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
# NAFNet 修復
# ---------------------------------------------------------------------------

def restore_crops(nafnet_model, padded_dir: Path, masks_dir: Path,
                  pad_info: dict, device: torch.device) -> dict:
    """NAFNetで修復。128×128入力 → 128×128出力。"""
    restored = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for stem in pad_info:
            lq_path = padded_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}.png"
            if not lq_path.exists() or not mask_path.exists():
                continue

            tmp_lq = tmpdir / f"{stem}_lq.png"
            tmp_mask = tmpdir / f"{stem}_mask.png"
            import shutil
            shutil.copy(lq_path, tmp_lq)
            shutil.copy(mask_path, tmp_mask)

            result = process_image(nafnet_model, tmp_lq, tmp_mask, device=str(device))
            if result is not None:
                restored[stem] = result
    return restored


# ---------------------------------------------------------------------------
# 貼り戻し
# ---------------------------------------------------------------------------

def paste_to_page(page_img: np.ndarray, restored_crops: dict,
                  masks_dir: Path, bboxes: list, pad_info: dict,
                  blend_mode: str = "mask") -> np.ndarray:
    """修復済みcropをページ画像の対応するbbox位置に貼り戻す。"""
    result = page_img.astype(np.float32)

    for i, bbox in enumerate(bboxes):
        stem = f"crop_{i:04d}"
        if stem not in restored_crops or stem not in pad_info:
            continue

        info = pad_info[stem]
        scale = info["scale"]
        off_x = info["off_x"]
        off_y = info["off_y"]
        orig_h = info["orig_h"]
        orig_w = info["orig_w"]

        restored_full = unpad_from_128(restored_crops[stem], orig_h, orig_w,
                                       scale, off_x, off_y)

        x1 = max(0, bbox["x1"])
        y1 = max(0, bbox["y1"])
        x2 = min(page_img.shape[1], bbox["x2"])
        y2 = min(page_img.shape[0], bbox["y2"])

        roi_h, roi_w = y2 - y1, x2 - x1
        if roi_h <= 0 or roi_w <= 0:
            continue

        if restored_full.shape[:2] != (roi_h, roi_w):
            restored_full = cv2.resize(restored_full, (roi_w, roi_h),
                                       interpolation=cv2.INTER_CUBIC)

        if blend_mode == "mask":
            mask_path = masks_dir / f"{stem}.png"
            if mask_path.exists():
                mask_128 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_128 is not None:
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
                    continue

        result[y1:y2, x1:x2] = restored_full.astype(np.float32)

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# ページ処理
# ---------------------------------------------------------------------------

def process_page(page_name: str, args, nafnet_model, device: torch.device) -> bool:
    rest_dir = Path(args.output_dir) / page_name
    result_path = rest_dir / "restored_page.png"
    if result_path.exists():
        return False

    padded_dir = rest_dir / "padded"
    masks_dir = rest_dir / "masks"
    pad_info_path = padded_dir / "pad_info.json"

    if not pad_info_path.exists():
        logger.warning("Missing pad_info.json (run run_unetpp_masks.py first): %s", page_name)
        return False

    with open(pad_info_path) as f:
        pad_info = json.load(f)

    bbox_json = Path(args.yolo_dir) / page_name / "bboxes.json"
    input_img_path = Path(args.input_dir) / page_name / "00_input.png"

    if not bbox_json.exists() or not input_img_path.exists():
        logger.warning("Missing files for: %s", page_name)
        return False

    with open(bbox_json) as f:
        bboxes = json.load(f)

    page_img = cv2.imread(str(input_img_path))
    if page_img is None:
        logger.warning("Failed to read page image: %s", input_img_path)
        return False

    # NAFNet修復
    restored = restore_crops(nafnet_model, padded_dir, masks_dir, pad_info, device)

    if not restored:
        logger.info("No crops restored for: %s", page_name)
        return False

    # 修復済みcropを保存
    crops_out = rest_dir / "crops"
    crops_out.mkdir(parents=True, exist_ok=True)
    for stem, img in restored.items():
        info = pad_info[stem]
        crop_restored = unpad_from_128(img, info["orig_h"], info["orig_w"],
                                       info["scale"], info["off_x"], info["off_y"])
        cv2.imwrite(str(crops_out / f"{stem}.png"), crop_restored)

    # ページに貼り戻し
    page_restored = paste_to_page(page_img, restored, masks_dir, bboxes,
                                  pad_info, args.blend_mode)
    rest_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(result_path), page_restored)
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("NAFNet restoration + paste-back pipeline")
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
    parser.add_argument("--blend_mode", default="mask", choices=["mask", "copy"])
    parser.add_argument("--pages", nargs="*", default=None,
                        help="処理するページ名を指定（省略時は全ページ）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading NAFNet ...")
    nafnet_model = load_nafnet(args.nafnet_config, args.nafnet_checkpoint,
                               device=str(device))

    if args.pages:
        page_names = args.pages
    else:
        page_names = sorted([
            d for d in os.listdir(args.output_dir)
            if os.path.isdir(os.path.join(args.output_dir, d))
        ])

    total = len(page_names)
    done = skipped = 0
    for i, page_name in enumerate(page_names):
        result = process_page(page_name, args, nafnet_model, device)
        if result:
            done += 1
            logger.info("[%d/%d] Done: %s", i + 1, total, page_name)
        else:
            skipped += 1

    logger.info("Finished. Processed: %d, Skipped: %d", done, skipped)


if __name__ == "__main__":
    main()
