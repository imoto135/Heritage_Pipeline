#!/usr/bin/env python3
"""
YOLOXで検出したcropをパディングし、UNet++でダメージマスクを生成する。
出力: output/restoration/<page>/padded/  ← 128×128パディング済みcrop + pad_info.json
     output/restoration/<page>/masks/   ← UNet++ソフトマスク (uint8 PNG)

実行例（1ページのみ）:
  conda run -n unetpp_env python scripts/run_unetpp_masks.py --pages 200003803_00010_1

全ページ一括:
  conda run -n unetpp_env python scripts/run_unetpp_masks.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import segmentation_models_pytorch as smp


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
        img_norm = (img_rgb - _MEAN) / _STD
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
# ページ処理
# ---------------------------------------------------------------------------

def process_page(page_name: str, args, unetpp_model, device: torch.device) -> bool:
    out_dir = Path(args.output_dir) / page_name
    masks_dir = out_dir / "masks"
    padded_dir = out_dir / "padded"

    # スキップ判定: pad_info.json が存在すれば処理済み
    pad_info_path = padded_dir / "pad_info.json"
    if pad_info_path.exists() and masks_dir.exists() and any(masks_dir.iterdir()):
        return False

    crops_dir = Path(args.yolo_dir) / page_name / "crops"
    bbox_json = Path(args.yolo_dir) / page_name / "bboxes.json"

    if not bbox_json.exists():
        logger.warning("Missing bboxes.json for: %s", page_name)
        return False

    with open(bbox_json) as f:
        bboxes = json.load(f)

    if len(bboxes) == 0:
        logger.info("No detections for: %s", page_name)
        return False

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
        pad_info[stem] = {"scale": scale, "off_x": off_x, "off_y": off_y,
                          "orig_h": orig_h, "orig_w": orig_w}

    if not padded_imgs:
        logger.info("No valid crops for: %s", page_name)
        return False

    # パディング済みcrop保存
    padded_dir.mkdir(parents=True, exist_ok=True)
    for stem, img in padded_imgs.items():
        cv2.imwrite(str(padded_dir / f"{stem}.png"), img)
    with open(pad_info_path, "w") as f:
        json.dump(pad_info, f)

    # マスク生成・保存
    masks = generate_masks(unetpp_model, padded_imgs, device, args.batch_size)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for stem, mask in masks.items():
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)

    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("UNet++ mask generation for restoration pipeline")
    parser.add_argument("--unetpp_weights", default="models/unet++/best_model.pth")
    parser.add_argument("--yolo_dir", default="output/yolo_detection")
    parser.add_argument("--output_dir", default="output/restoration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pages", nargs="*", default=None,
                        help="処理するページ名を指定（省略時は全ページ）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading UNet++ from %s ...", args.unetpp_weights)
    unetpp_model = load_unetpp(args.unetpp_weights, device)

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
        result = process_page(page_name, args, unetpp_model, device)
        if result:
            done += 1
            logger.info("[%d/%d] Done: %s", i + 1, total, page_name)
        else:
            skipped += 1

    logger.info("Finished. Processed: %d, Skipped: %d", done, skipped)


if __name__ == "__main__":
    main()
