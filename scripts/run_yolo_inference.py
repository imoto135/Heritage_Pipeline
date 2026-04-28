#!/usr/bin/env python3
"""全inference_inputフォルダに対してYOLOX推論を実行。処理済みはスキップ。"""

import os
import sys
import json
import argparse
import cv2
import torch
import numpy as np
from loguru import logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from modules.detection.yolox_detector import (
    Predictor, split_image, merge_outputs, image_demo_bunkatu
)


def main():
    parser = argparse.ArgumentParser("YOLOX batch inference with resume")
    parser.add_argument("--input_dir", default="data/inference_input")
    parser.add_argument("--output_dir", default="output/yolo_detection")
    parser.add_argument("-f", "--exp_file", default="modules/detection/nano.py")
    parser.add_argument("-c", "--ckpt", default="best_yolo.pth")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--conf", default=0.35, type=float)
    parser.add_argument("--nms", default=0.45, type=float)
    parser.add_argument("--tsize", default=640, type=int)
    args = parser.parse_args()

    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cuda" if args.device == "gpu" else "cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint: {}", args.ckpt)

    predictor = Predictor(model, exp, COCO_CLASSES, device=args.device)

    folders = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    total = len(folders)
    logger.info("Total folders: {}", total)

    os.makedirs(args.output_dir, exist_ok=True)

    done = 0
    skipped = 0
    for i, folder_name in enumerate(folders):
        input_img = os.path.join(args.input_dir, folder_name, "00_input.png")
        if not os.path.exists(input_img):
            logger.warning("Skipping (no 00_input.png): {}", folder_name)
            continue

        out_folder = os.path.join(args.output_dir, folder_name)
        if os.path.exists(os.path.join(out_folder, "bboxes.json")):
            skipped += 1
            if skipped % 50 == 1:
                logger.info("[{}/{}] Skip (already done): {}", i + 1, total, folder_name)
            continue

        logger.info("[{}/{}] Processing: {}", i + 1, total, folder_name)
        os.makedirs(os.path.join(out_folder, "crops"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "patches"), exist_ok=True)

        img = cv2.imread(input_img)
        if img is None:
            logger.warning("Failed to read image, skipping: {}", input_img)
            continue
        tiles, coords = split_image(img, tile_size=args.tsize, stride=args.tsize // 2)
        outputs_list = []

        for idx, tile in enumerate(tiles):
            outputs, _ = predictor.inference(tile)
            output = outputs[0] if outputs is not None else None
            outputs_list.append(output)

            img_info = {"raw_img": tile, "ratio": 1.0}
            vis_img = predictor.visual(output, img_info, predictor.confthre)
            cv2.imwrite(os.path.join(out_folder, "patches", f"patch_{idx:04d}.jpg"), vis_img)

        merged = merge_outputs(outputs_list, coords, nms_thresh=predictor.nmsthre,
                               original_img_size=img.shape[:2])
        img_info = {"raw_img": img, "ratio": 1.0}
        crops_folder = os.path.join(out_folder, "crops")
        result_image = predictor.visual(merged, img_info, predictor.confthre,
                                        save_folder=crops_folder)
        cv2.imwrite(os.path.join(out_folder, "00_input.png"), result_image)

        # bboxをJSONに保存
        bboxes = []
        if merged is not None:
            for det in merged.cpu().reshape(-1, 7).numpy():
                x1, y1, x2, y2 = map(int, det[:4])
                bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "score": float(det[4] * det[5])})
        with open(os.path.join(out_folder, "bboxes.json"), "w") as f:
            json.dump(bboxes, f)

        done += 1

    logger.info("Done. Processed: {}, Skipped (already done): {}", done, skipped)


if __name__ == "__main__":
    main()
