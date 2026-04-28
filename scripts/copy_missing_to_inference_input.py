#!/usr/bin/env python3
"""split_dataset/testにあってinference_inputにないページをコピー。"""

import os
import sys
import shutil
import cv2
from loguru import logger

SPLIT_TEST = "data/split_dataset/test"
INFER_INPUT = "data/inference_input"

os.makedirs(INFER_INPUT, exist_ok=True)

copied = 0
skipped = 0

for doc_id in sorted(os.listdir(SPLIT_TEST)):
    doc_dir = os.path.join(SPLIT_TEST, doc_id)
    if not os.path.isdir(doc_dir):
        continue
    for fname in sorted(os.listdir(doc_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        out_folder = os.path.join(INFER_INPUT, stem)
        out_img = os.path.join(out_folder, "00_input.png")
        if os.path.exists(out_img):
            skipped += 1
            continue
        os.makedirs(out_folder, exist_ok=True)
        src = os.path.join(doc_dir, fname)
        if ext.lower() == ".png":
            shutil.copy2(src, out_img)
        else:
            img = cv2.imread(src)
            if img is None:
                logger.warning("Failed to read: {}", src)
                continue
            cv2.imwrite(out_img, img)
        copied += 1
        if copied % 50 == 1:
            logger.info("Copied {}: {}", copied, stem)

logger.info("Done. Copied: {}, Skipped (already exists): {}", copied, skipped)
