#!/usr/bin/env python3
"""
coordinate.csv + 元ページ画像 → タイル分割 + COCOフォーマットJSON生成

出力:
  <output_dir>/images/   ... 640×640タイル画像
  <output_dir>/annotations/instances_<split>.json  ... COCOフォーマット
"""
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

TILE_SIZE = 640
STRIDE = 320  # 50%オーバーラップ（既存instances_train2017.jsonと同じ）
MIN_BBOX_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True,
                        help="data/split_dataset/<split>/ のパス")
    parser.add_argument("--full-dir", required=True,
                        help="data/full_dataset/ のパス（coordinate.csv置き場）")
    parser.add_argument("--output-dir", required=True,
                        help="出力先ディレクトリ")
    parser.add_argument("--split-name", default="train",
                        help="JSONのsplit名 (train/val/test)")
    return parser.parse_args()


def load_csv(csv_path: Path) -> dict:
    """image名 → [{x,y,w,h}, ...] のdict"""
    data = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["Image"].strip()
            x = float(row["X"])
            y = float(row["Y"])
            w = float(row["Width"])
            h = float(row["Height"])
            data[img_name].append((x, y, w, h))
    return data


def tile_image(img: np.ndarray, tile_size: int, stride: int):
    """画像をタイルに分割。(tile, x0, y0) のリストを返す"""
    H, W = img.shape[:2]
    tiles = []
    y = 0
    while True:
        x = 0
        while True:
            x1 = min(x + tile_size, W)
            y1 = min(y + tile_size, H)
            tile = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
            crop = img[y:y1, x:x1]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            tiles.append((tile, x, y))
            if x1 == W:
                break
            x += stride
        if y1 == H:
            break
        y += stride
    return tiles


def clip_bbox(bx, by, bw, bh, tx, ty, tile_size):
    """bboxをタイル座標系にクリップ。タイル内に無ければNone"""
    # タイル領域: [tx, tx+tile_size) x [ty, ty+tile_size)
    # bboxの絶対座標
    x1 = bx
    y1 = by
    x2 = bx + bw
    y2 = by + bh

    # タイル内にクリップ
    cx1 = max(x1, tx) - tx
    cy1 = max(y1, ty) - ty
    cx2 = min(x2, tx + tile_size) - tx
    cy2 = min(y2, ty + tile_size) - ty

    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw < MIN_BBOX_SIZE or ch < MIN_BBOX_SIZE:
        return None
    # bboxの中心がタイル内にあるもののみ採用
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if not (tx <= cx < tx + tile_size and ty <= cy < ty + tile_size):
        return None
    return (cx1, cy1, cw, ch)


def main():
    args = parse_args()
    split_dir = Path(args.split_dir)
    full_dir = Path(args.full_dir)
    out_dir = Path(args.output_dir)
    img_out_dir = out_dir / "images"
    ann_out_dir = out_dir / "annotations"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    ann_out_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "Kuzushiji YOLOX dataset"},
        "categories": [{"supercategory": "labels", "id": 1, "name": "labels"}],
        "images": [],
        "annotations": [],
    }
    image_id = 1
    ann_id = 1

    book_dirs = sorted(split_dir.iterdir())
    for book_dir in tqdm(book_dirs, desc="books"):
        if not book_dir.is_dir():
            continue
        book_id = book_dir.name

        csv_path = full_dir / book_id / f"{book_id}_coordinate.csv"
        if not csv_path.exists():
            print(f"[警告] CSVなし: {csv_path}")
            continue

        bboxes_by_page = load_csv(csv_path)

        for img_path in sorted(book_dir.glob("*.jpg")):
            # ページ名: 200003803_00002_2.jpg → 200003803_00002_2
            page_name = img_path.stem

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            page_bboxes = bboxes_by_page.get(page_name, [])

            tiles = tile_image(img, TILE_SIZE, STRIDE)
            for tile_idx, (tile, tx, ty) in enumerate(tiles):
                # 既存JSONの命名規則: {book}_{page}_{tile_idx:03d}.jpg
                tile_fname = f"{book_id}_{page_name}_{tile_idx:03d}.jpg"
                cv2.imwrite(str(img_out_dir / tile_fname), tile)

                coco["images"].append({
                    "id": image_id,
                    "file_name": tile_fname,
                    "width": TILE_SIZE,
                    "height": TILE_SIZE,
                    "date_captured": "2025",
                })

                for bx, by, bw, bh in page_bboxes:
                    clipped = clip_bbox(bx, by, bw, bh, tx, ty, TILE_SIZE)
                    if clipped is None:
                        continue
                    cx1, cy1, cw, ch = clipped
                    area = cw * ch
                    seg = [cx1, cy1, cx1 + cw, cy1, cx1 + cw, cy1 + ch, cx1, cy1 + ch]
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [cx1, cy1, cw, ch],
                        "area": area,
                        "segmentation": [seg],
                        "iscrowd": 0,
                    })
                    ann_id += 1

                image_id += 1

    out_json = ann_out_dir / f"instances_{args.split_name}2017.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\n完了: images={image_id-1}, annotations={ann_id-1}")
    print(f"  JSON: {out_json}")
    print(f"  画像: {img_out_dir}/")


if __name__ == "__main__":
    main()
