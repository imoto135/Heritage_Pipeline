"""
本文（くずし字主体）ページと、表紙・挿絵ページを自動判別するスクリプト。

判定方法:
  1. Otsu二値化で文字候補マスクを作る（黒=文字候補）
  2. 連結成分分析を行い、以下の特徴で「本文らしさ」を判定する
     - 黒画素比率: 一定範囲内（低すぎ=無地の表紙、高すぎ=濃い挿絵/汚損）
     - 連結成分の平均サイズ: 文字は小さい連結成分が多数、挿絵は大きい塊が少数
     - 連結成分数: 文字ページは数十〜数百個、表紙はほぼ0、挿絵は少数の大きい塊

出力: 判定結果をCSVに書き出す（コピーは行わない。--apply 指定時のみ実コピー）。

使い方:
    # ドライラン（判定結果をCSV出力するだけ）
    python scripts/classify_text_pages.py --input_dir data/train --output_csv tmp_work/page_classification.csv

    # 判定結果に基づいて本文ページのみ data/split_dataset_jogai にコピー
    python scripts/classify_text_pages.py --input_dir data/train --output_csv tmp_work/page_classification.csv \
        --apply --output_dir data/split_dataset_jogai
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm


def analyze_page(img_path: str,
                  min_black_ratio: float = 0.02,
                  max_black_ratio: float = 0.35,
                  min_components: int = 30,
                  max_mean_component_area_ratio: float = 0.0008) -> dict:
    """
    1ページを解析し、本文らしさの指標と判定結果を返す。

    Args:
        min_black_ratio: 黒画素比率がこれ未満なら無地（表紙等）と判定
        max_black_ratio: 黒画素比率がこれを超えると濃い挿絵/重度汚損と判定
        min_components: 文字候補の連結成分数がこれ未満なら本文ではないと判定
        max_mean_component_area_ratio: 連結成分の平均面積(画像全体に対する比率)が
            これを超えると「文字よりも大きい塊」=挿絵の線画/塗りと判定
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"path": img_path, "error": "read_failed", "is_text_page": False}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_px = h * w

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_candidate = cv2.bitwise_not(binary)  # 暗い画素=255(文字候補)

    black_ratio = float(np.count_nonzero(text_candidate)) / total_px

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_candidate, connectivity=8
    )
    # ラベル0は背景なので除く
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    num_components = len(areas)
    mean_component_area_ratio = float(areas.mean() / total_px) if num_components > 0 else 0.0

    is_text_page = (
        min_black_ratio <= black_ratio <= max_black_ratio
        and num_components >= min_components
        and mean_component_area_ratio <= max_mean_component_area_ratio
    )

    return {
        "path": img_path,
        "black_ratio": round(black_ratio, 4),
        "num_components": num_components,
        "mean_component_area_ratio": round(mean_component_area_ratio, 6),
        "is_text_page": is_text_page,
    }


def main():
    parser = argparse.ArgumentParser(description="本文ページと表紙/挿絵ページを自動判別する")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--apply", action="store_true", help="判定結果に基づき本文ページを実際にコピーする")
    parser.add_argument("--output_dir", default=None, help="--apply 時のコピー先ルート")
    args = parser.parse_args()

    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(args.input_dir, p), recursive=True)
    files = natsorted(set(files))

    if not files:
        print(f"画像が見つかりませんでした: {args.input_dir}")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results = []
    for f in tqdm(files, desc="ページ判定中"):
        results.append(analyze_page(f))

    with open(args.output_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "path", "black_ratio", "num_components",
            "mean_component_area_ratio", "is_text_page", "error",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    n_text = sum(1 for r in results if r.get("is_text_page"))
    n_total = len(results)
    print(f"本文ページ判定: {n_text}/{n_total} 枚 ({n_text/n_total*100:.1f}%)")
    print(f"判定結果を保存: {args.output_csv}")

    if args.apply:
        if not args.output_dir:
            raise ValueError("--apply 指定時は --output_dir が必要です")
        import shutil
        for r in results:
            if not r.get("is_text_page"):
                continue
            src = r["path"]
            rel = os.path.relpath(src, args.input_dir)
            dst = os.path.join(args.output_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"本文ページを {args.output_dir} にコピーしました。")


if __name__ == "__main__":
    main()
