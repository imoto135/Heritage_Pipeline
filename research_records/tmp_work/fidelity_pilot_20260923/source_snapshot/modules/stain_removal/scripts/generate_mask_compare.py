"""
3条件（大津法 / Sauvola法×2）でマスクを比較生成するスクリプト。

条件A: 大津法（現行）
条件B: Sauvola法 window_size=25, k=0.2
条件C: Sauvola法 window_size=51, k=0.1

マスクの定義:
  0（黒） = 文字領域
  255（白） = 背景領域

使い方:
    python scripts/generate_mask_compare.py --input_dir ./data/train

出力先:
    data/masks_compare/<condition>/<doc_id>/<basename>_mask.png
"""

import argparse
import os
import glob
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from natsort import natsorted
from tqdm import tqdm


CONDITIONS = {
    "A_otsu": {"method": "otsu"},
    "B_sauvola_w25_k02": {"method": "sauvola", "window_size": 25, "k": 0.2},
    "C_sauvola_w51_k01": {"method": "sauvola", "window_size": 51, "k": 0.1},
}


def generate_mask_otsu(gray: np.ndarray, kernel_size: int = 3, dilate_iter: int = 1) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    text_region = cv2.bitwise_not(binary)
    text_region = cv2.dilate(text_region, kernel, iterations=dilate_iter)
    return cv2.bitwise_not(text_region)


def generate_mask_sauvola(gray: np.ndarray, window_size: int, k: float,
                           kernel_size: int = 3, dilate_iter: int = 1) -> np.ndarray:
    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray >= thresh).astype(np.uint8) * 255  # 明るい=背景=255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    text_region = cv2.bitwise_not(binary)
    text_region = cv2.dilate(text_region, kernel, iterations=dilate_iter)
    return cv2.bitwise_not(text_region)


def process_image(img_path: str, cond_params: dict, kernel_size: int, dilate_iter: int) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"読み込み失敗: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cond_params["method"] == "otsu":
        return generate_mask_otsu(gray, kernel_size, dilate_iter)
    else:
        return generate_mask_sauvola(gray, cond_params["window_size"], cond_params["k"],
                                     kernel_size, dilate_iter)


def main():
    parser = argparse.ArgumentParser(description="3条件でマスクを比較生成")
    parser.add_argument("--input_dir", required=True, help="入力画像フォルダ")
    parser.add_argument("--output_dir", default=None,
                        help="出力先ルート（デフォルト: <input_dir>/../masks_compare）")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dilate_iter", type=int, default=1)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        output_root = os.path.join(os.path.dirname(input_dir), "masks_compare")

    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(input_dir, p), recursive=True)
    files = [f for f in files if "mask" not in os.path.basename(f).lower()]
    files = natsorted(set(files))

    if not files:
        print(f"画像が見つかりませんでした: {input_dir}")
        return

    print(f"対象画像数: {len(files)}  出力先: {output_root}")

    for img_path in tqdm(files, desc="マスク生成中"):
        rel = os.path.relpath(img_path, input_dir)
        base = os.path.splitext(os.path.basename(img_path))[0]

        for cond_name, cond_params in CONDITIONS.items():
            try:
                mask = process_image(img_path, cond_params, args.kernel_size, args.dilate_iter)
            except Exception as e:
                tqdm.write(f"スキップ [{cond_name}] ({e}): {img_path}")
                continue

            save_dir = os.path.join(output_root, cond_name, os.path.dirname(rel))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{base}_mask.png")
            cv2.imwrite(save_path, mask)

    print(f"\n完了。マスクを {output_root} に保存しました。")
    for cond_name in CONDITIONS:
        cond_dir = os.path.join(output_root, cond_name)
        count = len(glob.glob(os.path.join(cond_dir, "**/*.png"), recursive=True))
        print(f"  {cond_name}: {count} 枚")


if __name__ == "__main__":
    main()
