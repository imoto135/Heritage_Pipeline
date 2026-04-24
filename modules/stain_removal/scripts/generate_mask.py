"""
文献画像から文字領域マスク（02_text_black.png）を生成するスクリプト。

Otsu二値化 + モルフォロジー処理で文字領域を検出し、
推論スクリプト（main1.py / main_high_inference.py）が期待する形式で保存する。

マスクの定義:
  0（黒） = 文字領域（修復対象、Diffusionでは保持される）
  255（白） = 背景領域（修復対象外 or 自由生成）

使い方:
    # split_dataset/test 以下の全画像に対して生成
    python scripts/generate_mask.py --input_dir ./data/split_dataset/test --output_dir ./data/split_dataset/test_masked

    # 単一フォルダ（00_input.png がある場合）
    python scripts/generate_mask.py --input_dir ./data/split_dataset/test/100241706

設定:
    --input_dir   : 入力画像のルートフォルダ（doc_id/ 以下の .jpg / .png を再帰探索）
    --output_dir  : マスク画像の出力先（省略時は入力と同じ場所に 02_text_black.png を保存）
    --kernel_size : モルフォロジー処理のカーネルサイズ（デフォルト: 3）
    --dilate_iter : 膨張の繰り返し回数（デフォルト: 1、大きくすると文字領域が広がる）
"""

import argparse
import os
import glob
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm


def generate_mask(image_path: str, kernel_size: int = 3, dilate_iter: int = 1) -> np.ndarray:
    """
    1枚の文献画像からOtsu二値化でマスクを生成する。

    Returns:
        mask: uint8, shape (H, W), 0=文字, 255=背景
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu二値化: 暗い画素（文字）→ 0, 明るい画素（背景）→ 255
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 文字領域を少し膨張させて取りこぼしを減らす
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 文字(0)の周囲を広げるため、反転して膨張→反転
    text_region = cv2.bitwise_not(binary)                        # 文字=255
    text_region = cv2.dilate(text_region, kernel, iterations=dilate_iter)
    mask = cv2.bitwise_not(text_region)                          # 文字=0, 背景=255

    return mask


def process_directory(input_dir: str, output_dir: str | None,
                       kernel_size: int, dilate_iter: int) -> None:
    """input_dir 以下の .jpg / .png を再帰的に処理してマスクを保存する。"""
    patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png']
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(input_dir, p), recursive=True)
    # すでに生成済みのマスク自体を除外
    files = [f for f in files if '02_text_black' not in os.path.basename(f)]
    files = natsorted(set(files))

    if len(files) == 0:
        print(f"画像が見つかりませんでした: {input_dir}")
        return

    print(f"対象画像数: {len(files)}")

    for img_path in tqdm(files, desc="マスク生成中"):
        try:
            mask = generate_mask(img_path, kernel_size=kernel_size, dilate_iter=dilate_iter)
        except Exception as e:
            print(f"スキップ ({e}): {img_path}")
            continue

        # 出力先を決定
        if output_dir:
            rel = os.path.relpath(img_path, input_dir)
            rel_dir = os.path.dirname(rel)
            save_dir = os.path.join(output_dir, rel_dir)
        else:
            save_dir = os.path.dirname(img_path)

        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f'{base}_02_text_black.png')
        cv2.imwrite(save_path, mask)

    print("完了。")


def main():
    parser = argparse.ArgumentParser(description="文献画像からマスク（02_text_black.png）を生成")
    parser.add_argument('--input_dir', required=True,
                        help='入力画像フォルダ（再帰的に .jpg/.png を探索）')
    parser.add_argument('--output_dir', default=None,
                        help='マスク出力先フォルダ（省略時は入力と同じ場所）')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='モルフォロジー処理カーネルサイズ（デフォルト: 3）')
    parser.add_argument('--dilate_iter', type=int, default=1,
                        help='膨張の繰り返し回数（デフォルト: 1）')
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        kernel_size=args.kernel_size,
        dilate_iter=args.dilate_iter,
    )


if __name__ == '__main__':
    main()
