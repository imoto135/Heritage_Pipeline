"""
ACE + CSC（Color Space Clustering）でnpj論文の4クラスマスクを生成するスクリプト。

stain_removal/scripts/generate_mask.py と同じマスク値規約に合わせる:
  0（黒） = 文字領域
  255（白） = 背景・対象外領域

main1.py / main_high_inference.py が期待するファイル名で保存する:
  02_text_black.png : 黒文字マスク（DDRMのノイズマスクとして使用）
  03_text_red.png   : 赤文字マスク（DDRM後にオーバーレイするため使用）

使い方:
    python scripts/generate_mask_csc.py --input_dir ../../data/train/200021925 \
        --output_dir ./data/csc_output/200021925
"""

import argparse
import os
import glob
import sys

import cv2
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from ace import automatic_color_equalization
from csc import color_space_clustering


def process_image(img_path: str, output_dir: str, save_ace: bool) -> None:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {img_path}")

    ace_img = automatic_color_equalization(img)
    result = color_space_clustering(ace_img)

    # 黒文字マスク: 255=文字 を 0=文字 の規約に反転
    black_text_inv = 255 - result.black_text_mask
    red_text_inv = 255 - result.red_text_mask

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, '00_input.png'), img)
    cv2.imwrite(os.path.join(output_dir, '02_text_black.png'), black_text_inv)
    cv2.imwrite(os.path.join(output_dir, '03_text_red.png'), red_text_inv)
    cv2.imwrite(os.path.join(output_dir, '04_degradation.png'), result.degradation_mask)
    cv2.imwrite(os.path.join(output_dir, '05_background.png'), result.background_mask)
    if save_ace:
        cv2.imwrite(os.path.join(output_dir, '01_ace.png'), ace_img)


def process_directory(input_dir: str, output_dir: str, save_ace: bool) -> None:
    patterns = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(input_dir, p))
    files = natsorted(set(files))

    if not files:
        print(f"画像が見つかりませんでした: {input_dir}")
        return

    print(f"対象画像数: {len(files)}")
    for img_path in tqdm(files, desc="ACE+CSC マスク生成中"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(output_dir, base)
        try:
            process_image(img_path, save_dir, save_ace)
        except Exception as e:
            tqdm.write(f"スキップ ({e}): {img_path}")

    print("完了。")


def main():
    parser = argparse.ArgumentParser(description="ACE+CSCでnpj論文の4クラスマスクを生成")
    parser.add_argument('--input_dir', required=True, help='入力画像フォルダ')
    parser.add_argument('--output_dir', required=True, help='マスク出力先フォルダ')
    parser.add_argument('--save_ace', action='store_true', help='ACE補正済み画像も保存する')
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.save_ace)


if __name__ == '__main__':
    main()
