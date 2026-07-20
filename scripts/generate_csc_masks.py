"""
split_dataset_csc の ACE済み画像に CSC を適用してマスクを生成する。

Step 1: split_dataset_csc/test がなければ split_dataset/test に ACE をかけて生成
Step 2: split_dataset_csc/{train,val,test} の各画像に CSC を適用

出力先: data/split_dataset_csc_mask/
  <split>/
    <book_id>/
      <image_stem>/
        00_input.png       # ACE済み入力画像
        02_text_black.png  # 黒文字マスク（0=文字）
        03_text_red.png    # 赤文字マスク（0=文字）
        04_degradation.png
        05_background.png

使い方:
    cd /home/imoto/Heritage_Pipeline
    conda run -n diffusion python scripts/generate_csc_masks.py
    conda run -n diffusion python scripts/generate_csc_masks.py --split train
"""

import argparse
import glob
import os
import sys

import cv2
from natsort import natsorted
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, 'modules', 'ddrm_npj', 'scripts'))
from ace import automatic_color_equalization
from csc import color_space_clustering

SPLIT_DATASET     = os.path.join(REPO_ROOT, 'data', 'split_dataset')
SPLIT_DATASET_CSC = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc')
OUTPUT_BASE       = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc_mask')
SPLITS            = ['train', 'val', 'test']


def ensure_ace_split(split: str) -> None:
    """split_dataset/<split> の全冊を split_dataset_csc/<split> に ACE して生成する。"""
    src_dir = os.path.join(SPLIT_DATASET,     split)
    dst_dir = os.path.join(SPLIT_DATASET_CSC, split)

    all_imgs = natsorted(glob.glob(os.path.join(src_dir, '**', '*.jpg'), recursive=True))
    print(f'\n[ACE] {split}: {len(all_imgs)}枚 → {dst_dir}')

    for src_path in tqdm(all_imgs, desc=f'ACE ({split})'):
        rel      = os.path.relpath(src_path, src_dir)
        dst_path = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(dst_path):
            continue
        img = cv2.imread(src_path)
        if img is None:
            tqdm.write(f'読み込み失敗: {src_path}')
            continue
        ace_img = automatic_color_equalization(img, alpha=5.0, n_samples=200, seed=0)
        cv2.imwrite(dst_path, ace_img)


def process_split(split: str) -> None:
    """split_dataset_csc/<split> の ACE済み画像に CSC をかけてマスクを保存する。"""
    in_split  = os.path.join(SPLIT_DATASET_CSC, split)
    out_split = os.path.join(OUTPUT_BASE, split)

    book_dirs = natsorted([
        d for d in os.listdir(in_split)
        if os.path.isdir(os.path.join(in_split, d))
    ])

    all_imgs = []
    for book in book_dirs:
        imgs = natsorted(
            glob.glob(os.path.join(in_split, book, '*.jpg')) +
            glob.glob(os.path.join(in_split, book, '*.jpeg')) +
            glob.glob(os.path.join(in_split, book, '*.png'))
        )
        all_imgs.extend((book, img) for img in imgs)

    print(f'\n[CSC] {split}: {len(book_dirs)}冊 / {len(all_imgs)}枚')

    for book, img_path in tqdm(all_imgs, desc=f'CSC ({split})'):
        stem    = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(out_split, book, stem)

        if os.path.exists(os.path.join(out_dir, '02_text_black.png')):
            continue  # 冪等性

        ace_img = cv2.imread(img_path)
        if ace_img is None:
            tqdm.write(f'読み込み失敗: {img_path}')
            continue

        try:
            result = color_space_clustering(ace_img)
        except Exception as e:
            tqdm.write(f'CSC失敗 ({e}): {img_path}')
            continue

        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, '00_input.png'),       ace_img)
        cv2.imwrite(os.path.join(out_dir, '02_text_black.png'),  255 - result.black_text_mask)
        cv2.imwrite(os.path.join(out_dir, '03_text_red.png'),    255 - result.red_text_mask)
        cv2.imwrite(os.path.join(out_dir, '04_degradation.png'), result.degradation_mask)
        cv2.imwrite(os.path.join(out_dir, '05_background.png'),  result.background_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=SPLITS, default=None,
                        help='処理するsplitを指定（省略時は全split）')
    args = parser.parse_args()

    targets = [args.split] if args.split else SPLITS
    for split in targets:
        ensure_ace_split(split)
        process_split(split)

    print('\n完了。')


if __name__ == '__main__':
    main()
