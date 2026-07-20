"""
学習データにACE補正を適用して split_dataset_csc を生成するスクリプト。

入力:
  train: data/split_dataset/train → data/split_dataset_csc/train
  val:   data/split_dataset/val   → data/split_dataset_csc/val

使い方:
    cd modules/stain_removal
    python scripts/prepare_csc_dataset.py
"""

import os
import sys
import glob
from tqdm import tqdm

parent = os.path.abspath('.')
sys.path.insert(0, parent)

import cv2

_DDRM_SCRIPTS = os.path.join(parent, '..', 'ddrm_npj', 'scripts')
sys.path.insert(0, os.path.abspath(_DDRM_SCRIPTS))
from ace import automatic_color_equalization

SPLITS = [
    ('./data/split_dataset/train', './data/split_dataset_csc/train'),
    ('./data/split_dataset/val',   './data/split_dataset_csc/val'),
]


def process_split(src_dir, dst_dir):
    all_imgs = sorted(glob.glob(os.path.join(src_dir, '**', '*.jpg'), recursive=True))
    print(f'\n[{os.path.basename(dst_dir)}] 対象画像数: {len(all_imgs)}')

    for src_path in tqdm(all_imgs, desc=f'ACE処理中 ({os.path.basename(dst_dir)})'):
        rel = os.path.relpath(src_path, src_dir)
        dst_path = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(dst_path):
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f'読み込み失敗: {src_path}')
            continue

        ace_img = automatic_color_equalization(img, alpha=5.0, n_samples=200, seed=0)
        cv2.imwrite(dst_path, ace_img)

    print(f'[{os.path.basename(dst_dir)}] 完了。')


def main():
    for src_dir, dst_dir in SPLITS:
        process_split(src_dir, dst_dir)


if __name__ == '__main__':
    main()
