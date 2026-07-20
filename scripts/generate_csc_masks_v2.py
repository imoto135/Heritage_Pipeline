"""
改良版CSC（ddrm_npj_v2）でマスクを生成する。

現行版との違い（ddrm_npj_v2/scripts/csc.py）:
  Stage 1: GMM 3クラス → 2クラス（背景/非背景）
  Stage 2: GMM 2クラス → 3クラス（黒文字/赤文字/実劣化、RGB+Lab+Luv 9ch特徴）

ACE済み画像 (split_dataset_csc) は現行版と共有し、再生成しない。
出力先を split_dataset_csc_mask_v2 に分離して現行マスクとA/B比較できるようにする。
各ページの出力ディレクトリには、ACE済み画像(00_input.png)に加えてACE前の
元画像(00_original.png)も配置する。

使い方:
    cd /home/imoto/Heritage_Pipeline
    conda run -n diffusion python scripts/generate_csc_masks_v2.py --split test --workers 32
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from natsort import natsorted
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 改良版CSC（v2）をインポート
sys.path.insert(0, os.path.join(REPO_ROOT, 'modules', 'ddrm_npj_v2', 'scripts'))
from ace import automatic_color_equalization
from csc import color_space_clustering

SPLIT_DATASET     = os.path.join(REPO_ROOT, 'data', 'split_dataset')
SPLIT_DATASET_CSC = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc')
OUTPUT_BASE       = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc_mask_v2')
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


def _process_one(args: tuple) -> str | None:
    """1ページ分のCSC処理（別プロセスで実行される）。失敗時はエラー文字列を返す。"""
    split, book, img_path, out_dir = args

    if os.path.exists(os.path.join(out_dir, '02_text_black.png')):
        return None  # 冪等性

    ace_img = cv2.imread(img_path)
    if ace_img is None:
        return f'読み込み失敗: {img_path}'

    try:
        result = color_space_clustering(ace_img)
    except Exception as e:
        return f'CSC失敗 ({e}): {img_path}'

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, '00_input.png'),       ace_img)
    cv2.imwrite(os.path.join(out_dir, '02_text_black.png'),  255 - result.black_text_mask)
    cv2.imwrite(os.path.join(out_dir, '03_text_red.png'),    255 - result.red_text_mask)
    cv2.imwrite(os.path.join(out_dir, '04_degradation.png'), result.degradation_mask)
    cv2.imwrite(os.path.join(out_dir, '05_background.png'),  result.background_mask)

    # ACE前の元画像もそのまま配置する（色補正なしの原本を参照できるように）
    stem = os.path.splitext(os.path.basename(img_path))[0]
    raw_path = os.path.join(SPLIT_DATASET, split, book, stem + '.jpg')
    raw_img = cv2.imread(raw_path)
    if raw_img is not None:
        cv2.imwrite(os.path.join(out_dir, '00_original.png'), raw_img)
    else:
        return f'元画像が見つかりません: {raw_path}'

    return None


def process_split(split: str, workers: int = 1) -> None:
    """split_dataset_csc/<split> の ACE済み画像に v2 CSC をかけてマスクを保存する。"""
    in_split  = os.path.join(SPLIT_DATASET_CSC, split)
    out_split = os.path.join(OUTPUT_BASE, split)

    book_dirs = natsorted([
        d for d in os.listdir(in_split)
        if os.path.isdir(os.path.join(in_split, d))
    ])

    tasks = []
    for book in book_dirs:
        imgs = natsorted(
            glob.glob(os.path.join(in_split, book, '*.jpg')) +
            glob.glob(os.path.join(in_split, book, '*.jpeg')) +
            glob.glob(os.path.join(in_split, book, '*.png'))
        )
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = os.path.join(out_split, book, stem)
            tasks.append((split, book, img_path, out_dir))

    print(f'\n[CSC v2] {split}: {len(book_dirs)}冊 / {len(tasks)}枚 (workers={workers})')

    errors = []
    if workers <= 1:
        for t in tqdm(tasks, desc=f'CSC v2 ({split})'):
            err = _process_one(t)
            if err:
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f'CSC v2 ({split})'):
                err = fut.result()
                if err:
                    errors.append(err)

    for e in errors:
        tqdm.write(e)
    if errors:
        print(f'  エラー: {len(errors)}件')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=SPLITS, default=None,
                        help='処理するsplitを指定（省略時は全split）')
    parser.add_argument('--workers', type=int, default=1,
                        help='CSC計算の並列プロセス数（CPUバウンドなのでコア数まで有効）')
    args = parser.parse_args()

    targets = [args.split] if args.split else SPLITS
    for split in targets:
        ensure_ace_split(split)
        process_split(split, workers=args.workers)

    print('\n完了。')


if __name__ == '__main__':
    main()
