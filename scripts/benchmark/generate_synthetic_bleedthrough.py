"""
裏写り(文字の裏移り)の合成データ生成。

[[project_bleedthrough_verification]]で実証済みの物理モデル
（`200003803`で確認: side1のターゲットは前ページ(offset-1)のside2、
side2のターゲットは次ページ(offset+1)のside1が発生源）を使い、
隣接丁の実際の文字マスクを反転(鏡像)・減光・ぼかしてクリーンページに
合成することで、(clean, degraded, ghost_gt)の学習・検証用ペアを作る。

単一画像だけで動く裏写り検出器/除去器を学習するための教師データ。
擬似シミ生成(a.py/b.py、scripts/benchmark/generate_makura_benchmark.py)と
同じ「劣化モデルで学習データを合成する」という発想（文献調査で確認した
CDSR-CycleGANや劣化モデルベース浅いNNの手法と同じ枠組み）。

減光量(alpha)・ぼかし量(blur_sigma)のデフォルトは、確認済み実例
(200003803_00002_2、隣接丁page3_1との一致)の実測値
（ゴースト平均gray=175.5、背景平均gray=191.4、std=14.1）に合わせて
キャリブレーションしている。

使い方:
    conda run -n diffusion python scripts/benchmark/generate_synthetic_bleedthrough.py \
        --book 200003803 --n_pages 10 --out_dir tmp_work/synthetic_bleedthrough
"""

import argparse
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_ROOT = os.path.join(REPO_ROOT, 'data', 'split_dataset')
MASK_ROOT = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc_mask_v2')


def find_split(book: str) -> str:
    for split in ('test', 'val', 'train'):
        if os.path.isdir(os.path.join(MASK_ROOT, split, book)):
            return split
    raise FileNotFoundError(f'マスクが見つかりません: {book}')


def list_pages(book: str, split: str):
    d = os.path.join(MASK_ROOT, split, book)
    pages = []
    for name in os.listdir(d):
        parts = name.split('_')
        pages.append((int(parts[-2]), int(parts[-1])))
    return sorted(pages)


def adjacent_source(page: int, side: int):
    """[[project_bleedthrough_verification]]で確認済みの隣接丁ルール。
    side1 <- 前ページのside2 (offset-1) / side2 <- 次ページのside1 (offset+1)。"""
    if side == 1:
        return page - 1, 2
    else:
        return page + 1, 1


def load_text_mask(book: str, split: str, page: int, side: int):
    stem = f'{book}_{page:05d}_{side}'
    p = os.path.join(MASK_ROOT, split, book, stem, '02_text_black.png')
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return m < 127  # 0=文字(黒) なので反転


def load_clean_image(book: str, split: str, page: int, side: int):
    p = os.path.join(SPLIT_ROOT, split, book, f'{book}_{page:05d}_{side}.jpg')
    return cv2.imread(p)


def make_synthetic_ghost(target_shape, source_text_mask: np.ndarray,
                         alpha: float = 0.13, blur_sigma: float = 2.5,
                         edge_margin_ratio: float = 0.03):
    """
    source_text_mask(bool, 文字=True)を左右反転して target_shape にリサイズし、
    ぼかした「裏写り強度マップ」(0..1、1に近いほど濃い)を返す。
    紙をめくって裏から見ると鏡像になるため左右反転する。
    """
    H, W = target_shape
    flipped = np.fliplr(source_text_mask.astype(np.float32))
    flipped = cv2.resize(flipped, (W, H), interpolation=cv2.INTER_AREA)

    # ページ端(スキャン端・綴じ側)は実際の裏写りが物理的に生じにくい/検証もできないため除外
    my, mx = int(H * edge_margin_ratio), int(W * edge_margin_ratio)
    edge_mask = np.zeros((H, W), dtype=np.float32)
    edge_mask[my:H - my, mx:W - mx] = 1.0
    flipped *= edge_mask

    ghost_strength = cv2.GaussianBlur(flipped, (0, 0), sigmaX=blur_sigma)
    ghost_strength = np.clip(ghost_strength, 0, 1) * alpha
    return ghost_strength


def apply_synthetic_ghost(clean_img: np.ndarray, ghost_strength: np.ndarray):
    """ghost_strength(0..1、アルファブレンド率)だけ黒(0)へブレンドして裏写りを合成する。"""
    img = clean_img.astype(np.float32)
    a = ghost_strength[:, :, None]
    result = img * (1 - a)
    result = np.clip(result, 0, 255).astype(np.uint8)
    gt_mask = (ghost_strength > ghost_strength.max() * 0.15) if ghost_strength.max() > 0 else np.zeros_like(ghost_strength, dtype=bool)
    return result, gt_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', required=True,
                    help='隣接丁ルールが確認済みの本(例: 200003803)')
    ap.add_argument('--n_pages', type=int, default=0,
                    help='生成するページ数の上限(0で全ページ)')
    ap.add_argument('--alpha', type=float, default=0.13,
                    help='裏写りの最大濃さ(黒へのアルファブレンド率)。'
                         '実例の平均gray差(191→175程度)に合わせたデフォルト')
    ap.add_argument('--blur_sigma', type=float, default=2.5,
                    help='インク滲みを模したぼかし量(px)')
    ap.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'tmp_work', 'synthetic_bleedthrough'))
    args = ap.parse_args()

    split = find_split(args.book)
    pages = list_pages(args.book, split)
    pageset = set(pages)

    out_book = os.path.join(args.out_dir, args.book)
    clean_dir = os.path.join(out_book, 'clean')
    deg_dir = os.path.join(out_book, 'degraded')
    gt_dir = os.path.join(out_book, 'ghost_gt')
    for d in (clean_dir, deg_dir, gt_dir):
        os.makedirs(d, exist_ok=True)

    n_done = 0
    for page, side in pages:
        if args.n_pages and n_done >= args.n_pages:
            break
        src_page, src_side = adjacent_source(page, side)
        if (src_page, src_side) not in pageset:
            continue

        target_img = load_clean_image(args.book, split, page, side)
        src_text = load_text_mask(args.book, split, src_page, src_side)
        if target_img is None or src_text is None:
            continue

        H, W = target_img.shape[:2]
        ghost_strength = make_synthetic_ghost((H, W), src_text, args.alpha, args.blur_sigma)
        degraded, gt_mask = apply_synthetic_ghost(target_img, ghost_strength)

        stem = f'{args.book}_{page:05d}_{side}'
        cv2.imwrite(os.path.join(clean_dir, stem + '.png'), target_img)
        cv2.imwrite(os.path.join(deg_dir, stem + '.png'), degraded)
        cv2.imwrite(os.path.join(gt_dir, stem + '.png'), (gt_mask.astype(np.uint8) * 255))

        ghost_px = gt_mask.mean()
        gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gt_mask.sum() > 0:
            print(f'{stem}: source=p{src_page}_{src_side}  ghost_area={100*ghost_px:.2f}%  '
                 f'gray(degraded)={gray[gt_mask].mean():.1f}  gray(clean@same)={bg_gray[gt_mask].mean():.1f}')
        n_done += 1

    print(f'\n完了: {n_done}枚を {out_book} に生成')


if __name__ == '__main__':
    main()
