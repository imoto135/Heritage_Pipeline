"""
擬似劣化ベンチマーク生成（修士論文 Algorithm 2 の実装）。

1. テストsplitのCSC v2マスクから劣化率最小のクリーンページを自動選定（GT）
2. Algorithm 2で擬似シミを付与（text-preserving版 I_stained を使用）
3. 劣化画像に ACE + CSC v2 を適用してマスク生成（復元手法への入力）

出力構造:
    tmp_work/benchmark/
      clean/<stem>.png        # GT（クリーン原本）
      degraded/<stem>.png     # 擬似劣化入力
      stain_gt/<stem>.png     # 付与したシミ領域マスク（分析用）
      masks/<stem>/00_input.png, 02_text_black.png, ... # 劣化画像のCSC v2マスク

使い方:
    cd /home/imoto/Heritage_Pipeline
    conda run -n diffusion python scripts/benchmark/generate_pseudo_degradation.py \
        --n_pages 10 --intensity 1.0
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'modules', 'ddrm_npj_v2', 'scripts'))
from ace import automatic_color_equalization
from csc import color_space_clustering

CSC_MASK_V2 = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc_mask_v2', 'test')
SPLIT_TEST  = os.path.join(REPO_ROOT, 'data', 'split_dataset', 'test')
OUT_BASE    = os.path.join(REPO_ROOT, 'tmp_work', 'benchmark')


def select_clean_pages(n_pages: int, book: str = '200010454'):
    """CSC v2マスクの劣化クラス割合が最小のページを選ぶ（GT用クリーンページ）。"""
    scores = []
    for d in sorted(glob.glob(os.path.join(CSC_MASK_V2, book, '*'))):
        stem = os.path.basename(d)
        deg_p = os.path.join(d, '04_degradation.png')
        tb_p  = os.path.join(d, '02_text_black.png')
        if not (os.path.exists(deg_p) and os.path.exists(tb_p)):
            continue
        deg = cv2.imread(deg_p, cv2.IMREAD_GRAYSCALE)
        tb  = cv2.imread(tb_p,  cv2.IMREAD_GRAYSCALE)
        deg_ratio = (deg > 127).mean()
        text_ratio = (tb < 127).mean()
        # 文字がある本文ページに限定（表紙・白紙を除外）
        if text_ratio < 0.02:
            continue
        scores.append((deg_ratio, stem))
    scores.sort()
    return [s for _, s in scores[:n_pages]]


def pseudo_degrade(img_bgr: np.ndarray,
                   rho: float = 3e-4, sigma: float = 30.0,
                   alpha: float = 2.0, lam: float = 1.0,
                   stain_color_rgb=(150, 110, 80), tau: int = 150,
                   seed: int = 0):
    """
    修士論文 Algorithm 2: Pseudo-degradation image generation。
    text-preserving版 (I_stained) と付与シミマスクを返す。
    """
    rng = np.random.default_rng(seed)
    H, W = img_bgr.shape[:2]

    # (1) Text mask extraction
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    m_text = (gray < tau).astype(np.float32)

    # (2) Stain mask generation
    n_blobs = int(rho * H * W)
    S = np.zeros((H, W), dtype=np.uint8)
    ys = rng.integers(0, H, n_blobs)
    xs = rng.integers(0, W, n_blobs)
    S[ys, xs] = 1

    # (3) Distance transform and smoothing
    D = cv2.distanceTransform((1 - S).astype(np.uint8), cv2.DIST_L2, 5)
    m_dist = (D < sigma).astype(np.float32)
    ksigma = sigma / (2 * alpha)
    m_smooth = cv2.GaussianBlur(m_dist, (0, 0), sigmaX=ksigma)
    m_norm = (m_smooth - m_smooth.min()) / max(m_smooth.max() - m_smooth.min(), 1e-6)

    # (4) Noise generation and blob combination
    N = rng.random((H, W)).astype(np.float32)
    n_smooth = cv2.GaussianBlur(N, (0, 0), sigmaX=ksigma)
    n_norm = (n_smooth - n_smooth.min()) / max(n_smooth.max() - n_smooth.min(), 1e-6)
    m_blob = ((m_norm * n_norm) > 0.5).astype(np.float32)

    # (5) Color stain mask creation (BGRで作る)
    c_bgr = np.array(stain_color_rgb[::-1], dtype=np.float32)
    m_stain = m_blob[..., None] * c_bgr[None, None, :]

    # (6) Stain application (text-preserving)
    m_bg = 1.0 - m_text
    stain_active = (m_stain.mean(axis=2) > 10).astype(np.float32)
    m_blend = m_bg * stain_active
    blend3 = (m_blend * lam)[..., None]
    stained = img_bgr.astype(np.float32) * (1 - blend3) + m_stain * blend3
    stained = np.clip(stained, 0, 255).astype(np.uint8)

    return stained, (m_blend > 0).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pages',   type=int, default=10)
    parser.add_argument('--book',      default='200010454')
    parser.add_argument('--intensity', type=float, default=1.0)
    parser.add_argument('--seed',      type=int, default=0)
    args = parser.parse_args()

    for sub in ['clean', 'degraded', 'stain_gt']:
        os.makedirs(os.path.join(OUT_BASE, sub), exist_ok=True)

    stems = select_clean_pages(args.n_pages, args.book)
    print(f'クリーンページ {len(stems)} 枚選定: {stems}')

    for i, stem in enumerate(stems):
        src = os.path.join(SPLIT_TEST, args.book, stem + '.jpg')
        img = cv2.imread(src)
        if img is None:
            print(f'読み込み失敗: {src}')
            continue

        degraded, stain_mask = pseudo_degrade(
            img, lam=args.intensity, seed=args.seed + i)

        cv2.imwrite(os.path.join(OUT_BASE, 'clean',    stem + '.png'), img)
        cv2.imwrite(os.path.join(OUT_BASE, 'degraded', stem + '.jpg'), degraded)
        cv2.imwrite(os.path.join(OUT_BASE, 'stain_gt', stem + '.png'), stain_mask)

        # 劣化画像に ACE + CSC v2 → マスク生成
        out_dir = os.path.join(OUT_BASE, 'masks', stem)
        if os.path.exists(os.path.join(out_dir, '02_text_black.png')):
            print(f'[{i+1}/{len(stems)}] {stem}: マスク既存')
            continue
        os.makedirs(out_dir, exist_ok=True)
        ace_img = automatic_color_equalization(degraded, alpha=5.0, n_samples=200, seed=0)
        result = color_space_clustering(ace_img)
        cv2.imwrite(os.path.join(out_dir, '00_input.png'),       ace_img)
        cv2.imwrite(os.path.join(out_dir, '02_text_black.png'),  255 - result.black_text_mask)
        cv2.imwrite(os.path.join(out_dir, '03_text_red.png'),    255 - result.red_text_mask)
        cv2.imwrite(os.path.join(out_dir, '04_degradation.png'), result.degradation_mask)
        cv2.imwrite(os.path.join(out_dir, '05_background.png'),  result.background_mask)
        stain_ratio = (stain_mask > 0).mean()
        det_ratio = (result.degradation_mask > 127).mean()
        print(f'[{i+1}/{len(stems)}] {stem}: 付与シミ={100*stain_ratio:.1f}%, CSC検出={100*det_ratio:.1f}%')

    print('\n完了。')


if __name__ == '__main__':
    main()
