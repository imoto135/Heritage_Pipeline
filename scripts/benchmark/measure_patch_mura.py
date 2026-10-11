"""
パッチ推論(DDRM/DDNM, patch_size=128, overlap=32既定)の境界に生じる
「継ぎ目ムラ」を定量化する。

- 勾配比: 背景マスク内で、パッチ境界(stride=patch_size-overlapの倍数)に
  乗る行・列の平均|Sobel勾配| と、それ以外(内部)の平均|Sobel勾配|の比。
  1.0に近いほど境界が目立たない。>1は境界で勾配が跳ね上がる(継ぎ目が見える)ことを示す。
- ブロックstd: パッチグリッドに沿って背景をブロック分割し、
  各ブロックの平均輝度のstdを取る。値が大きいほどパッチ単位の濃淡ムラが強い。

背景マスク(05_background.png, 白=背景)で文字・赤字・劣化領域を除外して評価する。
"""
import argparse
import cv2
import numpy as np


def seam_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """パッチ境界(内部の継ぎ目のみ、画像端は除く)の座標一覧。"""
    seams = []
    pos = patch_size
    while pos < length:
        seams.append(pos)
        pos += stride
    # 開始位置側の継ぎ目(次パッチの先頭)も同様にstride間隔で発生する
    seams2 = []
    pos = stride
    while pos < length - patch_size + stride:
        seams2.append(pos)
        pos += stride
    return sorted(set(s for s in seams + seams2 if 0 < s < length))


def gradient_ratio(gray: np.ndarray, bg_mask: np.ndarray, patch_size: int, overlap: int) -> dict:
    stride = patch_size - overlap
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)

    H, W = gray.shape
    seam_rows = seam_positions(H, patch_size, stride)
    seam_cols = seam_positions(W, patch_size, stride)

    bg = bg_mask > 0
    seam_mask = np.zeros((H, W), dtype=bool)
    for r in seam_rows:
        seam_mask[max(0, r - 1):r + 2, :] = True
    for c in seam_cols:
        seam_mask[:, max(0, c - 1):c + 2] = True

    seam_bg = seam_mask & bg
    interior_bg = (~seam_mask) & bg

    if seam_bg.sum() < 100 or interior_bg.sum() < 100:
        return {'seam_grad_mean': float('nan'), 'interior_grad_mean': float('nan'),
                'gradient_ratio': float('nan'), 'n_seam_px': int(seam_bg.sum()),
                'n_interior_px': int(interior_bg.sum())}

    seam_grad_mean = float(grad[seam_bg].mean())
    interior_grad_mean = float(grad[interior_bg].mean())
    return {
        'seam_grad_mean': seam_grad_mean,
        'interior_grad_mean': interior_grad_mean,
        'gradient_ratio': seam_grad_mean / interior_grad_mean,
        'n_seam_px': int(seam_bg.sum()),
        'n_interior_px': int(interior_bg.sum()),
    }


def block_std(gray: np.ndarray, bg_mask: np.ndarray, patch_size: int, overlap: int,
               min_bg_ratio: float = 0.5) -> dict:
    stride = patch_size - overlap
    H, W = gray.shape
    bg = bg_mask > 0
    means = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            blk_gray = gray[y:y + stride, x:x + stride]
            blk_bg = bg[y:y + stride, x:x + stride]
            if blk_bg.size == 0:
                continue
            if blk_bg.mean() < min_bg_ratio:
                continue
            means.append(float(blk_gray[blk_bg].mean()))
    if len(means) < 4:
        return {'block_mean_std': float('nan'), 'n_blocks': len(means)}
    return {'block_mean_std': float(np.std(means)), 'n_blocks': len(means)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='評価対象の復元済み画像')
    ap.add_argument('--bg_mask', required=True, help='CSCの05_background.png(白=背景)')
    ap.add_argument('--patch_size', type=int, default=128)
    ap.add_argument('--overlap', type=int, default=32)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bg_mask = cv2.imread(args.bg_mask, cv2.IMREAD_GRAYSCALE)
    if bg_mask is None:
        raise FileNotFoundError(args.bg_mask)
    if bg_mask.shape != gray.shape:
        bg_mask = cv2.resize(bg_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    gr = gradient_ratio(gray, bg_mask, args.patch_size, args.overlap)
    bs = block_std(gray, bg_mask, args.patch_size, args.overlap)

    print(f"image: {args.image}")
    print(f"  gradient_ratio      = {gr['gradient_ratio']:.4f}  "
          f"(seam={gr['seam_grad_mean']:.3f}, interior={gr['interior_grad_mean']:.3f}, "
          f"n_seam={gr['n_seam_px']}, n_interior={gr['n_interior_px']})")
    print(f"  block_mean_std      = {bs['block_mean_std']:.4f}  (n_blocks={bs['n_blocks']})")


if __name__ == '__main__':
    main()
