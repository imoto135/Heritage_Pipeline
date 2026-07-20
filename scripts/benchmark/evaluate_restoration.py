"""
復元結果の定量評価（PSNR / SSIM / LPIPS）。

npj論文の評価プロトコルに準拠: クリーンGTと復元結果を比較し、
min / max / mean / var を手法ごとに集計する。

使い方:
    cd /home/imoto/Heritage_Pipeline
    conda run -n diffusion python scripts/benchmark/evaluate_restoration.py \
        --gt_dir tmp_work/benchmark/clean \
        --methods degraded=tmp_work/benchmark/degraded \
                  ddnm=tmp_work/benchmark/results/ddnm \
                  y0only=tmp_work/benchmark/results/y0only \
        --out tmp_work/benchmark/metrics.csv
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_pairs(gt_dir: str, method_dir: str):
    """GTと復元結果のペアを列挙する（拡張子は不問、stemで対応付け）。"""
    gt_files = {os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(gt_dir, '*'))}
    pairs = []
    for p in sorted(glob.glob(os.path.join(method_dir, '*'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in gt_files:
            pairs.append((stem, gt_files[stem], p))
    return pairs


def to_lpips_tensor(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return t.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--methods', nargs='+', required=True,
                        help='name=dir 形式で複数指定')
    parser.add_argument('--out', default=None, help='CSV出力先')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lpips_size', type=int, default=1024,
                        help='LPIPS計算時の最大辺（VRAM節約のため縮小）')
    args = parser.parse_args()

    import lpips as lpips_lib
    loss_fn = lpips_lib.LPIPS(net='alex').to(args.device)

    rows = []
    summary = {}
    for spec in args.methods:
        name, mdir = spec.split('=', 1)
        pairs = load_pairs(args.gt_dir, mdir)
        if not pairs:
            print(f'[WARN] {name}: ペアなし ({mdir})')
            continue

        psnrs, ssims, lps = [], [], []
        for stem, gt_p, res_p in pairs:
            gt = cv2.imread(gt_p)
            res = cv2.imread(res_p)
            if gt is None or res is None:
                print(f'[WARN] 読み込み失敗: {stem}')
                continue
            if res.shape != gt.shape:
                res = cv2.resize(res, (gt.shape[1], gt.shape[0]))

            p = peak_signal_noise_ratio(gt, res, data_range=255)
            s = structural_similarity(
                cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), data_range=255)

            # LPIPS（大きい画像は縮小して計算）
            H, W = gt.shape[:2]
            scale = min(1.0, args.lpips_size / max(H, W))
            if scale < 1.0:
                gt_s  = cv2.resize(gt,  (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
                res_s = cv2.resize(res, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
            else:
                gt_s, res_s = gt, res
            with torch.no_grad():
                lp = loss_fn(to_lpips_tensor(gt_s, args.device),
                             to_lpips_tensor(res_s, args.device)).item()

            psnrs.append(p); ssims.append(s); lps.append(lp)
            rows.append([name, stem, f'{p:.3f}', f'{s:.4f}', f'{lp:.4f}'])

        summary[name] = {
            'n': len(psnrs),
            'psnr': (np.min(psnrs), np.max(psnrs), np.mean(psnrs), np.var(psnrs)),
            'ssim': (np.min(ssims), np.max(ssims), np.mean(ssims), np.var(ssims)),
            'lpips': (np.min(lps), np.max(lps), np.mean(lps), np.var(lps)),
        }

    # 集計表示
    print('\n===== 集計 (min / max / mean / var) =====')
    hdr = f"{'method':12s} {'n':>3s}  {'PSNR':^28s}  {'SSIM':^30s}  {'LPIPS':^30s}"
    print(hdr)
    for name, s in summary.items():
        p, ss, lp = s['psnr'], s['ssim'], s['lpips']
        print(f"{name:12s} {s['n']:3d}  "
              f"{p[0]:6.2f}/{p[1]:6.2f}/{p[2]:6.2f}/{p[3]:6.3f}  "
              f"{ss[0]:.4f}/{ss[1]:.4f}/{ss[2]:.4f}/{ss[3]:.5f}  "
              f"{lp[0]:.4f}/{lp[1]:.4f}/{lp[2]:.4f}/{lp[3]:.5f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'stem', 'psnr', 'ssim', 'lpips'])
            w.writerows(rows)
        print(f'\nCSV保存: {args.out}')


if __name__ == '__main__':
    main()
