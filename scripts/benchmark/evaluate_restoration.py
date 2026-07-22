"""
復元結果の定量評価（PSNR / SSIM / LPIPS / CIEDE2000色差 / シミ除去F-measure）。

npj論文の評価プロトコルに準拠: クリーンGTと復元結果を比較し、
min / max / mean / var を手法ごとに集計する。

CIEDE2000（ΔE00）はCIE公式の知覚的色差指標で、文化財の色再現性評価で
標準的に使われる（例: "Modification of CIEDE2000 for Assessing Color
Quality of Image Archives"）。背景参照画像の設計差（外部参照 vs 自己参照）
による色味のズレを、構造指標(PSNR/SSIM)や知覚指標(LPIPS)とは独立に測る。
グレースケール(輝度)ベースの指標(PSNR/SSIM/F-measure)は、色情報のない
文献（白黒スキャン等）にも公平に適用できるよう、色を前提としない。

--stain_gt_dir を指定すると、generate_pseudo_degradation.py が出力する
シミ領域の正解マスク(stain_gt/<stem>.png)を使い、DIBCO等の文書二値化
コンペで一般的なF-measure/precision/recallに相当する指標を、シミ除去の
成否について計算する:
  recall    = シミ領域のうちGTに十分近づけられた画素の割合（除去率）
  precision = シミ以外の領域でGTに十分近い状態を保てた画素の割合（副作用の少なさ）
  f1        = recallとprecisionの調和平均

使い方:
    cd /home/imoto/Heritage_Pipeline
    conda run -n diffusion python scripts/benchmark/evaluate_restoration.py \
        --gt_dir tmp_work/benchmark/clean \
        --stain_gt_dir tmp_work/benchmark/stain_gt \
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
from skimage.color import deltaE_ciede2000, rgb2lab
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


def mean_delta_e2000(gt_bgr: np.ndarray, res_bgr: np.ndarray) -> float:
    """CIEDE2000（ΔE00）の画素平均。値が小さいほど色味がGTに近い。"""
    gt_lab  = rgb2lab(cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    res_lab = rgb2lab(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    return float(deltaE_ciede2000(gt_lab, res_lab).mean())


def stain_removal_scores(gt_bgr: np.ndarray, res_bgr: np.ndarray,
                         stain_mask: np.ndarray, tol: float = 15.0):
    """
    シミ除去のrecall/precision/F1（DIBCO等のF-measureに相当する枠組み）。
    輝度差のみで判定するため色の有無に依存しない。

    stain_mask: bool (H,W)、True=擬似シミを付与した画素
    tol: 復元結果がGTに「十分近い」とみなす輝度差の許容値
    """
    gt_gray  = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    close_to_gt = np.abs(res_gray - gt_gray) <= tol

    stain_px = stain_mask
    clean_px = ~stain_mask

    recall = float(close_to_gt[stain_px].mean()) if stain_px.any() else float('nan')
    precision = float(close_to_gt[clean_px].mean()) if clean_px.any() else float('nan')
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)
    return recall, precision, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--methods', nargs='+', required=True,
                        help='name=dir 形式で複数指定')
    parser.add_argument('--out', default=None, help='CSV出力先')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lpips_size', type=int, default=1024,
                        help='LPIPS計算時の最大辺（VRAM節約のため縮小）')
    parser.add_argument('--stain_gt_dir', default=None,
                        help='付与シミの正解マスクディレクトリ（指定時のみF-measure等を計算）')
    parser.add_argument('--stain_tol', type=float, default=15.0,
                        help='シミ除去判定の輝度差許容値')
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

        psnrs, ssims, lps, des = [], [], [], []
        recalls, precisions, f1s = [], [], []
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
            de = mean_delta_e2000(gt, res)

            rec = prec = f1 = float('nan')
            if args.stain_gt_dir:
                sg_p = os.path.join(args.stain_gt_dir, stem + '.png')
                sg = cv2.imread(sg_p, cv2.IMREAD_GRAYSCALE)
                if sg is not None:
                    if sg.shape != gt.shape[:2]:
                        sg = cv2.resize(sg, (gt.shape[1], gt.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    rec, prec, f1 = stain_removal_scores(gt, res, sg > 127, args.stain_tol)
                    recalls.append(rec); precisions.append(prec); f1s.append(f1)
                else:
                    print(f'[WARN] stain_gtなし: {stem}')

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

            psnrs.append(p); ssims.append(s); lps.append(lp); des.append(de)
            rows.append([name, stem, f'{p:.3f}', f'{s:.4f}', f'{lp:.4f}', f'{de:.4f}',
                        f'{rec:.4f}', f'{prec:.4f}', f'{f1:.4f}'])

        summary[name] = {
            'n': len(psnrs),
            'psnr': (np.min(psnrs), np.max(psnrs), np.mean(psnrs), np.var(psnrs)),
            'ssim': (np.min(ssims), np.max(ssims), np.mean(ssims), np.var(ssims)),
            'lpips': (np.min(lps), np.max(lps), np.mean(lps), np.var(lps)),
            'deltaE': (np.min(des), np.max(des), np.mean(des), np.var(des)),
        }
        if recalls:
            summary[name]['stain_f1'] = (np.mean(recalls), np.mean(precisions), np.mean(f1s))

    # 集計表示
    print('\n===== 集計 (min / max / mean / var) =====')
    hdr = (f"{'method':12s} {'n':>3s}  {'PSNR':^28s}  {'SSIM':^30s}  "
           f"{'LPIPS':^30s}  {'deltaE00':^28s}  {'stain R/P/F1':^22s}")
    print(hdr)
    for name, s in summary.items():
        p, ss, lp, de = s['psnr'], s['ssim'], s['lpips'], s['deltaE']
        stain_str = ''
        if 'stain_f1' in s:
            r, pr, f1 = s['stain_f1']
            stain_str = f"{r:.3f}/{pr:.3f}/{f1:.3f}"
        print(f"{name:12s} {s['n']:3d}  "
              f"{p[0]:6.2f}/{p[1]:6.2f}/{p[2]:6.2f}/{p[3]:6.3f}  "
              f"{ss[0]:.4f}/{ss[1]:.4f}/{ss[2]:.4f}/{ss[3]:.5f}  "
              f"{lp[0]:.4f}/{lp[1]:.4f}/{lp[2]:.4f}/{lp[3]:.5f}  "
              f"{de[0]:6.2f}/{de[1]:6.2f}/{de[2]:6.2f}/{de[3]:6.3f}  "
              f"{stain_str}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'stem', 'psnr', 'ssim', 'lpips', 'deltaE00',
                       'stain_recall', 'stain_precision', 'stain_f1'])
            w.writerows(rows)
        print(f'\nCSV保存: {args.out}')


if __name__ == '__main__':
    main()
