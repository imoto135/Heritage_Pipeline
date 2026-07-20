"""
Stage1（パッチDDRM、文字が鮮明だが背景にパッチ境界のムラ）と
Stage2（ページDDRM、背景は均一だが256px経由の大幅拡大で文字が崩れる）を
CSCマスクで合成し、双方の長所だけを組み合わせる後処理。

  背景・劣化: Stage2の出力（均一な背景）
  黒文字:     Stage1の出力（鮮明な文字。マスク境界は数px феザーブレンド）
  赤文字:     元画像からオーバーレイ（既存パイプラインと同じ、彩度の高い朱色を保持）

使い方:
    python scripts/postprocess_composite_stage.py \
        --stage1 <Stage1結果.png> \
        --stage2 <Stage2結果.png> \
        --mask_dir <CSCマスクディレクトリ> \
        --output <出力.png>
"""

import argparse
import os

import cv2
import numpy as np


def composite(stage1_bgr: np.ndarray, stage2_bgr: np.ndarray,
              text_mask01: np.ndarray, feather_px: int = 3) -> np.ndarray:
    if stage2_bgr.shape[:2] != stage1_bgr.shape[:2]:
        stage2_bgr = cv2.resize(stage2_bgr, (stage1_bgr.shape[1], stage1_bgr.shape[0]),
                                interpolation=cv2.INTER_LANCZOS4)

    alpha = text_mask01.astype(np.float32)
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=feather_px / 2.0)
    alpha = alpha[..., None]

    out = stage2_bgr.astype(np.float32) * (1 - alpha) + stage1_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Stage1文字 + Stage2背景 の合成後処理')
    parser.add_argument('--stage1', required=True, help='Stage1（パッチDDRM）の結果PNG')
    parser.add_argument('--stage2', required=True, help='Stage2（ページDDRM）の結果PNG')
    parser.add_argument('--mask_dir', required=True,
                        help='CSCマスクディレクトリ（02_text_black.png, 03_text_red.png, 00_input.png）')
    parser.add_argument('--output', required=True)
    parser.add_argument('--feather', type=int, default=3, help='文字マスク境界のフェザー幅(px)')
    parser.add_argument('--text_dilate', type=int, default=2,
                        help='文字マスクを膨張させるピクセル数（アンチエイリアス縁を含める）')
    parser.add_argument('--stage2_blur_sigma', type=float, default=0,
                        help='Stage2を使う前に強くぼかして文字形状のノイズを除去するsigma。'
                             '0なら無効（Stage2をそのまま使用）')
    args = parser.parse_args()

    stage1 = cv2.imread(args.stage1)
    stage2 = cv2.imread(args.stage2)
    if stage1 is None or stage2 is None:
        raise FileNotFoundError('Stage1/Stage2の結果画像が読み込めません')

    if args.stage2_blur_sigma > 0:
        # Stage2はページ全体を256pxで再構成する際、背景扱いの領域にも不正確な
        # 文字状の淡い形状が残ることがある。強いガウシアンブラーでその高周波
        # ノイズを消し、純粋な背景トーン（低周波成分）だけを合成に使う。
        stage2 = cv2.GaussianBlur(stage2, (0, 0), sigmaX=args.stage2_blur_sigma)

    tb_path = os.path.join(args.mask_dir, '02_text_black.png')
    tb = cv2.imread(tb_path, cv2.IMREAD_GRAYSCALE)
    if tb is None:
        raise FileNotFoundError(f'黒文字マスクが見つかりません: {tb_path}')
    if tb.shape != stage1.shape[:2]:
        tb = cv2.resize(tb, (stage1.shape[1], stage1.shape[0]), interpolation=cv2.INTER_NEAREST)
    text_mask01 = (tb < 127).astype(np.uint8)  # 0=文字, 255=背景の規約

    if args.text_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (args.text_dilate * 2 + 1, args.text_dilate * 2 + 1))
        text_mask01 = cv2.dilate(text_mask01, k)

    out = composite(stage1, stage2, text_mask01, feather_px=args.feather)

    # 赤文字を元画像から合成（既存パイプラインと同じ設計）
    tr_path = os.path.join(args.mask_dir, '03_text_red.png')
    orig_path = os.path.join(args.mask_dir, '00_input.png')
    if os.path.exists(tr_path) and os.path.exists(orig_path):
        tr = cv2.imread(tr_path, cv2.IMREAD_GRAYSCALE)
        orig = cv2.imread(orig_path)
        if tr.shape != out.shape[:2]:
            tr = cv2.resize(tr, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
            orig = cv2.resize(orig, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        red_mask = tr < 127
        out[red_mask] = orig[red_mask]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f'保存: {args.output}')


if __name__ == '__main__':
    main()
