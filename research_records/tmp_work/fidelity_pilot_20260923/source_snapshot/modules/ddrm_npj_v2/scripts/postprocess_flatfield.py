"""
DDRM Stage1出力のパッチ間ムラをフラットフィールド補正で緩和する後処理。

modules/ddnm_npj/scripts/pipeline_ddnm.py の --enhance（full_kept時）にある
背景フラットフィールド補正ロジックを、DDRM側の単体後処理として移植したもの。
GPU不要、モデル推論なし。

処理:
  illum = GaussianBlur(gray, sigmaX=50)  # 大域的な低周波照明マップ
  bg_target = median(gray)（05_background.pngで背景マスクした範囲、自動）
  gain = bg_target / illum
  out = input * gain（--whole_image指定時は画像全体、未指定時は背景領域のみ）

使い方:
    python scripts/postprocess_flatfield.py \
        --input  ../../tmp_work/ddrm_csc_test2_v2/.../result/xxx.png \
        --mask_dir ../../data/split_dataset_csc_mask_v2/test/200010454/200010454_00002_2 \
        --output ../../tmp_work/mura_diag_v2/flatfield_xxx.png
"""

import argparse
import os

import cv2
import numpy as np


def apply_flatfield(img_bgr: np.ndarray, bg_mask: np.ndarray | None,
                    whole_image: bool, sigma: float = 50.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    illum = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

    if bg_mask is not None and bg_mask.any():
        bg_target = float(np.median(gray[bg_mask]))
    else:
        bg_target = float(np.median(gray))

    gain = bg_target / np.clip(illum, 1, None)
    corrected = np.clip(img_bgr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)

    if whole_image or bg_mask is None:
        return corrected

    out = img_bgr.copy()
    out[bg_mask] = corrected[bg_mask]
    return out


def main():
    parser = argparse.ArgumentParser(description='DDRM出力へのフラットフィールド後処理')
    parser.add_argument('--input',  required=True, help='DDRM Stage1/Stage2の結果PNG')
    parser.add_argument('--mask_dir', default=None,
                        help='CSCマスクディレクトリ（05_background.pngを使ってbg_target自動決定・'
                             '背景限定適用に使用。未指定時は画像全体の中央値をbg_targetに使用')
    parser.add_argument('--output', required=True)
    parser.add_argument('--whole_image', action='store_true',
                        help='DDNM --enhance 同様に画像全体へゲインを適用する（未指定時は背景領域のみ）')
    parser.add_argument('--sigma', type=float, default=50.0,
                        help='低周波照明マップのGaussianBlur sigma（DDNM --enhance と同じデフォルト50）')
    args = parser.parse_args()

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(f'入力画像が読めません: {args.input}')

    bg_mask = None
    if args.mask_dir:
        bg_path = os.path.join(args.mask_dir, '05_background.png')
        if os.path.exists(bg_path):
            bg_gray = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
            if bg_gray.shape != img_bgr.shape[:2]:
                bg_gray = cv2.resize(bg_gray, (img_bgr.shape[1], img_bgr.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            bg_mask = bg_gray > 127
        else:
            print(f'[WARN] 背景マスクが見つかりません: {bg_path} → 画像全体の中央値を使用')

    out = apply_flatfield(img_bgr, bg_mask, args.whole_image, args.sigma)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f'保存: {args.output}')


if __name__ == '__main__':
    main()
