"""
枕草子(200017197)に対する擬似劣化ベンチマーク生成。

修士論文の実験で実際に使われた元スクリプト(`a.py`/`b.py`、リポジトリルート)の
シミ生成ロジックを**そのまま**再利用する（`scripts/benchmark/generate_pseudo_degradation.py`の
独自実装は乱数生成器(legacy np.random.seed vs default_rng)とシミの色(固定色1色 vs
ファイルごとのランダム色)が元スクリプトと異なり、同じseedでも同じ画像を再現できないため、
論文の数値と揃えたい場合は元スクリプトの関数をそのまま使う必要がある）。

元スクリプトとの違いは1点のみ: `apply_stain`がシミ領域マスク(blend_mask)も返すように
拡張している。この変更は乱数の消費順序に影響しない（すでに計算済みの値を捨てずに返す
だけ）ため、再現性は保たれる。

seedはファイル名でなく「665枚を昇順ソートした全体でのインデックス」から計算する
(元のb.pyのenumerate(sorted(files))と同じ)。ベンチマーク用にN枚だけ選ぶ場合も、
そのN枚が665枚中の何番目かを引き継いで計算するため、後でNを変えても
個々のページの劣化パターンは変わらない。

使い方:
    conda run -n diffusion python scripts/benchmark/generate_makura_benchmark.py \
        --n_pages 20 --intensities 0.3 0.5 0.7 1.0
"""

import argparse
import os

# 並列ワーカーごとのBLAS/OpenMPスレッド数をキャップする（cv2/numpy importより先に設定する
# 必要がある）。96コアの機械でProcessPoolExecutorのワーカーごとに無制限にスレッドを
# 使わせると、ワーカー数と内部スレッド数の掛け算でオーバーサブスクリプションが起きる。
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import sys

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'modules', 'ddrm_npj_v3', 'scripts'))
from ace import automatic_color_equalization_torch
from csc import color_space_clustering

SRC_DIR  = os.path.join(REPO_ROOT, 'data', 'makura_kiridashi')
MASK_DIR = os.path.join(REPO_ROOT, 'data', 'makura_csc_mask')
OUT_BASE = os.path.join(REPO_ROOT, 'tmp_work', 'makura_benchmark')


# ─────────────────────────────────────────────────────────
# 元スクリプト(b.py)のシミ生成ロジック(そのまま)
# ─────────────────────────────────────────────────────────

def get_text_mask(img, threshold=150):
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    mask = (gray < threshold).astype(np.uint8)[:, :, None]
    return mask


def generate_stain_mask(H, W, color=None, density=3e-4, size=30, roughness=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    numblobs = int(density * H * W)
    mask = np.zeros((H, W), 'i')
    for _ in range(numblobs):
        mask[np.random.randint(0, H), np.random.randint(0, W)] = 1

    dt = ndi.distance_transform_edt(1 - mask)
    mask = (dt < size).astype(np.float32)
    mask = ndi.gaussian_filter(mask, size / (2 * roughness))
    mask -= mask.min()
    mask /= mask.max()

    noise = np.random.rand(H, W)
    noise = ndi.gaussian_filter(noise, size / (2 * roughness))
    noise -= noise.min()
    noise /= noise.max()

    blob = (mask * noise > 0.5).astype(np.float32)

    if color is None:
        r = np.random.randint(170, 200)
        g = np.random.randint(110, 150)
        b = np.random.randint(90, 120)
        color = np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)

    stain_mask = (blob[:, :, None] * color).astype(np.uint8)
    return stain_mask, color


def apply_stain(img, stain_mask, text_mask=None, avoid_text=True, intensity=1.0):
    """元のapply_stainに、シミ領域マスク(blend_mask)を追加で返す変更のみ加えたもの。
    乱数は一切消費しないため元スクリプトとの再現性に影響しない。"""
    img = img.astype(np.float32)
    stain = stain_mask.astype(np.float32)

    if avoid_text and text_mask is not None:
        bg_mask = (1 - text_mask).astype(np.float32)
        stain_area = np.mean(stain, axis=2, keepdims=True) > 10
        blend_mask = bg_mask * stain_area
        result = img * (1 - blend_mask * intensity) + stain * (blend_mask * intensity)
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        stain_area = np.mean(stain, axis=2, keepdims=True) > 10
        blend_mask = stain_area.astype(np.float32)
        result = img * (1 - blend_mask * intensity) + stain * (blend_mask * intensity)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result, blend_mask


def generate_fixed_color(seed):
    np.random.seed(seed)
    r = np.random.randint(170, 200)
    g = np.random.randint(110, 150)
    b = np.random.randint(90, 120)
    return np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)


def select_clean_pages(all_files, n_pages, min_text_ratio=None):
    """
    665枚それぞれの事前生成済みCSC v3マスク(`data/makura_csc_mask/<stem>/04_degradation.png`、
    `scripts/generate_makura_csc_masks.py`で生成済み)を読み、劣化マスク比率が最も低い
    n_pages枚を選ぶ（generate_pseudo_degradation.pyのselect_clean_pagesと同じ考え方）。

    以前はここでACE+CSCを毎回自前計算していたが、`generate_makura_csc_masks.py`が
    既に全665枚のマスクを事前生成済みになったため、単純な画像読み込みだけで済む
    （[[project_csc_v3_makura]]参照）。GPU/並列処理は不要なほど高速。

    min_text_ratio: 文字面積比(02_text_black.pngの黒画素割合)がこの値未満のページを
    除外する。Noneなら全665ページの文字面積比の中央値を自動計算して使う。
    文字が少ないページほど誤検出されうる面積自体が少なく劣化マスク比率も低く出やすい
    ため、単純に劣化率最小のn_pages枚を選ぶと文字密度が偏ったページセットになって
    しまう（実測: 665ページ全体の文字面積比中央値5.03%に対し、フィルタなしで選んだ
    旧20ページの中央値は3.67%＝下位2.7パーセンタイルだった。ユーザー指摘により発覚、
    2026-07-31）。文字面積比が中央値以上のページに絞ってから劣化率で選定することで、
    典型的なページに近い文字密度を保ったまま、劣化の少なさも維持する
    （実測: 中央値以上に絞っても劣化率0.20〜0.39%と、フィルタなし版の0.10〜0.31%と
    ほぼ遜色ない）。
    """
    scores = []
    text_ratios = []
    missing = []
    for fname in all_files:
        stem = os.path.splitext(fname)[0]
        deg = cv2.imread(os.path.join(MASK_DIR, stem, '04_degradation.png'), cv2.IMREAD_GRAYSCALE)
        black = cv2.imread(os.path.join(MASK_DIR, stem, '02_text_black.png'), cv2.IMREAD_GRAYSCALE)
        if deg is None or black is None:
            missing.append(fname)
            continue
        text_ratio = float((black < 127).mean())
        text_ratios.append(text_ratio)
        scores.append((float((deg > 127).mean()), text_ratio, fname))

    if missing:
        raise RuntimeError(
            f'{len(missing)}枚のCSC v3マスクが見つかりません。先に'
            f'scripts/generate_makura_csc_masks.pyを実行してください: {missing[:5]}')

    if min_text_ratio is None:
        min_text_ratio = float(np.median(text_ratios))
    print(f'文字面積比の下限フィルタ: {100*min_text_ratio:.2f}% '
          f'({sum(1 for _, t, _ in scores if t >= min_text_ratio)}/{len(scores)}枚が該当)')

    filtered = [(d, f) for d, t, f in scores if t >= min_text_ratio]
    filtered.sort()
    return [f for _, f in filtered[:n_pages]]


# ─────────────────────────────────────────────────────────
# ベンチマーク生成
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pages', type=int, default=20,
                        help='665枚中、先頭から何枚をベンチマーク対象にするか')
    parser.add_argument('--intensities', type=float, nargs='+', default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument('--density', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=1000000, help='元スクリプトと同じseed=1000000')
    parser.add_argument('--min_text_ratio', type=float, default=None,
                        help='文字面積比の下限(0〜1)。省略時は665ページ全体の中央値を自動使用')
    args = parser.parse_args()

    all_files = sorted([f for f in os.listdir(SRC_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f'{SRC_DIR} 内の全ファイル数: {len(all_files)}')

    os.makedirs(OUT_BASE, exist_ok=True)
    target_files = set(select_clean_pages(all_files, args.n_pages, args.min_text_ratio))
    print(f'ベンチマーク対象: {len(target_files)}枚（劣化率が最も低いページを選定）')

    clean_dir = os.path.join(OUT_BASE, 'clean')
    os.makedirs(clean_dir, exist_ok=True)

    for intensity in args.intensities:
        param_str = f'intensity_{intensity:.2f}'
        degraded_dir  = os.path.join(OUT_BASE, 'degraded', param_str)
        stain_gt_dir  = os.path.join(OUT_BASE, 'stain_gt', param_str)
        masks_dir     = os.path.join(OUT_BASE, 'masks', param_str)
        os.makedirs(degraded_dir, exist_ok=True)
        os.makedirs(stain_gt_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        for global_idx, fname in enumerate(all_files):
            if fname not in target_files:
                continue
            stem = os.path.splitext(fname)[0]

            img_path = os.path.join(SRC_DIR, fname)
            img = io.imread(img_path)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            H, W = img.shape[:2]

            # クリーン原本の保存(強度に依存しないので1回だけ)
            clean_path = os.path.join(clean_dir, stem + '.png')
            if not os.path.exists(clean_path):
                io.imsave(clean_path, img)

            file_seed = args.seed + global_idx  # 元b.pyと同じ: 665枚全体でのindex
            fixed_color = generate_fixed_color(file_seed)
            stain_mask, _ = generate_stain_mask(H, W, color=fixed_color,
                                                density=args.density, seed=file_seed)
            text_mask = get_text_mask(img)

            img_no_text, blend_mask = apply_stain(img, stain_mask, text_mask,
                                                  avoid_text=True, intensity=intensity)

            # img_no_text/img はRGB(skimage.io)なので、ACE/CSC(BGR前提)用にBGRへ変換
            img_bgr = cv2.cvtColor(img_no_text, cv2.COLOR_RGB2BGR)

            io.imsave(os.path.join(degraded_dir, stem + '.png'), img_no_text)
            stain_gt_u8 = (blend_mask[:, :, 0] > 0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(stain_gt_dir, stem + '.png'), stain_gt_u8)

            # ACE + CSC v3 でマスク生成(復元パイプラインの入力用)
            out_dir = os.path.join(masks_dir, stem)
            if os.path.exists(os.path.join(out_dir, '02_text_black.png')):
                continue
            os.makedirs(out_dir, exist_ok=True)
            ace_img = automatic_color_equalization_torch(img_bgr, alpha=5.0, n_samples=200, seed=0, device='cuda')
            result = color_space_clustering(ace_img)
            cv2.imwrite(os.path.join(out_dir, '00_input.png'),       ace_img)
            cv2.imwrite(os.path.join(out_dir, '02_text_black.png'),  255 - result.black_text_mask)
            cv2.imwrite(os.path.join(out_dir, '03_text_red.png'),    255 - result.red_text_mask)
            cv2.imwrite(os.path.join(out_dir, '04_degradation.png'), result.degradation_mask)
            cv2.imwrite(os.path.join(out_dir, '05_background.png'),  result.background_mask)

            stain_ratio = (stain_gt_u8 > 0).mean()
            det_ratio = (result.degradation_mask > 127).mean()
            print(f'[{param_str}] {stem}: 付与シミ={100*stain_ratio:.1f}%, CSC検出={100*det_ratio:.1f}%')

    print('\n完了。')


if __name__ == '__main__':
    main()
