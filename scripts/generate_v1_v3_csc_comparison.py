"""
CSC v1(ddrm_npj、npj論文の原設計) と v3(ddrm_npj_v3、Stage再設計+赤文字連結成分ゲート+
黒文字救済ロジック) の出力を、同一ページ・同一ACE済み入力画像で見比べるための比較画像を
生成する。

ACEはv1/v3で共通の既存ACE済み画像を使う（CPU/GPU ACEの数値差を混ぜず、CSCアルゴリズム
自体の差だけを見せるため）。

出力先: data/masks_compare/<page>/
  00_input.png              : 共通のACE済み入力画像
  00_original.png           : 共通のACE前元画像（あれば）
  v1/02_text_black.png ...  : v1の各クラスマスク
  v3/02_text_black.png ...  : v3の各クラスマスク
  compare_<class>.png       : v1(左)/v3(右)を並べたラベル付き比較画像

使い方:
    conda run -n diffusion python scripts/generate_v1_v3_csc_comparison.py
"""

import importlib.util
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(REPO_ROOT, 'data', 'masks_compare')

PAGES = [
    {
        'name': '200017197_00008_0',
        'ace_input': os.path.join(REPO_ROOT, 'data/makura_csc_mask/200017197_00008_0/00_input.png'),
        'original': os.path.join(REPO_ROOT, 'data/makura_csc_mask/200017197_00008_0/00_original.png'),
        'note': 'ink-rescue閾値較正ページ（黒文字がかすれて劣化クラスタに誤分類されやすい例）',
    },
    {
        'name': '200010454_00002_2',
        'ace_input': os.path.join(REPO_ROOT, 'data/split_dataset_csc/test/200010454/200010454_00002_2.jpg'),
        'original': os.path.join(REPO_ROOT, 'data/test/200010454/200010454_00002_2.jpg'),
        'note': '赤文字連結成分ゲートの検証ページ（本物の朱印+茶シミの誤検出片）',
    },
    {
        'name': 'hnsd001_011',
        'ace_input': os.path.join(REPO_ROOT, 'data/split_dataset_csc/train/hnsd00000/hnsd001_011.jpg'),
        'original': os.path.join(REPO_ROOT, 'data/train/hnsd00000/hnsd001_011.jpg'),
        'note': 'ACEが背景の汚れの分散を増幅する例（背景std 16.5→31.5、黒文字検出率59%→30%）',
    },
]

CLASSES = ['text_black', 'text_red', 'degradation', 'background']


def _load_csc_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _masks_from_result(result):
    return {
        'text_black': 255 - result.black_text_mask,
        'text_red': 255 - result.red_text_mask,
        'degradation': result.degradation_mask,
        'background': result.background_mask,
    }


def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    img = img_bgr.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    bar_h = max(40, h // 30)
    cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(img, text, (10, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                bar_h / 45.0, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def _side_by_side(mask_v1: np.ndarray, mask_v3: np.ndarray, page: str, cls: str) -> np.ndarray:
    left = _label(mask_v1, f'v1  {page}  {cls}')
    right = _label(mask_v3, f'v3  {page}  {cls}')
    sep = np.full((left.shape[0], 8, 3), (0, 0, 255), dtype=np.uint8)
    return np.hstack([left, sep, right])


def main():
    csc_v1 = _load_csc_module('csc_v1', os.path.join(REPO_ROOT, 'modules/ddrm_npj/scripts/csc.py'))
    csc_v3 = _load_csc_module('csc_v3', os.path.join(REPO_ROOT, 'modules/ddrm_npj_v3/scripts/csc.py'))

    for page in PAGES:
        name = page['name']
        print(f'\n=== {name} ({page["note"]}) ===')
        out_dir = os.path.join(OUTPUT_BASE, name)
        v1_dir = os.path.join(out_dir, 'v1')
        v3_dir = os.path.join(out_dir, 'v3')
        os.makedirs(v1_dir, exist_ok=True)
        os.makedirs(v3_dir, exist_ok=True)

        ace_img = cv2.imread(page['ace_input'])
        if ace_img is None:
            print(f'  ACE済み画像が読めません: {page["ace_input"]}')
            continue
        cv2.imwrite(os.path.join(out_dir, '00_input.png'), ace_img)

        if os.path.exists(page['original']):
            original = cv2.imread(page['original'])
            if original is not None:
                cv2.imwrite(os.path.join(out_dir, '00_original.png'), original)

        result_v1 = csc_v1.color_space_clustering(ace_img)
        result_v3 = csc_v3.color_space_clustering(ace_img)

        masks_v1 = _masks_from_result(result_v1)
        masks_v3 = _masks_from_result(result_v3)

        for cls in CLASSES:
            m1 = masks_v1[cls]
            m3 = masks_v3[cls]
            cv2.imwrite(os.path.join(v1_dir, f'{cls}.png'), m1)
            cv2.imwrite(os.path.join(v3_dir, f'{cls}.png'), m3)

            composite = _side_by_side(m1, m3, name, cls)
            cv2.imwrite(os.path.join(out_dir, f'compare_{cls}.png'), composite)

            diff_px = int(np.count_nonzero(m1 != m3))
            total_px = m1.size
            print(f'  {cls:12s}: v1/v3差分 {diff_px:>9d} px / {total_px:>9d} px '
                  f'({100.0 * diff_px / total_px:5.2f}%)')

    print(f'\n完了。{OUTPUT_BASE} 以下を参照してください。')


if __name__ == '__main__':
    main()
