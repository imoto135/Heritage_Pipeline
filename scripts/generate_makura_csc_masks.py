"""
枕草子(200017197, data/makura_kiridashi/ 665枚)のACE+CSC v3マスクを一括生成する。

split_dataset側の`scripts/generate_csc_masks_v2.py`と同じ出力構造
(00_input.png/00_original.png/02_text_black.png/03_text_red.png/
04_degradation.png/05_background.png)を`data/makura_csc_mask/<stem>/`に保存する。
枕草子はtrain/val/testのbook分割に含まれない別本のため専用スクリプトとして分離した。

事前にマスクを保存しておくことで、`generate_makura_benchmark.py`側のページ選定は
（split_dataset_csc_mask_v2を使う`generate_pseudo_degradation.py`同様）既存マスクを
読むだけの高速な処理にできる。

ACEはGPU(torch)版(`automatic_color_equalization_torch`)を使う。CPU版は1枚69秒・
ピークRSS約46GBだったが、GPU版はRTX A6000実測で1枚0.5〜1秒・VRAM約8GB/プロセスに
収まる（詳細は`modules/ddrm_npj_v3/scripts/ace.py`のdocstring参照）。ワーカー数は
VRAM(49GB)を基準に既定4（4プロセス×約8GB=32GB、安全マージンあり）。

使い方:
    conda run -n diffusion python scripts/generate_makura_csc_masks.py --workers 4
"""

import argparse
import os

# 並列ワーカーごとのBLAS/OpenMPスレッド数をキャップする（cv2/numpy importより先に設定する
# 必要がある）。ワーカー数と内部スレッド数の掛け算によるオーバーサブスクリプションを防ぐ。
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'modules', 'ddrm_npj_v3', 'scripts'))
from ace import automatic_color_equalization_torch
from csc import color_space_clustering

SRC_DIR    = os.path.join(REPO_ROOT, 'data', 'makura_kiridashi')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data', 'makura_csc_mask')


def _worker_init():
    """cv2内部並列化(TBB/pthread)はOMP_NUM_THREADS等の環境変数キャップが効かないため、
    ワーカープロセス起動時に別途1スレッドに絞る。しないとn_workers×全コアの
    オーバーサブスクリプションでBrokenProcessPoolを誘発する。"""
    cv2.setNumThreads(1)


def _process_one(fname: str) -> str | None:
    """1ページ分のACE+CSC v3処理（別プロセスで実行される）。失敗時はエラー文字列を返す。"""
    stem = os.path.splitext(fname)[0]
    out_dir = os.path.join(OUTPUT_DIR, stem)

    if os.path.exists(os.path.join(out_dir, '02_text_black.png')):
        return None  # 冪等性（再実行時にスキップ）

    raw_img = cv2.imread(os.path.join(SRC_DIR, fname))
    if raw_img is None:
        return f'読み込み失敗: {fname}'

    try:
        ace_img = automatic_color_equalization_torch(raw_img, alpha=5.0, n_samples=200, seed=0, device='cuda')
        result = color_space_clustering(ace_img)
    except Exception as e:
        return f'ACE/CSC失敗 ({e}): {fname}'

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, '00_original.png'),    raw_img)
    cv2.imwrite(os.path.join(out_dir, '00_input.png'),       ace_img)
    cv2.imwrite(os.path.join(out_dir, '02_text_black.png'),  255 - result.black_text_mask)
    cv2.imwrite(os.path.join(out_dir, '03_text_red.png'),    255 - result.red_text_mask)
    cv2.imwrite(os.path.join(out_dir, '04_degradation.png'), result.degradation_mask)
    cv2.imwrite(os.path.join(out_dir, '05_background.png'),  result.background_mask)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    all_files = sorted(f for f in os.listdir(SRC_DIR)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    remaining = [f for f in all_files
                if not os.path.exists(os.path.join(OUTPUT_DIR, os.path.splitext(f)[0], '02_text_black.png'))]
    print(f'全{len(all_files)}枚中、未処理{len(remaining)}枚を{args.workers}並列で処理します。')

    errors = []
    attempt = 0
    while remaining:
        attempt += 1
        try:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
                futures = {ex.submit(_process_one, f): f for f in remaining}
                done_count = 0
                for fut in as_completed(futures):
                    err = fut.result()
                    if err:
                        errors.append(err)
                    done_count += 1
                    if done_count % 10 == 0 or done_count == len(futures):
                        print(f'  完了 {done_count}/{len(futures)}（試行{attempt}回目のバッチ内）', flush=True)
        except BrokenProcessPool as e:
            print(f'  ワーカープール異常終了(試行{attempt}回目): {e}', flush=True)
            if attempt >= 5:
                raise
            print('  完了済みファイルはスキップされるので、残りを再試行します...', flush=True)
        remaining = [f for f in all_files
                    if not os.path.exists(os.path.join(OUTPUT_DIR, os.path.splitext(f)[0], '02_text_black.png'))]

    for e in errors:
        print(e)
    if errors:
        print(f'エラー: {len(errors)}件')
    print('完了。')


if __name__ == '__main__':
    main()
