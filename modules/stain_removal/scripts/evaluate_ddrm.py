"""
完了済み推論結果と入力画像を比較評価するスクリプト。

入力と復元画像のkurtosis/brightness_std/entropyを計算し、
論文の数値と比較できる形でCSVとコンソールに出力する。

使い方:
    cd modules/stain_removal
    python scripts/evaluate_ddrm.py
"""

import csv
import os
import sys

import numpy as np
from PIL import Image
from natsort import natsorted

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)

from src.metrics import compute_metrics

# 完了済み冊（推論が全ページ完了しているもの）
RESULT_DIR = '../../results/otsu'
INPUT_DIR  = '../../data/test'
OUTPUT_CSV = '../../results/evaluation_otsu.csv'

COMPLETE_BOOKS = [
    '200003803', '200004107', '200005798', '200008003',
    '200010454', '200017458', '200018243', '200019865', '200021063',
]


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def evaluate_book(book_id: str):
    input_book_dir  = os.path.join(INPUT_DIR,  book_id)
    result_book_dir = os.path.join(RESULT_DIR, book_id)

    rows = []
    page_dirs = natsorted([
        d for d in os.listdir(result_book_dir)
        if os.path.isdir(os.path.join(result_book_dir, d))
    ])

    for page_stem in page_dirs:
        result_path = os.path.join(result_book_dir, page_stem, 'result', f'{page_stem}.png')
        input_path  = os.path.join(input_book_dir,  f'{page_stem}.jpg')
        if not os.path.exists(input_path):
            input_path = os.path.join(input_book_dir, f'{page_stem}.png')

        if not os.path.exists(result_path) or not os.path.exists(input_path):
            continue

        restored = load_image(result_path)
        original = load_image(input_path)

        m_restored = compute_metrics(restored)
        m_original = compute_metrics(original)

        row = {
            'book':          book_id,
            'page':          page_stem,
            'kurtosis_R_in': m_original['kurtosis_R'],
            'kurtosis_G_in': m_original['kurtosis_G'],
            'kurtosis_B_in': m_original['kurtosis_B'],
            'kurtosis_R_out': m_restored['kurtosis_R'],
            'kurtosis_G_out': m_restored['kurtosis_G'],
            'kurtosis_B_out': m_restored['kurtosis_B'],
            'brightness_std_in':  m_original['brightness_std'],
            'brightness_std_out': m_restored['brightness_std'],
            'entropy_in':  m_original['entropy'],
            'entropy_out': m_restored['entropy'],
        }
        rows.append(row)

    return rows


def main():
    all_rows = []
    for book_id in COMPLETE_BOOKS:
        print(f'評価中: {book_id} ...', end=' ', flush=True)
        rows = evaluate_book(book_id)
        all_rows.extend(rows)
        print(f'{len(rows)}ページ完了')

    if not all_rows:
        print('結果なし')
        return

    # CSV保存
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # 全体平均を表示
    metric_keys = [k for k in fieldnames if k not in ('book', 'page')]
    means = {k: np.mean([r[k] for r in all_rows]) for k in metric_keys}

    print(f'\n=== 全体平均 ({len(all_rows)}ページ) ===')
    print(f'{"指標":<25} {"入力":>10} {"復元後":>10} {"変化":>10}')
    print('-' * 58)
    pairs = [
        ('kurtosis_R', 'kurtosis_R_in', 'kurtosis_R_out'),
        ('kurtosis_G', 'kurtosis_G_in', 'kurtosis_G_out'),
        ('kurtosis_B', 'kurtosis_B_in', 'kurtosis_B_out'),
        ('brightness_std', 'brightness_std_in', 'brightness_std_out'),
        ('entropy',     'entropy_in',    'entropy_out'),
    ]
    for label, k_in, k_out in pairs:
        v_in  = means[k_in]
        v_out = means[k_out]
        diff  = v_out - v_in
        sign  = '+' if diff >= 0 else ''
        print(f'{label:<25} {v_in:>10.4f} {v_out:>10.4f} {sign}{diff:>9.4f}')

    print(f'\nCSV保存: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
