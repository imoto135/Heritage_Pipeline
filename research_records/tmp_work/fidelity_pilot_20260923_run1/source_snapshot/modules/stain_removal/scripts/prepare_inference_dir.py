"""推論用ディレクトリを準備するスクリプト。

testデータ（jpg）とtest_maskedデータ（マスクpng）を対応させて、
main_high_inference.pyが期待する構造に変換する:

  output_dir/<doc_id>_<page_id>_<side>/
      00_input.png
      02_text_black.png

使い方:
    python scripts/prepare_inference_dir.py
"""

import os
import sys
import shutil
from PIL import Image

TEST_DIR      = './data/split_dataset/test'
MASKED_DIR    = './data/split_dataset/test_masked'
OUTPUT_DIR    = './data/inference_input'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prepared = 0
    skipped  = 0

    for doc_id in sorted(os.listdir(MASKED_DIR)):
        masked_doc_dir = os.path.join(MASKED_DIR, doc_id)
        test_doc_dir   = os.path.join(TEST_DIR, doc_id)
        if not os.path.isdir(masked_doc_dir):
            continue

        for mask_fname in sorted(os.listdir(masked_doc_dir)):
            if not mask_fname.endswith('_02_text_black.png'):
                continue

            # 例: 200010454_00024_1_02_text_black.png → 200010454_00024_1
            page_key = mask_fname.replace('_02_text_black.png', '')
            img_fname = page_key + '.jpg'
            img_path  = os.path.join(test_doc_dir, img_fname)
            mask_path = os.path.join(masked_doc_dir, mask_fname)

            if not os.path.exists(img_path):
                print(f'[WARN] 対応画像なし: {img_path}')
                skipped += 1
                continue

            out_dir = os.path.join(OUTPUT_DIR, page_key)
            if os.path.exists(os.path.join(out_dir, '00_input.png')) and \
               os.path.exists(os.path.join(out_dir, '02_text_black.png')):
                skipped += 1
                continue

            os.makedirs(out_dir, exist_ok=True)

            # 入力画像をPNGに変換して保存
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(out_dir, '00_input.png'))

            # マスクをコピー
            shutil.copy(mask_path, os.path.join(out_dir, '02_text_black.png'))

            prepared += 1

    print(f'完了: {prepared}件準備, {skipped}件スキップ')
    print(f'出力先: {OUTPUT_DIR}')


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    main()
