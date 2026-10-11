"""
「CSCの劣化マスク(04_degradation.png)が捉えられない裏写りは、
DDRM/DDNMが原理的に除去できない」という仮説を検証するスクリプト。

背景（パイプライン設計の再確認、2026-07-31）:
  - DDRM本番パイプライン(pipeline_ddrm.py)のmissing領域は02_text_black.pngのみ。
    kept領域(文字以外すべて)は背景参照y_0へ拡散モデル経由で寄せられる。
  - 外部参照(isemonogatari.png等)を使う場合、y_0はこのページの内容と無関係な
    別文献のクリーンな紙面なので、kept領域の裏写りはCSCの劣化マスクの精度に
    関わらず基本的に消える構造になっている。
  - 一方、v15自己参照精緻化(pipeline_ddnm.py --degradation_only --full_kept)は
    04_degradation.pngだけをmissing(inpaint対象)とし、それ以外(文字+05_background
    と判定された画素)はそのまま保存する。つまり**04_degradation.pngが裏写りを
    見逃す(05_backgroundに分類する)と、その裏写りは自己参照背景にそのまま残り、
    最終出力にも残ってしまう**。これがv15自己参照方式(論文の主張の柱)における
    「DDRMが原理的に除去できない裏写り」の唯一の経路。

本スクリプト:
  [[project_bleedthrough_verification]]で隣接丁ルールが確認済みの3冊
  (200003803, 200018243, 200021071)について、
  generate_synthetic_bleedthrough.pyの物理モデル(隣接丁の文字マスクを反転・
  減光・ぼかす)を「学習データ合成」ではなく「実ページ上でどこに裏写りが
  物理的に出るはずかの予測」として転用する。予測されたghost強度が高い画素が、
  そのページの実際のCSC分類(04_degradation vs 05_background)でどちらの
  バケツに落ちているかを集計し、「CSCが最初から拾えていない(＝自己参照方式が
  原理的に触れない)裏写りがどれだけあるか」を定量化する。

使い方:
    conda run -n diffusion python scripts/benchmark/check_bleedthrough_csc_coverage.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_synthetic_bleedthrough import (  # noqa: E402
    adjacent_source, find_split, list_pages, load_text_mask, make_synthetic_ghost,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK_ROOT = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc_mask_v2')

CONFIRMED_BOOKS = ['200003803', '200018243', '200021071']

# make_synthetic_ghost / apply_synthetic_ghostのデフォルトと合わせる
ALPHA = 0.13
BLUR_SIGMA = 2.5
VIS_THRESH_RATIO = 0.15  # apply_synthetic_ghostのgt_mask定義(max比15%)と同じ


def load_csc_buckets(book, split, page, side):
    stem = f'{book}_{page:05d}_{side}'
    d = os.path.join(MASK_ROOT, split, book, stem)
    deg_p = os.path.join(d, '04_degradation.png')
    bg_p = os.path.join(d, '05_background.png')
    txt_p = os.path.join(d, '02_text_black.png')
    if not (os.path.exists(deg_p) and os.path.exists(bg_p) and os.path.exists(txt_p)):
        return None
    deg = cv2.imread(deg_p, cv2.IMREAD_GRAYSCALE) > 127
    bg = cv2.imread(bg_p, cv2.IMREAD_GRAYSCALE) > 127
    txt = cv2.imread(txt_p, cv2.IMREAD_GRAYSCALE) < 127
    return deg, bg, txt


def analyze_book(book):
    split = find_split(book)
    pages = list_pages(book, split)
    pageset = set(pages)

    n_pages_used = 0
    total_vis_px = 0
    caught_deg_px = 0     # CSCが劣化と判定(v15が触れる)
    missed_bg_px = 0      # CSCが背景(クリーン)と誤判定(v15が触れない=原理的に除去不可)
    overlap_text_px = 0   # 既存の黒文字ストロークと重なる(文字として別途再生成される)
    other_px = 0

    per_page_rows = []

    for page, side in pages:
        src_page, src_side = adjacent_source(page, side)
        if (src_page, src_side) not in pageset:
            continue
        src_text = load_text_mask(book, split, src_page, src_side)
        buckets = load_csc_buckets(book, split, page, side)
        if src_text is None or buckets is None:
            continue
        deg, bg, txt = buckets
        H, W = deg.shape

        ghost_strength = make_synthetic_ghost((H, W), src_text, ALPHA, BLUR_SIGMA)
        if ghost_strength.max() <= 0:
            continue
        vis = ghost_strength > (ghost_strength.max() * VIS_THRESH_RATIO)
        vis_area = int(vis.sum())
        if vis_area == 0:
            continue

        caught = int((vis & deg).sum())
        missed = int((vis & bg & ~deg).sum())
        overlap_t = int((vis & txt).sum())
        other = vis_area - caught - missed - overlap_t

        n_pages_used += 1
        total_vis_px += vis_area
        caught_deg_px += caught
        missed_bg_px += missed
        overlap_text_px += overlap_t
        other_px += max(other, 0)

        per_page_rows.append((page, side, src_page, src_side, vis_area,
                              caught, missed, overlap_t))

    return dict(book=book, n_pages_used=n_pages_used, total_vis_px=total_vis_px,
                caught_deg_px=caught_deg_px, missed_bg_px=missed_bg_px,
                overlap_text_px=overlap_text_px, other_px=other_px,
                per_page_rows=per_page_rows)


def main():
    all_results = []
    for book in CONFIRMED_BOOKS:
        r = analyze_book(book)
        all_results.append(r)
        tot = r['total_vis_px']
        if tot == 0:
            print(f'{book}: 有効ページなし')
            continue
        print(f"\n=== {book} ({r['n_pages_used']}ページ、予測ghost画素={tot}) ===")
        print(f"  CSC=degradation(v15が触れる) : {r['caught_deg_px']:8d} "
             f"({100*r['caught_deg_px']/tot:5.1f}%)")
        print(f"  CSC=background(見逃し=原理的に除去不可): {r['missed_bg_px']:8d} "
             f"({100*r['missed_bg_px']/tot:5.1f}%)")
        print(f"  既存黒文字と重なる(別扱い)    : {r['overlap_text_px']:8d} "
             f"({100*r['overlap_text_px']/tot:5.1f}%)")
        print(f"  その他(赤文字等)              : {r['other_px']:8d} "
             f"({100*r['other_px']/tot:5.1f}%)")

        # 見逃し率が高い上位ページを表示
        rows = sorted(r['per_page_rows'], key=lambda x: -(x[6] / max(x[4], 1)))
        print('  見逃し率(missed/vis_area)が高いページ上位5件:')
        for page, side, sp, ss, va, c, m, o in rows[:5]:
            print(f'    p{page}_{side} (source p{sp}_{ss}): vis={va:6d} '
                 f'missed={m:6d} ({100*m/va:5.1f}%)  caught={c:6d} ({100*c/va:5.1f}%)')

    tot_all = sum(r['total_vis_px'] for r in all_results)
    if tot_all > 0:
        caught_all = sum(r['caught_deg_px'] for r in all_results)
        missed_all = sum(r['missed_bg_px'] for r in all_results)
        print(f'\n=== 3冊合計 ===')
        print(f'予測ghost画素合計: {tot_all}')
        print(f'CSC捕捉率(v15が触れる): {100*caught_all/tot_all:.1f}%')
        print(f'CSC見逃し率(自己参照方式が原理的に除去不可): {100*missed_all/tot_all:.1f}%')


if __name__ == '__main__':
    main()
