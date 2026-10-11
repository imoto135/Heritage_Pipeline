"""Freeze three text pages and12 original-only ROIs before mask computation."""
import csv
import json
from pathlib import Path
import shutil

from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'tmp_work/unseen_thin_margin_validation_20261001'
SELECTION = {
    '200003076_00054_1': [
        ('fresh_page_edge', '上余白右端の紙端・折れ：損傷GTではない', [1950, 70, 330, 650]),
        ('fresh_margin_flecks', '上余白の小さな茶色い跡・原因未確定', [530, 140, 470, 300]),
        ('fresh_blank_margin', '対照：上余白の紙面', [1030, 140, 650, 270]),
        ('fresh_dark_body', '対照：濃い本文', [400, 650, 650, 850])],
    '200021853_00014_2': [
        ('fresh_header_ink', '対照：上余白に重なる文字と枠', [880, 160, 340, 210]),
        ('fresh_left_paper', '上余白左の薄い斑点・原因未確定', [80, 100, 400, 200]),
        ('fresh_mid_paper', '対照：見出し左の紙面', [540, 100, 340, 200]),
        ('fresh_body_ink', '対照：黒い本文', [100, 460, 550, 720])],
    '200021644_00003_2': [
        ('fresh_small_top_trace', '上余白の小さな薄い跡・原因未確定', [470, 80, 420, 240]),
        ('fresh_red_seal', '対照：上余白の朱印', [1550, 60, 340, 330]),
        ('fresh_blank_top', '対照：朱印左の紙面', [930, 70, 420, 270]),
        ('fresh_gray_body', '対照：灰色の本文', [450, 500, 700, 950])]}


def main():
    screen = json.loads((OUT/'screening.json').read_text())
    assert sha(OUT/'source_snapshot.py') == screen['source_sha256']
    assert not (OUT/'v6_masks').exists(), 'Selection must precede all new mask outputs'
    for relative, digest in screen['dependencies'].items(): assert sha(ROOT/relative) == digest
    inventory = {r['page']: r for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
    cases = []; inputs = {}
    review = {'200003076_00054_1': 'selected_text_page', '200021853_00014_2': 'selected_text_page',
        '200021644_00003_2': 'selected_text_page', '200014685_00009_2': 'excluded_illustration',
        'umgy007_011': 'excluded_illustration', '200015843_00103_1': 'excluded_illustration',
        '200021637_00019_2': 'excluded_decorative_background_for_main_evaluation'}
    for key, specs in SELECTION.items():
        page = inventory[key].copy(); assert page['book'] not in screen['excluded_books']
        image = read(ROOT/page['original']); H, W = image.shape[:2]; rois = []
        for rid, title, xywh in specs:
            x, y, w, h = xywh; assert 0 <= x < x+w <= W and 0 <= y < y+h <= H
            rois.append(dict(id=rid, title=title, xywh=xywh, baseline_failure=None))
            figure([image[y:y+h, x:x+w]], ['原画像・出力を見る前に固定'], OUT/'original_ROIs'/(rid+'.png'),
                title, '物理原因・損傷／筆記の画素GTなし。選定後に条件固定で分類・復元を適用。', '原画像のみ。新しいnpj/v6比較・候補マスク・復元の出力は未見。')
        page.update(rois=rois, contains_illustration=False, baseline_failure=None,
            selection_reason='original-only thin/brown trace screening; upper-margin writing/seal and body controls; no confirmed damage GT')
        cases.append(page)
        for k in ['original', 'npj_input', 'npj_black', 'npj_red', 'npj_degradation', 'npj_background']:
            inputs[page[k]] = sha(ROOT/page[k])
    data = dict(cases=cases, selection_before_new_v6_and_candidate=True, original_only=True,
        selected_pages=3, selected_books=3, selected_ROIs=12, screened_pages=screen['screened_pages'],
        excluded_books=screen['excluded_books'], pixel_GT=False, training_independence_unknown=True,
        parameters_tuned_after_new_outputs=False,
        frozen_rim_source_sha256=sha(ROOT/'scripts/benchmark/probe_paper_rim_lowfrequency_v2.py'),
        frozen_candidate_sources=screen['frozen_candidate_sources'],
        candidate='same v6 -> original margin/top-completion/rim -> core3.5 -> one-step near8 brown fragments',
        restoration='unchanged16-pair harmonic color + saved known-paper observed L seam; no tone/chroma/coarse variants',
        illustration_and_decorative_filter=review, inputs=inputs)
    dump(OUT/'baseline_review.json', data)
    dump(OUT/'visual_selection.json', dict(review=review, selection_sha256=sha(OUT/'baseline_review.json'),
        no_true_wormhole_or_tear_confirmed=True, upper_margin_actual_writing_control=True))
    stack([OUT/'original_details'/(key+'.png') for key in SELECTION], OUT/'selected_originals.png',
        '次の条件固定検証：原画像から先に選んだ3冊',
        ['紙端・小さな跡と上余白筆記・朱印・本文の対照。損傷の画素GTではない。', '3つのイラストページと1つの装飾背景ページを主評価から除外した。'])
    shutil.copyfile(__file__, OUT/'selection_runner_snapshot.py')
    dump(OUT/'selection_complete.json', dict(source_sha256=sha(Path(__file__)),
        baseline_review_sha256=sha(OUT/'baseline_review.json'), original_before_outputs=True,
        outputs={str(p.relative_to(OUT)): sha(p) for folder in ['original_ROIs', 'original_details'] for p in (OUT/folder).glob('*.png')},
        candidate_code_hashes=data['frozen_candidate_sources']))
    print('Original-only selection frozen:3 new books,12 ROIs', flush=True)


if __name__ == '__main__': main()
