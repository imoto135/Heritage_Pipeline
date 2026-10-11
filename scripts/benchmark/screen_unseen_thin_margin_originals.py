"""New original-only thin brown trace screening, excluding all30 development books."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, sha, FONT_PATH

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'tmp_work/unseen_thin_margin_validation_20261001'
FIRST = ROOT/'tmp_work/unseen_margin_validation_20261001'
cv2.setNumThreads(1)


def score(row):
    im = cv2.imread(str(ROOT/row['original']), cv2.IMREAD_REDUCED_COLOR_4)
    assert im is not None
    H, W = im.shape[:2]; b, g, r = cv2.split(im.astype(np.float32))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    light = cv2.GaussianBlur(gray, (0, 0), 12); drop = light-gray
    neutral_dark = ((drop >= 8)&(r-b <= 45)).astype(np.uint8)
    lines = cv2.morphologyEx(neutral_dark, cv2.MORPH_OPEN, np.ones((1, max(3, round(.28*W))), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    frame_y = [int(stats[i, 1]) for i in range(1, n)
               if stats[i, 2] >= .55*W and stats[i, 3] <= .03*H and .075*H <= stats[i, 1] <= .55*H]
    rule = min(frame_y) if frame_y else None
    band = np.zeros((H, W), bool)
    if rule is not None: band[round(.04*H):max(round(.04*H), rule-8), round(.06*W):round(.94*W)] = True
    brown = band&(r-b > 12)&(r-g > 2)&(drop > 6)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(brown.astype(np.uint8), connectivity=8)
    distance = cv2.distanceTransform(brown.astype(np.uint8), cv2.DIST_L2, 5)
    peak = np.zeros(n, np.float32); np.maximum.at(peak, labels.ravel(), distance.ravel())
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if 3 <= area <= .003*H*W and min(w, h) >= 2 and 4 <= max(w, h) <= .25*min(H, W) and peak[i] <= 2.1:
            candidates.append(dict(xywh=[x, y, w, h], area=area, peak_reduced=float(peak[i])))
    candidates.sort(key=lambda v: -v['area'])
    return dict(page=row['page'], book=row['book'], split=row['split'], original=row['original'],
        score=candidates[0]['area'] if candidates else 0, candidates=candidates[:8],
        thumbnail_size=[W, H], rule_y_reduced=rule)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    first = json.loads((FIRST/'baseline_review.json').read_text())
    excluded = set(first['excluded_books'])|{r['book'] for r in first['cases']}; assert len(excluded) == 30
    reject = set(); dependencies = {}
    for path in [ROOT/'tmp_work/paper_defect_screening_20261001/visual_selection.json', FIRST/'visual_selection.json']:
        dependencies[str(path.relative_to(ROOT))] = sha(path)
        data = json.loads(path.read_text())
        review = data.get('review', data.get('decisions', []))
        if isinstance(review, dict): review = [dict(page=key, status=value) for key, value in review.items()]
        for row in review:
            if row.get('status') in ['excluded_illustration', 'excluded_cover']: reject.add(row['page'])
    inventory_path = ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv'
    inventory = list(csv.DictReader(inventory_path.open()))
    candidates = [r for r in inventory if r['book'] not in excluded and r['page'] not in reject]
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(score, candidates), 1):
            results.append(result)
            if i % 500 == 0: print('New original thin screen:', i, '/', len(candidates), flush=True)
    results.sort(key=lambda r: (-r['score'], r['page'])); selected = []; books = set()
    for row in results:
        if row['book'] in books or not row['score']: continue
        books.add(row['book']); selected.append(row)
        if len(selected) == 24: break
    dependencies[str(inventory_path.relative_to(ROOT))] = sha(inventory_path)
    dependencies[str((FIRST/'baseline_review.json').relative_to(ROOT))] = sha(FIRST/'baseline_review.json')
    dump(OUT/'screening.json', dict(original_only=True, source_sha256=sha(Path(__file__)),
        excluded_books=sorted(excluded), rejected_pages=sorted(reject), screened_pages=len(candidates),
        candidates=selected, scores_not_GT=True, visual_illustration_filter_required=True,
        frozen_candidate_sources={name: sha(ROOT/'scripts/benchmark'/name) for name in
            ['probe_thin_margin_components.py', 'probe_anchored_thin_fragments.py', 'probe_known_paper_harmonic_fill_v2.py']},
        dependencies=dependencies, training_independence_unknown=True))
    dump(OUT/'screening_scores.json', results)
    font = ImageFont.truetype(FONT_PATH, 20)
    for start in range(0, len(selected), 6):
        canvas = Image.new('RGB', (1500, 1480), (250, 248, 243)); draw = ImageDraw.Draw(canvas)
        draw.text((14, 8), '原画像だけで選ぶ薄い茶色の候補。損傷／筆記GTではない。イラストは目視で除外。', font=font, fill='black')
        details = Image.new('RGB', (1500, 1000), (250, 248, 243)); dd = ImageDraw.Draw(details)
        dd.text((14, 8), '同じ原画像の上部。マスク・復元の出力は未見。', font=font, fill='black')
        for i, row in enumerate(selected[start:start+6]):
            x, y = (i % 3)*500, 50+(i//3)*700
            draw.text((x+12, y), row['page']+' · '+str(row['score']), font=font, fill='black')
            bgr = cv2.imread(str(ROOT/row['original']), cv2.IMREAD_REDUCED_COLOR_4)
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); pil.thumbnail((476, 640)); canvas.paste(pil, (x+12, y+35))
            tx, ty = (i % 3)*500, 50+(i//3)*470
            dd.text((tx+12, ty), row['page'], font=font, fill='black')
            top = Image.fromarray(cv2.cvtColor(bgr[:round(.36*bgr.shape[0])], cv2.COLOR_BGR2RGB)); top.thumbnail((476, 420)); details.paste(top, (tx+12, ty+35))
        canvas.save(OUT/f'original_sheet_{start//6+1:02d}.png'); details.save(OUT/f'original_top_{start//6+1:02d}.png')
    (OUT/'source_snapshot.py').write_text(Path(__file__).read_text())
    print('New original thin candidates:', len(selected), 'books', flush=True)


if __name__ == '__main__': main()
