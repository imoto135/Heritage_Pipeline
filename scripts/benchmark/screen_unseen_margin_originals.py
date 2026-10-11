"""Original-only screening for new books; no npj/v6/mask results are inspected."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, FONT_PATH, sha
import probe_paper_rim_lowfrequency_v2 as prior

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/unseen_margin_validation_20261001'
cv2.setNumThreads(1)


def score(row):
    original = cv2.imread(str(ROOT/row['original']), cv2.IMREAD_REDUCED_COLOR_8)
    assert original is not None
    H, W = original.shape[:2]
    b, g, r = cv2.split(original.astype(np.float32))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    light = cv2.GaussianBlur(gray, (0, 0), 8)
    band = np.zeros((H, W), bool); band[round(.025*H):round(.32*H), round(.06*W):round(.94*W)] = True
    brown = band & (r-b > 15) & (r-g > 3) & (light-gray > 10)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(brown.astype(np.uint8))
    blobs = []
    for k in range(1, n):
        x, y, w, h, area = map(int, stats[k])
        if area >= 10 and min(w, h) >= 4 and max(w, h) <= .3*min(H, W) and area/(w*h) >= .25:
            blobs.append(dict(xywh=[x, y, w, h], area=area))
    blobs.sort(key=lambda x: -x['area'])
    return dict(page=row['page'], book=row['book'], original=row['original'],
                split=row['split'], score=blobs[0]['area'] if blobs else 0,
                thumbnail_xywh=blobs[0]['xywh'] if blobs else None, thumbnail_size=[W, H])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = set()
    for source in prior.MARGIN.SOURCES:
        excluded.update(r['book'] for r in json.loads((source/'baseline_review.json').read_text())['cases'])
    for folder in ['warm_ink_hold_v6_20260930', 'warm_ink_hold_other_books_20260930']:
        path = ROOT/'tmp_work'/folder/'manifest.json'
        if path.exists(): excluded.update(r['book'] for r in json.loads(path.read_text()).get('cases', []) if 'book' in r)
    rejected = json.loads((ROOT/'tmp_work/paper_defect_screening_20261001/visual_selection.json').read_text())['review']
    rejected_pages = {r['page'] for r in rejected if r['status'] in ['excluded_illustration', 'excluded_cover']}
    inventory = list(csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open()))
    candidates = [r for r in inventory if r['book'] not in excluded and r['page'] not in rejected_pages]
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(score, candidates), 1):
            results.append(row)
            if i % 500 == 0: print('Unseen originals:', i, '/', len(candidates), flush=True)
    results.sort(key=lambda r: (-r['score'], r['page']))
    selected = []; books = set()
    for row in results:
        if row['book'] in books or row['score'] == 0: continue
        books.add(row['book']); selected.append(row)
        if len(selected) == 18: break
    dump(OUT/'screening.json', dict(excluded_books=sorted(excluded), screened_pages=len(candidates),
        candidates=selected, source_sha256=sha(Path(__file__)),
        original_only=True, scores_not_damage_GT=True, visual_illustration_filter_required=True,
        training_independence_unknown=True, frozen_rim_source_sha256=sha(Path(prior.__file__))))
    dump(OUT/'screening_scores.json', results)
    font = ImageFont.truetype(FONT_PATH, 20)
    for start in range(0, len(selected), 6):
        canvas = Image.new('RGB', (1500, 1460), (250, 248, 243)); draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), '原画像だけの候補確認。損傷・インクの正解ではない。イラストは目視で除外。', font=font, fill='black')
        for k, row in enumerate(selected[start:start+6]):
            x, y = (k % 3)*500, 50+(k//3)*700
            draw.text((x+12, y), row['page'], font=font, fill='black')
            draw.text((x+12, y+26), '元画像の上部・茶色候補 '+str(row['score']), font=font, fill='black')
            bgr = cv2.imread(str(ROOT/row['original']), cv2.IMREAD_REDUCED_COLOR_4)
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); im.thumbnail((476, 630))
            canvas.paste(im, (x+12, y+60))
        canvas.save(OUT/f'original_sheet_{start//6+1:02d}.png')
    print('Original-only candidates:', len(selected), 'books; excluded', len(excluded), flush=True)


if __name__ == '__main__': main()
