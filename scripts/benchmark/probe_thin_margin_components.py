"""Development ablation: loosen only the dark-core peak from 6 to 3.5 pixels."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_rim_lowfrequency_v2 as prior
import validate_unseen_margin_masks as new
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/thin_margin_core_pilot_20261001'
SETTINGS = dict(change='dark-core peak >=3.5 instead of >=6 only',
    other_component_gates='area128, brownseed32, size and old-band boundary unchanged',
    growth='existing 6px halo and Ldrop>=2; existing complete/rim edits retained',
    preservation='same safe domain, warm anchors, red/extra, long lines',
    full_page=True, ROI_gating=False, development_pages=18, development_ROIs=34,
    new_four_books_now_development=True, pixel_GT=False, restoration=False)


def select(connected, margin, seed, minimum_peak):
    H, W = margin.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), connectivity=8)
    seeds = np.bincount(labels.ravel(), weights=(seed&margin).ravel(), minlength=n)
    distance = cv2.distanceTransform(connected.astype(np.uint8), cv2.DIST_L2, 5)
    peak = np.zeros(n, np.float32); np.maximum.at(peak, labels.ravel(), distance.ravel())
    inner = cv2.erode(margin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    outer = np.bincount(labels.ravel(), weights=(connected&~inner).ravel(), minlength=n)
    accepted = np.zeros(n, bool); notes = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        valid = (128 <= area <= .01*H*W and seeds[i] >= 32 and peak[i] >= minimum_peak
                 and max(w, h) <= .2*min(H, W) and outer[i] == 0)
        accepted[i] = valid
        if valid:
            notes.append(dict(id=i, xywh=[x, y, w, h], area=area, seeds=int(seeds[i]),
                              peak=float(peak[i]), newly_accepted=bool(peak[i] < 6)))
    return accepted[labels]&connected, notes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [prior.OUT, new.OUT]: check_bundle(folder)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str(Path(prior.__file__).relative_to(ROOT)): sha(Path(prior.__file__)),
        str(Path(prior.MARGIN.__file__).relative_to(ROOT)): sha(Path(prior.MARGIN.__file__))}
    for folder in [prior.OUT, new.OUT]:
        for name in ['manifest.json', 'complete.json']: dependencies[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    sources = prior.MARGIN.SOURCES + [new.INPUT]
    for source in sources: dependencies[str((source/'baseline_review.json').relative_to(ROOT))] = sha(source/'baseline_review.json')
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); rows = []; pages = []
    for source in sources:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original'])
            v6 = {k: read(source/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
            features, _ = prior.MARGIN.load_features(key, original, v6)
            if source == new.INPUT:
                _, old_maps, _ = prior.MARGIN.infer(original, v6, features)
                margin = old_maps['top_margin']; connected = old_maps['dark_support']; expected = old_maps['accepted_core']
                directory = new.OUT/'pages'/key
                baseline = read(directory/'rim/edit.png', True)
                domain = read(directory/'features/safe_domain.png', True)
                guard = read(directory/'features/rim_guard.png', True)
            else:
                old_maps = prior.MARGIN.OUT/'candidate_pages'/key/'features'
                margin = read(old_maps/'top_margin.png', True); connected = read(old_maps/'dark_support.png', True)
                expected = read(old_maps/'accepted_core.png', True)
                directory = prior.OUT/'pages'/key
                baseline = read(directory/'rim_edit.png', True); domain = read(directory/'domain.png', True); guard = read(directory/'guard.png', True)
            original_core, _ = select(connected, margin, features['color_seed'], 6)
            assert np.array_equal(original_core, expected)
            relaxed, notes = select(connected, margin, features['color_seed'], 3.5)
            newly_selected = relaxed & ~original_core
            halo = cv2.dilate(newly_selected.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
            added = halo & domain & ~guard & (features['light_drop'] >= 2) & ~baseline
            edit = baseline | added
            assert not np.any(added & (guard | ~domain))
            dest = OUT/'pages'/key
            for name, value in dict(baseline_edit=baseline, edit=edit, added=added, new_core=newly_selected,
                strict_core=original_core, relaxed_core=relaxed, domain=domain, guard=guard).items(): write(dest/(name+'.png'), value)
            dump(dest/'components.json', notes)
            pages.append(dict(page=key, old_or_new='new4_now_development' if source == new.INPUT else 'old14',
                additional_pixels=int(added.sum()), newly_selected_components=sum(n['newly_accepted'] for n in notes)))
            regions = [dict(id=key+'_whole', title='ページ全体', xywh=[0, 0, original.shape[1], original.shape[0]])] + page['rois']
            for roi in regions:
                rid = roi['id']; x, y, w, h = roi['xywh']; s = (slice(y, y+h), slice(x, x+w)); crop = original[s]
                before = crop.copy(); after = crop.copy(); before[baseline[s]] = (0, 200, 255)
                after[baseline[s]] = (0, 200, 255); after[added[s]] = (255, 120, 0)
                prior.MARGIN.first.figure([crop, before, after], ['原画像', '黄＝固定の輪郭回収', '青＝薄いコアからの追加'],
                    OUT/'comparisons'/(rid+'.png'), roi['title']+' · 太さ条件だけの試作',
                    f'追加 {int(added[s].sum()):,}画素。紙面と墨を確定した正解ではない。復元なし。')
                rows.append(dict(id=rid, page=key, title=roi['title'], xywh=roi['xywh'], added_pixels=int(added[s].sum())))
                if not rid.endswith('_whole'):
                    for name, value in dict(original=crop, baseline_edit=baseline[s], edit=edit[s], added=added[s]).items():
                        write(OUT/'native'/rid/(name+'.png'), value)
            print('Thin-core ablation:', key, pages[-1], flush=True)
            del original, v6, features, margin, connected, original_core, expected, baseline, domain, guard, relaxed, added, edit
            gc.collect()
    assert len(pages) == 18 and len([r for r in rows if not r['id'].endswith('_whole')]) == 34
    for rel, digest in dependencies.items(): assert sha(ROOT/rel) == digest
    dump(OUT/'diagnostics.json', dict(pages=pages, regions=rows))
    dump(OUT/'verification.json', dict(status='passed', full_page=True, strict6_cores_reproduced=True,
        existing_edit_retained=True, guards_preserved=True, pages=18, ROIs=34, new_four_now_development=True, restoration=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.json'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: separate thin-core gate, 18 pages / 34 ROIs', flush=True)


if __name__ == '__main__': main()
