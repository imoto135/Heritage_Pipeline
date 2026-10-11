"""Recover small brown fragments only near already accepted thin components."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import restore_thin_margin_core as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/anchored_thin_fragment_pilot_20261001'
RADII = [8, 16]
SETTINGS = dict(baseline='frozen thin3.5 mask + unchanged 16-pair paper restorer',
    fragment_gate='area16..127, brownseed>=8, peak>=2, no margin-boundary contact, same size gate',
    anchor='frozen newly accepted thin dark cores; one step only, no recursive chaining',
    maximum_distance_px=RADII, growth='same halo6, Ldrop>=2, safe domain and old guards',
    full_page=True, ROI_label_gating=False, fixed_pages=18, ROIs=42,
    base_ambiguity='earlier thin-core stroke-like false/uncertain additions remain; not globally adopted',
    restoration='same raw/smooth harmonic + observed L seam, boundary>=16',
    tone_chroma_and_pair_gate_changes=False, development=True, pixel_GT=False, DDRM_recomputed=False)


def masks_for_page(page, source, original):
    key = page['page']; previous = prior.mask.OUT/'pages'/key
    base = read(previous/'edit.png', True); anchor = read(previous/'new_core.png', True)
    domain = read(previous/'domain.png', True); guard = read(previous/'guard.png', True)
    if not anchor.any(): return {r: dict(edit=base.copy(), added=np.zeros_like(base), fragment_core=np.zeros_like(base), notes=[]) for r in RADII}
    v6 = {k: read(source/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
    features, _ = prior.mask.prior.MARGIN.load_features(key, original, v6)
    _, maps, _ = prior.mask.prior.MARGIN.infer(original, v6, features)
    connected = maps['dark_support']; margin = maps['top_margin']
    n, labels, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), connectivity=8)
    seeds = np.bincount(labels.ravel(), weights=(features['color_seed']&margin).ravel(), minlength=n)
    dt = cv2.distanceTransform(connected.astype(np.uint8), cv2.DIST_L2, 5)
    peak = np.zeros(n, np.float32); np.maximum.at(peak, labels.ravel(), dt.ravel())
    away = cv2.distanceTransform((~anchor).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    gap = np.full(n, np.inf); np.minimum.at(gap, labels.ravel(), away.ravel())
    inner = cv2.erode(margin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary = np.bincount(labels.ravel(), weights=(connected&~inner).ravel(), minlength=n)
    results = {}; H, W = base.shape
    for radius in RADII:
        selected = np.zeros(n, bool); notes = []
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            valid = 16 <= area < 128 and seeds[i] >= 8 and peak[i] >= 2 and 0 < gap[i] <= radius and boundary[i] == 0 and max(w, h) <= .2*min(H, W)
            if valid:
                selected[i] = True
                notes.append(dict(id=i, xywh=[x, y, w, h], area=area, brown_seeds=int(seeds[i]), peak=float(peak[i]), distance=float(gap[i])))
        core = selected[labels]&connected
        halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        added = halo&domain&~guard&(features['light_drop'] >= 2)&~base
        results[radius] = dict(edit=base|added, added=added, fragment_core=core, notes=notes)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(prior.OUT); check_bundle(prior.mask.OUT)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        **{str((folder/name).relative_to(ROOT)): sha(folder/name)
           for folder in [prior.OUT, prior.mask.OUT] for name in ['manifest.json', 'complete.json']}}
    sources = prior.mask.prior.MARGIN.SOURCES+[prior.mask.new.INPUT]
    for source in sources:
        for pg in json.loads((source/'baseline_review.json').read_text())['cases']: dependencies[pg['original']] = sha(ROOT/pg['original'])
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    old_rows = json.loads((prior.OUT/'diagnostics.json').read_text()); rows = []; pages = []; poison = []
    for source in sources:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original']); masks = masks_for_page(page, source, original)
            before = read(prior.mask.OUT/'pages'/key/'edit.png', True); full_dir = OUT/'pages'/key
            page_record = dict(page=key, diagnostics={})
            for radius in RADII:
                mode = 'near'+str(radius); data = masks[radius]
                for name in ['edit', 'added', 'fragment_core']: write(full_dir/mode/(name+'.png'), data[name])
                dump(full_dir/mode/'fragments.json', data['notes'])
                page_record['diagnostics'][mode] = dict(added=int(data['added'].sum()), fragments=data['notes'])
            assert not np.any(masks[8]['edit']&~masks[16]['edit'])
            full_images = [original]
            for radius in [0]+RADII:
                show = original.copy(); show[before] = (0, 200, 255)
                if radius: show[masks[radius]['added']] = (255, 120, 0)
                full_images.append(show)
            figure(full_images, ['原画像', '黄＝薄いコアまで', '青＝近接8px追加', '青＝近接16px追加'],
                OUT/'whole_page_masks'/(key+'.png'), key+' · 小成分の近接条件を全ページで比較',
                '小成分の単独受理はしない。既存の筆画状の曖昧な候補も保持。復元なし。',
                '既存の保護・安全領域を維持。受理済み薄いコアへ一段だけ接続する。')
            pages.append(page_record)
            for old in [r for r in old_rows if r['page'] == key]:
                rid = old['id']; x, y, w, h = old['xywh']; cx, cy, cw, ch = old['context_xywh']
                cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
                context = original[cs]; crop = context[s]; lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                native = prior.OUT/'native'/rid; base = read(native/'thin16.png')
                known = read(native/'reference_context.png', True); active = read(native/'eligible_context.png', True)
                old_applied = read(native/'applied_context.png', True); outputs = {}; diagnostics = {}; dest = OUT/'native'/rid
                for radius in RADII:
                    mode = 'near'+str(radius); added = masks[radius]['added'][cs]
                    new_known = known&~added; new_active = active|added; md = dest/mode
                    if added.any() and new_active[s].any():
                        result = prior.fill.field_for(lab, new_known, new_active, 16)
                        fields, applied, used, notes, field, positions, qnotes, seam, average, sx, sy = result
                        value = prior.fill.frozen.compose(field, applied, base, s)
                        md.mkdir(parents=True, exist_ok=True); np.savez_compressed(md/'harmonic_fields_Lab.npz', **fields)
                        for name, item in dict(result_field_Lab=field, texture_seam=seam, texture_average=average, source_x=sx, source_y=sy).items():
                            np.save(md/(name+'.npy'), item, allow_pickle=False)
                        dump(md/'donors.json', dict(context_xywh=old['context_xywh'], patch_xy=positions, components=qnotes))
                        if applied[s].any():
                            poisoned = lab.copy(); poisoned[~new_known] = (240, -220, 132)
                            repeat = prior.fill.field_for(poisoned, new_known, new_active, 16)
                            assert np.array_equal(field, repeat[4]) and np.array_equal(applied, repeat[1]); poison.append(dict(id=rid, mode=mode))
                    else: value = base.copy(); applied = old_applied.copy(); used = np.zeros_like(known); notes = []
                    assert not np.any(np.any(value != base, axis=2)&~applied[s])
                    assert not np.any(np.any(value != crop, axis=2)&~masks[radius]['edit'][cs][s])
                    for name, item in dict(output=value, edit=masks[radius]['edit'][cs][s], added=added[s], reference_context=new_known,
                        eligible_context=new_active, applied_context=applied, used_context=used).items(): write(md/(name+'.png'), item)
                    dump(md/'components.json', notes); outputs[mode] = value
                    diagnostics[mode] = dict(added_edit=int(added[s].sum()), filled_added=int((applied[s]&added[s]).sum()),
                        newly_filled=int((applied[s]&~old_applied[s]).sum()), changed_vs_base=int(np.any(value != base, axis=2).sum()))
                for name, item in dict(original=crop, base=base, old_applied_context=old_applied).items(): write(dest/(name+'.png'), item)
                figure([crop, base, outputs['near8'], outputs['near16']],
                    ['原画像', '薄いコア・前回', '小成分接続8px', '小成分接続16px'], OUT/'comparisons'/(rid+'.png'),
                    old['title']+' · 受理コアに近い小成分の補完',
                    '同じ境界16組・紙色・明度模様。距離条件だけを比較。',
                    '既存の曖昧な曲線は除去の正解と扱わない。原画像を全42ROIに並べる。')
                rows.append(dict(**old, fragment_diagnostics=diagnostics)); print('Anchored fragment:', len(rows), '/42', rid, flush=True)
            del original, masks, before; gc.collect()
    assert len(rows) == 42 and len(pages) == 18
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', dict(pages=pages, regions=rows))
    dump(OUT/'verification.json', dict(status='passed', pages=18, ROIs=42, distance_levels=RADII,
        no_recursive_chaining=True, outside_applied_changes=0, outside_edit_changes=0,
        all_nonreference_poison_cases=poison, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: anchored small brown fragments, 18 pages /42 ROIs', flush=True)


if __name__ == '__main__': main()
