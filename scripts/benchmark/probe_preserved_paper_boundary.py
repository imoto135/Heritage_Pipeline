"""Bounded seam color correction from preserved light margin, not paper GT."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_color_reference_gate_v2 as references
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = references.ROOT
BASE = references.prior.OUT
OUT = ROOT/'tmp_work/preserved_paper_boundary_pilot_20261001'
RADII = [2, 4]
LIMITS = np.array([3., 1., 1.5], np.float32)
SETTINGS = dict(mask_applied_observed_texture_fixed=True, baseline='saved thin+8px fragments, original16-pair paper fill',
    added_color_sources='preserved outside ENTIRE edit, inside safe domain and outside guards; median L>=50, Ldrop[-6,8], |a-paper|<=3, |b-paper|<=6',
    uncertainty='added sources may include light stain or v6 hold; not clean-paper ground truth',
    operation='difference between broad-source and old-source raw/smooth harmonic color fields',
    bounds_Lab=LIMITS.tolist(), boundary_feather_px=RADII, boundary_support='distance<=1+3radius',
    original_damage_inside_edit_used=False, original_DDRM_guidance=False,
    outside_original_and_old_texture_preserved=True, ROIs=42, pages=18,
    classification_changes=False, development=True, pixel_GT=False, DDRM_recomputed=False)


def boundary_field(fields, applied):
    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
    return fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:, :, None]


def main():
    check_bundle(BASE)
    rows = json.loads((BASE/'diagnostics.json').read_text())['regions']; pages = []
    for source in references.prior.prior.mask.prior.MARGIN.SOURCES+[references.prior.prior.mask.new.INPUT]:
        pages += json.loads((source/'baseline_review.json').read_text())['cases']
    page_map = {p['page']: p for p in pages}; textures = {}; dependencies = {p['original']: sha(ROOT/p['original']) for p in pages}
    fill = references.prior.prior.fill
    # Reviewable preflight: reproduce baseline with its fixed saved donor signal.
    current = None
    for row in rows:
        rid = row['id']; key = row['page']; native = BASE/'native'/rid/'near8'
        if current != key: original = read(ROOT/page_map[key]['original']); current = key
        x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
        applied = read(native/'applied_context.png', True); known = read(native/'reference_context.png', True)
        source, seam = references.texture_source(rid, applied)
        textures[rid] = str(source.relative_to(ROOT)) if source else None
        if source:
            for name in ['texture_seam.npy', 'source_x.npy', 'source_y.npy', 'donors.json']: dependencies[str((source/name).relative_to(ROOT))] = sha(source/name)
        if applied[s].any():
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            fields, actual, *_ = fill.solve_fields(lab, known, applied, 16)
            assert np.array_equal(actual, applied)
            field = boundary_field(fields, applied)
            distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
            field[:, :, 0] += seam*np.clip((distance-1)/4., 0, 1)*applied
            base = read(native/'output.png')
            assert np.array_equal(fill.frozen.compose(field, applied, base, s), base)
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for name in ['manifest.json', 'complete.json']: dependencies[str((BASE/name).relative_to(ROOT))] = sha(BASE/name)
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); dump(OUT/'texture_sources.json', textures)
    results = []; poison = []
    for pg in pages:
        key = pg['page']; original = read(ROOT/pg['original'])
        median_lab = cv2.cvtColor(cv2.medianBlur(original, 3).astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        paper = references.prior.prior.mask.prior.MARGIN.first.paper_estimate(median_lab)
        drop = paper[:, :, 0]-median_lab[:, :, 0]; difference = np.abs(median_lab[:, :, 1:]-paper[:, :, 1:])
        mask_dir = BASE/'pages'/key/'near8'; full_edit = read(mask_dir/'edit.png', True)
        guard_dir = references.prior.prior.mask.OUT/'pages'/key
        domain = read(guard_dir/'domain.png', True); guard = read(guard_dir/'guard.png', True)
        available = ~full_edit&domain&~guard&(median_lab[:, :, 0] >= 50)&(drop >= -6)&(drop <= 8)&np.all(difference <= [3., 6.], axis=2)
        for row in [r for r in rows if r['page'] == key]:
            rid = row['id']; x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
            cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
            context = original[cs]; crop = context[s]; native = BASE/'native'/rid/'near8'; dest = OUT/'native'/rid
            applied = read(native/'applied_context.png', True); known = read(native/'reference_context.png', True)
            broad = known|available[cs]; assert not np.any(broad&full_edit[cs])
            base = read(native/'output.png'); outputs = {}; diagnostics = {}
            distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
            if applied[s].any():
                lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                old_fields, actual, used_old, old_notes = fill.solve_fields(lab, known, applied, 16)
                new_fields, actual_new, used_new, new_notes = fill.solve_fields(lab, broad, applied, 16)
                assert np.array_equal(actual, applied) and np.array_equal(actual_new, applied)
                old_boundary = boundary_field(old_fields, applied); new_boundary = boundary_field(new_fields, applied)
                source, seam = references.texture_source(rid, applied)
                base_field = old_boundary.copy(); base_field[:, :, 0] += seam*np.clip((distance-1)/4., 0, 1)*applied
                delta = np.clip(new_boundary-old_boundary, -LIMITS, LIMITS)*applied[:, :, None]
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/'baseline_fields_Lab.npz', **old_fields)
                np.savez_compressed(dest/'broad_fields_Lab.npz', **new_fields)
                np.save(dest/'base_field_Lab.npy', base_field, allow_pickle=False)
                np.save(dest/'bounded_delta_Lab.npy', delta, allow_pickle=False)
                write(dest/'used_baseline.png', used_old); write(dest/'used_broad.png', used_new)
                dump(dest/'components_broad.json', new_notes)
                changed = lab.copy(); changed[~broad] = (260, 180, -195)
                repeat = fill.solve_fields(changed, broad, applied, 16)
                for name in new_fields: assert np.array_equal(new_fields[name], repeat[0][name])
                poison.append(rid)
            for radius in RADII:
                mode = 'band'+str(radius); value = base.copy()
                if applied[s].any():
                    weight = np.exp(-(np.maximum(distance-1, 0)/radius)**2)*applied*(distance <= 1+3*radius)
                    correction = delta*weight[:, :, None]
                    field = base_field+correction; value = fill.frozen.compose(field, applied, base, s)
                    md = dest/mode; md.mkdir(parents=True, exist_ok=True)
                    np.save(md/'correction_Lab.npy', correction, allow_pickle=False)
                    np.save(md/'result_field_Lab.npy', field, allow_pickle=False)
                assert not np.any(np.any(value != base, axis=2)&~applied[s])
                assert not np.any(np.any(value != crop, axis=2)&~full_edit[cs][s])
                outputs[mode] = value; diagnostics[mode] = dict(changed_vs_base=int(np.any(value != base, axis=2).sum()))
                write(dest/mode/'output.png', value)
            for name, item in dict(original=crop, base=base, edit=full_edit[cs][s], applied_context=applied,
                baseline_reference_context=known, broad_color_context=broad).items(): write(dest/(name+'.png'), item)
            figure([crop, base]+[outputs['band'+str(r)] for r in RADII],
                ['原画像', '既知紙色のみ・前回', '保存境界色・2px帯', '保存境界色・4px帯'], OUT/'comparisons'/(rid+'.png'),
                row['title']+' · 編集外に残す明るい境界色との連続性',
                '補正はLab±3/1/1.5以内。模様・マスク・補間範囲は固定。',
                '追加参照は未編集の明るい余白。薄い汚れを含む可能性があり、紙面のGTではない。')
            results.append(dict(**row, preserved_boundary_diagnostics=diagnostics,
                additional_color_sources=int((broad&~known).sum()),
                newly_used_boundary_sources=int((used_new&~known).sum()) if applied[s].any() else 0))
            print('Preserved light boundary:', len(results), '/42', rid, flush=True)
        del original, median_lab, paper, drop, difference, full_edit, domain, guard, available; gc.collect()
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', results)
    dump(OUT/'verification.json', dict(status='passed', baseline_reproduced=True, ROIs=42, variants=84,
        outside_applied_changes=0, outside_edit_changes=0, texture_mask_applied_fixed=True,
        all_nonreference_poison_cases=poison, pixel_GT=False, uncertain_added_color_sources=True))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: bounded preserved-boundary color only, 84 variants', flush=True)


if __name__ == '__main__': main()
