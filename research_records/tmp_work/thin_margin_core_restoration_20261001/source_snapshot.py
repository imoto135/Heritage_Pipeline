"""One mask factor: thin cores through the unchanged 16-pair paper restorer."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_thin_margin_components as mask
import probe_paper_boundary_pair_gate_v2 as fill
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = mask.ROOT
OUT = ROOT/'tmp_work/thin_margin_core_restoration_20261001'
SETTINGS = dict(mask='frozen peak3.5 thin-core addition; other gates unchanged',
    restoration='same frozen 16-pair harmonic raw/smooth and observed L seam',
    reference='baseline known paper minus newly editable pixels',
    tone_or_chroma_changes=False, reference_pair_gate_changes=False,
    fixed_ROIs=34, additional_component_ROIs=8, pages=18,
    extra_ROIs='all newly selected whole-page components; causes unclassified',
    old14_fixed_baseline='saved rim fill and saved DDRM elsewhere',
    new4_fixed_and_extra_baseline='paper-only; original outside applied',
    development=True, pixel_GT=False, DDRM_recomputed=False)


def reference_maps(page, source, directory):
    baseline = read(directory/'baseline_edit.png', True)
    domain = read(directory/'domain.png', True); guard = read(directory/'guard.png', True)
    if source == fill.new.INPUT:
        refdir = fill.new.OUT/'reference_pages'/page['page']
        known = read(refdir/'rim_paper.png', True); active = read(refdir/'rim_eligible.png', True)
    else:
        known = read(fill.frozen.prior.prior.OUT/'reference_pages'/page['page']/'paper.png', True)&~baseline
        active = baseline&domain&~guard
    return known, active


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [mask.OUT, fill.OUT]: check_bundle(folder)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        **{str((folder/name).relative_to(ROOT)): sha(folder/name)
           for folder in [mask.OUT, fill.OUT] for name in ['manifest.json', 'complete.json']}}
    for source in mask.prior.MARGIN.SOURCES+[mask.new.INPUT]:
        for pg in json.loads((source/'baseline_review.json').read_text())['cases']: dependencies[pg['original']] = sha(ROOT/pg['original'])
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    fixed = {r['id']: r for r in json.loads((fill.OUT/'diagnostics.json').read_text())}
    rows = []; poison = []
    for source in mask.prior.MARGIN.SOURCES+[mask.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original']); H, W = original.shape[:2]
            directory = mask.OUT/'pages'/key
            before = read(directory/'baseline_edit.png', True); after = read(directory/'edit.png', True)
            added = read(directory/'added.png', True); known_full, active_full = reference_maps(page, source, directory)
            assert not np.any(known_full&before)
            regions = [dict(**r, fixed=True) for r in page['rois']]
            for component in json.loads((directory/'components.json').read_text()):
                if not component['newly_accepted']: continue
                x, y, w, h = component['xywh']; x0, y0 = max(0, x-45), max(0, y-45)
                x1, y1 = min(W, x+w+45), min(H, y+h+45)
                regions.append(dict(id=key+'_component_'+str(component['id']), title='原因未確定の追加成分 '+str(component['id']),
                    xywh=[x0, y0, x1-x0, y1-y0], fixed=False, component=component))
            for roi in regions:
                rid = roi['id']; x, y, w, h = roi['xywh']
                if roi['fixed']: cx, cy, cw, ch = fixed[rid]['context_xywh']
                else:
                    cx, cy = max(0, x-64), max(0, y-64)
                    cw, ch = min(W, x+w+64)-cx, min(H, y+h+64)-cy
                cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
                context = original[cs]; crop = context[s]; lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                known = known_full[cs]; active = active_full[cs]; local_added = added[cs]
                if roi['fixed']:
                    native = fill.OUT/'native'/rid
                    assert np.array_equal(known, read(native/'reference_context.png', True))
                    assert np.array_equal(active, read(native/'eligible_context.png', True))
                    base = read(native/'base.png'); old_applied = read(native/'old_applied_context.png', True)
                else:
                    original_result = fill.field_for(lab, known, active, 16)
                    old_applied = original_result[1]; base = fill.frozen.compose(original_result[4], old_applied, crop, s)
                dest = OUT/'native'/rid
                new_known = known&~local_added; new_active = active|local_added
                if local_added.any() and new_active[s].any():
                    result = fill.field_for(lab, new_known, new_active, 16)
                    fields, applied, used, notes, field, positions, qnotes, seam, average, sx, sy = result
                    value = fill.frozen.compose(field, applied, base, s)
                    dest.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest/'harmonic_fields_Lab.npz', **fields)
                    for name, item in dict(result_field_Lab=field, texture_seam=seam, texture_average=average, source_x=sx, source_y=sy).items():
                        np.save(dest/(name+'.npy'), item, allow_pickle=False)
                    dump(dest/'donors.json', dict(context_xywh=[cx, cy, cw, ch], patch_xy=positions, components=qnotes))
                    if applied[s].any():
                        poisoned = lab.copy(); poisoned[~new_known] = (250, -113, 190)
                        repeat = fill.field_for(poisoned, new_known, new_active, 16)
                        assert np.array_equal(field, repeat[4]) and np.array_equal(applied, repeat[1]); poison.append(rid)
                else:
                    value = base.copy(); applied = old_applied.copy(); used = np.zeros_like(known); notes = []
                assert not np.any(np.any(value != base, axis=2)&~applied[s])
                assert not np.any(np.any(value != crop, axis=2)&~after[cs][s])
                for name, item in dict(original=crop, base=base, thin16=value, edit=after[cs][s], added=local_added[s],
                    reference_context=new_known, eligible_context=new_active, applied_context=applied, old_applied_context=old_applied,
                    used_context=used).items(): write(dest/(name+'.png'), item)
                dump(dest/'components.json', notes)
                newly_filled = applied[s]&~old_applied[s]
                figure([crop, base, value], ['原画像', '太いコア・境界16組', '薄いコア追加・境界16組'],
                    OUT/'comparisons'/(rid+'.png'), roi['title']+' · 太さだけ変更して同じ紙面復元',
                    f'追加編集 {int(local_added[s].sum()):,}・追加補間 {int(newly_filled.sum()):,}画素。原因のGTなし。',
                    '紙色・境界16組・観測明度模様は従来どおり。分布紙色・色模様の試作は加えない。')
                rows.append(dict(id=rid, page=key, title=roi['title'], fixed_ROI=roi['fixed'], xywh=roi['xywh'],
                    context_xywh=[cx, cy, cw, ch], additional_edit=int(local_added[s].sum()),
                    newly_filled=int(newly_filled.sum()), filled_additional_edit=int((applied[s]&local_added[s]).sum()),
                    total_applied=int(applied[s].sum()), old_applied=int(old_applied[s].sum()),
                    changed_vs_base=int(np.any(value != base, axis=2).sum()),
                    component=roi.get('component'), base_elsewhere='saved_DDRM' if roi['fixed'] and source != mask.new.INPUT else 'original'))
                print('Thin mask to paper:', len(rows), '/42', rid, 'new fill', int(newly_filled.sum()), flush=True)
            del original, before, after, added, known_full, active_full; gc.collect()
    assert len(rows) == 42 and sum(r['fixed_ROI'] for r in rows) == 34
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', rows)
    dump(OUT/'verification.json', dict(status='passed', ROIs=42, old_fixed_ROIs=34, new_components=8,
        full_page_reference_consistent=True, all_nonreference_poison_cases=poison,
        outside_applied_changes=0, outside_edit_changes=0, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: thin mask into unchanged paper restorer, 42 ROIs', flush=True)


if __name__ == '__main__': main()
