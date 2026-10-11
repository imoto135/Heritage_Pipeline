"""New-book diagnostic: the same paper-only restoration for four fixed masks.

Unapplied pixels retain the original image. These are not DDRM outputs.
"""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_rim_lowfrequency_v2 as frozen
import validate_unseen_margin_masks as masks
from build_npj_v6_comparison import dump, read, sha, write
from collect_saved_npj_failures import saved_masks
from package_paper_blending_review import check_bundle

ROOT = frozen.ROOT
INPUT = masks.INPUT
OUT = INPUT/'paper_fill'
MODES = ['npj', 'v6', 'completed', 'rim']
SETTINGS = dict(mask_modes=MODES, scope='safe upper domain minus frozen warm anchors and long lines',
    restorer='frozen harmonic raw/smooth boundary mixture plus observed L texture with minimum-error seams',
    reference='unchanged paper Lab gate; each full mask excluded',
    outside_applied='original image, including unapplied edits; no saved or recomputed DDRM',
    npj='saved npj edit mask through the same conservative paper scope; not original npj pipeline',
    parameters_tuned_after_new_outputs=False, training=False, DDRM=False, pixel_GT=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(masks.OUT)
    selected = json.loads((INPUT/'baseline_review.json').read_text())
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str(Path(frozen.__file__).relative_to(ROOT)): sha(Path(frozen.__file__)),
        str(Path(frozen.prior.__file__).relative_to(ROOT)): sha(Path(frozen.prior.__file__)),
        str(Path(frozen.prior.prior.__file__).relative_to(ROOT)): sha(Path(frozen.prior.prior.__file__)),
        str((INPUT/'baseline_review.json').relative_to(ROOT)): sha(INPUT/'baseline_review.json')}
    for name in ['manifest.json', 'complete.json']: dependencies[str((masks.OUT/name).relative_to(ROOT))] = sha(masks.OUT/name)
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)), pages=4, regions=12)
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    rows = []; poison = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original']); H, W = original.shape[:2]
        directory = masks.OUT/'pages'/key
        v6 = {k: read(directory/'v6'/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
        npj = saved_masks(page)
        domain = read(directory/'features/safe_domain.png', True)
        guard = read(directory/'features/rim_guard.png', True)
        values = {'npj': dict(edit=npj['edit'], protect=npj['black'], red=npj['red'], hold=np.zeros((H, W), bool)), 'v6': v6}
        values.update({m: {k: read(directory/m/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red']} for m in ['completed', 'rim']})
        references = {}; eligibility = {}
        for mode, maps in values.items():
            edit = maps['edit']; protected = maps['protect'] | maps['hold'] | maps['red'] | guard
            known = frozen.prior.prior.previous.blending.previous.clean_reference(original, edit, protected&~edit, domain&~guard)[1]
            active = edit & domain & ~guard
            references[mode] = known; eligibility[mode] = active
            write(OUT/'reference_pages'/key/(mode+'_paper.png'), known)
            write(OUT/'reference_pages'/key/(mode+'_eligible.png'), active)
            assert not np.any(known & edit)
        for roi in page['rois']:
            rid = roi['id']; x, y, w, h = roi['xywh']; x0, y0 = max(0, x-64), max(0, y-64)
            x1, y1 = min(W, x+w+64), min(H, y+h+64)
            cs = (slice(y0, y1), slice(x0, x1)); s = (slice(y-y0, y-y0+h), slice(x-x0, x-x0+w))
            context = original[cs]; crop = context[s]; lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            write(OUT/'native'/rid/'original.png', crop)
            outputs = {}; stats = {}; cache = []
            for mode in MODES:
                edit = values[mode]['edit'][cs]; known = references[mode][cs]; active = eligibility[mode][cs]
                local = OUT/'native'/rid/mode; local.mkdir(parents=True, exist_ok=True)
                reused = next((item for item in cache if np.array_equal(item['known'], known) and np.array_equal(item['active'], active)), None)
                if reused is not None:
                    fields, applied, used, notes, result, positions, qnotes, seam, average, sx, sy = reused['data']
                elif active[s].any():
                    fields, applied, used, notes = frozen.prior.prior.solve_fields(lab, known, active)
                    if applied.any():
                        blended, positions, qnotes, seam, average, sx, sy = frozen.prior.infer(lab, known, applied,
                            fields['boundary_raw'], fields['boundary_smooth'])
                        result = blended['combined']
                    else:
                        result = np.zeros_like(lab); positions = []; qnotes = []
                        seam = np.zeros(active.shape, np.float32); average = seam.copy()
                        sx = np.full(active.shape, -1, np.int32); sy = sx.copy()
                else:
                    fields = {m: np.zeros_like(lab) for m in ['boundary_raw', 'boundary_smooth']}
                    applied = np.zeros_like(active); used = np.zeros_like(active); notes = []
                    result = np.zeros_like(lab); positions = []; qnotes = []
                    seam = np.zeros(active.shape, np.float32); average = seam.copy()
                    sx = np.full(active.shape, -1, np.int32); sy = sx.copy()
                data = (fields, applied, used, notes, result, positions, qnotes, seam, average, sx, sy)
                cache.append(dict(known=known, active=active, data=data))
                value = frozen.compose(result, applied, crop, s); outputs[mode] = value
                assert not np.any(np.any(value != crop, axis=2) & ~applied[s])
                assert not np.any(applied & ~active)
                assert not np.any(np.any(value != crop, axis=2) & ~edit[s])
                if applied[s].any() and reused is None:
                    changed = context.copy(); changed[edit] = (213, 3, 19)
                    plab = cv2.cvtColor(changed.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                    pf, pa, _, _ = frozen.prior.prior.solve_fields(plab, known, active)
                    assert np.array_equal(applied, pa)
                    pb = frozen.prior.infer(plab, known, pa, pf['boundary_raw'], pf['boundary_smooth'])[0]['combined']
                    assert np.array_equal(result, pb)
                    poison.append(dict(id=rid, mode=mode))
                for name, item in dict(output=value, edit=edit[s], eligible=active[s], applied=applied[s],
                    reference_context=known, eligible_context=active, applied_context=applied, used_context=used).items():
                    write(local/(name+'.png'), item)
                if applied.any():
                    np.savez_compressed(local/'harmonic_fields_Lab.npz', **fields)
                    for name, item in dict(result_field_Lab=result, texture_seam=seam, texture_average=average, source_x=sx, source_y=sy).items():
                        np.save(local/(name+'.npy'), item, allow_pickle=False)
                    dump(local/'donors.json', dict(context_xywh=[x0, y0, x1-x0, y1-y0], patch_xy=positions, components=qnotes))
                dump(local/'components.json', notes)
                stats[mode] = dict(edit_pixels=int(edit[s].sum()), eligible_pixels=int(active[s].sum()),
                    applied_pixels=int(applied[s].sum()), changed_pixels=int(np.any(value != crop, axis=2).sum()),
                    reference_pixels=int(known.sum()), donor_patches=len(positions),
                    reused_same_active_reference=reused is not None)
            frozen.prior.prior.previous.prior_ddrm.figure([crop]+[outputs[m] for m in MODES],
                ['原画像', 'npjマスク × 紙面補間', 'v6マスク × 紙面補間', '全成分回収 × 紙面補間', '輪郭回収 × 紙面補間'],
                OUT/'comparisons'/(rid+'.png'), roi['title']+' · 未使用冊の紙面復元診断',
                'すべて同じ紙色補間・観測模様。未補間の編集画素も原画像を保持。元npj全工程ではない。',
                'DDRMではない。本文全体を復元した画像ではない。固定条件の紙面だけを比較。')
            rows.append(dict(id=rid, page=key, title=roi['title'], xywh=roi['xywh'],
                context_xywh=[x0, y0, x1-x0, y1-y0], diagnostics=stats))
            dump(OUT/'progress.json', dict(completed=len(rows), total=12, latest=rid))
            print('Unseen paper:', len(rows), '/12', rid, {m: stats[m]['applied_pixels'] for m in MODES}, flush=True)
        del original, v6, npj, domain, guard, values, references, eligibility
        gc.collect()
    for rel, digest in dependencies.items(): assert sha(ROOT/rel) == digest
    dump(OUT/'diagnostics.json', rows)
    dump(OUT/'verification.json', dict(status='passed', pages=4, regions=12, all_unapplied_pixels_original=True,
        original_damage_poison_cases=poison, outside_edit_changed_pixels=0, frozen_paper_restorer=True,
        DDRM=False, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json'] and p.name not in ['manifest.json', 'complete.json']}))
    print('New-book same-paper-restorer diagnostic complete', flush=True)


if __name__ == '__main__': main()
