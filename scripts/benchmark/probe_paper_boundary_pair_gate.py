"""Development ablation of minimum adjacent reference pairs, 16 versus 8.

Preserve all previously filled pixels. Only newly accepted components are added.
"""
import gc
import inspect
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import probe_paper_rim_lowfrequency_v2 as frozen
import restore_unseen_margin_paper as new
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import check_bundle

ROOT = frozen.ROOT
OUT = ROOT/'tmp_work/paper_boundary_pair_gate_pilot_20261001'
SETTINGS = dict(change='minimum adjacent known-paper boundary pairs 16 -> 8',
    component_area_min=64, mask_and_reference_fixed=True, previous_filled_pixels_preserved=True,
    added_scope='only components not accepted at 16 but accepted at 8',
    guidance_texture='same existing raw/smooth harmonic boundary blend and known-paper L seams',
    old14_baseline='saved rim output, with saved DDRM elsewhere',
    new4_baseline='saved rim paper-only output, with original elsewhere',
    development=True, pages=18, ROIs=34, new4_now_development=True, DDRM=False, pixel_GT=False)

# Make the single changed acceptance condition reviewable as a saved source file.
# The numerical operator, values, and solver are copied verbatim from the freeze.
solver_source = inspect.getsource(frozen.prior.prior.solve_fields)
replacements = {
    'def solve_fields(original_lab,reference,active):': 'def solve_fields(original_lab,reference,active,minimum_pairs):',
    'if pairs<16:': 'if pairs<minimum_pairs:',
    "reason='fewer than 16 paper boundary pairs'": "reason=f'fewer than {minimum_pairs} paper boundary pairs'"}
for before, after in replacements.items():
    assert solver_source.count(before) == 1
    solver_source = solver_source.replace(before, after)
reference_lowpass = frozen.prior.prior.reference_lowpass
exec(solver_source, globals())


def field_for(lab, known, active, minimum):
    fields, applied, used, notes = solve_fields(lab, known, active, minimum)
    if applied.any():
        products, positions, qnotes, seam, average, sx, sy = frozen.prior.infer(lab, known, applied,
            fields['boundary_raw'], fields['boundary_smooth'])
        field = products['combined']
    else:
        field = np.zeros_like(lab); positions = []; qnotes = []
        seam = np.zeros(active.shape, np.float32); average = seam.copy()
        sx = np.full(active.shape, -1, np.int32); sy = sx.copy()
    return fields, applied, used, notes, field, positions, qnotes, seam, average, sx, sy


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in [frozen.OUT, new.OUT]: check_bundle(directory)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str(Path(frozen.prior.prior.__file__).relative_to(ROOT)): sha(Path(frozen.prior.prior.__file__)),
        str(Path(frozen.prior.__file__).relative_to(ROOT)): sha(Path(frozen.prior.__file__))}
    for directory in [frozen.OUT, new.OUT]:
        for name in ['manifest.json', 'complete.json']: dependencies[str((directory/name).relative_to(ROOT))] = sha(directory/name)
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); (OUT/'copied_solver.py').write_text(solver_source)
    old_rows = {r['id']: r for r in json.loads((frozen.OUT/'diagnostics.json').read_text())['regions']}
    new_rows = {r['id']: r for r in json.loads((new.OUT/'diagnostics.json').read_text())}
    rows = []; poison = []
    for source in frozen.MARGIN.SOURCES + [new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original'])
            for roi in page['rois']:
                rid = roi['id']; row = (new_rows if source == new.INPUT else old_rows)[rid]
                cx, cy, cw, ch = row['context_xywh']; x, y, w, h = row['xywh']
                cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
                context = original[cs]; crop = context[s]; lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                if source == new.INPUT:
                    native = new.OUT/'native'/rid/'rim'
                    known = read(native/'reference_context.png', True); active = read(native/'eligible_context.png', True)
                    old_applied = read(native/'applied_context.png', True); base = read(native/'output.png')
                    edit = read(new.masks.OUT/'pages'/key/'rim/edit.png', True)[cs]
                else:
                    native = frozen.OUT/'native'/rid
                    known = read(native/'rim_reference_context.png', True); active = read(native/'rim_eligible_context.png', True)
                    old_applied = read(native/'rim_applied_context.png', True); base = read(native/'rim.png')
                    edit = read(frozen.OUT/'pages'/key/'rim_edit.png', True)[cs]
                dest = OUT/'native'/rid; dest.mkdir(parents=True, exist_ok=True)
                assert not np.any(known & edit)
                extra = np.zeros_like(active); applied = old_applied.copy(); value = base.copy(); notes = []
                if active[s].any():
                    strict = field_for(lab, known, active, 16)
                    assert np.array_equal(strict[1][s], old_applied[s])
                    assert np.array_equal(frozen.compose(strict[4], strict[1], base, s), base)
                    result = field_for(lab, known, active, 8)
                    fields, applied, used, notes, field, positions, qnotes, seam, average, sx, sy = result
                    assert not np.any(old_applied[s] & ~applied[s])
                    extra = applied & ~strict[1]
                    value = frozen.compose(field, extra, base, s)
                    if extra[s].any():
                        changed = context.copy(); changed[edit] = (217, 2, 11)
                        repeat = field_for(cv2.cvtColor(changed.astype(np.float32)/255, cv2.COLOR_BGR2LAB), known, active, 8)
                        assert np.array_equal(field, repeat[4]) and np.array_equal(applied, repeat[1])
                        poison.append(rid)
                    if extra.any():
                        np.savez_compressed(dest/'harmonic_fields_Lab.npz', **fields)
                        for name, item in dict(result_field_Lab=field, texture_seam=seam, texture_average=average, source_x=sx, source_y=sy).items():
                            np.save(dest/(name+'.npy'), item, allow_pickle=False)
                        dump(dest/'donors.json', dict(context_xywh=row['context_xywh'], patch_xy=positions, components=qnotes))
                        write(dest/'used_context.png', used)
                assert not np.any(np.any(value != base, axis=2) & ~extra[s])
                assert not np.any(np.any(value != crop, axis=2) & ~edit[s])
                assert np.array_equal(value[old_applied[s]], base[old_applied[s]])
                for name, item in dict(original=crop, base=base, pairs8=value, edit=edit[s], reference_context=known,
                    eligible_context=active, old_applied_context=old_applied, applied8_context=applied,
                    newly_applied_context=extra, newly_applied=extra[s]).items(): write(dest/(name+'.png'), item)
                dump(dest/'components8.json', notes)
                frozen.prior.prior.previous.prior_ddrm.figure([crop, base, value],
                    ['原画像', '固定マスク・境界16組', '同じマスク・境界8組'], OUT/'comparisons'/(rid+'.png'),
                    roi['title']+' · 紙面参照の境界対数だけを変更',
                    f'追加補間 {int(extra[s].sum()):,}画素。既に補間した画素は完全保持。GTの成功数ではない。',
                    'マスク・参照・数値補間・模様条件は固定。以前16組で落ちた成分を追加する。DDRM再推論なし。')
                rows.append(dict(id=rid, page=key, title=roi['title'], xywh=row['xywh'], context_xywh=row['context_xywh'],
                    newly_applied_pixels=int(extra[s].sum()), old_applied_pixels=int(old_applied[s].sum()),
                    total_applied8_pixels=int(applied[s].sum()), changed_pixels=int(np.any(value != base, axis=2).sum())))
                print('Boundary-pair ablation:', len(rows), '/34', rid, 'new', int(extra[s].sum()), flush=True)
            del original
            gc.collect()
    assert len(rows) == 34
    for rel, digest in dependencies.items(): assert sha(ROOT/rel) == digest
    dump(OUT/'diagnostics.json', rows)
    dump(OUT/'verification.json', dict(status='passed', regions=34, strict16_native_reproduced=True,
        masks_and_references_fixed=True, old_applied_pixels_unchanged=True,
        original_damage_poison_cases=poison, outside_extra_changed_pixels=0, outside_edit_changed_pixels=0,
        new_four_books_now_development=True, DDRM_recomputed=False, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: 16 versus 8 adjacent paper pairs, 34 ROIs', flush=True)


if __name__ == '__main__': main()
