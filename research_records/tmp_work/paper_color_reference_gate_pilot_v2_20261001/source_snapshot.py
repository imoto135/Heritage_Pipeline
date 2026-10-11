"""Fixed edit and observed L texture; tighten only color-reference chroma gates."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_anchored_thin_fragments as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/paper_color_reference_gate_pilot_v2_20261001'
LEVELS = {'chroma75': [1.5, 3.0], 'chroma50': [1.0, 2.0]}
SETTINGS = dict(baseline='frozen thin core + 8px anchored fragment, 16-pair paper fill',
    color_reference_old_ab_tolerance=[2, 4], color_reference_new_ab_tolerance=LEVELS,
    other_reference_gates_and_mask_fixed=True, texture='exact saved L seam with original known-paper source coordinates',
    color_and_texture_references_separate=True, components='previously applied only; retain baseline if stricter reference lacks16 pairs',
    operator='same 4-neighbor raw/smooth harmonic paper Lab; no distributed guidance',
    input_gates='original full-page median3 Lab and same paper estimate',
    ROIs=42, pages=18, development=True, pixel_GT=False, DDRM_recomputed=False)


def texture_source(rid, applied):
    if not applied.any(): return None, np.zeros(applied.shape, np.float32)
    choices = [prior.OUT/'native'/rid/'near8', prior.prior.OUT/'native'/rid,
        prior.prior.fill.frozen.OUT/'native'/rid, prior.prior.fill.new.OUT/'native'/rid/'rim',
        prior.prior.fill.frozen.prior.OUT/'native'/rid]
    for directory in choices:
        path = directory/'texture_seam.npy'
        if path.exists():
            value = np.load(path, allow_pickle=False)
            assert value.shape == applied.shape and not value[~applied].any()
            return directory, value
    raise AssertionError(('Missing observed texture', rid))


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(prior.OUT)
    rows = json.loads((prior.OUT/'diagnostics.json').read_text())['regions']; pages = []
    for source in prior.prior.mask.prior.MARGIN.SOURCES+[prior.prior.mask.new.INPUT]:
        pages.extend(json.loads((source/'baseline_review.json').read_text())['cases'])
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        **{str((prior.OUT/name).relative_to(ROOT)): sha(prior.OUT/name) for name in ['manifest.json', 'complete.json']},
        **{p['original']: sha(ROOT/p['original']) for p in pages}}
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); results = []; poison = []
    for pg in pages:
        key = pg['page']; original = read(ROOT/pg['original'])
        median_lab = cv2.cvtColor(cv2.medianBlur(original, 3).astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        paper = prior.prior.mask.prior.MARGIN.first.paper_estimate(median_lab)
        chroma_difference = np.abs(median_lab[:, :, 1:]-paper[:, :, 1:])
        for row in [r for r in rows if r['page'] == key]:
            rid = row['id']; x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
            cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
            context = original[cs]; crop = context[s]; lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            native = prior.OUT/'native'/rid/'near8'; dest = OUT/'native'/rid
            base = read(native/'output.png'); known = read(native/'reference_context.png', True)
            old_applied = read(native/'applied_context.png', True); edit = read(native/'edit.png', True)
            source, seam = texture_source(rid, old_applied)
            if source:
                for name in ['texture_seam.npy', 'source_x.npy', 'source_y.npy', 'donors.json']:
                    dependencies[str((source/name).relative_to(ROOT))] = sha(source/name)
                sx = np.load(source/'source_x.npy', allow_pickle=False); sy = np.load(source/'source_y.npy', allow_pickle=False)
                selected = sx >= 0; assert known[sy[selected], sx[selected]].all()
            distance = cv2.distanceTransform(old_applied.astype(np.uint8), cv2.DIST_L2, 5)
            bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*old_applied
            tw = np.clip((distance-1)/4., 0, 1)*old_applied
            outputs = {}; diagnostics = {}
            for mode, thresholds in LEVELS.items():
                strict = known&np.all(chroma_difference[cs] <= np.array(thresholds, np.float32), axis=2)
                md = dest/mode
                if old_applied[s].any():
                    fields, applied, used, notes = prior.prior.fill.solve_fields(lab, strict, old_applied, 16)
                    assert not np.any(applied&~old_applied)
                    field = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
                    field[:, :, 0] += seam*tw
                    value = prior.prior.fill.frozen.compose(field, applied, base, s)
                    md.mkdir(parents=True, exist_ok=True); np.savez_compressed(md/'harmonic_fields_Lab.npz', **fields)
                    np.save(md/'result_field_Lab.npy', field, allow_pickle=False)
                    if applied[s].any():
                        poisoned = lab.copy(); poisoned[~strict] = (260, 125, -192)
                        repeat = prior.prior.fill.solve_fields(poisoned, strict, old_applied, 16)
                        for name in fields: assert np.array_equal(fields[name], repeat[0][name])
                        poison.append(dict(id=rid, mode=mode))
                else: value = base.copy(); applied = np.zeros_like(old_applied); used = np.zeros_like(known); notes = []
                assert not np.any(np.any(value != base, axis=2)&~applied[s])
                assert not np.any(np.any(value != crop, axis=2)&~edit)
                for name, item in dict(output=value, reference_context=strict, applied_context=applied, used_context=used).items():
                    write(md/(name+'.png'), item)
                dump(md/'components.json', notes); outputs[mode] = value
                diagnostics[mode] = dict(reference_pixels=int(strict.sum()), tone_updated_pixels=int(applied[s].sum()),
                    baseline_retained_due_insufficient_reference=int((old_applied[s]&~applied[s]).sum()),
                    changed_vs_base=int(np.any(value != base, axis=2).sum()))
            for name, item in dict(original=crop, base=base, baseline_reference_context=known, baseline_applied_context=old_applied,
                edit=edit).items(): write(dest/(name+'.png'), item)
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest/'reference_chroma_difference.npy', chroma_difference[cs], allow_pickle=False)
            dump(dest/'texture_source.json', dict(path=str(source.relative_to(ROOT)) if source else None,
                known_for_texture='baseline saved known paper; stricter gates apply to color interpolation only'))
            figure([crop, base]+[outputs[m] for m in LEVELS],
                ['原画像', '参照色差 a2/b4', '参照色差 a1.5/b3', '参照色差 a1/b2'], OUT/'comparisons'/(rid+'.png'),
                row['title']+' · 紙色参照の資格だけ', 'マスク・明度模様は固定。参照不足の成分は前回出力を保持。',
                '色補間の参照だけを絞る。模様の参照元は前回の観測紙面。DDRM再推論なし。')
            results.append(dict(**row, color_reference_diagnostics=diagnostics)); print('Color reference gate:', len(results), '/42', rid, flush=True)
        del original, median_lab, paper, chroma_difference; gc.collect()
    assert len(results) == 42
    # Texture files were discovered while reading; pin the final dependency set.
    if (OUT/'final_dependencies.json').exists(): assert json.loads((OUT/'final_dependencies.json').read_text()) == dependencies
    else: dump(OUT/'final_dependencies.json', dependencies)
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', results)
    dump(OUT/'verification.json', dict(status='passed', ROIs=42, variants=84, texture_fixed=True,
        paper_color_references_subset_of_baseline=True, insufficient_reference_fallback=True,
        all_nonreference_poison_cases=poison, outside_applied_changes=0, outside_edit_changes=0, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: paper color-reference subset, 84 variants', flush=True)


if __name__ == '__main__': main()
