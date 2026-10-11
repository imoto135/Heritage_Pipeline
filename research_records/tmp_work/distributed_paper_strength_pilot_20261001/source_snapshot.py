"""Fixed sigma32 observed-paper guidance; ablate only screening strength."""
import inspect
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import probe_distributed_paper_guidance as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/distributed_paper_strength_pilot_20261001'
DENOMINATORS = [64, 16]
SETTINGS = dict(sigma_px=32, baseline_strength='known reference mass /256',
    strengths=['known reference mass /64', 'known reference mass /16'],
    mask_reference_applied_texture_fixed=True, previously_filled_preserved=True,
    scope='only new 8-pair components; no thin-mask expansion',
    raw_smooth_boundary_and_feather_fixed=True, development_ROIs=34,
    no_nonreference_guidance=True, pixel_GT=False, DDRM_recomputed=False)

solver_source = inspect.getsource(prior.solve_screened)
for old, new in {
    'def solve_screened(lab, known, applied, sigma):': 'def solve_screened(lab, known, applied, sigma, screening_strength):',
    'mass/256.': 'mass*screening_strength'}.items():
    assert solver_source.count(old) == 1
    solver_source = solver_source.replace(old, new)
exec(solver_source, globals())


def main():
    # Reuse only the unchanged reference-lowpass alias needed by the copied solver.
    global prior
    original_module = prior
    prior = original_module.prior
    lab = np.full((32, 32, 3), (90, 3, 12), np.float32)
    active = np.zeros((32, 32), bool); active[10:22, 10:22] = True; lab[active] = 0
    for denominator in DENOMINATORS:
        fields, *_ = solve_screened(lab, ~active, active, 32, 1/denominator)
        for field in fields.values(): assert np.allclose(field[active], (90, 3, 12), atol=1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [original_module.OUT, prior.OUT]: check_bundle(folder)
    rows = json.loads((original_module.OUT/'diagnostics.json').read_text()); originals = {}
    for source in prior.frozen.MARGIN.SOURCES+[prior.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']: originals[page['page']] = page['original']
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str(Path(original_module.__file__).relative_to(ROOT)): sha(Path(original_module.__file__)),
        **{str((folder/name).relative_to(ROOT)): sha(folder/name)
           for folder in [original_module.OUT, prior.OUT] for name in ['manifest.json', 'complete.json']},
        **{path: sha(ROOT/path) for path in originals.values()}}
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); (OUT/'copied_solver.py').write_text(solver_source)
    results = []; poison = []; current = None
    for row in rows:
        rid = row['id']; key = row['page']; base_native = original_module.OUT/'native'/rid; src = prior.OUT/'native'/rid
        if current != key: original = read(ROOT/originals[key]); current = key
        x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
        lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        base = read(base_native/'guide32.png'); known = read(src/'reference_context.png', True)
        applied = read(src/'applied8_context.png', True); extra = read(src/'newly_applied_context.png', True)
        outputs = {}; diagnostics = {}; dest = OUT/'native'/rid
        if extra.any():
            strict = solve_screened(lab, known, applied, 32, 1/256)
            existing = np.load(base_native/'guide32/screened_fields_Lab.npz', allow_pickle=False)
            for name in strict[0]: assert np.array_equal(strict[0][name], existing[name])
        for denominator in DENOMINATORS:
            mode = 'strength'+str(denominator); value = base.copy(); md = dest/mode
            if extra.any():
                fields, guide, mass, strength, used, notes = solve_screened(lab, known, applied, 32, 1/denominator)
                seam = np.load(src/'texture_seam.npy', allow_pickle=False)
                field = original_module.field_with_texture(fields, applied, seam)
                value = prior.frozen.compose(field, extra, base, s)
                md.mkdir(parents=True, exist_ok=True); np.savez_compressed(md/'screened_fields_Lab.npz', **fields)
                for name, item in dict(guide_Lab=guide, reference_mass=mass, screening_strength=strength, result_field_Lab=field).items():
                    np.save(md/(name+'.npy'), item, allow_pickle=False)
                write(md/'used_context.png', used); dump(md/'components.json', notes)
                if extra[s].any():
                    poisoned = lab.copy(); poisoned[~known] = (240, 150, -160)
                    repeat = solve_screened(poisoned, known, applied, 32, 1/denominator)
                    for name in fields: assert np.array_equal(fields[name], repeat[0][name])
                    poison.append(dict(id=rid, denominator=denominator))
            assert not np.any(np.any(value != base, axis=2)&~extra[s])
            value_lab = cv2.cvtColor(value.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            outputs[mode] = value; diagnostics[mode] = dict(changed_vs_base=int(np.any(value != base, axis=2).sum()),
                median_L=float(np.median(value_lab[extra[s], 0])) if extra[s].any() else None)
            write(md/'output.png', value)
        for name, item in dict(original=context[s], base=base, newly_applied=extra[s]).items(): write(dest/(name+'.png'), item)
        figure([context[s], base]+[outputs['strength'+str(d)] for d in DENOMINATORS],
            ['原画像', '周囲紙色・弱 1/256', '周囲紙色・中 1/64', '周囲紙色・強 1/16'],
            OUT/'comparisons'/(rid+'.png'), row['title']+' · 同じ観測紙色への近づけ方だけ',
            'マスク・参照・σ32・明度模様を固定。旧補間画素は保持。',
            '内部の紙色項の強さだけを変更。非参照・原損傷の色は使わない。DDRM再推論なし。')
        results.append(dict(**row, strength_diagnostics=diagnostics)); print('Paper screening strength:', len(results), '/34', rid, flush=True)
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', results)
    dump(OUT/'verification.json', dict(status='passed', weak256_reproduced=True, constant_paper=True,
        outside_newly_applied_changes=0, all_nonreference_poison_cases=poison, ROIs=34, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: only screening strength varied, 34 ROIs', flush=True)


if __name__ == '__main__': main()
