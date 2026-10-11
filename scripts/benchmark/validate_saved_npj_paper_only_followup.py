"""Fixed paper-only follow-up on actual stored npj outputs and stored inputs.

Stored npj is never re-inferred. Two inputs differ from repository references;
classify their authentic saved input instead of borrowing mismatched masks.
"""
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump
from package_saved_npj_inference_v6_review import check
import check_red_positive_fixed as adapter
import prepare_training_free_fixed_check as foreground
import check_warm_hold_other_books as warm
import probe_saved_npj_role_separated_postprocess as frozen

OUT = ROOT/'tmp_work/saved_npj_paper_only_followup_pilot_20261002'
SELECTION = ROOT/'tmp_work/saved_npj_paper_followup_selection_20261002/selection.json'
PRIOR = frozen.OUT


def dependencies():
    selected = json.loads(SELECTION.read_text())
    deps = dict(selected['dependencies'])
    for rel, digest in deps.items(): assert sha(ROOT/rel) == digest, rel
    previous = check(PRIOR)
    assert previous['settings']['paper'] == frozen.SETTINGS['paper']
    assert sha(Path(frozen.__file__)) == previous['source_sha256']
    for m in [warm.v3, warm.v4, warm.v5, warm.v6]:
        p = m.OUT/'manifest.json'; manifest = json.loads(p.read_text())
        assert sha(Path(m.__file__)) == manifest['source_sha256']
        assert m.SETTINGS == manifest['settings']
        deps[str(Path(m.__file__).relative_to(ROOT))] = sha(Path(m.__file__))
        deps[str(p.relative_to(ROOT))] = sha(p)
    for directory, name in [(adapter.FROZEN, 'refine_training_free_protection.py'),
                            (adapter.COLOR, 'compare_training_free_color_shape.py'),
                            (adapter.PREPROCESS, 'compare_training_free_preprocessing.py')]:
        p = directory/'manifest.json'; expected = json.loads(p.read_text())['source_sha256']
        source = ROOT/'scripts/benchmark'/name
        assert sha(source) == expected
        deps[str(source.relative_to(ROOT))] = expected
        deps[str(p.relative_to(ROOT))] = sha(p)
    for p in [Path(__file__), Path(frozen.__file__), Path(adapter.__file__),
              Path(foreground.__file__), Path(warm.__file__), SELECTION,
              PRIOR/'manifest.json', PRIOR/'complete.json',
              ROOT/'scripts/benchmark/build_npj_v6_comparison.py',
              ROOT/'scripts/benchmark/package_saved_npj_inference_v6_review.py',
              warm.v6.OUT/'candidate_freeze.json', warm.v6.OUT/'complete.json']:
        deps[str(p.relative_to(ROOT))] = sha(p)
    assert json.loads((warm.v6.OUT/'candidate_freeze.json').read_text())['variant'] == 'split_direction'
    for case in selected['cases']:
        if 'v6_cache' not in case: continue
        expected = json.loads((warm.v6.OUT/'complete.json').read_text())['outputs']
        p = ROOT/case['v6_cache']
        for filename in ['split_direction_edit_allowed.png', 'split_direction_hold.png',
                         'split_direction_extra_hold.png', 'protect.png', 'red.png']:
            assert sha(p/filename) == expected[str((p/filename).relative_to(warm.v6.OUT))]
            deps[str((p/filename).relative_to(ROOT))] = sha(p/filename)
    return selected, deps


def paper_only(base, roles):
    keep = roles['protect'] | roles['hold'] | roles['red'] | roles['extra']
    guard = cv2.dilate(keep.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    update = roles['edit'] & ~guard & frozen.frame_guard(base.shape[:2], 1)
    result, diagnostic = frozen.harmonic(base, update)
    assert np.array_equal(result[~update], base[~update])
    assert np.array_equal(result[keep], base[keep])
    return result, update, guard, diagnostic


def prior_parity():
    results = []
    source = frozen.SOURCE
    for fid in ['upper_crop', 'full_page']:
        baseline = read(source/'native'/fid/'saved_npj.png')
        roles = {k:read(source/'native'/fid/(k+'.png'), True)
                 for k in ['edit', 'protect', 'hold', 'red', 'extra']}
        result, update, guard, _ = paper_only(baseline, roles)
        assert np.array_equal(result, read(PRIOR/'native'/fid/'paper_only.png'))
        assert np.array_equal(update, read(PRIOR/'native'/fid/'paper_update.png', True))
        assert np.array_equal(guard, read(PRIOR/'native'/fid/'paper_guard.png', True))
        results.append(dict(frame=fid, result_and_masks_exact=True))
    return results


def prepare_new(selected):
    new = [c for c in selected['cases'] if 'v6_cache' not in c]
    pilot = OUT/'classification/pilot'; review = OUT/'classification/input_review'
    color = OUT/'classification/color'
    inputs = {c['input']:sha(ROOT/c['input']) for c in new}
    dump(review/'manifest.json', dict(cases=[dict(id=c['id'], title=c['page'], original=c['input']) for c in new], inputs=inputs))
    # No display crop or semantic label enters classification.
    dump(review/'rois.json', {c['id']:[] for c in new})
    dump(pilot/'manifest.json', dict(settings=json.loads((adapter.PREPROCESS/'manifest.json').read_text())['settings'], inputs=inputs))
    for case in new:
        started = time.monotonic(); im = read(ROOT/case['input'])
        local, votes, normalized = foreground.foreground_inputs(im)
        for filename, a in [('original.png', im), ('sauvola_local/foreground.png', local),
                            ('scale_votes.png', votes), ('normalized.png', normalized)]:
            write(pilot/case['id']/filename, a)
        print('Fixed foreground:', case['id'], round(time.monotonic()-started, 1), 's', flush=True)
        del im, local, votes, normalized; gc.collect()
    dump(pilot/'complete.json', dict(outputs={str(p.relative_to(pilot)):sha(p) for p in sorted(pilot.rglob('*.png'))}))
    subprocess.run([sys.executable, '-B', str(ROOT/'scripts/benchmark/compare_training_free_color_shape.py'),
                    '--pilot', str(pilot), '--review', str(review), '--output', str(color)], check=True, cwd=ROOT)
    assert json.loads((color/'manifest.json').read_text())['parameters'] == json.loads((adapter.COLOR/'manifest.json').read_text())['parameters']
    for rel, digest in json.loads((color/'complete.json').read_text())['outputs'].items():
        assert sha(color/rel) == digest
    adapter.OUT = OUT/'classification'
    refine = adapter.refinement_function()
    previous_adapter = ROOT/'tmp_work/npj_failure_collection_20261001/refinement_adapter_snapshot.py'
    assert sha(adapter.OUT/'refinement_adapter_snapshot.py') == sha(previous_adapter)
    for case in new:
        started = time.monotonic(); fid = case['id']; im = read(ROOT/case['input'])
        ref, _ = adapter.refine_inputs(refine, color, pilot, fid)
        values = warm.warm_masks(im, ref, read(color/fid/'red_weak.png', True))
        extra = values[-1]['split_direction']
        roles = dict(edit=ref['edit_allowed'] & ~extra, hold=ref['hold'] | extra,
                     extra=extra, red=ref['red'], protect=ref['protect'])
        assert not (roles['edit'] & (roles['hold'] | roles['protect'] | roles['red'])).any()
        for k, a in roles.items(): write(OUT/'roles'/fid/(k+'.png'), a)
        dump(OUT/'roles'/fid/'features.json', dict(v3=values[1], v4=values[4], v5=values[6]))
        print('Fixed v6 on authentic saved input:', fid, round(time.monotonic()-started, 1), 's', flush=True)
        del im, ref, values, extra, roles; gc.collect()


def main():
    assert not (OUT/'manifest.json').exists(), 'Use a new directory for another run'
    OUT.mkdir(parents=True, exist_ok=True)
    selected, deps = dependencies()
    os.environ['MPLCONFIGDIR'] = str(OUT/'plot_cache')
    dump(OUT/'manifest.json', dict(settings=dict(paper=frozen.SETTINGS['paper'], stroke_disabled=True,
         classification='Frozen foreground/color/protection/v6 split_direction on authentic saved input frame',
         cached_roles_only_for_exact_input=True), cases=selected['cases'], dependencies=deps,
         source_sha256=sha(Path(__file__)), selection_sha256=sha(SELECTION),
         selection_before_outputs=True, independent_unseen_books=False, pixel_GT=False,
         baseline='Actual saved npj output, untouched; historical source/settings not fully reproduced'))
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    parity = prior_parity()
    prepare_new(selected)
    records = []
    for case in selected['cases']:
        fid = case['id']; im = read(ROOT/case['input']); base = read(ROOT/case['saved_npj'])
        assert im.shape == base.shape == tuple(case['input_shape'])
        if 'v6_cache' in case:
            mapping = dict(edit='split_direction_edit_allowed.png', hold='split_direction_hold.png',
                           extra='split_direction_extra_hold.png', protect='protect.png', red='red.png')
            roles = {k:read(ROOT/case['v6_cache']/name, True) for k, name in mapping.items()}
            assert case['reference_input_exact']
            frozen_case = next(c for c in json.loads((warm.v6.OUT/'manifest.json').read_text())['cases'] if c['id'] == case['page'])
            assert np.array_equal(im, read(ROOT/frozen_case['original']))
        else:
            roles = {k:read(OUT/'roles'/fid/(k+'.png'), True) for k in ['edit', 'hold', 'extra', 'protect', 'red']}
        result, update, guard, solve = paper_only(base, roles)
        poisoned = base.copy(); poisoned[update] = [255, 0, 255]
        poisoned_result, _, _, _ = paper_only(poisoned, roles)
        assert np.array_equal(poisoned_result, result)
        native = OUT/'native'/fid
        for k, a in dict(original=im, saved_npj=base, paper_only=result, paper_update=update,
                         paper_guard=guard, **roles).items():
            write(native/(k+'.png'), a)
            assert np.array_equal(read(native/(k+'.png'), a.dtype==bool), a)
        changed = np.any(result != base, axis=2)
        record = dict(id=fid, page=case['page'], edit_pixels=int(roles['edit'].sum()),
                      paper_update_pixels=int(update.sum()), changed_pixels=int(changed.sum()),
                      roles_pixels={k:int(a.sum()) for k, a in roles.items()},
                      reference_input_exact=case['reference_input_exact'],
                      reference_input_mae=case['reference_input_mae'], harmonic=solve,
                      outside_update_changed_pixels=int((changed & ~update).sum()),
                      protected_saved_npj_exact=True, stroke_update_pixels=0)
        records.append(record); print('Paper-only:', json.dumps(record), flush=True)
        del im, base, result, roles, poisoned, poisoned_result; gc.collect()
    for rel, digest in deps.items(): assert sha(ROOT/rel) == digest, rel
    dump(OUT/'verification.json', dict(status='passed', records=records, prior_paper_extraction_parity=parity,
         npj_rerun=False, diffusion_rerun=False, thresholds_changed=False, stroke_update_disabled=True,
         classification_uses_actual_saved_input=True, unknown_values_do_not_affect_paper_solution=True,
         outside_update_exact_saved_npj=True, protected_exact_saved_npj=True,
         ast_adapter_matches_saved_snapshot=True, pixel_GT=False, adopted=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Fixed paper-only follow-up frozen.', flush=True)


if __name__ == '__main__': main()
