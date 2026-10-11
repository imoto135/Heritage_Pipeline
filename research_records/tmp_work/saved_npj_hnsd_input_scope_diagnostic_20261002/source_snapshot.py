"""Separate native page context from changed saved-input pixels for hnsd.

Classification diagnostic only: no baseline, restoration or threshold changes.
"""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_inference_v6_review import check
from package_saved_npj_role_separated_review import figure
import validate_saved_npj_paper_only_followup as runner

OUT = ROOT/'tmp_work/saved_npj_hnsd_input_scope_diagnostic_20261002'
FOLLOWUP = runner.OUT
V6 = ROOT/'tmp_work/warm_ink_hold_v6_20260930'


def main():
    assert not (OUT/'manifest.json').exists(), 'Use a new output directory'
    OUT.mkdir(parents=True, exist_ok=True)
    previous = check(FOLLOWUP)
    case = next(c for c in previous['cases'] if c['id'] == 'crop_hnsd')
    original = read(ROOT/case['reference_original'])
    x, y, w, h = case['reference_xywh']; reference = original[y:y+h, x:x+w]
    stored = read(ROOT/case['input'])
    assert reference.shape == stored.shape == (620, 620, 3)
    native_reference = OUT/'input/reference_crop.png'; write(native_reference, reference)
    deps = {str((FOLLOWUP/name).relative_to(ROOT)):sha(FOLLOWUP/name)
            for name in ['manifest.json', 'complete.json', 'source_snapshot.py']}
    for p in [Path(__file__), Path(runner.__file__), ROOT/case['input'], ROOT/case['reference_original'],
              V6/'manifest.json', V6/'complete.json', V6/'candidate_freeze.json']:
        deps[str(p.relative_to(ROOT))] = sha(p)
    frozen = json.loads((V6/'complete.json').read_text())['outputs']
    mapping = dict(edit='split_direction_edit_allowed.png', hold='split_direction_hold.png',
                   extra='split_direction_extra_hold.png', protect='protect.png', red='red.png')
    full_crop_roles = {}
    for k, filename in mapping.items():
        p = V6/'hnsd001_011'/filename
        assert sha(p) == frozen[str(p.relative_to(V6))]
        a = read(p, True); assert a.shape == original.shape[:2]
        full_crop_roles[k] = a[y:y+h, x:x+w]
        deps[str(p.relative_to(ROOT))] = sha(p)
    source_case = next(c for c in json.loads((V6/'manifest.json').read_text())['cases'] if c['id'] == 'hnsd001_011')
    assert np.array_equal(original, read(ROOT/source_case['original']))
    stored_roles = {}
    for k in mapping:
        p = FOLLOWUP/'native/crop_hnsd'/(k+'.png')
        stored_roles[k] = read(p, True); deps[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'manifest.json', dict(dependencies=deps, source_sha256=sha(Path(__file__)),
         reference_xywh=case['reference_xywh'], threshold_changes=False, restoration=False,
         comparisons=['Same original pixels: full-page classification versus crop classification',
                      'Same cropped scope: original reference pixels versus historical saved-input pixels'],
         limitation='Historical full-image transformation unavailable; source/crop tests are not a complete 2x2'))
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    # This exact callable chain is already frozen by the follow-up. Route writes
    # to this diagnostic directory while leaving all historical artifacts intact.
    runner.OUT = OUT/'classification_run'
    runner.prepare_new(dict(cases=[dict(id='reference_crop', input=str(native_reference.relative_to(ROOT)), page='hnsd001_011 reference crop')]))
    crop_roles = {k:read(runner.OUT/'roles/reference_crop'/(k+'.png'), True) for k in mapping}
    sets = dict(original_page=full_crop_roles, reference_crop=crop_roles, saved_crop=stored_roles)
    regions = [('whole_crop', [0, 0, 620, 620]), ('dark_mass', [170, 300, 350, 210])]
    records = []
    for name, box in regions:
        rx, ry, rw, rh = box; s = np.s_[ry:ry+rh, rx:rx+rw]
        images = [reference[s]]
        for mode, image in [('original_page', reference), ('reference_crop', reference), ('saved_crop', stored)]:
            masks = sets[mode]
            painted = overlay(image, [('blue', masks['protect']), ('hold', masks['hold']), ('edit', masks['edit'])])
            images.append(painted[s])
        counts = {mode:{k:int(v[s].sum()) for k, v in roles.items()} for mode, roles in sets.items()}
        filename = name+'_scope_input_roles.png'
        figure(images, ['参照原画像', '原ページから分類・同位置', '同じ原画素のcropから分類', '濃淡の異なる保存crop分類'],
               OUT/filename, 'hnsd：紙面候補が変わる要因を分離 / '+name,
               ['青＝保護、紫＝保留、黄＝v6編集候補。右3列は分類表示で復元画像ではない。',
                f'編集候補：原ページ {counts["original_page"]["edit"]}／原crop {counts["reference_crop"]["edit"]}／保存crop {counts["saved_crop"]["edit"]}画素。',
                '原ページ→原cropは範囲の差、原crop→保存cropは同範囲で画素の差。GTではない。'])
        records.append(dict(region=name, xywh=box, roles=counts, figure=filename))
    for mode, roles in sets.items():
        for k, a in roles.items(): write(OUT/'native'/mode/(k+'.png'), a)
    write(OUT/'native/reference.png', reference); write(OUT/'native/saved_input.png', stored)
    dump(OUT/'verification.json', dict(status='passed', records=records,
         page_reference_pixels_exact=True, reference_crop_pixels_exact=True,
         stored_crop_reuses_exact_frozen_followup_masks=True, thresholds_changed=False,
         restoration=False, pixel_GT=False))
    for rel, digest in deps.items(): assert sha(ROOT/rel) == digest, rel
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(json.dumps(records), flush=True)


if __name__ == '__main__': main()
