"""Keep real saved npj output and separate paper repair from short stroke gaps.

Developmental postprocessing pilot. No npj/model rerun, no GT, no ROI gating.
Original contrast is used only inside observed, anchored short-gap candidates.
"""
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from build_npj_v6_comparison import ROOT, read, write, sha, dump
from package_saved_npj_inference_v6_review import check

SOURCE = ROOT / 'tmp_work/saved_npj_inference_v6_pilot_20261002'
OUT = ROOT / 'tmp_work/npj_role_separated_postprocess_pilot_20261002'
SETTINGS = dict(
    base='Actual saved npj image, unchanged outside explicit update masks',
    paper=dict(mask='Frozen v6 edit minus 2px guard around protect/hold/red/extra',
               frame_border_guard=1, solver='4-neighbour BGR Dirichlet harmonic fill',
               reference='Saved npj pixels immediately outside paper update'),
    stroke=dict(support='Frozen protect minus red/extra/hold/edit; original contrast >= 8 Lab L',
                paper_estimator='17px elliptical morphological closing, native saved frame',
                anchor_min_local_L_contrast=12, original_to_npj_L_loss_min=8,
                gap_area_min=2, gap_area_max=64, gap_max_side=12, frame_border_guard=8,
                contact='Direct 8-neighbour contact with >=2 distinct anchor components',
                anchor_component_min_area=8,
                render='Saved npj multiplied by min(original/original-local-paper, 1) in BGR'),
    processing_scope='Entire saved input frame; ROI coordinates never used in inference',
    selection='Same two development pages inspected before designing the pilot',
    npj_rerun=False, diffusion_rerun=False, pixel_GT=False, adopted=False)


def frame_guard(shape, radius):
    h, w = shape
    inner = np.zeros(shape, bool)
    if h > 2*radius and w > 2*radius:
        inner[radius:h-radius, radius:w-radius] = True
    return inner


def harmonic(base, mask):
    """Solve missing pixels from external neighbours, ignoring their old values."""
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    count = len(yy)
    if count == 0:
        return base.copy(), dict(pixels=0, max_residual=0.)
    assert frame_guard(mask.shape, 1)[mask].all()
    index = np.full(mask.shape, -1, np.int32)
    index[yy, xx] = np.arange(count)
    rows = [np.arange(count)]; cols = [np.arange(count)]
    values = [np.full(count, 4., np.float64)]
    rhs = np.zeros((count, 3), np.float64)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbours = index[yy+dy, xx+dx]
        unknown = neighbours >= 0
        rows.append(np.nonzero(unknown)[0]); cols.append(neighbours[unknown])
        values.append(np.full(int(unknown.sum()), -1., np.float64))
        known = ~unknown
        rhs[known] += base[yy[known]+dy, xx[known]+dx]
    matrix = coo_matrix((np.concatenate(values),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(count, count)).tocsr()
    solved = spsolve(matrix, rhs)
    assert np.isfinite(solved).all()
    residual = float(np.max(np.abs(matrix @ solved-rhs)))
    assert residual < 1e-7
    out = base.copy()
    out[mask] = np.clip(np.rint(solved), 0, 255).astype(np.uint8)
    assert np.array_equal(out[~mask], base[~mask])
    return out, dict(pixels=count, max_residual=residual,
                     min_solution=float(solved.min()), max_solution=float(solved.max()))


def infer(original, base, roles):
    shape = original.shape[:2]
    keep = roles['protect'] | roles['hold'] | roles['red'] | roles['extra']
    guard = cv2.dilate(keep.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    paper = roles['edit'] & ~guard & frame_guard(shape, 1)
    paper_only, solve = harmonic(base, paper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    L = cv2.cvtColor(original.astype(np.float32)/255., cv2.COLOR_BGR2LAB)[:, :, 0]
    N = cv2.cvtColor(base.astype(np.float32)/255., cv2.COLOR_BGR2LAB)[:, :, 0]
    original_paper = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel)
    npj_paper = cv2.morphologyEx(N, cv2.MORPH_CLOSE, kernel)
    eligible = (roles['protect'] & ~roles['hold'] & ~roles['red'] & ~roles['extra']
                & ~roles['edit'] & frame_guard(shape, 8))
    support = eligible & ((original_paper-L) >= 8)
    anchors = support & ((npj_paper-N) >= 12)
    weak = support & ~anchors & ((N-L) > 8)
    _, anchor_labels, anchor_stats, _ = cv2.connectedComponentsWithStats(
        anchors.astype(np.uint8), connectivity=8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    accepted = np.zeros(num, bool); components = []
    for i in range(1, num):
        x, y, w, h, area = map(int, stats[i])
        contacts = []
        small_enough = 2 <= area <= 64 and max(w, h) <= 12
        if small_enough:
            # Include one-pixel ring but no unrelated pixels within the rectangle.
            local = labels[y-1:y+h+1, x-1:x+w+1] == i
            near = cv2.dilate(local.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ids = np.unique(anchor_labels[y-1:y+h+1, x-1:x+w+1][near])
            contacts = [int(j) for j in ids if j > 0 and anchor_stats[j, 4] >= 8]
        accepted[i] = small_enough and len(contacts) >= 2
        components.append(dict(id=i, xywh=[x, y, w, h], pixels=area,
                               anchor_components=contacts, accepted=bool(accepted[i])))
    gap = accepted[labels] & weak
    assert not (gap & paper).any()
    local_paper_BGR = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
    ratio = np.minimum(original.astype(np.float64)/np.maximum(local_paper_BGR, 1), 1.)
    stroke_only = base.copy()
    stroke_only[gap] = np.clip(np.rint(base[gap].astype(np.float64)*ratio[gap]), 0, 255).astype(np.uint8)
    combined = paper_only.copy(); combined[gap] = stroke_only[gap]
    assert np.all(stroke_only[gap] <= base[gap])
    assert np.array_equal(combined[~(paper | gap)], base[~(paper | gap)])
    assert np.array_equal(combined[roles['red'] | roles['hold'] | roles['extra']],
                          base[roles['red'] | roles['hold'] | roles['extra']])
    return dict(paper_only=paper_only, stroke_only=stroke_only, combined=combined), dict(
        paper_update=paper, stroke_update=gap, original_support=support, npj_anchors=anchors,
        weak_candidate=weak, paper_guard=guard), dict(
        harmonic=solve, components=components, accepted_components=int(accepted.sum()),
        paper_update_pixels=int(paper.sum()), stroke_update_pixels=int(gap.sum()))


def solver_checks():
    yy, xx = np.mgrid[:25, :25]
    base = np.stack([150+xx+yy, 170+xx, 180+yy], axis=2).astype(np.uint8)
    mask = np.zeros((25, 25), bool); mask[7:18, 8:17] = True
    damaged = base.copy(); damaged[mask] = [0, 0, 0]
    fixed, stats = harmonic(damaged, mask)
    assert np.array_equal(fixed, base), 'Linear paper field must be reconstructed exactly'
    poisoned = damaged.copy(); poisoned[mask] = [255, 0, 255]
    fixed2, _ = harmonic(poisoned, mask)
    assert np.array_equal(fixed, fixed2), 'Unknown old pixels must not affect paper solution'
    return dict(linear_paper_exact=True, unknown_value_invariance=True, **stats)


def main():
    assert not (OUT/'complete.json').exists(), 'Pilot already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    source = check(SOURCE)
    dependencies = {str((SOURCE/name).relative_to(ROOT)):sha(SOURCE/name)
                    for name in ['manifest.json', 'complete.json', 'source_snapshot.py', 'verification.json']}
    for name in ['build_npj_v6_comparison.py', 'package_saved_npj_inference_v6_review.py', Path(__file__).name]:
        p = Path(__file__).parent/name; dependencies[str(p.relative_to(ROOT))] = sha(p)
    for frame in source['frames']:
        for name in ['original', 'saved_npj', 'edit', 'protect', 'hold', 'red', 'extra']:
            p = SOURCE/'native'/frame['id']/(name+'.png')
            dependencies[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'manifest.json', dict(settings=SETTINGS, dependencies=dependencies,
         source_sha256=sha(Path(__file__)), frames=source['frames'], regions=source['regions']))
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    checks = solver_checks(); records = []
    for spec in source['frames']:
        fid = spec['id']; src = SOURCE/'native'/fid; dest = OUT/'native'/fid
        original = read(src/'original.png'); base = read(src/'saved_npj.png')
        roles = {k:read(src/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
        images, masks, diagnostic = infer(original, base, roles)
        for name, a in dict(original=original, saved_npj=base, **images, **masks).items():
            write(dest/(name+'.png'), a)
            assert np.array_equal(read(dest/(name+'.png'), a.dtype==bool), a)
        # The paper reconstruction must ignore every old unknown pixel.
        poisoned = base.copy(); poisoned[masks['paper_update']] = [255, 0, 255]
        independent, _ = harmonic(poisoned, masks['paper_update'])
        assert np.array_equal(independent, images['paper_only'])
        dump(dest/'diagnostics.json', diagnostic)
        changed = {k:int(np.any(a != base, axis=2).sum()) for k, a in images.items()}
        records.append(dict(frame=fid, paper_update_pixels=diagnostic['paper_update_pixels'],
                            stroke_update_pixels=diagnostic['stroke_update_pixels'],
                            accepted_gap_components=diagnostic['accepted_components'], changed_pixels=changed,
                            harmonic_residual=diagnostic['harmonic']['max_residual']))
        print(fid, json.dumps(records[-1]), flush=True)
    for rel, digest in dependencies.items(): assert sha(ROOT/rel) == digest, rel
    dump(OUT/'verification.json', dict(status='passed', records=records, solver_checks=checks,
         paper_old_unknown_values_ignored=True, update_roles_disjoint=True,
         outside_updates_exact_saved_npj=True, red_hold_extra_exact_saved_npj=True,
         stroke_only_does_not_brighten_pixels=True, npj_rerun=False, diffusion_rerun=False,
         pixel_GT=False, independent_effectiveness=False, adopted=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p)
         for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Frozen role-separated npj postprocessing pilot.', flush=True)


if __name__ == '__main__': main()
